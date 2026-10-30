//! Frame synchronisation on known symbols.
//!
//! After the blind stages of a receiver (clock recovery, blind phase search), the
//! symbol stream is known up to a delay of a few symbols and, for a constellation with
//! an `M`-fold symmetry, a rotation by a multiple of `2π/M`. Both are resolved on a
//! block of known symbols `x[n]`, `n ∈ P` (a preamble or pilots):
//!
//! `c(d) = Σ_{n∈P} z[n + d] x*[n]`, `d̂ = argmax_{|d| ≤ D} |c(d)|`,
//!
//! and the rotation is `e^{-jφ̂}` with `φ̂` the phase of `c(d̂)`, rounded to the nearest
//! multiple of `2π/M` when a symmetry is given (the residual phase is then left to the
//! carrier recovery), exact otherwise.
//!
//! [`frame_sync`] estimates `(d̂, rotation)`, [`align`] applies them.
//!
//! # A sync word in binary data
//!
//! When a sync word $s_{0}, \dots, s_{L-1} \in \{\pm 1\}$ is embedded in equiprobable
//! binary data, frames of $N$ symbols, the correlation $\sum_{i} s_{i} \rho_{i+\mu}$ is not
//! the best rule: on the AWGN channel the most likely position maximises it minus a
//! correction for the data around (Massey 1972, eqs. (5)–(6)),
//!
//! $$
//! S(\mu) = \sum_{i=0}^{L-1} s_{i} \rho_{i+\mu} - \sum_{i=0}^{L-1} f(\rho_{i+\mu}), \qquad
//! f(x) = \frac{1}{2\gamma} \ln\cosh(2 \gamma x),
//! $$
//!
//! for the samples $\rho$ normalised to a unit signal amplitude and $\gamma = E/N_{0}$;
//! with the sign ambiguity of a binary PSK detector, $\ln\cosh(2\gamma \sum_{i} s_{i}
//! \rho_{i+\mu}) - \sum_{i} \ln\cosh(2\gamma \rho_{i+\mu})$ (their eq. (9)), against the
//! absolute correlation. [`locate_sync_word`] applies a [`SyncRule`].
//!
//! # Periodic references
//!
//! A measured capture usually carries a periodic sequence (a PRBS, or a block of
//! symbols repeated by an arbitrary waveform generator) and starts at an unknown point
//! of its period `P`. [`periodic_offset`] finds the offset `k̂` such that the received
//! symbol `z[n]` carries `x[(n + k̂) mod P]`, by the circular correlation of a window
//! of `z` with the reference computed by FFT (`O(P log P)` for all the offsets); the
//! receiver then shifts the reference instead of the samples.
//!
//! # Examples
//! ```
//! use com_dsp::sync::frame::{align, frame_sync};
//! use com_core::rng::Rng;
//! use com_math::C64;
//! // 16-QAM of unit energy
//! let c: Vec<C64> = (0..16).map(|i| C64::new((2 * (i % 4) - 3) as f64, (2 * (i / 4) - 3) as f64) / 10f64.sqrt()).collect();
//! let mut rng = Rng::new(1);
//! let x: Vec<C64> = (0..500).map(|_| c[rng.below(16)]).collect();
//! // the receiver output lags by 3 symbols and is rotated by π/2
//! let z: Vec<C64> = (0..500).map(|n| if n >= 3 { x[n - 3] * C64::new(0.0, 1.0) } else { C64::new(0.0, 0.0) }).collect();
//! let s = frame_sync(&z, &x, 100..200, 10, Some(4)).unwrap();
//! assert_eq!(s.delay, 3);
//! let y = align(&z, s.delay, s.rotation, x.len());
//! assert!((y[250] - x[250]).norm() < 1e-12);
//! ```

use com_core::{Result, ensure, ensure_shape};
use com_math::C64;
use com_math::fft;
use com_math::ln_1p_exp;
use std::f64::consts::TAU;
use std::ops::Range;

/// The result of [`frame_sync`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameSync {
    /// Delay `d̂` of the received stream: `z[n + d̂]` carries `x[n]`.
    pub delay: isize,
    /// Rotation to multiply the received stream with.
    pub rotation: C64,
    /// Normalised correlation `|c(d̂)| / √(Σ|z|² Σ|x|²)` over the known symbols, in
    /// `[0, 1]` (close to 1 when the synchronisation is reliable).
    pub quality: f64,
}

/// Delay and rotation of `z` against the known symbols `reference[positions]`.
///
/// `max_delay` bounds `|d|`; `symmetry` is the rotational symmetry `M` of the
/// constellation (4 for square QAM), or `None` to remove the exact phase.
///
/// # Errors
/// [`com_core::Error::Config`] if the positions do not lie in `reference`, or if no delay
/// in `[-max_delay, max_delay]` keeps them inside `z`.
pub fn frame_sync(
    z: &[C64],
    reference: &[C64],
    positions: Range<usize>,
    max_delay: usize,
    symmetry: Option<u32>,
) -> Result<FrameSync> {
    ensure(positions.end <= reference.len() && !positions.is_empty(), || {
        format!("the known symbols {positions:?} must lie in the {} reference symbols", reference.len())
    })?;
    let ex: f64 = reference[positions.clone()].iter().map(C64::norm_sqr).sum();
    let mut best: Option<(isize, C64, f64)> = None;
    let md = max_delay as isize;
    for d in -md..=md {
        let lo = positions.start as isize + d;
        let hi = positions.end as isize + d;
        if lo < 0 || hi > z.len() as isize {
            continue;
        }
        let mut c = C64::new(0.0, 0.0);
        let mut ez = 0.0;
        for n in positions.clone() {
            let v = z[(n as isize + d) as usize];
            c += v * reference[n].conj();
            ez += v.norm_sqr();
        }
        let q = if ez > 0.0 && ex > 0.0 { c.norm() / (ez * ex).sqrt() } else { 0.0 };
        if !matches!(best, Some(b) if q <= b.2) {
            best = Some((d, c, q));
        }
    }
    let (delay, c, quality) = best.ok_or_else(|| {
        com_core::Error::Config(format!(
            "no delay within ±{max_delay} keeps the known symbols {positions:?} inside the {} received samples",
            z.len()
        ))
    })?;
    let phi = c.arg();
    let phi = match symmetry {
        Some(m) if m > 0 => {
            let step = TAU / m as f64;
            (phi / step).round() * step
        }
        _ => phi,
    };
    Ok(FrameSync { delay, rotation: C64::from_polar(1.0, -phi), quality })
}

/// `y[n] = rotation · z[n + delay]` for `n = 0..len` (zero outside `z`).
pub fn align(z: &[C64], delay: isize, rotation: C64, len: usize) -> Vec<C64> {
    (0..len as isize)
        .map(|n| {
            let k = n + delay;
            if k >= 0 && (k as usize) < z.len() { z[k as usize] * rotation } else { C64::new(0.0, 0.0) }
        })
        .collect()
}

/// The result of [`periodic_offset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicOffset {
    /// The offset `k̂`: `z[n]` carries `reference[(n + k̂) mod P]`.
    pub offset: usize,
    /// Rotation to multiply the received stream with.
    pub rotation: C64,
    /// Normalised correlation over the window, in `[0, 1]`.
    pub quality: f64,
}

/// Offset of `z` in a periodic reference of period `reference.len()`.
///
/// The correlation uses `window` samples of `z` from `start` (at most one period);
/// `symmetry` rounds the rotation to a multiple of `2π/M` as in [`frame_sync`].
///
/// # Errors
/// [`com_core::Error::Config`] for an empty reference, [`com_core::Error::Shape`] for a
/// window outside `z`.
///
/// # Examples
/// ```
/// use com_dsp::sync::frame::periodic_offset;
/// use com_math::C64;
/// let p: Vec<C64> = (0..127).map(|k| C64::new(if (k * k) % 7 < 3 { 1.0 } else { -1.0 }, 0.0)).collect();
/// // a capture that starts 40 symbols into the period, rotated by π
/// let z: Vec<C64> = (0..300).map(|n| -p[(n + 40) % 127]).collect();
/// let r = periodic_offset(&z, &p, 0, 127, Some(4)).unwrap();
/// assert_eq!(r.offset, 40);
/// assert!((r.rotation + C64::new(1.0, 0.0)).norm() < 1e-12);
/// ```
pub fn periodic_offset(
    z: &[C64],
    reference: &[C64],
    start: usize,
    window: usize,
    symmetry: Option<u32>,
) -> Result<PeriodicOffset> {
    let p = reference.len();
    ensure(p > 0, || "the reference is empty".into())?;
    let l = window.min(p);
    ensure_shape(l > 0 && start + l <= z.len(), || {
        format!("the window {start}..{} is outside the {} received symbols", start + l, z.len())
    })?;
    // c(k) = Σ_t a[t] conj(r[(start + t + k) mod P]) = conj(IFFT(conj(A) · FFT(r'))[k])
    let mut a = vec![C64::new(0.0, 0.0); p];
    a[..l].copy_from_slice(&z[start..start + l]);
    let r: Vec<C64> = (0..p).map(|k| reference[(start + k) % p]).collect();
    let fa = fft::fft(&a);
    let fr = fft::fft(&r);
    let prod: Vec<C64> = fa.iter().zip(&fr).map(|(u, v)| u.conj() * v).collect();
    let c: Vec<C64> = fft::ifft(&prod).into_iter().map(|v| v.conj()).collect();
    let (k, ck) =
        c.iter().enumerate().fold((0, C64::new(0.0, 0.0)), |b, (k, v)| if v.norm() > b.1.norm() { (k, *v) } else { b });
    let ea: f64 = a.iter().map(C64::norm_sqr).sum();
    let er: f64 = (0..l).map(|t| reference[(start + t + k) % p].norm_sqr()).sum();
    let quality = if ea > 0.0 && er > 0.0 { ck.norm() / (ea * er).sqrt() } else { 0.0 };
    let phi = ck.arg();
    let phi = match symmetry {
        Some(m) if m > 0 => {
            let step = TAU / m as f64;
            (phi / step).round() * step
        }
        _ => phi,
    };
    Ok(PeriodicOffset { offset: k, rotation: C64::from_polar(1.0, -phi), quality })
}

/// The decision rule of [`locate_sync_word`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SyncRule {
    /// The correlation $\sum_{i} s_{i} \rho_{i+\mu}$ (its absolute value with a sign ambiguity).
    Correlation,
    /// The optimum rule of Massey (1972) at the SNR $E/N_{0}$ `snr` (linear).
    Optimum {
        /// $E/N_{0}$ of a symbol (linear).
        snr: f64,
    },
}

/// $\ln\cosh y$ without overflow.
fn ln_cosh(y: f64) -> f64 {
    y.abs() - std::f64::consts::LN_2 + ln_1p_exp(-2.0 * y.abs())
}

/// The position $\hat\mu \in [0, N)$ of the sync word `sync` ($\pm 1$) in a frame `r` of $N$
/// real detector outputs of unit signal amplitude, the frame read cyclically (the sync
/// word is periodic), by the rule `rule`; `ambiguous` when the sign of the detector
/// output is unknown (binary PSK, Massey 1972, Sec. III).
///
/// # Errors
/// [`com_core::Error::Config`] for an empty sync word, one longer than the frame or a
/// non-positive SNR.
///
/// # Examples
/// ```
/// use com_dsp::sync::frame::{locate_sync_word, SyncRule};
/// let barker7 = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];
/// let mut r = vec![1.0; 28];
/// r[10..17].copy_from_slice(&barker7);
/// assert_eq!(locate_sync_word(&r, &barker7, SyncRule::Optimum { snr: 1.0 }, false)?, 10);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn locate_sync_word(r: &[f64], sync: &[f64], rule: SyncRule, ambiguous: bool) -> Result<usize> {
    let (n, l) = (r.len(), sync.len());
    ensure(l >= 1 && l <= n, || format!("a sync word of {l} symbols in a frame of {n}"))?;
    if let SyncRule::Optimum { snr } = rule {
        ensure(snr > 0.0, || format!("the SNR must be positive, got {snr}"))?;
    }
    let statistic = |mu: usize| {
        let corr: f64 = (0..l).map(|i| sync[i] * r[(mu + i) % n]).sum();
        match (rule, ambiguous) {
            (SyncRule::Correlation, false) => corr,
            (SyncRule::Correlation, true) => corr.abs(),
            (SyncRule::Optimum { snr }, false) => {
                corr - (0..l).map(|i| ln_cosh(2.0 * snr * r[(mu + i) % n])).sum::<f64>() / (2.0 * snr)
            }
            (SyncRule::Optimum { snr }, true) => {
                ln_cosh(2.0 * snr * corr) - (0..l).map(|i| ln_cosh(2.0 * snr * r[(mu + i) % n])).sum::<f64>()
            }
        }
    };
    let mut best = (0, f64::NEG_INFINITY);
    for mu in 0..n {
        let v = statistic(mu);
        if v > best.1 {
            best = (mu, v);
        }
    }
    Ok(best.0)
}
