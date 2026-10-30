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
//! # Repetitions in the signal
//!
//! A preamble of two identical halves (Schmidl & Cox 1997), a repeated symbol (Moose
//! 1994) and the cyclic prefix of OFDM (van de Beek et al. 1997) are all found by the
//! correlation of the signal with itself `lag` samples later, summed over a window,
//!
//! $$
//! P(d) = \sum_{m=0}^{W-1} r^{*}[d + m]\, r[d + m + \text{lag}], \qquad
//! E(d) = \sum_{m=0}^{W-1} |r[d + m]|^{2},
//! $$
//!
//! [`delayed_correlation`] and [`windowed_energy`], for every $d$ in $O(n)$ instead of
//! $O(nW)$: the terms are cut into blocks of $W$, and the window that starts at $d$ is
//! the sum of the terms of its block from $d$ (accumulated from the end of the block)
//! and of the terms of the next block up to $d + W - 1$ (accumulated from its start),
//! the decomposition of van Herk (1992) and Gil & Werman (1993). No sum is subtracted
//! (unlike running sums): a window of zeros (a zero-padded capture) gives zero exactly,
//! the rounding error is bounded as that of the direct sum, and a window that starts a
//! block is the direct sum itself.
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
//!
//! # References
//!
//! 1. M. van Herk, "A fast algorithm for local minimum and maximum filters on
//!    rectangular and octagonal kernels," *Pattern Recognit. Lett.*, 13(7), 517–521, 1992.
//! 2. J. Gil and M. Werman, "Computing 2-D min, median, and max filters," *IEEE Trans.
//!    Pattern Anal. Mach. Intell.*, 15(5), 504–507, 1993.

use com_core::{Result, ensure, ensure_shape};
use com_math::C64;
use com_math::fft;
use com_math::ln_1p_exp;
use com_math::stats::Correlation;
use std::f64::consts::TAU;
use std::ops::{Add, Range};

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
/// [`com_core::Error::Config`] if the positions do not lie in `reference`,
/// [`com_core::Error::Shape`] if no delay in `[-max_delay, max_delay]` keeps them inside
/// `z` (a received block too short, as for the window of [`periodic_offset`]).
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
    let known = &reference[positions.clone()];
    let mut best: Option<(isize, C64, f64)> = None;
    let md = max_delay as isize;
    for d in -md..=md {
        let lo = positions.start as isize + d;
        let hi = positions.end as isize + d;
        if lo < 0 || hi > z.len() as isize {
            continue;
        }
        let c = Correlation::between(&z[lo as usize..hi as usize], known);
        let q = c.rho();
        if !matches!(best, Some(b) if q <= b.2) {
            best = Some((d, c.cross, q));
        }
    }
    let (delay, c, quality) = best.ok_or_else(|| {
        com_core::Error::Shape(format!(
            "no delay within ±{max_delay} keeps the known symbols {positions:?} inside the {} received samples",
            z.len()
        ))
    })?;
    Ok(FrameSync { delay, rotation: derotation(c, symmetry), quality })
}

/// The rotation $e^{-j\hat\varphi}$ that removes the phase $\hat\varphi$ of the
/// correlation `c`, rounded to the nearest multiple of $2\pi/M$ for a rotational
/// symmetry $M$ = `symmetry` (exact for `None` or 0).
fn derotation(c: C64, symmetry: Option<u32>) -> C64 {
    let phi = c.arg();
    let phi = match symmetry {
        Some(m) if m > 0 => {
            let step = TAU / m as f64;
            (phi / step).round() * step
        }
        _ => phi,
    };
    C64::from_polar(1.0, -phi)
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
    // the correlation of the FFT, with the energies of the window and of its reference
    let energy_a: f64 = a.iter().map(C64::norm_sqr).sum();
    let energy_b: f64 = (0..l).map(|t| reference[(start + t + k) % p].norm_sqr()).sum();
    let quality = Correlation { cross: ck, energy_a, energy_b }.rho();
    Ok(PeriodicOffset { offset: k, rotation: derotation(ck, symmetry), quality })
}

/// The correlation $P(d) = \sum_{m=0}^{W-1} r^{*}[d + m]\, r[d + m + \text{lag}]$ of `r`
/// with itself `lag` samples later over a window of $W$ = `window` samples, for every
/// start $d$ from 0 to `r.len() - lag - window` (empty if `r` is shorter).
///
/// # Examples
/// ```
/// use com_dsp::sync::frame::delayed_correlation;
/// use com_math::C64;
/// // a block repeated after 4 samples, rotated by 0.3 rad
/// let b = [C64::new(1.0, 0.0), C64::new(0.0, 1.0), C64::new(-1.0, 0.0), C64::new(0.5, 0.5)];
/// let mut r = b.to_vec();
/// r.extend(b.iter().map(|v| v * C64::from_polar(1.0, 0.3)));
/// let p = delayed_correlation(&r, 4, 4);
/// assert_eq!(p.len(), 1);
/// assert!((p[0].arg() - 0.3).abs() < 1e-12);
/// ```
pub fn delayed_correlation(r: &[C64], lag: usize, window: usize) -> Vec<C64> {
    if lag > r.len() {
        return Vec::new();
    }
    let terms: Vec<C64> = r.iter().zip(&r[lag..]).map(|(a, b)| a.conj() * b).collect();
    sliding_sums(&terms, window)
}

/// The energy $E(d) = \sum_{m=0}^{W-1} |r[d + m]|^{2}$ of `r` over a window of $W$ =
/// `window` samples, for every start $d$ from 0 to `r.len() - window` (empty if `r` is
/// shorter).
pub fn windowed_energy(r: &[C64], window: usize) -> Vec<f64> {
    let terms: Vec<f64> = r.iter().map(C64::norm_sqr).collect();
    sliding_sums(&terms, window)
}

/// The sums of the `t.len() - w + 1` windows of `w` consecutive terms of `t` (none if
/// `t` is shorter; zeros for `w = 0`), by the block prefix and suffix sums of the module
/// documentation.
fn sliding_sums<T: Copy + Default + Add<Output = T>>(t: &[T], w: usize) -> Vec<T> {
    let n = t.len();
    if w == 0 {
        return vec![T::default(); n + 1];
    }
    if n < w {
        return Vec::new();
    }
    // prefix[i]: the terms of the block of i up to i, from its start (the direct sum);
    // suffix[i]: the terms of the block of i from i, from its end
    let mut prefix = Vec::with_capacity(n);
    for (i, &v) in t.iter().enumerate() {
        let p = if i % w == 0 { T::default() + v } else { prefix[i - 1] + v };
        prefix.push(p);
    }
    let mut suffix = t.to_vec();
    for i in (0..n.saturating_sub(1)).rev() {
        if (i + 1) % w != 0 {
            suffix[i] = t[i] + suffix[i + 1];
        }
    }
    (0..=n - w).map(|d| if d % w == 0 { prefix[d + w - 1] } else { suffix[d] + prefix[d + w - 1] }).collect()
}

/// The decision rule of [`locate_sync_word`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SyncRule {
    /// The correlation $\sum_{i} s_{i} \rho_{i+\mu}$ (its absolute value with a sign ambiguity).
    Correlation,
    /// The optimum rule of Massey (1972) at the SNR $E/N_{0}$ `snr_lin` (linear).
    Optimum {
        /// $E/N_{0}$ of a symbol (linear).
        snr_lin: f64,
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
/// assert_eq!(locate_sync_word(&r, &barker7, SyncRule::Optimum { snr_lin: 1.0 }, false)?, 10);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn locate_sync_word(r: &[f64], sync: &[f64], rule: SyncRule, ambiguous: bool) -> Result<usize> {
    let (n, l) = (r.len(), sync.len());
    ensure(l >= 1 && l <= n, || format!("a sync word of {l} symbols in a frame of {n}"))?;
    if let SyncRule::Optimum { snr_lin } = rule {
        ensure(snr_lin > 0.0, || format!("the SNR must be positive, got {snr_lin}"))?;
    }
    let statistic = |mu: usize| {
        let corr: f64 = (0..l).map(|i| sync[i] * r[(mu + i) % n]).sum();
        match (rule, ambiguous) {
            (SyncRule::Correlation, false) => corr,
            (SyncRule::Correlation, true) => corr.abs(),
            (SyncRule::Optimum { snr_lin }, false) => {
                corr - (0..l).map(|i| ln_cosh(2.0 * snr_lin * r[(mu + i) % n])).sum::<f64>() / (2.0 * snr_lin)
            }
            (SyncRule::Optimum { snr_lin }, true) => {
                ln_cosh(2.0 * snr_lin * corr) - (0..l).map(|i| ln_cosh(2.0 * snr_lin * r[(mu + i) % n])).sum::<f64>()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_outside_the_received_block_is_a_shape_error_in_both_synchronisations() {
        let x: Vec<C64> = (0..20).map(|k| C64::from_polar(1.0, 0.7 * f64::from(k))).collect();
        assert!(matches!(frame_sync(&x[..5], &x, 0..10, 2, None), Err(com_core::Error::Shape(_))));
        assert!(matches!(periodic_offset(&x[..5], &x, 0, 10, None), Err(com_core::Error::Shape(_))));
        assert!(matches!(frame_sync(&x, &x, 15..25, 2, None), Err(com_core::Error::Config(_))));
    }

    #[test]
    fn the_windowed_sums_are_the_direct_sums_and_short_inputs_give_nothing() {
        // small integers: every sum is exact, whatever its order
        let r: Vec<C64> = (0..9).map(|n| C64::new(f64::from(n), 1.0 - f64::from(n % 3))).collect();
        let p = delayed_correlation(&r, 3, 4);
        assert_eq!(p.len(), 3);
        for (d, v) in p.iter().enumerate() {
            let want: C64 = (0..4).map(|m| r[d + m].conj() * r[d + m + 3]).sum();
            assert_eq!(*v, want);
        }
        let e = windowed_energy(&r, 4);
        assert_eq!(e.len(), 6);
        assert_eq!(e[2], (2..6).map(|n| r[n].norm_sqr()).sum::<f64>());
        assert!(delayed_correlation(&r, 6, 4).is_empty() && windowed_energy(&r, 10).is_empty());
        assert_eq!(delayed_correlation(&r, 10, 0), Vec::<C64>::new());
        assert_eq!(windowed_energy(&r, 0), vec![0.0; 10]);
    }

    /// The direct sums of the former code: the same values to the rounding of the sums,
    /// the same bits at the starts of the blocks, and exact zeros on zero padding.
    #[test]
    fn the_block_sums_are_the_former_direct_sums_to_the_rounding() {
        let mut rng = com_core::rng::Rng::new(4);
        let mut r: Vec<C64> = (0..2000).map(|_| rng.complex_gaussian() * 10.0).collect();
        r.extend(vec![C64::new(0.0, 0.0); 300]);
        for (lag, w) in [(64, 64), (256, 32), (1, 1), (3, 1000)] {
            let p = delayed_correlation(&r, lag, w);
            let e = windowed_energy(&r, w);
            assert_eq!((p.len(), e.len()), (r.len() - lag - w + 1, r.len() - w + 1));
            for (d, v) in p.iter().enumerate() {
                let want: C64 = (0..w).map(|m| r[d + m].conj() * r[d + m + lag]).sum();
                let scale: f64 = (0..w).map(|m| (r[d + m].conj() * r[d + m + lag]).norm()).sum();
                assert!((v - want).norm() <= 1e-15 * w as f64 * scale, "lag {lag}, window {w}, d {d}");
                if d % w == 0 {
                    assert_eq!(*v, want);
                }
                if d >= 2000 {
                    assert_eq!(*v, C64::new(0.0, 0.0));
                }
            }
            for (d, v) in e.iter().enumerate() {
                let want: f64 = (0..w).map(|m| r[d + m].norm_sqr()).sum();
                assert!((v - want).abs() <= 1e-15 * w as f64 * want, "window {w}, d {d}");
                if d >= 2000 {
                    assert_eq!(*v, 0.0);
                }
            }
        }
    }
}
