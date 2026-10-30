//! Filters: pulse shapes, FIR filtering, low-pass design, moving average.
//!
//! # Pulse shaping
//!
//! [`pulse_shape`] returns the taps of a transmit pulse at `sps` samples per symbol,
//! normalised to a unit sum (unit DC gain), on the symmetric time grid
//! `t_i = (-⌊L/2⌋ + i·2⌊L/2⌋/(L-1)) / sps`, `i = 0..L-1`. For an even `L` it is the grid
//! of OptiCommPy's `pulseShape` (a `numpy.linspace` of spacing `L/(L-1)` samples, so the
//! taps fall between the sampling instants); for an odd `L` the spacing is exactly one
//! sample and the taps sit on the sampling instants, the choice to make for matched
//! filtering (OptiCommPy writes `-L // 2`, which rounds to `-⌊L/2⌋ - 1` for an odd `L`
//! and shifts its grid by about half a sample):
//!
//! - raised cosine: `h(t) = sinc(t/T) cos(παt/T) / (1 - 4α²t²/T²)`;
//! - root-raised cosine: `h(t) = [sin(πt(1-α)/T) + 4αt/T cos(πt(1+α)/T)] / [πt/T (1-(4αt/T)²)]`,
//!   with its limits at `t = 0` and `|t| = T/(4α)`;
//! - rectangular: `sps` ones padded with `⌊sps/2⌋` zeros on each side;
//! - NRZ: `sps` ones convolved with a Gaussian `e^{-t²}` sampled on `[-2, 2]`.
//!
//! [`rrc_spectrum`] gives the exact root-raised-cosine *frequency* response on the bins
//! of a periodic block, the matched filter of the split-step simulations.
//!
//! # Filtering
//!
//! - [`fir_filter`] convolves with the `same` alignment of `scipy.signal`: the output
//!   has the length of the input and `y[n] = Σ_k h_k x[n + ⌊(L-1)/2⌋ - k]`, so a
//!   symmetric filter introduces no delay; long filters use the FFT;
//! - [`low_pass_fir`] designs a windowless sinc (`Rect`) or Gaussian low-pass FIR of
//!   cut-off `fc`;
//! - [`moving_average`] (centred, zeros outside).
//!
//! # Examples
//! ```
//! use com_dsp::filter::fir_filter;
//! use com_math::C64;
//! // a symmetric filter keeps the alignment: an impulse comes out centred
//! let mut x = vec![C64::new(0.0, 0.0); 9];
//! x[4] = C64::new(1.0, 0.0);
//! let y = fir_filter(&[0.25, 0.5, 0.25], &x);
//! assert!((y[4].re - 0.5).abs() < 1e-15 && (y[3].re - 0.25).abs() < 1e-15);
//! ```

use com_core::{Error, Result};
use com_math::fft;
use com_math::{C64, sinc};
use std::f64::consts::PI;

/// A transmit pulse shape.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PulseShape {
    /// Rectangular pulse of one symbol.
    Rect,
    /// Non-return-to-zero pulse with Gaussian edges.
    Nrz,
    /// Raised cosine of roll-off `rolloff` (Nyquist pulse).
    Rc {
        /// Roll-off factor `α ∈ (0, 1]`.
        rolloff: f64,
    },
    /// Root-raised cosine of roll-off `rolloff` (half of a Nyquist pair).
    Rrc {
        /// Roll-off factor `α ∈ (0, 1]`.
        rolloff: f64,
    },
}

impl PulseShape {
    /// The pulse of a name, `rect`, `nrz`, `rc` or `rrc` (case-insensitive), with the
    /// roll-off of the raised cosines.
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    pub fn from_name(kind: &str, rolloff: f64) -> Result<PulseShape> {
        match kind.to_ascii_lowercase().as_str() {
            "rect" => Ok(PulseShape::Rect),
            "nrz" => Ok(PulseShape::Nrz),
            "rc" => Ok(PulseShape::Rc { rolloff }),
            "rrc" => Ok(PulseShape::Rrc { rolloff }),
            other => Err(Error::Config(format!("pulse kind must be rect, nrz, rc or rrc, got {other:?}"))),
        }
    }
}

/// The symmetric time grid of the raised-cosine pulses, in symbols (see the module
/// documentation for the comparison with OptiCommPy).
fn pulse_grid(sps: usize, n_taps: usize) -> Vec<f64> {
    let h = (n_taps / 2) as f64;
    if n_taps == 1 {
        return vec![0.0];
    }
    (0..n_taps).map(|i| (-h + i as f64 * 2.0 * h / (n_taps as f64 - 1.0)) / sps as f64).collect()
}

/// Root-raised-cosine impulse response at the times `t` (symbol period `ts`).
pub fn rrc_taps(t: &[f64], alpha: f64, ts: f64) -> Vec<f64> {
    t.iter()
        .map(|&ti| {
            if ti == 0.0 {
                (1.0 / ts) * (1.0 + alpha * (4.0 / PI - 1.0))
            } else if alpha > 0.0 && (ti.abs() - ts / (4.0 * alpha)).abs() < 1e-12 * ts {
                let a = (1.0 + 2.0 / PI) * (PI / (4.0 * alpha)).sin();
                let b = (1.0 - 2.0 / PI) * (PI / (4.0 * alpha)).cos();
                alpha / (ts * 2f64.sqrt()) * (a + b)
            } else {
                let t1 = PI * ti / ts;
                let t2 = 4.0 * alpha * ti / ts;
                (1.0 / ts) * ((t1 * (1.0 - alpha)).sin() + t2 * (t1 * (1.0 + alpha)).cos()) / (t1 * (1.0 - t2 * t2))
            }
        })
        .collect()
}

/// Raised-cosine impulse response at the times `t` (symbol period `ts`).
pub fn rc_taps(t: &[f64], alpha: f64, ts: f64) -> Vec<f64> {
    t.iter()
        .map(|&ti| {
            if alpha > 0.0 && (ti.abs() - ts / (2.0 * alpha)).abs() < 1e-12 * ts {
                PI / (4.0 * ts) * sinc(1.0 / (2.0 * alpha))
            } else {
                (1.0 / ts) * sinc(ti / ts) * (PI * alpha * ti / ts).cos()
                    / (1.0 - 4.0 * alpha * alpha * ti * ti / (ts * ts))
            }
        })
        .collect()
}

/// Taps of a pulse at `sps` samples per symbol, normalised to a unit sum.
///
/// `n_taps` is the length of the raised-cosine pulses (ignored by `Rect` and `Nrz`).
///
/// # Errors
/// [`Error::Config`] for `sps = 0`, or `n_taps = 0` for a raised-cosine pulse.
pub fn pulse_shape(shape: PulseShape, sps: usize, n_taps: usize) -> Result<Vec<f64>> {
    let raised_cosine = matches!(shape, PulseShape::Rc { .. } | PulseShape::Rrc { .. });
    if sps == 0 || (raised_cosine && n_taps == 0) {
        return Err(Error::Config(format!("sps and n_taps must be positive, got {sps} and {n_taps}")));
    }
    let pulse: Vec<f64> = match shape {
        PulseShape::Rect => {
            let z = sps / 2;
            let mut p = vec![0.0; z];
            p.extend(std::iter::repeat_n(1.0, sps));
            p.extend(std::iter::repeat_n(0.0, z));
            p
        }
        PulseShape::Nrz => {
            let g: Vec<f64> = (0..sps)
                .map(|i| {
                    let t = if sps == 1 { 0.0 } else { -2.0 + 4.0 * i as f64 / (sps as f64 - 1.0) };
                    2.0 / PI.sqrt() * (-t * t).exp()
                })
                .collect();
            let mut p = vec![0.0; 2 * sps - 1];
            for i in 0..sps {
                for (j, gj) in g.iter().enumerate() {
                    p[i + j] += gj;
                }
            }
            p
        }
        PulseShape::Rc { rolloff } => rc_taps(&pulse_grid(sps, n_taps), rolloff, 1.0),
        PulseShape::Rrc { rolloff } => rrc_taps(&pulse_grid(sps, n_taps), rolloff, 1.0),
    };
    let s: f64 = pulse.iter().sum();
    Ok(pulse.iter().map(|v| v / s).collect())
}

/// Root of the raised-cosine spectrum on the `n_sym·sps` bins of a periodic block.
///
/// `H_k = √RC(f_k)` with `f_k` the frequency of bin `k` in symbol-rate units; the value
/// is 1 in the passband, so that shaping followed by matched filtering with this
/// response is exactly Nyquist on a periodic block.
pub fn rrc_spectrum(n_sym: usize, sps: usize, rolloff: f64) -> Vec<f64> {
    let lo = 0.5 * (1.0 - rolloff);
    let hi = 0.5 * (1.0 + rolloff);
    fft::fftfreq(n_sym * sps, 1.0)
        .into_iter()
        .map(|f| {
            let f = (f * sps as f64).abs();
            let rc = if f <= lo {
                1.0
            } else if f < hi {
                0.5 * (1.0 + (PI / rolloff * (f - lo)).cos())
            } else {
                0.0
            };
            rc.sqrt()
        })
        .collect()
}

/// Full linear convolution of `x` with complex taps `h` (direct or by FFT).
pub fn convolve_full(h: &[C64], x: &[C64]) -> Vec<C64> {
    if h.is_empty() || x.is_empty() {
        return Vec::new();
    }
    let n = x.len() + h.len() - 1;
    if h.len().min(x.len()) <= 32 {
        let mut y = vec![C64::new(0.0, 0.0); n];
        for (i, &xi) in x.iter().enumerate() {
            for (k, &hk) in h.iter().enumerate() {
                y[i + k] += hk * xi;
            }
        }
        return y;
    }
    // a long signal and a shorter filter: overlap-add with blocks of a few filter lengths
    if x.len() > 8 * h.len() {
        let nfft = (4 * h.len()).next_power_of_two().max(64);
        let block = nfft - h.len() + 1;
        let p = fft::plan(nfft);
        let mut hf = vec![C64::new(0.0, 0.0); nfft];
        hf[..h.len()].copy_from_slice(h);
        p.forward(&mut hf);
        let mut y = vec![C64::new(0.0, 0.0); n];
        let mut buf = vec![C64::new(0.0, 0.0); nfft];
        for start in (0..x.len()).step_by(block) {
            let seg = &x[start..(start + block).min(x.len())];
            for v in &mut buf {
                *v = C64::new(0.0, 0.0);
            }
            buf[..seg.len()].copy_from_slice(seg);
            p.forward(&mut buf);
            for (u, v) in buf.iter_mut().zip(&hf) {
                *u *= v;
            }
            p.inverse(&mut buf);
            let len = (seg.len() + h.len() - 1).min(n - start);
            for (yo, b) in y[start..start + len].iter_mut().zip(&buf) {
                *yo += b;
            }
        }
        return y;
    }
    let m = n.next_power_of_two();
    let p = fft::plan(m);
    let mut a = vec![C64::new(0.0, 0.0); m];
    let mut b = vec![C64::new(0.0, 0.0); m];
    a[..x.len()].copy_from_slice(x);
    b[..h.len()].copy_from_slice(h);
    p.forward(&mut a);
    p.forward(&mut b);
    for (u, v) in a.iter_mut().zip(&b) {
        *u *= v;
    }
    p.inverse(&mut a);
    a.truncate(n);
    a
}

/// FIR filtering with complex taps and the `same` alignment: as many samples as `x`
/// (zeros for no tap).
pub fn fir_filter_complex(h: &[C64], x: &[C64]) -> Vec<C64> {
    if h.is_empty() || x.is_empty() {
        return vec![C64::new(0.0, 0.0); x.len()];
    }
    let full = convolve_full(h, x);
    let start = (h.len() - 1) / 2;
    full[start..start + x.len()].to_vec()
}

/// FIR filtering with real taps and the `same` alignment (OptiCommPy `firFilter`).
pub fn fir_filter(h: &[f64], x: &[C64]) -> Vec<C64> {
    let hc: Vec<C64> = h.iter().map(|&v| C64::new(v, 0.0)).collect();
    fir_filter_complex(&hc, x)
}

/// Low-pass FIR filter types of [`low_pass_fir`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowPass {
    /// Truncated sinc (ideal rectangular response).
    Rect,
    /// Gaussian response with 3 dB bandwidth `fc`.
    Gauss,
}

impl LowPass {
    /// The name of the filter, as `from_str` reads it: `rect` or `gauss`.
    pub fn name(self) -> &'static str {
        match self {
            LowPass::Rect => "rect",
            LowPass::Gauss => "gauss",
        }
    }
}

impl std::str::FromStr for LowPass {
    type Err = Error;

    /// `rect` or `gauss` (`gaussian`), case-insensitive.
    fn from_str(s: &str) -> Result<LowPass> {
        match s.to_ascii_lowercase().as_str() {
            "rect" => Ok(LowPass::Rect),
            "gauss" | "gaussian" => Ok(LowPass::Gauss),
            other => Err(Error::Config(format!("unknown filter {other:?} (rect, gauss)"))),
        }
    }
}

/// `n`-tap low-pass FIR of cut-off `fc` at the sampling rate `fs`, unit DC gain
/// (OptiCommPy `lowPassFIR`).
pub fn low_pass_fir(fc: f64, fs: f64, n: usize, kind: LowPass) -> Vec<f64> {
    let fu = fc / fs;
    let d = (n as f64 - 1.0) / 2.0;
    let h: Vec<f64> = (0..n)
        .map(|i| {
            let m = i as f64 - d;
            match kind {
                LowPass::Rect => 2.0 * fu * sinc(2.0 * fu * m),
                LowPass::Gauss => {
                    (2.0 * PI / 2f64.ln()).sqrt() * fu * (-(2.0 / 2f64.ln()) * (PI * fu * m).powi(2)).exp()
                }
            }
        })
        .collect();
    let s: f64 = h.iter().sum();
    h.iter().map(|v| v / s).collect()
}

/// Centred moving average over `n` samples, zeros outside the signal
/// (OptiCommPy `movingAverage`): `y[i] = (1/n) Σ_{m=i+c-n+1}^{i+c} x[m]`, `c = ⌊(n-1)/2⌋`.
pub fn moving_average(x: &[C64], n: usize) -> Vec<C64> {
    let len = x.len() as isize;
    let n = n.max(1) as isize;
    let c = (n - 1) / 2;
    // prefix sums for O(len)
    let mut pre = vec![C64::new(0.0, 0.0); x.len() + 1];
    for i in 0..x.len() {
        pre[i + 1] = pre[i] + x[i];
    }
    (0..len)
        .map(|i| {
            let hi = (i + c).min(len - 1);
            let lo = (i + c - n + 1).max(0);
            if hi < lo { C64::new(0.0, 0.0) } else { (pre[(hi + 1) as usize] - pre[lo as usize]) / n as f64 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::*;

    #[test]
    fn a_pulse_needs_samples_and_taps() {
        assert!(pulse_shape(PulseShape::Nrz, 0, 16).is_err());
        assert!(pulse_shape(PulseShape::Rrc { rolloff: 0.2 }, 4, 0).is_err());
        assert_eq!(pulse_shape(PulseShape::Rect, 2, 0).unwrap(), vec![0.0, 0.5, 0.5, 0.0]);
    }

    #[test]
    fn the_low_pass_names_are_read_back() {
        for f in [LowPass::Rect, LowPass::Gauss] {
            assert_eq!(f.name().parse::<LowPass>().unwrap(), f);
        }
    }

    #[test]
    fn rrc_pair_is_nyquist_on_a_periodic_block() {
        let n_sym = 64;
        let sps = 4;
        let h = rrc_spectrum(n_sym, sps, 0.25);
        // RC = |H|² vanishes on the alias pairs f and f ± Rs: zero ISI
        let s: Vec<C64> = (0..n_sym).map(|k| C64::new((k % 3) as f64 - 1.0, (k % 5) as f64 - 2.0)).collect();
        let mut up = fft::fft(&upsample(&s, sps));
        for (v, hk) in up.iter_mut().zip(&h) {
            *v *= hk * hk;
        }
        // Y_m = S_m (1/sps) Σ_aliases RC = S_m / sps
        let y = downsample(&fft::ifft(&up), sps, 0);
        for (a, b) in y.iter().zip(&s) {
            assert!((a * sps as f64 - b).norm() < 1e-9, "{a} vs {b}");
        }
        assert!((h[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn rc_taps_are_nyquist() {
        // samples of the RC pulse at the symbol instants: 1 at 0, 0 elsewhere
        let t: Vec<f64> = (-5..=5).map(|k| k as f64).collect();
        let h = rc_taps(&t, 0.3, 1.0);
        for (k, v) in t.iter().zip(&h) {
            if *k == 0.0 {
                assert!((v - 1.0).abs() < 1e-12);
            } else {
                assert!(v.abs() < 1e-12, "t = {k}: {v}");
            }
        }
    }

    #[test]
    fn moving_average_and_resampling() {
        let x: Vec<C64> = (0..10).map(|n| C64::new(n as f64, 0.0)).collect();
        let y = moving_average(&x, 3);
        assert!((y[5].re - 5.0).abs() < 1e-12 && (y[0].re - 1.0 / 3.0).abs() < 1e-12);
        let up = resample_fft(&x.iter().map(|v| v * 0.0 + C64::new(1.0, 0.0)).collect::<Vec<_>>(), 30);
        assert!(up.iter().all(|v| (v - C64::new(1.0, 0.0)).norm() < 1e-12));
        let d = delay_fft(&x, 0.0, 1.0);
        assert!(d.iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-12));
    }

    #[test]
    fn long_and_short_convolutions_agree() {
        let h: Vec<C64> = (0..40).map(|k| C64::new((k as f64 * 0.3).sin(), 0.1 * k as f64)).collect();
        let x: Vec<C64> = (0..100).map(|k| C64::new((k as f64 * 0.7).cos(), 0.0)).collect();
        let fast = convolve_full(&h, &x);
        let mut slow = vec![C64::new(0.0, 0.0); 139];
        for i in 0..100 {
            for k in 0..40 {
                slow[i + k] += h[k] * x[i];
            }
        }
        assert!(fast.iter().zip(&slow).all(|(a, b)| (a - b).norm() < 1e-9));
    }
}
