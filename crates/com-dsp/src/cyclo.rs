//! Cyclostationary analysis: the cyclic autocorrelation and the blind estimation of the
//! symbol rate.
//!
//! # Model
//!
//! A linearly modulated signal $x(t) = \sum_{k} a_{k}\, g(t - kT)$ is cyclostationary: its
//! autocorrelation is periodic in $t$ with the period $T$. The cyclic autocorrelation at
//! the cycle frequency $\alpha$ and the lag $\tau$ (Gardner 1991),
//!
//! $$
//! R_{x}^{\alpha}(\tau) = \frac{1}{N} \sum_{n} x\[n + \tau\]\, x^{*}\[n\]\, e^{-j 2\pi \alpha n / f_{s}},
//! $$
//!
//! is non-zero at $\alpha = k/T$ only. At $\tau = 0$ it is the Fourier transform of the
//! instantaneous power $|x\[n\]|^{2}$: its line at $\alpha = 1/T$, of amplitude growing with
//! the excess bandwidth of the pulse (zero for a Nyquist pulse of roll-off 0), gives the
//! symbol rate. [`estimate_symbol_rate`] picks the strongest line of the spectrum of
//! $|x|^{2}$ (its mean removed) in a search band and refines it by a parabola through the
//! log-magnitudes of the three bins around the peak.
//!
//! # Examples
//! ```
//! use com_dsp::cyclo::estimate_symbol_rate;
//! use com_math::C64;
//! // a power that oscillates at 1/8 of the sampling rate: a symbol rate of fs/8
//! let x: Vec<C64> = (0..4096).map(|n| C64::new((1.0 + 0.3 * (std::f64::consts::PI * n as f64 / 4.0).cos()).sqrt(), 0.0)).collect();
//! let rs = estimate_symbol_rate(&x, 1.0, 0.05, 0.2)?;
//! assert!((rs - 0.125).abs() < 1e-6);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. W. A. Gardner, "Exploitation of spectral redundancy in cyclostationary signals,"
//!    *IEEE Signal Process. Mag.*, 8(2), 14–36, 1991.

use com_core::{Result, ensure, ensure_shape};
use com_math::C64;
use com_math::fft::fft;
use std::f64::consts::PI;

/// The cyclic autocorrelation $R_{x}^{\alpha}(\tau)$ of `x` at the cycle frequency
/// `alpha` (Hz) and the lag `lag` (samples), at the sampling rate `fs` (see the module).
pub fn cyclic_autocorrelation(x: &[C64], alpha: f64, lag: usize, fs: f64) -> C64 {
    if lag >= x.len() {
        return C64::new(0.0, 0.0);
    }
    let n = x.len() - lag;
    let w = -2.0 * PI * alpha / fs;
    (0..n).map(|k| x[k + lag] * x[k].conj() * C64::from_polar(1.0, w * k as f64)).sum::<C64>() / n as f64
}

/// The symbol rate (Hz) of `x`, sampled at `fs`: the strongest line of the spectrum of
/// $|x|^{2}$ between `f_min` and `f_max` (Hz), refined by parabolic interpolation (see
/// the module).
///
/// # Errors
/// [`com_core::Error::Shape`] for fewer than 16 samples, [`com_core::Error::Config`] for
/// a non-positive sampling rate or a search band outside $(0, f_{s}/2)$ narrower than
/// three bins.
pub fn estimate_symbol_rate(x: &[C64], fs: f64, f_min: f64, f_max: f64) -> Result<f64> {
    ensure_shape(x.len() >= 16, || format!("the symbol rate needs at least 16 samples, got {}", x.len()))?;
    ensure(fs > 0.0, || format!("the sampling rate must be positive, got {fs}"))?;
    let n = x.len();
    let bin = fs / n as f64;
    let (lo, hi) = ((f_min / bin).ceil().max(1.0) as usize, (f_max / bin).floor() as usize);
    ensure(f_min > 0.0 && f_max <= fs / 2.0 && hi >= lo + 2, || {
        format!("the search band [{f_min}, {f_max}] Hz must lie in (0, fs/2] and span three bins of {bin} Hz")
    })?;
    let mean = x.iter().map(C64::norm_sqr).sum::<f64>() / n as f64;
    let p: Vec<C64> = x.iter().map(|v| C64::new(v.norm_sqr() - mean, 0.0)).collect();
    let mag: Vec<f64> = fft(&p).iter().map(|v| v.norm()).collect();
    let k = (lo..=hi).max_by(|a, b| mag[*a].total_cmp(&mag[*b])).unwrap_or(lo);
    // a parabola through the log-magnitudes of k - 1, k, k + 1
    let ln = |i: usize| mag[i].max(1e-300).ln();
    let (a, b, c) = (ln(k - 1), ln(k), ln(k + 1));
    let den = a - 2.0 * b + c;
    let delta = if den.abs() > 1e-300 { (0.5 * (a - c) / den).clamp(-0.5, 0.5) } else { 0.0 };
    Ok((k as f64 + delta) * bin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cyclic_autocorrelation_of_a_tone_of_power() {
        // |x|² = 1 + cos(2π f0 n): the line of amplitude 1/2 at α = f0
        let f0 = 0.1;
        let x: Vec<C64> = (0..10_000).map(|n| C64::new((1.0 + (2.0 * PI * f0 * n as f64).cos()).sqrt(), 0.0)).collect();
        assert!((cyclic_autocorrelation(&x, f0, 0, 1.0).norm() - 0.5).abs() < 1e-3);
        assert!(cyclic_autocorrelation(&x, 0.137, 0, 1.0).norm() < 1e-3);
        assert!(estimate_symbol_rate(&x[..8], 1.0, 0.05, 0.2).is_err());
    }
}
