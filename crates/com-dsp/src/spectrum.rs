//! Spectral estimation: averaged periodograms.
//!
//! - [`welch`]: the averaged periodogram of windowed segments every `hop` samples;
//! - [`psd`]: Welch's estimate of the power spectral density (periodic Hann window, 50 %
//!   overlap), centred on the carrier and normalised so that it integrates to the
//!   average power;
//! - [`periodogram`]: the averaged periodogram of non-overlapping segments, without
//!   window (no leakage for segments aligned on the blocks of a cyclic signal).
//!
//! The metrics computed from a density (band power, adjacent channel leakage) are in
//! `com-metrics`.
//!
//! # Model
//!
//! With $K$ segments $x_{i}$ of length $N_{\mathrm{f}}$ and the window $w$,
//!
//! $$
//! \hat S(f_{k}) = \frac{1}{K f_{s} \sum_{n} w\[n\]^{2}} \sum_{i} \Bigl| \sum_{n} w\[n\]\\, x_{i}\[n\]\\,
//! e^{-j 2\pi k n / N_{\mathrm{f}}} \Bigr|^{2}
//! $$
//!
//! # Examples
//! ```
//! use com_dsp::spectrum::psd;
//! use com_core::rng::Rng;
//! let x = Rng::new(1).complex_noise(1 << 16, 2.0); // white, power 2
//! let (f, s) = psd(&x, 1.0, 256)?;
//! let power: f64 = s.iter().sum::<f64>() * (f[1] - f[0]);
//! assert!((power - 2.0).abs() < 0.05);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. P. D. Welch, "The use of fast Fourier transform for the estimation of power
//!    spectra," *IEEE Trans. Audio Electroacoust.*, 15(2), 70–73, 1967.

use crate::window::hann;
use com_core::{Result, ensure};
use com_math::C64;
use com_math::fft;

/// Welch's power spectral density of `x` sampled at `fs`, with segments of `nfft`
/// samples: the frequencies (centred, increasing) and the density (power per Hz).
///
/// # Errors
/// [`com_core::Error::Config`] if `x` is shorter than one segment or `fs` is not
/// positive.
pub fn psd(x: &[C64], fs: f64, nfft: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    welch(x, fs, nfft, Some(&hann(nfft)), nfft / 2)
}

/// The averaged periodogram of non-overlapping segments of `nfft` samples, without
/// window: the frequencies (centred) and the density. Segments aligned on the blocks of
/// a cyclic signal (OFDM without prefix, `nfft` = the block length) have no leakage, so
/// the out-of-band density is the regrowth of a nonlinearity alone.
///
/// # Errors
/// As [`psd`].
pub fn periodogram(x: &[C64], fs: f64, nfft: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    welch(x, fs, nfft, None, nfft)
}

/// The averaged periodogram of the segments of `nfft` samples every `hop` samples,
/// weighted by `window` (rectangular if `None`): the centred frequencies and the density
/// normalised by the energy of the window (see the module).
///
/// # Errors
/// [`com_core::Error::Config`] if `x` is shorter than one segment, `fs` is not positive,
/// `hop` is zero or the window does not have `nfft` taps.
pub fn welch(x: &[C64], fs: f64, nfft: usize, window: Option<&[f64]>, hop: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure(nfft >= 1 && x.len() >= nfft, || format!("{} samples for segments of {nfft}", x.len()))?;
    ensure(fs > 0.0, || format!("the sampling rate must be positive, got {fs}"))?;
    ensure(hop >= 1, || "the hop between segments must be at least 1".into())?;
    ensure(window.is_none_or(|w| w.len() == nfft), || format!("the window needs {nfft} taps"))?;
    let u: f64 = window.map_or(nfft as f64, |w| w.iter().map(|v| v * v).sum());
    let p = fft::plan(nfft);
    let mut acc = vec![0.0; nfft];
    let mut k = 0usize;
    let mut start = 0;
    let mut seg = vec![C64::new(0.0, 0.0); nfft];
    while start + nfft <= x.len() {
        seg.copy_from_slice(&x[start..start + nfft]);
        if let Some(w) = window {
            for (s, wi) in seg.iter_mut().zip(w) {
                *s *= wi;
            }
        }
        p.forward(&mut seg);
        for (a, s) in acc.iter_mut().zip(&seg) {
            *a += s.norm_sqr();
        }
        k += 1;
        start += hop;
    }
    let scale = 1.0 / (k as f64 * fs * u);
    let half = nfft / 2;
    let freqs = (0..nfft).map(|i| (i as f64 - half as f64) * fs / nfft as f64).collect();
    let dens = fft::fftshift(&acc).into_iter().map(|a| a * scale).collect();
    Ok((freqs, dens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_math::cis;

    #[test]
    fn a_tone_lands_in_its_bin_and_aligned_periodograms_do_not_leak() {
        let x: Vec<C64> = (0..8192).map(|n| cis(2.0 * std::f64::consts::PI * 0.125 * n as f64)).collect();
        let (f, s) = psd(&x, 1.0, 512).unwrap();
        let peak = f[s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap()];
        assert!((peak - 0.125).abs() < 1e-12);
        assert!((s.iter().sum::<f64>() * (f[1] - f[0]) - 1.0).abs() < 1e-6);
        // aligned on the period of a cyclic signal, the periodogram has no leakage
        let tone: Vec<C64> = (0..4096).map(|n| cis(2.0 * std::f64::consts::PI * 3.0 * n as f64 / 64.0)).collect();
        let (f, s) = periodogram(&tone, 1.0, 64).unwrap();
        assert!(s.iter().zip(&f).all(|(v, fr)| (*fr - 3.0 / 64.0).abs() < 1e-12 || *v < 1e-25));
        assert!(welch(&tone, 1.0, 64, Some(&[1.0; 3]), 64).is_err() && welch(&tone, 1.0, 64, None, 0).is_err());
    }
}
