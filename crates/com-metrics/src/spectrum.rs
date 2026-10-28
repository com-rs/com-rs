//! Envelope and spectral metrics: PAPR and its CCDF, power spectral density, adjacent
//! channel leakage.
//!
//! - [`papr_db`] (re-exported from `com-math`) and [`ccdf`], the empirical
//!   $\Pr(X > t)$ of any statistic;
//! - [`psd`]: Welch's estimate of the power spectral density (periodic Hann window, 50 %
//!   overlap), centred on the carrier and normalised so that it integrates to the
//!   average power;
//! - [`periodogram`]: the averaged periodogram of non-overlapping segments, without
//!   window (no leakage for segments aligned on the blocks of a cyclic signal);
//! - [`band_power`] and [`aclr_db`]: the adjacent channel leakage ratio, the power in
//!   the channel over the power in an adjacent channel of the same width — the measure
//!   of the spectral regrowth of a nonlinear amplifier (3GPP TS 36.104 defines the
//!   limits of LTE base stations, 45 dB).
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
//! use com_metrics::spectrum::{psd, band_power};
//! use com_core::rng::Rng;
//! let x = Rng::new(1).complex_noise(1 << 16, 2.0);            // white, power 2
//! let (f, s) = psd(&x, 1.0, 256)?;
//! assert!((band_power(&f, &s, -0.5, 0.5) - 2.0).abs() < 0.05);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. P. D. Welch, "The use of fast Fourier transform for the estimation of power
//!    spectra," *IEEE Trans. Audio Electroacoust.*, 15(2), 70–73, 1967.

use com_core::{Result, ensure};
use com_dsp::window::hann;
use com_math::C64;
use com_math::fft;

pub use com_math::papr_db;

/// The empirical complementary distribution $\Pr(X > t)$ of `values` at every
/// threshold.
pub fn ccdf(values: &[f64], thresholds: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len().max(1) as f64;
    thresholds.iter().map(|t| (v.len() - v.partition_point(|x| x <= t)) as f64 / n).collect()
}

/// Welch's power spectral density of `x` sampled at `fs`, with segments of `nfft`
/// samples: the frequencies (centred, increasing) and the density (power per Hz).
///
/// # Errors
/// [`com_core::Error::Config`] if `x` is shorter than one segment or `fs` is not
/// positive.
pub fn psd(x: &[C64], fs: f64, nfft: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure(nfft >= 2 && x.len() >= nfft, || format!("{} samples for segments of {nfft}", x.len()))?;
    ensure(fs > 0.0, || format!("the sampling rate must be positive, got {fs}"))?;
    let w = hann(nfft);
    let u: f64 = w.iter().map(|v| v * v).sum();
    let p = fft::plan(nfft);
    let mut acc = vec![0.0; nfft];
    let mut k = 0usize;
    let mut start = 0;
    let mut seg = vec![C64::new(0.0, 0.0); nfft];
    while start + nfft <= x.len() {
        for (s, (v, wi)) in seg.iter_mut().zip(x[start..start + nfft].iter().zip(&w)) {
            *s = v * wi;
        }
        p.forward(&mut seg);
        for (a, s) in acc.iter_mut().zip(&seg) {
            *a += s.norm_sqr();
        }
        k += 1;
        start += nfft / 2;
    }
    let scale = 1.0 / (k as f64 * fs * u);
    let half = nfft / 2;
    let freqs: Vec<f64> = (0..nfft).map(|i| (i as f64 - half as f64) * fs / nfft as f64).collect();
    let dens: Vec<f64> = (0..nfft).map(|i| acc[(i + nfft - half) % nfft] * scale).collect();
    Ok((freqs, dens))
}

/// The power of a density in the band $[f_{\mathrm{lo}}, f_{\mathrm{hi}})$.
pub fn band_power(freqs: &[f64], density: &[f64], f_lo: f64, f_hi: f64) -> f64 {
    let df = if freqs.len() > 1 { freqs[1] - freqs[0] } else { 0.0 };
    freqs.iter().zip(density).filter(|(f, _)| **f >= f_lo && **f < f_hi).map(|(_, s)| s * df).sum()
}

/// The averaged periodogram of non-overlapping segments of `nfft` samples, without
/// window: the frequencies (centred) and the density. Segments aligned on the blocks of
/// a cyclic signal (OFDM without prefix, `nfft` = the block length) have no leakage, so
/// the out-of-band density is the regrowth of a nonlinearity alone.
///
/// # Errors
/// As [`psd`].
pub fn periodogram(x: &[C64], fs: f64, nfft: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure(nfft >= 1 && x.len() >= nfft, || format!("{} samples for segments of {nfft}", x.len()))?;
    ensure(fs > 0.0, || format!("the sampling rate must be positive, got {fs}"))?;
    let p = fft::plan(nfft);
    let mut acc = vec![0.0; nfft];
    let mut k = 0usize;
    for seg in x.chunks_exact(nfft) {
        let mut v = seg.to_vec();
        p.forward(&mut v);
        for (a, s) in acc.iter_mut().zip(&v) {
            *a += s.norm_sqr();
        }
        k += 1;
    }
    let scale = 1.0 / (k as f64 * fs * nfft as f64);
    let half = nfft / 2;
    let freqs = (0..nfft).map(|i| (i as f64 - half as f64) * fs / nfft as f64).collect();
    let dens = (0..nfft).map(|i| acc[(i + nfft - half) % nfft] * scale).collect();
    Ok((freqs, dens))
}

/// The adjacent channel leakage ratio (dB) of a density: the power of the channel
/// $[-B/2, B/2)$ over the larger of the powers of the channels of width $B$ centred at
/// $\pm$`offset`.
///
/// # Errors
/// [`com_core::Error::Config`] if the adjacent channels leave the band of the density.
pub fn aclr_db(freqs: &[f64], density: &[f64], bandwidth: f64, offset: f64) -> Result<f64> {
    let top = freqs.last().copied().unwrap_or(0.0);
    ensure(
        offset + bandwidth / 2.0 <= top + (top - freqs.first().copied().unwrap_or(0.0)) / freqs.len().max(1) as f64,
        || "the adjacent channel lies outside the observed band".into(),
    )?;
    let main = band_power(freqs, density, -bandwidth / 2.0, bandwidth / 2.0);
    let lower = band_power(freqs, density, -offset - bandwidth / 2.0, -offset + bandwidth / 2.0);
    let upper = band_power(freqs, density, offset - bandwidth / 2.0, offset + bandwidth / 2.0);
    Ok(10.0 * (main / lower.max(upper).max(1e-300)).log10())
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use com_math::cis;

    #[test]
    fn ccdf_counts_exceedances() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(ccdf(&v, &[0.0, 2.0, 2.5, 4.0]), vec![1.0, 0.5, 0.5, 0.0]);
    }

    #[test]
    fn a_tone_lands_in_its_bin_and_a_band_limited_signal_has_a_high_aclr() {
        let x: Vec<C64> = (0..8192).map(|n| cis(2.0 * std::f64::consts::PI * 0.125 * n as f64)).collect();
        let (f, s) = psd(&x, 1.0, 512).unwrap();
        let peak = f[s.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap()];
        assert!((peak - 0.125).abs() < 1e-12);
        assert!((band_power(&f, &s, -0.5, 0.5) - 1.0).abs() < 1e-6);
        // white noise filtered to [-0.1, 0.1]
        let w = Rng::new(2).complex_noise(1 << 16, 1.0);
        let mut spec = fft::fft(&w);
        for (i, v) in spec.iter_mut().enumerate() {
            let fr = if i < w.len() / 2 { i as f64 } else { i as f64 - w.len() as f64 } / w.len() as f64;
            if fr.abs() > 0.1 {
                *v = C64::new(0.0, 0.0);
            }
        }
        let y = fft::ifft(&spec);
        let (f, s) = psd(&y, 1.0, 1024).unwrap();
        assert!(aclr_db(&f, &s, 0.2, 0.25).unwrap() > 40.0);
        let (f, s) = psd(&w, 1.0, 1024).unwrap();
        assert!(aclr_db(&f, &s, 0.2, 0.25).unwrap().abs() < 0.5);
        // aligned on the period of a cyclic signal, the periodogram has no leakage
        let tone: Vec<C64> = (0..4096).map(|n| cis(2.0 * std::f64::consts::PI * 3.0 * n as f64 / 64.0)).collect();
        let (f, s) = periodogram(&tone, 1.0, 64).unwrap();
        assert!(s.iter().zip(&f).all(|(v, fr)| (*fr - 3.0 / 64.0).abs() < 1e-12 || *v < 1e-25));
    }
}
