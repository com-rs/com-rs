//! Envelope and spectral metrics: PAPR and its CCDF, power spectral density, adjacent
//! channel leakage.
//!
//! - [`papr_db`] (re-exported from `com-math`) and [`ccdf`], the empirical
//!   $\Pr(X > t)$ of any statistic;
//! - [`psd`] and [`periodogram`] (re-exported from `com-dsp`): Welch's estimate of the
//!   power spectral density and the averaged periodogram without window;
//! - [`band_power`] and [`aclr_db`]: the adjacent channel leakage ratio, the power in
//!   the channel over the power in an adjacent channel of the same width — the measure
//!   of the spectral regrowth of a nonlinear amplifier (3GPP TS 36.104 defines the
//!   limits of LTE base stations, 45 dB).
//!
//! # Examples
//! ```
//! use com_metrics::spectrum::{psd, band_power};
//! use com_core::rng::Rng;
//! let x = Rng::new(1).complex_noise(1 << 16, 2.0); // white, power 2
//! let (f, s) = psd(&x, 1.0, 256)?;
//! assert!((band_power(&f, &s, -0.5, 0.5) - 2.0).abs() < 0.05);
//! # Ok::<(), com_core::Error>(())
//! ```

use com_core::{Result, ensure};

pub use com_dsp::spectrum::{periodogram, psd};
pub use com_math::papr_db;

/// The empirical complementary distribution $\Pr(X > t)$ of `values` at every
/// threshold.
pub fn ccdf(values: &[f64], thresholds: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len().max(1) as f64;
    thresholds.iter().map(|t| (v.len() - v.partition_point(|x| x <= t)) as f64 / n).collect()
}

/// The power of a density in the band $[f_{\mathrm{lo}}, f_{\mathrm{hi}})$.
pub fn band_power(freqs: &[f64], density: &[f64], f_lo: f64, f_hi: f64) -> f64 {
    let df = if freqs.len() > 1 { freqs[1] - freqs[0] } else { 0.0 };
    freqs.iter().zip(density).filter(|(f, _)| **f >= f_lo && **f < f_hi).map(|(_, s)| s * df).sum()
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
    Ok(com_math::lin2db(main / lower.max(upper).max(1e-300)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use com_math::{C64, cis, fft};

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
    }
}
