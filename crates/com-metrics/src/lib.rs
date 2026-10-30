//! # com-metrics — performance metrics (OptiCommPy `optic.comm.metrics`)
//!
//! Error counts and rates, SNR, EVM and information rates estimated from received and
//! transmitted symbols.
//!
//! | Module | Content |
//! |---|---|
//! | [`blind`] | estimates from the received samples alone: fourth-order cumulants, modulation classification by cumulants and by likelihood, the M2M4 SNR |
//! | [`errors`] | error counts and rates ([`bit_errors`], [`symbol_errors`], [`block_errors`], [`ber`]), SNR ([`snr_db`], [`effective_snr_db`]), [`ls_gain`], [`fast_ber_calc`], [`evm`], [`ber_to_q_db`] |
//! | [`information`] | [`entropy`], [`monte_carlo_gmi`], [`monte_carlo_mi`], [`information_rates`], [`posterior_rates`] |
//! | [`spectrum`] | PAPR and its CCDF, the adjacent channel leakage ratio, Welch's power spectral density (re-exported from `com_dsp::spectrum`) |
//! | [`stability`] | the frequency stability of an oscillator from its phase: Allan, modified Allan and Hadamard variances (IEEE Std 1139) |
//!
//! The items of [`errors`] and [`information`] are re-exported at the root.
//!
//! # Estimates from received and transmitted symbols
//!
//! [`fast_ber_calc`], [`evm`], [`monte_carlo_gmi`] and [`monte_carlo_mi`] follow the
//! pre-processing of OptiCommPy, so that their values can be compared with it:
//!
//! 1. for QAM and PSK, the received sequence is rotated by $\operatorname{mean}(x/z)$,
//!    which removes a constant phase (and gain) ambiguity;
//! 2. both sequences are normalised to unit mean power;
//! 3. the noise is $z - x$.
//!
//! The decisions use the normalised constellation of the family. The mutual
//! information estimates assume a Gaussian channel of the measured noise variance:
//!
//! - GMI (bit-wise, the achievable rate of a binary decoder after a bit-interleaved
//!   demapper): $\mathrm{GMI} = \sum_{k} \left[H/b - E \log_{2}(1 + e^{(2b_{k}-1) L_{k}})\right]$,
//!   with $L_{k}$ the exact LLRs (clipped at $\pm 500$) and $H$ the source entropy;
//! - MI (symbol-wise): $\mathrm{MI} = H(X) - E\left[-\log_{2} p(x \mid y)\right]$,
//!   $p(x \mid y) \propto p(x) e^{-|y-x|^{2}/\sigma^{2}}$.
//!
//! The rate estimators take the prior probabilities `px` of the points: uniform
//! symbols with `None`, a shaped source (probabilistic constellation shaping)
//! otherwise; the constellation is then normalised to unit energy under `px` and
//! $H(X)$ is the entropy of `px` ([`entropy`]).
//!
//! # Theory over AWGN
//!
//! [`theory_ber`], [`theory_ser`], [`theory_mi`] and [`theory_gmi`] are the closed forms
//! of `com_mapping::theory`, re-exported here.
//!
//! # Examples
//! ```
//! use com_metrics as metrics;
//! use com_mapping::constellation::{Constellation, ConstellationKind};
//! use com_core::rng::Rng;
//! use com_math::{db2lin, power};
//! let c = Constellation::qam(16).unwrap().normalized();
//! let mut rng = Rng::new(2);
//! let (_, tx) = com_mapping::source::symbols(&c, 50_000, None, &mut rng).unwrap();
//! // AWGN at Eb/N0 = 8 dB, 4 bits per symbol
//! let noise = rng.complex_noise(tx.len(), power(&tx) / (4.0 * db2lin(8.0)));
//! let rx: Vec<_> = tx.iter().zip(&noise).map(|(x, n)| x + n).collect();
//! let r = metrics::fast_ber_calc(&rx, &tx, &c).unwrap();
//! let theory = metrics::theory_ber(ConstellationKind::Qam, 16, 8.0).unwrap();
//! assert!((r.ber / theory - 1.0).abs() < 0.15);
//! // Es/N0 = Eb/N0 + 10 log10(b); the estimate normalises the noisy signal to unit power,
//! // which biases it up by about 10 log10(1 + N0/Es) (0.17 dB here), as in OptiCommPy
//! assert!((r.snr_db - (8.0 + com_math::lin2db(4.0))).abs() < 0.25);
//! let (gmi, ngmi) = metrics::monte_carlo_gmi(&rx, &tx, &c, None).unwrap();
//! assert!(gmi > 3.8 && gmi < 4.0 && (ngmi - gmi / 4.0).abs() < 1e-12);
//! ```

pub mod blind;
pub mod errors;
pub mod information;
pub mod spectrum;
pub mod stability;

use com_core::{Error, Result, ensure, same_len};
use com_dsp::signal::pnorm;
use com_mapping::constellation::ConstellationKind;
use com_math::C64;

pub use com_mapping::theory::{ber as theory_ber, gmi as theory_gmi, mi as theory_mi, ser as theory_ser};
pub use com_math::entropy;
pub use errors::{
    BerSerSnr, ber, ber_to_q_db, bit_errors, block_errors, effective_snr_db, evm, fast_ber_calc, ls_gain, snr_db,
    symbol_errors,
};
pub use information::{
    InformationRates, PosteriorRates, information_rates, monte_carlo_gmi, monte_carlo_mi, posterior_rates,
};

/// Rotate (QAM, PSK) and normalise as OptiCommPy does before its estimates.
fn preprocess(rx: &[C64], tx: &[C64], kind: ConstellationKind) -> Result<(Vec<C64>, Vec<C64>)> {
    same_len(rx.len(), tx.len(), "received and transmitted symbols")?;
    ensure(!rx.is_empty(), || "empty symbol sequences".into())?;
    let mut r = rx.to_vec();
    if matches!(kind, ConstellationKind::Qam | ConstellationKind::Psk) {
        let rot = tx.iter().zip(rx).map(|(t, x)| t / x).sum::<C64>() / rx.len() as f64;
        if !rot.is_finite() {
            return Err(Error::Numerical("phase ambiguity: a received symbol is zero".into()));
        }
        for v in &mut r {
            *v *= rot;
        }
    }
    Ok((pnorm(&r), pnorm(tx)))
}

/// Variance $\operatorname{mean} |x - \operatorname{mean} x|^{2}$ of a complex sequence (`numpy.var`).
fn variance(x: &[C64]) -> f64 {
    let m = x.iter().sum::<C64>() / x.len() as f64;
    x.iter().map(|v| (v - m).norm_sqr()).sum::<f64>() / x.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use com_mapping::constellation::Constellation;

    #[test]
    fn noiseless_estimates() {
        let c = Constellation::qam(16).unwrap();
        let mut rng = Rng::new(3);
        let tx: Vec<C64> = (0..1000).map(|_| c.points()[rng.below(16)]).collect();
        let rx: Vec<C64> = tx.iter().map(|v| v * C64::new(0.0, 2.0)).collect(); // gain and 90°
        let r = fast_ber_calc(&rx, &tx, &c).unwrap();
        assert_eq!((r.ber, r.ser), (0.0, 0.0));
        assert!(evm(&rx, &c, Some(&tx)).unwrap() < 1e-20);
        let noisy: Vec<C64> = tx.iter().map(|v| v + rng.complex_gaussian() * 0.01).collect();
        let mi = monte_carlo_mi(&noisy, &tx, &c, None).unwrap();
        assert!((mi - 4.0).abs() < 1e-6);
    }
}
