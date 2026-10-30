//! Blind estimates from the received samples alone: higher-order cumulants, automatic
//! modulation classification, the M2M4 estimator of the SNR and the Cramér–Rao bound
//! of SNR estimators.
//!
//! # Model
//!
//! The samples $r_{n} = x_{n} + w_{n}$ carry symbols $x_{n}$ of an unknown alphabet in
//! circular complex Gaussian noise of variance $N$, at one sample per symbol, with the
//! symbol timing and the carrier recovered (a constant phase rotation is allowed by the
//! cumulant features).
//!
//! **Cumulants.** With the moments $M_{pq} = \mathbb{E}[r^{p-q} (r^{*})^{q}]$ of a
//! zero-mean signal,
//!
//! $$
//! C_{20} = M_{20}, \quad C_{21} = M_{21}, \quad C_{40} = M_{40} - 3 M_{20}^{2}, \quad
//! C_{41} = M_{41} - 3 M_{20} M_{21}, \quad C_{42} = M_{42} - |M_{20}|^{2} - 2 M_{21}^{2}.
//! $$
//!
//! The cumulants of order four of Gaussian noise vanish, so that $C_{4k}$ is that of the
//! symbols whatever the SNR, while $C_{21} = S + N$. The normalised features
//! $\tilde{C}_{4k} = C_{4k}/C_{21}^{2}$ characterise the alphabet (Swami & Sadler 2000:
//! $-2$ for BPSK, $\tilde{C}_{40} = 1$ and $\tilde{C}_{42} = -1$ for QPSK, $-0.68$ for
//! 16-QAM); $|\tilde{C}_{40}|$ and $\tilde{C}_{42}$ do not depend on the carrier phase.
//! [`classify_cumulants`] picks the candidate whose theoretical features are nearest to
//! the estimated ones; with the noise variance known, $C_{21} - N$ replaces $C_{21}$.
//! With the noise unknown, $C_{21} = S + N$ shrinks the features by $(1 + 1/\mathrm{SNR})^{-2}$
//! (12 % at 10 dB: 16-QAM, 0.68, then looks like 64-QAM, 0.62): the classifier is then
//! for high SNR, and [`classify_ml_blind`] for the others. (Normalising by the power that
//! [`m2m4`] estimates under each candidate does not help: it makes $\tilde{C}_{42}$ equal
//! to the value of the candidate whatever the signal.)
//!
//! **Likelihood.** With the noise variance known and the phase recovered,
//! [`classify_ml`] picks the alphabet of largest average log-likelihood
//! $\sum_{n} \ln \frac{1}{M} \sum_{a} e^{-|r_{n} - a|^{2}/N}$ (Wei & Mendel 2000), the
//! optimal classifier, at a cost of $M$ exponentials per sample and candidate;
//! [`classify_ml_blind`] estimates the scale and the noise under every candidate with
//! [`m2m4`] first.
//!
//! **M2M4.** With $M_{2} = \mathbb{E}|r|^{2} = S + N$ and
//! $M_{4} = \mathbb{E}|r|^{4} = k_{a} S^{2} + 4 S N + 2 N^{2}$, $k_{a} = \mathbb{E}|a|^{4}/(\mathbb{E}|a|^{2})^{2}$
//! the kurtosis of the alphabet (1 for PSK), the signal power is
//!
//! $$
//! \hat{S} = \sqrt{\frac{2 M_{2}^{2} - M_{4}}{2 - k_{a}}}, \qquad \hat{N} = M_{2} - \hat{S}
//! $$
//!
//! ([`snr_m2m4`]; the eq. (37) of Pauluzzi & Beaulieu 2000 with the kurtosis 2 of
//! complex noise, and their eq. (39) for PSK, where $\hat{S} = \sqrt{2 M_{2}^{2} - M_{4}}$).
//! Its bias and normalised mean squared error tend to constants at high SNR (8-PSK on
//! 64 symbols: 5 % and 0.040, their Figs. 9–10), about 2.5 times the Cramér–Rao bound
//! [`snr_crb_nmse`] of the estimators that know the symbols.
//! For an alphabet of several amplitudes (QAM) the estimate degrades at high SNR: the
//! noise is then a small difference of moments that the sample kurtosis of the symbols
//! themselves makes fluctuate (16-QAM at 20 dB: errors beyond 20 % with 4000 samples,
//! and an infinite estimate when $\hat{S} \ge M_{2}$); for PSK, of constant modulus, it
//! does not.
//!
//! # Examples
//! ```
//! use com_metrics::blind::{classify_cumulants, Cumulants};
//! use com_mapping::constellation::Constellation;
//! let qpsk = Constellation::qam(4)?.normalized();
//! let t = Cumulants::of_constellation(&qpsk, None)?;
//! assert!((t.c40_normalized().norm() - 1.0).abs() < 1e-12 && (t.c42_normalized() + 1.0).abs() < 1e-12);
//! // noiseless QPSK symbols among BPSK, QPSK and 16-QAM
//! let r: Vec<_> = (0..400).map(|i| qpsk.points()[i % 4]).collect();
//! let candidates = [Constellation::psk(2)?.normalized(), qpsk, Constellation::qam(16)?.normalized()];
//! assert_eq!(classify_cumulants(&r, &candidates, None)?.index, 1);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. A. Swami and B. M. Sadler, "Hierarchical digital modulation classification using
//!    cumulants," *IEEE Trans. Commun.*, 48(3), 416–429, 2000.
//! 2. W. Wei and J. M. Mendel, "Maximum-likelihood classification for digital
//!    amplitude-phase modulations," *IEEE Trans. Commun.*, 48(2), 189–193, 2000.
//! 3. D. R. Pauluzzi and N. C. Beaulieu, "A comparison of SNR estimation techniques for
//!    the AWGN channel," *IEEE Trans. Commun.*, 48(10), 1681–1691, 2000.
//! 4. O. A. Dobre, A. Abdi, Y. Bar-Ness and W. Su, "Survey of automatic modulation
//!    classification techniques: classical approaches and new trends," *IET Commun.*,
//!    1(2), 137–156, 2007.

use crate::posterior::SymbolMetric;
use com_core::{Error, Result, ensure, ensure_shape};
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_math::stats::Moments;

/// The floor of the signal power $C_{21} - N$ that normalises the cumulants of
/// [`Cumulants::features`], so that an estimate at or below the noise stays finite.
const POWER_FLOOR: f64 = 1e-300;

/// The cumulants of order two and four of a zero-mean complex signal (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cumulants {
    /// $C_{20} = \mathbb{E}[r^{2}]$.
    pub c20: C64,
    /// $C_{21} = \mathbb{E}|r|^{2}$.
    pub c21: f64,
    /// $C_{40}$.
    pub c40: C64,
    /// $C_{41}$.
    pub c41: C64,
    /// $C_{42}$ (real).
    pub c42: f64,
}

impl Cumulants {
    /// The sample cumulants of `r` (the moments replaced by sample means).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for an empty signal.
    pub fn estimate(r: &[C64]) -> Result<Cumulants> {
        ensure_shape(!r.is_empty(), || "the cumulants need at least one sample".into())?;
        Ok(Cumulants::from_moments(Moments::sample(r)))
    }

    /// The cumulants of the points of `c` with the probabilities `px` (uniform if
    /// `None`): the theoretical features of the alphabet.
    ///
    /// # Errors
    /// Those of [`Constellation::priors`].
    pub fn of_constellation(c: &Constellation, px: Option<&[f64]>) -> Result<Cumulants> {
        let p = c.priors(px)?;
        Ok(Cumulants::from_moments(Moments::of_distribution(c.points(), &p)))
    }

    /// The cumulants of the moments (`com_math::stats::Moments`, the one accumulation
    /// of the sample, of the alphabet and of [`m2m4`]).
    fn from_moments(m: Moments) -> Cumulants {
        let Moments { m20, m21, m40, m41, m42 } = m;
        Cumulants {
            c20: m20,
            c21: m21,
            c40: m40 - m20 * m20 * 3.0,
            c41: m41 - m20 * m21 * 3.0,
            c42: m42 - m20.norm_sqr() - 2.0 * m21 * m21,
        }
    }

    /// $\tilde{C}_{40} = C_{40}/C_{21}^{2}$.
    pub fn c40_normalized(&self) -> C64 {
        self.c40 / (self.c21 * self.c21)
    }

    /// $\tilde{C}_{42} = C_{42}/C_{21}^{2}$.
    pub fn c42_normalized(&self) -> f64 {
        self.c42 / (self.c21 * self.c21)
    }

    /// The phase-invariant features $(|\tilde{C}_{40}|, \tilde{C}_{42})$, with the signal
    /// power $C_{21} - N$ when the noise variance `noise_var` is known.
    pub fn features(&self, noise_var: Option<f64>) -> (f64, f64) {
        let s = (self.c21 - noise_var.unwrap_or(0.0)).max(POWER_FLOOR);
        (self.c40.norm() / (s * s), self.c42 / (s * s))
    }
}

/// The decision of a classifier.
#[derive(Clone, Debug, PartialEq)]
pub struct Classification {
    /// The index of the decided candidate.
    pub index: usize,
    /// The score of every candidate: minus the distance of the features
    /// ([`classify_cumulants`]) or the mean log-likelihood per sample ([`classify_ml`]);
    /// the largest wins.
    pub scores: Vec<f64>,
}

/// A modulation classifier, the noise variance known or not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Classifier {
    /// [`classify_ml_blind`]: the likelihood, the scale and the noise estimated by M2M4
    /// under every candidate.
    MlBlind,
    /// [`classify_ml`]: the likelihood with the known noise variance.
    Ml {
        /// The noise variance.
        noise_var: f64,
    },
    /// [`classify_cumulants`]: the fourth-order cumulants, the noise variance
    /// subtracted when it is known.
    Cumulants {
        /// The noise variance, if known.
        noise_var: Option<f64>,
    },
}

impl Classifier {
    /// The classifier of a name: `ml` ([`Classifier::MlBlind`], which ignores
    /// `noise_var`), `ml_known` ([`Classifier::Ml`], which needs it) or `cumulants`
    /// (case-insensitive).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for another name, or `ml_known` without a noise
    /// variance.
    pub fn from_name(name: &str, noise_var: Option<f64>) -> Result<Classifier> {
        match name.to_ascii_lowercase().as_str() {
            "ml" => Ok(Classifier::MlBlind),
            "ml_known" => noise_var
                .map(|noise_var| Classifier::Ml { noise_var })
                .ok_or_else(|| Error::Config("ml_known needs the noise variance".into())),
            "cumulants" => Ok(Classifier::Cumulants { noise_var }),
            _ => Err(Error::Config(format!("unknown classifier {name:?}: ml, ml_known or cumulants"))),
        }
    }

    /// Classify `r` among `candidates`.
    ///
    /// # Errors
    /// As the function of the classifier.
    pub fn classify(&self, r: &[C64], candidates: &[Constellation]) -> Result<Classification> {
        match *self {
            Classifier::MlBlind => classify_ml_blind(r, candidates),
            Classifier::Ml { noise_var } => classify_ml(r, candidates, noise_var),
            Classifier::Cumulants { noise_var } => classify_cumulants(r, candidates, noise_var),
        }
    }
}

fn decide(scores: Vec<f64>) -> Classification {
    let index = scores.iter().enumerate().fold(0, |b, (i, s)| if *s > scores[b] { i } else { b });
    Classification { index, scores }
}

/// Classify `r` among `candidates` by the nearest fourth-order cumulant features
/// $(|\tilde{C}_{40}|, \tilde{C}_{42})$ (Swami & Sadler 2000), with the noise variance
/// `noise_var` subtracted from the power when it is known.
///
/// # Errors
/// [`com_core::Error::Shape`] for an empty signal, [`com_core::Error::Config`] for no
/// candidate.
pub fn classify_cumulants(r: &[C64], candidates: &[Constellation], noise_var: Option<f64>) -> Result<Classification> {
    ensure(!candidates.is_empty(), || "the classifier needs at least one candidate".into())?;
    let (a, b) = Cumulants::estimate(r)?.features(noise_var);
    let scores = candidates
        .iter()
        .map(|c| {
            let (ta, tb) = Cumulants::of_constellation(c, None)?.features(None);
            Ok(-((a - ta).powi(2) + (b - tb).powi(2)).sqrt())
        })
        .collect::<Result<Vec<f64>>>()?;
    Ok(decide(scores))
}

/// Classify `r` among `candidates` (each scaled to unit energy, the scale of the
/// symbols of `r`) by the largest mean log-likelihood in noise of variance `noise_var`
/// with equally likely points (Wei & Mendel 2000).
///
/// # Errors
/// [`com_core::Error::Shape`] for an empty signal, [`com_core::Error::Config`] for no
/// candidate or a non-positive noise variance.
pub fn classify_ml(r: &[C64], candidates: &[Constellation], noise_var: f64) -> Result<Classification> {
    ensure_shape(!r.is_empty(), || "the classifier needs at least one sample".into())?;
    ensure(!candidates.is_empty(), || "the classifier needs at least one candidate".into())?;
    ensure(noise_var > 0.0, || format!("the noise variance must be positive, got {noise_var}"))?;
    let scores = candidates
        .iter()
        .map(|c| {
            let ln_m = (c.order() as f64).ln();
            // the priors enter as -ln M after the sum
            let zeros = vec![0.0; c.order()];
            let mut metric = SymbolMetric::new(c.points(), &zeros, noise_var);
            let mut total = 0.0;
            for &y in r {
                total += metric.log_sum(y, 1.0, None) - ln_m;
            }
            total / r.len() as f64
        })
        .collect();
    Ok(decide(scores))
}

/// The signal power $\hat{S}$ and the noise power $\hat{N}$ of `r` by M2M4 for the alphabet
/// `c` (see the module).
///
/// # Errors
/// As [`snr_m2m4`].
pub fn m2m4(r: &[C64], c: &Constellation) -> Result<(f64, f64)> {
    ensure_shape(!r.is_empty(), || "the estimator needs at least one sample".into())?;
    let alphabet = Moments::of_distribution(c.points(), &c.priors(None)?);
    let ka = alphabet.m42 / (alphabet.m21 * alphabet.m21);
    ensure(ka < 2.0, || format!("M2M4 needs an alphabet of kurtosis below 2, got {ka}"))?;
    let Moments { m21: m2, m42: m4, .. } = Moments::sample(r);
    let s = ((2.0 * m2 * m2 - m4) / (2.0 - ka)).max(0.0).sqrt();
    Ok((s, m2 - s))
}

/// Classify `r` among `candidates` by the likelihood with the scale of the signal and
/// the noise variance unknown: under every candidate, $\hat{S}$ comes from [`m2m4`] and
/// $\hat{N}$ is the larger of the M2M4 estimate and of the mean squared distance to the
/// nearest points (accurate at high SNR, where M2M4 fails for QAM), and the score is the mean
/// log-likelihood $\frac{1}{L}\sum_{n} \ln \frac{1}{M} \sum_{a} \frac{1}{\pi \hat{N}}
/// e^{-|r_{n} - \sqrt{\hat{S}}\, a|^{2}/\hat{N}}$, with the points of unit energy, so
/// that the scores of the candidates compare.
///
/// # Errors
/// [`com_core::Error::Shape`] for an empty signal, [`com_core::Error::Config`] for no
/// candidate.
pub fn classify_ml_blind(r: &[C64], candidates: &[Constellation]) -> Result<Classification> {
    ensure_shape(!r.is_empty(), || "the classifier needs at least one sample".into())?;
    ensure(!candidates.is_empty(), || "the classifier needs at least one candidate".into())?;
    let scores = candidates
        .iter()
        .map(|c| {
            let unit = c.normalized();
            let (s, n) = m2m4(r, &unit)?;
            if s <= 0.0 {
                return Ok(f64::NEG_INFINITY);
            }
            let scaled: Vec<C64> = r.iter().map(|v| v / s.sqrt()).collect();
            // the noise: the larger of M2M4 and of the residual to the nearest points,
            // which stays accurate at high SNR where M2M4 fails for several amplitudes
            let residual = scaled.iter().map(|v| (v - unit.points()[unit.nearest(*v)]).norm_sqr()).sum::<f64>()
                / scaled.len() as f64;
            let noise = n.max(residual * s).max(1e-9 * s);
            let ll = classify_ml(&scaled, std::slice::from_ref(&unit), noise / s)?.scores[0];
            // back to the density of r: the Gaussian normalisation and the scale
            Ok(ll - (std::f64::consts::PI * noise / s).ln() - s.ln())
        })
        .collect::<Result<Vec<f64>>>()?;
    Ok(decide(scores))
}

/// The M2M4 estimate of the SNR $\hat{S}/\hat{N}$ (linear) of `r` for symbols of the
/// alphabet `c` (see the module); 0 when the moments give no positive signal power,
/// infinite when they give no noise.
///
/// # Errors
/// [`com_core::Error::Shape`] for an empty signal, [`com_core::Error::Config`] for an
/// alphabet of kurtosis 2 or more.
pub fn snr_m2m4(r: &[C64], c: &Constellation) -> Result<f64> {
    let (s, noise) = m2m4(r, c)?;
    Ok(if noise <= 0.0 { f64::INFINITY } else { s / noise })
}

/// The Cramér–Rao bound on the normalised mean squared error
/// $\mathbb{E}[(\hat{\rho} - \rho)^{2}]/\rho^{2}$ of an unbiased estimator of the SNR
/// $\rho$ (`snr_lin`, linear) from $N_{\mathrm{sym}}$ known PSK symbols (`n_symbols`) and
/// $N_{\mathrm{ss}}$ independent samples per symbol (`sps`) in complex AWGN
/// (Pauluzzi & Beaulieu 2000, eq. (65)), or twice it in real AWGN (`real`, their eq. (67)):
///
/// $$
/// \mathrm{NMSE}(\hat{\rho}) \ge \frac{2}{\rho N_{\mathrm{sym}}} +
/// \frac{1}{N_{\mathrm{ss}} N_{\mathrm{sym}}} .
/// $$
///
/// It tends to $1/(N_{\mathrm{ss}} N_{\mathrm{sym}})$ at high SNR: the noise power must
/// still be estimated.
///
/// # Examples
/// ```
/// use com_metrics::blind::snr_crb_nmse;
/// // 64 symbols at 3 dB: the lowest point of the bound of Pauluzzi & Beaulieu, Fig. 9
/// let crb = snr_crb_nmse(10f64.powf(0.3), 64, 1, false)?;
/// assert!((crb - 0.0313).abs() < 1e-4);
/// assert_eq!(snr_crb_nmse(1.0, 64, 1, true)?, 2.0 * snr_crb_nmse(1.0, 64, 1, false)?);
/// # Ok::<(), com_core::Error>(())
/// ```
///
/// # Errors
/// [`com_core::Error::Config`] for no symbol or no sample per symbol.
pub fn snr_crb_nmse(snr_lin: f64, n_symbols: usize, sps: usize, real: bool) -> Result<f64> {
    ensure(n_symbols > 0 && sps > 0, || format!("n_symbols and sps must be positive, got {n_symbols} and {sps}"))?;
    let n = n_symbols as f64;
    let complex = 2.0 / (snr_lin * n) + 1.0 / (sps as f64 * n);
    Ok(if real { 2.0 * complex } else { complex })
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn the_bound_needs_symbols_and_samples() {
        assert!(snr_crb_nmse(1.0, 0, 1, false).is_err() && snr_crb_nmse(1.0, 10, 0, false).is_err());
    }

    #[test]
    fn the_classifiers_are_read_by_name_and_call_their_function() {
        let cands = [Constellation::psk(2).unwrap(), Constellation::qam(16).unwrap().normalized()];
        let mut rng = Rng::new(4);
        let r: Vec<C64> = (0..500).map(|_| cands[1].points()[rng.below(16)] + rng.complex_gaussian() * 0.1).collect();
        let by = |name: &str, nv: Option<f64>| Classifier::from_name(name, nv).unwrap().classify(&r, &cands).unwrap();
        assert_eq!(by("ML", None), classify_ml_blind(&r, &cands).unwrap());
        assert_eq!(by("ml_known", Some(0.01)), classify_ml(&r, &cands, 0.01).unwrap());
        assert_eq!(by("cumulants", Some(0.01)), classify_cumulants(&r, &cands, Some(0.01)).unwrap());
        assert!(Classifier::from_name("ml_known", None).is_err() && Classifier::from_name("svm", None).is_err());
    }

    #[test]
    fn theoretical_cumulants_of_common_alphabets() {
        // normalised C40 and C42 (Swami & Sadler 2000, in magnitude for C40)
        let cases: [(Constellation, f64, f64); 5] = [
            (Constellation::psk(2).unwrap(), 2.0, -2.0),
            (Constellation::qam(4).unwrap(), 1.0, -1.0),
            (Constellation::psk(8).unwrap(), 0.0, -1.0),
            (Constellation::pam(4).unwrap(), 1.36, -1.36),
            (Constellation::qam(16).unwrap(), 0.68, -0.68),
        ];
        for (c, c40, c42) in cases {
            let (a, b) = Cumulants::of_constellation(&c, None).unwrap().features(None);
            assert!((a - c40).abs() < 1e-9 && (b - c42).abs() < 1e-9, "{:?} {}: {a} {b}", c.kind(), c.order());
        }
    }

    #[test]
    fn the_estimators_on_noisy_symbols() {
        let c = Constellation::qam(16).unwrap().normalized();
        let mut rng = Rng::new(1);
        let s2: f64 = 0.05;
        let r: Vec<C64> =
            (0..200_000).map(|_| c.points()[rng.below(16)] + rng.complex_gaussian() * s2.sqrt()).collect();
        // the fourth-order cumulants ignore the Gaussian noise
        let (a, b) = Cumulants::estimate(&r).unwrap().features(Some(s2));
        assert!((a - 0.68).abs() < 0.02 && (b + 0.68).abs() < 0.02, "{a} {b}");
        let snr = snr_m2m4(&r, &c).unwrap();
        assert!((snr * s2 - 1.0).abs() < 0.02, "{snr}");
        let cands =
            [Constellation::qam(4).unwrap().normalized(), c.clone(), Constellation::qam(64).unwrap().normalized()];
        assert_eq!(classify_ml(&r[..2000], &cands, s2).unwrap().index, 1);
        assert_eq!(classify_cumulants(&r, &cands, Some(s2)).unwrap().index, 1);
        assert!(classify_ml(&r, &cands, 0.0).is_err() && snr_m2m4(&[], &c).is_err());
        // unknown scale and noise: 2.3 times the amplitude
        let scaled: Vec<C64> = r[..4000].iter().map(|v| v * 2.3).collect();
        assert_eq!(classify_ml_blind(&scaled, &cands).unwrap().index, 1);
        let qpsk: Vec<C64> = noisy_qpsk(&mut rng);
        assert_eq!(classify_ml_blind(&qpsk, &cands).unwrap().index, 0);
    }

    fn noisy_qpsk(rng: &mut Rng) -> Vec<C64> {
        let c = Constellation::qam(4).unwrap().normalized();
        (0..2000).map(|_| (c.points()[rng.below(4)] + rng.complex_gaussian() * 0.2) * 0.7).collect::<Vec<_>>()
    }
}
