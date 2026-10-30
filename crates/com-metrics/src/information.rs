//! Information rates estimated from received and transmitted symbols: the generalised
//! and symbol-wise mutual information of a Gaussian channel of the measured noise
//! (OptiCommPy's estimators and consistent ones with the least-squares gain), and the
//! achievable rates of a receiver with its own posterior.
//!
//! # Model
//!
//! With the priors $p(x)$ of the points (uniform, or a shaped source) and the entropy
//! $H(X) = -\sum_{x} p(x) \log_{2} p(x)$ ([`entropy`]), the symbol-wise and the
//! bit-wise rates are
//!
//! $$
//! \mathrm{MI} = H(X) - E\left[-\log_{2} p(x \mid y)\right], \qquad
//! \mathrm{GMI} = H(X) - \sum_{k} E\left[\log_{2}\left(1 + e^{(2 b_{k} - 1) L_{k}}\right)\right],
//! $$
//!
//! with $p(x \mid y) \propto p(x) e^{-|y - x|^{2}/\sigma^{2}}$ and $L_{k}$ the exact LLR of
//! bit $k$ (Böcherer 2014).
//!
//! # References
//!
//! 1. G. Böcherer, "Achievable rates for shaped bit-metric decoding," arXiv:1410.8075,
//!    2014.
//! 2. G. Kaplan and S. Shamai, "Information rates and error exponents of compound
//!    channels with application to antipodal signaling in a fading environment," *AEÜ*,
//!    47(4), 228–239, 1993.

use crate::posterior::SymbolMetric;
use crate::{preprocess, variance};
use com_core::{Result, ensure, same_len};
use com_mapping::constellation::{Constellation, log_priors};
use com_mapping::demapper::LabelHalves;
use com_math::{C64, entropy, golden_max, ln_1p_exp};

/// The priors of the points of `c` (uniform if `px` is `None`), checked and normalised
/// to a sum of 1, and the constellation scaled to unit energy under them.
fn shaped(c: &Constellation, px: Option<&[f64]>) -> Result<(Constellation, Vec<f64>)> {
    let p = c.priors(px)?;
    Ok((c.normalized_under(&p)?, p))
}

/// The GMI (bits/symbol) of the bit LLRs of symbols of `b` bits whose transmitted bits
/// are `bits`, for a source of entropy `h`: $H - \sum_{k} \operatorname{mean} \ln(1 + e^{\pm L_{k}}) / \ln 2$.
fn gmi_from_llr(llr: &[f64], bits: &[u8], b: usize, h: f64) -> f64 {
    let mut gmi = 0.0;
    for k in 0..b {
        let mut acc = 0.0;
        let mut n = 0usize;
        for i in (k..llr.len()).step_by(b) {
            let l = llr[i].clamp(-500.0, 500.0);
            let s = 2.0 * f64::from(bits[i]) - 1.0;
            acc += ln_1p_exp(s * l) / std::f64::consts::LN_2;
            n += 1;
        }
        gmi += h / b as f64 - acc / n as f64;
    }
    gmi
}

/// The generalized mutual information of [`monte_carlo_gmi`] and its value per bit of
/// information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gmi {
    /// Generalized mutual information (bits/symbol).
    pub gmi: f64,
    /// $\mathrm{GMI}/H(X)$.
    pub ngmi: f64,
}

/// Generalized mutual information (bits/symbol) and its normalised value $\mathrm{GMI}/H$
/// (OptiCommPy `monteCarloGMI`, Gray labels): with the prior probabilities `px` of the
/// points (uniform if `None`, probabilistic shaping otherwise), the constellation is
/// normalised to unit energy under `px`, the LLRs include the priors and `H` is the
/// entropy of `px`.
///
/// # Errors
/// As [`fast_ber_calc`](crate::fast_ber_calc); [`com_core::Error::Shape`] or [`com_core::Error::Config`] for priors that are not
/// one non-negative value per point.
pub fn monte_carlo_gmi(rx: &[C64], tx: &[C64], c: &Constellation, px: Option<&[f64]>) -> Result<Gmi> {
    let (r, t) = preprocess(rx, tx, c.kind())?;
    let e: Vec<C64> = r.iter().zip(&t).map(|(a, b)| a - b).collect();
    // floored so that a noiseless input gives log2 M instead of 0/0
    let mut sigma2 = variance(&e).max(f64::MIN_POSITIVE);
    if c.kind().is_real() {
        sigma2 *= 2.0;
    }
    let (cn, p) = shaped(c, px)?;
    let b = cn.bits_per_symbol();
    let h = entropy(&p);
    let btx = cn.demodulate(&t);
    let llr = cn.llr(&r, sigma2, Some(&p), false)?;
    let gmi = gmi_from_llr(&llr, &btx, b, h);
    Ok(Gmi { gmi, ngmi: gmi / h })
}

/// Achievable information rates of a received sequence, see [`information_rates`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InformationRates {
    /// Symbol-wise mutual information (bits/symbol).
    pub mi: f64,
    /// Generalised mutual information of a bit-wise decoder (bits/symbol).
    pub gmi: f64,
    /// GMI per bit of information, $\mathrm{GMI}/H(X)$.
    pub ngmi: f64,
    /// Entropy of the source $H(X)$ (bits/symbol; $\log_{2} M$ for uniform symbols).
    pub entropy: f64,
    /// Effective SNR $|a|^{2} \sum |x|^{2} / \sum |y - a x|^{2}$ (dB).
    pub snr_db: f64,
}

/// Mutual information and GMI of a received sequence with the least-squares channel
/// gain: consistent estimators of the rates of an AWGN channel at every SNR.
///
/// The model is $y = a x + w$, $w \sim \mathcal{CN}(0, \sigma^{2})$, with $x$ the
/// transmitted points of the constellation (given by their `indices`) scaled to unit
/// energy under the priors `px` (uniform if `None`; the probabilities of a shaped
/// source otherwise),
///
/// $$
/// a = \frac{\sum_{k} y_{k} x_{k}^{*}}{\sum_{k} |x_{k}|^{2}}, \qquad
/// \sigma^{2} = \operatorname{mean} |y - a x|^{2};
/// $$
///
/// the rates are those of the decoder that uses this model and of the priors (the
/// exact LLRs of [`Constellation::llr`] for the GMI, with the entropy $H(X)$ of the
/// source: $\mathrm{GMI} = H(X) - \sum_{k} E \log_{2}(1 + e^{(2b_{k} - 1) L_{k}})$,
/// Böcherer 2014), so they converge to
/// [`theory_mi`](crate::theory_mi) and [`theory_gmi`](crate::theory_gmi) over AWGN.
///
/// [`monte_carlo_mi`] and [`monte_carlo_gmi`] follow OptiCommPy instead: they normalise
/// the power of the noisy sequence, which shrinks the signal by $1/\sqrt{1 + 1/\mathrm{SNR}}$ and
/// underestimates the rates at low SNR (by 0.06 bit for 16-QAM at 0 dB, 0.04 bit at
/// 5 dB; `specs/validation/information_rates.yaml`).
///
/// # Errors
/// [`com_core::Error::Shape`] for sequences of different lengths, [`com_core::Error::Config`] for an index
/// outside the constellation or an empty sequence.
///
/// # Examples
/// ```
/// use com_metrics as metrics;
/// use com_mapping::constellation::{Constellation, ConstellationKind};
/// use com_core::rng::Rng;
/// use com_math::{db2lin, power};
/// let c = Constellation::qam(16).unwrap().normalized();
/// let mut rng = Rng::new(1);
/// let (idx, x) = com_mapping::source::symbols(&c, 40_000, None, &mut rng).unwrap();
/// // AWGN at an SNR of 0 dB
/// let n = rng.complex_noise(x.len(), power(&x) / db2lin(0.0));
/// let y: Vec<_> = x.iter().zip(&n).map(|(a, b)| a + b).collect();
/// let r = metrics::information_rates(&y, &idx, &c, None).unwrap();
/// let mi = metrics::theory_mi(ConstellationKind::Qam, 16, 0.0, None, 40).unwrap();
/// assert!((r.mi - mi).abs() < 0.01);
/// ```
pub fn information_rates(
    rx: &[C64],
    indices: &[usize],
    c: &Constellation,
    px: Option<&[f64]>,
) -> Result<InformationRates> {
    same_len(rx.len(), indices.len(), "received symbols and transmitted indices")?;
    ensure(!rx.is_empty(), || "empty symbol sequences".into())?;
    let (cn, p) = shaped(c, px)?;
    let lp = log_priors(&p);
    let m = cn.order();
    ensure(indices.iter().all(|&i| i < m), || format!("an index is outside the {m} points"))?;
    let x: Vec<C64> = indices.iter().map(|&i| cn.points()[i]).collect();
    let ex: f64 = x.iter().map(C64::norm_sqr).sum();
    let a = rx.iter().zip(&x).map(|(y, x)| y * x.conj()).sum::<C64>() / ex;
    let err: f64 = rx.iter().zip(&x).map(|(y, x)| (y - a * x).norm_sqr()).sum();
    ensure(a.norm() > 0.0, || "the received sequence is uncorrelated with the transmitted one".into())?;
    // floored so that a noiseless input gives log2 M instead of 0/0
    let sigma2 = (err / rx.len() as f64 / a.norm_sqr()).max(f64::MIN_POSITIVE);
    let y: Vec<C64> = rx.iter().map(|v| v / a).collect();
    let h = entropy(&p);
    // MI = H(X) - mean[-log2 p(x|y)], p(x|y) = p(x) e^{-|y-x|²/σ²} / Σ_j p_j e^{-|y-x_j|²/σ²}
    let mut metric = SymbolMetric::new(cn.points(), &lp, sigma2);
    let acc: f64 = y.iter().zip(indices).map(|(&yk, &i)| metric.log_sum(yk, 1.0, Some(i))).sum();
    let mi = h - acc / rx.len() as f64 / std::f64::consts::LN_2;
    let b = cn.bits_per_symbol();
    let bits = com_mapping::constellation::indices_to_bits(indices, b);
    let llr = cn.llr(&y, sigma2, Some(&p), false)?;
    let gmi = gmi_from_llr(&llr, &bits, b, h);
    let snr_db = com_math::lin2db(a.norm_sqr() * ex / err.max(f64::MIN_POSITIVE));
    Ok(InformationRates { mi, gmi, ngmi: gmi / h, snr_db, entropy: h })
}

/// Achievable rates of a receiver that declares its own posterior, see
/// [`posterior_rates`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosteriorRates {
    /// Symbol-wise rate with the receiver's posterior as it is (bits/symbol).
    pub air: f64,
    /// Bit-wise rate with the LLRs of the receiver's posterior (bits/symbol).
    pub air_bmd: f64,
    /// Symbol-wise rate with the best scaling $s$ of the posterior's exponent.
    pub air_opt: f64,
    /// Bit-wise rate with the best scaling of the LLRs.
    pub air_bmd_opt: f64,
    /// The best scaling $s$ of the symbol-wise posterior (1 for a calibrated receiver).
    pub scale: f64,
    /// The best scaling of the LLRs.
    pub scale_bmd: f64,
    /// Calibration of the declared variance, $\hat\sigma^{2} / \operatorname{mean} |z - x|^{2}$ (1 for a calibrated
    /// receiver, below 1 for an over-confident one); unlike the scales, it is defined
    /// on every block.
    pub calibration: f64,
}

/// Maximise a unimodal function of $s > 0$ over $[10^{-4}, 10^{4}]$ (golden section on
/// $\ln s$, 60 steps).
fn maximise_scale(f: impl Fn(f64) -> f64) -> (f64, f64) {
    let s = golden_max(|t| f(t.exp()), (1e-4f64).ln(), 1e4f64.ln(), 60).exp();
    // the receiver's own scale wins ties (a saturated rate leaves s undetermined): report
    // s = 1 unless another scale gains more than 1e-6 bit
    let (v, v1) = (f(s), f(1.0));
    if v - v1 <= 1e-6 { (1.0, v1) } else { (s, v) }
}

/// Achievable information rates of a receiver with its **own** posterior: the rates of
/// a decoder that uses the receiver's beliefs, not those of a genie.
///
/// The receiver delivers `z` and a noise variance $\hat\sigma^{2}$ it believes in (a
/// learned receiver estimates it with its parameters); its posterior is
/// $q(x \mid z) \propto p(x) e^{-|z - x|^{2}/\hat\sigma^{2}}$ on the constellation of
/// unit energy under the priors `px` of the points (uniform if `None`), the scale of
/// [`information_rates`] and [`monte_carlo_gmi`], without any gain correction. The rate
/// of a decoder that uses $q$ is (mismatched decoding)
///
/// $$
/// \mathrm{AIR} = H(X) - E\left[-\log_{2} q(x \mid z)\right],
/// $$
///
/// a lower bound on the mutual information, equal to it when $q$ is the true posterior.
/// The bit-wise rate `air_bmd` uses the LLRs of $q$. Scaling the exponent by $s$
/// ($q_{s} \propto p(x) e^{-s|z - x|^{2}/\hat\sigma^{2}}$) and maximising over $s$ gives
/// the generalised mutual information of the decoding metric (Kaplan & Shamai 1993); the
/// best $s$ tells how calibrated the posterior is ($s < 1$: over-confident, $s > 1$:
/// under-confident), and `air_opt - air` what recalibrating it would gain. The best
/// scale is a property of the block: on a block without decision errors the rate grows
/// with $s$ up to the search bound $10^{4}$ (or stays flat, and $s = 1$ is reported);
/// `calibration`, $\hat\sigma^{2} / \operatorname{mean} |z - x|^{2}$, measures the same
/// miscalibration on every block. Compared with
/// [`information_rates`] (the genie gain and variance), the gap measures what the
/// receiver loses by its own estimates.
///
/// # Errors
/// [`com_core::Error::Shape`] for sequences of different lengths, [`com_core::Error::Config`] for an index
/// outside the constellation, an empty sequence or a variance that is not positive.
///
/// # Examples
/// ```
/// use com_metrics as metrics;
/// use com_mapping::constellation::Constellation;
/// use com_core::rng::Rng;
/// use com_math::{db2lin, power};
/// let c = Constellation::qam(16).unwrap().normalized();
/// let mut rng = Rng::new(3);
/// let (idx, x) = com_mapping::source::symbols(&c, 20_000, None, &mut rng).unwrap();
/// // AWGN at an SNR of 12 dB
/// let n = rng.complex_noise(x.len(), power(&x) / db2lin(12.0));
/// let z: Vec<_> = x.iter().zip(&n).map(|(a, b)| a + b).collect();
/// let true_s2 = 10f64.powf(-1.2);
/// // a calibrated receiver: its rate is the information rate, with s = 1
/// let good = metrics::posterior_rates(&z, &idx, &c, true_s2, None).unwrap();
/// let genie = metrics::information_rates(&z, &idx, &c, None).unwrap();
/// assert!((good.air - genie.mi).abs() < 0.01 && (good.scale - 1.0).abs() < 0.1);
/// assert!((good.calibration - 1.0).abs() < 0.05);
/// // an over-confident receiver (σ̂² four times too small) loses rate, which s recovers
/// let over = metrics::posterior_rates(&z, &idx, &c, true_s2 / 4.0, None).unwrap();
/// assert!(over.air < good.air - 0.05 && (over.scale - 0.25).abs() < 0.05);
/// assert!((over.calibration - 0.25).abs() < 0.02);
/// assert!((over.air_opt - good.air).abs() < 0.01);
/// ```
pub fn posterior_rates(
    z: &[C64],
    indices: &[usize],
    c: &Constellation,
    noise_var: f64,
    px: Option<&[f64]>,
) -> Result<PosteriorRates> {
    same_len(z.len(), indices.len(), "received symbols and transmitted indices")?;
    ensure(!z.is_empty(), || "empty symbol sequences".into())?;
    ensure(noise_var > 0.0 && noise_var.is_finite(), || {
        format!("the receiver's noise variance must be positive, got {noise_var}")
    })?;
    let (cn, p) = shaped(c, px)?;
    let m = cn.order();
    ensure(indices.iter().all(|&i| i < m), || format!("an index is outside the {m} points"))?;
    let b = cn.bits_per_symbol();
    let n = z.len() as f64;
    let lp = log_priors(&p);
    let h = entropy(&p);
    // the distances |z_k - x_j|² / σ̂² of a symbol, recomputed at every scale rather
    // than stored for the whole sequence (N × M values: 2 GB for 10⁶ symbols of 256-QAM)
    let distances = |zk: C64, row: &mut [f64]| {
        for (r, p) in row.iter_mut().zip(cn.points()) {
            *r = (zk - p).norm_sqr() / noise_var;
        }
    };
    let air_at = |s: f64| -> f64 {
        let mut metric = SymbolMetric::new(cn.points(), &lp, noise_var);
        let acc: f64 = z.iter().zip(indices).map(|(&zk, &i)| metric.log_sum(zk, s, Some(i))).sum();
        h - acc / n / std::f64::consts::LN_2
    };
    let bits = com_mapping::constellation::indices_to_bits(indices, b);
    let bmd_at = |s: f64| -> f64 {
        let mut acc = 0.0;
        let (mut row, mut metric) = (vec![0.0; m], vec![0.0; m]);
        for (k, &zk) in z.iter().enumerate() {
            distances(zk, &mut row);
            for (l, (r, p)) in metric.iter_mut().zip(row.iter().zip(&lp)) {
                *l = p - s * r;
            }
            for bit in 0..b {
                let own = bits[k * b + bit] as usize;
                // -log2 P(b = own | z) = log2(1 + e^{L_other - L_own}), L the log-sums
                // of the halves of the labels (the kernel of the demapper)
                let halves = LabelHalves::new(&metric, b - 1 - bit, false);
                acc += ln_1p_exp(halves.log_sum(1 - own) - halves.log_sum(own));
            }
        }
        h - acc / n / std::f64::consts::LN_2
    };
    let air = air_at(1.0);
    let air_bmd = bmd_at(1.0);
    let (scale, air_opt) = maximise_scale(air_at);
    let (scale_bmd, air_bmd_opt) = maximise_scale(bmd_at);
    let mse = z.iter().zip(indices).map(|(zk, &i)| (zk - cn.points()[i]).norm_sqr()).sum::<f64>() / n;
    let calibration = noise_var / mse.max(f64::MIN_POSITIVE);
    Ok(PosteriorRates { air, air_bmd, air_opt, air_bmd_opt, scale, scale_bmd, calibration })
}

/// Symbol-wise mutual information (bits/symbol) of a Gaussian channel of the measured
/// noise variance (OptiCommPy `monteCarloMI`): with the priors `px` of the points
/// (uniform if `None`), the constellation normalised to unit energy under them and
/// $\mathrm{MI} = H(X) - E\left[-\log_{2} p(x \mid y)\right]$; the transmitted symbols
/// are snapped to the constellation, so that the estimate never exceeds $H(X)$.
///
/// # Errors
/// As [`fast_ber_calc`](crate::fast_ber_calc).
pub fn monte_carlo_mi(rx: &[C64], tx: &[C64], c: &Constellation, px: Option<&[f64]>) -> Result<f64> {
    let (r, t) = preprocess(rx, tx, c.kind())?;
    let e: Vec<C64> = r.iter().zip(&t).map(|(a, b)| a - b).collect();
    // floored so that a noiseless input gives log2 M instead of 0/0
    let sigma2 = variance(&e).max(f64::MIN_POSITIVE);
    let scale = if c.kind().is_real() { 2.0 * sigma2 } else { sigma2 };
    let (cn, p) = shaped(c, px)?;
    let lp = log_priors(&p);
    let mut h_xgy = 0.0;
    let mut metric = SymbolMetric::new(cn.points(), &lp, scale);
    for (&y, x) in r.iter().zip(&t) {
        let log_py = metric.log_sum(y, 1.0, None);
        // the transmitted symbol is the constellation point nearest to the normalised
        // reference (OptiCommPy uses the normalised reference itself, which is slightly
        // off the grid after an empirical power normalisation and can push the estimate
        // above log2 M at high SNR)
        let log_pxy = metric.term(cn.nearest(*x));
        h_xgy -= (log_pxy - log_py) / std::f64::consts::LN_2;
    }
    Ok(entropy(&p) - h_xgy / r.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use com_math::{db2lin, power};

    /// The softplus that the rates used before `com_math::ln_1p_exp`.
    fn former_ln_1p_exp(x: f64) -> f64 {
        if x > 30.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
    }

    /// [`gmi_from_llr`] with the former softplus.
    fn former_gmi_from_llr(llr: &[f64], bits: &[u8], b: usize, h: f64) -> f64 {
        let mut gmi = 0.0;
        for k in 0..b {
            let (mut acc, mut n) = (0.0, 0usize);
            for i in (k..llr.len()).step_by(b) {
                let l = llr[i].clamp(-500.0, 500.0);
                acc += former_ln_1p_exp((2.0 * f64::from(bits[i]) - 1.0) * l) / std::f64::consts::LN_2;
                n += 1;
            }
            gmi += h / b as f64 - acc / n as f64;
        }
        gmi
    }

    /// The bit-metric rate of [`posterior_rates`] at the scale 1 as written before the
    /// shared kernel: `log_sum_exp_iter` over the halves, the former softplus and the
    /// scale of `c.normalized()`.
    fn former_air_bmd(z: &[C64], idx: &[usize], c: &Constellation, nv: f64, px: Option<&[f64]>) -> f64 {
        let cn = c.normalized();
        let (m, b) = (cn.order(), cn.bits_per_symbol());
        let (_, p) = shaped(c, px).unwrap();
        let lp = log_priors(&p);
        let bits = com_mapping::constellation::indices_to_bits(idx, b);
        let mut acc = 0.0;
        let mut row = vec![0.0; m];
        for (k, &zk) in z.iter().enumerate() {
            for (r, x) in row.iter_mut().zip(cn.points()) {
                *r = (zk - x).norm_sqr() / nv;
            }
            for bit in 0..b {
                let own = bits[k * b + bit] as usize;
                let lse = |val: usize| {
                    let has = |j: &usize| (j >> (b - 1 - bit)) & 1 == val;
                    com_math::log_sum_exp_iter((0..m).filter(has).map(|j| lp[j] - row[j]))
                };
                acc += former_ln_1p_exp(lse(1 - own) - lse(own));
            }
        }
        entropy(&p) - acc / z.len() as f64 / std::f64::consts::LN_2
    }

    fn link(
        c: &Constellation,
        n: usize,
        snr_db: f64,
        px: Option<&[f64]>,
        seed: u64,
    ) -> (Vec<usize>, Vec<C64>, Vec<C64>) {
        let mut rng = Rng::new(seed);
        let (idx, x) = com_mapping::source::symbols(c, n, px, &mut rng).unwrap();
        let w = rng.complex_noise(x.len(), power(&x) / db2lin(snr_db));
        let y = x.iter().zip(&w).map(|(a, b)| a + b).collect();
        (idx, x, y)
    }

    #[test]
    fn the_shared_softplus_moves_the_gmi_by_rounding_only() {
        let c = Constellation::qam(16).unwrap().normalized();
        for snr in [0.0, 10.0, 20.0] {
            let (idx, x, y) = link(&c, 20_000, snr, None, 5);
            let g = monte_carlo_gmi(&y, &x, &c, None).unwrap();
            // the former estimate, from the same LLRs
            let (r, t) = preprocess(&y, &x, c.kind()).unwrap();
            let e: Vec<C64> = r.iter().zip(&t).map(|(a, b)| a - b).collect();
            let sigma2 = variance(&e).max(f64::MIN_POSITIVE);
            let (cn, p) = shaped(&c, None).unwrap();
            let llr = cn.llr(&r, sigma2, Some(&p), false).unwrap();
            let former = former_gmi_from_llr(&llr, &cn.demodulate(&t), 4, entropy(&p));
            assert!((g.gmi - former).abs() < 1e-12, "{snr} dB: {former} -> {}", g.gmi);
            let rates = information_rates(&y, &idx, &c, None).unwrap();
            assert!(rates.gmi > 0.0 && rates.gmi <= 4.0 + 1e-12);
        }
    }

    #[test]
    fn the_bit_metric_rate_of_the_kernel_is_the_former_one() {
        let c = Constellation::qam(16).unwrap().normalized();
        for snr in [3.0, 12.0] {
            let (idx, _, y) = link(&c, 5_000, snr, None, 6);
            let nv = 10f64.powf(-snr / 10.0);
            let r = posterior_rates(&y, &idx, &c, nv, None).unwrap();
            let former = former_air_bmd(&y, &idx, &c, nv, None);
            assert!((r.air_bmd - former).abs() < 1e-12, "{snr} dB: {former} -> {}", r.air_bmd);
        }
    }
}
