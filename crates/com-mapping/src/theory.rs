//! Closed-form error rates and information rates of constellations over AWGN.
//!
//! - [`ber`] (Gray-mapped QAM, PSK and PAM, OptiCommPy `theoryBER`) and [`ser`] give the
//!   error rates as functions of $E_{b}/N_{0}$, and [`ebn0_for_ber`], [`ebn0_for_ser`]
//!   their inverses;
//! - [`ber_rayleigh_mrc`] gives the BER of BPSK and Gray QPSK with maximal-ratio
//!   combining of independent Rayleigh branches;
//! - [`mi`] and [`gmi`] give the symbol-wise and bit-wise mutual information by
//!   Gauss–Hermite quadrature (OptiCommPy `theoryMI`, `theoryGMI`), and
//!   [`constellation_mi`], [`constellation_gmi`] the same for any constellation and
//!   labelling, over AWGN or Rayleigh fading ([`Fading`]): the capacities of coded
//!   modulation and BICM;
//! - [`ber_dpsk`], [`ber_dbpsk_coherent`] and [`differential_phase_tail`] give the error
//!   rates of differential PSK ([`crate::differential`]);
//! - [`ber_gray_pam`], [`ber_gray_qam`] and [`ber_gray_rect_qam`] give the exact BER of
//!   Gray-mapped PAM, square and rectangular QAM (Cho and Yoon 2002), [`ber_exact`] the
//!   exact BER of a family where it exists, and [`phase_pdf`]
//!   the density of the phase of a carrier in Gaussian noise;
//! - [`ser_orthogonal_noncoherent`] gives the SER of $M$ orthogonal signals detected
//!   noncoherently (M-FSK, chirp spread spectrum).
//!
//! They are properties of a constellation and its labelling, so they live with the
//! constellations; `com-metrics` re-exports them as `theory_ber`, `theory_ser`,
//! `theory_mi` and `theory_gmi`. The functions of a family of constellations take the
//! family and the order first: `(kind, m, …)`.
//!
//! # Model
//!
//! For BPSK over AWGN,
//!
//! $$
//! P_{b} = Q\left(\sqrt{2 E_{b} / N_{0}}\right)
//! $$
//!
//! # Examples
//! ```
//! use com_mapping::{theory, constellation::ConstellationKind};
//! let pb = theory::ber(ConstellationKind::Qam, 4, 6.0).unwrap();
//! assert!((pb - 2.388e-3).abs() < 1e-6);
//! ```
//!
//! # References
//!
//! 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.
//! 2. R. F. Pawula, S. O. Rice and J. H. Roberts, "Distribution of the phase angle
//!    between two vectors perturbed by Gaussian noise," *IEEE Trans. Commun.*, 30(8),
//!    1828–1841, 1982.

use crate::constellation::{Constellation, ConstellationKind, log_priors, qam_side};
use com_core::{Error, Result, ensure};
use com_math::{C64, bessel_i0e, bisect, db2lin, entropy, log_sum_exp, qfunc};

/// The orders of the closed forms of [`ber`] and [`ser`]: a power of two of at least 2,
/// a square for QAM.
fn ensure_order(kind: ConstellationKind, m: usize) -> Result<()> {
    ensure(m >= 2 && m.is_power_of_two(), || format!("the order must be a power of two ≥ 2, got {m}"))?;
    ensure(kind != ConstellationKind::Qam || qam_side(m).is_some(), || {
        format!("the closed forms of QAM need a square order, got {m}")
    })
}

/// Theoretical bit error rate over AWGN with Gray mapping (OptiCommPy `theoryBER`).
///
/// - QAM (`L = √M`): `P_b = 2(1 - 1/L)/log2 L · Q(√(3 log2 L/(L²-1) · 2 E_b/N_0))`;
/// - PSK: `P_b = 2 Q(√(2 log2 M · E_b/N_0) sin(π/M)) / log2 M`, and the exact
///   `Q(√(2 E_b/N_0))` for `M = 2` (OptiCommPy applies the general formula, which counts
///   two nearest neighbours and doubles the BER of BPSK);
/// - PAM: `P_b = 2(M-1)/M · Q(√(6 log2 M/(M²-1) · E_b/N_0)) / log2 M`.
///
/// # Errors
/// [`Error::Config`] for another family, an order that is not a power of two (at least
/// 2), or a QAM that is not square.
pub fn ber(kind: ConstellationKind, m: usize, ebn0_db: f64) -> Result<f64> {
    ensure_order(kind, m)?;
    let g = db2lin(ebn0_db);
    let mf = m as f64;
    let k = mf.log2();
    match kind {
        ConstellationKind::Qam => {
            let l = mf.sqrt();
            Ok(2.0 * (1.0 - 1.0 / l) / l.log2() * qfunc((3.0 * l.log2() / (l * l - 1.0) * 2.0 * g).sqrt()))
        }
        ConstellationKind::Psk if m == 2 => Ok(qfunc((2.0 * g).sqrt())),
        ConstellationKind::Psk => Ok(2.0 * qfunc((2.0 * k * g).sqrt() * (std::f64::consts::PI / mf).sin()) / k),
        ConstellationKind::Pam => Ok(2.0 * (mf - 1.0) / mf * qfunc((6.0 * k / (mf * mf - 1.0) * g).sqrt()) / k),
        other => Err(Error::Config(format!("no theoretical BER for {other:?}"))),
    }
}

/// Theoretical symbol error rate over AWGN as a function of `E_b/N_0`.
///
/// - square QAM: exact, `P_s = 1 - (1 - p)²` with `p = 2(1 - 1/L) Q(√(3 E_s/N_0/(M-1)))`;
/// - PAM: exact, `P_s = 2(M-1)/M · Q(√(6 E_s/N_0/(M²-1)))`;
/// - PSK: the nearest-neighbour bound `2 Q(√(2 E_s/N_0) sin(π/M))` (exact for `M = 2`
///   when halved, tight for `M ≥ 4`).
///
/// # Errors
/// As [`ber`].
pub fn ser(kind: ConstellationKind, m: usize, ebn0_db: f64) -> Result<f64> {
    ensure_order(kind, m)?;
    let mf = m as f64;
    let es = db2lin(ebn0_db) * mf.log2();
    match kind {
        ConstellationKind::Qam => {
            let l = mf.sqrt();
            let p = 2.0 * (1.0 - 1.0 / l) * qfunc((3.0 * es / (mf - 1.0)).sqrt());
            Ok(1.0 - (1.0 - p) * (1.0 - p))
        }
        ConstellationKind::Pam => Ok(2.0 * (mf - 1.0) / mf * qfunc((6.0 * es / (mf * mf - 1.0)).sqrt())),
        ConstellationKind::Psk => {
            let p = 2.0 * qfunc((2.0 * es).sqrt() * (std::f64::consts::PI / mf).sin());
            Ok(if m == 2 { p / 2.0 } else { p })
        }
        other => Err(Error::Config(format!("no theoretical SER for {other:?}"))),
    }
}

/// The $E_{b}/N_{0}$ (dB) at which the BER of [`ber`] equals `target`: the bisection
/// (200 steps) of $[-10, 60]$ dB, where the BER decreases.
///
/// # Errors
/// [`Error::Config`] for a family without a closed form, or a target that no
/// $E_{b}/N_{0}$ of $[-10, 60]$ dB reaches.
///
/// # Examples
/// ```
/// use com_mapping::{theory, constellation::ConstellationKind};
/// // QPSK: Q(√(2 Eb/N0)) = 1e-3 at 6.79 dB
/// let e = theory::ebn0_for_ber(ConstellationKind::Qam, 4, 1e-3).unwrap();
/// assert!((e - 6.79).abs() < 0.01);
/// ```
pub fn ebn0_for_ber(kind: ConstellationKind, m: usize, target: f64) -> Result<f64> {
    ebn0_for(|e| ber(kind, m, e), "BER", target)
}

/// The $E_{b}/N_{0}$ (dB) at which the SER of [`ser`] equals `target`, as
/// [`ebn0_for_ber`].
///
/// # Errors
/// As [`ebn0_for_ber`].
///
/// # Examples
/// ```
/// use com_mapping::{theory, constellation::ConstellationKind};
/// let e = theory::ebn0_for_ser(ConstellationKind::Qam, 16, 1e-4).unwrap();
/// let s = theory::ser(ConstellationKind::Qam, 16, e).unwrap();
/// assert!((s / 1e-4 - 1.0).abs() < 1e-9);
/// ```
pub fn ebn0_for_ser(kind: ConstellationKind, m: usize, target: f64) -> Result<f64> {
    ebn0_for(|e| ser(kind, m, e), "SER", target)
}

/// The root of `rate(e) = target` on `[-10, 60]` dB for a decreasing `rate`.
fn ebn0_for(rate: impl Fn(f64) -> Result<f64>, what: &str, target: f64) -> Result<f64> {
    let (a, b) = (-10.0, 60.0);
    ensure(rate(a)? > target && target > rate(b)?, || format!("no Eb/N0 in [-10, 60] dB gives a {what} of {target}"))?;
    // the family has a closed form (checked above): `rate` cannot fail on the bracket
    Ok(bisect(|e| rate(e).is_ok_and(|v| v > target), a, b, 200))
}

/// The bit error rate of BPSK (or Gray QPSK, per bit) with maximal-ratio combining of
/// `branches` independent Rayleigh-faded branches of mean SNR per bit `mean_ebn0`
/// ($\bar\gamma$, linear) each (Proakis & Salehi, *Digital Communications*, 5th ed.,
/// eq. (13.4-15)):
///
/// $$
/// P_{b} = \left(\frac{1 - \mu}{2}\right)^{L} \sum_{k=0}^{L-1} \binom{L - 1 + k}{k}
/// \left(\frac{1 + \mu}{2}\right)^{k}, \qquad \mu = \sqrt{\frac{\bar\gamma}{1 + \bar\gamma}},
/// $$
///
/// with $\bar\gamma$ the mean $E_{b}/N_{0}$ per branch (linear) and $L$ the number of
/// branches; $L = 1$ is the flat Rayleigh channel, $P_{b} = \frac{1}{2}(1 - \mu)$.
///
/// # Errors
/// [`Error::Config`] for no branch.
///
/// # Examples
/// ```
/// use com_mapping::theory::ber_rayleigh_mrc;
/// let g = com_math::db2lin(10.0);
/// assert!((ber_rayleigh_mrc(g, 1)? - 0.5 * (1.0 - (g / (1.0 + g)).sqrt())).abs() < 1e-15);
/// // diversity: four branches at 10 dB each
/// assert!(ber_rayleigh_mrc(g, 4)? < 1e-5);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn ber_rayleigh_mrc(mean_ebn0_lin: f64, branches: usize) -> Result<f64> {
    ensure(branches >= 1, || "maximal-ratio combining needs at least one branch".into())?;
    let mu = (mean_ebn0_lin / (1.0 + mean_ebn0_lin)).sqrt();
    let l = branches;
    let mut binom = 1.0;
    let mut sum = 0.0;
    for k in 0..l {
        if k > 0 {
            binom *= (l - 1 + k) as f64 / k as f64;
        }
        sum += binom * ((1.0 + mu) / 2.0).powi(k as i32);
    }
    Ok(((1.0 - mu) / 2.0).powi(l as i32) * sum)
}

/// Normalised points, probabilities, `N_0` and quadrature nodes of a theory function.
pub(crate) type TheorySetup = (Vec<C64>, Vec<f64>, f64, Vec<(C64, f64)>);

/// The normalised points, the probabilities and the quadrature of a theory function of
/// a constellation, its labels included (`points()[i]` carries the bits of `i`).
pub(crate) fn setup(c: &Constellation, px: Option<&[f64]>, snr_db: f64, points: usize) -> Result<TheorySetup> {
    let p = c.priors(px)?;
    let pts = c.normalized_under(&p)?.points().to_vec();
    // OptiCommPy: a real constellation sees a real channel of variance 1/SNR, a complex
    // one a complex channel of variance 1/SNR (1/(2 SNR) per dimension)
    let real = c.kind().is_real();
    let sigma2 = db2lin(-snr_db);
    let n0 = if real { 2.0 * sigma2 } else { sigma2 };
    Ok((pts, p, n0, com_math::gaussian_nodes(sigma2, !real, points)))
}

/// The channel of [`constellation_mi`], [`constellation_gmi`] and of the cutoff rates of
/// [`crate::bicm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fading {
    /// No fading: $g = 1$.
    Awgn,
    /// Flat Rayleigh fading, $g \sim \mathcal{CN}(0, 1)$ independent from symbol to
    /// symbol (ideal interleaving), known at the receiver.
    Rayleigh,
}

impl std::str::FromStr for Fading {
    type Err = Error;

    /// `"awgn"` or `"rayleigh"`.
    fn from_str(name: &str) -> Result<Fading> {
        match name.to_ascii_lowercase().as_str() {
            "awgn" => Ok(Fading::Awgn),
            "rayleigh" => Ok(Fading::Rayleigh),
            other => Err(Error::Config(format!("unknown fading {other:?}: awgn, rayleigh"))),
        }
    }
}

/// The mean of `f(snr_db)` over the channel: `f(snr_db)` itself without fading, and
/// $\mathbb{E}\, f(\mathrm{snr} + 10 \log_{10} |g|^{2})$ over Rayleigh fading, the
/// integral over $u = \ln |g|^{2}$, of density $e^{u - e^{u}}$, by Simpson's rule with a
/// step of 1/8.
pub(crate) fn over_fading(f: impl Fn(f64) -> Result<f64>, snr_db: f64, fading: Fading) -> Result<f64> {
    if fading == Fading::Awgn {
        return f(snr_db);
    }
    // below e^{lo} the integrand is f(0) to first order, with the mass 1 - e^{-e^{lo}}
    let lo = (1e-4 / db2lin(snr_db).max(1.0)).ln();
    let hi = 40f64.ln();
    let n = (8.0 * (hi - lo)).ceil() as usize;
    let at = |u: f64| -> Result<f64> { Ok((u - u.exp()).exp() * f(snr_db + 10.0 * u / std::f64::consts::LN_10)?) };
    let tail = -(-lo.exp()).exp_m1() * f(snr_db + 10.0 * lo / std::f64::consts::LN_10)?;
    Ok(com_math::try_simpson(at, lo, hi, n)? + tail)
}

/// The Gauss–Hermite nodes per dimension of the information rates when the caller does
/// not choose ([`constellation_mi`], [`constellation_gmi`], the BICM cutoff rate): the
/// default of the `theory` checks of the specifications and of the Python closed forms.
pub const DEFAULT_QUADRATURE_POINTS: usize = 40;

/// The mutual information (bits/symbol) of the constellation `c` with the probabilities
/// `px` (uniform if `None`) over the channel `fading`: [`mi`] of any constellation,
/// averaged over the fading. With uniform inputs it is the capacity of coded modulation
/// (Caire, Taricco & Biglieri 1998, eq. (10)).
///
/// # Errors
/// [`Error::Config`] for invalid probabilities.
pub fn constellation_mi(
    c: &Constellation,
    snr_db: f64,
    px: Option<&[f64]>,
    fading: Fading,
    points: usize,
) -> Result<f64> {
    over_fading(|s| Ok(mi_of(&setup(c, px, s, points)?)), snr_db, fading)
}

/// The generalized mutual information (bits/symbol) of the bit-wise decoder of the
/// labelled constellation `c` (point `i` carries the bits of `i`) with the probabilities
/// `px` over the channel `fading`: [`gmi`] of any labelling, averaged over the fading.
/// With uniform inputs it is the capacity of BICM (Caire, Taricco & Biglieri 1998,
/// eq. (14)).
///
/// # Errors
/// [`Error::Config`] for invalid probabilities.
pub fn constellation_gmi(
    c: &Constellation,
    snr_db: f64,
    px: Option<&[f64]>,
    fading: Fading,
    points: usize,
) -> Result<f64> {
    over_fading(|s| Ok(gmi_of(&setup(c, px, s, points)?)), snr_db, fading)
}

/// Mutual information (bits/symbol) of a constellation over AWGN, by Gauss–Hermite
/// quadrature (OptiCommPy `theoryMI`).
///
/// `snr_db` is `E_s/σ²` with `E_s` the mean energy under `px` (uniform if `None`);
/// a real constellation (PAM, OOK) sees a real channel, a complex one a circular
/// complex channel, as in OptiCommPy. With `D_ij(w) = (|x_i - x_j + w|² - |w|²)/N_0`,
///
/// `MI = -Σ_i p_i E_w log2 Σ_j p_j e^{-D_ij(w)}`,
///
/// evaluated with `points` nodes per dimension (40 give 6 digits for 64-QAM).
///
/// # Errors
/// [`Error::Config`] for an invalid constellation or probabilities.
///
/// # Examples
/// ```
/// use com_mapping::{theory, constellation::ConstellationKind};
/// // the MI of 16-QAM approaches 4 bits at high SNR and stays below the capacity
/// let mi = theory::mi(ConstellationKind::Qam, 16, 10.0, None, 40).unwrap();
/// assert!(mi < (1.0 + 10.0f64).log2() && mi > 2.5);
/// let hi = theory::mi(ConstellationKind::Qam, 16, 30.0, None, 40).unwrap();
/// assert!((hi - 4.0).abs() < 1e-6);
/// ```
pub fn mi(kind: ConstellationKind, m: usize, snr_db: f64, px: Option<&[f64]>, points: usize) -> Result<f64> {
    constellation_mi(&Constellation::new(kind, m)?, snr_db, px, Fading::Awgn, points)
}

/// The kernel of [`mi`].
pub(crate) fn mi_of((x, p, n0, nodes): &TheorySetup) -> f64 {
    let (n0, lp) = (*n0, log_priors(p));
    let mut acc = 0.0;
    let mut terms = vec![0.0; x.len()];
    for (i, xi) in x.iter().enumerate() {
        if p[i] == 0.0 {
            continue;
        }
        let mut e = 0.0;
        for (w, a) in nodes {
            for (j, xj) in x.iter().enumerate() {
                terms[j] = lp[j] - ((xi - xj + w).norm_sqr() - w.norm_sqr()) / n0;
            }
            e += a * log_sum_exp(&terms);
        }
        acc += p[i] * e;
    }
    -acc / std::f64::consts::LN_2
}

/// Generalised mutual information (bits/symbol) of the bit-wise decoder of a Gray-labelled
/// constellation over AWGN, by Gauss–Hermite quadrature (OptiCommPy `theoryGMI`).
///
/// `GMI = H(X) - Σ_k H(B_k | Y)` with the labels of the points (point `i` carries the
/// bits of `i`); same channel conventions and parameters as [`mi`]. For uniform symbols
/// it is `Σ_k I(B_k; Y)`, OptiCommPy's formula; with shaping the bits are dependent and
/// `Σ_k I(B_k; Y)` can exceed the entropy of the source, while `H(X) - Σ_k H(B_k | Y)`
/// is the rate achievable by bit-metric decoding (Böcherer 2014) and the limit of the
/// Monte Carlo estimators (OptiCommPy's `monteCarloGMI` included).
///
/// G. Böcherer, "Achievable rates for shaped bit-metric decoding," arXiv:1410.8075,
/// 2014; A. Alvarado, T. Fehenberger, B. Chen and F. M. J. Willems, *J. Lightw.
/// Technol.*, 36(2), 424–439, 2018.
///
/// # Errors
/// [`Error::Config`] for an invalid constellation or probabilities.
pub fn gmi(kind: ConstellationKind, m: usize, snr_db: f64, px: Option<&[f64]>, points: usize) -> Result<f64> {
    constellation_gmi(&Constellation::new(kind, m)?, snr_db, px, Fading::Awgn, points)
}

/// The kernel of [`gmi`].
pub(crate) fn gmi_of(s: &TheorySetup) -> f64 {
    // GMI = H(X) - Σ_k H(B_k | Y): with shaped (dependent) bits, the sum of the
    // bit-wise mutual informations Σ_k I(B_k; Y) can exceed H(X) and is not achievable
    let mut gmi = entropy(&s.1);
    // -log2 P(b_k = b_k(i) | y)
    for h_cond in bit_expectations(s, |all, same| log_sum_exp(all) - same) {
        gmi -= h_cond / std::f64::consts::LN_2;
    }
    gmi
}

/// For every bit position `k`, $\sum_{i} p_{i} E_{w} f(\ell, \ell_{k})$ with
/// `ℓ[j] = ln p_j - (|x_i - x_j + w|² - |w|²)/N_0` (the log-metrics of $y = x_{i} + w$,
/// up to a common term) and $\ell_{k}$ their log-sum over the points whose bit `k`
/// equals that of `i`; the metrics of a noise sample are computed once for all bits.
pub(crate) fn bit_expectations((x, p, n0, nodes): &TheorySetup, f: impl Fn(&[f64], f64) -> f64) -> Vec<f64> {
    let m = x.len();
    let b = m.trailing_zeros() as usize;
    let lp = log_priors(p);
    let mut all = vec![0.0; m];
    let mut out = vec![0.0; b];
    let mut e = vec![0.0; b];
    for (i, xi) in x.iter().enumerate() {
        if p[i] == 0.0 {
            continue;
        }
        e.fill(0.0);
        for (w, a) in nodes {
            for (j, xj) in x.iter().enumerate() {
                all[j] = lp[j] - ((xi - xj + w).norm_sqr() - w.norm_sqr()) / n0;
            }
            for (k, ek) in e.iter_mut().enumerate() {
                let shift = b - 1 - k;
                let bi = (i >> shift) & 1;
                let same = com_math::log_sum_exp_iter((0..m).filter(|&j| (j >> shift) & 1 == bi).map(|j| all[j]));
                *ek += a * f(&all, same);
            }
        }
        for (o, ek) in out.iter_mut().zip(&e) {
            *o += p[i] * ek;
        }
    }
    out
}

/// The probability that the phase angle between two vectors of the same SNR `esn0_lin`
/// (linear, $E_{s}/N_{0}$ each) perturbed by independent Gaussian noise exceeds `psi`
/// in absolute value, $0 \le \psi \le \pi$ (Pawula, Rice & Roberts, 1982, the integral
/// form for equal SNRs):
///
/// $$
/// P(|\Delta\theta| > \psi) = \frac{\sin \psi}{2 \pi} \int_{-\pi/2}^{\pi/2}
/// \frac{\exp\left[-\gamma_{s} (1 - \cos \psi \cos t)\right]}{1 - \cos \psi \cos t} \\, dt,
/// $$
///
/// evaluated by Simpson's rule with 2 000 intervals (the integrand is smooth). It is
/// the symbol error rate of M-DPSK for $\psi = \pi/M$, and $\frac{1}{2} e^{-\gamma_{s}}$
/// for $\psi = \pi/2$ (DBPSK).
///
/// # Examples
/// ```
/// use com_mapping::theory::differential_phase_tail;
/// let g = 5.0;
/// assert!((differential_phase_tail(std::f64::consts::FRAC_PI_2, g) - 0.5 * (-g).exp()).abs() < 1e-9);
/// assert!((differential_phase_tail(0.0, g) - 1.0).abs() < 1e-9);
/// assert!(differential_phase_tail(std::f64::consts::PI, g).abs() < 1e-15);
/// ```
pub fn differential_phase_tail(psi: f64, esn0_lin: f64) -> f64 {
    let psi = psi.clamp(0.0, std::f64::consts::PI);
    if psi == 0.0 {
        return 1.0;
    }
    let (s, c) = psi.sin_cos();
    let f = |t: f64| {
        let d = 1.0 - c * t.cos();
        (-esn0_lin * d).exp() / d
    };
    let half = std::f64::consts::FRAC_PI_2;
    s / (2.0 * std::f64::consts::PI) * com_math::simpson(f, -half, half, 2000)
}

/// The terms of eq. (9) of Cho and Yoon for the bit `k` (1 = most significant) of
/// Gray-mapped `side`-PAM: the pairs $(X, Y)$ of Table I, so that
/// $P_{b}(k) = \frac{1}{I} \sum X \\, \mathrm{erfc}(Y \cdot \ldots)$, with
/// $X = (-1)^{\lfloor i 2^{k-1}/I \rfloor} (2^{k-1} - \lfloor i 2^{k-1}/I + \frac{1}{2} \rfloor)$
/// and $Y = 2i + 1$ for $i = 0 \dots (1 - 2^{-k}) I - 1$.
fn cho_yoon_terms(side: usize, k: usize) -> impl Iterator<Item = (f64, usize)> {
    let step = 1usize << (k - 1);
    (0..side - side / (1 << k)).map(move |i| {
        let sign = if (i * step / side) % 2 == 0 { 1.0 } else { -1.0 };
        let weight = step as f64 - ((i * step) as f64 / side as f64 + 0.5).floor();
        (sign * weight, 2 * i + 1)
    })
}

/// $\sum_{k} P_{b}(k)$ over the $\log_{2} I$ bits of Gray-mapped $I$-PAM, eq. (9) of Cho
/// and Yoon with $\mathrm{erfc}(Y a) = 2Q(Y x)$: `x` is the half distance between
/// neighbouring levels over the noise standard deviation.
fn gray_pam_bit_errors(side: usize, x: f64) -> f64 {
    let levels = side.trailing_zeros() as usize;
    (1..=levels)
        .map(|k| cho_yoon_terms(side, k).map(|(w, y)| w * 2.0 * qfunc(y as f64 * x)).sum::<f64>() / side as f64)
        .sum()
}

fn ensure_pam_side(side: usize, what: &str) -> Result<()> {
    ensure(side >= 2 && side.is_power_of_two(), || {
        format!("the exact Gray BER needs {what} a power of two ≥ 2, got {side}")
    })
}

/// Exact BER of Gray-mapped `m`-PAM over AWGN (K. Cho and D. Yoon, "On the general BER
/// expression of one- and two-dimensional amplitude modulations," *IEEE Trans. Commun.*
/// 50(7), 1074–1080, 2002, eqs. (9)–(10)), as a function of $E_{b}/N_{0}$ in dB:
///
/// $$P_{b} = \frac{1}{\log_{2} I} \sum_{k=1}^{\log_{2} I} P_{b}(k)\\,, \qquad
/// P_{b}(k) = \frac{1}{I} \sum_{i} X_{i,k} \\, \mathrm{erfc}\left((2i + 1)
/// \sqrt{\frac{3 \log_{2} I \cdot E_{b}/N_{0}}{I^{2} - 1}}\right),$$
///
/// with the coefficients $X_{i,k}$ of Table I of the paper. Unlike the
/// nearest-neighbour approximation of [`ber`], it tends to 1/2 at low SNR.
///
/// # Errors
/// [`Error::Config`] unless `m` is a power of two ≥ 2.
///
/// # Examples
/// ```
/// use com_mapping::theory::ber_gray_pam;
/// use com_math::{db2lin, qfunc};
/// // 4-PAM: (3Q(x) + 2Q(3x) - Q(5x))/4 with x = √(4/5 · E_b/N_0)
/// let x = (0.8 * db2lin(10.0)).sqrt();
/// let want = (3.0 * qfunc(x) + 2.0 * qfunc(3.0 * x) - qfunc(5.0 * x)) / 4.0;
/// assert!((ber_gray_pam(4, 10.0).unwrap() - want).abs() < 1e-15);
/// ```
pub fn ber_gray_pam(m: usize, ebn0_db: f64) -> Result<f64> {
    ensure_pam_side(m, "an order")?;
    let log2m = f64::from(m.trailing_zeros());
    let x = (6.0 * log2m * db2lin(ebn0_db) / ((m * m) as f64 - 1.0)).sqrt();
    Ok(gray_pam_bit_errors(m, x) / log2m)
}

/// Exact BER of Gray-mapped `mi × mj` rectangular QAM over AWGN, two Gray-mapped PAM
/// of `mi` and `mj` levels in quadrature (Cho and Yoon 2002, eqs. (19)–(22)):
///
/// $$P_{b} = \frac{1}{\log_{2}(IJ)} \left(\sum_{k=1}^{\log_{2} I} P_{I}(k) +
/// \sum_{l=1}^{\log_{2} J} P_{J}(l)\right),$$
///
/// the terms of [`ber_gray_pam`] with the argument
/// $(2i + 1)\sqrt{3 \log_{2}(IJ) \\, E_{b}/N_{0} / (I^{2} + J^{2} - 2)}$.
///
/// # Errors
/// [`Error::Config`] unless `mi` and `mj` are powers of two ≥ 2.
///
/// # Examples
/// ```
/// use com_mapping::theory::{ber_gray_qam, ber_gray_rect_qam};
/// // a square QAM is the case I = J
/// let (a, b) = (ber_gray_rect_qam(4, 4, 8.0).unwrap(), ber_gray_qam(16, 8.0).unwrap());
/// assert!((a - b).abs() < 1e-15);
/// // 32-QAM as 8 × 4 lies between 16- and 64-QAM
/// let p32 = ber_gray_rect_qam(8, 4, 12.0).unwrap();
/// assert!(p32 > ber_gray_qam(16, 12.0).unwrap() && p32 < ber_gray_qam(64, 12.0).unwrap());
/// ```
pub fn ber_gray_rect_qam(mi: usize, mj: usize, ebn0_db: f64) -> Result<f64> {
    ensure_pam_side(mi, "both sides")?;
    ensure_pam_side(mj, "both sides")?;
    let log2m = f64::from(mi.trailing_zeros() + mj.trailing_zeros());
    let x = (6.0 * log2m * db2lin(ebn0_db) / ((mi * mi + mj * mj) as f64 - 2.0)).sqrt();
    Ok((gray_pam_bit_errors(mi, x) + gray_pam_bit_errors(mj, x)) / log2m)
}

/// Exact BER of Gray-mapped square `m`-QAM over AWGN (Cho and Yoon 2002, eq. (16)), as
/// a function of `E_b/N_0` in dB: [`ber_gray_rect_qam`] with `I = J = √m`. Unlike the
/// nearest-neighbour approximation of [`ber`], it tends to 1/2 at low SNR, the BER of
/// random decisions; for 16-QAM, `P_b = (3Q(x) + 2Q(3x) - Q(5x))/4` with
/// `x = √(E_s/(5N_0))`.
///
/// # Errors
/// [`Error::Config`] for a non-square order.
///
/// # Examples
/// ```
/// use com_mapping::theory::{ber, ber_gray_qam};
/// use com_mapping::constellation::ConstellationKind;
/// let exact = ber_gray_qam(16, 10.0).unwrap();
/// let approx = ber(ConstellationKind::Qam, 16, 10.0).unwrap();
/// assert!(exact > approx && (exact / approx - 1.0).abs() < 0.03);
/// ```
pub fn ber_gray_qam(m: usize, ebn0_db: f64) -> Result<f64> {
    let side = qam_side(m).unwrap_or(0);
    ensure(side >= 2 && side.is_power_of_two(), || format!("the exact Gray BER needs a square QAM, got {m}"))?;
    ber_gray_rect_qam(side, side, ebn0_db)
}

/// Probability density of the phase error $\theta = \arg r - \varphi$ of a carrier
/// $r = A e^{j\varphi} + w$ in circular Gaussian noise, at the SNR
/// $\rho = A^{2}/\mathbb{E}|w|^{2}$ (linear) (S. O. Rice, *Bell Syst. Tech. J.* 27(1),
/// 1948; as eq. (8) of A. U. Ahmed, S. C. Thompson and J. R. Zeidler, "Threshold
/// extending receiver structures for CE-OFDM," *Proc. IEEE MILCOM*, 2007):
///
/// $$p(\theta) = \frac{e^{-\rho}}{2\pi}\left[1 + \sqrt{4\pi\rho}\\,\cos\theta\\,
/// e^{\rho\cos^{2}\theta}\\,\Phi\left(\sqrt{2\rho}\cos\theta\right)\right],
/// \qquad \theta \in (-\pi, \pi],$$
///
/// $\Phi$ the standard normal distribution function. It is uniform at $\rho = 0$ and
/// tends to $\mathcal{N}(0, 1/(2\rho))$ at high SNR; integrated over the decision
/// sectors it gives the error rates of PSK.
///
/// # Examples
/// ```
/// use com_mapping::theory::phase_pdf;
/// // uniform without signal, Gaussian of variance 1/(2ρ) at high SNR
/// assert!((phase_pdf(1.0, 0.0) - 1.0 / std::f64::consts::TAU).abs() < 1e-15);
/// let rho = 400.0;
/// let gauss = (rho / std::f64::consts::PI).sqrt();
/// assert!((phase_pdf(0.0, rho) / gauss - 1.0).abs() < 1e-12);
/// ```
pub fn phase_pdf(theta: f64, snr_lin: f64) -> f64 {
    let c = theta.cos();
    let s2 = theta.sin().powi(2);
    // e^{-ρ} √(4πρ) cos θ e^{ρ cos²θ} = √(4πρ) cos θ e^{-ρ sin²θ}, which does not overflow
    (-snr_lin).exp() / std::f64::consts::TAU
        + (snr_lin / std::f64::consts::PI).sqrt()
            * c
            * (-snr_lin * s2).exp()
            * (1.0 - qfunc((2.0 * snr_lin).sqrt() * c))
}

/// The exact bit error rate of a Gray-labelled constellation over AWGN, as a function of
/// $E_{b}/N_{0}$ (dB), where a closed form exists: [`ber_gray_pam`] for PAM,
/// [`ber_gray_qam`] for square QAM (Cho and Yoon 2002), and [`ber`] for BPSK and QPSK,
/// whose nearest-neighbour expression is exact.
///
/// # Errors
/// [`Error::Config`] for another constellation (8-PSK and beyond, OOK, rectangular QAM:
/// see [`ber`] and [`ber_gray_rect_qam`]).
///
/// # Examples
/// ```
/// use com_mapping::{theory, constellation::ConstellationKind};
/// let qam = theory::ber_exact(ConstellationKind::Qam, 16, 10.0).unwrap();
/// assert!((qam - theory::ber_gray_qam(16, 10.0).unwrap()).abs() < 1e-15);
/// assert!(theory::ber_exact(ConstellationKind::Psk, 8, 10.0).is_err());
/// ```
pub fn ber_exact(kind: ConstellationKind, m: usize, ebn0_db: f64) -> Result<f64> {
    match kind {
        ConstellationKind::Pam => ber_gray_pam(m, ebn0_db),
        ConstellationKind::Qam => ber_gray_qam(m, ebn0_db),
        ConstellationKind::Psk if m <= 4 => ber(kind, m, ebn0_db),
        _ => Err(Error::Config(format!("no exact Gray BER of {kind:?} of order {m}"))),
    }
}

/// Bit error rate of Gray-labelled M-DPSK with noncoherent detection over AWGN, as a
/// function of $E_{b}/N_{0}$ (dB): $\frac{1}{2} e^{-E_{b}/N_{0}}$ for DBPSK (Proakis &
/// Salehi, §4.5), and for $M \ge 4$ the average number of wrong bits over the decision
/// sectors of the differential phase, whose probabilities come from
/// [`differential_phase_tail`] at $\gamma_{s} = E_{b}/N_{0} \log_{2} M$:
///
/// $$
/// P_{b} = \frac{1}{\log_{2} M} \sum_{k=1}^{M-1} w_{k} \\,
/// P\left(\Delta\theta \in \left[\frac{(2k - 1)\pi}{M}, \frac{(2k + 1)\pi}{M}\right)\right),
/// $$
///
/// $w_{k}$ the Hamming weight of the Gray label of the transition $k$ (for DQPSK,
/// $P_{b} = \frac{1}{2}[P(|\Delta\theta| > \pi/4) + P(|\Delta\theta| > 3\pi/4)]$).
///
/// # Errors
/// [`Error::Config`] unless `m` is a power of two ≥ 2.
///
/// # Examples
/// ```
/// use com_mapping::theory::ber_dpsk;
/// let g = com_math::db2lin(8.0);
/// assert!((ber_dpsk(2, 8.0).unwrap() - 0.5 * (-g).exp()).abs() < 1e-12);
/// // DQPSK is worse than DBPSK at the same E_b/N_0, and better than D8PSK
/// assert!(ber_dpsk(4, 8.0).unwrap() > ber_dpsk(2, 8.0).unwrap());
/// assert!(ber_dpsk(8, 8.0).unwrap() > ber_dpsk(4, 8.0).unwrap());
/// ```
pub fn ber_dpsk(m: usize, ebn0_db: f64) -> Result<f64> {
    com_core::ensure(m >= 2 && m.is_power_of_two(), || format!("the order must be a power of two ≥ 2, got {m}"))?;
    let gb = db2lin(ebn0_db);
    if m == 2 {
        return Ok(0.5 * (-gb).exp());
    }
    let bits = m.trailing_zeros();
    let gs = gb * f64::from(bits);
    let pi = std::f64::consts::PI;
    // P(Δθ < ψ) on (-π, π]
    let cdf = |psi: f64| {
        if psi < 0.0 { 0.5 * differential_phase_tail(-psi, gs) } else { 1.0 - 0.5 * differential_phase_tail(psi, gs) }
    };
    let gray = crate::constellation::gray_code(bits)?;
    let mut errors = 0.0;
    for k in 1..m {
        let mut theta = 2.0 * pi * k as f64 / m as f64;
        if theta > pi {
            theta -= 2.0 * pi;
        }
        let (lo, hi) = (theta - pi / m as f64, theta + pi / m as f64);
        let p = if hi > pi { 1.0 - cdf(lo) + cdf(hi - 2.0 * pi) } else { cdf(hi) - cdf(lo) };
        errors += gray[k].count_ones() as f64 * p;
    }
    Ok(errors / f64::from(bits))
}

/// Bit error rate of DBPSK with coherent detection and differential decoding over AWGN
/// (Proakis & Salehi, §3.2-2): an isolated symbol error spoils two differences, so
/// $P_{b} = 2p(1 - p)$ with $p = Q(\sqrt{2 E_{b}/N_{0}})$.
///
/// # Examples
/// ```
/// use com_mapping::theory::ber_dbpsk_coherent;
/// let p = com_math::qfunc((2.0 * com_math::db2lin(6.0)).sqrt());
/// assert!((ber_dbpsk_coherent(6.0) - 2.0 * p * (1.0 - p)).abs() < 1e-15);
/// ```
pub fn ber_dbpsk_coherent(ebn0_db: f64) -> f64 {
    let p = qfunc((2.0 * db2lin(ebn0_db)).sqrt());
    2.0 * p * (1.0 - p)
}

/// Bit error rate of $M$-DPSK over AWGN with the detection `detection`: [`ber_dpsk`]
/// for the noncoherent detection, [`ber_dbpsk_coherent`] for the coherent detection with
/// differential decoding (defined for $M = 2$).
///
/// # Errors
/// [`Error::Config`] for an order that is not a power of two ≥ 2, or the coherent
/// detection with $M \ne 2$.
///
/// # Examples
/// ```
/// use com_mapping::differential::Detection;
/// use com_mapping::theory::{ber_dbpsk_coherent, ber_differential, ber_dpsk};
/// assert_eq!(ber_differential(4, 8.0, Detection::Noncoherent)?, ber_dpsk(4, 8.0)?);
/// assert_eq!(ber_differential(2, 6.0, Detection::Coherent)?, ber_dbpsk_coherent(6.0));
/// assert!(ber_differential(4, 6.0, Detection::Coherent).is_err());
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn ber_differential(m: usize, ebn0_db: f64, detection: crate::differential::Detection) -> Result<f64> {
    use crate::differential::Detection;
    match detection {
        Detection::Noncoherent => ber_dpsk(m, ebn0_db),
        Detection::Coherent if m == 2 => Ok(ber_dbpsk_coherent(ebn0_db)),
        Detection::Coherent => Err(Error::Config(format!(
            "coherent detection with differential decoding is defined for m = 2, got m = {m}"
        ))),
    }
}

/// Symbol error rate of $M$ orthogonal signals of equal energy detected noncoherently
/// over AWGN (noncoherent M-FSK, chirp spread spectrum), as a function of $E_{s}/N_{0}$
/// (dB): the magnitude of the matched filter of the sent signal is Rice of location
/// $v = \sqrt{2 E_{s}/N_{0}}$, the $M - 1$ others Rayleigh, all of unit scale, and the
/// largest wins (Proakis & Salehi, §4.5; Afisiadis et al. 2020, eq. (16)):
///
/// $$
/// P_{s} = \int_{0}^{\infty} y\, e^{-(y^{2} + v^{2})/2} I_{0}(v y)
/// \left[1 - \left(1 - e^{-y^{2}/2}\right)^{M - 1}\right] dy,
/// $$
///
/// by Simpson's rule on $v \pm 12$, which avoids the cancellation of the alternating
/// binomial sum for large $M$.
///
/// # Errors
/// [`Error::Config`] if `m` is below 2.
///
/// # Examples
/// ```
/// use com_mapping::theory::ser_orthogonal_noncoherent;
/// // binary: e^{-Es/(2 N0)} / 2
/// let g = com_math::db2lin(9.0);
/// assert!((ser_orthogonal_noncoherent(2, 9.0)? - 0.5 * (-g / 2.0).exp()).abs() < 1e-9);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn ser_orthogonal_noncoherent(m: usize, esn0_db: f64) -> Result<f64> {
    ensure(m >= 2, || format!("M orthogonal signals need M ≥ 2, got {m}"))?;
    // 1 - (1 - e^{-y²/2})^n without cancellation
    let one_minus_power = |y: f64, n: f64| {
        let p = (-y * y / 2.0).exp();
        if p >= 1.0 { 1.0 } else { -((n * (-p).ln_1p()).exp_m1()) }
    };
    let v = (2.0 * db2lin(esn0_db)).sqrt();
    let rice = |y: f64| y * (-(y - v) * (y - v) / 2.0).exp() * bessel_i0e(y * v);
    let (lo, hi) = ((v - 12.0).max(0.0), v + 12.0);
    let others = (m - 1) as f64;
    Ok(com_math::simpson(|y| rice(y) * one_minus_power(y, others), lo, hi, 4800).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_quadrature_has_converged() {
        let c = Constellation::qam(16).unwrap();
        for fading in [Fading::Awgn, Fading::Rayleigh] {
            let at = |n| constellation_mi(&c, 10.0, None, fading, n).unwrap();
            assert!((at(DEFAULT_QUADRATURE_POINTS) - at(80)).abs() < 1e-6, "{fading:?}");
        }
    }

    #[test]
    fn bpsk_theory_is_q_of_sqrt_2ebn0() {
        for eb in [0.0, 4.0, 8.0] {
            let g = db2lin(eb);
            let q = qfunc((2.0 * g).sqrt());
            assert!((ber(ConstellationKind::Psk, 2, eb).unwrap() / q - 1.0).abs() < 1e-9);
            assert!((ser(ConstellationKind::Psk, 2, eb).unwrap() / q - 1.0).abs() < 1e-9);
            assert!((ber(ConstellationKind::Qam, 4, eb).unwrap() / q - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn closed_forms_reject_orders_they_do_not_cover() {
        for (kind, m) in [
            (ConstellationKind::Qam, 8),
            (ConstellationKind::Qam, 32),
            (ConstellationKind::Pam, 6),
            (ConstellationKind::Psk, 1),
        ] {
            assert!(ber(kind, m, 10.0).is_err() && ser(kind, m, 10.0).is_err(), "{kind:?} {m}");
        }
    }

    #[test]
    fn dpsk_closed_forms() {
        // the integral reproduces the closed form of DBPSK at every SNR
        for g in [0.1f64, 1.0, 5.0, 20.0] {
            let want = 0.5 * (-g).exp();
            assert!((differential_phase_tail(std::f64::consts::FRAC_PI_2, g) / want - 1.0).abs() < 1e-8, "{g}");
        }
        // DQPSK: the sectors sum to one and the rate lies between QPSK and its double at 3 dB
        // more (the classical loss of differential detection)
        for eb in [4.0, 8.0, 12.0] {
            let gs = 2.0 * db2lin(eb);
            let t = |psi: f64| differential_phase_tail(psi, gs);
            let want = 0.5 * (t(std::f64::consts::FRAC_PI_4) + t(3.0 * std::f64::consts::FRAC_PI_4));
            // the sector sums cancel 1 - ½T, hence the rounding at 1e-12 of a 1e-5 rate
            let got = ber_dpsk(4, eb).unwrap();
            assert!((got / want - 1.0).abs() < 1e-9, "{eb}: {got:e} vs {want:e}");
            let qpsk = ber(ConstellationKind::Psk, 4, eb).unwrap();
            let qpsk_3db = ber(ConstellationKind::Psk, 4, eb - 3.0).unwrap();
            assert!(want > qpsk && want < 2.0 * qpsk_3db, "{eb}: {want} against {qpsk} and {qpsk_3db}");
        }
        assert!(ber_dpsk(3, 1.0).is_err());
    }

    /// Exact error count of Gray-mapped `side`-PAM by enumeration of the decision
    /// regions, with the labels of [`com_math::gray`]: `x` is the half distance between
    /// levels over the noise standard deviation.
    fn pam_bit_errors_by_enumeration(side: usize, x: f64) -> f64 {
        let mut total = 0.0;
        for j in 0..side {
            for r in 0..side {
                let d = r as f64 - j as f64;
                let lo = if r == 0 { 1.0 } else { qfunc((2.0 * d - 1.0) * x) };
                let hi = if r == side - 1 { 0.0 } else { qfunc((2.0 * d + 1.0) * x) };
                let wrong = (com_math::gray(j) ^ com_math::gray(r)).count_ones();
                total += f64::from(wrong) * (lo - hi);
            }
        }
        total / side as f64
    }

    #[test]
    fn the_cho_yoon_sums_count_the_errors_of_the_gray_labels() {
        for side in [2usize, 4, 8, 16, 32, 64] {
            for x in [0.05, 0.3, 1.0, 2.5] {
                let (a, b) = (gray_pam_bit_errors(side, x), pam_bit_errors_by_enumeration(side, x));
                assert!((a - b).abs() < 1e-13, "{side}-PAM at x = {x}: {a} vs {b}");
            }
        }
        // rectangular QAM: two PAM in quadrature at the same distance
        for (mi, mj) in [(4usize, 2usize), (8, 4), (16, 8), (32, 16)] {
            let eb = 9.0;
            let log2m = f64::from(mi.trailing_zeros() + mj.trailing_zeros());
            let x = (6.0 * log2m * db2lin(eb) / ((mi * mi + mj * mj) as f64 - 2.0)).sqrt();
            let want = (pam_bit_errors_by_enumeration(mi, x) + pam_bit_errors_by_enumeration(mj, x)) / log2m;
            assert!((ber_gray_rect_qam(mi, mj, eb).unwrap() - want).abs() < 1e-14, "{mi} × {mj}");
        }
        // random decisions at low SNR, BPSK and QPSK
        assert!((ber_gray_pam(16, -60.0).unwrap() - 0.5).abs() < 1e-3);
        let q = qfunc((2.0 * db2lin(5.0)).sqrt());
        assert!((ber_gray_pam(2, 5.0).unwrap() - q).abs() < 1e-15);
        assert!((ber_gray_qam(4, 5.0).unwrap() - q).abs() < 1e-15);
        assert!(ber_gray_pam(6, 5.0).is_err() && ber_gray_rect_qam(8, 1, 5.0).is_err());
    }

    #[test]
    fn the_phase_density_integrates_to_the_psk_error_rates() {
        let integral = |a: f64, b: f64, rho: f64| {
            let n = 20_000;
            let h = (b - a) / n as f64;
            (0..n).map(|i| phase_pdf(a + (i as f64 + 0.5) * h, rho)).sum::<f64>() * h
        };
        let pi = std::f64::consts::PI;
        for rho in [0.1, 1.0, 4.0, 20.0] {
            // to the accuracy of the midpoint rule
            assert!((integral(-pi, pi, rho) - 1.0).abs() < 1e-6, "{rho}: {}", integral(-pi, pi, rho));
            // BPSK: an error is Re r < 0, Q(√(2ρ)); QPSK: 2Q(√ρ) - Q(√ρ)²
            let bpsk = 2.0 * integral(pi / 2.0, pi, rho);
            assert!((bpsk - qfunc((2.0 * rho).sqrt())).abs() < 1e-6, "{rho}: {bpsk}");
            let q = qfunc(rho.sqrt());
            let qpsk = 2.0 * integral(pi / 4.0, pi, rho);
            assert!((qpsk - (2.0 * q - q * q)).abs() < 1e-6, "{rho}: {qpsk}");
        }
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    /// K. Cho and D. Yoon, "On the general BER expression of one- and two-dimensional
    /// amplitude modulations," IEEE Trans. Commun. 50(7), 2002, Table I (p. 1076): the
    /// coefficients $[X_{I}(k)]$ and $[Y_{I}(k)]$ of eq. (9) for $I$ = 2, 4, 8 and 16.
    #[test]
    fn cho_2002_table_1() {
        let table: [(usize, usize, &[f64]); 10] = [
            (2, 1, &[1.0]),
            (4, 1, &[1.0, 1.0]),
            (4, 2, &[2.0, 1.0, -1.0]),
            (8, 1, &[1.0; 4]),
            (8, 2, &[2.0, 2.0, 1.0, 1.0, -1.0, -1.0]),
            (8, 3, &[4.0, 3.0, -3.0, -2.0, 2.0, 1.0, -1.0]),
            (16, 1, &[1.0; 8]),
            (16, 2, &[2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]),
            (16, 3, &[4.0, 4.0, 3.0, 3.0, -3.0, -3.0, -2.0, -2.0, 2.0, 2.0, 1.0, 1.0, -1.0, -1.0]),
            (16, 4, &[8.0, 7.0, -7.0, -6.0, 6.0, 5.0, -5.0, -4.0, 4.0, 3.0, -3.0, -2.0, 2.0, 1.0, -1.0]),
        ];
        for (side, k, xs) in table {
            let (x, y): (Vec<f64>, Vec<usize>) = cho_yoon_terms(side, k).unzip();
            assert_eq!(x, xs, "X_{side}({k})");
            // Y is 1, 3, 5, …, 2[(1 - 2^-k) I - 1] + 1
            assert_eq!(y, (0..xs.len()).map(|i| 2 * i + 1).collect::<Vec<_>>(), "Y_{side}({k})");
        }
    }

    /// Cho and Yoon 2002, Sec. II-C and Fig. 2: "an additional 4–5 dB of SNR is
    /// required to transmit an extra bit to maintain an average BER of 1e-3". The exact
    /// BER gives 4.25, 4.62 and 4.88 dB from 4- to 32-PAM; the step is 3.73 dB from 2- to
    /// 4-PAM and grows to 5.06 and 5.20 dB beyond (64- and 128-PAM).
    #[test]
    fn cho_2002_fig_2_cost_of_an_extra_bit() {
        let required = |m: usize| bisect(|e| ber_gray_pam(m, e).unwrap() > 1e-3, -5.0, 60.0, 100);
        let r: Vec<f64> = (1..=7).map(|b| required(1 << b)).collect();
        assert!((r[0] - 6.79).abs() < 0.01, "BPSK at 1e-3: {}", r[0]);
        let steps: Vec<f64> = r.windows(2).map(|w| w[1] - w[0]).collect();
        for (i, s) in steps.iter().enumerate() {
            assert!((3.6..5.3).contains(s), "step {i}: {s}");
        }
        assert!(steps[1..4].iter().all(|s| (4.0..5.0).contains(s)), "{steps:?}");
    }
}
