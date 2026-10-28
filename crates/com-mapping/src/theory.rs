//! Closed-form error rates and information rates of constellations over AWGN.
//!
//! - [`ber`] (Gray-mapped QAM, PSK and PAM, OptiCommPy `theoryBER`) and [`ser`] give the
//!   error rates as functions of $E_{b}/N_{0}$, and [`ebn0_for_ber`], [`ebn0_for_ser`]
//!   their inverses;
//! - [`ber_rayleigh_mrc`] gives the BER of BPSK and Gray QPSK with maximal-ratio
//!   combining of independent Rayleigh branches;
//! - [`mi`] and [`gmi`] give the symbol-wise and bit-wise mutual information by
//!   Gauss–Hermite quadrature (OptiCommPy `theoryMI`, `theoryGMI`);
//! - [`ber_dpsk`], [`ber_dbpsk_coherent`] and [`differential_phase_tail`] give the error
//!   rates of differential PSK ([`crate::differential`]).
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
use com_math::{C64, bisect, db2lin, entropy, log_sum_exp, qfunc};

/// Theoretical bit error rate over AWGN with Gray mapping (OptiCommPy `theoryBER`).
///
/// - QAM (`L = √M`): `P_b = 2(1 - 1/L)/log2 L · Q(√(3 log2 L/(L²-1) · 2 E_b/N_0))`;
/// - PSK: `P_b = 2 Q(√(2 log2 M · E_b/N_0) sin(π/M)) / log2 M`, and the exact
///   `Q(√(2 E_b/N_0))` for `M = 2` (OptiCommPy applies the general formula, which counts
///   two nearest neighbours and doubles the BER of BPSK);
/// - PAM: `P_b = 2(M-1)/M · Q(√(6 log2 M/(M²-1) · E_b/N_0)) / log2 M`.
///
/// # Errors
/// [`Error::Config`] for another family.
pub fn ber(kind: ConstellationKind, m: usize, ebn0_db: f64) -> Result<f64> {
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
/// [`Error::Config`] for another family.
pub fn ser(kind: ConstellationKind, m: usize, ebn0_db: f64) -> Result<f64> {
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
/// with $L$ the number of branches (at least 1); $L = 1$ is the flat Rayleigh channel,
/// $P_{b} = \frac{1}{2}(1 - \mu)$.
///
/// # Examples
/// ```
/// use com_mapping::theory::ber_rayleigh_mrc;
/// let g = com_math::db2lin(10.0);
/// assert!((ber_rayleigh_mrc(g, 1) - 0.5 * (1.0 - (g / (1.0 + g)).sqrt())).abs() < 1e-15);
/// // diversity: four branches at 10 dB each
/// assert!(ber_rayleigh_mrc(g, 4) < 1e-5);
/// ```
pub fn ber_rayleigh_mrc(mean_ebn0: f64, branches: usize) -> f64 {
    let mu = (mean_ebn0 / (1.0 + mean_ebn0)).sqrt();
    let l = branches.max(1);
    let mut binom = 1.0;
    let mut sum = 0.0;
    for k in 0..l {
        if k > 0 {
            binom *= (l - 1 + k) as f64 / k as f64;
        }
        sum += binom * ((1.0 + mu) / 2.0).powi(k as i32);
    }
    ((1.0 - mu) / 2.0).powi(l as i32) * sum
}

/// Quadrature nodes `(w, weight)` of the noise, `E f(w) ≈ Σ weight f(w)`.
///
/// Complex channel: `w ~ CN(0, σ²)`, product rule on the two dimensions; real channel:
/// `w ~ N(0, σ²)` on the real axis.
fn noise_nodes(sigma2: f64, real: bool, points: usize) -> Vec<(C64, f64)> {
    let (t, a) = com_math::gauss_hermite(points.max(2));
    let pi = std::f64::consts::PI;
    if real {
        let s = (2.0 * sigma2).sqrt();
        t.iter().zip(&a).map(|(t, a)| (C64::new(s * t, 0.0), a / pi.sqrt())).collect()
    } else {
        let s = sigma2.sqrt();
        let mut v = Vec::with_capacity(t.len() * t.len());
        for (tr, ar) in t.iter().zip(&a) {
            for (ti, ai) in t.iter().zip(&a) {
                v.push((C64::new(s * tr, s * ti), ar * ai / pi));
            }
        }
        v
    }
}

/// Normalised points, probabilities, `N_0` and quadrature nodes of a theory function.
type TheorySetup = (Vec<C64>, Vec<f64>, f64, Vec<(C64, f64)>);

/// The normalised points, the probabilities and the quadrature of a theory function.
fn theory_setup(
    kind: ConstellationKind,
    m: usize,
    snr_db: f64,
    px: Option<&[f64]>,
    points: usize,
) -> Result<TheorySetup> {
    let c = Constellation::new(kind, m)?;
    let p = c.priors(px)?;
    let es: f64 = c.points().iter().zip(&p).map(|(x, q)| q * x.norm_sqr()).sum();
    let pts: Vec<C64> = c.points().iter().map(|x| x / es.sqrt()).collect();
    // OptiCommPy: a real constellation sees a real channel of variance 1/SNR, a complex
    // one a complex channel of variance 1/SNR (1/(2 SNR) per dimension)
    let sigma2 = db2lin(-snr_db);
    let n0 = if kind.is_real() { 2.0 * sigma2 } else { sigma2 };
    Ok((pts, p, n0, noise_nodes(sigma2, kind.is_real(), points)))
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
    let (x, p, n0, nodes) = theory_setup(kind, m, snr_db, px, points)?;
    let lp = log_priors(&p);
    let mut acc = 0.0;
    let mut terms = vec![0.0; x.len()];
    for (i, xi) in x.iter().enumerate() {
        if p[i] == 0.0 {
            continue;
        }
        let mut e = 0.0;
        for (w, a) in &nodes {
            for (j, xj) in x.iter().enumerate() {
                terms[j] = lp[j] - ((xi - xj + w).norm_sqr() - w.norm_sqr()) / n0;
            }
            e += a * log_sum_exp(&terms);
        }
        acc += p[i] * e;
    }
    Ok(-acc / std::f64::consts::LN_2)
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
    let (x, p, n0, nodes) = theory_setup(kind, m, snr_db, px, points)?;
    let b = m.trailing_zeros() as usize;
    let lp = log_priors(&p);
    // GMI = H(X) - Σ_k H(B_k | Y): with shaped (dependent) bits, the sum of the
    // bit-wise mutual informations Σ_k I(B_k; Y) can exceed H(X) and is not achievable
    let mut gmi = entropy(&p);
    let mut all = vec![0.0; m];
    for k in 0..b {
        let bit = |j: usize| (j >> (b - 1 - k)) & 1;
        let mut h_cond = 0.0;
        for (i, xi) in x.iter().enumerate() {
            if p[i] == 0.0 {
                continue;
            }
            let bi = bit(i);
            let mut e = 0.0;
            for (w, a) in &nodes {
                for (j, xj) in x.iter().enumerate() {
                    all[j] = lp[j] - ((xi - xj + w).norm_sqr() - w.norm_sqr()) / n0;
                }
                let same: Vec<f64> = (0..m).filter(|&j| bit(j) == bi).map(|j| all[j]).collect();
                // -log2 P(b_k = b_k(i) | y)
                e += a * (log_sum_exp(&all) - log_sum_exp(&same));
            }
            h_cond += p[i] * e;
        }
        gmi -= h_cond / std::f64::consts::LN_2;
    }
    Ok(gmi)
}

/// The probability that the phase angle between two vectors of the same SNR `es_n0`
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
pub fn differential_phase_tail(psi: f64, es_n0: f64) -> f64 {
    let psi = psi.clamp(0.0, std::f64::consts::PI);
    if psi == 0.0 {
        return 1.0;
    }
    let (s, c) = psi.sin_cos();
    let n = 2000usize;
    let h = std::f64::consts::PI / n as f64;
    let f = |t: f64| {
        let d = 1.0 - c * t.cos();
        (-es_n0 * d).exp() / d
    };
    let mut acc = f(-std::f64::consts::FRAC_PI_2) + f(std::f64::consts::FRAC_PI_2);
    for k in 1..n {
        let t = -std::f64::consts::FRAC_PI_2 + k as f64 * h;
        acc += if k % 2 == 1 { 4.0 } else { 2.0 } * f(t);
    }
    s / (2.0 * std::f64::consts::PI) * acc * h / 3.0
}

/// Exact BER of Gray-mapped square `m`-QAM over AWGN (K. Cho and D. Yoon, "On the
/// general BER expression of one- and two-dimensional amplitude modulations," IEEE
/// Trans. Commun. 50(7), 2002, eq. (16)), as a function of `E_b/N_0` in dB. Unlike the
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
    let levels = side.trailing_zeros() as usize;
    let es = db2lin(ebn0_db) * (m as f64).log2();
    let x = (3.0 * es / (m as f64 - 1.0)).sqrt();
    let mut total = 0.0;
    for k in 1..=levels {
        let step = 1usize << (k - 1);
        let count = side - side / (1 << k);
        let mut pk = 0.0;
        for i in 0..count {
            let sign = if (i * step / side) % 2 == 0 { 1.0 } else { -1.0 };
            let weight = step as f64 - ((i * step) as f64 / side as f64 + 0.5).floor();
            pk += sign * weight * 2.0 * qfunc((2 * i + 1) as f64 * x);
        }
        total += pk / side as f64;
    }
    Ok(total / levels as f64)
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
    let gray = crate::constellation::gray_code(bits);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
