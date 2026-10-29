//! Capacity of a MIMO channel realisation.
//!
//! With the total transmit power $P$, the receiver noise variance $\sigma^{2}$ and
//! $\rho = P / \sigma^{2}$:
//!
//! - without channel knowledge at the transmitter, equal power on every antenna:
//!
//! $$
//! C = \log_{2} \det\left(I_{n_{r}} + \frac{\rho}{n_{t}} H H^{H}\right);
//! $$
//!
//! - with channel knowledge, water-filling on the eigenmodes:
//!   $C = \sum_{i} \log_{2}(1 + p_{i} \sigma_{i}^{2})$ (see [`crate::precoding`]).
//!
//! The ergodic capacity of a fading channel is the mean of these over the channel
//! realisations. Over the i.i.d. Rayleigh channel (entries $\mathcal{CN}(0, 1)$) it has
//! the closed form of Telatar (1999, Theorem 2), [`ergodic_rayleigh`]: with
//! $m = \min(n_{r}, n_{t})$, $n = \max(n_{r}, n_{t})$ and the associated Laguerre
//! polynomials $L_{k}^{n-m}$,
//!
//! $$
//! C = \int_{0}^{\infty} \log_{2}\left(1 + \frac{\rho}{n_{t}} \lambda\right)
//! \sum_{k=0}^{m-1} \frac{k!}{(k+n-m)!} \left[L_{k}^{n-m}(\lambda)\right]^{2}
//! \lambda^{n-m} e^{-\lambda} \, d\lambda ,
//! $$
//!
//! the sum being $m$ times the density of an unordered eigenvalue of the Wishart matrix
//! $H H^{H}$ or $H^{H} H$ ([`wishart_eigenvalue_density`]). [`ergodic_rayleigh_limit`]
//! is its slope for $n_{r} = n_{t} \to \infty$ (eq. (13)). When $H$ is drawn once and
//! held, the rate $R$ is in outage with the probability [`outage_rayleigh`]
//! (eq. (16) and Example 7, for one antenna on one side), and [`outage_capacity_rayleigh`]
//! is the largest rate of outage probability $\varepsilon$ (eqs. (17), (18)).
//!
//! # Examples
//! ```
//! use com_mimo::capacity::{equal_power, water_filling};
//! use com_math::linalg::identity;
//! // two parallel unit channels at ρ = 10: 2 log2(1 + 5)
//! let h = identity(2);
//! assert!((equal_power(&h.view(), 10.0)? - 2.0 * 6f64.log2()).abs() < 1e-12);
//! assert!((water_filling(&h.view(), 10.0)? - 2.0 * 6f64.log2()).abs() < 1e-12);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. E. Telatar, "Capacity of multi-antenna Gaussian channels," *Eur. Trans.
//!    Telecommun.*, 10(6), 1999.
//! 2. G. J. Foschini and M. J. Gans, "On limits of wireless communications in a fading
//!    environment when using multiple antennas," *Wireless Pers. Commun.*, 6(3), 1998.

use crate::precoding;
use com_core::{Result, ensure};
use com_math::linalg::{hermitian, identity, log2_det_hpd, svd};
use com_math::{C64, bisect, gamma_p, ln_gamma, simpson};
use ndarray::ArrayView2;

/// Capacity with equal power on the $n_{t}$ antennas (bits per channel use).
///
/// # Errors
/// [`com_core::Error::Config`] for a negative SNR.
pub fn equal_power(h: &ArrayView2<'_, C64>, snr_lin: f64) -> Result<f64> {
    ensure(snr_lin >= 0.0, || format!("the SNR must be non-negative, got {snr_lin}"))?;
    let a = identity(h.nrows()) + h.dot(&hermitian(h)).mapv(|x| x * (snr_lin / h.ncols() as f64));
    log2_det_hpd(&a.view())
}

/// Capacity with water-filling on the eigenmodes (bits per channel use).
///
/// # Errors
/// [`com_core::Error::Config`] for a non-positive SNR.
pub fn water_filling(h: &ArrayView2<'_, C64>, snr_lin: f64) -> Result<f64> {
    let (_, s, _) = svd(h);
    let g: Vec<f64> = s.iter().map(|x| x * x).collect();
    let p = precoding::water_filling_powers(&g, snr_lin)?;
    Ok(g.iter().zip(&p).map(|(g, p)| (1.0 + g * p).log2()).sum())
}

/// The density of one unordered eigenvalue of the complex Wishart matrix $H H^{H}$,
/// $H$ of size $m \times n$ ($m \le n$) with i.i.d. $\mathcal{CN}(0, 1)$ entries
/// (Telatar 1999, eq. preceding (8)):
///
/// $$
/// p(\lambda) = \frac{1}{m} \sum_{k=0}^{m-1} \frac{k!}{(k+n-m)!}
/// \left[L_{k}^{n-m}(\lambda)\right]^{2} \lambda^{n-m} e^{-\lambda} .
/// $$
///
/// The normalized Laguerre functions
/// $\ell_{k} = \sqrt{k!/(k+\alpha)!}\, L_{k}^{\alpha}(\lambda)\, \lambda^{\alpha/2}
/// e^{-\lambda/2}$, $\alpha = n - m$, are computed by the three-term recurrence
/// $\sqrt{(k+1)(k+1+\alpha)}\, \ell_{k+1} = (2k+1+\alpha-\lambda)\, \ell_{k} -
/// \sqrt{k(k+\alpha)}\, \ell_{k-1}$, which stays in range where the polynomials
/// themselves overflow. `m` and `n` are swapped if `m > n`.
///
/// # Examples
/// ```
/// use com_mimo::capacity::wishart_eigenvalue_density;
/// // one antenna on each side: |h|² is exponential
/// assert!((wishart_eigenvalue_density(1, 1, 2.0) - (-2f64).exp()).abs() < 1e-15);
/// ```
pub fn wishart_eigenvalue_density(m: usize, n: usize, lambda: f64) -> f64 {
    let (m, n) = (m.min(n), m.max(n));
    if m == 0 || lambda < 0.0 {
        return 0.0;
    }
    let alpha = (n - m) as f64;
    let power = if alpha == 0.0 { 0.0 } else { 0.5 * alpha * lambda.ln() };
    let mut prev = 0.0;
    let mut cur = (power - 0.5 * lambda - 0.5 * ln_gamma(alpha + 1.0)).exp();
    let mut sum = cur * cur;
    for k in 0..m - 1 {
        let k = k as f64;
        let next = ((2.0 * k + 1.0 + alpha - lambda) * cur - (k * (k + alpha)).sqrt() * prev)
            / ((k + 1.0) * (k + 1.0 + alpha)).sqrt();
        (prev, cur) = (cur, next);
        sum += cur * cur;
    }
    sum / m as f64
}

/// Ergodic capacity (bits per channel use) of the i.i.d. Rayleigh channel with `n_rx`
/// receive and `n_tx` transmit antennas, the total SNR `snr_lin` (linear) spread equally
/// on the transmit antennas, the channel known at the receiver only: Telatar (1999),
/// Theorem 2, eq. (8), integrated by Simpson's rule in $\lambda$ up to well beyond the
/// largest eigenvalue after the change of variable $\lambda = u^{2}$ (the Laguerre
/// functions oscillate with a nearly constant period in $u$), to about $10^{-10}$
/// relative. It is the mean of
/// [`equal_power`] over the realisations of $H$.
///
/// # Errors
/// [`com_core::Error::Config`] for no antenna or a negative SNR.
///
/// # Examples
/// ```
/// use com_mimo::capacity::ergodic_rayleigh;
/// use com_math::expint_e1;
/// // one antenna: e^{1/ρ} E₁(1/ρ) nats
/// let rho = 10.0;
/// let c = ergodic_rayleigh(1, 1, rho)?;
/// assert!((c * 2f64.ln() - (1.0 / rho).exp() * expint_e1(1.0 / rho)).abs() < 1e-9);
/// // four antennas on each side: about four times as much at high SNR
/// assert!(ergodic_rayleigh(4, 4, 1e3)? > 3.7 * ergodic_rayleigh(1, 1, 1e3)?);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn ergodic_rayleigh(n_rx: usize, n_tx: usize, snr_lin: f64) -> Result<f64> {
    ensure(n_rx >= 1 && n_tx >= 1, || format!("need at least one antenna on each side, got {n_rx} × {n_tx}"))?;
    ensure(snr_lin >= 0.0, || format!("the SNR must be non-negative, got {snr_lin}"))?;
    let (m, n) = (n_rx.min(n_tx), n_rx.max(n_tx));
    // the largest eigenvalue concentrates at (√n + √m)², with fluctuations of order
    // √n + √m: the density is below e^{-40} beyond the upper limit
    let edge = (n as f64).sqrt() + (m as f64).sqrt();
    let upper = edge * edge + 10.0 * edge + 60.0;
    // λ = u²: the Laguerre functions oscillate with a nearly constant period in u
    let intervals = 400 + (120.0 * (upper * (m + n) as f64).sqrt()) as usize;
    let g = snr_lin / n_tx as f64;
    let c = simpson(
        |u| 2.0 * u * (g * u * u).ln_1p() * wishart_eigenvalue_density(m, n, u * u),
        0.0,
        upper.sqrt(),
        intervals,
    );
    Ok(m as f64 * c / std::f64::consts::LN_2)
}

/// The limit of $C / n$ (bits per channel use and per antenna) of
/// [`ergodic_rayleigh`] with $n$ antennas on each side, $n \to \infty$, at the total
/// SNR $\rho$ (Telatar 1999, eq. (13), from the quarter-circle law of the eigenvalues
/// of $H H^{H} / n$):
///
/// $$
/// \lim_{n \to \infty} \frac{C}{n} = \left[\ln \rho - 1 + \frac{\sqrt{1 + 4\rho} - 1}{2\rho} +
/// 2 \tanh^{-1} \frac{1}{\sqrt{1 + 4\rho}}\right] \frac{1}{\ln 2} .
/// $$
///
/// # Examples
/// ```
/// use com_mimo::capacity::{ergodic_rayleigh, ergodic_rayleigh_limit};
/// // twenty antennas are already within 0.5 % of the limit at 20 dB
/// let c = ergodic_rayleigh(20, 20, 100.0)? / 20.0;
/// assert!((c / ergodic_rayleigh_limit(100.0) - 1.0).abs() < 5e-3);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn ergodic_rayleigh_limit(snr_lin: f64) -> f64 {
    let s = (1.0 + 4.0 * snr_lin).sqrt();
    (snr_lin.ln() - 1.0 + (s - 1.0) / (2.0 * snr_lin) + 2.0 * (1.0 / s).atanh()) / std::f64::consts::LN_2
}

fn ensure_single_side(n_rx: usize, n_tx: usize) -> Result<()> {
    ensure(n_rx >= 1 && n_tx >= 1 && n_rx.min(n_tx) == 1, || {
        format!("the closed form needs one antenna on one side, got {n_rx} × {n_tx}")
    })
}

/// The outage probability $P(\log_{2} \det(I + \frac{\rho}{n_{t}} H H^{H}) < R)$ of a
/// rate `rate` (bits per channel use) over a Rayleigh channel drawn once, with equal
/// power on the transmit antennas and one antenna on one side. $\lVert h \rVert^{2}$ is
/// then a gamma variable of shape $k = \max(n_{r}, n_{t})$, and
///
/// $$
/// P_{\mathrm{out}} = P\left(k, \frac{n_{t} (2^{R} - 1)}{\rho}\right),
/// $$
///
/// the regularized incomplete gamma function [`com_math::gamma_p`] (Telatar 1999, eq.
/// (16) for $n_{t} = 1$ and Example 7 for $n_{r} = 1$, there in nats).
///
/// # Errors
/// [`com_core::Error::Config`] unless one side has a single antenna, or for a
/// non-positive SNR.
///
/// # Examples
/// ```
/// use com_mimo::capacity::outage_rayleigh;
/// // one antenna: 1 - e^{-(2^R - 1)/ρ}
/// let p = outage_rayleigh(1, 1, 10.0, 1.0)?;
/// assert!((p - (1.0 - (-0.1f64).exp())).abs() < 1e-15);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn outage_rayleigh(n_rx: usize, n_tx: usize, snr_lin: f64, rate: f64) -> Result<f64> {
    ensure_single_side(n_rx, n_tx)?;
    ensure(snr_lin > 0.0, || format!("the SNR must be positive, got {snr_lin}"))?;
    let k = n_rx.max(n_tx) as f64;
    Ok(gamma_p(k, n_tx as f64 * (rate * std::f64::consts::LN_2).exp_m1() / snr_lin))
}

/// The $\varepsilon$-capacity (bits per channel use) of [`outage_rayleigh`]: the rate
/// $R$ whose outage probability is `eps` (Telatar 1999, eqs. (17) and (18), Figs. 3 and
/// 5), $R = \log_{2}(1 + \rho x / n_{t})$ with $P(k, x) = \varepsilon$ solved by
/// bisection.
///
/// # Errors
/// [`com_core::Error::Config`] unless one side has a single antenna, for a
/// non-positive SNR or for `eps` outside $(0, 1)$.
///
/// # Examples
/// ```
/// use com_mimo::capacity::{outage_capacity_rayleigh, outage_rayleigh};
/// let r = outage_capacity_rayleigh(4, 1, 100.0, 1e-3)?;
/// assert!((outage_rayleigh(4, 1, 100.0, r)? / 1e-3 - 1.0).abs() < 1e-9);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn outage_capacity_rayleigh(n_rx: usize, n_tx: usize, snr_lin: f64, eps: f64) -> Result<f64> {
    ensure_single_side(n_rx, n_tx)?;
    ensure(snr_lin > 0.0, || format!("the SNR must be positive, got {snr_lin}"))?;
    ensure(eps > 0.0 && eps < 1.0, || format!("the outage probability must be in (0, 1), got {eps}"))?;
    let k = n_rx.max(n_tx) as f64;
    // P(k, x) < ε on the left of the quantile; search ln x for the dynamic range
    let ln_x = bisect(|u| gamma_p(k, u.exp()) < eps, -800.0, 10.0 + 2.0 * k.ln() + k.sqrt(), 200);
    Ok((snr_lin * ln_x.exp() / n_tx as f64).ln_1p() / std::f64::consts::LN_2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use ndarray::Array2;

    #[test]
    fn water_filling_never_loses_and_matches_at_high_snr() {
        let mut r = Rng::new(1);
        for _ in 0..50 {
            let h = Array2::from_shape_fn((3, 3), |_| r.complex_gaussian());
            for snr in [0.1, 1.0, 100.0] {
                assert!(water_filling(&h.view(), snr).unwrap() >= equal_power(&h.view(), snr).unwrap() - 1e-9);
            }
            let hi = 1e6;
            let gap = water_filling(&h.view(), hi).unwrap() - equal_power(&h.view(), hi).unwrap();
            assert!(gap < 0.01, "{gap}");
        }
    }

    #[test]
    fn the_wishart_density_is_normalized_with_the_mean_of_the_trace() {
        // ∫ p = 1 and ∫ λ p = n (E tr W = m n)
        for (m, n) in [(1, 1), (1, 4), (3, 3), (2, 7), (8, 8)] {
            let up = 4.0 * (m + n) as f64 + 80.0;
            let mass = com_math::simpson(|l| wishart_eigenvalue_density(m, n, l), 0.0, up, 20_000);
            let mean = com_math::simpson(|l| l * wishart_eigenvalue_density(m, n, l), 0.0, up, 20_000);
            assert!((mass - 1.0).abs() < 1e-8, "{m} × {n}: {mass}");
            assert!((mean / n as f64 - 1.0).abs() < 1e-8, "{m} × {n}: {mean}");
        }
    }

    #[test]
    fn the_ergodic_integral_has_converged() {
        // against the same integral on four times as many intervals
        for (nr, nt, snr) in [(1usize, 1usize, 1e-2), (8, 8, 100.0), (3, 20, 1e4), (32, 32, 10.0)] {
            let (m, n) = (nr.min(nt), nr.max(nt));
            let up = ((n as f64).sqrt() + (m as f64).sqrt()).powi(2) * 2.0 + 200.0;
            let g = snr / nt as f64;
            let fine = com_math::simpson(
                |u| 2.0 * u * (g * u * u).ln_1p() * wishart_eigenvalue_density(m, n, u * u),
                0.0,
                up.sqrt(),
                100_000,
            ) * m as f64
                / std::f64::consts::LN_2;
            let c = ergodic_rayleigh(nr, nt, snr).unwrap();
            assert!((c / fine - 1.0).abs() < 1e-10, "{nr} × {nt}: {c} vs {fine}");
        }
    }

    #[test]
    fn the_ergodic_capacity_is_the_mean_of_the_realisations() {
        let mut r = Rng::new(7);
        for (nr, nt, snr) in [(2usize, 2usize, 10.0), (4, 2, 3.0), (2, 4, 100.0), (1, 3, 10.0)] {
            let n = 20_000;
            let v: Vec<f64> = (0..n)
                .map(|_| {
                    let h = Array2::from_shape_fn((nr, nt), |_| r.complex_gaussian());
                    equal_power(&h.view(), snr).unwrap()
                })
                .collect();
            let mean = v.iter().sum::<f64>() / n as f64;
            let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n * (n - 1)) as f64).sqrt();
            let c = ergodic_rayleigh(nr, nt, snr).unwrap();
            assert!((c - mean).abs() < 4.0 * sd, "{nr} × {nt}: {c} vs {mean} ± {sd}");
        }
    }

    #[test]
    fn the_outage_probability_is_that_of_the_realisations() {
        let mut r = Rng::new(3);
        for (nr, nt, snr, rate) in [(1usize, 1usize, 10.0, 2.0), (4, 1, 3.0, 2.5), (1, 3, 30.0, 4.0)] {
            let n = 50_000;
            let hits = (0..n)
                .filter(|_| {
                    let h = Array2::from_shape_fn((nr, nt), |_| r.complex_gaussian());
                    equal_power(&h.view(), snr).unwrap() < rate
                })
                .count();
            let p = outage_rayleigh(nr, nt, snr, rate).unwrap();
            let sd = (p * (1.0 - p) / n as f64).sqrt();
            assert!((hits as f64 / n as f64 - p).abs() < 4.0 * sd, "{nr} × {nt}: {hits} vs {p}");
        }
        assert!(outage_rayleigh(2, 2, 10.0, 1.0).is_err());
        assert!(outage_capacity_rayleigh(1, 2, 10.0, 1.0).is_err());
    }

    #[test]
    fn the_capacity_of_a_rank_one_channel_is_one_mode() {
        // H = a b^H: one mode of gain ‖a‖² ‖b‖²
        let a = ndarray::arr2(&[[C64::new(1.0, 0.0)], [C64::new(0.0, 1.0)]]);
        let b = ndarray::arr2(&[[C64::new(1.0, 0.0), C64::new(1.0, 0.0)]]);
        let h = a.dot(&b);
        let c = water_filling(&h.view(), 5.0).unwrap();
        assert!((c - (1.0 + 5.0 * 4.0f64).log2()).abs() < 1e-12);
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    const NATS: f64 = std::f64::consts::LN_2;

    /// E. Telatar, "Capacity of multi-antenna Gaussian channels," Eur. Trans. Telecommun.
    /// 10(6), 1999, Remark 3 (the reciprocity $C(a, b, Pb) = C(b, a, Pa)$, while
    /// $C(a, b, P) \ne C(b, a, P)$) and Examples 3 and 4: with one transmit antenna the
    /// capacity follows $\ln(1 + Pr)$ as $r$ grows, with one receive antenna it tends to
    /// $\ln(1 + P)$ as $t$ grows.
    #[test]
    fn telatar_1999_remark_3_and_examples_3_4() {
        for (a, b, p) in [(1usize, 4usize, 10.0), (2, 5, 3.0), (3, 6, 100.0)] {
            let left = ergodic_rayleigh(a, b, p * b as f64).unwrap();
            let right = ergodic_rayleigh(b, a, p * a as f64).unwrap();
            assert!((left - right).abs() < 1e-9 * left, "{a}, {b}: {left} vs {right}");
            assert!((ergodic_rayleigh(a, b, p).unwrap() - ergodic_rayleigh(b, a, p).unwrap()).abs() > 0.1);
        }
        let p = 10.0;
        let gap = |r: usize| ergodic_rayleigh(r, 1, p).unwrap() * NATS - (p * r as f64).ln_1p();
        assert!(gap(4).abs() > gap(16).abs() && gap(16).abs() > gap(64).abs() && gap(64).abs() < 0.01, "{}", gap(64));
        let lim = ergodic_rayleigh(1, 200, p).unwrap() * NATS;
        assert!((lim - p.ln_1p()).abs() < 0.01, "{lim}");
    }

    /// Telatar 1999, Fig. 1 ($1 \le r, t \le 20$, $P = 20$ dB): its five contours are
    /// labelled 12.7, 25.3, 38, 50.7 and 63.4 nats, the multiples $k\, C_{\max} / 6$ of
    /// the largest value, $C(20, 20) = 76.0$ nats by eq. (8); Fig. 2 and eq. (13): with
    /// $r = t$ the capacity is nearly linear in $r$, $C(20, 20) / 20$ within 0.5 % of the
    /// limit (13) at every $P$ from 0 to 35 dB.
    #[test]
    fn telatar_1999_figs_1_2_and_eq_13() {
        let p = 100.0;
        let cmax = ergodic_rayleigh(20, 20, p).unwrap() * NATS;
        for (k, label) in [(1.0, 12.7), (2.0, 25.3), (3.0, 38.0), (4.0, 50.7), (5.0, 63.4)] {
            assert!((k * cmax / 6.0 - label).abs() < 0.05, "{k}: {}", k * cmax / 6.0);
        }
        for t in [1usize, 7, 13, 19] {
            assert!(ergodic_rayleigh(20, t, p).unwrap() * NATS < cmax);
        }
        for p_db in [0.0, 10.0, 20.0, 35.0] {
            let p = com_math::db2lin(p_db);
            let c = ergodic_rayleigh(20, 20, p).unwrap() / 20.0;
            assert!((c / ergodic_rayleigh_limit(p) - 1.0).abs() < 5e-3, "{p_db} dB: {c}");
        }
    }

    /// Telatar 1999, Fig. 3 (the $\varepsilon$-capacity of eq. (17) for $t = 1$): the
    /// ends of the curves at $r = 10$ read on panel (c), $P = 30$ dB, 8.85 nats for
    /// $\varepsilon = 10^{-1}$ and 7.75 nats for $10^{-4}$ (eq. (17) gives 8.74 and
    /// 7.70); Fig. 4 (one receive antenna): at $P = 15$ dB the curves of
    /// $t = 1, \dots, 10$ cross near $P(\Psi \le R) \approx 0.6$ at
    /// $R \approx 3.5$ nats $= \ln(1 + P)$, more transmit antennas being better below the
    /// crossing.
    #[test]
    fn telatar_1999_figs_3_4_outage() {
        let p = 1e3;
        let hi = outage_capacity_rayleigh(10, 1, p, 0.1).unwrap() * NATS;
        let lo = outage_capacity_rayleigh(10, 1, p, 1e-4).unwrap() * NATS;
        assert!((hi - 8.85).abs() < 0.15 && (lo - 7.75).abs() < 0.1, "{hi}, {lo}");
        let p = com_math::db2lin(15.0);
        let rate = p.ln_1p() / NATS;
        let probs: Vec<f64> = (1..=10).map(|t| outage_rayleigh(1, t, p, rate).unwrap()).collect();
        for w in probs.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.05, "{probs:?}");
        }
        assert!((probs[9] - 0.6).abs() < 0.1, "{probs:?}");
        let low = 2.0 / NATS;
        assert!(outage_rayleigh(1, 10, p, low).unwrap() < outage_rayleigh(1, 2, p, low).unwrap());
    }
}
