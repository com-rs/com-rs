//! Error rates over Rayleigh fading with an imperfect channel estimate (Chang & Su 2002).
//!
//! # Model
//!
//! The received sample of a subcarrier is $Y = HX + N$, $H$ zero-mean circular Gaussian
//! of variance $2\sigma_{1}^{2}$, $N$ of variance $2\sigma_{n}^{2}$; the receiver
//! equalises with an estimate $\hat{H}$ jointly Gaussian with $H$, of variance
//! $2\sigma_{2}^{2}$ and cross-correlation
//! $\frac{1}{2}\mathbb{E}[\hat{H} H^{*}] = \mu_{1} + j\mu_{2}$ — every linear pilot-assisted estimate (LS, LMMSE, interpolations) has this
//! form. With $\rho_{i} = \mu_{i}/(\sigma_{1}\sigma_{2})$, $\rho^{2} = \rho_{1}^{2} +
//! \rho_{2}^{2}$ and the mean SNR per bit $\bar{\gamma}_{b} = 2\sigma_{1}^{2}
//! \mathbb{E}|X|^{2}/(2\sigma_{n}^{2} K)$ ($K$ bits per symbol), the bit error
//! probabilities of M. X. Chang and Y. T. Su (*IEEE Trans. Wireless Commun.* 1(4), 2002)
//! are, for BPSK (their eq. 31) and Gray QPSK (eq. 32),
//!
//! $$
//! P_{b} = \frac{1}{2}\left[1 - \frac{\rho_{1}}{\sqrt{1 + 1/\bar{\gamma}_{b} -
//! \rho_{2}^{2}}}\right], \qquad
//! P_{b} = \frac{1}{2}\left[1 - \frac{1}{2}\frac{(\rho_{1} + \rho_{2})/\sqrt{2}}{\sqrt{1 +
//! \frac{1}{2\bar{\gamma}_{b}} - \frac{(\rho_{1} - \rho_{2})^{2}}{2}}} - \frac{1}{2}
//! \frac{(\rho_{1} - \rho_{2})/\sqrt{2}}{\sqrt{1 + \frac{1}{2\bar{\gamma}_{b}} -
//! \frac{(\rho_{1} + \rho_{2})^{2}}{2}}}\right],
//! $$
//!
//! for Gray 16-QAM their eq. 53, a sum of 24 terms of the coefficients of their Table I,
//! which reduces with $\hat{H} = H$ to their eq. 69,
//!
//! $$
//! P_{b} = \frac{1}{2} - \frac{3}{8}\frac{1}{\sqrt{1 + \frac{5}{2\bar{\gamma}_{b}}}} -
//! \frac{1}{4}\frac{1}{\sqrt{1 + \frac{5}{18\bar{\gamma}_{b}}}} + \frac{1}{8}
//! \frac{1}{\sqrt{1 + \frac{1}{10\bar{\gamma}_{b}}}},
//! $$
//!
//! (whose expansion is $P_{b} \approx 0.4972/\bar{\gamma}_{b}$ at high SNR; the paper prints
//! $0.4952/\bar{\gamma}_{b}$), and for differential detection over Jakes' fading of normalised Doppler $f_{D}T_{s}$
//! (eqs. 40, 41)
//!
//! $$
//! P_{b}^{\mathrm{DBPSK}} = \frac{1}{2}\left[1 - \frac{J_{0}(2\pi f_{D}T_{s})}{1 +
//! 1/\bar{\gamma}_{b}}\right], \qquad
//! P_{b}^{\mathrm{DQPSK}} = \frac{1}{2}\left[1 - \frac{J_{0}(2\pi f_{D}T_{s})}{\sqrt{(2 +
//! 1/\bar{\gamma}_{b})(1 + 1/(2\bar{\gamma}_{b})) - J_{0}^{2}(2\pi f_{D}T_{s})}}\right].
//! $$
//!
//! # Examples
//! ```
//! use com_mapping::imperfect_csi::{ber_bpsk_imperfect_csi, ber_16qam_imperfect_csi};
//! use com_mapping::theory::ber_rayleigh_mrc;
//! let g = 100.0;                                          // 20 dB
//! // a perfect estimate: the BER of flat Rayleigh fading
//! assert!((ber_bpsk_imperfect_csi(g, 1.0, 0.0) - ber_rayleigh_mrc(g, 1).unwrap()).abs() < 1e-15);
//! // an estimate correlated at 0.999: an error floor
//! assert!(ber_16qam_imperfect_csi(1e9, 0.999, 0.0, 1.0) > 1e-3);
//! ```
//!
//! # References
//!
//! 1. M.-X. Chang, Y. T. Su, "Performance analysis of equalized OFDM systems in Rayleigh
//!    fading," *IEEE Trans. Wireless Commun.* 1(4), 721–732, 2002.

use crate::constellation::{Constellation, ConstellationKind};
use com_core::{Error, Result};
use com_math::bessel_j0;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// The BER of the constellation `c` equalised by an estimate of correlations
/// $(\rho_{1}, \rho_{2})$ and of standard deviation ratio `sigma_ratio` at the mean SNR
/// per bit `mean_ebn0_lin` (linear): [`ber_bpsk_imperfect_csi`] for BPSK,
/// [`ber_qpsk_imperfect_csi`] for Gray QPSK (4-QAM or 4-PSK) and
/// [`ber_16qam_imperfect_csi`] for Gray 16-QAM (`sigma_ratio` is used by 16-QAM only:
/// the PSK decisions do not depend on $|\hat{H}|$).
///
/// # Errors
/// [`Error::Config`] for another constellation.
///
/// # Examples
/// ```
/// use com_mapping::constellation::Constellation;
/// use com_mapping::imperfect_csi::{ber_16qam_imperfect_csi, ber_imperfect_csi};
/// let c = Constellation::from_name("16qam")?;
/// assert_eq!(ber_imperfect_csi(&c, 1e5, 0.999, 0.0, 1.0)?, ber_16qam_imperfect_csi(1e5, 0.999, 0.0, 1.0));
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn ber_imperfect_csi(c: &Constellation, mean_ebn0_lin: f64, rho1: f64, rho2: f64, sigma_ratio: f64) -> Result<f64> {
    match (c.kind(), c.order()) {
        (ConstellationKind::Psk, 2) => Ok(ber_bpsk_imperfect_csi(mean_ebn0_lin, rho1, rho2)),
        (ConstellationKind::Qam | ConstellationKind::Psk, 4) => Ok(ber_qpsk_imperfect_csi(mean_ebn0_lin, rho1, rho2)),
        (ConstellationKind::Qam, 16) => Ok(ber_16qam_imperfect_csi(mean_ebn0_lin, rho1, rho2, sigma_ratio)),
        (kind, m) => Err(Error::Config(format!(
            "the BER with an imperfect estimate is for BPSK, QPSK and 16-QAM, got {m}-{}",
            kind.name()
        ))),
    }
}

/// The BER of BPSK equalised by an estimate of correlations $(\rho_{1}, \rho_{2})$ at the
/// mean SNR per bit `mean_ebn0_lin` (linear; Chang & Su eq. 31).
pub fn ber_bpsk_imperfect_csi(mean_ebn0_lin: f64, rho1: f64, rho2: f64) -> f64 {
    0.5 * (1.0 - rho1 / (1.0 + 1.0 / mean_ebn0_lin - rho2 * rho2).sqrt())
}

/// The BER of Gray QPSK equalised by an estimate of correlations $(\rho_{1}, \rho_{2})$
/// (Chang & Su eq. 32).
pub fn ber_qpsk_imperfect_csi(mean_ebn0_lin: f64, rho1: f64, rho2: f64) -> f64 {
    let d = 1.0 + 1.0 / (2.0 * mean_ebn0_lin);
    let (s, t) = (rho1 + rho2, rho1 - rho2);
    0.5 * (1.0
        - 0.5 * s * FRAC_1_SQRT_2 / (d - t * t / 2.0).sqrt()
        - 0.5 * t * FRAC_1_SQRT_2 / (d - s * s / 2.0).sqrt())
}

/// Three rows $(\mathrm{sgn}_{k}, a_{k}, b_{k})$ of Table I sharing $(\cos\theta_{k},
/// \sin\theta_{k})$.
type TableGroup = ([(f64, f64, f64); 3], f64, f64);

/// Table I of Chang & Su: $(\mathrm{sgn}_{k}, a_{k}, b_{k}, \cos\theta_{k},
/// \sin\theta_{k})$ of the 24 terms of their eq. 53.
fn table_i() -> [(f64, f64, f64, f64, f64); 24] {
    let (r2, r10) = (2f64.sqrt(), 10f64.sqrt());
    let mut t = [(0.0, 0.0, 0.0, 0.0, 0.0); 24];
    // groups of three rows sharing (cos θ, sin θ): (sgn, a, b) for each row
    let groups: [TableGroup; 8] = [
        ([(1.0, 2.0, -r2), (1.0, 0.0, r2), (1.0, 2.0, r2)], 1.0 / r2, 1.0 / r2),
        ([(1.0, 2.0, -r2), (1.0, 0.0, r2), (1.0, 2.0, r2)], 1.0 / r2, -1.0 / r2),
        ([(1.0, 2.0, -r10), (1.0, 0.0, r10), (1.0, 2.0, r10)], 1.0 / r10, 3.0 / r10),
        ([(1.0, -2.0, r10), (1.0, 0.0, r10), (-1.0, 2.0, r10)], 3.0 / r10, -1.0 / r10),
        ([(1.0, -2.0, r10), (1.0, 0.0, r10), (-1.0, 2.0, r10)], 3.0 / r10, 1.0 / r10),
        ([(1.0, 2.0, -r10), (1.0, 0.0, r10), (1.0, 2.0, r10)], 1.0 / r10, -3.0 / r10),
        ([(1.0, -2.0, 3.0 * r2), (1.0, 0.0, 3.0 * r2), (-1.0, 2.0, 3.0 * r2)], 1.0 / r2, 1.0 / r2),
        ([(1.0, -2.0, 3.0 * r2), (1.0, 0.0, 3.0 * r2), (-1.0, 2.0, 3.0 * r2)], 1.0 / r2, -1.0 / r2),
    ];
    for (g, (rows, c, s)) in groups.iter().enumerate() {
        for (i, &(sgn, a, b)) in rows.iter().enumerate() {
            t[3 * g + i] = (sgn, a, b, *c, *s);
        }
    }
    t
}

/// The BER of Gray 16-QAM equalised by an estimate of correlations $(\rho_{1},
/// \rho_{2})$ and of standard deviation ratio `sigma_ratio` $= \sigma_{2}/\sigma_{1}$
/// (Chang & Su eq. 53 and Table I; $\bar{\gamma}_{b} = 5\sigma_{1}^{2}/(2\sigma_{n}^{2})$
/// for the points $\{\pm 1, \pm 3\}^{2}$).
pub fn ber_16qam_imperfect_csi(mean_ebn0_lin: f64, rho1: f64, rho2: f64, sigma_ratio: f64) -> f64 {
    let rho_sq = rho1 * rho1 + rho2 * rho2;
    table_i()
        .iter()
        .map(|&(sgn, a, b, c, s)| {
            let u = b * (rho1 * c + rho2 * s) + a * sigma_ratio;
            sgn * (1.0 - u / (b * b * (1.0 - rho_sq) + u * u + 5.0 / (2.0 * mean_ebn0_lin)).sqrt())
        })
        .sum::<f64>()
        / 32.0
}

/// The BER of Gray 16-QAM over flat Rayleigh fading with the perfect channel (Chang & Su
/// eq. 69).
pub fn ber_16qam_rayleigh(mean_ebn0_lin: f64) -> f64 {
    let t = |c: f64| 1.0 / (1.0 + c / mean_ebn0_lin).sqrt();
    0.5 - 3.0 / 8.0 * t(5.0 / 2.0) - 0.25 * t(5.0 / 18.0) + 0.125 * t(1.0 / 10.0)
}

/// The BER of DBPSK over Jakes' Rayleigh fading of normalised Doppler `fd_ts` (Chang &
/// Su eq. 40).
pub fn ber_dbpsk_jakes(mean_ebn0_lin: f64, fd_ts: f64) -> f64 {
    0.5 * (1.0 - bessel_j0(2.0 * PI * fd_ts) / (1.0 + 1.0 / mean_ebn0_lin))
}

/// The BER of Gray DQPSK over Jakes' Rayleigh fading of normalised Doppler `fd_ts`
/// (Chang & Su eq. 41).
pub fn ber_dqpsk_jakes(mean_ebn0_lin: f64, fd_ts: f64) -> f64 {
    let j = bessel_j0(2.0 * PI * fd_ts);
    0.5 * (1.0 - j / ((2.0 + 1.0 / mean_ebn0_lin) * (1.0 + 1.0 / (2.0 * mean_ebn0_lin)) - j * j).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::ber_rayleigh_mrc;

    #[test]
    fn the_constellation_selects_the_closed_form() {
        let (g, r1, r2) = (30.0, 0.98, 0.05);
        let at = |name: &str| ber_imperfect_csi(&Constellation::from_name(name).unwrap(), g, r1, r2, 0.9);
        assert_eq!(at("bpsk").unwrap(), ber_bpsk_imperfect_csi(g, r1, r2));
        assert_eq!(at("QPSK").unwrap(), ber_qpsk_imperfect_csi(g, r1, r2));
        assert_eq!(at("4psk").unwrap(), ber_qpsk_imperfect_csi(g, r1, r2));
        assert_eq!(at("16-QAM").unwrap(), ber_16qam_imperfect_csi(g, r1, r2, 0.9));
        assert!(at("8psk").is_err() && at("64qam").is_err());
    }

    #[test]
    fn the_perfect_estimate_gives_the_known_results() {
        for g in [0.5, 3.0, 10.0, 100.0, 1e4] {
            // eq. 68: BPSK and QPSK have the BER of flat Rayleigh fading
            let r = ber_rayleigh_mrc(g, 1).unwrap();
            assert!((ber_bpsk_imperfect_csi(g, 1.0, 0.0) - r).abs() < 1e-15);
            assert!((ber_qpsk_imperfect_csi(g, 1.0, 0.0) - r).abs() < 1e-15);
            // eq. 53 with ρ1 = 1, ρ2 = 0, σ1 = σ2 is eq. 69: the transcription of Table I
            assert!((ber_16qam_imperfect_csi(g, 1.0, 0.0, 1.0) - ber_16qam_rayleigh(g)).abs() < 1e-14, "{g}");
            // eqs. 42, 43: slow fading
            assert!((ber_dbpsk_jakes(g, 0.0) - 0.5 / (1.0 + g)).abs() < 1e-15);
            let dq = 0.5 * (1.0 - 1.0 / (1.0 + 2.0 / g + 1.0 / (2.0 * g * g)).sqrt());
            assert!((ber_dqpsk_jakes(g, 0.0) - dq).abs() < 1e-15);
        }
        // the asymptotes of Sec. IV-D3: 1/(4γ) for BPSK; for 16-QAM the expansion of
        // eq. 69 gives (15/32 + 5/144 - 1/160)/γ = 0.49722/γ, where the paper prints
        // 0.4952/γ (a misprint, 0.4 % off)
        let g = 1e6;
        assert!((ber_bpsk_imperfect_csi(g, 1.0, 0.0) * 4.0 * g - 1.0).abs() < 1e-5);
        assert!((ber_16qam_rayleigh(g) * g - (15.0 / 32.0 + 5.0 / 144.0 - 1.0 / 160.0)).abs() < 1e-5);
    }
}
