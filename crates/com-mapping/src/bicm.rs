//! Cutoff rates of coded modulation (CM) and bit-interleaved coded modulation (BICM)
//! with uniform inputs (Caire, Taricco & Biglieri 1998, Sec. III); their capacities are
//! the mutual information and the generalized mutual information of
//! [`crate::theory::constellation_mi`] and [`crate::theory::constellation_gmi`].
//!
//! A constellation $\mathcal{X}$ of $M = 2^{m}$ points with unit mean energy carries
//! the label of `points()[i]` in the bits of `i` (Gray for the families of
//! [`Constellation::new`], set partitioning for [`Constellation::set_partitioning`]).
//! The channel is $y = g x + n$, $n \sim \mathcal{CN}(0, \sigma^{2})$ (a real
//! constellation sees a real channel, as in [`crate::theory`]), the SNR is
//! $1/\sigma^{2}$, and the gain $g$ is 1 ([`Fading::Awgn`]) or Rayleigh,
//! $\mathbb{E}|g|^{2} = 1$, known at the receiver ([`Fading::Rayleigh`], perfect CSI
//! and ideal interleaving; [`Fading`] is that of [`crate::theory`]). With $\mathcal{X}_{b}^{i}$ the points whose bit $i$ is $b$:
//!
//! - the CM capacity (eq. (10)), the mutual information
//!   ([`crate::theory::constellation_mi`] with `px = None`),
//!   $\tilde{C} = m - \mathbb{E}\left[\log_{2} \sum_{z \in \mathcal{X}} p(y|z) / p(y|x)\right]$;
//! - the BICM capacity (eq. (14)) ([`crate::theory::constellation_gmi`] with
//!   `px = None`),
//!   $\hat{C} = m - \sum_{i=1}^{m} \mathbb{E}_{b,y}\left[\log_{2} \sum_{z \in \mathcal{X}} p(y|z)
//!   / \sum_{z \in \mathcal{X}_{b}^{i}} p(y|z)\right] \le \tilde{C}$ (eq. (16)), the
//!   generalized mutual information;
//! - the CM cutoff rate (eq. (17)), in closed form, [`cm_cutoff_rate`],
//!   $\tilde{R}_{0} = m - \log_{2} \frac{1}{M} \sum_{x, z} \mathbb{E}_{g}\, e^{-|g|^{2}
//!   |x - z|^{2} / (4\sigma^{2})}$, where the mean over Rayleigh fading is
//!   $1 / (1 + |x - z|^{2} / (4\sigma^{2}))$;
//! - the BICM cutoff rate (eqs. (19), (21)), [`bicm_cutoff_rate`],
//!   $\hat{R}_{0} = m\left(1 - \log_{2}(1 + B)\right)$ with the Bhattacharyya factor
//!   $B = \frac{1}{m} \sum_{i} \mathbb{E}_{b,y}\sqrt{\sum_{z \in \mathcal{X}_{\bar{b}}^{i}}
//!   p(y|z) / \sum_{z \in \mathcal{X}_{b}^{i}} p(y|z)}$.
//!
//! The expectations over the noise use Gauss–Hermite quadrature with `points` nodes
//! per real dimension (the paper uses Monte Carlo); the mean over Rayleigh fading is
//! the integral over $u = \ln |g|^{2}$, of density $e^{u - e^{u}}$, by Simpson's rule
//! with a step of 1/8.
//!
//! # Examples
//! ```
//! use com_mapping::bicm::bicm_cutoff_rate;
//! use com_mapping::constellation::{Constellation, ConstellationKind};
//! use com_mapping::theory::{Fading, constellation_gmi, constellation_mi};
//! // 16-QAM at 10 dB: BICM with Gray labels loses little, with set partitioning much
//! let gray = Constellation::qam(16)?;
//! let sp = Constellation::set_partitioning(ConstellationKind::Qam, 16)?;
//! let cm = constellation_mi(&gray, 10.0, None, Fading::Awgn, 16)?;
//! let g = constellation_gmi(&gray, 10.0, None, Fading::Awgn, 16)?;
//! let s = constellation_gmi(&sp, 10.0, None, Fading::Awgn, 16)?;
//! assert!(cm > g && cm - g < 0.05 && g - s > 0.3);
//! // the cutoff rate lies below the capacity
//! assert!(bicm_cutoff_rate(&gray, 10.0, Fading::Awgn, 16)? < g);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. G. Caire, G. Taricco and E. Biglieri, "Bit-interleaved coded modulation," *IEEE
//!    Trans. Inf. Theory*, 44(3), 927–946, 1998.
//! 2. E. Zehavi, "8-PSK trellis codes for a Rayleigh channel," *IEEE Trans. Commun.*,
//!    40(5), 873–884, 1992.

use crate::constellation::Constellation;
use crate::theory::{Fading, TheorySetup, bit_expectations, over_fading, setup};
use com_core::Result;
use com_math::{db2lin, log_sum_exp_iter};

/// The CM cutoff rate (bits per channel use) of a constellation with uniform inputs,
/// eq. (17) of Caire et al. 1998 in closed form (see the module).
///
/// # Examples
/// ```
/// use com_mapping::bicm::cm_cutoff_rate;
/// use com_mapping::theory::Fading;
/// use com_mapping::constellation::Constellation;
/// // BPSK: 1 - log2(1 + e^{-Es/N0})
/// let c = Constellation::psk(2)?;
/// let r0 = cm_cutoff_rate(&c, 0.0, Fading::Awgn);
/// assert!((r0 - (1.0 - (1.0 + (-1f64).exp()).log2())).abs() < 1e-12);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn cm_cutoff_rate(c: &Constellation, snr_db: f64, fading: Fading) -> f64 {
    let x = c.normalized();
    let x = x.points();
    let m = x.len() as f64;
    // |x - z|²/(4σ²) for a complex channel, |x - z|²/(8σ²) for a real one
    let scale = db2lin(snr_db) / if c.kind().is_real() { 8.0 } else { 4.0 };
    let mut sum = 0.0;
    for xi in x {
        for xj in x {
            let a = scale * (xi - xj).norm_sqr();
            sum += match fading {
                Fading::Awgn => (-a).exp(),
                Fading::Rayleigh => 1.0 / (1.0 + a),
            };
        }
    }
    m.log2() - (sum / m).log2()
}

/// The BICM cutoff rate (bits per channel use) of a labelled constellation with uniform
/// inputs, eqs. (19) and (21) of Caire et al. 1998 (see the module).
///
/// # Errors
/// [`com_core::Error::Config`] for an invalid constellation.
pub fn bicm_cutoff_rate(c: &Constellation, snr_db: f64, fading: Fading, points: usize) -> Result<f64> {
    let b = c.bits_per_symbol() as f64;
    let factor = over_fading(|s| Ok(bhattacharyya(&setup(c, None, s, points)?)), snr_db, fading)?;
    Ok(b * (1.0 - factor.ln_1p() / std::f64::consts::LN_2))
}

/// The Bhattacharyya factor $B$ of eq. (19), from the log-metrics of every bit.
fn bhattacharyya(s: &TheorySetup) -> f64 {
    // √(Σ_{other} / Σ_{same}) = e^{(ln Σ_{other} - ln Σ_{same})/2}, with ln Σ_{other}
    // = ln(Σ_all - Σ_same) computed without cancellation from the log-sums
    let terms = bit_expectations(s, |all, same| {
        let total = log_sum_exp_iter(all.iter().copied());
        let other = total + (-(same - total).exp()).ln_1p();
        (0.5 * (other - same)).exp()
    });
    terms.iter().sum::<f64>() / terms.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constellation::ConstellationKind;
    use crate::theory::{constellation_gmi, constellation_mi};
    use com_core::rng::Rng;

    #[test]
    fn the_rayleigh_mean_of_the_cm_cutoff_rate_is_its_closed_form() {
        // the numerical mean over fading of the AWGN closed form (inside the log)
        let c = Constellation::qam(16).unwrap();
        let m = 16f64;
        for snr_db in [-5.0, 5.0, 20.0, 40.0] {
            let inner = |s: f64| Ok((m.log2() - cm_cutoff_rate(&c, s, Fading::Awgn)).exp2());
            let mean = over_fading(inner, snr_db, Fading::Rayleigh).unwrap();
            let want = m.log2() - mean.log2();
            let got = cm_cutoff_rate(&c, snr_db, Fading::Rayleigh);
            assert!((got - want).abs() < 1e-6, "{snr_db} dB: {got} vs {want}");
        }
    }

    #[test]
    fn the_capacities_over_rayleigh_fading_match_a_simulation() {
        // Monte Carlo of the information densities of eqs. (10) and (14), 16-QAM Gray
        let c = Constellation::qam(16).unwrap().normalized();
        let x = c.points();
        let snr_db = 8.0;
        let sigma2 = db2lin(-snr_db);
        let mut r = Rng::new(11);
        let n = 40_000;
        let (mut cm, mut bicm) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let i = r.below(16);
            let g = r.complex_gaussian();
            let y = g * x[i] + r.complex_gaussian() * sigma2.sqrt();
            let l: Vec<f64> = x.iter().map(|z| -(y - g * z).norm_sqr() / sigma2).collect();
            let all = log_sum_exp_iter(l.iter().copied());
            cm.push(4.0 - (all - l[i]) / std::f64::consts::LN_2);
            let mut v = 4.0;
            for k in 0..4 {
                let bit = (i >> (3 - k)) & 1;
                let same = log_sum_exp_iter((0..16).filter(|&j| (j >> (3 - k)) & 1 == bit).map(|j| l[j]));
                v -= (all - same) / std::f64::consts::LN_2;
            }
            bicm.push(v);
        }
        for (v, want) in [
            (&cm, constellation_mi(&c, snr_db, None, Fading::Rayleigh, 16).unwrap()),
            (&bicm, constellation_gmi(&c, snr_db, None, Fading::Rayleigh, 16).unwrap()),
        ] {
            let mean = v.iter().sum::<f64>() / n as f64;
            let sd = (v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n * (n - 1)) as f64).sqrt();
            assert!((mean - want).abs() < 4.0 * sd, "{mean} ± {sd} vs {want}");
        }
    }

    #[test]
    fn the_bhattacharyya_factor_of_bpsk_is_closed() {
        // one bit: B = e^{-Es/N0} over AWGN, 1/(1 + Es/N0) over Rayleigh fading
        let c = Constellation::psk(2).unwrap();
        for snr_db in [-3.0, 0.0, 6.0] {
            let s = db2lin(snr_db);
            for (fading, b) in [(Fading::Awgn, (-s).exp()), (Fading::Rayleigh, 1.0 / (1.0 + s))] {
                let want = 1.0 - b.ln_1p() / std::f64::consts::LN_2;
                let got = bicm_cutoff_rate(&c, snr_db, fading, 40).unwrap();
                assert!((got - want).abs() < 1e-6, "{snr_db} dB {fading:?}: {got} vs {want}");
                assert!((got - cm_cutoff_rate(&c, snr_db, fading)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn bicm_never_beats_cm_capacity_and_gray_beats_set_partitioning() {
        for (kind, m) in [(ConstellationKind::Psk, 8), (ConstellationKind::Qam, 16)] {
            let gray = Constellation::new(kind, m).unwrap();
            let sp = Constellation::set_partitioning(kind, m).unwrap();
            for snr_db in [0.0, 6.0, 12.0] {
                let cm = constellation_mi(&gray, snr_db, None, Fading::Awgn, 16).unwrap();
                let g = constellation_gmi(&gray, snr_db, None, Fading::Awgn, 16).unwrap();
                let s = constellation_gmi(&sp, snr_db, None, Fading::Awgn, 16).unwrap();
                assert!(cm >= g - 1e-12 && g > s, "{kind:?} {snr_db} dB: {cm} {g} {s}");
                // relabelling does not change the CM capacity
                assert!((constellation_mi(&sp, snr_db, None, Fading::Awgn, 16).unwrap() - cm).abs() < 1e-9);
            }
        }
    }
}

#[cfg(test)]
mod literature {
    use super::*;
    use crate::constellation::ConstellationKind;
    use crate::theory::{constellation_gmi, constellation_mi};
    use com_math::bisect;

    fn pair(kind: ConstellationKind, m: usize) -> (Constellation, Constellation) {
        (Constellation::new(kind, m).unwrap(), Constellation::set_partitioning(kind, m).unwrap())
    }

    /// G. Caire, G. Taricco and E. Biglieri, "Bit-interleaved coded modulation," IEEE
    /// Trans. Inf. Theory 44(3), 1998, Figs. 4 (AWGN) and 5 (Rayleigh fading, perfect
    /// CSI): values read on the figures within 0.1 bit (AWGN at 10 dB: CM 8PSK 2.7, CM
    /// 16QAM 3.15, BICM-SP 16QAM 2.4; Rayleigh: CM 16QAM 2.6 at 10 dB, BICM-SP 16QAM 3.0
    /// at 15 dB), and the text: Gray-labelled BICM is "very close" to CM (within 0.1
    /// bit from -5 to 20 dB), set partitioning "several decibels worse" (2.3 dB more
    /// SNR for 2 bits with 8PSK, 3.8 dB with 16QAM).
    #[test]
    fn caire_1998_figs_4_5_capacity() {
        let (p8, _) = pair(ConstellationKind::Psk, 8);
        let (q16, sp16) = pair(ConstellationKind::Qam, 16);
        for (got, want) in [
            (constellation_mi(&p8, 10.0, None, Fading::Awgn, 12).unwrap(), 2.7),
            (constellation_mi(&q16, 10.0, None, Fading::Awgn, 12).unwrap(), 3.15),
            (constellation_gmi(&sp16, 10.0, None, Fading::Awgn, 12).unwrap(), 2.4),
            (constellation_mi(&q16, 10.0, None, Fading::Rayleigh, 8).unwrap(), 2.6),
            (constellation_gmi(&sp16, 15.0, None, Fading::Rayleigh, 8).unwrap(), 3.0),
        ] {
            assert!((got - want).abs() < 0.1, "{got} vs {want}");
        }
        for (kind, m) in [(ConstellationKind::Psk, 4), (ConstellationKind::Psk, 8), (ConstellationKind::Qam, 16)] {
            let (gray, sp) = pair(kind, m);
            for snr_db in [-5.0, 0.0, 5.0, 10.0, 20.0] {
                let gap = constellation_mi(&gray, snr_db, None, Fading::Awgn, 12).unwrap()
                    - constellation_gmi(&gray, snr_db, None, Fading::Awgn, 12).unwrap();
                assert!((-1e-12..0.1).contains(&gap), "{kind:?} {m}, {snr_db} dB: {gap}");
            }
            if m > 4 {
                let need = |c: &Constellation| {
                    bisect(|s| constellation_gmi(c, s, None, Fading::Awgn, 12).unwrap() < 2.0, -10.0, 30.0, 30)
                };
                let loss = need(&sp) - need(&gray);
                assert!((2.0..6.0).contains(&loss), "{kind:?} {m}: {loss} dB");
            }
        }
    }
}
