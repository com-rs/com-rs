//! The uplink of multi-user massive MIMO: $K$ single-antenna users to a base station of
//! $M$ antennas, with linear receivers (Ngo, Larsson and Marzetta 2013).
//!
//! # Model
//!
//! The base station receives $y = \sqrt{p_{u}}\\, G x + n$, with $G = H D^{1/2}$ the
//! $M \times K$ channel: $H$ the small-scale fading (i.i.d. $\mathcal{CN}(0, 1)$, drawn
//! by `com_channel::fading::Mimo`), $D = \mathrm{diag}(\beta_{k})$ the large-scale fading
//! of the users, $x$ the symbols of unit energy, $n \sim \mathcal{CN}(0, I_{M})$ and
//! $p_{u}$ the normalised transmit SNR of every user. With perfect channel state
//! information (CSI), the receiver knows $G$. With estimated CSI, the users send
//! orthogonal pilots of $\tau \ge K$ symbols at the power $p_{u}$: the MMSE estimate
//! $\hat G$ has independent columns of variance $\tau p_{u} \beta_{k}^{2} / (\tau p_{u}
//! \beta_{k} + 1)$ per entry ([`Uplink::estimate_variance`]), and its error, independent
//! of $\hat G$, the variance $\varepsilon_{k} = \beta_{k} / (\tau p_{u} \beta_{k} + 1)$
//! ([`Uplink::error_variance`]). Then $y = \sqrt{p_{u}} \hat G x + n'$ with the noise and
//! estimation error $n'$ of variance $1 + p_{u} \sum_{i} \varepsilon_{i}$
//! ([`Uplink::effective_noise_variance`]). The achievable rate of user $k$ is
//! $\mathbb{E}\\, \log_{2}(1 + \mathrm{SINR}_{k})$ (eqs. (13), (37)): that of the
//! unbiased estimates of `crate::detection::Linear` on the channel $\sqrt{p_{u}} \hat G$
//! with the noise variance $1 + p_{u} \sum_{i} \varepsilon_{i}$.
//!
//! # Lower bounds
//!
//! [`Uplink::rate_bound`] gives the closed-form lower bounds on the achievable rate of
//! user $k$ (Propositions 2, 3, 6 and 7), with $p_{p} = \tau p_{u}$:
//!
//! | Receiver | Perfect CSI | Estimated CSI |
//! |---|---|---|
//! | MR | $\log_{2}\Bigl(1 + \frac{p_{u} (M - 1) \beta_{k}}{p_{u} \sum_{i \ne k} \beta_{i} + 1}\Bigr)$ | $\log_{2}\Bigl(1 + \frac{\tau p_{u}^{2} (M - 1) \beta_{k}^{2}}{p_{u} (\tau p_{u} \beta_{k} + 1) \sum_{i \ne k} \beta_{i} + (\tau + 1) p_{u} \beta_{k} + 1}\Bigr)$ |
//! | ZF | $\log_{2}(1 + p_{u} (M - K) \beta_{k})$ | $\log_{2}\Bigl(1 + \frac{\tau p_{u}^{2} (M - K) \beta_{k}^{2}}{(\tau p_{u} \beta_{k} + 1) \sum_{i} \frac{p_{u} \beta_{i}}{\tau p_{u} \beta_{i} + 1} + \tau p_{u} \beta_{k} + 1}\Bigr)$ |
//!
//! The bound of MR needs $M \ge 2$, that of ZF $M \ge K + 1$. As $M \to \infty$, with
//! $p_{u} = E_{u} / M$ and perfect CSI the rate tends to $\log_{2}(1 + \beta_{k} E_{u})$,
//! and with $p_{u} = E_{u} / \sqrt{M}$ and estimated CSI to $\log_{2}(1 + \tau
//! \beta_{k}^{2} E_{u}^{2})$ (Propositions 1 and 5): doubling $M$ saves 3 dB of transmit
//! power with perfect CSI and 1.5 dB with estimated CSI.
//!
//! # Examples
//! ```
//! use com_mimo::detection::Criterion;
//! use com_mimo::massive::Uplink;
//! // one user, perfect CSI: the MR and ZF bounds are log2(1 + p_u (M - 1) β)
//! let up = Uplink::new(8, vec![0.5], 2.0)?;
//! assert!((up.rate_bound(Criterion::Mr, 0)? - 8f64.log2()).abs() < 1e-12);
//! assert!((up.rate_bound(Criterion::Zf, 0)? - 8f64.log2()).abs() < 1e-12);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. H. Q. Ngo, E. G. Larsson and T. L. Marzetta, "Energy and spectral efficiency of
//!    very large multiuser MIMO systems," *IEEE Trans. Commun.*, 2013 (arXiv 1112.3810).

use com_core::{Error, Result, ensure};

use crate::detection::Criterion;

/// The uplink of $K$ users of large-scale fading `large_scale` to `antennas` antennas,
/// at the normalised transmit SNR `snr_lin` of every user (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Uplink {
    /// The number $M$ of base-station antennas.
    pub antennas: usize,
    /// The large-scale fading $\beta_{k}$ of every user.
    pub large_scale: Vec<f64>,
    /// The normalised transmit SNR $p_{u}$ of every user (linear).
    pub snr_lin: f64,
    /// The length $\tau$ of the pilots of the estimated CSI; perfect CSI if `None`.
    pub pilots: Option<usize>,
}

impl Uplink {
    /// The uplink with perfect CSI.
    ///
    /// # Errors
    /// [`Error::Config`] for no antenna or user, a large-scale fading that is not
    /// positive and finite, or an SNR that is not.
    pub fn new(antennas: usize, large_scale: Vec<f64>, snr_lin: f64) -> Result<Uplink> {
        ensure(antennas >= 1 && !large_scale.is_empty(), || {
            format!("the uplink needs antennas and users, got {antennas} and {}", large_scale.len())
        })?;
        ensure(large_scale.iter().all(|b| *b > 0.0 && b.is_finite()), || {
            "the large-scale fading of every user must be positive and finite".into()
        })?;
        ensure(snr_lin > 0.0 && snr_lin.is_finite(), || format!("snr_lin must be positive, got {snr_lin}"))?;
        Ok(Uplink { antennas, large_scale, snr_lin, pilots: None })
    }

    /// The uplink with the CSI estimated from orthogonal pilots of `tau` symbols.
    ///
    /// # Errors
    /// [`Error::Config`] for fewer pilot symbols than users.
    pub fn with_pilots(self, tau: usize) -> Result<Uplink> {
        ensure(tau >= self.large_scale.len(), || {
            format!("orthogonal pilots need tau >= K = {}, got {tau}", self.large_scale.len())
        })?;
        Ok(Uplink { pilots: Some(tau), ..self })
    }

    /// The variance $\varepsilon_{k}$ of every entry of the error of the channel
    /// estimate of every user, 0 with perfect CSI.
    pub fn error_variance(&self) -> Vec<f64> {
        match self.pilots {
            None => vec![0.0; self.large_scale.len()],
            Some(tau) => self.large_scale.iter().map(|b| b / (tau as f64 * self.snr_lin * b + 1.0)).collect(),
        }
    }

    /// The variance of every entry of the channel estimate $\hat G$ of every user:
    /// $\beta_{k} - \varepsilon_{k}$ ($\beta_{k}$ with perfect CSI).
    pub fn estimate_variance(&self) -> Vec<f64> {
        self.large_scale.iter().zip(self.error_variance()).map(|(b, e)| b - e).collect()
    }

    /// The variance $1 + p_{u} \sum_{i} \varepsilon_{i}$ of the noise and of the
    /// estimation error at every antenna.
    pub fn effective_noise_variance(&self) -> f64 {
        1.0 + self.snr_lin * self.error_variance().iter().sum::<f64>()
    }

    /// The closed-form lower bound on the achievable rate of user `k` (bit/s/Hz) with
    /// the receiver `receiver` (see the module).
    ///
    /// # Errors
    /// [`Error::Config`] for a user out of range, MMSE (whose bound is an approximation,
    /// not implemented), MR with one antenna or ZF with $M \le K$.
    pub fn rate_bound(&self, receiver: Criterion, k: usize) -> Result<f64> {
        let n_users = self.large_scale.len();
        ensure(k < n_users, || format!("user {k} of {n_users}"))?;
        let (m, p, bk) = (self.antennas as f64, self.snr_lin, self.large_scale[k]);
        let others: f64 = self.large_scale.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, b)| b).sum();
        let sinr = match receiver {
            Criterion::Mr => {
                ensure(self.antennas >= 2, || "the MR bound needs M >= 2".into())?;
                match self.pilots {
                    None => p * (m - 1.0) * bk / (p * others + 1.0),
                    Some(tau) => {
                        let t = tau as f64;
                        t * p * p * (m - 1.0) * bk * bk / (p * (t * p * bk + 1.0) * others + (t + 1.0) * p * bk + 1.0)
                    }
                }
            }
            Criterion::Zf => {
                ensure(self.antennas > n_users, || format!("the ZF bound needs M > K = {n_users}"))?;
                let dof = m - n_users as f64;
                match self.pilots {
                    None => p * dof * bk,
                    Some(tau) => {
                        let t = tau as f64;
                        let errors: f64 = self.large_scale.iter().map(|b| p * b / (t * p * b + 1.0)).sum();
                        t * p * p * dof * bk * bk / ((t * p * bk + 1.0) * errors + t * p * bk + 1.0)
                    }
                }
            }
            Criterion::Mmse => {
                return Err(Error::Config("the MMSE bound of Ngo et al. is an approximation: not implemented".into()));
            }
        };
        Ok((1.0 + sinr).log2())
    }

    /// The sum over the users of [`Uplink::rate_bound`] (bit/s/Hz).
    ///
    /// # Errors
    /// As [`Uplink::rate_bound`].
    pub fn sum_rate_bound(&self, receiver: Criterion) -> Result<f64> {
        (0..self.large_scale.len()).map(|k| self.rate_bound(receiver, k)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bounds_of_one_user_with_perfect_csi_are_those_of_maximum_ratio_combining() {
        let up = Uplink::new(16, vec![0.3], 4.0).unwrap();
        let expected = (1.0 + 4.0 * 15.0 * 0.3f64).log2();
        for r in [Criterion::Mr, Criterion::Zf] {
            assert!((up.rate_bound(r, 0).unwrap() - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn estimated_csi_splits_the_large_scale_fading() {
        let up = Uplink::new(32, vec![1.0, 0.25], 2.0).unwrap().with_pilots(2).unwrap();
        let (e, g) = (up.error_variance(), up.estimate_variance());
        assert!((e[0] - 1.0 / 5.0).abs() < 1e-15 && (e[1] - 0.25 / 2.0).abs() < 1e-15);
        assert!((g[0] + e[0] - 1.0).abs() < 1e-15 && (g[1] + e[1] - 0.25).abs() < 1e-15);
        assert!((up.effective_noise_variance() - (1.0 + 2.0 * (0.2 + 0.125))).abs() < 1e-15);
        // estimation costs rate
        for r in [Criterion::Mr, Criterion::Zf] {
            let perfect = Uplink { pilots: None, ..up.clone() };
            assert!(up.sum_rate_bound(r).unwrap() < perfect.sum_rate_bound(r).unwrap());
        }
    }

    #[test]
    fn the_power_scaling_laws_hold() {
        // Propositions 1 and 5: p_u = E_u / M (perfect CSI), E_u / √M (estimated CSI)
        let (eu, beta, k) = (10.0, vec![1.0, 0.5, 0.2], 3);
        let m = 1_000_000usize;
        let perfect = Uplink::new(m, beta.clone(), eu / m as f64).unwrap();
        // the estimated CSI converges as 1/√M: 0.06 bit away at 10⁶ antennas, 0.006 at 10⁸
        let m2 = 100_000_000usize;
        let estimated = Uplink::new(m2, beta.clone(), eu / (m2 as f64).sqrt()).unwrap().with_pilots(k).unwrap();
        for r in [Criterion::Mr, Criterion::Zf] {
            for (i, b) in beta.iter().enumerate() {
                let p = perfect.rate_bound(r, i).unwrap();
                let e = estimated.rate_bound(r, i).unwrap();
                assert!((p - (1.0 + b * eu).log2()).abs() < 1e-3, "{r:?} {i}: {p}");
                assert!((e - (1.0 + k as f64 * b * b * eu * eu).log2()).abs() < 1e-2, "{r:?} {i}: {e}");
            }
        }
    }

    #[test]
    fn invalid_uplinks_are_errors() {
        assert!(Uplink::new(0, vec![1.0], 1.0).is_err());
        assert!(Uplink::new(4, vec![], 1.0).is_err());
        assert!(Uplink::new(4, vec![0.0], 1.0).is_err());
        assert!(Uplink::new(4, vec![1.0], -1.0).is_err());
        assert!(Uplink::new(4, vec![1.0, 1.0], 1.0).unwrap().with_pilots(1).is_err());
        let up = Uplink::new(2, vec![1.0, 1.0], 1.0).unwrap();
        assert!(up.rate_bound(Criterion::Zf, 0).is_err() && up.rate_bound(Criterion::Mmse, 0).is_err());
        assert!(up.rate_bound(Criterion::Mr, 2).is_err() && up.rate_bound(Criterion::Mr, 0).is_ok());
    }
}
