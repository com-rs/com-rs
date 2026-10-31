//! A reconfigurable intelligent surface (RIS) of $N$ reflecting elements between a
//! source and a destination without line of sight (Basar et al. 2019, Sec. IV,
//! eqs. (9)–(15)).
//!
//! # Model
//!
//! The destination receives (eqs. (9)–(10))
//!
//! $$
//! r = \Bigl(\sum_{i=1}^{N} g_{i} e^{j\phi_{i}} h_{i}\Bigr) x + n = g^{T} \Phi h\\, x + n ,
//! $$
//!
//! with $h_{i}$ the channel from the source to element $i$, $g_{i}$ from element $i$ to
//! the destination, both i.i.d. $\mathcal{CN}(0, 1)$ (Rayleigh), $\phi_{i}$ the phase of
//! element $i$ and $n \sim \mathcal{CN}(0, N_{0})$. [`Ris::draw`] draws $h$ and $g$, and
//! [`RisChannel::cascade_gain`] is the gain $g^{T} \Phi h$ of a configuration of the
//! phases (the configuration is the receiver's: `com_mimo::precoding::ris_co_phasing`
//! maximises the SNR).
//!
//! # Co-phased elements
//!
//! With the phases that align every product, the SNR is $\gamma = A^{2} E_{s} / N_{0}$
//! with $A = \sum_{i} |h_{i}| |g_{i}|$ (eq. (13)). The product of two independent
//! Rayleigh amplitudes of unit power has the mean $\pi/4$ and the variance
//! $1 - \pi^{2}/16$, so by the central limit theorem $A$ is close to Gaussian of mean
//! $N\pi/4$ and variance $N(1 - \pi^{2}/16)$ for large $N$, and $\gamma$ is a scaled
//! non-central chi-square variable of one degree of freedom, of moment generating
//! function ([`Ris::clt_snr_mgf`], eq. (14))
//!
//! $$
//! M_{\gamma}(s) = \left(1 - s \frac{N (16 - \pi^{2}) E_{s}}{8 N_{0}}\right)^{-1/2}
//! \exp\left(\frac{s N^{2} \pi^{2} E_{s} / (16 N_{0})}{1 - s N (16 - \pi^{2}) E_{s}
//! / (8 N_{0})}\right)
//! $$
//!
//! and of mean $(N^{2} \pi^{2} + N (16 - \pi^{2})) E_{s} / (16 N_{0})$ ([`Ris::mean_snr`],
//! eq. (15)), which grows as $N^{2}$.
//!
//! # Examples
//! ```
//! use com_channel::ris::Ris;
//! use com_core::rng::Rng;
//! let ris = Ris::new(64)?;
//! let c = ris.draw(&mut Rng::new(1));
//! // the phases that align every product reach the gain Σ |h_i| |g_i|
//! let phases: Vec<f64> = c.h.iter().zip(&c.g).map(|(h, g)| -(h * g).arg()).collect();
//! let a: f64 = c.h.iter().zip(&c.g).map(|(h, g)| h.norm() * g.norm()).sum();
//! assert!((c.cascade_gain(&phases)?.norm() - a).abs() < 1e-12 * a);
//! assert!(ris.mean_snr(1.0) > 64.0 * 64.0 * 0.6);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. E. Basar, M. Di Renzo, J. de Rosny, M. Debbah, M.-S. Alouini and R. Zhang,
//!    "Wireless communications through reconfigurable intelligent surfaces," *IEEE
//!    Access*, 2019 (arXiv 1906.09490).

use std::f64::consts::PI;

use com_core::rng::Rng as SimRng;
use com_core::{Result, ensure, ensure_shape};
use com_math::C64;

/// A RIS of `n_elements` reflecting elements between two single-antenna terminals,
/// both hops i.i.d. Rayleigh (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ris {
    /// The number $N$ of reflecting elements.
    pub n_elements: usize,
}

/// One realisation of the two hops of a [`Ris`].
#[derive(Clone, Debug, PartialEq)]
pub struct RisChannel {
    /// The channels $h_{i}$ from the source to every element.
    pub h: Vec<C64>,
    /// The channels $g_{i}$ from every element to the destination.
    pub g: Vec<C64>,
}

impl Ris {
    /// A RIS of `n_elements` elements.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for no element.
    pub fn new(n_elements: usize) -> Result<Ris> {
        ensure(n_elements >= 1, || "a RIS needs at least one element".into())?;
        Ok(Ris { n_elements })
    }

    /// Draw the two hops: $h_{1}, g_{1}, h_{2}, g_{2}, \dots$ i.i.d. $\mathcal{CN}(0, 1)$.
    pub fn draw(&self, rng: &mut SimRng) -> RisChannel {
        let (mut h, mut g) = (Vec::with_capacity(self.n_elements), Vec::with_capacity(self.n_elements));
        for _ in 0..self.n_elements {
            h.push(rng.complex_gaussian());
            g.push(rng.complex_gaussian());
        }
        RisChannel { h, g }
    }

    /// The mean $N\pi/4$ and the variance $N(1 - \pi^{2}/16)$ of the amplitude $A =
    /// \sum_{i} |h_{i}| |g_{i}|$ of the co-phased elements.
    pub fn amplitude_moments(&self) -> (f64, f64) {
        let n = self.n_elements as f64;
        (n * PI / 4.0, n * (1.0 - PI * PI / 16.0))
    }

    /// The moment generating function $M_{\gamma}(s)$, $s \le 0$, of the SNR of the
    /// co-phased elements at `esn0_lin` $= E_{s}/N_{0}$ (linear) in the Gaussian
    /// approximation of $A$ (eq. (14)).
    pub fn clt_snr_mgf(&self, esn0_lin: f64, s: f64) -> f64 {
        let (mean, var) = self.amplitude_moments();
        // 1 - 2 s σ² Es/N0 and s μ² Es/N0, the non-central chi-square of A² Es/N0
        let d = 1.0 - 2.0 * s * var * esn0_lin;
        (s * mean * mean * esn0_lin / d).exp() / d.sqrt()
    }

    /// The mean SNR $(N^{2} \pi^{2} + N (16 - \pi^{2})) E_{s} / (16 N_{0})$ of the
    /// co-phased elements at `esn0_lin` $= E_{s}/N_{0}$ (linear), eq. (15): exact, as
    /// $\mathbb{E}[A^{2}]$ needs only the two moments of $A$.
    pub fn mean_snr(&self, esn0_lin: f64) -> f64 {
        let (mean, var) = self.amplitude_moments();
        (mean * mean + var) * esn0_lin
    }
}

impl RisChannel {
    /// The gain $g^{T} \Phi h = \sum_{i} g_{i} e^{j\phi_{i}} h_{i}$ of the phases
    /// `phases` (rad).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for a number of phases other than the elements.
    pub fn cascade_gain(&self, phases: &[f64]) -> Result<C64> {
        ensure_shape(phases.len() == self.h.len(), || {
            format!("{} phases for a RIS of {} elements", phases.len(), self.h.len())
        })?;
        Ok(self.h.iter().zip(&self.g).zip(phases).map(|((h, g), p)| g * C64::from_polar(1.0, *p) * h).sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mgf_is_that_of_eq_14_of_basar() {
        let (ris, e, s) = (Ris::new(16).unwrap(), 0.3, -2.0);
        let n = 16.0;
        let c = n * (16.0 - PI * PI) * e / 8.0;
        let expected = (1.0 / (1.0 - s * c)).sqrt() * (s * n * n * PI * PI * e / 16.0 / (1.0 - s * c)).exp();
        assert!((ris.clt_snr_mgf(e, s) - expected).abs() < 1e-15 * expected);
        assert_eq!(ris.clt_snr_mgf(e, 0.0), 1.0);
        let mean = (n * n * PI * PI + n * (16.0 - PI * PI)) * e / 16.0;
        assert!((ris.mean_snr(e) - mean).abs() < 1e-12 * mean);
    }

    #[test]
    fn the_mean_snr_of_co_phased_elements_is_eq_15() {
        let ris = Ris::new(8).unwrap();
        let mut rng = SimRng::new(15);
        let trials = 200_000;
        let mut acc = 0.0;
        for _ in 0..trials {
            let c = ris.draw(&mut rng);
            let a: f64 = c.h.iter().zip(&c.g).map(|(h, g)| h.norm() * g.norm()).sum();
            acc += a * a;
        }
        let measured = acc / trials as f64;
        assert!((measured / ris.mean_snr(1.0) - 1.0).abs() < 0.01, "{measured} against {}", ris.mean_snr(1.0));
    }

    #[test]
    fn invalid_inputs_are_errors() {
        assert!(Ris::new(0).is_err());
        let c = Ris::new(3).unwrap().draw(&mut SimRng::new(1));
        assert!(c.cascade_gain(&[0.0; 2]).is_err());
    }
}
