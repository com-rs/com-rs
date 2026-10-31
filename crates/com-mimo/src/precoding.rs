//! Linear precoding: eigenmode transmission with water-filling, and the zero-forcing
//! precoders of the multi-user downlink.
//!
//! # Eigenmodes
//!
//! With the SVD $H = U \Sigma V^{H}$, sending $x = V P^{1/2} s$ and filtering with
//! $U^{H}$ turns the channel into parallel scalar channels of gains $\sigma_{i}$:
//! $\tilde y_{i} = \sigma_{i} \sqrt{p_{i}}\\, s_{i} + \tilde w_{i}$. Water-filling chooses
//! the powers that maximise $\sum_{i} \log_{2}(1 + p_{i} g_{i})$ under
//! $\sum_{i} p_{i} = P$, with $g_{i} = \sigma_{i}^{2} / \sigma^{2}$:
//!
//! $$
//! p_{i} = \left(\mu - \frac{1}{g_{i}}\right)^{+}
//! $$
//!
//! # Multi-user downlink
//!
//! The rows of $H$ are the channels of single-antenna users. Zero forcing
//! $W \propto H^{H} (H H^{H})^{-1}$ removes the interference between users; the
//! regularised version $W \propto H^{H} (H H^{H} + \alpha I)^{-1}$, with
//! $\alpha = K \sigma^{2} / P$ for $K$ users, maximises the signal-to-leakage ratio. Both
//! are scaled to the total power $\lVert W \rVert_{F}^{2} = P$.
//!
//! # Reconfigurable intelligent surface
//!
//! A RIS of $N$ elements reflects $\sum_{i} g_{i} e^{j\phi_{i}} h_{i}$ (`com_channel::ris`):
//! [`ris_co_phasing`] gives the phases $\phi_{i} = -\arg(h_{i} g_{i})$ that align every
//! product, which maximise the SNR to $(\sum_{i} |h_{i}| |g_{i}|)^{2} E_{s}/N_{0}$
//! (Basar et al. 2019, eqs. (11)–(13), as maximal-ratio combining).
//!
//! # Examples
//! ```
//! use com_mimo::precoding::water_filling_powers;
//! let p = water_filling_powers(&[4.0, 1.0, 0.1], 1.0)?;
//! assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
//! assert_eq!(p[2], 0.0);                       // the weakest mode stays off
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. E. Telatar, "Capacity of multi-antenna Gaussian channels," *Eur. Trans.
//!    Telecommun.*, 10(6), 1999.
//! 2. C. B. Peel, B. M. Hochwald and A. L. Swindlehurst, "A vector-perturbation
//!    technique for near-capacity multiantenna multiuser communication — Part I:
//!    channel inversion and regularization," *IEEE Trans. Commun.*, 53(1), 2005.
//! 3. E. Basar, M. Di Renzo, J. de Rosny, M. Debbah, M.-S. Alouini and R. Zhang,
//!    "Wireless communications through reconfigurable intelligent surfaces," *IEEE
//!    Access*, 2019 (arXiv 1906.09490).

use crate::detection::FLOOR;
use com_core::signal::Symbols;
use com_core::traits::Precode;
use com_core::{Error, Result, ensure};
use com_math::C64;
use com_math::linalg::{CMat, hermitian, identity, inverse_hpd, svd};
use ndarray::{Array2, ArrayView2};

/// The water-filling powers of parallel channels of gains `gains` (SNR per unit
/// power) under the total power `total_power`.
///
/// # Errors
/// [`Error::Config`] for a negative gain or a non-positive power.
pub fn water_filling_powers(gains: &[f64], total_power: f64) -> Result<Vec<f64>> {
    ensure(total_power > 0.0, || format!("the total power must be positive, got {total_power}"))?;
    ensure(gains.iter().all(|g| *g >= 0.0 && g.is_finite()), || "the gains must be non-negative".into())?;
    let mut order: Vec<usize> = (0..gains.len()).filter(|&i| gains[i] > 0.0).collect();
    order.sort_by(|a, b| gains[*b].total_cmp(&gains[*a]));
    let mut p = vec![0.0; gains.len()];
    // the largest set of active modes whose water level stays above their floors
    for active in (1..=order.len()).rev() {
        let inv: f64 = order[..active].iter().map(|&i| 1.0 / gains[i]).sum();
        let mu = (total_power + inv) / active as f64;
        if mu > 1.0 / gains[order[active - 1]] {
            for &i in &order[..active] {
                p[i] = mu - 1.0 / gains[i];
            }
            break;
        }
    }
    Ok(p)
}

/// Eigenmode (SVD) transmission: precoder $V P^{1/2}$ and receive filter $U^{H}$.
#[derive(Clone, Debug, PartialEq)]
pub struct Eigenmode {
    /// Precoder, $n_{t} \times$ active streams.
    pub precoder: CMat,
    /// Receive filter, active streams $\times n_{r}$.
    pub combiner: CMat,
    /// Singular values of the active modes.
    pub gains: Vec<f64>,
    /// Power of every active mode.
    pub powers: Vec<f64>,
}

impl Eigenmode {
    /// The modes of `h` with water-filling of `total_power` against the receiver noise
    /// variance `noise_var` (equal power on every non-zero mode if `water_filling` is
    /// false); modes without power are dropped.
    ///
    /// # Errors
    /// As [`water_filling_powers`], and [`Error::Config`] for a non-positive noise variance or a
    /// zero channel.
    pub fn new(h: &ArrayView2<'_, C64>, total_power: f64, noise_var: f64, water_filling: bool) -> Result<Eigenmode> {
        ensure(noise_var > 0.0, || format!("the noise variance must be positive, got {noise_var}"))?;
        let (u, s, v) = svd(h);
        let nonzero: Vec<usize> = (0..s.len()).filter(|&i| s[i] > 1e-12 * s[0].max(FLOOR)).collect();
        ensure(!nonzero.is_empty(), || "the channel is zero".into())?;
        let powers_all = if water_filling {
            water_filling_powers(&s.iter().map(|x| x * x / noise_var).collect::<Vec<_>>(), total_power)?
        } else {
            let p = total_power / nonzero.len() as f64;
            (0..s.len()).map(|i| if nonzero.contains(&i) { p } else { 0.0 }).collect()
        };
        let active: Vec<usize> = (0..s.len()).filter(|&i| powers_all[i] > 0.0).collect();
        let precoder =
            Array2::from_shape_fn((h.ncols(), active.len()), |(r, c)| v[[r, active[c]]] * powers_all[active[c]].sqrt());
        let combiner = Array2::from_shape_fn((active.len(), h.nrows()), |(r, c)| u[[c, active[r]]].conj());
        Ok(Eigenmode {
            precoder,
            combiner,
            gains: active.iter().map(|&i| s[i]).collect(),
            powers: active.iter().map(|&i| powers_all[i]).collect(),
        })
    }

    /// Number of active streams.
    pub fn n_streams(&self) -> usize {
        self.powers.len()
    }
}

impl Precode for Eigenmode {
    fn precode_into(&self, x: &Symbols, out: &mut Symbols) -> Result<()> {
        precode_with(&self.precoder, x, out)
    }
}

/// $x = W s$ for every column of the streams `x` into `out`.
fn precode_with(w: &CMat, x: &Symbols, out: &mut Symbols) -> Result<()> {
    if x.n_streams() != w.ncols() {
        return Err(Error::Shape(format!("{} streams for a precoder of {}", x.n_streams(), w.ncols())));
    }
    out.assign(&w.dot(x.as_array()));
    Ok(())
}

/// A linear precoder $x = W s$ (one column of the streams at a time).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearPrecoder {
    /// The precoding matrix, $n_{t} \times$ streams.
    pub w: CMat,
}

impl LinearPrecoder {
    /// Zero forcing for the users of the rows of `h`, scaled to `total_power`.
    ///
    /// # Errors
    /// [`Error::Config`] if there are more users than transmit antennas,
    /// [`Error::Numerical`] for a rank-deficient channel.
    pub fn zero_forcing(h: &ArrayView2<'_, C64>, total_power: f64) -> Result<LinearPrecoder> {
        ensure(h.nrows() <= h.ncols(), || {
            format!("zero forcing serves at most {} users, not {}", h.ncols(), h.nrows())
        })?;
        LinearPrecoder::regularized(h, 0.0, total_power)
    }

    /// Regularised zero forcing $H^{H}(H H^{H} + \alpha I)^{-1}$, scaled to
    /// `total_power` (use $\alpha = K \sigma^{2} / P$).
    ///
    /// # Errors
    /// [`Error::Config`] for a negative `alpha` or a non-positive power,
    /// [`Error::Numerical`] for a singular system.
    pub fn regularized(h: &ArrayView2<'_, C64>, alpha: f64, total_power: f64) -> Result<LinearPrecoder> {
        ensure(alpha >= 0.0 && total_power > 0.0, || "alpha ≥ 0 and a positive power".into())?;
        let hh = hermitian(h);
        let g = h.dot(&hh) + identity(h.nrows()).mapv(|x| x * alpha);
        let inv = inverse_hpd(&g.view()).map_err(|_| Error::Numerical("the channel is rank deficient".into()))?;
        let w = hh.dot(&inv);
        let norm2: f64 = w.iter().map(C64::norm_sqr).sum();
        Ok(LinearPrecoder { w: w.mapv(|x| x * (total_power / norm2).sqrt()) })
    }
}

impl Precode for LinearPrecoder {
    fn precode_into(&self, x: &Symbols, out: &mut Symbols) -> Result<()> {
        precode_with(&self.w, x, out)
    }
}

/// The phases $\phi_{i} = -\arg(h_{i} g_{i})$ (rad) of the elements of a RIS that align
/// the products of its hops `h` (source to element) and `g` (element to destination):
/// the gain $\sum_{i} g_{i} e^{j\phi_{i}} h_{i}$ is then $\sum_{i} |h_{i}| |g_{i}|$.
///
/// # Errors
/// [`com_core::Error::Shape`] for hops of different lengths.
pub fn ris_co_phasing(h: &[C64], g: &[C64]) -> Result<Vec<f64>> {
    com_core::ensure_shape(h.len() == g.len(), || format!("hops of {} and {} elements", h.len(), g.len()))?;
    Ok(h.iter().zip(g).map(|(a, b)| -(a * b).arg()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn co_phased_elements_add_in_amplitude() {
        let mut r = Rng::new(3);
        let (h, g): (Vec<C64>, Vec<C64>) = (0..32).map(|_| (r.complex_gaussian(), r.complex_gaussian())).unzip();
        let phases = ris_co_phasing(&h, &g).unwrap();
        let gain: C64 = h.iter().zip(&g).zip(&phases).map(|((a, b), p)| b * C64::from_polar(1.0, *p) * a).sum();
        let amplitude: f64 = h.iter().zip(&g).map(|(a, b)| a.norm() * b.norm()).sum();
        assert!((gain - amplitude).norm() < 1e-12 * amplitude);
        assert!(ris_co_phasing(&h, &g[1..]).is_err());
    }

    fn channel(nr: usize, nt: usize, seed: u64) -> CMat {
        let mut r = Rng::new(seed);
        Array2::from_shape_fn((nr, nt), |_| r.complex_gaussian())
    }

    #[test]
    fn water_filling_satisfies_its_conditions() {
        let g = [3.0, 1.5, 0.4, 0.05];
        for p in [0.1, 1.0, 10.0, 100.0] {
            let w = water_filling_powers(&g, p).unwrap();
            assert!((w.iter().sum::<f64>() - p).abs() < 1e-12);
            // a common water level on the active modes, below the floor of the others
            let levels: Vec<f64> = (0..4).filter(|&i| w[i] > 0.0).map(|i| w[i] + 1.0 / g[i]).collect();
            assert!(levels.windows(2).all(|l| (l[0] - l[1]).abs() < 1e-12));
            for i in (0..4).filter(|&i| w[i] == 0.0) {
                assert!(1.0 / g[i] >= levels[0] - 1e-12);
            }
        }
        assert_eq!(water_filling_powers(&[1.0, 1.0], 2.0).unwrap(), vec![1.0, 1.0]);
        assert!(water_filling_powers(&[1.0], 0.0).is_err());
    }

    #[test]
    fn eigenmodes_diagonalise_the_channel() {
        let h = channel(4, 3, 1);
        let e = Eigenmode::new(&h.view(), 3.0, 0.1, false).unwrap();
        let d = e.combiner.dot(&h).dot(&e.precoder);
        for i in 0..e.n_streams() {
            for j in 0..e.n_streams() {
                let want = if i == j { e.gains[i] * e.powers[i].sqrt() } else { 0.0 };
                assert!((d[[i, j]] - C64::new(want, 0.0)).norm() < 1e-10);
            }
        }
        // at low SNR water-filling keeps the strongest mode only
        assert_eq!(Eigenmode::new(&h.view(), 1e-3, 1.0, true).unwrap().n_streams(), 1);
    }

    #[test]
    fn zero_forcing_removes_the_interference_between_users() {
        let h = channel(3, 4, 2);
        let p = LinearPrecoder::zero_forcing(&h.view(), 2.0).unwrap();
        let e = h.dot(&p.w);
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(e[[i, j]].norm() < 1e-10);
                }
            }
        }
        assert!((p.w.iter().map(C64::norm_sqr).sum::<f64>() - 2.0).abs() < 1e-12);
        assert!(LinearPrecoder::zero_forcing(&channel(5, 4, 3).view(), 1.0).is_err());
        let x = Symbols::new(Array2::from_elem((3, 5), C64::new(1.0, 0.0)));
        let mut out = Symbols::default();
        p.precode_into(&x, &mut out).unwrap();
        assert_eq!((out.n_streams(), out.len()), (4, 5));
    }
}
