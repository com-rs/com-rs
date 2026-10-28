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
//! realisations.
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
use com_math::C64;
use com_math::linalg::{hermitian, identity, log2_det_hpd, svd};
use ndarray::ArrayView2;

/// Capacity with equal power on the $n_{t}$ antennas (bits per channel use).
///
/// # Errors
/// [`com_core::Error::Config`] for a negative SNR.
pub fn equal_power(h: &ArrayView2<'_, C64>, snr: f64) -> Result<f64> {
    ensure(snr >= 0.0, || format!("the SNR must be non-negative, got {snr}"))?;
    let a = identity(h.nrows()) + h.dot(&hermitian(h)).mapv(|x| x * (snr / h.ncols() as f64));
    log2_det_hpd(&a.view())
}

/// Capacity with water-filling on the eigenmodes (bits per channel use).
///
/// # Errors
/// [`com_core::Error::Config`] for a non-positive SNR.
pub fn water_filling(h: &ArrayView2<'_, C64>, snr: f64) -> Result<f64> {
    let (_, s, _) = svd(h);
    let g: Vec<f64> = s.iter().map(|x| x * x).collect();
    let p = precoding::water_filling(&g, snr)?;
    Ok(g.iter().zip(&p).map(|(g, p)| (1.0 + g * p).log2()).sum())
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
    fn the_capacity_of_a_rank_one_channel_is_one_mode() {
        // H = a b^H: one mode of gain ‖a‖² ‖b‖²
        let a = ndarray::arr2(&[[C64::new(1.0, 0.0)], [C64::new(0.0, 1.0)]]);
        let b = ndarray::arr2(&[[C64::new(1.0, 0.0), C64::new(1.0, 0.0)]]);
        let h = a.dot(&b);
        let c = water_filling(&h.view(), 5.0).unwrap();
        assert!((c - (1.0 + 5.0 * 4.0f64).log2()).abs() < 1e-12);
    }
}
