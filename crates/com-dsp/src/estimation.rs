//! Waveform-agnostic channel estimation: least squares and linear MMSE.
//!
//! For known symbols $x_{i}$ received as $y_{i} = h_{i} x_{i} + w_{i}$,
//! $w_{i} \sim \mathcal{CN}(0, \sigma^{2})$:
//!
//! - least squares (LS): $\hat h_{i} = y_{i} / x_{i}$, unbiased, of error variance
//!   $\sigma^{2}/|x_{i}|^{2}$ ([`ls`]);
//! - linear MMSE: from the LS estimates $\hat h_{p}$ at the pilot positions $p$, the
//!   channel at the positions $q$ is
//!
//! $$
//! \hat h_{q} = R_{qp} \left(R_{pp} + \frac{\sigma^{2}}{E_{p}} I\right)^{-1} \hat h_{p},
//! $$
//!
//! with the channel correlations $R$ and the pilot energy $E_{p}$ ([`lmmse`], or
//! [`Lmmse`] to factor $R_{pp} + \frac{\sigma^{2}}{E_{p}} I$ once for the symbols that
//! share the pilots and correlations), the estimator of minimum mean square error among
//! the linear ones (van de Beek et al. 1995; Edfors et al. 1998); its error covariance is
//! $R_{qq} - R_{qp} (R_{pp} + \frac{\sigma^{2}}{E_{p}} I)^{-1} R_{pq}$ ([`lmmse_mse`]);
//! - [`frequency_correlation`], the correlation between subcarriers of a power-delay
//!   profile, which the LMMSE estimator needs.
//!
//! The interpolation of the estimates between the pilots is [`crate::interpolation`].
//!
//! # Examples
//! ```
//! use com_dsp::estimation::ls;
//! use com_math::C64;
//! let h = ls(&[C64::new(0.0, 2.0)], &[C64::new(1.0, 1.0)])?;
//! assert!((h[0] - C64::new(1.0, 1.0)).norm() < 1e-12);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J.-J. van de Beek, O. Edfors, M. Sandell, S. K. Wilson and P. O. Börjesson, "On
//!    channel estimation in OFDM systems," *Proc. IEEE VTC*, 815–819, 1995.
//! 2. O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson and P. O. Börjesson, "OFDM
//!    channel estimation by singular value decomposition," *IEEE Trans. Commun.*, 46(7),
//!    931–939, 1998.
//! 3. S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*,
//!    Prentice Hall, 1993, ch. 12.

use com_core::{Error, Result, ensure, ensure_shape, same_len};
use com_math::C64;
use com_math::linalg::{CMat, cholesky, hermitian, identity, solve_cholesky, solve_hpd};

/// The frequency correlation of a channel of power-delay profile `(delays, powers)`
/// between two subcarriers `delta` bins apart on a grid of `n_fft` bins,
/// $R(\Delta k) = \sum_{l} p_{l}\, e^{-j 2\pi \Delta k\, d_{l} / N}$, the delays in samples
/// of the IFFT of size $N$: the correlations of the LMMSE estimator.
///
/// # Panics
/// If `delays` and `powers` have different lengths.
///
/// # Examples
/// ```
/// use com_dsp::estimation::frequency_correlation;
/// // two equal paths 4 samples apart: |R| = |cos(πΔk·4/N)|
/// let r = frequency_correlation(&[0.0, 4.0], &[0.5, 0.5], 2.0, 16);
/// assert!(r.norm() < 1e-15);
/// ```
pub fn frequency_correlation(delays: &[f64], powers: &[f64], delta: f64, n_fft: usize) -> C64 {
    assert_eq!(delays.len(), powers.len(), "frequency_correlation: one power per delay");
    delays
        .iter()
        .zip(powers)
        .map(|(&d, &p)| com_math::cis(-2.0 * std::f64::consts::PI * delta * d / n_fft as f64) * p)
        .sum()
}

/// Least-squares estimates $y_{i}/x_{i}$.
///
/// # Errors
/// [`Error::Shape`] if `y` and `x` have different lengths.
pub fn ls(y: &[C64], x: &[C64]) -> Result<Vec<C64>> {
    same_len(y.len(), x.len(), "ls: received and known symbols")?;
    Ok(y.iter().zip(x).map(|(a, b)| a / b).collect())
}

/// $R_{pp} + \frac{\sigma^{2}}{E_{p}} I$.
pub(crate) fn regularized(r_pp: &CMat, noise_to_pilot: f64) -> CMat {
    r_pp + &identity(r_pp.nrows()).mapv(|v| v * noise_to_pilot)
}

/// The linear MMSE estimates at the positions of `r_qp` (rows) from the LS estimates
/// `h_ls` at the pilots, with `noise_to_pilot` = $\sigma^{2}/E_{p}$: [`Lmmse::new`] then
/// [`Lmmse::apply`], which factors the system again at each call.
///
/// # Errors
/// [`Error::Shape`] for inconsistent sizes, [`Error::Numerical`] for a singular system.
pub fn lmmse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, h_ls: &[C64]) -> Result<Vec<C64>> {
    ensure(noise_to_pilot >= 0.0, || "negative noise".into())?;
    ensure_shape(r_pp.nrows() == h_ls.len() && r_qp.ncols() == h_ls.len(), || {
        format!("{} pilots for correlations of {} pilots", h_ls.len(), r_pp.nrows())
    })?;
    Lmmse::new(r_qp.clone(), r_pp, noise_to_pilot)?.apply(h_ls)
}

/// The linear MMSE estimator of [`lmmse`] for one set of pilots and correlations, the
/// Cholesky factor of $R_{pp} + \frac{\sigma^{2}}{E_{p}} I$ computed once: each
/// [`Lmmse::apply`] costs two triangular substitutions and the product by $R_{qp}$,
/// $O(P^{2} + QP)$ for $P$ pilots and $Q$ estimated positions instead of $O(P^{3})$,
/// with the same operations as [`lmmse`] (the same results, bit for bit).
///
/// # Examples
/// ```
/// use com_dsp::estimation::{lmmse, Lmmse};
/// use com_math::C64;
/// use ndarray::Array2;
/// let r = Array2::from_shape_fn((4, 4), |(i, j)| C64::new(0.9f64.powi((i as i32 - j as i32).abs()), 0.0));
/// let est = Lmmse::new(r.clone(), &r, 0.1)?;
/// let h_ls = [C64::new(1.0, 0.0), C64::new(0.8, 0.1), C64::new(0.7, 0.3), C64::new(0.5, 0.2)];
/// assert_eq!(est.apply(&h_ls)?, lmmse(&r, &r, 0.1, &h_ls)?);
/// # Ok::<(), com_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Lmmse {
    r_qp: CMat,
    factor: CMat,
}

impl Lmmse {
    /// The estimator at the positions of `r_qp` (rows) from the pilots of `r_pp`, with
    /// `noise_to_pilot` = $\sigma^{2}/E_{p}$.
    ///
    /// # Errors
    /// [`Error::Config`] for a negative noise, [`Error::Shape`] unless `r_pp` is square
    /// with one column of `r_qp` per pilot, [`Error::Numerical`] for a singular system.
    pub fn new(r_qp: CMat, r_pp: &CMat, noise_to_pilot: f64) -> Result<Lmmse> {
        ensure(noise_to_pilot >= 0.0, || "negative noise".into())?;
        ensure_shape(r_pp.nrows() == r_pp.ncols() && r_qp.ncols() == r_pp.nrows(), || {
            format!("correlations of {}×{} pilots and {} columns", r_pp.nrows(), r_pp.ncols(), r_qp.ncols())
        })?;
        let factor = cholesky(&regularized(r_pp, noise_to_pilot).view())?;
        Ok(Lmmse { r_qp, factor })
    }

    /// The number of pilots $P$.
    pub fn n_pilots(&self) -> usize {
        self.factor.nrows()
    }

    /// The estimates at the positions of the rows of $R_{qp}$ from the LS estimates
    /// `h_ls` at the pilots.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `h_ls` has one estimate per pilot.
    pub fn apply(&self, h_ls: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(h_ls.len() == self.n_pilots(), || {
            format!("{} pilots for correlations of {} pilots", h_ls.len(), self.n_pilots())
        })?;
        let b =
            ndarray::Array2::from_shape_vec((h_ls.len(), 1), h_ls.to_vec()).map_err(|e| Error::Shape(e.to_string()))?;
        let z = solve_cholesky(&self.factor.view(), &b.view())?;
        Ok(self.r_qp.dot(&z).column(0).to_vec())
    }
}

/// The mean square error of every LMMSE estimate (the diagonal of its error covariance),
/// given the variances `r_qq_diag` of the channel at the estimated positions.
///
/// # Errors
/// As [`lmmse`]; [`Error::Shape`] unless `r_qq_diag` has one variance per row of `r_qp`.
pub fn lmmse_mse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, r_qq_diag: &[f64]) -> Result<Vec<f64>> {
    ensure_shape(r_qq_diag.len() == r_qp.nrows(), || {
        format!("{} variances for {} estimated positions", r_qq_diag.len(), r_qp.nrows())
    })?;
    let rpq = hermitian(&r_qp.view());
    let z = solve_hpd(&regularized(r_pp, noise_to_pilot).view(), &rpq.view())?;
    let g = r_qp.dot(&z);
    Ok(r_qq_diag.iter().enumerate().map(|(i, v)| (v - g[[i, i]].re).max(0.0)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "frequency_correlation: one power per delay")]
    fn frequency_correlation_rejects_a_power_per_delay_mismatch() {
        let _ = frequency_correlation(&[0.0, 1.0], &[1.0], 1.0, 16);
    }
    use com_core::rng::Rng;
    use ndarray::Array2;

    #[test]
    fn lmmse_beats_ls_and_follows_its_predicted_error() {
        // a smooth channel: R_ij = ρ^|i-j| with ρ = 0.95, 8 positions, SNR 0 dB
        let n = 8;
        let r = Array2::from_shape_fn((n, n), |(i, j)| C64::new(0.95f64.powi((i as i32 - j as i32).abs()), 0.0));
        let l = com_math::linalg::cholesky(&r.view()).unwrap();
        let mut rng = Rng::new(1);
        let (mut e_ls, mut e_mm) = (0.0, 0.0);
        for _ in 0..2000 {
            let g: Vec<C64> = (0..n).map(|_| rng.complex_gaussian()).collect();
            let h = com_math::linalg::matvec(&l.view(), &g);
            let y: Vec<C64> = h.iter().map(|v| v + rng.complex_gaussian()).collect();
            let x = vec![C64::new(1.0, 0.0); n];
            let hl = ls(&y, &x).unwrap();
            let hm = lmmse(&r, &r, 1.0, &hl).unwrap();
            e_ls += hl.iter().zip(&h).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
            e_mm += hm.iter().zip(&h).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
        }
        assert!((e_ls / 2000.0 / n as f64 - 1.0).abs() < 0.05);
        let predicted = lmmse_mse(&r, &r, 1.0, &vec![1.0; n]).unwrap().iter().sum::<f64>() / n as f64;
        assert!((e_mm / 2000.0 / n as f64 / predicted - 1.0).abs() < 0.08);
        assert!(predicted < 0.4);
    }

    /// The LMMSE estimate as it was computed before: [`solve_hpd`] for every vector.
    fn former_lmmse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, h_ls: &[C64]) -> Vec<C64> {
        let b = Array2::from_shape_vec((h_ls.len(), 1), h_ls.to_vec()).unwrap();
        let z = solve_hpd(&regularized(r_pp, noise_to_pilot).view(), &b.view()).unwrap();
        r_qp.dot(&z).column(0).to_vec()
    }

    #[test]
    fn the_factored_estimator_is_the_former_solve_for_every_symbol() {
        let mut rng = Rng::new(3);
        let (p, q) = (12, 40);
        let r_pp = Array2::from_shape_fn((p, p), |(i, j)| {
            frequency_correlation(&[0.0, 1.5, 4.0], &[0.6, 0.3, 0.1], (4 * i) as f64 - (4 * j) as f64, 64)
        });
        let r_qp = Array2::from_shape_fn((q, p), |(k, j)| {
            frequency_correlation(&[0.0, 1.5, 4.0], &[0.6, 0.3, 0.1], k as f64 - (4 * j) as f64, 64)
        });
        for noise in [1e-9, 0.01, 1.0] {
            let est = Lmmse::new(r_qp.clone(), &r_pp, noise).unwrap();
            for _ in 0..5 {
                let h: Vec<C64> = (0..p).map(|_| rng.complex_gaussian()).collect();
                assert_eq!(est.apply(&h).unwrap(), former_lmmse(&r_qp, &r_pp, noise, &h), "noise {noise}");
                assert_eq!(lmmse(&r_qp, &r_pp, noise, &h).unwrap(), est.apply(&h).unwrap());
            }
        }
        let est = Lmmse::new(r_qp.clone(), &r_pp, 0.1).unwrap();
        assert!(matches!(est.apply(&vec![C64::new(1.0, 0.0); p - 1]), Err(Error::Shape(_))));
        assert!(matches!(Lmmse::new(r_qp, &identity(p + 1), 0.1), Err(Error::Shape(_))));
        assert!(Lmmse::new(identity(3), &identity(3), -1.0).is_err());
    }

    #[test]
    fn ls_and_lmmse_mse_reject_inputs_of_different_lengths() {
        let one = C64::new(1.0, 0.0);
        assert!(matches!(ls(&[one; 3], &[one; 2]), Err(Error::Shape(_))));
        let r = identity(3);
        assert!(matches!(lmmse_mse(&r, &r, 0.1, &[1.0; 2]), Err(Error::Shape(_))));
        assert!(matches!(lmmse_mse(&r, &r, 0.1, &[1.0; 4]), Err(Error::Shape(_))));
        assert_eq!(lmmse_mse(&r, &r, 0.1, &[1.0; 3]).unwrap().len(), 3);
    }
}
