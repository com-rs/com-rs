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
//! with the channel correlations $R$ and the pilot energy $E_{p}$ ([`lmmse`]), the
//! estimator of minimum mean square error among the linear ones (van de Beek et al.
//! 1995; Edfors et al. 1998); its error covariance is
//! $R_{qq} - R_{qp} (R_{pp} + \frac{\sigma^{2}}{E_{p}} I)^{-1} R_{pq}$ ([`lmmse_mse`]);
//! - [`linear_interpolation`] between known positions (constant beyond the ends).
//!
//! # Examples
//! ```
//! use com_dsp::estimation::ls;
//! use com_math::C64;
//! let h = ls(&[C64::new(0.0, 2.0)], &[C64::new(1.0, 1.0)]);
//! assert!((h[0] - C64::new(1.0, 1.0)).norm() < 1e-12);
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

use com_core::{Error, Result, ensure, ensure_shape};
use com_math::C64;
use com_math::linalg::{CMat, hermitian, identity, solve_hpd};

/// Least-squares estimates $y_{i}/x_{i}$.
pub fn ls(y: &[C64], x: &[C64]) -> Vec<C64> {
    y.iter().zip(x).map(|(a, b)| a / b).collect()
}

fn regularized(r_pp: &CMat, noise_to_pilot: f64) -> CMat {
    r_pp + &identity(r_pp.nrows()).mapv(|v| v * noise_to_pilot)
}

/// The linear MMSE estimates at the positions of `r_qp` (rows) from the LS estimates
/// `h_ls` at the pilots, with `noise_to_pilot` = $\sigma^{2}/E_{p}$.
///
/// # Errors
/// [`Error::Shape`] for inconsistent sizes, [`Error::Numerical`] for a singular system.
pub fn lmmse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, h_ls: &[C64]) -> Result<Vec<C64>> {
    ensure(noise_to_pilot >= 0.0, || "negative noise".into())?;
    ensure_shape(r_pp.nrows() == h_ls.len() && r_qp.ncols() == h_ls.len(), || {
        format!("{} pilots for correlations of {} pilots", h_ls.len(), r_pp.nrows())
    })?;
    let b = ndarray::Array2::from_shape_vec((h_ls.len(), 1), h_ls.to_vec()).map_err(|e| Error::Shape(e.to_string()))?;
    let z = solve_hpd(&regularized(r_pp, noise_to_pilot).view(), &b.view())?;
    Ok(r_qp.dot(&z).column(0).to_vec())
}

/// The mean square error of every LMMSE estimate (the diagonal of its error covariance),
/// given the variances `r_qq_diag` of the channel at the estimated positions.
///
/// # Errors
/// As [`lmmse`].
pub fn lmmse_mse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, r_qq_diag: &[f64]) -> Result<Vec<f64>> {
    let rpq = hermitian(&r_qp.view());
    let z = solve_hpd(&regularized(r_pp, noise_to_pilot).view(), &rpq.view())?;
    let g = r_qp.dot(&z);
    Ok(r_qq_diag.iter().enumerate().map(|(i, v)| (v - g[[i, i]].re).max(0.0)).collect())
}

/// Linear interpolation of `values` known at the increasing `positions` onto
/// `0..n` (constant beyond the first and last positions).
///
/// # Panics
/// If `positions` is empty or of another length than `values`.
pub fn linear_interpolation(positions: &[usize], values: &[C64], n: usize) -> Vec<C64> {
    assert!(!positions.is_empty() && positions.len() == values.len(), "one value per known position");
    let mut out = Vec::with_capacity(n);
    let mut j = 0;
    for i in 0..n {
        while j + 1 < positions.len() && positions[j + 1] <= i {
            j += 1;
        }
        let v = if i <= positions[0] {
            values[0]
        } else if j + 1 >= positions.len() {
            values[positions.len() - 1]
        } else {
            let (a, b) = (positions[j], positions[j + 1]);
            let t = (i - a) as f64 / (b - a) as f64;
            values[j] * (1.0 - t) + values[j + 1] * t
        };
        out.push(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use ndarray::Array2;

    #[test]
    fn interpolation_is_exact_on_lines_and_lmmse_beats_ls() {
        let v: Vec<C64> = [0.0, 2.0, 4.0].iter().map(|x| C64::new(*x, -x)).collect();
        let out = linear_interpolation(&[2, 4, 6], &v, 9);
        assert_eq!(out[0], v[0]);
        assert!((out[3] - C64::new(1.0, -1.0)).norm() < 1e-12);
        assert_eq!(out[8], v[2]);
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
            let hl = ls(&y, &x);
            let hm = lmmse(&r, &r, 1.0, &hl).unwrap();
            e_ls += hl.iter().zip(&h).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
            e_mm += hm.iter().zip(&h).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
        }
        assert!((e_ls / 2000.0 / n as f64 - 1.0).abs() < 0.05);
        let predicted = lmmse_mse(&r, &r, 1.0, &vec![1.0; n]).unwrap().iter().sum::<f64>() / n as f64;
        assert!((e_mm / 2000.0 / n as f64 / predicted - 1.0).abs() < 0.08);
        assert!(predicted < 0.4);
    }
}
