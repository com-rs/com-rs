//! Dense complex linear algebra for the small matrices of MIMO processing.
//!
//! The matrices of a MIMO link are a few antennas wide, so the routines here favour
//! simplicity and accuracy over asymptotic speed: modified Gram–Schmidt with
//! re-orthogonalisation for the QR decomposition, Cholesky for Hermitian positive
//! definite systems, and the one-sided Jacobi method (Hestenes) for the singular value
//! decomposition, which is accurate to the relative precision of the singular values.
//! Matrices are `ndarray::Array2<C64>` ([`CMat`]).
//!
//! # Model
//!
//! - [`qr`]: $A = Q R$ with $Q^{H} Q = I$ and $R$ upper triangular with a real,
//!   non-negative diagonal ($A$ is $m \times n$, $m \ge n$);
//! - [`cholesky`]: $A = L L^{H}$ with $L$ lower triangular ($A$ Hermitian positive
//!   definite);
//! - [`svd`]: $A = U \Sigma V^{H}$ with the singular values in decreasing order.
//!
//! # Examples
//! ```
//! use com_math::linalg::{matmul, hermitian, svd};
//! use com_math::C64;
//! use ndarray::array;
//! let a = array![[C64::new(1.0, 1.0), C64::new(0.5, 0.0)], [C64::new(0.0, -2.0), C64::new(1.0, 0.0)]];
//! let (u, s, v) = svd(&a.view());
//! // U Σ V^H rebuilds A
//! let mut us = u.clone();
//! for (j, sj) in s.iter().enumerate() {
//!     us.column_mut(j).mapv_inplace(|x| x * *sj);
//! }
//! let b = matmul(&us.view(), &hermitian(&v.view()).view());
//! assert!(a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() < 1e-12));
//! ```
//!
//! # References
//!
//! 1. G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins
//!    University Press, 2013 (Sec. 5.2, 4.2, 8.6).

use crate::C64;
use com_core::{Error, Result};
use ndarray::{Array2, ArrayView2};

/// A dense complex matrix.
pub type CMat = Array2<C64>;

/// The identity of size `n`.
pub fn identity(n: usize) -> CMat {
    Array2::from_shape_fn((n, n), |(i, j)| if i == j { C64::new(1.0, 0.0) } else { C64::new(0.0, 0.0) })
}

/// The conjugate transpose $A^{H}$.
pub fn hermitian(a: &ArrayView2<'_, C64>) -> CMat {
    a.t().mapv(|x| x.conj())
}

/// The product $A B$.
///
/// # Panics
/// If the inner dimensions differ.
pub fn matmul(a: &ArrayView2<'_, C64>, b: &ArrayView2<'_, C64>) -> CMat {
    a.dot(b)
}

/// The product $A x$ of a matrix and a vector.
///
/// # Panics
/// If the dimensions differ.
pub fn matvec(a: &ArrayView2<'_, C64>, x: &[C64]) -> Vec<C64> {
    a.rows().into_iter().map(|r| r.iter().zip(x).map(|(u, v)| u * v).sum()).collect()
}

/// The Gram matrix $A^{H} A$.
pub fn gram(a: &ArrayView2<'_, C64>) -> CMat {
    hermitian(a).dot(a)
}

/// The Cholesky factor $L$ of a Hermitian positive definite matrix, $A = L L^{H}$.
///
/// # Errors
/// [`Error::Shape`] for a non-square matrix, [`Error::Numerical`] if the matrix is not
/// positive definite.
pub fn cholesky(a: &ArrayView2<'_, C64>) -> Result<CMat> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(Error::Shape(format!("Cholesky of a {}×{} matrix", n, a.ncols())));
    }
    let mut l = Array2::<C64>::zeros((n, n));
    for j in 0..n {
        let mut d = a[[j, j]].re;
        for k in 0..j {
            d -= l[[j, k]].norm_sqr();
        }
        if d.is_nan() || d <= 1e-300 {
            return Err(Error::Numerical("the matrix is not positive definite".into()));
        }
        let d = d.sqrt();
        l[[j, j]] = C64::new(d, 0.0);
        for i in j + 1..n {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]].conj();
            }
            l[[i, j]] = s / d;
        }
    }
    Ok(l)
}

/// Solve $A X = B$ for a Hermitian positive definite $A$ (by Cholesky).
///
/// # Errors
/// As [`cholesky`], and [`Error::Shape`] if $B$ has another number of rows.
pub fn solve_hpd(a: &ArrayView2<'_, C64>, b: &ArrayView2<'_, C64>) -> Result<CMat> {
    let l = cholesky(a)?;
    let n = l.nrows();
    if b.nrows() != n {
        return Err(Error::Shape(format!("system of size {n} with {} right-hand rows", b.nrows())));
    }
    let mut x = b.to_owned();
    for c in 0..x.ncols() {
        // forward: L y = b
        for i in 0..n {
            let mut s = x[[i, c]];
            for k in 0..i {
                s -= l[[i, k]] * x[[k, c]];
            }
            x[[i, c]] = s / l[[i, i]];
        }
        // backward: L^H x = y
        for i in (0..n).rev() {
            let mut s = x[[i, c]];
            for k in i + 1..n {
                s -= l[[k, i]].conj() * x[[k, c]];
            }
            x[[i, c]] = s / l[[i, i]];
        }
    }
    Ok(x)
}

/// The inverse of a Hermitian positive definite matrix.
///
/// # Errors
/// As [`cholesky`].
pub fn inverse_hpd(a: &ArrayView2<'_, C64>) -> Result<CMat> {
    solve_hpd(a, &identity(a.nrows()).view())
}

/// $\log_{2} \det A$ of a Hermitian positive definite matrix.
///
/// # Errors
/// As [`cholesky`].
pub fn log2_det_hpd(a: &ArrayView2<'_, C64>) -> Result<f64> {
    let l = cholesky(a)?;
    Ok(2.0 * (0..l.nrows()).map(|i| l[[i, i]].re.log2()).sum::<f64>())
}

/// The thin QR decomposition $A = Q R$ of an $m \times n$ matrix, $m \ge n$.
///
/// Modified Gram–Schmidt with one re-orthogonalisation pass ("twice is enough"); the
/// diagonal of $R$ is real and non-negative. A rank-deficient column gives a zero
/// diagonal entry and a zero column of $Q$.
///
/// # Errors
/// [`Error::Shape`] if $m < n$.
pub fn qr(a: &ArrayView2<'_, C64>) -> Result<(CMat, CMat)> {
    let (m, n) = a.dim();
    if m < n {
        return Err(Error::Shape(format!("thin QR of a {m}×{n} matrix needs m ≥ n")));
    }
    let mut q = a.to_owned();
    let mut r = Array2::<C64>::zeros((n, n));
    let scale = a.iter().map(C64::norm_sqr).sum::<f64>().sqrt().max(1e-300);
    for j in 0..n {
        for _pass in 0..2 {
            for k in 0..j {
                let c: C64 = (0..m).map(|i| q[[i, k]].conj() * q[[i, j]]).sum();
                r[[k, j]] += c;
                for i in 0..m {
                    let qk = q[[i, k]];
                    q[[i, j]] -= c * qk;
                }
            }
        }
        let norm = (0..m).map(|i| q[[i, j]].norm_sqr()).sum::<f64>().sqrt();
        if norm <= 1e-14 * scale {
            r[[j, j]] = C64::new(0.0, 0.0);
            for i in 0..m {
                q[[i, j]] = C64::new(0.0, 0.0);
            }
        } else {
            r[[j, j]] = C64::new(norm, 0.0);
            for i in 0..m {
                q[[i, j]] /= norm;
            }
        }
    }
    Ok((q, r))
}

/// The thin singular value decomposition $A = U \Sigma V^{H}$ by one-sided Jacobi.
///
/// For an $m \times n$ matrix with $k = \min(m, n)$, returns $U$ ($m \times k$), the
/// singular values in decreasing order ($k$ of them) and $V$ ($n \times k$). The
/// columns of $U$ that belong to zero singular values are zero.
pub fn svd(a: &ArrayView2<'_, C64>) -> (CMat, Vec<f64>, CMat) {
    let (m, n) = a.dim();
    if m < n {
        let (u, s, v) = svd(&hermitian(a).view());
        return (v, s, u);
    }
    let mut w = a.to_owned();
    let mut v = identity(n);
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let alpha: f64 = w.column(p).iter().map(C64::norm_sqr).sum();
                let beta: f64 = w.column(q).iter().map(C64::norm_sqr).sum();
                let gamma: C64 = w.column(p).iter().zip(w.column(q).iter()).map(|(x, y)| x.conj() * y).sum();
                let g = gamma.norm();
                if g <= 1e-15 * (alpha * beta).sqrt() || g == 0.0 {
                    continue;
                }
                rotated = true;
                // make the inner product real by a phase on column q, then rotate
                let phase = (gamma / g).conj();
                let zeta = (beta - alpha) / (2.0 * g);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for mat in [&mut w, &mut v] {
                    for i in 0..mat.nrows() {
                        let xp = mat[[i, p]];
                        let xq = mat[[i, q]] * phase;
                        mat[[i, p]] = xp * c - xq * s;
                        mat[[i, q]] = xp * s + xq * c;
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut order: Vec<(f64, usize)> =
        (0..n).map(|j| (w.column(j).iter().map(C64::norm_sqr).sum::<f64>().sqrt(), j)).collect();
    order.sort_by(|x, y| y.0.total_cmp(&x.0));
    let mut u = Array2::<C64>::zeros((m, n));
    let mut vs = Array2::<C64>::zeros((n, n));
    let mut sv = Vec::with_capacity(n);
    let top = order.first().map_or(0.0, |o| o.0);
    for (k, &(sk, j)) in order.iter().enumerate() {
        sv.push(sk);
        vs.column_mut(k).assign(&v.column(j));
        if sk > 1e-14 * top && sk > 0.0 {
            u.column_mut(k).assign(&w.column(j).mapv(|x| x / sk));
        }
    }
    (u, sv, vs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    fn random(m: usize, n: usize, seed: u64) -> CMat {
        let mut r = Rng::new(seed);
        Array2::from_shape_fn((m, n), |_| r.complex_gaussian())
    }

    fn close(a: &CMat, b: &CMat, tol: f64) -> bool {
        a.dim() == b.dim() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() < tol)
    }

    #[test]
    fn qr_is_orthonormal_and_triangular() {
        for (m, n, seed) in [(4, 4, 1), (6, 3, 2), (8, 8, 3)] {
            let a = random(m, n, seed);
            let (q, r) = qr(&a.view()).unwrap();
            assert!(close(&q.dot(&r), &a, 1e-12));
            assert!(close(&gram(&q.view()), &identity(n), 1e-12));
            for i in 0..n {
                assert!(r[[i, i]].im == 0.0 && r[[i, i]].re >= 0.0);
                for j in 0..i {
                    assert_eq!(r[[i, j]], C64::new(0.0, 0.0));
                }
            }
        }
    }

    #[test]
    fn cholesky_solve_and_inverse() {
        let h = random(6, 4, 4);
        let a = gram(&h.view()) + identity(4).mapv(|x| x * 0.1);
        let l = cholesky(&a.view()).unwrap();
        assert!(close(&l.dot(&hermitian(&l.view())), &a, 1e-12));
        let inv = inverse_hpd(&a.view()).unwrap();
        assert!(close(&a.dot(&inv), &identity(4), 1e-10));
        assert!(cholesky(&(identity(2).mapv(|x| -x)).view()).is_err());
        // log2 det of a diagonal matrix
        let d = Array2::from_diag(&ndarray::arr1(&[C64::new(2.0, 0.0), C64::new(8.0, 0.0)]));
        assert!((log2_det_hpd(&d.view()).unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn svd_rebuilds_the_matrix_with_orthonormal_factors() {
        for (m, n, seed) in [(4, 4, 5), (6, 2, 6), (2, 5, 7), (8, 8, 8)] {
            let a = random(m, n, seed);
            let (u, s, v) = svd(&a.view());
            let k = m.min(n);
            assert_eq!(s.len(), k);
            assert!(s.windows(2).all(|w| w[0] >= w[1]));
            let mut us = u.clone();
            for (j, sj) in s.iter().enumerate() {
                us.column_mut(j).mapv_inplace(|x| x * *sj);
            }
            assert!(close(&us.dot(&hermitian(&v.view())), &a, 1e-11), "{m}×{n}");
            assert!(close(&gram(&u.view()), &identity(k), 1e-11));
            assert!(close(&gram(&v.view()), &identity(k), 1e-11));
            // Σ s² = ‖A‖_F²
            let f: f64 = a.iter().map(C64::norm_sqr).sum();
            assert!((s.iter().map(|x| x * x).sum::<f64>() - f).abs() < 1e-10 * f);
        }
    }

    #[test]
    fn svd_of_a_rank_one_matrix() {
        let x = random(3, 1, 9);
        let y = random(1, 4, 10);
        let a = x.dot(&y);
        let (_, s, _) = svd(&a.view());
        assert!(s[0] > 1.0 && s[1..].iter().all(|v| *v < 1e-12 * s[0]));
    }
}
