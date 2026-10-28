//! Spatial correlation of the antennas of a MIMO channel.
//!
//! The Kronecker model separates the correlation of the receive antennas, $R_{r}$,
//! from that of the transmit antennas, $R_{t}$: with $G$ of independent
//! $\mathcal{CN}(0, 1)$ entries and $R = L L^{H}$ (Cholesky),
//!
//! $$
//! H = L_{r}\\, G\\, L_{t}^{T}, \qquad
//! \mathbb{E}\left[\mathrm{vec}(H)\\, \mathrm{vec}(H)^{H}\right] = R_{t} \otimes R_{r}
//! $$
//!
//! [`exponential_correlation`] gives the usual one-parameter model of a uniform linear
//! array, $R_{ij} = \rho^{\\,j - i}$ for $j \ge i$ (Hermitian).
//!
//! # Examples
//! ```
//! use com_channel::spatial::{exponential_correlation, Kronecker};
//! use com_math::C64;
//! let r = exponential_correlation(4, C64::new(0.7, 0.0));
//! assert!((r[[0, 3]].re - 0.343).abs() < 1e-12);
//! assert!(Kronecker::new(r.clone(), r).is_ok());
//! ```
//!
//! # References
//!
//! 1. J. P. Kermoal et al., "A stochastic MIMO radio channel model with experimental
//!    validation," *IEEE J. Sel. Areas Commun.*, 20(6), 2002.
//! 2. S. L. Loyka, "Channel capacity of MIMO architecture using the exponential
//!    correlation matrix," *IEEE Commun. Lett.*, 5(9), 2001.

use com_core::{Error, Result};
use com_math::C64;
use com_math::linalg::{CMat, cholesky};
use ndarray::Array2;

/// The exponential correlation matrix of `n` antennas, $R_{ij} = \rho^{\\,j-i}$ for
/// $j \ge i$ and its conjugate below the diagonal ($|\rho| < 1$ for a positive
/// definite matrix).
pub fn exponential_correlation(n: usize, rho: C64) -> CMat {
    Array2::from_shape_fn(
        (n, n),
        |(i, j)| {
            if j >= i { rho.powu((j - i) as u32) } else { rho.powu((i - j) as u32).conj() }
        },
    )
}

/// Receive and transmit correlation of the Kronecker model, with their Cholesky
/// factors.
#[derive(Clone, Debug, PartialEq)]
pub struct Kronecker {
    rx: CMat,
    tx: CMat,
    lrx: CMat,
    ltx: CMat,
}

impl Kronecker {
    /// The model with receive correlation `rx` and transmit correlation `tx`.
    ///
    /// # Errors
    /// [`Error::Config`] unless both are Hermitian positive definite with a unit
    /// diagonal.
    pub fn new(rx: CMat, tx: CMat) -> Result<Kronecker> {
        for (name, r) in [("receive", &rx), ("transmit", &tx)] {
            if r.diag().iter().any(|d| (d - C64::new(1.0, 0.0)).norm() > 1e-9) {
                return Err(Error::Config(format!("the {name} correlation needs a unit diagonal")));
            }
        }
        let lrx = cholesky(&rx.view())
            .map_err(|_| Error::Config("the receive correlation is not positive definite".into()))?;
        let ltx = cholesky(&tx.view())
            .map_err(|_| Error::Config("the transmit correlation is not positive definite".into()))?;
        Ok(Kronecker { rx, tx, lrx, ltx })
    }

    /// Receive correlation $R_{r}$.
    pub fn rx(&self) -> &CMat {
        &self.rx
    }

    /// Transmit correlation $R_{t}$.
    pub fn tx(&self) -> &CMat {
        &self.tx
    }

    /// $L_{r} G L_{t}^{T}$.
    ///
    /// # Panics
    /// If `g` is not `n_rx × n_tx`.
    pub fn correlate(&self, g: &CMat) -> CMat {
        self.lrx.dot(g).dot(&self.ltx.t())
    }
}
