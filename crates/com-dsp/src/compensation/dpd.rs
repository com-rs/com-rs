//! Digital predistortion (DPD) with memory polynomials and indirect learning.
//!
//! The predistorter $z = P(x)$ is placed before the amplifier $y = F(z)$ so that the
//! cascade is linear, $F(P(x)) \approx G x$. It is a memory polynomial,
//!
//! $$
//! z\[n\] = \sum_{k \in \mathcal{K}} \sum_{m=0}^{M-1} c_{km}\\, x\[n-m\]\\, \lvert x\[n-m\] \rvert^{k-1},
//! $$
//!
//! linear in its coefficients, so it is fitted by least squares. The indirect learning
//! architecture fits a *post*-inverse instead — the polynomial that maps the normalised
//! output $y/G$ back to the amplifier input $z$ — and copies it in front of the
//! amplifier; a few iterations of the loop converge (Eun & Powers 1997; Ding et al.
//! 2004; Morgan et al. 2006).
//!
//! Indirect learning needs a drive where the amplifier is monotonic (its AM/AM curve
//! increasing): beyond, the post-inverse it fits does not exist and the iterations
//! diverge. Keep the peaks below the saturation input, or back off.
//!
//! # Model
//!
//! With the basis matrix $\Phi(u)$ of the polynomial (one column per term $k, m$), the
//! fit solves the regularised normal equations
//!
//! $$
//! \left(\Phi^{H}\Phi + \lambda I\right) c = \Phi^{H} z, \qquad \Phi = \Phi(y / G).
//! $$
//!
//! # Examples
//! ```
//! use com_dsp::compensation::dpd::MemoryPolynomialDpd;
//! use com_math::C64;
//! // a static cubic compression, y = z - 0.1 z|z|², linearised
//! let pa = |z: &[C64]| z.iter().map(|v| v - v * v.norm_sqr() * 0.1).collect::<Vec<_>>();
//! let x: Vec<C64> = (0..4000).map(|n| C64::new((0.37 * n as f64).sin(), (0.21 * n as f64).cos()) * 0.6).collect();
//! let mut dpd = MemoryPolynomialDpd::new(vec![1, 3, 5], 1);
//! dpd.train_indirect(&x, |z| Ok(pa(z)), C64::new(1.0, 0.0), 4, 1e-9)?;
//! let y = pa(&dpd.apply(&x));
//! let err: f64 = y.iter().zip(&x).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / x.len() as f64;
//! assert!(err < 1e-4);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. C. Eun and E. J. Powers, "A new Volterra predistorter based on the indirect
//!    learning architecture," *IEEE Trans. Signal Process.*, 45(1), 223–227, 1997.
//! 2. J. Kim and K. Konstantinou, "Digital predistortion of wideband signals based on
//!    power amplifier model with memory," *Electron. Lett.*, 37(23), 1417–1418, 2001.
//! 3. L. Ding, G. T. Zhou, D. R. Morgan, Z. Ma, J. S. Kenney, J. Kim and C. R. Giardina,
//!    "A robust digital baseband predistorter constructed using memory polynomials,"
//!    *IEEE Trans. Commun.*, 52(1), 159–165, 2004.
//! 4. D. R. Morgan, Z. Ma, J. Kim, M. G. Zierdt and J. Pastalan, "A generalized memory
//!    polynomial model for digital predistortion of RF power amplifiers," *IEEE Trans.
//!    Signal Process.*, 54(10), 3852–3860, 2006.

use crate::filter::convolve_full;
use com_core::{Error, Result, ensure, ensure_shape};
use com_math::C64;
use com_math::linalg::{hermitian, identity, solve_hpd};
use ndarray::Array2;

/// The term of order `k` of a memory polynomial, $x\[n\] \lvert x\[n\] \rvert^{k-1}$,
/// shared by the predistorter and the amplifier model of `com-hardware`.
///
/// # Examples
/// ```
/// use com_dsp::compensation::dpd::polynomial_term;
/// use com_math::C64;
/// let t = polynomial_term(&[C64::new(0.0, 2.0)], 3);
/// assert_eq!(t[0], C64::new(0.0, 8.0));
/// ```
pub fn polynomial_term(x: &[C64], k: u32) -> Vec<C64> {
    let p = i32::try_from(k.saturating_sub(1)).unwrap_or(0);
    x.iter().map(|v| v * v.norm().powi(p)).collect()
}

/// The output of a memory polynomial: for every term, the order $k$ and the taps
/// $a_{k0}, \dots, a_{k,M-1}$,
///
/// $$
/// y\[n\] = \sum_{k} \sum_{m=0}^{M-1} a_{km}\\, x\[n-m\]\\, \lvert x\[n-m\] \rvert^{k-1},
/// $$
///
/// the samples before the block taken as zero: every term is the causal filtering of its
/// basis [`polynomial_term`] by its taps ([`convolve_full`], truncated to the block). The
/// one kernel of the predistorter ([`MemoryPolynomialDpd::apply`]) and of the amplifier
/// model of `com-hardware` (`MemoryPolynomial`).
///
/// # Examples
/// ```
/// use com_dsp::compensation::dpd::memory_polynomial;
/// use com_math::C64;
/// // y[n] = x[n] - 0.5 x[n-1] |x[n-1]|²
/// let x = [C64::new(1.0, 0.0), C64::new(0.0, 2.0)];
/// let y = memory_polynomial(&x, [(1, &[C64::new(1.0, 0.0)][..]), (3, &[C64::new(0.0, 0.0), C64::new(-0.5, 0.0)][..])]);
/// assert_eq!(y, [C64::new(1.0, 0.0), C64::new(-0.5, 2.0)]);
/// ```
pub fn memory_polynomial<'a>(x: &[C64], terms: impl IntoIterator<Item = (u32, &'a [C64])>) -> Vec<C64> {
    let mut y = vec![C64::new(0.0, 0.0); x.len()];
    for (k, taps) in terms {
        for (yn, v) in y.iter_mut().zip(convolve_full(taps, &polynomial_term(x, k))) {
            *yn += v;
        }
    }
    y
}

/// A memory-polynomial predistorter.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPolynomialDpd {
    /// Nonlinearity orders $k$ (odd orders are the usual choice: 1, 3, 5, …).
    pub orders: Vec<u32>,
    /// Memory depth $M$ (1: memoryless).
    pub memory: usize,
    /// Coefficients, order after order, tap after tap.
    pub coefficients: Vec<C64>,
}

impl MemoryPolynomialDpd {
    /// The orders of the predistorter of Ding et al. (2004): 1, 3 and 5.
    pub const DEFAULT_ORDERS: [u32; 3] = [1, 3, 5];
    /// The memory depth of the predistorter of Ding et al. (2004): 3 samples.
    pub const DEFAULT_MEMORY: usize = 3;

    /// The identity predistorter with the given orders and memory.
    pub fn new(orders: Vec<u32>, memory: usize) -> MemoryPolynomialDpd {
        let memory = memory.max(1);
        let mut coefficients = vec![C64::new(0.0, 0.0); orders.len() * memory];
        if let Some(i) = orders.iter().position(|k| *k == 1) {
            coefficients[i * memory] = C64::new(1.0, 0.0);
        }
        MemoryPolynomialDpd { orders, memory, coefficients }
    }

    /// The basis matrix $\Phi(u)$: one row per sample, one column per term.
    fn basis(&self, u: &[C64]) -> Array2<C64> {
        let cols = self.orders.len() * self.memory;
        let mut phi = Array2::<C64>::zeros((u.len(), cols));
        for (j, k) in self.orders.iter().enumerate() {
            let b = polynomial_term(u, *k);
            for m in 0..self.memory {
                for n in m..u.len() {
                    phi[[n, j * self.memory + m]] = b[n - m];
                }
            }
        }
        phi
    }

    /// Predistort a block ([`memory_polynomial`] with the coefficients of every order).
    ///
    /// # Panics
    /// If there are not `orders.len() × memory` coefficients.
    pub fn apply(&self, x: &[C64]) -> Vec<C64> {
        let memory = self.memory.max(1);
        assert_eq!(
            self.coefficients.len(),
            self.orders.len() * self.memory,
            "{} coefficients for {} orders of memory {}",
            self.coefficients.len(),
            self.orders.len(),
            self.memory
        );
        memory_polynomial(x, self.orders.iter().copied().zip(self.coefficients.chunks(memory)))
    }

    /// Fit the polynomial that maps `input` to `target` by regularised least squares.
    ///
    /// # Errors
    /// [`Error::Shape`] for inputs of different lengths, [`Error::Config`] for no terms,
    /// [`Error::Numerical`] for a singular system.
    pub fn fit(&mut self, input: &[C64], target: &[C64], regularization: f64) -> Result<()> {
        ensure(!self.orders.is_empty(), || "the predistorter has no terms".into())?;
        ensure_shape(input.len() == target.len(), || format!("{} inputs for {} targets", input.len(), target.len()))?;
        let phi = self.basis(input);
        let ph = hermitian(&phi.view());
        let a = ph.dot(&phi) + identity(phi.ncols()).mapv(|v| v * regularization.max(0.0));
        let b = ph
            .dot(&Array2::from_shape_vec((target.len(), 1), target.to_vec()).map_err(|e| Error::Shape(e.to_string()))?);
        let c = solve_hpd(&a.view(), &b.view())
            .map_err(|_| Error::Numerical("the least-squares system is singular".into()))?;
        self.coefficients = c.column(0).to_vec();
        Ok(())
    }

    /// Indirect learning: `iterations` times, predistort `x`, amplify with `pa`, fit the
    /// post-inverse from the output divided by the complex target gain `gain` to the
    /// amplifier input, and use it as the new predistorter.
    ///
    /// # Errors
    /// The errors of `pa`, as [`MemoryPolynomialDpd::fit`], and [`Error::Config`] for a
    /// zero gain.
    pub fn train_indirect(
        &mut self,
        x: &[C64],
        mut pa: impl FnMut(&[C64]) -> Result<Vec<C64>>,
        gain: C64,
        iterations: usize,
        regularization: f64,
    ) -> Result<()> {
        ensure(gain.norm() != 0.0, || "the target gain must not be zero".into())?;
        for _ in 0..iterations {
            let z = self.apply(x);
            let y: Vec<C64> = pa(&z)?.into_iter().map(|v| v / gain).collect();
            self.fit(&y, &z, regularization)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn the_default_predistorter_is_the_identity_of_ding_2004() {
        let d =
            MemoryPolynomialDpd::new(MemoryPolynomialDpd::DEFAULT_ORDERS.to_vec(), MemoryPolynomialDpd::DEFAULT_MEMORY);
        assert_eq!((d.orders.as_slice(), d.memory, d.coefficients.len()), (&[1, 3, 5][..], 3, 9));
        let x = [C64::new(0.3, -0.1), C64::new(0.2, 0.4)];
        assert_eq!(d.apply(&x), x.to_vec());
    }

    #[test]
    fn the_fit_recovers_a_memory_polynomial() {
        let mut r = Rng::new(1);
        let x: Vec<C64> = (0..2000).map(|_| r.complex_gaussian() * 0.5).collect();
        let truth = MemoryPolynomialDpd {
            orders: vec![1, 3],
            memory: 2,
            coefficients: vec![C64::new(1.0, 0.1), C64::new(-0.05, 0.02), C64::new(-0.2, 0.05), C64::new(0.03, 0.0)],
        };
        let y = truth.apply(&x);
        let mut fit = MemoryPolynomialDpd::new(vec![1, 3], 2);
        fit.fit(&x, &y, 0.0).unwrap();
        for (a, b) in fit.coefficients.iter().zip(&truth.coefficients) {
            assert!((a - b).norm() < 1e-10);
        }
        assert!(fit.fit(&x, &y[..10], 0.0).is_err());
    }

    /// `apply` evaluated the basis matrix times the coefficients before the kernel was
    /// shared with the amplifier: the two agree to the rounding of their sums.
    #[test]
    fn apply_is_the_basis_matrix_times_the_coefficients() {
        let mut r = Rng::new(7);
        let x: Vec<C64> = (0..3000).map(|_| r.complex_gaussian() * 0.4).collect();
        for (orders, memory) in [(vec![1, 3, 5], 3), (vec![1, 2, 3], 1), (vec![1, 3], 40)] {
            let coefficients: Vec<C64> = (0..orders.len() * memory).map(|_| r.complex_gaussian() * 0.3).collect();
            let dpd = MemoryPolynomialDpd { orders, memory, coefficients };
            let old = dpd.basis(&x).dot(&ndarray::Array1::from(dpd.coefficients.clone())).to_vec();
            let new = dpd.apply(&x);
            let scale = old.iter().map(|v| v.norm()).fold(0.0, f64::max);
            let worst = old.iter().zip(&new).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max) / scale;
            assert!(worst < 1e-14, "memory {memory}: {worst}");
        }
    }
}
