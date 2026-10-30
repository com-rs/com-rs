//! Probabilistic shaping: the Maxwell–Boltzmann distribution, the entropy and the
//! constant-composition distribution matcher.

use super::*;

/// Maxwell–Boltzmann probabilities `p(x) ∝ exp(-λ|x|²)` of the points.
///
/// Args:
///     constellation (Constellation): the alphabet.
///     lam (float): the shaping parameter `λ` (0 for uniform).
///
/// Returns:
///     numpy.ndarray: one probability per point, summing to one (float64).
#[pyfunction]
fn maxwell_boltzmann<'py>(py: Python<'py>, constellation: &PyConstellation, lam: f64) -> FArr<'py> {
    farr(py, sources::maxwell_boltzmann(&constellation.inner, lam))
}

/// The entropy ``H = -Σ p log2 p`` of a distribution (bits): the information per symbol of
/// a shaped source.
///
/// Args:
///     px (array_like): the probabilities.
///
/// Returns:
///     float: the entropy (bits).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.entropy([0.25, 0.25, 0.25, 0.25])
///     2.0
#[pyfunction]
fn entropy(px: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(metrics::entropy(&crate::fvec(px)?))
}

/// Constant-composition distribution matching (Schulte & Böcherer 2016): ``k`` uniform
/// bits to a sequence of ``n`` symbols of a fixed composition, by enumerative coding.
///
/// Args:
///     composition (list[int]): occurrences of every symbol in a block.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> m = comrs.Ccdm([2, 1, 1])
///     >>> b = np.zeros(m.k, dtype=np.uint8)
///     >>> m.dematch(m.match_bits(b)).tolist() == b.tolist()
///     True
#[pyclass(name = "Ccdm", module = "comrs")]
pub struct PyCcdm {
    inner: Ccdm,
}

#[pymethods]
impl PyCcdm {
    fn __repr__(&self) -> String {
        format!("Ccdm(n={}, k={})", self.inner.n(), self.inner.k())
    }

    #[new]
    fn new(composition: Vec<usize>) -> PyResult<Self> {
        Ok(PyCcdm { inner: Ccdm::new(composition).py()? })
    }

    /// The matcher of blocks of ``n`` symbols nearest to a distribution.
    ///
    /// Args:
    ///     p (array_like): the target distribution of the symbols.
    ///     n (int): symbols per block.
    ///
    /// Returns:
    ///     Ccdm: the matcher.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[staticmethod]
    fn from_distribution(p: &Bound<'_, PyAny>, n: i64) -> PyResult<Self> {
        Ok(PyCcdm { inner: Ccdm::from_distribution(&fvec(p)?, count(n, "n")?).py()? })
    }

    /// numpy.ndarray: Occurrences of every symbol in a block (int64).
    #[getter]
    fn composition<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.composition().to_vec())
    }

    /// int: Symbols per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// int: Bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// float: Bits per symbol, ``k / n``.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate()
    }

    /// numpy.ndarray: The distribution of the symbols, ``composition / n`` (float64).
    #[getter]
    fn distribution<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.distribution())
    }

    /// The symbols of one block of bits.
    ///
    /// Args:
    ///     bits (array_like): ``k`` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: ``n`` symbol indices (int64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn match_bits<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let s = self.inner.match_bits(&bvec(bits)?).py()?;
        Ok(iarr(py, s))
    }

    /// The bits of a block of symbols (a used-up symbol is replaced by the nearest
    /// available one).
    ///
    /// Args:
    ///     symbols (array_like): ``n`` symbol indices.
    ///
    /// Returns:
    ///     numpy.ndarray: ``k`` bits (uint8).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn dematch<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.dematch(&uvec(symbols)?).py()?.into_pyarray(py))
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(maxwell_boltzmann, m)?)?;
    m.add_function(wrap_pyfunction!(entropy, m)?)?;
    m.add_class::<PyCcdm>()?;
    Ok(())
}
