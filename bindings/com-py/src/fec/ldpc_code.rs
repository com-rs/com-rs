//! LDPC codes: progressive edge growth, a parity-check matrix, belief propagation.

use super::*;

/// An LDPC code: `LdpcCode.peg(n, k, var_degree=…, seed=…)` (progressive edge growth,
/// Hu, Eleftheriou & Arnold 2005) or `LdpcCode(n, rows)` from the rows of a
/// parity-check matrix. Decoded by belief propagation (see `set_decoder`): `rule` is
/// `"sum_product"` (the default, as in the specifications), `"min_sum"`,
/// `"normalized_min_sum"` (scaled by `alpha`, 0.8 by default) or `"offset_min_sum"`
/// (offset `alpha`); `layered` selects the layered schedule (Hocevar 2004, the
/// default); 50 iterations at most by default. The PEG construction uses the variable
/// degree 3 and the seed 1 by default.
///
/// Args:
///     n (int): coded bits per block.
///     rows (list[list[int]]): the positions of the ones of every row of the
///         parity-check matrix.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
#[pyclass(name = "LdpcCode", module = "comrs")]
pub struct PyLdpc {
    inner: LdpcCode,
}

#[pymethods]
impl PyLdpc {
    fn __repr__(&self) -> String {
        format!("LdpcCode(n={}, k={})", self.inner.n(), self.inner.k())
    }

    #[new]
    fn new(n: usize, rows: Vec<Vec<usize>>) -> PyResult<Self> {
        Ok(PyLdpc { inner: LdpcCode::from_rows(n, rows).py()? })
    }

    /// A code built by progressive edge growth.
    ///
    /// Args:
    ///     n (int): coded bits per block.
    ///     k (int): information bits per block.
    ///     var_degree (int): the degree of every variable node.
    ///     seed (int): the seed of the construction.
    ///
    /// Returns:
    ///     LdpcCode: the code.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> code = comrs.LdpcCode.peg(96, 48)
    ///     >>> code.n, code.k, code.girth() >= 6
    ///     (96, 48, True)
    #[staticmethod]
    #[pyo3(signature = (n, k, *, var_degree=ldpc::DEFAULT_VAR_DEGREE, seed=ldpc::DEFAULT_PEG_SEED))]
    fn peg(n: usize, k: usize, var_degree: usize, seed: u64) -> PyResult<Self> {
        Ok(PyLdpc { inner: LdpcCode::peg(n, k, var_degree, seed).py()? })
    }

    /// Choose the decoder.
    ///
    /// Args:
    ///     rule (str): the check-node rule, ``"sum_product"``, ``"min_sum"``,
    ///         ``"normalized_min_sum"`` (scaled by ``alpha``) or ``"offset_min_sum"``
    ///         (with the offset ``alpha``), as the specifications.
    ///     alpha (float): the scale or the offset of the min-sum rules.
    ///     layered (bool): the layered schedule (else flooding).
    ///     max_iterations (int): iterations at most.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    #[pyo3(signature = (
        rule=ldpc::DEFAULT_RULE.name(),
        *,
        alpha=ldpc::DEFAULT_ALPHA,
        layered=ldpc::DEFAULT_LAYERED,
        max_iterations=ldpc::DEFAULT_MAX_ITERATIONS,
    ))]
    fn set_decoder(&mut self, rule: &str, alpha: f64, layered: bool, max_iterations: usize) -> PyResult<()> {
        self.inner.rule = CheckRule::from_name(rule, alpha).py()?;
        self.inner.layered = layered;
        self.inner.max_iterations = max_iterations;
        Ok(())
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Coded bits per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// float: Code rate.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// list[list[int]]: The rows of the parity-check matrix (the positions of their
    /// ones).
    #[getter]
    fn rows(&self) -> Vec<Vec<usize>> {
        self.inner.rows().to_vec()
    }

    /// The girth of the Tanner graph.
    ///
    /// Returns:
    ///     int: the length of its shortest cycle.
    fn girth(&self) -> usize {
        self.inner.girth()
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` coded bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.LdpcCode.peg(96, 48).encode(comrs.bits(96, comrs.Rng(1))).size
    ///     192
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (belief propagation).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> code = comrs.LdpcCode.peg(96, 48)
    ///     >>> b = comrs.bits(48, comrs.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }

    /// Decode one block into the a posteriori LLRs of the coded bits.
    ///
    /// Args:
    ///     llr (array_like): `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int]: the `n` a posteriori LLRs and the iterations run.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn decode_llr<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<(FArr<'py>, usize)> {
        let l = fvec(llr)?;
        let (post, it) = py.detach(|| self.inner.decode_llr(&l)).py()?;
        Ok((farr(py, post), it))
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLdpc>()?;
    Ok(())
}
