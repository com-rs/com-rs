//! The LDPC code of the data channels and the polar code of the control channels of 5G
//! NR (3GPP TS 38.212).

use super::*;

/// The LDPC code of the data channels of 5G NR (3GPP TS 38.212, Sec. 5.3.2, 5.4.2): one
/// code block of `k` information bits rate-matched to `n` bits; base graph 1 or 2 chosen
/// as the shared channels do (or forced), the smallest lifting size `Z` with
/// `K_b Z ≥ k`, filler bits, the first `2Z` bits punctured, the circular buffer read
/// from the start of redundancy version `rv`, the bit interleaver of `qm` bits per
/// symbol; decoded by belief propagation.
///
/// Args:
///     k (int): information bits per block (at most 8448 for base graph 1, 3840 for 2).
///     e (int): transmitted bits per block `E` (the rate-matched length).
///     base_graph (int, optional): 1 or 2 (default: the rule of Sec. 7.2.2).
///     rv (int): the redundancy version, 0 to 3.
///     qm (int): bits per modulation symbol of the bit interleaver (1: none).
///     rule (str): the check-node rule (as `LdpcCode.set_decoder`).
///     alpha (float): the scale or the offset of the min-sum rules.
///     max_iterations (int): the maximum number of iterations.
///     layered (bool): the layered schedule (default), or flooding (``False``); both stop
///         early on a codeword.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.NrLdpc(100, 300)
///     >>> code.base_graph, code.lifting_size
///     (2, 18)
///     >>> b = comrs.bits(100, comrs.Rng(1))
///     >>> bool(np.array_equal(code.decode(4.0 - 8.0 * code.encode(b)), b))
///     True
#[pyclass(name = "NrLdpc", module = "comrs")]
pub struct PyNrLdpc {
    inner: comrs::fec::nr::NrLdpc,
}

#[pymethods]
impl PyNrLdpc {
    fn __repr__(&self) -> String {
        format!(
            "NrLdpc(k={}, n={}, base_graph={}, lifting_size={})",
            self.inner.k(),
            self.inner.n(),
            self.inner.base_graph(),
            self.inner.lifting_size()
        )
    }

    #[new]
    #[pyo3(signature = (
        k,
        e,
        *,
        base_graph = None,
        rv = 0,
        qm = 1,
        rule = ldpc::DEFAULT_RULE.name(),
        alpha = ldpc::DEFAULT_ALPHA,
        max_iterations = ldpc::DEFAULT_MAX_ITERATIONS,
        layered = ldpc::DEFAULT_LAYERED,
    ))]
    #[allow(clippy::too_many_arguments, reason = "keyword-only options of the Python constructor")]
    fn new(
        k: usize,
        e: usize,
        base_graph: Option<u8>,
        rv: u8,
        qm: usize,
        rule: &str,
        alpha: f64,
        max_iterations: usize,
        layered: bool,
    ) -> PyResult<Self> {
        let mut code = comrs::fec::nr::NrLdpc::with_options(k, e, base_graph, rv, qm).py()?;
        code.decoder_mut().rule = CheckRule::from_name(rule, alpha).py()?;
        code.decoder_mut().max_iterations = max_iterations;
        code.decoder_mut().layered = layered;
        Ok(PyNrLdpc { inner: code })
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Transmitted bits per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// int: The base graph, 1 or 2.
    #[getter]
    fn base_graph(&self) -> u8 {
        self.inner.base_graph()
    }

    /// float: Code rate `k / n`.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// int: The lifting size `Z`.
    #[getter]
    fn lifting_size(&self) -> usize {
        self.inner.lifting_size()
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs.
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// The polar code of the control channels of 5G NR (3GPP TS 38.212, Sec. 5.3.1, 5.4.1):
/// `k` payload bits with their CRC rate-matched to `n` bits (puncturing, shortening or
/// repetition by the sub-block interleaver), the frozen set of the polar sequence of
/// Table 5.3.1.2-1; decoded by CRC-aided SC list.
///
/// ``"uplink"``: the uplink control information (CRC6 and 3 parity-check bits for
/// 12 ≤ k ≤ 19, CRC11 above, the triangular interleaver of the coded bits).
/// ``"downlink"``: the downlink control information and the broadcast channel (CRC24C,
/// the input interleaver; the RNTI masking of a DCI is left to the caller).
///
/// Args:
///     k (int): payload bits per block.
///     e (int): transmitted bits per block `E` (the rate-matched length).
///     link (str): ``"uplink"`` or ``"downlink"``.
///     list_size (int): the list size `L`.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.NrPolar(30, 100, list_size=8)
///     >>> code.mother_length
///     128
///     >>> b = comrs.bits(30, comrs.Rng(1))
///     >>> bool(np.array_equal(code.decode(3.0 - 6.0 * code.encode(b)), b))
///     True
#[pyclass(name = "NrPolar", module = "comrs")]
pub struct PyNrPolar {
    inner: comrs::fec::nr::NrPolar,
}

#[pymethods]
impl PyNrPolar {
    fn __repr__(&self) -> String {
        format!("NrPolar(k={}, n={}, mother_length={})", self.inner.k(), self.inner.n(), self.inner.mother_length())
    }

    #[new]
    #[pyo3(signature = (k, e, *, link="uplink", list_size=8))]
    fn new(k: usize, e: usize, link: &str, list_size: usize) -> PyResult<Self> {
        let code = comrs::fec::nr::NrPolar::from_name(link, k, e).py()?;
        Ok(PyNrPolar { inner: code.with_list(list_size).py()? })
    }

    /// int: Payload bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Transmitted bits per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// float: Code rate `k / n`.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// int: The mother code length `N`.
    #[getter]
    fn mother_length(&self) -> usize {
        self.inner.mother_length()
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (CRC-aided SC list).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNrLdpc>()?;
    m.add_class::<PyNrPolar>()?;
    Ok(())
}
