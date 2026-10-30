//! Polar codes, decoded by SC or CRC-aided SC list.

use super::*;

/// A polar code of length `n` (a power of two) and `k` information bits, built by the
/// Gaussian approximation at `design_ebn0_db` (Trifonov 2012); decoded by SC
/// (Arıkan 2009), or SC list of size `list_size` (Tal & Vardy 2015) with the CRC `crc`
/// choosing the path (CA-SCL).
///
/// Args:
///     n (int): coded bits per block, a power of two.
///     k (int): information bits per block.
///     design_ebn0_db (float): the `Eb/N0` of the construction (dB).
///     crc (str, optional): the CRC of the list decoder (a name of `Crc`).
///     list_size (int): the list size `L` (1 for SC).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.PolarCode(128, 64, crc="crc6", list_size=8)
///     >>> b = comrs.bits(64, comrs.Rng(2))
///     >>> llr = 4.0 * (1.0 - 2.0 * code.encode(b))
///     >>> bool(np.array_equal(code.decode(llr), b))
///     True
#[pyclass(name = "PolarCode", module = "comrs")]
pub struct PyPolar {
    inner: PolarCode,
}

#[pymethods]
impl PyPolar {
    fn __repr__(&self) -> String {
        format!("PolarCode(n={}, k={})", self.inner.n(), self.inner.k())
    }

    #[new]
    #[pyo3(signature = (n, k, *, design_ebn0_db=2.0, crc=None, list_size=1))]
    fn new(n: usize, k: usize, design_ebn0_db: f64, crc: Option<&str>, list_size: usize) -> PyResult<Self> {
        let mut code = PolarCode::new(n, k, design_ebn0_db).py()?;
        if let Some(c) = crc {
            code = code.with_crc(Crc::from_name(c).py()?).py()?;
        }
        Ok(PyPolar { inner: code.with_list(list_size) })
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

    /// numpy.ndarray: The positions of the information and CRC bits (int64).
    #[getter]
    fn info_positions<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.info_positions().to_vec())
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
    ///     >>> comrs.PolarCode(64, 32).encode(comrs.bits(64, comrs.Rng(1))).size
    ///     128
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (SC or CA-SCL).
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
    ///     >>> code = comrs.PolarCode(64, 32)
    ///     >>> b = comrs.bits(32, comrs.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPolar>()?;
    Ok(())
}
