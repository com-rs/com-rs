//! Constellations, their mapping and demapping, and the Gray code.

use super::*;

/// A constellation with Gray labels: point `i` carries the bits of `i`, MSB first.
///
/// The points and their labels are those of OptiCommPy (`GrayMapping`): square QAM
/// as the product of two PAM listed row by row from the top in a "snake" order, PSK
/// on `e^{j2πk/M}`, PAM on `-(M-1), …, -1, 1, …, M-1`, OOK on `{0, 1}`; the `i`-th
/// point of the construction order gets the Gray label `i ^ (i >> 1)`. The grids keep
/// their integer spacing; call `normalized()` for unit mean energy.
///
/// Args:
///     kind (str): `"qam"`, `"psk"`, `"pam"` or `"ook"`.
///     m (int): the order, a power of two (`2` for OOK).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> c = comrs.Constellation("qam", 16).normalized()
///     >>> round(c.energy, 12)
///     1.0
///     >>> bits = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
///     >>> c.demodulate(c.modulate(bits)).tolist()
///     [0, 0, 0, 0, 1, 1, 1, 1]
#[pyclass(name = "Constellation", module = "comrs")]
#[derive(Clone)]
pub struct PyConstellation {
    /// The Rust constellation.
    pub inner: Constellation,
}

#[pymethods]
impl PyConstellation {
    #[new]
    fn new(kind: &str, m: usize) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::new(self::kind(kind)?, m).py()? })
    }

    /// A custom constellation from its points, labelled by their index.
    ///
    /// Args:
    ///     points (array_like): the `M` points (complex), `M` a power of two.
    ///
    /// Returns:
    ///     Constellation: the constellation (kind `"custom"`).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Constellation.from_points([1, -1]).order
    ///     2
    #[staticmethod]
    fn from_points(points: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::from_points(cvec(points)?).py()? })
    }

    /// The points of `Constellation(kind, m)` with a set-partitioning labelling
    /// (Ungerboeck): natural labels for PSK, Fig. 2(a) of Caire, Taricco & Biglieri
    /// (1998) for 16-QAM.
    ///
    /// Args:
    ///     kind (str): `"psk"` or `"qam"` (16-QAM only).
    ///     m (int): the order.
    ///
    /// Returns:
    ///     Constellation: the constellation (kind `"custom"`).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Constellation.set_partitioning("qam", 16).points[0b1001]
    ///     np.complex128(-3+3j)
    #[staticmethod]
    fn set_partitioning(kind: &str, m: usize) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::set_partitioning(self::kind(kind)?, m).py()? })
    }

    /// A constellation from the name the specifications use (``"16-QAM"``, ``"QPSK"``,
    /// ``"8PSK"``, ``"4-PAM"``, ``"OOK"``; case, dashes and spaces are ignored), with
    /// its integer grid (call `normalized()` for unit energy).
    ///
    /// Args:
    ///     name (str): the name.
    ///
    /// Returns:
    ///     Constellation: the constellation.
    ///
    /// Raises:
    ///     ValueError: for an unknown family or an invalid order.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> c = comrs.Constellation.from_name("16-QAM")
    ///     >>> c.kind, c.order
    ///     ('qam', 16)
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::from_name(name).py()? })
    }

    /// str: The family, `"qam"`, `"psk"`, `"pam"`, `"ook"` or `"custom"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind().name()
    }

    /// int: The number of points `M`.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// int: Bits per symbol, `log2 M`.
    #[getter]
    fn bits_per_symbol(&self) -> usize {
        self.inner.bits_per_symbol()
    }

    /// numpy.ndarray: The points, in label order (a copy, complex128).
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> CArr<'py> {
        carr(py, self.inner.points().to_vec())
    }

    /// float: Mean energy of the points under uniform probabilities.
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy()
    }

    /// Mean energy of the points under the probabilities `px`.
    ///
    /// Args:
    ///     px (array_like): the probabilities of the points.
    ///
    /// Returns:
    ///     float: `Σ p_i |s_i|²`.
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn energy_with(&self, px: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.energy_with(&crate::fvec(px)?))
    }

    /// The same constellation scaled to unit mean energy.
    ///
    /// Returns:
    ///     Constellation: the normalised constellation.
    fn normalized(&self) -> Self {
        PyConstellation { inner: self.inner.normalized() }
    }

    /// The same constellation with every point multiplied by `s`.
    ///
    /// Args:
    ///     s (float): the scale.
    ///
    /// Returns:
    ///     Constellation: the scaled constellation.
    fn scaled(&self, s: f64) -> Self {
        PyConstellation { inner: self.inner.scaled(s) }
    }

    /// Symbol indices of a bit sequence (`log2 M` bits per symbol, MSB first).
    ///
    /// Args:
    ///     bits (array_like): the bits, a multiple of `log2 M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the indices (int64).
    ///
    /// Raises:
    ///     ValueError: if the length is not a multiple of `log2 M`.
    fn indices<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let idx = self.inner.indices(&bvec(bits)?).py()?;
        Ok(iarr(py, idx))
    }

    /// Symbols of a bit sequence (OptiCommPy `modulateGray`).
    ///
    /// Args:
    ///     bits (array_like): the bits, a multiple of `log2 M`.
    ///
    /// Returns:
    ///     numpy.ndarray: one symbol per `log2 M` bits (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn modulate<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&bvec(bits)?).py()?))
    }

    /// Symbols of an index sequence.
    ///
    /// Args:
    ///     indices (array_like): the indices, below `M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the points (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn symbols<'py>(&self, py: Python<'py>, indices: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let idx = uvec(indices)?;
        if let Some(bad) = idx.iter().find(|&&i| i >= self.inner.order()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("index {bad} out of range")));
        }
        Ok(carr(py, self.inner.symbols(&idx)))
    }

    /// Index of the nearest point of every sample (minimum-distance decisions).
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the indices (int64).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn decide<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let z = cvec(z)?;
        Ok(iarr(py, self.inner.decide(&z)))
    }

    /// The nearest point of every sample.
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the points (complex128).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn project<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.project(&cvec(z)?)))
    }

    /// Hard-decision bits of every sample (OptiCommPy `demodulateGray`).
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: `log2 M` bits per sample (uint8).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn demodulate<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.demodulate(&cvec(z)?).into_pyarray(py))
    }

    /// Bit log-likelihood ratios `ln p(b=0|y) - ln p(b=1|y)`, `log2 M` per sample.
    ///
    /// The likelihood of point `s` is `exp(-|y - s|²/σ²)` (OptiCommPy `calcLLR`).
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///     noise_var (float): complex noise variance `E|w|²` (for a real constellation
    ///         over a real channel, twice the variance of the real noise).
    ///     px (array_like, optional): prior probabilities of the points.
    ///     max_log (bool): max-log approximation instead of log-sum-exp.
    ///
    /// Returns:
    ///     numpy.ndarray: `log2 M` LLRs per sample, MSB first (float64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> c = comrs.Constellation("qam", 4).normalized()
    ///     >>> llr = c.llr(c.modulate([0, 1]), 0.1)
    ///     >>> bool(llr[0] > 0 and llr[1] < 0)
    ///     True
    #[pyo3(signature = (z, noise_var, *, px = None, max_log = false))]
    fn llr<'py>(
        &self,
        py: Python<'py>,
        z: &Bound<'py, PyAny>,
        noise_var: f64,
        px: Option<&Bound<'py, PyAny>>,
        max_log: bool,
    ) -> PyResult<FArr<'py>> {
        let z = cvec(z)?;
        let px = px.map(crate::fvec).transpose()?;
        let c = &self.inner;
        let v = py.detach(|| c.llr(&z, noise_var, px.as_deref(), max_log)).py()?;
        Ok(farr(py, v))
    }

    fn __len__(&self) -> usize {
        self.inner.order()
    }

    fn __repr__(&self) -> String {
        format!("comrs.Constellation('{}', {}, energy={:.6})", self.kind(), self.inner.order(), self.inner.energy())
    }
}

/// The reflected binary Gray code of `n_bits` bits: `g[i] = i ^ (i >> 1)`.
///
/// Args:
///     n_bits (int): the number of bits.
///
/// Returns:
///     numpy.ndarray: the `2^n_bits` code words (int64).
///
/// Raises:
///     ValueError: for more than 24 bits (more than 16 777 216 code words).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.gray_code(2).tolist()
///     [0, 1, 3, 2]
#[pyfunction]
fn gray_code(py: Python<'_>, n_bits: u32) -> PyResult<IArr<'_>> {
    Ok(iarr(py, modulation::gray_code(n_bits).py()?))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConstellation>()?;
    m.add_function(wrap_pyfunction!(gray_code, m)?)?;
    Ok(())
}
