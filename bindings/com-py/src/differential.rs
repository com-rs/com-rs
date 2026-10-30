//! Differential M-PSK: encoding, noncoherent and coherent detection, and the closed
//! forms of its error rates.

use crate::{BArr, CArr, IArr, IntoPy, bvec, carr, cvec, iarr, map_real, uvec};
use comrs::mapping::differential::{self, Detection, Dpsk};
use comrs::mapping::theory;
use numpy::IntoPyArray;
use pyo3::prelude::*;

/// Differential M-PSK (DBPSK, DQPSK, D8PSK, …) with Gray-labelled phase transitions.
///
/// The information is carried by the phase transitions: with `a_k` the differential
/// phase index of symbol `k` and `c_{-1} = 0` the reference, `c_k = (a_k + c_{k-1}) mod
/// M` and `x_k = exp(j 2 pi c_k / M)`, so `n` information symbols are sent as `n + 1`
/// unit-energy PSK symbols. The bits of a symbol are the Gray label of its transition,
/// so the nearest wrong transitions differ by one bit. The noncoherent detector
/// decides the transition nearest to `arg(r_k conj(r_{k-1}))`, which a constant phase
/// rotation of the received symbols leaves unchanged (Proakis & Salehi, *Digital
/// Communications*, 5th ed., 2008, Sec. 4.5); its bit error rate over AWGN is
/// `theory_ber_dpsk(m, ebn0_db)`. Coherent decisions on the absolute phase followed by
/// differential decoding spoil two differences per symbol error, hence `2p(1 - p)`
/// for DBPSK (`theory_ber_dpsk(2, ebn0_db, detection="coherent")`, Sec. 3.2-2).
///
/// The methods work at two levels: bits (`modulate`, `demodulate`,
/// `demodulate_coherent`, and `indices`/`bits` between bits and differential indices)
/// and phase indices (`encode` and `decode` between differential and absolute indices,
/// `detect` and `detect_coherent` from received symbols to indices).
///
/// Args:
///     m (int): the order, a power of two (2 for DBPSK, 4 for DQPSK).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> dqpsk = comrs.Dpsk(4)
///     >>> bits = np.array([0, 1, 1, 1, 0, 0], dtype=np.uint8)
///     >>> x = dqpsk.modulate(bits)
///     >>> x.shape
///     (4,)
///     >>> dqpsk.demodulate(x * np.exp(1j * 1.234)).tolist()
///     [0, 1, 1, 1, 0, 0]
#[pyclass(name = "Dpsk", module = "comrs")]
#[derive(Clone)]
pub struct PyDpsk {
    inner: Dpsk,
}

#[pymethods]
impl PyDpsk {
    #[new]
    fn new(m: usize) -> PyResult<Self> {
        Ok(PyDpsk { inner: Dpsk::new(m).py()? })
    }

    /// int: The order `M`.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// int: Bits per symbol, `log2 M`.
    #[getter]
    fn bits_per_symbol(&self) -> usize {
        self.inner.bits_per_symbol()
    }

    /// The differential phase indices of a bit stream (`log2 M` bits per symbol, MSB
    /// first, Gray-labelled transitions).
    ///
    /// Args:
    ///     bits (array_like): the bits, a multiple of `log2 M`.
    ///
    /// Returns:
    ///     numpy.ndarray: one differential index per symbol (int64).
    ///
    /// Raises:
    ///     ValueError: if the length is not a multiple of `log2 M`.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Dpsk(4).indices([0, 0, 0, 1, 1, 1, 1, 0]).tolist()
    ///     [0, 1, 2, 3]
    fn indices<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, self.inner.indices(&bvec(bits)?).py()?))
    }

    /// The bits of a sequence of differential phase indices (the inverse of `indices`).
    ///
    /// Args:
    ///     indices (array_like): differential indices, below `M`.
    ///
    /// Returns:
    ///     numpy.ndarray: `log2 M` bits per index (uint8).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Dpsk(4).bits([0, 1, 2, 3]).tolist()
    ///     [0, 0, 0, 1, 1, 1, 1, 0]
    fn bits<'py>(&self, py: Python<'py>, indices: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.bits(&uvec(indices)?).py()?.into_pyarray(py))
    }

    /// Differential encoding of phase indices: the `n + 1` absolute indices
    /// `c_{-1} = 0, c_0, …, c_{n-1}` with `c_k = (a_k + c_{k-1}) mod M`.
    ///
    /// Args:
    ///     indices (array_like): the `n` differential phase indices, below `M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the `n + 1` absolute phase indices (int64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Dpsk(4).encode([0, 1, 3, 2, 1]).tolist()
    ///     [0, 0, 1, 0, 2, 3]
    fn encode<'py>(&self, py: Python<'py>, indices: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, differential::encode(&uvec(indices)?, self.inner.order()).py()?))
    }

    /// Differential decoding: the `n` differences `a_k = (c_k - c_{k-1}) mod M` of
    /// `n + 1` absolute phase indices (the inverse of `encode`).
    ///
    /// Args:
    ///     coded (array_like): the `n + 1` absolute phase indices, below `M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the `n` differential phase indices (int64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Dpsk(4).decode([0, 0, 1, 0, 2, 3]).tolist()
    ///     [0, 1, 3, 2, 1]
    fn decode<'py>(&self, py: Python<'py>, coded: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, differential::decode(&uvec(coded)?, self.inner.order()).py()?))
    }

    /// Noncoherent detection: the differential phase index nearest to
    /// `arg(r_k conj(r_{k-1}))` for every pair of consecutive received symbols,
    /// invariant to a constant phase rotation (Proakis & Salehi, 5th ed., Sec. 4.5).
    ///
    /// Args:
    ///     r (array_like): the `n + 1` received symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: the `n` differential phase indices (int64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> r = np.exp(1j * (0.3 + np.array([0.0, np.pi / 2, 0.0])))
    ///     >>> comrs.Dpsk(4).detect(r).tolist()
    ///     [1, 3]
    fn detect<'py>(&self, py: Python<'py>, r: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, differential::detect(&cvec(r)?, self.inner.order()).py()?))
    }

    /// Coherent decisions: the absolute phase index nearest to every received symbol
    /// (the hard decision of M-PSK); `decode` then removes the differential encoding.
    ///
    /// Args:
    ///     r (array_like): the received symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: one absolute phase index per symbol (int64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> comrs.Dpsk(4).detect_coherent(np.exp(1j * np.pi / 2 * np.array([0, 1, 1, 2]))).tolist()
    ///     [0, 1, 1, 2]
    fn detect_coherent<'py>(&self, py: Python<'py>, r: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, differential::detect_coherent(&cvec(r)?, self.inner.order()).py()?))
    }

    /// The unit-energy PSK symbols carrying a bit stream (the reference symbol first).
    ///
    /// Args:
    ///     bits (array_like): `n` bits, a multiple of `log2 M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the `n / log2 M + 1` symbols (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Dpsk(2).modulate([1, 0]).real.round(12).tolist()
    ///     [1.0, -1.0, -1.0]
    fn modulate<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&bvec(bits)?).py()?))
    }

    /// Noncoherent demodulation: the transition nearest to `arg(r_k conj(r_{k-1}))`
    /// (no carrier phase needed), as bits.
    ///
    /// Args:
    ///     r (array_like): the `n + 1` received symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: the `n log2 M` bits (uint8).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> d = comrs.Dpsk(2)
    ///     >>> d.demodulate(d.modulate([1, 0, 1]) * 1j).tolist()
    ///     [1, 0, 1]
    fn demodulate<'py>(&self, py: Python<'py>, r: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.demodulate(&cvec(r)?).py()?.into_pyarray(py))
    }

    /// Coherent demodulation with differential decoding: the phase index nearest to
    /// every symbol, then the differences, as bits.
    ///
    /// Args:
    ///     r (array_like): the `n + 1` received symbols (carrier phase recovered).
    ///
    /// Returns:
    ///     numpy.ndarray: the `n log2 M` bits (uint8).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> d = comrs.Dpsk(4)
    ///     >>> d.demodulate_coherent(d.modulate([1, 1, 0, 1])).tolist()
    ///     [1, 1, 0, 1]
    fn demodulate_coherent<'py>(&self, py: Python<'py>, r: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.demodulate_coherent(&cvec(r)?).py()?.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("Dpsk({})", self.inner.order())
    }
}

/// Bit error rate of Gray-labelled M-DPSK over AWGN.
///
/// With noncoherent detection (the default): `½ exp(-Eb/N0)` for DBPSK (Proakis &
/// Salehi, *Digital Communications*, 5th ed., 2008, Sec. 4.5); for `M >= 4` the average
/// number of wrong bits over the decision sectors of the differential phase, whose
/// probabilities come from the distribution of the phase angle between two vectors of
/// the same SNR `Es/N0 = Eb/N0 log2 M` perturbed by Gaussian noise (Pawula, Rice &
/// Roberts, *IEEE Trans. Commun.* 30(8), 1982, Sec. IV; see
/// `differential_phase_tail`). For DQPSK, `Pb = ½ [P(|dθ| > π/4) + P(|dθ| > 3π/4)]`.
///
/// With coherent detection and differential decoding (DBPSK only): an isolated symbol
/// error spoils two differences, so `Pb = 2p(1 - p)` with `p = Q(sqrt(2 Eb/N0))`
/// (Proakis & Salehi, 5th ed., Sec. 3.2-2).
///
/// Args:
///     m (int): the order, a power of two.
///     ebn0_db (float | array_like): `Eb/N0` (dB).
///     detection (str): `"noncoherent"` or `"coherent"` (with differential decoding,
///         defined for `m = 2`).
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every `Eb/N0`.
///
/// Raises:
///     ValueError: for an unknown detection, or coherent detection with `m != 2`.
///
/// Examples:
///     >>> import comrs, math
///     >>> round(comrs.theory_ber_dpsk(2, 8.0) / (0.5 * math.exp(-comrs.db2lin(8.0))), 12)
///     1.0
///     >>> comrs.theory_ber_dpsk(4, 8.0) > comrs.theory_ber_dpsk(2, 8.0)
///     True
///     >>> p = comrs.qfunc(math.sqrt(2 * comrs.db2lin(6.0)))
///     >>> round(comrs.theory_ber_dpsk(2, 6.0, detection="coherent") / (2 * p * (1 - p)), 12)
///     1.0
#[pyfunction]
#[pyo3(signature = (m, ebn0_db, *, detection = "noncoherent"))]
fn theory_ber_dpsk(m: usize, ebn0_db: &Bound<'_, PyAny>, detection: &str) -> PyResult<Py<PyAny>> {
    let detection: Detection = detection.parse().py()?;
    map_real(ebn0_db, |e| theory::ber_differential(m, e, detection).py())
}

/// The probability that the phase angle between two vectors of the same SNR perturbed
/// by independent Gaussian noise exceeds `psi` in absolute value, `0 <= psi <= pi`.
///
/// The integral form of Pawula, Rice & Roberts (*IEEE Trans. Commun.* 30(8), 1982,
/// Sec. IV) for equal SNRs, by Simpson's rule. It is the symbol error rate of M-DPSK
/// for `psi = pi / M`, and `½ exp(-Es/N0)` for `psi = pi / 2`.
///
/// Args:
///     psi (float): the threshold angle (rad).
///     esn0 (float | array_like): `Es/N0` of each vector (linear).
///
/// Returns:
///     float | numpy.ndarray: `P(|dθ| > psi)` at every `Es/N0`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs, math
///     >>> round(comrs.differential_phase_tail(math.pi / 2, 5.0) / (0.5 * math.exp(-5.0)), 8)
///     1.0
#[pyfunction]
fn differential_phase_tail(psi: f64, esn0: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(esn0, |g| Ok(theory::differential_phase_tail(psi, g)))
}

/// Register the differential PSK bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDpsk>()?;
    m.add_function(wrap_pyfunction!(theory_ber_dpsk, m)?)?;
    m.add_function(wrap_pyfunction!(differential_phase_tail, m)?)?;
    Ok(())
}
