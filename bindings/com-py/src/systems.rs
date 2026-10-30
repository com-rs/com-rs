//! Standard-compliant chains: DVB-S2.

use crate::comm::PyConstellation;
use crate::{BArr, CArr, FArr, IntoPy, bvec, carr, cvec, farr};
use comrs::systems::dvbs2::{self, Dvbs2, FrameSize, Modcod};
use numpy::IntoPyArray;
use pyo3::prelude::*;

/// A DVB-S2 transmitter and receiver (ETSI EN 302 307-1) of one MODCOD, constant coding
/// and modulation.
///
/// The transmitter takes a BBFRAME of ``k`` bits (see :func:`dvbs2_bbframe`), encodes it
/// (BCH, then the LDPC code of the standard), interleaves the bits (8PSK, 16APSK,
/// 32APSK), maps them on the constellation of the MODCOD and builds the PLFRAME: the
/// PLHEADER in π/2-BPSK, the slots of 90 symbols, pilot blocks of 36 symbols every 16
/// slots, and the Gold scrambling sequence number ``scrambling_code``. Every symbol has
/// unit energy, so that a noise of variance ``1/(Es/N0)`` gives the abscissa of the
/// performance table of the standard (:attr:`threshold_db`). The receiver assumes the
/// frame found and the carrier recovered.
///
/// Args:
///     modulation (str): ``"QPSK"``, ``"8PSK"``, ``"16APSK"`` or ``"32APSK"``.
///     rate (str): ``"1/4"`` … ``"9/10"`` (the 28 MODCODs of Table 12).
///     frame (str): ``"normal"`` or ``"64800"`` (64 800 bits), ``"short"`` or ``"16200"``
///         (16 200 bits).
///     pilots (bool): pilot blocks.
///     scrambling_code (int): the number ``n`` of the Gold sequence.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> link = comrs.Dvbs2("8PSK", "3/5", frame="short")
///     >>> link.k, link.plframe_len
///     (9552, 5598)
///     >>> bb = comrs.dvbs2_bbframe(np.zeros(link.k - 80, dtype=np.uint8), link.k)
///     >>> x = link.modulate(bb)
///     >>> comrs.dvbs2_decode_plheader(x[:90])
///     (12, True, True)
///     >>> bool(np.array_equal(link.decode(x, 1e-3), bb))
///     True
#[pyclass(name = "Dvbs2", module = "comrs")]
pub struct PyDvbs2 {
    inner: Dvbs2,
}

#[pymethods]
impl PyDvbs2 {
    fn __repr__(&self) -> String {
        format!(
            "Dvbs2(modcod={}, k={}, plframe_len={})",
            self.inner.modcod().number().map_or_else(|_| "?".to_string(), |m| m.to_string()),
            self.inner.k(),
            self.inner.plframe_len()
        )
    }

    #[new]
    #[pyo3(signature = (modulation, rate, *, frame = "normal", pilots = true, scrambling_code = 0))]
    fn new(modulation: &str, rate: &str, frame: &str, pilots: bool, scrambling_code: usize) -> PyResult<Self> {
        let frame: FrameSize = frame.parse().py()?;
        let modcod = Modcod::from_name(modulation, rate).py()?;
        Ok(PyDvbs2 { inner: Dvbs2::new(modcod, frame, pilots, scrambling_code).py()? })
    }

    /// int: Bits of a BBFRAME (``K_bch``, Tables 5a and 5b).
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Symbols of a PLFRAME.
    #[getter]
    fn plframe_len(&self) -> usize {
        self.inner.plframe_len()
    }

    /// int: The MODCOD field of the PLHEADER (1 to 28, Table 12).
    #[getter]
    fn modcod(&self) -> PyResult<u8> {
        self.inner.modcod().number().py()
    }

    /// float: The ideal Es/N0 (dB) at a PER of 1e-7 of the normal frame (Table 13).
    #[getter]
    fn threshold_db(&self) -> f64 {
        self.inner.modcod().threshold_db()
    }

    /// Constellation: The constellation of the MODCOD (``points[l]`` carries the label
    /// ``l``).
    #[getter]
    fn constellation(&self) -> PyConstellation {
        PyConstellation { inner: self.inner.constellation().clone() }
    }

    /// int: LDPC iterations (sum-product, layered; read and write).
    #[getter]
    fn get_iterations(&self) -> usize {
        self.inner.fec().ldpc().max_iterations
    }

    #[setter]
    fn set_iterations(&mut self, n: usize) {
        self.inner.fec_mut().ldpc_mut().max_iterations = n;
    }

    /// The PLFRAME of a BBFRAME.
    ///
    /// Args:
    ///     bbframe (array_like): ``k`` bits (see :func:`dvbs2_bbframe`).
    ///
    /// Returns:
    ///     numpy.ndarray: the ``plframe_len`` unit-energy symbols (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn modulate<'py>(&self, py: Python<'py>, bbframe: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let b = bvec(bbframe)?;
        let x = py.detach(|| self.inner.modulate(&b)).py()?;
        Ok(carr(py, x))
    }

    /// The deinterleaved bit LLRs of the FECFRAME of a received PLFRAME.
    ///
    /// Args:
    ///     rx (array_like): the received PLFRAME (``plframe_len`` symbols).
    ///     noise_var (float): complex noise variance per symbol.
    ///     max_log (bool): max-log LLRs instead of the exact ones.
    ///
    /// Returns:
    ///     numpy.ndarray: one LLR per bit of the FECFRAME (positive for a 0).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (rx, noise_var, *, max_log = false))]
    fn llr<'py>(&self, py: Python<'py>, rx: &Bound<'py, PyAny>, noise_var: f64, max_log: bool) -> PyResult<FArr<'py>> {
        let y = cvec(rx)?;
        let l = py.detach(|| self.inner.llr(&y, noise_var, max_log)).py()?;
        Ok(farr(py, l))
    }

    /// The BBFRAME of a received PLFRAME (LDPC, then BCH decoding).
    ///
    /// Args:
    ///     rx (array_like): the received PLFRAME (``plframe_len`` symbols).
    ///     noise_var (float): complex noise variance per symbol.
    ///
    /// Returns:
    ///     numpy.ndarray: the ``k`` bits of the BBFRAME (uint8).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn decode<'py>(&self, py: Python<'py>, rx: &Bound<'py, PyAny>, noise_var: f64) -> PyResult<BArr<'py>> {
        let y = cvec(rx)?;
        let b = py.detach(|| self.inner.decode(&y, noise_var)).py()?;
        Ok(b.into_pyarray(py))
    }
}

/// The BBFRAME of ``k_bch`` bits of a generic continuous stream: a BBHEADER (CCM, single
/// stream, DFL = ``k_bch - 80``, CRC-8) and the ``k_bch - 80`` data bits, randomised by the
/// PRBS of the standard.
///
/// Args:
///     data (array_like): the ``k_bch - 80`` data bits.
///     k_bch (int): the bits of the BBFRAME (``Dvbs2.k``).
///     rolloff (int): the roll-off field of the BBHEADER (0: 0.35, 1: 0.25, 2: 0.20).
///
/// Returns:
///     numpy.ndarray: the ``k_bch`` bits of the BBFRAME (uint8).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (data, k_bch, *, rolloff = 0))]
fn dvbs2_bbframe<'py>(py: Python<'py>, data: &Bound<'py, PyAny>, k_bch: usize, rolloff: u8) -> PyResult<BArr<'py>> {
    Ok(dvbs2::bbframe(&bvec(data)?, k_bch, rolloff).py()?.into_pyarray(py))
}

/// The data bits of a received BBFRAME and whether the CRC-8 of its BBHEADER holds.
///
/// Args:
///     bbframe (array_like): the bits of a BBFRAME.
///
/// Returns:
///     tuple[numpy.ndarray, bool]: the data bits (uint8, derandomised) and the CRC check.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn dvbs2_bbframe_data<'py>(py: Python<'py>, bbframe: &Bound<'py, PyAny>) -> PyResult<(BArr<'py>, bool)> {
    let (d, ok) = dvbs2::bbframe_data(&bvec(bbframe)?).py()?;
    Ok((d.into_pyarray(py), ok))
}

/// The MODCOD field, the short-frame flag and the pilot flag of a received PLHEADER (90
/// symbols), by maximum likelihood over the 128 PLS codewords.
///
/// Args:
///     rx (array_like): the 90 received symbols of the PLHEADER.
///
/// Returns:
///     tuple[int, bool, bool]: the MODCOD, the short-frame flag and the pilot flag.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn dvbs2_decode_plheader(rx: &Bound<'_, PyAny>) -> PyResult<(u8, bool, bool)> {
    dvbs2::decode_plheader(&cvec(rx)?).py()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDvbs2>()?;
    m.add_function(wrap_pyfunction!(dvbs2_bbframe, m)?)?;
    m.add_function(wrap_pyfunction!(dvbs2_bbframe_data, m)?)?;
    m.add_function(wrap_pyfunction!(dvbs2_decode_plheader, m)?)?;
    Ok(())
}
