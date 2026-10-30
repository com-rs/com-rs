//! 5G NR: the numerology, the synchronisation signal block, the cell search and the DMRS
//! of the PDSCH.

use crate::{CArr, FArr, IntoPy, carr, farr, mat};
use comrs::waveform::ofdm::nr::{self, Numerology, PdschDmrs, SSB_SUBCARRIERS, SSB_SYMBOLS, SsbElement};
use num_complex::Complex64 as C64;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The numerology `mu` of NR (the subcarrier spacing `2^mu · 15` kHz; `mu = 0` is also
/// the numerology of LTE).
///
/// Args:
///     mu (int): 0 to 4.
///     extended_cp (bool): the extended cyclic prefix (`mu = 2`, or LTE with `mu = 0`).
///     n_fft (int): the FFT size of the sampling rate and the prefixes (a multiple of
///         128).
///
/// Returns:
///     dict: ``subcarrier_spacing`` (Hz), ``slots_per_subframe``, ``slots_per_frame``,
///     ``symbols_per_slot``, ``sample_rate`` (Hz), ``cp_lengths`` (samples, the symbols
///     of one subframe).
///
/// Examples:
///     >>> import comrs
///     >>> info = comrs.nr_numerology(0, n_fft=2048)   # LTE: 30.72 MS/s
///     >>> info["sample_rate"], info["cp_lengths"][:2].tolist()
///     (30720000.0, [160, 144])
#[pyfunction]
#[pyo3(signature = (mu, *, extended_cp = false, n_fft = 2048))]
fn nr_numerology(py: Python<'_>, mu: u32, extended_cp: bool, n_fft: usize) -> PyResult<Bound<'_, PyDict>> {
    let n = Numerology::new(mu, extended_cp).py()?;
    let d = PyDict::new(py);
    d.set_item("subcarrier_spacing", n.subcarrier_spacing())?;
    d.set_item("slots_per_subframe", n.slots_per_subframe())?;
    d.set_item("slots_per_frame", n.slots_per_frame())?;
    d.set_item("symbols_per_slot", n.symbols_per_slot())?;
    d.set_item("sample_rate", n.sample_rate(n_fft))?;
    d.set_item("cp_lengths", crate::iarr(py, n.cp_lengths(n_fft).py()?))?;
    Ok(d)
}

/// The columns of a `(K, L)` grid (subcarriers × symbols), symbol-major.
fn grid_in(grid: &Bound<'_, PyAny>) -> PyResult<(Vec<C64>, usize)> {
    let a = crate::carray(grid)?;
    if a.ndim() != 2 {
        return Err(PyValueError::new_err("expected a 2-D grid (subcarriers × symbols)"));
    }
    let (k, l) = (a.shape()[0], a.shape()[1]);
    Ok(((0..l).flat_map(|j| (0..k).map(move |i| (i, j))).map(|(i, j)| a[[i, j]]).collect(), k))
}

/// A symbol-major grid as a `(K, L)` array.
fn grid_out<'py>(py: Python<'py>, g: &[C64], k: usize) -> PyResult<Bound<'py, PyArray2<C64>>> {
    let l = g.len() / k;
    mat(py, (0..k).map(|i| (0..l).map(|j| g[j * k + i]).collect()).collect())
}

/// OFDM-modulate a grid of NR: the subcarrier `k` at the frequency `(k - K/2) Δf`, the
/// cyclic prefixes of the numerology (the first symbol of every half subframe longer),
/// the IFFT scaled by `1/√K` (unit mean power for unit-energy symbols).
///
/// Args:
///     grid (array_like): the grid, shape `(K, L)` (subcarriers × OFDM symbols), `K`
///         even (`12 N_RB`).
///     mu (int): the numerology.
///     n_fft (int): the FFT size (a multiple of 128, at least `K`).
///     extended_cp (bool): the extended cyclic prefix.
///     first_symbol (int): the index in the subframe of the first symbol of the grid.
///
/// Returns:
///     numpy.ndarray: the samples at `n_fft · Δf` (complex128).
#[pyfunction]
#[pyo3(signature = (grid, mu, n_fft, *, extended_cp = false, first_symbol = 0))]
fn nr_ofdm_modulate<'py>(
    py: Python<'py>,
    grid: &Bound<'py, PyAny>,
    mu: u32,
    n_fft: usize,
    extended_cp: bool,
    first_symbol: usize,
) -> PyResult<CArr<'py>> {
    let (g, k) = grid_in(grid)?;
    let n = Numerology::new(mu, extended_cp).py()?;
    let carrier = nr::Carrier { numerology: n, n_subcarriers: k, n_fft, first_symbol };
    Ok(carr(py, carrier.modulate(&g).py()?))
}

/// The NR primary synchronisation signal of `N_ID2` (TS 38.211, Sec. 7.4.2.2): the BPSK
/// m-sequence of length 127.
///
/// Args:
///     n_id2 (int): 0, 1 or 2.
///
/// Returns:
///     numpy.ndarray: 127 values ±1 (float64).
#[pyfunction]
fn nr_pss(py: Python<'_>, n_id2: u32) -> PyResult<FArr<'_>> {
    Ok(farr(py, nr::pss(n_id2).py()?))
}

/// The NR secondary synchronisation signal of `(N_ID1, N_ID2)` (TS 38.211, Sec.
/// 7.4.2.3): the BPSK Gold sequence of length 127.
///
/// Args:
///     n_id1 (int): 0 to 335.
///     n_id2 (int): 0, 1 or 2.
///
/// Returns:
///     numpy.ndarray: 127 values ±1 (float64).
#[pyfunction]
fn nr_sss(py: Python<'_>, n_id1: u32, n_id2: u32) -> PyResult<FArr<'_>> {
    Ok(farr(py, nr::sss(n_id1, n_id2).py()?))
}

/// The 144 QPSK values of the PBCH DMRS (TS 38.211, Sec. 7.4.1.4).
///
/// Args:
///     n_cell_id (int): the cell identity, 0 to 1007.
///     i_ssb_bar (int): the SSB index `ī_SSB`, 0 to 7.
///
/// Returns:
///     numpy.ndarray: the values (complex128), in the order of their mapping.
#[pyfunction]
#[pyo3(signature = (n_cell_id, i_ssb_bar = 0))]
fn nr_pbch_dmrs(py: Python<'_>, n_cell_id: u32, i_ssb_bar: u32) -> PyResult<CArr<'_>> {
    Ok(carr(py, nr::pbch_dmrs(n_cell_id, i_ssb_bar).py()?))
}

/// The linear indices `l · 240 + k` of the resource elements of one kind in the SSB,
/// frequency first (the order of the mapping; the indices of MATLAB minus one).
///
/// Args:
///     n_cell_id (int): the cell identity (it shifts the PBCH DMRS).
///     kind (str): ``"pss"``, ``"sss"``, ``"pbch"`` or ``"pbch_dmrs"``.
///
/// Returns:
///     numpy.ndarray: the indices (int64).
#[pyfunction]
fn nr_ssb_indices<'py>(py: Python<'py>, n_cell_id: u32, kind: &str) -> PyResult<crate::IArr<'py>> {
    let k = match kind {
        "pss" => SsbElement::Pss,
        "sss" => SsbElement::Sss,
        "pbch" => SsbElement::Pbch,
        "pbch_dmrs" => SsbElement::PbchDmrs,
        other => return Err(PyValueError::new_err(format!("unknown kind {other:?}: pss, sss, pbch or pbch_dmrs"))),
    };
    Ok(crate::iarr(py, nr::ssb_indices(n_cell_id, k).into_iter().map(|(k, l)| l * SSB_SUBCARRIERS + k).collect()))
}

/// The SSB of a cell: the PSS and the SSS of unit amplitude, the PBCH DMRS and the 432
/// PBCH symbols.
///
/// Args:
///     n_cell_id (int): the cell identity, 0 to 1007.
///     i_ssb_bar (int): the SSB index `ī_SSB`, 0 to 7.
///     pbch (array_like | None): the 432 PBCH symbols (zeros when None).
///
/// Returns:
///     numpy.ndarray: the grid, shape `(240, 4)` (complex128).
#[pyfunction]
#[pyo3(signature = (n_cell_id, i_ssb_bar = 0, pbch = None))]
fn nr_ssb_grid<'py>(
    py: Python<'py>,
    n_cell_id: u32,
    i_ssb_bar: u32,
    pbch: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<C64>>> {
    let d = pbch.map(crate::cvec).transpose()?;
    let g = nr::ssb_grid(n_cell_id, i_ssb_bar, d.as_deref()).py()?;
    grid_out(py, &g, SSB_SUBCARRIERS)
}

/// The cell identity of a received SSB, by the PSS then the SSS (coherently, with the flat
/// channel estimated on the PSS).
///
/// Args:
///     grid (array_like): the received SSB, shape `(240, 4)`.
///
/// Returns:
///     tuple[int, float, float]: the cell identity and the normalised correlations of
///     the PSS and of the SSS (1 without noise).
#[pyfunction]
fn nr_detect_cell_id(grid: &Bound<'_, PyAny>) -> PyResult<(u32, f64, f64)> {
    let (g, k) = grid_in(grid)?;
    if k != SSB_SUBCARRIERS || g.len() != SSB_SUBCARRIERS * SSB_SYMBOLS {
        return Err(PyValueError::new_err("expected an SSB grid of shape (240, 4)"));
    }
    nr::detect_cell_id(&g).py()
}

/// The DMRS of a PDSCH of mapping type A over a slot of 14 symbols, single-symbol, port
/// 1000 (TS 38.211, Sec. 7.4.1.1).
///
/// Args:
///     n_rb (int): the resource blocks of the grid (from common resource block 0).
///     config_type (int): 1 (one subcarrier in two) or 2 (two in six).
///     type_a_position (int): the first DMRS symbol, 2 or 3.
///     additional_position (int): 0 to 3: the symbols `{l0}`, `{l0, 11}`,
///         `{l0, 7, 11}`, `{l0, 5, 8, 11}`.
///     n_id (int): the scrambling identity.
///     slot (int): the slot in the frame.
///
/// Returns:
///     dict: ``symbols`` (the DMRS symbols), ``subcarriers``, ``values`` (shape
///     `(len(subcarriers), len(symbols))`, complex128) and ``indices`` (the linear
///     indices `l · 12 n_rb + k` of the grid, frequency first).
#[pyfunction]
#[pyo3(signature = (n_rb, *, config_type = 1, type_a_position = 2, additional_position = 0, n_id = 0, slot = 0))]
fn nr_pdsch_dmrs(
    py: Python<'_>,
    n_rb: usize,
    config_type: u8,
    type_a_position: usize,
    additional_position: usize,
    n_id: u32,
    slot: u32,
) -> PyResult<Bound<'_, PyDict>> {
    let d = PdschDmrs { config_type, type_a_position, additional_position, n_id, slot };
    let symbols = d.symbols().py()?;
    let ks = d.subcarriers(n_rb);
    let k_grid = 12 * n_rb;
    let values: Vec<Vec<C64>> = symbols.iter().map(|&l| d.values(n_rb, l)).collect();
    let out = PyDict::new(py);
    out.set_item("symbols", crate::iarr(py, symbols.clone()))?;
    out.set_item("subcarriers", crate::iarr(py, ks.clone()))?;
    out.set_item("values", mat(py, (0..ks.len()).map(|i| values.iter().map(|v| v[i]).collect()).collect())?)?;
    let idx: Vec<usize> = symbols.iter().flat_map(|&l| ks.iter().map(move |&k| l * k_grid + k)).collect();
    out.set_item("indices", crate::iarr(py, idx))?;
    Ok(out)
}

/// Register the NR functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nr_numerology, m)?)?;
    m.add_function(wrap_pyfunction!(nr_ofdm_modulate, m)?)?;
    m.add_function(wrap_pyfunction!(nr_pss, m)?)?;
    m.add_function(wrap_pyfunction!(nr_sss, m)?)?;
    m.add_function(wrap_pyfunction!(nr_pbch_dmrs, m)?)?;
    m.add_function(wrap_pyfunction!(nr_ssb_indices, m)?)?;
    m.add_function(wrap_pyfunction!(nr_ssb_grid, m)?)?;
    m.add_function(wrap_pyfunction!(nr_detect_cell_id, m)?)?;
    m.add_function(wrap_pyfunction!(nr_pdsch_dmrs, m)?)?;
    Ok(())
}
