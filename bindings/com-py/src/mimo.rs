//! MIMO: channels, detection, space-time codes, precoding and capacity.
//!
//! Channel matrices are `(n_rx, n_tx)` arrays, or `(n, n_rx, n_tx)` stacks with one
//! matrix per received vector; received vectors are `(n, n_rx)` arrays (one row per
//! channel use).

use crate::comm::PyConstellation;
use crate::rng::PyRng;
use crate::{CArr, FArr, IntoPy, carray, farr};
use comrs::channel::fading::Mimo;
use comrs::channel::spatial::{Kronecker, exponential_correlation as exp_corr};
use comrs::mimo::capacity;
use comrs::mimo::detection::{Linear, MimoDetector};
use comrs::mimo::precoding::{Eigenmode, LinearPrecoder};
use comrs::mimo::receiver::DetectorKind;
use comrs::mimo::stbc::Ostbc;
use ndarray::{Array2, Array3, ArrayView2, Axis};
use num_complex::Complex64 as C64;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type Detector = Box<dyn MimoDetector + Send + Sync>;

/// One channel matrix, or one per received vector.
enum Channels {
    One(Array2<C64>),
    Many(Array3<C64>),
}

impl Channels {
    fn get(&self, i: usize) -> ArrayView2<'_, C64> {
        match self {
            Channels::One(h) => h.view(),
            Channels::Many(h) => h.index_axis(Axis(0), i),
        }
    }
}

fn channels(h: &Bound<'_, PyAny>) -> PyResult<Channels> {
    let a = carray(h)?;
    match a.ndim() {
        2 => Ok(Channels::One(a.into_dimensionality().map_err(|e| PyValueError::new_err(e.to_string()))?)),
        3 => Ok(Channels::Many(a.into_dimensionality().map_err(|e| PyValueError::new_err(e.to_string()))?)),
        d => Err(PyValueError::new_err(format!(
            "the channel must be (n_rx, n_tx) or (n, n_rx, n_tx), got {d} dimensions"
        ))),
    }
}

/// The received vectors as `(n, n_rx)` (a 1-D array is one vector).
fn received(y: &Bound<'_, PyAny>) -> PyResult<(Array2<C64>, bool)> {
    let a = carray(y)?;
    match a.ndim() {
        1 => {
            let n = a.len();
            Ok((a.into_shape_with_order((1, n)).map_err(|e| PyValueError::new_err(e.to_string()))?, true))
        }
        2 => Ok((a.into_dimensionality().map_err(|e| PyValueError::new_err(e.to_string()))?, false)),
        d => Err(PyValueError::new_err(format!("the received vectors must be (n, n_rx), got {d} dimensions"))),
    }
}

fn detector(method: &str, c: &comrs::mapping::constellation::Constellation, k: usize) -> PyResult<Detector> {
    Ok(DetectorKind::from_name(method, k).py()?.detector(c))
}

/// Run `f` on every received vector with its channel; rows of the result.
fn per_vector<T: Send>(
    h: &Channels,
    y: &Array2<C64>,
    f: impl Fn(&ArrayView2<'_, C64>, &[C64]) -> comrs::Result<Vec<T>> + Sync,
) -> comrs::Result<Vec<Vec<T>>> {
    if let Channels::Many(hs) = h {
        if hs.shape()[0] != y.nrows() {
            return Err(comrs::Error::Shape(format!(
                "{} channel matrices for {} received vectors",
                hs.shape()[0],
                y.nrows()
            )));
        }
    }
    (0..y.nrows()).map(|i| f(&h.get(i), &y.row(i).to_vec())).collect()
}

fn rows_to_array<T: numpy::Element + Clone>(py: Python<'_>, rows: Vec<Vec<T>>, single: bool) -> PyResult<Py<PyAny>> {
    if single {
        let row = rows.into_iter().next().unwrap_or_default();
        return Ok(row.into_pyarray(py).into_any().unbind());
    }
    Ok(PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))?.into_any().unbind())
}

/// Flat MIMO channel matrices: Rayleigh or Rician fading, with exponential antenna
/// correlation (Kronecker model).
///
/// `H = √(K/(K+1)) H_LOS + √(1/(K+1)) L_r G L_tᵀ` with `G` of independent `CN(0, 1)`
/// entries, `L Lᴴ` the exponential correlation `[R]_ij = ρ^(j-i)`, and `H_LOS` all
/// ones; every entry has unit mean power.
///
/// Args:
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas.
///     rng (Rng): the generator.
///     n (int, optional): number of independent matrices (one per channel use).
///     k_factor (float): Rician factor `K` (linear, 0 for Rayleigh).
///     rx_correlation (complex): correlation `ρ` of adjacent receive antennas.
///     tx_correlation (complex): correlation `ρ` of adjacent transmit antennas.
///
/// Returns:
///     numpy.ndarray: `(n_rx, n_tx)`, or `(n, n_rx, n_tx)` with `n`.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.mimo_channel(4, 2, comrs.Rng(1), n=10).shape
///     (10, 4, 2)
#[pyfunction]
#[pyo3(signature = (n_rx, n_tx, rng, *, n=None, k_factor=0.0, rx_correlation=C64::new(0.0, 0.0), tx_correlation=C64::new(0.0, 0.0)))]
#[allow(clippy::too_many_arguments)]
fn mimo_channel(
    py: Python<'_>,
    n_rx: usize,
    n_tx: usize,
    mut rng: PyRefMut<'_, PyRng>,
    n: Option<usize>,
    k_factor: f64,
    rx_correlation: C64,
    tx_correlation: C64,
) -> PyResult<Py<PyAny>> {
    let correlation = if rx_correlation.norm() > 0.0 || tx_correlation.norm() > 0.0 {
        Some(Kronecker::new(exp_corr(n_rx, rx_correlation), exp_corr(n_tx, tx_correlation)).py()?)
    } else {
        None
    };
    let ch = Mimo { k_factor, correlation, ..Mimo::rayleigh(n_rx, n_tx) };
    match n {
        None => Ok(ch.draw(&mut rng.inner).py()?.into_pyarray(py).into_any().unbind()),
        Some(n) => {
            let mut out = Array3::<C64>::zeros((n, n_rx, n_tx));
            for i in 0..n {
                out.index_axis_mut(Axis(0), i).assign(&ch.draw(&mut rng.inner).py()?);
            }
            Ok(out.into_pyarray(py).into_any().unbind())
        }
    }
}

/// The exponential correlation matrix `[R]_ij = ρ^(j-i)` (Hermitian) of `n` antennas.
///
/// Args:
///     n (int): number of antennas.
///     rho (complex): correlation `ρ` of adjacent antennas, `|ρ| ≤ 1`.
///
/// Returns:
///     numpy.ndarray: the `(n, n)` matrix (complex128).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.exponential_correlation(3, 0.5).real.tolist()
///     [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]]
#[pyfunction]
fn exponential_correlation<'py>(py: Python<'py>, n: usize, rho: C64) -> Bound<'py, PyArray2<C64>> {
    exp_corr(n, rho).into_pyarray(py)
}

/// Detect the transmitted symbols of a MIMO link `y = H x + w`.
///
/// Methods: `"zf"`, `"mmse"` (linear, unbiased), `"zf_sic"`, `"mmse_sic"` (ordered
/// successive interference cancellation, V-BLAST), `"ml"` (exhaustive search),
/// `"sphere"` (exact ML by the Schnorr–Euchner sphere decoder), `"kbest"` (K-best tree
/// search keeping `k` paths).
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received vectors, `(n, n_rx)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna `E|w_i|²`.
///     method (str): the detector.
///     k (int): paths kept by `"kbest"` (8, as the ``mimo_rx`` block).
///
/// Returns:
///     numpy.ndarray: the labels of the decided points, `(n, n_tx)` (or `(n_tx,)`).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> c = comrs.Constellation("qam", 4).normalized()
///     >>> h = comrs.mimo_channel(4, 2, comrs.Rng(3))
///     >>> x = c.points[[1, 2]]
///     >>> comrs.mimo_detect(h, h @ x, c, 1e-3, "sphere").tolist()
///     [1, 2]
#[pyfunction]
#[pyo3(signature = (h, y, constellation, noise_var, method="mmse", *, k=DetectorKind::DEFAULT_PATHS))]
fn mimo_detect(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
    k: usize,
) -> PyResult<Py<PyAny>> {
    let (hs, (ys, single)) = (channels(h)?, received(y)?);
    let d = detector(method, &constellation.inner, k)?;
    let rows = py.detach(|| per_vector(&hs, &ys, |h, y| d.detect(h, y, noise_var))).py()?;
    let rows: Vec<Vec<i64>> = rows.into_iter().map(|r| r.into_iter().map(|v| v as i64).collect()).collect();
    rows_to_array(py, rows, single)
}

/// Bit log-likelihood ratios of a MIMO link (positive for a 0, most significant bit
/// first, stream after stream): exact per-stream LLRs after the unbiased linear filter
/// for `"zf"` and `"mmse"`, max-log over all hypotheses for `"ml"`, over the final
/// list for `"kbest"`.
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received vectors, `(n, n_rx)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna `E|w_i|²`.
///     method (str): `"zf"`, `"mmse"`, `"ml"` or `"kbest"` (the SIC and sphere
///         detectors have no soft output).
///     k (int): paths kept by `"kbest"`.
///
/// Returns:
///     numpy.ndarray: `(n, n_tx · log2 M)` (or one row), float64.
#[pyfunction]
#[pyo3(signature = (h, y, constellation, noise_var, method="mmse", *, k=DetectorKind::DEFAULT_PATHS))]
fn mimo_llr(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
    k: usize,
) -> PyResult<Py<PyAny>> {
    let (hs, (ys, single)) = (channels(h)?, received(y)?);
    let d = detector(method, &constellation.inner, k)?;
    let rows = py.detach(|| per_vector(&hs, &ys, |h, y| d.llr(h, y, noise_var))).py()?;
    rows_to_array(py, rows, single)
}

/// The unbiased estimates of a linear detector and their noise variances.
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received vectors, `(n, n_rx)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna `E|w_i|²`.
///     method (str): `"zf"` or `"mmse"`.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the estimates and their variances, both
///     `(n, n_tx)` (or `(n_tx,)`).
#[pyfunction]
#[pyo3(signature = (h, y, constellation, noise_var, method="mmse"))]
fn mimo_estimate(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (hs, (ys, single)) = (channels(h)?, received(y)?);
    let lin = match method {
        "zf" => Linear::zf(constellation.inner.clone()),
        "mmse" => Linear::mmse(constellation.inner.clone()),
        other => return Err(PyValueError::new_err(format!("method must be zf or mmse, got {other:?}"))),
    };
    let rows =
        py.detach(|| per_vector(&hs, &ys, |h, y| lin.estimate(h, y, noise_var).map(|(z, v)| vec![(z, v)]))).py()?;
    let (z, v): (Vec<Vec<C64>>, Vec<Vec<f64>>) = rows.into_iter().map(|mut r| r.remove(0)).unzip();
    Ok((rows_to_array(py, z, single)?, rows_to_array(py, v, single)?))
}

/// An orthogonal space-time block code (Alamouti, or the rate-3/4 codes for 3 and 4
/// antennas), with linear maximum-likelihood decoding.
///
/// A block sends `n_symbols` symbols over `n_slots` channel uses from `n_tx`
/// antennas, with a unit average transmitted energy per channel use. Decoding turns
/// the block into one scalar channel per symbol of noise variance
/// `σ²/(c² ‖H‖_F²)`, a diversity of `n_tx · n_rx`. Build it with `Ostbc.alamouti()` or
/// `Ostbc.rate_3_4(n_tx)`.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.Ostbc.alamouti()
///     >>> s = np.array([1 + 1j, -1 + 1j]) / np.sqrt(2)
///     >>> h = comrs.mimo_channel(1, 2, comrs.Rng(2))
///     >>> est, var = code.decode(h, h @ code.encode(s), 0.01)
///     >>> np.allclose(est, s)
///     True
#[pyclass(name = "Ostbc", module = "comrs")]
pub struct PyOstbc {
    inner: Ostbc,
}

#[pymethods]
impl PyOstbc {
    /// Alamouti's code: two antennas, rate one.
    ///
    /// Returns:
    ///     Ostbc: the code.
    #[staticmethod]
    fn alamouti() -> Self {
        PyOstbc { inner: Ostbc::alamouti() }
    }

    /// The rate-3/4 code for 3 or 4 antennas.
    ///
    /// Args:
    ///     n_tx (int): 3 or 4.
    ///
    /// Returns:
    ///     Ostbc: the code.
    #[staticmethod]
    fn rate_3_4(n_tx: usize) -> PyResult<Self> {
        Ok(PyOstbc { inner: Ostbc::rate_3_4(n_tx).py()? })
    }

    /// str: Name of the code.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// int: Transmit antennas.
    #[getter]
    fn n_tx(&self) -> usize {
        self.inner.n_tx()
    }

    /// int: Channel uses per block.
    #[getter]
    fn n_slots(&self) -> usize {
        self.inner.n_slots()
    }

    /// int: Symbols per block.
    #[getter]
    fn n_symbols(&self) -> usize {
        self.inner.n_symbols()
    }

    /// float: Symbols per channel use.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate()
    }

    /// The signal of the antennas.
    ///
    /// Args:
    ///     symbols (array_like): a multiple of `n_symbols` symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n_tx, n_slots · blocks)` (complex128).
    fn encode<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(self.inner.encode(&crate::cvec(symbols)?).py()?.into_pyarray(py))
    }

    /// Decode the received blocks (linear maximum likelihood).
    ///
    /// Args:
    ///     h (array_like): the channel, `(n_rx, n_tx)`, or `(blocks, n_rx, n_tx)`.
    ///     y (array_like): the received blocks, `(n_rx, n_slots · blocks)`.
    ///     noise_var (float): noise variance per receive antenna.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: the symbol estimates and their noise
    ///     variances.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        h: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        noise_var: f64,
    ) -> PyResult<(CArr<'py>, FArr<'py>)> {
        let ya: Array2<C64> = carray(y)?
            .into_dimensionality()
            .map_err(|_| PyValueError::new_err("y must be (n_rx, n_slots · blocks)"))?;
        let (est, var) = match channels(h)? {
            Channels::One(h) => py.detach(|| self.inner.decode(&h.view(), &ya.view(), noise_var)),
            Channels::Many(hs) => {
                let list: Vec<Array2<C64>> = hs.outer_iter().map(|m| m.to_owned()).collect();
                py.detach(|| self.inner.decode_blocks(&list, &ya.view(), noise_var))
            }
        }
        .py()?;
        Ok((est.into_pyarray(py), farr(py, var)))
    }

    fn __repr__(&self) -> String {
        format!("Ostbc({:?}, n_tx={}, rate={})", self.inner.name(), self.inner.n_tx(), self.inner.rate())
    }
}

/// Water-filling powers of parallel channels: `p_i = (μ - 1/g_i)⁺` with `Σ p_i` the
/// total power.
///
/// Args:
///     gains (array_like): the gains `g_i` (SNR per unit power).
///     total_power (float): the total power.
///
/// Returns:
///     numpy.ndarray: the power of every channel (float64).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.water_filling([1.0, 1.0], 2.0).tolist()
///     [1.0, 1.0]
#[pyfunction]
fn water_filling<'py>(
    py: Python<'py>,
    gains: &Bound<'py, PyAny>,
    total_power: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    Ok(farr(py, comrs::mimo::precoding::water_filling(&crate::fvec(gains)?, total_power).py()?))
}

/// Capacity of MIMO channel realisations (bits per channel use) at the SNR
/// `ρ = P/σ²`: `log2 det(I + ρ/n_tx H Hᴴ)` with equal power, or water-filling on the
/// eigenmodes.
///
/// Args:
///     h (array_like): `(n_rx, n_tx)` or `(n, n_rx, n_tx)`.
///     snr_db (float): total transmit power over the noise variance per receive antenna
///         (dB).
///     water_filling (bool): channel knowledge at the transmitter.
///
/// Returns:
///     float | numpy.ndarray: the capacity of every realisation.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> round(comrs.mimo_capacity(np.eye(2), 10 * np.log10(10.0)), 6)
///     5.169925
#[pyfunction]
#[pyo3(signature = (h, snr_db, *, water_filling=false))]
fn mimo_capacity(py: Python<'_>, h: &Bound<'_, PyAny>, snr_db: f64, water_filling: bool) -> PyResult<Py<PyAny>> {
    let snr = 10f64.powf(snr_db / 10.0);
    let f = |m: &ArrayView2<'_, C64>| {
        if water_filling { capacity::water_filling(m, snr) } else { capacity::equal_power(m, snr) }
    };
    match channels(h)? {
        Channels::One(m) => Ok(f(&m.view()).py()?.into_pyobject(py)?.into_any().unbind()),
        Channels::Many(hs) => {
            let c = py.detach(|| hs.outer_iter().map(|m| f(&m)).collect::<comrs::Result<Vec<f64>>>()).py()?;
            Ok(farr(py, c).into_any().unbind())
        }
    }
}

/// Eigenmode (SVD) transmission over `h`: precoder `V P^(1/2)`, receive filter `Uᴴ`,
/// gains and powers of the active modes (water-filling of `total_power` against
/// `noise_var`, or equal power).
///
/// Args:
///     h (array_like): one channel `(n_rx, n_tx)`.
///     total_power (float): the total transmit power.
///     noise_var (float): noise variance per receive antenna.
///     water_filling (bool): water-filling (else equal power on every mode).
///
/// Returns:
///     dict: ``precoder`` `(n_tx, streams)`, ``combiner`` `(streams, n_rx)`, ``gains``,
///     ``powers``.
#[pyfunction]
#[pyo3(signature = (h, total_power, noise_var, *, water_filling=true))]
fn eigenmode<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    total_power: f64,
    noise_var: f64,
    water_filling: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let Channels::One(m) = channels(h)? else {
        return Err(PyValueError::new_err("eigenmode takes one (n_rx, n_tx) channel"));
    };
    let e = Eigenmode::new(&m.view(), total_power, noise_var, water_filling).py()?;
    let d = PyDict::new(py);
    d.set_item("precoder", e.precoder.into_pyarray(py))?;
    d.set_item("combiner", e.combiner.into_pyarray(py))?;
    d.set_item("gains", farr(py, e.gains))?;
    d.set_item("powers", farr(py, e.powers))?;
    Ok(d)
}

/// The (regularised) zero-forcing precoder `Hᴴ(H Hᴴ + αI)⁻¹` of the users of the rows
/// of `h`, scaled to `total_power` (`α = 0` for zero forcing, `K σ²/P` for the
/// regularised version).
///
/// Args:
///     h (array_like): the channel `(n_users, n_tx)`.
///     total_power (float): the total transmit power.
///     alpha (float): the regularisation `α`.
///
/// Returns:
///     numpy.ndarray: the precoder `(n_tx, n_users)` (complex128).
#[pyfunction]
#[pyo3(signature = (h, *, total_power=1.0, alpha=0.0))]
fn zf_precoder<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    total_power: f64,
    alpha: f64,
) -> PyResult<Bound<'py, PyArray2<C64>>> {
    let Channels::One(m) = channels(h)? else {
        return Err(PyValueError::new_err("zf_precoder takes one (n_users, n_tx) channel"));
    };
    let p = if alpha == 0.0 {
        LinearPrecoder::zero_forcing(&m.view(), total_power)
    } else {
        LinearPrecoder::regularized(&m.view(), alpha, total_power)
    }
    .py()?;
    Ok(p.w.into_pyarray(py))
}

/// Register the MIMO bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOstbc>()?;
    m.add_function(wrap_pyfunction!(mimo_channel, m)?)?;
    m.add_function(wrap_pyfunction!(exponential_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_detect, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_llr, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(water_filling, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(eigenmode, m)?)?;
    m.add_function(wrap_pyfunction!(zf_precoder, m)?)?;
    Ok(())
}
