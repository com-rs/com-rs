//! MIMO: channels, detection, space-time codes, precoding and capacity.
//!
//! Channel matrices are `(n_rx, n_tx)` arrays, or `(n, n_rx, n_tx)` stacks with one
//! matrix per received vector; received vectors are `(n, n_rx)` arrays (one row per
//! channel use).

use crate::comm::PyConstellation;
use crate::rng::PyRng;
use crate::{CArr, FArr, IntoPy, carray, count, farr};
use comrs::channel::fading::Mimo;
use comrs::channel::spatial::{Kronecker, exponential_correlation as exp_corr};
use comrs::mimo::capacity;
use comrs::mimo::detection::{Linear, MimoDetector};
use comrs::mimo::message_passing::{self as mp, AmpVariance, Domain, OampLinear, Prior};
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

fn detector(
    method: &str,
    c: &comrs::mapping::constellation::Constellation,
    k: usize,
    iterations: usize,
) -> PyResult<Detector> {
    Ok(DetectorKind::from_name(method, k).py()?.with_iterations(iterations).py()?.detector(c))
}

/// The rows of a per-vector result, stored flat (`width` values per vector).
struct Rows<T> {
    data: Vec<T>,
    width: usize,
}

impl<T> Rows<T> {
    fn map<U>(self, f: impl FnMut(T) -> U) -> Rows<U> {
        Rows { data: self.data.into_iter().map(f).collect(), width: self.width }
    }
}

/// Run `f` on every received vector with its channel, the rows of the result in one
/// buffer (a vector is passed without a copy when its row is contiguous).
fn per_vector<T: Send>(
    h: &Channels,
    y: &Array2<C64>,
    f: impl Fn(&ArrayView2<'_, C64>, &[C64]) -> comrs::Result<Vec<T>> + Sync,
) -> comrs::Result<Rows<T>> {
    if let Channels::Many(hs) = h {
        if hs.shape()[0] != y.nrows() {
            return Err(comrs::Error::Shape(format!(
                "{} channel matrices for {} received vectors",
                hs.shape()[0],
                y.nrows()
            )));
        }
    }
    let mut out = Rows { data: Vec::new(), width: 0 };
    for i in 0..y.nrows() {
        let row = y.row(i);
        let r = match row.as_slice() {
            Some(v) => f(&h.get(i), v)?,
            None => f(&h.get(i), &row.to_vec())?,
        };
        if i == 0 {
            out.width = r.len();
            out.data.reserve(r.len() * y.nrows());
        } else {
            comrs::core::same_len(r.len(), out.width, "values per vector")?;
        }
        out.data.extend(r);
    }
    Ok(out)
}

fn rows_to_array<T: numpy::Element>(py: Python<'_>, rows: Rows<T>, single: bool) -> PyResult<Py<PyAny>> {
    if single {
        return Ok(rows.data.into_pyarray(py).into_any().unbind());
    }
    let n = rows.data.len().checked_div(rows.width).unwrap_or(0);
    let a = Array2::from_shape_vec((n, rows.width), rows.data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.into_pyarray(py).into_any().unbind())
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
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.mimo_channel(4, 2, comrs.Rng(1), n=10).shape
///     (10, 4, 2)
#[pyfunction]
#[pyo3(signature = (n_rx, n_tx, rng, *, n=None, k_factor=0.0, rx_correlation=C64::new(0.0, 0.0), tx_correlation=C64::new(0.0, 0.0)))]
#[allow(clippy::too_many_arguments, reason = "the antennas and the options of the channel, by keyword")]
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
/// Raises:
///     ValueError: for a negative count.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.exponential_correlation(3, 0.5).real.tolist()
///     [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]]
#[pyfunction]
fn exponential_correlation<'py>(py: Python<'py>, n: i64, rho: C64) -> PyResult<Bound<'py, PyArray2<C64>>> {
    Ok(exp_corr(count(n, "n")?, rho).into_pyarray(py))
}

/// Detect the transmitted symbols of a MIMO link `y = H x + w`.
///
/// Methods: `"zf"`, `"mmse"` (linear, unbiased), `"zf_sic"`, `"mmse_sic"` (ordered
/// successive interference cancellation, V-BLAST), `"ml"` (exhaustive search),
/// `"sphere"` (exact ML by the Schnorr–Euchner sphere decoder), `"kbest"` (K-best tree
/// search keeping `k` paths), `"amp"` (LAMA, the complex Bayesian AMP of Jeon et al.)
/// and `"oamp"` (OAMP-LMMSE of Ma & Ping), both for large systems.
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received vectors, `(n, n_rx)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna `E|w_i|²`.
///     method (str): the detector.
///     k (int): paths kept by `"kbest"` (8, as the ``mimo_rx`` block).
///     iterations (int): iterations of `"amp"` and `"oamp"`.
///
/// Returns:
///     numpy.ndarray: the labels of the decided points, `(n, n_tx)` (or `(n_tx,)`).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> c = comrs.Constellation("qam", 4).normalized()
///     >>> h = comrs.mimo_channel(4, 2, comrs.Rng(3))
///     >>> x = c.points[[1, 2]]
///     >>> comrs.mimo_detect(h, h @ x, c, 1e-3, "sphere").tolist()
///     [1, 2]
#[pyfunction]
#[pyo3(signature = (
    h, y, constellation, noise_var, method="mmse", *, k=DetectorKind::DEFAULT_PATHS,
    iterations=DetectorKind::DEFAULT_ITERATIONS
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn mimo_detect(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
    k: usize,
    iterations: usize,
) -> PyResult<Py<PyAny>> {
    let (hs, (ys, single)) = (channels(h)?, received(y)?);
    let d = detector(method, &constellation.inner, k, iterations)?;
    let rows = py.detach(|| per_vector(&hs, &ys, |h, y| d.detect(h, y, noise_var))).py()?;
    rows_to_array(py, rows.map(|v| v as i64), single)
}

/// Bit log-likelihood ratios of a MIMO link (positive for a 0, most significant bit
/// first, stream after stream): exact per-stream LLRs after the unbiased linear filter
/// for `"zf"` and `"mmse"`, max-log over all hypotheses for `"ml"`, over the final
/// list for `"kbest"`, exact LLRs of the decoupled channel for `"amp"` and `"oamp"`.
///
/// Args:
///     h (array_like): the channel, `(n_rx, n_tx)` for every vector or `(n, n_rx, n_tx)`.
///     y (array_like): the received vectors, `(n, n_rx)` (or one vector `(n_rx,)`).
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna `E|w_i|²`.
///     method (str): `"zf"`, `"mmse"`, `"ml"`, `"kbest"`, `"amp"` or `"oamp"` (the SIC
///         and sphere detectors have no soft output).
///     k (int): paths kept by `"kbest"`.
///     iterations (int): iterations of `"amp"` and `"oamp"`.
///
/// Returns:
///     numpy.ndarray: `(n, n_tx · log2 M)` (or one row), float64.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (
    h, y, constellation, noise_var, method="mmse", *, k=DetectorKind::DEFAULT_PATHS,
    iterations=DetectorKind::DEFAULT_ITERATIONS
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn mimo_llr(
    py: Python<'_>,
    h: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
    k: usize,
    iterations: usize,
) -> PyResult<Py<PyAny>> {
    let (hs, (ys, single)) = (channels(h)?, received(y)?);
    let d = detector(method, &constellation.inner, k, iterations)?;
    let rows = py.detach(|| per_vector(&hs, &ys, |h, y| d.llr(h, y, noise_var))).py()?;
    rows_to_array(py, rows, single)
}

fn prior(c: &PyConstellation, px: Option<&Bound<'_, PyAny>>) -> PyResult<Prior> {
    let px = px.map(crate::fvec).transpose()?;
    Prior::with_probs(&c.inner, px.as_deref()).py()
}

fn domain(real: bool) -> Domain {
    if real { Domain::Real } else { Domain::Complex }
}

fn steps<'py>(py: Python<'py>, se: &[mp::StateEvolutionStep]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("noise_var", farr(py, se.iter().map(|s| s.noise_var).collect()))?;
    d.set_item("mse", farr(py, se.iter().map(|s| s.mse).collect()))?;
    Ok(d)
}

/// The iterations of LAMA or OAMP on one received vector: the MMSE estimate of every
/// iteration, and the decoupled channel of the last one (Jeon et al. 2018; Ma & Ping
/// 2017).
///
/// Args:
///     h (array_like): the channel `(n_rx, n_tx)`.
///     y (array_like): the received vector `(n_rx,)`.
///     constellation (Constellation): the alphabet of every stream.
///     noise_var (float): noise variance per receive antenna `E|w_i|²`.
///     method (str): `"amp"` (LAMA) or `"oamp"`.
///     iterations (int): iterations.
///     linear (str): the linear estimator of OAMP, `"lmmse"`, `"pinv"` or `"mf"`.
///     variance (str): the variance of LAMA, `"postulated"` (Algorithm 2) or
///         `"residual"` (eq. 38, better in small systems).
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///     real (bool): a real model (real channel, symbols and noise).
///
/// Returns:
///     dict: ``estimates`` (`(iterations, n_tx)`, the MMSE estimate of every iteration),
///     ``z`` (`(n_tx,)`, the input of the last denoiser: the symbols in Gaussian noise),
///     ``noise_var`` (its variance) and ``noise_vars`` (the variance of every iteration).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> c = comrs.Constellation("qam", 4).normalized()
///     >>> rng = comrs.Rng(1)
///     >>> h = comrs.mimo_channel(64, 32, rng)
///     >>> x = c.points[rng.integers(4, 32)]
///     >>> out = comrs.message_passing(h, h @ x, c, 1e-4, "amp", 15)
///     >>> bool(np.allclose(out["estimates"][-1], x, atol=1e-3))
///     True
#[pyfunction]
#[pyo3(signature = (
    h, y, constellation, noise_var, method="amp", iterations=DetectorKind::DEFAULT_ITERATIONS, *,
    linear="lmmse", variance="postulated", px=None, real=false
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn message_passing<'py>(
    py: Python<'py>,
    h: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    method: &str,
    iterations: usize,
    linear: &str,
    variance: &str,
    px: Option<&Bound<'py, PyAny>>,
    real: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let Channels::One(h) = channels(h)? else {
        return Err(PyValueError::new_err("message_passing takes one channel matrix (n_rx, n_tx)"));
    };
    let y = crate::cvec(y)?;
    let prior = prior(constellation, px)?;
    let f = domain(real);
    let trace = match method.parse::<DetectorKind>().py()? {
        DetectorKind::Amp(_) => {
            let v: AmpVariance = variance.parse().py()?;
            py.detach(|| mp::lama(&h.view(), &y, noise_var, &prior, f, iterations, v)).py()?
        }
        DetectorKind::Oamp(_) => {
            let l: OampLinear = linear.parse().py()?;
            py.detach(|| mp::oamp(&h.view(), &y, noise_var, &prior, f, iterations, l)).py()?
        }
        _ => return Err(PyValueError::new_err(format!("unknown method {method:?}: amp or oamp"))),
    };
    let n = h.ncols();
    let flat: Vec<C64> = trace.estimates.iter().flatten().copied().collect();
    let est =
        Array2::from_shape_vec((trace.estimates.len(), n), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let d = PyDict::new(py);
    d.set_item("estimates", est.into_pyarray(py))?;
    d.set_item("z", trace.z.clone().into_pyarray(py))?;
    d.set_item("noise_var", trace.noise_var)?;
    d.set_item("noise_vars", farr(py, trace.noise_vars))?;
    Ok(d)
}

/// The state evolution of LAMA (Jeon et al. 2018, Corollary 2): the variance of the
/// decoupled channel and the predicted MSE of every iteration, in the large-system limit
/// with columns of unit norm.
///
/// Args:
///     constellation (Constellation): the alphabet of every stream.
///     beta (float): the system ratio `n_tx / n_rx`.
///     noise_var (float): noise variance per receive antenna, relative to columns of unit
///         norm (`β Es / SNR`).
///     iterations (int): iterations.
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///     real (bool): a real model.
///
/// Returns:
///     dict: ``noise_var`` and ``mse``, one value per iteration.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comrs
///     >>> c = comrs.Constellation("qam", 4).normalized()
///     >>> se = comrs.amp_state_evolution(c, 0.5, 0.05, 20)
///     >>> bool(se["mse"][-1] < se["mse"][0])
///     True
#[pyfunction]
#[pyo3(signature = (constellation, beta, noise_var, iterations, *, px=None, real=false))]
fn amp_state_evolution<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    beta: f64,
    noise_var: f64,
    iterations: usize,
    px: Option<&Bound<'py, PyAny>>,
    real: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p = prior(constellation, px)?;
    let se = py.detach(|| mp::amp_state_evolution(&p, beta, noise_var, iterations, domain(real)));
    steps(py, &se)
}

/// The recovery thresholds of LAMA (Jeon et al. 2018, Table II): below ``beta_min`` LAMA
/// is optimal at every SNR, below ``beta_max`` it recovers the noiseless signal;
/// ``n0_min`` and ``n0_max`` are the critical noise variances at those ratios.
///
/// Args:
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///     real (bool): a real model.
///
/// Returns:
///     dict: ``beta_min``, ``beta_max``, ``n0_min`` and ``n0_max``.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comrs
///     >>> t = comrs.amp_thresholds(comrs.Constellation("qam", 16).normalized())
///     >>> round(t["beta_min"], 3), round(t["beta_max"], 3)
///     (0.983, 1.363)
#[pyfunction]
#[pyo3(signature = (constellation, *, px=None, real=false))]
fn amp_thresholds<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    px: Option<&Bound<'py, PyAny>>,
    real: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p = prior(constellation, px)?;
    let t = py.detach(|| mp::amp_thresholds(&p, domain(real)));
    let d = PyDict::new(py);
    for (k, v) in [("beta_min", t.beta_min), ("beta_max", t.beta_max), ("n0_min", t.n0_min), ("n0_max", t.n0_max)] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// The empirical state evolution of OAMP with the optimal denoiser (Ma & Ping 2017,
/// eqs. 32 and 39) for the singular values of a channel.
///
/// Args:
///     constellation (Constellation): the alphabet of every stream.
///     singular_values (array_like): the singular values of the channel.
///     n_tx (int): the number of columns of the channel.
///     noise_var (float): noise variance per receive antenna.
///     iterations (int): iterations.
///     linear (str): the linear estimator, `"lmmse"`, `"pinv"` or `"mf"`.
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///     real (bool): a real model.
///
/// Returns:
///     dict: ``noise_var`` (`τ²` of every iteration) and ``mse``.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> c = comrs.Constellation("qam", 4).normalized()
///     >>> h = comrs.mimo_channel(128, 64, comrs.Rng(1)) / np.sqrt(128)
///     >>> lam = np.linalg.svd(h, compute_uv=False)
///     >>> se = comrs.oamp_state_evolution(c, lam, 64, 0.3, 10)
///     >>> bool(se["mse"][-1] < se["mse"][0])
///     True
#[pyfunction]
#[pyo3(signature = (
    constellation, singular_values, n_tx, noise_var, iterations, *, linear="lmmse", px=None, real=false
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn oamp_state_evolution<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    singular_values: &Bound<'py, PyAny>,
    n_tx: usize,
    noise_var: f64,
    iterations: usize,
    linear: &str,
    px: Option<&Bound<'py, PyAny>>,
    real: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p = prior(constellation, px)?;
    let sv = crate::fvec(singular_values)?;
    let l: OampLinear = linear.parse().py()?;
    let se = py.detach(|| mp::oamp_state_evolution(&p, &sv, n_tx, noise_var, iterations, l, domain(real)));
    steps(py, &se)
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
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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
    let lin = match method.parse::<DetectorKind>().py()? {
        DetectorKind::Zf => Linear::zf(constellation.inner.clone()),
        DetectorKind::Mmse => Linear::mmse(constellation.inner.clone()),
        _ => return Err(PyValueError::new_err(format!("method must be zf or mmse, got {method:?}"))),
    };
    let rows = py
        .detach(|| {
            per_vector(&hs, &ys, |h, y| lin.estimate(h, y, noise_var).map(|(z, v)| z.into_iter().zip(v).collect()))
        })
        .py()?;
    let (z, v): (Vec<C64>, Vec<f64>) = rows.data.into_iter().unzip();
    let width = rows.width;
    Ok((rows_to_array(py, Rows { data: z, width }, single)?, rows_to_array(py, Rows { data: v, width }, single)?))
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.water_filling_powers([1.0, 1.0], 2.0).tolist()
///     [1.0, 1.0]
#[pyfunction]
fn water_filling_powers<'py>(
    py: Python<'py>,
    gains: &Bound<'py, PyAny>,
    total_power: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    Ok(farr(py, comrs::mimo::precoding::water_filling_powers(&crate::fvec(gains)?, total_power).py()?))
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
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> round(comrs.mimo_capacity(np.eye(2), 10 * np.log10(10.0)), 6)
///     5.169925
#[pyfunction]
#[pyo3(signature = (h, snr_db, *, water_filling=false))]
fn mimo_capacity(py: Python<'_>, h: &Bound<'_, PyAny>, snr_db: f64, water_filling: bool) -> PyResult<Py<PyAny>> {
    let snr = comrs::math::db2lin(snr_db);
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

/// The ergodic capacity (bits per channel use) of the i.i.d. Rayleigh MIMO channel with
/// equal power on the transmit antennas, in closed form (Telatar 1999, Theorem 2): the
/// mean of `mimo_capacity` over the realisations.
///
/// Args:
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas.
///     snr_db (float | array_like): total transmit power over the noise variance per
///         receive antenna (dB).
///
/// Returns:
///     float | numpy.ndarray: the capacity at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.ergodic_capacity(1, 1, 10.0), 4)   # log2(e) e^0.1 E1(0.1)
///     2.9065
#[pyfunction]
fn ergodic_capacity(n_rx: usize, n_tx: usize, snr_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    crate::map_real(snr_db, |s| capacity::ergodic_rayleigh(n_rx, n_tx, comrs::math::db2lin(s)).py())
}

/// The probability that a Rayleigh channel drawn once cannot carry `rate` bits per
/// channel use with equal power, one antenna on one side (Telatar 1999, eq. (16) and
/// Example 7).
///
/// Args:
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas (`n_rx` or `n_tx` is 1).
///     snr_db (float): the SNR (dB).
///     rate (float | array_like): the rate (bits per channel use).
///
/// Returns:
///     float | numpy.ndarray: the outage probability at every rate.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn outage_probability(n_rx: usize, n_tx: usize, snr_db: f64, rate: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    crate::map_real(rate, |r| capacity::outage_rayleigh(n_rx, n_tx, comrs::math::db2lin(snr_db), r).py())
}

/// The ε-capacity (bits per channel use): the rate of outage probability `eps` over a
/// Rayleigh channel drawn once, one antenna on one side (Telatar 1999, eqs. (17), (18)).
///
/// Args:
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas (`n_rx` or `n_tx` is 1).
///     snr_db (float | array_like): the SNR (dB).
///     eps (float): the outage probability, in (0, 1).
///
/// Returns:
///     float | numpy.ndarray: the rate at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn outage_capacity(n_rx: usize, n_tx: usize, snr_db: &Bound<'_, PyAny>, eps: f64) -> PyResult<Py<PyAny>> {
    crate::map_real(snr_db, |s| capacity::outage_capacity_rayleigh(n_rx, n_tx, comrs::math::db2lin(s), eps).py())
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
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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
    m.add_function(wrap_pyfunction!(message_passing, m)?)?;
    m.add_function(wrap_pyfunction!(amp_state_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(amp_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(oamp_state_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(water_filling_powers, m)?)?;
    m.add_function(wrap_pyfunction!(mimo_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(ergodic_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(outage_probability, m)?)?;
    m.add_function(wrap_pyfunction!(outage_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(eigenmode, m)?)?;
    m.add_function(wrap_pyfunction!(zf_precoder, m)?)?;
    Ok(())
}
