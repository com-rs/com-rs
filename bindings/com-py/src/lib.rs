//! Python bindings of `comrs` (the `comrs._comrs` extension module).
//!
//! The bindings follow the Rust API closely, with NumPy arrays in and out:
//!
//! - signals are 1-D `complex128` arrays (any array-like is converted, so real arrays
//!   and lists work too); bits are `uint8` arrays, symbol indices `int64` arrays;
//! - the random generators are explicit `PyRng` objects (`comrs.Rng(seed)`),
//!   passed to every function that draws, so that a simulation is reproducible from
//!   its seeds;
//! - the Rust errors become `ValueError` (invalid parameters and shapes, missing
//!   condition keys), `ArithmeticError` (numerical failures) and `OSError`;
//! - the long computations (split-step propagation, adaptive equalisation, blind phase
//!   search, filtering, declarative campaigns) release the GIL;
//! - `run_spec` runs a declarative campaign with the parallel Rust runner, and the
//!   Python Monte Carlo runner (`comrs.sim`) runs Python trials on the same runner
//!   ([`sim`]), with one thread;
//! - the closed forms take a scalar (and return a float) or an array-like (and return
//!   an array of the same shape), see [`map_real`].
//!
//! The submodules group the bindings by theme; they are flattened into one Python
//! module, re-exported by the `comrs` package.

use num_complex::Complex64 as C64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyArithmeticError, PyOSError, PyValueError};
use pyo3::prelude::*;

mod cdl;
mod chain;
mod comm;
mod devices;
mod differential;
mod dsp;
mod fec;
mod fec_codes;
mod fiber;
mod gn;
mod mimo;
mod ofdm;
mod ofdm_link;
mod rng;
mod sim;
mod spec;
mod systems;
mod waveform;

/// A 1-D complex array returned to Python.
pub(crate) type CArr<'py> = Bound<'py, PyArray1<C64>>;
/// A 1-D real array returned to Python.
pub(crate) type FArr<'py> = Bound<'py, PyArray1<f64>>;
/// A 1-D integer array returned to Python.
pub(crate) type IArr<'py> = Bound<'py, PyArray1<i64>>;
/// A 1-D array of bits returned to Python.
pub(crate) type BArr<'py> = Bound<'py, PyArray1<u8>>;
/// A 2-D complex array returned to Python.
pub(crate) type CMat<'py> = Bound<'py, PyArray2<C64>>;
/// A 2-D real array returned to Python.
pub(crate) type FMat<'py> = Bound<'py, PyArray2<f64>>;

/// Convert a `comrs` error into the matching Python exception.
pub(crate) fn err(e: comrs::Error) -> PyErr {
    match e {
        comrs::Error::Config(_) | comrs::Error::Shape(_) | comrs::Error::Condition(_) => {
            PyValueError::new_err(e.to_string())
        }
        comrs::Error::Numerical(_) => PyArithmeticError::new_err(e.to_string()),
        comrs::Error::Io(_) => PyOSError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// A constellation family from its name (`qam`, `psk`, `pam`, `ook`).
pub(crate) fn kind(name: &str) -> PyResult<comrs::mapping::constellation::ConstellationKind> {
    name.parse().py()
}

/// `Result<T, comrs::Error>` into `PyResult<T>`.
pub(crate) trait IntoPy<T> {
    /// Map the error with [`err`].
    fn py(self) -> PyResult<T>;
}

impl<T> IntoPy<T> for comrs::Result<T> {
    fn py(self) -> PyResult<T> {
        self.map_err(err)
    }
}

/// Call `numpy.ascontiguousarray(x, dtype)`.
pub(crate) fn as_numpy<'py>(x: &Bound<'py, PyAny>, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
    x.py().import("numpy")?.call_method1("ascontiguousarray", (x, dtype))
}

/// A 1-D array-like as complex samples.
pub(crate) fn cvec(x: &Bound<'_, PyAny>) -> PyResult<Vec<C64>> {
    let a = as_numpy(x, "complex128")?;
    let a = a.cast::<PyArray1<C64>>().map_err(|_| PyValueError::new_err("expected a 1-D array of complex samples"))?;
    Ok(a.readonly().as_array().to_vec())
}

/// A 1-D array-like as real values.
pub(crate) fn fvec(x: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let a = as_numpy(x, "float64")?;
    let a = a.cast::<PyArray1<f64>>().map_err(|_| PyValueError::new_err("expected a 1-D array of real values"))?;
    Ok(a.readonly().as_array().to_vec())
}

/// A 1-D array-like of bits (`0`/`1`).
pub(crate) fn bvec(x: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let a = as_numpy(x, "uint8")?;
    let a = a.cast::<PyArray1<u8>>().map_err(|_| PyValueError::new_err("expected a 1-D array of bits"))?;
    let v = a.readonly().as_array().to_vec();
    if v.iter().any(|&b| b > 1) {
        return Err(PyValueError::new_err("bits must be 0 or 1"));
    }
    Ok(v)
}

/// A 1-D array-like of non-negative integers.
pub(crate) fn uvec(x: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let a = as_numpy(x, "int64")?;
    let a = a.cast::<PyArray1<i64>>().map_err(|_| PyValueError::new_err("expected a 1-D array of integers"))?;
    a.readonly()
        .as_array()
        .iter()
        .map(|&v| usize::try_from(v).map_err(|_| PyValueError::new_err(format!("negative index {v}"))))
        .collect()
}

/// A 2-D array-like (rows = modes) as complex samples.
pub(crate) fn cmat(x: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<C64>>> {
    let a = as_numpy(x, "complex128")?;
    let a = a
        .cast::<PyArray2<C64>>()
        .map_err(|_| PyValueError::new_err("expected a 2-D array of complex samples (modes × samples)"))?;
    Ok(a.readonly().as_array().rows().into_iter().map(|r| r.to_vec()).collect())
}

/// A complex array of any number of dimensions.
pub(crate) fn carray(x: &Bound<'_, PyAny>) -> PyResult<ndarray::ArrayD<C64>> {
    let a = as_numpy(x, "complex128")?;
    let a = a.cast::<numpy::PyArrayDyn<C64>>().map_err(|_| PyValueError::new_err("expected a complex array"))?;
    Ok(a.readonly().as_array().to_owned())
}

/// The streams of a `(n,)` or `(n_streams, n)` complex array, and whether it was 1-D.
pub(crate) fn streams_in(x: &Bound<'_, PyAny>) -> PyResult<(Vec<Vec<C64>>, bool)> {
    let a = carray(x)?;
    match a.ndim() {
        1 => Ok((vec![a.iter().copied().collect()], true)),
        2 => Ok((a.outer_iter().map(|r| r.iter().copied().collect()).collect(), false)),
        _ => Err(PyValueError::new_err("x must be (n,) or (n_tx, n)")),
    }
}

/// The streams of a signal as a NumPy array: `(n,)` for one stream when the input was
/// 1-D (`one_d`, from [`streams_in`]), `(n_streams, n)` otherwise.
pub(crate) fn streams_out(py: Python<'_>, y: &comrs::core::signal::Signal, one_d: bool) -> Py<PyAny> {
    if y.n_streams() == 1 && one_d {
        carr(py, y.stream(0).to_vec()).into_any().unbind()
    } else {
        y.as_array().clone().into_pyarray(py).into_any().unbind()
    }
}

/// Complex samples as a NumPy array.
pub(crate) fn carr(py: Python<'_>, v: Vec<C64>) -> CArr<'_> {
    v.into_pyarray(py)
}

/// A vector of indices as a NumPy array of `int64`.
pub(crate) fn iarr(py: Python<'_>, v: Vec<usize>) -> IArr<'_> {
    v.into_iter().map(|i| i as i64).collect::<Vec<_>>().into_pyarray(py)
}

/// A Python object (a dict, a list, a scalar) as a JSON value, through `json.dumps`.
pub(crate) fn json(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let text: String = py.import("json")?.call_method1("dumps", (v,))?.extract()?;
    serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// A JSON value as a Python object, through `json.loads`.
pub(crate) fn from_json<'py>(py: Python<'py>, v: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?.call_method1("loads", (v.to_string(),))
}

/// The generator of an optional Python `Rng` argument.
pub(crate) fn rng_of<'a>(rng: &'a mut Option<PyRefMut<'_, rng::PyRng>>) -> Option<&'a mut comrs::core::rng::Rng> {
    rng.as_deref_mut().map(|g| &mut g.inner)
}

/// Real values as a NumPy array.
pub(crate) fn farr(py: Python<'_>, v: Vec<f64>) -> FArr<'_> {
    v.into_pyarray(py)
}

/// A real scalar or array-like mapped element by element by `f`: a Python float for a
/// scalar (a number or a 0-d array), an array of the same shape otherwise (the
/// closed forms of the package accept both).
pub(crate) fn map_real(x: &Bound<'_, PyAny>, f: impl Fn(f64) -> PyResult<f64>) -> PyResult<Py<PyAny>> {
    let py = x.py();
    if x.is_instance_of::<pyo3::types::PyFloat>() || x.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(f(x.extract::<f64>()?)?.into_pyobject(py)?.into_any().unbind());
    }
    let a = py.import("numpy")?.call_method1("asarray", (x, "float64"))?;
    let a = a.cast::<numpy::PyArrayDyn<f64>>().map_err(|_| PyValueError::new_err("expected real values"))?;
    let ro = a.readonly();
    let v = ro.as_array();
    if v.ndim() == 0 {
        let x0 = v.iter().next().copied().unwrap_or(f64::NAN);
        return Ok(f(x0)?.into_pyobject(py)?.into_any().unbind());
    }
    let out = v.iter().map(|&x| f(x)).collect::<PyResult<Vec<f64>>>()?;
    let out = ndarray::ArrayD::from_shape_vec(v.raw_dim(), out).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py).into_any().unbind())
}

/// A count given as a Python integer: a negative one is a `ValueError` (not the
/// `OverflowError` of an unsigned conversion).
pub(crate) fn count(n: i64, what: &str) -> PyResult<usize> {
    usize::try_from(n).map_err(|_| PyValueError::new_err(format!("{what} must be non-negative, got {n}")))
}

/// Taps `(delay in s, power in dB)` from an `(n, 2)` array-like (a list of pairs works).
pub(crate) fn taps_in(x: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    let a = as_numpy(x, "float64")?;
    let bad = || PyValueError::new_err("the taps are an (n, 2) array of (delay in s, power in dB)");
    let a = a.cast::<PyArray2<f64>>().map_err(|_| bad())?;
    let ro = a.readonly();
    let v = ro.as_array();
    if v.ncols() != 2 {
        return Err(bad());
    }
    Ok(v.rows().into_iter().map(|r| (r[0], r[1])).collect())
}

/// Taps `(delay in s, power in dB)` as an `(n, 2)` array.
pub(crate) fn taps_out(py: Python<'_>, taps: Vec<(f64, f64)>) -> PyResult<FMat<'_>> {
    mat(py, taps.into_iter().map(|(d, p)| vec![d, p]).collect())
}

/// Rows of equal length as a 2-D NumPy array.
pub(crate) fn mat<T: numpy::Element + Clone>(py: Python<'_>, rows: Vec<Vec<T>>) -> PyResult<Bound<'_, PyArray2<T>>> {
    PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Communication algorithms in Rust: sources, modulation, metrics, DSP, fibre and
/// device models (see the `comrs` package documentation).
#[pymodule]
fn _comrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    rng::register(m)?;
    comm::register(m)?;
    dsp::register(m)?;
    fiber::register(m)?;
    gn::register(m)?;
    devices::register(m)?;
    mimo::register(m)?;
    ofdm::register(m)?;
    ofdm_link::register(m)?;
    cdl::register(m)?;
    differential::register(m)?;
    fec::register(m)?;
    fec_codes::register(m)?;
    chain::register(m)?;
    spec::register(m)?;
    sim::register(m)?;
    systems::register(m)?;
    waveform::register(m)?;
    Ok(())
}
