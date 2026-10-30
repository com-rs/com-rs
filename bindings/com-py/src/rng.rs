//! The random generator and the math helpers.

use crate::{CArr, FArr, IArr, IntoPy, carr, count, farr, map_real};
use comrs::core::rng::{Rng, stream};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A seeded random generator (xoshiro256** seeded through SplitMix64).
///
/// The same generator as the Rust crate: a seed gives the same draws in Rust and in
/// Python. Every function of `comrs` that draws takes one as its `rng` argument and
/// advances it.
///
/// Args:
///     seed (int): the seed (any 64-bit unsigned integer).
///
/// Examples:
///     >>> import comrs
///     >>> rng = comrs.Rng(1)
///     >>> rng.uniform(3).shape
///     (3,)
///     >>> comrs.Rng(7).next_u64() == comrs.Rng(7).next_u64()
///     True
#[pyclass(name = "Rng", module = "comrs")]
pub struct PyRng {
    /// The Rust generator.
    pub inner: Rng,
}

#[pymethods]
impl PyRng {
    #[new]
    fn new(seed: u64) -> Self {
        PyRng { inner: Rng::new(seed) }
    }

    /// The state of the generator (pickling: a copy continues its draws).
    fn __getstate__(&self) -> [u64; 4] {
        self.inner.state()
    }

    fn __setstate__(&mut self, state: [u64; 4]) -> PyResult<()> {
        self.inner = Rng::from_state(state).py()?;
        Ok(())
    }

    fn __getnewargs__(&self) -> (u64,) {
        (0,)
    }

    fn __copy__(&self) -> Self {
        PyRng { inner: self.inner.clone() }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        PyRng { inner: self.inner.clone() }
    }

    fn __repr__(&self) -> String {
        let s = self.inner.state();
        format!("Rng(state=[{:#018x}, {:#018x}, {:#018x}, {:#018x}])", s[0], s[1], s[2], s[3])
    }

    /// The generator of trial `trial` of condition `condition` of experiment `name`
    /// derived from a base seed (the streams of the Monte Carlo runner).
    ///
    /// Args:
    ///     seed (int): base seed.
    ///     name (str): experiment name (hashed with FNV-1a).
    ///     condition (int): condition index.
    ///     trial (int): trial index.
    ///
    /// Returns:
    ///     Rng: an independent generator.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> a = comrs.Rng.stream(1, "exp", 0, 3).uniform(2)
    ///     >>> bool((a == comrs.Rng.stream(1, "exp", 0, 3).uniform(2)).all())
    ///     True
    #[staticmethod]
    fn stream(seed: u64, name: &str, condition: u64, trial: u64) -> Self {
        PyRng { inner: stream(seed, name, condition, trial) }
    }

    /// The next raw 64-bit draw.
    ///
    /// Returns:
    ///     int: an unsigned 64-bit integer.
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Uniform draws in `[0, 1)`.
    ///
    /// Args:
    ///     n (int): number of draws.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` values (float64).
    ///
    /// Raises:
    ///     ValueError: for a negative count.
    fn uniform<'py>(&mut self, py: Python<'py>, n: i64) -> PyResult<FArr<'py>> {
        let n = count(n, "n")?;
        Ok(farr(py, (0..n).map(|_| self.inner.uniform()).collect()))
    }

    /// Standard normal draws.
    ///
    /// Args:
    ///     n (int): number of draws.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` values (float64).
    ///
    /// Raises:
    ///     ValueError: for a negative count.
    fn gaussian<'py>(&mut self, py: Python<'py>, n: i64) -> PyResult<FArr<'py>> {
        let n = count(n, "n")?;
        Ok(farr(py, (0..n).map(|_| self.inner.gaussian()).collect()))
    }

    /// Circular complex Gaussian draws of variance `noise_var` (`E|w|² = noise_var`).
    ///
    /// Args:
    ///     n (int): number of draws.
    ///     noise_var (float): the variance `E|w|²`.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` values (complex128).
    ///
    /// Raises:
    ///     ValueError: for a negative count.
    #[pyo3(signature = (n, noise_var = 1.0))]
    fn complex_noise<'py>(&mut self, py: Python<'py>, n: i64, noise_var: f64) -> PyResult<CArr<'py>> {
        let n = count(n, "n")?;
        Ok(carr(py, self.inner.complex_noise(n, noise_var)))
    }

    /// Uniform integers in `[0, high)`.
    ///
    /// Args:
    ///     high (int): the exclusive upper bound (positive).
    ///     n (int): number of draws.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` values (int64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn integers<'py>(&mut self, py: Python<'py>, high: i64, n: i64) -> PyResult<IArr<'py>> {
        let (high, n) = (count(high, "high")?, count(n, "n")?);
        if high == 0 {
            return Err(PyValueError::new_err("high must be positive"));
        }
        Ok((0..n).map(|_| self.inner.below(high) as i64).collect::<Vec<_>>().into_pyarray(py))
    }
}

/// Convert decibels to a linear ratio, `10^{x/10}`.
///
/// Args:
///     x_db (float | array_like): the ratio (dB).
///
/// Returns:
///     float | numpy.ndarray: the linear ratio.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.db2lin(20.0), comrs.db2lin([0.0, 10.0]).tolist()
///     (100.0, [1.0, 10.0])
#[pyfunction]
fn db2lin(x_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x_db, |v| Ok(comrs::math::db2lin(v)))
}

/// Convert a linear ratio to decibels, `10 log10 x`.
///
/// Args:
///     x (float | array_like): the linear ratio.
///
/// Returns:
///     float | numpy.ndarray: the ratio (dB).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.lin2db(100.0)
///     20.0
#[pyfunction]
fn lin2db(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x, |v| Ok(comrs::math::lin2db(v)))
}

/// Convert a power in dBm to watts, `10^{(P - 30)/10}`.
///
/// Args:
///     p_dbm (float | array_like): the power (dBm).
///
/// Returns:
///     float | numpy.ndarray: the power (W).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.dbm2w(30.0)
///     1.0
#[pyfunction]
fn dbm2w(p_dbm: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(p_dbm, |v| Ok(comrs::math::dbm2w(v)))
}

/// Convert a power in watts to dBm, `10 log10 P + 30`.
///
/// Args:
///     p_w (float | array_like): the power (W).
///
/// Returns:
///     float | numpy.ndarray: the power (dBm).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.w2dbm(1e-3)
///     0.0
#[pyfunction]
fn w2dbm(p_w: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(p_w, |v| Ok(comrs::math::w2dbm(v)))
}

/// The Gaussian tail function `Q(x) = erfc(x/√2)/2`.
///
/// Args:
///     x (float | array_like): the argument.
///
/// Returns:
///     float | numpy.ndarray: `Q(x)`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.qfunc(0.0), 6), round(comrs.qfunc(3.0), 6)
///     (0.5, 0.00135)
#[pyfunction]
fn qfunc(x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(x, |v| Ok(comrs::math::qfunc(v)))
}

/// The Q factor (dB) of a bit error rate, `20 log10(√2 erfc⁻¹(2 BER))`.
///
/// Args:
///     ber (float | array_like): the bit error rate, in `(0, 0.5)`.
///
/// Returns:
///     float | numpy.ndarray: the Q factor (dB).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.ber_to_q_db(1e-3), 3)
///     9.8
#[pyfunction]
fn ber_to_q_db(ber: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ber, |v| Ok(comrs::metrics::ber_to_q_db(v)))
}

/// Register the generator and the math helpers.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRng>()?;
    m.add_function(wrap_pyfunction!(db2lin, m)?)?;
    m.add_function(wrap_pyfunction!(lin2db, m)?)?;
    m.add_function(wrap_pyfunction!(dbm2w, m)?)?;
    m.add_function(wrap_pyfunction!(w2dbm, m)?)?;
    m.add_function(wrap_pyfunction!(qfunc, m)?)?;
    m.add_function(wrap_pyfunction!(ber_to_q_db, m)?)?;
    Ok(())
}
