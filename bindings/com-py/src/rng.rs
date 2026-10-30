//! The random generator (the unit conversions are in [`crate::math`]).

use crate::{CArr, FArr, IArr, IntoPy, carr, count, farr};
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

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRng>()?;
    Ok(())
}
