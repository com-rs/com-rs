//! Power amplifiers: the memoryless models of Rapp and Saleh, the memory polynomial,
//! their Bussgang decomposition and the digital predistorter.

use crate::{CArr, IntoPy, carr, cvec};
use comrs::dsp::compensation::dpd::MemoryPolynomialDpd;
use comrs::dsp::nonlinear;
use comrs::hardware::amplifier::{MemoryPolynomial, Model, Rapp, Saleh};
use num_complex::Complex64 as C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Bussgang's decomposition `y = K x + d` of a memoryless amplifier driven by a circular
/// complex Gaussian signal: the complex gain and the power of the distortion.
///
/// The parameters of each model default to those of :func:`pa_rapp` and
/// :func:`pa_saleh` (Rapp with `p = 2`, Saleh's fit of 1981).
///
/// Args:
///     model (str): `"rapp"` (gain 1, `saturation`, `smoothness`) or
///         `"saleh"` (`alpha_a`, `beta_a`, `alpha_p`, `beta_p`).
///     power (float): the input power.
///     saturation (float): Rapp's saturation amplitude.
///     smoothness (float): Rapp's smoothness `p`.
///     alpha_a (float): Saleh's AM/AM gain.
///     beta_a (float): Saleh's AM/AM compression.
///     alpha_p (float): Saleh's AM/PM gain (rad).
///     beta_p (float): Saleh's AM/PM compression.
///
/// Returns:
///     tuple[complex, float]: `K` and `σ_d²`.
///
/// Raises:
///     ValueError: for an unknown model.
///
/// Examples:
///     The TWTA of Bouhadda et al. (2014, Table 3) at 4 dB of input back-off, `φ₀ = π/6`:
///
///     >>> import comrs, math
///     >>> k, d = comrs.amplifier_bussgang("saleh", 10 ** -0.4, alpha_a=1.0, beta_a=1.0,
///     ...                                 alpha_p=math.pi / 6, beta_p=1.0)
///     >>> round(k.real, 4), round(k.imag, 4)
///     (0.5917, 0.1106)
#[pyfunction]
#[pyo3(signature = (
    model,
    power,
    *,
    saturation = Rapp::default().saturation,
    smoothness = Rapp::default().smoothness,
    alpha_a = Saleh::default().alpha_a,
    beta_a = Saleh::default().beta_a,
    alpha_p = Saleh::default().alpha_p,
    beta_p = Saleh::default().beta_p,
))]
#[allow(clippy::too_many_arguments, reason = "the parameters of both amplifier models, by keyword")]
fn amplifier_bussgang(
    model: &str,
    power: f64,
    saturation: f64,
    smoothness: f64,
    alpha_a: f64,
    beta_a: f64,
    alpha_p: f64,
    beta_p: f64,
) -> PyResult<(C64, f64)> {
    let b = match model.parse::<Model>().py()? {
        Model::Rapp => nonlinear::bussgang_of(&Rapp { saturation, smoothness, ..Rapp::default() }, power),
        Model::Saleh => nonlinear::bussgang_of(&Saleh { alpha_a, beta_a, alpha_p, beta_p }, power),
        other => return Err(PyValueError::new_err(format!("no Bussgang decomposition for {other:?}"))),
    };
    Ok((b.gain, b.distortion))
}

/// Rapp's solid-state amplifier: `A(r) = g r / (1 + (g r / A_sat)^(2p))^(1/(2p))`, no
/// AM/PM (Rapp 1991).
///
/// Args:
///     x (array_like): the input envelope.
///     gain (float): the small-signal gain `g`.
///     saturation (float): the saturation amplitude `A_sat`.
///     smoothness (float): the smoothness `p`.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
#[pyfunction]
#[pyo3(signature = (x, *, gain=Rapp::default().gain, saturation=Rapp::default().saturation, smoothness=Rapp::default().smoothness))]
fn pa_rapp<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    gain: f64,
    saturation: f64,
    smoothness: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Rapp { gain, saturation, smoothness }.amplify(&cvec(x)?).py()?))
}

/// The static predistorter of Rapp's amplifier: the inverse AM/AM, up to `max_output`
/// times the saturation (the cascade is then an ideal envelope limiter).
///
/// Args:
///     x (array_like): the input envelope.
///     gain (float): the small-signal gain of the amplifier.
///     saturation (float): its saturation amplitude.
///     smoothness (float): its smoothness.
///     max_output (float): the largest output, relative to the saturation.
///
/// Returns:
///     numpy.ndarray: the predistorted input (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
#[pyfunction]
#[pyo3(signature = (
    x,
    *,
    gain = Rapp::default().gain,
    saturation = Rapp::default().saturation,
    smoothness = Rapp::default().smoothness,
    max_output = 0.99,
))]
fn rapp_predistort<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    gain: f64,
    saturation: f64,
    smoothness: f64,
    max_output: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Rapp { gain, saturation, smoothness }.predistort(&cvec(x)?, max_output).py()?))
}

/// Saleh's travelling-wave tube amplifier (AM/AM and AM/PM; Saleh 1981, default
/// coefficients of his fit).
///
/// Args:
///     x (array_like): the input envelope.
///     alpha_a (float): AM/AM coefficient `α_a`.
///     beta_a (float): AM/AM coefficient `β_a`.
///     alpha_p (float): AM/PM coefficient `α_φ`.
///     beta_p (float): AM/PM coefficient `β_φ`.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
#[pyfunction]
#[pyo3(signature = (
    x,
    *,
    alpha_a = Saleh::default().alpha_a,
    beta_a = Saleh::default().beta_a,
    alpha_p = Saleh::default().alpha_p,
    beta_p = Saleh::default().beta_p,
))]
fn pa_saleh<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    alpha_a: f64,
    beta_a: f64,
    alpha_p: f64,
    beta_p: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Saleh { alpha_a, beta_a, alpha_p, beta_p }.amplify(&cvec(x)?).py()?))
}

/// The memory-polynomial amplifier of Ding et al. (2004): orders 1, 3, 5, memory 3.
///
/// Args:
///     x (array_like): the input envelope.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
#[pyfunction]
fn pa_memory_polynomial<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
    Ok(carr(py, MemoryPolynomial::ding2004().amplify(&cvec(x)?)))
}

/// A memory-polynomial digital predistorter, trained by indirect learning (Eun & Powers
/// 1997; Ding et al. 2004; Morgan et al. 2006).
///
/// Args:
///     orders (list[int]): nonlinearity orders (odd: 1, 3, 5, …).
///     memory (int): memory depth `M` (samples).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> g = 1.0513 + 0.0904j                      # small-signal gain of the amplifier
///     >>> x = 0.2 * comrs.Rng(1).complex_noise(20_000, 1.0)
///     >>> dpd = comrs.MemoryPolynomialDpd([1, 3, 5], 3)
///     >>> dpd.train_indirect(x, comrs.pa_memory_polynomial, gain=g, iterations=2)
///     >>> nmse = lambda y: float(np.mean(np.abs(y / g - x) ** 2) / np.mean(np.abs(x) ** 2))
///     >>> nmse(comrs.pa_memory_polynomial(dpd.apply(x))) < nmse(comrs.pa_memory_polynomial(x)) / 5
///     True
///
/// The drive must stay where the amplifier is monotonic: beyond, the post-inverse that
/// indirect learning fits has no meaning and the iterations diverge.
#[pyclass(name = "MemoryPolynomialDpd", module = "comrs")]
pub struct PyDpd {
    inner: MemoryPolynomialDpd,
}

#[pymethods]
impl PyDpd {
    fn __repr__(&self) -> String {
        format!("MemoryPolynomialDpd(orders={:?}, memory={})", self.inner.orders(), self.inner.memory())
    }

    #[new]
    #[pyo3(signature = (
        orders = MemoryPolynomialDpd::DEFAULT_ORDERS.to_vec(),
        memory = MemoryPolynomialDpd::DEFAULT_MEMORY,
    ))]
    fn new(orders: Vec<u32>, memory: usize) -> Self {
        PyDpd { inner: MemoryPolynomialDpd::new(orders, memory) }
    }

    /// numpy.ndarray: The coefficients, order after order, tap after tap (complex128).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> CArr<'py> {
        carr(py, self.inner.coefficients())
    }

    /// Predistort a block.
    ///
    /// Args:
    ///     x (array_like): the input envelope.
    ///
    /// Returns:
    ///     numpy.ndarray: the predistorted envelope (complex128).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn apply<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let v = cvec(x)?;
        Ok(carr(py, py.detach(|| self.inner.apply(&v))))
    }

    /// Fit the polynomial mapping `x` to `y` by regularised least squares.
    ///
    /// Args:
    ///     x (array_like): the input of the polynomial.
    ///     y (array_like): the output it should give.
    ///     regularization (float): the Tikhonov regularisation.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (x, y, *, regularization=1e-9))]
    fn fit(&mut self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, regularization: f64) -> PyResult<()> {
        self.inner.fit(&cvec(x)?, &cvec(y)?, regularization).py()
    }

    /// Indirect learning against an amplifier.
    ///
    /// Args:
    ///     x (array_like): the training signal.
    ///     pa (callable): the amplifier, an array to an array.
    ///     gain (complex): the target complex gain of the cascade.
    ///     iterations (int): iterations of the indirect learning.
    ///     regularization (float): the regularisation of every fit.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (x, pa, *, gain=C64::new(1.0, 0.0), iterations=3, regularization=1e-9))]
    fn train_indirect(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        pa: &Bound<'_, PyAny>,
        gain: C64,
        iterations: usize,
        regularization: f64,
    ) -> PyResult<()> {
        let x = cvec(x)?;
        // the exception raised by `pa`, passed through the Rust error
        let mut raised = None;
        let amplify = |z: &[C64]| {
            pa.call1((carr(py, z.to_vec()),)).and_then(|y| cvec(&y)).map_err(|e| {
                raised = Some(e);
                comrs::Error::Config("the amplifier raised an exception".into())
            })
        };
        let r = self.inner.train_indirect(&x, amplify, gain, iterations, regularization);
        match raised {
            Some(e) => Err(e),
            None => r.py(),
        }
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(amplifier_bussgang, m)?)?;
    m.add_class::<PyDpd>()?;
    m.add_function(wrap_pyfunction!(pa_rapp, m)?)?;
    m.add_function(wrap_pyfunction!(rapp_predistort, m)?)?;
    m.add_function(wrap_pyfunction!(pa_saleh, m)?)?;
    m.add_function(wrap_pyfunction!(pa_memory_polynomial, m)?)?;
    Ok(())
}
