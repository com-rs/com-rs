//! The acceptance rule of the expectations of the specifications and of the validation
//! cards (`comrs verify`).

use crate::fvec;
use comrs::spec::verify::Tolerance;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The acceptance rule of a measured value `v` against an expected value `e`, as
/// ``comrs verify`` and the validation cards apply it: `|v - e| <= atol + rtol |e|`
/// (with ``atol`` or ``rtol``), or `e/factor <= v <= factor e` for a positive `e` (with
/// ``factor``); a value passes when one of the given rules accepts it.
///
/// Args:
///     atol (float | None): the absolute tolerance.
///     rtol (float | None): the tolerance relative to the expected value.
///     factor (float | None): the factor, at least 1.
///
/// Examples:
///     >>> import comrs
///     >>> t = comrs.Tolerance(factor=1.6)
///     >>> t.accepts(0.30, 0.26), t.accepts(0.5, 0.26)
///     (True, False)
///     >>> comrs.Tolerance(rtol=0.05).accepts([1.0, 2.05], [1.0, 2.0])
///     True
#[pyclass(name = "Tolerance", module = "comrs")]
#[derive(Clone)]
pub struct PyTolerance {
    inner: Tolerance,
}

#[pymethods]
impl PyTolerance {
    #[new]
    #[pyo3(signature = (*, atol = None, rtol = None, factor = None))]
    fn new(atol: Option<f64>, rtol: Option<f64>, factor: Option<f64>) -> Self {
        PyTolerance { inner: Tolerance { atol, rtol, factor } }
    }

    /// float | None: The absolute tolerance.
    #[getter]
    fn atol(&self) -> Option<f64> {
        self.inner.atol
    }

    /// float | None: The relative tolerance.
    #[getter]
    fn rtol(&self) -> Option<f64> {
        self.inner.rtol
    }

    /// float | None: The factor.
    #[getter]
    fn factor(&self) -> Option<f64> {
        self.inner.factor
    }

    /// bool: Whether a rule is given (an empty tolerance accepts nothing).
    #[getter]
    fn is_set(&self) -> bool {
        self.inner.is_set()
    }

    /// Whether every measured value is accepted for its expected value.
    ///
    /// Args:
    ///     value (float | array_like): the measured values.
    ///     expected (float | array_like): the expected values, one for all or one per
    ///         measured value.
    ///
    /// Returns:
    ///     bool: whether all are accepted.
    ///
    /// Raises:
    ///     ValueError: for arrays of different sizes, or values that are not real numbers.
    fn accepts(&self, value: &Bound<'_, PyAny>, expected: &Bound<'_, PyAny>) -> PyResult<bool> {
        let flat = |x: &Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
            fvec(&crate::as_numpy(x, "float64")?.call_method1("reshape", (-1,))?)
        };
        let (v, e) = (flat(value)?, flat(expected)?);
        if e.len() != 1 && e.len() != v.len() {
            return Err(PyValueError::new_err(format!("{} values for {} expected values", v.len(), e.len())));
        }
        Ok(v.iter().enumerate().all(|(i, &x)| self.inner.accepts(x, e[if e.len() == 1 { 0 } else { i }])))
    }

    fn __repr__(&self) -> String {
        let opt = |x: Option<f64>| x.map_or_else(|| "None".to_string(), |v| v.to_string());
        format!(
            "Tolerance(atol={}, rtol={}, factor={})",
            opt(self.inner.atol),
            opt(self.inner.rtol),
            opt(self.inner.factor)
        )
    }
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTolerance>()
}
