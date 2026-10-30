//! The Monte Carlo runner of Python: Python trials on the Rust runner.
//!
//! [`PyScenario`] implements [`Scenario`] by calling the Python trial (and setup) under
//! the GIL; the runner runs it with one thread (Python blocks are not thread-safe; a
//! chain runs on more when its caller or its specification asks for them), so
//! that `comrs.sim.run` and the Python path of `Chain.run` share the Rust runner, its
//! statistics, its stopping rules (parsed by Rust from the `stop` mapping of a
//! specification), its statuses, its condition labels and its CSV and JSON loggers.
//! The exception of a Python trial is raised again unchanged, with a note naming the
//! trial and the condition.

use crate::chain::PyFrame;
use crate::rng::PyRng;
use crate::spec::{condition_dict, result_with};
use crate::{IntoPy, json};
use comrs::core::rng::Rng;
use comrs::sim::callbacks::ProgressBar;
use comrs::sim::grid::{Condition, Grid, Value};
use comrs::sim::loggers::{CsvLogger, JsonLogger};
use comrs::sim::runner::{ConditionResult, Runner};
use comrs::sim::scenario::{Record, Scenario};
use comrs::sim::stats::{ErrorRate, Estimate, Mean};
use comrs::sim::stopping::Stop;
use comrs::spec::{RunOptions, Spec};
use comrs::spec::metrics::Metrics;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

/// The message of the Rust error that carries a Python exception through the runner.
const RAISED: &str = "the Python code raised an exception";

/// A condition from a Python mapping: booleans, integers, reals and strings keep their
/// type; any other value is labelled by its `str`.
pub(crate) fn condition_of(d: &Bound<'_, PyAny>) -> PyResult<Condition> {
    let d = d.cast::<PyDict>().map_err(|_| PyTypeError::new_err("a condition is a dict"))?;
    let mut c = Condition::new();
    for (k, v) in d.iter() {
        let key: String = k.extract().map_err(|_| PyTypeError::new_err("the keys of a condition are strings"))?;
        let value = if v.is_instance_of::<PyBool>() {
            Value::Bool(v.extract()?)
        } else if v.is_instance_of::<PyInt>() {
            Value::I64(v.extract()?)
        } else if v.is_instance_of::<PyFloat>() {
            Value::F64(v.extract()?)
        } else if v.is_instance_of::<PyString>() {
            Value::Str(v.extract()?)
        } else if let Ok(i) = v.extract::<i64>() {
            Value::I64(i)
        } else if let Ok(x) = v.extract::<f64>() {
            Value::F64(x)
        } else {
            Value::Str(v.str()?.to_string())
        };
        c.set(&key, value);
    }
    Ok(c)
}

/// A stopping rule, checked after every batch of trials of a condition; rules combine
/// with ``|`` (any of them) and ``&`` (all of them).
///
/// The rules of the Rust runner (``com_sim::stopping::Stop``): :meth:`max_trials`,
/// :meth:`min_trials`, :meth:`min_errors`, :meth:`precision` and :meth:`max_time`. A rule
/// must be bounded (contain ``max_trials`` or ``max_time`` on every branch of an
/// ``&``). :meth:`from_spec` parses the ``stop`` mapping of a specification.
///
/// Examples:
///     >>> from comrs.sim import Stop
///     >>> Stop.min_errors("ber", 100) | Stop.max_trials(1000)
///     (errors(ber) ≥ 100 or max_trials(1000))
///     >>> Stop.from_spec({"min_errors": {"ber": 100}, "max_trials": 1000}).bounded
///     True
#[pyclass(name = "Stop", module = "comrs.sim", frozen)]
#[derive(Clone)]
pub struct PyStop {
    pub(crate) inner: Stop,
}

#[pymethods]
impl PyStop {
    /// Stop after a number of trials.
    ///
    /// Args:
    ///     n (int): the number of trials.
    ///
    /// Returns:
    ///     Stop: the rule.
    #[staticmethod]
    fn max_trials(n: u64) -> Self {
        PyStop { inner: Stop::max_trials(n) }
    }

    /// Never stop before a number of trials (combine with ``&``).
    ///
    /// Args:
    ///     n (int): the number of trials.
    ///
    /// Returns:
    ///     Stop: the rule.
    #[staticmethod]
    fn min_trials(n: u64) -> Self {
        PyStop { inner: Stop::min_trials(n) }
    }

    /// Stop after a number of errors of an error rate.
    ///
    /// Args:
    ///     metric (str): the error rate.
    ///     errors (int): the number of errors.
    ///
    /// Returns:
    ///     Stop: the rule.
    #[staticmethod]
    fn min_errors(metric: &str, errors: u64) -> Self {
        PyStop { inner: Stop::min_errors(metric, errors) }
    }

    /// Stop when the confidence interval of a metric is within a relative half-width of
    /// its value.
    ///
    /// Args:
    ///     metric (str): the metric.
    ///     relative (float): the relative half-width.
    ///     confidence (float): the level of the interval.
    ///
    /// Returns:
    ///     Stop: the rule.
    #[staticmethod]
    #[pyo3(signature = (metric, relative, *, confidence = 0.95))]
    fn precision(metric: &str, relative: f64, confidence: f64) -> Self {
        PyStop { inner: Stop::Precision { metric: metric.to_string(), relative, confidence } }
    }

    /// Stop after an amount of wall time on the condition (machine-dependent).
    ///
    /// Args:
    ///     seconds (float): the time (s).
    ///
    /// Returns:
    ///     Stop: the rule.
    ///
    /// Raises:
    ///     ValueError: if ``seconds`` is negative, infinite or NaN.
    #[staticmethod]
    fn max_time(seconds: f64) -> PyResult<Self> {
        let d = Duration::try_from_secs_f64(seconds)
            .map_err(|_| PyValueError::new_err(format!("max_time needs a finite, non-negative time, got {seconds}")))?;
        Ok(PyStop { inner: Stop::max_time(d) })
    }

    /// The rule of the ``stop`` mapping of a specification, parsed as the Rust runner
    /// parses it: the targets (``min_errors``, ``precision``, with ``min_trials``) or
    /// the caps (``max_trials``, ``max_time_s``); ``min_trials`` alone caps the trials,
    /// and an empty mapping gives 100 trials.
    ///
    /// Args:
    ///     stop (dict): the mapping.
    ///
    /// Returns:
    ///     Stop: the rule.
    ///
    /// Examples:
    ///     >>> from comrs.sim import Stop
    ///     >>> Stop.from_spec({"min_trials": 50})
    ///     max_trials(50)
    #[staticmethod]
    fn from_spec(py: Python<'_>, stop: &Bound<'_, PyAny>) -> PyResult<Self> {
        let v = serde_json::json!({ "name": "stop", "link": ["source"], "stop": json(py, stop)? });
        Ok(PyStop { inner: Spec::from_json(v).py()?.stop })
    }

    /// bool: Whether the rule can stop (a rule made only of lower bounds runs forever).
    #[getter]
    fn bounded(&self) -> bool {
        self.inner.is_bounded()
    }

    fn __or__(&self, other: &PyStop) -> PyStop {
        PyStop { inner: self.inner.clone().or(other.inner.clone()) }
    }

    fn __and__(&self, other: &PyStop) -> PyStop {
        PyStop { inner: self.inner.clone().and(other.inner.clone()) }
    }

    fn __eq__(&self, other: &PyStop) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        self.inner.to_string()
    }
}

/// An error rate: errors out of trials, with a Wilson score interval
/// (``com_sim::stats::ErrorRate``).
///
/// Args:
///     errors (int): errors counted.
///     total (int): trials counted.
///
/// Examples:
///     >>> from comrs.sim import ErrorRate
///     >>> r = ErrorRate(3, 1000)
///     >>> r.value, [round(v, 5) for v in r.interval(0.95)]
///     (0.003, [0.00102, 0.00878])
#[pyclass(name = "ErrorRate", module = "comrs.sim")]
#[derive(Clone)]
pub struct PyErrorRate {
    inner: ErrorRate,
}

#[pymethods]
impl PyErrorRate {
    #[new]
    #[pyo3(signature = (errors = 0, total = 0))]
    fn new(errors: u64, total: u64) -> Self {
        PyErrorRate { inner: ErrorRate { errors, total } }
    }

    /// int: Errors counted.
    #[getter]
    fn errors(&self) -> u64 {
        self.inner.errors
    }

    /// int: Trials counted.
    #[getter]
    fn total(&self) -> u64 {
        self.inner.total
    }

    /// float: The rate (``nan`` without trials).
    #[getter]
    fn value(&self) -> f64 {
        self.inner.rate()
    }

    /// int: Number of errors (the count of the stopping rules).
    #[getter]
    fn count(&self) -> u64 {
        self.inner.errors
    }

    /// Count errors.
    ///
    /// Args:
    ///     errors (int): errors among the new trials.
    ///     total (int): new trials.
    fn add(&mut self, errors: u64, total: u64) {
        self.inner.add(errors, total);
    }

    /// Add the counts of another estimate.
    ///
    /// Args:
    ///     other (ErrorRate): the other estimate.
    fn merge(&mut self, other: &PyErrorRate) {
        self.inner.merge(&other.inner);
    }

    /// The Wilson score interval, valid down to zero errors (``(0, 1)`` without trials).
    ///
    /// Args:
    ///     confidence (float): the level of the interval.
    ///
    /// Returns:
    ///     tuple[float, float]: the bounds.
    #[pyo3(signature = (confidence = 0.95))]
    fn interval(&self, confidence: f64) -> (f64, f64) {
        self.inner.wilson(confidence)
    }

    /// Half-width of the interval relative to the rate (``inf`` without errors).
    ///
    /// Args:
    ///     confidence (float): the level of the interval.
    ///
    /// Returns:
    ///     float: the relative half-width.
    #[pyo3(signature = (confidence = 0.95))]
    fn relative_precision(&self, confidence: f64) -> f64 {
        Estimate::Rate(self.inner).relative_precision(confidence)
    }

    fn __eq__(&self, other: &PyErrorRate) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("ErrorRate(errors={}, total={})", self.inner.errors, self.inner.total)
    }
}

/// A running mean and variance (Welford), merged exactly (Chan et al.), with a normal
/// interval (``com_sim::stats::Mean``).
///
/// Examples:
///     >>> from comrs.sim import Mean
///     >>> m = Mean()
///     >>> for x in [1.0, 2.0, 3.0]:
///     ...     m.add(x)
///     >>> m.value, m.std, m.count
///     (2.0, 1.0, 3)
#[pyclass(name = "Mean", module = "comrs.sim")]
#[derive(Clone)]
pub struct PyMean {
    inner: Mean,
}

#[pymethods]
impl PyMean {
    #[new]
    fn new() -> Self {
        PyMean { inner: Mean::default() }
    }

    /// float: The mean (``nan`` without values).
    #[getter]
    fn value(&self) -> f64 {
        self.inner.mean()
    }

    /// int: Number of values.
    #[getter]
    fn count(&self) -> u64 {
        self.inner.count()
    }

    /// float: Unbiased sample variance (``nan`` with fewer than two values).
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// float: Sample standard deviation.
    #[getter]
    fn std(&self) -> f64 {
        self.inner.std()
    }

    /// float: Standard error of the mean, ``std/√n``.
    #[getter]
    fn sem(&self) -> f64 {
        self.inner.sem()
    }

    /// Add one value.
    ///
    /// Args:
    ///     x (float): the value.
    fn add(&mut self, x: f64) {
        self.inner.add(x);
    }

    /// Merge the values of another estimate.
    ///
    /// Args:
    ///     other (Mean): the other estimate.
    fn merge(&mut self, other: &PyMean) {
        self.inner.merge(&other.inner);
    }

    /// The normal interval of the mean, ``mean ± z·std/√n`` (unbounded with fewer than
    /// two values).
    ///
    /// Args:
    ///     confidence (float): the level of the interval.
    ///
    /// Returns:
    ///     tuple[float, float]: the bounds.
    #[pyo3(signature = (confidence = 0.95))]
    fn interval(&self, confidence: f64) -> (f64, f64) {
        self.inner.interval(confidence)
    }

    /// Half-width of the interval relative to the mean (``inf`` for a zero mean).
    ///
    /// Args:
    ///     confidence (float): the level of the interval.
    ///
    /// Returns:
    ///     float: the relative half-width.
    #[pyo3(signature = (confidence = 0.95))]
    fn relative_precision(&self, confidence: f64) -> f64 {
        Estimate::Mean(self.inner).relative_precision(confidence)
    }

    fn __eq__(&self, other: &PyMean) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("Mean(value={}, count={})", self.inner.mean(), self.inner.count())
    }
}

/// An estimate as its Python class.
fn estimate_obj(py: Python<'_>, e: &Estimate) -> PyResult<Py<PyAny>> {
    Ok(match e {
        Estimate::Rate(r) => Py::new(py, PyErrorRate { inner: *r })?.into_any(),
        Estimate::Mean(m) => Py::new(py, PyMean { inner: *m })?.into_any(),
    })
}

/// The metrics of a record as a dict of estimates.
fn metrics_dict<'py>(py: Python<'py>, rec: &Record) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, e) in rec.iter() {
        d.set_item(k, estimate_obj(py, e)?)?;
    }
    Ok(d)
}

/// The metrics of one trial, or merged over many (``com_sim::scenario::Record``):
/// :meth:`errors` counts errors (an error rate), :meth:`value` adds values (a mean).
///
/// Args:
///     trial (int): the index of the trial of its condition (0 when merged), for the
///         blocks that save or label a trial.
///
/// Examples:
///     >>> from comrs.sim import Record
///     >>> rec = Record()
///     >>> rec.errors("ber", 3, 100)
///     >>> rec.value("snr_db", 9.5)
///     >>> rec["ber"].value, sorted(rec.metrics)
///     (0.03, ['ber', 'snr_db'])
#[pyclass(name = "Record", module = "comrs.sim")]
pub struct PyRecord {
    inner: Record,
}

#[pymethods]
impl PyRecord {
    #[new]
    #[pyo3(signature = (trial = 0))]
    fn new(trial: u64) -> Self {
        PyRecord { inner: Record::for_trial(trial) }
    }

    /// int: The index of the trial of its condition (0 when merged).
    #[getter]
    fn trial(&self) -> u64 {
        self.inner.trial()
    }

    /// dict[str, ErrorRate | Mean]: The metrics, sorted by name (copies).
    #[getter]
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metrics_dict(py, &self.inner)
    }

    /// Count errors of an error rate.
    ///
    /// Args:
    ///     name (str): the error rate.
    ///     errors (int): errors among the events.
    ///     total (int): events.
    ///
    /// Raises:
    ///     TypeError: if ``name`` is a mean.
    fn errors(&mut self, name: &str, errors: u64, total: u64) -> PyResult<()> {
        if matches!(self.inner.get(name), Some(Estimate::Mean(_))) {
            return Err(PyTypeError::new_err(format!("metric {name:?} is a mean, not a rate")));
        }
        self.inner.errors(name, errors, total);
        Ok(())
    }

    /// Add a value to a mean (non-finite values are ignored).
    ///
    /// Args:
    ///     name (str): the mean.
    ///     x (float): the value.
    ///
    /// Raises:
    ///     TypeError: if ``name`` is an error rate.
    fn value(&mut self, name: &str, x: f64) -> PyResult<()> {
        if matches!(self.inner.get(name), Some(Estimate::Rate(_))) {
            return Err(PyTypeError::new_err(format!("metric {name:?} is a rate, not a mean")));
        }
        self.inner.value(name, x);
        Ok(())
    }

    /// Merge the metrics of another record.
    ///
    /// Args:
    ///     other (Record): the other record.
    ///
    /// Raises:
    ///     TypeError: for a metric that is a rate in one and a mean in the other.
    fn merge(&mut self, other: &PyRecord) -> PyResult<()> {
        for (k, e) in other.inner.iter() {
            if let Some(mine) = self.inner.get(k) {
                if std::mem::discriminant(mine) != std::mem::discriminant(e) {
                    return Err(PyTypeError::new_err(format!("metric {k:?} is reported as both a rate and a mean")));
                }
            }
        }
        self.inner.merge(&other.inner);
        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let e = self.inner.get(name).ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        estimate_obj(py, e)
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.get(name).is_some()
    }

    fn __repr__(&self) -> String {
        let names: Vec<&str> = self.inner.iter().map(|(k, _)| k.as_str()).collect();
        format!("Record(trial={}, metrics={names:?})", self.inner.trial())
    }
}

/// The result of one condition of a campaign (``com_sim::runner::ConditionResult``).
#[pyclass(name = "ConditionResult", module = "comrs.sim", frozen)]
pub struct PyConditionResult {
    inner: ConditionResult,
    condition: Py<PyAny>,
}

#[pymethods]
impl PyConditionResult {
    /// int: Index of the condition in the grid.
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    /// dict: The condition (as given).
    #[getter]
    fn condition(&self, py: Python<'_>) -> Py<PyAny> {
        self.condition.clone_ref(py)
    }

    /// str: How it ended: ``completed``, ``stopped``, ``skipped`` or ``dev_run``.
    #[getter]
    fn status(&self) -> &'static str {
        self.inner.status.as_str()
    }

    /// int: Trials run.
    #[getter]
    fn trials(&self) -> u64 {
        self.inner.trials
    }

    /// float: Wall time of the condition, setup included (s).
    #[getter]
    fn wall_s(&self) -> f64 {
        self.inner.wall_s
    }

    /// dict[str, ErrorRate | Mean]: The metrics merged over the trials.
    #[getter]
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metrics_dict(py, &self.inner.record)
    }

    /// A flat row, as the CSV logger writes it (``com_sim::runner::ConditionResult::row``).
    ///
    /// Args:
    ///     confidence (float): the level of the intervals.
    ///
    /// Returns:
    ///     dict: ``index``, the condition (its values as given), ``status``,
    ///     ``trials``, ``wall_s``, and for every metric ``m``: ``m``, ``m_lo``,
    ///     ``m_hi``, ``m_count``, ``m_total``.
    #[pyo3(signature = (confidence = 0.95))]
    fn row<'py>(&self, py: Python<'py>, confidence: f64) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in self.inner.row(confidence) {
            d.set_item(k, crate::spec::value(py, &v)?)?;
        }
        // the values of the condition as given, in place of their labels
        if let Ok(c) = self.condition.bind(py).cast::<PyDict>() {
            for (k, v) in c.iter() {
                d.set_item(k, v)?;
            }
        }
        Ok(d)
    }

    /// The result as a dict, the format of :func:`comrs.run_spec`.
    ///
    /// Args:
    ///     confidence (float): the level of the intervals.
    ///
    /// Returns:
    ///     dict: ``index``, ``condition``, ``status``, ``trials``, ``wall_s`` and
    ///     ``metrics`` (``{name: {value, lo, hi, kind, count, total}}``).
    #[pyo3(signature = (confidence = 0.95))]
    fn to_dict<'py>(&self, py: Python<'py>, confidence: f64) -> PyResult<Bound<'py, PyDict>> {
        result_with(py, &self.inner, self.condition.bind(py).clone(), confidence)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ConditionResult(index={}, condition={}, status={:?}, trials={})",
            self.inner.index,
            self.condition.bind(py).repr()?,
            self.inner.status.as_str(),
            self.inner.trials
        ))
    }
}

/// What a trial of a [`PyScenario`] calls.
enum Trial {
    /// `trial(condition, rng, record[, state])`, the trials of `comrs.sim.run`.
    Function(Py<PyAny>),
    /// The blocks of a chain: `setup(condition)` returns `apply(frame, rng)`; the
    /// frame is then recorded by the metrics of the specification, as a declarative
    /// link records it.
    Chain(Box<Spec>),
}

/// A scenario whose setup and trials are Python callables (see the module
/// documentation).
struct PyScenario {
    name: String,
    trial: Trial,
    setup: Option<Py<PyAny>>,
    /// The conditions of the grid and their Python dicts, matched in order.
    conditions: Vec<(Condition, Py<PyAny>)>,
    cursor: Mutex<usize>,
    /// The first Python exception, raised again when the runner returns.
    raised: Mutex<Option<PyErr>>,
}

/// The state of a condition: its Python dict, the result of the setup, the metrics.
struct PyState {
    condition: Py<PyAny>,
    state: Option<Py<PyAny>>,
    metrics: Option<Metrics>,
}

impl PyScenario {
    /// Keep the Python exception and return the Rust error that stops the runner.
    fn fail(&self, e: PyErr) -> comrs::Error {
        if let Ok(mut r) = self.raised.lock() {
            if r.is_none() {
                *r = Some(e);
            }
        }
        comrs::Error::Config(RAISED.into())
    }

    /// The Python dict of a condition: the next one in order equal to it.
    fn python_condition(&self, py: Python<'_>, c: &Condition) -> comrs::Result<Py<PyAny>> {
        let mut cursor = self.cursor.lock().map_err(|_| comrs::Error::Config("poisoned lock".into()))?;
        let found = (*cursor..self.conditions.len())
            .chain(0..*cursor)
            .find(|&i| self.conditions[i].0 == *c)
            .ok_or_else(|| comrs::Error::Config(format!("unknown condition ({})", c.label())))?;
        *cursor = found + 1;
        Ok(self.conditions[found].1.clone_ref(py))
    }
}

impl Scenario for PyScenario {
    type State = PyState;

    fn name(&self) -> &str {
        &self.name
    }

    fn setup(&self, condition: &Condition) -> comrs::Result<PyState> {
        Python::attach(|py| {
            let cond = self.python_condition(py, condition)?;
            let state = match &self.setup {
                Some(f) => Some(f.call1(py, (cond.clone_ref(py),)).map_err(|e| self.fail(e))?),
                None => None,
            };
            let metrics = match &self.trial {
                Trial::Function(_) => None,
                Trial::Chain(spec) => Some(spec.link_for(condition)?.metrics),
            };
            Ok(PyState { condition: cond, state, metrics })
        })
    }

    fn trial(&self, s: &PyState, _: &Condition, rng: &mut Rng, record: &mut Record) -> comrs::Result<()> {
        Python::attach(|py| {
            let generator =
                Py::new(py, PyRng { inner: std::mem::replace(rng, Rng::new(0)) }).map_err(|e| self.fail(e))?;
            let outcome = match &self.trial {
                Trial::Function(f) => {
                    let rec =
                        Py::new(py, PyRecord { inner: Record::for_trial(record.trial()) }).map_err(|e| self.fail(e))?;
                    let cond = s.condition.clone_ref(py);
                    let called = match &s.state {
                        Some(state) => {
                            f.call1(py, (cond, generator.clone_ref(py), rec.clone_ref(py), state.clone_ref(py)))
                        }
                        None => f.call1(py, (cond, generator.clone_ref(py), rec.clone_ref(py))),
                    };
                    called.map(|_| record.merge(&rec.borrow(py).inner))
                }
                Trial::Chain(_) => {
                    let inner = comrs::spec::blocks::Frame { trial: record.trial(), ..Default::default() };
                    let frame = Py::new(py, PyFrame { inner }).map_err(|e| self.fail(e))?;
                    let apply =
                        s.state.as_ref().ok_or_else(|| comrs::Error::Config("a chain needs its blocks".into()))?;
                    let metrics =
                        s.metrics.as_ref().ok_or_else(|| comrs::Error::Config("a chain needs its metrics".into()))?;
                    apply.call1(py, (frame.clone_ref(py), generator.clone_ref(py))).and_then(|_| {
                        // what `Link::trial` records after the blocks
                        metrics.record_frame(&frame.borrow(py).inner, record).map_err(crate::err)
                    })
                }
            };
            *rng = std::mem::replace(&mut generator.borrow_mut(py).inner, Rng::new(0));
            outcome.map_err(|e| self.fail(e))
        })
    }
}

/// Run a Python scenario on the Rust runner, with the GIL released between the trials;
/// a Python exception is raised again with a note naming where it happened.
fn run_scenario(py: Python<'_>, scenario: &PyScenario, grid: &Grid, runner: Runner) -> PyResult<Vec<ConditionResult>> {
    let out = py.detach(|| runner.run(scenario, grid));
    match out {
        Ok(r) => Ok(r),
        Err(e) => {
            let raised = scenario.raised.lock().ok().and_then(|mut r| r.take());
            match raised {
                Some(pe) => {
                    let msg = match &e {
                        comrs::Error::Config(m) => m.clone(),
                        other => other.to_string(),
                    };
                    let place = msg.strip_suffix(RAISED).map_or(msg.as_str(), |s| s.trim_end_matches([':', ' ']));
                    let _ = pe.value(py).call_method1("add_note", (format!("{place} failed"),));
                    Err(pe)
                }
                None => Err(crate::err(e)),
            }
        }
    }
}

/// Run a Python trial on every condition, on the Rust runner (see `comrs.sim.run`).
///
/// Args:
///     trial (callable): ``trial(condition, rng, record[, state])``.
///     conditions (list[dict]): the conditions.
///     name (str): the scenario name.
///     seed (int): the base seed.
///     batch (int): trials between two checks of the stopping rule.
///     stop (Stop | None): the stopping rule (100 trials by default).
///     setup (callable | None): ``setup(condition) -> state``.
///     confidence (float): level of the intervals of the reports.
///     progress (bool): progress lines on the standard error.
///     csv_path (str | os.PathLike | None): the CSV report.
///     json_path (str | os.PathLike | None): the JSON report.
///
/// Returns:
///     list[ConditionResult]: one result per condition.
#[pyfunction]
#[pyo3(name = "_sim_run", signature = (trial, conditions, *, name, seed = 0, batch = 32, stop = None, setup = None, confidence = 0.95, progress = false, csv_path = None, json_path = None))]
#[allow(clippy::too_many_arguments)]
fn sim_run(
    py: Python<'_>,
    trial: Py<PyAny>,
    conditions: &Bound<'_, PyAny>,
    name: String,
    seed: u64,
    batch: u64,
    stop: Option<PyRef<'_, PyStop>>,
    setup: Option<Py<PyAny>>,
    confidence: f64,
    progress: bool,
    csv_path: Option<PathBuf>,
    json_path: Option<PathBuf>,
) -> PyResult<Vec<PyConditionResult>> {
    let mut conds = Vec::new();
    for c in conditions.try_iter()? {
        let d = PyDict::new(py);
        d.update(c?.cast::<pyo3::types::PyMapping>().map_err(|_| PyTypeError::new_err("a condition is a dict"))?)?;
        conds.push((condition_of(d.as_any())?, d.into_any().unbind()));
    }
    let grid = Grid::from_conditions(conds.iter().map(|(c, _)| c.clone()).collect());
    // Python trials are not thread-safe: one thread
    let mut runner = Runner::new()
        .seed(seed)
        .batch(batch)
        .threads(1)
        .stop(stop.map_or(Stop::max_trials(100), |s| s.inner.clone()))
        .confidence(confidence);
    if progress {
        runner = runner.callback(ProgressBar::default());
    }
    if let Some(p) = csv_path {
        runner = runner.logger(CsvLogger::new(p));
    }
    if let Some(p) = json_path {
        runner = runner.logger(JsonLogger::new(p));
    }
    let scenario = PyScenario {
        name,
        trial: Trial::Function(trial),
        setup,
        conditions: conds,
        cursor: Mutex::new(0),
        raised: Mutex::new(None),
    };
    let results = run_scenario(py, &scenario, &grid, runner)?;
    Ok(results
        .into_iter()
        .map(|r| {
            let condition = scenario.conditions[r.index].1.clone_ref(py);
            PyConditionResult { inner: r, condition }
        })
        .collect())
}

/// Run a chain with Python blocks on the Rust runner: the runner of the specification
/// `spec` (``com_spec::Spec::runner``: its seed, batch, stopping rule, confidence,
/// callbacks and outputs, overridden by the arguments given; its link is ignored), the
/// blocks of a condition from `setup(condition)`, a callable `apply(frame, rng)`.
///
/// Python blocks are not thread-safe: the chain runs on one thread unless `threads`, or
/// the ``threads`` of the specification, asks for more (the results do not depend on
/// it when the blocks draw from the generator they are given only).
///
/// Args:
///     spec (dict): the settings of the campaign.
///     setup (callable): ``setup(condition) -> apply``.
///     progress (bool): progress lines on the standard error (or those of the
///         specification).
///     stop (Stop | None): a rule instead of the one of the specification.
///     threads (int | None): worker threads (0: all cores).
///     batch (int | None): trials between two checks of the stopping rule.
///     fast_dev_run (bool): one batch per condition, the stopping rule ignored.
///     csv_path (str | os.PathLike | None): the CSV report, instead of the one of the
///         specification.
///     json_path (str | os.PathLike | None): the JSON report, instead of the one of the
///         specification.
///
/// Returns:
///     list[dict]: one dict per condition, as :func:`comrs.run_spec`.
///
/// Raises:
///     ValueError: for an invalid specification or stopping rule.
#[pyfunction]
#[pyo3(
    name = "_run_chain",
    signature = (spec, setup, *, progress = false, stop = None, threads = None, batch = None, fast_dev_run = false, csv_path = None, json_path = None)
)]
#[allow(clippy::too_many_arguments, reason = "the options of a run, keyword-only in Python")]
fn run_chain<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyAny>,
    setup: Py<PyAny>,
    progress: bool,
    stop: Option<PyRef<'py, PyStop>>,
    threads: Option<usize>,
    batch: Option<u64>,
    fast_dev_run: bool,
    csv_path: Option<PathBuf>,
    json_path: Option<PathBuf>,
) -> PyResult<Bound<'py, PyList>> {
    let mut v = json(py, spec)?;
    if let Some(o) = v.as_object_mut() {
        o.insert("link".into(), serde_json::json!(["source"]));
    }
    let spec = Spec::from_json(v).py()?;
    let grid = spec.grid.clone();
    let conditions = grid
        .conditions()
        .into_iter()
        .map(|c| Ok((c.clone(), condition_dict(py, &c)?.into_any().unbind())))
        .collect::<PyResult<Vec<_>>>()?;
    let threads = threads.or_else(|| spec.source.get("threads").map(|_| spec.threads)).or(Some(1));
    let opts = RunOptions {
        threads,
        batch,
        fast_dev_run,
        progress: progress.then_some(true),
        profile: None,
        csv: csv_path,
        json: json_path,
    };
    let mut runner = spec.runner(&opts);
    if let Some(s) = stop {
        runner = runner.stop(s.inner.clone());
    }
    let confidence = spec.confidence;
    let scenario = PyScenario {
        name: spec.name.clone(),
        trial: Trial::Chain(Box::new(spec)),
        setup: Some(setup),
        conditions,
        cursor: Mutex::new(0),
        raised: Mutex::new(None),
    };
    let results = run_scenario(py, &scenario, &grid, runner)?;
    let items = results
        .iter()
        .map(|r| result_with(py, r, scenario.conditions[r.index].1.bind(py).clone(), confidence))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// The names of the columns of a metric in a flat row (``ConditionResult.row``, the CSV
/// report and :meth:`comrs.Results.rows`).
///
/// Args:
///     name (str): the metric.
///
/// Returns:
///     list[str]: ``name``, ``name_lo``, ``name_hi``, ``name_count``, ``name_total``.
#[pyfunction]
#[pyo3(name = "_metric_columns")]
fn metric_columns(name: &str) -> Vec<String> {
    comrs::sim::runner::metric_columns(name).to_vec()
}

/// The two-sided normal quantile of a confidence level (1.96 for 0.95).
///
/// Args:
///     confidence (float): the level, in `(0, 1)`.
///
/// Returns:
///     float: the quantile `z`.
#[pyfunction]
fn z_score(confidence: f64) -> PyResult<f64> {
    if !(0.0..1.0).contains(&confidence) {
        return Err(PyValueError::new_err(format!("the confidence must be in [0, 1), got {confidence}")));
    }
    Ok(comrs::sim::stats::z_score(confidence))
}

/// Register the runner bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStop>()?;
    m.add_class::<PyErrorRate>()?;
    m.add_class::<PyMean>()?;
    m.add_class::<PyRecord>()?;
    m.add_class::<PyConditionResult>()?;
    m.add_function(wrap_pyfunction!(sim_run, m)?)?;
    m.add_function(wrap_pyfunction!(run_chain, m)?)?;
    m.add_function(wrap_pyfunction!(z_score, m)?)?;
    m.add_function(wrap_pyfunction!(metric_columns, m)?)?;
    Ok(())
}
