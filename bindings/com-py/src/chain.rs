//! The native side of `comrs.Chain`: the state of a link (`Frame`), one block of the
//! registry built from its parameters (`LinkBlock`), and the metrics of a frame.

use crate::comm::PyConstellation;
use crate::rng::PyRng;
use crate::{CArr, IntoPy, carr, cvec, iarr, json};
use comrs::sim::scenario::Record;
use comrs::sim::stats::Estimate;
use comrs::spec::blocks::{self, Block, BuildContext, Frame};
use comrs::spec::metrics::Metrics;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The state carried along a link during one trial: the current samples and what the
/// metrics need (the transmitted bits, symbol indices and symbols, the constellation).
/// Python blocks read and write `samples` (and `sps`); the other fields are set by the
/// source.
///
/// Examples:
///     >>> import comrs
///     >>> f = comrs.Frame()
///     >>> f.samples = [1, -1]
///     >>> f.samples.dtype, f.n_streams
///     (dtype('complex128'), 1)
#[pyclass(name = "Frame", module = "comrs")]
#[derive(Clone)]
pub struct PyFrame {
    pub(crate) inner: Frame,
}

#[pymethods]
impl PyFrame {
    /// An empty frame (the first block of a link must be a source).
    #[new]
    fn new() -> Self {
        PyFrame { inner: Frame::default() }
    }

    /// numpy.ndarray: Current samples of antenna 0 (complex128; read and write).
    #[getter]
    fn get_samples<'py>(&self, py: Python<'py>) -> CArr<'py> {
        carr(py, self.inner.samples.clone())
    }

    #[setter]
    fn set_samples(&mut self, x: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.samples = cvec(x)?;
        Ok(())
    }

    /// float: Samples per symbol (read and write).
    #[getter]
    fn get_sps(&self) -> f64 {
        self.inner.sps
    }

    #[setter]
    fn set_sps(&mut self, sps: f64) {
        self.inner.sps = sps;
    }

    /// int: The index of the trial of the condition (set by the runners; 0 outside a
    /// campaign), for the blocks that save or label a trial (read and write).
    #[getter]
    fn get_trial(&self) -> u64 {
        self.inner.trial
    }

    #[setter]
    fn set_trial(&mut self, trial: u64) {
        self.inner.trial = trial;
    }

    /// float: Symbol rate (Bd).
    #[getter]
    fn symbol_rate(&self) -> f64 {
        self.inner.symbol_rate
    }

    /// numpy.ndarray: Transmitted bits (uint8).
    #[getter]
    fn tx_bits<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
        self.inner.bits.clone().into_pyarray(py)
    }

    /// numpy.ndarray: Transmitted symbol indices (int64).
    #[getter]
    fn tx_indices<'py>(&self, py: Python<'py>) -> crate::IArr<'py> {
        iarr(py, self.inner.indices.clone())
    }

    /// numpy.ndarray: Transmitted symbols (complex128).
    #[getter]
    fn tx_symbols<'py>(&self, py: Python<'py>) -> CArr<'py> {
        carr(py, self.inner.symbols.clone())
    }

    /// Constellation | None: The constellation of the source (unit energy), if any.
    #[getter]
    fn constellation(&self) -> Option<PyConstellation> {
        self.inner.constellation.clone().map(|inner| PyConstellation { inner })
    }

    /// int: Number of antenna streams (1 for a single antenna).
    #[getter]
    fn n_streams(&self) -> usize {
        self.inner.n_streams()
    }

    /// list[numpy.ndarray]: The samples of every antenna (antenna 0 is `samples`).
    #[getter]
    fn streams<'py>(&self, py: Python<'py>) -> Vec<CArr<'py>> {
        self.inner.streams().map(|x| carr(py, x.clone())).collect()
    }

    /// numpy.ndarray: Noise variance of every received symbol after equalisation or
    /// detection (float64).
    #[getter]
    fn symbol_noise<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.inner.receiver.symbol_noise.clone().into_pyarray(py)
    }

    /// float | None: Noise variance per sample added by the channel blocks, if any.
    #[getter]
    fn noise_var(&self) -> Option<f64> {
        self.inner.channel.noise_var
    }

    /// numpy.ndarray: Information bits of the coded blocks (uint8).
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
        self.inner.coding.payload.clone().into_pyarray(py)
    }

    /// numpy.ndarray: Information bits decided by the decoder (uint8).
    #[getter]
    fn decoded<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u8>> {
        self.inner.coding.decoded.clone().into_pyarray(py)
    }

    /// numpy.ndarray: Bit LLRs of the demapper or the detector (positive: bit 0).
    #[getter]
    fn llr<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.inner.coding.llr.clone().into_pyarray(py)
    }

    /// int: Coded blocks per trial.
    #[getter]
    fn n_blocks(&self) -> usize {
        self.inner.coding.n_blocks
    }

    /// list[tuple[str, float]]: Diagnostic values recorded by the blocks.
    #[getter]
    fn values(&self) -> Vec<(String, f64)> {
        self.inner.values.clone()
    }

    /// Record a diagnostic value (reported as a mean by the runners).
    ///
    /// Args:
    ///     name (str): the name of the metric.
    ///     value (float): the value.
    fn record(&mut self, name: String, value: f64) {
        self.inner.values.push((name, value));
    }

    /// A copy of the frame.
    ///
    /// Returns:
    ///     Frame: an independent copy.
    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame({} samples, sps = {}, {} symbols)",
            self.inner.samples.len(),
            self.inner.sps,
            self.inner.symbols.len()
        )
    }
}

/// One block of the registry of declarative links (see `spec_blocks()`), built from
/// its specification `{"type": ..., <parameters>}`.
///
/// Args:
///     spec (dict): the block, its type and its parameters (no variable left).
///     fiber (dict, optional): the ``fiber`` of the specification, for the optical
///         blocks.
///     name (str, optional): the name of the campaign, for the blocks that write files.
///     condition (dict, optional): the condition, labelled as the Rust runner labels it
///         (``k=v, k=v``), for the blocks that write files.
///     base_dir (str | os.PathLike, optional): the directory the relative paths of the
///         block are resolved against (the directory of its specification file).
///
/// Examples:
///     >>> import comrs
///     >>> f = comrs.Frame()
///     >>> comrs.LinkBlock({"type": "source", "constellation": "QPSK", "n_symbols": 8}).apply(f, comrs.Rng(1))
///     >>> f.samples.size
///     8
#[pyclass(name = "LinkBlock", module = "comrs")]
pub struct PyLinkBlock {
    inner: Box<dyn Block>,
}

#[pymethods]
impl PyLinkBlock {
    #[new]
    #[pyo3(signature = (spec, *, fiber=None, name=None, condition=None, base_dir=None))]
    fn new(
        py: Python<'_>,
        spec: &Bound<'_, PyAny>,
        fiber: Option<&Bound<'_, PyAny>>,
        name: Option<String>,
        condition: Option<&Bound<'_, PyDict>>,
        base_dir: Option<std::path::PathBuf>,
    ) -> PyResult<Self> {
        let v = json(py, spec)?;
        let condition = match condition {
            Some(c) if !c.is_empty() => Some(crate::sim::condition_of(c)?.label()),
            _ => None,
        };
        let ctx = BuildContext { fiber: fiber.map(|f| json(py, f)).transpose()?, base_dir, name, condition };
        Ok(PyLinkBlock { inner: blocks::build(&v, &ctx).py()? })
    }

    /// Apply the block to a frame in place.
    ///
    /// Args:
    ///     frame (Frame): the frame, modified.
    ///     rng (Rng): the generator of the block.
    fn apply(&self, frame: &mut PyFrame, mut rng: PyRefMut<'_, PyRng>) -> PyResult<()> {
        self.inner.apply(&mut frame.inner, &mut rng.inner).py()
    }
}

/// The metrics of one frame, as a declarative campaign records them.
///
/// Args:
///     frame (Frame): the frame after the link.
///     metrics (dict): the ``metrics`` mapping of a specification (``report``, …).
///
/// Returns:
///     dict: ``{name: {"kind": "rate", "errors", "total"}}`` or
///     ``{name: {"kind": "mean", "value", "count"}}``.
///
/// Examples:
///     >>> import comrs
///     >>> f = comrs.Frame()
///     >>> comrs.LinkBlock({"type": "source", "constellation": "QPSK", "n_symbols": 8}).apply(f, comrs.Rng(1))
///     >>> comrs.frame_metrics(f, {"report": ["ser"]})["ser"]
///     {'kind': 'rate', 'errors': 0, 'total': 8}
#[pyfunction]
fn frame_metrics<'py>(py: Python<'py>, frame: &PyFrame, metrics: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let m = Metrics::from_json(&json(py, metrics)?).py()?;
    let mut rec = Record::new();
    m.record(&frame.inner, &mut rec).py()?;
    for (k, v) in &frame.inner.values {
        rec.value(k, *v);
    }
    rec.merge(&frame.inner.probes);
    let out = PyDict::new(py);
    for (name, e) in rec.iter() {
        let d = PyDict::new(py);
        if let Estimate::Rate(r) = e {
            d.set_item("kind", "rate")?;
            d.set_item("errors", r.errors)?;
            d.set_item("total", r.total)?;
        } else {
            d.set_item("kind", "mean")?;
            d.set_item("value", e.value())?;
            d.set_item("count", e.count())?;
        }
        out.set_item(name, d)?;
    }
    Ok(out)
}

/// A YAML text as JSON (the YAML parser of the specifications, for `Chain.from_spec`).
///
/// Args:
///     text (str): the YAML text.
///
/// Returns:
///     str: the same document as JSON.
#[pyfunction]
#[pyo3(name = "_yaml_to_json")]
fn yaml_to_json(text: &str) -> PyResult<String> {
    let v: serde_json::Value =
        serde_norway::from_str(text).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(v.to_string())
}

/// The labels of the blocks of a link as the Rust runner names them (the `label`, else
/// the type; a repeated one takes the position of its block, counted from 1).
///
/// Args:
///     link (list): the blocks of a link, names or mappings with a ``type``.
///
/// Returns:
///     list[str]: one label per block.
#[pyfunction]
#[pyo3(name = "_link_labels")]
fn link_labels(py: Python<'_>, link: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let v = json(py, link)?;
    let items = v.as_array().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("the link is a list of blocks"))?;
    comrs::spec::link_labels(items).py()
}

/// Substitute the variables of a condition in a value of a specification, as the Rust
/// runner does: a string that is exactly ``$name`` or ``${name}`` takes the value of the
/// variable, ``${name}`` inside a longer string its text, and ``$$`` is a dollar sign.
///
/// Args:
///     value (Any): a value of a specification (a string, a list, a dict, a scalar).
///     condition (dict): the values of the grid variables.
///
/// Returns:
///     Any: the value with the variables substituted.
///
/// Raises:
///     ValueError: for a variable that the condition does not define.
#[pyfunction]
#[pyo3(name = "_substitute")]
fn substitute<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    condition: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let cond = crate::sim::condition_of(condition)?;
    crate::from_json(py, &comrs::spec::substitute(&json(py, value)?, &cond).py()?)
}

/// The variable that a string of a specification is exactly (``name`` for ``$name`` or
/// ``${name}``), None for any other string.
///
/// Args:
///     text (str): the string.
///
/// Returns:
///     str | None: the name of the variable.
#[pyfunction]
#[pyo3(name = "_spec_variable")]
fn spec_variable(text: &str) -> Option<&str> {
    comrs::spec::variable(text)
}

/// Register the chain bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrame>()?;
    m.add_class::<PyLinkBlock>()?;
    m.add_function(wrap_pyfunction!(frame_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(yaml_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(link_labels, m)?)?;
    m.add_function(wrap_pyfunction!(substitute, m)?)?;
    m.add_function(wrap_pyfunction!(spec_variable, m)?)?;
    Ok(())
}
