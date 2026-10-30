//! Declarative campaigns from Python: the Rust runner on all cores.

use crate::{IntoPy, from_json, json};
use comrs::sim::runner::ConditionResult;
use comrs::sim::stats::Estimate;
use comrs::spec::{Format, RunOptions, Spec, blocks, run, search, verify};
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use std::path::PathBuf;

/// A specification from a path, a YAML/JSON text or a mapping.
///
/// A string of one line that is neither a file nor a mapping (a misspelt path) raises
/// `FileNotFoundError`.
pub(crate) fn load(py: Python<'_>, spec: &Bound<'_, PyAny>) -> PyResult<Spec> {
    if let Ok(p) = spec.extract::<PathBuf>() {
        if !spec.is_instance_of::<PyString>() || p.is_file() {
            return Spec::from_path(&p).py();
        }
    }
    if let Ok(text) = spec.extract::<String>() {
        let scalar = || matches!(serde_norway::from_str::<serde_json::Value>(&text), Ok(v) if !v.is_object());
        if !text.contains('\n') && scalar() {
            return Err(PyFileNotFoundError::new_err(format!(
                "no specification file {text:?} (and the text is not a specification)"
            )));
        }
        return Spec::parse(&text, Format::Yaml).py();
    }
    if spec.is_instance_of::<PyDict>() {
        return Spec::parse(&json(py, spec)?.to_string(), Format::Json).py();
    }
    Err(PyValueError::new_err("spec must be a path, a YAML text or a dict"))
}

pub(crate) fn value<'py>(py: Python<'py>, v: &comrs::sim::grid::Value) -> PyResult<Bound<'py, PyAny>> {
    use comrs::sim::grid::Value;
    Ok(match v {
        Value::F64(x) => x.into_pyobject(py)?.into_any(),
        Value::I64(x) => x.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
    })
}

/// A condition as a dict.
pub(crate) fn condition_dict<'py>(py: Python<'py>, c: &comrs::sim::grid::Condition) -> PyResult<Bound<'py, PyDict>> {
    let cond = PyDict::new(py);
    for (k, v) in c.entries() {
        cond.set_item(k, value(py, v)?)?;
    }
    Ok(cond)
}

fn result<'py>(py: Python<'py>, r: &ConditionResult, confidence: f64) -> PyResult<Bound<'py, PyDict>> {
    result_with(py, r, condition_dict(py, &r.condition)?.into_any(), confidence)
}

/// The dict of one condition (``index``, ``condition``, ``status``, ``trials``,
/// ``wall_s``, ``metrics``), with the condition given as a Python object.
pub(crate) fn result_with<'py>(
    py: Python<'py>,
    r: &ConditionResult,
    cond: Bound<'py, PyAny>,
    confidence: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("index", r.index)?;
    d.set_item("condition", cond)?;
    d.set_item("status", r.status.as_str())?;
    d.set_item("trials", r.trials)?;
    d.set_item("wall_s", r.wall_s)?;
    let metrics = PyDict::new(py);
    for (k, e) in r.record.iter() {
        let m = PyDict::new(py);
        let (lo, hi) = e.interval(confidence);
        m.set_item("value", e.value())?;
        m.set_item("lo", lo)?;
        m.set_item("hi", hi)?;
        m.set_item("kind", if matches!(e, Estimate::Rate(_)) { "rate" } else { "mean" })?;
        m.set_item("count", e.count())?;
        m.set_item("total", e.total())?;
        metrics.set_item(k, m)?;
    }
    d.set_item("metrics", metrics)?;
    Ok(d)
}

/// Run a declarative campaign with the Rust runner (parallel, reproducible).
///
/// The specification describes the grid of conditions, the link as a list of blocks,
/// the metrics and the stopping rule (see `comrs.spec_blocks()` for the blocks and the
/// documentation of the Rust crate `com-spec` for the format).
///
/// Args:
///     spec (str | os.PathLike | dict): a YAML, JSON or TOML file, a YAML text, or the
///         specification as a dict.
///     threads (int, optional): worker threads (default: the specification, 0 = all).
///     batch (int, optional): trials between two checks of the stopping rule.
///     fast_dev_run (bool): one batch per condition, to check a specification.
///     progress (bool): print one line per condition on the standard error.
///     profile (bool, optional): time every block, reported as the metrics
///         ``<label>.time_ms`` and ``link.time_ms`` (``Results.profile`` tabulates them).
///     csv_path (str | os.PathLike, optional): CSV report (default: the
///         specification's).
///     json_path (str | os.PathLike, optional): JSON report (default: the
///         specification's).
///     base_dir (str | os.PathLike, optional): the directory the relative paths of the
///         blocks are resolved against (default: the directory of the specification
///         file, else the working directory).
///
/// Returns:
///     Results: one dict per condition with ``index``, ``condition``, ``status``,
///     ``trials``, ``wall_s`` and ``metrics`` (``{name: {value, lo, hi, kind, count,
///     total}}``), in a :class:`comrs.Results` (a ``list``) that keeps the
///     specification.
///
/// Raises:
///     ValueError: for an invalid specification; ArithmeticError for a failed trial.
///
/// Examples:
///     >>> import comrs
///     >>> res = comrs.run_spec('''
///     ... name: doc
///     ... grid: {ebn0_db: [4]}
///     ... link: [{type: source, constellation: QPSK, n_symbols: 1000},
///     ...        {type: awgn_ebn0, ebn0_db: $ebn0_db}]
///     ... metrics: {report: [ber]}
///     ... stop: {max_trials: 50}
///     ... ''')
///     >>> res[0]["trials"], res[0]["metrics"]["ber"]["kind"]
///     (64, 'rate')
#[pyfunction]
#[pyo3(signature = (spec, *, threads = None, batch = None, fast_dev_run = false, progress = false, profile = None, csv_path = None, json_path = None, base_dir = None))]
#[allow(clippy::too_many_arguments)]
fn run_spec<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyAny>,
    threads: Option<usize>,
    batch: Option<u64>,
    fast_dev_run: bool,
    progress: bool,
    profile: Option<bool>,
    csv_path: Option<PathBuf>,
    json_path: Option<PathBuf>,
    base_dir: Option<PathBuf>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut spec = load(py, spec)?;
    if base_dir.is_some() {
        spec.base_dir = base_dir;
    }
    let opts =
        RunOptions { threads, batch, fast_dev_run, progress: Some(progress), profile, csv: csv_path, json: json_path };
    let results = py.detach(|| run(&spec, &opts)).py()?;
    let items = results.iter().map(|r| result(py, r, spec.confidence)).collect::<PyResult<Vec<_>>>()?;
    results_of(py, PyList::new(py, items)?, &spec.source)
}

/// A list of condition dicts as a `comrs.Results` keeping the specification.
pub(crate) fn results_of<'py>(
    py: Python<'py>,
    items: Bound<'py, PyList>,
    spec: &serde_json::Value,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("comrs.results")?.getattr("Results")?.call1((items, from_json(py, spec)?))
}

/// A specification read by the Rust parser, as `spec.load` returns it, and the
/// directory of its file.
///
/// Args:
///     spec (str | os.PathLike | dict): as for `run_spec`.
///
/// Returns:
///     tuple[dict, os.PathLike | None]: the specification as parsed (checked), and the directory
///     that the relative paths of its blocks are resolved against (None for a text or a
///     dict).
///
/// Raises:
///     FileNotFoundError: for a string that is neither a file nor a specification.
///     ValueError: for an invalid specification.
#[pyfunction]
#[pyo3(name = "_load_spec")]
fn load_spec<'py>(py: Python<'py>, spec: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Option<PathBuf>)> {
    let spec = load(py, spec)?;
    Ok((from_json(py, &spec.source)?, spec.base_dir))
}

/// The conditions of a specification, in order, as dicts.
///
/// Args:
///     spec (str | os.PathLike | dict): as for `run_spec`.
///
/// Returns:
///     list[dict]: one dict per condition, the last grid axis varying fastest.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.spec_conditions({"name": "c", "grid": {"m": [4, 16], "snr_db": [0, 10]},
///     ...                        "link": ["source"]})
///     [{'m': 4, 'snr_db': 0}, {'m': 4, 'snr_db': 10}, {'m': 16, 'snr_db': 0}, {'m': 16, 'snr_db': 10}]
#[pyfunction]
fn spec_conditions<'py>(py: Python<'py>, spec: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let spec = load(py, spec)?;
    let items = spec.conditions().iter().map(|c| condition_dict(py, c)).collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// The blocks of the declarative links, their parameters and their ports.
///
/// Returns:
///     dict[str, dict]: for every block, its ``summary``, ``starts`` (it starts a link),
///     ``needs`` and ``gives`` (the facts of the frame it reads and may add: ``samples``,
///     ``symbols``, ``constellation``, ``noise``, ``llr``…) and its ``params``
///     (``{name: {default, doc, unit, choices, type, minimum, exclusive_minimum,
///     maximum, required}}``, ``type`` being a JSON Schema).
///
/// Examples:
///     >>> import comrs
///     >>> b = comrs.spec_blocks()["awgn_ebn0"]
///     >>> b["params"]["ebn0_db"]["required"], b["gives"]
///     (True, ['noise'])
#[pyfunction]
fn spec_blocks(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for b in blocks::registry() {
        let d = PyDict::new(py);
        d.set_item("summary", b.summary)?;
        d.set_item("starts", b.starts)?;
        d.set_item("needs", b.needs.iter().map(|f| f.name()).collect::<Vec<_>>())?;
        d.set_item("gives", b.gives.iter().map(|f| f.name()).collect::<Vec<_>>())?;
        let params = PyDict::new(py);
        for p in &b.params {
            let q = PyDict::new(py);
            q.set_item("default", p.default_text())?;
            q.set_item("doc", p.doc)?;
            q.set_item("unit", p.unit)?;
            q.set_item("choices", p.choices.to_vec())?;
            q.set_item("type", from_json(py, &p.schema)?)?;
            q.set_item("minimum", p.ge)?;
            q.set_item("exclusive_minimum", p.gt)?;
            q.set_item("maximum", p.le)?;
            q.set_item("required", matches!(p.default, blocks::ParamDefault::Required))?;
            params.set_item(p.name, q)?;
        }
        d.set_item("params", params)?;
        out.set_item(b.name, d)?;
    }
    Ok(out)
}

/// The JSON Schema of a block of a declarative link: one alternative per block, with
/// the type, default, unit, accepted values and bounds of every parameter, and the
/// ports of the block (``x-needs``, ``x-gives``), for editors and graphical front ends.
///
/// Returns:
///     dict: the schema (draft 2020-12).
///
/// Examples:
///     >>> import comrs
///     >>> s = comrs.spec_schema()
///     >>> awgn = next(a for a in s["oneOf"] if a.get("title") == "awgn")
///     >>> awgn["required"], awgn["x-gives"]
///     (['type', 'snr_db'], ['noise'])
#[pyfunction]
fn spec_schema(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    from_json(py, &blocks::json_schema())
}

/// Check the ports of a link of block types: every block finds the facts it needs
/// among those the blocks before it give.
///
/// Args:
///     kinds (list[str]): the block types, in the order of the link.
///
/// Raises:
///     ValueError: naming the first block that misses a fact and the blocks that give it.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.spec_check_link(["source", "awgn", "noise_estimate"])
///     >>> try:
///     ...     comrs.spec_check_link(["fec_encode", "map", "demap"])
///     ... except ValueError as e:
///     ...     print(str(e).split(":")[1].strip())
///     link block 3 (demap) needs noise, which no block before it gives
#[pyfunction]
fn spec_check_link(kinds: Vec<String>) -> PyResult<()> {
    blocks::check_link(&kinds.iter().map(String::as_str).collect::<Vec<_>>()).py()
}

/// Run a specification and check its expected results (closed forms, published curves).
///
/// Args:
///     spec (str | os.PathLike | dict): as for `run_spec`; it must have an ``expect``
///         list.
///     threads (int, optional): worker threads.
///     progress (bool): print one line per condition.
///
/// Returns:
///     dict: ``results`` (a list of dicts, as `run_spec`), ``checks`` (one dict per check with ``name``,
///     ``source``, ``condition``, ``metric``, ``expected``, ``value``, ``lo``, ``hi``,
///     ``passed``) and ``passed`` (all checks passed).
#[pyfunction]
#[pyo3(signature = (spec, *, threads = None, progress = false))]
fn verify_spec<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyAny>,
    threads: Option<usize>,
    progress: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = load(py, spec)?;
    if spec.expect.is_empty() {
        return Err(PyValueError::new_err("the specification has no expect list"));
    }
    let opts = RunOptions { threads, progress: Some(progress), ..RunOptions::default() };
    let out = PyDict::new(py);
    let checks = if spec.search.is_some() {
        let found = py.detach(|| search::search(&spec, &opts)).py()?;
        let items = found.iter().map(|r| search_result(py, r)).collect::<PyResult<Vec<_>>>()?;
        out.set_item("results", PyList::new(py, items)?)?;
        verify::verify_search(&spec.expect, &found).py()?
    } else {
        let results = py.detach(|| run(&spec, &opts)).py()?;
        let items = results.iter().map(|r| result(py, r, spec.confidence)).collect::<PyResult<Vec<_>>>()?;
        out.set_item("results", PyList::new(py, items)?)?;
        verify::verify(&spec.expect, &results, spec.confidence).py()?
    };
    let cs = PyList::empty(py);
    for c in &checks {
        let d = PyDict::new(py);
        d.set_item("name", &c.name)?;
        d.set_item("source", &c.source)?;
        d.set_item("condition", &c.condition)?;
        d.set_item("metric", &c.metric)?;
        d.set_item("expected", c.expected)?;
        d.set_item("value", c.measured.0)?;
        d.set_item("lo", c.measured.1)?;
        d.set_item("hi", c.measured.2)?;
        d.set_item("passed", c.passed)?;
        cs.append(d)?;
    }
    out.set_item("checks", cs)?;
    out.set_item("passed", checks.iter().all(|c| c.passed))?;
    Ok(out)
}

fn search_result<'py>(py: Python<'py>, r: &search::SearchResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let cond = PyDict::new(py);
    for (k, v) in r.condition.entries() {
        cond.set_item(k, value(py, v)?)?;
    }
    d.set_item("condition", cond)?;
    d.set_item("along", &r.along)?;
    d.set_item("metric", &r.metric)?;
    d.set_item("target", r.target)?;
    d.set_item("value", r.value)?;
    d.set_item("lo", r.lo)?;
    d.set_item("hi", r.hi)?;
    d.set_item("converged", r.converged)?;
    let evals = PyList::empty(py);
    for e in &r.evaluations {
        let q = PyDict::new(py);
        q.set_item("x", e.x)?;
        q.set_item("value", e.value)?;
        q.set_item("lo", e.lo)?;
        q.set_item("hi", e.hi)?;
        q.set_item("trials", e.trials)?;
        evals.append(q)?;
    }
    d.set_item("evaluations", evals)?;
    Ok(d)
}

/// Search the operating point of a specification: the value of a variable at which a
/// metric reaches a target, for every combination of the other grid axes.
///
/// The specification needs a ``search`` section (``along``, ``range``, ``metric``,
/// ``target``, and optionally ``tolerance``, ``log``, ``max_evaluations``). The bracket
/// is halved until it is narrower than the tolerance; every evaluation runs the stopping
/// rule of the specification on the same random streams (common random numbers); the
/// crossing is interpolated in ``log10`` of the metric, with a confidence interval from
/// those of the bracketing estimates.
///
/// Args:
///     spec (str | os.PathLike | dict): as for `run_spec`.
///     threads (int, optional): worker threads.
///     batch (int, optional): trials between two checks of the stopping rule.
///     progress (bool): print every evaluation on the standard error.
///     csv_path (str | os.PathLike, optional): CSV report (one row per combination).
///     json_path (str | os.PathLike, optional): JSON report (with every evaluation).
///
/// Returns:
///     list[dict]: per combination ``condition``, ``along``, ``metric``, ``target``,
///     ``value``, ``lo``, ``hi``, ``converged`` and ``evaluations`` (``x``, ``value``,
///     ``lo``, ``hi``, ``trials``).
///
/// Raises:
///     ValueError: without a search section, or when the target is not bracketed.
#[pyfunction]
#[pyo3(signature = (spec, *, threads = None, batch = None, progress = false, csv_path = None, json_path = None))]
fn search_spec<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyAny>,
    threads: Option<usize>,
    batch: Option<u64>,
    progress: bool,
    csv_path: Option<PathBuf>,
    json_path: Option<PathBuf>,
) -> PyResult<Bound<'py, PyList>> {
    let spec = load(py, spec)?;
    let opts = RunOptions {
        threads,
        batch,
        progress: Some(progress),
        csv: csv_path,
        json: json_path,
        ..RunOptions::default()
    };
    let found = py.detach(|| search::search(&spec, &opts)).py()?;
    let items = found.iter().map(|r| search_result(py, r)).collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// Read a SigMF recording (SigMF 1.x: every data type, little or big endian, several
/// channels, capture headers, trailing bytes, `.sigmf` archives).
///
/// Args:
///     path (str | os.PathLike): the `.sigmf-meta` or `.sigmf-data` file, their base
///         name, or a `.sigmf` archive.
///     channel (int | None): the channel to read; None: every channel (an array of
///         shape `(num_channels, n)` when there are several).
///     autoscale (bool): divide integer samples by their full scale `2^(b-1)` (the
///         values of `sigmf-python`); otherwise integers are converted without scaling
///         (unsigned types centred by `2^(b-1)`).
///
/// Returns:
///     tuple[numpy.ndarray, float | None, dict]: the samples (complex128), the sampling
///     rate `fs` (Hz) and the metadata (``meta["captures"]``,
///     ``meta["annotations"]``…).
///
/// Examples:
///     >>> import comrs, numpy as np, tempfile, os
///     >>> base = os.path.join(tempfile.mkdtemp(), "x")
///     >>> comrs.write_sigmf(base, np.array([1, 1j]), 1e6, frequency=2.4e9,
///     ...                   annotations=[{"sample_start": 1, "label": "QPSK"}])
///     >>> x, fs, meta = comrs.read_sigmf(base)
///     >>> meta["captures"][0]["core:frequency"], meta["annotations"][0]["core:label"]
///     (2400000000.0, 'QPSK')
#[pyfunction]
#[pyo3(signature = (path, *, channel = None, autoscale = false))]
fn read_sigmf(
    py: Python<'_>,
    path: PathBuf,
    channel: Option<usize>,
    autoscale: bool,
) -> PyResult<(Py<PyAny>, Option<f64>, Bound<'_, PyAny>)> {
    let r = comrs::io::read_sigmf(&path).py()?;
    let scale = if autoscale { r.full_scale().unwrap_or(1.0) } else { 1.0 };
    let get = |k: usize| r.channel(k).into_iter().map(|v| v / scale).collect::<Vec<_>>();
    let x = match channel {
        Some(k) if k >= r.num_channels => {
            return Err(PyValueError::new_err(format!("channel {k} of a recording of {}", r.num_channels)));
        }
        Some(k) => crate::carr(py, get(k)).into_any().unbind(),
        None if r.num_channels == 1 => crate::carr(py, get(0)).into_any().unbind(),
        None => crate::mat(py, (0..r.num_channels).map(get).collect())?.into_any().unbind(),
    };
    let meta = from_json(py, &r.meta)?;
    Ok((x, r.sample_rate, meta))
}

/// Write a SigMF recording `base.sigmf-meta` and `base.sigmf-data`.
///
/// Args:
///     base (str | os.PathLike): path without the extensions.
///     samples (array_like): complex samples.
///     fs (float): sampling rate (Hz).
///     datatype (str): the sample format, as the ``dump`` block: ``cf64`` (the
///         default), ``cf32`` or ``ci16``; every SigMF type is accepted
///         (``[c|r][f32|f64|i32|i16|i8|u32|u16|u8][_le|_be]``; integers are rounded and
///         saturated: scale the samples first).
///     frequency (float | None): the centre frequency of the capture (Hz).
///     annotations (list[dict] | None): annotations, with the keys ``sample_start``,
///         and optionally ``sample_count``, ``label``, ``comment``,
///         ``freq_lower_edge``, ``freq_upper_edge``.
///     global_meta (dict, optional): keys added to the ``global`` object of the
///         metadata (a description, the symbol rate, …).
///
/// Examples:
///     >>> import comrs, numpy as np, tempfile, os
///     >>> base = os.path.join(tempfile.mkdtemp(), "x")
///     >>> comrs.write_sigmf(base, np.array([1, 1j]), 1e6)
///     >>> x, fs, meta = comrs.read_sigmf(base)
///     >>> x.tolist(), fs, meta["global"]["core:datatype"]
///     ([(1+0j), 1j], 1000000.0, 'cf64_le')
#[pyfunction]
#[pyo3(signature = (base, samples, fs, *, datatype = "cf64", frequency = None, annotations = None, global_meta = None))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn write_sigmf(
    py: Python<'_>,
    base: PathBuf,
    samples: &Bound<'_, PyAny>,
    fs: f64,
    datatype: &str,
    frequency: Option<f64>,
    annotations: Option<&Bound<'_, PyList>>,
    global_meta: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let dt = comrs::io::Datatype::parse(datatype).py()?;
    let extra: Option<serde_json::Value> = match global_meta {
        Some(d) => Some(json(py, d)?),
        None => None,
    };
    let mut ann = Vec::new();
    for a in annotations.iter().flat_map(|l| l.iter()) {
        let a = json(py, &a)?;
        let known = ["sample_start", "sample_count", "label", "comment", "freq_lower_edge", "freq_upper_edge"];
        if let Some(k) = a.as_object().and_then(|o| o.keys().find(|k| !known.contains(&k.as_str()))) {
            return Err(PyValueError::new_err(format!("unknown annotation key {k:?} (one of {known:?})")));
        }
        let start = a["sample_start"]
            .as_u64()
            .ok_or_else(|| PyValueError::new_err("every annotation needs a non-negative integer sample_start"))?;
        ann.push(comrs::io::Annotation {
            sample_start: start as usize,
            sample_count: a["sample_count"].as_u64().map(|v| v as usize),
            label: a["label"].as_str().map(str::to_string),
            comment: a["comment"].as_str().map(str::to_string),
            freq_lower_edge: a["freq_lower_edge"].as_f64(),
            freq_upper_edge: a["freq_upper_edge"].as_f64(),
        });
    }
    comrs::io::write_sigmf_annotated(&base, &crate::cvec(samples)?, dt, fs, frequency, extra.as_ref(), &ann).py()
}

/// Register the declarative functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_spec, m)?)?;
    m.add_function(wrap_pyfunction!(load_spec, m)?)?;
    m.add_function(wrap_pyfunction!(spec_conditions, m)?)?;
    m.add_function(wrap_pyfunction!(spec_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(spec_schema, m)?)?;
    m.add_function(wrap_pyfunction!(spec_check_link, m)?)?;
    m.add_function(wrap_pyfunction!(verify_spec, m)?)?;
    m.add_function(wrap_pyfunction!(search_spec, m)?)?;
    m.add_function(wrap_pyfunction!(read_sigmf, m)?)?;
    m.add_function(wrap_pyfunction!(write_sigmf, m)?)?;
    Ok(())
}
