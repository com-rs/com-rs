//! Loggers: where the results go.
//!
//! A [`Logger`] receives every [`ConditionResult`] as soon as the condition ends, and
//! the [`RunReport`] at the end of the run.
//!
//! - [`CsvLogger`] writes one row per condition **as it completes** (and flushes), so
//!   that a long campaign leaves usable partial results. The columns are those of
//!   [`ConditionResult::row`]: `index`, the keys of the condition, `status`, `trials`,
//!   `wall_s`, then for every metric `<m>`, `<m>_lo`, `<m>_hi` (the confidence
//!   interval) and `<m>_count`, `<m>_total` (errors and trials of a rate, or the number
//!   of values of a mean); those of the first condition that ran.
//! - [`JsonLogger`] writes the full report at the end of the run: the run information
//!   ([`run_header`]: scenario, seed, batch, threads, stopping rule, confidence, library
//!   version and commit, configuration, start time), the wall time, and every condition
//!   with its metrics
//!   and, under `dropped`, the number of values each metric dropped (non-finite or of
//!   the other kind, see [`Record::dropped`](crate::scenario::Record::dropped); an
//!   empty mapping when none was).
//!
//! # Examples
//! ```
//! use com_sim::prelude::*;
//! use com_mapping::{constellation::{Constellation, ConstellationKind}, source as sources};
//! use com_channel::awgn as channels;
//! use com_metrics as metrics;
//! use com_sim::scenario::from_fn;
//! let dir = std::env::temp_dir().join("comrs-doc-loggers");
//! std::fs::create_dir_all(&dir).unwrap();
//! let s = from_fn("toss", |_, rng, rec| { rec.errors("p", rng.bit() as u64, 1); Ok(()) });
//! Runner::new()
//!     .stop(Stop::max_trials(64))
//!     .logger(CsvLogger::new(dir.join("toss.csv")))
//!     .logger(JsonLogger::new(dir.join("toss.json")))
//!     .run(&s, &Grid::new().axis("n", [1usize, 2]))
//!     .unwrap();
//! let csv = std::fs::read_to_string(dir.join("toss.csv")).unwrap();
//! assert!(csv.starts_with("index,n,status,trials,wall_s,p,p_lo,p_hi,p_count,p_total\n"));
//! assert_eq!(csv.lines().count(), 3);
//! let json = std::fs::read_to_string(dir.join("toss.json")).unwrap();
//! assert!(json.contains("\"scenario\": \"toss\""));
//! ```

use crate::grid::Value;
use crate::json;
use crate::runner::{ConditionResult, RunInfo, RunReport};
use crate::stats::Estimate;
use com_core::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// A destination of the results.
pub trait Logger: Send {
    /// Before the first condition.
    ///
    /// # Errors
    /// [`com_core::Error::Io`] if the output cannot be created.
    fn start(&mut self, _info: &RunInfo) -> Result<()> {
        Ok(())
    }
    /// After every condition (`confidence`: level of the intervals).
    ///
    /// # Errors
    /// [`com_core::Error::Io`] if the output cannot be written.
    fn log(&mut self, result: &ConditionResult, confidence: f64) -> Result<()>;
    /// After the last condition.
    ///
    /// # Errors
    /// [`com_core::Error::Io`] if the output cannot be written.
    fn finish(&mut self, _report: &RunReport) -> Result<()> {
        Ok(())
    }
}

/// A CSV field (RFC 4180): `s` itself, or quoted with its quotes doubled when it holds
/// a comma, a quote or a line break.
///
/// # Examples
/// ```
/// use com_sim::loggers::csv_field;
/// assert_eq!(csv_field("snr_db"), "snr_db");
/// assert_eq!(csv_field("a, \"b\""), "\"a, \"\"b\"\"\"");
/// ```
pub fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

/// Create the parent directories of `path` (none for a bare file name).
///
/// # Errors
/// [`com_core::Error::Io`] if a directory cannot be created.
pub fn create_parent_dir(path: &Path) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}

/// One CSV row per condition ([`ConditionResult::row`]), written as the conditions
/// complete.
///
/// The header is written with the first condition that ran trials (skipped conditions
/// before it are held back and written right after the header). The values of the
/// condition are written as they are, the wall time to the microsecond, the estimates
/// and their intervals in scientific notation (empty when not finite), the counts as
/// integers.
pub struct CsvLogger {
    path: PathBuf,
    out: Option<BufWriter<File>>,
    header: bool,
    keys: Vec<String>,
    columns: Vec<String>,
    pending: Vec<(ConditionResult, f64)>,
}

impl CsvLogger {
    /// A logger writing to `path` (created or truncated at the start of the run; the
    /// parent directories are created).
    pub fn new(path: impl Into<PathBuf>) -> CsvLogger {
        CsvLogger {
            path: path.into(),
            out: None,
            header: false,
            keys: Vec::new(),
            columns: Vec::new(),
            pending: Vec::new(),
        }
    }

    fn write_header(&mut self, r: &ConditionResult, confidence: f64) -> Result<()> {
        self.keys = r.condition.keys().map(str::to_string).collect();
        self.columns = r.row(confidence).into_iter().map(|(k, _)| k).collect();
        let header: Vec<String> = self.columns.iter().map(|k| csv_field(k)).collect();
        if let Some(out) = self.out.as_mut() {
            writeln!(out, "{}", header.join(","))?;
        }
        self.header = true;
        Ok(())
    }

    /// A cell of the row: a value of the condition or of the head as it is, the wall
    /// time to the microsecond, an estimate or a bound in scientific notation.
    fn cell(column: &str, v: &Value, metric: bool) -> String {
        match v {
            Value::F64(x) if !metric && column == "wall_s" => format!("{x:.6}"),
            Value::F64(x) if metric && x.is_finite() => format!("{x:e}"),
            Value::F64(_) if metric => String::new(),
            other => csv_field(&other.to_string()),
        }
    }

    fn write_row(&mut self, r: &ConditionResult, confidence: f64) -> Result<()> {
        let cells = r.row(confidence);
        // the head of the row (index, condition, status, trials, wall_s) by position, the
        // condition by its keys and the metrics by their names (a metric may have the
        // name of a variable: snr_db)
        let (head, metrics) = cells.split_at(4 + r.condition.entries().len());
        let (columns_head, columns_metrics) = self.columns.split_at(4 + self.keys.len());
        let condition = &head[1..head.len() - 3];
        let mut row = vec![Self::cell("index", &head[0].1, false)];
        row.extend(
            self.keys.iter().map(|k| {
                condition.iter().find(|(c, _)| c == k).map(|(_, v)| Self::cell(k, v, false)).unwrap_or_default()
            }),
        );
        row.extend(head[head.len() - 3..].iter().map(|(k, v)| Self::cell(k, v, false)));
        debug_assert_eq!(row.len(), columns_head.len());
        row.extend(
            columns_metrics
                .iter()
                .map(|c| metrics.iter().find(|(k, _)| k == c).map(|(_, v)| Self::cell(c, v, true)).unwrap_or_default()),
        );
        if let Some(out) = self.out.as_mut() {
            writeln!(out, "{}", row.join(","))?;
            out.flush()?;
        }
        Ok(())
    }
}

impl Logger for CsvLogger {
    fn start(&mut self, _: &RunInfo) -> Result<()> {
        create_parent_dir(&self.path)?;
        self.out = Some(BufWriter::new(File::create(&self.path)?));
        self.header = false;
        self.pending.clear();
        Ok(())
    }

    fn log(&mut self, r: &ConditionResult, confidence: f64) -> Result<()> {
        if !self.header {
            if r.trials == 0 {
                self.pending.push((r.clone(), confidence));
                return Ok(());
            }
            self.write_header(r, confidence)?;
            for (p, c) in std::mem::take(&mut self.pending) {
                self.write_row(&p, c)?;
            }
        }
        self.write_row(r, confidence)
    }

    fn finish(&mut self, _: &RunReport) -> Result<()> {
        if !self.header {
            if let Some((first, c)) = self.pending.first().cloned() {
                self.write_header(&first, c)?;
            }
            for (p, c) in std::mem::take(&mut self.pending) {
                self.write_row(&p, c)?;
            }
        }
        if let Some(out) = self.out.as_mut() {
            out.flush()?;
        }
        Ok(())
    }
}

/// The full report as JSON, written at the end of the run.
pub struct JsonLogger {
    path: PathBuf,
}

impl JsonLogger {
    /// A logger writing to `path` at the end of the run.
    pub fn new(path: impl Into<PathBuf>) -> JsonLogger {
        JsonLogger { path: path.into() }
    }
}

/// A value of a condition as JSON (a number, a string or a boolean).
///
/// # Examples
/// ```
/// use com_sim::grid::Value;
/// assert_eq!(com_sim::loggers::value_json(&Value::Str("qam".into())), "\"qam\"");
/// assert_eq!(com_sim::loggers::value_json(&Value::F64(f64::NAN)), "null");
/// ```
pub fn value_json(v: &Value) -> String {
    match v {
        Value::F64(x) => json::number(*x),
        Value::I64(x) => x.to_string(),
        Value::Str(s) => json::string(s),
        Value::Bool(b) => b.to_string(),
    }
}

fn estimate_json(e: &Estimate, confidence: f64) -> String {
    let (lo, hi) = e.interval(confidence);
    let mut f = vec![("value", json::number(e.value())), ("lo", json::number(lo)), ("hi", json::number(hi))];
    match e {
        Estimate::Rate(r) => {
            f.push(("kind", json::string("rate")));
            f.push(("errors", r.errors.to_string()));
            f.push(("total", r.total.to_string()));
        }
        Estimate::Mean(m) => {
            f.push(("kind", json::string("mean")));
            f.push(("count", m.count().to_string()));
            f.push(("std", json::number(m.std())));
        }
    }
    json::object(&f)
}

/// The fields of the run information that head every JSON report (of a run, of a search
/// and of a verification), encoded: `scenario`, `seed`, `batch`, `threads`, `stop`,
/// `confidence`, `version`, `commit`, `config` (the configuration as a JSON document,
/// `null` if unknown) and `started_unix_s`.
pub fn run_header(i: &RunInfo) -> Vec<(&'static str, String)> {
    vec![
        ("scenario", json::string(&i.scenario)),
        ("seed", i.seed.to_string()),
        ("batch", i.batch.to_string()),
        ("threads", i.threads.to_string()),
        ("stop", json::string(&i.stop)),
        ("confidence", json::number(i.confidence)),
        ("version", json::string(&i.version)),
        ("commit", json::string(&i.commit)),
        ("config", i.config.clone().unwrap_or_else(|| "null".into())),
        ("started_unix_s", i.started_unix_s.to_string()),
    ]
}

/// Write a JSON document to `path`, its parent directories created.
///
/// # Errors
/// [`com_core::Error::Io`] if the file cannot be written.
pub fn write_json(path: &Path, doc: &str) -> Result<()> {
    create_parent_dir(path)?;
    std::fs::write(path, format!("{doc}\n"))?;
    Ok(())
}

impl Logger for JsonLogger {
    fn log(&mut self, _: &ConditionResult, _: f64) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self, rep: &RunReport) -> Result<()> {
        let i = &rep.info;
        let conditions: Vec<String> = rep
            .results
            .iter()
            .map(|r| {
                let cond: Vec<(&str, String)> =
                    r.condition.entries().iter().map(|(k, v)| (k.as_str(), value_json(v))).collect();
                let metrics: Vec<(&str, String)> =
                    r.record.iter().map(|(k, e)| (k.as_str(), estimate_json(e, i.confidence))).collect();
                let dropped: Vec<(&str, String)> =
                    r.record.dropped_iter().map(|(k, n)| (k.as_str(), n.to_string())).collect();
                json::object(&[
                    ("index", r.index.to_string()),
                    ("condition", json::object(&cond)),
                    ("status", json::string(r.status.as_str())),
                    ("trials", r.trials.to_string()),
                    ("wall_s", json::number(r.wall_s)),
                    ("metrics", json::object(&metrics)),
                    ("dropped", json::object(&dropped)),
                ])
            })
            .collect();
        let mut fields = run_header(i);
        fields.push(("wall_s", json::number(rep.wall_s)));
        fields.push(("conditions", json::array(&conditions)));
        write_json(&self.path, &json::object(&fields))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use crate::scenario::from_fn;

    #[test]
    fn the_json_report_lists_the_dropped_values() {
        let dir = std::env::temp_dir().join(format!("comrs-dropped-{}", std::process::id()));
        let path = dir.join("r.json");
        let s = from_fn("drops", |_, _, rec| {
            rec.value("snr_db", f64::INFINITY);
            rec.value("evm", 0.1);
            Ok(())
        });
        Runner::new().batch(4).stop(Stop::max_trials(6)).logger(JsonLogger::new(&path)).run(&s, &Grid::new()).unwrap();
        let json = std::fs::read_to_string(&path).unwrap();
        assert!(json.contains("\"dropped\": {\"snr_db\": 6}"), "{json}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
