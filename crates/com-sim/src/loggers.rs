//! Loggers: where the results go.
//!
//! A [`Logger`] receives every [`ConditionResult`] as soon as the condition ends, and
//! the [`RunReport`] at the end of the run.
//!
//! - [`CsvLogger`] writes one row per condition **as it completes** (and flushes), so
//!   that a long campaign leaves usable partial results. Columns: `index`, the keys of
//!   the condition, `status`, `trials`, `wall_s`, then for every metric `<m>`,
//!   `<m>_lo`, `<m>_hi` (the confidence interval) and `<m>_count`, `<m>_total` (errors and
//!   trials of a rate, or the number of values of a mean). The columns are those of the
//!   first condition that ran.
//! - [`JsonLogger`] writes the full report at the end of the run: the run information
//!   (scenario, seed, batch, threads, stopping rule, confidence, library version and
//!   commit, configuration, start time, wall time) and every condition with its metrics
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

/// One CSV row per condition, written as the conditions complete.
///
/// The header is written with the first condition that ran trials (skipped conditions
/// before it are held back and written right after the header).
pub struct CsvLogger {
    path: PathBuf,
    out: Option<BufWriter<File>>,
    header: bool,
    keys: Vec<String>,
    metrics: Vec<String>,
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
            metrics: Vec::new(),
            pending: Vec::new(),
        }
    }

    fn write_header(&mut self, r: &ConditionResult) -> Result<()> {
        self.keys = r.condition.keys().map(str::to_string).collect();
        self.metrics = r.record.iter().map(|(k, _)| k.clone()).collect();
        let mut header = vec!["index".to_string()];
        header.extend(self.keys.iter().map(|k| csv_field(k)));
        header.extend(["status", "trials", "wall_s"].map(String::from));
        for m in &self.metrics {
            for suffix in ["", "_lo", "_hi", "_count", "_total"] {
                header.push(csv_field(&format!("{m}{suffix}")));
            }
        }
        if let Some(out) = self.out.as_mut() {
            writeln!(out, "{}", header.join(","))?;
        }
        self.header = true;
        Ok(())
    }

    fn write_row(&mut self, r: &ConditionResult, confidence: f64) -> Result<()> {
        let mut row = vec![r.index.to_string()];
        for k in &self.keys {
            row.push(r.condition.get(k).map(|v| csv_field(&v.to_string())).unwrap_or_default());
        }
        row.extend([r.status.as_str().to_string(), r.trials.to_string(), format!("{:.6}", r.wall_s)]);
        for m in &self.metrics {
            match r.metric(m) {
                Some(e) => {
                    let (lo, hi) = e.interval(confidence);
                    row.extend(
                        [e.value(), lo, hi].iter().map(
                            |v| {
                                if v.is_finite() { format!("{v:e}") } else { String::new() }
                            },
                        ),
                    );
                    row.extend([e.count().to_string(), e.total().to_string()]);
                }
                None => row.extend(std::iter::repeat_n(String::new(), 5)),
            }
        }
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
            self.write_header(r)?;
            for (p, c) in std::mem::take(&mut self.pending) {
                self.write_row(&p, c)?;
            }
        }
        self.write_row(r, confidence)
    }

    fn finish(&mut self, _: &RunReport) -> Result<()> {
        if !self.header {
            if let Some((first, _)) = self.pending.first().cloned() {
                self.write_header(&first)?;
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

fn value_json(v: &Value) -> String {
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
        let doc = json::object(&[
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
            ("wall_s", json::number(rep.wall_s)),
            ("conditions", json::array(&conditions)),
        ]);
        if let Some(dir) = self.path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir)?;
            }
        }
        std::fs::write(&self.path, doc + "\n")?;
        Ok(())
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
