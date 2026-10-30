//! Running a specification ([`run`]) and the profile of its blocks ([`profile`]).

use crate::{Spec, SpecScenario};
use com_core::Result;
use com_sim::callbacks::ProgressBar;
use com_sim::loggers::{CsvLogger, JsonLogger};
use com_sim::runner::{ConditionResult, Runner};
use std::path::PathBuf;

/// Options of [`run`] that override the specification.
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Worker threads.
    pub threads: Option<usize>,
    /// Trials per batch.
    pub batch: Option<u64>,
    /// One batch per condition.
    pub fast_dev_run: bool,
    /// Print one line per condition.
    pub progress: Option<bool>,
    /// Time every block (the metrics `<label>.time_ms`, see [`profile`]).
    pub profile: Option<bool>,
    /// CSV output.
    pub csv: Option<PathBuf>,
    /// JSON output.
    pub json: Option<PathBuf>,
}

impl Spec {
    /// The runner of the specification without its callbacks and loggers: its seed,
    /// batch, threads, stopping rule, confidence and source (the configuration of the
    /// JSON report), the options overriding the batch and the threads, and
    /// `fast_dev_run` (what [`search`](crate::search::search) evaluates every point with).
    pub(crate) fn bare_runner(&self, opts: &RunOptions) -> Runner {
        Runner::new()
            .seed(self.seed)
            .batch(opts.batch.unwrap_or(self.batch))
            .threads(opts.threads.unwrap_or(self.threads))
            .stop(self.stop.clone())
            .confidence(self.confidence)
            .config(self.source.to_string())
            .fast_dev_run(opts.fast_dev_run)
    }

    /// The runner of the specification: its seed, batch, threads, stopping rule,
    /// confidence and source, its callbacks (the progress lines, `skip_below`) and its
    /// loggers (the CSV and JSON reports), each overridden by `opts` when it gives one.
    /// [`run`] runs it on the scenario of the specification; a front end that runs the
    /// trials otherwise (the Python chains with Python blocks) runs it on its own
    /// scenario.
    ///
    /// # Examples
    /// ```
    /// use com_spec::{RunOptions, Spec, SpecScenario};
    /// let spec = Spec::from_yaml("name: t\nthreads: 2\nlink: [source]\nstop: {max_trials: 3}")?;
    /// let opts = RunOptions { threads: Some(1), ..RunOptions::default() };
    /// let r = spec.runner(&opts).run(&SpecScenario::new(&spec), &spec.grid)?;
    /// assert_eq!(r[0].trials, 3);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn runner(&self, opts: &RunOptions) -> Runner {
        let mut r = self.bare_runner(opts);
        if opts.progress.unwrap_or(self.progress) {
            let mut bar = ProgressBar::default();
            bar.confidence = self.confidence;
            r = r.callback(bar);
        }
        if let Some(skip) = &self.skip_below {
            r = r.callback(skip.clone());
        }
        if let Some(p) = opts.csv.as_ref().or(self.csv.as_ref()) {
            r = r.logger(CsvLogger::new(p));
        }
        if let Some(p) = opts.json.as_ref().or(self.json.as_ref()) {
            r = r.logger(JsonLogger::new(p));
        }
        r
    }
}

/// Run a specification: its runner ([`Spec::runner`]) on its scenario
/// ([`SpecScenario`]).
///
/// # Errors
/// The errors of the blocks, of the stopping rule (an unbounded rule) and of the
/// loggers.
pub fn run(spec: &Spec, opts: &RunOptions) -> Result<Vec<ConditionResult>> {
    spec.runner(opts).run(&SpecScenario::new(spec).profile(opts.profile.unwrap_or(spec.profile)), &spec.grid)
}

/// One block of a [`profile`]: its mean time per trial and its share of the link.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileRow {
    /// The label of the block (`link` for the whole link).
    pub label: String,
    /// Mean time per trial (ms), averaged over the conditions that report it.
    pub time_ms: f64,
    /// Share of the time of the link, in `[0, 1]`.
    pub share: f64,
}

/// Where the time goes: the blocks of a campaign run with `profile`, in the order of
/// the link, from the metrics `<label>.time_ms` of the results (an empty list when the
/// campaign was not profiled). `labels` gives the order, [`Link::labels`](crate::Link::labels); the last row
/// is the whole link.
pub fn profile(results: &[ConditionResult], labels: &[String]) -> Vec<ProfileRow> {
    let mean = |name: &str| {
        let v: Vec<f64> =
            results.iter().filter_map(|r| r.record.get(name).map(com_sim::stats::Estimate::value)).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let Some(total) = mean("link.time_ms") else { return Vec::new() };
    let mut rows: Vec<ProfileRow> = labels
        .iter()
        .filter_map(|l| {
            let time_ms = mean(&format!("{l}.time_ms"))?;
            Some(ProfileRow { label: l.clone(), time_ms, share: if total > 0.0 { time_ms / total } else { 0.0 } })
        })
        .collect();
    rows.push(ProfileRow { label: "link".into(), time_ms: total, share: 1.0 });
    rows
}

/// The profile of a specification's results: the labels of its blocks come from its
/// first condition ([`link_labels`](crate::link_labels), without building the link).
///
/// # Errors
/// [`com_core::Error::Config`] for a variable that the condition does not define or a
/// block that is neither a name nor a mapping with a type.
pub fn profile_of(spec: &Spec, results: &[ConditionResult]) -> Result<Vec<ProfileRow>> {
    let Some(cond) = spec.conditions().into_iter().next() else { return Ok(Vec::new()) };
    let link = spec.link.iter().map(|b| crate::substitute(b, &cond)).collect::<Result<Vec<_>>>()?;
    Ok(profile(results, &crate::link_labels(&link)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::Frame;
    use crate::io;
    use com_core::rng::stream;

    #[test]
    fn profile_times_every_block_in_the_order_of_the_link() {
        let spec = Spec::from_yaml(
            "name: prof\ngrid: {ebn0_db: [4, 8]}\nlink: [{type: source, constellation: QPSK, n_symbols: 512}, {type: pulse_shape, sps: 2}, {type: awgn_ebn0, ebn0_db: $ebn0_db}, {type: matched_filter}, {type: decimate}]\nmetrics: {report: [ber], skip_start: 16, skip_end: 16}\nstop: {max_trials: 4}\nprofile: true",
        )
        .unwrap();
        let results = run(&spec, &RunOptions::default()).unwrap();
        let names: Vec<&String> =
            results[0].record.iter().map(|(k, _)| k).filter(|k| k.ends_with(".time_ms")).collect();
        assert_eq!(names.len(), 6, "{names:?}");
        let rows = profile_of(&spec, &results).unwrap();
        let labels: Vec<&str> = rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["source", "pulse_shape", "awgn_ebn0", "matched_filter", "decimate", "link"]);
        assert!(rows.iter().all(|r| r.time_ms >= 0.0 && r.share >= 0.0 && r.share <= 1.0));
        assert!((rows.last().unwrap().share - 1.0).abs() < 1e-12);
        // not profiled: no timing metric, an empty table
        let spec =
            Spec::from_yaml("name: plain\nlink: [{type: source, n_symbols: 64}]\nstop: {max_trials: 1}").unwrap();
        let results = run(&spec, &RunOptions::default()).unwrap();
        assert!(results[0].record.iter().all(|(k, _)| !k.ends_with(".time_ms")));
        assert_eq!(profile_of(&spec, &results).unwrap(), Vec::<ProfileRow>::new());
        // the option overrides the specification
        let results = run(&spec, &RunOptions { profile: Some(true), ..RunOptions::default() }).unwrap();
        assert!(results[0].record.iter().any(|(k, _)| k == "link.time_ms"));
    }

    #[test]
    fn dump_saves_the_first_trials_of_every_condition() {
        let dir = std::env::temp_dir().join(format!("comrs-dump-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let spec = Spec::from_yaml(&format!(
            "name: dumps\ngrid: {{snr_db: [5, 10]}}\nbatch: 8\nlink: [{{type: source, constellation: QPSK, n_symbols: 128, symbol_rate: 1e6}}, {{type: pulse_shape, sps: 4}}, {{type: dump, dir: {}, trials: 2, label: tap}}, {{type: awgn, snr_db: $snr_db}}, {{type: matched_filter}}, {{type: decimate}}]\nstop: {{max_trials: 8}}",
            dir.display()
        ))
        .unwrap();
        run(&spec, &RunOptions::default()).unwrap();
        let cond = dir.join("dumps").join("snr_db=10");
        let mut files: Vec<String> =
            std::fs::read_dir(&cond).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        files.sort();
        assert_eq!(files, ["tap_0.sigmf-data", "tap_0.sigmf-meta", "tap_1.sigmf-data", "tap_1.sigmf-meta"]);
        let rec = io::read_sigmf(cond.join("tap_0.sigmf-meta")).unwrap();
        assert_eq!(rec.samples.len(), 512);
        assert_eq!(rec.sample_rate, Some(4e6));
        assert_eq!(rec.meta["global"]["comrs:sps"], 4.0);
        assert_eq!(rec.meta["global"]["comrs:block"], "tap");
        // trial 0 (by index, whatever the order the threads ran them in) is
        // reproducible from its stream
        let link = spec.link_for(&spec.conditions()[1]).unwrap();
        let mut rng = stream(spec.seed, &spec.name, 1, 0);
        let mut s = Frame::default();
        for b in &link.blocks[..2] {
            b.apply(&mut s, &mut rng).unwrap();
        }
        assert_eq!(s.samples, rec.samples);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
