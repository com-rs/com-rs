//! Callbacks: hooks into the run.
//!
//! A [`Callback`] is called at the start and the end of the run, of every condition and
//! after every batch of trials. It observes (progress display, monitoring) or steers
//! the run through the [`Action`] it returns: skip a condition before it starts, stop a
//! condition early, or stop the whole run after a condition. Every method has a default
//! that does nothing, so a callback implements only the hooks it needs.
//!
//! Built-in callbacks:
//!
//! - [`ProgressBar`]: one line per condition on the standard error (trials, wall time,
//!   the estimate and its interval for every metric), and a periodic progress line on
//!   long conditions;
//! - [`SkipAfter`]: the classical shortcut of error-rate curves: once an error rate
//!   falls below a floor along an axis (for instance the BER along an SNR sweep), the
//!   following points of the same sweep are skipped, since they would need too many
//!   trials and bring no information.
//!
//! # Examples
//! ```
//! use com_sim::prelude::*;
//! use com_mapping::{constellation::{Constellation, ConstellationKind}, source as sources};
//! use com_channel::awgn as channels;
//! use com_metrics as metrics;
//! use com_sim::scenario::from_fn;
//! // error probability 1, 1e-2, 1e-4, 1e-6: the point at 30 dB is skipped once the
//! // estimate at 20 dB falls below 1e-3
//! let s = from_fn("decay", |cond, rng, rec| {
//!     let p = 10f64.powf(-cond.f64("snr_db")? / 5.0);
//!     rec.errors("ber", (rng.uniform() < p) as u64, 1);
//!     Ok(())
//! });
//! let grid = Grid::new().axis("snr_db", [0.0, 10.0, 20.0, 30.0]);
//! let r = Runner::new()
//!     .stop(Stop::max_trials(2000))
//!     .callback(SkipAfter::below("ber", 1e-3, "snr_db"))
//!     .run(&s, &grid)
//!     .unwrap();
//! let status: Vec<&str> = r.iter().map(|c| c.status.as_str()).collect();
//! assert_eq!(status, ["completed", "completed", "completed", "skipped"]);
//! ```

use crate::grid::{Condition, Value};
use crate::runner::{ConditionResult, RunInfo, secs};
use crate::stopping::Progress;
use std::io::Write;
use std::time::Instant;

/// What a callback asks the runner to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Go on.
    Continue,
    /// At the start of a condition: do not run it; after a batch: stop the condition.
    Skip,
    /// After a condition: run no further condition.
    StopRun,
}

/// Hooks into a run (all optional).
pub trait Callback: Send {
    /// Before the first condition.
    fn on_run_start(&mut self, _info: &RunInfo) {}
    /// Before a condition: [`Action::Skip`] does not run it.
    fn on_condition_start(&mut self, _index: usize, _condition: &Condition) -> Action {
        Action::Continue
    }
    /// After every batch of trials: [`Action::Skip`] stops the condition.
    fn on_batch_end(&mut self, _progress: &Progress<'_>) -> Action {
        Action::Continue
    }
    /// After a condition (run or skipped): [`Action::StopRun`] ends the run.
    fn on_condition_end(&mut self, _result: &ConditionResult) -> Action {
        Action::Continue
    }
    /// After the last condition.
    fn on_run_end(&mut self, _results: &[ConditionResult]) {}
}

/// Progress lines on the standard error.
#[derive(Clone, Debug)]
pub struct ProgressBar {
    /// Seconds between two progress lines within a long condition.
    pub every_s: f64,
    /// Confidence of the printed intervals.
    pub confidence: f64,
    n: usize,
    last: Option<Instant>,
}

impl Default for ProgressBar {
    fn default() -> Self {
        ProgressBar { every_s: 5.0, confidence: 0.95, n: 0, last: None }
    }
}

impl ProgressBar {
    /// A progress display with a line every `every_s` seconds within a condition.
    pub fn new(every_s: f64) -> ProgressBar {
        ProgressBar { every_s, ..ProgressBar::default() }
    }
}

impl Callback for ProgressBar {
    fn on_run_start(&mut self, info: &RunInfo) {
        self.n = info.n_conditions;
        eprintln!(
            "[{}] {} conditions, seed {}, batches of {}, {} threads, stop: {}",
            info.scenario, info.n_conditions, info.seed, info.batch, info.threads, info.stop
        );
    }

    fn on_condition_start(&mut self, _: usize, _: &Condition) -> crate::callbacks::Action {
        self.last = Some(Instant::now());
        Action::Continue
    }

    fn on_batch_end(&mut self, p: &Progress<'_>) -> Action {
        let now = Instant::now();
        if self.last.is_some_and(|t| now.duration_since(t) >= secs(self.every_s)) {
            self.last = Some(now);
            eprint!(
                "\r  condition {}/{}: {} trials, {:.0} s",
                p.condition + 1,
                self.n,
                p.trials,
                p.elapsed.as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }
        Action::Continue
    }

    fn on_condition_end(&mut self, r: &ConditionResult) -> Action {
        let metrics: Vec<String> = r
            .record
            .iter()
            .map(|(k, e)| {
                let (lo, hi) = e.interval(self.confidence);
                format!("{k} = {:.4e} [{:.2e}, {:.2e}]", e.value(), lo, hi)
            })
            .collect();
        eprintln!(
            "\r[{}/{}] ({}) {}: {} trials, {:.2} s{}{}",
            r.index + 1,
            self.n,
            r.condition.label(),
            r.status.as_str(),
            r.trials,
            r.wall_s,
            if metrics.is_empty() { "" } else { "  " },
            metrics.join(", ")
        );
        Action::Continue
    }
}

/// Skip the rest of a sweep once an error rate is below a floor.
///
/// After a condition whose metric is below `threshold` (with at least one trial), the
/// later conditions that have the same values on every key except `along` are skipped.
/// The grid should list the values of `along` in the direction of decreasing error
/// rates (increasing SNR), which is the case when `along` is its last axis.
#[derive(Clone, Debug)]
pub struct SkipAfter {
    /// The error-rate metric.
    pub metric: String,
    /// The floor.
    pub threshold: f64,
    /// The swept key.
    pub along: String,
    done: Vec<Vec<(String, Value)>>,
}

impl SkipAfter {
    /// Skip along `along` once `metric` is below `threshold`.
    pub fn below(metric: &str, threshold: f64, along: &str) -> SkipAfter {
        SkipAfter { metric: metric.to_string(), threshold, along: along.to_string(), done: Vec::new() }
    }

    fn others(&self, c: &Condition) -> Vec<(String, Value)> {
        c.entries().iter().filter(|(k, _)| *k != self.along).cloned().collect()
    }
}

impl Callback for SkipAfter {
    fn on_condition_start(&mut self, _: usize, c: &Condition) -> Action {
        if self.done.contains(&self.others(c)) { Action::Skip } else { Action::Continue }
    }

    fn on_condition_end(&mut self, r: &ConditionResult) -> Action {
        if r.trials > 0 && r.metric(&self.metric).is_some_and(|e| e.value() < self.threshold) {
            let key = self.others(&r.condition);
            if !self.done.contains(&key) {
                self.done.push(key);
            }
        }
        Action::Continue
    }
}
