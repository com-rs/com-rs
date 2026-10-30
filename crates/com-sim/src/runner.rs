//! The runner: the `Trainer` of a Monte Carlo campaign.
//!
//! [`Runner::run`] takes a [`Scenario`] and a [`Grid`] and, for every condition in
//! order:
//!
//! 1. asks the callbacks whether to run it ([`Callback::on_condition_start`]);
//! 2. builds the state with [`Scenario::setup`];
//! 3. runs batches of `batch` trials in parallel, trial `t` drawing from
//!    `stream(seed, name, condition index, t)`, merges their records in trial order,
//!    and after every batch evaluates the stopping rule and calls
//!    [`Callback::on_batch_end`]; after the first batch it checks that the targets of
//!    the rule are recorded ([`Stop::check_metrics`]), and the last batch ends at the
//!    bound of the rule on the trials ([`Stop::trial_limit`]);
//! 4. hands the [`ConditionResult`] to the callbacks and the loggers.
//!
//! The results are identical whatever the number of threads (see [`crate`]). The batch
//! size only sets how often the rule is checked (a larger batch keeps every core busy, a
//! smaller one stops closer to an error or precision target); a bound on the trials is
//! met exactly, so that `max_trials(1000)` runs 1000 trials with batches of 32 as with
//! batches of 1.
//! Every condition runs at least one trial.
//!
//! # Examples
//! ```
//! use com_sim::prelude::*;
//! use com_mapping::{constellation::{Constellation, ConstellationKind}, source as sources};
//! use com_channel::awgn as channels;
//! use com_metrics as metrics;
//! use com_sim::scenario::from_fn;
//! let coin = from_fn("coin", |cond, rng, rec| {
//!     let p = cond.f64("p")?;
//!     rec.errors("heads", (rng.uniform() < p) as u64, 1);
//!     Ok(())
//! });
//! let grid = Grid::new().axis("p", [0.1, 0.5]);
//! let one = Runner::new().seed(3).threads(1).batch(50).stop(Stop::max_trials(1000))
//!     .run(&coin, &grid).unwrap();
//! let four = Runner::new().seed(3).threads(4).batch(50).stop(Stop::max_trials(1000))
//!     .run(&coin, &grid).unwrap();
//! assert_eq!(one[0].metric("heads"), four[0].metric("heads"));   // thread-count independent
//! assert!((one[1].metric("heads").unwrap().value() - 0.5).abs() < 0.05);
//! ```

use crate::callbacks::{Action, Callback};
use crate::grid::{Condition, Grid};
use crate::loggers::Logger;
use crate::scenario::{Record, Scenario};
use crate::stats::Estimate;
use crate::stopping::{Progress, Stop};
use com_core::rng::stream;
use com_core::{Error, Result};
use rayon::prelude::*;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How a condition ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// The stopping rule was met.
    Completed,
    /// A callback stopped it before the rule was met.
    Stopped,
    /// A callback skipped it: no trial was run.
    Skipped,
    /// [`Runner::fast_dev_run`]: one batch only.
    DevRun,
}

impl Status {
    /// `completed`, `stopped`, `skipped` or `dev_run`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Completed => "completed",
            Status::Stopped => "stopped",
            Status::Skipped => "skipped",
            Status::DevRun => "dev_run",
        }
    }
}

/// The result of one condition.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditionResult {
    /// Index of the condition in the grid.
    pub index: usize,
    /// The condition.
    pub condition: Condition,
    /// How it ended.
    pub status: Status,
    /// Number of trials run.
    pub trials: u64,
    /// Wall time of the condition, setup included (s).
    pub wall_s: f64,
    /// The metrics merged over the trials, with the counts of the values dropped
    /// ([`Record::dropped`]).
    pub record: Record,
}

impl ConditionResult {
    /// The estimate of a metric.
    pub fn metric(&self, name: &str) -> Option<&Estimate> {
        self.record.get(name)
    }
}

/// What the callbacks and loggers know about the run.
#[derive(Clone, Debug, PartialEq)]
pub struct RunInfo {
    /// Name of the scenario.
    pub scenario: String,
    /// Base seed.
    pub seed: u64,
    /// Trials per batch.
    pub batch: u64,
    /// Number of worker threads.
    pub threads: usize,
    /// The stopping rule.
    pub stop: String,
    /// Confidence level of the reported intervals.
    pub confidence: f64,
    /// Number of conditions of the grid.
    pub n_conditions: usize,
    /// Start of the run (seconds since the Unix epoch).
    pub started_unix_s: u64,
    /// Version of the library ([`crate::VERSION`]).
    pub version: String,
    /// Commit of the library ([`crate::COMMIT`]).
    pub commit: String,
    /// The configuration that produced the run, as a JSON document (a declarative
    /// specification sets it), if known.
    pub config: Option<String>,
}

/// The summary of a finished run.
#[derive(Clone, Debug, PartialEq)]
pub struct RunReport {
    /// The run information.
    pub info: RunInfo,
    /// Every condition, in order.
    pub results: Vec<ConditionResult>,
    /// Total wall time (s).
    pub wall_s: f64,
}

/// The campaign runner (builder; see the module documentation).
pub struct Runner {
    seed: u64,
    batch: u64,
    threads: usize,
    stop: Stop,
    confidence: f64,
    fast_dev_run: bool,
    config: Option<String>,
    callbacks: Vec<Box<dyn Callback>>,
    loggers: Vec<Box<dyn Logger>>,
}

impl Default for Runner {
    fn default() -> Self {
        Runner::new()
    }
}

impl Runner {
    /// Seed 0, batches of 32 trials, all cores, 100 trials per condition, 95 % intervals.
    pub fn new() -> Runner {
        Runner {
            seed: 0,
            batch: 32,
            threads: 0,
            stop: Stop::max_trials(100),
            confidence: 0.95,
            fast_dev_run: false,
            config: None,
            callbacks: Vec::new(),
            loggers: Vec::new(),
        }
    }

    /// The configuration that produced the run, a JSON document recorded in the run
    /// information (and the JSON report) next to the version, the commit and the seed.
    pub fn config(mut self, json: impl Into<String>) -> Runner {
        self.config = Some(json.into());
        self
    }

    /// The base seed of every stream.
    pub fn seed(mut self, seed: u64) -> Runner {
        self.seed = seed;
        self
    }

    /// Trials per batch (at least 1): the stopping rule is checked after every batch.
    pub fn batch(mut self, n: u64) -> Runner {
        self.batch = n.max(1);
        self
    }

    /// Worker threads (0: one per core).
    pub fn threads(mut self, n: usize) -> Runner {
        self.threads = n;
        self
    }

    /// The stopping rule of every condition.
    pub fn stop(mut self, rule: Stop) -> Runner {
        self.stop = rule;
        self
    }

    /// Confidence level of the intervals reported by the loggers (default 0.95), in
    /// $(0, 1)$ ([`Runner::run`] refuses another value).
    pub fn confidence(mut self, c: f64) -> Runner {
        self.confidence = c;
        self
    }

    /// Run one batch per condition and ignore the stopping rule (a quick check).
    pub fn fast_dev_run(mut self, on: bool) -> Runner {
        self.fast_dev_run = on;
        self
    }

    /// Add a callback.
    pub fn callback(mut self, c: impl Callback + 'static) -> Runner {
        self.callbacks.push(Box::new(c));
        self
    }

    /// Add a logger.
    pub fn logger(mut self, l: impl Logger + 'static) -> Runner {
        self.loggers.push(Box::new(l));
        self
    }

    /// Run the scenario on every condition of the grid.
    ///
    /// # Errors
    /// [`Error::Config`] for a confidence level outside $(0, 1)$ (of the runner or of a
    /// precision target, see [`check_confidence`](crate::stopping::check_confidence));
    /// for a stopping rule that can never stop, or whose target is a
    /// metric that the first batch of a condition does not record (or `min_errors` on a
    /// mean, see [`Stop::check_metrics`]); the first error of a `setup` or of a trial, of
    /// the same kind, with the condition and the trial in the message; the errors of the
    /// loggers.
    pub fn run<S: Scenario>(mut self, scenario: &S, grid: &Grid) -> Result<Vec<ConditionResult>> {
        crate::stopping::check_confidence(self.confidence, "the runner")?;
        self.stop.check_confidences()?;
        if !self.fast_dev_run && !self.stop.is_bounded() {
            return Err(Error::Config(format!(
                "the stopping rule `{}` can run forever: add Stop::max_trials or Stop::max_time with `or`",
                self.stop
            )));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .map_err(|e| Error::Config(format!("thread pool: {e}")))?;
        let conditions = grid.conditions();
        let info = RunInfo {
            scenario: scenario.name().to_string(),
            seed: self.seed,
            batch: self.batch,
            threads: pool.current_num_threads(),
            stop: if self.fast_dev_run { "fast_dev_run".into() } else { self.stop.to_string() },
            confidence: self.confidence,
            n_conditions: conditions.len(),
            started_unix_s: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
            version: crate::VERSION.into(),
            commit: crate::COMMIT.into(),
            config: self.config.clone(),
        };
        for c in &mut self.callbacks {
            c.on_run_start(&info);
        }
        for l in &mut self.loggers {
            l.start(&info)?;
        }
        let t_run = Instant::now();
        let mut results = Vec::with_capacity(conditions.len());
        let mut stop_run = false;
        for (index, cond) in conditions.into_iter().enumerate() {
            // every callback sees the start of the condition (no short circuit), unless
            // the run was stopped
            let mut skip = stop_run;
            if !stop_run {
                for c in &mut self.callbacks {
                    if c.on_condition_start(index, &cond) == Action::Skip {
                        skip = true;
                    }
                }
            }
            let res = if skip {
                ConditionResult {
                    index,
                    condition: cond,
                    status: Status::Skipped,
                    trials: 0,
                    wall_s: 0.0,
                    record: Record::new(),
                }
            } else {
                self.run_condition(scenario, index, cond, &pool)?
            };
            for c in &mut self.callbacks {
                if c.on_condition_end(&res) == Action::StopRun {
                    stop_run = true;
                }
            }
            for l in &mut self.loggers {
                l.log(&res, self.confidence)?;
            }
            results.push(res);
        }
        for c in &mut self.callbacks {
            c.on_run_end(&results);
        }
        let report = RunReport { info, results, wall_s: t_run.elapsed().as_secs_f64() };
        for l in &mut self.loggers {
            l.finish(&report)?;
        }
        Ok(report.results)
    }

    fn run_condition<S: Scenario>(
        &mut self,
        scenario: &S,
        index: usize,
        cond: Condition,
        pool: &rayon::ThreadPool,
    ) -> Result<ConditionResult> {
        let t0 = Instant::now();
        let state = scenario.setup(&cond).map_err(|e| e.context(format!("setup of ({})", cond.label())))?;
        let name = scenario.name().to_string();
        let mut record = Record::new();
        let mut trials = 0u64;
        // the last batch ends at the bound on the trials (one batch, whole, for a dev run)
        let limit = if self.fast_dev_run { None } else { self.stop.trial_limit() };
        let status = loop {
            let end = limit.map_or(trials + self.batch, |n| (trials + self.batch).min(n).max(trials + 1));
            let range = trials..end;
            let seed = self.seed;
            let recs: Vec<Result<Record>> = pool.install(|| {
                range
                    .into_par_iter()
                    .map(|t| {
                        let mut rng = stream(seed, &name, index as u64, t);
                        let mut rec = Record::for_trial(t);
                        scenario
                            .trial(&state, &cond, &mut rng, &mut rec)
                            .map(|()| rec)
                            .map_err(|e| e.context(format!("trial {t} of ({})", cond.label())))
                    })
                    .collect()
            });
            for r in recs {
                record.merge(&r?);
            }
            if trials == 0 {
                self.stop.check_metrics(&record).map_err(|e| e.context(format!("({})", cond.label())))?;
            }
            trials = end;
            let progress = Progress { condition: index, trials, record: &record, elapsed: t0.elapsed() };
            let mut stopped = false;
            for c in &mut self.callbacks {
                if c.on_batch_end(&progress) == Action::Skip {
                    stopped = true;
                }
            }
            if self.fast_dev_run {
                break Status::DevRun;
            }
            if self.stop.is_met(&progress) {
                break Status::Completed;
            }
            if stopped {
                break Status::Stopped;
            }
        };
        Ok(ConditionResult { index, condition: cond, status, trials, wall_s: t0.elapsed().as_secs_f64(), record })
    }
}

/// Wall time as a [`Duration`] from seconds (helper of the callbacks): zero for a
/// negative or NaN value, [`Duration::MAX`] for one too large (`∞`).
pub(crate) fn secs(s: f64) -> Duration {
    if s.is_nan() || s <= 0.0 { Duration::ZERO } else { Duration::try_from_secs_f64(s).unwrap_or(Duration::MAX) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::from_fn;

    fn coin() -> impl Scenario {
        from_fn("coin", |_, rng, rec| {
            rec.errors("heads", u64::from(rng.uniform() < 0.3), 1);
            rec.value("u", rng.uniform());
            Ok(())
        })
    }

    #[test]
    fn max_trials_is_met_exactly_whatever_the_batch_and_the_threads() {
        let grid = Grid::new().axis("x", [1usize]);
        let run = |batch: u64, threads: usize| {
            Runner::new()
                .seed(5)
                .batch(batch)
                .threads(threads)
                .stop(Stop::max_trials(1000))
                .run(&coin(), &grid)
                .unwrap()
        };
        let reference = run(1000, 1);
        for (batch, threads) in [(32, 1), (32, 4), (7, 3), (1, 2), (5000, 2)] {
            let r = run(batch, threads);
            assert_eq!(r[0].trials, 1000, "batch {batch}");
            assert_eq!(r[0].record, reference[0].record, "batch {batch}, {threads} threads");
        }
        // an error target stops at the end of a batch, the cap exactly
        let rule = Stop::min_errors("heads", 1_000_000).or(Stop::max_trials(70));
        let r = Runner::new().batch(32).stop(rule).run(&coin(), &grid).unwrap();
        assert_eq!(r[0].trials, 70);
        // at least one trial
        let r = Runner::new().batch(32).stop(Stop::max_trials(0)).run(&coin(), &grid).unwrap();
        assert_eq!(r[0].trials, 1);
    }

    #[test]
    fn a_stopping_target_that_is_not_recorded_is_an_error() {
        let grid = Grid::new().axis("x", [1usize, 2]);
        let rule = Stop::min_errors("ber", 100).or(Stop::max_trials(10_000));
        let e = Runner::new().batch(4).stop(rule).run(&coin(), &grid).unwrap_err().to_string();
        assert!(e.contains("`ber`") && e.contains("recorded: heads, u") && e.contains("x=1"), "{e}");
        let rule = Stop::min_errors("u", 100).or(Stop::max_trials(10_000));
        let e = Runner::new().batch(4).stop(rule).run(&coin(), &grid).unwrap_err().to_string();
        assert!(e.contains("is a mean"), "{e}");
        // a dev run (`comrs check`) checks it too
        let rule = Stop::precision("evm", 0.1).or(Stop::max_trials(10));
        assert!(Runner::new().fast_dev_run(true).stop(rule).run(&coin(), &grid).is_err());
    }

    #[test]
    fn a_confidence_outside_zero_one_is_an_error() {
        let grid = Grid::new();
        for c in [95.0, 0.0, 1.0, f64::NAN] {
            let e = Runner::new().confidence(c).stop(Stop::max_trials(1)).run(&coin(), &grid).unwrap_err();
            assert!(e.to_string().contains("between 0 and 1"), "{c}: {e}");
        }
        let rule = Stop::precision_at("u", 0.1, 95.0).or(Stop::max_trials(4));
        assert!(Runner::new().stop(rule).run(&coin(), &grid).is_err());
        let rule = Stop::precision_at("u", 1e-9, 0.5).or(Stop::max_trials(4));
        assert_eq!(Runner::new().batch(2).stop(rule).run(&coin(), &grid).unwrap()[0].trials, 4);
    }

    #[test]
    fn seconds_never_panic() {
        assert_eq!(secs(f64::INFINITY), Duration::MAX);
        assert_eq!(secs(f64::NAN), Duration::ZERO);
        assert_eq!(secs(-1.0), Duration::ZERO);
        assert_eq!(secs(1.5), Duration::from_millis(1500));
    }

    #[test]
    fn the_dropped_values_reach_the_result() {
        let s = from_fn("inf", |_, rng, rec| {
            rec.value("snr_db", if rng.uniform() < 0.5 { f64::INFINITY } else { 10.0 });
            Ok(())
        });
        let r = Runner::new().batch(8).stop(Stop::max_trials(64)).run(&s, &Grid::new()).unwrap();
        let kept = r[0].metric("snr_db").unwrap().count();
        assert!(kept > 0 && r[0].record.dropped("snr_db") > 0);
        assert_eq!(kept + r[0].record.dropped("snr_db"), 64);
    }
}
