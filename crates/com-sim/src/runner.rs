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
use crate::grid::{Condition, Grid, Value};
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

    /// The result as one flat row of named values: `index`, the entries of the
    /// condition, `status`, `trials`, `wall_s`, then the columns of every metric
    /// ([`metric_columns`]) with its interval at `confidence`, in the order of the
    /// names: the columns of the CSV report ([`CsvLogger`](crate::loggers::CsvLogger))
    /// and the rows of the Python results.
    ///
    /// # Examples
    /// ```
    /// use com_sim::prelude::*;
    /// use com_sim::scenario::from_fn;
    /// let s = from_fn("coin", |_, rng, rec| { rec.errors("p", rng.bit() as u64, 1); Ok(()) });
    /// let r = Runner::new().stop(Stop::max_trials(8)).run(&s, &Grid::new().axis("n", [1usize])).unwrap();
    /// let names: Vec<String> = r[0].row(0.95).into_iter().map(|(k, _)| k).collect();
    /// assert_eq!(names, ["index", "n", "status", "trials", "wall_s", "p", "p_lo", "p_hi", "p_count", "p_total"]);
    /// ```
    pub fn row(&self, confidence: f64) -> Vec<(String, Value)> {
        let count = |n: u64| Value::I64(i64::try_from(n).unwrap_or(i64::MAX));
        let mut row = vec![("index".to_string(), count(self.index as u64))];
        row.extend(self.condition.entries().iter().cloned());
        row.push(("status".into(), Value::Str(self.status.as_str().into())));
        row.push(("trials".into(), count(self.trials)));
        row.push(("wall_s".into(), Value::F64(self.wall_s)));
        for (k, e) in self.record.iter() {
            let (lo, hi) = e.interval(confidence);
            let values = [Value::F64(e.value()), Value::F64(lo), Value::F64(hi), count(e.count()), count(e.total())];
            row.extend(metric_columns(k).into_iter().zip(values));
        }
        row
    }
}

/// The names of the columns of the metric `name` in a flat row
/// ([`ConditionResult::row`]): `name`, `name_lo` and `name_hi` (the estimate and its
/// confidence interval), `name_count` and `name_total` (the errors and the events of a
/// rate, the number of values of a mean twice).
///
/// # Examples
/// ```
/// assert_eq!(com_sim::runner::metric_columns("ber"), ["ber", "ber_lo", "ber_hi", "ber_count", "ber_total"]);
/// ```
pub fn metric_columns(name: &str) -> [String; 5] {
    ["", "_lo", "_hi", "_count", "_total"].map(|suffix| format!("{name}{suffix}"))
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

    /// What the callbacks and the loggers know about a run of this runner on the
    /// scenario `scenario` over `n_conditions` conditions, started now (all the cores
    /// counted for `threads: 0`): the header of every report, that of a run, of a search
    /// and of a verification.
    pub fn run_info(&self, scenario: &str, n_conditions: usize) -> RunInfo {
        RunInfo {
            scenario: scenario.to_string(),
            seed: self.seed,
            batch: self.batch,
            threads: if self.threads == 0 { rayon::current_num_threads() } else { self.threads },
            stop: if self.fast_dev_run { "fast_dev_run".into() } else { self.stop.to_string() },
            confidence: self.confidence,
            n_conditions,
            started_unix_s: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()),
            version: crate::VERSION.into(),
            commit: crate::COMMIT.into(),
            config: self.config.clone(),
        }
    }

    /// Run the scenario on every condition of the grid.
    ///
    /// # Errors
    /// [`Error::Config`] for a grid that gives a key twice ([`Grid::check`]); for a
    /// confidence level outside $(0, 1)$ (of the runner or of a precision target, see
    /// [`check_confidence`](crate::stopping::check_confidence));
    /// for a stopping rule that can never stop, or whose target is a
    /// metric that the first batch of a condition does not record (or `min_errors` on a
    /// mean, see [`Stop::check_metrics`]); the first error of a `setup` or of a trial, of
    /// the same kind, with the condition and the trial in the message; the errors of the
    /// loggers.
    pub fn run<S: Scenario>(mut self, scenario: &S, grid: &Grid) -> Result<Vec<ConditionResult>> {
        self.check(grid)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .map_err(|e| Error::Config(format!("thread pool: {e}")))?;
        let conditions = grid.conditions();
        let info = RunInfo { threads: pool.current_num_threads(), ..self.run_info(scenario.name(), conditions.len()) };
        self.start(&info)?;
        let t_run = Instant::now();
        let mut results = Vec::with_capacity(conditions.len());
        let mut stop_run = false;
        for (index, cond) in conditions.into_iter().enumerate() {
            let res = self.condition(scenario, index, cond, &pool, stop_run)?;
            stop_run |= self.end_condition(&res)?;
            results.push(res);
        }
        self.finish(info, results, t_run)
    }

    /// The checks of [`Runner::run`] before it starts: the confidence levels, the grid and
    /// a stopping rule that stops.
    fn check(&self, grid: &Grid) -> Result<()> {
        crate::stopping::check_confidence(self.confidence, "the runner")?;
        self.stop.check_confidences()?;
        grid.check()?;
        if !self.fast_dev_run && !self.stop.is_bounded() {
            return Err(Error::Config(format!(
                "the stopping rule `{}` can run forever: add Stop::max_trials or Stop::max_time with `or`",
                self.stop
            )));
        }
        Ok(())
    }

    /// The start of the run, for every callback, then every logger.
    fn start(&mut self, info: &RunInfo) -> Result<()> {
        for c in &mut self.callbacks {
            c.on_run_start(info);
        }
        for l in &mut self.loggers {
            l.start(info)?;
        }
        Ok(())
    }

    /// One condition: skipped if the run was stopped or a callback skips it, run
    /// otherwise.
    fn condition<S: Scenario>(
        &mut self,
        scenario: &S,
        index: usize,
        cond: Condition,
        pool: &rayon::ThreadPool,
        stop_run: bool,
    ) -> Result<ConditionResult> {
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
        if skip {
            Ok(ConditionResult {
                index,
                condition: cond,
                status: Status::Skipped,
                trials: 0,
                wall_s: 0.0,
                record: Record::new(),
            })
        } else {
            self.run_condition(scenario, index, cond, pool)
        }
    }

    /// The end of a condition, for every callback, then every logger; whether a callback
    /// stops the run.
    fn end_condition(&mut self, res: &ConditionResult) -> Result<bool> {
        let mut stop_run = false;
        for c in &mut self.callbacks {
            if c.on_condition_end(res) == Action::StopRun {
                stop_run = true;
            }
        }
        for l in &mut self.loggers {
            l.log(res, self.confidence)?;
        }
        Ok(stop_run)
    }

    /// The end of the run, for every callback, then every logger with the wall time since
    /// `t_run`; the results.
    fn finish(&mut self, info: RunInfo, results: Vec<ConditionResult>, t_run: Instant) -> Result<Vec<ConditionResult>> {
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
                if matches!(c.on_batch_end(&progress), Action::StopCondition | Action::Skip) {
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
    fn a_callback_stops_a_condition_after_a_batch() {
        struct Once;
        impl Callback for Once {
            fn on_batch_end(&mut self, _: &Progress<'_>) -> Action {
                Action::StopCondition
            }
        }
        let r = Runner::new().batch(8).stop(Stop::max_trials(100)).callback(Once).run(&coin(), &Grid::new()).unwrap();
        assert_eq!((r[0].status.clone(), r[0].trials), (Status::Stopped, 8));
        assert_eq!(Stop::max_trials(5).and(Stop::min_trials(2)).to_string(), "(max_trials(5) and min_trials(2))");
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
