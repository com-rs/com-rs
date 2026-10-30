//! When a condition has run enough trials.
//!
//! A [`Stop`] rule is evaluated after every batch of trials of a condition, on the
//! merged metrics. The usual rules of error-rate estimation:
//!
//! - [`Stop::max_trials`]: a fixed budget of trials (the default is 100);
//! - [`Stop::min_errors`]: at least `k` errors of an error rate, the classical rule of
//!   BER simulations (with `k = 100` the relative standard deviation of the estimate is
//!   about 10 %); it needs an error rate (a mean has no errors: use
//!   [`Stop::precision`]);
//! - [`Stop::precision`]: the half-width of the confidence interval of a metric below a
//!   fraction of its value;
//! - [`Stop::min_trials`]: a lower bound on the trials, to combine with the others;
//! - [`Stop::max_time`]: a wall-time budget per condition (the only rule that depends
//!   on the machine, and therefore breaks the thread-count independence of the
//!   results).
//!
//! Rules combine with [`Stop::or`] (any of them) and [`Stop::and`] (all of them): a
//! typical BER rule is `min_errors("ber", 200).or(max_trials(10⁵))`, which stops at 200
//! errors or after 10⁵ trials when the BER is too low to reach them.
//!
//! The runner checks the rule against the metrics of the first batch of every
//! condition ([`Stop::check_metrics`]): a target on a metric that the scenario does not
//! record, or `min_errors` on a mean, would never be met and is an error. It also ends
//! the last batch at the bound of the rule on the trials ([`Stop::trial_limit`]), so
//! that `max_trials(1000)` runs 1000 trials whatever the batch size.
//!
//! # Examples
//! ```
//! use com_sim::stopping::{Stop, Progress};
//! use com_sim::scenario::Record;
//! use std::time::Duration;
//! let rule = Stop::min_errors("ber", 100).or(Stop::max_trials(1000)).and(Stop::min_trials(10));
//! let mut rec = Record::new();
//! rec.errors("ber", 150, 10_000);
//! let p = Progress { condition: 0, trials: 5, record: &rec, elapsed: Duration::ZERO };
//! assert!(!rule.is_met(&p));            // enough errors but fewer than 10 trials
//! let p = Progress { trials: 12, ..p };
//! assert!(rule.is_met(&p));
//! assert_eq!(rule.to_string(), "((errors(ber) ≥ 100 or trials ≥ 1000) and trials ≥ 10)");
//! ```

use crate::scenario::Record;
use crate::stats::Estimate;
use com_core::{Error, Result};
use std::fmt;
use std::time::Duration;

/// The state of a condition seen by the stopping rule and the callbacks.
#[derive(Clone, Copy, Debug)]
pub struct Progress<'a> {
    /// Index of the condition in the grid.
    pub condition: usize,
    /// Trials run so far.
    pub trials: u64,
    /// Metrics merged over these trials.
    pub record: &'a Record,
    /// Wall time spent on the condition.
    pub elapsed: Duration,
}

/// A stopping rule.
#[derive(Clone, Debug, PartialEq)]
pub enum Stop {
    /// At least this many trials.
    MaxTrials(u64),
    /// Never before this many trials (use with [`Stop::and`]).
    MinTrials(u64),
    /// At least `errors` errors of the error rate `metric` (never met by a mean).
    MinErrors {
        /// The error-rate metric.
        metric: String,
        /// The number of errors.
        errors: u64,
    },
    /// Relative half-width of the confidence interval of `metric` at most `relative`.
    Precision {
        /// The metric.
        metric: String,
        /// The relative half-width.
        relative: f64,
        /// The confidence level of the interval.
        confidence: f64,
    },
    /// At least this much wall time on the condition.
    MaxTime(Duration),
    /// Any of the rules.
    Any(Vec<Stop>),
    /// All of the rules.
    All(Vec<Stop>),
}

impl Stop {
    /// Stop after `n` trials.
    pub fn max_trials(n: u64) -> Stop {
        Stop::MaxTrials(n)
    }

    /// Never stop before `n` trials.
    pub fn min_trials(n: u64) -> Stop {
        Stop::MinTrials(n)
    }

    /// Stop after `errors` errors of the error rate `metric`.
    pub fn min_errors(metric: &str, errors: u64) -> Stop {
        Stop::MinErrors { metric: metric.to_string(), errors }
    }

    /// Stop when the 95 % interval of `metric` is within `±relative` of its value.
    pub fn precision(metric: &str, relative: f64) -> Stop {
        Stop::precision_at(metric, relative, 0.95)
    }

    /// Stop when the interval of level `confidence` of `metric` is within `±relative` of
    /// its value (the runner refuses a level outside $(0, 1)$, see [`check_confidence`]).
    ///
    /// # Examples
    /// ```
    /// use com_sim::stopping::Stop;
    /// assert_eq!(Stop::precision_at("snr_db", 0.01, 0.99).to_string(), "precision(snr_db) ≤ 0.01 at 0.99");
    /// ```
    pub fn precision_at(metric: &str, relative: f64, confidence: f64) -> Stop {
        Stop::Precision { metric: metric.to_string(), relative, confidence }
    }

    /// Stop after `d` of wall time on the condition.
    pub fn max_time(d: Duration) -> Stop {
        Stop::MaxTime(d)
    }

    /// Either rule.
    pub fn or(self, other: Stop) -> Stop {
        Stop::Any(vec![self, other])
    }

    /// Both rules.
    pub fn and(self, other: Stop) -> Stop {
        Stop::All(vec![self, other])
    }

    /// Whether the rule is met.
    pub fn is_met(&self, p: &Progress<'_>) -> bool {
        match self {
            Stop::MaxTrials(n) | Stop::MinTrials(n) => p.trials >= *n,
            Stop::MinErrors { metric, errors } => {
                matches!(p.record.get(metric), Some(Estimate::Rate(r)) if r.errors >= *errors)
            }
            Stop::Precision { metric, relative, confidence } => {
                p.record.get(metric).is_some_and(|e| e.relative_precision(*confidence) <= *relative)
            }
            Stop::MaxTime(d) => p.elapsed >= *d,
            Stop::Any(v) => v.iter().any(|s| s.is_met(p)),
            Stop::All(v) => v.iter().all(|s| s.is_met(p)),
        }
    }

    /// Whether the rule can stop at all (a rule made only of lower bounds would run
    /// forever).
    pub fn is_bounded(&self) -> bool {
        match self {
            Stop::MaxTrials(_) | Stop::MaxTime(_) => true,
            Stop::MinTrials(_) | Stop::MinErrors { .. } | Stop::Precision { .. } => false,
            Stop::Any(v) => v.iter().any(Stop::is_bounded),
            Stop::All(v) => v.iter().all(Stop::is_bounded),
        }
    }

    /// The number of trials at which the rule is met whatever the metrics and the
    /// time, if there is one: `n` for [`Stop::max_trials`] and [`Stop::min_trials`], the
    /// smallest bound of an [`Stop::or`], the largest of an [`Stop::and`] whose rules
    /// all have one. The runner ends its last batch there.
    ///
    /// # Examples
    /// ```
    /// use com_sim::stopping::Stop;
    /// assert_eq!(Stop::min_errors("ber", 100).or(Stop::max_trials(1000)).trial_limit(), Some(1000));
    /// assert_eq!(Stop::max_trials(1000).and(Stop::min_trials(10)).trial_limit(), Some(1000));
    /// assert_eq!(Stop::min_errors("ber", 100).and(Stop::max_trials(1000)).trial_limit(), None);
    /// ```
    pub fn trial_limit(&self) -> Option<u64> {
        match self {
            Stop::MaxTrials(n) | Stop::MinTrials(n) => Some(*n),
            Stop::MinErrors { .. } | Stop::Precision { .. } | Stop::MaxTime(_) => None,
            Stop::Any(v) => v.iter().filter_map(Stop::trial_limit).min(),
            Stop::All(v) => v.iter().map(Stop::trial_limit).collect::<Option<Vec<u64>>>()?.into_iter().max(),
        }
    }

    /// The metrics the rule targets, with whether the target needs an error rate.
    fn targets<'a>(&'a self, out: &mut Vec<(&'a str, bool)>) {
        match self {
            Stop::MinErrors { metric, .. } => out.push((metric, true)),
            Stop::Precision { metric, .. } => out.push((metric, false)),
            Stop::MaxTrials(_) | Stop::MinTrials(_) | Stop::MaxTime(_) => {}
            Stop::Any(v) | Stop::All(v) => v.iter().for_each(|s| s.targets(out)),
        }
    }

    /// Check the confidence levels of the precision targets of the rule
    /// ([`check_confidence`]).
    ///
    /// # Errors
    /// [`Error::Config`] for a level outside $(0, 1)$.
    pub fn check_confidences(&self) -> Result<()> {
        match self {
            Stop::Precision { confidence, .. } => check_confidence(*confidence, &format!("the stopping rule `{self}`")),
            Stop::Any(v) | Stop::All(v) => v.iter().try_for_each(Stop::check_confidences),
            Stop::MaxTrials(_) | Stop::MinTrials(_) | Stop::MinErrors { .. } | Stop::MaxTime(_) => Ok(()),
        }
    }

    /// Check that every target of the rule can be met by the metrics of `record` (the
    /// record of a first batch of trials).
    ///
    /// # Errors
    /// [`Error::Config`] for a target on a metric that `record` does not hold (the
    /// message names the recorded ones), or [`Stop::min_errors`] on a mean.
    ///
    /// # Examples
    /// ```
    /// use com_sim::stopping::Stop;
    /// use com_sim::scenario::Record;
    /// let mut rec = Record::new();
    /// rec.errors("ber", 3, 1000);
    /// rec.value("snr_db", 9.5);
    /// assert!(Stop::min_errors("ber", 100).or(Stop::max_trials(10)).check_metrics(&rec).is_ok());
    /// let e = Stop::min_errors("bler", 100).or(Stop::max_trials(10)).check_metrics(&rec).unwrap_err();
    /// assert!(e.to_string().contains("bler") && e.to_string().contains("ber, snr_db"));
    /// assert!(Stop::min_errors("snr_db", 100).or(Stop::max_trials(10)).check_metrics(&rec).is_err());
    /// ```
    pub fn check_metrics(&self, record: &Record) -> Result<()> {
        let mut targets = Vec::new();
        self.targets(&mut targets);
        for (metric, needs_rate) in targets {
            if !record.knows(metric) {
                let names: Vec<&str> = record.iter().map(|(k, _)| k.as_str()).collect();
                return Err(Error::Config(format!(
                    "the stopping rule `{self}` targets the metric `{metric}`, which is not recorded (recorded: {}): it would never be met",
                    if names.is_empty() { "none".to_string() } else { names.join(", ") }
                )));
            }
            if needs_rate && matches!(record.get(metric), Some(Estimate::Mean(_))) {
                return Err(Error::Config(format!(
                    "the stopping rule `{self}`: min_errors needs an error rate, and `{metric}` is a mean (use precision)"
                )));
            }
        }
        Ok(())
    }
}

/// Check a confidence level: a probability strictly between 0 and 1 (`0.95`, not `95`),
/// `what` naming where it was given.
///
/// # Errors
/// [`Error::Config`] for a level outside $(0, 1)$ or not a number.
///
/// # Examples
/// ```
/// use com_sim::stopping::check_confidence;
/// assert!(check_confidence(0.95, "confidence").is_ok());
/// let e = check_confidence(95.0, "confidence").unwrap_err().to_string();
/// assert!(e.contains("between 0 and 1"), "{e}");
/// ```
pub fn check_confidence(c: f64, what: &str) -> Result<()> {
    if c > 0.0 && c < 1.0 {
        Ok(())
    } else {
        Err(Error::Config(format!("{what}: the confidence level must be between 0 and 1 (0.95 for 95 %), got {c}")))
    }
}

impl fmt::Display for Stop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stop::MaxTrials(n) | Stop::MinTrials(n) => write!(f, "trials ≥ {n}"),
            Stop::MinErrors { metric, errors } => write!(f, "errors({metric}) ≥ {errors}"),
            Stop::Precision { metric, relative, confidence } => {
                write!(f, "precision({metric}) ≤ {relative} at {confidence}")
            }
            Stop::MaxTime(d) => write!(f, "time ≥ {:.1} s", d.as_secs_f64()),
            Stop::Any(v) => {
                write!(f, "({})", v.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(" or "))
            }
            Stop::All(v) => {
                write!(f, "({})", v.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(" and "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_target_on_a_metric_not_recorded_is_an_error() {
        let mut rec = Record::new();
        rec.errors("ber", 1, 100);
        rec.value("snr_db", f64::INFINITY); // dropped, but recorded
        let rule = Stop::min_errors("ber", 10).or(Stop::max_trials(5));
        assert!(rule.check_metrics(&rec).is_ok());
        assert!(Stop::precision("snr_db", 0.1).or(Stop::max_trials(5)).check_metrics(&rec).is_ok());
        let e = Stop::precision("evm", 0.1).or(Stop::max_trials(5)).check_metrics(&rec).unwrap_err().to_string();
        assert!(e.contains("`evm`") && e.contains("recorded: ber"), "{e}");
        let e = Stop::min_errors("ber", 1).or(Stop::max_trials(5)).check_metrics(&Record::new()).unwrap_err();
        assert!(e.to_string().contains("recorded: none"), "{e}");
        assert!(Stop::max_trials(5).check_metrics(&Record::new()).is_ok());
    }

    #[test]
    fn min_errors_on_a_mean_is_an_error_and_never_met() {
        let mut rec = Record::new();
        rec.values("snr_db", [1.0; 50]);
        let rule = Stop::min_errors("snr_db", 10);
        let e = rule.check_metrics(&rec).unwrap_err().to_string();
        assert!(e.contains("is a mean"), "{e}");
        let p = Progress { condition: 0, trials: 50, record: &rec, elapsed: Duration::ZERO };
        assert!(!rule.is_met(&p));
    }

    #[test]
    fn the_trial_limit_of_combined_rules() {
        assert_eq!(Stop::max_trials(7).trial_limit(), Some(7));
        assert_eq!(Stop::max_time(Duration::from_secs(1)).trial_limit(), None);
        assert_eq!(Stop::max_trials(7).or(Stop::max_trials(3)).trial_limit(), Some(3));
        assert_eq!(Stop::max_time(Duration::from_secs(1)).or(Stop::max_trials(3)).trial_limit(), Some(3));
        assert_eq!(Stop::precision("x", 0.1).and(Stop::min_trials(4)).or(Stop::max_trials(9)).trial_limit(), Some(9));
        assert_eq!(Stop::max_time(Duration::from_secs(1)).and(Stop::max_trials(3)).trial_limit(), None);
    }
}
