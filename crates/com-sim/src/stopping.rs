//! When a condition has run enough trials.
//!
//! A [`Stop`] rule is evaluated after every batch of trials of a condition, on the
//! merged metrics. The usual rules of error-rate estimation:
//!
//! - [`Stop::max_trials`]: a fixed budget of trials (the default is 100);
//! - [`Stop::min_errors`]: at least `k` errors of an error rate, the classical rule of
//!   BER simulations (with `k = 100` the relative standard deviation of the estimate is
//!   about 10 %);
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
    /// At least `errors` errors of the error rate `metric`.
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
        Stop::Precision { metric: metric.to_string(), relative, confidence: 0.95 }
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
            Stop::MinErrors { metric, errors } => p.record.get(metric).is_some_and(|e| e.count() >= *errors),
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
