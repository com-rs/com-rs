//! Running statistics with confidence intervals.
//!
//! - [`ErrorRate`] counts errors among trials of a Bernoulli event (bit, symbol or block
//!   errors). Its estimate is `p̂ = k/n` and its interval the **Wilson score interval**
//!   at confidence `1 - α`,
//!
//!   `(p̂ + z²/2n ± z √(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)`, `z = Φ⁻¹(1 - α/2)`,
//!
//!   which, unlike the normal approximation, stays inside `[0, 1]` and is informative
//!   with few or no errors (with `k = 0` its upper end is about `z²/n`).
//! - [`Mean`] accumulates real values with Welford's algorithm (numerically stable
//!   running mean and variance) and merges two accumulators exactly (Chan, Golub and
//!   LeVeque); its interval is the normal interval of the mean, `x̄ ± z s/√n`.
//! - [`Estimate`] is either of the two, with the same interface.
//!
//! # Examples
//! ```
//! use com_sim::stats::{ErrorRate, Mean, Estimate, z_score};
//! let mut e = ErrorRate::default();
//! e.add(3, 1000);
//! e.add(7, 1000);
//! assert_eq!(e.rate(), 5e-3);
//! let (lo, hi) = e.wilson(0.95);
//! assert!(lo < 5e-3 && 5e-3 < hi);
//! let mut m = Mean::default();
//! for v in [1.0, 2.0, 3.0, 4.0] { m.add(v); }
//! assert_eq!((m.mean(), m.variance()), (2.5, 5.0 / 3.0));
//! assert!((z_score(0.95) - 1.959964).abs() < 1e-5);
//! let est = Estimate::Rate(e);
//! assert_eq!(est.value(), 5e-3);
//! ```

use com_math::erfcinv;

/// `z = Φ⁻¹(1 - α/2)` of a two-sided confidence level `1 - α` (e.g. 1.96 for 0.95).
pub fn z_score(confidence: f64) -> f64 {
    std::f64::consts::SQRT_2 * erfcinv(1.0 - confidence.clamp(0.0, 1.0 - 1e-16))
}

/// Errors among trials of a Bernoulli event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ErrorRate {
    /// Number of errors `k`.
    pub errors: u64,
    /// Number of trials `n`.
    pub total: u64,
}

impl ErrorRate {
    /// Add `errors` errors among `total` new trials.
    pub fn add(&mut self, errors: u64, total: u64) {
        self.errors += errors;
        self.total += total;
    }

    /// Merge another count.
    pub fn merge(&mut self, o: &ErrorRate) {
        self.add(o.errors, o.total);
    }

    /// `k / n` (`NaN` without trials).
    pub fn rate(&self) -> f64 {
        if self.total == 0 { f64::NAN } else { self.errors as f64 / self.total as f64 }
    }

    /// Wilson score interval at the given confidence (`(0, 1)` without trials).
    pub fn wilson(&self, confidence: f64) -> (f64, f64) {
        if self.total == 0 {
            return (0.0, 1.0);
        }
        let n = self.total as f64;
        let p = self.rate();
        let z = z_score(confidence);
        let z2 = z * z;
        let centre = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
        let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / (1.0 + z2 / n);
        ((centre - half).max(0.0), (centre + half).min(1.0))
    }
}

/// Running mean and variance of real values (Welford).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(clippy::struct_field_names, reason = "the running mean of Welford's algorithm")]
pub struct Mean {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Mean {
    /// Add one value.
    pub fn add(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    /// Merge another accumulator (exact, Chan et al.).
    pub fn merge(&mut self, o: &Mean) {
        if o.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *o;
            return;
        }
        let n = self.n + o.n;
        let d = o.mean - self.mean;
        self.mean += d * o.n as f64 / n as f64;
        self.m2 += o.m2 + d * d * self.n as f64 * o.n as f64 / n as f64;
        self.n = n;
    }

    /// Number of values.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Mean (`NaN` without values).
    pub fn mean(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.mean }
    }

    /// Unbiased sample variance (`NaN` with fewer than two values).
    pub fn variance(&self) -> f64 {
        if self.n < 2 { f64::NAN } else { self.m2 / (self.n - 1) as f64 }
    }

    /// Sample standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Standard error of the mean `s/√n`.
    pub fn sem(&self) -> f64 {
        self.std() / (self.n as f64).sqrt()
    }

    /// Normal confidence interval of the mean, `mean ± z s/√n` (unbounded with fewer
    /// than two values, so that a precision rule never stops on a single value).
    pub fn interval(&self, confidence: f64) -> (f64, f64) {
        if self.n < 2 {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let h = z_score(confidence) * self.sem();
        (self.mean - h, self.mean + h)
    }
}

/// The estimate of one metric: an error rate or a mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Estimate {
    /// Error counts.
    Rate(ErrorRate),
    /// Values.
    Mean(Mean),
}

impl Estimate {
    /// The point estimate (rate or mean).
    pub fn value(&self) -> f64 {
        match self {
            Estimate::Rate(r) => r.rate(),
            Estimate::Mean(m) => m.mean(),
        }
    }

    /// The confidence interval (Wilson or normal).
    pub fn interval(&self, confidence: f64) -> (f64, f64) {
        match self {
            Estimate::Rate(r) => r.wilson(confidence),
            Estimate::Mean(m) => m.interval(confidence),
        }
    }

    /// Half-width of the interval relative to the estimate (`∞` for a zero estimate).
    pub fn relative_precision(&self, confidence: f64) -> f64 {
        let (lo, hi) = self.interval(confidence);
        let v = self.value().abs();
        if v == 0.0 || !v.is_finite() { f64::INFINITY } else { 0.5 * (hi - lo) / v }
    }

    /// Number of errors (rates) or of values (means).
    pub fn count(&self) -> u64 {
        match self {
            Estimate::Rate(r) => r.errors,
            Estimate::Mean(m) => m.count(),
        }
    }

    /// Number of trials (rates) or of values (means).
    pub fn total(&self) -> u64 {
        match self {
            Estimate::Rate(r) => r.total,
            Estimate::Mean(m) => m.count(),
        }
    }

    /// Merge an estimate of the same kind (a mismatch keeps `self` unchanged and
    /// returns `false`).
    pub fn merge(&mut self, o: &Estimate) -> bool {
        match (self, o) {
            (Estimate::Rate(a), Estimate::Rate(b)) => {
                a.merge(b);
                true
            }
            (Estimate::Mean(a), Estimate::Mean(b)) => {
                a.merge(b);
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_with_no_error_and_merges() {
        let r = ErrorRate { errors: 0, total: 1000 };
        let (lo, hi) = r.wilson(0.95);
        assert_eq!(lo, 0.0);
        assert!((hi - 3.83e-3).abs() < 1e-4, "{hi}");
        let mut a = Mean::default();
        let mut b = Mean::default();
        let mut all = Mean::default();
        for i in 0..100 {
            let v = (i as f64 * 0.37).sin();
            if i < 30 {
                a.add(v);
            } else {
                b.add(v);
            }
            all.add(v);
        }
        a.merge(&b);
        assert!((a.mean() - all.mean()).abs() < 1e-14 && (a.variance() - all.variance()).abs() < 1e-14);
    }
}
