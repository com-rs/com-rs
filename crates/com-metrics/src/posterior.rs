//! The log-sum-exp of the Gaussian posterior of the points of a constellation, shared by
//! the information rates ([`crate::information`]) and the likelihood classifiers
//! ([`crate::blind`]).
//!
//! # Model
//!
//! For a received $y$, the points $x_{j}$ of log-priors $\ln p_{j}$, the noise variance
//! $\sigma^{2}$ and a scaling $s$ of the exponent, the log-metrics are
//! $\ell_{j} = \ln p_{j} - s |y - x_{j}|^{2} / \sigma^{2}$; relative to a reference
//! point $x_{i}$,
//!
//! $$
//! \ln \sum_{j} e^{\ell_{j} - \ell_{i}} = -\ln q_{s}(x_{i} \mid y),
//! $$
//!
//! the information of $x_{i}$ under the posterior $q_{s}$, and without a reference
//! $\ln \sum_{j} e^{\ell_{j}}$, the log-likelihood of $y$ up to the Gaussian
//! normalisation. The terms are formed as
//! $(\ln p_{j} - \ln p_{i}) - s\\,((|y - x_{j}|^{2} - |y - x_{i}|^{2}) / \sigma^{2})$
//! (with $\ln p_{i} = |y - x_{i}|^{2} = 0$ without a reference): the term of the
//! reference is exactly 0, whatever the size of the distances against $\sigma^{2}$.

use com_math::{C64, log_sum_exp};

/// The log-metrics of the points of a constellation (see the module), with the scratch
/// of one received symbol.
pub(crate) struct SymbolMetric<'a> {
    points: &'a [C64],
    log_px: &'a [f64],
    noise_var: f64,
    terms: Vec<f64>,
}

impl<'a> SymbolMetric<'a> {
    /// The metric of `points` with the log-priors `log_px` (zeros for a likelihood
    /// without priors) in noise of variance `noise_var`.
    pub(crate) fn new(points: &'a [C64], log_px: &'a [f64], noise_var: f64) -> SymbolMetric<'a> {
        debug_assert_eq!(points.len(), log_px.len());
        SymbolMetric { points, log_px, noise_var, terms: vec![0.0; points.len()] }
    }

    /// $\ln \sum_{j} e^{\ell_{j} - \ell_{i}}$ for the received `y`, the scale `s` and
    /// the reference point `i` ($-\ln q_{s}(x_{i} \mid y)$), or $\ln \sum_{j} e^{\ell_{j}}$
    /// when `reference` is `None`.
    pub(crate) fn log_sum(&mut self, y: C64, s: f64, reference: Option<usize>) -> f64 {
        let (lp_ref, d_ref) = reference.map_or((0.0, 0.0), |i| (self.log_px[i], (y - self.points[i]).norm_sqr()));
        for ((t, pt), lp) in self.terms.iter_mut().zip(self.points).zip(self.log_px) {
            *t = lp - lp_ref - s * (((y - pt).norm_sqr() - d_ref) / self.noise_var);
        }
        log_sum_exp(&self.terms)
    }

    /// The term $\ell_{j} - \ell_{i}$ of the point `j` in the last [`SymbolMetric::log_sum`]
    /// ($\ell_{j}$ without a reference).
    pub(crate) fn term(&self, j: usize) -> f64 {
        self.terms[j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_mapping::constellation::{Constellation, log_priors};

    #[test]
    fn the_sums_are_those_of_the_former_estimators_to_the_last_bit() {
        let c = Constellation::qam(16).unwrap().normalized();
        let lp = log_priors(&c.priors(Some(&(1..=16).map(f64::from).collect::<Vec<_>>())).unwrap());
        let mut metric = SymbolMetric::new(c.points(), &lp, 0.07);
        let mut rng = com_core::rng::Rng::new(4);
        for k in 0..200 {
            let i = k % 16;
            let y = c.points()[i] + rng.complex_gaussian() * 0.3;
            // information_rates: the terms relative to the transmitted point
            let d0 = (y - c.points()[i]).norm_sqr();
            let rel: Vec<f64> =
                c.points().iter().zip(&lp).map(|(p, l)| l - lp[i] - ((y - p).norm_sqr() - d0) / 0.07).collect();
            assert_eq!(metric.log_sum(y, 1.0, Some(i)), log_sum_exp(&rel));
            // monte_carlo_mi and classify_ml: the absolute log-metrics
            let abs: Vec<f64> = c.points().iter().zip(&lp).map(|(p, l)| -(y - p).norm_sqr() / 0.07 + l).collect();
            assert_eq!(metric.log_sum(y, 1.0, None), log_sum_exp(&abs));
            assert_eq!(metric.term(i), abs[i]);
        }
        // the term of the reference is 0 even when the distances overflow against σ²
        let mut tiny = SymbolMetric::new(c.points(), &lp, f64::MIN_POSITIVE);
        assert_eq!(tiny.log_sum(c.points()[3] + 1e-3, 1.0, Some(3)), 0.0);
    }
}
