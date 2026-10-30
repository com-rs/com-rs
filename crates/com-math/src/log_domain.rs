//! The log domain of probabilities and LLRs, without overflow: [`ln_1p_exp`],
//! [`max_star`], [`log_sum_exp`] and [`log_sum_exp_iter`] (re-exported at the root).

/// `ln(1 + e^x)` (the softplus) without overflow.
///
/// # Examples
/// ```
/// use com_math::ln_1p_exp;
/// assert!((ln_1p_exp(0.0) - 2f64.ln()).abs() < 1e-15);
/// assert_eq!(ln_1p_exp(1000.0), 1000.0);
/// ```
#[inline]
pub fn ln_1p_exp(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// The `max*` operation `ln(e^a + e^b)` of the log-MAP algorithms, `max(a, b)` when
/// `exact` is false (the max-log approximation) or when the maximum is not finite.
#[inline]
pub fn max_star(a: f64, b: f64, exact: bool) -> f64 {
    let m = a.max(b);
    if !exact || !m.is_finite() { m } else { m + (-(a - b).abs()).exp().ln_1p() }
}

/// `ln Σ_i e^{v_i}` without overflow (the log domain of probabilities and LLRs).
///
/// # Examples
/// ```
/// use com_math::log_sum_exp;
/// assert!((log_sum_exp(&[1000.0, 1000.0]) - (1000.0 + 2f64.ln())).abs() < 1e-9);
/// assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
/// ```
pub fn log_sum_exp(v: &[f64]) -> f64 {
    log_sum_exp_iter(v.iter().copied())
}

/// [`log_sum_exp`] of the values of an iterator, traversed twice (no allocation).
///
/// # Examples
/// ```
/// use com_math::{log_sum_exp, log_sum_exp_iter};
/// let v = [0.5, -2.0, 3.0, 1.0];
/// // the terms of even index only
/// let even = log_sum_exp_iter(v.iter().step_by(2).copied());
/// assert_eq!(even, log_sum_exp(&[0.5, 3.0]));
/// ```
pub fn log_sum_exp_iter<I: Iterator<Item = f64> + Clone>(v: I) -> f64 {
    let m = v.clone().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.map(|x| (x - m).exp()).sum::<f64>().ln()
}
