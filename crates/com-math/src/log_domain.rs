//! The log domain of probabilities and LLRs, without overflow: [`ln_1p_exp`],
//! [`max_star`] and its choice [`Jacobian`], [`log_sum_exp`] and [`log_sum_exp_iter`]
//! (re-exported at the root).

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

/// The Jacobian logarithm of the log-MAP algorithms and of the soft demappers, exact or
/// approximated by its maximum:
///
/// $$
/// \ln\left(e^{a} + e^{b}\right) = \max(a, b) + \ln\left(1 + e^{-\lvert a - b \rvert}\right)
/// \approx \max(a, b).
/// $$
///
/// One choice for [`max_star`], the BCJR and turbo decoders of `com-fec` and the
/// demappers of `com-mapping`; a specification keeps the boolean key `max_log`
/// ([`Jacobian::from_max_log`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Jacobian {
    /// The exact logarithm (log-MAP, log-sum-exp).
    #[default]
    Exact,
    /// The max-log approximation, $\max(a, b)$.
    MaxLog,
}

impl Jacobian {
    /// [`Jacobian::MaxLog`] when `max_log` is true, [`Jacobian::Exact`] otherwise (the
    /// boolean key `max_log` of the specifications and of the Python API).
    ///
    /// # Examples
    /// ```
    /// use com_math::Jacobian;
    /// assert_eq!(Jacobian::from_max_log(true), Jacobian::MaxLog);
    /// assert_eq!(Jacobian::from_max_log(false), Jacobian::default());
    /// ```
    pub fn from_max_log(max_log: bool) -> Jacobian {
        if max_log { Jacobian::MaxLog } else { Jacobian::Exact }
    }

    /// Whether the logarithm is exact.
    pub fn is_exact(self) -> bool {
        self == Jacobian::Exact
    }
}

/// The `max*` operation `ln(e^a + e^b)` of the log-MAP algorithms; `max(a, b)` with
/// [`Jacobian::MaxLog`] or when the maximum is not finite.
///
/// # Examples
/// ```
/// use com_math::{Jacobian, max_star};
/// assert!((max_star(0.0, 0.0, Jacobian::Exact) - 2f64.ln()).abs() < 1e-15);
/// assert_eq!(max_star(0.0, -1.0, Jacobian::MaxLog), 0.0);
/// ```
#[inline]
pub fn max_star(a: f64, b: f64, jacobian: Jacobian) -> f64 {
    let m = a.max(b);
    if !jacobian.is_exact() || !m.is_finite() { m } else { m + (-(a - b).abs()).exp().ln_1p() }
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
