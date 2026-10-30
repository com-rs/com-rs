//! One-dimensional searches: bisection, regula falsi and golden-section maximisation,
//! the largest element of a sequence, and the greatest common divisor.
//!
//! # Model
//!
//! [`bisect`] halves a bracket $[a, b]$ of a predicate that holds below a threshold
//! $x^{\star}$ and fails above it: at every step, with $c = \frac{1}{2}(a + b)$,
//!
//! $$
//! [a, b] \leftarrow \begin{cases} [c, b] & \text{if the predicate holds at } c, \\\\
//! [a, c] & \text{otherwise,} \end{cases}
//! $$
//!
//! and it returns the middle of the last bracket, within $2^{-(n+1)}(b - a)$ of
//! $x^{\star}$ after $n$ steps.
//!
//! [`regula_falsi`] finds a root of a continuous function $f$ whose values at the ends of
//! the bracket have opposite signs, by the Illinois variant of the false position
//! (Dowell & Jarratt 1971): the next point is the root of the chord,
//!
//! $$
//! c = \frac{a f(b) - b f(a)}{f(b) - f(a)},
//! $$
//!
//! which replaces the end of the same sign; when the same end is replaced twice in a row,
//! the value kept at the other end is halved, which restores a superlinear convergence.
//!
//! [`golden_max`] shrinks a bracket of a unimodal function by the golden ratio
//! $\varphi^{-1} = (\sqrt{5} - 1)/2$ at every step, with the two interior points
//! $c = b - \varphi^{-1}(b - a)$ and $d = a + \varphi^{-1}(b - a)$; one evaluation per
//! step, the bracket length multiplied by $\varphi^{-n}$ after $n$ steps (Kiefer 1953).
//!
//! They keep the order of their operations (`0.5 * (lo + hi)`, no `f64::midpoint`;
//! `docs/adr/0005-lint-exceptions.md`), so that the published values computed with them
//! do not change.
//!
//! # Examples
//! ```
//! use com_math::{bisect, gcd, golden_max};
//! // √2: x² < 2 below the root
//! let r = bisect(|x| x * x < 2.0, 0.0, 2.0, 60);
//! assert!((r - 2f64.sqrt()).abs() < 1e-15);
//! // the maximum of 1 - (x - 0.3)² on [0, 1]
//! let x = golden_max(|x| 1.0 - (x - 0.3) * (x - 0.3), 0.0, 1.0, 60);
//! assert!((x - 0.3).abs() < 1e-8);
//! assert_eq!(gcd(12, 18), 6);
//! ```
//!
//! # References
//!
//! 1. J. Kiefer, "Sequential minimax search for a maximum," *Proc. Amer. Math. Soc.*,
//!    4(3), 502–506, 1953.
//! 2. W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical
//!    Recipes*, 3rd ed., Cambridge University Press, 2007, Sec. 9.1 (bisection) and
//!    Sec. 10.2 (golden section search).
//! 3. M. Dowell and P. Jarratt, "A modified regula falsi method for computing the root of
//!    an equation," *BIT* 11(2), 168–174, 1971.

/// The threshold of a predicate by `iters` bisection steps of the bracket `[lo, hi]`:
/// `below(x)` must hold below the threshold and fail above it (see the module).
///
/// # Examples
/// ```
/// use com_math::bisect;
/// // the median of the exponential distribution: 1 - e^{-x} = 1/2
/// let m = bisect(|x| 1.0 - (-x).exp() < 0.5, 0.0, 10.0, 100);
/// assert!((m - 2f64.ln()).abs() < 1e-14);
/// ```
pub fn bisect(mut below: impl FnMut(f64) -> bool, lo: f64, hi: f64, iters: usize) -> f64 {
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        if below(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// A root of `f` between `a` and `b`, where `f` takes values of opposite signs, by at most
/// `iters` steps of the Illinois regula falsi (see the module): the first point $c$ of
/// $|f(c)| <$ `f_tol`, or the point computed when the bracket is shorter than `x_tol`;
/// the middle of the bracket after `iters` steps.
///
/// # Examples
/// ```
/// use com_math::search::regula_falsi;
/// let r = regula_falsi(|x| x * x * x - 2.0, 0.0, 2.0, 1e-12, 1e-14, 100);
/// assert!((r - 2f64.cbrt()).abs() < 1e-12);
/// ```
pub fn regula_falsi(mut f: impl FnMut(f64) -> f64, a: f64, b: f64, x_tol: f64, f_tol: f64, iters: usize) -> f64 {
    let (mut a, mut b) = (a, b);
    let (mut fa, mut fb) = (f(a), f(b));
    // the end replaced at the previous step: -1 for a, 1 for b
    let mut side = 0;
    for _ in 0..iters {
        let c = (a * fb - b * fa) / (fb - fa);
        let fc = f(c);
        if fc.abs() < f_tol || (b - a).abs() < x_tol {
            return c;
        }
        if (fc < 0.0) == (fa < 0.0) {
            (a, fa) = (c, fc);
            if side == -1 {
                fb /= 2.0;
            }
            side = -1;
        } else {
            (b, fb) = (c, fc);
            if side == 1 {
                fa /= 2.0;
            }
            side = 1;
        }
    }
    0.5 * (a + b)
}

/// The maximiser of a unimodal function `f` on `[a, b]` by `iters` steps of the golden
/// section search (see the module): the middle of the last bracket.
///
/// # Examples
/// ```
/// use com_math::golden_max;
/// // x e^{-x} is largest at x = 1
/// let x = golden_max(|x| x * (-x).exp(), 0.0, 5.0, 80);
/// assert!((x - 1.0).abs() < 1e-7);
/// ```
pub fn golden_max(mut f: impl FnMut(f64) -> f64, a: f64, b: f64, iters: usize) -> f64 {
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (a, b);
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..iters {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// The greatest common divisor of `a` and `b` (Euclid's algorithm; `gcd(a, 0) = a`).
///
/// # Examples
/// ```
/// use com_math::gcd;
/// assert_eq!(gcd(255, 11), 1);
/// assert_eq!(gcd(0, 7), 7);
/// ```
pub fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The index and the value of the largest element of `values`, the first one on ties
/// (`None` for an empty sequence or one of NaN only; NaN elements are skipped): the
/// decision of the detectors that pick the strongest candidate.
///
/// # Examples
/// ```
/// use com_math::argmax;
/// assert_eq!(argmax([0.5, 2.0, -1.0, 2.0]), Some((1, 2.0)));
/// assert_eq!(argmax([f64::NAN, -3.0]), Some((1, -3.0)));
/// assert_eq!(argmax(std::iter::empty()), None);
/// ```
pub fn argmax(values: impl IntoIterator<Item = f64>) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, v) in values.into_iter().enumerate() {
        if !v.is_nan() && best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The folds the detectors wrote before they called `argmax`, with an initial value
    /// that no element beats: the same index on ties and on every sequence.
    #[test]
    fn argmax_is_the_former_fold() {
        let fold = |v: &[f64]| {
            v.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (k, &x)| if x > b.1 { (k, x) } else { b })
        };
        let mut r = com_core::rng::Rng::new(3);
        for n in 1..40 {
            let v: Vec<f64> = (0..n).map(|_| (r.gaussian() * 3.0).round()).collect();
            assert_eq!(argmax(v.iter().copied()), Some(fold(&v)), "{v:?}");
        }
        assert_eq!(argmax([f64::NAN, f64::NAN]), None);
        assert_eq!(argmax([f64::NEG_INFINITY]), Some((0, f64::NEG_INFINITY)));
    }

    #[test]
    fn bisection_and_golden_section_reach_their_targets() {
        assert_eq!(bisect(|_| true, 0.0, 1.0, 0), 0.5);
        assert!(bisect(|_| true, 0.0, 1.0, 60) > 1.0 - 1e-15);
        let x = golden_max(|x| -(x - 2.5).abs(), -10.0, 10.0, 100);
        assert!((x - 2.5).abs() < 1e-9);
        assert_eq!((gcd(1, 1), gcd(36, 24), gcd(17, 0)), (1, 12, 17));
    }

    #[test]
    fn regula_falsi_finds_the_roots_of_increasing_and_decreasing_functions() {
        let up = regula_falsi(|x| x.exp() - 3.0, 0.0, 5.0, 1e-13, 1e-15, 200);
        assert!((up - 3f64.ln()).abs() < 1e-12);
        let down = regula_falsi(|x| 1.0 - x * x, 0.0, 4.0, 1e-13, 1e-15, 200);
        assert!((down - 1.0).abs() < 1e-12);
        // no step: the middle of the bracket
        assert_eq!(regula_falsi(|x| x, -1.0, 3.0, 0.0, 0.0, 0), 1.0);
    }
}
