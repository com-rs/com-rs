//! One-dimensional searches: bisection and golden-section maximisation, and the
//! greatest common divisor.
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
//! [`golden_max`] shrinks a bracket of a unimodal function by the golden ratio
//! $\varphi^{-1} = (\sqrt{5} - 1)/2$ at every step, with the two interior points
//! $c = b - \varphi^{-1}(b - a)$ and $d = a + \varphi^{-1}(b - a)$; one evaluation per
//! step, the bracket length multiplied by $\varphi^{-n}$ after $n$ steps (Kiefer 1953).
//!
//! Both keep the order of their operations (`0.5 * (lo + hi)`, no `f64::midpoint`;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisection_and_golden_section_reach_their_targets() {
        assert_eq!(bisect(|_| true, 0.0, 1.0, 0), 0.5);
        assert!(bisect(|_| true, 0.0, 1.0, 60) > 1.0 - 1e-15);
        let x = golden_max(|x| -(x - 2.5).abs(), -10.0, 10.0, 100);
        assert!((x - 2.5).abs() < 1e-9);
        assert_eq!((gcd(1, 1), gcd(36, 24), gcd(17, 0)), (1, 12, 17));
    }
}
