//! Special functions: the error function and its inverse, the Gaussian tail, the
//! Fresnel integrals, Bessel functions, the gamma function and the regularized
//! incomplete gamma function, the exponential integral (all re-exported at the root).
//!
//! The power series share one private summation (the terms until the last falls below
//! $10^{-17}$ of the sum), the continued fractions one modified Lentz iteration (the
//! partial denominators floored at $10^{-300}$, until a factor is within $10^{-16}$ of
//! 1), and the periodic integrands one midpoint rule; the order of the operations of
//! each function is the one it had before they were shared (the same results, bit for
//! bit).
//!
//! # References
//!
//! 1. W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical
//!    Recipes*, 3rd ed., Cambridge University Press, 2007, §5.2 (the modified Lentz
//!    method), §6.2 and §6.3.
//! 2. M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*, Dover, 1965.

use crate::search::bisect;
use crate::{C64, cis};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign};

/// The smallest magnitude of a partial denominator of the modified Lentz method: a
/// smaller one is replaced by it (Press et al., §5.2).
const LENTZ_TINY: f64 = 1e-300;

/// The convergence of the modified Lentz method: the last factor $\Delta_{k}$ is within
/// this of 1.
const LENTZ_TOLERANCE: f64 = 1e-16;

/// The convergence of a power series: the last term is below this fraction of the sum.
const SERIES_TOLERANCE: f64 = 1e-17;

/// The real or complex numbers the series and continued fractions are evaluated in.
trait Scalar: Copy + Add<Output = Self> + Mul<Output = Self> + MulAssign + AddAssign + Div<f64, Output = Self> {
    /// [`LENTZ_TINY`] in this field.
    fn tiny() -> Self;
    /// The modulus.
    fn magnitude(self) -> f64;
    /// `a · x`.
    fn scale(a: f64, x: Self) -> Self;
    /// `a / x`.
    fn over(a: f64, x: Self) -> Self;
    /// `1 / x`.
    fn recip(self) -> Self;
    /// `|x - 1|`.
    fn distance_to_one(self) -> f64;
}

impl Scalar for f64 {
    fn tiny() -> f64 {
        LENTZ_TINY
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
    fn scale(a: f64, x: f64) -> f64 {
        a * x
    }
    fn over(a: f64, x: f64) -> f64 {
        a / x
    }
    fn recip(self) -> f64 {
        1.0 / self
    }
    fn distance_to_one(self) -> f64 {
        (self - 1.0).abs()
    }
}

impl Scalar for C64 {
    fn tiny() -> C64 {
        C64::new(LENTZ_TINY, 0.0)
    }
    fn magnitude(self) -> f64 {
        self.norm()
    }
    fn scale(a: f64, x: C64) -> C64 {
        a * x
    }
    fn over(a: f64, x: C64) -> C64 {
        a / x
    }
    fn recip(self) -> C64 {
        self.inv()
    }
    fn distance_to_one(self) -> f64 {
        (self - 1.0).norm()
    }
}

/// The power series $\sum_{k} t_{k} / d_{k}$ from the partial sum `sum`, with $t_{k} =
/// t_{k-1} r_{k}$ from `term` $= t_{0}$, the ratios `ratio(k)` and the divisors
/// `divisor(k)` for $k = 1, 2, \dots$ below `max_terms`, until the last term is below
/// [`SERIES_TOLERANCE`] of the sum.
fn series<T: Scalar>(
    mut term: T,
    mut sum: T,
    max_terms: usize,
    mut ratio: impl FnMut(usize) -> T,
    divisor: impl Fn(usize) -> f64,
) -> T {
    for k in 1..max_terms {
        term *= ratio(k);
        let t = term / divisor(k);
        sum += t;
        if t.magnitude() < SERIES_TOLERANCE * sum.magnitude() {
            break;
        }
    }
    sum
}

/// The modified Lentz method (Press et al., §5.2) for the continued fraction
/// $f = b_{0} + \cfrac{a_{1}}{b_{1} + \cfrac{a_{2}}{b_{2} + \dotsb}}$ from the state
/// $(f_{0}, C_{0}, D_{0})$ and the terms $(a_{k}, b_{k})$ = `term(k)`, $k = 1, 2, \dots$
/// below `max_terms`: $D_{k} = 1/(b_{k} + a_{k} D_{k-1})$, $C_{k} = b_{k} + a_{k}/C_{k-1}$
/// (both floored at [`LENTZ_TINY`]), $f_{k} = f_{k-1} C_{k} D_{k}$, until $|C_{k} D_{k} -
/// 1|$ is below [`LENTZ_TOLERANCE`].
fn lentz<T: Scalar>(f0: T, c0: T, d0: T, max_terms: usize, mut term: impl FnMut(usize) -> (f64, T)) -> T {
    let (mut f, mut c, mut d) = (f0, c0, d0);
    for k in 1..max_terms {
        let (a, b) = term(k);
        d = b + T::scale(a, d);
        if d.magnitude() < LENTZ_TINY {
            d = T::tiny();
        }
        c = b + T::over(a, c);
        if c.magnitude() < LENTZ_TINY {
            c = T::tiny();
        }
        d = d.recip();
        let delta = c * d;
        f *= delta;
        if delta.distance_to_one() < LENTZ_TOLERANCE {
            break;
        }
    }
    f
}

/// The midpoint rule on $n$ points of $\frac{1}{\pi} \int_{0}^{\pi} f(\theta)\, d\theta$,
/// $\frac{1}{n} \sum_{k} f(h (k + \frac{1}{2}))$ with $h = \pi/n$ (exponentially
/// convergent on a smooth periodic integrand).
fn midpoint_mean(f: impl Fn(f64) -> f64, n: usize) -> f64 {
    let h = std::f64::consts::PI / n as f64;
    (0..n).map(|k| f(h * (k as f64 + 0.5))).sum::<f64>() / n as f64
}

/// Complementary error function $\operatorname{erfc}(x) = \frac{2}{\sqrt{\pi}}
/// \int_{x}^{\infty} e^{-t^{2}}\, dt$.
///
/// The rational Chebyshev approximations of Cody (1969), as in his `CALERF` (SPECFUN,
/// 1990): $1 - x R_{1}(x^{2})$ for $|x| \le 0.46875$,
/// $e^{-x^{2}} R_{2}(x)$ up to 4 and $\frac{e^{-x^{2}}}{x} \left(\frac{1}{\sqrt{\pi}} +
/// \frac{1}{x^{2}} R_{3}(1/x^{2})\right)$ beyond, $e^{-x^{2}}$ computed in two factors so
/// that its rounding does not grow with $x^{2}$; $2 - \operatorname{erfc}(-x)$ for
/// $x < -0.46875$. Its relative error is below $10^{-15}$ wherever the result is a
/// normal number (against mpmath, the tail included: $\operatorname{erfc}(26) \approx
/// 5.7 \cdot 10^{-296}$); it underflows to 0 beyond 27.3.
///
/// # Examples
/// ```
/// use com_math::erfc;
/// assert!((erfc(1.0) / 0.157_299_207_050_285_13 - 1.0).abs() < 1e-15);
/// assert!((erfc(10.0) / 2.088_487_583_762_544_8e-45 - 1.0).abs() < 1e-15);
/// ```
///
/// # References
///
/// 1. W. J. Cody, "Rational Chebyshev approximations for the error function," *Math.
///    Comp.*, 23(107), 631–637, 1969.
/// 2. W. J. Cody, "Algorithm 715: SPECFUN — a portable FORTRAN package of special
///    function routines and test drivers," *ACM Trans. Math. Softw.*, 19(1), 22–32, 1993.
#[allow(
    clippy::unreadable_literal,
    clippy::excessive_precision,
    reason = "the coefficients of Cody's CALERF, digit for digit"
)]
pub fn erfc(x: f64) -> f64 {
    const A: [f64; 5] = [
        3.16112374387056560e00,
        1.13864154151050156e02,
        3.77485237685302021e02,
        3.20937758913846947e03,
        1.85777706184603153e-1,
    ];
    const B: [f64; 4] =
        [2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03];
    const C: [f64; 9] = [
        5.64188496988670089e-1,
        8.88314979438837594e00,
        6.61191906371416295e01,
        2.98635138197400131e02,
        8.81952221241769090e02,
        1.71204761263407058e03,
        2.05107837782607147e03,
        1.23033935479799725e03,
        2.15311535474403846e-8,
    ];
    const D: [f64; 8] = [
        1.57449261107098347e01,
        1.17693950891312499e02,
        5.37181101862009858e02,
        1.62138957456669019e03,
        3.29079923573345963e03,
        4.36261909014324716e03,
        3.43936767414372164e03,
        1.23033935480374942e03,
    ];
    const P: [f64; 6] = [
        3.05326634961232344e-1,
        3.60344899949804439e-1,
        1.25781726111229246e-1,
        1.60837851487422766e-2,
        6.58749161529837803e-4,
        1.63153871373020978e-2,
    ];
    const Q: [f64; 5] = [
        2.56852019228982242e00,
        1.87295284992346725e00,
        5.27905102951428412e-1,
        6.05183413124413191e-2,
        2.33520497626869185e-3,
    ];
    const FRAC_1_SQRT_PI: f64 = 5.6418958354775628695e-1;
    let y = x.abs();
    if y <= 0.46875 {
        let ysq = if y > 1.11e-16 { y * y } else { 0.0 };
        let (mut num, mut den) = (A[4] * ysq, ysq);
        for i in 0..3 {
            num = (num + A[i]) * ysq;
            den = (den + B[i]) * ysq;
        }
        return 1.0 - x * (num + A[3]) / (den + B[3]);
    }
    // erfc(27.3) is below the smallest subnormal number
    let r = if y > 27.3 {
        0.0
    } else {
        let r = if y <= 4.0 {
            let (mut num, mut den) = (C[8] * y, y);
            for i in 0..7 {
                num = (num + C[i]) * y;
                den = (den + D[i]) * y;
            }
            (num + C[7]) / (den + D[7])
        } else {
            let ysq = 1.0 / (y * y);
            let (mut num, mut den) = (P[5] * ysq, ysq);
            for i in 0..4 {
                num = (num + P[i]) * ysq;
                den = (den + Q[i]) * ysq;
            }
            (FRAC_1_SQRT_PI - ysq * (num + P[4]) / (den + Q[4])) / y
        };
        // e^{-y²} = e^{-u²} e^{-(y - u)(y + u)} with u = y rounded down to 1/16: exact u²
        let u = (y * 16.0).trunc() / 16.0;
        (-u * u).exp() * (-(y - u) * (y + u)).exp() * r
    };
    if x < 0.0 { 2.0 - r } else { r }
}

/// Inverse of [`erfc`] on `(0, 2)`, by bisection (`±∞` at the ends).
pub fn erfcinv(y: f64) -> f64 {
    if y <= 0.0 {
        return f64::INFINITY;
    }
    if y >= 2.0 {
        return f64::NEG_INFINITY;
    }
    bisect(|x| erfc(x) > y, -27.0, 27.0, 200)
}

/// Gaussian tail probability `Q(x) = P(N(0,1) > x) = ½ erfc(x/√2)`.
#[inline]
pub fn qfunc(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

/// Inverse of [`qfunc`] on $(0, 1)$: $Q^{-1}(p) = \sqrt{2}\, \operatorname{erfc}^{-1}(2p)$
/// (`±∞` at the ends).
///
/// # Examples
/// ```
/// use com_math::{qfunc, qfuncinv};
/// assert!((qfuncinv(0.025) - 1.959_964).abs() < 1e-6);
/// assert!((qfunc(qfuncinv(1e-6)) / 1e-6 - 1.0).abs() < 1e-9);
/// ```
#[inline]
pub fn qfuncinv(p: f64) -> f64 {
    std::f64::consts::SQRT_2 * erfcinv(2.0 * p)
}

/// The standard normal density $\varphi(x) = e^{-x^{2}/2}/\sqrt{2\pi}$ (its
/// distribution function is $1 - Q(x)$, [`qfunc`]).
///
/// # Examples
/// ```
/// use com_math::normal_pdf;
/// assert!((normal_pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-16);
/// ```
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt()
}

/// The Fresnel integrals $C(x) = \int_{0}^{x} \cos(\pi t^{2}/2)\, dt$ and
/// $S(x) = \int_{0}^{x} \sin(\pi t^{2}/2)\, dt$, as `(C, S)`, to about $10^{-15}$.
///
/// For $|x| < 1.5$ the power series $C + jS = \sum_{k} (j\pi/2)^{k} x^{2k+1} / (k!\,(2k+1))$
/// (Abramowitz & Stegun 7.3.11, 7.3.13); beyond, the relation to the error function
/// $C + jS = \frac{1+j}{2} \operatorname{erf}(z)$, $z = \frac{\sqrt{\pi}}{2}(1-j)x$
/// (A&S 7.3.22), with $\operatorname{erfc}(z) = \frac{e^{-z^{2}}}{\sqrt{\pi}}
/// \cfrac{1}{z + \cfrac{1/2}{z + \cfrac{1}{z + \cfrac{3/2}{z + \dotsb}}}}$
/// (A&S 7.1.14, valid for $\operatorname{Re} z > 0$), evaluated by the modified Lentz
/// method. $C$ and $S$ are odd.
pub fn fresnel(x: f64) -> (f64, f64) {
    let ax = x.abs();
    let v = if ax < 1.5 {
        let a = C64::new(0.0, std::f64::consts::FRAC_PI_2 * ax * ax);
        series(C64::new(ax, 0.0), C64::new(ax, 0.0), 60, |k| a / k as f64, |k| (2 * k + 1) as f64)
    } else {
        let z = C64::new(1.0, -1.0) * (std::f64::consts::PI.sqrt() / 2.0 * ax);
        // modified Lentz for z + (1/2)/(z + 1/(z + (3/2)/(z + ...)))
        let f = lentz(z, z, C64::new(0.0, 0.0), 100_000, |k| (k as f64 / 2.0, z));
        // e^{-z²} = e^{jπx²/2}
        let erfc = cis(std::f64::consts::FRAC_PI_2 * ax * ax) / (std::f64::consts::PI.sqrt() * f);
        C64::new(0.5, 0.5) * (1.0 - erfc)
    };
    (v.re.copysign(x), v.im.copysign(x))
}

/// The Bessel function of the first kind of order zero, $J_{0}(x) = \frac{1}{\pi}
/// \int_{0}^{\pi} \cos(x \sin\theta)\, d\theta$, by the midpoint rule on that
/// periodic integrand (exponentially convergent: $64 + 2|x|$ points give the machine
/// precision), as for the correlation $J_{0}(2\pi f_{D} \tau)$ of Jakes' fading.
pub fn bessel_j0(x: f64) -> f64 {
    let n = 64 + 2 * x.abs().ceil() as usize;
    midpoint_mean(|theta| (x * theta.sin()).cos(), n)
}

/// The natural logarithm of the modified Bessel function of the second kind $K_{\nu}(x)$
/// of real order $\nu$ and $x > 0$, from $K_{\nu}(x) = \int_{0}^{\infty} e^{-x \cosh t}
/// \cosh(\nu t)\, dt$ by the trapezoidal rule (exponentially convergent on this even,
/// entire integrand), summed in the log domain so that neither the large values at
/// small $x$ and large $\nu$ nor the small ones at large $x$ overflow; to about
/// $10^{-13}$ relative. The limits: $-\infty$ for $x = +\infty$ ($K_{\nu}(x) \to 0$),
/// $+\infty$ for an infinite order, and the leading term $\ln\left(\tfrac{1}{2}
/// \Gamma(\nu)\right) + \nu \ln(2/x)$ when $\nu / x$ overflows (a subnormal $x$; for
/// $\nu \ge 0.05$, where the next term is below $e^{-70}$ relative); `NaN` for
/// $x \le 0$, a `NaN` argument, or an overflowing $\nu / x$ with a smaller order.
///
/// # Examples
/// ```
/// use com_math::{bessel_k, ln_bessel_k};
/// // K₀(1) and K₁(1) (Abramowitz & Stegun, Table 9.8); K_{1/2}(x) = √(π/2x) e^{-x}
/// assert!((bessel_k(0.0, 1.0) - 0.421_024_438_240_708_3).abs() < 1e-13);
/// assert!((bessel_k(1.0, 1.0) - 0.601_907_230_197_234_6).abs() < 1e-13);
/// let x = 3.7;
/// let half = (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp();
/// assert!((bessel_k(0.5, x) / half - 1.0).abs() < 1e-13);
/// // small x: K_ν(x) ≈ ½ Γ(ν) (2/x)^ν, here about 10^{129}
/// let asymptotic = (0.5f64).ln() + com_math::ln_gamma(30.0) + 30.0 * 2000f64.ln();
/// assert!((ln_bessel_k(30.0, 1e-3) - asymptotic).abs() < 1e-3);
/// ```
pub fn ln_bessel_k(nu: f64, x: f64) -> f64 {
    if x.is_nan() || nu.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let nu = nu.abs();
    if x == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    if nu == f64::INFINITY {
        return f64::INFINITY;
    }
    if nu >= 0.05 && !(nu / x).is_finite() {
        // K_ν(x) = ½ Γ(ν) (2/x)^ν (1 + O((x/2)^{2ν} + x²)), the correction below e^{-70}
        return -std::f64::consts::LN_2 + ln_gamma(nu) + nu * (std::f64::consts::LN_2 - x.ln());
    }
    // ln cosh(νt) - x cosh t, without overflow
    let f = |t: f64| {
        let a = nu * t;
        a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2 - x * t.cosh()
    };
    // the maximum of the integrand: sinh t = (ν/x) tanh(νt) ≈ ν/x
    let t_max = if nu > 0.0 { (nu / x).asinh() } else { 0.0 };
    let peak = f(t_max);
    // step: a fraction of the width of the peak, (x cosh t_max)^{-1/2}
    let width = 1.0 / (x * t_max.cosh()).sqrt().max(1.0);
    let h = 0.1 * width.min(1.0);
    let mut acc = 0.5 * (f(0.0) - peak).exp();
    let mut k = 1usize;
    loop {
        let t = k as f64 * h;
        let v = f(t) - peak;
        if v.is_nan() {
            return f64::NAN;
        }
        acc += v.exp();
        if t > t_max && v < -50.0 {
            break;
        }
        k += 1;
    }
    peak + (acc * h).ln()
}

/// The modified Bessel function of the second kind $K_{\nu}(x)$ of real order:
/// `exp(`[`ln_bessel_k`]`)`.
pub fn bessel_k(nu: f64, x: f64) -> f64 {
    ln_bessel_k(nu, x).exp()
}

/// The scaled modified Bessel function of the first kind of order zero,
/// $e^{-|x|} I_{0}(x)$, with $I_{0}(x) = \frac{1}{\pi} \int_{0}^{\pi} e^{x\cos\theta}
/// \, d\theta$: the midpoint rule on the periodic integrand
/// $e^{|x|(\cos\theta - 1)}$ for $|x| \le 60$ (exponentially convergent), and the
/// asymptotic series $\frac{1}{\sqrt{2\pi x}} \sum_{k} \frac{((2k-1)!!)^{2}}{k!
/// \, (8x)^{k}}$ beyond; to about $10^{-14}$ relative. The density of the Rice
/// distribution, $y e^{-(y - v)^{2}/2} \, e^{-yv} I_{0}(yv)$, needs the scaled form.
///
/// # Examples
/// ```
/// use com_math::bessel_i0e;
/// // I₀(1) = 1.266065877752008, I₀(10) = 2815.716628466254
/// assert!((bessel_i0e(1.0) * 1f64.exp() - 1.266_065_877_752_008).abs() < 1e-13);
/// assert!((bessel_i0e(10.0) * 10f64.exp() / 2815.716_628_466_254 - 1.0).abs() < 1e-13);
/// assert!((bessel_i0e(1e4) * (2.0 * std::f64::consts::PI * 1e4).sqrt() - 1.0).abs() < 2e-5);
/// ```
pub fn bessel_i0e(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 60.0 {
        let n = 32 + 8 * ax.sqrt().ceil() as usize + ax.ceil() as usize;
        midpoint_mean(|theta| (ax * (theta.cos() - 1.0)).exp(), n)
    } else {
        // Σ_k ((2k-1)!!)² / (k! (8x)^k): the terms fall fast for x > 60
        let ratio = |k: usize| {
            let odd = (2 * k - 1) as f64;
            odd * odd / (k as f64 * 8.0 * ax)
        };
        series(1.0, 1.0, 30, ratio, |_| 1.0) / (std::f64::consts::TAU * ax).sqrt()
    }
}

/// The error function $\operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x}
/// e^{-t^{2}}\, dt$: its Maclaurin series $\frac{2}{\sqrt{\pi}} \sum_{k} \frac{(-1)^{k}
/// x^{2k+1}}{k!\,(2k+1)}$ for $|x| < 2$ (relative error about $10^{-15}$, small
/// arguments included), $1 - \operatorname{erfc}(x)$ beyond (absolute error about
/// $10^{-16}$). Odd.
///
/// # Examples
/// ```
/// use com_math::erf;
/// assert!((erf(1e-4) / 1.128_379_163_334_248_7e-4 - 1.0).abs() < 1e-14);
/// assert!((erf(0.9) - 0.796_908_212_422_831_9).abs() < 1e-15);
/// ```
pub fn erf(x: f64) -> f64 {
    if x.abs() >= 2.0 {
        return (1.0 - erfc(x.abs())).copysign(x);
    }
    let x2 = x * x;
    series(x, x, 60, |k| -x2 / k as f64, |k| (2 * k + 1) as f64) * std::f64::consts::FRAC_2_SQRT_PI
}

/// The logarithm of the gamma function, $\ln \Gamma(x)$ for $x > 0$, by the Lanczos
/// approximation ($g = 7$, nine coefficients; relative error about $10^{-15}$) and the
/// reflection formula $\Gamma(x)\Gamma(1-x) = \pi / \sin(\pi x)$ below $1/2$.
///
/// # Examples
/// ```
/// use com_math::ln_gamma;
/// // Γ(10) = 9! = 362 880, Γ(1/2) = √π
/// assert!((ln_gamma(10.0) - 362_880f64.ln()).abs() < 1e-13);
/// assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
/// ```
#[allow(clippy::unreadable_literal, reason = "the Lanczos coefficients, digit for digit")]
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + G + 0.5;
    let a = C[0] + C.iter().enumerate().skip(1).map(|(i, c)| c / (x + i as f64)).sum::<f64>();
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// The regularized lower incomplete gamma function
/// $P(a, x) = \frac{1}{\Gamma(a)} \int_{0}^{x} u^{a-1} e^{-u}\, du$, $a > 0$, $x \ge 0$:
/// the distribution function of a gamma variable of shape $a$ and unit scale (a
/// $\chi^{2}$ variable with $2a$ degrees of freedom, halved). The series
/// $e^{-x} x^{a} \sum_{k} x^{k} / \Gamma(a + k + 1)$ for $x < a + 1$, and $1 - Q(a, x)$
/// with the continued fraction of $Q$ (modified Lentz) beyond (Press et al., *Numerical
/// Recipes*, §6.2); relative error about $10^{-14}$ on the side computed directly.
///
/// # Examples
/// ```
/// use com_math::gamma_p;
/// // P(1, x) = 1 - e^{-x}
/// assert!((gamma_p(1.0, 0.5) - (1.0 - (-0.5f64).exp())).abs() < 1e-15);
/// assert!((gamma_p(20.0, 5.0) / 3.452_135_820_914_455e-7 - 1.0).abs() < 1e-12);
/// ```
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let ln_front = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        let mut ap = a;
        let ratio = |_| {
            ap += 1.0;
            x / ap
        };
        series(1.0 / a, 1.0 / a, 10_001, ratio, |_| 1.0) * ln_front.exp()
    } else {
        // Q(a, x) = e^{-x} x^a / Γ(a) · 1/(x+1-a- 1·(1-a)/(x+3-a- 2(2-a)/(x+5-a- …)))
        let mut b = x + 1.0 - a;
        let d = 1.0 / b;
        let term = |i: usize| {
            b += 2.0;
            (-(i as f64) * (i as f64 - a), b)
        };
        let h = lentz(d, 1.0 / LENTZ_TINY, d, 10_000, term);
        1.0 - ln_front.exp() * h
    }
}

/// The exponential integral $E_{1}(x) = \int_{x}^{\infty} \frac{e^{-t}}{t}\, dt$,
/// $x > 0$: the series $-\gamma - \ln x - \sum_{k \ge 1} \frac{(-x)^{k}}{k\, k!}$ for
/// $x \le 1$, the continued fraction $\cfrac{e^{-x}}{x + 1 - \cfrac{1}{x + 3 -
/// \cfrac{4}{x + 5 - \dotsb}}}$ beyond (Press et al., *Numerical Recipes*, §6.3);
/// relative error about $10^{-15}$. $e^{1/\rho} E_{1}(1/\rho)$ is the ergodic capacity
/// (nats) of the Rayleigh channel at the mean SNR $\rho$.
///
/// # Examples
/// ```
/// use com_math::expint_e1;
/// assert!((expint_e1(1.0) - 0.219_383_934_395_520_5).abs() < 1e-15);
/// assert!((expint_e1(10.0) / 4.156_968_929_685_325e-6 - 1.0).abs() < 1e-14);
/// ```
pub fn expint_e1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x <= 1.0 {
        const EULER: f64 = 0.577_215_664_901_532_9;
        -EULER - x.ln() - series(1.0, 0.0, 60, |k| -x / k as f64, |k| k as f64)
    } else {
        // e^{-x} · 1/(x + 1 - 1/(x + 3 - 4/(x + 5 - …)))
        let mut b = x + 1.0;
        let d = 1.0 / b;
        let term = |i: usize| {
            b += 2.0;
            (-((i * i) as f64), b)
        };
        lentz(d, 1.0 / LENTZ_TINY, d, 10_000, term) * (-x).exp()
    }
}
