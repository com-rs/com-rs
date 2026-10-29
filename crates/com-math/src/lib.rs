//! # com-math — numerical methods without communication semantics
//!
//! Special functions, quadrature, FFT plans, nearest-point search and the split-step
//! solver of the nonlinear Schrödinger equation. Nothing here knows what a symbol or a
//! channel is: the crates above give these methods their meaning.
//!
//! # Contents
//!
//! | Module | Content |
//! |---|---|
//! | [`fft`] | cached FFT plans with the NumPy conventions, `fftfreq` |
//! | [`linalg`] | QR, Cholesky, Hermitian systems and the SVD of small complex matrices |
//! | [`nearest`] | [`nearest::Slicer`], the nearest point of a finite set |
//! | [`nlse`] | the symmetric split-step Fourier solvers of the scalar NLSE and of the Manakov equation |
//! | [`gf2`] | polynomials and matrices over GF(2) |
//! | [`gf2m`] | the Galois fields $\mathrm{GF}(2^{m})$ of the BCH and Reed–Solomon codes |
//! | [`bigint`] | unsigned integers of arbitrary size, for enumerative coding |
//! | [`bits`] | the Gray code [`gray`], [`gray_inverse`], and the conversions [`bits_to_index`], [`index_to_bits`] (most significant bit first) |
//! | [`search`] | [`bisect`], [`golden_max`] and [`gcd`] |
//! | [`sequences`] | linear-feedback shift registers and the Gold sequence of 3GPP |
//!
//! At the root:
//!
//! - the log domain: [`log_sum_exp`] (`ln Σ e^{v_i}`), [`ln_1p_exp`] (`ln(1 + e^x)`),
//!   [`max_star`] (`ln(e^a + e^b)`), all without overflow;
//! - [`C64`]: the complex type of every signal (`num_complex::Complex<f64>`), and
//!   [`cis`], `e^{jθ}`;
//! - decibels: [`db2lin`], [`lin2db`] (power ratios), [`db2amp`], [`amp2db`] (amplitude
//!   ratios), [`dbm2w`], [`w2dbm`], [`db2np`], [`np2db`] (attenuation per unit length);
//! - [`sinc`], `sin(πx)/(πx)`, [`diric`], the periodic sinc `sin(nx/2)/(n sin(x/2))`, and
//!   [`effective_length`], `(1 - e^{-αL})/α`;
//! - special functions: [`erf`], [`erfc`], [`erfcinv`], [`qfunc`] (the Gaussian tail
//!   `Q(x) = ½ erfc(x/√2)`), its inverse [`qfuncinv`], [`fresnel`] (the Fresnel
//!   integrals $C$ and $S$), [`bessel_j0`], [`bessel_i0e`], [`ln_gamma`], [`gamma_p`] (the regularized incomplete gamma
//!   function), [`expint_e1`] (the exponential integral);
//! - [`entropy`], $-\sum_{i} p_{i} \log_{2} p_{i}$;
//! - signal statistics: [`power`] (mean `|x|²`), [`energy`], [`papr_db`];
//! - quadrature: [`simpson`], [`try_simpson`] and [`simpson_weight`] (composite
//!   Simpson's rule), [`gauss_hermite`] (`∫ f(t) e^{-t²} dt`) and [`gauss_legendre`]
//!   (`∫_{-1}^{1} f(t) dt`);
//! - physical constants [`PLANCK`], [`BOLTZMANN`], [`ELEMENTARY_CHARGE`],
//!   [`SPEED_OF_LIGHT`].
//!
//! # Examples
//! ```
//! use com_math::*;
//! assert!((db2lin(3.0) - 1.9953).abs() < 1e-4);
//! assert!((dbm2w(0.0) - 1e-3).abs() < 1e-15);
//! assert!((qfunc(0.0) - 0.5).abs() < 1e-7);
//! let z = cis(std::f64::consts::FRAC_PI_2);
//! assert!((z - C64::new(0.0, 1.0)).norm() < 1e-15);
//! ```

pub mod bigint;
pub mod bits;
pub mod fft;
pub mod gf2;
pub mod gf2m;
pub mod linalg;
pub mod nearest;
pub mod nlse;
pub mod search;
pub mod sequences;

pub use bits::{bits_to_index, gray, gray_inverse, index_to_bits};
pub use search::{bisect, gcd, golden_max};

pub use com_core::C64;

/// Planck constant (J·s).
pub const PLANCK: f64 = 6.626_070_15e-34;
/// Boltzmann constant (J/K).
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Elementary charge (C).
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// Speed of light in vacuum (m/s).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// `e^{jθ} = cos θ + j sin θ`.
#[inline]
pub fn cis(theta: f64) -> C64 {
    let (s, c) = theta.sin_cos();
    C64::new(c, s)
}

/// Power ratio in dB to linear: `10^{x/10}`.
#[inline]
pub fn db2lin(x_db: f64) -> f64 {
    10f64.powf(x_db / 10.0)
}

/// Linear power ratio to dB: `10 log10(x)`.
#[inline]
pub fn lin2db(x: f64) -> f64 {
    10.0 * x.log10()
}

/// Amplitude ratio in dB to linear: `10^{x/20}`.
#[inline]
pub fn db2amp(x_db: f64) -> f64 {
    10f64.powf(x_db / 20.0)
}

/// Linear amplitude ratio to dB: `20 log10(x)`.
#[inline]
pub fn amp2db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// Attenuation in dB per unit length to nepers (power): `α_dB / (10 log10 e)`.
#[inline]
pub fn db2np(x_db: f64) -> f64 {
    x_db / (10.0 * std::f64::consts::LOG10_E)
}

/// Attenuation in nepers (power) per unit length to dB: `10 log10(e) · α`.
#[inline]
pub fn np2db(x_np: f64) -> f64 {
    10.0 * std::f64::consts::LOG10_E * x_np
}

/// The normalised sinc `sin(πx)/(πx)`, 1 at 0.
#[inline]
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) }
}

/// The effective length `(1 - e^{-αL})/α` of an attenuation `α` over `length` (`L`
/// itself when `αL` is negligible).
#[inline]
pub fn effective_length(alpha: f64, length: f64) -> f64 {
    if alpha * length < 1e-12 { length } else { (1.0 - (-alpha * length).exp()) / alpha }
}

/// Power in dBm to watts: `10^{P/10} · 1 mW`.
#[inline]
pub fn dbm2w(p_dbm: f64) -> f64 {
    1e-3 * db2lin(p_dbm)
}

/// Power in watts to dBm.
#[inline]
pub fn w2dbm(p_w: f64) -> f64 {
    lin2db(p_w / 1e-3)
}

/// Mean power `(1/N) Σ |x_n|²` of a complex signal (0 for an empty slice).
pub fn power(x: &[C64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    energy(x) / x.len() as f64
}

/// Energy `Σ |x_n|²` of a complex signal.
pub fn energy(x: &[C64]) -> f64 {
    x.iter().map(C64::norm_sqr).sum()
}

/// Complementary error function `erfc(x) = (2/√π) ∫_x^∞ e^{-t²} dt`.
///
/// Chebyshev fit of Press et al. (*Numerical Recipes*, §6.2), with a relative error
/// below `1.2·10⁻⁷` on the whole real line, which is ample for error rates down to
/// `10⁻³⁰⁰`.
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { r } else { 2.0 - r }
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

/// Composite Simpson's rule $\int_{a}^{b} f(t)\, dt$ on `n` intervals (`n` even, at
/// least 2), exact for cubics.
///
/// # Examples
/// ```
/// use com_math::simpson;
/// assert!((simpson(|t| t * t * t, 0.0, 2.0, 2) - 4.0).abs() < 1e-15);
/// assert!((simpson(f64::sin, 0.0, std::f64::consts::PI, 200) - 2.0).abs() < 1e-8);
/// ```
pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    try_simpson(|t| Ok::<f64, std::convert::Infallible>(f(t)), a, b, n).unwrap_or_else(|never| match never {})
}

/// [`simpson`] of a fallible integrand: the first error stops the quadrature.
///
/// # Errors
/// The first error of `f`.
///
/// # Examples
/// ```
/// use com_math::try_simpson;
/// let v: Result<f64, String> = try_simpson(|t| Ok(t * t), 0.0, 3.0, 4);
/// assert!((v.unwrap() - 9.0).abs() < 1e-14);
/// let e: Result<f64, String> = try_simpson(|t| if t > 1.0 { Err("beyond 1".into()) } else { Ok(t) }, 0.0, 2.0, 4);
/// assert!(e.is_err());
/// ```
pub fn try_simpson<E>(f: impl Fn(f64) -> Result<f64, E>, a: f64, b: f64, n: usize) -> Result<f64, E> {
    let n = n.max(2).div_ceil(2) * 2;
    let h = (b - a) / n as f64;
    let mut acc = f(a)? + f(b)?;
    for k in 1..n {
        acc += simpson_weight(k, n) * f(a + k as f64 * h)?;
    }
    Ok(acc * h / 3.0)
}

/// The weight of node `k` of Simpson's rule on `n` intervals (`n` even), before the
/// factor $h/3$: 1 at the ends, 4 at the odd nodes, 2 at the even inner ones.
#[inline]
pub fn simpson_weight(k: usize, n: usize) -> f64 {
    if k == 0 || k == n {
        1.0
    } else if k % 2 == 1 {
        4.0
    } else {
        2.0
    }
}

/// The Dirichlet (periodic sinc) function of MATLAB, $\operatorname{diric}(x, n) =
/// \frac{\sin(n x/2)}{n \sin(x/2)}$, of period $2\pi$ ($4\pi$) for odd (even) $n$, with its
/// limits $(-1)^{k(n-1)}$ at $x = 2\pi k$; $\sin(n\theta)/\sin\theta = n
/// \operatorname{diric}(2\theta, n)$ is the array factor of $n$ equally spaced terms.
///
/// # Examples
/// ```
/// use com_math::diric;
/// assert_eq!(diric(0.0, 8), 1.0);
/// assert_eq!(diric(2.0 * std::f64::consts::PI, 8), -1.0);
/// let x = 0.3;
/// assert!((diric(x, 5) - (2.5 * x).sin() / (5.0 * (0.5 * x).sin())).abs() < 1e-15);
/// ```
pub fn diric(x: f64, n: usize) -> f64 {
    let half = 0.5 * x;
    let den = half.sin();
    let nf = n as f64;
    if den.abs() < 1e-12 {
        // x = 2πk: the limit (-1)^{k(n-1)}
        let k = (half / std::f64::consts::PI).round() as i64;
        return if (k * (n as i64 - 1)).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
    }
    (nf * half).sin() / (nf * den)
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
        let (mut term, mut sum) = (C64::new(ax, 0.0), C64::new(ax, 0.0));
        for k in 1..60 {
            term *= a / k as f64;
            let t = term / (2 * k + 1) as f64;
            sum += t;
            if t.norm() < 1e-17 * sum.norm() {
                break;
            }
        }
        sum
    } else {
        let z = C64::new(1.0, -1.0) * (std::f64::consts::PI.sqrt() / 2.0 * ax);
        // modified Lentz for z + (1/2)/(z + 1/(z + (3/2)/(z + ...)))
        let tiny = 1e-300;
        let (mut f, mut c, mut d) = (z, z, C64::new(0.0, 0.0));
        for k in 1..100_000 {
            let a = k as f64 / 2.0;
            d = z + a * d;
            if d.norm() < tiny {
                d = C64::new(tiny, 0.0);
            }
            c = z + a / c;
            if c.norm() < tiny {
                c = C64::new(tiny, 0.0);
            }
            d = d.inv();
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).norm() < 1e-16 {
                break;
            }
        }
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
    let h = std::f64::consts::PI / n as f64;
    (0..n).map(|k| (x * (h * (k as f64 + 0.5)).sin()).cos()).sum::<f64>() / n as f64
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
        let h = std::f64::consts::PI / n as f64;
        (0..n).map(|k| (ax * ((h * (k as f64 + 0.5)).cos() - 1.0)).exp()).sum::<f64>() / n as f64
    } else {
        // Σ_k ((2k-1)!!)² / (k! (8x)^k): the terms fall fast for x > 60
        let mut term = 1.0;
        let mut sum = 1.0;
        for k in 1..30 {
            let odd = (2 * k - 1) as f64;
            term *= odd * odd / (k as f64 * 8.0 * ax);
            sum += term;
            if term < 1e-17 * sum {
                break;
            }
        }
        sum / (std::f64::consts::TAU * ax).sqrt()
    }
}

/// The error function $\operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x}
/// e^{-t^{2}}\, dt$: its Maclaurin series $\frac{2}{\sqrt{\pi}} \sum_{k} \frac{(-1)^{k}
/// x^{2k+1}}{k!\,(2k+1)}$ for $|x| < 2$ (relative error about $10^{-15}$, small
/// arguments included), $1 - \operatorname{erfc}(x)$ beyond (absolute error below
/// $10^{-9}$). Odd.
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
    let (mut term, mut sum) = (x, x);
    for k in 1..60 {
        term *= -x2 / k as f64;
        let t = term / (2 * k + 1) as f64;
        sum += t;
        if t.abs() < 1e-17 * sum.abs() {
            break;
        }
    }
    sum * std::f64::consts::FRAC_2_SQRT_PI
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
        let (mut ap, mut term) = (a, 1.0 / a);
        let mut sum = term;
        for _ in 0..10_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term < sum * 1e-17 {
                break;
            }
        }
        sum * ln_front.exp()
    } else {
        // Q(a, x) = e^{-x} x^a / Γ(a) · 1/(x+1-a- 1·(1-a)/(x+3-a- 2(2-a)/(x+5-a- …)))
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
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
        let (mut term, mut sum) = (1.0, 0.0);
        for k in 1..60 {
            term *= -x / k as f64;
            let t = term / k as f64;
            sum += t;
            if t.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        -EULER - x.ln() - sum
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -((i * i) as f64);
            b += 2.0;
            d = 1.0 / (an * d + b);
            c = b + an / c;
            let delta = c * d;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h * (-x).exp()
    }
}

/// Peak-to-average power ratio `max|x|² / mean|x|²` in dB (`-inf` for an empty or
/// zero block).
///
/// # Examples
/// ```
/// use com_math::{papr_db, C64};
/// let x = [C64::new(2.0, 0.0), C64::new(0.0, 0.0)];
/// assert!((papr_db(&x) - 10.0 * 2f64.log10()).abs() < 1e-12);   // peak 4, mean 2
/// ```
pub fn papr_db(x: &[C64]) -> f64 {
    let peak = x.iter().map(C64::norm_sqr).fold(0.0, f64::max);
    let mean = power(x);
    if mean > 0.0 { lin2db(peak / mean) } else { f64::NEG_INFINITY }
}

/// Nodes and weights of the `n`-point Gauss–Hermite rule,
/// `∫ f(t) e^{-t²} dt ≈ Σ_k w_k f(t_k)`, exact for polynomials of degree `2n - 1`.
///
/// The nodes are the roots of the Hermite polynomial `H_n`, found by Newton's method
/// on the orthonormal recurrence (Numerical Recipes, `gauher`); they come in
/// decreasing order. An expectation over `X ~ N(0, s²)` is
/// `E f(X) ≈ Σ_k w_k f(√2 s t_k) / √π`.
///
/// # Examples
/// ```
/// use com_math::gauss_hermite;
/// let (t, w) = gauss_hermite(20);
/// let pi = std::f64::consts::PI;
/// assert!((w.iter().sum::<f64>() - pi.sqrt()).abs() < 1e-12);
/// // E X⁴ = 3 for X ~ N(0, 1)
/// let m4: f64 = t.iter().zip(&w).map(|(t, w)| w * (2f64.sqrt() * t).powi(4)).sum::<f64>() / pi.sqrt();
/// assert!((m4 - 3.0).abs() < 1e-10);
/// ```
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    const PIM4: f64 = 0.751_125_544_464_942_5; // π^(-1/4)
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    let mut z = 0.0f64;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.85575 * (2.0 * nf + 1.0).powf(-0.16667),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut pp = 1.0;
        for _ in 0..200 {
            let (mut p1, mut p2) = (PIM4, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = z * (2.0 / (jf + 1.0)).sqrt() * p2 - (jf / (jf + 1.0)).sqrt() * p3;
            }
            pp = (2.0 * nf).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 1e-15 * z.abs().max(1.0) {
                break;
            }
        }
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// Nodes and weights of the `n`-point Gauss–Legendre rule on $[-1, 1]$,
/// `∫ f(t) dt ≈ Σ_k w_k f(t_k)`, exact for polynomials of degree `2n - 1`.
///
/// The nodes are the roots of the Legendre polynomial $P_{n}$, found by Newton's method
/// on its recurrence (Numerical Recipes, `gauleg`); they come in increasing order. On
/// $[a, b]$ the nodes are $\frac{a + b}{2} + \frac{b - a}{2} t_{k}$ and the weights
/// $\frac{b - a}{2} w_{k}$.
///
/// # Examples
/// ```
/// use com_math::gauss_legendre;
/// let (t, w) = gauss_legendre(5);
/// assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
/// // exact for t⁸: 2/9
/// let m8: f64 = t.iter().zip(&w).map(|(t, w)| w * t.powi(8)).sum();
/// assert!((m8 - 2.0 / 9.0).abs() < 1e-14);
/// assert!((t[4] - 0.906_179_845_938_664).abs() < 1e-15);
/// ```
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut pp = 1.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf + 1.0) * z * p2 - jf * p3) / (jf + 1.0);
            }
            pp = nf * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// Quadrature nodes `(w, weight)` of Gaussian noise, `E f(W) ≈ Σ weight f(w)`, from the
/// `n`-point [`gauss_hermite`] rule: `W ~ CN(0, σ²)` by the product rule on the two
/// dimensions (`complex`), or `W ~ N(0, σ²)` on the real axis. The nodes of negligible
/// weight (below `1e-300`) are left out.
///
/// # Examples
/// ```
/// use com_math::gaussian_nodes;
/// // E|W|² = σ² for W ~ CN(0, σ²); E W² = σ² for W ~ N(0, σ²)
/// for complex in [true, false] {
///     let m2: f64 = gaussian_nodes(0.3, complex, 16).iter().map(|(w, p)| p * w.norm_sqr()).sum();
///     assert!((m2 - 0.3).abs() < 1e-12);
/// }
/// ```
pub fn gaussian_nodes(sigma2: f64, complex: bool, n: usize) -> Vec<(C64, f64)> {
    let (t, a) = gauss_hermite(n.max(2));
    let pi = std::f64::consts::PI;
    if complex {
        let s = sigma2.sqrt();
        t.iter()
            .zip(&a)
            .flat_map(|(tr, ar)| t.iter().zip(&a).map(move |(ti, ai)| (C64::new(s * tr, s * ti), ar * ai / pi)))
            .filter(|(_, w)| *w > 1e-300)
            .collect()
    } else {
        let s = (2.0 * sigma2).sqrt();
        t.iter().zip(&a).map(|(t, a)| (C64::new(s * t, 0.0), a / pi.sqrt())).collect()
    }
}

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

/// The entropy (bits) of a distribution, the zero probabilities left out:
///
/// $$
/// H(p) = -\sum_{i:\, p_{i} > 0} p_{i} \log_{2} p_{i}.
/// $$
///
/// # Examples
/// ```
/// use com_math::entropy;
/// assert_eq!(entropy(&[0.25; 4]), 2.0);
/// assert_eq!(entropy(&[1.0, 0.0]), 0.0);
/// ```
pub fn entropy(p: &[f64]) -> f64 {
    p.iter().filter(|q| **q > 0.0).map(|q| -q * q.log2()).sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unreadable_literal, reason = "the values of SciPy, digit for digit")]
    fn gamma_e1_and_erf_against_scipy() {
        // scipy.special.gammaln, gammainc, exp1, erf (SciPy 1.18)
        for (x, v) in
            [(0.5, 0.5723649429247), (3.7, 1.428072326665388), (10.0, 12.801827480081469), (100.5, 361.43554046777757)]
        {
            assert!((ln_gamma(x) - v).abs() < 1e-13 * v.abs().max(1.0), "lgamma {x}");
        }
        assert!(ln_gamma(1.0).abs() < 1e-15 && ln_gamma(2.0).abs() < 1e-15);
        for (a, x, v) in [
            (2.5, 1.0, 0.15085496391539038),
            (10.0, 6.2, 0.09837933549011063),
            (10.0, 20.0, 0.9950045876916924),
            (20.0, 5.0, 3.452135820914455e-7),
            (0.5, 30.0, 0.9999999999999906),
        ] {
            assert!((gamma_p(a, x) / v - 1.0).abs() < 1e-12, "P({a}, {x}) = {} vs {v}", gamma_p(a, x));
        }
        for (x, v) in [
            (1e-3, 6.331539364136149),
            (0.3, 0.9056766516758468),
            (2.5, 0.024914917870269736),
            (50.0, 3.783264029550459e-24),
        ] {
            assert!((expint_e1(x) / v - 1.0).abs() < 1e-13, "E1({x})");
        }
        for (x, v) in
            [(0.1, 0.1124629160182849), (1.9, 0.9927904292352575), (2.1, 0.997020533343667), (3.5, 0.9999992569016276)]
        {
            assert!((erf(x) - v).abs() < 1e-9, "erf({x})");
            assert_eq!(erf(-x), -erf(x));
        }
    }

    #[test]
    fn bessel_i0e_matches_scipy() {
        // scipy.special.i0e (SciPy 1.11), on both sides of the switch at 60
        let table = [
            (0.0, 1.0),
            (0.5, 0.645_035_270_449_150_1),
            (1.0, 0.465_759_607_593_640_43),
            (5.0, 0.183_540_812_609_328_34),
            (10.0, 0.127_833_337_163_428_6),
            (30.0, 0.073_145_946_482_237_3),
            (59.9, 0.051_654_795_368_473_484),
            (60.1, 0.051_568_411_420_294_53),
            (100.0, 0.039_944_379_299_096_68),
            (1e3, 0.012_617_240_455_891_257),
            (1e5, 0.001_261_567_837_976_776_6),
        ];
        for (x, want) in table {
            let got = bessel_i0e(x);
            assert!((got / want - 1.0).abs() < 1e-13, "{x}: {got} vs {want}");
            assert_eq!(bessel_i0e(-x), got);
        }
    }

    #[test]
    #[allow(clippy::unreadable_literal, reason = "the values of SciPy, digit for digit")]
    fn bessel_j0_against_scipy() {
        // scipy.special.j0
        for (x, v) in [
            (0.0, 1.0),
            (0.0628, 0.9990142830026578),
            (1.0, 0.7651976865579665),
            (2.404825557695773, 0.0),
            (5.0, -0.1775967713143383),
            (10.0, -0.2459357644513483),
            (30.0, -0.08636798358104023),
            (-3.0, -0.2600519549019335),
        ] {
            assert!((bessel_j0(x) - v).abs() < 1e-14, "J0({x}) = {} vs {v}", bessel_j0(x));
        }
    }

    #[test]
    #[allow(clippy::unreadable_literal, reason = "the values of SciPy, digit for digit")]
    fn fresnel_against_scipy() {
        // scipy.special.fresnel 1.x (which returns (S, C)): (x, C, S)
        let table = [
            (0.1, 0.09999753262708506, 0.0005235895476122108),
            (0.5, 0.4923442258714464, 0.06473243285999929),
            (1.0, 0.779893400376823, 0.4382591473903547),
            (1.4999, 0.4453535730062187, 0.6975432066559986),
            (1.5, 0.44526117603982146, 0.697504960082093),
            (2.0, 0.48825340607534073, 0.34341567836369824),
            (3.7, 0.5419456621544874, 0.5749803498874729),
            (10.0, 0.49989869420551575, 0.46816997858488224),
            (31.4, 0.5006397897427959, 0.51011704529959),
            (-2.5, -0.45741300964177706, -0.6191817558195929),
        ];
        for (x, c, s) in table {
            let (fc, fs) = fresnel(x);
            assert!((fc - c).abs() < 1e-14 && (fs - s).abs() < 1e-14, "{x}: ({fc}, {fs}) vs ({c}, {s})");
        }
        assert_eq!(fresnel(0.0), (0.0, 0.0));
        // C, S → 1/2 at infinity
        let (c, s) = fresnel(1e4);
        assert!((c - 0.5).abs() < 1e-4 && (s - 0.5).abs() < 1e-4);
    }

    #[test]
    fn erfc_reference_values() {
        // values of erfc from tables
        for (x, v) in [(0.0, 1.0), (0.5, 0.479_500_122), (1.0, 0.157_299_207), (3.0, 2.209_049_7e-5)] {
            assert!((erfc(x) / v - 1.0).abs() < 2e-7, "erfc({x})");
            assert!((erfc(-x) - (2.0 - v)).abs() < 1e-7);
        }
        assert!((erfcinv(erfc(1.3)) - 1.3).abs() < 1e-9);
    }

    #[test]
    fn decibels_round_trip() {
        assert!((lin2db(db2lin(7.3)) - 7.3).abs() < 1e-12);
        assert!((w2dbm(dbm2w(-3.0)) + 3.0).abs() < 1e-12);
    }
}
