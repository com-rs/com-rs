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
//! - decibels: [`db2lin`], [`lin2db`] (power ratios), [`dbm2w`], [`w2dbm`], [`db2np`],
//!   [`np2db`] (attenuation per unit length);
//! - [`sinc`], `sin(πx)/(πx)`, and [`effective_length`], `(1 - e^{-αL})/α`;
//! - special functions: [`erfc`], [`erfcinv`], [`qfunc`] (the Gaussian tail
//!   `Q(x) = ½ erfc(x/√2)`), [`fresnel`] (the Fresnel integrals $C$ and $S$), [`bessel_j0`];
//! - [`entropy`], $-\sum_{i} p_{i} \log_{2} p_{i}$;
//! - signal statistics: [`power`] (mean `|x|²`), [`energy`], [`papr_db`];
//! - quadrature: [`gauss_hermite`], the nodes and weights of `∫ f(t) e^{-t²} dt`;
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
    let n = n.max(2).div_ceil(2) * 2;
    let h = (b - a) / n as f64;
    let mut acc = f(a) + f(b);
    for k in 1..n {
        acc += if k % 2 == 1 { 4.0 } else { 2.0 } * f(a + k as f64 * h);
    }
    acc * h / 3.0
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
    if mean > 0.0 { 10.0 * (peak / mean).log10() } else { f64::NEG_INFINITY }
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
