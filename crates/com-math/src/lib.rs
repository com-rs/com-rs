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
//! | [`geometry`] | vectors of the three-dimensional space: scalar product, norm, direction |
//! | [`linalg`] | QR, Cholesky, Hermitian systems and the SVD of small complex matrices |
//! | [`nearest`] | [`nearest::Slicer`], the nearest point of a finite set |
//! | [`special`] | special functions: the error function, the Gaussian tail, Fresnel, Bessel, gamma, the exponential integral |
//! | [`quadrature`] | Simpson's rule, the Gauss–Hermite and Gauss–Legendre rules |
//! | [`log_domain`] | `ln(1 + e^x)`, `max*` and the log-sum-exp without overflow |
//! | [`stats`] | the moments of order two and four of a complex sequence, the normalised correlation of two |
//! | [`nlse`] | the symmetric split-step Fourier solvers of the scalar NLSE and of the Manakov equation |
//! | [`gf2`] | polynomials and matrices over GF(2) |
//! | [`gf2m`] | the Galois fields $\mathrm{GF}(2^{m})$ of the BCH and Reed–Solomon codes |
//! | [`bigint`] | unsigned integers of arbitrary size, for enumerative coding |
//! | [`bits`] | the Gray code [`gray`], [`gray_inverse`], and the conversions [`bits_to_index`], [`index_to_bits`] (most significant bit first) |
//! | [`search`] | [`bisect`], [`golden_max`] and [`gcd`] |
//! | [`sequences`] | linear-feedback shift registers and the Gold sequence of 3GPP |
//!
//! At the root (the functions of [`special`], [`quadrature`] and [`log_domain`] are
//! re-exported there):
//!
//! - the log domain: [`log_sum_exp`] (`ln Σ e^{v_i}`), [`ln_1p_exp`] (`ln(1 + e^x)`),
//!   [`max_star`] (`ln(e^a + e^b)`), all without overflow;
//! - [`C64`]: the complex type of every signal (`num_complex::Complex<f64>`), and
//!   [`cis`], `e^{jθ}`;
//! - decibels: [`db2lin`], [`lin2db`] (power ratios), [`db2amp`], [`amp2db`] (amplitude
//!   ratios), [`dbm2w`], [`w2dbm`], [`db2np`], [`np2db`] (attenuation per unit length),
//!   defined once in [`com_core::units`] next to the unit newtypes and re-exported here;
//! - [`sinc`], `sin(πx)/(πx)`, [`diric`], the periodic sinc `sin(nx/2)/(n sin(x/2))`, and
//!   [`effective_length`], `(1 - e^{-αL})/α`;
//! - special functions: [`erf`], [`erfc`], [`erfcinv`], [`qfunc`] (the Gaussian tail
//!   `Q(x) = ½ erfc(x/√2)`), its inverse [`qfuncinv`], [`fresnel`] (the Fresnel
//!   integrals $C$ and $S$), [`bessel_j0`], [`bessel_i0e`], [`bessel_k`] and
//!   [`ln_bessel_k`], [`ln_gamma`], [`gamma_p`] (the regularized incomplete gamma
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
pub mod geometry;
pub mod gf2;
pub mod gf2m;
pub mod linalg;
pub mod log_domain;
pub mod nearest;
pub mod nlse;
pub mod quadrature;
pub mod search;
pub mod sequences;
pub mod special;
pub mod stats;

pub use bits::{bits_to_index, gray, gray_inverse, index_to_bits};
pub use com_core::units::{amp2db, db2amp, db2lin, db2np, dbm2w, lin2db, np2db, w2dbm};
pub use log_domain::{ln_1p_exp, log_sum_exp, log_sum_exp_iter, max_star};
pub use quadrature::{gauss_hermite, gauss_legendre, gaussian_nodes, simpson, simpson_weight, try_simpson};
pub use search::{bisect, gcd, golden_max};
pub use special::{
    bessel_i0e, bessel_j0, bessel_k, erf, erfc, erfcinv, expint_e1, fresnel, gamma_p, ln_bessel_k, ln_gamma,
    normal_pdf, qfunc, qfuncinv,
};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_bessel_k_returns_its_limits_instead_of_looping() {
        assert_eq!(ln_bessel_k(0.0, f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(ln_bessel_k(2.5, f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(ln_bessel_k(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(bessel_k(1.0, f64::INFINITY), 0.0);
        assert!(ln_bessel_k(f64::NAN, 1.0).is_nan());
        assert!(ln_bessel_k(1.0, f64::NAN).is_nan());
        assert!(ln_bessel_k(1.0, 0.0).is_nan());
        // ν/x overflows: the leading term ½Γ(ν)(2/x)^ν, which the quadrature matches
        // just before the overflow
        let lead = |nu: f64, x: f64| -std::f64::consts::LN_2 + ln_gamma(nu) + nu * (std::f64::consts::LN_2 - x.ln());
        assert!((ln_bessel_k(2.0, 1e-300) / lead(2.0, 1e-300) - 1.0).abs() < 1e-12);
        assert_eq!(ln_bessel_k(2.0, 1e-310), lead(2.0, 1e-310));
        assert!(ln_bessel_k(0.01, 1e-320).is_nan());
    }

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
    #[allow(clippy::unreadable_literal, clippy::excessive_precision, reason = "the values of mpmath, digit for digit")]
    fn erfc_reference_values() {
        // mpmath 1.3 with 40 digits, rounded to 20: every branch of Cody's approximations
        // (|x| ≤ 0.46875, up to 4, beyond) and the tail down to 5.7·10⁻²⁹⁶
        for (x, v) in [
            (0.0, 1.0),
            (1e-10, 0.99999999988716208329),
            (0.1, 0.8875370839817151078),
            (0.3, 0.67137324054087257236),
            (0.46875, 0.50738652678206200841),
            (0.5, 0.47950012218695346232),
            (0.75, 0.2888443663464848684),
            (1.0, 0.15729920705028513066),
            (1.5, 0.033894853524689272933),
            (2.0, 0.0046777349810472658379),
            (3.0, 0.000022090496998585441373),
            (4.0, 1.5417257900280018852e-8),
            (4.5, 1.9661604415428874763e-10),
            (5.0, 1.5374597944280348502e-12),
            (6.0, 2.1519736712498913117e-17),
            (8.0, 1.122429717298292708e-29),
            (10.0, 2.088487583762544757e-45),
            (15.0, 7.2129941724512066666e-100),
            (20.0, 5.3958656116079009289e-176),
            (26.0, 5.6631924088561428465e-296),
        ] {
            assert!((erfc(x) / v - 1.0).abs() < 1e-15, "erfc({x}) = {} against {v}", erfc(x));
            assert!((erfc(-x) - (2.0 - v)).abs() < 1e-15, "erfc(-{x})");
        }
        for (x, v) in [(-0.3, 1.3286267594591274276), (-1.0, 1.8427007929497148693), (-3.0, 1.9999779095030014146)] {
            assert!((erfc(x) / v - 1.0).abs() < 1e-15, "erfc({x})");
        }
        // the subnormal range, then zero, the limits and NaN
        assert!((erfc(27.0) / 5.237048923789255685e-319 - 1.0).abs() < 1e-3);
        assert_eq!((erfc(27.5), erfc(f64::INFINITY), erfc(f64::NEG_INFINITY)), (0.0, 0.0, 2.0));
        assert!(erfc(f64::NAN).is_nan());
        assert!((erfcinv(erfc(1.3)) - 1.3).abs() < 1e-9);
        // Q(4.753424308822899) = 1.0000000000000013·10⁻⁶ (mpmath), within the rounding of
        // x/√2 amplified by x² ≈ 23
        assert!((qfunc(4.753_424_308_822_899) / 1.000_000_000_000_001_3e-6 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn decibels_round_trip() {
        assert!((lin2db(db2lin(7.3)) - 7.3).abs() < 1e-12);
        assert!((w2dbm(dbm2w(-3.0)) + 3.0).abs() < 1e-12);
    }
}
