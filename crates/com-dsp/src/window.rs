//! Windows for spectral estimation.
//!
//! [`hann`] is the periodic Hann window $w\[n\] = \sin^{2}(\pi n / N)$, $n = 0, \dots, N-1$,
//! whose 50 % overlapped copies sum to a constant (the choice of Welch's method).
//! [`chebyshev`] is the Dolph–Chebyshev window, whose sidelobes are all at the same
//! level below the main lobe, the narrowest main lobe for that level (Harris 1978,
//! Sec. V-H): the filter of UFMC (Schaich & Wild 2014).
//!
//! # Examples
//! ```
//! use com_dsp::window::hann;
//! let w = hann(8);
//! assert_eq!(w[0], 0.0);
//! assert!((w[4] - 1.0).abs() < 1e-15);
//! ```
//!
//! # References
//!
//! 1. F. J. Harris, "On the use of windows for harmonic analysis with the discrete
//!    Fourier transform," *Proc. IEEE*, 66(1), 51–83, 1978.

use com_math::C64;
use std::f64::consts::PI;

/// The periodic Hann window of length `n`.
pub fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| (PI * i as f64 / n as f64).sin().powi(2)).collect()
}

/// The Dolph–Chebyshev window of length `n` whose sidelobes are `attenuation_db` below
/// its main lobe, normalised to a maximum of 1 (as `scipy.signal.windows.chebwin`): the
/// inverse DFT of the samples $T_{n-1}(x_{0} \cos(\pi k / n))$ of the Chebyshev
/// polynomial, $x_{0} = \cosh(\operatorname{arccosh}(10^{A/20}) / (n - 1))$.
///
/// # Examples
/// ```
/// use com_dsp::window::chebyshev;
/// let w = chebyshev(7, 40.0);
/// assert!((w[0] - 0.159_410_69).abs() < 1e-8 && (w[3] - 1.0).abs() < 1e-15);
/// assert!((w[0] - w[6]).abs() < 1e-15);
/// ```
pub fn chebyshev(n: usize, attenuation_db: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let order = (n - 1) as f64;
    let beta = ((10f64.powf(attenuation_db.abs() / 20.0)).acosh() / order).cosh();
    let odd = n % 2 == 1;
    let p: Vec<C64> = (0..n)
        .map(|k| {
            let x = beta * (PI * k as f64 / n as f64).cos();
            let t = if x > 1.0 {
                (order * x.acosh()).cosh()
            } else if x < -1.0 {
                (if odd { 1.0 } else { -1.0 }) * (order * (-x).acosh()).cosh()
            } else {
                (order * x.acos()).cos()
            };
            if odd { C64::new(t, 0.0) } else { com_math::cis(PI * k as f64 / n as f64) * t }
        })
        .collect();
    let w: Vec<f64> = com_math::fft::fft(&p).iter().map(|v| v.re).collect();
    let full: Vec<f64> = if odd {
        let half = n.div_ceil(2);
        w[1..half].iter().rev().chain(&w[..half]).copied().collect()
    } else {
        let half = n / 2 + 1;
        w[1..half].iter().rev().chain(&w[1..half]).copied().collect()
    };
    let max = full.iter().copied().fold(f64::MIN, f64::max);
    full.into_iter().map(|v| v / max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unreadable_literal, reason = "the values of SciPy, digit for digit")]
    fn chebyshev_is_that_of_scipy() {
        // scipy.signal.windows.chebwin (SciPy 1.18)
        for (n, at, head) in [
            (8usize, 40.0, [0.14609713, 0.41790422, 0.75944595, 1.0]),
            (43, 40.0, [0.1731554, 0.11451651, 0.14967948, 0.18969298]),
            (74, 60.0, [0.02603358, 0.02045551, 0.02816156, 0.03750065]),
        ] {
            let w = chebyshev(n, at);
            assert_eq!(w.len(), n);
            for (a, b) in w.iter().zip(head) {
                assert!((a - b).abs() < 1e-8, "{n}: {a} vs {b}");
            }
            assert!(w.iter().zip(w.iter().rev()).all(|(a, b)| (a - b).abs() < 1e-12));
        }
    }
}
