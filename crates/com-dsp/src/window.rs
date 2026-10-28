//! Windows for spectral estimation.
//!
//! [`hann`] is the periodic Hann window $w\[n\] = \sin^{2}(\pi n / N)$, $n = 0, \dots, N-1$,
//! whose 50 % overlapped copies sum to a constant (the choice of Welch's method).
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

use std::f64::consts::PI;

/// The periodic Hann window of length `n`.
pub fn hann(n: usize) -> Vec<f64> {
    (0..n).map(|i| (PI * i as f64 / n as f64).sin().powi(2)).collect()
}
