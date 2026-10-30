//! Frequency stability of an oscillator from its phase: the two-sample (Allan), modified
//! and Hadamard variances of IEEE Std 1139-2022.
//!
//! # Model
//!
//! With the phase $\phi_{k}$ (rad) sampled every $\tau_{0}$, the time deviation is
//! $x_{k} = \phi_{k} / (2\pi\nu_{0})$ for a carrier $\nu_{0}$; the estimators below take
//! $x_{k} = \phi_{k}/(2\pi)$, so that they return the variances of the *absolute*
//! frequency (Hz²): divide by $\nu_{0}^{2}$ for those of the fractional frequency $y$. At
//! the averaging time $\tau = m\tau_{0}$, with the overlapping estimators of the standard
//! (Annex A),
//!
//! $$
//! \sigma^{2}(\tau) = \frac{\langle (x_{k+2m} - 2x_{k+m} + x_{k})^{2} \rangle}{2\tau^{2}}, \qquad
//! \mathrm{Mod}\,\sigma^{2}(\tau) = \frac{\langle (\frac{1}{m}\sum_{i=k}^{k+m-1} x_{i+2m} - 2x_{i+m} + x_{i})^{2} \rangle}{2\tau^{2}},
//! $$
//!
//! $$
//! H\sigma^{2}(\tau) = \frac{\langle (x_{k+3m} - 3x_{k+2m} + 3x_{k+m} - x_{k})^{2} \rangle}{6\tau^{2}}.
//! $$
//!
//! For white frequency noise of one-sided density $S_{\nu}(f) = h_{0}$ (a Lorentzian line of
//! full width $\Delta\nu = \pi h_{0}$, the Wiener phase of `com_hardware::oscillator`), they
//! are $h_{0}/(2\tau)$, $h_{0}/(4\tau)$ (IEEE Std 1139-2022, Table B.2) and $h_{0}/(2\tau)$:
//! the Hadamard variance of its eq. (A.31) equals the Allan variance for white frequency
//! noise, by its eq. (B.2), where Table B.2 prints $h_{0}/(3\tau)$.
//!
//! # Examples
//! ```
//! use com_metrics::stability::allan_variance;
//! // a phase ramp (a constant frequency offset) has no Allan variance
//! let phi: Vec<f64> = (0..1000).map(|k| 0.01 * k as f64).collect();
//! assert!(allan_variance(&phi, 1e-6, 10)?.abs() < 1e-12);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. IEEE Std 1139-2022, *IEEE Standard Definitions of Physical Quantities for Fundamental
//!    Frequency and Time Metrology—Random Instabilities*, Annex A and Table B.2.

use com_core::{Result, ensure, ensure_shape};
use std::f64::consts::TAU;

/// Check the averaging factor `m` and the sampling period `ts` (a configuration), then
/// the `n` phase samples against them and the `span` (2 or 3) of the differences (a
/// length).
fn check(n: usize, ts: f64, m: usize, span: usize) -> Result<()> {
    ensure(m >= 1 && ts > 0.0, || {
        format!("the averaging factor must be positive and the period too, got {m} and {ts} s")
    })?;
    ensure_shape(n > span * m, || {
        format!("{n} phase samples for an averaging factor of {m} (at least {} needed)", span * m + 1)
    })
}

/// The overlapping two-sample (Allan) variance of the frequency (Hz²) at $\tau = m\tau_{0}$
/// from the phase `phase_rad` sampled every `ts` seconds (see the module).
///
/// # Errors
/// [`com_core::Error::Shape`] for fewer than $2m + 1$ samples, [`com_core::Error::Config`]
/// for a zero `m` or a non-positive `ts`.
pub fn allan_variance(phase_rad: &[f64], ts: f64, m: usize) -> Result<f64> {
    check(phase_rad.len(), ts, m, 2)?;
    let x = |k: usize| phase_rad[k] / TAU;
    let terms = phase_rad.len() - 2 * m;
    let s: f64 = (0..terms).map(|k| (x(k + 2 * m) - 2.0 * x(k + m) + x(k)).powi(2)).sum();
    let tau = m as f64 * ts;
    Ok(s / (2.0 * tau * tau * terms as f64))
}

/// The modified Allan variance of the frequency (Hz²) at $\tau = m\tau_{0}$ (see the
/// module).
///
/// # Errors
/// [`com_core::Error::Shape`] for fewer than $3m$ samples, [`com_core::Error::Config`]
/// for a zero `m` or a non-positive `ts`.
pub fn modified_allan_variance(phase_rad: &[f64], ts: f64, m: usize) -> Result<f64> {
    check(phase_rad.len() + 1, ts, m, 3)?;
    let x = |k: usize| phase_rad[k] / TAU;
    let d = |i: usize| x(i + 2 * m) - 2.0 * x(i + m) + x(i);
    let terms = phase_rad.len() + 1 - 3 * m;
    // running sum of m consecutive second differences
    let mut window: f64 = (0..m).map(d).sum();
    let mut s = window * window;
    for k in 1..terms {
        window += d(k + m - 1) - d(k - 1);
        s += window * window;
    }
    let tau = m as f64 * ts;
    Ok(s / (2.0 * (m * m) as f64 * tau * tau * terms as f64))
}

/// The overlapping Hadamard variance of the frequency (Hz²) at $\tau = m\tau_{0}$ (see the
/// module).
///
/// # Errors
/// [`com_core::Error::Shape`] for fewer than $3m + 1$ samples, [`com_core::Error::Config`]
/// for a zero `m` or a non-positive `ts`.
pub fn hadamard_variance(phase_rad: &[f64], ts: f64, m: usize) -> Result<f64> {
    check(phase_rad.len(), ts, m, 3)?;
    let x = |k: usize| phase_rad[k] / TAU;
    let terms = phase_rad.len() - 3 * m;
    let s: f64 = (0..terms).map(|k| (x(k + 3 * m) - 3.0 * x(k + 2 * m) + 3.0 * x(k + m) - x(k)).powi(2)).sum();
    let tau = m as f64 * ts;
    Ok(s / (6.0 * tau * tau * terms as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frequency_offset_is_invisible_and_short_series_are_refused() {
        let phi: Vec<f64> = (0..200).map(|k| 0.3 * k as f64 + 1.0).collect();
        for f in [allan_variance, modified_allan_variance, hadamard_variance] {
            assert!(f(&phi, 1e-3, 5).unwrap().abs() < 1e-12);
            assert!(matches!(f(&phi, 1e-3, 100), Err(com_core::Error::Shape(_))));
            assert!(matches!(f(&phi, 1e-3, 0), Err(com_core::Error::Config(_))));
            assert!(matches!(f(&phi, 0.0, 5), Err(com_core::Error::Config(_))));
        }
        // a linear drift of the frequency: zero Hadamard variance, not Allan
        let drift: Vec<f64> = (0..200).map(|k| 1e-3 * (k * k) as f64).collect();
        assert!(hadamard_variance(&drift, 1.0, 5).unwrap().abs() < 1e-12);
        assert!(allan_variance(&drift, 1.0, 5).unwrap() > 1e-9);
    }
}
