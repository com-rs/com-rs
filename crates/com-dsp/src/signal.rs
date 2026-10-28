//! Elementary operations on signals: normalisations and frequency shift.
//!
//! [`pnorm`] (unit mean power), [`anorm`] (unit peak amplitude), [`freq_shift`]
//! (`x e^{j2πΔf t}`).
//!
//! # Examples
//! ```
//! use com_dsp::signal::pnorm;
//! use com_math::{power, C64};
//! let x = pnorm(&[C64::new(3.0, 0.0), C64::new(0.0, 4.0)]);
//! assert!((power(&x) - 1.0).abs() < 1e-12);
//! ```

use com_math::{C64, cis, power};
use std::f64::consts::PI;

/// Scale to unit mean power.
pub fn pnorm(x: &[C64]) -> Vec<C64> {
    let p = power(x).sqrt();
    if p == 0.0 {
        return x.to_vec();
    }
    x.iter().map(|v| v / p).collect()
}

/// Scale to unit peak amplitude.
pub fn anorm(x: &[C64]) -> Vec<C64> {
    let m = x.iter().map(|v| v.norm()).fold(0.0, f64::max);
    if m == 0.0 {
        return x.to_vec();
    }
    x.iter().map(|v| v / m).collect()
}

/// Frequency shift `x[n] e^{j2πΔf n/fs}`.
pub fn freq_shift(x: &[C64], delta_f: f64, fs: f64) -> Vec<C64> {
    x.iter().enumerate().map(|(n, &v)| v * cis(2.0 * PI * delta_f * n as f64 / fs)).collect()
}
