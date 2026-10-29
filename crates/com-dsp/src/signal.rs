//! Elementary operations on signals: normalisations and frequency shift.
//!
//! [`pnorm`] (unit mean power), [`anorm`] (unit peak amplitude), [`freq_shift`]
//! (`x e^{j2πΔf t}`), [`map_envelope`] (a memoryless nonlinearity of the envelope:
//! amplifiers, clipping, companding).
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

/// Apply a memoryless nonlinearity of the envelope in place: every nonzero sample `v`
/// becomes `v · g(|v|)`, `g(r)` the complex gain at the envelope `r` (the AM/AM
/// conversion over `r`, times `e^{j AM/PM}`); zero samples stay zero.
///
/// # Examples
/// ```
/// use com_dsp::signal::map_envelope;
/// use com_math::C64;
/// // an ideal limiter at 1, keeping the phase
/// let mut x = vec![C64::new(3.0, 4.0), C64::new(0.3, 0.0), C64::new(0.0, 0.0)];
/// map_envelope(&mut x, |r| C64::new(r.min(1.0) / r, 0.0));
/// assert!((x[0] - C64::new(0.6, 0.8)).norm() < 1e-15 && x[1] == C64::new(0.3, 0.0));
/// assert_eq!(x[2], C64::new(0.0, 0.0));
/// ```
pub fn map_envelope(x: &mut [C64], g: impl Fn(f64) -> C64) {
    for v in x {
        let r = v.norm();
        if r > 0.0 {
            *v *= g(r);
        }
    }
}
