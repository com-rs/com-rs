//! Elementary operations on signals: normalisations, frequency shift, cyclic prefix.
//!
//! [`pnorm`] (unit mean power), [`anorm`] (unit peak amplitude), [`freq_shift`]
//! (`x e^{j2πΔf t}`) and its in-place kernel [`rotate`], [`map_envelope`] (a memoryless
//! nonlinearity of the envelope: amplifiers, clipping, companding), and the cyclic
//! prefix of a block, [`extend_with_prefix`] and [`strip_prefix`] (OFDM, CE-OFDM, OTFS,
//! 5G NR).
//!
//! # Examples
//! ```
//! use com_dsp::signal::pnorm;
//! use com_math::{power, C64};
//! let x = pnorm(&[C64::new(3.0, 0.0), C64::new(0.0, 4.0)]);
//! assert!((power(&x) - 1.0).abs() < 1e-12);
//! ```

use com_core::{Result, ensure_shape};
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
    let mut y = x.to_vec();
    rotate(&mut y, 2.0 * PI * delta_f / fs, 0.0);
    y
}

/// Rotate in place by a linear phase, `x[n] ← x[n] e^{jω(n - n_0)}`: a frequency shift
/// of `ω` rad/sample whose phase is zero at the sample `n_0` (not necessarily an
/// integer: `(N - 1)/2` centres it on the block).
///
/// # Examples
/// ```
/// use com_dsp::signal::rotate;
/// use com_math::C64;
/// let mut x = vec![C64::new(1.0, 0.0); 3];
/// rotate(&mut x, std::f64::consts::FRAC_PI_2, 1.0);
/// assert!((x[0] - C64::new(0.0, -1.0)).norm() < 1e-15 && x[1] == C64::new(1.0, 0.0));
/// ```
pub fn rotate(x: &mut [C64], omega: f64, n0: f64) {
    for (n, v) in x.iter_mut().enumerate() {
        *v *= cis(omega * (n as f64 - n0));
    }
}

/// Append `block` to `out`, preceded by its cyclic prefix: its last `n_cp` samples.
///
/// # Errors
/// [`com_core::Error::Shape`] if the prefix is longer than the block.
///
/// # Examples
/// ```
/// use com_dsp::signal::{extend_with_prefix, strip_prefix};
/// let mut out = Vec::new();
/// extend_with_prefix(&mut out, &[1, 2, 3, 4], 2)?;
/// assert_eq!(out, [3, 4, 1, 2, 3, 4]);
/// assert_eq!(strip_prefix(&out, 2)?, [1, 2, 3, 4]);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn extend_with_prefix<T: Copy>(out: &mut Vec<T>, block: &[T], n_cp: usize) -> Result<()> {
    ensure_shape(n_cp <= block.len(), || format!("a cyclic prefix of {n_cp} samples for a block of {}", block.len()))?;
    out.extend_from_slice(&block[block.len() - n_cp..]);
    out.extend_from_slice(block);
    Ok(())
}

/// The block without its first `n_cp` samples (the cyclic prefix).
///
/// # Errors
/// [`com_core::Error::Shape`] if the prefix is longer than the block.
pub fn strip_prefix<T>(block: &[T], n_cp: usize) -> Result<&[T]> {
    ensure_shape(n_cp <= block.len(), || format!("a cyclic prefix of {n_cp} samples for a block of {}", block.len()))?;
    Ok(&block[n_cp..])
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
