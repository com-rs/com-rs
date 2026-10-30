//! Elementary operations on signals: normalisations, frequency shift, cyclic prefix.
//!
//! [`pnorm`] (unit mean power), [`anorm`] (unit peak amplitude), [`freq_shift`]
//! (`x e^{j2πΔf t}`) and its in-place kernel [`rotate`], [`map_envelope`] (a memoryless
//! nonlinearity of the envelope: amplifiers, clipping, companding), and the cyclic
//! prefix of a block, [`extend_with_prefix`] and [`strip_prefix`] (OFDM, CE-OFDM, OTFS,
//! 5G NR); the frequencies of a grid of channels, [`wdm_grid`], and the channel of a
//! multiplex brought to baseband, [`wdm_demultiplex`] (a frequency shift then a
//! band-limiting resampling).
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
pub fn freq_shift(x: &[C64], offset_hz: f64, fs: f64) -> Vec<C64> {
    let mut y = x.to_vec();
    rotate(&mut y, 2.0 * PI * offset_hz / fs, 0.0);
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

/// The centre frequencies of `n` channels spaced by `spacing_hz` (Hz), relative to the
/// carrier, as OptiCommPy's `simpleWDMTx`: $(k - \lfloor N/2 \rfloor) \Delta f$ for
/// $k = 0, \dots, N - 1$, shifted by $\Delta f / 2$ when $N$ is even (the grid is then
/// symmetric about the carrier).
///
/// # Examples
/// ```
/// use com_dsp::signal::wdm_grid;
/// assert_eq!(wdm_grid(3, 50e9), vec![-50e9, 0.0, 50e9]);
/// assert_eq!(wdm_grid(2, 50e9), vec![-25e9, 25e9]);
/// ```
pub fn wdm_grid(n: usize, spacing_hz: f64) -> Vec<f64> {
    let shift = if n % 2 == 0 { spacing_hz / 2.0 } else { 0.0 };
    (0..n).map(|k| (k as f64 - (n / 2) as f64) * spacing_hz + shift).collect()
}

/// The channel of centre frequency `f` (Hz) of a multiplex `x` sampled at `fs`, brought
/// to baseband ([`freq_shift`] by $-f$) and band-limited by resampling to `n_out` samples
/// ([`crate::resample::resample_fft`]): an ideal coherent receiver with its local
/// oscillator on the channel.
///
/// # Examples
/// ```
/// use com_dsp::signal::{freq_shift, wdm_demultiplex};
/// use com_math::C64;
/// // a tone 1 GHz above the carrier comes back to DC
/// let x = freq_shift(&vec![C64::new(1.0, 0.0); 64], 1e9, 8e9);
/// let y = wdm_demultiplex(&x, 1e9, 8e9, 16);
/// assert!(y.iter().all(|v| (v - C64::new(1.0, 0.0)).norm() < 1e-9));
/// ```
pub fn wdm_demultiplex(x: &[C64], f: f64, fs: f64, n_out: usize) -> Vec<C64> {
    crate::resample::resample_fft(&freq_shift(x, -f, fs), n_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The demultiplexer shifted the multiplex itself, with the phase `-2π f k / fs`,
    /// before it called [`freq_shift`], whose phase `(-2π f / fs) k` rounds differently:
    /// the outputs differ by the rounding of phases up to `2π f n / fs`.
    #[test]
    fn the_demultiplexer_shifts_as_before_to_the_rounding_of_the_phase() {
        let mut rng = com_core::rng::Rng::new(5);
        let x: Vec<C64> = (0..8192).map(|_| rng.complex_gaussian()).collect();
        let (fs, f) = (256e9, 75e9);
        let shifted: Vec<C64> = x.iter().enumerate().map(|(k, v)| v * cis(-2.0 * PI * f * k as f64 / fs)).collect();
        let old = crate::resample::resample_fft(&shifted, 2048);
        let new = wdm_demultiplex(&x, f, fs, 2048);
        let err = old.iter().zip(&new).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
        let sig = old.iter().map(C64::norm_sqr).sum::<f64>().sqrt();
        // the phase reaches 2π · 75/256 · 8192 ≈ 1.5·10⁴ rad, rounded to about 2·10⁻¹² rad
        assert!(err / sig < 1e-11, "{}", err / sig);
    }

    /// The grid of the WDM transmitter, whose copy without the half-spacing shift the
    /// tests of the GN model used.
    #[test]
    fn the_grid_is_symmetric_about_the_carrier() {
        for n in 1..8 {
            let g = wdm_grid(n, 50e9);
            assert!(g.iter().zip(g.iter().rev()).all(|(a, b)| (a + b).abs() < 1e-3), "{n}: {g:?}");
            assert!(g.windows(2).all(|w| (w[1] - w[0] - 50e9).abs() < 1e-3));
        }
    }
}
