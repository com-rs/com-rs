//! Resampling: up- and down-sampling, decimation, band-limited resampling and
//! fractional delay.
//!
//! - [`upsample`] inserts `factor - 1` zeros after every sample, [`downsample`] keeps one
//!   sample in `factor`, [`decimate`] first picks the sampling phase of maximum variance
//!   (OptiCommPy `decimate`);
//! - [`resample_fft`] resamples a periodic block exactly (zero-padding or truncation of
//!   its spectrum);
//! - [`delay_fft`] delays a periodic block by a fractional number of samples.
//!
//! # Examples
//! ```
//! use com_dsp::resample::{downsample, upsample};
//! use com_math::C64;
//! // decimation after upsampling recovers the samples
//! let s: Vec<C64> = (0..5).map(|n| C64::new(n as f64, 1.0)).collect();
//! assert_eq!(downsample(&upsample(&s, 3), 3, 0), s);
//! ```

use com_core::{Result, ensure, ensure_shape};
use com_math::fft;
use com_math::{C64, cis};
use std::f64::consts::PI;

/// Insert `factor - 1` zeros after every sample (nothing for a factor of 0).
pub fn upsample(x: &[C64], factor: usize) -> Vec<C64> {
    if factor == 0 {
        return Vec::new();
    }
    let mut y = vec![C64::new(0.0, 0.0); x.len() * factor];
    for (i, &v) in x.iter().enumerate() {
        y[i * factor] = v;
    }
    y
}

/// Keep the samples `offset, offset + factor, …`.
pub fn downsample(x: &[C64], factor: usize, offset: usize) -> Vec<C64> {
    x.iter().skip(offset).step_by(factor.max(1)).copied().collect()
}

/// Decimate from `sps_in` to `sps_out` samples per symbol at the sampling phase of
/// maximum variance (OptiCommPy `decimate`, a simple timing recovery).
///
/// # Errors
/// [`com_core::Error::Config`] unless `sps_out` divides `sps_in`,
/// [`com_core::Error::Shape`] unless the length is a multiple of `sps_in`.
pub fn decimate(x: &[C64], sps_in: usize, sps_out: usize) -> Result<Vec<C64>> {
    ensure(sps_in >= 1 && sps_out >= 1 && sps_in % sps_out == 0, || format!("{sps_out} does not divide {sps_in}"))?;
    ensure_shape(x.len() % sps_in == 0, || format!("{} samples is not a multiple of sps_in = {sps_in}", x.len()))?;
    let n = x.len() / sps_in;
    let var = |phase: usize| {
        let col: Vec<C64> = (0..n).map(|k| x[k * sps_in + phase]).collect();
        let m = col.iter().sum::<C64>() / n as f64;
        col.iter().map(|v| (v - m).norm_sqr()).sum::<f64>()
    };
    let best = (0..sps_in).fold((f64::NEG_INFINITY, 0), |acc, p| {
        let v = var(p);
        if v > acc.0 { (v, p) } else { acc }
    });
    let len = x.len();
    let factor = sps_in / sps_out;
    Ok((0..len / factor).map(|k| x[(k * factor + best.1) % len]).collect())
}

/// Exact band-limited resampling of a periodic block to `n_out` samples.
pub fn resample_fft(x: &[C64], n_out: usize) -> Vec<C64> {
    let n_in = x.len();
    if n_in == 0 || n_out == 0 {
        return vec![C64::new(0.0, 0.0); n_out];
    }
    let xs = fft::fft(x);
    let mut ys = vec![C64::new(0.0, 0.0); n_out];
    let scale = n_out as f64 / n_in as f64;
    for (i, v) in ys.iter_mut().enumerate() {
        let k = if i < n_out.div_ceil(2) { i as i64 } else { i as i64 - n_out as i64 };
        let inside = n_in >= n_out || (k >= -((n_in / 2) as i64) && k < n_in.div_ceil(2) as i64);
        if inside {
            *v = xs[k.rem_euclid(n_in as i64) as usize] * scale;
        }
    }
    fft::ifft(&ys)
}

/// Delay a periodic block by `delay` seconds at the sampling rate `fs` (fractional
/// delays allowed): `Y(f) = X(f) e^{-j2πf·delay}`.
pub fn delay_fft(x: &[C64], delay: f64, fs: f64) -> Vec<C64> {
    let f = fft::fftfreq(x.len(), 1.0 / fs);
    fft::filter(x, f.into_iter().map(|fk| cis(-2.0 * PI * fk * delay)))
}
