//! Chromatic-dispersion compensation.
//!
//! The dispersion of a length $L$ of fibre is the all-pass response
//! $e^{j \beta_{2} \omega^{2} L / 2}$; its compensation multiplies by the conjugate.
//! Nothing here is specific to optics but the name of the parameter: the blocks take
//! the group-velocity dispersion `β₂` (s²/km) and the length (km).
//!
//! - [`cdc`]: exact compensation on a periodic block, `X(ω) e^{-j β₂ ω² L/2}`;
//! - [`cd_fir_taps`]: an FIR of `n` taps by frequency sampling of the same response
//!   (inverse DFT of `e^{-jβ₂ω_k²L/2}` on `n` bins, centred), applied by
//!   [`crate::filter::fir_filter_complex`]; [`cd_taps_needed`] gives the length rule
//!   of OptiCommPy's `edc`, `2⌈6.67 |β₂| L R_s² (f_s/R_s)⌉`.
//!
//! # Examples
//! ```
//! use com_dsp::compensation::cd::cdc;
//! use com_math::C64;
//! let b2 = -2.168e-23;                              // SSMF at 1550 nm (s²/km)
//! let x: Vec<C64> = (0..256).map(|n| C64::new((n % 5) as f64, 0.0)).collect();
//! // dispersion of 80 km (a compensation with the opposite sign), then compensation
//! let y = cdc(&cdc(&x, -b2, 80.0, 64e9), b2, 80.0, 64e9);
//! assert!(x.iter().zip(&y).all(|(a, b)| (a - b).norm() < 1e-9));
//! ```

use com_math::fft;
use com_math::{C64, cis};

/// Compensate the dispersion `beta2_s2_km` of `length_km` on a block sampled at `fs`.
pub fn cdc(x: &[C64], beta2_s2_km: f64, length_km: f64, fs: f64) -> Vec<C64> {
    let mut y = fft::fft(x);
    for (v, h) in y.iter_mut().zip(inverse_dispersion(x.len(), beta2_s2_km, length_km, fs)) {
        *v *= h;
    }
    fft::ifft(&y)
}

/// The inverse dispersion `e^{-jβ₂ω²L/2}` on the `n` angular frequencies of a block
/// sampled at `fs`.
fn inverse_dispersion(n: usize, beta2_s2_km: f64, length_km: f64, fs: f64) -> Vec<C64> {
    let w = fft::angular_frequencies(n, fs);
    w.iter().map(|wk| cis(-beta2_s2_km / 2.0 * wk * wk * length_km)).collect()
}

/// Number of taps of the dispersion-compensating FIR (OptiCommPy `edc` rule).
pub fn cd_taps_needed(beta2_s2_km: f64, length_km: f64, symbol_rate: f64, fs: f64) -> usize {
    let b2 = beta2_s2_km.abs();
    (2.0 * (6.67 * b2 * length_km * symbol_rate * symbol_rate * (fs / symbol_rate)).ceil()) as usize
}

/// `n` centred taps of the dispersion compensation of `length_km` by frequency sampling.
pub fn cd_fir_taps(beta2_s2_km: f64, length_km: f64, fs: f64, n: usize) -> Vec<C64> {
    let t = fft::ifft(&inverse_dispersion(n, beta2_s2_km, length_km, fs));
    // centre: tap i of the FIR is the impulse response at time i - ⌊n/2⌋
    (0..n).map(|i| t[(i + n - n / 2) % n]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::fir_filter_complex;
    use com_core::rng::Rng;

    #[test]
    fn cdc_and_its_fir_invert_the_dispersion() {
        // SSMF: D = 17 ps/(nm km) at 193.1 THz
        let b2 = -2.1749e-23;
        let mut r = Rng::new(1);
        let n = 4096;
        let fs = 64e9;
        let x: Vec<C64> = (0..n).map(|_| r.complex_gaussian()).collect();
        // band-limit to ±fs/4
        let mut xs = fft::fft(&x);
        for (v, fk) in xs.iter_mut().zip(fft::fftfreq(n, 1.0)) {
            if fk.abs() > 0.25 {
                *v = C64::new(0.0, 0.0);
            }
        }
        let x = fft::ifft(&xs);
        // the lossless linear channel of 20 km is the compensation of the opposite β₂
        let y = cdc(&x, -b2, 20.0, fs);
        let z = cdc(&y, b2, 20.0, fs);
        assert!(x.iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-9));
        let taps = cd_fir_taps(b2, 20.0, fs, 255);
        let zf = fir_filter_complex(&taps, &y);
        let mid = &zf[512..n - 512];
        let err: f64 = mid.iter().zip(&x[512..n - 512]).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>()
            / x[512..n - 512].iter().map(C64::norm_sqr).sum::<f64>();
        assert!(err < 1e-3, "relative error {err}");
        // 2⌈6.67 |β₂| L R_s² (f_s/R_s)⌉ = 2⌈5.92⌉ for 20 km of SSMF at 32 GBd, 2 sps
        assert_eq!(cd_taps_needed(b2, 20.0, 32e9, fs), 12);
    }
}
