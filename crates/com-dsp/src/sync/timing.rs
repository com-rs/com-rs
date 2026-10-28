//! Clock recovery (OptiCommPy `optic.dsp.clockRecovery`).
//!
//! The signal is at two samples per symbol. The Gardner timing-error detector compares
//! the sample between two symbols with the difference of the two symbols,
//!
//! `e = Re{x[1]* (x[2] - x[0])}` ([`gardner_ted`]), or, for Nyquist pulses,
//! `e = |x[1]|² (|x[0]|² - |x[2]|²)` ([`gardner_ted_nyquist`]),
//!
//! which is zero on average when the samples `x[0]`, `x[2]` sit on the symbol instants.
//! [`interpolate_cubic`] evaluates the cubic Lagrange (Farrow) interpolator of four
//! samples at a fractional position `μ ∈ [-1, 1]` (OptiCommPy `interpolator`), and
//! [`GardnerLoop`] closes the loop: every other output sample, the detector output goes
//! through a proportional-integral filter (`k_p`, `k_i`) that moves the fractional
//! sampling instant, with one sample skipped or repeated when it leaves `[-1, 1]`
//! (OptiCommPy `gardnerClockRecovery`).
//!
//! # Examples
//! ```
//! use com_dsp::sync::timing::{gardner_ted, interpolate_cubic};
//! use com_math::C64;
//! // a symmetric transition sampled at the right instants gives no error
//! let e = gardner_ted(&[C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(-1.0, 0.0)]);
//! assert_eq!(e, 0.0);
//! let x = [C64::new(1.0, 0.0), C64::new(2.0, 0.0), C64::new(3.0, 0.0), C64::new(4.0, 0.0)];
//! assert!((interpolate_cubic(&x, 0.0) - x[2]).norm() < 1e-15);   // μ = 0 returns x[2]
//! ```

use com_math::C64;

/// Gardner timing error of three consecutive samples at two samples per symbol.
pub fn gardner_ted(x: &[C64; 3]) -> f64 {
    (x[1].conj() * (x[2] - x[0])).re
}

/// Gardner timing error for Nyquist pulses (power version).
pub fn gardner_ted_nyquist(x: &[C64; 3]) -> f64 {
    x[1].norm_sqr() * (x[0].norm_sqr() - x[2].norm_sqr())
}

/// Cubic interpolation of four samples at the fractional position `t` (`t = 0` returns
/// `x[2]`, `t = 1` returns `x[3]`, `t = -1` returns `x[1]`).
pub fn interpolate_cubic(x: &[C64; 4], t: f64) -> C64 {
    let t2 = t * t;
    let t3 = t2 * t;
    x[0] * (-t3 / 6.0 + t / 6.0)
        + x[1] * (t3 / 2.0 + t2 / 2.0 - t)
        + x[2] * (-t3 / 2.0 - t2 + t / 2.0 + 1.0)
        + x[3] * (t3 / 6.0 + t2 / 2.0 + t / 3.0)
}

/// The Gardner clock-recovery loop.
#[derive(Clone, Debug, PartialEq)]
pub struct GardnerLoop {
    /// Proportional gain of the loop filter.
    pub kp: f64,
    /// Integral gain of the loop filter.
    pub ki: f64,
    /// Use the Nyquist (power) detector.
    pub nyquist: bool,
    /// Largest clock deviation handled (ppm): sizes the output.
    pub max_ppm: f64,
}

impl Default for GardnerLoop {
    /// `k_p = 10⁻³`, `k_i = 10⁻⁶`, Nyquist detector, 500 ppm (OptiCommPy defaults).
    fn default() -> Self {
        GardnerLoop { kp: 1e-3, ki: 1e-6, nyquist: true, max_ppm: 500.0 }
    }
}

/// The output of [`GardnerLoop::run`].
#[derive(Clone, Debug, PartialEq)]
pub struct ClockRecoveryOutput {
    /// Resampled signal at two samples per symbol.
    pub signal: Vec<C64>,
    /// Fractional sampling instant of every output sample.
    pub timing: Vec<f64>,
}

impl GardnerLoop {
    /// Recover the clock of a signal at (about) two samples per symbol.
    pub fn run(&self, x: &[C64]) -> ClockRecoveryOutput {
        let mut sig = x.to_vec();
        sig.push(C64::new(0.0, 0.0));
        let n_samples = sig.len();
        let ln = ((1.0 - self.max_ppm / 1e6) * n_samples as f64) as usize;
        let mut out = vec![C64::new(0.0, 0.0); ln];
        let mut timing = vec![0.0; ln];
        let (mut int_part, mut t_nco) = (0.0, 0.0);
        let (mut n, mut m) = (2usize, 2usize);
        let mut last = 0;
        while n + 1 < ln && m + 2 < n_samples {
            out[n] = interpolate_cubic(&[sig[m - 2], sig[m - 1], sig[m], sig[m + 1]], t_nco);
            if n % 2 == 0 {
                let w = [out[n - 2], out[n - 1], out[n]];
                let ted = if self.nyquist { gardner_ted_nyquist(&w) } else { gardner_ted(&w) };
                int_part += self.ki * ted;
                t_nco -= self.kp * ted + int_part;
            }
            if t_nco > 1.0 {
                t_nco -= 1.0;
                n -= 1;
            } else if t_nco < -1.0 {
                t_nco += 1.0;
                n += 2;
                m += 1;
            } else {
                n += 1;
                m += 1;
            }
            if n < ln {
                timing[n] = t_nco;
            }
            last = last.max(n);
        }
        out.truncate(last.min(ln));
        timing.truncate(last.min(ln));
        ClockRecoveryOutput { signal: out, timing }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{PulseShape, fir_filter, pulse_shape};
    use crate::resample::{delay_fft, upsample};
    use com_core::rng::Rng;

    #[test]
    fn interpolator_is_exact_on_cubics() {
        let f = |t: f64| 1.0 + 2.0 * t - 0.5 * t * t + 0.25 * t * t * t;
        let x = [f(-2.0), f(-1.0), f(0.0), f(1.0)].map(|v| C64::new(v, 0.0));
        for t in [-0.7, 0.0, 0.3, 0.9] {
            assert!((interpolate_cubic(&x, t).re - f(t)).abs() < 1e-12);
        }
    }

    #[test]
    fn gardner_loop_locks_on_a_delayed_signal() {
        let mut r = Rng::new(1);
        let s: Vec<C64> = (0..4000)
            .map(|_| C64::new(if r.bit() == 1 { 1.0 } else { -1.0 }, if r.bit() == 1 { 1.0 } else { -1.0 }))
            .collect();
        let h = pulse_shape(PulseShape::Rrc { rolloff: 0.5 }, 2, 64);
        let mut x = fir_filter(&h, &upsample(&s, 2));
        x = fir_filter(&h, &x);
        let x = delay_fft(&x, 0.4, 1.0); // 0.4 sample timing offset
        let out = GardnerLoop { kp: 5e-3, ki: 1e-5, nyquist: false, ..GardnerLoop::default() }.run(&x);
        // the loop converges to a constant fractional instant
        let t = &out.timing[6000..7000];
        let mean = t.iter().sum::<f64>() / t.len() as f64;
        let var = t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / t.len() as f64;
        assert!(var < 1e-3, "timing variance {var}");
    }
}
