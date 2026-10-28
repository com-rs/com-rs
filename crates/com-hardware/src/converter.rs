//! Digital-to-analogue and analogue-to-digital converters: uniform quantisation with
//! clipping, the effective number of bits, sampling-clock jitter, the analogue
//! bandwidth and the zero-order hold, with the closed forms they are checked against.
//!
//! The in-phase and quadrature components of a complex signal go through two identical
//! converters.
//!
//! # Model
//!
//! A [`Quantizer`] of $N$ bits and full scale $A$ is uniform and mid-rise: its step is
//! $\Delta = 2A/2^{N}$ and its levels are $\pm\Delta/2, \pm 3\Delta/2, \dots, \pm(A - \Delta/2)$,
//!
//! $$
//! q(x) = \min\left(\max\left(\Delta \left(\left\lfloor x/\Delta \right\rfloor + \tfrac{1}{2}\right),
//! -A + \tfrac{\Delta}{2}\right), A - \tfrac{\Delta}{2}\right).
//! $$
//!
//! - **Granular noise.** Inside the range, the error of a fine quantiser is uniform over a
//!   step, of variance $\Delta^{2}/12$ (Bennett 1948): a full-scale sine has the SQNR
//!   $10 \log_{10}(\tfrac{3}{2}\, 4^{N}) = 6.02 N + 1.76$ dB ([`sqnr_sine_db`]).
//! - **Gaussian inputs.** For $X \sim \mathcal{N}(0, \sigma^{2})$ (an OFDM signal, per
//!   dimension), the loading factor $\lambda = A/\sigma$ trades the granular noise against
//!   the clipping (overload) noise; [`gaussian_mse`] is their exact sum, cell by cell,
//!   with the overload cells in closed form
//!   $\int_{c}^{\infty} (u - y)^{2} \varphi(u)\\, du = (1 + y^{2}) Q(c) + (c - 2y) \varphi(c)$,
//!   and [`optimal_loading`] the $\lambda$ that minimises it.
//! - **Effective number of bits.** An ADC of $\mathrm{ENOB} < N$ bits has the SINAD
//!   $6.02\\, \mathrm{ENOB} + 1.76$ dB on a full-scale sine (IEEE Std 1241): [`Adc`] adds
//!   white noise of variance $\tfrac{\Delta^{2}}{12} (4^{N - \mathrm{ENOB}} - 1)$ per
//!   dimension before the quantiser ([`enob_from_sinad_db`] inverts the relation).
//! - **Jitter.** Sampling at $t_{n} + \delta_{n}$, $\delta_{n} \sim \mathcal{N}(0, \sigma_{j}^{2})$,
//!   gives to the first order $x(t_{n}) + \delta_{n} x'(t_{n})$, the derivative taken
//!   in the frequency domain (the frame is treated as periodic): for a sine of frequency
//!   $f$, $\mathrm{SNR} = -20 \log_{10}(2\pi f \sigma_{j})$ ([`jitter_snr_db`], Walden 1999).
//! - **Zero-order hold.** [`Dac`] holds every sample for `hold` samples of the output:
//!   the response $\sin(\pi f/f_{s}) / (H \sin(\pi f/(H f_{s})))$, which tends to
//!   $\mathrm{sinc}(f/f_{s})$ ([`zero_order_hold_gain`]).
//! - **Bandwidth.** An optional Gaussian low-pass filter of 3 dB bandwidth $B$ models the
//!   analogue front end (before the ADC, after the DAC).
//!
//! # Examples
//! ```
//! use com_hardware::converter::{Quantizer, sqnr_sine_db};
//! use com_math::C64;
//! let q = Quantizer::new(3, 1.0)?;                         // step 0.25, levels ±0.125 … ±0.875
//! assert_eq!(q.quantize(0.3), 0.375);
//! assert_eq!(q.quantize(5.0), 0.875);                      // clipped to the last level
//! let mut x = vec![C64::new(-0.01, 0.6)];
//! q.apply(&mut x);
//! assert_eq!(x[0], C64::new(-0.125, 0.625));
//! assert!((sqnr_sine_db(8) - 49.92).abs() < 0.01);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. W. R. Bennett, "Spectra of quantized signals," *Bell Syst. Tech. J.*, 27(3),
//!    446–472, 1948.
//! 2. R. H. Walden, "Analog-to-digital converter survey and analysis," *IEEE J. Sel.
//!    Areas Commun.*, 17(4), 539–550, 1999.
//! 3. IEEE Std 1241-2010, *Standard for terminology and test methods for
//!    analog-to-digital converters*.

use com_core::ensure;
use com_core::prelude::*;
use com_dsp::filter::{LowPass, fir_filter, low_pass_fir};
use com_math::fft::{fft, fftfreq, ifft};
use com_math::{erfc, golden_max};
use std::f64::consts::PI;

/// A uniform mid-rise quantiser of `bits` bits over $[-A, A]$ (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantizer {
    /// Resolution $N$ (bits).
    pub bits: u32,
    /// Full scale $A$: the levels span $[-A + \Delta/2, A - \Delta/2]$.
    pub full_scale: f64,
}

impl Quantizer {
    /// A quantiser of `bits` bits and full scale `full_scale`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless $1 \le N \le 24$ and $A > 0$ is finite.
    pub fn new(bits: u32, full_scale: f64) -> Result<Quantizer> {
        ensure((1..=24).contains(&bits), || format!("a converter has 1 to 24 bits, got {bits}"))?;
        ensure(full_scale > 0.0 && full_scale.is_finite(), || {
            format!("the full scale must be positive and finite, got {full_scale}")
        })?;
        Ok(Quantizer { bits, full_scale })
    }

    /// The quantiser whose full scale is `loading` times `rms` (the RMS value of one
    /// dimension of the signal: an automatic gain control before the converter).
    ///
    /// # Errors
    /// As [`Quantizer::new`].
    pub fn with_loading(bits: u32, rms: f64, loading: f64) -> Result<Quantizer> {
        Quantizer::new(bits, loading * rms)
    }

    /// The step $\Delta = 2A/2^{N}$.
    pub fn step(&self) -> f64 {
        2.0 * self.full_scale / 2f64.powi(self.bits as i32)
    }

    /// The level of a real value.
    pub fn quantize(&self, x: f64) -> f64 {
        let d = self.step();
        let top = self.full_scale - d / 2.0;
        (d * ((x / d).floor() + 0.5)).clamp(-top, top)
    }

    /// Quantise the in-phase and quadrature components in place.
    pub fn apply(&self, x: &mut [C64]) {
        for v in x {
            *v = C64::new(self.quantize(v.re), self.quantize(v.im));
        }
    }
}

/// The RMS value of one dimension of a complex signal, $\sqrt{\overline{|x|^{2}}/2}$:
/// the reference of the loading factor (an automatic gain control before the
/// converter).
pub fn rms_per_dimension(x: &[C64]) -> f64 {
    if x.is_empty() { 0.0 } else { (x.iter().map(C64::norm_sqr).sum::<f64>() / (2.0 * x.len() as f64)).sqrt() }
}

/// The SQNR of a full-scale sine through an ideal quantiser of `bits` bits:
/// $10 \log_{10}(\tfrac{3}{2}\, 4^{N})$ dB.
pub fn sqnr_sine_db(bits: u32) -> f64 {
    10.0 * 1.5f64.log10() + 20.0 * f64::from(bits) * 2f64.log10()
}

/// The effective number of bits of a SINAD measured on a full-scale sine:
/// $(\mathrm{SINAD} - 1.76)/6.02$.
pub fn enob_from_sinad_db(sinad_db: f64) -> f64 {
    (sinad_db - 10.0 * 1.5f64.log10()) / (20.0 * 2f64.log10())
}

/// The SNR of a sine of frequency `f` (Hz) sampled with an RMS clock jitter of
/// `jitter_s` seconds: $-20 \log_{10}(2\pi f \sigma_{j})$ dB.
pub fn jitter_snr_db(f: f64, jitter_s: f64) -> f64 {
    -20.0 * (2.0 * PI * f * jitter_s).log10()
}

/// The amplitude response at `f` of a zero-order hold of `hold` output samples per
/// input sample at the input rate `fs`: $|\sin(\pi f/f_{s}) / (H \sin(\pi f/(H f_{s})))|$.
pub fn zero_order_hold_gain(f: f64, fs: f64, hold: usize) -> f64 {
    let h = hold.max(1) as f64;
    let x = PI * f / fs;
    if x.abs() < 1e-12 { 1.0 } else { (x.sin() / (h * (x / h).sin())).abs() }
}

/// $\Phi(u)$ and $\varphi(u)$ of the standard normal distribution.
fn normal(u: f64) -> (f64, f64) {
    (0.5 * erfc(-u / std::f64::consts::SQRT_2), (-0.5 * u * u).exp() / (2.0 * PI).sqrt())
}

/// The exact mean squared error of `q` (one dimension) for $X \sim \mathcal{N}(0, \sigma^{2})$:
/// the granular cells by Gauss–Legendre quadrature, the two overload cells in closed
/// form (see the module).
pub fn gaussian_mse(q: &Quantizer, sigma: f64) -> f64 {
    // five-point Gauss–Legendre on [-1, 1]: exact up to degree 9
    const X: [f64; 5] =
        [0.0, -0.538_469_310_105_683_1, 0.538_469_310_105_683_1, -0.906_179_845_938_664, 0.906_179_845_938_664];
    const W: [f64; 5] = [
        0.568_888_888_888_888_9,
        0.478_628_670_499_366_5,
        0.478_628_670_499_366_5,
        0.236_926_885_056_189_1,
        0.236_926_885_056_189_1,
    ];
    let (d, a) = (q.step() / sigma, q.full_scale / sigma);
    let levels = 1usize << q.bits;
    // the overload cells beyond ±(A - Δ), of level ±(A - Δ/2), are symmetric
    let (c, y) = (a - d, a - d / 2.0);
    let (phi_c, pdf_c) = normal(c);
    let tail = (1.0 + y * y) * (1.0 - phi_c) + (c - 2.0 * y) * pdf_c;
    let mut inner = 0.0;
    for k in 1..levels - 1 {
        let lo = -a + k as f64 * d;
        let mid = lo + d / 2.0;
        inner += X.iter().zip(&W).map(|(x, w)| w * (x * d / 2.0).powi(2) * normal(mid + x * d / 2.0).1).sum::<f64>()
            * d
            / 2.0;
    }
    let tails = if levels == 2 {
        // one bit: the two cells are the half-lines, of levels ±Δ/2
        let y = d / 2.0;
        let (_, pdf0) = normal(0.0);
        2.0 * ((1.0 + y * y) * 0.5 - 2.0 * y * pdf0)
    } else {
        2.0 * tail
    };
    sigma * sigma * (inner + tails)
}

/// The loading factor $\lambda = A/\sigma$ that minimises [`gaussian_mse`] for `bits` bits.
pub fn optimal_loading(bits: u32) -> f64 {
    golden_max(|l| -gaussian_mse(&Quantizer { bits, full_scale: l }, 1.0), 0.2, 12.0, 120)
}

/// An analogue-to-digital converter: the front-end bandwidth, the clock jitter, the
/// noise of its effective number of bits and the quantiser, in that order (see the
/// module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Adc {
    /// The quantiser.
    pub quantizer: Quantizer,
    /// Effective number of bits (`None`: the resolution of the quantiser).
    pub enob: Option<f64>,
    /// RMS sampling-clock jitter (s).
    pub jitter_s: f64,
    /// 3 dB bandwidth of the analogue front end (Hz; `None`: unlimited).
    pub bandwidth_hz: Option<f64>,
}

impl Adc {
    /// An ideal ADC: the quantiser alone.
    pub fn ideal(quantizer: Quantizer) -> Adc {
        Adc { quantizer, enob: None, jitter_s: 0.0, bandwidth_hz: None }
    }

    /// Convert `x`, sampled at `fs` (Hz).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for an ENOB outside $(0, N]$, a negative jitter or a
    /// non-positive bandwidth or sampling rate.
    pub fn convert(&self, x: &[C64], fs: f64, rng: &mut Rng) -> Result<Vec<C64>> {
        let n = self.quantizer.bits;
        ensure(fs > 0.0, || format!("the sampling rate must be positive, got {fs}"))?;
        ensure(self.jitter_s >= 0.0, || format!("the jitter must be non-negative, got {}", self.jitter_s))?;
        let mut y = band_limit(x, fs, self.bandwidth_hz)?;
        if self.jitter_s > 0.0 && y.len() > 1 {
            // x(t + δ) ≈ x(t) + δ x'(t), x' = IFFT(j 2π f X)
            let freqs = fftfreq(y.len(), 1.0 / fs);
            let spec: Vec<C64> = fft(&y).iter().zip(&freqs).map(|(v, f)| v * C64::new(0.0, 2.0 * PI * f)).collect();
            let deriv = ifft(&spec);
            for (v, dv) in y.iter_mut().zip(&deriv) {
                *v += dv * (rng.gaussian() * self.jitter_s);
            }
        }
        if let Some(e) = self.enob {
            ensure(e > 0.0 && e <= f64::from(n), || format!("the ENOB must be in (0, {n}], got {e}"))?;
            let d = self.quantizer.step();
            let sd = (d * d / 12.0 * (4f64.powf(f64::from(n) - e) - 1.0)).sqrt();
            for v in &mut y {
                *v += C64::new(rng.gaussian() * sd, rng.gaussian() * sd);
            }
        }
        self.quantizer.apply(&mut y);
        Ok(y)
    }
}

impl Impair for Adc {
    fn impair_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let fs = x.sample_rate().0;
        let streams = (0..x.n_streams()).map(|s| self.convert(x.stream(s), fs, rng)).collect::<Result<Vec<_>>>()?;
        *out = Signal::from_streams(&streams, x.sample_rate())?;
        Ok(())
    }
}

/// A digital-to-analogue converter: the quantiser, the zero-order hold of `hold` output
/// samples per input sample, then the analogue bandwidth (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dac {
    /// The quantiser.
    pub quantizer: Quantizer,
    /// Output samples per input sample (1: no hold, the output at the input rate).
    pub hold: usize,
    /// 3 dB bandwidth of the analogue output (Hz; `None`: unlimited).
    pub bandwidth_hz: Option<f64>,
}

impl Dac {
    /// An ideal DAC at the input rate: the quantiser alone.
    pub fn ideal(quantizer: Quantizer) -> Dac {
        Dac { quantizer, hold: 1, bandwidth_hz: None }
    }

    /// Convert `x`, sampled at `fs` (Hz): the output is at `hold` × `fs`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for zero hold, a non-positive bandwidth or sampling rate.
    pub fn convert(&self, x: &[C64], fs: f64) -> Result<Vec<C64>> {
        ensure(self.hold >= 1, || "the hold of a DAC must be at least one sample".into())?;
        ensure(fs > 0.0, || format!("the sampling rate must be positive, got {fs}"))?;
        let mut q = x.to_vec();
        self.quantizer.apply(&mut q);
        let held: Vec<C64> = q.iter().flat_map(|v| std::iter::repeat_n(*v, self.hold)).collect();
        band_limit(&held, fs * self.hold as f64, self.bandwidth_hz)
    }
}

impl Impair for Dac {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let fs = x.sample_rate().0;
        let streams = (0..x.n_streams()).map(|s| self.convert(x.stream(s), fs)).collect::<Result<Vec<_>>>()?;
        *out = Signal::from_streams(&streams, Hertz(fs * self.hold as f64))?;
        Ok(())
    }
}

/// The Gaussian low-pass filter of 3 dB bandwidth `bandwidth` (none if `None`).
fn band_limit(x: &[C64], fs: f64, bandwidth: Option<f64>) -> Result<Vec<C64>> {
    let Some(b) = bandwidth else { return Ok(x.to_vec()) };
    ensure(b > 0.0 && b.is_finite(), || format!("the bandwidth must be positive, got {b}"))?;
    // a Gaussian response falls by 60 dB within about 2.5/B: taps over ±4 fs/B
    let n = (8.0 * fs / b).ceil().clamp(9.0, 4097.0) as usize | 1;
    Ok(fir_filter(&low_pass_fir(b, fs, n, LowPass::Gauss), x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_steps_and_clipping() {
        let q = Quantizer::new(2, 2.0).unwrap(); // step 1, levels ±0.5, ±1.5
        let out: Vec<f64> = [-9.0, -1.2, -0.2, 0.0, 0.7, 1.99, 9.0].iter().map(|x| q.quantize(*x)).collect();
        assert_eq!(out, vec![-1.5, -1.5, -0.5, 0.5, 0.5, 1.5, 1.5]);
        assert!(Quantizer::new(0, 1.0).is_err() && Quantizer::new(8, 0.0).is_err());
    }

    #[test]
    fn the_gaussian_mse_is_granular_plus_overload() {
        // fine quantiser, large loading: Δ²/12
        let q = Quantizer::new(12, 6.0).unwrap();
        let d = q.step();
        assert!((gaussian_mse(&q, 1.0) / (d * d / 12.0) - 1.0).abs() < 1e-3);
        // one bit at the optimal level ±√(2/π) σ: σ²(1 - 2/π) (Max 1960)
        let one = Quantizer::new(1, 2.0 * (2.0 / PI).sqrt()).unwrap();
        assert!((gaussian_mse(&one, 1.0) - (1.0 - 2.0 / PI)).abs() < 1e-12);
        // the optimal loading grows with the resolution
        assert!(optimal_loading(4) < optimal_loading(8));
    }

    #[test]
    fn a_dac_holds_its_samples() {
        let dac = Dac { quantizer: Quantizer::new(8, 1.0).unwrap(), hold: 3, bandwidth_hz: None };
        let y = dac.convert(&[C64::new(0.5, -0.25)], 1.0).unwrap();
        assert_eq!(y.len(), 3);
        assert!(y.iter().all(|v| (v - C64::new(0.503_906_25, -0.246_093_75)).norm() < 1e-12));
        assert!((zero_order_hold_gain(0.0, 1.0, 4) - 1.0).abs() < 1e-12);
    }
}
