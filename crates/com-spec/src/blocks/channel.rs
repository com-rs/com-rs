//! Channels and impairments of one or several streams: noise, phase noise, frequency offset, linear channels, IQ imbalance, converters.

use super::schema::ParamType;
use super::{Block, Extension, Frame};
use com_channel::awgn::{ebn0_noise_variance, snr_noise_variance};
use com_core::Error;
use com_core::Result;
use com_core::ensure;
use com_core::rng::Rng;
use com_dsp::filter;
use com_hardware::converter::{Adc, Dac, Quantizer};
use com_hardware::iq;
use com_hardware::oscillator::{PhaseOrigin, frequency_offset, increment_sd, wiener_phase};
use com_math::C64;
use com_math::cis;
use serde::Deserialize;
use serde_json::Value as Json;
use std::f64::consts::PI;

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            "awgn",
            "Add white Gaussian noise for an SNR in a bandwidth.",
            AwgnSpec,
            needs[Samples],
            gives[Noise],
            |q, _, _| Ok(Box::new(Awgn::new(&q))),
        ),
        builtin!(
            "awgn_ebn0",
            "Add white Gaussian noise for an E_b/N_0 (matched-filter definition).",
            AwgnEbn0,
            needs [Samples, Constellation],
            gives [Noise],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            "phase_noise", "Wiener phase noise.", PhaseNoiseSpec, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(PhaseNoise::new(&q)?)),
        ),
        builtin!(
            "cfo", "Carrier frequency offset.", CfoSpec, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(Cfo::new(&q)?)),
        ),
        builtin!(
            per_polarization "fir", "Linear channel with complex taps.", FirSpec, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(Fir::new(&q)?)),
        ),
        builtin!(
            per_polarization "rx_iq", "Receiver IQ imbalance [Re y; Im y] = T [Re x; Im x].", RxIq, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            per_polarization "tx_iq", "Transmitter IQ imbalance y = x + β x*.", TxIq, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            per_polarization "adc",
            "Analogue-to-digital converter: front-end bandwidth, clock jitter, ENOB noise and uniform quantisation with clipping.",
            AdcSpec,
            needs[Samples],
            gives[],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            per_polarization "dac",
            "Digital-to-analogue converter: uniform quantisation with clipping and output bandwidth.",
            DacSpec,
            needs[Samples],
            gives[],
            |q, _, _| Ok(Box::new(q)),
        ),
    ]
}

// ----------------------------------------------------------------------- channels

block_params! {
    struct AwgnSpec {
        snr_db: f64 [required] "dB", "signal-to-noise ratio";
        reference: NoiseReference ["signal"] one_of ["signal", "unit"],
            "signal: measured power (OptiCommPy); unit: E_s = 1 whatever the samples, σ² = 10^(-SNR/10) per symbol";
        bandwidth_hz: Option<f64> [absent "symbol rate"] "Hz" > 0, "noise bandwidth B: σ² = (f_s/B) P / SNR";
    }
}

/// The power the SNR of `awgn` refers to.
#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum NoiseReference {
    /// The measured power of the signal.
    Signal,
    /// Unit-energy symbols.
    Unit,
}

impl ParamType for NoiseReference {
    fn schema() -> Json {
        String::schema()
    }
}

struct Awgn {
    snr_db: f64,
    unit: bool,
    bandwidth: Option<f64>,
}

impl Awgn {
    fn new(p: &AwgnSpec) -> Awgn {
        Awgn { snr_db: p.snr_db, unit: p.reference == NoiseReference::Unit, bandwidth: p.bandwidth_hz }
    }
}

impl Block for Awgn {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let b = self.bandwidth.unwrap_or(s.symbol_rate);
        // the power of the samples (that of the channel output before fading, if any)
        let p = if self.unit { 1.0 / s.sps } else { s.reference_power() };
        add_noise(s, snr_noise_variance(p, self.snr_db, s.fs(), b), rng);
        Ok(())
    }
}

/// Add complex white noise of variance `noise_var` to every antenna and record it.
fn add_noise(s: &mut Frame, noise_var: f64, rng: &mut Rng) {
    for v in s.streams_mut() {
        com_channel::awgn::add_noise(v, noise_var, rng);
    }
    s.channel.noise_var = Some(noise_var);
}

block_params! {
    struct AwgnEbn0 {
        ebn0_db: f64 [required] "dB", "E_b/N_0";
    }
}

impl Block for AwgnEbn0 {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        // information bits per symbol: log2 M times the code rate, or the entropy of a
        // shaped source
        let b = match (s.coding.code_rate, &s.pmf) {
            (Some(r), _) => s.constellation("awgn_ebn0")?.bits_per_symbol() as f64 * r,
            (None, Some(p)) => com_metrics::entropy(p),
            (None, None) => s.constellation("awgn_ebn0")?.bits_per_symbol() as f64,
        };
        // the power of the samples over every antenna, before fading if any (that of
        // com_channel::awgn::awgn_ebn0 for a single stream)
        let p = s.reference_power();
        add_noise(s, ebn0_noise_variance(p, self.ebn0_db, b, s.sps), rng);
        Ok(())
    }
}

block_params! {
    struct PhaseNoiseSpec {
        linewidth_hz: Option<f64> [absent] "Hz" >= 0, "laser linewidth, increments of variance 2πΔν/f_s";
        sigma_rad: Option<f64> [absent] "rad" >= 0, "increment standard deviation per sample, instead of the linewidth";
        origin: String ["zero"] one_of ["zero", "step"],
            "zero: φ[0] = 0, the first sample is not rotated (OptiCommPy); step: φ[0] = Δ[0], the first sample is rotated too";
    }
}

struct PhaseNoise {
    linewidth_hz: Option<f64>,
    sigma_rad: Option<f64>,
    origin: PhaseOrigin,
}

impl PhaseNoise {
    fn new(p: &PhaseNoiseSpec) -> Result<PhaseNoise> {
        ensure(p.linewidth_hz.is_some() != p.sigma_rad.is_some(), || {
            "phase_noise: give linewidth_hz or sigma_rad".into()
        })?;
        let origin = p.origin.parse().map_err(|e: Error| e.context("phase_noise"))?;
        Ok(PhaseNoise { linewidth_hz: p.linewidth_hz, sigma_rad: p.sigma_rad, origin })
    }
}

impl Block for PhaseNoise {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let sd = match (self.linewidth_hz, self.sigma_rad) {
            (Some(lw), _) => increment_sd(lw, s.fs())?,
            (_, Some(sg)) => sg,
            _ => 0.0,
        };
        let rot: Vec<C64> = wiener_phase(sd, s.samples.len(), self.origin, rng).into_iter().map(cis).collect();
        // one oscillator for every antenna
        for x in s.streams_mut() {
            for (v, r) in x.iter_mut().zip(&rot) {
                *v *= r;
            }
        }
        Ok(())
    }
}

block_params! {
    struct CfoSpec {
        offset_hz: Option<f64> [absent] "Hz", "offset";
        omega_rad: Option<f64> [absent] "rad/sample", "offset per sample, instead of offset_hz";
        centred: bool [false], "phase zero at the middle of the block, e^{jω(n - N/2)}, instead of the first sample";
    }
}

struct Cfo {
    offset_hz: Option<f64>,
    omega_rad: Option<f64>,
    centred: bool,
}

impl Cfo {
    fn new(p: &CfoSpec) -> Result<Cfo> {
        ensure(p.offset_hz.is_some() != p.omega_rad.is_some(), || "cfo: give offset_hz or omega_rad".into())?;
        Ok(Cfo { offset_hz: p.offset_hz, omega_rad: p.omega_rad, centred: p.centred })
    }
}

impl Block for Cfo {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let w = self.omega_rad.unwrap_or_else(|| 2.0 * PI * self.offset_hz.unwrap_or(0.0) / s.fs());
        for x in s.streams_mut() {
            frequency_offset(x, w, self.centred);
        }
        Ok(())
    }
}

block_params! {
    struct FirSpec {
        h: Vec<ComplexNumber> [required], "taps, [[re, im], …] or real numbers";
        alignment: Alignment ["causal"] one_of ["causal", "centred", "centered"],
            "causal: y[n] = Σ h[k] x[n-k], the delay of the taps kept; centred: the taps centred on the samples (OptiCommPy firFilter)";
    }
}

/// The alignment of the output of `fir` on its input.
#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Alignment {
    /// $y\[n\] = \sum_{k} h\[k\] x\[n-k\]$.
    Causal,
    /// Centred on the middle tap (OptiCommPy `firFilter`).
    #[serde(alias = "centered")]
    Centred,
}

impl ParamType for Alignment {
    fn schema() -> Json {
        String::schema()
    }
}

json_param! {
    /// A complex number: `[re, im]` or a real number.
    ComplexNumber,
    serde_json::json!({"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}]})
}

struct Fir {
    h: Vec<C64>,
    causal: bool,
}

/// Parse a complex number written as `[re, im]` or a real number.
pub(crate) fn complex(v: &Json) -> Result<C64> {
    match v {
        Json::Number(n) => Ok(C64::new(n.as_f64().unwrap_or(f64::NAN), 0.0)),
        Json::Array(a) if a.len() == 2 => match (a[0].as_f64(), a[1].as_f64()) {
            (Some(r), Some(i)) => Ok(C64::new(r, i)),
            _ => Err(Error::Config(format!("a complex number is [re, im], got {v}"))),
        },
        other => Err(Error::Config(format!("a complex number is [re, im] or a real number, got {other}"))),
    }
}

impl Fir {
    fn new(p: &FirSpec) -> Result<Fir> {
        ensure(!p.h.is_empty(), || "fir: h needs at least one tap".into())?;
        let h = p.h.iter().map(|v| complex(&v.0)).collect::<Result<Vec<_>>>()?;
        Ok(Fir { h, causal: p.alignment == Alignment::Causal })
    }
}

impl Block for Fir {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        s.samples = if self.causal {
            let mut y = filter::convolve_full(&self.h, &s.samples);
            y.truncate(s.samples.len());
            y
        } else {
            filter::fir_filter_complex(&self.h, &s.samples)
        };
        Ok(())
    }
}

block_params! {
    struct RxIq {
        t: [f64; 4] [required], "row-major real 2×2 matrix [t00, t01, t10, t11]";
    }
}

impl Block for RxIq {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let t = self.t;
        iq::RxIq { t: [[t[0], t[1]], [t[2], t[3]]] }.apply(&mut s.samples);
        Ok(())
    }
}

block_params! {
    struct TxIq {
        beta: [f64; 2] [required], "[re, im] of β";
    }
}

impl Block for TxIq {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        iq::TxIq { beta: C64::new(self.beta[0], self.beta[1]) }.apply(&mut s.samples);
        Ok(())
    }
}

// ---------------------------------------------------------------------- converters

block_params! {
    struct AdcSpec {
        bits: u32 [required] "bits" >= 1, "resolution N (1 to 24)";
        full_scale: Option<f64> [absent "loading × RMS"] > 0, "full scale A of each dimension (levels ±(A - Δ/2), Δ = 2A/2^N)";
        loading: Option<f64> [absent "optimal for a Gaussian input"] > 0,
            "full scale in RMS values of one dimension of the samples (an automatic gain control)";
        enob: Option<f64> [absent "bits"] > 0, "effective number of bits (≤ bits)";
        jitter_s: f64 [0.0] "s" >= 0, "RMS sampling-clock jitter";
        bandwidth_hz: Option<f64> [absent "unlimited"] "Hz" > 0, "3 dB bandwidth of the front end (Gaussian)";
    }
}

impl Block for AdcSpec {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let quantizer = Quantizer::for_signal(self.bits, self.full_scale, self.loading, &s.samples)
            .map_err(|e| e.context("adc"))?;
        let adc = Adc { quantizer, enob: self.enob, jitter_s: self.jitter_s, bandwidth_hz: self.bandwidth_hz };
        s.samples = adc.convert(&s.samples, s.fs(), rng).map_err(|e| e.context("adc"))?;
        Ok(())
    }
}

block_params! {
    struct DacSpec {
        bits: u32 [required] "bits" >= 1, "resolution N (1 to 24)";
        full_scale: Option<f64> [absent "loading × RMS"] > 0, "full scale A of each dimension";
        loading: Option<f64> [absent "optimal for a Gaussian input"] > 0,
            "full scale in RMS values of one dimension of the samples";
        bandwidth_hz: Option<f64> [absent "unlimited"] "Hz" > 0, "3 dB bandwidth of the output (Gaussian)";
    }
}

impl Block for DacSpec {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let quantizer = Quantizer::for_signal(self.bits, self.full_scale, self.loading, &s.samples)
            .map_err(|e| e.context("dac"))?;
        let dac = Dac { quantizer, hold: 1, bandwidth_hz: self.bandwidth_hz };
        s.samples = dac.convert(&s.samples, s.fs()).map_err(|e| e.context("dac"))?;
        Ok(())
    }
}
