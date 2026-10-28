//! Channels and impairments of one or several streams: noise, phase noise, frequency offset, linear channels, IQ imbalance.

use super::schema::ParamType;
use super::{Block, BlockInfo, BuildContext, Frame, Params};
use com_channel::awgn::{ebn0_noise_variance, snr_noise_variance};
use com_core::Error;
use com_core::Result;
use com_core::ensure;
use com_core::rng::Rng;
use com_dsp::filter;
use com_hardware::iq;
use com_hardware::oscillator::{PhaseOrigin, frequency_offset, increment_sd, wiener_phase};
use com_math::C64;
use com_math::{cis, power};
use serde::Deserialize;
use serde_json::Value as Json;
use std::f64::consts::PI;

/// The documentation of the blocks of this module.
pub(super) fn infos() -> Vec<BlockInfo> {
    vec![
        block_info!(
            "awgn",
            "Add white Gaussian noise for an SNR in a bandwidth.",
            AwgnSpec,
            needs[Samples],
            gives[Noise],
        ),
        block_info!(
            "awgn_ebn0",
            "Add white Gaussian noise for an E_b/N_0 (matched-filter definition).",
            AwgnEbn0,
            needs [Samples, Constellation],
            gives [Noise],
        ),
        block_info!("phase_noise", "Wiener phase noise.", PhaseNoiseSpec, needs [Samples], gives []),
        block_info!("cfo", "Carrier frequency offset.", CfoSpec, needs [Samples], gives []),
        block_info!(per_polarization "fir", "Linear channel with complex taps.", FirSpec, needs [Samples], gives []),
        block_info!(per_polarization "rx_iq", "Receiver IQ imbalance [Re y; Im y] = T [Re x; Im x].", RxIq, needs [Samples], gives []),
        block_info!(per_polarization "tx_iq", "Transmitter IQ imbalance y = x + β x*.", TxIq, needs [Samples], gives []),
    ]
}

/// Build a block of this module, or `None` for another type.
pub(super) fn build(k: &str, _label: &str, p: &Json, _ctx: &BuildContext) -> Option<Result<Box<dyn Block>>> {
    let p = p.clone();
    let block = move || -> Result<Option<Box<dyn Block>>> {
        Ok(Some(match k {
            "awgn" => Box::new(Awgn::new(&AwgnSpec::parse(k, p)?)),
            "awgn_ebn0" => Box::new(AwgnEbn0::parse(k, p)?),
            "phase_noise" => Box::new(PhaseNoise::new(&PhaseNoiseSpec::parse(k, p)?)?),
            "cfo" => Box::new(Cfo::new(&CfoSpec::parse(k, p)?)?),
            "fir" => Box::new(Fir::new(&FirSpec::parse(k, p)?)?),
            "rx_iq" => Box::new(RxIq::parse(k, p)?),
            "tx_iq" => Box::new(TxIq::parse(k, p)?),
            _ => return Ok(None),
        }))
    };
    block().transpose()
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
        let p = if self.unit {
            1.0 / s.sps
        } else if s.antennas.is_empty() && s.channel.faded_power.is_none() {
            power(&s.samples)
        } else {
            s.reference_power()
        };
        add_noise(s, snr_noise_variance(p, self.snr_db, s.fs(), b), rng);
        Ok(())
    }
}

/// Add complex white noise of variance `sigma2` to every antenna and record it.
fn add_noise(s: &mut Frame, sigma2: f64, rng: &mut Rng) {
    let sd = sigma2.sqrt();
    for v in s.streams_mut().flatten() {
        *v += rng.complex_gaussian() * sd;
    }
    s.channel.noise_var = Some(sigma2);
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
        // an uncoded single stream: the power of its samples (com_channel::awgn::awgn_ebn0)
        let p = if s.antennas.is_empty()
            && s.coding.code_rate.is_none()
            && s.pmf.is_none()
            && s.channel.faded_power.is_none()
        {
            power(&s.samples)
        } else {
            s.reference_power()
        };
        add_noise(s, ebn0_noise_variance(p, self.ebn0_db, b, s.sps), rng);
        Ok(())
    }
}

block_params! {
    struct PhaseNoiseSpec {
        linewidth_hz: Option<f64> [absent] "Hz" >= 0, "laser linewidth, increments of variance 2πΔν/f_s";
        sigma: Option<f64> [absent] "rad" >= 0, "increment standard deviation per sample, instead of the linewidth";
        origin: String ["zero"] one_of ["zero", "step"],
            "zero: φ[0] = 0, the first sample is not rotated (OptiCommPy); step: φ[0] = Δ[0], the first sample is rotated too";
    }
}

struct PhaseNoise {
    linewidth_hz: Option<f64>,
    sigma: Option<f64>,
    origin: PhaseOrigin,
}

impl PhaseNoise {
    fn new(p: &PhaseNoiseSpec) -> Result<PhaseNoise> {
        ensure(p.linewidth_hz.is_some() != p.sigma.is_some(), || "phase_noise: give linewidth_hz or sigma".into())?;
        let origin = p.origin.parse().map_err(|e: Error| e.context("phase_noise"))?;
        Ok(PhaseNoise { linewidth_hz: p.linewidth_hz, sigma: p.sigma, origin })
    }
}

impl Block for PhaseNoise {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let sd = match (self.linewidth_hz, self.sigma) {
            (Some(lw), _) => increment_sd(lw, s.fs()),
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
        hz: Option<f64> [absent] "Hz", "offset";
        omega: Option<f64> [absent] "rad/sample", "offset per sample, instead of hz";
        centred: bool [false], "phase zero at the middle of the block, e^{jω(n - N/2)}, instead of the first sample";
    }
}

struct Cfo {
    hz: Option<f64>,
    omega: Option<f64>,
    centred: bool,
}

impl Cfo {
    fn new(p: &CfoSpec) -> Result<Cfo> {
        ensure(p.hz.is_some() != p.omega.is_some(), || "cfo: give hz or omega".into())?;
        Ok(Cfo { hz: p.hz, omega: p.omega, centred: p.centred })
    }
}

impl Block for Cfo {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let w = self.omega.unwrap_or_else(|| 2.0 * PI * self.hz.unwrap_or(0.0) / s.fs());
        for x in s.streams_mut() {
            frequency_offset(x, w, self.centred);
        }
        Ok(())
    }
}

block_params! {
    struct FirSpec {
        h: Vec<Json> [required], "taps, [[re, im], …] or real numbers";
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
        let h = p.h.iter().map(complex).collect::<Result<Vec<_>>>()?;
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
