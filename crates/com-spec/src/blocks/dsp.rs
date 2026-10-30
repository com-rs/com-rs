//! Receiver DSP: shaping and matched filtering, resampling and decimation, clock recovery, adaptive equalisation, carrier recovery, frame synchronisation.

use super::schema::ParamType;
use super::{Block, Extension, Frame, NumberOrList, part, per_pol};
use com_core::Error;
use com_core::Result;
use com_core::ensure;
use com_core::rng::Rng;
use com_dsp::equalizer::Algorithm;
use com_dsp::equalizer::Singularity;
use com_dsp::equalizer::{AdaptiveEqualizer, EqualizerOutput};
use com_dsp::filter;
use com_dsp::filter::PulseShape;
use com_dsp::resample;
use com_dsp::signal;
use com_dsp::sync::carrier;
use com_dsp::sync::frame;
use com_dsp::sync::timing::GardnerLoop;
use com_math::C64;
use com_math::dbm2w;
use serde::Deserialize;
use serde_json::Value as Json;
use std::f64::consts::{FRAC_PI_2, PI};

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            per_polarization "pulse_shape",
            "Upsample and filter with a transmit pulse, then normalise to unit power.",
            PulseSpec,
            needs [Samples],
            gives [],
            |q, _, _| Ok(Box::new(PulseShaping::new(&q)?)),
        ),
        builtin!(
            "normalize", "Scale the samples to a mean power.", Normalize, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            per_polarization "resample",
            "Band-limited resampling of the block (FFT) to a number of samples per symbol.",
            Resample,
            needs [Samples],
            gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            per_polarization "matched_filter",
            "Filter with a receive pulse at the current samples per symbol.",
            MatchedSpec,
            needs [Samples],
            gives [],
            |q, _, _| Ok(Box::new(MatchedFilter::new(q)?)),
        ),
        builtin!(
            per_polarization "decimate", "Reduce the samples per symbol.", DecimateSpec, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(Decimate::new(&q))),
        ),
        builtin!(
            per_polarization "gardner", "Gardner clock recovery at 2 samples per symbol.", Gardner, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            "adaptive_equalizer",
            "Adaptive FIR equaliser, output at 1 sample per symbol; the n × n butterfly over the polarisations or antennas.",
            EqualizerSpec,
            needs [Samples, Constellation],
            gives [],
            |q, label, _| Ok(Box::new(Equalizer::new(&q, label)?)),
        ),
        builtin!(
            per_polarization "foe",
            "Frequency-offset estimation by the M-th power and correction; records foe_hz.",
            FoeSpec,
            needs [Samples],
            gives [],
            |q, label, _| Ok(Box::new(Foe { p: q, label })),
        ),
        builtin!(
            per_polarization "bps",
            "Blind phase search, unwrapped with a period of π/2, and correction.",
            Bps,
            needs [Samples, Constellation],
            gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            per_polarization "viterbi", "Viterbi–Viterbi phase estimation and correction.", Viterbi, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            per_polarization "ddpll", "Decision-directed phase-locked loop.", Ddpll, needs [Samples, Constellation], gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            "frame_sync",
            "Delay and rotation on known symbols, output aligned on the transmitted symbols; records delay and quality.",
            FrameSyncSpec,
            needs [Samples, Symbols],
            gives [],
            |q, label, _| Ok(Box::new(FrameSync { p: q, label })),
        ),
    ]
}

block_params! {
    /// A pulse: the parameters shared by `pulse_shape`, `matched_filter` and `wdm_tx`.
    #[derive(Clone, Debug)]
    pub(super) struct PulseSettings {
        kind: String ["rrc"] one_of ["rect", "nrz", "rc", "rrc"], "the pulse";
        rolloff: f64 [0.1] >= 0 <= 1, "roll-off of rc and rrc";
        n_taps: usize [1025] >= 1, "taps of rc and rrc (odd: on the sampling instants)";
    }
}

impl PulseSettings {
    /// The taps of the pulse at `sps` samples per symbol.
    pub(super) fn taps(&self, sps: usize) -> Result<Vec<f64>> {
        ensure(sps >= 1, || "sps must be positive".into())?;
        Ok(filter::pulse_shape(PulseShape::from_name(&self.kind, self.rolloff)?, sps, self.n_taps))
    }
}

block_params! {
    struct PulseSpec {
        pulse: PulseSettings [flatten], "the transmit pulse";
        sps: usize [2] >= 1, "samples per symbol";
    }
}

struct PulseShaping {
    taps: Vec<f64>,
    sps: usize,
}

impl PulseShaping {
    fn new(p: &PulseSpec) -> Result<PulseShaping> {
        Ok(PulseShaping { taps: p.pulse.taps(p.sps)?, sps: p.sps })
    }
}

impl Block for PulseShaping {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let k = s.int_sps("pulse_shape")?;
        ensure(k == 1, || "pulse_shape expects symbols (1 sample per symbol)".into())?;
        s.samples = signal::pnorm(&filter::fir_filter(&self.taps, &resample::upsample(&s.samples, self.sps)));
        s.sps = self.sps as f64;
        Ok(())
    }
}

block_params! {
    struct Normalize {
        power_dbm: Option<f64> [absent "unit power"] "dBm", "mean power";
    }
}

impl Block for Normalize {
    /// Every sample divided by the RMS value over all the streams (as
    /// `com_dsp::signal::pnorm` for one stream), then multiplied by the amplitude of
    /// the power.
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let a = self.power_dbm.map_or(1.0, |p| dbm2w(p).sqrt());
        let rms = s.mean_power().sqrt();
        s.streams_mut().flatten().for_each(|v| *v = if rms == 0.0 { *v * a } else { *v / rms * a });
        Ok(())
    }
}

block_params! {
    struct MatchedSpec {
        pulse: PulseSettings [flatten], "the receive pulse";
    }
}

/// The matched filter: its taps at the samples per symbol of the signal.
struct MatchedFilter(PulseSettings);

impl MatchedFilter {
    fn new(p: MatchedSpec) -> Result<MatchedFilter> {
        p.pulse.taps(2)?;
        Ok(MatchedFilter(p.pulse))
    }
}

impl Block for MatchedFilter {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let sps = s.int_sps("matched_filter")?;
        let h = self.0.taps(sps)?;
        s.samples = filter::fir_filter(&h, &s.samples);
        Ok(())
    }
}

block_params! {
    struct DecimateSpec {
        sps_out: usize [1] >= 1, "output samples per symbol";
        method: DecimationMethod ["max_variance"] one_of ["max_variance", "offset"],
            "max_variance (OptiCommPy decimate) or offset";
        offset: usize [0], "sampling phase of the offset method";
    }
}

/// The sampling phase of `decimate`.
#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DecimationMethod {
    /// The phase of maximum variance (OptiCommPy `decimate`).
    MaxVariance,
    /// A given phase.
    Offset,
}

impl ParamType for DecimationMethod {
    fn schema() -> Json {
        String::schema()
    }
}

struct Decimate {
    sps_out: usize,
    offset: Option<usize>,
}

impl Decimate {
    fn new(p: &DecimateSpec) -> Decimate {
        let offset = match p.method {
            DecimationMethod::MaxVariance => None,
            DecimationMethod::Offset => Some(p.offset),
        };
        Decimate { sps_out: p.sps_out, offset }
    }
}

impl Block for Decimate {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let sps = s.int_sps("decimate")?;
        ensure(sps % self.sps_out == 0, || format!("decimate: {} does not divide {sps}", self.sps_out))?;
        let n = s.samples.len() / sps * sps;
        s.samples.truncate(n);
        s.samples = match self.offset {
            None => resample::decimate(&s.samples, sps, self.sps_out)?,
            Some(o) => resample::downsample(&s.samples, sps / self.sps_out, o),
        };
        s.sps = self.sps_out as f64;
        Ok(())
    }
}

block_params! {
    struct Gardner {
        kp: f64 [1e-3], "proportional gain";
        ki: f64 [1e-6], "integral gain";
        nyquist: bool [true], "detector for Nyquist pulses";
    }
}

impl Block for Gardner {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        ensure(s.int_sps("gardner")? == 2, || "gardner needs 2 samples per symbol".into())?;
        let g = GardnerLoop { kp: self.kp, ki: self.ki, nyquist: self.nyquist, max_ppm: 500.0 };
        let mut y = g.run(&signal::pnorm(&s.samples)).signal;
        y.truncate(y.len() / 2 * 2);
        s.samples = y;
        Ok(())
    }
}

block_params! {
    struct EqualizerSpec {
        algorithm: Schedule ["cma"], "cma, rde, nlms, dd-lms, or a schedule [[algorithm, symbols], …]";
        n_taps: usize [15] >= 1, "taps";
        mu: NumberOrList [1e-3] > 0,
            "step size, or one per stage of the schedule (the last one holds for the following stages)";
        passes: usize [1] >= 1, "passes over the block";
        kikuchi: Option<usize> [absent],
            "2 × 2: Kikuchi's initialisation against the singularity, output 0 alone during this number of symbols, then output 1 as its unitary complement; records correlation and singular";
    }
}

json_param! {
    /// The algorithm of `adaptive_equalizer`: a name, or a schedule of stages.
    Schedule,
    serde_json::json!({"anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer", "minimum": 0}], "minItems": 2, "maxItems": 2}}
    ]})
}

struct Equalizer {
    schedule: Vec<(Algorithm, usize)>,
    n_taps: usize,
    mu: Vec<f64>,
    passes: usize,
    singularity: Singularity,
    label: String,
}

impl Equalizer {
    fn new(p: &EqualizerSpec, label: String) -> Result<Equalizer> {
        let schedule = match &p.algorithm.0 {
            Json::String(a) => vec![(a.parse()?, usize::MAX)],
            Json::Array(steps) => steps
                .iter()
                .map(|st| match st {
                    Json::Array(pair) if pair.len() == 2 => match (pair[0].as_str(), pair[1].as_u64()) {
                        (Some(a), Some(n)) => Ok((a.parse()?, n as usize)),
                        _ => Err(Error::Config(format!("a schedule step is [algorithm, symbols], got {st}"))),
                    },
                    _ => Err(Error::Config(format!("a schedule step is [algorithm, symbols], got {st}"))),
                })
                .collect::<Result<_>>()?,
            other => return Err(Error::Config(format!("algorithm must be a name or a schedule, got {other}"))),
        };
        let singularity = p.kikuchi.map_or(Singularity::None, Singularity::Kikuchi);
        let mu: Vec<f64> = match &p.mu.0 {
            Json::Array(v) => v
                .iter()
                .map(|x| x.as_f64().ok_or_else(|| Error::Config("adaptive_equalizer: mu must be numbers".into())))
                .collect::<Result<_>>()?,
            v => vec![v.as_f64().ok_or_else(|| Error::Config("adaptive_equalizer: mu must be a number".into()))?],
        };
        ensure(!mu.is_empty() && mu.iter().all(|m| *m > 0.0), || {
            "adaptive_equalizer: mu must be positive step sizes".into()
        })?;
        Ok(Equalizer { schedule, n_taps: p.n_taps, mu, passes: p.passes, singularity, label })
    }
}

impl Block for Equalizer {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let sps = s.int_sps("adaptive_equalizer")?;
        let eq = AdaptiveEqualizer {
            n_taps: self.n_taps,
            sps,
            mu: self.mu[0],
            stage_mu: self.mu.clone(),
            schedule: self.schedule.clone(),
            passes: self.passes,
            singularity: self.singularity,
        };
        let c = s.constellation("adaptive_equalizer")?.clone();
        // every polarisation (or antenna) is a mode of the butterfly
        let n = s.n_streams();
        let needs_ref = self.schedule.iter().any(|(a, _)| *a == Algorithm::Nlms);
        let reference: Vec<Vec<C64>> = if needs_ref {
            (0..n).map(|i| part(&s.symbols, i, n, "symbols")).collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        let x: Vec<Vec<C64>> = s.streams().map(|v| signal::pnorm(v)).collect();
        let out = eq.run(&x, if needs_ref { Some(&reference) } else { None }, c.points())?;
        if let Some(rho) = out.output_correlation() {
            s.record(&self.label, "correlation", rho);
            s.record(&self.label, "singular", f64::from(u8::from(rho > EqualizerOutput::SINGULAR_CORRELATION)));
        }
        s.set_streams(out.symbols);
        s.sps = 1.0;
        Ok(())
    }
}

block_params! {
    struct FoeSpec {
        m: u32 [4] >= 1, "rotational symmetry";
        refined: bool [true], "refine the periodogram peak between the bins";
    }
}

struct Foe {
    p: FoeSpec,
    label: String,
}

impl Block for Foe {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let fs = s.fs();
        let (y, fo) = if self.p.refined {
            carrier::fourth_power_foe_refined(&s.samples, fs, self.p.m)?
        } else {
            carrier::fourth_power_foe(&s.samples, fs, self.p.m)?
        };
        s.samples = y;
        s.record(&self.label, "foe_hz", fo);
        Ok(())
    }
}

block_params! {
    struct Bps {
        n: usize [35] "symbols", "half window";
        b: usize [64] >= 1, "test phases";
    }
}

impl Block for Bps {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let pts = s.constellation("bps")?.points().to_vec();
        let z = signal::pnorm(&s.samples);
        let theta = carrier::unwrap(&carrier::bps(&z, self.n, &pts, self.b)?, FRAC_PI_2);
        s.samples = carrier::derotate(&z, &theta, 1.0)?;
        Ok(())
    }
}

block_params! {
    struct Viterbi {
        n: usize [35] "symbols", "window";
        m: u32 [4] >= 1, "power";
    }
}

impl Block for Viterbi {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let theta = carrier::viterbi_viterbi(&s.samples, self.n, self.m);
        s.samples = carrier::derotate(&s.samples, &theta, 1.0)?;
        Ok(())
    }
}

block_params! {
    struct Ddpll {
        kv: f64 [0.1], "VCO gain";
        tau1: f64 [1.0 / (2.0 * PI * 10e6) => "1/(2π 10 MHz)"] "s" > 0, "loop-filter time constant";
        tau2: f64 [1.0 / (2.0 * PI * 10e6) => "1/(2π 10 MHz)"] "s" > 0, "loop-filter time constant";
    }
}

impl Block for Ddpll {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let pts = s.constellation("ddpll")?.points().to_vec();
        let z = signal::pnorm(&s.samples);
        let theta = carrier::ddpll(&z, 1.0 / s.symbol_rate, self.kv, self.tau1, self.tau2, &pts, None);
        s.samples = carrier::derotate(&z, &theta, 1.0)?;
        Ok(())
    }
}

block_params! {
    struct FrameSyncSpec {
        start: usize [0], "first known symbol";
        length: usize [512] >= 1, "number of known symbols";
        max_delay: usize [64] "symbols", "largest delay searched";
        symmetry: u32 [4], "rotational symmetry of the constellation; 0 removes the exact phase";
        exclude: bool [true], "exclude the known symbols from the metrics";
    }
}

struct FrameSync {
    p: FrameSyncSpec,
    label: String,
}

impl Block for FrameSync {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        if !s.layout.polarisations || s.n_streams() != 2 {
            return self.single(s);
        }
        // the order of the outputs of a blind 2 × 2 equaliser is arbitrary: keep the
        // order whose known symbols correlate best, then synchronise each polarisation
        s.require_one_sps("frame_sync")?;
        let p = &self.p;
        let n = s.symbols.len() / 2;
        let symmetry = if p.symmetry == 0 { None } else { Some(p.symmetry) };
        let q = |z: &[C64], pol: usize| {
            frame::frame_sync(z, &s.symbols[pol * n..(pol + 1) * n], p.start..p.start + p.length, p.max_delay, symmetry)
                .map_or(0.0, |r| r.quality)
        };
        let direct = q(&s.samples, 0) + q(&s.antennas[0], 1);
        let swapped = q(&s.samples, 1) + q(&s.antennas[0], 0);
        let swap = swapped > direct;
        if swap {
            std::mem::swap(&mut s.samples, &mut s.antennas[0]);
        }
        s.record(&self.label, "pol_swap", f64::from(u8::from(swap)));
        per_pol(s, rng, &|sub, _| self.single(sub))
    }
}

impl FrameSync {
    fn single(&self, s: &mut Frame) -> Result<()> {
        s.require_one_sps("frame_sync")?;
        let p = &self.p;
        let range = p.start..p.start + p.length;
        let symmetry = if p.symmetry == 0 { None } else { Some(p.symmetry) };
        let r = frame::frame_sync(&s.samples, &s.symbols, range.clone(), p.max_delay, symmetry)?;
        let len_in = s.samples.len() as isize;
        s.samples = frame::align(&s.samples, r.delay, r.rotation, s.symbols.len());
        // the symbols whose received sample falls outside the block are excluded
        for (i, e) in s.excluded.iter_mut().enumerate() {
            let k = i as isize + r.delay;
            if k < 0 || k >= len_in {
                *e = true;
            }
        }
        if p.exclude {
            let n = s.excluded.len();
            for i in range.filter(|&i| i < n) {
                s.excluded[i] = true;
            }
        }
        s.record(&self.label, "delay", r.delay as f64);
        s.record(&self.label, "quality", r.quality);
        Ok(())
    }
}

block_params! {
    struct Resample {
        sps: f64 [2.0] > 0, "output samples per symbol";
    }
}

impl Block for Resample {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let n_out = (s.samples.len() as f64 * self.sps / s.sps).round() as usize;
        s.samples = resample::resample_fft(&s.samples, n_out);
        s.sps = self.sps;
        Ok(())
    }
}
