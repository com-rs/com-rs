//! Receiver DSP: shaping and matched filtering, resampling and decimation, clock recovery, adaptive equalisation, carrier recovery, frame synchronisation.

use super::schema::ParamType;
use super::{Block, BlockInfo, BuildContext, Frame, Params, part, per_pol};
use com_core::Error;
use com_core::Result;
use com_core::ensure;
use com_core::rng::Rng;
use com_dsp::equalizer::AdaptiveEqualizer;
use com_dsp::equalizer::Algorithm;
use com_dsp::equalizer::Singularity;
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

/// The documentation of the blocks of this module.
pub(super) fn infos() -> Vec<BlockInfo> {
    vec![
        block_info!(
            per_polarization "pulse_shape",
            "Upsample and filter with a transmit pulse, then normalise to unit power.",
            PulseSpec,
            needs [Samples],
            gives [],
        ),
        block_info!("normalize", "Scale the samples to a mean power.", Normalize, needs [Samples], gives []),
        block_info!(
            per_polarization "resample",
            "Band-limited resampling of the block (FFT) to a number of samples per symbol.",
            Resample,
            needs [Samples],
            gives [],
        ),
        block_info!(
            per_polarization "matched_filter",
            "Filter with a receive pulse at the current samples per symbol.",
            MatchedSpec,
            needs [Samples],
            gives [],
        ),
        block_info!(per_polarization "decimate", "Reduce the samples per symbol.", DecimateSpec, needs [Samples], gives []),
        block_info!(per_polarization "gardner", "Gardner clock recovery at 2 samples per symbol.", Gardner, needs [Samples], gives []),
        block_info!(
            "adaptive_equalizer",
            "Adaptive FIR equaliser, output at 1 sample per symbol; the n × n butterfly over the polarisations or antennas.",
            EqualizerSpec,
            needs [Samples, Constellation],
            gives [],
        ),
        block_info!(
            per_polarization "foe",
            "Frequency-offset estimation by the M-th power and correction; records foe_hz.",
            FoeSpec,
            needs [Samples],
            gives [],
        ),
        block_info!(
            per_polarization "bps",
            "Blind phase search, unwrapped with a period of π/2, and correction.",
            Bps,
            needs [Samples, Constellation],
            gives [],
        ),
        block_info!(per_polarization "viterbi", "Viterbi–Viterbi phase estimation and correction.", Viterbi, needs [Samples], gives []),
        block_info!(per_polarization "ddpll", "Decision-directed phase-locked loop.", Ddpll, needs [Samples, Constellation], gives []),
        block_info!(
            "frame_sync",
            "Delay and rotation on known symbols, output aligned on the transmitted symbols; records delay and quality.",
            FrameSyncSpec,
            needs [Samples, Symbols],
            gives [],
        ),
    ]
}

/// Build a block of this module, or `None` for another type.
pub(super) fn build(k: &str, label: &str, p: &Json, _ctx: &BuildContext) -> Option<Result<Box<dyn Block>>> {
    let (p, label) = (p.clone(), label.to_string());
    let block = move || -> Result<Option<Box<dyn Block>>> {
        Ok(Some(match k {
            "pulse_shape" => Box::new(PulseShaping::new(&PulseSpec::parse(k, p)?)?),
            "normalize" => Box::new(Normalize::parse(k, p)?),
            "resample" => Box::new(Resample::parse(k, p)?),
            "matched_filter" => Box::new(MatchedFilter::new(MatchedSpec::parse(k, p)?)?),
            "decimate" => Box::new(Decimate::new(&DecimateSpec::parse(k, p)?)),
            "gardner" => Box::new(Gardner::parse(k, p)?),
            "adaptive_equalizer" => Box::new(Equalizer::new(&EqualizerSpec::parse(k, p)?, label)?),
            "foe" => Box::new(Foe { p: FoeSpec::parse(k, p)?, label }),
            "bps" => Box::new(Bps::parse(k, p)?),
            "viterbi" => Box::new(Viterbi::parse(k, p)?),
            "ddpll" => Box::new(Ddpll::parse(k, p)?),
            "frame_sync" => Box::new(FrameSync { p: FrameSyncSpec::parse(k, p)?, label }),
            _ => return Ok(None),
        }))
    };
    block().transpose()
}

block_params! {
    struct PulseSpec {
        kind: String ["rrc"] one_of ["rect", "nrz", "rc", "rrc"], "transmit pulse";
        sps: usize [2] >= 1, "samples per symbol";
        rolloff: f64 [0.1] >= 0 <= 1, "roll-off of rc and rrc";
        n_taps: usize [1025] >= 1, "taps of rc and rrc (odd: on the sampling instants)";
    }
}

pub(super) fn pulse(kind: &str, sps: usize, n_taps: usize, rolloff: f64) -> Result<Vec<f64>> {
    ensure(sps >= 1 && n_taps >= 1, || "sps and n_taps must be positive".into())?;
    Ok(filter::pulse_shape(PulseShape::from_name(kind, rolloff)?, sps, n_taps))
}

struct PulseShaping {
    taps: Vec<f64>,
    sps: usize,
}

impl PulseShaping {
    fn new(p: &PulseSpec) -> Result<PulseShaping> {
        Ok(PulseShaping { taps: pulse(&p.kind, p.sps, p.n_taps, p.rolloff)?, sps: p.sps })
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
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let a = self.power_dbm.map_or(1.0, |p| dbm2w(p).sqrt());
        if s.antennas.is_empty() {
            s.samples = signal::pnorm(&s.samples).into_iter().map(|v| v * a).collect();
        } else {
            let p = s.mean_power();
            let g = if p > 0.0 { a / p.sqrt() } else { a };
            s.streams_mut().flatten().for_each(|v| *v *= g);
        }
        Ok(())
    }
}

block_params! {
    struct MatchedSpec {
        kind: String ["rrc"] one_of ["rect", "nrz", "rc", "rrc"], "receive pulse";
        rolloff: f64 [0.1] >= 0 <= 1, "roll-off";
        n_taps: usize [1025] >= 1, "taps";
    }
}

struct MatchedFilter {
    kind: String,
    rolloff: f64,
    n_taps: usize,
}

impl MatchedFilter {
    fn new(p: MatchedSpec) -> Result<MatchedFilter> {
        pulse(&p.kind, 2, p.n_taps, p.rolloff)?;
        Ok(MatchedFilter { kind: p.kind, rolloff: p.rolloff, n_taps: p.n_taps })
    }
}

impl Block for MatchedFilter {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let sps = s.int_sps("matched_filter")?;
        let h = pulse(&self.kind, sps, self.n_taps, self.rolloff)?;
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
        algorithm: Json ["cma"], "cma, rde, nlms, dd-lms, or a schedule [[algorithm, symbols], …]";
        n_taps: usize [15] >= 1, "taps";
        mu: Json [1e-3] > 0,
            "step size, or one per stage of the schedule (the last one holds for the following stages)";
        passes: usize [1] >= 1, "passes over the block";
        kikuchi: Option<usize> [absent],
            "2 × 2: Kikuchi's initialisation against the singularity, output 0 alone during this number of symbols, then output 1 as its unitary complement; records correlation and singular";
    }
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
        let schedule = match &p.algorithm {
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
        let mu: Vec<f64> = match &p.mu {
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
        let reference: Vec<Vec<C64>> = (0..n).map(|i| part(&s.symbols, i, n)).collect();
        let needs_ref = self.schedule.iter().any(|(a, _)| *a == Algorithm::Nlms);
        let x: Vec<Vec<C64>> = s.streams().map(|v| signal::pnorm(v)).collect();
        let out = eq.run(&x, if needs_ref { Some(&reference) } else { None }, c.points())?;
        if out.symbols.len() == 2 {
            // |correlation| of the two outputs over the second half: near 1 when both
            // converged to the same polarisation (the singularity)
            let (a, b) = (&out.symbols[0], &out.symbols[1]);
            let h = a.len() / 2;
            let c: C64 = a[h..].iter().zip(&b[h..]).map(|(u, v)| u * v.conj()).sum();
            let e = |v: &[C64]| v.iter().map(C64::norm_sqr).sum::<f64>();
            let rho = c.norm() / (e(&a[h..]) * e(&b[h..])).sqrt().max(f64::MIN_POSITIVE);
            s.record(&self.label, "correlation", rho);
            s.record(&self.label, "singular", f64::from(u8::from(rho > 0.5)));
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
