//! Single-polarisation fibre propagation and its compensation: linear fibre, split-step Fourier, EDFA, dispersion compensation, digital back-propagation.

use super::{Block, BuildContext, Extension, FIBER_DOC, FiberSpec, Frame, Mapping};
use com_channel::optical::edfa::Edfa;
use com_channel::optical::fiber::FiberParams;
use com_channel::optical::fiber::linear_fiber_channel;
use com_channel::optical::fiber::{Amplification, Precision, Ssfm};
use com_core::Error;
use com_core::Result;
use com_core::ensure;
use com_core::rng::Rng;
use com_dsp::compensation::cd;
use serde_json::Value as Json;

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            per_polarization "fiber",
            "Linear fibre (loss and dispersion, OptiCommPy linearFiberChannel).",
            FiberBlock,
            needs [Samples],
            gives [],
            |q, _, ctx| {
                let fiber = ctx.fiber(fiber_json(q.fiber.as_ref()))?;
                let length_km = q.length_km.unwrap_or(fiber.total_km());
                Ok(Box::new(LinearFiber { length_km, fiber, compensate: q.compensate_loss }))
            },
        ),
        builtin!(
            per_polarization "ssfm", "Split-step Fourier propagation over the link, one polarisation.", SsfmBlock, needs [Samples], gives [],
            |q, _, ctx| Ok(Box::new(SplitStep::new(&q, ctx)?)),
        ),
        builtin!(
            per_polarization "edfa", "Erbium-doped fibre amplifier with ASE.", EdfaBlock, needs [Samples], gives [],
            |q, _, _| Ok(Box::new(EdfaStage(Edfa { gain_db: q.gain_db, nf_db: q.nf_db, fc_hz: q.fc_hz }))),
        ),
        builtin!(
            per_polarization "cdc", "Chromatic-dispersion compensation in the frequency domain.", CdcBlock, needs [Samples], gives [],
            |q, _, ctx| {
                let fiber = ctx.fiber(fiber_json(q.fiber.as_ref()))?;
                Ok(Box::new(Cdc { length_km: q.length_km.unwrap_or(fiber.total_km()), fiber }))
            },
        ),
        builtin!(
            per_polarization "dbp", "Digital back-propagation by the split-step method.", DbpBlock, needs [Samples], gives [],
            |q, _, ctx| Ok(Box::new(Dbp::new(&q, ctx)?)),
        ),
    ]
}

/// The fibre of a block as given, merged over that of the specification by the context.
pub(super) fn fiber_json(f: Option<&Mapping<FiberSpec>>) -> Option<&Json> {
    f.map(|m| &m.value)
}

block_params! {
    struct FiberBlock {
        fiber: Option<Mapping<FiberSpec>> [absent "spec fiber"], FIBER_DOC;
        length_km: Option<f64> [absent "total"] "km" >= 0, "length; the whole link by default";
        compensate_loss: bool [true], "restore the power with an ideal gain e^(αL/2)";
    }
}

struct LinearFiber {
    fiber: FiberParams,
    length_km: f64,
    compensate: bool,
}

impl Block for LinearFiber {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let mut y = linear_fiber_channel(&s.samples, &self.fiber, self.length_km, s.fs());
        if self.compensate {
            let g = (self.fiber.alpha_np_km() / 2.0 * self.length_km).exp();
            for v in &mut y {
                *v *= g;
            }
        }
        s.samples = y;
        Ok(())
    }
}

block_params! {
    /// The amplifiers after every span: the parameters shared by `ssfm` and `manakov`.
    pub(super) struct AmplifierSettings {
        amplification: String ["edfa"] one_of ["edfa", "ideal", "none"],
            "edfa (gain αL, ASE), ideal (noiseless gain) or none; with fiber.raman, the amplifier restores every frequency";
        noise: bool [true], "draw the ASE of the EDFAs and the spontaneous Raman noise";
    }
}

impl AmplifierSettings {
    /// The amplification after every span.
    pub(super) fn amplification(&self) -> Result<Amplification> {
        self.amplification.parse()
    }

    /// Whether the amplifiers and the Raman pumps add their noise.
    pub(super) fn noise(&self) -> bool {
        self.noise
    }
}

block_params! {
    struct SsfmBlock {
        fiber: Option<Mapping<FiberSpec>> [absent "spec fiber"], FIBER_DOC;
        step_km: f64 [1.0] "km" > 0, "step";
        amplifiers: AmplifierSettings [flatten], "the amplifiers after every span";
        precision: String ["f64"] one_of ["f64", "f32"],
            "f64, or f32 (about twice as fast, relative error about 1e-6 √steps; not with Raman)";
    }
}

struct SplitStep {
    ssfm: Ssfm,
    noise: bool,
    precision: Precision,
}

impl SplitStep {
    fn new(q: &SsfmBlock, ctx: &BuildContext) -> Result<SplitStep> {
        let fiber = ctx.fiber(fiber_json(q.fiber.as_ref()))?;
        let ssfm = Ssfm {
            fiber,
            step_km: q.step_km,
            amplification: q.amplifiers.amplification()?,
            raman: ctx.raman(fiber_json(q.fiber.as_ref()))?,
        };
        let precision: Precision = q.precision.parse().map_err(|e: Error| e.context("ssfm"))?;
        ensure(!(precision == Precision::F32 && ssfm.raman.is_some()), || {
            "ssfm: the Raman amplification is simulated in double precision only".into()
        })?;
        Ok(SplitStep { ssfm, noise: q.amplifiers.noise(), precision })
    }
}

impl Block for SplitStep {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let rng = if self.noise { Some(rng) } else { None };
        s.samples = self.ssfm.propagate_with(&s.samples, s.fs(), rng, self.precision)?;
        Ok(())
    }
}

block_params! {
    struct EdfaBlock {
        gain_db: f64 [required] "dB", "gain";
        nf_db: f64 [4.5] "dB", "noise figure";
        fc_hz: f64 [193.1e12] "Hz" > 0, "carrier frequency";
    }
}

struct EdfaStage(Edfa);

impl Block for EdfaStage {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        s.samples = self.0.apply(&s.samples, s.fs(), rng)?;
        Ok(())
    }
}

// ------------------------------------------------------------------------ receiver

block_params! {
    struct CdcBlock {
        fiber: Option<Mapping<FiberSpec>> [absent "spec fiber"], FIBER_DOC;
        length_km: Option<f64> [absent "total"] "km", "compensated length";
    }
}

struct Cdc {
    fiber: FiberParams,
    length_km: f64,
}

impl Block for Cdc {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        s.samples = cd::cdc(&s.samples, self.fiber.beta2_s2_km(), self.length_km, s.fs());
        Ok(())
    }
}

block_params! {
    struct DbpBlock {
        fiber: Option<Mapping<FiberSpec>> [absent "spec fiber"], FIBER_DOC;
        steps_per_span: usize [1] >= 1, "symmetric steps per span";
        amplification: String ["edfa"] one_of ["edfa", "ideal", "none"], "the amplification of the forward link";
        nonlinear: bool [true], "false: linear back-propagation (dispersion compensation and loss)";
        nonlinear_scale: f64 [1.0] >= 0, "ξ: multiplies the nonlinear phase (optimised DBP with few steps, Ip & Kahn 2008)";
    }
}

struct Dbp {
    ssfm: Ssfm,
    steps: usize,
    xi: f64,
}

impl Dbp {
    fn new(q: &DbpBlock, ctx: &BuildContext) -> Result<Dbp> {
        let mut fiber = ctx.fiber(fiber_json(q.fiber.as_ref()))?;
        if !q.nonlinear {
            fiber.gamma_w_km = 0.0;
        }
        let ssfm = Ssfm { fiber, step_km: 1.0, amplification: q.amplification.parse()?, raman: None };
        Ok(Dbp { ssfm, steps: q.steps_per_span, xi: q.nonlinear_scale })
    }
}

impl Block for Dbp {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        s.samples = self.ssfm.backpropagate_scaled(&s.samples, s.fs(), self.steps, self.xi)?;
        Ok(())
    }
}
