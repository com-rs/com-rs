//! Optical blocks: wavelength-division multiplexing and dual-polarisation nonlinear
//! propagation.
//!
//! | Block | Does |
//! |---|---|
//! | `wdm_tx` | the channel under test and `n_channels - 1` independently modulated neighbours on a frequency grid, shaped, at a power per channel, with laser phase noise |
//! | `manakov` | dual-polarisation propagation over the multi-span link by the Manakov equation (OptiCommPy's `manakovSSF`), EDFAs with ASE |
//! | `pmd` | a random rotation of the polarisations and polarisation-mode dispersion (coarse-step model of Marcuse, Menyuk & Wai), a new realisation every trial |
//! | `manakov_dbp` | dual-polarisation digital back-propagation of the link (the Manakov equation with the coefficients negated) |
//! | `wdm_rx` | the channel under test brought to baseband and low-pass filtered (an ideal coherent receiver with its local oscillator on the channel) |
//!
//! # Dual polarisation
//!
//! `source` with `pols: 2` draws the symbols of both polarisations: `symbols` holds those
//! of polarisation x then those of y, `samples` carries x and the first entry of
//! `antennas` carries y. The blocks written for one polarisation (pulse shaping,
//! dispersion compensation, matched filter, clock recovery, frequency and phase
//! estimation, frame synchronisation) then process each polarisation with its own
//! symbols; `adaptive_equalizer` is the 2 × 2 butterfly that undoes the rotations of
//! the polarisation; the metrics pool both polarisations.
//!
//! # Example
//!
//! ```yaml
//! fiber: {n_spans: 10, span_km: 80, d_ps_nm_km: 16, gamma_w_km: 1.3}
//! link:
//!   - {type: source, constellation: 16-QAM, n_symbols: 8192, symbol_rate: 32e9, pols: 2}
//!   - {type: wdm_tx, n_channels: 5, spacing_hz: 50e9, sps: 8, power_dbm: $power_dbm, rolloff: 0.1}
//!   - manakov
//!   - {type: wdm_rx, sps: 2}
//!   - cdc
//!   - {type: matched_filter, rolloff: 0.1}
//!   - decimate
//!   - {type: frame_sync, length: 512}
//! metrics: {report: [ber, snr_eff_db, gmi], skip_start: 64, skip_end: 64}
//! ```
//!
//! # Raman amplification
//!
//! A `raman` mapping in the fibre parameters (of the specification or of a block) pumps
//! every span of `ssfm`, `manakov` and of the theory `gn_snr_db` (see
//! `com_channel::optical::raman`):
//!
//! ```yaml
//! fiber:
//!   n_spans: 3
//!   span_km: 100
//!   raman:
//!     pumps:                       # any number, co- or counter-propagating
//!       - {wavelength_nm: 1425, power_mw: 300, direction: counter}
//!       - {frequency_thz: 205.0, power_mw: 100, direction: co, alpha_db_km: 0.25}
//!     isrs: true                   # the channels also exchange power (default)
//!     aeff_um2: 80                 # effective area (default)
//!     temperature_k: 300           # for the spontaneous noise (default)
//!     step_km: 0.1                 # grid of the power equations (default)
//!     slice_hz: 25e9               # slices of the spectrum of the field (default)
//!     # gain: {offsets_thz: [...], gain_m_w: [...], reference_thz: 206.18}
//! ```
//!
//! The amplifier after every span (`amplification`) then restores every frequency to
//! its launch power. `dbp` and `manakov_dbp` back-propagate the profile of lumped
//! amplifiers.
//!
//! # References
//!
//! 1. E. P. da Silva and D. A. A. Mello, OptiCommPy, `simpleWDMTx` and `manakovSSF`.
//! 2. P. Poggiolini, "The GN model of non-linear propagation in uncoherent optical
//!    systems," *J. Lightw. Technol.*, 30(24), 3857–3879, 2012 (the reference of the
//!    WDM campaigns, see `com_channel::optical::gn`).

use super::dsp::PulseSettings;
use super::fiber::AmplifierSettings;
use super::fiber::fiber_json;
use super::{Block, BuildContext, Extension, FIBER_DOC, FiberSpec, Frame, Mapping, NumberOrList};
use com_channel::optical::manakov::ManakovSsfm;
use com_channel::optical::pmd::{Jones, PmdLink};
use com_core::rng::Rng;
use com_core::signal::DualPol;
use com_core::{Error, Result, ensure};
use com_dsp::signal::{wdm_demultiplex, wdm_grid};
use com_math::nlse::StepRule;
use com_math::{C64, dbm2w};
use com_systems::wdm::WdmTransmitter;
use serde_json::Value as Json;

/// The WDM frame between `wdm_tx` and `wdm_rx`.
#[derive(Clone, Debug)]
pub struct WdmFrame {
    /// The centre frequencies of the channels relative to the carrier (Hz).
    pub freqs_hz: Vec<f64>,
    /// The index of the channel under test.
    pub cut: usize,
}

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            "wdm_tx",
            "WDM transmitter: the channel under test and independently modulated neighbours on the grid of OptiCommPy's simpleWDMTx.",
            WdmTxSpec,
            needs[Samples],
            gives[Wdm],
            |q, _, _| Ok(Box::new(WdmTx::new(&q)?)),
        ),
        builtin!(
            "manakov",
            "Dual-polarisation nonlinear propagation over the link (Manakov equation, OptiCommPy's manakovSSF), EDFAs with ASE.",
            ManakovSpec,
            needs [Samples],
            gives [],
            |q, label, ctx| Ok(Box::new(ManakovBlock::new(&q, ctx, label)?)),
        ),
        builtin!(
            "pmd",
            "Random rotation of the polarisations and polarisation-mode dispersion of a dual-polarisation signal, a new realisation every trial.",
            PmdBlock,
            needs [Samples],
            gives [],
            |q, label, _| Ok(Box::new(Pmd { p: q, label })),
        ),
        builtin!(
            "manakov_dbp",
            "Dual-polarisation digital back-propagation of the link (Manakov equation, coefficients negated).",
            DbpSpec,
            needs [Samples],
            gives [],
            |q, _, ctx| Ok(Box::new(ManakovDbpBlock::new(&q, ctx)?)),
        ),
        builtin!(
            "wdm_rx",
            "Bring the channel under test to baseband and keep its band (ideal low-pass by resampling).",
            WdmRx,
            needs [Samples, Wdm],
            gives [],
            |q, _, _| Ok(Box::new(q)),
        ),
    ]
}

// ------------------------------------------------------------------------- wdm_tx

block_params! {
    struct WdmTxSpec {
        n_channels: usize [5] >= 1, "channels N; the grid is (k - ⌊N/2⌋) Δf, shifted by Δf/2 when N is even";
        spacing_hz: f64 [50e9] "Hz" > 0, "grid spacing Δf";
        sps: usize [8] >= 1, "samples per symbol of the multiplex (the sampling rate must cover the band)";
        power_dbm: NumberOrList [0.0] "dBm", "launch power per channel, both polarisations; a list gives one per channel";
        pulse: PulseSettings [flatten], "the pulse of every channel";
        linewidth_hz: f64 [0.0] "Hz" >= 0, "linewidth of the transmit lasers";
        channel: Option<usize> [absent "centre"], "index of the channel under test (⌊N/2⌋ by default)";
        mzm_scale: Option<f64> [absent],
            "drive the IQ modulator of OptiCommPy at this peak amplitude V/V_π (none: a linear modulator)";
    }
}

struct WdmTx {
    tx: WdmTransmitter,
    cut: usize,
}

impl WdmTx {
    fn new(p: &WdmTxSpec) -> Result<WdmTx> {
        let powers_dbm: Vec<f64> = match &p.power_dbm.0 {
            Json::Array(v) => v
                .iter()
                .map(|x| x.as_f64().ok_or_else(|| Error::Config("wdm_tx: power_dbm must be numbers".into())))
                .collect::<Result<_>>()?,
            v => vec![
                v.as_f64().ok_or_else(|| Error::Config("wdm_tx: power_dbm must be a number".into()))?;
                p.n_channels
            ],
        };
        ensure(powers_dbm.len() == p.n_channels, || {
            format!("wdm_tx: {} powers for {} channels", powers_dbm.len(), p.n_channels)
        })?;
        let cut = p.channel.unwrap_or(p.n_channels / 2);
        ensure(cut < p.n_channels, || format!("wdm_tx: channel {cut} out of the {} channels", p.n_channels))?;
        let tx = WdmTransmitter {
            freqs_hz: wdm_grid(p.n_channels, p.spacing_hz),
            powers_w: powers_dbm.into_iter().map(dbm2w).collect(),
            taps: p.pulse.taps(p.sps)?,
            sps: p.sps,
            linewidth_hz: p.linewidth_hz,
            mzm_scale: p.mzm_scale,
        };
        Ok(WdmTx { tx, cut })
    }
}

impl Block for WdmTx {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        s.require_one_sps("wdm_tx")?;
        // the neighbours carry the format of the channel under test (Gaussian symbols
        // after a gaussian source)
        let c = s.constellation.clone();
        let pols = s.n_streams();
        let n_sym = s.samples.len();
        let cut_streams = s.take_streams();
        let out = self.tx.multiplex(pols, n_sym, s.symbol_rate, rng, |ch, pol, rng| {
            if ch == self.cut {
                cut_streams[pol].clone()
            } else {
                match &c {
                    Some(c) => (0..n_sym).map(|_| c.points()[rng.below(c.order())]).collect(),
                    None => (0..n_sym).map(|_| rng.complex_gaussian()).collect(),
                }
            }
        })?;
        s.set_streams(out);
        s.sps = self.tx.sps as f64;
        s.in_flight.wdm = Some(WdmFrame { freqs_hz: self.tx.freqs_hz.clone(), cut: self.cut });
        Ok(())
    }
}

// ------------------------------------------------------------------------ manakov

block_params! {
    struct ManakovSpec {
        fiber: Option<Mapping<FiberSpec>> [absent "spec fiber"], FIBER_DOC;
        max_nl_phase_rad: f64 [2e-2] "rad" > 0, "largest nonlinear phase rotation per step";
        step_km: Option<f64> [absent] "km" > 0, "a fixed step instead of the adaptive one";
        nonlinear_step: String ["trapezoidal"] one_of ["trapezoidal", "midpoint"],
            "the nonlinear phase of a step: the trapezoidal mean of the power at its ends (OptiCommPy, iterated) or the power at its middle (the scalar solver; more accurate in WDM)";
        max_iterations: usize [10] >= 1, "trapezoidal iterations per step at most";
        tolerance: f64 [1e-5] > 0, "convergence tolerance of the iterations";
        amplifiers: AmplifierSettings [flatten], "the amplifiers after every span";
    }
}

struct ManakovBlock {
    link: ManakovSsfm,
    noise: bool,
    label: String,
}

impl ManakovBlock {
    fn new(p: &ManakovSpec, ctx: &BuildContext, label: String) -> Result<ManakovBlock> {
        let fiber = ctx.fiber(fiber_json(p.fiber.as_ref()))?;
        let rule = p.step_km.map_or(StepRule::NonlinearPhase(p.max_nl_phase_rad), StepRule::Fixed);
        Ok(ManakovBlock {
            link: ManakovSsfm {
                fiber,
                rule,
                nonlinear: p.nonlinear_step.parse()?,
                max_iterations: p.max_iterations,
                tolerance: p.tolerance,
                amplification: p.amplifiers.amplification()?,
                raman: ctx.raman(fiber_json(p.fiber.as_ref()))?,
            },
            noise: p.amplifiers.noise(),
            label,
        })
    }
}

impl Block for ManakovBlock {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        ensure(s.n_streams() <= 2, || "manakov propagates one or two polarisations".into())?;
        let fs = s.fs();
        let mut streams = s.take_streams();
        let single = streams.len() == 1;
        if single {
            streams.push(vec![C64::new(0.0, 0.0); streams[0].len()]);
        }
        let y = streams.pop().unwrap_or_default();
        let x = streams.pop().unwrap_or_default();
        let mut u = DualPol::new(x, y)?;
        let steps = self.link.propagate(&mut u, fs, if self.noise { Some(rng) } else { None })?;
        s.record(&self.label, "steps", steps as f64);
        s.set_streams(if single { vec![u.x] } else { vec![u.x, u.y] });
        Ok(())
    }
}

// ----------------------------------------------------------------------------- pmd

block_params! {
    struct PmdBlock {
        mean_dgd_ps: f64 [0.0] "ps" >= 0, "mean differential group delay of the link; 0 for a rotation only";
        sections: usize [20] >= 1, "birefringent sections of the coarse-step model";
        theta_deg: Option<f64> [absent] "deg", "a fixed real rotation by this angle instead, without DGD";
    }
}

/// The `pmd` block: its parameters and the prefix of its recorded values.
struct Pmd {
    p: PmdBlock,
    label: String,
}

impl Block for Pmd {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let PmdBlock { mean_dgd_ps, sections, theta_deg } = self.p;
        ensure(s.n_streams() == 2, || "pmd needs two polarisations (source with pols: 2)".into())?;
        let (x, rest) = (&mut s.samples, &mut s.antennas);
        if let Some(t) = theta_deg {
            let j = Jones::rotation(t.to_radians());
            for (u, v) in x.iter_mut().zip(rest[0].iter_mut()) {
                (*u, *v) = j.apply(*u, *v);
            }
            return Ok(());
        }
        // the coarse-step model: a rotation only without DGD
        let n = if mean_dgd_ps == 0.0 { 1 } else { sections };
        let link = PmdLink::with_mean_dgd(mean_dgd_ps * 1e-12, n, rng);
        let fs = s.fs();
        let mut u = DualPol::new(std::mem::take(&mut s.samples), std::mem::take(&mut s.antennas[0]))?;
        link.apply(&mut u, fs)?;
        (s.samples, s.antennas[0]) = (u.x, u.y);
        s.record(&self.label, "rms_dgd_ps", link.rms_dgd_s() * 1e12);
        Ok(())
    }
}

// ---------------------------------------------------------------------- manakov_dbp

block_params! {
    struct DbpSpec {
        fiber: Option<Mapping<FiberSpec>> [absent "spec fiber"], FIBER_DOC;
        steps_per_span: usize [1] >= 1, "symmetric steps per span";
        nonlinear_scale: f64 [1.0], "ξ: multiplies the nonlinear phase (0: dispersion compensation only)";
        amplification: String ["edfa"] one_of ["edfa", "ideal", "none"], "the amplification of the forward link";
    }
}

struct ManakovDbpBlock(com_dsp::compensation::dbp::Dbp);

impl ManakovDbpBlock {
    fn new(p: &DbpSpec, ctx: &BuildContext) -> Result<ManakovDbpBlock> {
        let link = ManakovSsfm {
            amplification: p.amplification.parse()?,
            ..ManakovSsfm::new(ctx.fiber(fiber_json(p.fiber.as_ref()))?)
        };
        Ok(ManakovDbpBlock(link.dbp(p.steps_per_span, p.nonlinear_scale)))
    }
}

impl Block for ManakovDbpBlock {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        ensure(s.n_streams() <= 2, || "manakov_dbp back-propagates one or two polarisations".into())?;
        let fs = s.fs();
        let n = s.samples.len();
        if s.antennas.is_empty() {
            let mut y = vec![C64::new(0.0, 0.0); n];
            self.0.run_dual(&mut s.samples, &mut y, fs)
        } else {
            let (x, rest) = (&mut s.samples, &mut s.antennas);
            self.0.run_dual(x, &mut rest[0], fs)
        }
    }
}

// ------------------------------------------------------------------------- wdm_rx

block_params! {
    struct WdmRx {
        sps: usize [2] >= 1, "output samples per symbol";
    }
}

impl Block for WdmRx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let w = s.in_flight.wdm.take().ok_or_else(|| Error::Config("wdm_rx needs a wdm_tx before it".into()))?;
        let sps_in = s.int_sps("wdm_rx")?;
        ensure(self.sps >= 1 && self.sps <= sps_in, || {
            format!("wdm_rx: sps must be between 1 and the {sps_in} samples per symbol of the multiplex")
        })?;
        let fs = s.fs();
        let f = w.freqs_hz[w.cut];
        let n_out = s.samples.len() / sps_in * self.sps;
        for x in s.streams_mut() {
            *x = wdm_demultiplex(x, f, fs, n_out);
        }
        s.sps = self.sps as f64;
        Ok(())
    }
}
