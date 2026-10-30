//! The OFDM blocks: `ofdm_tx` and `ofdm_rx`.

use super::mimo::{Detection, DetectorSpec, finish, layer_gain};
use super::{Csi, OfdmFrame};
use crate::blocks::schema::ParamType;
use crate::blocks::{Block, Frame, NameOr, Named};
use com_core::rng::Rng;
use com_core::units::Hertz;
use com_core::{Error, Result, ensure};
use com_math::C64;
use com_waveform::ofdm::Ofdm;
use com_waveform::ofdm::estimation::{FrequencyInterpolation, PowerDelayProfile};
use com_waveform::ofdm::grid::{PilotPattern, ResourceGrid};
use com_waveform::ofdm::link::{FrameDetection, transmit};
use serde::Deserialize;
use serde_json::Value as Json;

// ------------------------------------------------------------------------ ofdm_tx

block_params! {
    pub(super) struct OfdmTxSpec {
        n_fft: usize [256], "subcarrier grid N";
        n_subcarriers: usize [200], "active subcarriers K";
        n_cp: usize [16] "samples", "cyclic prefix (Nyquist-rate samples)";
        oversampling: usize [1] >= 1, "oversampling L";
        dc_null: bool [false], "leave the DC subcarrier empty";
        n_symbols: usize [14] >= 1, "OFDM symbols per grid";
        pilots: NameOr<PilotSpec> [serde_json::json!({"pattern": "lattice", "freq": 3, "time": 4})
            => "{pattern: lattice, freq: 3, time: 4}"],
            "pattern block {period}, comb {spacing}, lattice {freq, time} or none; with several antennas the pilots of a symbol are dealt to the antennas in turn";
        subcarrier_spacing_hz: f64 [15e3] "Hz" > 0, "subcarrier spacing Δf";
        seed: u64 [0], "seed of the pilot symbols";
    }
}

block_params! {
    /// The pilots of `ofdm_tx`.
    struct PilotSpec {
        pattern: String [required] one_of ["block", "comb", "lattice", "none"], "the pattern";
        period: usize [4] "OFDM symbols" >= 1, "pilot symbols every period symbols (block)";
        spacing: usize [4] "subcarriers" >= 1, "pilot subcarriers every spacing subcarriers (comb)";
        freq: usize [3] "subcarriers" >= 1, "spacing in frequency (lattice)";
        time: usize [4] "OFDM symbols" >= 1, "spacing in time (lattice)";
    }
}

fn pilot_pattern(v: &NameOr<PilotSpec>) -> Result<PilotPattern> {
    match v.get("ofdm_tx pilots")? {
        Named::Name(s) if s.eq_ignore_ascii_case("none") => Ok(PilotPattern::None),
        Named::Name(s) => Err(Error::Config(format!("ofdm_tx: pilots must be none or a mapping, got {s:?}"))),
        Named::Params(p) => match p.pattern.to_ascii_lowercase().as_str() {
            "block" => PilotPattern::block(p.period),
            "comb" => PilotPattern::comb(p.spacing),
            "lattice" => PilotPattern::lattice(p.freq, p.time),
            // `one_of` leaves only `none`
            _ => Ok(PilotPattern::None),
        }
        .map_err(|e| e.context("ofdm_tx")),
    }
}

pub(super) struct OfdmTx {
    ofdm: Ofdm,
    grid: ResourceGrid,
    scs: f64,
}

impl OfdmTx {
    pub(super) fn new(p: &OfdmTxSpec) -> Result<OfdmTx> {
        let ofdm = Ofdm {
            n_fft: p.n_fft,
            n_active: p.n_subcarriers,
            oversampling: p.oversampling,
            n_cp: p.n_cp,
            dc_null: p.dc_null,
            dft_spread: false,
            sample_rate: Hertz(p.subcarrier_spacing_hz * (p.oversampling * p.n_fft) as f64),
        };
        ofdm.check()?;
        let grid = ResourceGrid {
            n_subcarriers: p.n_subcarriers,
            n_symbols: p.n_symbols,
            pattern: pilot_pattern(&p.pilots)?,
            seed: p.seed,
        };
        ensure(grid.n_data() > 0, || "ofdm_tx: the grid has no data resource element".into())?;
        Ok(OfdmTx { ofdm, grid, scs: p.subcarrier_spacing_hz })
    }
}

impl Block for OfdmTx {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        s.require_one_sps("ofdm_tx")?;
        let (out, frame) = transmit(&self.ofdm, &self.grid, s.take_streams(), rng).map_err(|e| e.context("ofdm_tx"))?;
        let (k, ln) = (self.ofdm.n_active, self.ofdm.fft_len());
        s.set_streams(out);
        s.sps = ln as f64 / k as f64;
        s.symbol_rate = k as f64 * self.scs;
        s.in_flight.ofdm = Some(frame);
        Ok(())
    }
}

// ------------------------------------------------------------------------ ofdm_rx

block_params! {
    pub(super) struct OfdmRxSpec {
        estimation: Estimation ["perfect"] one_of ["perfect", "ls", "lmmse"],
            "perfect (the channel of the channel block), ls (linear interpolation) or lmmse (uniform delay profile over the prefix)";
        detection: DetectorSpec [flatten], "the detector";
    }
}

/// The channel estimation of `ofdm_rx`.
#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Estimation {
    /// The channel the channel block drew.
    Perfect,
    /// Least squares with linear interpolation.
    Ls,
    /// LMMSE with a uniform delay profile over the prefix.
    Lmmse,
}

impl ParamType for Estimation {
    fn schema() -> Json {
        String::schema()
    }
}

pub(super) struct OfdmRx {
    estimation: Estimation,
    detection: Detection,
}

impl OfdmRx {
    pub(super) fn new(p: &OfdmRxSpec) -> Result<OfdmRx> {
        Ok(OfdmRx { estimation: p.estimation, detection: Detection::new(&p.detection)? })
    }
}

/// The channel of every resource element of grid `g` as the channel block drew it
/// (`[re][rx][tx]` flattened): taken in the middle of every OFDM symbol, the identity
/// without a channel block.
fn perfect_channel(s: &Frame, f: &OfdmFrame, nr: usize, g: usize) -> Result<Vec<C64>> {
    let (k, l_n, nt) = (f.grid.n_subcarriers, f.grid.n_symbols, f.n_ports);
    let mut h = vec![C64::new(0.0, 0.0); k * l_n * nr * nt];
    let Some(csi) = s.channel.csi.as_ref() else {
        ensure(nr == nt, || {
            "ofdm_rx: perfect estimation without a channel block needs as many receive as transmit antennas".into()
        })?;
        for re in 0..k * l_n {
            for a in 0..nr {
                h[(re * nr + a) * nt + a] = C64::new(1.0, 0.0);
            }
        }
        return Ok(h);
    };
    let bins = f.active_bins();
    for (l, n) in f.symbol_middles(g).into_iter().enumerate() {
        let resp = match csi {
            Csi::Taps(t) => t.frequency_response(n, &bins, f.ofdm.fft_len()),
            Csi::Flat { .. } => {
                let m = csi.flat(n).ok_or_else(|| Error::Config("ofdm_rx: empty channel".into()))?;
                (0..k).flat_map(|_| m.iter().copied()).collect()
            }
        };
        ensure(resp.len() == k * nr * nt, || {
            format!("ofdm_rx: the channel has another number of antennas than the {nr} × {nt} of the link")
        })?;
        h[l * k * nr * nt..(l + 1) * k * nr * nt].copy_from_slice(&resp);
    }
    Ok(h)
}

impl Block for OfdmRx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let f = s.in_flight.ofdm.take().ok_or_else(|| Error::Config("ofdm_rx needs an ofdm_tx before it".into()))?;
        let receiver = self.detection.receiver(s.constellation("ofdm_rx")?)?;
        let streams = s.take_streams();
        let (nr, nt) = (streams.len(), f.n_ports);
        let noise_var = s.channel.detector_noise_var();
        let method = match self.estimation {
            Estimation::Perfect => None,
            Estimation::Ls => Some(FrequencyInterpolation::Linear),
            Estimation::Lmmse => {
                let max_delay = (f.ofdm.n_cp * f.ofdm.oversampling).max(1);
                Some(FrequencyInterpolation::Lmmse(PowerDelayProfile::uniform(max_delay, f.ofdm.fft_len())?))
            }
        };
        let gain = layer_gain(s, nt);
        let state: &Frame = s;
        let out = f.detect(
            &streams,
            |g, rx| match &method {
                Some(m) => f.estimate(rx, noise_var, m),
                None => perfect_channel(state, &f, nr, g),
            },
            &receiver,
            &FrameDetection { gain, alamouti: state.layout.alamouti, noise_var },
        )?;
        s.symbol_rate *= nt as f64;
        finish(s, out)
    }
}
