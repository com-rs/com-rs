//! Wireless blocks: several antennas, fading channels, OFDM and MIMO detection.
//!
//! | Block | Does |
//! |---|---|
//! | `mimo_tx` | spreads the symbols over $n_{t}$ antennas: spatial multiplexing (one layer per antenna, power $1/n_{t}$ each) or Alamouti's code |
//! | `rayleigh` | flat Rayleigh or Rician fading, $n_{r} \times n_{t}$, constant over the trial or over coherence blocks, with exponential antenna correlation |
//! | `tdl` | a tapped delay line (EPA, EVA, ETU, ITU Pedestrian A and Vehicular A, exponential) with Doppler |
//! | `cdl` | a clustered delay line of 3GPP TR 38.901 (CDL-A to CDL-E) between uniform linear arrays, with the Doppler of a moving receiver |
//! | `ofdm_tx` | maps every antenna's symbols on a resource grid with pilots and modulates it |
//! | `ofdm_rx` | demodulates, estimates the channel (perfect, LS or LMMSE) and detects every resource element |
//! | `mimo_rx` | detects a single-carrier link over flat fading with the known channel |
//!
//! # Scales
//!
//! The OFDM modulator gives the samples the power $K/(L N)$ ($K$ active subcarriers,
//! IFFT of size $L N$), so that a resource element of unit energy spreads over
//! $L N / K$ samples: `ofdm_tx` sets the samples per symbol to $L N / K$ and the symbol
//! rate to the rate of the resource elements $K \Delta f$. The noise blocks then set the
//! SNR per resource element: `awgn` with `snr_db` gives $E_{s}/N_{0}$ on every data
//! resource element of unit energy, and the noise of every demodulated resource
//! element has the variance of the noise per sample. Spatial multiplexing divides the
//! power among the antennas, so that the SNR is the total transmitted energy per
//! resource element over the noise of one receive antenna.
//!
//! # Example
//!
//! ```yaml
//! link:
//!   - {type: source, constellation: 16-QAM, n_symbols: 4800}
//!   - {type: mimo_tx, n_tx: 2}
//!   - {type: ofdm_tx, n_fft: 256, n_subcarriers: 200, n_cp: 18, n_symbols: 14}
//!   - {type: tdl, profile: EPA, doppler_hz: 5, n_rx: 2}
//!   - {type: awgn, snr_db: $snr_db}
//!   - {type: ofdm_rx, estimation: lmmse, detector: mmse}
//! ```
//!
//! # References
//!
//! 1. S. M. Alamouti, "A simple transmit diversity technique for wireless
//!    communications," *IEEE J. Sel. Areas Commun.*, 16(8), 1451–1458, 1998.
//! 2. O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson and P. O. Börjesson, "OFDM
//!    channel estimation by singular value decomposition," *IEEE Trans. Commun.*,
//!    46(7), 931–939, 1998.
//! 3. 3GPP TS 36.101, *E-UTRA; User Equipment radio transmission and reception*,
//!    Annex B.2 (EPA, EVA, ETU).

use super::schema::ParamType;
use super::{Block, BlockInfo, BuildContext, Frame, Params};
use crate::params::parse;
use com_channel::cdl::{Array, Cdl, Profile as CdlProfile};
use com_channel::fading::Mimo;
use com_channel::multipath::{Profile, Taps, Tdl};
use com_channel::spatial::{Kronecker, exponential_correlation};
use com_core::rng::Rng;
use com_core::{Error, Result, ensure};
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_math::linalg::CMat;
use com_mimo::receiver::{DetectorKind, Receiver, SoftOutput};
use com_mimo::stbc::Ostbc;
use com_waveform::ofdm::Ofdm;
use com_waveform::ofdm::estimation::{FrequencyInterpolation, PowerDelayProfile};
use com_waveform::ofdm::grid::{PilotPattern, ResourceGrid};
use com_waveform::ofdm::link::transmit;
use serde::Deserialize;
use serde_json::Value as Json;

/// The channel of a trial, as a genie-aided receiver knows it.
#[derive(Clone, Debug)]
pub enum Csi {
    /// Flat fading: one matrix $n_{r} \times n_{t}$ per block of `coherence` samples.
    Flat {
        /// The matrices, block after block.
        h: Vec<CMat>,
        /// Samples per block.
        coherence: usize,
    },
    /// The taps of a tapped delay line.
    Taps(Taps),
}

impl Csi {
    /// The flat channel at sample `n`.
    fn flat(&self, n: usize) -> Option<&CMat> {
        match self {
            Csi::Flat { h, coherence } => h.get(n / (*coherence).max(1)).or_else(|| h.last()),
            Csi::Taps(_) => None,
        }
    }
}

pub use com_waveform::ofdm::link::OfdmFrame;

/// The documentation of the wireless blocks.
pub(super) fn infos() -> Vec<BlockInfo> {
    vec![
        block_info!(
            "mimo_tx",
            "Spread the symbols over n_tx antennas: spatial multiplexing (power 1/n_tx per antenna) or Alamouti's code.",
            MimoTxSpec,
            needs [Samples],
            gives [],
        ),
        block_info!(
            "rayleigh",
            "Flat fading y = H x on every antenna; records the channel for a genie-aided receiver.",
            RayleighSpec,
            needs[Samples],
            gives[Csi],
        ),
        block_info!(
            "tdl",
            "Tapped-delay-line multipath fading with Clarke/Jakes Doppler; records the taps for a genie-aided receiver.",
            TdlSpec,
            needs[Samples],
            gives[Csi],
        ),
        block_info!(
            "cdl",
            "Clustered delay line of 3GPP TR 38.901 between uniform linear arrays (one tap per cluster, random ray phases and coupling); records the taps for a genie-aided receiver.",
            CdlSpec,
            needs[Samples],
            gives[Csi],
        ),
        block_info!(
            "ofdm_tx",
            "Map every antenna's symbols on resource grids with pilots and OFDM-modulate them (power K/(LN) per sample).",
            OfdmTxSpec,
            needs[Samples],
            gives[Ofdm],
        ),
        block_info!(
            "ofdm_rx",
            "Demodulate the OFDM grids, estimate the channel and detect every data resource element; output the symbols and their noise variances.",
            OfdmRxSpec,
            needs [Samples, Constellation, Ofdm],
            gives [Noise, Llr],
        ),
        block_info!(
            "mimo_rx",
            "Detect a single-carrier link over flat fading with the known channel; output the symbols and their noise variances.",
            DetectorSpec,
            needs [Samples, Constellation, Csi],
            gives [Noise, Llr],
        ),
    ]
}

/// Build a wireless block, or `None` for another type.
pub(super) fn build(kind: &str, _label: &str, p: &Json, _ctx: &BuildContext) -> Option<Result<Box<dyn Block>>> {
    let p = p.clone();
    let b: Result<Box<dyn Block>> = match kind {
        "mimo_tx" => MimoTxSpec::parse(kind, p).and_then(|q| MimoTx::new(&q)).map(|b| Box::new(b) as Box<dyn Block>),
        "rayleigh" => RayleighSpec::parse(kind, p).and_then(Rayleigh::new).map(|b| Box::new(b) as _),
        "tdl" => TdlSpec::parse(kind, p).and_then(|q| TdlBlock::new(&q)).map(|b| Box::new(b) as _),
        "cdl" => CdlSpec::parse(kind, p).and_then(|q| CdlBlock::new(&q)).map(|b| Box::new(b) as _),
        "ofdm_tx" => OfdmTxSpec::parse(kind, p).and_then(|q| OfdmTx::new(&q)).map(|b| Box::new(b) as _),
        "ofdm_rx" => OfdmRxSpec::parse(kind, p).and_then(|q| OfdmRx::new(&q)).map(|b| Box::new(b) as _),
        "mimo_rx" => DetectorSpec::parse(kind, p).and_then(|q| Detection::new(&q)).map(|d| Box::new(MimoRx(d)) as _),
        _ => return None,
    };
    Some(b)
}

// ------------------------------------------------------------------------ mimo_tx

block_params! {
    struct MimoTxSpec {
        n_tx: usize [2] >= 1, "transmit antennas";
        scheme: Scheme ["spatial"] one_of ["spatial", "alamouti"],
            "spatial (V-BLAST, symbol i on antenna i mod n_tx) or alamouti (n_tx = 2)";
    }
}

/// The transmit scheme of `mimo_tx`.
#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Scheme {
    /// Spatial multiplexing.
    Spatial,
    /// Alamouti's code.
    Alamouti,
}

impl ParamType for Scheme {
    fn schema() -> Json {
        String::schema()
    }
}

struct MimoTx {
    n_tx: usize,
    alamouti: bool,
}

impl MimoTx {
    fn new(p: &MimoTxSpec) -> Result<MimoTx> {
        let alamouti = p.scheme == Scheme::Alamouti;
        ensure(!alamouti || p.n_tx == 2, || "mimo_tx: Alamouti's code has two antennas".into())?;
        Ok(MimoTx { n_tx: p.n_tx, alamouti })
    }
}

impl Block for MimoTx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        ensure(s.antennas.is_empty(), || "mimo_tx: the signal already has several antennas".into())?;
        s.require_one_sps("mimo_tx")?;
        let x = std::mem::take(&mut s.samples);
        let nt = self.n_tx;
        ensure(x.len() % nt == 0, || format!("mimo_tx: {} symbols is not a multiple of {nt} antennas", x.len()))?;
        let streams: Vec<Vec<C64>> = if self.alamouti {
            let m = Ostbc::alamouti().encode(&x)?;
            m.rows().into_iter().map(|r| r.to_vec()).collect()
        } else {
            let a = 1.0 / (nt as f64).sqrt();
            (0..nt).map(|t| x.iter().skip(t).step_by(nt).map(|v| v * a).collect()).collect()
        };
        s.set_streams(streams);
        s.layout.layers = nt;
        s.layout.alamouti = self.alamouti;
        Ok(())
    }
}

// ----------------------------------------------------------------------- rayleigh

block_params! {
    struct RayleighSpec {
        n_rx: usize [1] >= 1, "receive antennas";
        coherence: usize [0] "samples", "samples per channel matrix (0: one matrix per trial, 1: fast fading)";
        k_factor: f64 [0.0] >= 0, "Rician factor K (linear), 0 for Rayleigh fading";
        rx_correlation: f64 [0.0], "exponential correlation ρ of the receive antennas, in (-1, 1)";
        tx_correlation: f64 [0.0], "exponential correlation ρ of the transmit antennas, in (-1, 1)";
    }
}

struct Rayleigh {
    p: RayleighSpec,
}

impl Rayleigh {
    fn new(p: RayleighSpec) -> Result<Rayleigh> {
        ensure(p.rx_correlation.abs() < 1.0 && p.tx_correlation.abs() < 1.0, || {
            "rayleigh: the correlations must be in (-1, 1)".into()
        })?;
        Ok(Rayleigh { p })
    }
}

impl Block for Rayleigh {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        // the transmitted power, summed over the antennas: the reference of the noise
        s.channel.faded_power = Some(s.mean_power() * s.n_streams() as f64);
        let x = s.take_streams();
        let (nt, nr, n) = (x.len(), self.p.n_rx, x[0].len());
        let correlation = if self.p.rx_correlation != 0.0 || self.p.tx_correlation != 0.0 {
            Some(Kronecker::new(
                exponential_correlation(nr, C64::new(self.p.rx_correlation, 0.0)),
                exponential_correlation(nt, C64::new(self.p.tx_correlation, 0.0)),
            )?)
        } else {
            None
        };
        let coh = if self.p.coherence == 0 { n.max(1) } else { self.p.coherence };
        let model = Mimo { n_rx: nr, n_tx: nt, k_factor: self.p.k_factor, los: None, correlation, coherence: coh };
        let (y, hs) = model.apply(&x, rng)?;
        s.set_streams(y);
        s.channel.csi = Some(Csi::Flat { h: hs, coherence: coh });
        Ok(())
    }
}

// ---------------------------------------------------------------------------- tdl

block_params! {
    struct TdlSpec {
        profile: Json ["EPA"],
            "EPA, EVA, ETU (3GPP TS 36.101), pedestrian_a, pedestrian_b, vehicular_a, vehicular_b (ITU-R M.1225, Tables 4 and 5), or {rms_delay_s, resolution_s} (exponential)";
        doppler_hz: f64 [0.0] "Hz" >= 0, "maximum Doppler shift; 0 for a channel constant over the trial";
        n_rx: usize [1] >= 1, "receive antennas";
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExponentialSpec {
    rms_delay_s: f64,
    resolution_s: f64,
}

struct TdlBlock {
    taps: Vec<(f64, f64)>,
    doppler: f64,
    n_rx: usize,
}

impl TdlBlock {
    fn new(p: &TdlSpec) -> Result<TdlBlock> {
        let profile = match &p.profile {
            Json::String(name) => name.parse()?,
            other => {
                let e: ExponentialSpec = parse("tdl profile", other.clone())?;
                ensure(e.rms_delay_s > 0.0 && e.resolution_s > 0.0, || {
                    "tdl: rms_delay_s and resolution_s must be positive".into()
                })?;
                Profile::Exponential { rms_delay: e.rms_delay_s, resolution: e.resolution_s }
            }
        };
        Ok(TdlBlock { taps: profile.taps(), doppler: p.doppler_hz, n_rx: p.n_rx })
    }
}

impl Block for TdlBlock {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        // the transmitted power, summed over the antennas: the reference of the noise
        s.channel.faded_power = Some(s.mean_power() * s.n_streams() as f64);
        let x = s.take_streams();
        let (nt, n) = (x.len(), x[0].len());
        let tdl = Tdl::new(self.taps.clone(), self.doppler, s.fs(), self.n_rx, nt)?;
        let taps = tdl.draw(n, rng);
        let y = tdl.apply(&x, &taps)?;
        s.set_streams(y);
        s.channel.csi = Some(Csi::Taps(taps));
        Ok(())
    }
}

// ---------------------------------------------------------------------------- cdl

block_params! {
    struct CdlSpec {
        profile: String ["CDL-A"] one_of ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"],
            "the profile of TR 38.901 Tables 7.7.1-1 to 7.7.1-5 (A, B, C: NLOS; D, E: LOS)";
        delay_spread_s: f64 [100e-9] "s" >= 0, "desired RMS delay spread (TR 38.901 §7.7.3)";
        n_rx: usize [1] >= 1, "receive antennas, a vertical uniform linear array";
        spacing: f64 [0.5] "wavelengths" > 0, "element spacing of both arrays";
        velocity_mps: f64 [0.0] "m/s" >= 0, "speed of the receiver along the array axis; 0 for a channel constant over the trial";
        fc_hz: f64 [3.5e9] "Hz" > 0, "carrier frequency (the wavelength of the Doppler shifts)";
    }
}

struct CdlBlock {
    profile: CdlProfile,
    delay_spread: f64,
    n_rx: usize,
    spacing: f64,
    velocity: f64,
    carrier: f64,
}

impl CdlBlock {
    fn new(p: &CdlSpec) -> Result<CdlBlock> {
        Ok(CdlBlock {
            profile: p.profile.parse()?,
            delay_spread: p.delay_spread_s,
            n_rx: p.n_rx,
            spacing: p.spacing,
            velocity: p.velocity_mps,
            carrier: p.fc_hz,
        })
    }
}

impl Block for CdlBlock {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        s.channel.faded_power = Some(s.mean_power() * s.n_streams() as f64);
        let x = s.take_streams();
        let (nt, n) = (x.len(), x[0].len());
        let mut cdl = Cdl::new(
            self.profile,
            self.delay_spread,
            s.fs(),
            Array::linear(nt, self.spacing),
            Array::linear(self.n_rx, self.spacing),
        )?;
        cdl.velocity = [0.0, self.velocity, 0.0];
        cdl.carrier_frequency = self.carrier;
        let taps = cdl.draw(n, rng)?;
        let y = taps.apply(&x)?;
        s.set_streams(y);
        s.channel.csi = Some(Csi::Taps(taps));
        Ok(())
    }
}

// ------------------------------------------------------------------------ ofdm_tx

block_params! {
    struct OfdmTxSpec {
        n_fft: usize [256], "subcarrier grid N";
        n_subcarriers: usize [200], "active subcarriers K";
        n_cp: usize [16] "samples", "cyclic prefix (Nyquist-rate samples)";
        oversampling: usize [1] >= 1, "oversampling L";
        dc_null: bool [false], "leave the DC subcarrier empty";
        n_symbols: usize [14] >= 1, "OFDM symbols per grid";
        pilots: Json [serde_json::json!({"pattern": "lattice", "freq": 3, "time": 4})
            => "{pattern: lattice, freq: 3, time: 4}"],
            "pattern block {period}, comb {spacing}, lattice {freq, time} or none; with several antennas the pilots of a symbol are dealt to the antennas in turn";
        subcarrier_spacing_hz: f64 [15e3] "Hz" > 0, "subcarrier spacing Δf";
        seed: u64 [0], "seed of the pilot symbols";
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PilotSpec {
    pattern: String,
    #[serde(default = "d_four")]
    period: usize,
    #[serde(default = "d_four")]
    spacing: usize,
    #[serde(default = "d_three")]
    freq: usize,
    #[serde(default = "d_four")]
    time: usize,
}

fn d_four() -> usize {
    4
}
fn d_three() -> usize {
    3
}

fn pilot_pattern(v: &Json) -> Result<PilotPattern> {
    if let Json::String(s) = v {
        ensure(s == "none", || format!("ofdm_tx: pilots must be none or a mapping, got {s:?}"))?;
        return Ok(PilotPattern::None);
    }
    let p: PilotSpec = parse("ofdm_tx pilots", v.clone())?;
    ensure(p.period >= 1 && p.spacing >= 1 && p.freq >= 1 && p.time >= 1, || {
        "ofdm_tx: the pilot spacings must be at least 1".into()
    })?;
    PilotPattern::from_name(&p.pattern, p.period, p.spacing, p.freq, p.time).map_err(|e| e.context("ofdm_tx"))
}

struct OfdmTx {
    ofdm: Ofdm,
    grid: ResourceGrid,
    scs: f64,
}

impl OfdmTx {
    fn new(p: &OfdmTxSpec) -> Result<OfdmTx> {
        let ofdm = Ofdm {
            n_fft: p.n_fft,
            n_active: p.n_subcarriers,
            oversampling: p.oversampling,
            n_cp: p.n_cp,
            dc_null: p.dc_null,
            dft_spread: false,
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

// ---------------------------------------------------------------------- detection

block_params! {
    struct DetectorSpec {
        detector: String ["mmse"] one_of ["zf", "mmse", "sic", "ml", "sphere", "kbest"],
            "zf, mmse, sic (MMSE-SIC), ml, sphere or kbest";
        k: usize [8] >= 1, "paths of kbest";
        llr: bool [false], "also compute the bit LLRs with the detector (zf, mmse, ml, kbest)";
    }
}

/// The detector of a receiver block and whether it outputs LLRs.
struct Detection {
    kind: DetectorKind,
    llr: bool,
}

impl Detection {
    fn new(p: &DetectorSpec) -> Result<Detection> {
        let kind = DetectorKind::from_name(&p.detector, p.k)?;
        ensure(!p.llr || kind.has_soft_output(), || {
            "the sic and sphere detectors have no soft output: use llr: false and a demap block".into()
        })?;
        Ok(Detection { kind, llr: p.llr })
    }

    fn receiver(&self, c: &Constellation) -> Result<Receiver> {
        Receiver::new(self.kind, c.clone(), self.llr)
    }
}

/// The amplitude of every layer: $1/\sqrt{n_{t}}$ with spatial multiplexing (the
/// receivers detect with the effective channel $H/\sqrt{n_{t}}$), 1 otherwise (Alamouti's
/// code carries its own scale).
fn layer_gain(s: &Frame, nt: usize) -> f64 {
    if s.layout.alamouti { 1.0 } else { 1.0 / (nt as f64).sqrt() }
}

/// Write the detected symbols into the signal: one sample per transmitted symbol.
fn finish(s: &mut Frame, mut d: SoftOutput) -> Result<()> {
    let n = s.symbols.len();
    ensure(d.symbols.len() >= n, || {
        format!("the receiver detected {} of the {n} transmitted symbols", d.symbols.len())
    })?;
    d.symbols.truncate(n);
    d.noise_var.truncate(n);
    s.samples = d.symbols;
    s.antennas.clear();
    s.receiver.symbol_noise = d.noise_var;
    if !d.llr.is_empty() {
        let b = s.constellation("detector")?.bits_per_symbol();
        d.llr.truncate(n * b);
        s.coding.llr = d.llr;
    }
    s.sps = 1.0;
    Ok(())
}

// ------------------------------------------------------------------------ ofdm_rx

block_params! {
    struct OfdmRxSpec {
        estimation: Estimation ["perfect"] one_of ["perfect", "ls", "lmmse"],
            "perfect (the channel of the channel block), ls (linear interpolation) or lmmse (uniform delay profile over the prefix)";
        detector: String ["mmse"] one_of ["zf", "mmse", "sic", "ml", "sphere", "kbest"],
            "zf, mmse, sic (MMSE-SIC), ml, sphere or kbest";
        k: usize [8] >= 1, "paths of kbest";
        llr: bool [false], "also compute the bit LLRs with the detector (zf, mmse, ml, kbest)";
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

struct OfdmRx {
    estimation: Estimation,
    detection: Detection,
}

impl OfdmRx {
    fn new(p: &OfdmRxSpec) -> Result<OfdmRx> {
        let estimation = p.estimation;
        let detection = Detection::new(&DetectorSpec { detector: p.detector.clone(), k: p.k, llr: p.llr })?;
        Ok(OfdmRx { estimation, detection })
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
        let noise_var = s.channel.noise_var.unwrap_or(1e-12).max(1e-12);
        let method = match self.estimation {
            Estimation::Perfect => None,
            Estimation::Ls => Some(FrequencyInterpolation::Linear),
            Estimation::Lmmse => {
                let max_delay = (f.ofdm.n_cp * f.ofdm.oversampling).max(1);
                Some(FrequencyInterpolation::Lmmse(PowerDelayProfile::uniform(max_delay, f.ofdm.fft_len())))
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
            gain,
            state.layout.alamouti,
            noise_var,
        )?;
        s.symbol_rate *= nt as f64;
        finish(s, out)
    }
}

// ------------------------------------------------------------------------ mimo_rx

struct MimoRx(Detection);

impl Block for MimoRx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let receiver = self.0.receiver(s.constellation("mimo_rx")?)?;
        let csi =
            s.channel.csi.clone().ok_or_else(|| Error::Config("mimo_rx needs a rayleigh block before it".into()))?;
        ensure(matches!(csi, Csi::Flat { .. }), || {
            "mimo_rx needs flat fading (rayleigh); use ofdm_rx after tdl".into()
        })?;
        let streams = s.take_streams();
        let n = streams[0].len();
        let noise_var = s.channel.noise_var.unwrap_or(1e-12).max(1e-12);
        let mut out = SoftOutput::default();
        let a = layer_gain(s, csi.flat(0).map_or(1, ndarray::ArrayBase::ncols));
        receiver
            .detect_uses(
                0..n,
                s.layout.alamouti,
                noise_var,
                |i| csi.flat(i).map(|h| h.mapv(|v| v * a)).ok_or_else(|| Error::Config("empty channel".into())),
                |i| streams.iter().map(|x| x[i]).collect(),
                &mut out,
            )
            .map_err(|e| e.context("mimo_rx"))?;
        finish(s, out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BuildContext, build};
    use super::*;
    use serde_json::json;

    fn run(link: &[Json], seed: u64) -> Frame {
        let mut rng = Rng::new(seed);
        let mut s = Frame::default();
        for b in link {
            build(b, &BuildContext::default()).unwrap().apply(&mut s, &mut rng).unwrap();
        }
        s
    }

    fn max_error(s: &Frame) -> f64 {
        assert_eq!(s.samples.len(), s.symbols.len());
        s.samples.iter().zip(&s.symbols).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max)
    }

    #[test]
    fn noiseless_siso_ofdm_over_multipath_is_exact() {
        for est in ["perfect", "ls", "lmmse"] {
            let s = run(
                &[
                    json!({"type": "source", "constellation": "16-QAM", "n_symbols": 1000}),
                    json!({"type": "ofdm_tx", "n_fft": 64, "n_subcarriers": 48, "n_cp": 8, "pilots": {"pattern": "block", "period": 2}, "n_symbols": 4}),
                    json!({"type": "tdl", "profile": "EPA"}),
                    json!({"type": "ofdm_rx", "estimation": est, "detector": "zf"}),
                ],
                1,
            );
            // pilot symbols on every subcarrier and a static channel: LS is exact; LMMSE smooths a little
            let tol = if est == "lmmse" { 1e-3 } else { 1e-9 };
            assert!(max_error(&s) < tol, "{est}: {}", max_error(&s));
        }
    }

    #[test]
    fn noiseless_mimo_ofdm_with_orthogonal_pilots_is_exact() {
        let s = run(
            &[
                json!({"type": "source", "constellation": "QPSK", "n_symbols": 2000}),
                json!({"type": "mimo_tx", "n_tx": 2}),
                json!({"type": "ofdm_tx", "n_fft": 64, "n_subcarriers": 48, "n_cp": 16, "pilots": {"pattern": "block", "period": 2}, "n_symbols": 4}),
                json!({"type": "tdl", "profile": {"rms_delay_s": 2e-7, "resolution_s": 1e-6}, "n_rx": 2}),
                json!({"type": "ofdm_rx", "estimation": "perfect", "detector": "ml"}),
            ],
            2,
        );
        assert!(max_error(&s) < 1e-9);
    }

    #[test]
    fn the_snr_is_set_per_resource_element() {
        let s = run(
            &[
                json!({"type": "source", "constellation": "QPSK", "n_symbols": 40_000}),
                json!({"type": "ofdm_tx", "n_fft": 256, "n_subcarriers": 160, "oversampling": 2, "pilots": "none"}),
                json!({"type": "awgn", "snr_db": 10.0}),
                json!({"type": "ofdm_rx", "detector": "zf"}),
            ],
            3,
        );
        let mse: f64 =
            s.samples.iter().zip(&s.symbols).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / s.samples.len() as f64;
        assert!((mse / 0.1 - 1.0).abs() < 0.05, "{mse}");
        assert!((s.receiver.symbol_noise[0] / 0.1 - 1.0).abs() < 0.02);
    }

    #[test]
    fn alamouti_over_flat_fading_is_exact_without_noise() {
        let s = run(
            &[
                json!({"type": "source", "constellation": "16-QAM", "n_symbols": 400}),
                json!({"type": "mimo_tx", "scheme": "alamouti"}),
                json!({"type": "rayleigh", "n_rx": 1, "coherence": 2}),
                json!({"type": "mimo_rx"}),
            ],
            4,
        );
        assert!(max_error(&s) < 1e-9);
    }

    #[test]
    fn spatial_multiplexing_with_soft_outputs() {
        let s = run(
            &[
                json!({"type": "source", "constellation": "QPSK", "n_symbols": 400}),
                json!({"type": "mimo_tx", "n_tx": 2}),
                json!({"type": "rayleigh", "n_rx": 2, "coherence": 1}),
                json!({"type": "awgn", "snr_db": 40.0, "reference": "unit"}),
                json!({"type": "mimo_rx", "detector": "ml", "llr": true}),
            ],
            5,
        );
        assert_eq!(s.coding.llr.len(), 800);
        let hard: Vec<u8> = s.coding.llr.iter().map(|l| u8::from(*l < 0.0)).collect();
        assert_eq!(hard, s.bits);
    }
}
