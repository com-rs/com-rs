//! The MIMO blocks: `mimo_tx` (spatial multiplexing, Alamouti's code), the detector
//! shared by the receivers, and `mimo_rx` over flat fading.

use super::Csi;
use crate::blocks::schema::ParamType;
use crate::blocks::{Block, Frame};
use com_core::rng::Rng;
use com_core::{Error, Result, ensure};
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_mimo::receiver::{DetectorKind, Receiver, SoftOutput};
use com_mimo::stbc::Ostbc;
use serde::Deserialize;
use serde_json::Value as Json;

// ------------------------------------------------------------------------ mimo_tx

block_params! {
    pub(super) struct MimoTxSpec {
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

pub(super) struct MimoTx {
    n_tx: usize,
    alamouti: bool,
}

impl MimoTx {
    pub(super) fn new(p: &MimoTxSpec) -> Result<MimoTx> {
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

// ---------------------------------------------------------------------- detection

block_params! {
    /// The detector of a receiver block (`mimo_rx`, `ofdm_rx`), its parameters shared.
    pub(super) struct DetectorSpec {
        detector: String ["mmse"] one_of (DetectorKind::NAMES),
            "mr (matched filter), zf, mmse, sic (MMSE-SIC), zf_sic (ZF-SIC), ml, sphere, kbest, amp (LAMA) or oamp (OAMP-LMMSE)";
        k: usize [DetectorKind::DEFAULT_PATHS => "8"] >= 1, "paths of kbest";
        iterations: usize [DetectorKind::DEFAULT_ITERATIONS => "10"] >= 1, "iterations of amp and oamp";
        llr: bool [false], "also compute the bit LLRs with the detector (mr, zf, mmse, ml, kbest, amp, oamp)";
    }
}

/// The detector of a receiver block and whether it outputs LLRs.
pub(super) struct Detection {
    kind: DetectorKind,
    llr: bool,
}

impl Detection {
    pub(super) fn new(p: &DetectorSpec) -> Result<Detection> {
        let kind = DetectorKind::from_name(&p.detector, p.k)?.with_iterations(p.iterations)?;
        ensure(!p.llr || kind.has_soft_output(), || {
            format!("the {} detector has no soft output: use llr: false and a demap block", kind.name())
        })?;
        Ok(Detection { kind, llr: p.llr })
    }

    pub(super) fn receiver(&self, c: &Constellation) -> Result<Receiver> {
        Receiver::new(self.kind, c.clone(), self.llr)
    }
}

/// The amplitude of every layer: $1/\sqrt{n_{t}}$ with spatial multiplexing (the
/// receivers detect with the effective channel $H/\sqrt{n_{t}}$), 1 otherwise (Alamouti's
/// code carries its own scale).
pub(super) fn layer_gain(s: &Frame, nt: usize) -> f64 {
    if s.layout.alamouti { 1.0 } else { 1.0 / (nt as f64).sqrt() }
}

/// Write the detected symbols into the signal: one sample per transmitted symbol.
pub(super) fn finish(s: &mut Frame, mut d: SoftOutput) -> Result<()> {
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

// ------------------------------------------------------------------------ mimo_rx

pub(super) struct MimoRx(pub(super) Detection);

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
        let noise_var = s.channel.detector_noise_var();
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
