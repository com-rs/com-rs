//! The blocks of a declarative link and their registry.
//!
//! A link is a list of blocks applied in order to a [`Frame`]. The first block is a
//! `source`, which draws the transmitted bits and symbols; every other block transforms
//! the current samples (a channel, a device, a DSP stage, a trained receiver) and may
//! record diagnostic values (an estimated frequency offset, the iterations of a fit).
//! [`registry`] lists every block with its parameters, their defaults and their meaning;
//! the command `comrs blocks` prints it.
//!
//! A block is written as a mapping with a `type`, an optional `label` (the prefix of the
//! values it records, the type by default) and its parameters; a block without
//! parameters can be written as its name alone. Unknown parameters are errors.
//!
//! ```yaml
//! link:
//!   - {type: source, constellation: 16-QAM, n_symbols: 4096, symbol_rate: 32e9}
//!   - {type: pulse_shape, sps: 2, rolloff: 0.1}
//!   - {type: awgn, snr_db: $snr_db}
//!   - {type: matched_filter, rolloff: 0.1}
//!   - decimate
//! ```
//!
//! The wireless blocks ([`wireless`]) add antennas, fading channels, OFDM and MIMO
//! detection; the coding blocks ([`coding`]) add channel codes, mapping of coded bits
//! and soft demapping.
//!
//! # The model of a trial
//!
//! [`Frame`] carries one trial along the link. A simulated source sets the
//! transmitted symbols (`symbols`, `indices`, `bits`) and the samples; a coded link
//! starts with `fec_encode`, which draws the information bits (`coding.payload`) and
//! sets the coded bits, then `map`, which sets the symbols. With several transmit
//! antennas or two polarisations the samples of stream 0 stay in `samples` and those of
//! the others go to `antennas`, and [`Layout`] says what they are. The channel blocks
//! act on every stream and record what a genie-aided receiver knows
//! ([`ChannelState`]); a transmitter block leaves its frame to its receiver block in
//! [`InFlight`]. The receiver blocks bring the signal back to one sample per transmitted
//! symbol, aligned on the symbols, as the metrics need, and declare the noise of their
//! outputs ([`ReceiverState`]); the soft demapper and the decoder then fill
//! `coding.llr` and `coding.decoded`.
//!
//! # The families of blocks
//!
//! | Module | Blocks |
//! |---|---|
//! | [`source`] | simulated symbols, measured recordings |
//! | [`channel`] | noise, phase noise, frequency offset, linear channels, IQ imbalance |
//! | [`fiber`] | single-polarisation fibre propagation and its compensation |
//! | [`optical`] | WDM, dual-polarisation propagation, PMD, back-propagation |
//! | [`dsp`] | shaping, filtering, resampling, clock and carrier recovery, equalisation, frame synchronisation |
//! | [`wireless`] | antennas, fading, OFDM, MIMO detection |
//! | [`coding`] | channel codes, probabilistic amplitude shaping, mapping, soft demapping |
//! | [`ce_ofdm`] | constant-envelope OFDM |
//! | [`cpm`] | continuous-phase modulation (CPFSK, MSK, GMSK) |
//! | [`systems`] | the DVB-S2 chain of `com-systems` (ADR 0010) |
//! | [`analysis`] | probes, recordings of the samples (`dump`), the declared posterior of a receiver |
//!
//! Every module documents its blocks next to their code (`infos`); a test checks
//! that the parameters a block accepts and those it documents are the same.

use com_core::rng::Rng;
use com_core::{Error, Result};
use com_mapping::constellation::{Constellation, ConstellationKind};
use serde_json::Value as Json;

#[macro_use]
mod registry;
#[macro_use]
pub mod schema;
mod context;
mod frame;
mod per_pol;

pub mod analysis;
pub mod ce_ofdm;
pub mod channel;
pub mod coding;
pub mod cpm;
pub mod dsp;
pub mod fiber;
pub mod optical;
pub mod source;
pub mod systems;
pub mod wireless;

#[cfg(test)]
pub(crate) mod tests;

pub use crate::params::parse;
pub use context::BuildContext;
use context::FIBER_DOC;
pub(crate) use context::merge_fiber;
pub use frame::{CaptureState, ChannelState, Coding, Frame, InFlight, Layout, ReceiverState};
pub use optical::WdmFrame;
use per_pol::{part, per_pol};
pub use registry::{BlockInfo, Extension, build, check_link, json_schema, register, registry};
pub use schema::{Fact, ParamDefault, ParamInfo, Params};
pub use wireless::{Csi, OfdmFrame};

/// A block of a link.
pub trait Block: Send + Sync {
    /// Apply the block to the signal of one trial, drawing from `rng` only.
    ///
    /// # Errors
    /// Any error of the underlying algorithm (inconsistent signal, invalid parameter).
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()>;
    /// Whether the block is trained once per condition before the trials.
    fn needs_training(&self) -> bool {
        false
    }
    /// Train on the signal that reaches the block on an independent training block.
    ///
    /// # Errors
    /// Any error of the training.
    fn train(&mut self, _s: &Frame, _rng: &mut Rng) -> Result<()> {
        Ok(())
    }
}

/// Parse a constellation, normalised to unit energy: a name of
/// [`Constellation::from_name`] (`16-QAM`, `qam16`, `QPSK`, `BPSK`, `8-PSK`, `4-PAM`,
/// `OOK`) or a mapping `{kind, m}`.
///
/// # Errors
/// [`Error::Config`] for an unknown family or a missing or invalid order.
pub fn parse_constellation(v: &Json) -> Result<Constellation> {
    let c = match v {
        Json::Object(o) => {
            let kind: ConstellationKind = o.get("kind").and_then(Json::as_str).unwrap_or("qam").parse()?;
            let m = o
                .get("m")
                .and_then(Json::as_u64)
                .ok_or_else(|| Error::Config("constellation needs an order m".into()))?;
            Constellation::new(kind, m as usize)?
        }
        Json::String(s) => Constellation::from_name(s)?,
        other => return Err(Error::Config(format!("constellation must be a name or {{kind, m}}, got {other}"))),
    };
    Ok(c.normalized())
}
