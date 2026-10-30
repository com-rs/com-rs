//! The ports of a block: the [`Fact`]s of a frame, the check of a link, and the
//! check of a block while it runs.

use crate::blocks::{Block, BlockInfo, Frame};
use com_core::rng::Rng;
use com_core::{Error, Result};
use std::borrow::Borrow;

/// What the [`Frame`] of a trial holds at some point of a link.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Fact {
    /// Transmitted bits (the labels of the symbols, or the coded bits before `map`).
    Bits,
    /// Transmitted symbols.
    Symbols,
    /// The constellation of the transmitted symbols.
    Constellation,
    /// Samples.
    Samples,
    /// A noise variance: of the channel, declared by the receiver or per symbol.
    Noise,
    /// The fading of the trial (channel state information).
    Csi,
    /// The information bits of a coded link.
    Payload,
    /// Bit LLRs.
    Llr,
    /// Decoded information bits.
    Decoded,
    /// An OFDM frame between `ofdm_tx` and `ofdm_rx`.
    Ofdm,
    /// A WDM grid between `wdm_tx` and `wdm_rx`.
    Wdm,
    /// The reference sequence of a measured capture, not yet aligned.
    Capture,
}

impl Fact {
    /// Every fact.
    pub const ALL: [Fact; 12] = [
        Fact::Bits,
        Fact::Symbols,
        Fact::Constellation,
        Fact::Samples,
        Fact::Noise,
        Fact::Csi,
        Fact::Payload,
        Fact::Llr,
        Fact::Decoded,
        Fact::Ofdm,
        Fact::Wdm,
        Fact::Capture,
    ];

    /// The name of the fact (`noise`, `llr`…).
    pub fn name(self) -> &'static str {
        match self {
            Fact::Bits => "bits",
            Fact::Symbols => "symbols",
            Fact::Constellation => "constellation",
            Fact::Samples => "samples",
            Fact::Noise => "noise",
            Fact::Csi => "csi",
            Fact::Payload => "payload",
            Fact::Llr => "llr",
            Fact::Decoded => "decoded",
            Fact::Ofdm => "ofdm",
            Fact::Wdm => "wdm",
            Fact::Capture => "capture",
        }
    }

    /// Whether the frame holds the fact.
    pub fn holds(self, s: &Frame) -> bool {
        match self {
            Fact::Bits => !s.bits.is_empty(),
            Fact::Symbols => !s.symbols.is_empty(),
            Fact::Constellation => s.constellation.is_some(),
            Fact::Samples => !s.samples.is_empty(),
            Fact::Noise => {
                s.receiver.noise_var.is_some() || s.channel.noise_var.is_some() || !s.receiver.symbol_noise.is_empty()
            }
            Fact::Csi => s.channel.csi.is_some(),
            Fact::Payload => !s.coding.payload.is_empty(),
            Fact::Llr => !s.coding.llr.is_empty(),
            Fact::Decoded => !s.coding.decoded.is_empty(),
            Fact::Ofdm => s.in_flight.ofdm.is_some(),
            Fact::Wdm => s.in_flight.wdm.is_some(),
            Fact::Capture => !s.capture.reference.is_empty(),
        }
    }
}

/// Check the ports of a link: every block finds the facts it needs among those that
/// the blocks before it give.
///
/// # Errors
/// [`Error::Config`] naming the first block that misses a fact, and the blocks that give
/// it.
pub fn check_link<B: Borrow<BlockInfo>>(kinds: &[&str], registry: &[B]) -> Result<()> {
    let registry: Vec<&BlockInfo> = registry.iter().map(Borrow::borrow).collect();
    let mut have: Vec<Fact> = Vec::new();
    for (i, k) in kinds.iter().enumerate() {
        let Some(b) = registry.iter().find(|b| b.name == *k) else { continue };
        if let Some(f) = b.needs.iter().find(|f| !have.contains(f)) {
            let givers: Vec<&str> = registry.iter().filter(|g| g.gives.contains(f)).map(|g| g.name).collect();
            return Err(Error::Config(format!(
                "link block {} ({k}) needs {}, which no block before it gives: add one of {givers:?} before it",
                i + 1,
                f.name()
            )));
        }
        have.extend(b.gives.iter().copied());
    }
    Ok(())
}

/// A block that checks its declaration while it runs (debug builds): the facts it
/// gains must be declared in `gives` (what it needs is checked before the link runs,
/// by [`check_link`]).
#[cfg_attr(not(debug_assertions), allow(dead_code, reason = "built in debug builds only"))]
pub(crate) struct Checked {
    pub(crate) inner: Box<dyn Block>,
    pub(crate) kind: String,
    pub(crate) gives: &'static [Fact],
}

impl Block for Checked {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let before: Vec<Fact> = Fact::ALL.into_iter().filter(|f| f.holds(s)).collect();
        self.inner.apply(s, rng)?;
        if let Some(f) = Fact::ALL.into_iter().find(|f| f.holds(s) && !before.contains(f) && !self.gives.contains(f)) {
            panic!("block {:?} gave {}, which it does not declare in gives", self.kind, f.name());
        }
        Ok(())
    }

    fn needs_training(&self) -> bool {
        self.inner.needs_training()
    }

    fn train(&mut self, s: &Frame, rng: &mut Rng) -> Result<()> {
        self.inner.train(s, rng)
    }
}
