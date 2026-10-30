//! The state of one trial along a link: [`Frame`] and the contexts it carries.

use super::{Csi, OfdmFrame, WdmFrame};
use com_core::{Error, Result, ensure};
use com_mapping::constellation::{Constellation, indices_to_bits};
use com_math::C64;

/// The state carried along a link during one trial: what was transmitted, the
/// current samples, and the contexts that some blocks set for the following ones.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// The constellation of the source (unit mean energy for uniform symbols).
    pub constellation: Option<Constellation>,
    /// The probabilities of the points of the constellation (probabilistic shaping);
    /// uniform if `None`. The rate metrics, the soft demapper and `awgn_ebn0` use them.
    pub pmf: Option<Vec<f64>>,
    /// Transmitted bits (Gray labels of the constellation, MSB first).
    pub bits: Vec<u8>,
    /// Transmitted symbol indices.
    pub indices: Vec<usize>,
    /// Transmitted symbols (of polarisation x, then y, with two polarisations).
    pub symbols: Vec<C64>,
    /// Symbols excluded from the metrics (pilots, preambles), one flag per symbol.
    pub excluded: Vec<bool>,
    /// Current samples of antenna (or polarisation) 0.
    pub samples: Vec<C64>,
    /// The samples of the other antennas or polarisations (1, 2, …); empty for one.
    pub antennas: Vec<Vec<C64>>,
    /// Current samples per symbol.
    pub sps: f64,
    /// Symbol rate (Bd; 1 for a normalised time axis).
    pub symbol_rate: f64,
    /// What the streams are: antennas, layers, polarisations.
    pub layout: Layout,
    /// The channel of the trial, as a genie-aided receiver knows it.
    pub channel: ChannelState,
    /// What the receiver declares about its outputs.
    pub receiver: ReceiverState,
    /// The information bits, the LLRs and the decisions of a coded link.
    pub coding: Coding,
    /// The frames between a transmitter block and its receiver (OFDM, WDM).
    pub in_flight: InFlight,
    /// A measured capture.
    pub capture: CaptureState,
    /// The index of the trial of the condition (set by the runner; 0 outside a
    /// campaign), for the blocks that save or label a trial.
    pub trial: u64,
    /// Diagnostic values recorded by the blocks, reported as means.
    pub values: Vec<(String, f64)>,
    /// The metrics measured by the `probe` blocks, merged into the results of the trial.
    pub probes: com_sim::scenario::Record,
}

/// What the streams of a [`Frame`] are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    /// The streams are the polarisations of one optical channel, each with its own
    /// transmitted symbols: the blocks written for one polarisation process each of
    /// them.
    pub polarisations: bool,
    /// The layers of a multi-antenna transmitter (`mimo_tx`); 0 or 1 for one antenna.
    pub layers: usize,
    /// The transmitter uses Alamouti's code (`mimo_tx` with `scheme: alamouti`).
    pub alamouti: bool,
}

/// The channel of a trial, as a genie-aided receiver knows it.
#[derive(Clone, Debug, Default)]
pub struct ChannelState {
    /// The fading of the trial.
    pub csi: Option<Csi>,
    /// Variance of the complex noise added per sample by the noise blocks.
    pub noise_var: Option<f64>,
    /// Mean power of the signal that entered the fading channel: the reference power of
    /// the noise blocks after a fading block, so that the SNR is an average over the
    /// fading and not the SNR of every realisation.
    pub faded_power: Option<f64>,
}

impl ChannelState {
    /// The noise variance per sample for the detectors, floored at `1e-12` (and `1e-12`
    /// when no noise block ran).
    pub fn detector_noise_var(&self) -> f64 {
        self.noise_var.unwrap_or(1e-12).max(1e-12)
    }
}

/// What the receiver declares about its outputs.
#[derive(Clone, Debug, Default)]
pub struct ReceiverState {
    /// The noise variance the receiver believes in, on the scale of the unit-energy
    /// constellation: with the samples, the receiver's posterior
    /// $q(x \mid z) \propto e^{-\lvert z - x \rvert^{2} / \hat{\sigma}^{2}}$ (declared by `noise_estimate`, or by a learned
    /// receiver).
    pub noise_var: Option<f64>,
    /// Noise variance of every received symbol after equalisation or detection (empty:
    /// unknown).
    pub symbol_noise: Vec<f64>,
}

/// The coded blocks of a trial.
#[derive(Clone, Debug, Default)]
pub struct Coding {
    /// Information bits of the coded blocks, block after block.
    pub payload: Vec<u8>,
    /// Coded blocks per trial.
    pub n_blocks: usize,
    /// Code rate (information bits per coded bit); `None` for an uncoded link.
    pub code_rate: Option<f64>,
    /// Bit LLRs of the demapper or the detector (positive: bit 0 more likely).
    pub llr: Vec<f64>,
    /// Information bits decided by the decoder.
    pub decoded: Vec<u8>,
}

/// The frames between a transmitter block and its receiver.
#[derive(Clone, Debug, Default)]
pub struct InFlight {
    /// What `ofdm_rx` needs to undo `ofdm_tx`.
    pub ofdm: Option<OfdmFrame>,
    /// The WDM grid between `wdm_tx` and `wdm_rx`.
    pub wdm: Option<WdmFrame>,
}

/// A measured capture (`capture`, `reference_sync`).
#[derive(Clone, Debug, Default)]
pub struct CaptureState {
    /// One period of the transmitted indices (empty for a simulated source).
    pub reference: Vec<usize>,
    /// The transmitted symbols are not yet aligned on the samples (before
    /// `reference_sync`).
    pub unaligned: bool,
}

impl Frame {
    /// Sampling rate `symbol_rate · sps` (Hz).
    pub fn fs(&self) -> f64 {
        self.symbol_rate * self.sps
    }

    /// Number of transmitted symbols.
    pub fn n_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// The constellation, or an error naming the block that needs it.
    ///
    /// # Errors
    /// [`Error::Config`] if no source has set the constellation before `block`.
    pub fn constellation(&self, block: &str) -> Result<&Constellation> {
        self.constellation.as_ref().ok_or_else(|| {
            Error::Config(format!(
                "block {block:?} needs a constellation (it comes before the source, or the source is gaussian)"
            ))
        })
    }

    /// Set the transmitted symbols from indices of the constellation.
    pub(super) fn set_indices(&mut self, idx: Vec<usize>) {
        if let Some(c) = &self.constellation {
            self.symbols = c.symbols(&idx);
            self.bits = indices_to_bits(&idx, c.bits_per_symbol());
        }
        self.excluded = vec![false; idx.len()];
        self.indices = idx;
    }

    /// The integer samples per symbol, or an error naming the block.
    ///
    /// # Errors
    /// [`Error::Config`] unless the signal is aligned at an integer number of samples per symbol.
    pub fn int_sps(&self, block: &str) -> Result<usize> {
        let k = self.sps.round();
        ensure(k >= 1.0 && (self.sps - k).abs() < 1e-9, || {
            format!("block {block:?} needs an integer number of samples per symbol, the signal has {}", self.sps)
        })?;
        Ok(k as usize)
    }

    /// Check that the signal is at one sample per symbol: the symbols of a source, or
    /// the output of a decimation.
    ///
    /// # Errors
    /// [`Error::Config`] naming `block` unless the signal is at one sample per symbol.
    pub fn require_one_sps(&self, block: &str) -> Result<()> {
        ensure(self.sps == 1.0, || {
            format!("block {block:?} needs the signal at 1 sample per symbol, it has {} samples per symbol", self.sps)
        })
    }

    /// Check that the samples are at one sample per symbol and aligned on the symbols.
    ///
    /// # Errors
    /// [`Error::Config`] unless the signal is aligned at one sample per symbol.
    pub fn require_symbols(&self, block: &str) -> Result<()> {
        ensure(!self.capture.unaligned, || {
            format!(
                "block {block:?} needs the transmitted symbols aligned on the capture: add reference_sync before it"
            )
        })?;
        ensure(self.sps == 1.0 && self.samples.len() == self.symbols.len(), || {
            format!(
                "block {block:?} needs the signal at 1 sample per symbol aligned on the {} transmitted symbols \
                 (it has {} samples at {} samples per symbol): place it after decimation and frame synchronisation",
                self.symbols.len(),
                self.samples.len(),
                self.sps
            )
        })
    }

    /// Record a diagnostic value.
    pub fn record(&mut self, label: &str, name: &str, value: f64) {
        self.values.push((format!("{label}.{name}"), value));
    }

    /// Number of antenna streams (1 for a single antenna).
    pub fn n_streams(&self) -> usize {
        1 + self.antennas.len()
    }

    /// The samples of every antenna.
    pub fn streams(&self) -> impl Iterator<Item = &Vec<C64>> {
        std::iter::once(&self.samples).chain(&self.antennas)
    }

    /// The samples of every antenna, mutable.
    pub fn streams_mut(&mut self) -> impl Iterator<Item = &mut Vec<C64>> {
        std::iter::once(&mut self.samples).chain(&mut self.antennas)
    }

    /// Take the samples of every antenna, leaving the signal without samples.
    pub fn take_streams(&mut self) -> Vec<Vec<C64>> {
        let mut out = vec![std::mem::take(&mut self.samples)];
        out.append(&mut self.antennas);
        out
    }

    /// Set the samples of every antenna (at least one stream).
    pub fn set_streams(&mut self, mut streams: Vec<Vec<C64>>) {
        if streams.is_empty() {
            streams.push(Vec::new());
        }
        self.samples = streams.remove(0);
        self.antennas = streams;
    }

    /// The reference power of a noise block: the power entering the fading channel if
    /// there is one, else the mean power of the samples.
    pub fn reference_power(&self) -> f64 {
        self.channel.faded_power.unwrap_or_else(|| self.mean_power())
    }

    /// Mean power of the samples over every antenna: their energy
    /// ([`com_math::energy`]) over their number.
    pub fn mean_power(&self) -> f64 {
        let (e, n) = self.streams().fold((0.0, 0usize), |(e, n), x| (e + com_math::energy(x), n + x.len()));
        if n == 0 { 0.0 } else { e / n as f64 }
    }

    /// The signal with the polarisations of a dual-polarisation channel concatenated
    /// into one stream, aligned on `symbols` (the signal itself otherwise).
    pub fn pooled(&self) -> std::borrow::Cow<'_, Frame> {
        if self.layout.polarisations && !self.antennas.is_empty() {
            let mut s = self.clone();
            let mut all = std::mem::take(&mut s.samples);
            for a in std::mem::take(&mut s.antennas) {
                all.extend(a);
            }
            s.samples = all;
            std::borrow::Cow::Owned(s)
        } else {
            std::borrow::Cow::Borrowed(self)
        }
    }

    /// The noise variance of every symbol the receiver sees: `symbol_noise`, else the
    /// declared posterior `noise_var`, else the channel's `noise_var`.
    ///
    /// # Errors
    /// [`Error::Config`] if no block has set the noise variance.
    pub fn symbol_noise(&self, block: &str) -> Result<Vec<f64>> {
        let known = &self.receiver.symbol_noise;
        if known.len() == self.samples.len() && !known.is_empty() {
            return Ok(known.clone());
        }
        let v = self.receiver.noise_var.or(self.channel.noise_var).ok_or_else(|| {
            Error::Config(format!(
                "block {block:?} needs the noise variance: add a noise block (awgn, awgn_ebn0) or noise_estimate before it"
            ))
        })?;
        Ok(vec![v; self.samples.len()])
    }
}
