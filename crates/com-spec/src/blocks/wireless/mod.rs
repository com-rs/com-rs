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

use super::Extension;
use com_channel::multipath::Taps;
use com_math::linalg::CMat;

mod fading;
mod mimo;
mod ofdm;

use fading::{CdlBlock, CdlSpec, Rayleigh, RayleighSpec, TdlBlock, TdlSpec};
use mimo::{Detection, DetectorSpec, MimoRx, MimoTx, MimoTxSpec};
use ofdm::{OfdmRx, OfdmRxSpec, OfdmTx, OfdmTxSpec};

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

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            "mimo_tx",
            "Spread the symbols over n_tx antennas: spatial multiplexing (power 1/n_tx per antenna) or Alamouti's code.",
            MimoTxSpec,
            needs [Samples],
            gives [],
            |q, _, _| Ok(Box::new(MimoTx::new(&q)?)),
        ),
        builtin!(
            "rayleigh",
            "Flat fading y = H x on every antenna; records the channel for a genie-aided receiver.",
            RayleighSpec,
            needs[Samples],
            gives[Csi],
            |q, _, _| Ok(Box::new(Rayleigh { p: q })),
        ),
        builtin!(
            "tdl",
            "Tapped-delay-line multipath fading with Clarke/Jakes Doppler; records the taps for a genie-aided receiver.",
            TdlSpec,
            needs[Samples],
            gives[Csi],
            |q, _, _| Ok(Box::new(TdlBlock::new(&q)?)),
        ),
        builtin!(
            "cdl",
            "Clustered delay line of 3GPP TR 38.901 between uniform linear arrays (one tap per cluster, random ray phases and coupling); records the taps for a genie-aided receiver.",
            CdlSpec,
            needs[Samples],
            gives[Csi],
            |q, _, _| Ok(Box::new(CdlBlock::new(&q)?)),
        ),
        builtin!(
            "ofdm_tx",
            "Map every antenna's symbols on resource grids with pilots and OFDM-modulate them (power K/(LN) per sample).",
            OfdmTxSpec,
            needs[Samples],
            gives[Ofdm],
            |q, _, _| Ok(Box::new(OfdmTx::new(&q)?)),
        ),
        builtin!(
            "ofdm_rx",
            "Demodulate the OFDM grids, estimate the channel and detect every data resource element; output the symbols and their noise variances.",
            OfdmRxSpec,
            needs [Samples, Constellation, Ofdm],
            gives [Noise, Llr],
            |q, _, _| Ok(Box::new(OfdmRx::new(&q)?)),
        ),
        builtin!(
            "mimo_rx",
            "Detect a single-carrier link over flat fading with the known channel; output the symbols and their noise variances.",
            DetectorSpec,
            needs [Samples, Constellation, Csi],
            gives [Noise, Llr],
            |q, _, _| Ok(Box::new(MimoRx(Detection::new(&q)?))),
        ),
    ]
}

#[cfg(test)]
mod tests;
