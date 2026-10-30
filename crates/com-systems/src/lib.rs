//! # com-systems — reference chains of standards
//!
//! Standard-compliant chains assembled from the blocks of layers 0 to 2. New algorithms
//! do not belong here: they go to the functional crates.
//!
//! | Module | Standard |
//! |---|---|
//! | [`dvbs2`] | DVB-S2 (ETSI EN 302 307-1): BBFRAME, BCH and LDPC ([`dvbs2::fec`]), bit interleaving, QPSK to 32APSK, PL framing with pilots and Gold scrambling |
//! | [`pas`] | probabilistic amplitude shaping (Böcherer, Steiner & Schulte 2015): a constant-composition matcher, Gray PAM and a systematic LDPC code whose parity bits are the signs |
//! | [`wdm`] | a WDM transmitter (OptiCommPy's `simpleWDMTx`): pulse shaping, lasers with phase noise and IQ modulators on a grid of channels |
//!
//! A chain of another standard that assembles several crates of layer 2 has its place
//! here (see the placement rules of `docs/ARCHITECTURE.md`); the 5G NR numerology, grid
//! and codes live in `com-waveform` and `com-fec`, which own them.

pub mod dvbs2;
pub mod pas;
pub mod wdm;
