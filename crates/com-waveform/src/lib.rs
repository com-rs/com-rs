//! # com-waveform — time–frequency structures
//!
//! Waveforms carry mapped symbols, so this crate uses `com-mapping` (the sibling
//! dependency allowed by the architecture guide) and, for MIMO-OFDM, the detectors of
//! `com-mimo` (ADR 0009).
//!
//! | Module | Content |
//! |---|---|
//! | [`ofdm`] | OFDM and DFT-spread OFDM: oversampled modulator, cyclic prefix, demodulator; resource grid and pilots, LS/LMMSE channel estimation, Schmidl & Cox synchronisation, MIMO-OFDM detection |
//! | [`papr`] | peak-to-average power ratio: theory of its distribution, clipping (Bussgang), and the reduction techniques (clipping and filtering, SLM, PTS, tone reservation, ACE, companding) |
//! | [`ce_ofdm`] | constant-envelope OFDM (Thompson et al. 2008): modulator, phase demodulator (principal or unwrapped phase), and the error rates of its linear regime |
//! | [`cpm`] | continuous-phase modulation (CPFSK, MSK, GMSK, raised-cosine pulses): modulator with an exact phase state, coherent maximum-likelihood sequence detector, quadrature detector of MSK, differential detector, the MSK spectrum and the Gaussian pulse in closed form, minimum distance |
//!
//! Single-carrier waveforms and the OFDM resource grid (pilots, channel estimation)
//! are planned (see `docs/ARCHITECTURE.md`).

pub mod ce_ofdm;
pub mod cpm;
pub mod ofdm;
pub mod papr;
