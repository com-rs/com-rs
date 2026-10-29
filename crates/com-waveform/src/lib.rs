//! # com-waveform — time–frequency structures
//!
//! Waveforms carry mapped symbols, so this crate uses `com-mapping` (the sibling
//! dependency allowed by the architecture guide) and, for MIMO-OFDM, the detectors of
//! `com-mimo` (ADR 0009).
//!
//! | Module | Content |
//! |---|---|
//! | [`ofdm`] | OFDM and DFT-spread OFDM: oversampled modulator, cyclic prefix, demodulator; resource grid and pilots, LS/LMMSE channel estimation, Schmidl & Cox synchronisation, MIMO-OFDM detection |
//! | [`otfs`] | orthogonal time frequency space: ISFFT and Heisenberg transform with rectangular pulses and one prefix per frame, the sparse effective delay–Doppler channel (Raviteja et al. 2019) |
//! | [`papr`] | peak-to-average power ratio: theory of its distribution, clipping (Bussgang), and the reduction techniques (clipping and filtering, SLM, PTS, tone reservation, ACE, companding) |
//! | [`ce_ofdm`] | constant-envelope OFDM (Thompson et al. 2008): modulator, phase demodulator (principal or unwrapped phase), and the error rates of its linear regime |
//! | [`css`] | chirp spread spectrum (the frequency-shift chirp modulation of LoRa): modulator at any oversampling, noncoherent DFT receiver, exact error rates, cross-correlation, spectrum in Fresnel functions, same-SF interference (Chiani & Elzanaty 2019, Afisiadis et al. 2020) |
//! | [`fbmc`] | filter-bank multicarrier with offset QAM (FBMC/OQAM): PHYDYAS prototype, polyphase modulator and demodulator, OQAM staggering, intrinsic interference |
//! | [`ufmc`] | universal filtered multicarrier: OFDM filtered per sub-band by Dolph–Chebyshev filters, no prefix, receiver on the even bins of a DFT of twice the size (Schaich & Wild 2014) |
//! | [`cpm`] | continuous-phase modulation (CPFSK, MSK, GMSK, raised-cosine pulses): modulator with an exact phase state, coherent maximum-likelihood sequence detector, quadrature detector of MSK, differential detector, the MSK spectrum and the Gaussian pulse in closed form, minimum distance |
//!
//! Single-carrier waveforms and the OFDM resource grid (pilots, channel estimation)
//! are planned (see `docs/ARCHITECTURE.md`).

pub mod ce_ofdm;
pub mod cpm;
pub mod css;
pub mod fbmc;
pub mod ofdm;
pub mod otfs;
pub mod papr;
pub mod ufmc;
