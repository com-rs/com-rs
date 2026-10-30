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
//! # Conventions
//!
//! A modem is a configuration with public fields, built by a constructor (`new`, or
//! `Css::lora`), changed by struct update and checked by `check` when it is used, with
//! two exceptions: FBMC samples its prototype once in `new` and reads its fields through
//! getters, and CPM tabulates its phase response in `CpmConfig::build`. The modems that
//! map complex symbols to samples (OFDM, UFMC, OTFS, CE-OFDM, FBMC) implement
//! `Modulate` and `Demodulate` of `com-core`, at their `sample_rate` (1 Hz by default,
//! time in samples), and refuse an input at another rate; CSS and CPM carry symbol
//! indices, which `Symbols` cannot (`docs/adr/0012-index-symbol-waveforms.md`). The
//! vocabulary is shared: `n_fft` the size of the transform (the
//! subcarriers of the grid), `n_active` the modulated subcarriers, `n_cp` the cyclic
//! prefix, `block_len()` the samples of a block, prefix included. `oversampling` is the
//! number of samples per Nyquist-rate sample (OFDM, its prefix counted at the Nyquist
//! rate) or per chip (CSS); CE-OFDM folds it into its DFT size and counts its prefix in
//! samples of that DFT. The channels the waveforms cross are in `com-channel`.
//!
//! Single-carrier waveforms are planned (see `docs/ARCHITECTURE.md`).

pub mod ce_ofdm;
pub mod cpm;
pub mod css;
pub mod fbmc;
pub mod ofdm;
pub mod otfs;
pub mod papr;
pub mod ufmc;
