//! # com-hardware — transmitter and receiver non-idealities
//!
//! Component models only: propagation effects are in `com-channel`, so any impairment
//! combines with any channel.
//!
//! | Module | Content |
//! |---|---|
//! | [`amplifier`] | power amplifiers: Rapp (SSPA), Saleh (TWTA), memory polynomial ([`Impair`](com_core::traits::Impair) blocks) |
//! | [`gaussian_input`] | an envelope nonlinearity driven by a complex Gaussian (OFDM) signal: the series of its output autocorrelation and the spectrum of its distortion (Banelli & Cacopardi 2000) |
//! | [`oscillator`] | Wiener phase noise and carrier frequency offset ([`Impair`](com_core::traits::Impair) blocks) |
//! | [`converter`] | DAC and ADC: uniform quantisation with clipping, effective number of bits, clock jitter, analogue bandwidth, zero-order hold, and their closed forms ([`Impair`](com_core::traits::Impair) blocks) |
//! | [`iq`] | transmitter and receiver IQ imbalance ([`Impair`](com_core::traits::Impair) blocks) |
//! | [`optical`] | lasers, phase and Mach–Zehnder modulators, the IQ modulator, photodiodes with shot and thermal noise, the 90° hybrid and the coherent receiver (feature `optical`) |

pub mod amplifier;
pub mod converter;
pub mod gaussian_input;
pub mod iq;
#[cfg(feature = "optical")]
pub mod optical;
pub mod oscillator;
