//! # com-mimo — multiple-antenna processing
//!
//! Detection, space-time coding, precoding and capacity on the generic flat model
//! $y = H x + w$, with $H$ the $n_{r} \times n_{t}$ channel matrix. The channel models
//! themselves (Rayleigh and Rician fading, antenna correlation) are in `com-channel`
//! (`fading::Mimo`, `spatial`); waveform-specific resource mapping (per-subcarrier
//! detection of MIMO-OFDM) belongs to `com-waveform`.
//!
//! | Module | Content |
//! |---|---|
//! | [`detection`] | ZF, MMSE, ordered SIC (V-BLAST), exhaustive ML, sphere decoder, K-best; hard and soft outputs, the estimates of the symbols ([`detection::Estimates`]) |
//! | [`message_passing`] | LAMA (complex Bayesian AMP) and OAMP for large systems, their state evolution and the recovery thresholds of LAMA |
//! | [`receiver`] | the receiver of a link: a detector of [`detection`] by name, applied channel use after channel use (spatial layers or Alamouti blocks), with soft outputs |
//! | [`stbc`] | orthogonal space-time block codes: Alamouti, the rate-3/4 codes for 3 and 4 antennas; linear decoding |
//! | [`precoding`] | SVD (eigenmode) transmission with water-filling, ZF and regularised ZF precoders |
//! | [`capacity`] | capacity of a channel realisation with equal power or water-filling; the ergodic and outage capacities of i.i.d. Rayleigh fading, the eigenvalue density of a Wishart matrix |
//!
//! `com-mimo` uses the constellations of `com-mapping` (ADR 0008).

pub mod capacity;
pub mod detection;
pub mod message_passing;
pub mod precoding;
pub mod receiver;
pub mod stbc;
