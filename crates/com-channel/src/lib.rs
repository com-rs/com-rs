//! # com-channel — propagation media
//!
//! The medium only: component non-idealities (lasers, modulators, photodiodes,
//! oscillators) are in `com-hardware`, so any impairment combines with any channel.
//!
//! | Module | Content |
//! |---|---|
//! | [`cdl`] | the clustered delay lines CDL-A to CDL-E of 3GPP TR 38.901 ([`cdl::Cdl`]): ray geometry, antenna arrays, polarisation, Doppler from the velocity |
//! | [`awgn`] | additive white Gaussian noise at an $E_{s}/N_{0}$ ([`awgn::Awgn`]) or of a given variance ([`awgn::Noise`]) |
//! | [`fading`] | flat fading: single-antenna Rayleigh ([`fading::Rayleigh`]) and MIMO Rayleigh or Rician, block or fast ([`fading::Mimo`]) |
//! | [`multipath`] | frequency-selective fading: tapped delay lines (EPA, EVA, ETU, ITU-R Pedestrian/Vehicular A, exponential) with Clarke/Jakes Doppler, SISO or MIMO; the delay–Doppler channel of discrete paths of OTFS |
//! | [`spatial`] | antenna correlation: the Kronecker model, exponential correlation |
//! | [`optical`] | fibre parameters, the linear fibre channel, the split-step Fourier method (scalar) and the Manakov-PMD equation (two polarisations), the EDFA, Raman amplification and ISRS, the GN model (feature `optical`) |
//!
//! # Examples
//! ```
//! use com_channel::optical::fiber::FiberParams;
//! let f = FiberParams::default();                   // SSMF, 80 km spans
//! assert!((f.beta2_ps2_km() + 21.75).abs() < 0.01);  // ps²/km at 193.1 THz, D = 17
//! ```

pub mod awgn;
pub mod cdl;
pub mod fading;
pub mod multipath;
#[cfg(feature = "optical")]
pub mod optical;
pub mod spatial;
