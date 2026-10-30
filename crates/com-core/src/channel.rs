//! The description of a channel realisation shared by the media that draw it and the
//! waveforms that take it.
//!
//! A doubly-dispersive channel of a few discrete paths, each with its complex gain
//! $h_{i}$, its delay $l_{i}$ (samples) and a single Doppler shift $\nu_{i}$ (Hz), acts on
//! a signal sampled at $f_{s}$ as
//!
//! $$
//! r\[n\] = \sum_{i=1}^{P} h_{i}\, e^{j 2\pi \nu_{i} (n - l_{i}) / f_{s}}\, s\[n - l_{i}\].
//! $$
//!
//! `com_channel::multipath::DelayDoppler` draws and applies it;
//! `com_waveform::otfs::Otfs::effective_channel` takes its paths. Both crates sit on the
//! same layer, so the type of a path lives here (it contains no algorithm).
//!
//! # Examples
//! ```
//! use com_core::channel::Path;
//! use com_core::C64;
//! let p = Path { gain: C64::new(1.0, 0.0), delay: 2, doppler_hz: 1e3 };
//! assert_eq!(p.delay, 2);
//! ```

use crate::C64;

/// One path of a delay–Doppler channel: a complex gain, a delay (samples) and a single
/// Doppler shift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Path {
    /// Complex gain $h_{i}$.
    pub gain: C64,
    /// Delay $l_{i}$ (samples).
    pub delay: usize,
    /// Doppler shift $\nu_{i}$ (Hz).
    pub doppler_hz: f64,
}
