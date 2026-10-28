//! Synchronisation: carrier frequency and phase, symbol timing, frame.
//!
//! - [`carrier`]: frequency-offset estimation by the M-th power periodogram, phase
//!   recovery by Viterbi–Viterbi, blind phase search (BPS) and a decision-directed PLL;
//! - [`timing`]: Gardner timing-error detector, cubic interpolator and the Gardner
//!   clock-recovery loop;
//! - [`frame`]: frame synchronisation (delay and rotation) on known symbols.

pub mod carrier;
pub mod frame;
pub mod timing;
