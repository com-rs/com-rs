//! Compensation of deterministic channel effects at the receiver.
//!
//! - [`cd`]: chromatic-dispersion compensation, an all-pass frequency-domain filter
//!   (exact on a block, or as an FIR);
//! - [`dbp`]: digital back-propagation, the split-step solver of `com_math::nlse` run
//!   with the coefficients of the link negated;
//! - [`dpd`]: digital predistortion of a power amplifier, memory polynomials trained
//!   by indirect learning, and the static predistorter of a memoryless model.

pub mod cd;
pub mod dbp;
pub mod dpd;
