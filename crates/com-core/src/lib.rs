//! # com-core — the vocabulary every com-rs crate speaks
//!
//! Typed containers, verb traits, unit newtypes, the random generator and the error
//! type. It contains no algorithm: the blocks that implement these traits live in the
//! crates above it (see `docs/ARCHITECTURE.md`).
//!
//! | Module | Content |
//! |---|---|
//! | [`channel`] | [`Path`], one path of a delay–Doppler channel realisation |
//! | [`signal`] | [`Bits`], [`Llr`], [`Symbols`], [`Labels`], [`Signal`] (streams × elements), [`DualPol`] |
//! | [`traits`] | [`Encode`], [`Decode`], [`Map`], [`Demap`] (with [`Demapping`] and [`Jacobian`]), [`Modulate`], [`Demodulate`], [`Precode`], [`Detect`], [`Propagate`], [`Impair`] |
//! | [`units`] | [`Db`], [`EbN0`], [`EsN0`], [`Hertz`], [`Seconds`] |
//! | [`rng`] | [`Rng`] (xoshiro256\*\*), [`SimRng`], [`stream`], [`batch_rng`] |
//! | [`error`] | [`Error`], [`Result`], [`ensure`], [`ensure_shape`], [`expect_len`], [`same_len`] |
//!
//! # Conventions
//!
//! - Complex baseband samples are [`C64`] (`num_complex::Complex<f64>`).
//! - Physical quantities are in SI units unless the name says otherwise (`_db`, `_dbm`,
//!   `_km`, `_ps_nm_km`, …).
//! - There is no global or thread-local generator anywhere: every random block takes a
//!   `&mut SimRng`, so every simulation is reproducible from its seed.
//!
//! [`Path`]: channel::Path
//! [`Bits`]: signal::Bits
//! [`Llr`]: signal::Llr
//! [`Symbols`]: signal::Symbols
//! [`Labels`]: signal::Labels
//! [`Signal`]: signal::Signal
//! [`DualPol`]: signal::DualPol
//! [`Encode`]: traits::Encode
//! [`Decode`]: traits::Decode
//! [`Map`]: traits::Map
//! [`Demap`]: traits::Demap
//! [`Demapping`]: traits::Demapping
//! [`Jacobian`]: traits::Jacobian
//! [`Modulate`]: traits::Modulate
//! [`Demodulate`]: traits::Demodulate
//! [`Precode`]: traits::Precode
//! [`Detect`]: traits::Detect
//! [`Propagate`]: traits::Propagate
//! [`Impair`]: traits::Impair
//! [`Db`]: units::Db
//! [`EbN0`]: units::EbN0
//! [`EsN0`]: units::EsN0
//! [`Hertz`]: units::Hertz
//! [`Seconds`]: units::Seconds
//! [`Rng`]: rng::Rng
//! [`SimRng`]: rng::SimRng
//! [`stream`]: rng::stream
//! [`batch_rng`]: rng::batch_rng

pub mod channel;
pub mod error;
pub mod rng;
pub mod signal;
pub mod traits;
pub mod units;

pub use error::{Error, Result, ensure, ensure_shape, expect_len, same_len};

/// Complex number with `f64` components (the type of every complex sample).
pub type C64 = num_complex::Complex<f64>;

/// The items used by most blocks.
pub mod prelude {
    pub use crate::C64;
    pub use crate::error::{Error, Result};
    pub use crate::rng::{Rng, SimRng, batch_rng, stream};
    pub use crate::signal::{Bits, DualPol, Labels, Llr, Signal, Symbols};
    pub use crate::traits::{
        Decode, Demap, Demapping, Demodulate, Detect, Encode, Impair, Jacobian, Map, Modulate, Precode, Propagate,
    };
    pub use crate::units::{Db, EbN0, EsN0, Hertz, Seconds};
}
