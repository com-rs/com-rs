//! The error type of com-rs.
//!
//! Every fallible function returns [`Result<T>`], whose error says what kind of mistake
//! was made: an invalid configuration (a parameter out of its domain), inconsistent
//! sizes, a numerical failure (a non-finite value, a singular system), a missing value
//! in a simulation condition, or an input/output error.
//!
//! The crates of layers 0 to 4 share this one enum, re-exported as `comrs::Error`, so
//! that `?` composes across blocks of different crates inside a simulation trial (see
//! `docs/adr/0003-shared-error-type.md`).
//!
//! # Examples
//! ```
//! use com_core::{ensure, Error};
//! let err = ensure(false, || "order must be a power of two, got 6".into()).unwrap_err();
//! assert!(matches!(err, Error::Config(_)));
//! assert_eq!(err.context("qam").to_string(), "invalid configuration: qam: order must be a power of two, got 6");
//! ```

use std::fmt;

/// The error of every fallible operation of com-rs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A parameter is outside its domain (the message names it and its value).
    #[error("invalid configuration: {0}")]
    Config(String),
    /// Two inputs have inconsistent lengths or shapes.
    #[error("inconsistent shapes: {0}")]
    Shape(String),
    /// A numerical routine failed (non-finite value, singular system, no convergence).
    #[error("numerical failure: {0}")]
    Numerical(String),
    /// A simulation condition does not hold the requested key, or holds another type.
    #[error("condition: {0}")]
    Condition(String),
    /// An input/output error (a logger, a recording).
    #[error("input/output: {0}")]
    Io(#[from] std::io::Error),
}

/// `Result` with the error type of com-rs.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// The same error with `context` prepended to its message (the kind is kept, so
    /// that a configuration error found while building a condition stays one).
    #[must_use]
    pub fn context(self, context: impl fmt::Display) -> Error {
        match self {
            Error::Config(m) => Error::Config(format!("{context}: {m}")),
            Error::Shape(m) => Error::Shape(format!("{context}: {m}")),
            Error::Numerical(m) => Error::Numerical(format!("{context}: {m}")),
            Error::Condition(m) => Error::Condition(format!("{context}: {m}")),
            Error::Io(e) => Error::Io(std::io::Error::new(e.kind(), format!("{context}: {e}"))),
        }
    }
}

/// Return [`Error::Config`] with the message of `msg` unless `ok`.
///
/// # Errors
/// [`Error::Config`] when `ok` is false.
pub fn ensure(ok: bool, msg: impl FnOnce() -> String) -> Result<()> {
    if ok { Ok(()) } else { Err(Error::Config(msg())) }
}

/// Return [`Error::Shape`] with the message of `msg` unless `ok` (inconsistent lengths
/// or shapes; [`ensure`] for a parameter outside its domain).
///
/// # Errors
/// [`Error::Shape`] when `ok` is false.
///
/// # Examples
/// ```
/// use com_core::{ensure_shape, Error};
/// let err = ensure_shape(false, || "3 rows for a code of 2 outputs".into()).unwrap_err();
/// assert!(matches!(err, Error::Shape(_)));
/// assert_eq!(err.to_string(), "inconsistent shapes: 3 rows for a code of 2 outputs");
/// ```
pub fn ensure_shape(ok: bool, msg: impl FnOnce() -> String) -> Result<()> {
    if ok { Ok(()) } else { Err(Error::Shape(msg())) }
}

/// Return [`Error::Shape`] unless a block of `got` elements has the `want` expected
/// (`what` names the elements).
///
/// # Errors
/// [`Error::Shape`] when the lengths differ.
pub fn expect_len(got: usize, want: usize, what: &str) -> Result<()> {
    if got == want { Ok(()) } else { Err(Error::Shape(format!("{got} {what} for a block of {want}"))) }
}

/// Return [`Error::Shape`] unless `a == b` (`what` names the two inputs).
///
/// # Errors
/// [`Error::Shape`] when the lengths differ.
pub fn same_len(a: usize, b: usize, what: &str) -> Result<()> {
    if a == b { Ok(()) } else { Err(Error::Shape(format!("{what}: lengths {a} and {b} differ"))) }
}
