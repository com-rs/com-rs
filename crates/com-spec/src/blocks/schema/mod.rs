//! The single description of a block: its parameters and its ports.
//!
//! The parameters of a block are declared once, with
//! [`block_params!`](crate::block_params): the name and the Rust type of every
//! parameter, its default, its unit, its accepted values, its bounds and its meaning.
//! The macro writes the structure, its parser ([`Params`]) and its documentation
//! ([`ParamInfo`]), so that what a block accepts, what `comrs blocks` prints, what the
//! Python classes of `comrs.blocks` take and the JSON Schema of the links
//! ([`json_schema`]) cannot differ. A block of another crate declares its parameters the
//! same way (see [`register`](super::register)).
//!
//! ```
//! use com_spec::blocks::Params;
//! com_spec::block_params! {
//!     struct AwgnSpec {
//!         snr_db: f64 [required] "dB", "signal-to-noise ratio";
//!         reference: String ["signal"] one_of ["signal", "unit"], "power reference of the SNR";
//!         bandwidth_hz: Option<f64> [absent "symbol rate"] "Hz" > 0, "noise bandwidth";
//!     }
//! }
//! let p = AwgnSpec::parse("awgn", serde_json::json!({"snr_db": 10})).unwrap();
//! assert_eq!((p.snr_db, p.reference.as_str(), p.bandwidth_hz), (10.0, "signal", None));
//! assert_eq!(AwgnSpec::params()[2].default_text(), "symbol rate");
//! assert!(AwgnSpec::parse("awgn", serde_json::json!({"snr_db": 10, "bandwidth_hz": 0})).is_err());
//! ```
//!
//! A default is `required`, `absent` (an `Option` left to `None`, with the text that says
//! what the block then does) or a JSON value, read as the value of a specification
//! would be (`VALUE => "TEXT"` shows it as `TEXT`). Accepted values (`one_of`) are
//! compared without case; bounds (`>`, `>=`, `<=`, `<`) apply to a number and to every
//! number of a list.
//!
//! A parameter whose value is a mapping of parameters of its own (the fibre of the
//! optical blocks, the pilots of `ofdm_tx`) is declared with `block_params!` too, and
//! typed [`Mapping`] (or [`NameOr`], a name or such a mapping): its keys reach the
//! documentation, the JSON Schema and the Python classes, and are checked as those of
//! a block when the block parses it.
//!
//! # Ports
//!
//! A block also declares what it reads from the [`Frame`](super::Frame) of a trial and what it may
//! add to it, as [`Fact`]s: `awgn` needs `samples` and gives `noise`, `demap` needs
//! `samples`, `symbols`, `constellation` and `noise` and gives `llr`. [`check_link`]
//! checks a link before it runs, so that a missing block is named at once (`demap
//! needs noise: add awgn, awgn_ebn0, noise_estimate… before it`), and a graphical editor
//! can tell which blocks may follow which. In debug builds every block also checks
//! what it gives while it runs: a fact that it adds to the frame must be declared in its
//! `gives` (what it needs is checked once, before the link runs, by [`check_link`]).

#[macro_use]
mod macros;
mod fact;
mod json_schema;
mod param;

pub(crate) use fact::Checked;
pub use fact::{Fact, check_link};
pub use json_schema::{json_schema, object_schema, variants_schema};
pub(crate) use param::NumberOrList;
pub use param::{Mapping, NameOr, Named, ParamDefault, ParamInfo, ParamType, Params, choice, mapping, shown, take};
