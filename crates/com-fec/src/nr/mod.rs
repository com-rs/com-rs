//! The channel codes of 5G NR (3GPP TS 38.212): the LDPC code of the data channels
//! ([`NrLdpc`]) and the polar code of the control and broadcast channels
//! ([`NrPolar`]), with their rate matching, from the tables of the standard
//! ([`tables`]).

pub mod ldpc;
pub mod polar;
pub mod tables;

pub use ldpc::NrLdpc;
pub use polar::NrPolar;
