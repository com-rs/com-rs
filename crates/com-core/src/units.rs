//! Unit newtypes for the quantities that are routinely confused.
//!
//! A decibel value is a [`Db`], never a bare `f64`; an `E_{b}/N_{0}` is an [`EbN0`] and
//! an `E_{s}/N_{0}` an [`EsN0`], so that passing one where the other is expected is a
//! compile error. Conversions are explicit methods, never `From` impls that hide a
//! formula.
//!
//! The decibel conversions of plain numbers ([`db2lin`], [`lin2db`], [`db2amp`],
//! [`amp2db`], [`dbm2w`], [`w2dbm`], [`db2np`], [`np2db`]) are defined here, once: the
//! methods of the newtypes call them, and `com_math` re-exports them (`com_math::db2lin`
//! is this function), so that a value in dB converts to the same bits whichever way it
//! is written.
//!
//! # Model
//!
//! With `k` coded bits per symbol and a code rate `R`,
//!
//! $$
//! \left.\frac{E_{s}}{N_{0}}\right|_{\mathrm{dB}} = \left.\frac{E_{b}}{N_{0}}\right|_{\mathrm{dB}} + 10 \log_{10}(k R)
//! $$
//!
//! # Examples
//! ```
//! use com_core::units::{Db, EbN0};
//! let es = EbN0(Db(4.0)).to_es_n0(2.0, 1.0);       // QPSK, uncoded
//! assert!((es.0.0 - 7.0103).abs() < 1e-4);
//! assert!((Db(3.0).to_linear() - 1.9953).abs() < 1e-4);
//! assert!((Db::from_linear(10.0).0 - 10.0).abs() < 1e-12);
//! // the newtype and the function are one conversion
//! assert_eq!(Db(3.0).to_linear(), com_core::units::db2lin(3.0));
//! // 2.6 information bits per symbol (a shaped or coded constellation)
//! let es = EbN0(Db(4.0)).to_es_n0(2.6, 1.0);
//! assert!((es.0.0 - (4.0 + com_core::units::lin2db(2.6))).abs() < 1e-12);
//! ```

/// Power ratio in dB to linear: `10^{x/10}`.
#[inline]
pub fn db2lin(x_db: f64) -> f64 {
    10f64.powf(x_db / 10.0)
}

/// Linear power ratio to dB: `10 log10(x)`.
#[inline]
pub fn lin2db(x: f64) -> f64 {
    10.0 * x.log10()
}

/// Amplitude ratio in dB to linear: `10^{x/20}`.
#[inline]
pub fn db2amp(x_db: f64) -> f64 {
    10f64.powf(x_db / 20.0)
}

/// Linear amplitude ratio to dB: `20 log10(x)`.
#[inline]
pub fn amp2db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// Attenuation in dB per unit length to nepers (power): `α_dB / (10 log10 e)`.
#[inline]
pub fn db2np(x_db: f64) -> f64 {
    x_db / (10.0 * std::f64::consts::LOG10_E)
}

/// Attenuation in nepers (power) per unit length to dB: `10 log10(e) · α`.
#[inline]
pub fn np2db(x_np: f64) -> f64 {
    10.0 * std::f64::consts::LOG10_E * x_np
}

/// Power in dBm to watts: `10^{P/10} · 1 mW`.
#[inline]
pub fn dbm2w(p_dbm: f64) -> f64 {
    1e-3 * db2lin(p_dbm)
}

/// Power in watts to dBm.
#[inline]
pub fn w2dbm(p_w: f64) -> f64 {
    lin2db(p_w / 1e-3)
}

/// A power ratio in decibels.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub struct Db(pub f64);

impl Db {
    /// The linear power ratio `10^{x/10}` ([`db2lin`]).
    pub fn to_linear(self) -> f64 {
        db2lin(self.0)
    }

    /// The decibel value `10 log_{10}(x)` of a linear power ratio ([`lin2db`]).
    pub fn from_linear(x: f64) -> Db {
        Db(lin2db(x))
    }
}

/// Energy per information bit over the noise spectral density, `E_{b}/N_{0}`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub struct EbN0(pub Db);

/// Energy per symbol over the noise spectral density, `E_{s}/N_{0}`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub struct EsN0(pub Db);

impl EbN0 {
    /// Converts to `E_{s}/N_{0}` for `bits_per_symbol` coded bits per symbol at code
    /// rate `rate`: $E_{s}/N_{0} = E_{b}/N_{0} + 10 \log_{10}(k R)$ dB. The number of bits
    /// is real: the entropy of a shaped constellation, for instance.
    pub fn to_es_n0(self, bits_per_symbol: f64, rate: f64) -> EsN0 {
        EsN0(Db(self.0.0 + lin2db(bits_per_symbol * rate)))
    }
}

impl EsN0 {
    /// Converts to `E_{b}/N_{0}` for `bits_per_symbol` coded bits per symbol at code
    /// rate `rate` (the inverse of [`EbN0::to_es_n0`]).
    pub fn to_eb_n0(self, bits_per_symbol: f64, rate: f64) -> EbN0 {
        EbN0(Db(self.0.0 - lin2db(bits_per_symbol * rate)))
    }
}

/// A frequency or a rate (sample rate, symbol rate), in hertz.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub struct Hertz(pub f64);

/// A duration, in seconds.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub struct Seconds(pub f64);

impl Hertz {
    /// The period `1/f`.
    pub fn period(self) -> Seconds {
        Seconds(1.0 / self.0)
    }
}

impl Seconds {
    /// The frequency `1/T`.
    pub fn frequency(self) -> Hertz {
        Hertz(1.0 / self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_round_trip() {
        let eb = EbN0(Db(6.0));
        let back = eb.to_es_n0(4.0, 0.75).to_eb_n0(4.0, 0.75);
        assert!((back.0.0 - 6.0).abs() < 1e-12);
        assert!((Hertz(32e9).period().frequency().0 - 32e9).abs() < 1e-3);
        assert!((Db::from_linear(Db(-7.5).to_linear()).0 + 7.5).abs() < 1e-12);
    }
}
