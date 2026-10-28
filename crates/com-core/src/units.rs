//! Unit newtypes for the quantities that are routinely confused.
//!
//! A decibel value is a [`Db`], never a bare `f64`; an `E_{b}/N_{0}` is an [`EbN0`] and
//! an `E_{s}/N_{0}` an [`EsN0`], so that passing one where the other is expected is a
//! compile error. Conversions are explicit methods, never `From` impls that hide a
//! formula.
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
//! let es = EbN0(Db(4.0)).to_es_n0(2, 1.0);         // QPSK, uncoded
//! assert!((es.0.0 - 7.0103).abs() < 1e-4);
//! assert!((Db(3.0).to_linear() - 1.9953).abs() < 1e-4);
//! assert!((Db::from_linear(10.0).0 - 10.0).abs() < 1e-12);
//! ```

/// A power ratio in decibels.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Default)]
pub struct Db(pub f64);

impl Db {
    /// The linear power ratio `10^{x/10}`.
    pub fn to_linear(self) -> f64 {
        10f64.powf(self.0 / 10.0)
    }

    /// The decibel value `10 log_{10}(x)` of a linear power ratio.
    pub fn from_linear(x: f64) -> Db {
        Db(10.0 * x.log10())
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
    /// rate `rate`.
    pub fn to_es_n0(self, bits_per_symbol: u32, rate: f64) -> EsN0 {
        EsN0(Db(self.0.0 + 10.0 * (f64::from(bits_per_symbol) * rate).log10()))
    }
}

impl EsN0 {
    /// Converts to `E_{b}/N_{0}` for `bits_per_symbol` coded bits per symbol at code
    /// rate `rate` (the inverse of [`EbN0::to_es_n0`]).
    pub fn to_eb_n0(self, bits_per_symbol: u32, rate: f64) -> EbN0 {
        EbN0(Db(self.0.0 - 10.0 * (f64::from(bits_per_symbol) * rate).log10()))
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
        let back = eb.to_es_n0(4, 0.75).to_eb_n0(4, 0.75);
        assert!((back.0.0 - 6.0).abs() < 1e-12);
        assert!((Hertz(32e9).period().frequency().0 - 32e9).abs() < 1e-3);
        assert!((Db::from_linear(Db(-7.5).to_linear()).0 + 7.5).abs() < 1e-12);
    }
}
