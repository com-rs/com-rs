//! One verb trait per role in the chain.
//!
//! Methods taking `&self` are pure; `&mut self` signals internal state (memory,
//! adaptation). Every trait has an `_into` method writing into a caller-owned buffer,
//! which it resizes and fills in place (never replaces), so that a loop passing the same
//! buffer allocates no output once its shape is stable; scratch buffers may still
//! allocate (`docs/adr/0011-into-reuses-the-output.md`). All but [`Precode`] and
//! [`Detect`] also have an allocating method that calls it. Randomness is always
//! explicit: a random block receives `&mut SimRng`.
//!
//! The waveform traits are generic over the symbols they carry, [`Symbols`] by default:
//! a waveform of complex symbols (OFDM) implements `Modulate`, one that carries the
//! index of a symbol (chirp spread spectrum, continuous-phase modulation) implements
//! `Modulate<Labels>` (`docs/adr/0012-index-symbol-waveforms.md`). [`Demap`] takes the
//! a priori log-probabilities of the points and the [`Jacobian`] logarithm through
//! [`Demapping`], uniform and exact by default.
//!
//! | Trait | Role | Input → output |
//! |---|---|---|
//! | [`Encode`] / [`Decode`] | channel coding | [`Bits`] → [`Bits`], [`Llr`] → [`Bits`] |
//! | [`Map`] / [`Demap`] | constellation mapping | [`Bits`] → [`Symbols`] → [`Llr`] |
//! | [`Modulate`] / [`Demodulate`] | waveform | [`Symbols`] or [`Labels`](crate::signal::Labels) → [`Signal`] → [`Symbols`] or [`Labels`](crate::signal::Labels) |
//! | [`Precode`] / [`Detect`] | MIMO | [`Symbols`] → [`Symbols`] |
//! | [`Propagate`] | propagation medium | [`Signal`] → [`Signal`] |
//! | [`Impair`] | component non-ideality | [`Signal`] → [`Signal`] |
//!
//! # Examples
//! ```
//! use com_core::prelude::*;
//!
//! /// A noiseless gain, the smallest `Impair`.
//! struct Gain(f64);
//!
//! impl Impair for Gain {
//!     fn impair_into(&mut self, x: &Signal, _: &mut SimRng, out: &mut Signal) -> Result<()> {
//!         out.resize(x.n_streams(), x.len(), x.sample_rate());
//!         out.as_array_mut().assign(&(x.as_array() * self.0));
//!         Ok(())
//!     }
//! }
//!
//! let x = Signal::from_stream(vec![C64::new(1.0, 0.0); 4], Hertz(1.0))?;
//! let y = Gain(2.0).impair(&x, &mut batch_rng(1, 0))?;
//! assert_eq!(y.stream(0)[3], C64::new(2.0, 0.0));
//! # Ok::<(), com_core::Error>(())
//! ```

use crate::error::Result;
use crate::rng::SimRng;
use crate::signal::{Bits, Llr, Signal, Symbols};
use crate::units::Hertz;

/// Channel encoding of information bits into codewords.
pub trait Encode {
    /// Code rate `k/n`.
    fn rate(&self) -> f64;

    /// Encode `info` into `out`.
    ///
    /// # Errors
    /// When the length of `info` is not a multiple of the information length.
    fn encode_into(&self, info: &Bits, out: &mut Bits) -> Result<()>;

    /// Encode `info` (allocates, calls [`Encode::encode_into`]).
    ///
    /// # Errors
    /// As [`Encode::encode_into`].
    fn encode(&self, info: &Bits) -> Result<Bits> {
        let mut out = Bits::default();
        self.encode_into(info, &mut out)?;
        Ok(out)
    }
}

/// Channel decoding of log-likelihood ratios into information bits.
pub trait Decode {
    /// Decode `llr` into `out`.
    ///
    /// # Errors
    /// When the length of `llr` is not a multiple of the codeword length.
    fn decode_into(&mut self, llr: &Llr, out: &mut Bits) -> Result<()>;

    /// Decode `llr` (allocates, calls [`Decode::decode_into`]).
    ///
    /// # Errors
    /// As [`Decode::decode_into`].
    fn decode(&mut self, llr: &Llr) -> Result<Bits> {
        let mut out = Bits::default();
        self.decode_into(llr, &mut out)?;
        Ok(out)
    }
}

/// Mapping of bits onto constellation symbols.
pub trait Map {
    /// Map `bits` into `out`.
    ///
    /// # Errors
    /// When the length of a stream is not a multiple of the bits per symbol.
    fn map_into(&self, bits: &Bits, out: &mut Symbols) -> Result<()>;

    /// Map `bits` (allocates, calls [`Map::map_into`]).
    ///
    /// # Errors
    /// As [`Map::map_into`].
    fn map(&self, bits: &Bits) -> Result<Symbols> {
        let mut out = Symbols::default();
        self.map_into(bits, &mut out)?;
        Ok(out)
    }
}

/// The Jacobian logarithm of the log-MAP algorithms and of the soft demappers, exact or
/// approximated by its maximum:
///
/// $$
/// \ln\left(e^{a} + e^{b}\right) = \max(a, b) + \ln\left(1 + e^{-\lvert a - b \rvert}\right)
/// \approx \max(a, b).
/// $$
///
/// One choice for [`Demap`] ([`Demapping::jacobian`]), `max_star` of `com-math` (which
/// re-exports it as `com_math::Jacobian`), the BCJR and turbo decoders of `com-fec` and
/// the demappers of `com-mapping`; a specification keeps the boolean key `max_log`
/// ([`Jacobian::from_max_log`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Jacobian {
    /// The exact logarithm (log-MAP, log-sum-exp).
    #[default]
    Exact,
    /// The max-log approximation, $\max(a, b)$.
    MaxLog,
}

impl Jacobian {
    /// [`Jacobian::MaxLog`] when `max_log` is true, [`Jacobian::Exact`] otherwise (the
    /// boolean key `max_log` of the specifications and of the Python API).
    ///
    /// # Examples
    /// ```
    /// use com_core::traits::Jacobian;
    /// assert_eq!(Jacobian::from_max_log(true), Jacobian::MaxLog);
    /// assert_eq!(Jacobian::from_max_log(false), Jacobian::default());
    /// ```
    pub fn from_max_log(max_log: bool) -> Jacobian {
        if max_log { Jacobian::MaxLog } else { Jacobian::Exact }
    }

    /// Whether the logarithm is exact.
    pub fn is_exact(self) -> bool {
        self == Jacobian::Exact
    }
}

/// The a priori information and the approximation of a soft demapper
/// ([`Demap::demap_with_into`]). With the log-priors $\lambda_{x} = \ln P(x)$ of the
/// points $x$ of the alphabet and the variance $\sigma^{2}$ of the complex noise, the
/// LLR of a bit of the received symbol $y$ is
///
/// $$
/// L(b) = \ln \sum_{x:\\, b = 0} e^{\lambda_{x} - |y - x|^{2} / \sigma^{2}}
///      - \ln \sum_{x:\\, b = 1} e^{\lambda_{x} - |y - x|^{2} / \sigma^{2}},
/// $$
///
/// each sum exact or max-log by [`Demapping::jacobian`]. A constant added to every
/// $\lambda_{x}$ cancels, so the log-priors need not be normalised. The default, uniform
/// priors and the exact logarithm, is that of [`Demap::demap_into`].
///
/// # Examples
/// ```
/// use com_core::traits::{Demapping, Jacobian};
/// let lp = [-0.5f64, -1.5];
/// let d = Demapping { log_priors: Some(&lp), jacobian: Jacobian::MaxLog };
/// assert_eq!(Demapping::default(), Demapping { log_priors: None, jacobian: Jacobian::Exact });
/// assert_ne!(d, Demapping::default());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Demapping<'a> {
    /// The a priori log-probabilities $\ln P(x)$, one per point in the order of the
    /// labels (probabilistic shaping); `None` for uniform priors, which need no table.
    pub log_priors: Option<&'a [f64]>,
    /// The Jacobian logarithm of the sums over the labels: exact or max-log.
    pub jacobian: Jacobian,
}

/// Soft demapping of received symbols into log-likelihood ratios.
pub trait Demap {
    /// Demap `y`, received with a complex noise variance `noise_var`, into `out`, with
    /// the priors and the logarithm of `demapping` (see [`Demapping`]).
    ///
    /// # Errors
    /// [`crate::Error::Config`] for a noise variance that is not positive and finite, or
    /// log-priors with a NaN or $+\infty$, or none finite; [`crate::Error::Shape`] for
    /// log-priors of another length than the alphabet.
    fn demap_with_into(&self, y: &Symbols, noise_var: f64, demapping: Demapping<'_>, out: &mut Llr) -> Result<()>;

    /// Demap `y`, received with a complex noise variance `noise_var`, into `out`: the
    /// exact LLRs under uniform priors ([`Demap::demap_with_into`] with the default
    /// [`Demapping`]).
    ///
    /// # Errors
    /// As [`Demap::demap_with_into`].
    fn demap_into(&self, y: &Symbols, noise_var: f64, out: &mut Llr) -> Result<()> {
        self.demap_with_into(y, noise_var, Demapping::default(), out)
    }

    /// Demap `y` with `demapping` (allocates, calls [`Demap::demap_with_into`]).
    ///
    /// # Errors
    /// As [`Demap::demap_with_into`].
    fn demap_with(&self, y: &Symbols, noise_var: f64, demapping: Demapping<'_>) -> Result<Llr> {
        let mut out = Llr::default();
        self.demap_with_into(y, noise_var, demapping, &mut out)?;
        Ok(out)
    }

    /// Demap `y` (allocates, calls [`Demap::demap_into`]).
    ///
    /// # Errors
    /// As [`Demap::demap_into`].
    fn demap(&self, y: &Symbols, noise_var: f64) -> Result<Llr> {
        let mut out = Llr::default();
        self.demap_into(y, noise_var, &mut out)?;
        Ok(out)
    }
}

/// Modulation of symbols onto a waveform: complex [`Symbols`] by default, or the
/// [`Labels`](crate::signal::Labels) of a waveform that carries the index of a symbol (`Modulate<Labels>`).
pub trait Modulate<In = Symbols> {
    /// Modulate `x` into `out`.
    ///
    /// # Errors
    /// When the number of symbols does not fit the waveform structure.
    fn modulate_into(&self, x: &In, out: &mut Signal) -> Result<()>;

    /// Modulate `x` (allocates, calls [`Modulate::modulate_into`]).
    ///
    /// # Errors
    /// As [`Modulate::modulate_into`].
    fn modulate(&self, x: &In) -> Result<Signal> {
        let mut out = Signal::zeros(0, 0, Hertz(1.0));
        self.modulate_into(x, &mut out)?;
        Ok(out)
    }
}

/// Demodulation of a waveform back to symbols: complex [`Symbols`] by default, or the
/// decided [`Labels`](crate::signal::Labels) of a waveform that carries the index of a symbol
/// (`Demodulate<Labels>`).
pub trait Demodulate<Out = Symbols> {
    /// Demodulate `y` into `out`.
    ///
    /// # Errors
    /// When the number of samples does not fit the waveform structure.
    fn demodulate_into(&mut self, y: &Signal, out: &mut Out) -> Result<()>;

    /// Demodulate `y` (allocates, calls [`Demodulate::demodulate_into`]).
    ///
    /// # Errors
    /// As [`Demodulate::demodulate_into`].
    fn demodulate(&mut self, y: &Signal) -> Result<Out>
    where
        Out: Default,
    {
        let mut out = Out::default();
        self.demodulate_into(y, &mut out)?;
        Ok(out)
    }
}

/// MIMO precoding of streams of symbols.
pub trait Precode {
    /// Precode `x` into `out`.
    ///
    /// # Errors
    /// When the number of streams does not match the precoder.
    fn precode_into(&self, x: &Symbols, out: &mut Symbols) -> Result<()>;
}

/// Detection of transmitted symbols from received streams.
pub trait Detect {
    /// Detect the streams of `y`, received with a noise variance `noise_var`, into `out`.
    ///
    /// # Errors
    /// When the number of streams does not match the channel.
    fn detect_into(&mut self, y: &Symbols, noise_var: f64, out: &mut Symbols) -> Result<()>;
}

/// Propagation through a medium. Randomness is always explicit.
pub trait Propagate {
    /// Propagate `x` into `out`, drawing the random numbers from `rng`.
    ///
    /// # Errors
    /// When the input does not fit the medium (empty signal, wrong number of streams).
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()>;

    /// Propagate `x` (allocates, calls [`Propagate::propagate_into`]).
    ///
    /// # Errors
    /// As [`Propagate::propagate_into`].
    fn propagate(&mut self, x: &Signal, rng: &mut SimRng) -> Result<Signal> {
        let mut out = Signal::zeros(0, 0, x.sample_rate());
        self.propagate_into(x, rng, &mut out)?;
        Ok(out)
    }
}

/// A component non-ideality (amplifier, oscillator, converter…).
pub trait Impair {
    /// Apply the impairment to `x` into `out`, drawing the random numbers from `rng`.
    ///
    /// # Errors
    /// When the input does not fit the component.
    fn impair_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()>;

    /// Apply the impairment to `x` (allocates, calls [`Impair::impair_into`]).
    ///
    /// # Errors
    /// As [`Impair::impair_into`].
    fn impair(&mut self, x: &Signal, rng: &mut SimRng) -> Result<Signal> {
        let mut out = Signal::zeros(0, 0, x.sample_rate());
        self.impair_into(x, rng, &mut out)?;
        Ok(out)
    }
}
