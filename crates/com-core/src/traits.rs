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
//! | Trait | Role | Input → output |
//! |---|---|---|
//! | [`Encode`] / [`Decode`] | channel coding | [`Bits`] → [`Bits`], [`Llr`] → [`Bits`] |
//! | [`Map`] / [`Demap`] | constellation mapping | [`Bits`] → [`Symbols`] → [`Llr`] |
//! | [`Modulate`] / [`Demodulate`] | waveform | [`Symbols`] → [`Signal`] → [`Symbols`] |
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

/// Soft demapping of received symbols into log-likelihood ratios.
pub trait Demap {
    /// Demap `y`, received with a complex noise variance `noise_var`, into `out`.
    ///
    /// # Errors
    /// For a non-positive noise variance.
    fn demap_into(&self, y: &Symbols, noise_var: f64, out: &mut Llr) -> Result<()>;

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

/// Modulation of symbols onto a waveform.
pub trait Modulate {
    /// Modulate `x` into `out`.
    ///
    /// # Errors
    /// When the number of symbols does not fit the waveform structure.
    fn modulate_into(&self, x: &Symbols, out: &mut Signal) -> Result<()>;

    /// Modulate `x` (allocates, calls [`Modulate::modulate_into`]).
    ///
    /// # Errors
    /// As [`Modulate::modulate_into`].
    fn modulate(&self, x: &Symbols) -> Result<Signal> {
        let mut out = Signal::zeros(0, 0, Hertz(1.0));
        self.modulate_into(x, &mut out)?;
        Ok(out)
    }
}

/// Demodulation of a waveform back to symbols.
pub trait Demodulate {
    /// Demodulate `y` into `out`.
    ///
    /// # Errors
    /// When the number of samples does not fit the waveform structure.
    fn demodulate_into(&mut self, y: &Signal, out: &mut Symbols) -> Result<()>;

    /// Demodulate `y` (allocates, calls [`Demodulate::demodulate_into`]).
    ///
    /// # Errors
    /// As [`Demodulate::demodulate_into`].
    fn demodulate(&mut self, y: &Signal) -> Result<Symbols> {
        let mut out = Symbols::default();
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
