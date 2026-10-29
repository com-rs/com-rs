//! The interface shared by the block codes.
//!
//! A [`BlockCode`] maps $k$ information bits to $n$ coded bits and decodes $n$ LLRs
//! (positive for a 0, the convention of `com_core::signal`) back to $k$ bits. Its
//! provided methods encode and decode streams of whole blocks, and every code of the
//! crate implements the verb traits [`Encode`](com_core::traits::Encode) and
//! [`Decode`](com_core::traits::Decode) on the multi-stream containers.

use com_core::{Error, Result};

/// A code of $k$ information bits and $n$ coded bits per block.
pub trait BlockCode {
    /// Information bits per block.
    fn k(&self) -> usize;

    /// Coded bits per block.
    fn n(&self) -> usize;

    /// Encode one block of $k$ bits.
    ///
    /// # Errors
    /// [`Error::Shape`] if `info` does not have $k$ bits.
    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>>;

    /// Decode one block of $n$ LLRs into $k$ bits.
    ///
    /// # Errors
    /// [`Error::Shape`] if `llr` does not have $n$ values.
    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>>;

    /// Code rate $k/n$.
    fn code_rate(&self) -> f64 {
        self.k() as f64 / self.n() as f64
    }

    /// Encode whole blocks.
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of bits is not a multiple of $k$.
    fn encode_stream(&self, info: &[u8]) -> Result<Vec<u8>> {
        let k = self.k();
        if k == 0 || info.len() % k != 0 {
            return Err(Error::Shape(format!("{} bits is not a multiple of k = {k}", info.len())));
        }
        let mut out = Vec::with_capacity(info.len() / k * self.n());
        for b in info.chunks(k) {
            out.extend(self.encode_block(b)?);
        }
        Ok(out)
    }

    /// Decode whole blocks.
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of LLRs is not a multiple of $n$.
    fn decode_stream(&self, llr: &[f64]) -> Result<Vec<u8>> {
        let n = self.n();
        if n == 0 || llr.len() % n != 0 {
            return Err(Error::Shape(format!("{} LLRs is not a multiple of n = {n}", llr.len())));
        }
        let mut out = Vec::with_capacity(llr.len() / n * self.k());
        for b in llr.chunks(n) {
            out.extend(self.decode_block(b)?);
        }
        Ok(out)
    }
}

pub(crate) use com_core::expect_len;

/// `Encode` and `Decode` on the multi-stream containers for a [`BlockCode`], every
/// stream encoded (decoded) as whole blocks: `container_traits!(Code)`, or with generic
/// parameters `container_traits!(Punctured<C>, C: BlockCode)`.
macro_rules! container_traits {
    ($t:ty) => {
        $crate::block::container_traits!($t,);
    };
    ($t:ty, $($g:tt)*) => {
        impl<$($g)*> com_core::traits::Encode for $t {
            fn rate(&self) -> f64 {
                $crate::block::BlockCode::code_rate(self)
            }

            fn encode_into(
                &self,
                info: &com_core::signal::Bits,
                out: &mut com_core::signal::Bits,
            ) -> com_core::Result<()> {
                let rows = (0..info.n_streams())
                    .map(|s| $crate::block::BlockCode::encode_stream(self, info.stream_slice(s)))
                    .collect::<com_core::Result<Vec<_>>>()?;
                *out = com_core::signal::Bits::from_streams(&rows)?;
                Ok(())
            }
        }

        impl<$($g)*> com_core::traits::Decode for $t {
            fn decode_into(
                &mut self,
                llr: &com_core::signal::Llr,
                out: &mut com_core::signal::Bits,
            ) -> com_core::Result<()> {
                let rows = (0..llr.n_streams())
                    .map(|s| $crate::block::BlockCode::decode_stream(self, llr.stream_slice(s)))
                    .collect::<com_core::Result<Vec<_>>>()?;
                *out = com_core::signal::Bits::from_streams(&rows)?;
                Ok(())
            }
        }
    };
}
pub(crate) use container_traits;

/// The LLRs of BPSK ($0 \mapsto +1$, $1 \mapsto -1$) received in real Gaussian noise of
/// variance `noise_var` per dimension, $2 y / \sigma^{2}$.
pub fn bpsk_llr(y: &[f64], noise_var: f64) -> Vec<f64> {
    y.iter().map(|v| 2.0 * v / noise_var).collect()
}

/// The BPSK symbols $1 - 2b$ of bits.
pub fn bpsk(bits: &[u8]) -> Vec<f64> {
    bits.iter().map(|b| 1.0 - 2.0 * f64::from(*b)).collect()
}

pub(crate) use com_math::max_star;

/// Helpers of the tests of the codes.
#[cfg(test)]
pub(crate) mod testing {
    use com_core::rng::Rng;

    /// The LLRs of the bits `c` sent in BPSK over a real AWGN channel of variance `s2`.
    pub(crate) fn awgn_llr(c: &[u8], s2: f64, rng: &mut Rng) -> Vec<f64> {
        c.iter().map(|b| 2.0 * (1.0 - 2.0 * f64::from(*b) + s2.sqrt() * rng.gaussian()) / s2).collect()
    }
}
