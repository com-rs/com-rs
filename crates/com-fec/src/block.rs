//! The interface shared by the block codes.
//!
//! A [`BlockCode`] maps $k$ information bits to $n$ coded bits and decodes $n$ LLRs
//! (positive for a 0, the convention of `com_core::signal`) back to $k$ bits. Its
//! provided methods encode and decode streams of whole blocks, and every code of the
//! crate implements the verb traits [`Encode`](com_core::traits::Encode) and
//! [`Decode`](com_core::traits::Decode) on the multi-stream containers. A code whose
//! decoder gives the a posteriori LLRs of its information bits also implements
//! [`SoftDecode`].

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

    /// Encode one block of $k$ bits into the $n$ values of `out` (ADR 0011: the default
    /// copies [`BlockCode::encode_block`]; a code overrides it when a profile shows the
    /// copy).
    ///
    /// # Errors
    /// [`Error::Shape`] if `info` does not have $k$ bits or `out` $n$ values.
    fn encode_block_into(&self, info: &[u8], out: &mut [u8]) -> Result<()> {
        let c = self.encode_block(info)?;
        com_core::same_len(out.len(), c.len(), "coded bits")?;
        out.copy_from_slice(&c);
        Ok(())
    }

    /// Decode one block of $n$ LLRs into the $k$ values of `out` (as
    /// [`BlockCode::encode_block_into`]).
    ///
    /// # Errors
    /// [`Error::Shape`] if `llr` does not have $n$ values or `out` $k$ values.
    fn decode_block_into(&self, llr: &[f64], out: &mut [u8]) -> Result<()> {
        let d = self.decode_block(llr)?;
        com_core::same_len(out.len(), d.len(), "decoded bits")?;
        out.copy_from_slice(&d);
        Ok(())
    }

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
        whole_blocks(info.len(), k, "bits")?;
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
        whole_blocks(llr.len(), n, "LLRs")?;
        let mut out = Vec::with_capacity(llr.len() / n * self.k());
        for b in llr.chunks(n) {
            out.extend(self.decode_block(b)?);
        }
        Ok(out)
    }
}

/// A block code whose decoder gives the a posteriori LLRs of the information bits: the
/// soft output of an iterative receiver (turbo equalisation, iterative demapping) or of
/// a concatenation.
///
/// Implemented by the [`LdpcCode`](crate::ldpc::LdpcCode) (belief propagation), the
/// [`ConvolutionalCode`](crate::conv::ConvolutionalCode) (BCJR, log-MAP), the
/// [`TurboCode`](crate::turbo::TurboCode), the [`NrLdpc`](crate::nr::NrLdpc) code, and
/// the punctured codes [`Punctured`](crate::puncture::Punctured) and
/// [`PuncturedTurbo`](crate::puncture::PuncturedTurbo), which put back a zero LLR at the
/// punctured positions. The list decoder of the polar codes gives hard candidates only
/// ([`PolarCode::decode_list`](crate::polar::PolarCode::decode_list)).
///
/// # Model
///
/// With the channel LLRs $L_{c}$ of the $n$ coded bits and the a priori LLRs $L_{a}$ of
/// the $k$ information bits $u_{i}$ (zero if `None`),
///
/// $$
/// L_{\mathrm{app}}(u_{i}) = \ln \frac{P(u_{i} = 0 \mid L_{c}, L_{a})}{P(u_{i} = 1 \mid L_{c}, L_{a})},
/// $$
///
/// exact for the BCJR algorithm, approximated by the iterative decoders; the extrinsic
/// information of the code is $L_{\mathrm{app}} - L_{a}$ minus the channel LLR of a
/// systematic bit. The hard decisions of $L_{\mathrm{app}}$ are those of
/// [`BlockCode::decode_block`] for the iterative codes; a convolutional code decodes its
/// blocks by Viterbi (maximum-likelihood sequence) and its soft output by BCJR
/// (maximum a posteriori bit), which may differ on a few bits.
pub trait SoftDecode: BlockCode {
    /// The a posteriori LLRs of the $k$ information bits into `out`, from the $n$
    /// channel LLRs `llr` and the $k$ a priori LLRs `apriori`; returns the iterations
    /// of the decoder (1 for a non-iterative one, 0 when the input was already a
    /// codeword for the LDPC codes).
    ///
    /// # Errors
    /// [`Error::Shape`] if `llr` does not have $n$ values, or `apriori` or `out` $k$.
    fn app_into(&self, llr: &[f64], apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize>;

    /// [`SoftDecode::app_into`] into a new vector.
    ///
    /// # Errors
    /// As [`SoftDecode::app_into`].
    ///
    /// # Examples
    /// ```
    /// use com_fec::block::{BlockCode, SoftDecode};
    /// use com_fec::turbo::TurboCode;
    /// let code = TurboCode::lte(40)?;
    /// let info: Vec<u8> = (0..40).map(|i| (i % 3 == 1) as u8).collect();
    /// let llr: Vec<f64> = code.encode_block(&info)?.iter().map(|&b| if b == 0 { 2.0 } else { -2.0 }).collect();
    /// let app = code.app(&llr, None)?;
    /// assert_eq!(com_core::signal::hard_decisions(&app), info);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    fn app(&self, llr: &[f64], apriori: Option<&[f64]>) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.k()];
        self.app_into(llr, apriori, &mut out)?;
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
                let (k, n) = ($crate::block::BlockCode::k(self), $crate::block::BlockCode::n(self));
                $crate::block::whole_blocks(info.len(), k, "bits")?;
                out.resize(info.n_streams(), info.len() / k * n);
                for s in 0..info.n_streams() {
                    let o = out.stream_slice_mut(s);
                    for (b, c) in info.stream_slice(s).chunks(k).zip(o.chunks_mut(n)) {
                        $crate::block::BlockCode::encode_block_into(self, b, c)?;
                    }
                }
                Ok(())
            }
        }

        impl<$($g)*> com_core::traits::Decode for $t {
            fn decode_into(
                &mut self,
                llr: &com_core::signal::Llr,
                out: &mut com_core::signal::Bits,
            ) -> com_core::Result<()> {
                let (k, n) = ($crate::block::BlockCode::k(self), $crate::block::BlockCode::n(self));
                $crate::block::whole_blocks(llr.len(), n, "LLRs")?;
                out.resize(llr.n_streams(), llr.len() / n * k);
                for s in 0..llr.n_streams() {
                    let o = out.stream_slice_mut(s);
                    for (l, d) in llr.stream_slice(s).chunks(n).zip(o.chunks_mut(k)) {
                        $crate::block::BlockCode::decode_block_into(self, l, d)?;
                    }
                }
                Ok(())
            }
        }
    };
}
pub(crate) use container_traits;

/// [`Error::Shape`] unless `len` values make whole blocks of `block` (for the container
/// traits and the streams).
pub(crate) fn whole_blocks(len: usize, block: usize, what: &str) -> Result<()> {
    if block == 0 || len % block != 0 {
        return Err(Error::Shape(format!("{len} {what} is not a multiple of the block length {block}")));
    }
    Ok(())
}

pub(crate) use com_math::{Jacobian, max_star};

/// Helpers of the tests of the codes.
#[cfg(test)]
pub(crate) mod testing {
    use com_core::rng::Rng;

    /// The LLRs of the bits `c` sent in BPSK ($1 - 2b$) over a real AWGN channel of
    /// variance `s2`: one Gaussian draw per bit, the LLRs $2 y / \sigma^{2}$ (the link of
    /// `com_validation::bpsk_awgn_llr`; the code crate holds no modulation).
    pub(crate) fn awgn_llr(c: &[u8], s2: f64, rng: &mut Rng) -> Vec<f64> {
        let s = s2.sqrt();
        c.iter().map(|&b| 2.0 * ((1.0 - 2.0 * f64::from(b)) + s * rng.gaussian()) / s2).collect()
    }
}
