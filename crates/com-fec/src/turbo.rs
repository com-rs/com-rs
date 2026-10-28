//! Turbo codes: two recursive systematic codes in parallel, decoded iteratively.
//!
//! The encoder of Berrou, Glavieux & Thitimajshima (1993) in the form of LTE (3GPP TS
//! 36.212, Sec. 5.1.3.2): the $K$ information bits enter a recursive systematic code
//! $(13, 15)_{8}$ of constraint length 4, and their interleaved copy a second one. The
//! codeword is the systematic bits, the two parity sequences and the 12 tail bits that
//! terminate both encoders, $n = 3K + 12$.
//!
//! The decoder exchanges extrinsic information between two BCJR decoders: each
//! computes the a posteriori LLRs of the information bits with the extrinsic LLRs of
//! the other as a priori, and passes on the difference
//! $L_{e} = L_{\mathrm{app}} - L_{a} - L_{s}$. A few iterations reach within about 1 dB of
//! capacity for long blocks.
//!
//! # Examples
//! ```
//! use com_fec::block::{BlockCode, bpsk_llr};
//! use com_fec::turbo::TurboCode;
//! let code = TurboCode::lte(40)?;                         // K = 40, QPP (3, 10)
//! let info: Vec<u8> = (0..40).map(|i| (i % 3 == 1) as u8).collect();
//! let c = code.encode_block(&info)?;
//! assert_eq!(c.len(), 3 * 40 + 12);
//! let y: Vec<f64> = c.iter().map(|b| 1.0 - 2.0 * f64::from(*b)).collect();
//! assert_eq!(code.decode_block(&bpsk_llr(&y, 1.0))?, info);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. C. Berrou, A. Glavieux and P. Thitimajshima, "Near Shannon limit error-correcting
//!    coding and decoding: turbo-codes," *Proc. IEEE ICC*, 1064–1070, 1993.
//! 2. 3GPP TS 36.212, *E-UTRA; Multiplexing and channel coding*, Sec. 5.1.3.2.
//! 3. J. Sun and O. Y. Takeshita, "Interleavers for turbo codes using permutation
//!    polynomials over integer rings," *IEEE Trans. Inf. Theory*, 51(1), 101–119, 2005.

use crate::block::{BlockCode, container_traits, expect_len, hard};
use crate::conv::{Trellis, bcjr};
use crate::interleaving::Interleaver;
use com_core::Result;

/// A parallel concatenation of two $(13, 15)_{8}$ recursive systematic codes.
#[derive(Clone, Debug, PartialEq)]
pub struct TurboCode {
    /// The constituent trellis.
    pub trellis: Trellis,
    /// The interleaver of the second encoder ($K$ positions).
    pub interleaver: Interleaver,
    /// Decoding iterations.
    pub iterations: usize,
    /// log-MAP (true) or max-log-MAP decoding.
    pub exact: bool,
}

impl TurboCode {
    /// The code of LTE with the quadratic permutation polynomial interleaver of $K = 40$
    /// ($f_{1} = 3$, $f_{2} = 10$) or a random interleaver of the same length otherwise.
    ///
    /// # Errors
    /// Never for `k = 40`.
    pub fn lte(k: usize) -> Result<TurboCode> {
        let interleaver = if k == 40 { Interleaver::qpp(40, 3, 10)? } else { Interleaver::random(k, 0x7u64) };
        Ok(TurboCode::with_interleaver(interleaver))
    }

    /// The code with a given interleaver (its length is $K$), 8 iterations of log-MAP.
    pub fn with_interleaver(interleaver: Interleaver) -> TurboCode {
        let trellis = Trellis::recursive(4, 0o13, 0o15).unwrap_or_else(|_| unreachable!("a valid code"));
        TurboCode { trellis, interleaver, iterations: 8, exact: true }
    }

    fn tail_len(&self) -> usize {
        self.trellis.constraint as usize - 1
    }

    /// Split a coded block into its systematic, parity and tail parts, as the LLR
    /// sequences of the two constituent decoders (systematic and parity interleaved
    /// per step, the tail appended).
    fn constituents(&self, llr: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let k = self.k();
        let m = self.tail_len();
        let (sys, rest) = llr.split_at(k);
        let (p1, rest) = rest.split_at(k);
        let (p2, tail) = rest.split_at(k);
        let sys2 = self.interleaver.interleave(sys);
        let mut a = Vec::with_capacity(2 * (k + m));
        let mut b = Vec::with_capacity(2 * (k + m));
        for i in 0..k {
            a.extend([sys[i], p1[i]]);
            b.extend([sys2[i], p2[i]]);
        }
        a.extend(&tail[..2 * m]);
        b.extend(&tail[2 * m..]);
        (a, b)
    }
}

impl BlockCode for TurboCode {
    fn k(&self) -> usize {
        self.interleaver.len()
    }

    fn n(&self) -> usize {
        3 * self.k() + 4 * self.tail_len()
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        let k = self.k();
        expect_len(info.len(), k, "bits")?;
        let c1 = self.trellis.encode(info, true);
        let c2 = self.trellis.encode(&self.interleaver.interleave(info), true);
        let mut out = info.to_vec();
        out.extend((0..k).map(|i| c1[2 * i + 1]));
        out.extend((0..k).map(|i| c2[2 * i + 1]));
        out.extend(&c1[2 * k..]);
        out.extend(&c2[2 * k..]);
        Ok(out)
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        expect_len(llr.len(), self.n(), "LLRs")?;
        let k = self.k();
        let m = self.tail_len();
        let (a, b) = self.constituents(llr);
        let sys = &llr[..k];
        let sys2 = self.interleaver.interleave(sys);
        let mut ext2 = vec![0.0; k]; // extrinsic of decoder 2, deinterleaved
        let mut app = sys.to_vec();
        for _ in 0..self.iterations.max(1) {
            let mut la1 = ext2.clone();
            la1.extend(vec![0.0; m]);
            let app1 = bcjr(&self.trellis, &a, &la1, true, self.exact)?;
            let ext1: Vec<f64> = (0..k).map(|i| app1[i] - ext2[i] - sys[i]).collect();
            let mut la2 = self.interleaver.interleave(&ext1);
            la2.extend(vec![0.0; m]);
            let app2 = bcjr(&self.trellis, &b, &la2, true, self.exact)?;
            let e2: Vec<f64> = (0..k).map(|i| app2[i] - la2[i] - sys2[i]).collect();
            ext2 = self.interleaver.deinterleave(&e2);
            app = self.interleaver.deinterleave(&app2[..k]);
        }
        Ok(hard(&app))
    }
}

container_traits!(TurboCode);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::testing::awgn_llr;
    use crate::bounds::noise_variance;
    use com_core::rng::Rng;

    #[test]
    fn iterations_correct_more_errors() {
        let mut code = TurboCode::with_interleaver(Interleaver::random(512, 9));
        let mut r = Rng::new(2);
        let (mut e1, mut e8) = (0usize, 0usize);
        for _ in 0..20 {
            let info: Vec<u8> = (0..512).map(|_| r.bit()).collect();
            let c = code.encode_block(&info).unwrap();
            // Eb/N0 = 1 dB at rate ≈ 1/3
            let s2 = noise_variance(1.0, code.code_rate());
            let y = awgn_llr(&c, s2, &mut r);
            code.iterations = 1;
            e1 += code.decode_block(&y).unwrap().iter().zip(&info).filter(|(a, b)| a != b).count();
            code.iterations = 8;
            e8 += code.decode_block(&y).unwrap().iter().zip(&info).filter(|(a, b)| a != b).count();
        }
        assert!(e8 * 10 < e1.max(1), "1 iteration: {e1} errors, 8 iterations: {e8}");
    }
}
