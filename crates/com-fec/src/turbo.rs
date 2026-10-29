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

use crate::block::{BlockCode, container_traits, expect_len};
use crate::conv::{Trellis, bcjr};
use crate::interleaving::Interleaver;
use com_core::Result;
use com_core::signal::hard_decisions;

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

/// The QPP coefficients $(K, f_{1}, f_{2})$ of the turbo interleaver of LTE, 3GPP TS 36.212,
/// Table 5.1.3-3 (transcribed from the table of Sionna, `turbo_coeffs.csv`, Apache-2.0,
/// see `THIRD_PARTY.md`; every entry is checked to be a permutation, and the codewords
/// of four sizes against the reference vectors of Sionna).
#[rustfmt::skip]
const LTE_QPP: [(usize, usize, usize); 188] = [
    (40, 3, 10), (48, 7, 12), (56, 19, 42), (64, 7, 16), (72, 7, 18), (80, 11, 20), (88, 5, 22),
    (96, 11, 24), (104, 7, 26), (112, 41, 84), (120, 103, 90), (128, 15, 32), (136, 9, 34),
    (144, 17, 108), (152, 9, 38), (160, 21, 120), (168, 101, 84), (176, 21, 44), (184, 57, 46),
    (192, 23, 48), (200, 13, 50), (208, 27, 52), (216, 11, 36), (224, 27, 56), (232, 85, 58),
    (240, 29, 60), (248, 33, 62), (256, 15, 32), (264, 17, 198), (272, 33, 68), (280, 103, 210),
    (288, 19, 36), (296, 19, 74), (304, 37, 76), (312, 19, 78), (320, 21, 120), (328, 21, 82),
    (336, 115, 84), (344, 193, 86), (352, 21, 44), (360, 133, 90), (368, 81, 46), (376, 45, 94),
    (384, 23, 48), (392, 243, 98), (400, 151, 40), (408, 155, 102), (416, 25, 52), (424, 51, 106),
    (432, 47, 72), (440, 91, 110), (448, 29, 168), (456, 29, 114), (464, 247, 58), (472, 29, 118),
    (480, 89, 180), (488, 91, 122), (496, 157, 62), (504, 55, 84), (512, 31, 64), (528, 17, 66),
    (544, 35, 68), (560, 227, 420), (576, 65, 96), (592, 19, 74), (608, 37, 76), (624, 41, 234),
    (640, 39, 80), (656, 185, 82), (672, 43, 252), (688, 21, 86), (704, 155, 44), (720, 79, 120),
    (736, 139, 92), (752, 23, 94), (768, 217, 48), (784, 25, 98), (800, 17, 80), (816, 127, 102),
    (832, 25, 52), (848, 239, 106), (864, 17, 48), (880, 137, 110), (896, 215, 112), (912, 29, 114),
    (928, 15, 58), (944, 147, 118), (960, 29, 60), (976, 59, 122), (992, 65, 124), (1008, 55, 84),
    (1024, 31, 64), (1056, 17, 66), (1088, 171, 204), (1120, 67, 140), (1152, 35, 72),
    (1184, 19, 74), (1216, 39, 76), (1248, 19, 78), (1280, 199, 240), (1312, 21, 82),
    (1344, 211, 252), (1376, 21, 86), (1408, 43, 88), (1440, 149, 60), (1472, 45, 92),
    (1504, 49, 846), (1536, 71, 48), (1568, 13, 28), (1600, 17, 80), (1632, 25, 102),
    (1664, 183, 104), (1696, 55, 954), (1728, 127, 96), (1760, 27, 110), (1792, 29, 112),
    (1824, 29, 114), (1856, 57, 116), (1888, 45, 354), (1920, 31, 120), (1952, 59, 610),
    (1984, 185, 124), (2016, 113, 420), (2048, 31, 64), (2112, 17, 66), (2176, 171, 136),
    (2240, 209, 420), (2304, 253, 216), (2368, 367, 444), (2432, 265, 456), (2496, 181, 468),
    (2560, 39, 80), (2624, 27, 164), (2688, 127, 504), (2752, 143, 172), (2816, 43, 88),
    (2880, 29, 300), (2944, 45, 92), (3008, 157, 188), (3072, 47, 96), (3136, 13, 28),
    (3200, 111, 240), (3264, 443, 204), (3328, 51, 104), (3392, 51, 212), (3456, 451, 192),
    (3520, 257, 220), (3584, 57, 336), (3648, 313, 228), (3712, 271, 232), (3776, 179, 236),
    (3840, 331, 120), (3904, 363, 244), (3968, 375, 248), (4032, 127, 168), (4096, 31, 64),
    (4160, 33, 130), (4224, 43, 264), (4288, 33, 134), (4352, 477, 408), (4416, 35, 138),
    (4480, 233, 280), (4544, 357, 142), (4608, 337, 480), (4672, 37, 146), (4736, 71, 444),
    (4800, 71, 120), (4864, 37, 152), (4928, 39, 462), (4992, 127, 234), (5056, 39, 158),
    (5120, 39, 80), (5184, 31, 96), (5248, 113, 902), (5312, 41, 166), (5376, 251, 336),
    (5440, 43, 170), (5504, 21, 86), (5568, 43, 174), (5632, 45, 176), (5696, 45, 178),
    (5760, 161, 120), (5824, 89, 182), (5888, 323, 184), (5952, 47, 186), (6016, 23, 94),
    (6080, 47, 190), (6144, 263, 480),
];

/// The QPP coefficients $(f_{1}, f_{2})$ of the turbo interleaver of LTE for the block
/// size `k` (TS 36.212, Table 5.1.3-3), `None` for a size outside the table.
pub fn lte_qpp(k: usize) -> Option<(usize, usize)> {
    LTE_QPP.binary_search_by_key(&k, |e| e.0).ok().map(|i| (LTE_QPP[i].1, LTE_QPP[i].2))
}

impl TurboCode {
    /// The code of LTE: the quadratic permutation polynomial interleaver of [`lte_qpp`]
    /// for the 188 block sizes of TS 36.212 (40 to 6144), a random interleaver of the
    /// same length for any other size.
    ///
    /// # Errors
    /// Never for the sizes of the table.
    pub fn lte(k: usize) -> Result<TurboCode> {
        let interleaver = match lte_qpp(k) {
            Some((f1, f2)) => Interleaver::qpp(k, f1, f2)?,
            None => Interleaver::random(k, 0x7u64),
        };
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
        Ok(hard_decisions(&app))
    }
}

container_traits!(TurboCode);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lte_qpp_table_gives_permutations() {
        assert_eq!((LTE_QPP[0], LTE_QPP[187]), ((40, 3, 10), (6144, 263, 480)));
        assert!(LTE_QPP.windows(2).all(|w| w[0].0 < w[1].0));
        for &(k, f1, f2) in &LTE_QPP {
            assert!(Interleaver::qpp(k, f1, f2).is_ok(), "K = {k}");
        }
        assert_eq!(lte_qpp(432), Some((47, 72)));
        assert_eq!(lte_qpp(41), None);
    }
    use crate::block::testing::awgn_llr;
    use crate::bounds::bpsk_noise_variance;
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
            let s2 = bpsk_noise_variance(1.0, code.code_rate());
            let y = awgn_llr(&c, s2, &mut r);
            code.iterations = 1;
            e1 += code.decode_block(&y).unwrap().iter().zip(&info).filter(|(a, b)| a != b).count();
            code.iterations = 8;
            e8 += code.decode_block(&y).unwrap().iter().zip(&info).filter(|(a, b)| a != b).count();
        }
        assert!(e8 * 10 < e1.max(1), "1 iteration: {e1} errors, 8 iterations: {e8}");
    }
}
