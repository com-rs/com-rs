//! Puncturing of the convolutional and turbo codes: rate-compatible patterns, the
//! punctured block codes and the free distance of a punctured code.
//!
//! # Model
//!
//! A rate-$1/n$ code emits $n$ bits per trellis step; a [`Puncturer`] of period $P$
//! keeps the bit $i$ of step $t$ when the entry $(i, t \bmod P)$ of its pattern, an
//! $n \times P$ matrix of 0 and 1, is 1. With $w$ ones in the pattern, the rate becomes
//! $P / w$ (Cain, Clark & Geist 1979). The receiver puts back a zero LLR at every
//! punctured position: in the Viterbi and BCJR branch metric $\sum_{i} (1 - 2c_{i})
//! L_{i} / 2$ of [`crate::conv`] a zero contributes nothing, which is the standard
//! treatment of a dummy ("stolen") bit (IEEE 802.11-2020, Sec. 17.3.5.6).
//!
//! The patterns of the standards are provided as constructors:
//!
//! | Constructor | Source | Pattern (row $i$ = output $i$) |
//! |---|---|---|
//! | [`Puncturer::ieee80211_2_3`] | IEEE 802.11-2020, Sec. 17.3.5.6, Figure 17-9 | A: 1 1, B: 1 0 |
//! | [`Puncturer::ieee80211_3_4`] | IEEE 802.11-2020, Sec. 17.3.5.6, Figure 17-9 | A: 1 1 0, B: 1 0 1 |
//! | [`Puncturer::ieee80211_5_6`] | IEEE 802.11-2020, Sec. 19.3.11.6, Figure 19-11 | A: 1 1 0 1 0, B: 1 0 1 0 1 |
//! | [`Puncturer::dvbs_2_3`] | ETSI EN 300 421, Table 1 | Y: 1 1, X: 1 0 |
//! | [`Puncturer::dvbs_3_4`] | ETSI EN 300 421, Table 1 | Y: 1 1 0, X: 1 0 1 |
//! | [`Puncturer::dvbs_5_6`] | ETSI EN 300 421, Table 1 | Y: 1 1 0 1 0, X: 1 0 1 0 1 |
//! | [`Puncturer::dvbs_7_8`] | ETSI EN 300 421, Table 1 | Y: 1 1 1 1 0 1 0, X: 1 0 0 0 1 0 1 |
//! | [`Puncturer::turbo_1_2`] | Berrou, Glavieux & Thitimajshima 1993, Sec. II | systematic: 1 1, parity 1: 1 0, parity 2: 0 1 |
//!
//! all for the $(133, 171)_{8}$ code of constraint length 7 in the output order of
//! [`ConvolutionalCode::k7`](crate::conv::ConvolutionalCode::k7) ($133_{8}$ first), but
//! the last, for the three streams of a turbo code. IEEE 802.11 emits A from $133_{8}$
//! and B from $171_{8}$, so its rows are in the printed order; DVB-S emits X from
//! $G_{1} = 171_{8}$ and Y from $G_{2} = 133_{8}$ (EN 300 421, Sec. 4.4), so the rows of
//! its Table 1 are given Y first. The two standards then puncture the same codes: the
//! patterns of rates 2/3, 3/4 and 5/6 are defined once, by IEEE 802.11, and the DVB-S
//! constructors of these rates are aliases kept under the name of their standard.
//!
//! A rate matching that transmits some bits of a word, once or several times, is a
//! [`BitSelection`]: output bit $t$ carries bit $\pi(t)$ of the word, and the receiver
//! sums the LLRs of every bit over its copies (zero for a bit never sent). A puncturer
//! is one ([`Puncturer::selection`]), as are the circular buffers of 5G NR
//! ([`crate::nr`]).
//!
//! [`Punctured`] wraps a [`BlockCode`] whose coded bits are in step-major order (the
//! [`ConvolutionalCode`](crate::conv::ConvolutionalCode), tail included), and
//! [`PuncturedTurbo`] a [`TurboCode`] whose codeword is the
//! systematic, first-parity and second-parity streams (the pattern is applied across
//! the three streams at every information index; the tail bits are kept). The
//! rate matching of 3GPP (TS 36.212, Sec. 5.1.4: sub-block interleaving and a circular
//! buffer) is not a periodic pattern and is not provided.
//!
//! [`free_distance`] computes the free distance of a punctured code from its trellis, a
//! shortest path on the $2^{K-1} \times P$ states of the periodically time-varying
//! trellis; it reproduces the tables of Yasuda, Kashiki & Hirata (1984).
//!
//! # Examples
//! ```
//! use com_fec::block::BlockCode;
//! use com_fec::conv::ConvolutionalCode;
//! use com_fec::puncture::{Punctured, Puncturer, free_distance};
//! let code = Punctured::new(ConvolutionalCode::k7(100)?, Puncturer::ieee80211_3_4())?;
//! assert_eq!(code.n(), 35 * 4 + 2);                 // 4 bits kept of every 6, 2 of the last 2
//! let info: Vec<u8> = (0..100).map(|i| ((i * 5) % 7 < 3) as u8).collect();
//! let c = code.encode_block(&info)?;
//! let llr: Vec<f64> = c.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect(); // noiseless
//! assert_eq!(code.decode_block(&llr)?, info);
//! assert_eq!(free_distance(code.code().trellis(), code.puncturer())?, 5);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. B. Cain, G. C. Clark and J. M. Geist, "Punctured convolutional codes of rate
//!    $(n-1)/n$ and simplified maximum likelihood decoding," *IEEE Trans. Inf. Theory*,
//!    25(1), 97–100, 1979.
//! 2. Y. Yasuda, K. Kashiki and Y. Hirata, "High-rate punctured convolutional codes for
//!    soft decision Viterbi decoding," *IEEE Trans. Commun.*, 32(3), 315–319, 1984,
//!    Table I.
//! 3. IEEE Std 802.11-2020, Sec. 17.3.5.6 (Figure 17-9) and Sec. 19.3.11.6
//!    (Figure 19-11).
//! 4. ETSI EN 300 421 V1.1.2, *Digital Video Broadcasting (DVB); Framing structure,
//!    channel coding and modulation for 11/12 GHz satellite services*, 1997, Sec. 4.5,
//!    Table 1.
//! 5. C. Berrou, A. Glavieux and P. Thitimajshima, "Near Shannon limit error-correcting
//!    coding and decoding: turbo-codes," *Proc. IEEE ICC*, 1064–1070, 1993, Sec. II.

use crate::block::{BlockCode, SoftDecode, container_traits, expect_len};
use crate::conv::Trellis;
use crate::turbo::{TurboCode, TurboDecoder};
use com_core::{Error, Result, ensure, ensure_shape};

/// The bits of a word of `full_len` bits that a rate matching transmits: output bit $t$
/// carries bit `positions()[t]` of the word (a bit may be sent several times, or never).
///
/// # Examples
/// ```
/// use com_fec::puncture::BitSelection;
/// // a word of 4 bits, bit 3 dropped, bit 0 repeated
/// let sel = BitSelection::new(vec![0, 1, 2, 0], 4)?;
/// assert_eq!(sel.select(&[1u8, 0, 1, 1])?, [1, 0, 1, 1]);
/// let mut full = [0.0; 4];
/// sel.accumulate_into(&[2.0, -1.0, 0.5, 3.0], &mut full)?;
/// assert_eq!(full, [5.0, -1.0, 0.5, 0.0]);
/// # Ok::<(), com_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSelection {
    positions: Vec<usize>,
    full_len: usize,
}

impl BitSelection {
    /// The selection of the bits `positions` of a word of `full_len` bits.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a position beyond the word.
    pub fn new(positions: Vec<usize>, full_len: usize) -> Result<BitSelection> {
        ensure(positions.iter().all(|&p| p < full_len), || {
            format!("a selected bit lies beyond the word of {full_len} bits")
        })?;
        Ok(BitSelection { positions, full_len })
    }

    /// The number of transmitted bits.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether nothing is transmitted.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The length of the word.
    pub fn full_len(&self) -> usize {
        self.full_len
    }

    /// The bit of the word carried by every transmitted bit.
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// The transmitted bits of a word.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if `word` does not have [`BitSelection::full_len`] values.
    pub fn select<T: Copy>(&self, word: &[T]) -> Result<Vec<T>> {
        expect_len(word.len(), self.full_len, "bits of the word")?;
        Ok(self.positions.iter().map(|&p| word[p]).collect())
    }

    /// The LLRs of the bits of the word from the LLRs of the transmitted bits into
    /// `out`: the sum over the copies of every bit, zero for a bit never sent.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if `llr` does not have [`BitSelection::len`] values or
    /// `out` [`BitSelection::full_len`].
    pub fn accumulate_into(&self, llr: &[f64], out: &mut [f64]) -> Result<()> {
        expect_len(llr.len(), self.positions.len(), "LLRs")?;
        expect_len(out.len(), self.full_len, "LLRs of the word")?;
        out.fill(0.0);
        for (&p, &l) in self.positions.iter().zip(llr) {
            out[p] += l;
        }
        Ok(())
    }

    /// [`BitSelection::accumulate_into`] into a new vector.
    ///
    /// # Errors
    /// As [`BitSelection::accumulate_into`].
    pub fn word_llr(&self, llr: &[f64]) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.full_len];
        self.accumulate_into(llr, &mut out)?;
        Ok(out)
    }
}

/// A periodic puncturing pattern over the $n$ outputs of a code (see the module).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Puncturer {
    n_out: usize,
    period: usize,
    /// `keep[p * n_out + i]`: output `i` of the step of phase `p` is transmitted.
    keep: Vec<bool>,
}

impl Puncturer {
    /// The puncturer of a pattern given by rows: `pattern[i][p]` is 1 when the output
    /// `i` is kept at the steps of phase `p`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for no row, rows of different or zero length, an entry
    /// other than 0 or 1, or a pattern that keeps nothing.
    pub fn new(pattern: &[&[u8]]) -> Result<Puncturer> {
        let n_out = pattern.len();
        ensure(n_out >= 1, || "puncturing: the pattern needs at least one row".into())?;
        let period = pattern[0].len();
        ensure(period >= 1, || "puncturing: the pattern needs at least one column".into())?;
        ensure(pattern.iter().all(|r| r.len() == period), || "puncturing: the rows must have the same length".into())?;
        ensure(pattern.iter().flat_map(|r| r.iter()).all(|&b| b <= 1), || {
            "puncturing: the pattern entries must be 0 or 1".into()
        })?;
        let keep: Vec<bool> = (0..period).flat_map(|p| pattern.iter().map(move |r| r[p] == 1)).collect();
        ensure(keep.iter().any(|&k| k), || "puncturing: the pattern keeps no bit".into())?;
        Ok(Puncturer { n_out, period, keep })
    }

    /// The pattern that keeps every bit of a rate-$1/n$ code.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for $n = 0$.
    pub fn identity(n_out: usize) -> Result<Puncturer> {
        let row = [1u8];
        let rows: Vec<&[u8]> = vec![&row; n_out];
        Puncturer::new(&rows)
    }

    fn of(pattern: &[&[u8]]) -> Puncturer {
        Puncturer::new(pattern).unwrap_or_else(|_| unreachable!("a valid pattern"))
    }

    /// Rate 2/3 of IEEE 802.11-2020, Sec. 17.3.5.6, Figure 17-9: A 1 1, B 1 0.
    pub fn ieee80211_2_3() -> Puncturer {
        Puncturer::of(&[&[1, 1], &[1, 0]])
    }

    /// Rate 3/4 of IEEE 802.11-2020, Sec. 17.3.5.6, Figure 17-9: A 1 1 0, B 1 0 1.
    pub fn ieee80211_3_4() -> Puncturer {
        Puncturer::of(&[&[1, 1, 0], &[1, 0, 1]])
    }

    /// Rate 5/6 of IEEE 802.11-2020, Sec. 19.3.11.6, Figure 19-11: A 1 1 0 1 0,
    /// B 1 0 1 0 1.
    pub fn ieee80211_5_6() -> Puncturer {
        Puncturer::of(&[&[1, 1, 0, 1, 0], &[1, 0, 1, 0, 1]])
    }

    /// Rate 2/3 of ETSI EN 300 421, Table 1: X 1 0, Y 1 1 (rows Y, X: see the module),
    /// the pattern of [`Puncturer::ieee80211_2_3`] (an alias, under the name of the
    /// standard).
    pub fn dvbs_2_3() -> Puncturer {
        Puncturer::ieee80211_2_3()
    }

    /// Rate 3/4 of ETSI EN 300 421, Table 1: X 1 0 1, Y 1 1 0 (rows Y, X), the pattern
    /// of [`Puncturer::ieee80211_3_4`] (an alias).
    pub fn dvbs_3_4() -> Puncturer {
        Puncturer::ieee80211_3_4()
    }

    /// Rate 5/6 of ETSI EN 300 421, Table 1: X 1 0 1 0 1, Y 1 1 0 1 0 (rows Y, X), the
    /// pattern of [`Puncturer::ieee80211_5_6`] (an alias).
    pub fn dvbs_5_6() -> Puncturer {
        Puncturer::ieee80211_5_6()
    }

    /// Rate 7/8 of ETSI EN 300 421, Table 1: X 1 0 0 0 1 0 1, Y 1 1 1 1 0 1 0 (rows Y, X).
    pub fn dvbs_7_8() -> Puncturer {
        Puncturer::of(&[&[1, 1, 1, 1, 0, 1, 0], &[1, 0, 0, 0, 1, 0, 1]])
    }

    /// The rate-1/2 turbo code of Berrou, Glavieux & Thitimajshima (1993), Sec. II: the
    /// systematic bits and the parity bits of the two encoders alternately
    /// (systematic 1 1, parity 1: 1 0, parity 2: 0 1), for [`PuncturedTurbo`].
    pub fn turbo_1_2() -> Puncturer {
        Puncturer::of(&[&[1, 1], &[1, 0], &[0, 1]])
    }

    /// The pattern of a name: `ieee80211_2_3`, `ieee80211_3_4`, `ieee80211_5_6`,
    /// `dvbs_2_3`, `dvbs_3_4`, `dvbs_5_6`, `dvbs_7_8` (the convolutional codes) or
    /// `turbo_1_2` (the turbo codes), case-insensitive: the constructors of the same
    /// names.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for another name.
    ///
    /// # Examples
    /// ```
    /// use com_fec::puncture::Puncturer;
    /// assert_eq!(Puncturer::from_name("dvbs_7_8")?, Puncturer::dvbs_7_8());
    /// assert!(Puncturer::from_name("dvbs_9_10").is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn from_name(name: &str) -> Result<Puncturer> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "ieee80211_2_3" => Puncturer::ieee80211_2_3(),
            "ieee80211_3_4" => Puncturer::ieee80211_3_4(),
            "ieee80211_5_6" => Puncturer::ieee80211_5_6(),
            "dvbs_2_3" => Puncturer::dvbs_2_3(),
            "dvbs_3_4" => Puncturer::dvbs_3_4(),
            "dvbs_5_6" => Puncturer::dvbs_5_6(),
            "dvbs_7_8" => Puncturer::dvbs_7_8(),
            "turbo_1_2" => Puncturer::turbo_1_2(),
            other => {
                return Err(Error::Config(format!(
                    "unknown puncturing pattern {other:?} (ieee80211_2_3, ieee80211_3_4, ieee80211_5_6, \
                     dvbs_2_3, dvbs_3_4, dvbs_5_6, dvbs_7_8, turbo_1_2)"
                )));
            }
        })
    }

    /// Outputs per step $n$ (the rows of the pattern).
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// The period $P$ (the columns of the pattern).
    pub fn period(&self) -> usize {
        self.period
    }

    /// The bits kept per period $w$.
    pub fn kept_per_period(&self) -> usize {
        self.keep.iter().filter(|&&k| k).count()
    }

    /// Whether output `i` of step `step` is transmitted.
    pub fn keeps(&self, step: usize, i: usize) -> bool {
        self.keep[(step % self.period) * self.n_out + i]
    }

    /// The number of transmitted bits of `n_coded` coded bits.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if `n_coded` is not a multiple of $n$.
    pub fn punctured_len(&self, n_coded: usize) -> Result<usize> {
        self.check_coded(n_coded)?;
        Ok((0..n_coded).filter(|&j| self.keep[j % (self.period * self.n_out)]).count())
    }

    /// The transmitted bits of `n_coded` coded bits in step-major order, as a
    /// [`BitSelection`].
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if `n_coded` is not a multiple of $n$.
    pub fn selection(&self, n_coded: usize) -> Result<BitSelection> {
        self.check_coded(n_coded)?;
        let cycle = self.period * self.n_out;
        Ok(BitSelection { positions: (0..n_coded).filter(|&j| self.keep[j % cycle]).collect(), full_len: n_coded })
    }

    fn check_coded(&self, n_coded: usize) -> Result<()> {
        if n_coded % self.n_out == 0 {
            Ok(())
        } else {
            Err(Error::Shape(format!("{n_coded} coded bits is not a multiple of the {} outputs", self.n_out)))
        }
    }

    /// The transmitted bits of coded bits in step-major order (the $n$ outputs of step
    /// 0, then of step 1, …).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if the number of bits is not a multiple of $n$.
    pub fn puncture(&self, bits: &[u8]) -> Result<Vec<u8>> {
        self.selection(bits.len())?.select(bits)
    }

    /// The LLRs of the `n_coded` coded bits from the LLRs of the transmitted ones: a
    /// zero at every punctured position.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if `n_coded` is not a multiple of $n$ or `llr` does
    /// not have [`Puncturer::punctured_len`] values.
    pub fn depuncture(&self, llr: &[f64], n_coded: usize) -> Result<Vec<f64>> {
        self.selection(n_coded)?.word_llr(llr)
    }
}

/// A block code whose coded bits (in step-major order) are punctured: the
/// [`ConvolutionalCode`](crate::conv::ConvolutionalCode).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Punctured<C> {
    /// The mother code.
    code: C,
    /// The pattern.
    puncturer: Puncturer,
    /// The transmitted coded bits of a block.
    selection: BitSelection,
}

impl<C: BlockCode> Punctured<C> {
    /// The code `code` punctured by `puncturer`.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if the coded length of the mother code is not a
    /// multiple of the outputs of the pattern.
    pub fn new(code: C, puncturer: Puncturer) -> Result<Punctured<C>> {
        let selection = puncturer.selection(code.n())?;
        Ok(Punctured { code, puncturer, selection })
    }

    /// The mother code (its settings are those it had when punctured).
    pub fn code(&self) -> &C {
        &self.code
    }

    /// The pattern.
    pub fn puncturer(&self) -> &Puncturer {
        &self.puncturer
    }

    /// The transmitted coded bits of a block.
    pub fn selection(&self) -> &BitSelection {
        &self.selection
    }
}

impl<C: BlockCode> BlockCode for Punctured<C> {
    fn k(&self) -> usize {
        self.code.k()
    }

    fn n(&self) -> usize {
        self.selection.len()
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        self.selection.select(&self.code.encode_block(info)?)
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        self.code.decode_block(&self.selection.word_llr(llr)?)
    }
}

impl<C: SoftDecode> SoftDecode for Punctured<C> {
    /// The soft output of the mother code, a zero LLR at every punctured position.
    fn app_into(&self, llr: &[f64], apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize> {
        self.code.app_into(&self.selection.word_llr(llr)?, apriori, out)
    }
}

/// A turbo code whose systematic and parity streams are punctured by a pattern of three
/// rows (systematic, parity 1, parity 2) applied at every information index; the tail
/// bits are kept.
#[derive(Clone, Debug, PartialEq)]
pub struct PuncturedTurbo {
    /// The mother code.
    code: TurboCode,
    /// The pattern (three rows).
    puncturer: Puncturer,
    /// The transmitted bits of the codeword of the mother code (systematic, parity 1,
    /// parity 2, tail).
    selection: BitSelection,
}

impl PuncturedTurbo {
    /// The turbo code `code` punctured by the three-row pattern `puncturer`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if the pattern does not have three rows.
    pub fn new(code: TurboCode, puncturer: Puncturer) -> Result<PuncturedTurbo> {
        ensure(puncturer.n_out() == 3, || {
            format!("puncturing a turbo code needs a pattern of 3 rows, got {}", puncturer.n_out())
        })?;
        let k = code.k();
        // the pattern runs over the steps (systematic, parity 1, parity 2) of every
        // index: step-major bit j is bit (j mod 3) k + j / 3 of the three streams
        let mut positions: Vec<usize> =
            puncturer.selection(3 * k)?.positions().iter().map(|&j| (j % 3) * k + j / 3).collect();
        positions.extend(3 * k..code.n());
        let selection = BitSelection { positions, full_len: code.n() };
        Ok(PuncturedTurbo { code, puncturer, selection })
    }

    /// The mother code.
    pub fn code(&self) -> &TurboCode {
        &self.code
    }

    /// The pattern.
    pub fn puncturer(&self) -> &Puncturer {
        &self.puncturer
    }

    /// The settings of the decoder of the mother code, which can change without
    /// changing the code.
    pub fn decoder_mut(&mut self) -> &mut TurboDecoder {
        &mut self.code.decoder
    }

    /// The transmitted bits of the codeword of the mother code.
    pub fn selection(&self) -> &BitSelection {
        &self.selection
    }
}

impl BlockCode for PuncturedTurbo {
    fn k(&self) -> usize {
        self.code.k()
    }

    fn n(&self) -> usize {
        self.selection.len()
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        self.selection.select(&self.code.encode_block(info)?)
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        self.code.decode_block(&self.selection.word_llr(llr)?)
    }
}

impl SoftDecode for PuncturedTurbo {
    /// The soft output of the turbo code, a zero LLR at every punctured position.
    fn app_into(&self, llr: &[f64], apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize> {
        self.code.app_into(&self.selection.word_llr(llr)?, apriori, out)
    }
}

container_traits!(Punctured<C>, C: BlockCode);
container_traits!(PuncturedTurbo);

/// The free distance of the code of `trellis` punctured by `puncturer`: the smallest
/// weight of a transmitted path that leaves the zero state and returns to it, over
/// every phase of the pattern. A shortest path over the $2^{K-1} P$ (state, phase)
/// nodes of the time-varying trellis (Bellman–Ford, the weights being the numbers of
/// kept ones of the branches).
///
/// # Errors
/// [`com_core::Error::Shape`] if the pattern does not have the $n$ rows of the trellis.
pub fn free_distance(trellis: &Trellis, puncturer: &Puncturer) -> Result<u32> {
    let n = trellis.n_out;
    ensure_shape(puncturer.n_out() == n, || {
        format!("the pattern has {} rows for a code of {n} outputs", puncturer.n_out())
    })?;
    let (ns, p) = (trellis.n_states(), puncturer.period());
    let weight = |o: u32, phase: usize| {
        (0..n).filter(|&i| puncturer.keeps(phase, i) && (o >> (n - 1 - i)) & 1 == 1).count() as u32
    };
    let mut best = u32::MAX;
    for start in 0..p {
        // the diverging branch: input 1 from state 0 at phase `start`
        let mut dist = vec![u32::MAX; ns * p];
        let s1 = trellis.next[0][1];
        let w1 = weight(trellis.out[0][1], start);
        let ph1 = (start + 1) % p;
        dist[s1 * p + ph1] = dist[s1 * p + ph1].min(w1);
        if s1 == 0 {
            best = best.min(w1);
        }
        // relax until stable (every shortest path is simple: at most ns·p steps)
        for _ in 0..ns * p {
            let mut changed = false;
            for s in 0..ns {
                for ph in 0..p {
                    let d = dist[s * p + ph];
                    if d == u32::MAX || s == 0 {
                        continue;
                    }
                    for u in 0..2 {
                        let s2 = trellis.next[s][u];
                        let d2 = d + weight(trellis.out[s][u], ph);
                        let ph2 = (ph + 1) % p;
                        if s2 == 0 {
                            best = best.min(d2);
                        } else if d2 < dist[s2 * p + ph2] {
                            dist[s2 * p + ph2] = d2;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::testing::awgn_llr;
    use crate::bounds::bpsk_noise_variance;
    use crate::conv::ConvolutionalCode;
    use com_core::rng::Rng;

    #[test]
    fn puncture_and_depuncture_restore_the_positions() {
        let p = Puncturer::ieee80211_3_4();
        assert_eq!((p.n_out(), p.period(), p.kept_per_period()), (2, 3, 4));
        let bits: Vec<u8> = (0..12).map(|i| (i % 3 == 0) as u8).collect();
        // steps (A, B): kept A0 B0 A1 B2 A3 B3 A4 B5
        let punctured = p.puncture(&bits).unwrap();
        assert_eq!(punctured, [bits[0], bits[1], bits[2], bits[5], bits[6], bits[7], bits[8], bits[11]]);
        let llr: Vec<f64> = punctured
            .iter()
            .enumerate()
            .map(|(i, &b)| if b == 1 { -(i as f64 + 1.0) } else { i as f64 + 1.0 })
            .collect();
        let full = p.depuncture(&llr, 12).unwrap();
        assert_eq!(full.len(), 12);
        for (j, v) in full.iter().enumerate() {
            if p.keeps(j / 2, j % 2) {
                assert_eq!(u8::from(*v < 0.0), bits[j]);
            } else {
                assert_eq!(*v, 0.0, "a punctured position carries a zero LLR");
            }
        }
        assert!(p.puncture(&bits[..5]).is_err());
        assert!(p.depuncture(&llr[..7], 12).is_err());
        assert!(Puncturer::new(&[&[1, 1], &[1]]).is_err());
        assert!(Puncturer::new(&[&[0, 0], &[0, 0]]).is_err());
        assert!(Puncturer::new(&[&[2]]).is_err());
        assert!(Puncturer::new(&[]).is_err());
        // a partial period at the end
        assert_eq!(p.punctured_len(8).unwrap(), 6);
    }

    /// The streams of a turbo codeword in step-major order, (systematic, parity 1,
    /// parity 2) per index, as `PuncturedTurbo` wrote them before its selection.
    fn former_streams_to_steps<T: Copy>(k: usize, v: &[T]) -> Vec<T> {
        (0..k).flat_map(|i| [v[i], v[k + i], v[2 * k + i]]).collect()
    }

    /// The depuncturing written position by position before the selections.
    fn former_depuncture(p: &Puncturer, llr: &[f64], n_coded: usize) -> Vec<f64> {
        let cycle = p.period * p.n_out;
        let mut src = llr.iter();
        (0..n_coded).map(|j| if p.keep[j % cycle] { src.next().copied().unwrap_or(0.0) } else { 0.0 }).collect()
    }

    #[test]
    fn the_selections_give_the_former_punctured_words_and_llrs() {
        let mut rng = Rng::new(12);
        for p in [Puncturer::ieee80211_3_4(), Puncturer::dvbs_7_8(), Puncturer::turbo_1_2()] {
            for n_coded in [0, p.n_out(), 7 * p.n_out(), 120] {
                if n_coded % p.n_out() != 0 {
                    continue;
                }
                let bits: Vec<u8> = (0..n_coded).map(|_| rng.bit()).collect();
                let cycle = p.period * p.n_out;
                let former: Vec<u8> =
                    bits.iter().enumerate().filter(|(j, _)| p.keep[j % cycle]).map(|(_, &b)| b).collect();
                assert_eq!(p.puncture(&bits).unwrap(), former);
                let llr: Vec<f64> = (0..former.len()).map(|_| rng.gaussian()).collect();
                let (new, old) = (p.depuncture(&llr, n_coded).unwrap(), former_depuncture(&p, &llr, n_coded));
                assert!(new.len() == old.len() && new.iter().zip(&old).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
        // the punctured turbo code: the former decoder input, the streams of the
        // depunctured body and the tail
        let turbo = PuncturedTurbo::new(TurboCode::lte(40).unwrap(), Puncturer::turbo_1_2()).unwrap();
        let llr: Vec<f64> = (0..turbo.n()).map(|_| rng.gaussian()).collect();
        let body = former_depuncture(&Puncturer::turbo_1_2(), &llr[..80], 120);
        let mut former: Vec<f64> =
            (0..3).flat_map(|s| (0..40).map(move |i| (s, i))).map(|(s, i)| body[3 * i + s]).collect();
        former.extend_from_slice(&llr[80..]);
        let new = turbo.selection().word_llr(&llr).unwrap();
        assert!(new.iter().zip(&former).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(BitSelection::new(vec![0, 4], 4).is_err());
    }

    #[test]
    fn the_soft_output_passes_through_the_puncturing() {
        let mut rng = Rng::new(13);
        let code = Punctured::new(ConvolutionalCode::k7(50).unwrap(), Puncturer::ieee80211_2_3()).unwrap();
        let llr: Vec<f64> = (0..code.n()).map(|_| rng.gaussian()).collect();
        let app = code.app(&llr, None).unwrap();
        let mother = code.code().app(&code.puncturer().depuncture(&llr, code.code().n()).unwrap(), None).unwrap();
        assert!(app.iter().zip(&mother).all(|(a, b)| a.to_bits() == b.to_bits()));
        let mut turbo = PuncturedTurbo::new(TurboCode::lte(40).unwrap(), Puncturer::turbo_1_2()).unwrap();
        turbo.decoder_mut().iterations = 3;
        let llr: Vec<f64> = (0..turbo.n()).map(|_| rng.gaussian()).collect();
        assert_eq!(turbo.code().decoder.iterations, 3);
        assert_eq!(
            com_core::signal::hard_decisions(&turbo.app(&llr, None).unwrap()),
            turbo.decode_block(&llr).unwrap()
        );
    }

    #[test]
    fn punctured_codes_decode_without_error_at_high_snr() {
        let mut rng = Rng::new(7);
        for puncturer in [
            Puncturer::ieee80211_2_3(),
            Puncturer::ieee80211_3_4(),
            Puncturer::ieee80211_5_6(),
            Puncturer::dvbs_2_3(),
            Puncturer::dvbs_3_4(),
            Puncturer::dvbs_5_6(),
            Puncturer::dvbs_7_8(),
        ] {
            let code = Punctured::new(ConvolutionalCode::k7(300).unwrap(), puncturer).unwrap();
            let info: Vec<u8> = (0..300).map(|_| rng.bit()).collect();
            let c = code.encode_block(&info).unwrap();
            assert_eq!(c.len(), code.n());
            // Eb/N0 = 10 dB
            let y = awgn_llr(&c, bpsk_noise_variance(10.0, code.code_rate()), &mut rng);
            assert_eq!(code.decode_block(&y).unwrap(), info, "{:?}", code.puncturer());
            // the hard-decision decoder also treats the dummy bits as zero metrics
            let mut hard = code.code().clone();
            hard.hard_decision = true;
            let hard_code = Punctured::new(hard, code.puncturer().clone()).unwrap();
            assert_eq!(hard_code.decode_block(&y).unwrap(), info);
        }
        let turbo = PuncturedTurbo::new(TurboCode::lte(40).unwrap(), Puncturer::turbo_1_2()).unwrap();
        assert_eq!(turbo.n(), 2 * 40 + 12);
        assert!((turbo.code_rate() - 40.0 / 92.0).abs() < 1e-12);
        let info: Vec<u8> = (0..40).map(|_| rng.bit()).collect();
        let c = turbo.encode_block(&info).unwrap();
        let mother = turbo.code().encode_block(&info).unwrap();
        assert_eq!(
            &c[..80],
            &former_streams_to_steps(40, &mother[..120])
                .iter()
                .enumerate()
                .filter(|(j, _)| Puncturer::turbo_1_2().keep[j % 6])
                .map(|(_, &b)| b)
                .collect::<Vec<u8>>()[..]
        );
        assert_eq!(c[80..], mother[120..]);
        let y = awgn_llr(&c, bpsk_noise_variance(6.0, turbo.code_rate()), &mut rng);
        assert_eq!(turbo.decode_block(&y).unwrap(), info);
        assert!(PuncturedTurbo::new(TurboCode::lte(40).unwrap(), Puncturer::ieee80211_2_3()).is_err());
    }

    #[test]
    fn the_rate_3_4_code_lies_between_rate_1_2_and_uncoded_bpsk() {
        // Eb/N0 = 4 dB: uncoded BPSK has P_b = Q(√(2·2.51)) = 0.0125; soft-decision Viterbi
        // decoding of the K = 7 rate-1/2 code is below 1e-4 and of the rate-3/4 code
        // about 5e-4
        let ebn0 = 4.0;
        let (k, blocks) = (2000, 25);
        let half = ConvolutionalCode::k7(k).unwrap();
        let three_quarters = Punctured::new(ConvolutionalCode::k7(k).unwrap(), Puncturer::ieee80211_3_4()).unwrap();
        let mut rng = Rng::new(34);
        let (mut e_half, mut e_34, mut e_uncoded) = (0usize, 0usize, 0usize);
        for _ in 0..blocks {
            let info: Vec<u8> = (0..k).map(|_| rng.bit()).collect();
            let errors = |decoded: &[u8]| decoded.iter().zip(&info).filter(|(a, b)| a != b).count();
            let c = half.encode_block(&info).unwrap();
            let y = awgn_llr(&c, bpsk_noise_variance(ebn0, half.code_rate()), &mut rng);
            e_half += errors(&half.decode_block(&y).unwrap());
            let c = three_quarters.encode_block(&info).unwrap();
            let y = awgn_llr(&c, bpsk_noise_variance(ebn0, three_quarters.code_rate()), &mut rng);
            e_34 += errors(&three_quarters.decode_block(&y).unwrap());
            let y = awgn_llr(&info, bpsk_noise_variance(ebn0, 1.0), &mut rng);
            e_uncoded += errors(&com_core::signal::hard_decisions(&y));
        }
        // 50 000 bits: about 625 uncoded errors, 25 at rate 3/4 and a few at rate 1/2
        // (this seed: 644, 24 and 0); the counts are Poisson-like with standard
        // deviations of 25, 5 and 2, so the factors of two below hold with margins of
        // many standard deviations
        assert!(e_half < e_34 && e_34 < e_uncoded, "rate 1/2: {e_half}, rate 3/4: {e_34}, uncoded: {e_uncoded}");
        assert!(
            e_34 * 2 < e_uncoded && e_half * 2 < e_34,
            "rate 1/2: {e_half}, rate 3/4: {e_34}, uncoded: {e_uncoded}"
        );
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    /// Y. Yasuda, K. Kashiki and Y. Hirata, "High-rate punctured convolutional codes for
    /// soft decision Viterbi decoding," IEEE Trans. Commun. 32(3), 1984, Table I: the
    /// free distances of the codes punctured from the K = 7 (133, 171) code are 6 at
    /// rate 2/3, 5 at 3/4, 4 at 5/6 and 3 at 7/8 (the mother code: 10). The patterns of
    /// DVB-S (ETSI EN 300 421, Table 1) and of 802.11 are those of the paper; the swapped
    /// rows at rate 2/3 (keeping both bits of the 171 output instead of the 133 one)
    /// give 5, not 6, which checks the output order of the constructors.
    #[test]
    fn yasuda_1984_table_i_free_distances_of_the_k7_punctured_codes() {
        let t = Trellis::feedforward(7, &[0o133, 0o171]).unwrap();
        assert_eq!(free_distance(&t, &Puncturer::identity(2).unwrap()).unwrap(), 10);
        assert_eq!(free_distance(&t, &Puncturer::dvbs_2_3()).unwrap(), 6);
        assert_eq!(free_distance(&t, &Puncturer::dvbs_3_4()).unwrap(), 5);
        assert_eq!(free_distance(&t, &Puncturer::dvbs_5_6()).unwrap(), 4);
        assert_eq!(free_distance(&t, &Puncturer::dvbs_7_8()).unwrap(), 3);
        assert_eq!(free_distance(&t, &Puncturer::ieee80211_2_3()).unwrap(), 6);
        assert_eq!(free_distance(&t, &Puncturer::ieee80211_3_4()).unwrap(), 5);
        assert_eq!(free_distance(&t, &Puncturer::ieee80211_5_6()).unwrap(), 4);
        assert!(free_distance(&t, &Puncturer::turbo_1_2()).is_err());
        assert_eq!(free_distance(&t, &Puncturer::new(&[&[1, 0], &[1, 1]]).unwrap()).unwrap(), 5);
    }

    /// ETSI EN 300 421, Table 1, read with the rows Y (from $G_{2} = 133_{8}$) then X
    /// (from $G_{1} = 171_{8}$): the patterns of rates 2/3, 3/4 and 5/6 are those of IEEE
    /// 802.11-2020, Figures 17-9 and 19-11, which the DVB-S constructors alias.
    #[test]
    fn etsi_en_300_421_table_1_is_the_80211_pattern_at_the_shared_rates() {
        // (X, Y) as printed in Table 1
        let table: [(&[u8], &[u8], Puncturer); 3] = [
            (&[1, 0], &[1, 1], Puncturer::dvbs_2_3()),
            (&[1, 0, 1], &[1, 1, 0], Puncturer::dvbs_3_4()),
            (&[1, 0, 1, 0, 1], &[1, 1, 0, 1, 0], Puncturer::dvbs_5_6()),
        ];
        for (x, y, alias) in table {
            assert_eq!(Puncturer::new(&[y, x]).unwrap(), alias);
        }
    }

    /// Yasuda, Kashiki & Hirata (1984), Tables I and II in full: the 84 deleting maps of
    /// the codes of rates 2/3 to 13/14 punctured from the rate-1/2 codes of maximal free
    /// distance with $\nu = 2$ to 8 (generators (5, 7) to (561, 753), the map of the first
    /// generator first) have the free distances of Table II, and the mother codes 5, 6, 7,
    /// 8, 10, 10 and 12.
    #[test]
    fn yasuda_1984_tables_i_ii_every_punctured_code() {
        let generators =
            [(0o5, 0o7), (0o15, 0o17), (0o23, 0o35), (0o53, 0o75), (0o133, 0o171), (0o247, 0o371), (0o561, 0o753)];
        let mother: [u32; 7] = [5, 6, 7, 8, 10, 10, 12];
        // (map of the first generator, map of the second, d), nu = 2 to 8
        let table: [[(&str, &str, u32); 7]; 12] = [
            // rate 2/3, nu = 2 to 8
            [
                ("10", "11", 3),
                ("11", "10", 4),
                ("11", "10", 4),
                ("10", "11", 6),
                ("11", "10", 6),
                ("10", "11", 7),
                ("11", "10", 7),
            ],
            // rate 3/4, nu = 2 to 8
            [
                ("101", "110", 3),
                ("110", "101", 4),
                ("101", "110", 3),
                ("100", "111", 4),
                ("110", "101", 5),
                ("110", "101", 6),
                ("111", "100", 6),
            ],
            // rate 4/5, nu = 2 to 8
            [
                ("1011", "1100", 2),
                ("1011", "1100", 3),
                ("1010", "1101", 3),
                ("1000", "1111", 4),
                ("1111", "1000", 4),
                ("1010", "1101", 5),
                ("1101", "1010", 5),
            ],
            // rate 5/6, nu = 2 to 8
            [
                ("10111", "11000", 2),
                ("10100", "11011", 3),
                ("10111", "11000", 3),
                ("10000", "11111", 4),
                ("11010", "10101", 4),
                ("11100", "10011", 4),
                ("10110", "11001", 5),
            ],
            // rate 6/7, nu = 2 to 8
            [
                ("101111", "110000", 2),
                ("100011", "111100", 2),
                ("101010", "110101", 3),
                ("110110", "101001", 3),
                ("111010", "100101", 3),
                ("101001", "110110", 4),
                ("110110", "101001", 4),
            ],
            // rate 7/8, nu = 2 to 8
            [
                ("1011111", "1100000", 2),
                ("1000010", "1111101", 2),
                ("1010011", "1101100", 3),
                ("1011101", "1100010", 3),
                ("1111010", "1000101", 3),
                ("1010100", "1101011", 4),
                ("1101011", "1010100", 4),
            ],
            // rate 8/9, nu = 2 to 8
            [
                ("10111111", "11000000", 2),
                ("10000011", "11111100", 2),
                ("10100011", "11011100", 3),
                ("11100010", "10011101", 3),
                ("11110100", "10001011", 3),
                ("10110110", "11001001", 3),
                ("11100000", "10011111", 4),
            ],
            // rate 9/10, nu = 2 to 8
            [
                ("101111111", "110000000", 2),
                ("101000000", "110111111", 2),
                ("111110011", "100001100", 2),
                ("100001111", "111110000", 3),
                ("111101110", "100010001", 3),
                ("101100110", "110011001", 4),
                ("111000101", "100111010", 4),
            ],
            // rate 10/11, nu = 2 to 8
            [
                ("1011111111", "1100000000", 2),
                ("1000000011", "1111111100", 2),
                ("1000000101", "1111111010", 2),
                ("1001110100", "1110001011", 3),
                ("1110110111", "1001001000", 3),
                ("1001000011", "1110111100", 3),
                ("1000101100", "1111010011", 4),
            ],
            // rate 11/12, nu = 2 to 8
            [
                ("10111111111", "11000000000", 2),
                ("10000000010", "11111111101", 2),
                ("10101101101", "11010010010", 2),
                ("10001110100", "11110001011", 3),
                ("11110111110", "10001000001", 3),
                ("10110000110", "11001111001", 3),
                ("11000010001", "10111101110", 4),
            ],
            // rate 12/13, nu = 2 to 8
            [
                ("101111111111", "110000000000", 2),
                ("100000000011", "111111111100", 2),
                ("101101111011", "110010000100", 2),
                ("110100110110", "101011001001", 3),
                ("111111110101", "100000001010", 3),
                ("100100001100", "111011110011", 3),
                ("110000011010", "101111100101", 4),
            ],
            // rate 13/14, nu = 2 to 8
            [
                ("1011111111111", "1100000000000", 2),
                ("1010000000000", "1101111111111", 2),
                ("1110110110111", "1001001001000", 2),
                ("1100011000100", "1011100111011", 3),
                ("1101000001111", "1010111110000", 3),
                ("1010100100000", "1101011011111", 3),
                ("1100000100001", "1011111011110", 4),
            ],
        ];
        let bits = |s: &str| s.bytes().map(|b| b - b'0').collect::<Vec<u8>>();
        for (j, &(g1, g2)) in generators.iter().enumerate() {
            let t = Trellis::feedforward(j as u32 + 3, &[g1, g2]).unwrap();
            assert_eq!(free_distance(&t, &Puncturer::identity(2).unwrap()).unwrap(), mother[j], "nu = {}", j + 2);
            for row in &table {
                let (a, b, d) = row[j];
                let (a, b) = (bits(a), bits(b));
                let p = Puncturer::new(&[&a, &b]).unwrap();
                assert_eq!(free_distance(&t, &p).unwrap(), d, "nu = {}, rate {}/{}", j + 2, a.len(), a.len() + 1);
            }
        }
    }
}
