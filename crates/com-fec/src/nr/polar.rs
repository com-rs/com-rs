//! The polar code of the control and broadcast channels of 5G NR (TS 38.212, Sec.
//! 5.3.1, 5.4.1, 6.3.1, 7.1, 7.3).
//!
//! # Model
//!
//! $A$ payload bits receive a CRC ($L$ bits) and become $K = A + L$ bits, interleaved by
//! the pattern of Table 5.3.1.1-1 on the downlink ($I_{\mathrm{IL}} = 1$). The mother
//! code length is $N = 2^{n}$ with $n = \max(\min(n_{1}, n_{2}, n_{\max}), 5)$,
//! $n_{1} = \lceil \log_{2} E \rceil - 1$ when $E \le \frac{9}{8} 2^{\lceil \log_{2} E
//! \rceil - 1}$ and $K/E < 9/16$ (else $\lceil \log_{2} E \rceil$), $n_{2} = \lceil
//! \log_{2}(8K) \rceil$ (Sec. 5.3.1). The sub-block interleaver of Sec. 5.4.1.1 cuts the
//! codeword into 32 blocks permuted by Table 5.4.1.1-1, $J(n) = P(\lfloor 32 n / N
//! \rfloor) N/32 + n \bmod N/32$; the $E$ transmitted bits are, when $E < N$, the last
//! $E$ (puncturing, $K/E \le 7/16$) or the first $E$ (shortening) of the interleaved
//! word, and when $E \ge N$ the word repeated. The frozen set holds the positions
//! $J(n)$ that are not transmitted — and, with puncturing, the first $\lceil 3N/4 - E/2
//! \rceil$ (or $\lceil 9N/16 - E/4 \rceil$) positions — and the least reliable of the
//! others in the order of the polar sequence of Table 5.3.1.2-1; the $K + n_{\mathrm{PC}}$
//! most reliable carry the bits and the parity-check bits ([`crate::polar::PolarCode::with_positions`]).
//! On the uplink, the triangular interleaver of Sec. 5.4.1.3 ($I_{\mathrm{BIL}} = 1$)
//! writes the $E$ bits row by row into rows of $T, T - 1, \dots$ cells, $T(T+1)/2 \ge E$,
//! and reads them column by column.
//!
//! - [`NrPolar::uplink`]: the uplink control information (Sec. 6.3.1.2–6.3.1.4): CRC6
//!   and 3 parity-check bits for $12 \le A \le 19$ (one of minimum row weight when
//!   $E - K + 3 > 192$), CRC11 for $A \ge 20$, $n_{\max} = 10$, $I_{\mathrm{BIL}} = 1$;
//!   one code block ($A < 360$ or $E < 1088$, and $A < 1013$);
//! - [`NrPolar::downlink`]: the downlink control information and the broadcast channel
//!   (Sec. 7.3, 7.1): CRC24C, $I_{\mathrm{IL}} = 1$, $n_{\max} = 9$. The CRC is that of
//!   the payload: the 24 ones prepended to a DCI and its masking by an RNTI are left to
//!   the caller.
//!
//! The decoder undoes the interleavers and the rate matching on the LLRs (summing the
//! repetitions; zero on the punctured bits, a large positive value on the shortened
//! ones, which are zeros), runs the SC list decoder of [`crate::polar`] and keeps the
//! first path whose deinterleaved bits pass the CRC.
//!
//! # Examples
//! ```
//! use com_fec::block::BlockCode;
//! use com_fec::nr::NrPolar;
//! let code = NrPolar::uplink(30, 100)?.with_list(8);      // A = 30 bits, E = 100
//! assert_eq!(code.mother_length(), 128);
//! let info: Vec<u8> = (0..30).map(|i| (i % 4 == 1) as u8).collect();
//! let c = code.encode_block(&info)?;
//! let llr: Vec<f64> = c.iter().map(|b| if *b == 0 { 3.0 } else { -3.0 }).collect();
//! assert_eq!(code.decode_block(&llr)?, info);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 38.212 V18.5.0, *NR; Multiplexing and channel coding*, Sec. 5.3.1, 5.4.1,
//!    6.3.1, 7.1, 7.3 (ETSI TS 138 212, 2025).
//! 2. V. Bioglio, C. Condo, I. Land, "Design of polar codes in 5G New Radio," *IEEE
//!    Commun. Surveys Tuts.* 23(1), 29–40, 2021.

use super::tables::{POLAR_INTERLEAVER, POLAR_SEQUENCE, SUBBLOCK_PATTERN};
use crate::block::{BlockCode, container_traits, expect_len};
use crate::crc::Crc;
use crate::polar::PolarCode;
use com_core::{Result, ensure};

/// The LLR of a shortened bit (known to be 0).
const KNOWN: f64 = 30.0;

/// The polar code of 5G NR for one code block (see the module).
#[derive(Clone, Debug)]
pub struct NrPolar {
    a: usize,
    e: usize,
    crc: Crc,
    /// $\Pi$: interleaved bit $k$ is bit $\Pi(k)$ of the payload and CRC (identity
    /// without input interleaving).
    il: Vec<usize>,
    /// The bit of the mother codeword carried by every output bit.
    map: Vec<usize>,
    /// The mother codeword bits that shortening sets to zero.
    shortened: Vec<usize>,
    /// The mother code, its positions and its list size.
    pub code: PolarCode,
}

/// $\lceil \log_{2} x \rceil$.
fn ceil_log2(x: usize) -> u32 {
    x.next_power_of_two().trailing_zeros()
}

impl NrPolar {
    /// The code of the uplink control information: `a` bits, `e` transmitted bits.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for $A < 12$ (the small block codes of Sec. 5.3.3),
    /// a payload that needs two code blocks, or $E$ too small for the code.
    pub fn uplink(a: usize, e: usize) -> Result<NrPolar> {
        ensure(a >= 12, || format!("the uplink polar code takes A ≥ 12 bits, got {a}"))?;
        ensure(a < 1013 && !(a >= 360 && e >= 1088), || {
            format!("A = {a}, E = {e} needs two code blocks (Sec. 6.3.1.2.1), not supported")
        })?;
        let (crc, n_pc) = if a <= 19 { (Crc::crc6(), 3) } else { (Crc::crc11(), 0) };
        let k = a + crc.width() as usize;
        let n_pc_wm = usize::from(n_pc > 0 && e + n_pc > k + 192);
        NrPolar::build(a, e, crc, n_pc, n_pc_wm, false, true, 10)
    }

    /// The code of the downlink control information and of the broadcast channel: `a`
    /// bits, `e` transmitted bits.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for $A = 0$, $A + 24 > 164$ (the input interleaver
    /// of Table 5.3.1.1-1), or $E$ too small for the code.
    pub fn downlink(a: usize, e: usize) -> Result<NrPolar> {
        ensure(a >= 1 && a + 24 <= 164, || format!("the downlink polar code takes 1 ≤ A ≤ 140 bits, got {a}"))?;
        NrPolar::build(a, e, Crc::crc24c(), 0, 0, true, false, 9)
    }

    #[allow(clippy::too_many_arguments, reason = "the parameters of Sec. 5.3.1 and 5.4.1")]
    fn build(
        a: usize,
        e: usize,
        crc: Crc,
        n_pc: usize,
        n_pc_wm: usize,
        i_il: bool,
        i_bil: bool,
        n_max: u32,
    ) -> Result<NrPolar> {
        let k = a + crc.width() as usize;
        ensure(e >= 1, || "E must be positive".to_string())?;
        let ce = ceil_log2(e);
        let n1 = if ce >= 1 && 8 * e <= 9 * (1 << (ce - 1)) && 16 * k < 9 * e { ce - 1 } else { ce };
        let n2 = ceil_log2(8 * k);
        let n_exp = n1.min(n2).min(n_max).max(5);
        let n = 1usize << n_exp;
        ensure(k + n_pc <= e && k + n_pc <= n, || format!("K + n_PC = {} exceeds E = {e} or N = {n}", k + n_pc))?;
        // the sub-block interleaver J(n)
        let b = n / 32;
        let j: Vec<usize> = (0..n).map(|m| SUBBLOCK_PATTERN[32 * m / n] as usize * b + m % b).collect();
        // the frozen positions of rate matching, Sec. 5.4.1.1
        let mut frozen = vec![false; n];
        let puncturing = e < n && 16 * k <= 7 * e;
        let mut shortened = Vec::new();
        if e < n {
            if puncturing {
                for &p in &j[..n - e] {
                    frozen[p] = true;
                }
                let t = if 4 * e >= 3 * n {
                    // ceil(3N/4 - E/2)
                    (6 * n - 4 * e).div_ceil(8)
                } else {
                    // ceil(9N/16 - E/4)
                    (9 * n - 4 * e).div_ceil(16)
                };
                for f in frozen.iter_mut().take(t) {
                    *f = true;
                }
            } else {
                for &p in &j[e..] {
                    frozen[p] = true;
                    shortened.push(p);
                }
            }
        }
        // the K + n_PC most reliable of the others, in the order of the polar sequence
        let reliable: Vec<usize> =
            POLAR_SEQUENCE.iter().map(|&q| q as usize).filter(|&q| q < n && !frozen[q]).collect();
        let used: Vec<usize> = reliable[reliable.len() - (k + n_pc)..].to_vec();
        let mut pc = Vec::new();
        if n_pc > 0 {
            // n_PC - n_PC^wm parity checks on the least reliable positions of Q_I
            pc.extend_from_slice(&used[..n_pc - n_pc_wm]);
            if n_pc_wm > 0 {
                // and one of minimum row weight 2^popcount among the K most reliable,
                // the most reliable on ties
                let best = used[n_pc..]
                    .iter()
                    .enumerate()
                    .min_by_key(|&(r, &p)| (p.count_ones(), std::cmp::Reverse(r)))
                    .map(|(_, &p)| p)
                    .ok_or_else(|| com_core::Error::Config("no position for the parity check".into()))?;
                pc.push(best);
            }
        }
        let info: Vec<usize> = used.iter().copied().filter(|p| !pc.contains(p)).collect();
        let code = PolarCode::with_positions(n, &info, &pc)?;
        // the input interleaver of Sec. 5.3.1.1
        let il: Vec<usize> = if i_il {
            POLAR_INTERLEAVER.iter().map(|&m| m as usize).filter(|&m| m >= 164 - k).map(|m| m - (164 - k)).collect()
        } else {
            (0..k).collect()
        };
        // bit selection, Sec. 5.4.1.2: indices into y, then into d through J
        let sel: Vec<usize> = (0..e)
            .map(|t| {
                if e >= n {
                    t % n
                } else if puncturing {
                    t + n - e
                } else {
                    t
                }
            })
            .map(|y| j[y])
            .collect();
        let map = if i_bil {
            let mut t = 0;
            while t * (t + 1) / 2 < e {
                t += 1;
            }
            // row i holds T - i cells, written row by row
            let mut cell = vec![vec![None; t]; t];
            let mut idx = 0;
            for (i, row) in cell.iter_mut().enumerate() {
                for c in row.iter_mut().take(t - i) {
                    if idx < e {
                        *c = Some(sel[idx]);
                    }
                    idx += 1;
                }
            }
            let mut out = Vec::with_capacity(e);
            for col in 0..t {
                for row in cell.iter().take(t - col) {
                    if let Some(v) = row[col] {
                        out.push(v);
                    }
                }
            }
            out
        } else {
            sel
        };
        Ok(NrPolar { a, e, crc, il, map, shortened, code })
    }

    /// The same code decoded by SC list with `list` paths.
    #[must_use]
    pub fn with_list(mut self, list: usize) -> NrPolar {
        self.code = self.code.with_list(list);
        self
    }

    /// The mother code length $N$.
    pub fn mother_length(&self) -> usize {
        self.code.n()
    }

    /// The bit of the mother codeword carried by every output bit.
    pub fn bit_map(&self) -> &[usize] {
        &self.map
    }
}

impl BlockCode for NrPolar {
    fn k(&self) -> usize {
        self.a
    }

    fn n(&self) -> usize {
        self.e
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.a, "bits")?;
        let c = self.crc.attach(info);
        let interleaved: Vec<u8> = self.il.iter().map(|&p| c[p]).collect();
        let d = self.code.encode_block(&interleaved)?;
        Ok(self.map.iter().map(|&i| d[i]).collect())
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        expect_len(llr.len(), self.e, "LLRs")?;
        let mut d = vec![0.0; self.code.n()];
        for (&i, &l) in self.map.iter().zip(llr) {
            d[i] += l;
        }
        for &p in &self.shortened {
            d[p] = KNOWN;
        }
        let k = self.il.len();
        let mut first = None;
        for cand in self.code.decode_list(&d)? {
            let mut c = vec![0u8; k];
            for (&p, &b) in self.il.iter().zip(&cand) {
                c[p] = b;
            }
            if self.crc.check(&c) {
                return Ok(c[..self.a].to_vec());
            }
            first.get_or_insert(c);
        }
        Ok(first.map_or_else(|| vec![0; self.a], |c| c[..self.a].to_vec()))
    }
}

container_traits!(NrPolar);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_are_permutations() {
        let mut q: Vec<u16> = POLAR_SEQUENCE.to_vec();
        q.sort_unstable();
        assert!(q.iter().enumerate().all(|(i, &v)| v as usize == i));
        let mut il: Vec<u8> = POLAR_INTERLEAVER.to_vec();
        il.sort_unstable();
        assert!(il.iter().enumerate().all(|(i, &v)| v as usize == i));
        let mut p: Vec<u8> = SUBBLOCK_PATTERN.to_vec();
        p.sort_unstable();
        assert!(p.iter().enumerate().all(|(i, &v)| v as usize == i));
    }

    #[test]
    fn puncturing_shortening_repetition_and_parity_checks_round_trip() {
        // (A, E): shortening, puncturing, repetition, PC bits with and without the
        // minimum-weight one, downlink
        for (a, e, down) in
            [(30, 45, false), (30, 200, false), (40, 600, false), (14, 100, false), (14, 300, false), (39, 108, true)]
        {
            let code = if down { NrPolar::downlink(a, e) } else { NrPolar::uplink(a, e) }.unwrap().with_list(8);
            let info: Vec<u8> = (0..a).map(|i| ((i * 5 + a) % 3 == 0) as u8).collect();
            let c = code.encode_block(&info).unwrap();
            assert_eq!(c.len(), e);
            let llr: Vec<f64> = c.iter().map(|b| if *b == 0 { 2.0 } else { -2.0 }).collect();
            assert_eq!(code.decode_block(&llr).unwrap(), info, "A {a}, E {e}");
        }
        assert!(NrPolar::uplink(11, 100).is_err() && NrPolar::uplink(400, 2000).is_err());
    }
}
