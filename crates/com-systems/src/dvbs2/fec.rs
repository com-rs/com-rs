//! The forward error correction of DVB-S2 (ETSI EN 302 307-1, § 5.3): an outer BCH
//! code and an inner LDPC code of 64 800 bits (normal FECFRAME) or 16 200 bits (short
//! FECFRAME).
//!
//! # Model
//!
//! - **BCH** ([`com_fec::bch`]): $K_{\mathrm{bch}}$ bits of the BBFRAME, $N_{\mathrm{bch}} =
//!   K_{\mathrm{ldpc}}$ bits, correcting $t$ = 8, 10 or 12 errors; the generator is the
//!   product of the minimal polynomials $g_{1}, \dots, g_{t}$ of $\alpha, \alpha^{3},
//!   \dots, \alpha^{2t-1}$ in $\mathrm{GF}(2^{16})$, $p(x) = 1 + x^{2} + x^{3} + x^{5} +
//!   x^{16}$ (normal frames, Table 6a), or in $\mathrm{GF}(2^{14})$, $p(x) = 1 + x + x^{3}
//!   + x^{5} + x^{14}$ (short frames, Table 6b).
//! - **LDPC** ([`com_fec::ldpc::LdpcCode::accumulate`]): an irregular repeat-accumulate
//!   code. Information bit $i_{360 g + l}$ of group $g$ enters the parity checks
//!   $(x + l q) \bmod (N - K)$ for every address $x$ of row $g$ of the tables of Annexes
//!   B and C, with $q = (N - K)/360$; the parity bits are then accumulated,
//!   $p_{j} \leftarrow p_{j} \oplus p_{j-1}$ (§ 5.3.2).
//!
//! The decoder runs the layered sum-product algorithm of [`com_fec::ldpc`] (50 iterations
//! by default, the reference of the performance table of the standard; the normalised
//! min-sum loses 0.2 dB at rate 1/2 and does not reach the thresholds of the rates 1/4
//! and 1/3), then the BCH decoder on the hard decisions of the information bits.
//!
//! # Examples
//! ```
//! use com_systems::dvbs2::fec::{CodeRate, Dvbs2Fec, FrameSize};
//! let fec = Dvbs2Fec::new(FrameSize::Short, CodeRate::R1_2)?;
//! assert_eq!((fec.k(), fec.n()), (7032, 16200));     // Table 5b
//! let bits: Vec<u8> = (0..fec.k()).map(|i| (i % 7 == 3) as u8).collect();
//! let c = fec.encode(&bits)?;
//! let llr: Vec<f64> = c.iter().map(|b| if *b == 0 { 4.0 } else { -4.0 }).collect();
//! assert_eq!(fec.decode(&llr)?, bits);
//! // the frame sizes and the rates are read from their names
//! assert_eq!("16200".parse::<FrameSize>()?, FrameSize::Short);
//! assert_eq!("2/3".parse::<CodeRate>()?, CodeRate::R2_3);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. ETSI EN 302 307-1 V1.4.1, *Digital Video Broadcasting (DVB); Second generation
//!    framing structure, channel coding and modulation systems for Broadcasting,
//!    Interactive Services, News Gathering and other broadband satellite applications;
//!    Part 1: DVB-S2*, 2014.
//! 2. A. Morello and V. Mignone, "DVB-S2: The second generation standard for satellite
//!    broad-band services," *Proc. IEEE*, 94(1), 210–227, 2006.
//! 3. AFF3CT, the tables of `Tools/Code/LDPC/Standard/DVBS2`, identical to Annexes B
//!    and C.

use super::tables;
use com_core::{Error, Result, expect_len};
use com_fec::bch::Bch;
use com_fec::block::{BlockCode, hard};
use com_fec::ldpc::{CheckRule, LdpcCode};
use std::str::FromStr;

/// The FECFRAME length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameSize {
    /// 64 800 bits.
    Normal,
    /// 16 200 bits.
    Short,
}

impl FrameSize {
    /// The number of coded bits $n_{\mathrm{ldpc}}$ of the FECFRAME.
    pub fn n(self) -> usize {
        match self {
            FrameSize::Normal => 64800,
            FrameSize::Short => 16200,
        }
    }
}

impl FromStr for FrameSize {
    type Err = Error;

    /// The frame size from its name or its length: `normal` or `64800`, `short` or
    /// `16200` (case-insensitive).
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    fn from_str(s: &str) -> Result<FrameSize> {
        match s.trim().to_ascii_lowercase().as_str() {
            "normal" | "64800" => Ok(FrameSize::Normal),
            "short" | "16200" => Ok(FrameSize::Short),
            other => Err(Error::Config(format!("DVB-S2 frame must be normal (64800) or short (16200), got {other:?}"))),
        }
    }
}

/// The code rates of DVB-S2 (nominal: the short frames of rates 1/4 to 4/5 have
/// slightly lower effective rates, Table 5b).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs, reason = "the names are the rates")]
#[non_exhaustive]
pub enum CodeRate {
    R1_4,
    R1_3,
    R2_5,
    R1_2,
    R3_5,
    R2_3,
    R3_4,
    R4_5,
    R5_6,
    R8_9,
    R9_10,
}

impl CodeRate {
    /// Every rate, in increasing order.
    pub const ALL: [CodeRate; 11] = [
        CodeRate::R1_4,
        CodeRate::R1_3,
        CodeRate::R2_5,
        CodeRate::R1_2,
        CodeRate::R3_5,
        CodeRate::R2_3,
        CodeRate::R3_4,
        CodeRate::R4_5,
        CodeRate::R5_6,
        CodeRate::R8_9,
        CodeRate::R9_10,
    ];

    /// The name of the rate.
    pub fn name(self) -> &'static str {
        match self {
            CodeRate::R1_4 => "1/4",
            CodeRate::R1_3 => "1/3",
            CodeRate::R2_5 => "2/5",
            CodeRate::R1_2 => "1/2",
            CodeRate::R3_5 => "3/5",
            CodeRate::R2_3 => "2/3",
            CodeRate::R3_4 => "3/4",
            CodeRate::R4_5 => "4/5",
            CodeRate::R5_6 => "5/6",
            CodeRate::R8_9 => "8/9",
            CodeRate::R9_10 => "9/10",
        }
    }
}

impl FromStr for CodeRate {
    type Err = Error;

    /// The rate from its name: `1/4`, `2/3`, `9/10` …
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    fn from_str(s: &str) -> Result<CodeRate> {
        CodeRate::ALL
            .iter()
            .copied()
            .find(|r| r.name() == s.trim())
            .ok_or_else(|| Error::Config(format!("DVB-S2 code rate must be 1/4, 1/3, … 9/10, got {s:?}")))
    }
}

/// The parameters of one code: $K_{\mathrm{bch}}$, $N_{\mathrm{bch}} = K_{\mathrm{ldpc}}$,
/// $t$ (Tables 5a and 5b) and the LDPC table.
struct Params {
    k_bch: usize,
    k_ldpc: usize,
    t: usize,
    table: &'static [u16],
}

fn params(frame: FrameSize, rate: CodeRate) -> Result<Params> {
    use CodeRate::{R1_2, R1_3, R1_4, R2_3, R2_5, R3_4, R3_5, R4_5, R5_6, R8_9, R9_10};
    use tables::{
        N16200_P1800, N16200_P2880, N16200_P3600, N16200_P4320, N16200_P5400, N16200_P6480, N16200_P9000, N16200_P9720,
        N16200_P10800, N16200_P12960, N64800_P6480, N64800_P7200, N64800_P10800, N64800_P12960, N64800_P16200,
        N64800_P21600, N64800_P25920, N64800_P32400, N64800_P38880, N64800_P43200, N64800_P48600,
    };
    let p = |k_bch, k_ldpc, t, table| Params { k_bch, k_ldpc, t, table };
    Ok(match frame {
        FrameSize::Normal => match rate {
            R1_4 => p(16008, 16200, 12, N64800_P48600),
            R1_3 => p(21408, 21600, 12, N64800_P43200),
            R2_5 => p(25728, 25920, 12, N64800_P38880),
            R1_2 => p(32208, 32400, 12, N64800_P32400),
            R3_5 => p(38688, 38880, 12, N64800_P25920),
            R2_3 => p(43040, 43200, 10, N64800_P21600),
            R3_4 => p(48408, 48600, 12, N64800_P16200),
            R4_5 => p(51648, 51840, 12, N64800_P12960),
            R5_6 => p(53840, 54000, 10, N64800_P10800),
            R8_9 => p(57472, 57600, 8, N64800_P7200),
            R9_10 => p(58192, 58320, 8, N64800_P6480),
        },
        FrameSize::Short => match rate {
            R1_4 => p(3072, 3240, 12, N16200_P12960),
            R1_3 => p(5232, 5400, 12, N16200_P10800),
            R2_5 => p(6312, 6480, 12, N16200_P9720),
            R1_2 => p(7032, 7200, 12, N16200_P9000),
            R3_5 => p(9552, 9720, 12, N16200_P6480),
            R2_3 => p(10632, 10800, 12, N16200_P5400),
            R3_4 => p(11712, 11880, 12, N16200_P4320),
            R4_5 => p(12432, 12600, 12, N16200_P3600),
            R5_6 => p(13152, 13320, 12, N16200_P2880),
            R8_9 => p(14232, 14400, 12, N16200_P1800),
            R9_10 => return Err(Error::Config("DVB-S2 has no short FECFRAME of rate 9/10".into())),
        },
    })
}

/// The parity checks of the LDPC code of `table` (Annexes B and C) for $N$ coded bits
/// and $K$ information bits: the rows of $H$.
fn ldpc_rows(n: usize, k: usize, table: &[u16]) -> Vec<Vec<usize>> {
    let m = n - k;
    let q = m / 360;
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut i = 0;
    let mut bit = 0;
    while i < table.len() {
        let count = usize::from(table[i]);
        let addresses = &table[i + 1..i + 1 + count];
        for l in 0..360 {
            for &x in addresses {
                rows[(usize::from(x) + l * q) % m].push(bit);
            }
            bit += 1;
        }
        i += 1 + count;
    }
    for (r, row) in rows.iter_mut().enumerate() {
        if r > 0 {
            row.push(k + r - 1);
        }
        row.push(k + r);
    }
    rows
}

/// The BCH and LDPC codes of one FECFRAME (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Dvbs2Fec {
    frame: FrameSize,
    rate: CodeRate,
    bch: Bch,
    ldpc: LdpcCode,
}

impl Dvbs2Fec {
    /// The code of a frame size and a rate.
    ///
    /// # Errors
    /// [`Error::Config`] for the short frame of rate 9/10, which the standard does not
    /// define.
    pub fn new(frame: FrameSize, rate: CodeRate) -> Result<Dvbs2Fec> {
        let p = params(frame, rate)?;
        let n = frame.n();
        let bch = match frame {
            FrameSize::Normal => Bch::new(16, 0b1_0000_0000_0010_1101, p.t, p.k_ldpc)?,
            FrameSize::Short => Bch::new(14, 0b100_0000_0010_1011, p.t, p.k_ldpc)?,
        };
        debug_assert_eq!(bch.k(), p.k_bch);
        let mut ldpc = LdpcCode::accumulate(n, ldpc_rows(n, p.k_ldpc, p.table))?;
        // sum-product: the normalised min-sum loses 0.2 dB at rate 1/2 and fails at the
        // thresholds of Table 13 for the rates 1/4 and 1/3
        ldpc.rule = CheckRule::SumProduct;
        Ok(Dvbs2Fec { frame, rate, bch, ldpc })
    }

    /// The frame size.
    pub fn frame(&self) -> FrameSize {
        self.frame
    }

    /// The code rate.
    pub fn rate(&self) -> CodeRate {
        self.rate
    }

    /// Information bits per FECFRAME $K_{\mathrm{bch}}$.
    pub fn k(&self) -> usize {
        self.bch.k()
    }

    /// Coded bits per FECFRAME $n_{\mathrm{ldpc}}$.
    pub fn n(&self) -> usize {
        self.ldpc.n()
    }

    /// The outer code.
    pub fn bch(&self) -> &Bch {
        &self.bch
    }

    /// The inner code (its decoder settings are public fields).
    pub fn ldpc(&self) -> &LdpcCode {
        &self.ldpc
    }

    /// The inner code, to change its decoder settings.
    pub fn ldpc_mut(&mut self) -> &mut LdpcCode {
        &mut self.ldpc
    }

    /// The FECFRAME of $K_{\mathrm{bch}}$ bits: the BBFRAME, the BCH parity, the LDPC
    /// parity.
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of bits.
    pub fn encode(&self, bits: &[u8]) -> Result<Vec<u8>> {
        let b = self.bch.encode(bits)?;
        self.ldpc.encode_block(&b)
    }

    /// The $K_{\mathrm{bch}}$ information bits of the LLRs of a FECFRAME (positive for a
    /// 0): LDPC decoding, then BCH decoding of the hard decisions.
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of LLRs.
    pub fn decode(&self, llr: &[f64]) -> Result<Vec<u8>> {
        Ok(self.decode_detailed(llr)?.0)
    }

    /// [`Dvbs2Fec::decode`] with the LDPC iterations and the number of errors the BCH
    /// decoder corrected (`None` if it detected an uncorrectable word).
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of LLRs.
    pub fn decode_detailed(&self, llr: &[f64]) -> Result<(Vec<u8>, usize, Option<usize>)> {
        expect_len(llr.len(), self.n(), "LLRs")?;
        let (post, iterations) = self.ldpc.decode_llr(llr)?;
        let h = hard(&post[..self.bch.n()]);
        let (bits, corrected) = self.bch.decode(&h)?;
        Ok((bits, iterations, corrected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    /// The LLRs of the bits `c` sent in BPSK over a real AWGN channel of variance `s2`.
    fn awgn_llr(c: &[u8], s2: f64, rng: &mut Rng) -> Vec<f64> {
        c.iter().map(|b| 2.0 * (1.0 - 2.0 * f64::from(*b) + s2.sqrt() * rng.gaussian()) / s2).collect()
    }

    #[test]
    fn every_code_has_the_dimensions_of_tables_5a_and_5b() {
        for frame in [FrameSize::Normal, FrameSize::Short] {
            for rate in CodeRate::ALL {
                if frame == FrameSize::Short && rate == CodeRate::R9_10 {
                    assert!(Dvbs2Fec::new(frame, rate).is_err());
                    continue;
                }
                let fec = Dvbs2Fec::new(frame, rate).unwrap();
                let p = params(frame, rate).unwrap();
                assert_eq!((fec.k(), fec.bch.n()), (p.k_bch, p.k_ldpc), "{frame:?} {}", rate.name());
                assert_eq!(fec.ldpc.k(), p.k_ldpc);
            }
        }
    }

    #[test]
    fn codewords_satisfy_the_parity_checks_and_decode() {
        let mut rng = Rng::new(7);
        for rate in [CodeRate::R1_4, CodeRate::R2_3, CodeRate::R8_9] {
            let fec = Dvbs2Fec::new(FrameSize::Short, rate).unwrap();
            let bits: Vec<u8> = (0..fec.k()).map(|_| rng.bit()).collect();
            let c = fec.encode(&bits).unwrap();
            assert!(fec.ldpc.is_codeword(&c));
            // BPSK over AWGN at an SNR well above the threshold
            let llr = awgn_llr(&c, 0.25, &mut rng);
            assert_eq!(fec.decode(&llr).unwrap(), bits, "{}", rate.name());
        }
    }

    #[test]
    fn the_variable_degrees_are_those_of_the_standard() {
        // normal frame of rate 1/2 (Table 7a … the degree distribution of the codes):
        // 12 960 bits of degree 8, 19 440 of degree 3, the parity bits of degree 2
        let fec = Dvbs2Fec::new(FrameSize::Normal, CodeRate::R1_2).unwrap();
        let mut deg = vec![0usize; 64800];
        for r in fec.ldpc.rows() {
            for &v in r {
                deg[v] += 1;
            }
        }
        let count = |d: usize| deg[..32400].iter().filter(|x| **x == d).count();
        assert_eq!((count(8), count(3)), (12960, 19440));
        assert_eq!(deg[32400..64799].iter().filter(|x| **x == 2).count(), 32399);
        assert_eq!(deg[64799], 1);
    }
}
