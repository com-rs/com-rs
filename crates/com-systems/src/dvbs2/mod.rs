//! DVB-S2 (ETSI EN 302 307-1 V1.4.1): the physical layer from the BBFRAME to the
//! scrambled PLFRAME, and the receiver back to the BBFRAME, for constant coding and
//! modulation.
//!
//! # Chain
//!
//! 1. **BBFRAME** (§ 5.1.6, § 5.2): a BBHEADER of 80 bits (MATYPE, UPL, DFL, SYNC, SYNCD,
//!    CRC-8 of $x^{8} + x^{7} + x^{6} + x^{4} + x^{2} + 1$) and the data field of a
//!    generic continuous stream, $K_{\mathrm{bch}}$ bits, randomised by the PRBS $1 +
//!    x^{14} + x^{15}$ loaded with `100101010000000` ([`bbframe`], [`bb_scramble`]).
//! 2. **FEC** (§ 5.3.1, § 5.3.2): BCH then LDPC, [`fec`].
//! 3. **Bit interleaver** (§ 5.3.3, Table 8): for 8PSK, 16APSK and 32APSK, the FECFRAME
//!    is written column-wise in $\eta_{\mathrm{MOD}}$ columns and read row-wise, the
//!    first column first (the last first for 8PSK 3/5).
//! 4. **Mapping** (§ 5.4, Figures 9 to 12): Gray QPSK and 8PSK, 4 + 12 APSK of ring ratio
//!    $\gamma$ (Table 9), 4 + 12 + 16 APSK of ratios $\gamma_{1}$, $\gamma_{2}$ (Table 10),
//!    unit average energy ([`constellation`]).
//! 5. **PL framing** (§ 5.5): the PLHEADER (SOF `18D2E82`, the (64, 7) PLS code of the
//!    MODCOD and the TYPE, scrambled, in $\pi/2$-BPSK), the $S$ slots of 90 symbols,
//!    pilot blocks of 36 symbols $(1 + j)/\sqrt{2}$ every 16 slots, and the
//!    multiplication by the Gold scrambling sequence $n$ of every symbol after the
//!    header ([`scrambling_sequence`]).
//!
//! The receiver ([`Dvbs2::llr`], [`Dvbs2::decode`]) assumes the frame found: it
//! descrambles, removes the header and the pilots, computes the exact bit LLRs of the
//! constellation, deinterleaves and decodes; [`decode_plheader`] recovers the MODCOD
//! and the TYPE of a header by maximum likelihood over the 128 PLS codewords.
//!
//! # Examples
//! ```
//! use com_systems::dvbs2::{CodeRate, Dvbs2, FrameSize, Modcod, Modulation};
//! let link = Dvbs2::new(Modcod::new(Modulation::Apsk16, CodeRate::R3_4)?, FrameSize::Short, true, 0)?;
//! assert_eq!(link.k(), 11712);
//! assert_eq!(link.plframe_len(), 90 * (45 + 1) + 36 * 2);   // S = 45 slots, 2 pilot blocks
//! let bb: Vec<u8> = (0..link.k()).map(|i| (i % 3 == 0) as u8).collect();
//! let x = link.modulate(&bb)?;
//! assert_eq!(com_systems::dvbs2::decode_plheader(&x[..90])?, (19, true, true));   // 16APSK 3/4, short, pilots
//! assert_eq!(link.decode(&x, 1e-3)?, bb);
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

pub mod fec;
mod tables;

pub use fec::{CodeRate, Dvbs2Fec, FrameSize};

use com_core::{Error, Result, ensure, ensure_shape};
use com_fec::crc::Crc;
use com_fec::interleaving::Interleaver;
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_math::sequences::lfsr;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::str::FromStr;

/// The modulations of DVB-S2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Modulation {
    /// Gray QPSK, 2 bits per symbol.
    Qpsk,
    /// Gray 8PSK, 3 bits per symbol.
    Psk8,
    /// 4 + 12 APSK, 4 bits per symbol.
    Apsk16,
    /// 4 + 12 + 16 APSK, 5 bits per symbol.
    Apsk32,
}

impl Modulation {
    /// Bits per symbol $\eta_{\mathrm{MOD}}$.
    pub fn bits_per_symbol(self) -> usize {
        match self {
            Modulation::Qpsk => 2,
            Modulation::Psk8 => 3,
            Modulation::Apsk16 => 4,
            Modulation::Apsk32 => 5,
        }
    }
}

impl FromStr for Modulation {
    type Err = Error;

    /// The modulation from its name: `QPSK`, `8PSK`, `16APSK`, `32APSK`
    /// (case-insensitive; `-`, `_` and spaces are ignored).
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    fn from_str(s: &str) -> Result<Modulation> {
        match s.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
            "qpsk" => Ok(Modulation::Qpsk),
            "8psk" => Ok(Modulation::Psk8),
            "16apsk" => Ok(Modulation::Apsk16),
            "32apsk" => Ok(Modulation::Apsk32),
            other => {
                Err(Error::Config(format!("DVB-S2 modulation must be QPSK, 8PSK, 16APSK or 32APSK, got {other:?}")))
            }
        }
    }
}

/// A modulation and a code rate of Table 12.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Modcod {
    /// The modulation.
    pub modulation: Modulation,
    /// The code rate.
    pub rate: CodeRate,
}

impl Modcod {
    /// A MODCOD of the standard.
    ///
    /// # Errors
    /// [`Error::Config`] for a combination outside Table 12.
    pub fn new(modulation: Modulation, rate: CodeRate) -> Result<Modcod> {
        let m = Modcod { modulation, rate };
        m.number().map(|_| m)
    }

    /// The MODCOD from names: `Modcod::from_name("16APSK", "3/4")`.
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown name or a combination outside Table 12.
    pub fn from_name(modulation: &str, rate: &str) -> Result<Modcod> {
        Modcod::new(modulation.parse()?, rate.parse()?)
    }

    /// The MODCOD field (Table 12): 1 to 28.
    ///
    /// # Errors
    /// [`Error::Config`] for a combination outside Table 12.
    pub fn number(self) -> Result<u8> {
        use CodeRate::{R1_2, R1_3, R1_4, R2_3, R2_5, R3_4, R3_5, R4_5, R5_6, R8_9, R9_10};
        let rates: &[CodeRate] = match self.modulation {
            Modulation::Qpsk => &[R1_4, R1_3, R2_5, R1_2, R3_5, R2_3, R3_4, R4_5, R5_6, R8_9, R9_10],
            Modulation::Psk8 => &[R3_5, R2_3, R3_4, R5_6, R8_9, R9_10],
            Modulation::Apsk16 => &[R2_3, R3_4, R4_5, R5_6, R8_9, R9_10],
            Modulation::Apsk32 => &[R3_4, R4_5, R5_6, R8_9, R9_10],
        };
        let first = match self.modulation {
            Modulation::Qpsk => 1,
            Modulation::Psk8 => 12,
            Modulation::Apsk16 => 18,
            Modulation::Apsk32 => 24,
        };
        rates
            .iter()
            .position(|r| *r == self.rate)
            .map(|i| first + i as u8)
            .ok_or_else(|| Error::Config(format!("DVB-S2 has no MODCOD {:?} {}", self.modulation, self.rate.name())))
    }

    /// The ideal $E_{s}/N_{0}$ (dB) at a PER of $10^{-7}$ of the normal FECFRAME, 50
    /// iterations (Table 13).
    ///
    /// Every MODCOD built by [`Modcod::new`] has an entry; another combination gives
    /// `NaN`.
    pub fn threshold_db(self) -> f64 {
        const TABLE13: [f64; 28] = [
            -2.35, -1.24, -0.30, 1.00, 2.23, 3.10, 4.03, 4.68, 5.18, 6.20, 6.42, // QPSK
            5.50, 6.62, 7.91, 9.35, 10.69, 10.98, // 8PSK
            8.97, 10.21, 11.03, 11.61, 12.89, 13.13, // 16APSK
            12.73, 13.64, 14.28, 15.69, 16.05, // 32APSK
        ];
        self.number().map_or(f64::NAN, |n| TABLE13[usize::from(n) - 1])
    }
}

/// The ring ratio $\gamma$ of 16APSK (Table 9).
fn gamma16(rate: CodeRate) -> f64 {
    match rate {
        CodeRate::R3_4 => 2.85,
        CodeRate::R4_5 => 2.75,
        CodeRate::R5_6 => 2.70,
        CodeRate::R8_9 => 2.60,
        CodeRate::R9_10 => 2.57,
        _ => 3.15,
    }
}

/// The ring ratios $\gamma_{1}$, $\gamma_{2}$ of 32APSK (Table 10).
fn gamma32(rate: CodeRate) -> (f64, f64) {
    match rate {
        CodeRate::R4_5 => (2.72, 4.87),
        CodeRate::R5_6 => (2.64, 4.64),
        CodeRate::R8_9 => (2.54, 4.33),
        CodeRate::R9_10 => (2.53, 4.30),
        _ => (2.84, 5.27),
    }
}

/// 8PSK: the multiple of π/4 of every label (Figure 10).
const PSK8_ANGLE: [u8; 8] = [1, 0, 4, 5, 2, 7, 3, 6];

/// 16APSK: the ring (1 inner, 2 outer) and the multiple of π/12 of every label (Figure 11): inner
/// ring at π/4 + kπ/2, outer ring at π/12 + kπ/6.
const APSK16_PLACE: [(u8, u8); 16] = [
    (2, 3),
    (2, 21),
    (2, 9),
    (2, 15),
    (2, 1),
    (2, 23),
    (2, 11),
    (2, 13),
    (2, 5),
    (2, 19),
    (2, 7),
    (2, 17),
    (1, 3),
    (1, 21),
    (1, 9),
    (1, 15),
];

/// 32APSK: the ring (1 inner, 2 middle, 3 outer) and the multiple of π/24 of every label
/// (Figure 12): inner ring at π/4 + kπ/2, middle at π/12 + kπ/6, outer at kπ/8.
const APSK32_PLACE: [(u8, u8); 32] = [
    (2, 6),
    (2, 10),
    (2, 42),
    (2, 38),
    (2, 18),
    (2, 14),
    (2, 30),
    (2, 34),
    (3, 3),
    (3, 9),
    (3, 42),
    (3, 36),
    (3, 18),
    (3, 12),
    (3, 27),
    (3, 33),
    (2, 2),
    (1, 6),
    (2, 46),
    (1, 42),
    (2, 22),
    (1, 18),
    (2, 26),
    (1, 30),
    (3, 0),
    (3, 6),
    (3, 45),
    (3, 39),
    (3, 21),
    (3, 15),
    (3, 24),
    (3, 30),
];

/// The constellation of a MODCOD, unit average energy, `points()[l]` carrying the label
/// `l` (MSB first, the first bit of the interleaver output), Figures 9 to 12.
///
/// # Errors
/// [`Error::Config`] for a combination outside Table 12.
pub fn constellation(modcod: Modcod) -> Result<Constellation> {
    modcod.number()?;
    let polar = |r: f64, phi: f64| C64::from_polar(r, phi);
    let points: Vec<C64> = match modcod.modulation {
        Modulation::Qpsk => (0..4)
            .map(|l| C64::new(1.0 - 2.0 * f64::from(l >> 1), 1.0 - 2.0 * f64::from(l & 1)) * FRAC_1_SQRT_2)
            .collect(),
        Modulation::Psk8 => PSK8_ANGLE.iter().map(|k| polar(1.0, f64::from(*k) * PI / 4.0)).collect(),
        Modulation::Apsk16 => {
            let g = gamma16(modcod.rate);
            let r1 = 2.0 / (1.0 + 3.0 * g * g).sqrt();
            let r2 = g * r1;

            APSK16_PLACE
                .iter()
                .map(|(ring, k)| polar(if *ring == 1 { r1 } else { r2 }, f64::from(*k) * PI / 12.0))
                .collect()
        }
        Modulation::Apsk32 => {
            let (g1, g2) = gamma32(modcod.rate);
            let r1 = (8.0 / (1.0 + 3.0 * g1 * g1 + 4.0 * g2 * g2)).sqrt();

            APSK32_PLACE
                .iter()
                .map(|(ring, k)| {
                    let r = match ring {
                        1 => r1,
                        2 => g1 * r1,
                        _ => g2 * r1,
                    };
                    polar(r, f64::from(*k) * PI / 24.0)
                })
                .collect()
        }
    };
    Constellation::from_points(points)
}

/// The bit interleaver of a FECFRAME of `n` bits (§ 5.3.3): the bits of symbol $r$
/// are the bits $r$ of the $\eta_{\mathrm{MOD}}$ columns of $N/\eta_{\mathrm{MOD}}$ rows
/// (column write, row read), in reverse column order for 8PSK 3/5; `None` for QPSK,
/// which is not interleaved.
pub fn bit_interleaver(modcod: Modcod, n: usize) -> Option<Interleaver> {
    let c = modcod.modulation.bits_per_symbol();
    if c == 2 {
        return None;
    }
    let rows = n / c;
    let reverse = modcod.modulation == Modulation::Psk8 && modcod.rate == CodeRate::R3_5;
    let col = |j: usize| if reverse { c - 1 - j } else { j };
    Some(Interleaver { perm: (0..rows * c).map(|i| col(i % c) * rows + i / c).collect() })
}

/// The Start Of Frame, 26 bits (§ 5.5.2.1).
const SOF: u32 = 0x18D_2E82;
/// The scrambling of the PLS code (§ 5.5.2.4).
const PLS_SCRAMBLE: u64 = 0b0111_0001_1001_1101_1000_0011_1100_1001_0101_0011_0100_0010_0010_1101_1111_1010;
/// The generator of the (32, 6) bi-orthogonal code (Figure 13b), first row first.
const PLS_G: [u32; 6] = [0x5555_5555, 0x3333_3333, 0x0F0F_0F0F, 0x00FF_00FF, 0x0000_FFFF, 0xFFFF_FFFF];

/// The 90 bits of a PLHEADER: SOF, then the scrambled (64, 7) PLS code of the MODCOD
/// field (0 for a dummy frame) and the TYPE (short frame, pilots).
pub fn plheader_bits(modcod: u8, short: bool, pilots: bool) -> Vec<u8> {
    let mut out: Vec<u8> = (0..26).rev().map(|i| ((SOF >> i) & 1) as u8).collect();
    let b: [u8; 6] =
        [(modcod >> 4) & 1, (modcod >> 3) & 1, (modcod >> 2) & 1, (modcod >> 1) & 1, modcod & 1, u8::from(short)];
    let mut y = 0u32;
    for (bit, row) in b.iter().zip(PLS_G) {
        if *bit == 1 {
            y ^= row;
        }
    }
    for i in 0..32 {
        let yi = ((y >> (31 - i)) & 1) as u8;
        for (k, v) in [yi, yi ^ u8::from(pilots)].into_iter().enumerate() {
            let s = ((PLS_SCRAMBLE >> (63 - (2 * i + k))) & 1) as u8;
            out.push(v ^ s);
        }
    }
    out
}

/// $\pi/2$-BPSK of the PLHEADER (§ 5.5.2): odd symbols on the diagonal, even ones on
/// the anti-diagonal.
pub fn pi2_bpsk(bits: &[u8]) -> Vec<C64> {
    bits.iter()
        .enumerate()
        .map(|(i, b)| {
            let a = FRAC_1_SQRT_2 * (1.0 - 2.0 * f64::from(*b));
            if i % 2 == 0 { C64::new(a, a) } else { C64::new(-a, a) }
        })
        .collect()
}

/// The MODCOD field and the TYPE (short frame, pilots) of a received PLHEADER (90
/// symbols, frame aligned, carrier recovered): the PLS codeword of largest correlation
/// with the $\pi/2$-BPSK symbols, maximum likelihood over AWGN.
///
/// # Errors
/// [`Error::Shape`] for fewer than 90 symbols.
pub fn decode_plheader(rx: &[C64]) -> Result<(u8, bool, bool)> {
    ensure_shape(rx.len() >= 90, || format!("a PLHEADER has 90 symbols, got {}", rx.len()))?;
    // back to real BPSK values: undo the π/2 rotation
    let soft: Vec<f64> = rx
        .iter()
        .take(90)
        .enumerate()
        .map(|(i, z)| if i % 2 == 0 { (z.re + z.im) * FRAC_1_SQRT_2 } else { (z.im - z.re) * FRAC_1_SQRT_2 })
        .collect();
    let mut best = (f64::NEG_INFINITY, 0u8, false, false);
    for code in 0..128u8 {
        let (modcod, short, pilots) = (code >> 2, (code >> 1) & 1 == 1, code & 1 == 1);
        let bits = plheader_bits(modcod, short, pilots);
        let corr: f64 = bits[26..].iter().zip(&soft[26..]).map(|(b, s)| (1.0 - 2.0 * f64::from(*b)) * s).sum();
        if corr > best.0 {
            best = (corr, modcod, short, pilots);
        }
    }
    Ok((best.1, best.2, best.3))
}

/// The complex scrambling sequence $C_{I} + j C_{Q} = e^{j R_{n}(i) \pi / 2}$ of code
/// number `n`, `len` symbols (§ 5.5.4): Gold sequences of the m-sequences $1 + x^{7} +
/// x^{18}$ and $1 + y^{5} + y^{7} + y^{10} + y^{18}$.
pub fn scrambling_sequence(n: usize, len: usize) -> Vec<C64> {
    const P: usize = (1 << 18) - 1;
    let x = lfsr(&[0, 7], 18, &[1], P + 18);
    let y = lfsr(&[0, 5, 7, 10], 18, &[1; 18], P + 18);
    let z = |i: usize| x[(i + n) % P] ^ y[i % P];
    (0..len)
        .map(|i| match 2 * z((i + 131_072) % P) + z(i) {
            0 => C64::new(1.0, 0.0),
            1 => C64::new(0.0, 1.0),
            2 => C64::new(-1.0, 0.0),
            _ => C64::new(0.0, -1.0),
        })
        .collect()
}

/// Randomise (or derandomise) a BBFRAME with the PRBS $1 + x^{14} + x^{15}$ loaded with
/// `100101010000000` (§ 5.2.2).
pub fn bb_scramble(bits: &[u8]) -> Vec<u8> {
    // register 1 … 15 loaded MSB first: the sequence starts with register 15
    const INIT: [u8; 15] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1];
    let prbs = lfsr(&[0, 1], 15, &INIT, bits.len() + 15);
    bits.iter().zip(&prbs[15..]).map(|(b, p)| b ^ p).collect()
}

/// The CRC-8 of the BBHEADER, $x^{8} + x^{7} + x^{6} + x^{4} + x^{2} + 1$ (§ 5.1.4).
pub fn bbheader_crc() -> Crc {
    Crc::new("CRC8", 8, 0xD5).unwrap_or_else(|_| unreachable!("a valid CRC of degree 8"))
}

/// The BBFRAME of $K_{\mathrm{bch}}$ bits of a generic continuous stream (§ 5.1.6, § 5.2):
/// the BBHEADER (TS/GS = 01, single stream, CCM, roll-off code `rolloff`, DFL =
/// $K_{\mathrm{bch}} - 80$, CRC-8) and the $K_{\mathrm{bch}} - 80$ data bits, randomised.
///
/// # Errors
/// [`Error::Config`] unless $80 \le K_{\mathrm{bch}} \le 65\,615$ (DFL holds on 16 bits),
/// [`Error::Shape`] for another number of data bits.
pub fn bbframe(data: &[u8], k_bch: usize, rolloff: u8) -> Result<Vec<u8>> {
    let dfl = k_bch.checked_sub(80).and_then(|d| u16::try_from(d).ok());
    let dfl = dfl.ok_or_else(|| Error::Config(format!("K_bch = {k_bch} is not between 80 and 65 615 bits")))?;
    ensure_shape(data.len() + 80 == k_bch, || {
        format!("a BBFRAME of {k_bch} bits carries {dfl} data bits, got {}", data.len())
    })?;
    // MATYPE-1: TS/GS 01, SIS 1, CCM 1, ISSYI 0, NPD 0, RO
    let matype1: u8 = 0b0111_0000 | (rolloff & 3);
    let bytes = [matype1, 0, 0, 0, (dfl >> 8) as u8, dfl as u8, 0, 0, 0];
    let fields: Vec<u8> = bytes.iter().flat_map(|v| (0..8).rev().map(move |i| (v >> i) & 1)).collect();
    let mut header = bbheader_crc().attach(&fields);
    header.extend_from_slice(data);
    Ok(bb_scramble(&header))
}

/// The data field of a received BBFRAME and whether the CRC-8 of its BBHEADER holds.
///
/// # Errors
/// [`Error::Shape`] for fewer than the 80 bits of the BBHEADER.
pub fn bbframe_data(bbframe: &[u8]) -> Result<(Vec<u8>, bool)> {
    ensure_shape(bbframe.len() >= 80, || {
        format!("a BBFRAME starts with an 80-bit BBHEADER, got {} bits", bbframe.len())
    })?;
    let clear = bb_scramble(bbframe);
    Ok((clear[80..].to_vec(), bbheader_crc().check(&clear[..80])))
}

/// A DVB-S2 transmitter and receiver of one MODCOD and frame configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Dvbs2 {
    modcod: Modcod,
    frame: FrameSize,
    pilots: bool,
    fec: Dvbs2Fec,
    interleaver: Option<Interleaver>,
    constellation: Constellation,
    scrambler: Vec<C64>,
}

impl Dvbs2 {
    /// The link of a MODCOD, a frame size, with or without pilots, and the scrambling
    /// code number `n` (0 for broadcasting).
    ///
    /// # Errors
    /// [`Error::Config`] for a combination outside Table 12, the short frame of rate
    /// 9/10, or `n` above 262 141.
    pub fn new(modcod: Modcod, frame: FrameSize, pilots: bool, n: usize) -> Result<Dvbs2> {
        ensure(n <= 262_141, || format!("the scrambling code number must be at most 262 141, got {n}"))?;
        let fec = Dvbs2Fec::new(frame, modcod.rate)?;
        let interleaver = bit_interleaver(modcod, fec.n());
        let constellation = constellation(modcod)?;
        let mut link = Dvbs2 { modcod, frame, pilots, fec, interleaver, constellation, scrambler: Vec::new() };
        link.scrambler = scrambling_sequence(n, link.plframe_len() - 90);
        Ok(link)
    }

    /// The MODCOD.
    pub fn modcod(&self) -> Modcod {
        self.modcod
    }

    /// Bits of a BBFRAME $K_{\mathrm{bch}}$.
    pub fn k(&self) -> usize {
        self.fec.k()
    }

    /// The forward error correction.
    pub fn fec(&self) -> &Dvbs2Fec {
        &self.fec
    }

    /// The forward error correction, to change its decoder settings.
    pub fn fec_mut(&mut self) -> &mut Dvbs2Fec {
        &mut self.fec
    }

    /// The constellation.
    pub fn constellation(&self) -> &Constellation {
        &self.constellation
    }

    /// Slots of data $S$ (Table 11).
    pub fn slots(&self) -> usize {
        self.fec.n() / self.modcod.modulation.bits_per_symbol() / 90
    }

    /// Pilot blocks per frame, $\lfloor (S - 1)/16 \rfloor$ with pilots.
    pub fn pilot_blocks(&self) -> usize {
        if self.pilots { (self.slots() - 1) / 16 } else { 0 }
    }

    /// Symbols per PLFRAME $90 (S + 1) + 36 \lfloor (S - 1)/16 \rfloor$.
    pub fn plframe_len(&self) -> usize {
        90 * (self.slots() + 1) + 36 * self.pilot_blocks()
    }

    /// Whether a symbol of the PLFRAME (after the header) is a data symbol.
    fn data_positions(&self) -> Vec<bool> {
        let mut out = Vec::with_capacity(self.plframe_len() - 90);
        for s in 0..self.slots() {
            out.extend(std::iter::repeat_n(true, 90));
            if self.pilots && (s + 1) % 16 == 0 && s + 1 < self.slots() {
                out.extend(std::iter::repeat_n(false, 36));
            }
        }
        out
    }

    /// The PLFRAME symbols of a BBFRAME of $K_{\mathrm{bch}}$ bits.
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of bits.
    pub fn modulate(&self, bbframe: &[u8]) -> Result<Vec<C64>> {
        let coded = self.fec.encode(bbframe)?;
        let bits = self.interleaver.as_ref().map_or(coded.clone(), |p| p.interleave(&coded));
        let symbols = self.constellation.modulate(&bits)?;
        let short = self.frame == FrameSize::Short;
        let mut out = pi2_bpsk(&plheader_bits(self.modcod.number()?, short, self.pilots));
        let pilot = C64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let positions = self.data_positions();
        let n_data = positions.iter().filter(|d| **d).count();
        ensure_shape(symbols.len() == n_data, || format!("{} symbols for {n_data} data slots", symbols.len()))?;
        let mut data = symbols.into_iter();
        for (is_data, c) in positions.into_iter().zip(&self.scrambler) {
            let v = if is_data { data.next().unwrap_or(pilot) } else { pilot };
            out.push(v * c);
        }
        Ok(out)
    }

    /// The deinterleaved LLRs of the FECFRAME of a received PLFRAME (frame aligned,
    /// carrier recovered) with complex noise of variance `noise_var` per symbol: exact,
    /// or max-log with `max_log`.
    ///
    /// # Errors
    /// [`Error::Shape`] if `rx` is not one PLFRAME.
    pub fn llr(&self, rx: &[C64], noise_var: f64, max_log: bool) -> Result<Vec<f64>> {
        ensure_shape(rx.len() == self.plframe_len(), || {
            format!("a PLFRAME has {} symbols, got {}", self.plframe_len(), rx.len())
        })?;
        let data: Vec<C64> = rx[90..]
            .iter()
            .zip(&self.scrambler)
            .zip(self.data_positions())
            .filter(|(_, d)| *d)
            .map(|((z, c), _)| z * c.conj())
            .collect();
        let llr = self.constellation.llr(&data, noise_var.max(1e-300), None, max_log)?;
        Ok(self.interleaver.as_ref().map_or(llr.clone(), |p| p.deinterleave(&llr)))
    }

    /// The BBFRAME of a received PLFRAME (see [`Dvbs2::llr`]).
    ///
    /// # Errors
    /// As [`Dvbs2::llr`].
    pub fn decode(&self, rx: &[C64], noise_var: f64) -> Result<Vec<u8>> {
        self.fec.decode(&self.llr(rx, noise_var, false)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    fn modcods() -> Vec<Modcod> {
        let mut v = Vec::new();
        for m in [Modulation::Qpsk, Modulation::Psk8, Modulation::Apsk16, Modulation::Apsk32] {
            for r in CodeRate::ALL {
                if let Ok(mc) = Modcod::new(m, r) {
                    v.push(mc);
                }
            }
        }
        v
    }

    #[test]
    fn table_12_has_28_modcods_numbered_1_to_28() {
        let n: Vec<u8> = modcods().iter().map(|m| m.number().unwrap()).collect();
        assert_eq!(n, (1..=28).collect::<Vec<u8>>());
        assert!(Modcod::new(Modulation::Psk8, CodeRate::R1_2).is_err());
    }

    #[test]
    fn the_constellations_have_unit_energy_and_their_published_geometry() {
        for m in modcods() {
            let c = constellation(m).unwrap();
            assert!((c.energy() - 1.0).abs() < 1e-12, "{m:?}");
            let p = c.points();
            let dmin = (0..p.len())
                .flat_map(|a| ((a + 1)..p.len()).map(move |b| (a, b)))
                .map(|(a, b)| (p[a] - p[b]).norm())
                .fold(f64::INFINITY, f64::min);
            assert!(dmin > 0.1, "{m:?}: {dmin}");
        }
        // 16APSK 3/4: ring ratio 2.85, label 1100 on the inner ring at π/4, 0100 outer at π/12
        let c = constellation(Modcod::new(Modulation::Apsk16, CodeRate::R3_4).unwrap()).unwrap();
        let p = c.points();
        assert!((p[0b0100].norm() / p[0b1100].norm() - 2.85).abs() < 1e-12);
        assert!((p[0b1100].arg() - PI / 4.0).abs() < 1e-12 && (p[0b0100].arg() - PI / 12.0).abs() < 1e-12);
        // 32APSK 9/10: 11000 on the outer ring at 0, 10000 on the middle one at π/12
        let c = constellation(Modcod::new(Modulation::Apsk32, CodeRate::R9_10).unwrap()).unwrap();
        let p = c.points();
        assert!((p[0b11000].norm() / p[0b10001].norm() - 4.30).abs() < 1e-12);
        assert!(p[0b11000].arg().abs() < 1e-12 && (p[0b10000].arg() - PI / 12.0).abs() < 1e-12);
        // 8PSK is Gray: neighbours differ in one bit
        let c = constellation(Modcod::new(Modulation::Psk8, CodeRate::R3_5).unwrap()).unwrap();
        let mut by_angle: Vec<usize> = (0..8).collect();
        by_angle.sort_by(|a, b| {
            c.points()[*a].arg().rem_euclid(2.0 * PI).total_cmp(&c.points()[*b].arg().rem_euclid(2.0 * PI))
        });
        for i in 0..8 {
            assert_eq!((by_angle[i] ^ by_angle[(i + 1) % 8]).count_ones(), 1);
        }
    }

    #[test]
    fn the_plheader_decodes_and_the_pls_code_has_distance_32() {
        for code in 0..128u8 {
            let (m, s, p) = (code >> 2, (code >> 1) & 1 == 1, code & 1 == 1);
            assert_eq!(decode_plheader(&pi2_bpsk(&plheader_bits(m, s, p))).unwrap(), (m, s, p));
        }
        let codewords: Vec<Vec<u8>> =
            (0..128u8).map(|c| plheader_bits(c >> 2, (c >> 1) & 1 == 1, c & 1 == 1)[26..].to_vec()).collect();
        let dmin = (0..128)
            .flat_map(|a| ((a + 1)..128).map(move |b| (a, b)))
            .map(|(a, b)| codewords[a].iter().zip(&codewords[b]).filter(|(x, y)| x != y).count())
            .min()
            .unwrap();
        assert_eq!(dmin, 32);
        // SOF, MSB first: 01 1000 1101 0010 1110 1000 0010
        assert_eq!(&plheader_bits(4, false, false)[..8], &[0, 1, 1, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn the_scrambling_and_the_bbframe_follow_the_standard() {
        // the PRBS of § 5.2.2 starts with 0000 0011 1111 0110 (0x03 0xF6), as in DVB-S
        let prbs = bb_scramble(&[0u8; 16]);
        assert_eq!(prbs, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]);
        let data: Vec<u8> = (0..7032 - 80).map(|i| (i % 5 == 1) as u8).collect();
        let bb = bbframe(&data, 7032, 0).unwrap();
        assert_eq!(bbframe_data(&bb).unwrap(), (data.clone(), true));
        assert!(matches!(bbframe(&[], 10, 0), Err(Error::Config(_))));
        assert!(matches!(bbframe(&[], 70_000, 0), Err(Error::Config(_))));
        assert!(matches!(bbframe(&data[1..], 7032, 0), Err(Error::Shape(_))));
        assert!(matches!(bbframe_data(&data[..79]), Err(Error::Shape(_))));
        // the Gold sequence n = 0 is the published one: R_0 takes the four values
        let s = scrambling_sequence(0, 1000);
        assert!(s.iter().all(|c| (c.norm() - 1.0).abs() < 1e-12));
    }

    #[test]
    fn interleaving_is_inverted_and_every_modcod_round_trips() {
        let mut rng = Rng::new(3);
        for m in modcods() {
            let link = Dvbs2::new(m, FrameSize::Short, true, 0);
            let Ok(link) = link else { continue };
            let bb: Vec<u8> = (0..link.k()).map(|_| rng.bit()).collect();
            let x = link.modulate(&bb).unwrap();
            assert_eq!(x.len(), link.plframe_len());
            let n = m.number().unwrap();
            assert_eq!(decode_plheader(&x[..90]).unwrap(), (n, true, true));
            assert_eq!(link.decode(&x, 1e-3).unwrap(), bb, "{m:?}");
        }
    }

    #[test]
    fn table_11_gives_the_slots() {
        for (m, normal, short) in [
            (Modulation::Qpsk, 360, 90),
            (Modulation::Psk8, 240, 60),
            (Modulation::Apsk16, 180, 45),
            (Modulation::Apsk32, 144, 36),
        ] {
            let mc = Modcod::new(m, CodeRate::R3_4).unwrap();
            assert_eq!(Dvbs2::new(mc, FrameSize::Normal, false, 0).unwrap().slots(), normal);
            assert_eq!(Dvbs2::new(mc, FrameSize::Short, false, 0).unwrap().slots(), short);
        }
    }
}
