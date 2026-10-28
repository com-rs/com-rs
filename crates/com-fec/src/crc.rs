//! Cyclic redundancy checks.
//!
//! A CRC of degree $L$ appends the remainder of $D^{L} a(D)$ divided by the generator
//! polynomial $g(D)$ to the information bits $a(D)$ (most significant bit first, zero
//! initial state, no reflection, no final XOR — the convention of 3GPP TS 38.212,
//! Sec. 5.1). The polynomials of 5G NR are provided.
//!
//! | Constructor | $g(D)$ | Use in NR |
//! |---|---|---|
//! | [`Crc::crc24a`] | `0x864CFB` | transport blocks |
//! | [`Crc::crc24b`] | `0x800063` | code blocks |
//! | [`Crc::crc24c`] | `0xB2B117` | downlink control (polar) |
//! | [`Crc::crc16`] | `0x1021` | small transport blocks |
//! | [`Crc::crc11`] | `0x621` | uplink control (polar) |
//! | [`Crc::crc6`] | `0x21` | uplink control (polar) |
//!
//! # Examples
//! ```
//! use com_fec::crc::Crc;
//! let crc = Crc::crc24a();
//! let mut word = crc.attach(&[1, 0, 1, 1, 0, 0, 1]);
//! assert_eq!(word.len(), 7 + 24);
//! assert!(crc.check(&word));
//! word[3] ^= 1;
//! assert!(!crc.check(&word));
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 38.212, *NR; Multiplexing and channel coding*, Sec. 5.1.
//! 2. W. W. Peterson and D. T. Brown, "Cyclic codes for error detection," *Proc. IRE*,
//!    49(1), 228–235, 1961.

use com_core::{Error, Result, ensure};

/// A cyclic redundancy check of degree `width` and generator `poly` (without its
/// leading term), valid by construction ([`Crc::new`] checks them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Crc {
    name: &'static str,
    width: u32,
    poly: u64,
}

impl Crc {
    /// A CRC of degree `width` (1 to 64) and generator `poly` without its leading term.
    ///
    /// # Errors
    /// [`Error::Config`] for a degree outside 1 to 64 or a generator with bits above it.
    pub fn new(name: &'static str, width: u32, poly: u64) -> Result<Crc> {
        ensure((1..=64).contains(&width), || format!("the CRC degree must be 1 to 64, got {width}"))?;
        ensure(width == 64 || poly < (1u64 << width), || {
            format!("the generator {poly:#x} exceeds the degree {width}")
        })?;
        Ok(Crc { name, width, poly })
    }

    /// Name of the CRC.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Degree $L$.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Generator polynomial without its leading term $D^{L}$.
    pub fn poly(&self) -> u64 {
        self.poly
    }

    /// CRC24A of 5G NR.
    pub fn crc24a() -> Crc {
        Crc { name: "CRC24A", width: 24, poly: 0x86_4CFB }
    }

    /// CRC24B of 5G NR.
    pub fn crc24b() -> Crc {
        Crc { name: "CRC24B", width: 24, poly: 0x80_0063 }
    }

    /// CRC24C of 5G NR.
    pub fn crc24c() -> Crc {
        Crc { name: "CRC24C", width: 24, poly: 0xB2_B117 }
    }

    /// CRC16 of 5G NR (CCITT).
    pub fn crc16() -> Crc {
        Crc { name: "CRC16", width: 16, poly: 0x1021 }
    }

    /// CRC11 of 5G NR.
    pub fn crc11() -> Crc {
        Crc { name: "CRC11", width: 11, poly: 0x621 }
    }

    /// CRC6 of 5G NR.
    pub fn crc6() -> Crc {
        Crc { name: "CRC6", width: 6, poly: 0x21 }
    }

    /// A CRC of 5G NR by name: `crc24a`, `crc24b`, `crc24c`, `crc16`, `crc11` or `crc6`
    /// (case-insensitive).
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    pub fn from_name(name: &str) -> Result<Crc> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "crc24a" => Crc::crc24a(),
            "crc24b" => Crc::crc24b(),
            "crc24c" => Crc::crc24c(),
            "crc16" => Crc::crc16(),
            "crc11" => Crc::crc11(),
            "crc6" => Crc::crc6(),
            other => {
                return Err(Error::Config(format!(
                    "unknown CRC {other:?}: crc24a, crc24b, crc24c, crc16, crc11 or crc6"
                )));
            }
        })
    }

    /// The $L$ parity bits of `bits`.
    pub fn checksum(&self, bits: &[u8]) -> Vec<u8> {
        // 1 ≤ L ≤ 64 by construction; the mask of 64 bits wraps
        let top = 1u64 << (self.width - 1);
        let mask = (top << 1).wrapping_sub(1);
        let mut reg = 0u64;
        for &b in bits {
            let fb = ((reg & top) != 0) ^ (b & 1 == 1);
            reg = (reg << 1) & mask;
            if fb {
                reg ^= self.poly;
            }
        }
        (0..self.width).rev().map(|i| ((reg >> i) & 1) as u8).collect()
    }

    /// `bits` followed by their parity bits.
    pub fn attach(&self, bits: &[u8]) -> Vec<u8> {
        let mut out = bits.to_vec();
        out.extend(self.checksum(bits));
        out
    }

    /// Whether a word (information then parity bits) passes the check.
    pub fn check(&self, word: &[u8]) -> bool {
        let w = self.width as usize;
        word.len() >= w && self.checksum(&word[..word.len() - w]) == word[word.len() - w..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_remainder_of_a_codeword_is_zero_and_every_single_error_is_detected() {
        for crc in [Crc::crc24a(), Crc::crc24b(), Crc::crc24c(), Crc::crc16(), Crc::crc11(), Crc::crc6()] {
            let word = crc.attach(&[1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0]);
            // a codeword is divisible by g(D): its own checksum is zero
            assert!(crc.checksum(&word).iter().all(|b| *b == 0), "{}", crc.name);
            for i in 0..word.len() {
                let mut e = word.clone();
                e[i] ^= 1;
                assert!(!crc.check(&e), "{} misses an error at {i}", crc.name);
            }
        }
        // CRC-16/CCITT ("XMODEM" convention: zero init) of the ASCII "123456789" is 0x31C3
        let msg: Vec<u8> = b"123456789".iter().flat_map(|c| (0..8).rev().map(move |i| (c >> i) & 1)).collect();
        let v = Crc::crc16().checksum(&msg).iter().fold(0u32, |a, b| (a << 1) | u32::from(*b));
        assert_eq!(v, 0x31C3);
        // the degrees 1 and 64 are valid
        for w in [1, 64] {
            let c = Crc::new("edge", w, 1).unwrap();
            assert!(c.check(&c.attach(&[1, 0, 1])));
        }
        assert!(Crc::new("bad", 65, 1).is_err() && Crc::new("bad", 4, 16).is_err());
        assert_eq!(Crc::from_name("CRC24A").unwrap(), Crc::crc24a());
    }
}

/// Checks against the published tables (`references/`).
#[cfg(test)]
mod literature {
    use super::*;

    /// 3GPP TS 38.212 V18.5.0, Sec. 5.1 (p. 11): the generator polynomials of the six
    /// CRCs of NR, written by their exponents.
    #[test]
    fn ts_38_212_sec_5_1_generator_polynomials() {
        let exponents = |crc: &Crc| -> Vec<u32> {
            let mut e: Vec<u32> = (0..crc.width).filter(|i| (crc.poly >> i) & 1 == 1).collect();
            e.push(crc.width);
            e.sort_unstable_by(|a, b| b.cmp(a));
            e
        };
        assert_eq!(exponents(&Crc::crc24a()), [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0]);
        assert_eq!(exponents(&Crc::crc24b()), [24, 23, 6, 5, 1, 0]);
        assert_eq!(exponents(&Crc::crc24c()), [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]);
        assert_eq!(exponents(&Crc::crc16()), [16, 12, 5, 0]);
        assert_eq!(exponents(&Crc::crc11()), [11, 10, 9, 5, 0]);
        assert_eq!(exponents(&Crc::crc6()), [6, 5, 0]);
    }
}
