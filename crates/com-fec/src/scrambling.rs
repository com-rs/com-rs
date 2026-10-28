//! Scrambling by the length-31 Gold sequence of 3GPP.
//!
//! The pseudo-random sequence of 3GPP TS 38.211, Sec. 5.2.1:
//!
//! $$
//! c(n) = \left(x_{1}(n + N_{c}) + x_{2}(n + N_{c})\right) \bmod 2, \qquad N_{c} = 1600,
//! $$
//!
//! with $x_{1}(n+31) = (x_{1}(n+3) + x_{1}(n)) \bmod 2$, $x_{1}(0) = 1$, $x_{1}(1..30) = 0$, and
//! $x_{2}(n+31) = (x_{2}(n+3) + x_{2}(n+2) + x_{2}(n+1) + x_{2}(n)) \bmod 2$ initialised by
//! $c_{\mathrm{init}} = \sum_{i} x_{2}(i) 2^{i}$. Scrambling adds $c(n)$ to the bits;
//! descrambling LLRs flips the sign of those where $c(n) = 1$.
//!
//! # Examples
//! ```
//! use com_fec::scrambling::Scrambler;
//! let s = Scrambler { c_init: 0x1234 };
//! let bits = [1, 0, 0, 1, 1, 1, 0, 0];
//! assert_eq!(s.scramble(&s.scramble(&bits)), bits);
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 38.211, *NR; Physical channels and modulation*, Sec. 5.2.1.

/// The sequence of a linear-feedback shift register of order `order`: the first
/// `order` bits are `init` (zero-padded), then $x_{i + \mathrm{order}} = \sum_{t} x_{i + t}
/// \bmod 2$ over the `taps`, up to `len` bits in all (the Fibonacci form of every
/// m-sequence, Gold sequence and PRBS of the standards).
///
/// # Examples
/// ```
/// use com_fec::scrambling::lfsr;
/// // x³ + x + 1: the m-sequence of period 7
/// let x = lfsr(&[0, 1], 3, &[1, 0, 0], 10);
/// assert_eq!(x, vec![1, 0, 0, 1, 0, 1, 1, 1, 0, 0]);
/// ```
pub fn lfsr(taps: &[usize], order: usize, init: &[u8], len: usize) -> Vec<u8> {
    let mut x = vec![0u8; len.max(order)];
    for (v, &b) in x.iter_mut().zip(init.iter().take(order)) {
        *v = b & 1;
    }
    for i in 0..len.saturating_sub(order) {
        x[i + order] = taps.iter().fold(0, |acc, &t| acc ^ x[i + t]);
    }
    x.truncate(len);
    x
}

/// The Gold scrambler of 3GPP with the initial value `c_init`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scrambler {
    /// Initial value of the second sequence (31 bits).
    pub c_init: u32,
}

impl Scrambler {
    /// The first `n` bits of $c$.
    pub fn sequence(&self, n: usize) -> Vec<u8> {
        const NC: usize = 1600;
        let len = n + NC + 31;
        let init2: Vec<u8> = (0..31).map(|i| ((self.c_init >> i) & 1) as u8).collect();
        let x1 = lfsr(&[0, 3], 31, &[1], len);
        let x2 = lfsr(&[0, 1, 2, 3], 31, &init2, len);
        (0..n).map(|i| x1[i + NC] ^ x2[i + NC]).collect()
    }

    /// Scramble (or descramble) bits.
    pub fn scramble(&self, bits: &[u8]) -> Vec<u8> {
        bits.iter().zip(self.sequence(bits.len())).map(|(b, c)| b ^ c).collect()
    }

    /// Descramble LLRs.
    pub fn descramble_llr(&self, llr: &[f64]) -> Vec<f64> {
        llr.iter().zip(self.sequence(llr.len())).map(|(l, c)| if c == 1 { -l } else { *l }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sequence_is_balanced_and_depends_on_its_seed() {
        let a = Scrambler { c_init: 1 }.sequence(10_000);
        let b = Scrambler { c_init: 2 }.sequence(10_000);
        let ones = a.iter().filter(|v| **v == 1).count();
        assert!((ones as f64 / 1e4 - 0.5).abs() < 0.02);
        assert_ne!(a, b);
        let llr = [1.0, -2.0, 3.0];
        let s = Scrambler { c_init: 7 };
        assert_eq!(s.descramble_llr(&s.descramble_llr(&llr)), llr);
    }
}
