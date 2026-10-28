//! Unsigned integers of arbitrary size, with the few operations that enumerative coding
//! needs: products and exact quotients by machine integers, sums, differences,
//! comparisons, and conversions from and to bit strings.
//!
//! # Examples
//! ```
//! use com_math::bigint::BigUint;
//! // C(100, 50) = 100891344545564193334812497256, beyond 64 bits
//! let mut c = BigUint::from(1);
//! for i in 1..=50u64 {
//!     c = c.mul_small(50 + i).div_small(i).0;
//! }
//! assert_eq!(c.bits(), 97);
//! let back = BigUint::from_bits(&c.to_bits(97));
//! assert_eq!(back, c);
//! ```

use std::cmp::Ordering;

/// An unsigned integer of arbitrary size (64-bit limbs, least significant first).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BigUint {
    limbs: Vec<u64>,
}

impl From<u64> for BigUint {
    fn from(v: u64) -> BigUint {
        let mut b = BigUint { limbs: vec![v] };
        b.trim();
        b
    }
}

impl BigUint {
    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// The number of significant bits (0 for zero): `⌊log2 v⌋ + 1`.
    pub fn bits(&self) -> usize {
        self.limbs.last().map_or(0, |top| 64 * (self.limbs.len() - 1) + (64 - top.leading_zeros() as usize))
    }

    /// The product by a machine integer.
    #[must_use]
    pub fn mul_small(&self, k: u64) -> BigUint {
        let mut out = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = 0u128;
        for &l in &self.limbs {
            let v = u128::from(l) * u128::from(k) + carry;
            out.push(v as u64);
            carry = v >> 64;
        }
        if carry > 0 {
            out.push(carry as u64);
        }
        let mut b = BigUint { limbs: out };
        b.trim();
        b
    }

    /// The quotient and the remainder of the division by a machine integer.
    ///
    /// # Panics
    /// If `k` is zero.
    #[must_use]
    pub fn div_small(&self, k: u64) -> (BigUint, u64) {
        assert!(k != 0, "division by zero");
        let mut out = vec![0u64; self.limbs.len()];
        let mut rem = 0u128;
        for (i, &l) in self.limbs.iter().enumerate().rev() {
            let v = (rem << 64) | u128::from(l);
            out[i] = (v / u128::from(k)) as u64;
            rem = v % u128::from(k);
        }
        let mut b = BigUint { limbs: out };
        b.trim();
        (b, rem as u64)
    }

    /// The sum.
    #[must_use]
    pub fn add(&self, o: &BigUint) -> BigUint {
        let n = self.limbs.len().max(o.limbs.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry = 0u128;
        for i in 0..n {
            let v = u128::from(*self.limbs.get(i).unwrap_or(&0)) + u128::from(*o.limbs.get(i).unwrap_or(&0)) + carry;
            out.push(v as u64);
            carry = v >> 64;
        }
        if carry > 0 {
            out.push(carry as u64);
        }
        let mut b = BigUint { limbs: out };
        b.trim();
        b
    }

    /// The difference `self - o`.
    ///
    /// # Panics
    /// If `o > self`.
    #[must_use]
    pub fn sub(&self, o: &BigUint) -> BigUint {
        assert!(*self >= *o, "negative difference");
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0u64;
        for (i, &l) in self.limbs.iter().enumerate() {
            let r = *o.limbs.get(i).unwrap_or(&0);
            let (d1, b1) = l.overflowing_sub(r);
            let (d2, b2) = d1.overflowing_sub(borrow);
            out.push(d2);
            borrow = u64::from(b1 || b2);
        }
        let mut b = BigUint { limbs: out };
        b.trim();
        b
    }

    /// The integer whose binary digits are `bits`, most significant first.
    pub fn from_bits(bits: &[u8]) -> BigUint {
        let mut limbs = vec![0u64; bits.len().div_ceil(64)];
        for (i, &b) in bits.iter().rev().enumerate() {
            if b != 0 {
                limbs[i / 64] |= 1 << (i % 64);
            }
        }
        let mut v = BigUint { limbs };
        v.trim();
        v
    }

    /// The `n` lowest binary digits, most significant first.
    pub fn to_bits(&self, n: usize) -> Vec<u8> {
        (0..n).rev().map(|i| self.limbs.get(i / 64).map_or(0, |l| ((l >> (i % 64)) & 1) as u8)).collect()
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, o: &BigUint) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for BigUint {
    fn cmp(&self, o: &BigUint) -> Ordering {
        self.limbs.len().cmp(&o.limbs.len()).then_with(|| self.limbs.iter().rev().cmp(o.limbs.iter().rev()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_agrees_with_u128() {
        let vals: [u128; 5] = [0, 1, 0xFFFF_FFFF_FFFF_FFFF, 0x1_0000_0000_0000_0001, 12_345_678_901_234_567_890_123];
        let big = |v: u128| BigUint::from_bits(&(0..128).rev().map(|i| ((v >> i) & 1) as u8).collect::<Vec<_>>());
        let val = |b: &BigUint| b.to_bits(128).iter().fold(0u128, |a, &x| (a << 1) | u128::from(x));
        for &a in &vals {
            for &b in &vals {
                assert_eq!(val(&big(a).add(&big(b))), a.wrapping_add(b));
                if a >= b {
                    assert_eq!(val(&big(a).sub(&big(b))), a - b);
                }
                assert_eq!(big(a).cmp(&big(b)), a.cmp(&b));
            }
            assert_eq!(val(&big(a).mul_small(977)), a * 977);
            let (q, r) = big(a).div_small(977);
            assert_eq!((val(&q), u128::from(r)), (a / 977, a % 977));
        }
    }
}
