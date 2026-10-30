//! Arithmetic over GF(2): binary polynomials and dense binary matrices on bitsets.
//!
//! The channel codes build their generators and parity structures from these; nothing
//! here knows what a code is.
//!
//! # Examples
//! ```
//! use com_math::gf2::{Matrix, poly_mul};
//! // (1 + x)(1 + x) = 1 + x² over GF(2)
//! assert_eq!(poly_mul(&[1, 1], &[1, 1]), vec![1, 0, 1]);
//! // the parity-check matrix of the (3, 1) repetition code has rank 2
//! let mut h = Matrix::from_rows(3, &[vec![0, 1], vec![1, 2]]);
//! assert_eq!(h.reduce().len(), 2);
//! ```

/// Product of binary polynomials, lowest degree first (bits 0 or 1); the empty
/// polynomial (no coefficient) is zero, and so is its product with any other.
pub fn poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u8; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 1 {
            for (j, &y) in b.iter().enumerate() {
                out[i + j] ^= y;
            }
        }
    }
    out
}

/// The remainder $r(x) = m(x)\, x^{r} \bmod g(x)$ of the message `msg` by the generator
/// `generator` of degree $r$: the $r$ parity bits of a systematic cyclic code or a CRC.
///
/// The message is in transmission order (its first bit the coefficient of the highest
/// degree), the generator lowest degree first with $g_{r} = 1$; the remainder is
/// returned highest degree first. A shift register of $r$ bits on 64-bit words, any
/// degree (the CRCs of 5G NR, the BCH codes of DVB-S2 of degree 192).
///
/// # Examples
/// ```
/// use com_math::gf2::poly_rem;
/// // the (7, 4) Hamming code of g(x) = 1 + x + x³: x³ · x³ = x⁶ ≡ x² + 1
/// assert_eq!(poly_rem(&[1, 0, 0, 0], &[1, 1, 0, 1]), [1, 0, 1]);
/// assert!(poly_rem(&[1, 0, 1], &[1]).is_empty());
/// ```
pub fn poly_rem(msg: &[u8], generator: &[u8]) -> Vec<u8> {
    let r = generator.len().saturating_sub(1);
    if r == 0 {
        return Vec::new();
    }
    // the register of r bits in 64-bit words, bit d the coefficient of x^d
    let words = r.div_ceil(64);
    let mut g = vec![0u64; words];
    for (d, &b) in generator[..r].iter().enumerate() {
        if b & 1 == 1 {
            g[d / 64] |= 1 << (d % 64);
        }
    }
    let mut reg = vec![0u64; words];
    let top = r - 1;
    for &b in msg {
        let fb = ((reg[top / 64] >> (top % 64)) & 1) as u8 ^ (b & 1);
        // shift left by one
        for w in (0..words).rev() {
            let carry = if w > 0 { reg[w - 1] >> 63 } else { 0 };
            reg[w] = (reg[w] << 1) | carry;
        }
        if r % 64 != 0 {
            reg[words - 1] &= (1u64 << (r % 64)) - 1;
        }
        if fb == 1 {
            for (x, y) in reg.iter_mut().zip(&g) {
                *x ^= y;
            }
        }
    }
    (0..r).rev().map(|d| ((reg[d / 64] >> (d % 64)) & 1) as u8).collect()
}

/// A dense binary matrix, every row a bitset of `n_cols` bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    n_cols: usize,
    rows: Vec<Vec<u64>>,
}

impl Matrix {
    /// The matrix of `n_cols` columns whose rows list the columns that hold a 1.
    pub fn from_rows(n_cols: usize, rows: &[Vec<usize>]) -> Matrix {
        let words = n_cols.div_ceil(64);
        let rows = rows
            .iter()
            .map(|r| {
                let mut w = vec![0u64; words];
                for &c in r {
                    w[c / 64] ^= 1 << (c % 64);
                }
                w
            })
            .collect();
        Matrix { n_cols, rows }
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The entry at row `r`, column `c`.
    pub fn get(&self, r: usize, c: usize) -> bool {
        (self.rows[r][c / 64] >> (c % 64)) & 1 == 1
    }

    /// Reduced row echelon form in place, the pivots searched from the last column
    /// to the first: returns the pivot column of every leading row, in that order.
    /// The remaining rows are zero.
    pub fn reduce(&mut self) -> Vec<usize> {
        let mut pivots = Vec::new();
        let mut r = 0;
        for c in (0..self.n_cols).rev() {
            if r == self.rows.len() {
                break;
            }
            let Some(p) = (r..self.rows.len()).find(|&i| self.get(i, c)) else { continue };
            self.rows.swap(r, p);
            for i in 0..self.rows.len() {
                if i != r && self.get(i, c) {
                    let (a, b) = if i < r {
                        let (x, y) = self.rows.split_at_mut(r);
                        (&mut x[i], &y[0])
                    } else {
                        let (x, y) = self.rows.split_at_mut(i);
                        (&mut y[0], &x[r])
                    };
                    for (u, v) in a.iter_mut().zip(b.iter()) {
                        *u ^= v;
                    }
                }
            }
            pivots.push(c);
            r += 1;
        }
        pivots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The remainder by long division of m(x) x^r, written directly.
    fn long_division(msg: &[u8], g: &[u8]) -> Vec<u8> {
        let r = g.len() - 1;
        // highest degree first: the message then r zeros
        let mut a: Vec<u8> = msg.iter().copied().chain(std::iter::repeat_n(0, r)).collect();
        for i in 0..msg.len() {
            if a[i] == 1 {
                for d in 0..=r {
                    a[i + d] ^= g[r - d];
                }
            }
        }
        a[msg.len()..].to_vec()
    }

    #[test]
    fn the_remainder_is_that_of_the_long_division_at_any_degree() {
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut bit = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 1) as u8
        };
        for r in [1usize, 5, 24, 63, 64, 65, 128, 192] {
            let mut g: Vec<u8> = (0..r).map(|_| bit()).collect();
            g.push(1);
            for len in [0usize, 1, 7, 100] {
                let msg: Vec<u8> = (0..len).map(|_| bit()).collect();
                assert_eq!(poly_rem(&msg, &g), long_division(&msg, &g), "r = {r}, {len} bits");
            }
        }
    }

    #[test]
    fn the_product_with_the_empty_polynomial_is_empty_instead_of_an_underflow() {
        let none: Vec<u8> = Vec::new();
        assert_eq!(poly_mul(&[], &[]), none);
        assert_eq!((poly_mul(&[1, 1], &[]), poly_mul(&[], &[1])), (none.clone(), none));
    }

    #[test]
    fn polynomial_products_and_rank() {
        assert_eq!(poly_mul(&[1, 1, 0, 1], &[1]), vec![1, 1, 0, 1]);
        assert_eq!(poly_mul(&[1, 1], &[1, 1, 1]), vec![1, 0, 0, 1]);
        // two equal rows have rank 1; the reduction leaves the second one zero
        let mut m = Matrix::from_rows(70, &[vec![3, 69], vec![3, 69], vec![0]]);
        let pivots = m.reduce();
        assert_eq!(pivots, vec![69, 0]);
        assert!(m.get(0, 69) && m.get(0, 3) && !m.get(2, 3) && !m.get(2, 69));
        assert_eq!((m.n_rows(), m.n_cols()), (3, 70));
    }
}
