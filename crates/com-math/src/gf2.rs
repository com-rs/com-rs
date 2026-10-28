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

/// Product of binary polynomials, lowest degree first (bits 0 or 1).
pub fn poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
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
