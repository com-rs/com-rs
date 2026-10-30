//! Interleavers: permutations of the coded bits (or of their LLRs).
//!
//! An [`Interleaver`] holds a permutation $\pi$: the interleaved sequence is
//! $y_{i} = x_{\pi(i)}$. [`Interleaver::block`] writes row by row and reads column by
//! column; [`Interleaver::random`] draws a uniform permutation from a seed; the
//! quadratic permutation polynomial of LTE turbo codes is
//! [`Interleaver::qpp`], $\pi(i) = (f_{1} i + f_{2} i^{2}) \bmod K$ (Sun & Takeshita 2005).
//!
//! # Examples
//! ```
//! use com_fec::interleaving::Interleaver;
//! let p = Interleaver::block(2, 3);
//! assert_eq!(p.interleave(&[0, 1, 2, 3, 4, 5]), vec![0, 3, 1, 4, 2, 5]);
//! assert_eq!(p.deinterleave(&p.interleave(&[9, 8, 7, 6, 5, 4])), vec![9, 8, 7, 6, 5, 4]);
//! ```
//!
//! # References
//!
//! 1. J. Sun and O. Y. Takeshita, "Interleavers for turbo codes using permutation
//!    polynomials over integer rings," *IEEE Trans. Inf. Theory*, 51(1), 101–119, 2005.

use com_core::rng::Rng;
use com_core::{Result, ensure};

/// A permutation of the positions of a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interleaver {
    /// $\pi$: output position $i$ takes input position `perm[i]`.
    perm: Vec<usize>,
}

impl Interleaver {
    /// The interleaver of the permutation `perm`: output position $i$ takes input
    /// position `perm[i]`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if `perm` is not a permutation of $0, \dots, n - 1$.
    ///
    /// # Examples
    /// ```
    /// use com_fec::interleaving::Interleaver;
    /// let p = Interleaver::from_permutation(vec![2, 0, 1])?;
    /// assert_eq!(p.interleave(&['a', 'b', 'c']), ['c', 'a', 'b']);
    /// assert!(Interleaver::from_permutation(vec![0, 0, 1]).is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn from_permutation(perm: Vec<usize>) -> Result<Interleaver> {
        ensure_permutation(&perm, || "the interleaver is not a permutation".into())?;
        Ok(Interleaver { perm })
    }

    /// $\pi$: output position $i$ takes input position `perm()[i]`.
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Write row by row into `rows × cols`, read column by column.
    pub fn block(rows: usize, cols: usize) -> Interleaver {
        Interleaver { perm: (0..rows * cols).map(|i| (i % rows) * cols + i / rows).collect() }
    }

    /// A uniform random permutation of `n` positions (seeded).
    pub fn random(n: usize, seed: u64) -> Interleaver {
        let mut perm: Vec<usize> = (0..n).collect();
        Rng::new(seed).shuffle(&mut perm);
        Interleaver { perm }
    }

    /// The quadratic permutation polynomial $\pi(i) = (f_{1} i + f_{2} i^{2}) \bmod K$.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if the polynomial is not a permutation.
    pub fn qpp(k: usize, f1: usize, f2: usize) -> Result<Interleaver> {
        let perm: Vec<usize> = (0..k).map(|i| ((f1 * i) % k + (f2 * ((i * i) % k)) % k) % k).collect();
        ensure_permutation(&perm, || format!("({f1}, {f2}) is not a permutation polynomial modulo {k}"))?;
        Ok(Interleaver { perm })
    }

    /// Length of the block.
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// $y_{i} = x_{\pi(i)}$.
    ///
    /// # Panics
    /// If `x` is shorter than the permutation.
    pub fn interleave<T: Copy>(&self, x: &[T]) -> Vec<T> {
        self.perm.iter().map(|&p| x[p]).collect()
    }

    /// The inverse permutation.
    ///
    /// # Panics
    /// If `y` is shorter than the permutation.
    pub fn deinterleave<T: Copy + Default>(&self, y: &[T]) -> Vec<T> {
        let mut x = vec![T::default(); self.perm.len()];
        for (i, &p) in self.perm.iter().enumerate() {
            x[p] = y[i];
        }
        x
    }
}

/// [`com_core::Error::Config`] with the message of `msg` unless `perm` holds every
/// position below its length once.
fn ensure_permutation(perm: &[usize], msg: impl FnOnce() -> String) -> Result<()> {
    let mut seen = vec![false; perm.len()];
    for &p in perm {
        if p >= perm.len() || seen[p] {
            return ensure(false, msg);
        }
        seen[p] = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutations_invert() {
        let x: Vec<u32> = (0..40).collect();
        for p in [Interleaver::random(40, 3), Interleaver::block(5, 8), Interleaver::qpp(40, 3, 10).unwrap()] {
            let mut sorted = p.perm.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, x.iter().map(|v| *v as usize).collect::<Vec<_>>());
            assert_eq!(p.deinterleave(&p.interleave(&x)), x);
        }
        // the QPP of LTE for K = 40 is (3, 10); (2, 10) is not a permutation
        assert!(Interleaver::qpp(40, 2, 10).is_err());
    }
}
