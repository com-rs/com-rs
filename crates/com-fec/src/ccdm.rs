//! Constant-composition distribution matching (CCDM): uniform bits to sequences of
//! symbols of a prescribed distribution, the shaping stage of probabilistic amplitude
//! shaping (`com_systems::pas`).
//!
//! # Model
//!
//! A composition $(n_{1}, \dots, n_{A})$, $\sum_{a} n_{a} = n$, defines the set of the
//! $\binom{n}{n_{1}, \dots, n_{A}} = n! / \prod_{a} n_{a}!$ sequences of $n$ symbols in which
//! symbol $a$ appears $n_{a}$ times: their empirical distribution is exactly
//! $n_{a} / n$. The matcher maps $k = \lfloor \log_{2} \binom{n}{n_{1}, \dots, n_{A}}
//! \rfloor$ uniform bits, read as an integer, to the sequence of that rank in the
//! lexicographic order of the set (enumerative coding, Cover 1973), and the dematcher
//! computes the rank back. The rate $k/n$ approaches the entropy of the target
//! distribution as $n$ grows, with a rate loss of the order of $\log n / n$ (Schulte &
//! Böcherer 2016, who implement the same set by arithmetic coding).
//!
//! The ranks are exact integers of about $n H$ bits ([`com_math::bigint::BigUint`]); a
//! symbol costs $O(A)$ big-integer products and quotients by machine integers.
//!
//! [`Ccdm::from_distribution`] chooses the composition nearest to a target
//! distribution (largest remainders). [`Ccdm::dematch`] accepts any sequence: a symbol
//! that the composition has used up (after a decoding error) is replaced by the nearest
//! available one, so that wrong sequences still give $k$ bits.
//!
//! # Examples
//! ```
//! use com_fec::ccdm::Ccdm;
//! // 4 amplitudes with probabilities 0.4, 0.3, 0.2, 0.1, blocks of 100 amplitudes
//! let dm = Ccdm::from_distribution(&[0.4, 0.3, 0.2, 0.1], 100)?;
//! assert_eq!(dm.composition(), &[40, 30, 20, 10]);
//! assert_eq!(dm.k(), 175);   // log2 of 100!/(40! 30! 20! 10!) = 175.03; 100 H = 184.6
//! let bits: Vec<u8> = (0..dm.k()).map(|i| ((i * 7) % 3 == 0) as u8).collect();
//! let a = dm.matched(&bits)?;
//! assert_eq!(a.iter().filter(|&&s| s == 0).count(), 40);
//! assert_eq!(dm.dematch(&a)?, bits);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. P. Schulte and G. Böcherer, "Constant composition distribution matching," *IEEE
//!    Trans. Inf. Theory*, 62(1), 430–434, 2016.
//! 2. T. M. Cover, "Enumerative source coding," *IEEE Trans. Inf. Theory*, 19(1),
//!    73–77, 1973.

use crate::block::expect_len;
use com_core::{Result, ensure};
use com_math::bigint::BigUint;

/// A constant-composition distribution matcher (see the module).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ccdm {
    composition: Vec<usize>,
    n: usize,
    k: usize,
    total: BigUint,
}

impl Ccdm {
    /// The matcher of a composition: `composition[a]` occurrences of symbol `a`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for an empty composition or one that carries no bit.
    pub fn new(composition: Vec<usize>) -> Result<Ccdm> {
        let n: usize = composition.iter().sum();
        ensure(n > 0, || "the composition is empty".into())?;
        // the multinomial coefficient as a product of binomial coefficients
        let mut total = BigUint::from(1);
        let mut left = n as u64;
        for &c in &composition {
            for i in 1..=c as u64 {
                total = total.mul_small(left - c as u64 + i).div_small(i).0;
            }
            left -= c as u64;
        }
        let k = total.bits().saturating_sub(1);
        ensure(k >= 1, || "the composition has a single sequence: it carries no bit".into())?;
        Ok(Ccdm { composition, n, k, total })
    }

    /// The matcher of blocks of `n` symbols whose composition is the nearest to the
    /// distribution `p` (largest remainders: `⌊n p_a⌋`, then one more for the largest
    /// remainders until the sum is `n`).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for probabilities that are not non-negative with a
    /// positive sum, and as [`Ccdm::new`].
    pub fn from_distribution(p: &[f64], n: usize) -> Result<Ccdm> {
        let s: f64 = p.iter().sum();
        ensure(!p.is_empty() && s > 0.0 && p.iter().all(|v| *v >= 0.0), || {
            "the distribution must be non-negative with a positive sum".into()
        })?;
        let target: Vec<f64> = p.iter().map(|v| v / s * n as f64).collect();
        let mut c: Vec<usize> = target.iter().map(|t| t.floor() as usize).collect();
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| {
            (target[b] - target[b].floor()).total_cmp(&(target[a] - target[a].floor())).then(a.cmp(&b))
        });
        let missing = n - c.iter().sum::<usize>();
        for &a in order.iter().take(missing) {
            c[a] += 1;
        }
        Ccdm::new(c)
    }

    /// The composition.
    pub fn composition(&self) -> &[usize] {
        &self.composition
    }

    /// Symbols per block $n$.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Bits per block $k$.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The rate $k/n$ (bits per symbol).
    pub fn rate(&self) -> f64 {
        self.k as f64 / self.n as f64
    }

    /// The distribution of the symbols, $n_{a}/n$.
    pub fn distribution(&self) -> Vec<f64> {
        self.composition.iter().map(|&c| c as f64 / self.n as f64).collect()
    }

    /// The sequence of $n$ symbols of `bits` ($k$ bits, the rank, most significant
    /// first).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if `bits` does not have $k$ bits.
    pub fn matched(&self, bits: &[u8]) -> Result<Vec<usize>> {
        expect_len(bits.len(), self.k, "bits")?;
        let mut r = BigUint::from_bits(bits);
        let mut count = self.composition.clone();
        let mut total = self.total.clone();
        let mut left = self.n as u64;
        let mut out = Vec::with_capacity(self.n);
        for _ in 0..self.n {
            for (a, c) in count.iter_mut().enumerate() {
                if *c == 0 {
                    continue;
                }
                // the sequences that start with a
                let with_a = total.mul_small(*c as u64).div_small(left).0;
                if r < with_a {
                    out.push(a);
                    total = with_a;
                    *c -= 1;
                    left -= 1;
                    break;
                }
                r = r.sub(&with_a);
            }
        }
        Ok(out)
    }

    /// The $k$ bits of a sequence of $n$ symbols (its rank, most significant first); a
    /// symbol the composition has used up is replaced by the nearest available one.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if `symbols` does not have $n$ symbols.
    pub fn dematch(&self, symbols: &[usize]) -> Result<Vec<u8>> {
        expect_len(symbols.len(), self.n, "symbols")?;
        let mut r = BigUint::default();
        let mut count = self.composition.clone();
        let mut total = self.total.clone();
        let mut left = self.n as u64;
        for &s in symbols {
            let s = (0..count.len()).filter(|&a| count[a] > 0).min_by_key(|&a| (a.abs_diff(s), a)).unwrap_or(0);
            for &c in count.iter().take(s) {
                if c > 0 {
                    r = r.add(&total.mul_small(c as u64).div_small(left).0);
                }
            }
            total = total.mul_small(count[s] as u64).div_small(left).0;
            count[s] -= 1;
            left -= 1;
        }
        Ok(r.to_bits(self.k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn matching_is_a_bijection_onto_the_composition() {
        let dm = Ccdm::new(vec![3, 2, 1]).unwrap(); // 60 sequences, k = 5
        assert_eq!(dm.k(), 5);
        let mut seen = std::collections::HashSet::new();
        for r in 0..32u8 {
            let bits: Vec<u8> = (0..5).rev().map(|i| (r >> i) & 1).collect();
            let a = dm.matched(&bits).unwrap();
            assert_eq!(a.iter().filter(|&&s| s == 0).count(), 3);
            assert_eq!(dm.dematch(&a).unwrap(), bits);
            assert!(seen.insert(a));
        }
    }

    #[test]
    fn long_blocks_approach_the_entropy() {
        let p = [0.35, 0.3, 0.2, 0.15];
        let h: f64 = p.iter().map(|v: &f64| -v * v.log2()).sum();
        let mut rng = Rng::new(1);
        for n in [64, 512, 4096] {
            let dm = Ccdm::from_distribution(&p, n).unwrap();
            let bits: Vec<u8> = (0..dm.k()).map(|_| rng.bit()).collect();
            assert_eq!(dm.dematch(&dm.matched(&bits).unwrap()).unwrap(), bits);
            // rate loss below 4 log2(n)/n
            assert!(h - dm.rate() < 4.0 * (n as f64).log2() / n as f64, "{n}: {}", h - dm.rate());
        }
    }

    #[test]
    fn a_wrong_sequence_still_gives_k_bits() {
        let dm = Ccdm::from_distribution(&[0.5, 0.3, 0.2], 50).unwrap();
        let wrong = vec![2; 50];
        assert_eq!(dm.dematch(&wrong).unwrap().len(), dm.k());
    }
}
