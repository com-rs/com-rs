//! Reproducible random numbers.
//!
//! The generator is **xoshiro256\*\*** (Blackman & Vigna, 2018) seeded through
//! **SplitMix64**: fast, statistically excellent for simulation, and identical across
//! platforms, so that a stream drawn here is the same as one drawn from the same seed by
//! another implementation of the generator (a learned receiver trained in Python, say).
//!
//! Monte Carlo campaigns never share a generator between trials: [`stream`] derives an
//! independent generator from `(base_seed, experiment, condition, trial)`, so a trial is
//! reproducible on its own, whatever the number of threads and the order in which the
//! trials run (see the crate `com-sim`).
//!
//! # Mathematics
//!
//! - [`Rng::uniform`] returns `(next_u64() >> 11) · 2⁻⁵³ ∈ [0, 1)`;
//! - [`Rng::gaussian`] is the Box–Muller transform `√(-2 ln(1-u₁)) cos(2π u₂)` (the sine
//!   branch is dropped, so that one call consumes exactly two uniforms);
//! - [`Rng::complex_gaussian`] is circular with `E|z|² = 1`;
//! - [`Rng::gamma`] is the method of Marsaglia & Tsang (2000), a rejection from a
//!   transformed Gaussian;
//! - [`stream`]: `seed = sm(sm(sm(base ⊕ fnv1a(experiment)) ⊕ condition) ⊕ trial)` with
//!   `sm` one SplitMix64 step; [`member`]: `seed ⊕ index · φ`, the members of a family of
//!   deterministic sequences shared by a transmitter and its receiver.
//!
//! # Examples
//! ```
//! use com_core::rng::{stream, Rng};
//! let mut a = Rng::new(42);
//! let mut b = Rng::new(42);
//! assert_eq!(a.next_u64(), b.next_u64());
//! // two trials of the same condition get independent streams
//! let x = stream(1, "ber", 3, 0).uniform();
//! let y = stream(1, "ber", 3, 1).uniform();
//! assert_ne!(x, y);
//! assert_eq!(x, stream(1, "ber", 3, 0).uniform());
//! ```
//!
//! # References
//!
//! 1. D. Blackman and S. Vigna, "Scrambled linear pseudorandom number generators,"
//!    *ACM Trans. Math. Softw.*, 47(4), 2021 (arXiv:1805.01407, 2018).
//! 2. G. Marsaglia and W. W. Tsang, "A simple method for generating gamma variables,"
//!    *ACM Trans. Math. Softw.*, 26(3), 363–372, 2000.

use crate::C64;

/// Advance a SplitMix64 state and return its output.
#[inline]
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// FNV-1a 64-bit hash of a byte string (hashes the experiment names of [`stream`]).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The xoshiro256\*\* generator (256 bits of state).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// A generator seeded from one `u64` through SplitMix64.
    pub fn new(seed: u64) -> Rng {
        let mut st = seed;
        let s = [splitmix64(&mut st), splitmix64(&mut st), splitmix64(&mut st), splitmix64(&mut st)];
        Rng { s }
    }

    /// The 256 bits of the state, to save a generator and resume it with
    /// [`Rng::from_state`] (the pickling of the Python generator).
    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    /// The generator of a state saved by [`Rng::state`]: it continues the draws of the
    /// saved generator.
    ///
    /// # Errors
    /// [`crate::Error::Config`] for the all-zero state, a fixed point of xoshiro256\*\*
    /// that no seed gives.
    pub fn from_state(s: [u64; 4]) -> crate::Result<Rng> {
        crate::ensure(s != [0; 4], || "the all-zero state is not a state of xoshiro256**".into())?;
        Ok(Rng { s })
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform on `[0, 1)` with 53 random bits.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform integer in `0..n` (`n ≥ 1`).
    ///
    /// # Panics
    /// In debug builds, for `n = 0` (an empty range has no element: release builds
    /// return 0 rather than check it in the inner loops).
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n >= 1, "Rng::below(0): the range 0..0 is empty");
        ((self.uniform() * n as f64) as usize).min(n.saturating_sub(1))
    }

    /// One fair random bit.
    #[inline]
    pub fn bit(&mut self) -> u8 {
        (self.next_u64() >> 63) as u8
    }

    /// Standard Gaussian `N(0, 1)`.
    #[inline]
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * (1.0 - u1).ln()).sqrt();
        r * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Circular complex Gaussian with `E|z|² = 1` (each component of variance ½).
    #[inline]
    pub fn complex_gaussian(&mut self) -> C64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * (1.0 - u1).ln()).sqrt() * std::f64::consts::FRAC_1_SQRT_2;
        let (s, c) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        C64::new(r * c, r * s)
    }

    /// A gamma variable of shape `shape` and unit scale (mean `shape`), by the method of
    /// Marsaglia & Tsang (2000): for a shape below 1, the draw of shape `shape + 1` times
    /// $U^{1/\text{shape}}$. A non-positive or non-finite shape returns `NaN`.
    ///
    /// # Examples
    /// ```
    /// use com_core::rng::Rng;
    /// let mut r = Rng::new(1);
    /// let n = 100_000;
    /// let mean = (0..n).map(|_| r.gamma(2.5)).sum::<f64>() / n as f64;
    /// assert!((mean - 2.5).abs() < 0.03);
    /// ```
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if !(shape > 0.0 && shape.is_finite()) {
            return f64::NAN;
        }
        if shape < 1.0 {
            let u = self.uniform();
            return self.gamma(shape + 1.0) * (1.0 - u).powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.gaussian();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = 1.0 - self.uniform();
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// A uniform random permutation of `v` in place (Fisher–Yates, Durstenfeld's
    /// version): for $i = n - 1, \dots, 1$, swap $v_{i}$ and $v_{j}$ with $j$ uniform in
    /// $\\{0, \dots, i\\}$ ([`Rng::below`]), $n - 1$ draws in all.
    ///
    /// # Examples
    /// ```
    /// use com_core::rng::Rng;
    /// let mut v: Vec<usize> = (0..10).collect();
    /// Rng::new(3).shuffle(&mut v);
    /// let mut sorted = v.clone();
    /// sorted.sort_unstable();
    /// assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    /// ```
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }

    /// `n` circular complex Gaussian samples of variance `sigma2` (`E|z|² = σ²`).
    pub fn complex_noise(&mut self, n: usize, sigma2: f64) -> Vec<C64> {
        let s = sigma2.max(0.0).sqrt();
        (0..n).map(|_| self.complex_gaussian() * s).collect()
    }
}

/// The generator every random block receives (`&mut SimRng`).
///
/// A concrete type rather than a generic parameter, so that the block traits stay
/// object-safe and every call is monomorphic. It is [`Rng`] (xoshiro256\*\*), see
/// `docs/adr/0002-keep-xoshiro-rng.md`.
pub type SimRng = Rng;

/// An independent, reproducible generator for one batch of frames.
///
/// Keyed by the batch index, never by the worker, so that results do not depend on the
/// number of threads. It is the stream `(master_seed, "", 0, batch)` of [`stream`].
///
/// # Examples
/// ```
/// use com_core::rng::batch_rng;
/// assert_eq!(batch_rng(42, 3).next_u64(), batch_rng(42, 3).next_u64());
/// assert_ne!(batch_rng(42, 3).next_u64(), batch_rng(42, 4).next_u64());
/// ```
pub fn batch_rng(master_seed: u64, batch: u64) -> SimRng {
    stream(master_seed, "", 0, batch)
}

/// The generator of the member `index` of a family of deterministic sequences drawn from
/// `seed` (the pilots of the REs of a grid, the phase sequences of SLM):
/// `Rng::new(seed ⊕ index · φ)`, $\varphi$ the 64-bit golden ratio of SplitMix64. Unlike
/// [`stream`] it is a plain function of the two numbers, which the transmitter and the
/// receiver share.
///
/// # Examples
/// ```
/// use com_core::rng::member;
/// assert_eq!(member(7, 3).next_u64(), member(7, 3).next_u64());
/// assert_ne!(member(7, 3).next_u64(), member(7, 4).next_u64());
/// ```
pub fn member(seed: u64, index: u64) -> Rng {
    Rng::new(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// An independent generator for `(base_seed, experiment, condition, trial)`.
pub fn stream(base_seed: u64, experiment: &str, condition: u64, trial: u64) -> Rng {
    let mut st = base_seed ^ fnv1a64(experiment.as_bytes());
    let a = splitmix64(&mut st);
    let mut st = a ^ condition;
    let b = splitmix64(&mut st);
    let mut st = b ^ trial;
    let c = splitmix64(&mut st);
    Rng::new(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "is empty")]
    fn below_zero_is_a_broken_invariant_in_debug_builds() {
        let _ = Rng::new(1).below(0);
    }

    #[test]
    fn a_saved_state_resumes_the_draws() {
        let mut a = Rng::new(42);
        a.next_u64();
        let mut b = Rng::from_state(a.state()).unwrap();
        assert_eq!((0..5).map(|_| a.next_u64()).collect::<Vec<_>>(), (0..5).map(|_| b.next_u64()).collect::<Vec<_>>());
        assert!(Rng::from_state([0; 4]).is_err());
    }

    #[test]
    fn reference_values() {
        // xoshiro256** seeded by SplitMix64 from 0 (reference C implementation)
        assert_eq!(Rng::new(0).next_u64(), 0x99EC_5F36_CB75_F2B4);
        assert_eq!(fnv1a64(b"a"), 0xAF63_DC4C_8601_EC8C);
    }

    #[test]
    fn moments() {
        let mut r = Rng::new(7);
        let n = 200_000;
        let (mut m, mut v, mut b) = (0.0, 0.0, 0u64);
        for _ in 0..n {
            let z = r.complex_gaussian();
            m += z.re;
            v += z.norm_sqr();
            b += r.bit() as u64;
        }
        assert!((m / n as f64).abs() < 0.01);
        assert!((v / n as f64 - 1.0).abs() < 0.01);
        assert!((b as f64 / n as f64 - 0.5).abs() < 0.01);
    }
}
