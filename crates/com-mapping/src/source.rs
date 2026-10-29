//! Sources of bits and symbols (OptiCommPy `optic.comm.sources`).
//!
//! - [`bits`]: independent fair bits from a seeded generator;
//! - [`prbs`]: maximal-length pseudo-random binary sequences of the ITU-T O.150 family
//!   (orders 7, 9, 11, 15, 20, 23, 31), produced by a Fibonacci linear-feedback shift
//!   register: the sequence of order `n` has period `2ⁿ - 1` and contains every
//!   non-zero `n`-bit word exactly once per period;
//! - [`symbols`]: random symbols of a constellation with a given probability mass
//!   function, uniform by default, or Maxwell–Boltzmann shaped with
//!   [`maxwell_boltzmann`] (probabilistic shaping, `p(x) ∝ exp(-λ|x|²)`).
//!
//! # Examples
//! ```
//! use com_mapping::source;
//! use com_core::rng::Rng;
//! let b = source::bits(1000, &mut Rng::new(3));
//! assert!(b.iter().all(|&v| v <= 1));
//! let p = source::prbs(7, 127 * 2, 1).unwrap();
//! assert_eq!(&p[..127], &p[127..]);           // period 2⁷ - 1
//! assert_eq!(p[..127].iter().filter(|&&v| v == 1).count(), 64);
//! ```

use crate::constellation::Constellation;
use com_core::Result;
use com_core::rng::Rng;
use com_math::C64;

/// `n` independent fair bits (`0` or `1`).
pub fn bits(n: usize, rng: &mut Rng) -> Vec<u8> {
    (0..n).map(|_| rng.bit()).collect()
}

/// Feedback taps `(n, k)` of the PRBS polynomials `xⁿ + xᵏ + 1` (ITU-T O.150).
const PRBS_TAPS: [(u32, u32); 7] = [(7, 6), (9, 5), (11, 9), (15, 14), (20, 3), (23, 18), (31, 28)];

/// `length` bits of the PRBS of order `order`, from the register state `seed`.
///
/// The register is a Fibonacci LFSR ([`com_math::sequences::lfsr`]) with the polynomial
/// `xⁿ + xᵏ + 1` of the order;
/// `seed` gives its initial content (its `n` low bits; an all-zero register is replaced
/// by 1, since it would lock the sequence at zero). The output bit is the feedback bit.
///
/// # Errors
/// [`com_core::Error::Config`] for an order outside `{7, 9, 11, 15, 20, 23, 31}`.
pub fn prbs(order: u32, length: usize, seed: u64) -> Result<Vec<u8>> {
    let (n, k) = PRBS_TAPS
        .iter()
        .copied()
        .find(|&(n, _)| n == order)
        .ok_or_else(|| com_core::Error::Config(format!("PRBS order {order} not in 7, 9, 11, 15, 20, 23, 31")))?;
    let (n, k) = (n as usize, k as usize);
    let reg = match seed & ((1u64 << n) - 1) {
        0 => 1,
        r => r,
    };
    // the register holds the last n outputs, the newest in its bit 0: the output bit
    // b(t) = b(t - n) ⊕ b(t - k) is the LFSR x(i + n) = x(i) ⊕ x(i + n - k)
    let init: Vec<u8> = (0..n).map(|i| ((reg >> (n - 1 - i)) & 1) as u8).collect();
    Ok(com_math::sequences::lfsr(&[0, n - k], n, &init, n + length).split_off(n))
}

/// Maxwell–Boltzmann probabilities `p_i ∝ exp(-λ |x_i|²)` over the points of `c`.
///
/// `λ = 0` gives the uniform distribution; larger `λ` favour the low-energy points
/// (probabilistic constellation shaping).
pub fn maxwell_boltzmann(c: &Constellation, lambda: f64) -> Vec<f64> {
    let w: Vec<f64> = c.points().iter().map(|p| (-lambda * p.norm_sqr()).exp()).collect();
    let s: f64 = w.iter().sum();
    w.iter().map(|v| v / s).collect()
}

/// `n` random symbols of `c` drawn with the probabilities `px` (uniform if `None`,
/// normalised to a sum of 1 otherwise, as [`Constellation::priors`]).
///
/// Returns the indices (Gray labels) and the symbols.
///
/// # Errors
/// As [`Constellation::priors`].
pub fn symbols(c: &Constellation, n: usize, px: Option<&[f64]>, rng: &mut Rng) -> Result<(Vec<usize>, Vec<C64>)> {
    let m = c.order();
    let idx: Vec<usize> = if px.is_none() {
        (0..n).map(|_| rng.below(m)).collect()
    } else {
        let mut cdf = Vec::with_capacity(m);
        let mut acc = 0.0;
        for v in c.priors(px)? {
            acc += v;
            cdf.push(acc);
        }
        (0..n)
            .map(|_| {
                let u = rng.uniform() * acc;
                cdf.partition_point(|&c| c <= u).min(m - 1)
            })
            .collect()
    };
    let x = idx.iter().map(|&i| c.points()[i]).collect();
    Ok((idx, x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prbs_periods_and_balance() {
        for &(n, _) in &PRBS_TAPS[..4] {
            let period = (1usize << n) - 1;
            let p = prbs(n, 2 * period, 0xACE1).unwrap();
            assert_eq!(&p[..period], &p[period..], "order {n}");
            let ones = p[..period].iter().filter(|&&v| v == 1).count();
            assert_eq!(ones, period.div_ceil(2), "order {n}");
        }
        assert!(prbs(8, 10, 1).is_err());
    }

    #[test]
    fn shaped_symbols_follow_the_distribution() {
        let c = Constellation::qam(16).unwrap().normalized();
        let px = maxwell_boltzmann(&c, 1.0);
        let (idx, _) = symbols(&c, 200_000, Some(&px), &mut Rng::new(5)).unwrap();
        for i in [0usize, 5, 15] {
            let f = idx.iter().filter(|&&k| k == i).count() as f64 / idx.len() as f64;
            assert!((f - px[i]).abs() < 0.005, "point {i}: {f} vs {}", px[i]);
        }
    }
}
