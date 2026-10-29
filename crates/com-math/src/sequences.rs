//! Binary pseudo-random sequences: linear-feedback shift registers and the length-31
//! Gold sequence of 3GPP.
//!
//! # Model
//!
//! A Fibonacci register of order $r$ produces $x(i + r) = \sum_{t \in T} x(i + t) \bmod 2$
//! from its first $r$ bits ([`lfsr`]): every m-sequence and Gold sequence of the
//! standards is one or a sum of two. The pseudo-random sequence of 3GPP (TS 38.211,
//! Sec. 5.2.1; TS 36.211, Sec. 7.2) is
//!
//! $$
//! c(n) = \left(x_{1}(n + N_{c}) + x_{2}(n + N_{c})\right) \bmod 2, \qquad N_{c} = 1600,
//! $$
//!
//! with $x_{1}(n+31) = (x_{1}(n+3) + x_{1}(n)) \bmod 2$, $x_{1}(0) = 1$,
//! $x_{1}(1..30) = 0$, and $x_{2}(n+31) = (x_{2}(n+3) + x_{2}(n+2) + x_{2}(n+1) +
//! x_{2}(n)) \bmod 2$ initialised by $c_{\mathrm{init}} = \sum_{i} x_{2}(i) 2^{i}$
//! ([`gold_3gpp`]): the scrambling of the coded bits and the sequences of the reference
//! signals.
//!
//! # Examples
//! ```
//! use com_math::sequences::{gold_3gpp, lfsr};
//! // x³ + x + 1: the m-sequence of period 7
//! assert_eq!(lfsr(&[0, 1], 3, &[1, 0, 0], 10), vec![1, 0, 0, 1, 0, 1, 1, 1, 0, 0]);
//! assert_eq!(gold_3gpp(0, 8).len(), 8);
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
/// use com_math::sequences::lfsr;
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

/// The first `n` bits of the Gold sequence $c$ of 3GPP with the initial value `c_init`
/// (31 bits; see the module).
pub fn gold_3gpp(c_init: u32, n: usize) -> Vec<u8> {
    const NC: usize = 1600;
    let len = n + NC + 31;
    let init2: Vec<u8> = (0..31).map(|i| ((c_init >> i) & 1) as u8).collect();
    let x1 = lfsr(&[0, 3], 31, &[1], len);
    let x2 = lfsr(&[0, 1, 2, 3], 31, &init2, len);
    (0..n).map(|i| x1[i + NC] ^ x2[i + NC]).collect()
}
