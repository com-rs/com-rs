//! The algebraic decoding shared by the BCH and Reed–Solomon codes: the
//! Berlekamp–Massey algorithm and the Chien search over $\mathrm{GF}(2^{m})$.
//!
//! The acceptance tests differ between the two codes and stay with them: the BCH
//! decoder accepts a locator of degree at most $t$ with as many roots as its degree,
//! the Reed–Solomon decoder a locator whose length $L$ satisfies $2L \le 2t - s$ and
//! equals its degree, with as many roots as the degree of $\Lambda(x)\Gamma(x)$.
//!
//! # References
//!
//! 1. J. L. Massey, "Shift-register synthesis and BCH decoding," *IEEE Trans. Inf.
//!    Theory*, 15(1), 122–127, 1969.
//! 2. R. T. Chien, "Cyclic decoding procedures for Bose-Chaudhuri-Hocquenghem codes,"
//!    *IEEE Trans. Inf. Theory*, 10(4), 357–363, 1964.

use com_math::gf2m::Field;

/// The shortest linear recurrence of the syndromes `syn` (Massey 1969): the connection
/// polynomial $\Lambda(x)$, lowest degree first and without trailing zeros, and the
/// length $L$ of the recurrence.
pub(crate) fn berlekamp_massey(f: &Field, syn: &[u32]) -> (Vec<u32>, usize) {
    let mut lambda = vec![1u32];
    let mut b_poly = vec![1u32];
    let (mut l, mut shift, mut b_disc) = (0usize, 1usize, 1u32);
    for step in 0..syn.len() {
        let mut d = syn[step];
        for i in 1..=l.min(lambda.len() - 1) {
            d ^= f.mul(lambda[i], syn[step - i]);
        }
        if d == 0 {
            shift += 1;
            continue;
        }
        let coef = f.div(d, b_disc);
        let mut next = lambda.clone();
        if next.len() < b_poly.len() + shift {
            next.resize(b_poly.len() + shift, 0);
        }
        for (i, &bv) in b_poly.iter().enumerate() {
            next[i + shift] ^= f.mul(coef, bv);
        }
        if 2 * l <= step {
            b_poly = lambda;
            l = step + 1 - l;
            b_disc = d;
            shift = 1;
        } else {
            shift += 1;
        }
        lambda = next;
    }
    while lambda.len() > 1 && lambda.last() == Some(&0) {
        lambda.pop();
    }
    (lambda, l)
}

/// $p(\alpha^{e}) = \sum_{d} p_{d} \alpha^{d e}$ (coefficients lowest degree first).
pub(crate) fn eval_at_power(f: &Field, p: &[u32], e: usize) -> u32 {
    p.iter().enumerate().fold(0u32, |acc, (d, &a)| acc ^ f.mul(a, f.pow(d * e)))
}

/// The Chien search: the positions $i < n$ whose locator $Z_{i} = \alpha^{e_{i}}$
/// (`locator_exp(i)` gives $e_{i}$, reduced modulo $2^{m} - 1$) is the inverse of a root
/// of $p$, $p(Z_{i}^{-1}) = 0$, in increasing order.
pub(crate) fn chien_search(f: &Field, p: &[u32], n: usize, locator_exp: impl Fn(usize) -> usize) -> Vec<usize> {
    let order = f.order();
    (0..n).filter(|&i| eval_at_power(f, p, (order - locator_exp(i)) % order) == 0).collect()
}
