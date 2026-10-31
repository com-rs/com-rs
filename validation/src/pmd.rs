//! The PMD vectors of a link of `PmdLink` (J. P. Gordon and H. Kogelnik, *Proc. Natl. Acad.
//! Sci.* 97(9), 2000, eqs. (5.18)–(5.19)), with the Pauli matrices of that paper, for the
//! PMD tests.

use comrs::C64;
use comrs::channel::optical::pmd::{Jones, PmdLink};
use comrs::core::signal::DualPol;

/// The Pauli matrices of Gordon & Kogelnik (row-major): $\sigma_{1} =
/// \operatorname{diag}(1, -1)$, $\sigma_{2}$ the exchange, $\sigma_{3}$ with $\mp j$.
pub fn pauli() -> [[C64; 4]; 3] {
    let (o, l, j) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 1.0));
    [[l, o, o, -l], [o, l, l, o], [o, -j, j, o]]
}

/// The trace of the product of two 2 × 2 matrices.
pub fn trace_product(a: &[C64; 4], b: &[C64; 4]) -> C64 {
    a[0] * b[0] + a[1] * b[2] + a[2] * b[1] + a[3] * b[3]
}

/// The vector $\vec{v}$ of a traceless Hermitian matrix $H = \frac{1}{2} \vec{v} \cdot
/// \vec{\sigma}$, and the largest deviation from Hermitian and traceless.
pub fn stokes_of(h: &Jones) -> ([f64; 3], f64) {
    let m = &h.0;
    let v = pauli().map(|s| trace_product(m, &s));
    let defect = [v[0].im, v[1].im, v[2].im, (m[0] + m[3]).norm(), (m[1] - m[2].conj()).norm()]
        .iter()
        .fold(0.0_f64, |a, b| a.max(b.abs()));
    ([v[0].re, v[1].re, v[2].re], defect)
}

fn scale(j: &Jones, s: C64) -> Jones {
    Jones(j.0.map(|v| v * s))
}

/// The input and output PMD vectors (s) of `link` at the angular frequency `w` (eqs.
/// (5.18)–(5.19), $U_{\omega}$ by a central difference over 1 MHz) and the defect of the
/// Hermitian expansions.
pub fn pmd_vectors(link: &PmdLink, w: f64) -> ([f64; 3], [f64; 3], f64) {
    let dw = 2.0 * std::f64::consts::PI * 1e6;
    let r = link.lumped_response(&[w - dw, w, w + dw]);
    let (u, du) = (r[1], Jones(std::array::from_fn(|k| (r[2].0[k] - r[0].0[k]) * (0.5 / dw))));
    let j = C64::new(0.0, 1.0);
    let (tin, d1) = stokes_of(&scale(&u.adjoint().mul(&du), j));
    let (tout, d2) = stokes_of(&scale(&du.mul(&u.adjoint()), j));
    (tin, tout, d1.max(d2))
}

/// The Stokes vector $\hat{s} = \langle s | \vec{\sigma} | s \rangle$ of a unit Jones vector.
pub fn stokes(x: C64, y: C64) -> [f64; 3] {
    pauli().map(|s| (x.conj() * (s[0] * x + s[1] * y) + y.conj() * (s[2] * x + s[3] * y)).re)
}

/// The unit Jones vector of the unit Stokes vector `s`.
pub fn jones_of(s: [f64; 3]) -> (C64, C64) {
    let theta = s[0].clamp(-1.0, 1.0).acos();
    (C64::new((theta / 2.0).cos(), 0.0), C64::from_polar((theta / 2.0).sin(), s[2].atan2(s[1])))
}

/// The dot product of two vectors of three components.
pub fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The first two moments of the intensity of a dual-polarisation field sampled at `fs`
/// over a block centred on its middle: the centre of gravity (s) and the rms width (s).
pub fn moments(u: &DualPol, fs: f64) -> (f64, f64) {
    let n = u.x.len();
    let (mut m0, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for k in 0..n {
        let p = u.x[k].norm_sqr() + u.y[k].norm_sqr();
        let t = (k as f64 - n as f64 / 2.0) / fs;
        m0 += p;
        m1 += t * p;
        m2 += t * t * p;
    }
    let mean = m1 / m0;
    (mean, (m2 / m0 - mean * mean).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_jones_vector_of_a_stokes_vector_has_that_stokes_vector() {
        for s in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.6, 0.0, 0.8], [0.36, -0.48, 0.8]] {
            let (x, y) = jones_of(s);
            assert!((dot(&stokes(x, y), &s) - 1.0).abs() < 1e-12, "{s:?}");
        }
    }
}
