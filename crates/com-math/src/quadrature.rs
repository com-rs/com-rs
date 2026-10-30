//! Quadrature: the composite Simpson rule ([`simpson`], [`try_simpson`],
//! [`simpson_weight`]) and the Gauss rules ([`gauss_hermite`], [`gauss_legendre`], and
//! [`gaussian_nodes`], the nodes of a Gaussian expectation), re-exported at the root.

use crate::C64;

/// The weight below which a node of [`gaussian_nodes`] is left out.
const NEGLIGIBLE_WEIGHT: f64 = 1e-300;

/// Composite Simpson's rule $\int_{a}^{b} f(t)\, dt$ on `n` intervals (`n` even, at
/// least 2), exact for cubics.
///
/// # Examples
/// ```
/// use com_math::simpson;
/// assert!((simpson(|t| t * t * t, 0.0, 2.0, 2) - 4.0).abs() < 1e-15);
/// assert!((simpson(f64::sin, 0.0, std::f64::consts::PI, 200) - 2.0).abs() < 1e-8);
/// ```
pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    try_simpson(|t| Ok::<f64, std::convert::Infallible>(f(t)), a, b, n).unwrap_or_else(|never| match never {})
}

/// [`simpson`] of a fallible integrand: the first error stops the quadrature.
///
/// # Errors
/// The first error of `f`.
///
/// # Examples
/// ```
/// use com_math::try_simpson;
/// let v: Result<f64, String> = try_simpson(|t| Ok(t * t), 0.0, 3.0, 4);
/// assert!((v.unwrap() - 9.0).abs() < 1e-14);
/// let e: Result<f64, String> = try_simpson(|t| if t > 1.0 { Err("beyond 1".into()) } else { Ok(t) }, 0.0, 2.0, 4);
/// assert!(e.is_err());
/// ```
pub fn try_simpson<E>(f: impl Fn(f64) -> Result<f64, E>, a: f64, b: f64, n: usize) -> Result<f64, E> {
    let n = n.max(2).div_ceil(2) * 2;
    let h = (b - a) / n as f64;
    let mut acc = f(a)? + f(b)?;
    for k in 1..n {
        acc += simpson_weight(k, n) * f(a + k as f64 * h)?;
    }
    Ok(acc * h / 3.0)
}

/// The weight of node `k` of Simpson's rule on `n` intervals (`n` even), before the
/// factor $h/3$: 1 at the ends, 4 at the odd nodes, 2 at the even inner ones.
#[inline]
pub fn simpson_weight(k: usize, n: usize) -> f64 {
    if k == 0 || k == n {
        1.0
    } else if k % 2 == 1 {
        4.0
    } else {
        2.0
    }
}

/// Nodes and weights of the `n`-point Gauss–Hermite rule,
/// `∫ f(t) e^{-t²} dt ≈ Σ_k w_k f(t_k)`, exact for polynomials of degree `2n - 1`.
///
/// The nodes are the roots of the Hermite polynomial `H_n`, found by Newton's method
/// on the orthonormal recurrence (Numerical Recipes, `gauher`); they come in
/// decreasing order. An expectation over `X ~ N(0, s²)` is
/// `E f(X) ≈ Σ_k w_k f(√2 s t_k) / √π`.
///
/// # Examples
/// ```
/// use com_math::gauss_hermite;
/// let (t, w) = gauss_hermite(20);
/// let pi = std::f64::consts::PI;
/// assert!((w.iter().sum::<f64>() - pi.sqrt()).abs() < 1e-12);
/// // E X⁴ = 3 for X ~ N(0, 1)
/// let m4: f64 = t.iter().zip(&w).map(|(t, w)| w * (2f64.sqrt() * t).powi(4)).sum::<f64>() / pi.sqrt();
/// assert!((m4 - 3.0).abs() < 1e-10);
/// ```
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    const PIM4: f64 = 0.751_125_544_464_942_5; // π^(-1/4)
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    let mut z = 0.0f64;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.85575 * (2.0 * nf + 1.0).powf(-0.16667),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut pp = 1.0;
        for _ in 0..200 {
            let (mut p1, mut p2) = (PIM4, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = z * (2.0 / (jf + 1.0)).sqrt() * p2 - (jf / (jf + 1.0)).sqrt() * p3;
            }
            pp = (2.0 * nf).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 1e-15 * z.abs().max(1.0) {
                break;
            }
        }
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// Nodes and weights of the `n`-point Gauss–Legendre rule on $[-1, 1]$,
/// `∫ f(t) dt ≈ Σ_k w_k f(t_k)`, exact for polynomials of degree `2n - 1`.
///
/// The nodes are the roots of the Legendre polynomial $P_{n}$, found by Newton's method
/// on its recurrence (Numerical Recipes, `gauleg`); they come in increasing order. On
/// $[a, b]$ the nodes are $\frac{a + b}{2} + \frac{b - a}{2} t_{k}$ and the weights
/// $\frac{b - a}{2} w_{k}$.
///
/// # Examples
/// ```
/// use com_math::gauss_legendre;
/// let (t, w) = gauss_legendre(5);
/// assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
/// // exact for t⁸: 2/9
/// let m8: f64 = t.iter().zip(&w).map(|(t, w)| w * t.powi(8)).sum();
/// assert!((m8 - 2.0 / 9.0).abs() < 1e-14);
/// assert!((t[4] - 0.906_179_845_938_664).abs() < 1e-15);
/// ```
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut pp = 1.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf + 1.0) * z * p2 - jf * p3) / (jf + 1.0);
            }
            pp = nf * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// Quadrature nodes `(w, weight)` of Gaussian noise, `E f(W) ≈ Σ weight f(w)`, from the
/// `n`-point [`gauss_hermite`] rule: `W ~ CN(0, σ²)` by the product rule on the two
/// dimensions (`complex`), or `W ~ N(0, σ²)` on the real axis. The nodes of negligible
/// weight (below `1e-300`) are left out.
///
/// # Examples
/// ```
/// use com_math::gaussian_nodes;
/// // E|W|² = σ² for W ~ CN(0, σ²); E W² = σ² for W ~ N(0, σ²)
/// for complex in [true, false] {
///     let m2: f64 = gaussian_nodes(0.3, complex, 16).iter().map(|(w, p)| p * w.norm_sqr()).sum();
///     assert!((m2 - 0.3).abs() < 1e-12);
/// }
/// ```
pub fn gaussian_nodes(noise_var: f64, complex: bool, n: usize) -> Vec<(C64, f64)> {
    let (t, a) = gauss_hermite(n.max(2));
    let pi = std::f64::consts::PI;
    if complex {
        let s = noise_var.sqrt();
        t.iter()
            .zip(&a)
            .flat_map(|(tr, ar)| t.iter().zip(&a).map(move |(ti, ai)| (C64::new(s * tr, s * ti), ar * ai / pi)))
            .filter(|(_, w)| *w > NEGLIGIBLE_WEIGHT)
            .collect()
    } else {
        let s = (2.0 * noise_var).sqrt();
        t.iter().zip(&a).map(|(t, a)| (C64::new(s * t, 0.0), a / pi.sqrt())).collect()
    }
}
