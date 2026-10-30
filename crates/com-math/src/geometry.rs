//! Vectors of the three-dimensional Euclidean space, as `[f64; 3]`: the scalar product,
//! the difference, the norm and the direction of a vector.
//!
//! The geometry of the antenna arrays of the CDL channel and of the indoor optical
//! channel (positions, directions, orientations) is written with these.
//!
//! # Examples
//! ```
//! use com_math::geometry::{difference, dot, norm, unit};
//! let (a, b) = ([1.0, 2.0, 2.0], [1.0, 0.0, 0.0]);
//! assert_eq!(dot(a, b), 1.0);
//! assert_eq!(norm(a), 3.0);
//! assert_eq!(difference(a, b), [0.0, 2.0, 2.0]);
//! assert_eq!(unit([0.0, 0.0, -2.0]), Some([0.0, 0.0, -1.0]));
//! ```

/// A vector of the three-dimensional space.
pub type Vec3 = [f64; 3];

/// The scalar product $a \cdot b = a_{0} b_{0} + a_{1} b_{1} + a_{2} b_{2}$.
#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The difference $a - b$.
#[inline]
pub fn difference(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// The Euclidean norm $\lVert a \rVert = \sqrt{a \cdot a}$.
#[inline]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// The direction $a / \lVert a \rVert$ of `a`; `None` for a zero or non-finite norm.
pub fn unit(a: Vec3) -> Option<Vec3> {
    let n = norm(a);
    (n > 0.0 && n.is_finite()).then(|| [a[0] / n, a[1] / n, a[2] / n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_direction_has_unit_norm_and_a_zero_vector_has_none() {
        let u = unit([3.0, -4.0, 12.0]).unwrap();
        assert!((norm(u) - 1.0).abs() < 1e-15);
        assert_eq!(u, [3.0 / 13.0, -4.0 / 13.0, 12.0 / 13.0]);
        assert_eq!(unit([0.0; 3]), None);
        assert_eq!(unit([f64::INFINITY, 0.0, 0.0]), None);
        assert_eq!(unit([f64::NAN, 0.0, 0.0]), None);
    }
}
