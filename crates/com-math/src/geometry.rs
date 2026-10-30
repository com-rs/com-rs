//! Vectors of the three-dimensional Euclidean space, as `[f64; 3]`: the scalar product,
//! the difference, the norm and the direction of a vector, and the unit vector of
//! spherical angles.
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
//!
//! # Spherical angles
//!
//! [`from_spherical`] takes the zenith (polar) angle $\theta$ from $+z$ and the azimuth
//! $\phi$ from $+x$ towards $+y$, in radians (ISO 80000-2; the angles of 3GPP TR 38.901,
//! (7.5-23)):
//!
//! $$
//! \hat{r}(\theta, \phi) = (\sin\theta \cos\phi,\ \sin\theta \sin\phi,\ \cos\theta)^{T}.
//! $$
//!
//! An elevation $\epsilon$ above the $xy$ plane is the zenith $\theta = \pi/2 - \epsilon$.

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

/// The unit vector $\hat{r}(\theta, \phi) = (\sin\theta \cos\phi, \sin\theta \sin\phi,
/// \cos\theta)$ of the zenith `zenith_rad` (from $+z$) and the azimuth `azimuth_rad` (from
/// $+x$ towards $+y$), see the module.
///
/// # Examples
/// ```
/// use com_math::geometry::from_spherical;
/// use std::f64::consts::FRAC_PI_2;
/// assert_eq!(from_spherical(0.0, 1.0), [0.0, 0.0, 1.0]);
/// let y = from_spherical(FRAC_PI_2, FRAC_PI_2);
/// assert!((y[1] - 1.0).abs() < 1e-15 && y[0].abs() < 1e-15 && y[2].abs() < 1e-15);
/// ```
#[inline]
pub fn from_spherical(zenith_rad: f64, azimuth_rad: f64) -> Vec3 {
    let (st, ct) = zenith_rad.sin_cos();
    let (sp, cp) = azimuth_rad.sin_cos();
    [st * cp, st * sp, ct]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spherical_direction_has_unit_norm() {
        for (t, p) in [(0.3, -2.0), (1.2, 0.7), (3.0, 3.1)] {
            let r = from_spherical(t, p);
            assert!((norm(r) - 1.0).abs() < 1e-15);
            assert!((r[2] - f64::cos(t)).abs() < 1e-15 && (r[1].atan2(r[0]) - p).abs() < 1e-12);
        }
    }

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
