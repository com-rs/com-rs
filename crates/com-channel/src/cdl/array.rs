//! Antenna arrays and element patterns of TR 38.901 Section 7.3.

use com_math::db2lin;

/// The radiation power pattern of an antenna element.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// $A = 1$ in every direction.
    Isotropic,
    /// TR 38.901 Table 7.3-1: $A_{\mathrm{dB}}(\theta, \phi) = G_{E, \max} -
    /// \min\\{-(A_{V}(\theta) + A_{H}(\phi)), A_{\max}\\}$ with
    /// $A_{V} = -\min\\{12 ((\theta - 90°) / \theta_{3\mathrm{dB}})^{2}, \mathrm{SLA}_{V}\\}$,
    /// $A_{H} = -\min\\{12 (\phi / \phi_{3\mathrm{dB}})^{2}, A_{\max}\\}$,
    /// $\theta_{3\mathrm{dB}} = \phi_{3\mathrm{dB}} = 65°$, $\mathrm{SLA}_{V} = A_{\max} =
    /// 30$ dB, $G_{E, \max} = 8$ dBi, boresight along $x$ ($\theta = 90°$, $\phi = 0$).
    Sectored,
}

impl Pattern {
    /// The power gain $A$ (dB) in the direction of zenith `zenith` and azimuth
    /// `azimuth` (degrees).
    ///
    /// # Examples
    /// ```
    /// use com_channel::cdl::Pattern;
    /// assert_eq!(Pattern::Sectored.gain_db(90.0, 0.0), 8.0);
    /// assert!((Pattern::Sectored.gain_db(90.0, 65.0) - (8.0 - 12.0)).abs() < 1e-12);
    /// assert_eq!(Pattern::Isotropic.gain_db(10.0, 120.0), 0.0);
    /// ```
    pub fn gain_db(self, zenith: f64, azimuth: f64) -> f64 {
        match self {
            Pattern::Isotropic => 0.0,
            Pattern::Sectored => {
                let phi = (azimuth + 180.0).rem_euclid(360.0) - 180.0;
                let a_v = -(12.0 * ((zenith - 90.0) / 65.0).powi(2)).min(30.0);
                let a_h = -(12.0 * (phi / 65.0).powi(2)).min(30.0);
                8.0 - (-(a_v + a_h)).min(30.0)
            }
        }
    }
}

/// An antenna array: the position of every element (wavelengths, global coordinates),
/// its slant angle (degrees) and the common element pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    /// Positions $\bar{d} / \lambda$ of the elements, $(x, y, z)$.
    pub positions: Vec<[f64; 3]>,
    /// Slant angle $\zeta$ of every element (degrees): 0 vertical, $\pm 45$ for a
    /// dual-polarised pair.
    pub slants: Vec<f64>,
    /// Element pattern.
    pub pattern: Pattern,
}

impl Array {
    /// A uniform linear array of `n` vertical isotropic elements along $y$, `spacing`
    /// wavelengths apart, the first element at the origin.
    ///
    /// # Examples
    /// ```
    /// use com_channel::cdl::Array;
    /// let a = Array::linear(4, 0.5);
    /// assert_eq!(a.n_elements(), 4);
    /// assert_eq!(a.positions[3], [0.0, 1.5, 0.0]);
    /// ```
    pub fn linear(n: usize, spacing: f64) -> Array {
        Array {
            positions: (0..n).map(|i| [0.0, i as f64 * spacing, 0.0]).collect(),
            slants: vec![0.0; n],
            pattern: Pattern::Isotropic,
        }
    }

    /// A uniform linear array of `n` dual-polarised isotropic pairs along $y$,
    /// `spacing` wavelengths apart: `2n` elements, the pair at position `i` being the
    /// elements `2i` (slant $+45°$) and `2i + 1` (slant $-45°$).
    pub fn linear_dual(n: usize, spacing: f64) -> Array {
        Array {
            positions: (0..2 * n).map(|i| [0.0, (i / 2) as f64 * spacing, 0.0]).collect(),
            slants: (0..2 * n).map(|i| if i % 2 == 0 { 45.0 } else { -45.0 }).collect(),
            pattern: Pattern::Isotropic,
        }
    }

    /// Number of elements.
    pub fn n_elements(&self) -> usize {
        self.positions.len()
    }

    /// The field $(F_{\theta}, F_{\phi})$ of element `i` in the direction of zenith
    /// `zenith` and azimuth `azimuth` (degrees): $\sqrt{A}\\,(\cos\zeta, \sin\zeta)$.
    ///
    /// # Panics
    /// If `i` is not an element.
    pub fn field(&self, i: usize, zenith: f64, azimuth: f64) -> (f64, f64) {
        let a = db2lin(self.pattern.gain_db(zenith, azimuth)).sqrt();
        let (s, c) = self.slants[i].to_radians().sin_cos();
        (a * c, a * s)
    }
}
