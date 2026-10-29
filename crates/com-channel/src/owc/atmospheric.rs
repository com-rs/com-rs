//! The outdoor optical wireless channel (free-space optics, FSO): the random fading of
//! the received intensity by atmospheric turbulence and by pointing errors, and the
//! attenuation of the weather.
//!
//! # Model
//!
//! The received intensity is $I = h_{l}\, h_{a}\, h_{p}\, I_{0}$: the deterministic loss
//! $h_{l}$ of the weather, the turbulence $h_{a}$ of unit mean and the misalignment
//! $h_{p}$, all constant over thousands of symbols (the coherence time is 0.1 to 10 ms;
//! Khalighi & Uysal 2014, Sec. II).
//!
//! **Turbulence** ([`Turbulence`]). The strength of the fluctuations is the Rytov
//! variance $\sigma_{1}^{2} = 1.23 C_{n}^{2} k^{7/6} L^{11/6}$ ([`rytov_variance`]) and their
//! measure the scintillation index $\sigma_{I}^{2} = \langle I^{2}\rangle / \langle I
//! \rangle^{2} - 1$. In the gamma–gamma model the intensity is the product of two
//! independent gamma variables of unit mean, the large- and small-scale eddies, of
//! parameters $\alpha$ and $\beta$ (Al-Habash, Andrews & Phillips 2001, eqs. (10)–(15)):
//!
//! $$
//! p(I) = \frac{2(\alpha\beta)^{(\alpha+\beta)/2}}{\Gamma(\alpha)\Gamma(\beta)}
//! I^{(\alpha+\beta)/2 - 1} K_{\alpha-\beta}\left(2\sqrt{\alpha\beta I}\right), \qquad
//! \sigma_{I}^{2} = \frac{1}{\alpha} + \frac{1}{\beta} + \frac{1}{\alpha\beta},
//! $$
//!
//! with $\alpha = 1/\sigma_{x}^{2}$, $\beta = 1/\sigma_{y}^{2}$ and, for a plane wave and a
//! negligible inner scale (their eqs. (18)–(19)),
//!
//! $$
//! \sigma_{x}^{2} = \exp\left[\frac{0.49\sigma_{1}^{2}}{(1 + 1.11\sigma_{1}^{12/5})^{7/6}}\right] - 1,
//! \qquad
//! \sigma_{y}^{2} = \exp\left[\frac{0.51\sigma_{1}^{2}}{(1 + 0.69\sigma_{1}^{12/5})^{5/6}}\right] - 1,
//! $$
//!
//! and for a spherical wave the same with $\beta_{0}^{2} = 0.4\sigma_{1}^{2}$ and $0.56$ in
//! place of $1.11$ (their eqs. (21), (25)–(26)) ([`GammaGamma::from_rytov`]). The
//! exponent of $\sigma_{y}^{2}$ is $5/6$, that of the theory they cite: eqs. (19) and
//! (26) print $7/6$, but the parameters of their Figs. 3 and 6–8 and the scintillation
//! index of 1.214 they quote for $\sigma_{1}^{2} = 25$ follow from $5/6$. With
//! $\beta = 1$ the gamma–gamma model is the K distribution of strong turbulence.
//! The log-normal model of weak turbulence ($\ln I$ Gaussian) and the negative
//! exponential model of saturation complete [`Turbulence`].
//!
//! **Pointing errors** ([`PointingError`]). A Gaussian beam of radius $w_{z}$ at the
//! receiver, a circular aperture of radius $a$ and a radial displacement $r$ of
//! Rayleigh distribution of parameter $\sigma_{s}$ (independent Gaussian sway in both
//! directions) collect the fraction $h_{p} = A_{0} \exp(-2r^{2}/w_{z,\mathrm{eq}}^{2})$, with
//! $v = \sqrt{\pi} a/(\sqrt{2} w_{z})$, $A_{0} = \operatorname{erf}^{2}(v)$ and
//! $w_{z,\mathrm{eq}}^{2} = w_{z}^{2} \sqrt{\pi}\operatorname{erf}(v) / (2v e^{-v^{2}})$
//! (Farid & Hranilovic 2007, quoted by Khalighi & Uysal 2014).
//!
//! **Weather** ([`attenuation_db_per_km`]). The attenuation of haze and fog for a
//! visibility $V$: $\beta_{\lambda} = \frac{3.91}{V} (\lambda / 550\,\mathrm{nm})^{-q}$ in
//! km⁻¹, with $q$ of Kim et al. (2001), quoted.
//!
//! [`Fso`] draws the channel ([`Fso::gain`]) and applies it to a signal of intensity
//! ([`com_core::traits::Propagate`]), one draw per call.
//!
//! # Examples
//! ```
//! use com_channel::owc::atmospheric::{GammaGamma, Wave};
//! // a plane wave in the saturation regime: α = 8.05, β = 1.03, σ_I² = 1.214
//! // (Al-Habash, Andrews & Phillips 2001, Fig. 3 and Sec. 4)
//! let g = GammaGamma::from_rytov(25.0, Wave::Plane)?;
//! assert!((g.alpha - 8.05).abs() < 0.01 && (g.beta - 1.03).abs() < 0.01);
//! assert!((g.scintillation_index() - 1.214).abs() < 1e-3);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. M. A. Al-Habash, L. C. Andrews and R. L. Phillips, "Mathematical model for the
//!    irradiance probability density function of a laser beam propagating through
//!    turbulent media," *Opt. Eng.*, 40(8), 1554–1562, 2001.
//! 2. M. A. Khalighi and M. Uysal, "Survey on free space optical communication: a
//!    communication theory perspective," *IEEE Commun. Surveys Tuts.*, 16(4),
//!    2231–2258, 2014.
//! 3. A. A. Farid and S. Hranilovic, "Outage capacity optimization for free-space
//!    optical links with pointing errors," *J. Lightw. Technol.*, 25(7), 1702–1710, 2007.
//! 4. I. I. Kim, B. McArthur and E. Korevaar, "Comparison of laser beam propagation at
//!    785 nm and 1550 nm in fog and haze for optical wireless communications," *Proc.
//!    SPIE* 4214, 26–37, 2001.

use com_core::ensure;
use com_core::prelude::*;
use com_math::{erf, ln_bessel_k, ln_gamma};
use std::f64::consts::PI;

/// The Rytov variance $\sigma_{1}^{2} = 1.23 C_{n}^{2} k^{7/6} L^{11/6}$ of a plane wave of
/// wavelength `wavelength_m` over `length_m` of turbulence of structure parameter
/// `cn2` (m^{-2/3}); typical $C_{n}^{2}$ range from $10^{-17}$ to $10^{-13}$.
pub fn rytov_variance(cn2: f64, wavelength_m: f64, length_m: f64) -> f64 {
    let k = 2.0 * PI / wavelength_m;
    1.23 * cn2 * k.powf(7.0 / 6.0) * length_m.powf(11.0 / 6.0)
}

/// The propagation model of the scintillation theory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wave {
    /// A plane wave.
    Plane,
    /// A spherical wave.
    Spherical,
}

/// The gamma–gamma distribution of the intensity, of unit mean (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaGamma {
    /// $\alpha$, the effective number of large-scale cells.
    pub alpha: f64,
    /// $\beta$, the effective number of small-scale cells.
    pub beta: f64,
}

impl GammaGamma {
    /// The parameters predicted by the Rytov variance `rytov` $\sigma_{1}^{2}$ for a
    /// negligible inner scale (Al-Habash, Andrews & Phillips 2001, eqs. (14), (18)–(19)
    /// and (21), (25)–(26), with the exponent $5/6$, see the module).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a non-positive Rytov variance.
    pub fn from_rytov(rytov: f64, wave: Wave) -> Result<GammaGamma> {
        ensure(rytov > 0.0 && rytov.is_finite(), || format!("the Rytov variance must be positive, got {rytov}"))?;
        let (s2, large) = match wave {
            Wave::Plane => (rytov, 1.11),
            Wave::Spherical => (0.4 * rytov, 0.56),
        };
        let p = s2.powf(6.0 / 5.0); // σ^{12/5} = (σ²)^{6/5}
        let sx2 = (0.49 * s2 / (1.0 + large * p).powf(7.0 / 6.0)).exp_m1();
        let sy2 = (0.51 * s2 / (1.0 + 0.69 * p).powf(5.0 / 6.0)).exp_m1();
        Ok(GammaGamma { alpha: 1.0 / sx2, beta: 1.0 / sy2 })
    }

    /// The scintillation index $1/\alpha + 1/\beta + 1/(\alpha\beta)$.
    pub fn scintillation_index(&self) -> f64 {
        1.0 / self.alpha + 1.0 / self.beta + 1.0 / (self.alpha * self.beta)
    }

    /// The density $p(I)$ (0 for $I \le 0$), computed in the log domain.
    pub fn pdf(&self, i: f64) -> f64 {
        if i <= 0.0 {
            return 0.0;
        }
        let (a, b) = (self.alpha, self.beta);
        let ln = std::f64::consts::LN_2 + 0.5 * (a + b) * (a * b).ln() - ln_gamma(a) - ln_gamma(b)
            + (0.5 * (a + b) - 1.0) * i.ln()
            + ln_bessel_k(a - b, 2.0 * (a * b * i).sqrt());
        ln.exp()
    }

    /// A draw: the product of two gamma variables of shapes $\alpha$ and $\beta$ and unit
    /// mean.
    pub fn draw(&self, rng: &mut Rng) -> f64 {
        rng.gamma(self.alpha) / self.alpha * rng.gamma(self.beta) / self.beta
    }

    fn check(&self) -> Result<()> {
        ensure(self.alpha > 0.0 && self.beta > 0.0, || {
            format!("the gamma–gamma parameters must be positive, got {} and {}", self.alpha, self.beta)
        })
    }
}

/// The fading of the intensity by turbulence, of unit mean.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Turbulence {
    /// No turbulence: $h_{a} = 1$.
    None,
    /// Log-normal fading of weak turbulence: $\ln h_{a}$ Gaussian of variance
    /// $\ln(1 + \sigma_{I}^{2})$ and mean $-\ln(1 + \sigma_{I}^{2})/2$.
    LogNormal {
        /// The scintillation index $\sigma_{I}^{2}$.
        scintillation_index: f64,
    },
    /// Gamma–gamma fading, from weak to strong turbulence.
    GammaGamma(GammaGamma),
    /// The negative exponential fading of the saturation regime ($\sigma_{I}^{2} = 1$).
    NegativeExponential,
}

impl Turbulence {
    /// The scintillation index $\sigma_{I}^{2}$.
    pub fn scintillation_index(&self) -> f64 {
        match self {
            Turbulence::None => 0.0,
            Turbulence::LogNormal { scintillation_index } => *scintillation_index,
            Turbulence::GammaGamma(g) => g.scintillation_index(),
            Turbulence::NegativeExponential => 1.0,
        }
    }

    /// The density of $h_{a}$ (`NaN` without turbulence, a Dirac at 1).
    pub fn pdf(&self, h: f64) -> f64 {
        match self {
            Turbulence::None => f64::NAN,
            _ if h <= 0.0 => 0.0,
            Turbulence::LogNormal { scintillation_index } => {
                let s2 = scintillation_index.ln_1p();
                let m = -0.5 * s2;
                (-(h.ln() - m).powi(2) / (2.0 * s2)).exp() / (h * (2.0 * PI * s2).sqrt())
            }
            Turbulence::GammaGamma(g) => g.pdf(h),
            Turbulence::NegativeExponential => (-h).exp(),
        }
    }

    /// A draw of $h_{a}$.
    pub fn draw(&self, rng: &mut Rng) -> f64 {
        match self {
            Turbulence::None => 1.0,
            Turbulence::LogNormal { scintillation_index } => {
                let s2 = scintillation_index.ln_1p();
                (s2.sqrt() * rng.gaussian() - 0.5 * s2).exp()
            }
            Turbulence::GammaGamma(g) => g.draw(rng),
            Turbulence::NegativeExponential => -(1.0 - rng.uniform()).ln(),
        }
    }

    fn check(&self) -> Result<()> {
        match self {
            Turbulence::LogNormal { scintillation_index } => ensure(*scintillation_index > 0.0, || {
                format!("the scintillation index must be positive, got {scintillation_index}")
            }),
            Turbulence::GammaGamma(g) => g.check(),
            _ => Ok(()),
        }
    }
}

/// The misalignment of a Gaussian beam on a circular aperture (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointingError {
    /// Beam radius $w_{z}$ at the receiver (m).
    pub beam_radius_m: f64,
    /// Radius $a$ of the receiving aperture (m).
    pub aperture_radius_m: f64,
    /// Standard deviation $\sigma_{s}$ of the displacement in each direction (m).
    pub jitter_m: f64,
}

impl PointingError {
    fn v(&self) -> f64 {
        PI.sqrt() * self.aperture_radius_m / (std::f64::consts::SQRT_2 * self.beam_radius_m)
    }

    /// The fraction $A_{0} = \operatorname{erf}^{2}(v)$ collected without displacement.
    pub fn a0(&self) -> f64 {
        erf(self.v()).powi(2)
    }

    /// The equivalent beam radius $w_{z,\mathrm{eq}}$ (m).
    pub fn equivalent_radius_m(&self) -> f64 {
        let v = self.v();
        (self.beam_radius_m.powi(2) * PI.sqrt() * erf(v) / (2.0 * v * (-v * v).exp())).sqrt()
    }

    /// The fraction $h_{p}$ collected at the radial displacement `r_m`.
    pub fn collected(&self, r_m: f64) -> f64 {
        self.a0() * (-2.0 * r_m * r_m / self.equivalent_radius_m().powi(2)).exp()
    }

    /// The density of $h_{p}$, $\frac{\xi^{2}}{A_{0}^{\xi^{2}}} h_{p}^{\xi^{2} - 1}$ on
    /// $(0, A_{0}]$ with $\xi = w_{z,\mathrm{eq}}/(2\sigma_{s})$.
    pub fn pdf(&self, h: f64) -> f64 {
        let a0 = self.a0();
        if h <= 0.0 || h > a0 {
            return 0.0;
        }
        let xi2 = (self.equivalent_radius_m() / (2.0 * self.jitter_m)).powi(2);
        xi2 / a0.powf(xi2) * h.powf(xi2 - 1.0)
    }

    /// A draw of $h_{p}$: a Rayleigh displacement of parameter $\sigma_{s}$.
    pub fn draw(&self, rng: &mut Rng) -> f64 {
        let r = self.jitter_m * (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
        self.collected(r)
    }

    fn check(&self) -> Result<()> {
        ensure(self.beam_radius_m > 0.0 && self.aperture_radius_m > 0.0 && self.jitter_m > 0.0, || {
            "the beam radius, the aperture radius and the jitter must be positive".into()
        })
    }
}

/// The attenuation (dB/km) of haze and fog for the visibility `visibility_km` at the
/// wavelength `wavelength_nm`: $10 \log_{10}(e)\, \frac{3.91}{V} (\lambda/550)^{-q}$, with
/// $q = 1.6$ above 50 km, 1.3 above 6 km, $0.16V + 0.34$ above 1 km, $V - 0.5$ above
/// 0.5 km and 0 below (Kim et al. 2001).
///
/// # Errors
/// [`com_core::Error::Config`] for a non-positive visibility or wavelength.
pub fn attenuation_db_per_km(visibility_km: f64, wavelength_nm: f64) -> Result<f64> {
    ensure(visibility_km > 0.0 && wavelength_nm > 0.0, || {
        format!("the visibility and the wavelength must be positive, got {visibility_km} km and {wavelength_nm} nm")
    })?;
    let v = visibility_km;
    let q = if v > 50.0 {
        1.6
    } else if v > 6.0 {
        1.3
    } else if v > 1.0 {
        0.16 * v + 0.34
    } else if v > 0.5 {
        v - 0.5
    } else {
        0.0
    };
    Ok(10.0 * std::f64::consts::LOG10_E * 3.91 / v * (wavelength_nm / 550.0).powf(-q))
}

/// A free-space optical link: weather loss, turbulence and pointing errors, drawn once
/// per call and applied to the intensity (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fso {
    /// The deterministic loss (dB): weather and geometric loss.
    pub loss_db: f64,
    /// The turbulence.
    pub turbulence: Turbulence,
    /// The pointing errors; `None` for a perfectly aligned link.
    pub pointing: Option<PointingError>,
}

impl Fso {
    /// Check the link.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid turbulence or pointing parameters.
    pub fn check(&self) -> Result<()> {
        self.turbulence.check()?;
        self.pointing.as_ref().map_or(Ok(()), PointingError::check)
    }

    /// A draw of the channel gain $h = h_{l} h_{a} h_{p}$ of the intensity.
    ///
    /// # Errors
    /// As [`Fso::check`].
    pub fn gain(&self, rng: &mut Rng) -> Result<f64> {
        self.check()?;
        let hp = self.pointing.as_ref().map_or(1.0, |p| p.draw(rng));
        Ok(com_math::db2lin(-self.loss_db) * self.turbulence.draw(rng) * hp)
    }
}

impl Propagate for Fso {
    /// One gain per call, the same for every stream and sample (the fading is constant
    /// over a frame).
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let h = self.gain(rng)?;
        x.map_streams(out, |_, xs| xs.iter().map(|v| v * h).collect());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gamma_gamma_draws_follow_their_moments_and_the_density_integrates_to_one() {
        let mut rng = Rng::new(5);
        for rytov in [0.2, 2.0, 25.0] {
            let g = GammaGamma::from_rytov(rytov, Wave::Plane).unwrap();
            let n = 200_000;
            let x: Vec<f64> = (0..n).map(|_| g.draw(&mut rng)).collect();
            let m1 = x.iter().sum::<f64>() / n as f64;
            let m2 = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
            assert!((m1 - 1.0).abs() < 0.01, "{rytov}: mean {m1}");
            let si = m2 / (m1 * m1) - 1.0;
            assert!((si / g.scintillation_index() - 1.0).abs() < 0.05, "{rytov}: {si} vs {}", g.scintillation_index());
            let total = com_math::simpson(|i| g.pdf(i), 1e-9, 30.0, 60_000);
            let mean = com_math::simpson(|i| i * g.pdf(i), 1e-9, 30.0, 60_000);
            assert!((total - 1.0).abs() < 1e-3 && (mean - 1.0).abs() < 2e-3, "{rytov}: {total} {mean}");
        }
    }

    #[test]
    fn the_other_models_have_unit_mean_and_their_index() {
        let mut rng = Rng::new(9);
        let n = 200_000;
        for t in [Turbulence::LogNormal { scintillation_index: 0.3 }, Turbulence::NegativeExponential] {
            let x: Vec<f64> = (0..n).map(|_| t.draw(&mut rng)).collect();
            let m1 = x.iter().sum::<f64>() / n as f64;
            let si = x.iter().map(|v| v * v).sum::<f64>() / n as f64 / (m1 * m1) - 1.0;
            assert!((m1 - 1.0).abs() < 0.01 && (si - t.scintillation_index()).abs() < 0.05, "{t:?}: {m1} {si}");
            let total = com_math::simpson(|h| t.pdf(h), 1e-9, 40.0, 40_000);
            assert!((total - 1.0).abs() < 1e-3, "{t:?}: {total}");
        }
    }

    #[test]
    fn pointing_errors_follow_their_density() {
        let p = PointingError { beam_radius_m: 2.5, aperture_radius_m: 0.1, jitter_m: 0.3 };
        let total = com_math::simpson(|h| p.pdf(h), 1e-12, p.a0(), 20_000);
        assert!((total - 1.0).abs() < 1e-3, "{total}");
        // the mean of the draws against that of the density
        let mut rng = Rng::new(2);
        let n = 100_000;
        let m = (0..n).map(|_| p.draw(&mut rng)).sum::<f64>() / n as f64;
        let want = com_math::simpson(|h| h * p.pdf(h), 1e-12, p.a0(), 20_000);
        assert!((m / want - 1.0).abs() < 0.01, "{m} vs {want}");
        // a small aperture in a wide beam collects about 2a²/w²
        assert!((p.a0() / (2.0 * 0.01 / 6.25) - 1.0).abs() < 0.01);
    }

    #[test]
    fn the_weather_attenuation_and_invalid_settings() {
        // clear air (23 km) at 1550 nm: about 0.2 dB/km; fog (0.2 km): 85 dB/km
        assert!((attenuation_db_per_km(23.0, 1550.0).unwrap() - 0.19).abs() < 0.01);
        assert!((attenuation_db_per_km(0.2, 1550.0).unwrap() - 84.9).abs() < 0.1);
        assert!(attenuation_db_per_km(0.0, 1550.0).is_err() && GammaGamma::from_rytov(0.0, Wave::Plane).is_err());
        let bad = Fso { loss_db: 0.0, turbulence: Turbulence::LogNormal { scintillation_index: -1.0 }, pointing: None };
        assert!(bad.gain(&mut Rng::new(1)).is_err());
    }
}
