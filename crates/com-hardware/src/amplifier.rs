//! Power amplifier models.
//!
//! A memoryless amplifier is described by its AM/AM and AM/PM conversions: an input
//! $x = r e^{j\theta}$ gives $y = A(r)\\, e^{j(\theta + \Phi(r))}$. The memory polynomial
//! adds the memory of wideband amplifiers.
//!
//! | Model | Use | $A(r)$, $\Phi(r)$ |
//! |---|---|---|
//! | [`Rapp`] | solid-state amplifier (SSPA) | $\dfrac{g r}{\left(1 + (g r / A_{\mathrm{sat}})^{2p}\right)^{1/(2p)}}$, no AM/PM |
//! | [`Saleh`] | travelling-wave tube (TWTA) | $\dfrac{\alpha_{a} r}{1 + \beta_{a} r^{2}}$, $\dfrac{\alpha_{\phi} r^{2}}{1 + \beta_{\phi} r^{2}}$ |
//! | [`MemoryPolynomial`] | wideband amplifier with memory | $y\[n\] = \sum_{k} \sum_{m} a_{km}\\, x\[n-m\]\\, \lvert x\[n-m\] \rvert^{k-1}$ |
//!
//! The operating point is set by the input back-off: the ratio of the input power of
//! saturation to the average input power. [`Rapp::bussgang`] and [`Saleh::bussgang`]
//! give the Bussgang decomposition of the models under a Gaussian input
//! (`com_dsp::nonlinear::bussgang`); [`Rapp::predistort`] is the static predistorter of
//! `com_dsp::compensation::dpd` with the inverse of the model.
//!
//! The parameters are checked by `check` when the block is used (`amplify`,
//! `impair_into`): a zero saturation or a negative coefficient returns
//! [`Error::Config`] instead of NaN samples.
//!
//! # Examples
//! ```
//! use com_hardware::amplifier::{Rapp, Saleh};
//! let pa = Rapp { gain: 1.0, saturation: 1.0, smoothness: 2.0 };
//! assert!((pa.am_am(0.01) - 0.01).abs() < 1e-9);      // linear at low power
//! assert!(Rapp { saturation: 0.0, ..pa }.check().is_err());
//! assert!(pa.am_am(100.0) < 1.0);                      // saturates
//! let twta = Saleh::default();
//! assert!((twta.saturation_input() - 1.0 / 1.1517f64.sqrt()).abs() < 1e-12);
//! ```
//!
//! # References
//!
//! 1. C. Rapp, "Effects of HPA-nonlinearity on a 4-DPSK/OFDM-signal for a digital sound
//!    broadcasting system," *Proc. 2nd European Conf. on Satellite Communications*,
//!    Liège, 179–184, 1991.
//! 2. A. A. M. Saleh, "Frequency-independent and frequency-dependent nonlinear models of
//!    TWT amplifiers," *IEEE Trans. Commun.*, 29(11), 1715–1720, 1981.
//! 3. J. Kim and K. Konstantinou, "Digital predistortion of wideband signals based on
//!    power amplifier model with memory," *Electron. Lett.*, 37(23), 1417–1418, 2001.
//! 4. L. Ding, G. T. Zhou, D. R. Morgan, Z. Ma, J. S. Kenney, J. Kim and C. R. Giardina,
//!    "A robust digital baseband predistorter constructed using memory polynomials,"
//!    *IEEE Trans. Commun.*, 52(1), 159–165, 2004.

use com_core::ensure;
use com_core::prelude::*;
use com_dsp::compensation::dpd::{memory_polynomial, static_predistort};
use com_dsp::nonlinear::{Bussgang, Envelope, bussgang_of};
use com_dsp::signal::map_envelope;
use com_math::cis;

/// Apply a memoryless AM/AM, AM/PM conversion in place.
fn memoryless(y: &mut [C64], am_am: impl Fn(f64) -> f64, am_pm: impl Fn(f64) -> f64) {
    map_envelope(y, |r| am_am(r) / r * cis(am_pm(r)));
}

/// `Ok` if `v` is positive and finite.
fn positive(v: f64, what: &str) -> Result<()> {
    ensure(v > 0.0 && v.is_finite(), || format!("the {what} must be positive and finite, got {v}"))
}

/// `Ok` if `v` is non-negative and finite.
fn non_negative(v: f64, what: &str) -> Result<()> {
    ensure(v >= 0.0 && v.is_finite(), || format!("the {what} must be non-negative and finite, got {v}"))
}

/// The memoryless amplifier models, by name.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// [`Rapp`].
    Rapp,
    /// [`Saleh`].
    Saleh,
}

impl std::str::FromStr for Model {
    type Err = Error;

    /// `rapp` or `saleh` (case-insensitive).
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    fn from_str(s: &str) -> Result<Model> {
        match s.to_ascii_lowercase().as_str() {
            "rapp" => Ok(Model::Rapp),
            "saleh" => Ok(Model::Saleh),
            _ => Err(Error::Config(format!("unknown amplifier {s:?}: rapp, saleh"))),
        }
    }
}

/// Rapp's model of a solid-state amplifier: a smooth envelope limiter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rapp {
    /// Small-signal amplitude gain $g$.
    pub gain: f64,
    /// Output saturation amplitude $A_{\mathrm{sat}}$.
    pub saturation: f64,
    /// Smoothness $p$ (large $p$: an ideal limiter; 2 to 3 for practical SSPAs).
    pub smoothness: f64,
}

impl Default for Rapp {
    /// Unit gain and saturation, $p = 2$.
    fn default() -> Self {
        Rapp { gain: 1.0, saturation: 1.0, smoothness: 2.0 }
    }
}

impl Rapp {
    /// The largest output, as a fraction of $A_{\mathrm{sat}}$, that
    /// [`Rapp::inverse_am_am`] inverts: the inverse diverges at the saturation, so the
    /// output is clamped just below it.
    pub const INVERSE_LIMIT: f64 = 1.0 - 1e-12;

    /// The largest output, as a fraction of $A_{\mathrm{sat}}$, that [`Rapp::predistort`]
    /// targets (the input it needs grows without bound at the saturation).
    pub const PREDISTORTION_LIMIT: f64 = 1.0 - 1e-6;

    /// Validate the parameters.
    ///
    /// # Errors
    /// [`Error::Config`] unless the gain, the saturation and the smoothness are positive
    /// and finite.
    pub fn check(&self) -> Result<()> {
        positive(self.gain, "gain of the Rapp model")?;
        positive(self.saturation, "saturation of the Rapp model")?;
        positive(self.smoothness, "smoothness of the Rapp model")
    }

    /// AM/AM conversion.
    pub fn am_am(&self, r: f64) -> f64 {
        let p = self.smoothness;
        self.gain * r / (1.0 + (self.gain * r / self.saturation).powf(2.0 * p)).powf(1.0 / (2.0 * p))
    }

    /// Input amplitude that drives the output to saturation in the linear model,
    /// $A_{\mathrm{sat}} / g$ (the reference of the input back-off).
    pub fn saturation_input(&self) -> f64 {
        self.saturation / self.gain
    }

    /// The input amplitude that gives the output amplitude `r` ($r < A_{\mathrm{sat}}$):
    /// the inverse of the AM/AM conversion, the output clamped at
    /// [`Rapp::INVERSE_LIMIT`] $A_{\mathrm{sat}}$.
    pub fn inverse_am_am(&self, r: f64) -> f64 {
        let p = self.smoothness;
        let u = (r / self.saturation).min(Rapp::INVERSE_LIMIT);
        r / self.gain / (1.0 - u.powf(2.0 * p)).powf(1.0 / (2.0 * p))
    }

    /// The static predistorter of the model ([`static_predistort`] with
    /// [`Rapp::inverse_am_am`]): the amplitude that makes the amplifier output $g r$, up
    /// to `max_output` $A_{\mathrm{sat}}$ — the cascade is then an ideal envelope limiter.
    /// A fraction above [`Rapp::PREDISTORTION_LIMIT`] is lowered to it.
    ///
    /// # Errors
    /// The errors of [`Rapp::check`]; [`Error::Config`] unless `max_output` lies in
    /// $[0, 1)$.
    pub fn predistort(&self, x: &[C64], max_output: f64) -> Result<Vec<C64>> {
        self.check()?;
        ensure((0.0..1.0).contains(&max_output), || {
            format!("the maximum output is a fraction of the saturation in [0, 1), got {max_output}")
        })?;
        let cap = max_output.min(Rapp::PREDISTORTION_LIMIT) * self.saturation;
        Ok(static_predistort(x, |r| self.inverse_am_am(r), self.gain, cap))
    }

    /// Amplify a block.
    ///
    /// # Errors
    /// The errors of [`Rapp::check`].
    pub fn amplify(&self, x: &[C64]) -> Result<Vec<C64>> {
        self.check()?;
        let mut y = x.to_vec();
        memoryless(&mut y, |r| self.am_am(r), |_| 0.0);
        Ok(y)
    }

    /// The Bussgang decomposition of the amplifier driven by a circular Gaussian input of
    /// power `power` ([`bussgang_of`] of its [`Envelope`]).
    ///
    /// # Errors
    /// The errors of [`Rapp::check`].
    pub fn bussgang(&self, power: f64) -> Result<Bussgang> {
        self.check()?;
        Ok(bussgang_of(self, power))
    }
}

impl Envelope for Rapp {
    /// $A(r)$, no AM/PM.
    fn response(&self, r: f64) -> C64 {
        C64::new(self.am_am(r), 0.0)
    }
}

impl Impair for Rapp {
    /// # Errors
    /// The errors of [`Rapp::check`].
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        self.check()?;
        x.map_streams_in_place(out, |_, y| memoryless(y, |r| self.am_am(r), |_| 0.0));
        Ok(())
    }
}

/// Saleh's model of a travelling-wave tube amplifier (AM/AM and AM/PM).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Saleh {
    /// $\alpha_{a}$.
    pub alpha_a: f64,
    /// $\beta_{a}$.
    pub beta_a: f64,
    /// $\alpha_{\phi}$ (rad).
    pub alpha_p: f64,
    /// $\beta_{\phi}$.
    pub beta_p: f64,
}

impl Default for Saleh {
    /// The TWT fitted by Saleh (1981): $\alpha_{a} = 2.1587$, $\beta_{a} = 1.1517$,
    /// $\alpha_{\phi} = 4.0033$, $\beta_{\phi} = 9.1040$.
    fn default() -> Self {
        Saleh { alpha_a: 2.1587, beta_a: 1.1517, alpha_p: 4.0033, beta_p: 9.1040 }
    }
}

impl Saleh {
    /// Validate the parameters.
    ///
    /// # Errors
    /// [`Error::Config`] unless $\alpha_{a} > 0$ and $\beta_{a}$, $\alpha_{\phi}$,
    /// $\beta_{\phi}$ are non-negative, all finite.
    pub fn check(&self) -> Result<()> {
        positive(self.alpha_a, "alpha_a of the Saleh model")?;
        non_negative(self.beta_a, "beta_a of the Saleh model")?;
        non_negative(self.alpha_p, "alpha_p of the Saleh model")?;
        non_negative(self.beta_p, "beta_p of the Saleh model")
    }

    /// AM/AM conversion.
    pub fn am_am(&self, r: f64) -> f64 {
        self.alpha_a * r / (1.0 + self.beta_a * r * r)
    }

    /// AM/PM conversion (rad).
    pub fn am_pm(&self, r: f64) -> f64 {
        self.alpha_p * r * r / (1.0 + self.beta_p * r * r)
    }

    /// Input amplitude of the maximum output, $1/\sqrt{\beta_{a}}$.
    pub fn saturation_input(&self) -> f64 {
        1.0 / self.beta_a.sqrt()
    }

    /// Amplify a block.
    ///
    /// # Errors
    /// The errors of [`Saleh::check`].
    pub fn amplify(&self, x: &[C64]) -> Result<Vec<C64>> {
        self.check()?;
        let mut y = x.to_vec();
        memoryless(&mut y, |r| self.am_am(r), |r| self.am_pm(r));
        Ok(y)
    }

    /// The Bussgang decomposition of the amplifier driven by a circular Gaussian input of
    /// power `power` ([`bussgang_of`] of its [`Envelope`]).
    ///
    /// # Errors
    /// The errors of [`Saleh::check`].
    pub fn bussgang(&self, power: f64) -> Result<Bussgang> {
        self.check()?;
        Ok(bussgang_of(self, power))
    }
}

impl Envelope for Saleh {
    /// $A(r) e^{j\Phi(r)}$.
    fn response(&self, r: f64) -> C64 {
        C64::from_polar(self.am_am(r), self.am_pm(r))
    }
}

impl Impair for Saleh {
    /// # Errors
    /// The errors of [`Saleh::check`].
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        self.check()?;
        x.map_streams_in_place(out, |_, y| memoryless(y, |r| self.am_am(r), |r| self.am_pm(r)));
        Ok(())
    }
}

/// A memory polynomial amplifier, $y\[n\] = \sum_{k} \sum_{m=0}^{M-1} a_{km}\\, x\[n-m\]\\,
/// \lvert x\[n-m\] \rvert^{k-1}$ (Kim & Konstantinou 2001), the samples before the block being zero.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryPolynomial {
    /// For every nonlinearity order $k$, the taps $a_{k0}, \dots, a_{k,M-1}$.
    pub terms: Vec<(u32, Vec<C64>)>,
}

impl MemoryPolynomial {
    /// The amplifier of Ding et al. (2004): odd orders 1, 3, 5 and a memory of three
    /// samples, extracted from a class-AB amplifier by Kim & Konstantinou (2001).
    pub fn ding2004() -> MemoryPolynomial {
        let c = C64::new;
        MemoryPolynomial {
            terms: vec![
                (1, vec![c(1.0513, 0.0904), c(-0.0680, -0.0023), c(0.0289, -0.0054)]),
                (3, vec![c(-0.0542, -0.2900), c(0.2234, 0.2317), c(-0.0621, -0.0932)]),
                (5, vec![c(-0.9657, -0.7028), c(-0.2451, -0.3735), c(0.1229, 0.1508)]),
            ],
        }
    }

    /// Amplify a block: every term is the causal filtering of its basis
    /// ($y\[n\] = \sum_{k} \sum_{m} a_{km}\, x\[n - m\] |x\[n - m\]|^{k - 1}$), the samples
    /// before the block taken as zero ([`memory_polynomial`], the kernel of the
    /// predistorter).
    pub fn amplify(&self, x: &[C64]) -> Vec<C64> {
        memory_polynomial(x, self.terms.iter().map(|(k, taps)| (*k, taps.as_slice())))
    }

    /// The small-signal gain $a_{10}$ (the first tap of the linear term).
    pub fn linear_gain(&self) -> C64 {
        self.terms.iter().find(|(k, _)| *k == 1).and_then(|(_, t)| t.first().copied()).unwrap_or(C64::new(0.0, 0.0))
    }
}

impl Impair for MemoryPolynomial {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        x.map_streams(out, |_, s| self.amplify(s));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rapp_is_linear_then_saturates_and_its_inverse_is_exact() {
        let pa = Rapp { gain: 2.0, saturation: 1.5, smoothness: 3.0 };
        assert!((pa.am_am(1e-3) - 2e-3).abs() < 1e-12);
        // the 1 dB compression point lies below the saturation input
        assert!(pa.am_am(pa.saturation_input()) < 1.5);
        for r in [0.1, 0.5, 1.0, 1.4] {
            assert!((pa.am_am(pa.inverse_am_am(r)) - r).abs() < 1e-9);
        }
        // the predistorted amplifier is linear up to the cap
        let x: Vec<C64> = (1..60).map(|i| C64::new(0.0, f64::from(i) * 0.01)).collect();
        let y = pa.amplify(&pa.predistort(&x, 0.95).unwrap()).unwrap();
        for (a, b) in y.iter().zip(&x) {
            if 2.0 * b.norm() < 0.95 * 1.5 {
                assert!((a - b * 2.0).norm() < 1e-9);
            }
        }
    }

    /// The amplifiers copied every stream and allocated a second time before they
    /// amplified the output in place, and the predistorter evaluated the inverse itself
    /// before it called `com_dsp::compensation::dpd::static_predistort`: the samples are
    /// bit-identical.
    #[test]
    fn the_in_place_blocks_and_the_shared_predistorter_are_the_former_ones() {
        let mut r = Rng::new(8);
        let x: Vec<C64> = (0..2000).map(|_| r.complex_gaussian() * 0.7).collect();
        let sig = Signal::from_streams(&[x.clone(), x.iter().map(|v| v * 0.5).collect()], Hertz(1.0)).unwrap();
        let rapp = Rapp { gain: 1.3, saturation: 0.9, smoothness: 2.5 };
        let saleh = Saleh::default();
        let former = |am_am: &dyn Fn(f64) -> f64, am_pm: &dyn Fn(f64) -> f64, s: &[C64]| {
            let mut y = s.to_vec();
            map_envelope(&mut y, |r| am_am(r) / r * cis(am_pm(r)));
            y
        };
        let ra = Impair::impair(&mut rapp.clone(), &sig, &mut r).unwrap();
        let sa = Impair::impair(&mut saleh.clone(), &sig, &mut r).unwrap();
        for s in 0..2 {
            assert_eq!(ra.stream(s), &former(&|r| rapp.am_am(r), &|_| 0.0, sig.stream(s))[..]);
            assert_eq!(sa.stream(s), &former(&|r| saleh.am_am(r), &|r| saleh.am_pm(r), sig.stream(s))[..]);
        }
        let cap = 0.95 * rapp.saturation;
        let old = former(&|r| rapp.inverse_am_am((rapp.gain * r).min(cap)), &|_| 0.0, &x);
        assert_eq!(rapp.predistort(&x, 0.95).unwrap(), old);
        assert!(rapp.predistort(&x, 1.0).is_err() && Rapp { smoothness: 0.0, ..rapp }.amplify(&x).is_err());
        assert!(
            Saleh { beta_a: -1.0, ..saleh }.check().is_err() && Saleh { alpha_a: 0.0, ..saleh }.amplify(&x).is_err()
        );
    }

    #[test]
    fn the_models_are_parsed_by_name() {
        assert_eq!("Rapp".parse::<Model>().unwrap(), Model::Rapp);
        assert_eq!("saleh".parse::<Model>().unwrap(), Model::Saleh);
        assert!("twta".parse::<Model>().is_err());
    }

    #[test]
    fn saleh_peaks_at_its_saturation_input() {
        let pa = Saleh::default();
        let r0 = pa.saturation_input();
        assert!(pa.am_am(r0) > pa.am_am(r0 * 0.95) && pa.am_am(r0) > pa.am_am(r0 * 1.05));
        assert!((pa.am_am(r0) - pa.alpha_a / (2.0 * pa.beta_a.sqrt())).abs() < 1e-12);
    }

    #[test]
    fn the_memory_polynomial_is_linear_at_low_power() {
        let pa = MemoryPolynomial::ding2004();
        let x = vec![C64::new(1e-4, 0.0); 10];
        let y = pa.amplify(&x);
        let lin: C64 = pa.terms[0].1.iter().sum::<C64>() * 1e-4;
        assert!((y[9] - lin).norm() < 1e-9);
        assert_eq!(pa.linear_gain(), C64::new(1.0513, 0.0904));
    }

    /// The amplifier evaluated its terms itself before the kernel was shared with the
    /// predistorter: the results are bit-identical.
    #[test]
    fn the_shared_kernel_is_the_former_evaluation() {
        use com_dsp::compensation::dpd::polynomial_term;
        use com_dsp::filter::convolve_full;
        let mut r = Rng::new(3);
        let x: Vec<C64> = (0..5000).map(|_| r.complex_gaussian() * 0.3).collect();
        let long =
            MemoryPolynomial { terms: vec![(1, vec![C64::new(0.9, 0.1); 40]), (3, vec![C64::new(-0.1, 0.0); 2])] };
        for pa in [MemoryPolynomial::ding2004(), long] {
            let mut old = vec![C64::new(0.0, 0.0); x.len()];
            for (k, taps) in &pa.terms {
                for (yn, v) in old.iter_mut().zip(convolve_full(taps, &polynomial_term(&x, *k))) {
                    *yn += v;
                }
            }
            assert_eq!(pa.amplify(&x), old);
        }
    }
}

#[cfg(test)]
mod literature {
    use super::{Rapp, Saleh};
    use com_dsp::nonlinear::{bussgang, bussgang_of};
    use com_math::C64;

    /// H. Bouhadda, H. Shaiek, D. Roviras, R. Zayani, Y. Medjahdi and R. Bouallegue,
    /// "Theoretical analysis of BER performance of nonlinearly amplified FBMC/OQAM and
    /// OFDM signals," EURASIP J. Adv. Signal Process. 2014:60, Tables 2 and 3: Bussgang's
    /// gain and the distortion power of the SSPA (Rapp, $v = 1$) and of the TWTA (Saleh
    /// with $A_{\mathrm{sat}} = 1$, $\phi_{0} = 0$ and $\pi/6$) at input back-offs of 4, 6
    /// and 8 dB from the saturation, against their "simulation" columns (10⁶ symbols):
    /// the gains within 2·10⁻³, the distortion powers within 1.5 %. Their "theoretical"
    /// columns, from a polynomial fit of order 10 (eqs. (33)–(34)), are further from the
    /// exact values (0.5904 + 0.1068j against 0.5917 + 0.1106j at 4 dB).
    #[test]
    fn bouhadda_2014_tables_2_3() {
        let sspa = Rapp { gain: 1.0, saturation: 1.0, smoothness: 1.0 };
        let twta = |phi0: f64| Saleh { alpha_a: 1.0, beta_a: 1.0, alpha_p: phi0, beta_p: 1.0 };
        let rows: [(&str, f64, f64, C64, f64); 9] = [
            ("SSPA", 4.0, 0.0, C64::new(0.7699, 0.0), 4.4759e-3),
            ("SSPA", 6.0, 0.0, C64::new(0.8307, 0.0), 1.8950e-3),
            ("SSPA", 8.0, 0.0, C64::new(0.8798, 0.0), 7.0742e-4),
            ("TWTA", 4.0, 0.0, C64::new(0.6042, 0.0), 1.0330e-2),
            ("TWTA", 6.0, 0.0, C64::new(0.6976, 0.0), 4.9589e-3),
            ("TWTA", 8.0, 0.0, C64::new(0.7784, 0.0), 2.1228e-3),
            ("TWTA", 4.0, std::f64::consts::FRAC_PI_6, C64::new(0.5917, 0.1106), 1.1339e-2),
            ("TWTA", 6.0, std::f64::consts::FRAC_PI_6, C64::new(0.6887, 0.0995), 5.5901e-3),
            ("TWTA", 8.0, std::f64::consts::FRAC_PI_6, C64::new(0.7727, 0.0830), 2.4194e-3),
        ];
        for (model, ibo, phi0, k_table, d_table) in rows {
            // IBO = A_sat² / σ² with A_sat = 1 (eq. (14))
            let power = com_math::db2lin(-ibo);
            let b = if model == "SSPA" {
                bussgang(|r| C64::new(sspa.am_am(r), 0.0), power)
            } else {
                let t = twta(phi0);
                bussgang(|r| C64::from_polar(t.am_am(r), t.am_pm(r)), power)
            };
            // the models as envelopes give the same decomposition
            let of_model = if model == "SSPA" { bussgang_of(&sspa, power) } else { bussgang_of(&twta(phi0), power) };
            assert_eq!(of_model, b);
            let method = if model == "SSPA" { sspa.bussgang(power) } else { twta(phi0).bussgang(power) }.unwrap();
            assert_eq!(method, b);
            let (k, d) = (b.gain, b.distortion);
            assert!((k - k_table).norm() < 2e-3, "{model} {ibo} dB: K = {k}");
            assert!((d / d_table - 1.0).abs() < 0.015, "{model} {ibo} dB: σ_d² = {d}");
        }
    }
}
