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
//! saturation to the average input power.
//!
//! # Examples
//! ```
//! use com_hardware::amplifier::{Rapp, Saleh};
//! let pa = Rapp { gain: 1.0, saturation: 1.0, smoothness: 2.0 };
//! assert!((pa.am_am(0.01) - 0.01).abs() < 1e-9);      // linear at low power
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

use com_core::prelude::*;
use com_dsp::compensation::dpd::polynomial_term;
use com_dsp::signal::map_envelope;
use com_math::cis;

/// Apply a memoryless AM/AM, AM/PM conversion.
fn memoryless(x: &[C64], am_am: impl Fn(f64) -> f64, am_pm: impl Fn(f64) -> f64) -> Vec<C64> {
    let mut y = x.to_vec();
    map_envelope(&mut y, |r| am_am(r) / r * cis(am_pm(r)));
    y
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
    /// the inverse of the AM/AM conversion.
    pub fn inverse_am_am(&self, r: f64) -> f64 {
        let p = self.smoothness;
        let u = (r / self.saturation).min(1.0 - 1e-12);
        r / self.gain / (1.0 - u.powf(2.0 * p)).powf(1.0 / (2.0 * p))
    }

    /// The static predistorter of the model: the amplitude that makes the amplifier
    /// output $g r$, up to `max_output` (a fraction of $A_{\mathrm{sat}}$, below one) —
    /// the cascade is then an ideal envelope limiter.
    pub fn predistort(&self, x: &[C64], max_output: f64) -> Vec<C64> {
        let cap = max_output.clamp(0.0, 1.0 - 1e-6) * self.saturation;
        memoryless(x, |r| self.inverse_am_am((self.gain * r).min(cap)), |_| 0.0)
    }

    /// Amplify a block.
    pub fn amplify(&self, x: &[C64]) -> Vec<C64> {
        memoryless(x, |r| self.am_am(r), |_| 0.0)
    }
}

impl Impair for Rapp {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        x.map_streams(out, |_, s| self.amplify(s));
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
    pub fn amplify(&self, x: &[C64]) -> Vec<C64> {
        memoryless(x, |r| self.am_am(r), |r| self.am_pm(r))
    }
}

impl Impair for Saleh {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        x.map_streams(out, |_, s| self.amplify(s));
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

    /// Amplify a block.
    pub fn amplify(&self, x: &[C64]) -> Vec<C64> {
        let mut y = vec![C64::new(0.0, 0.0); x.len()];
        for (k, taps) in &self.terms {
            let basis = polynomial_term(x, *k);
            for (m, a) in taps.iter().enumerate() {
                for n in m..x.len() {
                    y[n] += a * basis[n - m];
                }
            }
        }
        y
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
        let y = pa.amplify(&pa.predistort(&x, 0.95));
        for (a, b) in y.iter().zip(&x) {
            if 2.0 * b.norm() < 0.95 * 1.5 {
                assert!((a - b * 2.0).norm() < 1e-9);
            }
        }
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
}
