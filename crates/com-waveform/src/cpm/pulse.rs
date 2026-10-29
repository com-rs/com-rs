//! The frequency pulses of CPM, the closed forms of the Gaussian pulse of GMSK and the
//! power spectral density of MSK.

use com_core::{Error, Result};
use com_math::{normal_pdf, qfunc};
use std::f64::consts::{PI, TAU};

/// The frequency pulse $g(t)$ of a CPM signal (see the module documentation).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrequencyPulse {
    /// $L$REC: constant over $L$ symbols (1REC is CPFSK; with $h = 1/2$, MSK).
    Rectangular,
    /// $L$RC: a raised cosine over $L$ symbols.
    RaisedCosine,
    /// GMSK: the Gaussian pulse of Murota & Hirade with the normalised bandwidth `bt`,
    /// truncated to $L$ symbols.
    Gaussian {
        /// The 3 dB bandwidth of the Gaussian filter times the symbol duration, $BT$.
        bt: f64,
    },
}

impl FrequencyPulse {
    /// The short name of the pulse, `rec`, `rc` or `gmsk` (read back by
    /// [`FrequencyPulse::from_name`]).
    pub fn name(&self) -> &'static str {
        match self {
            FrequencyPulse::Rectangular => "rec",
            FrequencyPulse::RaisedCosine => "rc",
            FrequencyPulse::Gaussian { .. } => "gmsk",
        }
    }

    /// The phase response $q(t)$ of the pulse of `length` symbols at the time `t` in
    /// symbol durations: 0 before 0, $1/2$ after `length`.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::cpm::FrequencyPulse;
    /// assert_eq!(FrequencyPulse::Rectangular.phase_response(0.5, 1), 0.25);
    /// assert_eq!(FrequencyPulse::RaisedCosine.phase_response(3.0, 3), 0.5);
    /// let g = FrequencyPulse::Gaussian { bt: 0.3 };
    /// assert!((g.phase_response(1.5, 3) - 0.25).abs() < 1e-12);     // symmetric pulse
    /// ```
    pub fn phase_response(&self, t: f64, length: usize) -> f64 {
        let l = length as f64;
        if t <= 0.0 {
            return 0.0;
        }
        if t >= l {
            return 0.5;
        }
        match *self {
            FrequencyPulse::Rectangular => t / (2.0 * l),
            FrequencyPulse::RaisedCosine => t / (2.0 * l) - (TAU * t / l).sin() / (2.0 * TAU),
            FrequencyPulse::Gaussian { bt } => {
                // the pulse centred at L/2, truncated to [0, L] and scaled to the area 1/2
                let lo = gmsk_phase(-l / 2.0, bt);
                let hi = gmsk_phase(l / 2.0, bt);
                0.5 * (gmsk_phase(t - l / 2.0, bt) - lo) / (hi - lo)
            }
        }
    }

    /// Parse `rec` (`rectangular`), `rc` (`raised-cosine`) or `gmsk` (`gaussian`) with
    /// the bandwidth `bt` for the Gaussian pulse.
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    pub fn from_name(name: &str, bt: f64) -> Result<FrequencyPulse> {
        match name.to_ascii_lowercase().as_str() {
            "rec" | "rectangular" => Ok(FrequencyPulse::Rectangular),
            "rc" | "raised-cosine" => Ok(FrequencyPulse::RaisedCosine),
            "gmsk" | "gaussian" => Ok(FrequencyPulse::Gaussian { bt }),
            other => Err(Error::Config(format!("pulse must be rec, rc or gmsk, got {other:?}"))),
        }
    }
}

/// $\int Q(x) \\, dx = x Q(x) - \phi(x)$.
fn q_integral(x: f64) -> f64 {
    x * qfunc(x) - normal_pdf(x)
}

/// The Gaussian frequency pulse of GMSK, centred at 0, with $T = 1$ (Murota & Hirade,
/// 1981): the response of the Gaussian low-pass filter of 3 dB bandwidth $B$ to a
/// rectangular pulse of one symbol,
///
/// $$
/// g(t) = \frac{1}{2}\left[Q\left(\frac{2 \pi B}{\sqrt{\ln 2}}\left(t - \frac{1}{2}\right)\right) -
/// Q\left(\frac{2 \pi B}{\sqrt{\ln 2}}\left(t + \frac{1}{2}\right)\right)\right], \qquad
/// \int_{-\infty}^{\infty} g(t) \\, dt = \frac{1}{2}.
/// $$
///
/// # Examples
/// ```
/// use com_waveform::cpm::gmsk_pulse;
/// // the pulse is symmetric and its area is 1/2 (trapezoid over ±4 symbols)
/// let n = 8000;
/// let area: f64 = (0..=n).map(|i| gmsk_pulse(-4.0 + 8.0 * i as f64 / n as f64, 0.3) * 8.0 / n as f64).sum();
/// assert!((area - 0.5).abs() < 1e-6);
/// assert!((gmsk_pulse(0.7, 0.3) - gmsk_pulse(-0.7, 0.3)).abs() < 1e-15);
/// ```
pub fn gmsk_pulse(t: f64, bt: f64) -> f64 {
    let kappa = TAU * bt / 2f64.ln().sqrt();
    0.5 * (qfunc(kappa * (t - 0.5)) - qfunc(kappa * (t + 0.5)))
}

/// The phase response of the Gaussian pulse, $q(t) = \int_{-\infty}^{t} g(\tau) \\, d\tau$
/// in closed form (0 at $-\infty$, $1/4$ at 0, $1/2$ at $+\infty$).
///
/// # Examples
/// ```
/// use com_waveform::cpm::gmsk_phase;
/// assert!((gmsk_phase(0.0, 0.3) - 0.25).abs() < 1e-12);
/// assert!((gmsk_phase(10.0, 0.3) - 0.5).abs() < 1e-12);
/// ```
pub fn gmsk_phase(t: f64, bt: f64) -> f64 {
    let kappa = TAU * bt / 2f64.ln().sqrt();
    0.5 + (q_integral(kappa * (t - 0.5)) - q_integral(kappa * (t + 0.5))) / (2.0 * kappa)
}

/// The power spectral density of MSK at the normalised frequency `ft` $= f T$, with
/// $T = 1$ and a unit power (Proakis & Salehi, §3.4-3):
///
/// $$
/// S(f) = \frac{16 T}{\pi^{2}} \left[\frac{\cos 2 \pi f T}{1 - 16 f^{2} T^{2}}\right]^{2},
/// $$
///
/// finite at $f T = \pm 1/4$ (where it equals $T$), zero at $f T = \pm 3/4, \pm 5/4, \dots$
///
/// # Examples
/// ```
/// use com_waveform::cpm::msk_psd;
/// assert!((msk_psd(0.0) - 16.0 / std::f64::consts::PI.powi(2)).abs() < 1e-12);
/// assert!((msk_psd(0.25) - 1.0).abs() < 1e-6);
/// assert!(msk_psd(0.75).abs() < 1e-12);
/// ```
pub fn msk_psd(ft: f64) -> f64 {
    let d = 1.0 - 16.0 * ft * ft;
    let ratio = if d.abs() < 1e-7 { PI / 4.0 } else { (TAU * ft).cos() / d };
    16.0 / (PI * PI) * ratio * ratio
}
