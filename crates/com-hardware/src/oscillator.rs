//! Oscillator impairments: Wiener phase noise and carrier frequency offset.
//!
//! - [`phase_noise`]: the Wiener process of an oscillator (or a laser) of linewidth
//!   `Δν` (OptiCommPy `phaseNoise`); [`wiener_phase`]: the same for a given increment
//!   standard deviation and [`PhaseOrigin`];
//! - [`frequency_offset`]: the rotation of a constant frequency offset, from the first
//!   sample or from the middle of the block;
//! - [`PhaseNoise`] and [`Cfo`]: the [`Impair`] blocks that rotate every stream of a
//!   [`Signal`] by the phase noise, or by a constant frequency offset.
//!
//! # Model
//!
//! $$
//! \phi_{n+1} = \phi_{n} + \Delta_{n}, \qquad \Delta_{n} \sim \mathcal{N}(0, 2\pi \Delta\nu T_{s}),
//! \qquad \phi_{0} = 0
//! $$
//!
//! ([`PhaseOrigin::Step`] draws $\phi_{0} = \Delta_{0}$ instead: the first sample is
//! rotated too), and a frequency offset $\omega = 2\pi \Delta f / f_{s}$ rotates sample
//! $n$ by $e^{j\omega (n - n_{c})}$, with $n_{c} = 0$, or $n_{c} = N/2$ for a block of
//! $N$ samples whose phase is zero at its middle.
//!
//! # Examples
//! ```
//! use com_core::prelude::*;
//! use com_hardware::oscillator::{Cfo, PhaseNoise};
//! let x = Signal::from_stream(vec![C64::new(1.0, 0.0); 1000], Hertz(1e9))?;
//! let y = PhaseNoise { linewidth: Hertz(1e5) }.impair(&x, &mut batch_rng(1, 0))?;
//! assert!(y.stream(0).iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));   // constant envelope
//! let z = Cfo { offset: Hertz(1e6), centred: false }.impair(&x, &mut batch_rng(1, 0))?;
//! assert!((z.stream(0)[250].arg() - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
//! # Ok::<(), com_core::Error>(())
//! ```

use com_core::ensure;
use com_core::prelude::*;
use com_math::cis;
use std::f64::consts::PI;

/// Wiener phase noise of linewidth `linewidth_hz` over `n` samples spaced by `ts`
/// seconds (radians, `φ[0] = 0`): [`wiener_phase`] with the increments of
/// [`increment_sd`] at $f_{s} = 1/T_{s}$.
///
/// # Errors
/// [`Error::Config`] for a negative or non-finite linewidth, or a sample spacing that is
/// not positive and finite.
pub fn phase_noise(linewidth_hz: f64, n: usize, ts: f64, rng: &mut Rng) -> Result<Vec<f64>> {
    ensure(ts > 0.0 && ts.is_finite(), || format!("the sample spacing must be positive, got {ts} s"))?;
    let sd = increment_sd(linewidth_hz, 1.0 / ts)?;
    Ok(wiener_phase(sd, n, PhaseOrigin::Zero, rng))
}

/// The standard deviation $\sqrt{2\pi \Delta\nu / f_{s}}$ of the increments of the
/// Wiener phase of linewidth `linewidth_hz` sampled at `fs`.
///
/// # Errors
/// [`Error::Config`] for a negative or non-finite linewidth, or a sampling rate that is
/// not positive and finite.
pub fn increment_sd(linewidth_hz: f64, fs: f64) -> Result<f64> {
    ensure(linewidth_hz >= 0.0 && linewidth_hz.is_finite(), || {
        format!("the linewidth must be non-negative, got {linewidth_hz} Hz")
    })?;
    ensure(fs > 0.0 && fs.is_finite(), || format!("the sampling rate must be positive, got {fs} Hz"))?;
    Ok((2.0 * PI * linewidth_hz / fs).sqrt())
}

/// Where a Wiener phase starts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PhaseOrigin {
    /// $\phi_{0} = 0$: the first sample is not rotated (OptiCommPy `phaseNoise`).
    #[default]
    Zero,
    /// $\phi_{0} = \Delta_{0}$: an increment is drawn before the first sample too, which
    /// is rotated.
    Step,
}

impl std::str::FromStr for PhaseOrigin {
    type Err = Error;

    /// `zero` or `step` (case-insensitive).
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    fn from_str(s: &str) -> Result<PhaseOrigin> {
        match s.to_ascii_lowercase().as_str() {
            "zero" => Ok(PhaseOrigin::Zero),
            "step" => Ok(PhaseOrigin::Step),
            _ => Err(Error::Config(format!("the phase origin must be zero or step, got {s:?}"))),
        }
    }
}

/// A Wiener phase over `n` samples with increments of standard deviation `sd` (radians),
/// starting at `origin`: one Gaussian draw per increment, sample after sample.
pub fn wiener_phase(sd: f64, n: usize, origin: PhaseOrigin, rng: &mut Rng) -> Vec<f64> {
    let step = origin == PhaseOrigin::Step;
    let mut phi = Vec::with_capacity(n);
    let mut p = 0.0;
    for k in 0..n {
        if k > 0 || step {
            p += sd * rng.gaussian();
        }
        phi.push(p);
    }
    phi
}

/// Rotate `x` by the frequency offset `omega` (rad/sample): sample $n$ by
/// $e^{j\omega (n - n_{c})}$, $n_{c} = N/2$ if `centred` (the phase is zero at the middle
/// of the block), $0$ otherwise.
pub fn frequency_offset(x: &mut [C64], omega: f64, centred: bool) {
    let nc = if centred { x.len() as f64 / 2.0 } else { 0.0 };
    com_dsp::signal::rotate(x, omega, nc);
}

/// The Wiener phase noise of an oscillator, drawn independently for every stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseNoise {
    /// Linewidth `Δν` (the sum of the transmitter and receiver linewidths for a link).
    pub linewidth: Hertz,
}

impl Impair for PhaseNoise {
    /// # Errors
    /// The errors of [`increment_sd`] (a negative linewidth, a signal without a positive
    /// sampling rate).
    fn impair_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let sd = increment_sd(self.linewidth.0, x.sample_rate().0)?;
        x.map_streams_in_place(out, |_, y| {
            let phi = wiener_phase(sd, y.len(), PhaseOrigin::Zero, rng);
            for (v, p) in y.iter_mut().zip(phi) {
                *v *= cis(p);
            }
        });
        Ok(())
    }
}

/// A carrier frequency offset, `x[n] e^{j2πΔf (n - n_c)/f_s}` on every stream
/// ([`frequency_offset`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfo {
    /// Offset `Δf`.
    pub offset: Hertz,
    /// Zero phase at the middle of the signal ($n_{c} = N/2$) instead of its first
    /// sample.
    pub centred: bool,
}

impl Impair for Cfo {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let w = 2.0 * PI * self.offset.0 / x.sample_rate().0;
        x.map_streams_in_place(out, |_, y| frequency_offset(y, w, self.centred));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wiener_increments_have_the_linewidth_variance() {
        let phi = phase_noise(1e6, 100_001, 1e-9, &mut Rng::new(2)).unwrap();
        let inc: Vec<f64> = phi.windows(2).map(|w| w[1] - w[0]).collect();
        let v = inc.iter().map(|d| d * d).sum::<f64>() / inc.len() as f64;
        assert!((v / (2.0 * PI * 1e6 * 1e-9) - 1.0).abs() < 0.02);
    }

    #[test]
    fn the_origins_differ_by_the_first_increment() {
        let zero = wiener_phase(0.1, 50, PhaseOrigin::Zero, &mut Rng::new(3));
        let step = wiener_phase(0.1, 50, PhaseOrigin::Step, &mut Rng::new(3));
        assert_eq!(zero[0], 0.0);
        assert!(step[0] != 0.0);
        let pn = phase_noise(0.01 / (2.0 * PI), 50, 1.0, &mut Rng::new(3)).unwrap();
        assert!(zero.iter().zip(&pn).all(|(a, b)| (a - b).abs() < 1e-12));
        let mut x = vec![C64::new(1.0, 0.0); 8];
        frequency_offset(&mut x, 0.3, true);
        assert!((x[4] - C64::new(1.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn zero_impairments_are_the_identity() {
        let x = Signal::from_streams(
            &[(0..64).map(|n| C64::new(f64::from(n), 1.0)).collect(), vec![C64::new(0.5, -0.5); 64]],
            Hertz(1e9),
        )
        .unwrap();
        let mut rng = Rng::new(0);
        assert_eq!(PhaseNoise { linewidth: Hertz(0.0) }.impair(&x, &mut rng).unwrap(), x);
        assert_eq!(Cfo { offset: Hertz(0.0), centred: false }.impair(&x, &mut rng).unwrap(), x);
        assert_eq!(Cfo { offset: Hertz(0.0), centred: true }.impair(&x, &mut rng).unwrap(), x);
        assert!(PhaseNoise { linewidth: Hertz(-1.0) }.impair(&x, &mut rng).is_err());
        assert!(phase_noise(1e3, 4, 0.0, &mut rng).is_err() && increment_sd(f64::NAN, 1.0).is_err());
    }

    /// The blocks rotated a copy of every stream before they rotated the output in place,
    /// and the phase noise took the increments of `phase_noise` (the deviation
    /// $\sqrt{2\pi\Delta\nu T_{s}}$) before those of `increment_sd`
    /// ($\sqrt{2\pi\Delta\nu / f_{s}}$): the same draws, rotations within the rounding of
    /// the deviation.
    #[test]
    fn the_in_place_blocks_are_the_former_ones() {
        let x = Signal::from_streams(
            &[(0..500).map(|n| C64::new(f64::from(n).sin(), 1.0)).collect(), vec![C64::new(0.5, -0.5); 500]],
            Hertz(3e9),
        )
        .unwrap();
        let (lw, ts) = (2e6, x.sample_rate().period().0);
        let mut rng = Rng::new(4);
        let former: Vec<Vec<C64>> = (0..2)
            .map(|s| {
                let phi = wiener_phase((2.0 * PI * lw * ts).sqrt(), 500, PhaseOrigin::Zero, &mut rng);
                x.stream(s).iter().zip(phi).map(|(v, p)| v * cis(p)).collect()
            })
            .collect();
        let y = PhaseNoise { linewidth: Hertz(lw) }.impair(&x, &mut Rng::new(4)).unwrap();
        for (s, f) in former.iter().enumerate() {
            let err = y.stream(s).iter().zip(f).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-13, "{err}");
        }
        let mut cfo = x.stream(1).to_vec();
        frequency_offset(&mut cfo, 2.0 * PI * 1e6 / 3e9, true);
        assert_eq!(Cfo { offset: Hertz(1e6), centred: true }.impair(&x, &mut rng).unwrap().stream(1), &cfo[..]);
    }
}
