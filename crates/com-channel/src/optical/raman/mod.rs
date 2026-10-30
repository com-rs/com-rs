//! Distributed Raman amplification and inter-channel stimulated Raman scattering
//! (ISRS): the power profiles of the signals and of any number of co- and
//! counter-propagating pumps along a span, and the spontaneous Raman noise.
//!
//! # Model
//!
//! Every wave $k$ (a signal, a slice of the spectrum of a field, or a pump) of frequency
//! $f_{k}$ and power $P_{k}(z)$ obeys the coupled power equations of stimulated Raman
//! scattering (Bromage 2004, Eq. (1); Agrawal, *Nonlinear Fiber Optics*, ch. 8)
//!
//! $$
//! \pm \frac{d P_{k}}{d z} = P_{k} \left(-\alpha_{k} + \sum_{j} C_{kj} P_{j}\right),
//! $$
//!
//! with $+$ for the waves that travel with the signals and $-$ for the
//! counter-propagating pumps (the Raman gain of silica is the same in both directions).
//! The coefficient of a wave $k$ from a wave $j$ of higher frequency is the gain
//! efficiency of the fibre
//!
//! $$
//! C_{kj} = \frac{g_{R}(f_{j} - f_{k})}{A_{\mathrm{eff}}} \frac{f_{j}}{f_{\mathrm{ref}}},
//! \qquad f_{j} > f_{k},
//! $$
//!
//! the measured gain spectrum $g_{R}$ of a pump at $f_{\mathrm{ref}}$ ([`RamanGain`])
//! scaled with the pump frequency, and the wave of higher frequency loses the photons
//! that the other gains: $C_{jk} = -(f_{j}/f_{k}) C_{kj}$ (photon-number
//! conservation; without loss, $\sum_{k} \pm P_{k}/f_{k}$ is constant along $z$).
//!
//! The co-propagating waves are known at $z = 0$ and the counter-propagating pumps at
//! $z = L$: a two-point boundary-value problem. [`Raman::solve`] integrates the
//! logarithms of the powers by the classical fourth-order Runge-Kutta method on a
//! uniform grid, forward for the co-propagating waves with the counter-propagating
//! powers of the previous sweep (linear between the nodes, so that the scheme is of
//! second order when both directions are present: $10^{-6}$ dB with the default step
//! of 100 m), then backward for the counter-propagating ones, until
//! the logarithms change by less than a tolerance between sweeps (the relaxation
//! method of the multi-pump design literature; Perlin & Winful 2002), with
//! under-relaxation if the sweeps oscillate.
//!
//! # Spontaneous emission
//!
//! The spontaneous Raman scattering of the pumps adds to signal $k$ a noise of power
//! spectral density, both polarisations,
//!
//! $$
//! N_{k}(L) = \int_{0}^{L} 2 h f_{k} \sum_{j \in \mathrm{pumps}} C_{kj} P_{j}(z)\\,
//!   (1 + \eta_{kj})\\, \frac{G_{k}(L)}{G_{k}(z)}\\, dz,
//! \qquad \eta_{kj} = \frac{1}{e^{h (f_{j} - f_{k}) / (k_{B} T)} - 1},
//! $$
//!
//! with $G_{k}(z) = P_{k}(z)/P_{k}(0)$ and $\eta$ the phonon occupancy at the
//! temperature $T$ (Bromage 2004, Eq. (4); the spontaneous term of GNPy's
//! `RamanSolver`). As in GNPy, the spontaneous scattering of the signals themselves
//! (Stokes and anti-Stokes noise between channels, about 10 % of the noise of the
//! pumps for 40 channels of 1 mW) and double Rayleigh backscattering are neglected.
//!
//! # Use in the links
//!
//! [`super::fiber::Ssfm`], [`super::manakov::ManakovSsfm`] and [`super::gn::GnModel`]
//! take a [`Raman`] configuration: the split-step solvers apply the profile of
//! every span as a gain that depends on the position and on the frequency (the
//! spectrum of the field is cut into slices of [`Raman::slice_hz`] whose average powers
//! are the waves of the equations, the quasi-static treatment of ISRS in split-step
//! simulations), add the spontaneous noise at the end of the span and, after it, the
//! amplifier restores every frequency to its launch power (an EDFA with an ideal
//! gain-flattening filter, which adds its ASE only where the Raman gain did not
//! compensate the loss). Without pumps and with [`Raman::isrs`], the profile is the
//! tilt of the spectrum by ISRS alone.
//!
//! # Examples
//! ```
//! use com_channel::optical::fiber::FiberParams;
//! use com_channel::optical::raman::{Direction, Raman, RamanPump};
//! // a counter-propagating pump at 1450 nm of 400 mW in 100 km of standard fibre
//! let fiber = FiberParams { span_km: 100.0, ..FiberParams::default() };
//! let raman = Raman {
//!     pumps: vec![RamanPump::at_wavelength(1450.0, 0.4, Direction::Counter)],
//!     ..Raman::default()
//! };
//! let signal = [(fiber.fc_hz, 1e-3)];
//! let p = raman.solve(&fiber, &signal)?;
//! // the pump compensates 13 of the 20 dB of loss of the span
//! assert!((p.net_gain_db(0) + 20.0 - p.on_off_gain_db(0) ).abs() < 1e-9);
//! assert!(p.on_off_gain_db(0) > 10.0 && p.on_off_gain_db(0) < 16.0);
//! assert!(p.ase_psd[0] > 0.0);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. Bromage, "Raman amplification for fiber communications systems," *J. Lightw.
//!    Technol.*, 22(1), 79–93, 2004.
//! 2. V. E. Perlin and H. G. Winful, "Optimal design of flat-gain wide-band fiber Raman
//!    amplifiers," *J. Lightw. Technol.*, 20(2), 250–254, 2002.
//! 3. G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013, ch. 8.
//! 4. GNPy (Telecom Infra Project), `gnpy.core.science_utils.RamanSolver` and the gain
//!    spectrum of standard single-mode fibre of `gnpy.core.parameters`.
//! 5. D. Semrau, R. I. Killey and P. Bayvel, "A closed-form approximation of the
//!    Gaussian noise model in the presence of inter-channel stimulated Raman
//!    scattering," *J. Lightw. Technol.*, 37(9), 1924–1936, 2019.

mod gain;
mod solver;

pub use gain::RamanGain;

use super::edfa::ase_density;
use super::fiber::{Amplification, FiberParams};
use com_core::Result;
use com_core::rng::Rng;
use com_math::{C64, db2lin, fft, nm2hz, np2db};

/// The direction of propagation of a pump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// With the signals (launched at $z = 0$).
    Co,
    /// Against the signals (launched at $z = L$).
    Counter,
}

/// A Raman pump.
#[derive(Clone, Debug, PartialEq)]
pub struct RamanPump {
    /// Frequency (Hz).
    pub frequency_hz: f64,
    /// Power launched into the fibre at its end of the span (W).
    pub power_w: f64,
    /// Direction of propagation.
    pub direction: Direction,
    /// Attenuation at the pump wavelength (dB/km); that of the fibre if `None`.
    pub alpha_db_km: Option<f64>,
}

impl RamanPump {
    /// A pump of `power_w` at the vacuum wavelength `wavelength_nm`, with the
    /// attenuation of the fibre.
    pub fn at_wavelength(wavelength_nm: f64, power_w: f64, direction: Direction) -> RamanPump {
        RamanPump { frequency_hz: nm2hz(wavelength_nm), power_w, direction, alpha_db_km: None }
    }
}

/// The Raman amplification and ISRS of the spans of a link (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Raman {
    /// The pumps of every span.
    pub pumps: Vec<RamanPump>,
    /// The gain spectrum of the fibre.
    pub gain: RamanGain,
    /// Effective area of the fibre (µm²).
    pub aeff_um2: f64,
    /// Temperature of the fibre (K), for the spontaneous emission.
    pub temperature_k: f64,
    /// The Raman interaction between the signals (ISRS); without it the signals only
    /// interact with the pumps (the pumps always interact with each other).
    pub isrs: bool,
    /// Step of the integration grid (km).
    pub step_km: f64,
    /// Convergence tolerance of the sweeps: the largest change of $\ln P$.
    pub tolerance: f64,
    /// Largest number of forward-backward sweeps.
    pub max_iterations: usize,
    /// Width of the slices of the spectrum of a field (Hz) that are the signal waves of
    /// the split-step solvers.
    pub slice_hz: f64,
}

impl Default for Raman {
    /// No pump, the gain of standard single-mode fibre, 80 µm², 300 K, ISRS, steps of
    /// 100 m, tolerance $10^{-10}$, at most 500 sweeps, slices of 25 GHz.
    fn default() -> Self {
        Raman {
            pumps: Vec::new(),
            gain: RamanGain::ssmf(),
            aeff_um2: 80.0,
            temperature_k: 300.0,
            isrs: true,
            step_km: 0.1,
            tolerance: 1e-10,
            max_iterations: 500,
            slice_hz: 25e9,
        }
    }
}

/// The power profiles of a span (see [`Raman::solve`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RamanProfile {
    /// The positions of the grid (km), from 0 to the span length.
    pub z_km: Vec<f64>,
    /// The frequency of every wave (Hz): the signals in their order, then the pumps.
    pub freqs_hz: Vec<f64>,
    /// The direction of every wave.
    pub directions: Vec<Direction>,
    /// The attenuation of every wave (Np/km).
    pub alpha_np_km: Vec<f64>,
    /// The number of signals.
    pub n_signals: usize,
    /// $\ln P_{k}(z)$ (W) of every wave on the grid.
    pub log_power: Vec<Vec<f64>>,
    /// The spontaneous Raman noise of every signal at the end of the span, both
    /// polarisations (W/Hz).
    pub ase_psd: Vec<f64>,
    /// The number of forward-backward sweeps.
    pub iterations: usize,
}

impl RamanProfile {
    /// The power of wave `k` along the span (W).
    pub fn power_w(&self, k: usize) -> Vec<f64> {
        self.log_power[k].iter().map(|l| l.exp()).collect()
    }

    /// The power of wave `k` along the span (dBm).
    pub fn power_dbm(&self, k: usize) -> Vec<f64> {
        self.log_power[k].iter().map(|l| 10.0 * std::f64::consts::LOG10_E * l + 30.0).collect()
    }

    /// The net gain of signal `k` over the span, $P(L)/P(0)$ (dB).
    pub fn net_gain_db(&self, k: usize) -> f64 {
        let l = &self.log_power[k];
        np2db(l[l.len() - 1] - l[0])
    }

    /// The on-off gain of signal `k`: the net gain with the pumps and ISRS against the
    /// loss alone (dB).
    pub fn on_off_gain_db(&self, k: usize) -> f64 {
        let length = self.z_km[self.z_km.len() - 1];
        self.net_gain_db(k) + np2db(self.alpha_np_km[k]) * length
    }

    /// $\ln (P_{k}(z)/P_{k}(0))$ of signal `k` at `z` (km), linear between the nodes.
    pub fn log_gain(&self, k: usize, z: f64) -> f64 {
        let (i, t) = self.node(z);
        let l = &self.log_power[k];
        let a = l[i] + t * (l[(i + 1).min(l.len() - 1)] - l[i]);
        a - l[0]
    }

    fn node(&self, z: f64) -> (usize, f64) {
        let m = self.z_km.len() - 1;
        let h = self.z_km[m] / m as f64;
        let x = (z / h).clamp(0.0, m as f64);
        let i = (x.floor() as usize).min(m.saturating_sub(1));
        (i, x - i as f64)
    }

    /// The weights that interpolate the signals linearly in frequency at `freqs_hz`
    /// (constant beyond the extreme signals).
    pub(crate) fn interpolator(&self, freqs_hz: &[f64]) -> SpectralInterp {
        let mut order: Vec<usize> = (0..self.n_signals).collect();
        order.sort_by(|a, b| self.freqs_hz[*a].total_cmp(&self.freqs_hz[*b]));
        let fs: Vec<f64> = order.iter().map(|k| self.freqs_hz[*k]).collect();
        let weights = freqs_hz
            .iter()
            .map(|f| {
                if fs.len() == 1 || *f <= fs[0] {
                    return (order[0], order[0], 0.0);
                }
                if *f >= fs[fs.len() - 1] {
                    let k = order[fs.len() - 1];
                    return (k, k, 0.0);
                }
                let i = fs.partition_point(|v| v <= f).clamp(1, fs.len() - 1);
                (order[i - 1], order[i], (f - fs[i - 1]) / (fs[i] - fs[i - 1]))
            })
            .collect();
        SpectralInterp { weights }
    }

    /// $\ln (P(z)/P(0))$ at `z` of every frequency of `w`, into `out`.
    pub(crate) fn log_gain_at(&self, w: &SpectralInterp, z: f64, out: &mut [f64]) {
        let g: Vec<f64> = (0..self.n_signals).map(|k| self.log_gain(k, z)).collect();
        for (o, (a, b, t)) in out.iter_mut().zip(&w.weights) {
            *o = g[*a] + t * (g[*b] - g[*a]);
        }
    }

    /// The spontaneous noise at the end of the span of every frequency of `w`, both
    /// polarisations (W/Hz).
    pub(crate) fn ase_at(&self, w: &SpectralInterp) -> Vec<f64> {
        w.weights.iter().map(|(a, b, t)| self.ase_psd[*a] + t * (self.ase_psd[*b] - self.ase_psd[*a])).collect()
    }
}

/// Linear interpolation over the signals of a [`RamanProfile`] at given frequencies.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpectralInterp {
    weights: Vec<(usize, usize, f64)>,
}

/// One span of a split-step simulation with Raman amplification: the profile of the
/// slices of the field and the gains of its frequency bins.
#[derive(Clone, Debug)]
pub(crate) struct FieldSpan {
    /// The profile of the span.
    pub(crate) profile: RamanProfile,
    interp: SpectralInterp,
    /// The absolute frequency of every bin (Hz, FFT order).
    freqs: Vec<f64>,
    length_km: f64,
}

impl Raman {
    /// The slices of the spectrum of the polarisations `fields` sampled at `fs` around
    /// the carrier `fc` (Hz): the absolute centre frequency of every slice of
    /// [`Raman::slice_hz`] and its average power (W, all polarisations).
    pub(crate) fn spectral_slices(&self, fields: &[&[C64]], fs: f64, fc: f64) -> Vec<(f64, f64)> {
        let mut slices: std::collections::BTreeMap<i64, f64> = std::collections::BTreeMap::new();
        for x in fields {
            let n = x.len();
            if n == 0 {
                continue;
            }
            let spectrum = fft::fft(x);
            let nn = (n * n) as f64;
            for (v, f) in spectrum.iter().zip(fft::fftfreq(n, 1.0 / fs)) {
                *slices.entry((f / self.slice_hz).round() as i64).or_insert(0.0) += v.norm_sqr() / nn;
            }
        }
        slices.into_iter().map(|(q, p)| (fc + q as f64 * self.slice_hz, p)).collect()
    }

    /// The span of `fiber` for the polarisations `fields` sampled at `fs` (see
    /// [`FieldSpan`]).
    ///
    /// # Errors
    /// As [`Raman::solve`].
    pub(crate) fn field_span(&self, fiber: &FiberParams, fields: &[&[C64]], fs: f64) -> Result<FieldSpan> {
        let slices = self.spectral_slices(fields, fs, fiber.fc_hz);
        let profile = self.solve(fiber, &slices)?;
        let n = fields.first().map_or(0, |x| x.len());
        let freqs: Vec<f64> = fft::fftfreq(n, 1.0 / fs).iter().map(|f| fiber.fc_hz + f).collect();
        let interp = profile.interpolator(&freqs);
        Ok(FieldSpan { profile, interp, freqs, length_km: fiber.span_km })
    }
}

impl FieldSpan {
    /// The gain of the field at every bin from `z0` to `z1` (km), $\sqrt{G(z_{1}, f) /
    /// G(z_{0}, f)}$: the [`com_math::nlse::FieldGain`] of the span.
    pub(crate) fn field_gain(&self, z0: f64, z1: f64, out: &mut [f64]) {
        let mut a = vec![0.0; out.len()];
        self.profile.log_gain_at(&self.interp, z0, &mut a);
        self.profile.log_gain_at(&self.interp, z1, out);
        for (o, a) in out.iter_mut().zip(&a) {
            *o = ((*o - a) / 2.0).exp();
        }
    }

    /// The power gain that restores every bin to its launch power at the end of the
    /// span, $1/G(L, f)$.
    pub(crate) fn restoring_gain(&self) -> Vec<f64> {
        let mut g = vec![0.0; self.freqs.len()];
        self.profile.log_gain_at(&self.interp, self.length_km, &mut g);
        g.iter().map(|l| (-l).exp()).collect()
    }

    /// The end of the span on one polarisation `x`: the spontaneous Raman noise (half of
    /// the power spectral density of the two polarisations), then the amplifier, and its
    /// ASE where its gain exceeds 1 (`Edfa`, noise figure `nf_db`): the noise is drawn
    /// from `rng`, none without one.
    pub(crate) fn amplify(
        &self,
        x: &mut [C64],
        fs: f64,
        amplification: &Amplification,
        nf_db: f64,
        rng: Option<&mut Rng>,
    ) {
        let n = x.len();
        if n == 0 {
            return;
        }
        let g = match amplification {
            Amplification::None => vec![1.0; n],
            Amplification::Edfa | Amplification::Ideal => self.restoring_gain(),
        };
        let p = fft::plan(n);
        p.forward(x);
        for (v, gb) in x.iter_mut().zip(&g) {
            *v *= gb.sqrt();
        }
        if let Some(rng) = rng {
            let ase = self.profile.ase_at(&self.interp);
            let nf = db2lin(nf_db);
            let mut w: Vec<C64> = (0..n).map(|_| rng.complex_gaussian()).collect();
            p.forward(&mut w);
            for (b, (v, wb)) in x.iter_mut().zip(&w).enumerate() {
                let edfa = match amplification {
                    Amplification::Edfa if g[b] > 1.0 => ase_density(g[b], nf, self.freqs[b]),
                    _ => 0.0,
                };
                *v += wb * ((ase[b] / 2.0 * g[b] + edfa) * fs).sqrt();
            }
        }
        p.inverse(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_math::{BOLTZMANN, PLANCK, SPEED_OF_LIGHT};

    fn fiber(span_km: f64) -> FiberParams {
        FiberParams { span_km, ..FiberParams::default() }
    }

    #[test]
    fn without_pumps_and_isrs_the_profile_is_the_loss() {
        let raman = Raman { isrs: false, ..Raman::default() };
        let f = fiber(80.0);
        let p = raman.solve(&f, &[(193e12, 1e-3), (194e12, 2e-3)]).unwrap();
        for k in 0..2 {
            assert!((p.net_gain_db(k) + 16.0).abs() < 1e-9, "{}", p.net_gain_db(k));
            assert!(p.on_off_gain_db(k).abs() < 1e-9);
        }
        assert_eq!(p.ase_psd, vec![0.0, 0.0]);
        // 1 mW and 2 mW at the start of the span, 16 dB less at its end
        let dbm = p.power_dbm(1);
        assert!((dbm[0] - 10.0 * 2f64.log10()).abs() < 1e-12 && (dbm[0] - dbm[dbm.len() - 1] - 16.0).abs() < 1e-9);
    }

    #[test]
    fn an_undepleted_pump_gives_the_closed_form_gain() {
        // small signal: G_on-off = exp(C P_p L_eff,p) for a co- or a counter-pump
        let f = fiber(80.0);
        let gain = RamanGain::ssmf();
        let fp = SPEED_OF_LIGHT / 1450e-9;
        let fs = SPEED_OF_LIGHT / 1550e-9;
        let c = gain.efficiency(fs, fp, 80e-12);
        let ap = 0.25 / (10.0 * std::f64::consts::LOG10_E);
        let leff = (1.0 - (-ap * 80.0).exp()) / ap;
        for direction in [Direction::Co, Direction::Counter] {
            let pump = RamanPump { alpha_db_km: Some(0.25), ..RamanPump::at_wavelength(1450.0, 0.3, direction) };
            let raman = Raman { pumps: vec![pump], ..Raman::default() };
            let p = raman.solve(&f, &[(fs, 1e-9)]).unwrap();
            let expected = 10.0 * std::f64::consts::LOG10_E * c * 0.3 * leff;
            assert!(
                (p.on_off_gain_db(0) - expected).abs() < 1e-3,
                "{direction:?}: {} vs {expected}",
                p.on_off_gain_db(0)
            );
        }
    }

    #[test]
    fn photons_are_conserved_without_loss() {
        let f = FiberParams { alpha_db_km: 0.0, span_km: 50.0, ..FiberParams::default() };
        let pumps = vec![
            RamanPump::at_wavelength(1425.0, 0.4, Direction::Co),
            RamanPump::at_wavelength(1455.0, 0.3, Direction::Co),
        ];
        let raman = Raman { pumps, ..Raman::default() };
        let signals: Vec<(f64, f64)> = (0..8).map(|k| (190e12 + k as f64 * 0.6e12, 5e-3)).collect();
        let p = raman.solve(&f, &signals).unwrap();
        let photons =
            |i: usize| -> f64 { (0..p.freqs_hz.len()).map(|k| p.log_power[k][i].exp() / p.freqs_hz[k]).sum() };
        let last = p.z_km.len() - 1;
        assert!((photons(last) / photons(0) - 1.0).abs() < 1e-9);
        // and the pumps transferred power to the signals
        assert!(p.net_gain_db(0) > 3.0);
    }

    #[test]
    fn counter_pumps_meet_their_boundary_conditions_and_the_equations() {
        let f = fiber(100.0);
        let pumps = vec![
            RamanPump::at_wavelength(1424.0, 0.25, Direction::Counter),
            RamanPump::at_wavelength(1452.0, 0.2, Direction::Counter),
            RamanPump::at_wavelength(1480.0, 0.1, Direction::Co),
        ];
        let raman = Raman { pumps: pumps.clone(), ..Raman::default() };
        let signals: Vec<(f64, f64)> = (0..10).map(|k| (191e12 + k as f64 * 0.5e12, 1e-3)).collect();
        let p = raman.solve(&f, &signals).unwrap();
        let m = p.z_km.len() - 1;
        for (k, pump) in pumps.iter().enumerate() {
            let at = if pump.direction == Direction::Counter { m } else { 0 };
            assert!((p.log_power[10 + k][at].exp() / pump.power_w - 1.0).abs() < 1e-12);
        }
        // a finer grid gives the same profiles (to 10⁻⁵ dB)
        let fine = Raman { step_km: 0.025, ..raman }.solve(&f, &signals).unwrap();
        for k in 0..p.freqs_hz.len() {
            let d = (p.log_power[k][m] - fine.log_power[k][4 * m]).abs();
            assert!(d < 3e-6, "{k}: {d}");
        }
    }

    #[test]
    fn isrs_tilts_the_spectrum_towards_low_frequencies() {
        // 10 THz of signals at 1 mW per 50 GHz: the lowest channels gain from the highest
        let f = fiber(80.0);
        let raman = Raman::default();
        let signals: Vec<(f64, f64)> = (0..200).map(|k| (186e12 + k as f64 * 50e9, 1e-3)).collect();
        let p = raman.solve(&f, &signals).unwrap();
        let tilt = p.net_gain_db(0) - p.net_gain_db(199);
        // Zirngibl's triangular approximation: tilt ≈ 10 log10 e · C' P_tot L_eff B with
        // C' the slope of the gain efficiency
        let gain = RamanGain::ssmf();
        let slope = gain.efficiency(193e12, 193e12 + 5e12, 80e-12) / 5e12;
        let expected = 10.0 * std::f64::consts::LOG10_E * slope * 0.2 * f.effective_length_km() * 9.95e12;
        assert!(tilt > 0.0 && (tilt / expected - 1.0).abs() < 0.15, "{tilt} vs {expected}");
    }

    #[test]
    fn the_spontaneous_noise_matches_the_undepleted_integral() {
        let f = fiber(80.0);
        let pump = RamanPump { alpha_db_km: Some(0.25), ..RamanPump::at_wavelength(1450.0, 0.3, Direction::Counter) };
        let raman = Raman { pumps: vec![pump], ..Raman::default() };
        let fs = SPEED_OF_LIGHT / 1550e-9;
        let p = raman.solve(&f, &[(fs, 1e-9)]).unwrap();
        // independent quadrature with the closed-form profiles
        let fp = SPEED_OF_LIGHT / 1450e-9;
        let c = RamanGain::ssmf().efficiency(fs, fp, 80e-12);
        let (a, ap) = (f.alpha_np_km(), 0.25 / (10.0 * std::f64::consts::LOG10_E));
        let pp = |z: f64| 0.3 * (-ap * (80.0 - z)).exp();
        let lg = |z: f64| -a * z + c * 0.3 * ((-ap * (80.0 - z)).exp() - (-ap * 80.0).exp()) / ap;
        let eta = 1.0 / (PLANCK * (fp - fs) / (BOLTZMANN * 300.0)).exp_m1();
        let n = 80_000;
        let dz = 80.0 / n as f64;
        let integral: f64 = (0..n)
            .map(|i| {
                let z = (i as f64 + 0.5) * dz;
                c * pp(z) * (lg(80.0) - lg(z)).exp()
            })
            .sum::<f64>()
            * dz;
        let expected = 2.0 * PLANCK * fs * (1.0 + eta) * integral;
        assert!((p.ase_psd[0] / expected - 1.0).abs() < 1e-4, "{} vs {expected}", p.ase_psd[0]);
    }

    fn tones(n: usize, fs: f64, offsets: &[(f64, f64)]) -> Vec<C64> {
        // tones at the frequencies of the bins nearest to the offsets, of the given powers
        (0..n)
            .map(|t| {
                offsets
                    .iter()
                    .map(|(f, p)| {
                        let k = (f / fs * n as f64).round();
                        C64::from_polar(p.sqrt(), 2.0 * std::f64::consts::PI * k * t as f64 / n as f64)
                    })
                    .sum()
            })
            .collect()
    }

    fn tone_power(x: &[C64], fs: f64, f: f64) -> f64 {
        let n = x.len();
        let k = ((f / fs * n as f64).round() as i64).rem_euclid(n as i64) as usize;
        fft::fft(x)[k].norm_sqr() / (n * n) as f64
    }

    #[test]
    fn the_split_step_follows_the_profile_of_every_slice() {
        use super::super::fiber::Ssfm;
        // two tones 1.2 THz apart, linear fibre, a counter pump, no amplifier
        let fiber = FiberParams { gamma_w_km: 0.0, d_ps_nm_km: 0.0, span_km: 60.0, ..FiberParams::default() };
        let (fs, n) = (4.096e12, 4096);
        let x = tones(n, fs, &[(-0.6e12, 20e-3), (0.6e12, 20e-3)]);
        let raman = Raman {
            pumps: vec![RamanPump::at_wavelength(1450.0, 0.3, Direction::Counter)],
            slice_hz: 100e9,
            ..Raman::default()
        };
        let ssfm =
            Ssfm { fiber: fiber.clone(), step_km: 0.5, amplification: Amplification::None, raman: Some(raman.clone()) };
        let y = ssfm.propagate(&x, fs, None).unwrap();
        let p = raman.solve(&fiber, &raman.spectral_slices(&[&x], fs, fiber.fc_hz)).unwrap();
        for f in [-0.6e12, 0.6e12] {
            let k = p.freqs_hz[..p.n_signals].iter().position(|v| (v - fiber.fc_hz - f).abs() < 1.0).unwrap();
            let measured = com_math::lin2db(tone_power(&y, fs, f) / tone_power(&x, fs, f));
            assert!((measured - p.net_gain_db(k)).abs() < 1e-6, "{f}: {measured} vs {}", p.net_gain_db(k));
        }
    }

    #[test]
    fn without_pumps_nor_isrs_the_raman_links_are_the_edfa_links() {
        use super::super::edfa::Edfa;
        use super::super::fiber::Ssfm;
        use super::super::manakov::ManakovSsfm;
        use com_core::signal::DualPol;
        let fiber = FiberParams { n_spans: 2, span_km: 50.0, ..FiberParams::default() };
        let (fs, n) = (64e9, 1024);
        let mut r = Rng::new(3);
        let x: Vec<C64> = r.complex_noise(n, 1e-3);
        let none = Some(Raman { isrs: false, ..Raman::default() });
        let a = Ssfm { amplification: Amplification::Ideal, ..Ssfm::new(fiber.clone(), 0.5) };
        let b = Ssfm { raman: none.clone(), ..a.clone() };
        let (ya, yb) = (a.propagate(&x, fs, None).unwrap(), b.propagate(&x, fs, None).unwrap());
        let err = ya.iter().zip(&yb).map(|(u, v)| (u - v).norm()).fold(0.0, f64::max);
        assert!(err < 1e-9 * 0.03, "{err}");
        let a = ManakovSsfm { amplification: Amplification::Ideal, ..ManakovSsfm::new(fiber.clone()) };
        let b = ManakovSsfm { raman: none.clone(), ..a.clone() };
        let mut ua = DualPol::new(x.clone(), x.iter().map(|v| v * 0.5).collect()).unwrap();
        let mut ub = ua.clone();
        a.propagate(&mut ua, fs, None).unwrap();
        b.propagate(&mut ub, fs, None).unwrap();
        let err = ua.x.iter().zip(&ub.x).chain(ua.y.iter().zip(&ub.y)).map(|(u, v)| (u - v).norm()).fold(0.0, f64::max);
        assert!(err < 1e-9 * 0.03, "{err}");
        // and the noise of the EDFAs: the variance of the noise added to a silent field
        let silent = vec![C64::new(0.0, 0.0); 1 << 16];
        let e = Ssfm { raman: none, ..Ssfm::new(FiberParams { n_spans: 1, ..fiber.clone() }, 1.0) };
        let y = e.propagate(&silent, fs, Some(&mut Rng::new(9))).unwrap();
        let var = y.iter().map(C64::norm_sqr).sum::<f64>() / y.len() as f64;
        let expected = Edfa { gain_db: 10.0, nf_db: fiber.nf_db, fc_hz: fiber.fc_hz }.noise_variance(fs);
        assert!((var / expected - 1.0).abs() < 0.02, "{var} vs {expected}");
    }
}

/// Checks against the published closed forms (`references/`).
#[cfg(test)]
mod literature {
    use super::*;
    use crate::optical::fiber::FiberParams;

    /// D. Semrau and P. Bayvel, "The Gaussian noise model in the presence of inter-channel
    /// stimulated Raman scattering," J. Lightw. Technol. 36(14), 2018 (arXiv 1801.02460),
    /// eq. (10) (p. 4): with a linear Raman gain slope `C_r`, the ISRS power transfer of
    /// a flat band of `B` after one span is `Δρ [dB] = 4.3 P_tot C_r L_eff B` between the
    /// band edges; the text reports the ODE and this closed form within 0.18 dB at
    /// 28 dBm (Fig. 1, 100 km, 0.2 dB/km). Checked at 19 and 22 dBm over 10 THz, within
    /// 5 % of the transfer plus 0.05 dB: the gain of the library scales with the pump
    /// frequency (`RamanGain::efficiency`), which the constant slope of the paper does
    /// not, and the band spans ±2.6 % of the carrier.
    #[test]
    fn semrau_2018_eq_10_isrs_power_transfer() {
        let fiber = FiberParams { span_km: 100.0, alpha_db_km: 0.2, ..FiberParams::default() };
        let c_r = 0.028; // 1/(W km THz), the fibre of the paper's Sec. IV
        let aeff_um2 = 80.0;
        // g_R A_eff (m/W) = C_r Δf A_eff / 1e3, linear up to 14 THz, zero beyond
        let gain =
            RamanGain::new(vec![0.0, 14e12], vec![0.0, c_r * 14.0 * aeff_um2 * 1e-12 / 1e3], fiber.fc_hz).unwrap();
        let raman = Raman { gain, aeff_um2, isrs: true, ..Raman::default() };
        let (n, spacing) = (101usize, 100e9);
        let l_eff = fiber.effective_length_km();
        for p_tot_dbm in [19.0, 22.0] {
            let p_ch = com_math::dbm2w(p_tot_dbm) / n as f64;
            let signals: Vec<(f64, f64)> = (0..n).map(|i| (fiber.fc_hz + (i as f64 - 50.0) * spacing, p_ch)).collect();
            let profile = raman.solve(&fiber, &signals).unwrap();
            let measured = profile.net_gain_db(0) - profile.net_gain_db(n - 1);
            let band_thz = (n - 1) as f64 * spacing * 1e-12;
            let paper = 4.3 * com_math::dbm2w(p_tot_dbm) * c_r * l_eff * band_thz;
            assert!(
                (measured - paper).abs() < 0.05 * paper + 0.05,
                "{p_tot_dbm} dBm: {measured:.3} dB vs {paper:.3} dB"
            );
        }
    }
}
