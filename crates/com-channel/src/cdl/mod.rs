//! Clustered delay line channels of 3GPP TR 38.901 (CDL-A to CDL-E).
//!
//! A clustered delay line is a tapped delay line whose taps are clusters of $M = 20$
//! rays, each ray leaving the transmit array in a direction (AOD, ZOD) and reaching the
//! receive array from a direction (AOA, ZOA). The geometry gives every ray its phase
//! across the antenna elements and its Doppler shift, so the spatial correlation of the
//! channel and its time variation follow from the angles of the table rather than from
//! a correlation matrix. [`Profile`] holds the five tables of TR 38.901 Section 7.7.1
//! verbatim (CDL-A, B, C without line of sight; CDL-D, E with a specular component in
//! the first cluster); [`Cdl`] builds the channel of one of them for two [`Array`]s and
//! applies it to a multi-stream signal through the tapped delay line of
//! [`crate::multipath::Taps`].
//!
//! # Model
//!
//! Section 7.7.1 of TR 38.901 generates the channel of a CDL profile by the steps of
//! Section 7.5 without the large-scale parameters. The normalised delays of the table
//! are scaled by the desired RMS delay spread $\mathrm{DS}$ (Section 7.7.3, (7.7-1)),
//! the cluster powers of the table are normalised to a unit sum, and the rays of cluster
//! $n$ take the angles (step 1, with the offsets $\alpha_{m}$ of Table 7.5-3)
//!
//! $$
//! \tau_{n} = \tau_{n, \mathrm{model}}\\, \mathrm{DS}, \qquad
//! \phi_{n, m, \mathrm{AOA}} = \phi_{n, \mathrm{AOA}} + c_{\mathrm{ASA}}\\, \alpha_{m}, \qquad
//! \theta_{n, m, \mathrm{ZOA}} = \theta_{n, \mathrm{ZOA}} + c_{\mathrm{ZSA}}\\, \alpha_{m},
//! $$
//!
//! and likewise for AOD and ZOD with $c_{\mathrm{ASD}}$ and $c_{\mathrm{ZSD}}$. The
//! departure and arrival rays of a cluster are coupled at random (step 2), the
//! cross-polarisation power ratio is $\kappa = 10^{X / 10}$ with $X$ the XPR of the
//! table (step 3), and four initial phases
//! $\Phi^{\theta\theta}_{n, m}, \Phi^{\theta\phi}_{n, m}, \Phi^{\phi\theta}_{n, m}, \Phi^{\phi\phi}_{n, m}$,
//! uniform on $(-\pi, \pi)$, are drawn per ray (step 4). The coefficient between
//! transmit element $s$ and receive element $u$ of cluster $n$ is then (7.5-22)
//!
//! $$
//! h_{u, s, n}(t) = \sqrt{\frac{P_{n}}{M}} \sum_{m = 1}^{M}
//! \begin{bmatrix} F_{\mathrm{rx}, u, \theta} \\\\ F_{\mathrm{rx}, u, \phi} \end{bmatrix}^{T}
//! \begin{bmatrix}
//! e^{j \Phi^{\theta\theta}_{n, m}} & \sqrt{\kappa^{-1}}\\, e^{j \Phi^{\theta\phi}_{n, m}} \\\\
//! \sqrt{\kappa^{-1}}\\, e^{j \Phi^{\phi\theta}_{n, m}} & e^{j \Phi^{\phi\phi}_{n, m}}
//! \end{bmatrix}
//! \begin{bmatrix} F_{\mathrm{tx}, s, \theta} \\\\ F_{\mathrm{tx}, s, \phi} \end{bmatrix}
//! e^{j 2\pi \hat{r}_{\mathrm{rx}, n, m}^{T} \bar{d}_{\mathrm{rx}, u} / \lambda}\\,
//! e^{j 2\pi \hat{r}_{\mathrm{tx}, n, m}^{T} \bar{d}_{\mathrm{tx}, s} / \lambda}\\,
//! e^{j 2\pi \hat{r}_{\mathrm{rx}, n, m}^{T} \bar{v}\\, t / \lambda},
//! $$
//!
//! with the field patterns evaluated at the angles of the ray, the spherical unit
//! vectors $\hat{r} = (\sin\theta \cos\phi, \sin\theta \sin\phi, \cos\theta)^{T}$
//! (7.5-23, 7.5-24), the element positions $\bar{d}$ and the velocity $\bar{v}$ of the
//! receiver. The specular component of CDL-D and CDL-E (7.5-29) has no random phase
//! and the polarisation matrix $\mathrm{diag}(1, -1)$; its distance phase
//! $e^{-j 2\pi d_{3D} / \lambda_{0}}$ is a constant, taken as 1 since the CDL tables
//! carry no geometry. Its power and that of the diffuse rays of cluster 1 are the two
//! rows of the table (K-factor 13.3 dB for CDL-D, 22 dB for CDL-E).
//!
//! The realisation is a tapped delay line with one tap per cluster at the delay
//! $\mathrm{round}(\tau_{n} f_{s})$ samples (two clusters may share a sample; their
//! coefficients then add in the convolution). The splitting of the strongest clusters
//! into sub-clusters of Section 7.5 step 11 (Table 7.5-5) is not applied.
//!
//! # Antenna arrays
//!
//! An [`Array`] lists its elements by position (in wavelengths, global coordinates:
//! the elements of a uniform linear array lie along $y$, broadside along $x$), slant
//! angle $\zeta$ and element pattern, without orientation (the local coordinate system
//! is the global one, Section 7.1). The field of an element of pattern $A(\theta, \phi)$
//! is (7.3-4, 7.3-5, model-2 of Section 7.3.2)
//!
//! $$
//! F_{\theta}(\theta, \phi) = \sqrt{A(\theta, \phi)} \cos\zeta, \qquad
//! F_{\phi}(\theta, \phi) = \sqrt{A(\theta, \phi)} \sin\zeta,
//! $$
//!
//! with $\zeta = 0$ for a vertical element and $\pm 45°$ for the two elements of a
//! dual-polarised pair. [`Pattern::Isotropic`] has $A = 1$; [`Pattern::Sectored`] is
//! the radiation power pattern of Table 7.3-1.
//!
//! # Left out
//!
//! The angle scaling of Section 7.7.5.1 (a desired angular spread per angle type) is
//! not implemented: it needs the angular spread of the model computed as in Annex A
//! over clusters and rays, and the standard is not explicit on whether the ray offsets
//! are scaled with the cluster angles. The delays and angles of the tables are used as
//! they are.
//!
//! # Examples
//! ```
//! use com_core::prelude::*;
//! use com_channel::cdl::{Array, Cdl, Profile};
//! let ch = Cdl::new(Profile::C, 300e-9, 30.72e6, Array::linear(2, 0.5), Array::linear(4, 0.5))?;
//! let x = Signal::zeros(2, 256, Hertz(30.72e6));
//! let (y, taps) = ch.fade(&x, &mut batch_rng(1, 0))?;
//! assert_eq!(y.n_streams(), 4);
//! assert_eq!(taps.delays().len(), 24);           // one tap per cluster
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TR 38.901 V17.0.0 (2022-03), *Study on channel model for frequencies from
//!    0.5 to 100 GHz*: Section 7.3 (antenna modelling, Table 7.3-1), Section 7.5 step 11
//!    (equations 7.5-22 to 7.5-29), Table 7.5-3 (ray offsets), Section 7.7.1 and Tables
//!    7.7.1-1 to 7.7.1-5 (CDL-A to CDL-E), Section 7.7.3 (delay spread scaling, 7.7-1).

mod array;
mod tables;

pub use array::{Array, Pattern};
pub use tables::{Cluster, Profile, RAY_OFFSETS, RAYS_PER_CLUSTER, Spreads};

use crate::multipath::Taps;
use array::{dot, unit_vector};
use com_core::prelude::*;
use com_core::{ensure, same_len};
use com_math::{SPEED_OF_LIGHT, cis, db2lin};
use std::f64::consts::PI;

/// A CDL channel between two arrays (see the module).
///
/// Its delays are put on the sampling grid of the delay spread and rate given at
/// construction ([`Cdl::delay_spread`], [`Cdl::sample_rate`]): the channel applies only
/// to signals of that rate.
#[derive(Clone, Debug, PartialEq)]
pub struct Cdl {
    clusters: Vec<Cluster>,
    los: Option<Cluster>,
    spreads: Spreads,
    powers: Vec<f64>,
    delays: Vec<usize>,
    delay_spread: f64,
    sample_rate: f64,
    /// Transmit array.
    pub tx: Array,
    /// Receive array.
    pub rx: Array,
    /// Velocity $\bar{v}$ of the receiver (m/s, global coordinates); zero for a channel
    /// constant in time.
    pub velocity: [f64; 3],
    /// Carrier frequency (Hz), which sets the wavelength of the Doppler shifts.
    pub carrier_frequency: f64,
}

impl Cdl {
    /// The channel of `profile` with the desired RMS delay spread `delay_spread` (s),
    /// sampled at `sample_rate` (Hz), between the arrays `tx` and `rx`, without motion
    /// (set [`Cdl::velocity`] and [`Cdl::carrier_frequency`] for Doppler).
    ///
    /// # Errors
    /// [`Error::Config`] for a negative delay spread, a non-positive rate, an array
    /// without element or with a slant per element missing.
    ///
    /// # Examples
    /// ```
    /// use com_channel::cdl::{Array, Cdl, Profile};
    /// let mut ch = Cdl::new(Profile::D, 30e-9, 61.44e6, Array::linear(1, 0.5), Array::linear_dual(2, 0.5))?;
    /// ch.velocity = [10.0, 0.0, 0.0];
    /// ch.carrier_frequency = 3.5e9;
    /// assert!((ch.max_doppler() - 116.74).abs() < 0.01);     // 10 m/s at 3.5 GHz
    /// assert_eq!(ch.delays().last(), Some(&23));             // 12.525 × 30 ns at 16.3 ns per sample
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn new(profile: Profile, delay_spread: f64, sample_rate: f64, tx: Array, rx: Array) -> Result<Cdl> {
        Cdl::custom(profile.table().to_vec(), profile.los(), profile.spreads(), delay_spread, sample_rate, tx, rx)
    }

    /// A channel of arbitrary clusters (in the units of the tables, powers in dB of any
    /// normalisation), an optional specular component `los` added to the first
    /// cluster, and per-cluster spreads.
    ///
    /// # Errors
    /// [`Error::Config`] for no cluster, a negative delay, a negative spread, a
    /// negative delay spread, a non-positive rate, an array without element or with a
    /// slant per element missing.
    pub fn custom(
        clusters: Vec<Cluster>,
        los: Option<Cluster>,
        spreads: Spreads,
        delay_spread: f64,
        sample_rate: f64,
        tx: Array,
        rx: Array,
    ) -> Result<Cdl> {
        ensure(!clusters.is_empty(), || "a CDL channel needs at least one cluster".into())?;
        ensure(clusters.iter().all(|c| c.delay >= 0.0), || "negative cluster delay".into())?;
        ensure(spreads.asd >= 0.0 && spreads.asa >= 0.0 && spreads.zsd >= 0.0 && spreads.zsa >= 0.0, || {
            "negative angular spread".into()
        })?;
        ensure(delay_spread >= 0.0, || format!("negative delay spread {delay_spread}"))?;
        ensure(sample_rate > 0.0, || format!("the sampling rate must be positive, got {sample_rate}"))?;
        for (name, a) in [("transmit", &tx), ("receive", &rx)] {
            ensure(a.n_elements() >= 1, || format!("the {name} array needs at least one element"))?;
            same_len(a.slants.len(), a.positions.len(), "slants and positions of the array")?;
        }
        let mut powers: Vec<f64> = clusters.iter().map(|c| db2lin(c.power_db)).collect();
        if let Some(s) = los {
            powers[0] += db2lin(s.power_db);
        }
        let tot: f64 = powers.iter().sum();
        let delays = clusters.iter().map(|c| (c.delay * delay_spread * sample_rate).round() as usize).collect();
        Ok(Cdl {
            clusters,
            los,
            spreads,
            powers: powers.iter().map(|p| p / tot).collect(),
            delays,
            delay_spread,
            sample_rate,
            tx,
            rx,
            velocity: [0.0; 3],
            carrier_frequency: 3.5e9,
        })
    }

    /// The clusters (the specular component apart).
    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }

    /// Delays of the clusters on the sampling grid (samples), one tap each.
    pub fn delays(&self) -> &[usize] {
        &self.delays
    }

    /// Desired RMS delay spread $\mathrm{DS}$ (s).
    pub fn delay_spread(&self) -> f64 {
        self.delay_spread
    }

    /// Sampling rate of the delays and of the time variation (Hz).
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Normalised powers $P_{n}$ of the clusters (unit sum, the specular component
    /// counted in the first).
    pub fn powers(&self) -> &[f64] {
        &self.powers
    }

    /// Receive antennas.
    pub fn n_rx(&self) -> usize {
        self.rx.n_elements()
    }

    /// Transmit antennas.
    pub fn n_tx(&self) -> usize {
        self.tx.n_elements()
    }

    /// The maximum Doppler shift $|\bar{v}| / \lambda$ (Hz).
    pub fn max_doppler(&self) -> f64 {
        dot(self.velocity, self.velocity).sqrt() * self.carrier_frequency / SPEED_OF_LIGHT
    }

    /// Draw a realisation: the ray coupling and initial phases of every cluster, from
    /// which [`Rays::taps`] gives the coefficients of any stretch of time.
    ///
    /// # Errors
    /// [`Error::Config`] for a non-positive carrier frequency with a non-zero velocity.
    pub fn draw_rays(&self, rng: &mut SimRng) -> Result<Rays> {
        let moving = self.velocity != [0.0; 3];
        ensure(self.carrier_frequency > 0.0 || !moving, || {
            format!("the carrier frequency must be positive for a moving receiver, got {}", self.carrier_frequency)
        })?;
        let inv_lambda = if moving { self.carrier_frequency / SPEED_OF_LIGHT } else { 0.0 };
        let kappa_inv_sqrt = db2lin(-self.spreads.xpr_db).sqrt();
        let (n_rx, n_tx) = (self.n_rx(), self.n_tx());
        let total = self.clusters.iter().map(|c| db2lin(c.power_db)).sum::<f64>()
            + self.los.map_or(0.0, |s| db2lin(s.power_db));
        let norm = 1.0 / total;
        let mut clusters = Vec::with_capacity(self.clusters.len());
        for (n, c) in self.clusters.iter().enumerate() {
            let mut rays = Vec::with_capacity(RAYS_PER_CLUSTER + 1);
            if n == 0 {
                if let Some(s) = self.los {
                    // (7.5-29): no random phase, polarisation matrix diag(1, -1)
                    let amp = (db2lin(s.power_db) * norm).sqrt();
                    rays.push(self.ray(s.aod, s.aoa, s.zod, s.zoa, amp, [0.0, 0.0, 0.0, PI], 0.0, inv_lambda));
                }
            }
            // step 1: ray angles; step 2: random coupling of the departure and arrival rays
            let aoa: Vec<f64> = RAY_OFFSETS.iter().map(|a| c.aoa + self.spreads.asa * a).collect();
            let zoa = permuted(RAY_OFFSETS.iter().map(|a| c.zoa + self.spreads.zsa * a).collect(), rng);
            let aod = permuted(RAY_OFFSETS.iter().map(|a| c.aod + self.spreads.asd * a).collect(), rng);
            let zod = permuted(RAY_OFFSETS.iter().map(|a| c.zod + self.spreads.zsd * a).collect(), rng);
            let amp = (db2lin(c.power_db) * norm / RAYS_PER_CLUSTER as f64).sqrt();
            for m in 0..RAYS_PER_CLUSTER {
                // step 4: initial phases Φ^θθ, Φ^θφ, Φ^φθ, Φ^φφ uniform on (-π, π)
                let phases = [
                    rng.uniform() * 2.0 * PI - PI,
                    rng.uniform() * 2.0 * PI - PI,
                    rng.uniform() * 2.0 * PI - PI,
                    rng.uniform() * 2.0 * PI - PI,
                ];
                rays.push(self.ray(aod[m], aoa[m], zod[m], zoa[m], amp, phases, kappa_inv_sqrt, inv_lambda));
            }
            clusters.push(rays);
        }
        Ok(Rays { delays: self.delays.clone(), n_rx, n_tx, sample_rate: self.sample_rate, clusters })
    }

    /// The ray of amplitude `amp` and angles given: its Doppler shift and the complex
    /// amplitude of every `[rx][tx]` link, the field terms and array phases of
    /// (7.5-22) included.
    fn ray(
        &self,
        aod: f64,
        aoa: f64,
        zod: f64,
        zoa: f64,
        amp: f64,
        phases: [f64; 4],
        kappa_inv_sqrt: f64,
        inv_lambda: f64,
    ) -> Ray {
        let r_rx = unit_vector(zoa, aoa);
        let r_tx = unit_vector(zod, aod);
        let m = [cis(phases[0]), cis(phases[1]) * kappa_inv_sqrt, cis(phases[2]) * kappa_inv_sqrt, cis(phases[3])];
        // per transmit element: array phase, F_tx,θ, F_tx,φ
        let tx: Vec<(C64, f64, f64)> = (0..self.n_tx())
            .map(|s| {
                let (ft, fp) = self.tx.field(s, zod, aod);
                (cis(2.0 * PI * dot(r_tx, self.tx.positions[s])), ft, fp)
            })
            .collect();
        let mut amplitude = Vec::with_capacity(self.n_rx() * self.n_tx());
        for u in 0..self.n_rx() {
            let (ft, fp) = self.rx.field(u, zoa, aoa);
            let e_rx = cis(2.0 * PI * dot(r_rx, self.rx.positions[u]));
            for &(e_tx, ft_tx, fp_tx) in &tx {
                // [F_rx,θ F_rx,φ] M [F_tx,θ; F_tx,φ]
                let pol = (m[0] * ft_tx + m[1] * fp_tx) * ft + (m[2] * ft_tx + m[3] * fp_tx) * fp;
                amplitude.push(pol * e_tx * e_rx * amp);
            }
        }
        Ray { aod, aoa, zod, zoa, doppler: dot(r_rx, self.velocity) * inv_lambda, amplitude }
    }

    /// Draw the taps for `n_samples` samples from time 0 (a realisation applied by
    /// [`Rays::taps`]).
    ///
    /// # Errors
    /// As [`Cdl::draw_rays`].
    pub fn draw(&self, n_samples: usize, rng: &mut SimRng) -> Result<Taps> {
        Ok(self.draw_rays(rng)?.taps(0, n_samples))
    }

    /// Apply the channel to `x` (`n_tx` streams), the samples before the signal being
    /// zero: the `n_rx` received streams and the taps of the realisation.
    ///
    /// # Errors
    /// [`Error::Shape`] if `x` does not have `n_tx` streams, [`Error::Config`] if it is
    /// not sampled at [`Cdl::sample_rate`], and as [`Cdl::draw_rays`].
    pub fn fade(&self, x: &Signal, rng: &mut SimRng) -> Result<(Signal, Taps)> {
        same_len(x.n_streams(), self.n_tx(), "streams and transmit antennas")?;
        crate::multipath::same_rate(x.sample_rate(), self.sample_rate, "the CDL channel")?;
        let n = x.n_samples();
        let taps = self.draw(n, rng)?;
        let y = taps.apply(&x.to_streams())?;
        Ok((Signal::from_streams(&y, x.sample_rate())?, taps))
    }
}

impl Propagate for Cdl {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        out.assign(&self.fade(x, rng)?.0);
        Ok(())
    }
}

/// A random permutation of `v` ([`SimRng::shuffle`]).
fn permuted(mut v: Vec<f64>, rng: &mut SimRng) -> Vec<f64> {
    rng.shuffle(&mut v);
    v
}

/// One ray of a realisation: its angles, Doppler shift and the complex amplitude of
/// every link.
#[derive(Clone, Debug, PartialEq)]
pub struct Ray {
    /// Azimuth of departure (degrees).
    pub aod: f64,
    /// Azimuth of arrival (degrees).
    pub aoa: f64,
    /// Zenith of departure (degrees).
    pub zod: f64,
    /// Zenith of arrival (degrees).
    pub zoa: f64,
    /// Doppler shift $\nu = \hat{r}_{\mathrm{rx}}^{T} \bar{v} / \lambda$ (Hz).
    pub doppler: f64,
    /// Amplitude of the ray on every link, `[rx][tx]` flattened: everything of
    /// (7.5-22) but the Doppler term.
    pub amplitude: Vec<C64>,
}

/// A realisation of a [`Cdl`] channel: the rays of every cluster, with their coupling,
/// phases and Doppler shifts drawn. The coefficients at any time follow
/// deterministically, so the taps of consecutive stretches join seamlessly.
#[derive(Clone, Debug, PartialEq)]
pub struct Rays {
    delays: Vec<usize>,
    n_rx: usize,
    n_tx: usize,
    sample_rate: f64,
    clusters: Vec<Vec<Ray>>,
}

impl Rays {
    /// The rays of cluster `n` (21 for the first cluster of a LOS profile, the specular
    /// one first; 20 otherwise).
    ///
    /// # Panics
    /// If `n` is not a cluster.
    pub fn cluster(&self, n: usize) -> &[Ray] {
        &self.clusters[n]
    }

    /// Number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Delays of the taps (samples).
    pub fn delays(&self) -> &[usize] {
        &self.delays
    }

    /// The taps from sample `start` for `n_samples` samples,
    /// $h_{u, s, n}(k / f_{s}) = \sum_{m} a_{u, s, n, m}\\, e^{j 2\pi \nu_{n, m} k / f_{s}}$
    /// for $k = \mathrm{start}, \ldots, \mathrm{start} + n_{\mathrm{samples}} - 1$.
    ///
    /// # Examples
    /// ```
    /// use com_core::prelude::*;
    /// use com_channel::cdl::{Array, Cdl, Profile};
    /// let mut ch = Cdl::new(Profile::A, 100e-9, 1e6, Array::linear(1, 0.5), Array::linear(2, 0.5))?;
    /// ch.velocity = [30.0, 0.0, 0.0];
    /// let rays = ch.draw_rays(&mut batch_rng(5, 0))?;
    /// let whole = rays.taps(0, 64);
    /// let second_half = rays.taps(32, 32);
    /// assert_eq!(whole.coefficient(3, 40, 1, 0), second_half.coefficient(3, 8, 1, 0));
    /// # Ok::<(), com_core::Error>(())
    /// ```
    ///
    /// Without Doppler (every ray of zero shift, a receiver at rest) the taps are the
    /// same at every sample and are stored once ([`Taps::is_time_invariant`]).
    pub fn taps(&self, start: usize, n_samples: usize) -> Taps {
        let links = self.n_rx * self.n_tx;
        // a zero shift gives the phase factor 1 at every time: the coefficients of the
        // first sample are those of every sample, bit for bit
        let invariant = self.clusters.iter().flatten().all(|ray| ray.doppler == 0.0);
        let stored = if invariant { n_samples.min(1) } else { n_samples };
        let coeffs = self
            .clusters
            .iter()
            .map(|rays| {
                let mut c = vec![C64::new(0.0, 0.0); stored * links];
                for k in 0..stored {
                    let t = (start + k) as f64 / self.sample_rate;
                    for ray in rays {
                        let e = cis(2.0 * PI * ray.doppler * t);
                        for (out, a) in c[k * links..(k + 1) * links].iter_mut().zip(&ray.amplitude) {
                            *out += a * e;
                        }
                    }
                }
                c
            })
            .collect();
        // the sizes are consistent by construction
        let taps = if invariant && n_samples > 0 {
            Taps::time_invariant(self.delays.clone(), self.n_rx, self.n_tx, n_samples, coeffs)
        } else {
            Taps::new(self.delays.clone(), self.n_rx, self.n_tx, n_samples, coeffs)
        };
        taps.unwrap_or_else(|_| {
            unreachable!("the taps of a realisation have one coefficient vector per cluster of the right length")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::tables::*;
    use super::*;
    use crate::multipath::rms_delay_spread;

    fn vertical(profile: Profile, n_rx: usize) -> Cdl {
        Cdl::new(profile, 100e-9, 1e6, Array::linear(1, 0.5), Array::linear(n_rx, 0.5)).unwrap()
    }

    #[test]
    fn errors_on_invalid_configurations() {
        assert!(Cdl::new(Profile::A, -1.0, 1e6, Array::linear(1, 0.5), Array::linear(1, 0.5)).is_err());
        assert!(Cdl::new(Profile::A, 1e-7, 0.0, Array::linear(1, 0.5), Array::linear(1, 0.5)).is_err());
        assert!(Cdl::new(Profile::A, 1e-7, 1e6, Array::linear(0, 0.5), Array::linear(1, 0.5)).is_err());
        let bad = Array { slants: vec![0.0], ..Array::linear(2, 0.5) };
        assert!(Cdl::new(Profile::A, 1e-7, 1e6, Array::linear(1, 0.5), bad).is_err());
        let mut ch = vertical(Profile::A, 1);
        ch.velocity = [1.0, 0.0, 0.0];
        ch.carrier_frequency = 0.0;
        assert!(ch.draw_rays(&mut Rng::new(0)).is_err());
        let ch = vertical(Profile::A, 1);
        assert!(ch.fade(&Signal::zeros(2, 8, Hertz(1e6)), &mut Rng::new(0)).is_err());
        assert!("CDL-X".parse::<Profile>().is_err());
        assert_eq!((ch.delay_spread(), ch.sample_rate()), (100e-9, 1e6));
        assert_eq!("b".parse::<Profile>().unwrap(), Profile::B);
    }

    #[test]
    fn a_cdl_rejects_a_signal_of_another_sample_rate() {
        let mut ch = vertical(Profile::C, 2);
        ch.velocity = [0.0, 15.0, 0.0];
        assert!(ch.fade(&Signal::zeros(1, 64, Hertz(1e6)), &mut Rng::new(0)).is_ok());
        // four times oversampled: the delays and the Doppler phase per sample would be wrong
        let x = Signal::zeros(1, 64, Hertz(4e6));
        assert!(matches!(ch.fade(&x, &mut Rng::new(0)), Err(Error::Config(_))));
        let mut out = Signal::zeros(2, 0, Hertz(1.0));
        assert!(ch.propagate_into(&x, &mut Rng::new(0), &mut out).is_err());
    }

    #[test]
    fn powers_are_normalised_and_the_pdp_has_the_desired_spread() {
        for p in [Profile::A, Profile::B, Profile::C, Profile::D, Profile::E] {
            let tot: f64 = p.cluster_powers().iter().sum();
            assert!((tot - 1.0).abs() < 1e-12);
            let pdp = p.pdp(300e-9);
            assert_eq!(pdp.len(), p.n_clusters());
            let ds = rms_delay_spread(&pdp);
            // the tables are normalised to a unit spread within the rounding of their entries (< 2 %)
            assert!((ds / 300e-9 - 1.0).abs() < 0.02, "{p:?}: {ds:e}");
            let ch = Cdl::new(p, 300e-9, 1e9, Array::linear(1, 0.5), Array::linear(1, 0.5)).unwrap();
            for (a, b) in ch.powers().iter().zip(p.cluster_powers()) {
                assert!((a - b).abs() < 1e-15);
            }
            assert_eq!(ch.delays()[1], (p.table()[1].delay * 300.0).round() as usize);
        }
    }

    #[test]
    fn mean_tap_powers_match_the_normalised_cluster_powers() {
        // One realisation gives |h_n|² of relative standard deviation ≈ 1 (a sum of 20
        // unit phasors is close to Rayleigh); the average of 4000 draws has a relative
        // standard error of 1.6 %, so 5 % is three standard errors.
        let ch = vertical(Profile::A, 1);
        let mut rng = Rng::new(11);
        let trials = 4000;
        let mut p = vec![0.0; ch.powers().len()];
        for _ in 0..trials {
            let t = ch.draw(1, &mut rng).unwrap();
            for (n, acc) in p.iter_mut().enumerate() {
                *acc += t.coefficient(n, 0, 0, 0).norm_sqr();
            }
        }
        for (n, (&got, &want)) in p.iter().zip(ch.powers()).enumerate() {
            let got = got / f64::from(trials);
            assert!((got / want - 1.0).abs() < 0.05, "cluster {n}: {got:e} against {want:e}");
        }
    }

    #[test]
    fn doppler_of_every_ray_is_bounded_and_the_tap_spectrum_too() {
        let mut ch = Cdl::new(Profile::B, 100e-9, 4000.0, Array::linear(1, 0.5), Array::linear(1, 0.5)).unwrap();
        ch.velocity = [20.0, -10.0, 3.0];
        ch.carrier_frequency = 2e9;
        let f_max = ch.max_doppler(); // 150 Hz
        let rays = ch.draw_rays(&mut Rng::new(3)).unwrap();
        for n in 0..rays.n_clusters() {
            for r in rays.cluster(n) {
                assert!(r.doppler.abs() <= f_max * (1.0 + 1e-12), "{} > {f_max}", r.doppler);
            }
        }
        // spectrum of tap 5 over 4096 samples (bin 0.98 Hz) with a Hann window: the
        // sidelobes fall as 1/f³, so the energy more than 16 bins past f_max is < 1e-3
        let n = 4096;
        let taps = rays.taps(0, n);
        let h: Vec<C64> = (0..n)
            .map(|k| {
                let w = 0.5 - 0.5 * (2.0 * PI * k as f64 / n as f64).cos();
                taps.coefficient(5, k, 0, 0) * w
            })
            .collect();
        let spec = com_math::fft::fft(&h);
        let bin = ch.sample_rate() / n as f64;
        let (mut inside, mut outside) = (0.0, 0.0);
        for (k, s) in spec.iter().enumerate() {
            let f = if k < n / 2 { k as f64 } else { k as f64 - n as f64 } * bin;
            if f.abs() <= f_max + 16.0 * bin { inside += s.norm_sqr() } else { outside += s.norm_sqr() }
        }
        assert!(outside / (inside + outside) < 1e-3, "{}", outside / (inside + outside));
    }

    #[test]
    fn same_seed_same_channel_and_chunks_join() {
        let mut ch = vertical(Profile::C, 2);
        ch.velocity = [0.0, 15.0, 0.0];
        let x = Signal::from_streams(&[(0..100).map(|i| C64::new(i as f64, 1.0)).collect()], Hertz(1e6)).unwrap();
        let (y1, t1) = ch.fade(&x, &mut batch_rng(9, 2)).unwrap();
        let (y2, t2) = ch.fade(&x, &mut batch_rng(9, 2)).unwrap();
        assert_eq!(t1, t2);
        assert_eq!(y1.as_array(), y2.as_array());
        assert_ne!(t1, ch.fade(&x, &mut batch_rng(9, 3)).unwrap().1);
        let rays = ch.draw_rays(&mut batch_rng(9, 2)).unwrap();
        let (a, b) = (rays.taps(0, 37), rays.taps(37, 63));
        for n in 0..rays.n_clusters() {
            for k in 0..100 {
                for r in 0..2 {
                    let want = t1.coefficient(n, k, r, 0);
                    let got = if k < 37 { a.coefficient(n, k, r, 0) } else { b.coefficient(n, k - 37, r, 0) };
                    assert_eq!(got, want);
                }
            }
        }
    }

    #[test]
    fn adjacent_elements_of_a_half_wavelength_ula_are_correlated_by_the_arrival_angle() {
        // one cluster, no angular spread, no cross-polarisation: h_u = Σ_m a_m e^{jπ u sin φ}
        for aoa in [0.0, 30.0, -70.0, 120.0] {
            let cluster = cl(0.0, 0.0, 10.0, aoa, 95.0, 90.0);
            let spreads = Spreads { asd: 0.0, asa: 0.0, zsd: 0.0, zsa: 0.0, xpr_db: 300.0 };
            let ch =
                Cdl::custom(vec![cluster], None, spreads, 100e-9, 1e6, Array::linear(1, 0.5), Array::linear(3, 0.5))
                    .unwrap();
            let t = ch.draw(1, &mut Rng::new(4)).unwrap();
            let (h0, h1, h2) = (t.coefficient(0, 0, 0, 0), t.coefficient(0, 0, 1, 0), t.coefficient(0, 0, 2, 0));
            let want = cis(PI * aoa.to_radians().sin());
            assert!((h1 / h0 - want).norm() < 1e-12, "aoa {aoa}: {}", h1 / h0);
            assert!((h2 / h1 - want).norm() < 1e-12);
        }
    }

    #[test]
    fn the_sectored_pattern_and_dual_polarisation_follow_the_field_equations() {
        assert!((Pattern::Sectored.gain_db(90.0, 180.0) - (8.0 - 30.0)).abs() < 1e-12);
        assert!((Pattern::Sectored.gain_db(0.0, 0.0) - (8.0 - 12.0 * (90.0f64 / 65.0).powi(2))).abs() < 1e-12);
        assert!((Pattern::Sectored.gain_db(90.0, -65.0) - (8.0 - 12.0)).abs() < 1e-12);
        let a = Array { pattern: Pattern::Sectored, ..Array::linear_dual(1, 0.5) };
        let (ft, fp) = a.field(0, 90.0, 0.0);
        assert!((ft - db2lin(8.0).sqrt() * 0.5f64.sqrt()).abs() < 1e-12);
        assert!((fp - ft).abs() < 1e-12);
        let (_, fp) = a.field(1, 90.0, 0.0);
        assert!(fp < 0.0);
        // a dual-polarised link at XPR κ: E|h|² = P (1 + 1/κ) / 2 (co-polar: Φ^θθ and Φ^φφ
        // of weight ½, cross terms ½ κ⁻¹); 3000 draws → 1.8 % standard error, 6 % ≈ 3σ
        let ch = Cdl::new(Profile::E, 30e-9, 1e6, Array::linear_dual(1, 0.5), Array::linear_dual(1, 0.5)).unwrap();
        let mut rng = Rng::new(8);
        let mut p = 0.0;
        for _ in 0..3000 {
            p += ch.draw(1, &mut rng).unwrap().coefficient(1, 0, 0, 0).norm_sqr();
        }
        let want = ch.powers()[1] * (1.0 + db2lin(-8.0)) / 2.0;
        assert!((p / 3000.0 / want - 1.0).abs() < 0.06, "{}", p / 3000.0 / want);
    }

    #[test]
    fn a_static_channel_is_a_convolution_of_the_streams() {
        let ch = Cdl::new(Profile::D, 1e-6, 1e6, Array::linear(2, 0.5), Array::linear(1, 0.5)).unwrap();
        let mut rng = Rng::new(5);
        let x: Vec<Vec<C64>> = (0..2).map(|_| (0..40).map(|_| rng.complex_gaussian()).collect()).collect();
        let sig = Signal::from_streams(&x, Hertz(1e6)).unwrap();
        let (y, taps) = ch.fade(&sig, &mut rng).unwrap();
        let mut want = vec![C64::new(0.0, 0.0); 40];
        for (l, &d) in taps.delays().iter().enumerate() {
            for t in 0..2 {
                for i in d..40 {
                    want[i] += taps.coefficient(l, 0, 0, t) * x[t][i - d];
                }
            }
        }
        for i in 0..40 {
            assert!((y.stream(0)[i] - want[i]).norm() < 1e-12);
        }
        assert!(taps.delays().contains(&13), "{:?}", taps.delays()); // 12.525 µs
        // stored once, the same coefficient at every sample
        assert!(taps.is_time_invariant());
        assert_eq!(taps.coefficient(2, 0, 0, 1), taps.coefficient(2, 39, 0, 1));
    }
}

/// Checks against the published tables (3GPP TR 38.901 V17.0.0).
#[cfg(test)]
mod literature {
    use super::tables::*;
    use super::*;
    use crate::multipath::rms_delay_spread;
    use com_math::lin2db;

    const ALL: [Profile; 5] = [Profile::A, Profile::B, Profile::C, Profile::D, Profile::E];

    /// TR 38.901 V17.0.0, Tables 7.7.1-1 to 7.7.1-5 (Section 7.7.1, pp. 77–80): the
    /// number of clusters of every profile and the rows of several clusters, verbatim.
    #[test]
    #[allow(clippy::approx_constant, reason = "0.6366 is the delay of cluster 6 of CDL-C")]
    fn tr_38901_tables_7_7_1_1_to_5_entries() {
        assert_eq!(ALL.map(Profile::n_clusters), [23, 23, 24, 13, 14]);
        // Table 7.7.1-1 (CDL-A), clusters 1, 9 and 23
        assert_eq!(CDL_A[0], cl(0.0, -13.4, -178.1, 51.3, 50.2, 125.4));
        assert_eq!(CDL_A[8], cl(0.7618, -7.5, -81.7, -41.9, 55.2, 56.0));
        assert_eq!(CDL_A[22], cl(9.6586, -29.7, -56.2, 55.2, 144.9, 156.1));
        // Table 7.7.1-2 (CDL-B), clusters 1, 16 and 23
        assert_eq!(CDL_B[0], cl(0.0, 0.0, 9.3, -173.3, 105.8, 78.9));
        assert_eq!(CDL_B[15], cl(1.7842, -1.9, 30.6, -139.1, 103.0, 78.0));
        assert_eq!(CDL_B[22], cl(4.7834, -11.3, -77.6, -60.4, 115.4, 62.3));
        // Table 7.7.1-3 (CDL-C), clusters 3, 6 and 24
        assert_eq!(CDL_C[2], cl(0.2219, -3.5, -22.8, 120.0, 98.6, 72.1));
        assert_eq!(CDL_C[5], cl(0.6366, 0.0, 0.3, 170.4, 99.2, 75.3));
        assert_eq!(CDL_C[23], cl(8.6523, -22.8, -123.8, 33.6, 107.8, 57.0));
        // Table 7.7.1-4 (CDL-D), cluster 1 (specular and Laplacian rows) and 13
        assert_eq!(Profile::D.los(), Some(cl(0.0, -0.2, 0.0, -180.0, 98.5, 81.5)));
        assert_eq!(CDL_D[0], cl(0.0, -13.5, 0.0, -180.0, 98.5, 81.5));
        assert_eq!(CDL_D[12], cl(12.525, -27.7, -44.4, -78.7, 103.8, 71.3));
        // Table 7.7.1-5 (CDL-E), cluster 1 (both rows) and 14
        assert_eq!(Profile::E.los(), Some(cl(0.0, -0.03, 0.0, -180.0, 99.6, 80.4)));
        assert_eq!(CDL_E[0], cl(0.0, -22.03, 0.0, -180.0, 99.6, 80.4));
        assert_eq!(CDL_E[13], cl(20.6419, -29.2, 57.6, -72.4, 97.1, 86.4));
        // the per-cluster parameters, last rows of each table
        let s = ALL.map(Profile::spreads);
        assert_eq!([s[0].asd, s[0].asa, s[0].zsd, s[0].zsa, s[0].xpr_db], [5.0, 11.0, 3.0, 3.0, 10.0]);
        assert_eq!([s[1].asd, s[1].asa, s[1].zsd, s[1].zsa, s[1].xpr_db], [10.0, 22.0, 3.0, 7.0, 8.0]);
        assert_eq!([s[2].asd, s[2].asa, s[2].zsd, s[2].zsa, s[2].xpr_db], [2.0, 15.0, 3.0, 7.0, 7.0]);
        assert_eq!([s[3].asd, s[3].asa, s[3].zsd, s[3].zsa, s[3].xpr_db], [5.0, 8.0, 3.0, 3.0, 11.0]);
        assert_eq!([s[4].asd, s[4].asa, s[4].zsd, s[4].zsa, s[4].xpr_db], [5.0, 11.0, 3.0, 7.0, 8.0]);
        // Table 7.5-3: the ray offsets, symmetric, 20 rays
        assert_eq!(RAY_OFFSETS[18], 2.1551);
        assert!(RAY_OFFSETS.chunks(2).all(|p| p[0] == -p[1]));
    }

    /// TR 38.901 V17.0.0, Section 7.7.3: the delays of the tables are normalised to a
    /// unit RMS delay spread and (7.7-1) scales them by the desired one. Computed from
    /// the tables (the specular row of CDL-D/E included), the normalised spreads are
    /// 1.00006 (A), 0.99999 (B), 1.00000 (C), 0.99372 (D) and 1.00002 (E): the entries
    /// are rounded to 4 decimals (3 for CDL-D) and 0.1 dB, hence within 2 %, and the
    /// scaled profile has the desired spread to the same accuracy.
    #[test]
    fn tr_38901_section_7_7_3_unit_delay_spread_and_its_scaling() {
        let measured = [1.00006, 0.99999, 1.00000, 0.99372, 1.00002];
        for (p, m) in ALL.iter().zip(measured) {
            let ds = rms_delay_spread(&p.pdp(1.0));
            assert!((ds - 1.0).abs() < 0.02, "{p:?}: {ds}");
            assert!((ds - m).abs() < 1e-5, "{p:?}: {ds} against the documented {m}");
            let scaled = rms_delay_spread(&p.pdp(100e-9));
            assert!((scaled - ds * 100e-9).abs() < 1e-15, "{p:?}: {scaled:e}");
        }
    }

    /// TR 38.901 V17.0.0, Tables 7.7.1-4 and 7.7.1-5: the K-factor of the first cluster
    /// is 13.3 dB (CDL-D, −0.2 against −13.5 dB) and 22 dB (CDL-E, −0.03 against −22.03
    /// dB). The specular component (7.5-29) is deterministic: over the random phases of
    /// the Laplacian rays, the mean of $h_{1}$ is the specular component and the power
    /// about the mean is the Laplacian power. With 10 000 draws the diffuse power is
    /// estimated to 1 % (0.04 dB), so 0.2 dB is far beyond three standard errors.
    #[test]
    fn tr_38901_tables_7_7_1_4_and_5_k_factor_of_the_first_cluster() {
        assert!((Profile::D.k_factor_db().unwrap_or(0.0) - 13.3).abs() < 1e-12);
        assert!((Profile::E.k_factor_db().unwrap_or(0.0) - 22.0).abs() < 1e-12);
        assert_eq!(Profile::A.k_factor_db(), None);
        for (p, want) in [(Profile::D, 13.3), (Profile::E, 22.0)] {
            let ch = Cdl::new(p, 100e-9, 1e6, Array::linear(1, 0.5), Array::linear(1, 0.5)).unwrap();
            let mut rng = Rng::new(21);
            let trials = 10_000;
            let draws: Vec<C64> = (0..trials).map(|_| ch.draw(1, &mut rng).unwrap().coefficient(0, 0, 0, 0)).collect();
            let mean = draws.iter().sum::<C64>() / trials as f64;
            let diffuse = draws.iter().map(|h| (h - mean).norm_sqr()).sum::<f64>() / trials as f64;
            let specular = db2lin(p.los().unwrap_or(CDL_D_LOS).power_db) * p.normalisation();
            // the mean of 10 000 draws of the Laplacian part is 0 ± √(P_lap / 10⁴), 0.4 % of
            // the specular amplitude for CDL-D: 2 % on the power is five standard errors
            assert!((mean.norm_sqr() / specular - 1.0).abs() < 2e-2, "{p:?}: specular {}", mean.norm_sqr());
            let k_db = lin2db(mean.norm_sqr() / diffuse);
            assert!((k_db - want).abs() < 0.2, "{p:?}: K = {k_db} dB against {want}");
            // the first tap carries the two rows of the table
            assert!((ch.powers()[0] - specular * (1.0 + db2lin(-want))).abs() < 1e-12);
        }
    }
}
