//! The indoor optical wireless channel: Lambertian emitters (LED lights, infrared
//! transmitters), a detector with a field of view, and the reflections of an empty
//! rectangular room, as an impulse response of the received optical power.
//!
//! # Model
//!
//! An emitter of power $P$ at $\mathbf{r}_{S}$ pointing along $\hat{\mathbf{n}}_{S}$
//! radiates the generalized Lambertian intensity $R(\phi) = \frac{n + 1}{2\pi} P
//! \cos^{n}\phi$ ([`Emitter`]), of mode $n = -\ln 2 / \ln\cos\Phi_{1/2}$ for a semi-angle at
//! half power $\Phi_{1/2}$ ([`lambertian_mode`]). A detector of area $A$, field of view
//! $\Psi$, optical filter gain $T_{s}$ and concentrator gain $g(\psi) =
//! n_{c}^{2}/\sin^{2}\Psi$ ([`Detector`]) at the distance $R$ receives along the line of
//! sight, after the delay $R/c$, the power (Barry et al. 1993, eqs. (4)–(9); Komine &
//! Nakagawa 2004, eqs. (6)–(8))
//!
//! $$
//! P_{r} = P \frac{n + 1}{2\pi R^{2}} \cos^{n}\phi \; A \cos\psi \; T_{s}\, g(\psi), \qquad
//! 0 \le \psi \le \Psi .
//! $$
//!
//! The walls, ceiling and floor are ideal diffuse (Lambertian) reflectors of
//! reflectivity $\rho$ ([`Room`]): a reflecting element of area $\Delta A$ is a detector
//! with a field of view of 90° and then a source of mode 1 and power $\rho\, dP$, so that
//! the light reflected $k$ times is computed recursively from the light reflected
//! $k - 1$ times (Barry et al. 1993, eqs. (10)–(13)):
//!
//! $$
//! h^{(k)}(t; S, R) = \sum_{i} h^{(0)}(t; S, \mathcal{E}_{i}) \otimes
//! h^{(k-1)}(t; \mathcal{E}_{i}, R),
//! $$
//!
//! the surfaces being divided into elements ([`Resolution`]: $N_{x}, N_{y}, N_{z}$ per
//! dimension, a resolution per order of reflection) and time into bins of width $\Delta
//! t$. [`Room::illuminate`] carries the light from the emitters to the elements once;
//! [`Illumination::impulse_responses`] then gives, for any detector, the response of
//! the line of sight and of each order of reflection ([`ImpulseResponse`]: the power
//! received in each bin, for the powers of the emitters).
//!
//! With the impulse response, [`ImpulseResponse::bit_split`] splits the power received
//! by on–off keying with rectangular pulses into the part inside the bit and the part
//! that spills into the next ones (Komine & Nakagawa 2004, eqs. (17) and (19)), and
//! [`illuminance_lx`] gives the illuminance of the lights (their eq. (5)).
//!
//! # Examples
//! ```
//! use com_channel::owc::indoor::{direction, lambertian_mode, Detector, Emitter, Resolution, Room};
//! // a Lambertian source at the ceiling of a 5 × 5 × 3 m room pointing down, a
//! // detector of 1 cm² on the floor pointing up (Barry et al. 1993, configuration A)
//! let room = Room::uniform([5.0, 5.0, 3.0], 0.8)?;
//! let down = direction(-std::f64::consts::FRAC_PI_2, 0.0);
//! let up = direction(std::f64::consts::FRAC_PI_2, 0.0);
//! let source = Emitter::new([2.5, 2.5, 3.0], down, lambertian_mode(60f64.to_radians())?, 1.0);
//! let detector = Detector::new([0.5, 1.0, 0.0], up, 1e-4, 85f64.to_radians());
//! let lit = room.illuminate(&[source], &Resolution::new(0.2e-9, vec![[50, 50, 30]]))?;
//! let h = lit.impulse_responses(&detector)?;
//! // the line of sight carries 1.23 µW (their Fig. 3), the first reflections about 0.5 µW
//! assert!((h[0].total_w() / 1.23e-6 - 1.0).abs() < 0.01);
//! assert!((h[1].total_w() / 0.505e-6 - 1.0).abs() < 0.05);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. R. Barry, J. M. Kahn, W. J. Krause, E. A. Lee and D. G. Messerschmitt,
//!    "Simulation of multipath impulse response for indoor wireless optical channels,"
//!    *IEEE J. Sel. Areas Commun.*, 11(3), 367–379, 1993.
//! 2. T. Komine and M. Nakagawa, "Fundamental analysis for visible-light communication
//!    system using LED lights," *IEEE Trans. Consum. Electron.*, 50(1), 100–107, 2004.

use com_core::{Error, Result, ensure};
use com_math::geometry::{self, Vec3, dot, norm, sub};
use com_math::{C64, SPEED_OF_LIGHT, cis};
use std::f64::consts::{FRAC_PI_2, PI};

fn unit(a: Vec3) -> Result<Vec3> {
    geometry::unit(a).ok_or_else(|| Error::Config("a direction must be a non-zero vector".into()))
}

/// The Lambertian mode $n = -\ln 2 / \ln\cos\Phi_{1/2}$ of a source of semi-angle at half
/// power $\Phi_{1/2}$ (`semi_angle_rad`): 1 for 60°, 0.65 for the 70° of Komine &
/// Nakagawa (2004), whose text prints the formula without its minus sign.
///
/// # Errors
/// [`com_core::Error::Config`] unless $0 < \Phi_{1/2} < \pi/2$.
pub fn lambertian_mode(semi_angle_rad: f64) -> Result<f64> {
    ensure(semi_angle_rad > 0.0 && semi_angle_rad < FRAC_PI_2, || {
        format!("the semi-angle must lie in (0, π/2), got {semi_angle_rad} rad")
    })?;
    Ok(-std::f64::consts::LN_2 / semi_angle_rad.cos().ln())
}

/// The unit vector of elevation `elevation_rad` above the horizontal plane and azimuth
/// `azimuth_rad` from $x$ towards $y$ (the angles of Table I of Barry et al. 1993:
/// $-\pi/2$ points straight down, $\pi/2$ straight up).
pub fn direction(elevation_rad: f64, azimuth_rad: f64) -> [f64; 3] {
    let (se, ce) = elevation_rad.sin_cos();
    let (sa, ca) = azimuth_rad.sin_cos();
    [ce * ca, ce * sa, se]
}

/// A generalized Lambertian emitter (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    /// Position (m).
    pub position: [f64; 3],
    /// Direction of the axis (unit vector, see [`direction`]).
    pub direction: [f64; 3],
    /// Lambertian mode $n$ ([`lambertian_mode`]).
    pub mode: f64,
    /// Optical power (W).
    pub power_w: f64,
}

impl Emitter {
    /// An emitter at `position` pointing along `direction`.
    pub fn new(position: [f64; 3], direction: [f64; 3], mode: f64, power_w: f64) -> Emitter {
        Emitter { position, direction, mode, power_w }
    }

    /// The radiant intensity (W/sr) at the angle of cosine `cos_phi` from the axis.
    fn intensity(&self, cos_phi: f64) -> f64 {
        if cos_phi <= 0.0 { 0.0 } else { self.power_w * (self.mode + 1.0) / (2.0 * PI) * cos_phi.powf(self.mode) }
    }

    fn check(&self) -> Result<Emitter> {
        ensure(self.mode >= 0.0 && self.power_w >= 0.0, || {
            format!("an emitter needs a non-negative mode and power, got {} and {} W", self.mode, self.power_w)
        })?;
        Ok(Emitter { direction: unit(self.direction)?, ..*self })
    }
}

/// A detector: area, field of view, optical filter and concentrator (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detector {
    /// Position (m).
    pub position: [f64; 3],
    /// Direction of the normal (unit vector, see [`direction`]).
    pub direction: [f64; 3],
    /// Physical area $A$ (m²).
    pub area_m2: f64,
    /// Field of view $\Psi$ (rad): the largest angle of incidence detected.
    pub fov_rad: f64,
    /// Gain $T_{s}$ of the optical filter.
    pub filter_gain: f64,
    /// Refractive index $n_{c}$ of a non-imaging concentrator of gain
    /// $n_{c}^{2}/\sin^{2}\Psi$; `None` without a concentrator.
    pub concentrator_index: Option<f64>,
}

impl Detector {
    /// A bare detector (no filter, no concentrator).
    pub fn new(position: [f64; 3], direction: [f64; 3], area_m2: f64, fov_rad: f64) -> Detector {
        Detector { position, direction, area_m2, fov_rad, filter_gain: 1.0, concentrator_index: None }
    }

    /// The effective area $A \cos\psi\, T_{s}\, g(\psi)$ at the angle of incidence of cosine
    /// `cos_psi` (0 outside the field of view).
    fn effective_area(&self, cos_psi: f64) -> f64 {
        if cos_psi <= 0.0 || cos_psi < self.fov_rad.cos() {
            return 0.0;
        }
        let g = self.concentrator_index.map_or(1.0, |n| n * n / self.fov_rad.sin().powi(2));
        self.area_m2 * cos_psi * self.filter_gain * g
    }

    fn check(&self) -> Result<Detector> {
        ensure(self.area_m2 > 0.0 && self.fov_rad > 0.0 && self.fov_rad <= FRAC_PI_2, || {
            format!(
                "a detector needs a positive area and a field of view in (0, π/2], got {} m² and {} rad",
                self.area_m2, self.fov_rad
            )
        })?;
        ensure(self.filter_gain >= 0.0 && self.concentrator_index.is_none_or(|n| n >= 1.0), || {
            "the filter gain must be non-negative and the refractive index at least 1".into()
        })?;
        Ok(Detector { direction: unit(self.direction)?, ..*self })
    }
}

/// The reflectivities of the six surfaces of a [`Room`]: north and south are the walls
/// $x = L$ and $x = 0$, west and east the walls $y = W$ and $y = 0$ (north along $x$
/// and west along $y$, as in Barry et al. 1993).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reflectance {
    /// Wall $x = L$.
    pub north: f64,
    /// Wall $x = 0$.
    pub south: f64,
    /// Wall $y = 0$.
    pub east: f64,
    /// Wall $y = W$.
    pub west: f64,
    /// Ceiling $z = H$.
    pub ceiling: f64,
    /// Floor $z = 0$.
    pub floor: f64,
}

/// An empty rectangular room $[0, L] \times [0, W] \times [0, H]$ with diffuse walls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Room {
    /// $(L, W, H)$ (m).
    pub size_m: [f64; 3],
    /// The reflectivities of the surfaces.
    pub reflectance: Reflectance,
}

/// The time bins and, for every order of reflection $k = 1, 2, \dots$, the number of
/// elements $(N_{x}, N_{y}, N_{z})$ into which the length, width and height are divided
/// (Barry et al. 1993, Table I): the number of reflections is the length of
/// `partitions`.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    /// Width $\Delta t$ of a time bin (s).
    pub bin_s: f64,
    /// $(N_{x}, N_{y}, N_{z})$ of each order of reflection.
    pub partitions: Vec<[usize; 3]>,
}

impl Resolution {
    /// Bins of `bin_s` and the partitions of each order of reflection.
    pub fn new(bin_s: f64, partitions: Vec<[usize; 3]>) -> Resolution {
        Resolution { bin_s, partitions }
    }
}

/// The power received in successive time bins: bin `i` holds the power of the light
/// arriving between $i \Delta t$ and $(i + 1)\Delta t$ after the emission (W, for the
/// powers of the emitters).
#[derive(Clone, Debug, PartialEq)]
pub struct ImpulseResponse {
    /// Width $\Delta t$ of a bin (s).
    pub bin_s: f64,
    /// Power received in each bin (W).
    pub power_w: Vec<f64>,
}

impl ImpulseResponse {
    /// The total received power $\int h(t)\, dt$ (W), the DC gain for an emitter of 1 W.
    pub fn total_w(&self) -> f64 {
        self.power_w.iter().sum()
    }

    /// The start of the first bin that holds power (s); `None` for no power.
    pub fn first_arrival_s(&self) -> Option<f64> {
        self.power_w.iter().position(|&p| p > 0.0).map(|i| i as f64 * self.bin_s)
    }

    /// The sum of two responses of the same bins.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for different bin widths.
    pub fn add(&self, other: &ImpulseResponse) -> Result<ImpulseResponse> {
        ensure(self.bin_s == other.bin_s, || "impulse responses of different bin widths".into())?;
        let n = self.power_w.len().max(other.power_w.len());
        let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
        Ok(ImpulseResponse {
            bin_s: self.bin_s,
            power_w: (0..n).map(|i| at(&self.power_w, i) + at(&other.power_w, i)).collect(),
        })
    }

    /// The frequency response $H(f) = \sum_{i} P_{i} e^{-j 2\pi f (i + 1/2) \Delta t}$, the
    /// power of every bin at its centre (Barry et al. 1993, eq. (15)).
    pub fn frequency_response(&self, f_hz: f64) -> C64 {
        self.power_w.iter().enumerate().map(|(i, p)| cis(-2.0 * PI * f_hz * (i as f64 + 0.5) * self.bin_s) * *p).sum()
    }

    /// The −3 dB bandwidth (Hz): the first frequency at which $|H(f)| / H(0)$ falls to
    /// $10^{-3/20}$, searched by steps of `step_hz` up to `max_hz` and refined by
    /// bisection; `None` if it does not fall below within `max_hz`.
    pub fn bandwidth_3db_hz(&self, step_hz: f64, max_hz: f64) -> Option<f64> {
        let h0 = self.total_w();
        if !(h0 > 0.0 && step_hz > 0.0) {
            return None;
        }
        let level = 10f64.powf(-3.0 / 20.0);
        let below = |f: f64| self.frequency_response(f).norm() / h0 < level;
        let mut f = step_hz;
        while f <= max_hz {
            if below(f) {
                return Some(com_math::bisect(|x| !below(x), f - step_hz, f, 60));
            }
            f += step_hz;
        }
        None
    }

    /// The power of on–off keying with rectangular pulses of duration `bit_s` received
    /// inside the bit and after it, `(signal, intersymbol interference)`, the bit
    /// starting at `origin_s`: a bin centred $\tau$ after the origin gives the fraction
    /// $(T - \tau)/T$ of its power to the bit and the rest to the next ones (Komine &
    /// Nakagawa 2004, eqs. (17) and (19), with the time counted from the first arrival).
    pub fn bit_split(&self, bit_s: f64, origin_s: f64) -> (f64, f64) {
        let (mut signal, mut isi) = (0.0, 0.0);
        for (i, p) in self.power_w.iter().enumerate() {
            let tau = ((i as f64 + 0.5) * self.bin_s - origin_s).max(0.0);
            let inside = ((bit_s - tau) / bit_s).clamp(0.0, 1.0);
            signal += p * inside;
            isi += p * (1.0 - inside);
        }
        (signal, isi)
    }
}

/// A sparse histogram: the bins `start..start + values.len()`.
#[derive(Clone, Debug, Default)]
struct Histogram {
    start: usize,
    values: Vec<f64>,
}

impl Histogram {
    /// Add `p` at the bin `bin`.
    fn deposit(&mut self, bin: usize, p: f64) {
        if self.values.is_empty() {
            self.start = bin;
            self.values.push(p);
            return;
        }
        if bin < self.start {
            let grow = self.start - bin;
            let mut v = vec![0.0; grow];
            v.append(&mut self.values);
            self.values = v;
            self.start = bin;
        }
        let i = bin - self.start;
        if i >= self.values.len() {
            self.values.resize(i + 1, 0.0);
        }
        self.values[i] += p;
    }

    /// Add `scale` times `other` delayed by `shift` bins.
    fn add_shifted(&mut self, other: &Histogram, shift: usize, scale: f64) {
        if other.values.is_empty() || scale == 0.0 {
            return;
        }
        let lo = other.start + shift;
        let hi = lo + other.values.len();
        if self.values.is_empty() {
            self.start = lo;
            self.values = vec![0.0; hi - lo];
        } else {
            if lo < self.start {
                let mut v = vec![0.0; self.start - lo];
                v.append(&mut self.values);
                self.values = v;
                self.start = lo;
            }
            if hi > self.start + self.values.len() {
                self.values.resize(hi - self.start, 0.0);
            }
        }
        let off = lo - self.start;
        for (d, s) in self.values[off..off + other.values.len()].iter_mut().zip(&other.values) {
            *d += s * scale;
        }
    }

    fn into_response(self, bin_s: f64) -> ImpulseResponse {
        let mut power_w = vec![0.0; self.start];
        power_w.extend(self.values);
        ImpulseResponse { bin_s, power_w }
    }
}

/// A reflecting element: centre, inward normal, area and reflectivity.
#[derive(Clone, Copy, Debug)]
struct Element {
    centre: Vec3,
    normal: Vec3,
    area: f64,
    rho: f64,
}

impl Room {
    /// A room of size `(L, W, H)` (m) whose six surfaces have the reflectivity `rho`.
    ///
    /// # Errors
    /// As [`Room::check`].
    pub fn uniform(size_m: [f64; 3], rho: f64) -> Result<Room> {
        let r = Reflectance { north: rho, south: rho, east: rho, west: rho, ceiling: rho, floor: rho };
        let room = Room { size_m, reflectance: r };
        room.check()?;
        Ok(room)
    }

    /// Check the room.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a non-positive size or a reflectivity outside $[0, 1]$.
    pub fn check(&self) -> Result<()> {
        ensure(self.size_m.iter().all(|s| *s > 0.0 && s.is_finite()), || {
            format!("the size of a room must be positive, got {:?}", self.size_m)
        })?;
        let r = self.reflectance;
        ensure([r.north, r.south, r.east, r.west, r.ceiling, r.floor].iter().all(|p| (0.0..=1.0).contains(p)), || {
            "the reflectivities must lie in [0, 1]".into()
        })
    }

    /// The reflecting elements of the partition `(N_x, N_y, N_z)`, those of zero
    /// reflectivity left out.
    fn elements(&self, [nx, ny, nz]: [usize; 3]) -> Vec<Element> {
        let [l, w, h] = self.size_m;
        let (dx, dy, dz) = (l / nx as f64, w / ny as f64, h / nz as f64);
        let r = self.reflectance;
        let mut out = Vec::with_capacity(2 * (nx * ny + nx * nz + ny * nz));
        let mut wall = |rho: f64, normal: Vec3, n1: usize, n2: usize, at: &dyn Fn(f64, f64) -> Vec3, area: f64| {
            if rho <= 0.0 {
                return;
            }
            for i in 0..n1 {
                for j in 0..n2 {
                    out.push(Element { centre: at(i as f64 + 0.5, j as f64 + 0.5), normal, area, rho });
                }
            }
        };
        wall(r.south, [1.0, 0.0, 0.0], ny, nz, &|a, b| [0.0, a * dy, b * dz], dy * dz);
        wall(r.north, [-1.0, 0.0, 0.0], ny, nz, &|a, b| [l, a * dy, b * dz], dy * dz);
        wall(r.east, [0.0, 1.0, 0.0], nx, nz, &|a, b| [a * dx, 0.0, b * dz], dx * dz);
        wall(r.west, [0.0, -1.0, 0.0], nx, nz, &|a, b| [a * dx, w, b * dz], dx * dz);
        wall(r.floor, [0.0, 0.0, 1.0], nx, ny, &|a, b| [a * dx, b * dy, 0.0], dx * dy);
        wall(r.ceiling, [0.0, 0.0, -1.0], nx, ny, &|a, b| [a * dx, b * dy, h], dx * dy);
        out
    }

    /// Carry the light of `emitters` to the reflecting elements of every order of
    /// reflection of `resolution`, once for any number of detectors.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for an invalid room, emitter or resolution.
    pub fn illuminate(&self, emitters: &[Emitter], resolution: &Resolution) -> Result<Illumination> {
        self.check()?;
        ensure(resolution.bin_s > 0.0, || "the time bins must have a positive width".into())?;
        ensure(resolution.partitions.iter().all(|p| p.iter().all(|n| *n >= 1)), || {
            "every partition needs at least one element per dimension".into()
        })?;
        let emitters = emitters.iter().map(Emitter::check).collect::<Result<Vec<_>>>()?;
        let bin = resolution.bin_s;
        let delay_bins = |d: f64| (d / SPEED_OF_LIGHT / bin).floor() as usize;
        let mut orders = Vec::with_capacity(resolution.partitions.len());
        for (k, &partition) in resolution.partitions.iter().enumerate() {
            let elements = self.elements(partition);
            // the light of the emitters on every element, then k bounces between elements
            let mut state: Vec<Histogram> = elements
                .iter()
                .map(|e| {
                    let mut hist = Histogram::default();
                    for s in &emitters {
                        let v = sub(e.centre, s.position);
                        let d = norm(v);
                        let cos_phi = dot(s.direction, v) / d;
                        let cos_theta = -dot(e.normal, v) / d;
                        if cos_theta > 0.0 {
                            let p = s.intensity(cos_phi) * e.area * cos_theta / (d * d);
                            if p > 0.0 {
                                hist.deposit(delay_bins(d), p);
                            }
                        }
                    }
                    hist
                })
                .collect();
            for _ in 0..k {
                state = elements
                    .iter()
                    .map(|to| {
                        let mut hist = Histogram::default();
                        for (from, h) in elements.iter().zip(&state) {
                            if h.values.is_empty() {
                                continue;
                            }
                            let v = sub(to.centre, from.centre);
                            let d = norm(v);
                            let (cos_phi, cos_theta) = (dot(from.normal, v) / d, -dot(to.normal, v) / d);
                            if cos_phi > 0.0 && cos_theta > 0.0 {
                                let g = from.rho * cos_phi / PI * to.area * cos_theta / (d * d);
                                hist.add_shifted(h, delay_bins(d), g);
                            }
                        }
                        hist
                    })
                    .collect();
            }
            orders.push((elements, state));
        }
        Ok(Illumination { emitters, bin_s: bin, orders })
    }
}

/// The light of a set of emitters carried to the elements of a room, for every order
/// of reflection ([`Room::illuminate`]).
#[derive(Clone, Debug)]
pub struct Illumination {
    emitters: Vec<Emitter>,
    bin_s: f64,
    /// For every order $k$: the elements and the light they receive after $k - 1$
    /// reflections.
    orders: Vec<(Vec<Element>, Vec<Histogram>)>,
}

impl Illumination {
    /// The impulse responses at `detector`: the line of sight, then the light reflected
    /// once, twice… (one [`ImpulseResponse`] per order, of the same bins).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for an invalid detector.
    pub fn impulse_responses(&self, detector: &Detector) -> Result<Vec<ImpulseResponse>> {
        let det = detector.check()?;
        let bin = self.bin_s;
        let delay_bins = |d: f64| (d / SPEED_OF_LIGHT / bin).floor() as usize;
        let mut out = Vec::with_capacity(self.orders.len() + 1);
        let mut los = Histogram::default();
        for s in &self.emitters {
            let v = sub(det.position, s.position);
            let d = norm(v);
            let p = s.intensity(dot(s.direction, v) / d) * det.effective_area(-dot(det.direction, v) / d) / (d * d);
            if p > 0.0 {
                los.deposit(delay_bins(d), p);
            }
        }
        out.push(los);
        for (elements, state) in &self.orders {
            let mut hist = Histogram::default();
            for (e, h) in elements.iter().zip(state) {
                if h.values.is_empty() {
                    continue;
                }
                let v = sub(det.position, e.centre);
                let d = norm(v);
                let cos_phi = dot(e.normal, v) / d;
                if cos_phi > 0.0 {
                    let g = e.rho * cos_phi / PI * det.effective_area(-dot(det.direction, v) / d) / (d * d);
                    hist.add_shifted(h, delay_bins(d), g);
                }
            }
            out.push(hist);
        }
        // one length for all, so that they add bin by bin
        let len = out.iter().map(|h| h.start + h.values.len()).max().unwrap_or(0);
        Ok(out
            .into_iter()
            .map(|h| {
                let mut r = h.into_response(bin);
                r.power_w.resize(len, 0.0);
                r
            })
            .collect())
    }
}

/// The horizontal illuminance (lx) at `point` on a surface of normal `normal` from LED
/// chips at the positions and along the directions of `emitters`, each of centre
/// luminous intensity `intensity_cd` and Lambertian mode `mode`:
/// $E = \sum I(0) \cos^{n}\phi \cos\psi / D^{2}$ (Komine & Nakagawa 2004, eq. (5)).
pub fn illuminance_lx(emitters: &[Emitter], intensity_cd: f64, point: [f64; 3], normal: [f64; 3]) -> f64 {
    emitters
        .iter()
        .map(|s| {
            let v = sub(point, s.position);
            let d = norm(v);
            let (cos_phi, cos_psi) = (dot(s.direction, v) / d, -dot(normal, v) / d);
            if cos_phi > 0.0 && cos_psi > 0.0 { intensity_cd * cos_phi.powf(s.mode) * cos_psi / (d * d) } else { 0.0 }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_of_sight_is_the_closed_form_and_energy_is_conserved() {
        // a source straight above a detector: P (n + 1) A / (2π d²)
        let down = direction(-FRAC_PI_2, 0.0);
        let up = direction(FRAC_PI_2, 0.0);
        let s = Emitter::new([1.0, 1.0, 3.0], down, 1.0, 2.0);
        let det = Detector::new([1.0, 1.0, 1.0], up, 1e-4, 1.0);
        let room = Room {
            reflectance: Reflectance { north: 0.0, south: 0.0, east: 0.0, west: 0.0, ceiling: 0.0, floor: 0.0 },
            ..Room::uniform([2.0, 2.0, 3.0], 0.0).unwrap()
        };
        let h = room.illuminate(&[s], &Resolution::new(1e-10, vec![])).unwrap().impulse_responses(&det).unwrap();
        assert!((h[0].total_w() - 2.0 * 2.0 * 1e-4 / (2.0 * PI * 4.0)).abs() < 1e-15);
        assert!((h[0].first_arrival_s().unwrap() - 2.0 / SPEED_OF_LIGHT).abs() < 1e-10);
        // a white closed room returns all the light to its walls: the power on the
        // elements after one reflection is ρ times the power emitted (up to the
        // discretisation)
        let room = Room::uniform([2.0, 2.0, 2.0], 0.5).unwrap();
        let s = Emitter::new([1.0, 1.0, 1.0], down, 1.0, 1.0);
        let lit = room.illuminate(&[s], &Resolution::new(1e-10, vec![[20, 20, 20], [20, 20, 20]])).unwrap();
        let first: f64 = lit.orders[0].1.iter().flat_map(|h| h.values.iter()).sum();
        let second: f64 = lit.orders[1].1.iter().flat_map(|h| h.values.iter()).sum();
        assert!((first - 1.0).abs() < 0.02, "{first}");
        assert!((second / (0.5 * first) - 1.0).abs() < 0.05, "{second}");
    }

    #[test]
    fn invalid_settings_are_rejected() {
        assert!(lambertian_mode(0.0).is_err() && lambertian_mode(FRAC_PI_2).is_err());
        assert!((lambertian_mode(60f64.to_radians()).unwrap() - 1.0).abs() < 1e-12);
        assert!(Room::uniform([1.0, 0.0, 1.0], 0.5).is_err() && Room::uniform([1.0, 1.0, 1.0], 1.5).is_err());
        let room = Room::uniform([1.0, 1.0, 1.0], 0.5).unwrap();
        let bad = Emitter::new([0.5, 0.5, 1.0], [0.0, 0.0, 0.0], 1.0, 1.0);
        assert!(room.illuminate(&[bad], &Resolution::new(1e-10, vec![])).is_err());
        let lit = room.illuminate(&[], &Resolution::new(1e-10, vec![[2, 2, 2]])).unwrap();
        let det = Detector::new([0.5, 0.5, 0.0], [0.0, 0.0, 1.0], 0.0, 1.0);
        assert!(lit.impulse_responses(&det).is_err());
    }

    #[test]
    fn the_bit_split_and_the_bandwidth_of_simple_responses() {
        // two equal pulses half a bit apart: 3/4 of the power in the bit
        let mut p = vec![0.0; 100];
        p[0] = 1.0;
        p[50] = 1.0;
        let h = ImpulseResponse { bin_s: 1e-10, power_w: p };
        let (s, i) = h.bit_split(1e-8, 0.0);
        assert!((s - (0.995 + 0.495)).abs() < 1e-12 && (s + i - 2.0).abs() < 1e-12);
        // a single bin is flat beyond any reasonable band
        let delta = ImpulseResponse { bin_s: 1e-10, power_w: vec![1.0] };
        assert!(delta.bandwidth_3db_hz(1e6, 1e9).is_none());
        // two equal pulses τ apart: |H| = |cos(πfτ)|, −3 dB at f = acos(10^{-3/20})/(πτ)
        let b = h.bandwidth_3db_hz(1e6, 1e9).unwrap();
        let want = 10f64.powf(-3.0 / 20.0).acos() / (PI * 5e-9);
        assert!((b / want - 1.0).abs() < 1e-6, "{b} vs {want}");
    }
}
