//! A near-optimum receiver of CE-OFDM by generalised approximate message passing
//! (GAMP), after M. J. López Morales, R. Dinis and A. García Armada, "Near-optimal
//! detection of CE-OFDM signals with high power efficiency via GAMP-based receivers,"
//! *Proc. IEEE Globecom Workshops*, 2022, Algorithm 1.
//!
//! # Model
//!
//! The real and imaginary parts of the $K$ data symbols of a block form
//! $\mathbf{u} \in \mathbb{R}^{2K}$, and the OFDM signal is $\mathbf{x} = \mathbf{A}
//! \mathbf{u}$ with $A_{n,k} = \frac{2}{\sqrt{2K}} \cos\frac{2\pi k n}{N}$ on the real
//! parts and $-\frac{2}{\sqrt{2K}} \sin\frac{2\pi k n}{N}$ on the imaginary ones (the
//! modulator of [`super::CeOfdm`]); the receiver sees $r_{n} = e^{j2\pi h x_{n}} + w_{n}$,
//! $w_{n} \sim \mathcal{CN}(0, \sigma^{2})$. Every row of $|\mathbf{A}|^{2}$ sums to 2
//! and every column to $N/K$, so scalar variances are exact on average.
//!
//! # Algorithm
//!
//! Damped sum-product GAMP (S. Rangan, P. Schniter, A. K. Fletcher and S. Sarkar, "On the
//! convergence of approximate message passing with arbitrary matrices," *IEEE Trans.
//! Inf. Theory* 65(9), 2019): the output step computes the mean and variance of
//! $x_{n}$ under the Gaussian message $\mathcal{N}(\hat{p}_{n}, \tau_{p})$ and the
//! likelihood $e^{-|r_{n} - e^{j2\pi h x}|^{2}/\sigma^{2}} \propto
//! e^{c_{n}\cos(2\pi h x - \arg r_{n})}$, $c_{n} = 2|r_{n}|/\sigma^{2}$, by the midpoint
//! rule (eq. (12) of the paper) or, when the likelihood is sharp ($c_{n} \ge 25$), by its
//! Laplace approximation, a mixture of Gaussians over the branches $\arg r_{n} + 2\pi m$;
//! the input step is the posterior of the levels of the constellation. The linear steps
//! are FFTs.
//!
//! # Differences with the paper
//!
//! The paper estimates the $N M$ bins of a Hermitian-symmetric spectrum as independent
//! complex symbols; here the unknowns are the $2K$ real coordinates, so the symmetry is
//! exact and the priors are the per-axis levels of a square QAM (or PAM, BPSK). The
//! damping is a constant (the paper ties it to $E_{b}/N_{0}$).
//!
//! # Size of the system
//!
//! GAMP assumes a large, random system. At $2\pi h = 1.4$ and $E_{b}/N_{0} = 4$ dB (QPSK,
//! $J = 4$), its BER is $1.6 \cdot 10^{-2}$, $4.5 \cdot 10^{-3}$, $1.0 \cdot 10^{-3}$ and
//! $8.1 \cdot 10^{-4}$ for $K = 63$, 127, 255 and 511 data subcarriers, against
//! $8.5 \cdot 10^{-4}$ for the optimum asymptotic BER; the paper uses $N = 1024$
//! ($K = 511$). Structured data, whose blocks have high peaks, can trap it.

use super::{CeOfdm, TAU};
use com_core::{Result, ensure, ensure_shape};
use com_mapping::constellation::Constellation;
use com_math::C64;

/// The GAMP receiver (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Gamp {
    /// Iterations per block.
    pub iterations: usize,
    /// Damping factor $\beta \in (0, 1]$ (1: no damping).
    pub damping: f64,
    /// Points of the midpoint rule of the output step.
    pub points: usize,
}

impl Default for Gamp {
    /// 50 iterations (as the paper), a damping of 0.5 and 64 points.
    fn default() -> Gamp {
        Gamp { iterations: 50, damping: 0.5, points: 64 }
    }
}

/// The levels of the real and imaginary parts of a constellation that is their
/// Cartesian product.
fn axis_levels(c: &Constellation) -> Result<(Vec<f64>, Vec<f64>)> {
    let unique = |values: Vec<f64>| {
        let mut v = values;
        v.sort_by(f64::total_cmp);
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        v
    };
    let re = unique(c.points().iter().map(|p| p.re).collect());
    let im = unique(c.points().iter().map(|p| p.im).collect());
    let near = |levels: &[f64], v: f64| levels.iter().any(|l| (l - v).abs() < 1e-9);
    let product = re.len() * im.len() == c.order() && c.points().iter().all(|p| near(&re, p.re) && near(&im, p.im));
    ensure(product, || {
        "GAMP needs a constellation whose real and imaginary parts are independent (square QAM, PAM, BPSK)".into()
    })?;
    Ok((re, im))
}

/// The mean and variance of the levels under the Gaussian message `N(r, tr)`.
fn input_moments(levels: &[f64], r: f64, tr: f64) -> (f64, f64) {
    let logw: Vec<f64> = levels.iter().map(|l| -(r - l).powi(2) / (2.0 * tr)).collect();
    let top = logw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for (l, w) in levels.iter().zip(&logw) {
        let w = (w - top).exp();
        s0 += w;
        s1 += w * l;
        s2 += w * l * l;
    }
    let mean = s1 / s0;
    (mean, (s2 / s0 - mean * mean).max(0.0))
}

/// The mean and variance of `x` under the message `N(p, tp)` and the likelihood
/// `exp(c cos(κx - ψ))` (see the module).
fn output_moments(p: f64, tp: f64, c: f64, psi: f64, kappa: f64, points: usize) -> (f64, f64) {
    let sd = tp.sqrt();
    if c <= 0.0 {
        return (p, tp);
    }
    // width of one branch of the likelihood
    let wl = 1.0 / (kappa * c.sqrt());
    if c >= 25.0 {
        // Laplace: a Gaussian of variance v around every branch (ψ + 2πm)/κ
        let v = wl * wl;
        let spread = 8.0 * (tp + v).sqrt();
        let m_lo = ((kappa * (p - spread) - psi) / TAU).floor() as i64;
        let m_hi = ((kappa * (p + spread) - psi) / TAU).ceil() as i64;
        let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
        let mut terms = Vec::with_capacity((m_hi - m_lo + 1) as usize);
        for m in m_lo..=m_hi {
            let mu = (psi + TAU * m as f64) / kappa;
            terms.push((mu, -(mu - p).powi(2) / (2.0 * (tp + v))));
        }
        let top = terms.iter().map(|t| t.1).fold(f64::NEG_INFINITY, f64::max);
        let var = tp * v / (tp + v);
        for (mu, lw) in terms {
            let w = (lw - top).exp();
            let mean = (mu * tp + p * v) / (tp + v);
            s0 += w;
            s1 += w * mean;
            s2 += w * (var + mean * mean);
        }
        let mean = s1 / s0;
        return (mean, (s2 / s0 - mean * mean).max(0.0));
    }
    // the midpoint rule over ±7 standard deviations of the message, fine enough for the
    // likelihood
    let span = 14.0 * sd;
    let step = (span / points as f64).min(wl / 4.0);
    let n = ((span / step).ceil() as usize).clamp(points, 4096);
    let h = span / n as f64;
    let z0 = p - 7.0 * sd;
    let logw = |z: f64| -(z - p).powi(2) / (2.0 * tp) + c * (kappa * z - psi).cos();
    let top = (0..n).map(|i| logw(z0 + (i as f64 + 0.5) * h)).fold(f64::NEG_INFINITY, f64::max);
    let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let z = z0 + (i as f64 + 0.5) * h;
        let w = (logw(z) - top).exp();
        s0 += w;
        s1 += w * z;
        s2 += w * z * z;
    }
    let mean = s1 / s0;
    (mean, (s2 / s0 - mean * mean).max(0.0))
}

impl Gamp {
    /// The posterior means of the data symbols of every block (prefix removed) of the
    /// received samples `r`, for symbols of the constellation `c` (as transmitted) and the
    /// noise variance per sample `noise_variance` ([`CeOfdm::noise_variance`]).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters or a constellation whose real
    /// and imaginary parts are not independent, [`com_core::Error::Shape`] for a number
    /// of samples that is not a multiple of `N + N_cp`.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::ce_ofdm::{CeOfdm, gamp::Gamp};
    /// use com_mapping::constellation::Constellation;
    /// use com_core::rng::Rng;
    /// // 2πh = 1.4: the phase leaves (-π, π] and the arctangent receiver fails
    /// let ce = CeOfdm { n_data: 63, oversampling: 4, h: 1.4 / std::f64::consts::TAU, n_cp: 0 };
    /// let c = Constellation::qam(4).unwrap().normalized();
    /// let (_, d) = com_mapping::source::symbols(&c, 63, None, &mut Rng::new(1)).unwrap();
    /// let r = ce.modulate(&d).unwrap();
    /// let e = Gamp::default().detect(&ce, &r, &c, 1e-4).unwrap();
    /// assert_eq!(c.decide(&e), c.decide(&d));
    /// ```
    pub fn detect(&self, ce: &CeOfdm, r: &[C64], c: &Constellation, noise_variance: f64) -> Result<Vec<C64>> {
        ce.check()?;
        ensure(self.iterations >= 1 && self.points >= 2, || "GAMP needs one iteration and two points".into())?;
        ensure(self.damping > 0.0 && self.damping <= 1.0, || {
            format!("the damping must lie in (0, 1], got {}", self.damping)
        })?;
        ensure(noise_variance > 0.0, || format!("the noise variance must be positive, got {noise_variance}"))?;
        let (k, l) = (ce.n_data, ce.block_len());
        ensure_shape(r.len() % l == 0, || format!("{} samples is not a multiple of N + N_cp = {l}", r.len()))?;
        let (re, im) = axis_levels(c)?;
        let mut out = Vec::with_capacity(r.len() / l * k);
        for block in r.chunks(l) {
            out.extend(self.block(ce, &block[ce.n_cp..], &re, &im, noise_variance));
        }
        Ok(out)
    }

    fn block(&self, ce: &CeOfdm, r: &[C64], re: &[f64], im: &[f64], noise_variance: f64) -> Vec<C64> {
        let (k, n) = (ce.n_data, ce.n_fft());
        let kappa = TAU * ce.h;
        let a = 2.0 / (2.0 * k as f64).sqrt();
        let beta = self.damping;
        // A u is the modulator's OFDM signal; Aᵀ y the data bins of the DFT, scaled
        let forward = |u: &[C64]| ce.synthesize(u);
        let adjoint = |y: &[f64]| -> Vec<C64> { ce.data_bins(y).into_iter().map(|v| v * a).collect() };
        let prior = |levels: &[f64]| input_moments(levels, 0.0, f64::INFINITY);
        let (m_re, v_re) = prior(re);
        let (m_im, v_im) = prior(im);
        let mut u_hat = vec![C64::new(m_re, m_im); k];
        let mut u_bar = u_hat.clone();
        let mut tu = (v_re + v_im) / 2.0;
        let mut s_hat = vec![0.0; n];
        let mut ts = 0.0;
        let likelihood: Vec<(f64, f64)> = r.iter().map(|v| (2.0 * v.norm() / noise_variance, v.arg())).collect();
        for t in 0..self.iterations {
            // output linear and nonlinear steps
            let tp = 2.0 * tu;
            let au = forward(&u_hat);
            let mut ts_new = 0.0;
            for (i, (&(cn, psi), x)) in likelihood.iter().zip(&au).enumerate() {
                let p = x - tp * s_hat[i];
                let (z, tz) = output_moments(p, tp, cn, psi, kappa, self.points);
                let s = (z - p) / tp;
                s_hat[i] = if t == 0 { s } else { (1.0 - beta) * s_hat[i] + beta * s };
                // a posterior variance above the prior one is a rounding error
                ts_new += (1.0 - tz / tp).max(0.0) / tp;
            }
            ts_new /= n as f64;
            ts = if t == 0 { ts_new } else { (1.0 - beta) * ts + beta * ts_new };
            // input linear and nonlinear steps
            for (b, h) in u_bar.iter_mut().zip(&u_hat) {
                *b = if t == 0 { *h } else { (1.0 - beta) * *b + beta * h };
            }
            let tr = k as f64 / (n as f64 * ts.max(1e-300));
            let back = adjoint(&s_hat);
            let mut next = Vec::with_capacity(k);
            let mut tu_new = 0.0;
            for (b, g) in u_bar.iter().zip(&back) {
                let rv = b + g * tr;
                let (mr, vr) = input_moments(re, rv.re, tr);
                let (mi, vi) = input_moments(im, rv.im, tr);
                next.push(C64::new(mr, mi));
                tu_new += vr + vi;
            }
            if next.iter().any(|v| !v.is_finite()) {
                break;
            }
            u_hat = next;
            tu = tu_new / (2 * k) as f64;
            // the decisions are firm: the variances would vanish and the steps divide by them
            if tu < 1e-9 * (v_re + v_im) {
                break;
            }
        }
        u_hat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ce_ofdm::Unwrap;
    use com_core::rng::Rng;

    #[test]
    fn the_output_step_matches_the_quadrature_and_the_input_step_the_levels() {
        // Laplace (sharp likelihood) against a fine midpoint rule
        for psi in [0.5, 2.0, 3.0, -3.1] {
            let (m1, v1) = output_moments(0.3, 0.8, 30.0, psi, 1.4, 64);
            let (sd, n) = (0.8f64.sqrt(), 400_000);
            let h = 16.0 * sd / n as f64;
            let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
            for i in 0..n {
                let z = 0.3 - 8.0 * sd + (i as f64 + 0.5) * h;
                let w = (-(z - 0.3).powi(2) / 1.6 + 30.0 * ((1.4 * z - psi).cos() - 1.0)).exp();
                (s0, s1, s2) = (s0 + w, s1 + w * z, s2 + w * z * z);
            }
            let (m2, v2) = (s1 / s0, s2 / s0 - (s1 / s0).powi(2));
            // the Laplace approximation errs by O(1/c)
            assert!((m1 - m2).abs() < 0.05 && (v1 - v2).abs() < 0.05 * v2.max(0.01), "{psi}: {m1} {v1} vs {m2} {v2}");
        }
        // a flat likelihood leaves the message
        assert_eq!(output_moments(0.2, 0.5, 0.0, 1.0, 1.0, 64), (0.2, 0.5));
        let (m, v) = input_moments(&[-1.0, 1.0], 0.0, f64::INFINITY);
        assert!(m.abs() < 1e-15 && (v - 1.0).abs() < 1e-15);
        let (m, v) = input_moments(&[-1.0, 1.0], 0.9, 0.01);
        assert!((m - 1.0).abs() < 1e-6 && v < 1e-6);
    }

    #[test]
    fn gamp_detects_where_the_phase_demodulator_fails() {
        // 2πh = 1.4 (López Morales et al. 2022, Fig. 3): the phase wraps, the arctangent
        // receiver floors at a SER of 0.1 and more, GAMP decodes
        let c = Constellation::qam(4).unwrap().normalized();
        let ce = CeOfdm { n_data: 63, oversampling: 4, h: 1.4 / TAU, n_cp: 0 };
        let mut rng = Rng::new(3);
        let (_, d) = com_mapping::source::symbols(&c, 63 * 8, None, &mut rng).unwrap();
        for ebn0 in [8.0, 30.0] {
            let sigma2 = ce.noise_variance(ebn0 + 10.0 * 2f64.log10());
            let r: Vec<C64> =
                ce.modulate(&d).unwrap().into_iter().map(|v| v + rng.complex_gaussian() * sigma2.sqrt()).collect();
            let ser = |e: &[C64]| {
                c.decide(e).iter().zip(c.decide(&d)).filter(|(a, b)| *a != b).count() as f64 / d.len() as f64
            };
            let gamp = Gamp::default().detect(&ce, &r, &c, sigma2).unwrap();
            let atan = ce.demodulate(&r, Unwrap::None).unwrap();
            assert!(ser(&gamp) == 0.0 && ser(&atan) > 0.1, "{ebn0} dB: GAMP {} arctangent {}", ser(&gamp), ser(&atan));
        }
        // a constellation that is not a product of its axes is refused
        let psk8 = Constellation::psk(8).unwrap();
        assert!(Gamp::default().detect(&ce, &vec![C64::new(1.0, 0.0); ce.block_len()], &psk8, 0.1).is_err());
    }
}
