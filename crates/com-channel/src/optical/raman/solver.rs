//! The boundary-value problem of the coupled power equations: forward and backward
//! Runge-Kutta sweeps with relaxation, and the spontaneous Raman noise.

use super::{Direction, Raman, RamanProfile};
use crate::optical::fiber::FiberParams;
use com_core::{Error, Result, ensure};
use com_math::{BOLTZMANN, PLANCK, db2np};

/// The waves of the equations.
struct Wave {
    f: f64,
    alpha: f64,
    p_in: f64,
    counter: bool,
    signal: bool,
}

/// The smallest power of a wave: the logarithms stay finite for empty slices.
const POWER_FLOOR_W: f64 = 1e-30;

impl Raman {
    fn check(&self) -> Result<()> {
        ensure(self.aeff_um2 > 0.0, || format!("raman: aeff_um2 must be positive, got {}", self.aeff_um2))?;
        ensure(self.temperature_k > 0.0, || {
            format!("raman: temperature_k must be positive, got {}", self.temperature_k)
        })?;
        ensure(self.step_km > 0.0, || format!("raman: step_km must be positive, got {}", self.step_km))?;
        ensure(self.tolerance > 0.0 && self.max_iterations >= 1, || {
            "raman: tolerance must be positive and max_iterations at least 1".into()
        })?;
        ensure(self.slice_hz > 0.0, || format!("raman: slice_hz must be positive, got {}", self.slice_hz))?;
        for p in &self.pumps {
            ensure(p.frequency_hz > 0.0 && p.power_w >= 0.0, || {
                format!("raman: a pump needs a positive frequency and a non-negative power, got {p:?}")
            })?;
        }
        Ok(())
    }

    /// The power profiles of one span of `fiber` for the co-propagating `signals`
    /// (absolute frequency in Hz, launch power in W) and the pumps.
    ///
    /// # Errors
    /// [`Error::Config`] for invalid parameters or no signal, [`Error::Numerical`] if
    /// the sweeps do not converge.
    pub fn solve(&self, fiber: &FiberParams, signals: &[(f64, f64)]) -> Result<RamanProfile> {
        self.check()?;
        ensure(!signals.is_empty(), || "raman: no signal".into())?;
        ensure(fiber.span_km > 0.0, || "raman: the span must have a positive length".into())?;
        ensure(signals.iter().all(|(f, p)| *f > 0.0 && *p >= 0.0), || {
            "raman: signals need positive frequencies and non-negative powers".into()
        })?;
        let alpha = fiber.alpha_np_km();
        let mut waves: Vec<Wave> = signals
            .iter()
            .map(|(f, p)| Wave { f: *f, alpha, p_in: p.max(POWER_FLOOR_W), counter: false, signal: true })
            .collect();
        waves.extend(self.pumps.iter().map(|p| Wave {
            f: p.frequency_hz,
            alpha: p.alpha_db_km.map_or(alpha, db2np),
            p_in: p.power_w.max(POWER_FLOOR_W),
            counter: p.direction == Direction::Counter,
            signal: false,
        }));
        let n = waves.len();
        let aeff = self.aeff_um2 * 1e-12;
        let mut c = vec![vec![0.0; n]; n];
        for k in 0..n {
            for j in 0..n {
                if !self.isrs && waves[k].signal && waves[j].signal {
                    continue;
                }
                let (fk, fj) = (waves[k].f, waves[j].f);
                c[k][j] = if fj > fk {
                    self.gain.efficiency(fk, fj, aeff)
                } else if fj < fk {
                    -(fk / fj) * self.gain.efficiency(fj, fk, aeff)
                } else {
                    0.0
                };
            }
        }
        let length = fiber.span_km;
        let m = ((length / self.step_km).ceil() as usize).max(1);
        let h = length / m as f64;
        let z: Vec<f64> = (0..=m).map(|i| i as f64 * h).collect();
        // the initial guess: the loss alone
        let mut y: Vec<Vec<f64>> = waves
            .iter()
            .map(|w| z.iter().map(|zi| w.p_in.ln() - w.alpha * if w.counter { length - zi } else { *zi }).collect())
            .collect();
        let co: Vec<usize> = (0..n).filter(|k| !waves[*k].counter).collect();
        let counter: Vec<usize> = (0..n).filter(|k| waves[*k].counter).collect();
        let sys = System { waves: &waves, c: &c, h };
        let mut iterations = 0;
        if counter.is_empty() {
            sys.sweep(&mut y, &co, true, 1.0);
            iterations = 1;
        } else {
            let mut omega: f64 = 1.0;
            let mut last = f64::INFINITY;
            loop {
                iterations += 1;
                let r_co = sys.sweep(&mut y, &co, true, 1.0);
                let r_cnt = sys.sweep(&mut y, &counter, false, omega);
                let residue = r_co.max(r_cnt / omega);
                if !residue.is_finite() {
                    return Err(Error::Numerical("raman: the power profiles diverge".into()));
                }
                if residue < self.tolerance {
                    break;
                }
                if iterations >= self.max_iterations {
                    return Err(Error::Numerical(format!(
                        "raman: no convergence after {iterations} sweeps (change {residue:.2e})"
                    )));
                }
                if residue > last {
                    omega = (omega / 2.0).max(1.0 / 64.0);
                }
                last = residue;
            }
        }
        let ase_psd = (0..signals.len()).map(|k| self.spontaneous(&waves, &c, &y, k, h)).collect();
        Ok(RamanProfile {
            z_km: z,
            freqs_hz: waves.iter().map(|w| w.f).collect(),
            directions: waves.iter().map(|w| if w.counter { Direction::Counter } else { Direction::Co }).collect(),
            alpha_np_km: waves.iter().map(|w| w.alpha).collect(),
            n_signals: signals.len(),
            log_power: y,
            ase_psd,
            iterations,
        })
    }

    /// The spontaneous noise of signal `k` at the end of the span (both polarisations),
    /// by the trapezoidal rule on the grid.
    fn spontaneous(&self, waves: &[Wave], c: &[Vec<f64>], y: &[Vec<f64>], k: usize, h: f64) -> f64 {
        let fk = waves[k].f;
        let sources: Vec<(usize, f64)> = (0..waves.len())
            .filter(|j| waves[*j].f > fk && !waves[*j].signal && c[k][*j] > 0.0)
            .map(|j| {
                let x = PLANCK * (waves[j].f - fk) / (BOLTZMANN * self.temperature_k);
                (j, c[k][j] * (1.0 + 1.0 / x.exp_m1()))
            })
            .collect();
        if sources.is_empty() {
            return 0.0;
        }
        let yk = &y[k];
        let end = yk[yk.len() - 1];
        let integrand: Vec<f64> = (0..yk.len())
            .map(|i| sources.iter().map(|(j, cj)| cj * y[*j][i].exp()).sum::<f64>() * (end - yk[i]).exp())
            .collect();
        let integral = h * (integrand.iter().sum::<f64>() - (integrand[0] + integrand[integrand.len() - 1]) / 2.0);
        2.0 * PLANCK * fk * integral
    }
}

/// The equations on the grid.
struct System<'a> {
    waves: &'a [Wave],
    c: &'a [Vec<f64>],
    h: f64,
}

impl System<'_> {
    /// $d \ln P_{k} / dz$ along $+z$ of the waves of `group` for their logarithms
    /// `own` and the logarithms `fixed` of every wave (the others are read there): the
    /// rate $-\alpha_{k} + \sum_{j} C_{kj} P_{j}$, opposite for the counter-propagating
    /// waves.
    fn rates(&self, group: &[usize], own: &[f64], fixed: &[f64], out: &mut [f64]) {
        let mut p: Vec<f64> = fixed.iter().map(|l| l.exp()).collect();
        for (g, l) in group.iter().zip(own) {
            p[*g] = l.exp();
        }
        for (o, k) in out.iter_mut().zip(group) {
            let gain: f64 = self.c[*k].iter().zip(&p).map(|(c, p)| c * p).sum();
            let r = -self.waves[*k].alpha + gain;
            *o = if self.waves[*k].counter { -r } else { r };
        }
    }

    /// Integrate the waves of `group` over the grid (forward from $z = 0$ or backward
    /// from $z = L$) by RK4 with the other waves fixed, and relax the new profiles by
    /// `omega`: the largest change of the logarithms.
    fn sweep(&self, y: &mut [Vec<f64>], group: &[usize], forward: bool, omega: f64) -> f64 {
        let m = y[0].len() - 1;
        let n = y.len();
        let g = group.len();
        let at = |y: &[Vec<f64>], i: usize| -> Vec<f64> { (0..n).map(|k| y[k][i]).collect() };
        let mut new: Vec<Vec<f64>> = group.iter().map(|k| y[*k].clone()).collect();
        let (start, step) = if forward { (0, 1i64) } else { (m, -1i64) };
        let h = if forward { self.h } else { -self.h };
        let mut s: Vec<f64> = group.iter().map(|k| y[*k][start]).collect();
        let (mut k1, mut k2, mut k3, mut k4) = (vec![0.0; g], vec![0.0; g], vec![0.0; g], vec![0.0; g]);
        let mut tmp = vec![0.0; g];
        let mut i = start;
        for _ in 0..m {
            let next = (i as i64 + step) as usize;
            let (fa, fb) = (at(y, i), at(y, next));
            let fm: Vec<f64> = fa.iter().zip(&fb).map(|(a, b)| (a + b) / 2.0).collect();
            // the backward sweep integrates the same derivative along +z with h < 0
            self.rates(group, &s, &fa, &mut k1);
            for q in 0..g {
                tmp[q] = s[q] + h / 2.0 * k1[q];
            }
            self.rates(group, &tmp, &fm, &mut k2);
            for q in 0..g {
                tmp[q] = s[q] + h / 2.0 * k2[q];
            }
            self.rates(group, &tmp, &fm, &mut k3);
            for q in 0..g {
                tmp[q] = s[q] + h * k3[q];
            }
            self.rates(group, &tmp, &fb, &mut k4);
            for q in 0..g {
                s[q] += h / 6.0 * (k1[q] + 2.0 * k2[q] + 2.0 * k3[q] + k4[q]);
                new[q][next] = s[q];
            }
            i = next;
        }
        let mut change: f64 = 0.0;
        for (q, k) in group.iter().enumerate() {
            for (old, v) in y[*k].iter_mut().zip(&new[q]) {
                let relaxed = *old + omega * (v - *old);
                change = change.max((relaxed - *old).abs());
                *old = relaxed;
            }
        }
        change
    }
}
