//! Pilot-based gain correction in the presence of a cochannel interferer (Cimini 1985,
//! Sec. II-C to II-F).
//!
//! # Model
//!
//! Every sample of the desired signal $D$ (unit power) reaches the receiver through a
//! flat Rayleigh gain $Z_{d} = A_{d} e^{j\theta_{d}}$, $\mathbb{E} A_{d}^{2} = 1$, with a
//! cochannel interferer $I$ of independent gain $Z_{i}$, $\gamma = 1/\mathrm{SIR}$:
//!
//! $$
//! R = Z_{d} D + \sqrt{\gamma}\, Z_{i} I .
//! $$
//!
//! A pilot gives the complex envelope $Z_{p}$, equal to $Z_{d}$ in flat fading and
//! correlated with it, $\mathbb{E}[Z_{d} Z_{p}^{*}] = \rho\, e^{j\phi}$, when the pilot
//! is $s$ apart in frequency. The correction $\hat{D} = R / Z_{c}$ either follows the
//! fades only down to a limit $\varepsilon$ (amplitude relative to the RMS gain),
//! $Z_{c} = \max(A_{p}, \varepsilon)\, e^{j\theta_{p}}$ ([`gain_limited`], eq. (16)), or
//! uses the optimum factor $Z_{c} = Z_{d}\left[1 + \gamma A_{i}^{2} / A_{d}^{2}\right]$
//! of eq. (15), which needs the amplitude of the interferer. Unlimited correction
//! ($\varepsilon = 0$) boosts the interferer in the deep fades of the desired signal.
//!
//! The signal-to-distortion ratio $\mathrm{SDR} = 1 / \mathbb{E}|\hat{D} - D|^{2}$ is, for
//! gain-limited correction (eq. (25), and eq. (20) for $\rho = 1$, flat fading),
//!
//! $$
//! \mathrm{SDR}^{-1} = \frac{\mathrm{SIR} + 1}{\mathrm{SIR}}\, \frac{1 - e^{-\varepsilon^{2}}}
//! {\varepsilon^{2}} + 1 - \frac{\rho \sqrt{\pi} \cos\phi}{\varepsilon}
//! \operatorname{erf}(\varepsilon) + \left(1 - \rho^{2} + \frac{1}{\mathrm{SIR}}\right)
//! E_{1}(\varepsilon^{2}),
//! $$
//!
//! [`sdr_gain_limited`], and for the optimum correction (eq. (21))
//! $\mathrm{SDR}^{-1} = \frac{1}{\mathrm{SIR} - 1}\left(\frac{\mathrm{SIR}}{\mathrm{SIR} - 1}
//! \ln \mathrm{SIR} - 1\right)$, [`sdr_optimum`].
//!
//! Three points of the paper do not hold as printed, and are corrected here:
//!
//! - eq. (19) lacks the $-1$ of the first term, $\overline{(A_{d} \min(1/A_{d},
//!   1/\varepsilon) - 1)^{2}}$, without which eq. (20) does not follow;
//! - the coherence bandwidth of the two-ray model of eq. (23), where $\rho^{2} = 0.5$, is
//!   $s = \pi / (2\Delta)$ rad/s, $1/(4\Delta)$ Hz, not "$1/\Delta$";
//! - the phase of the correlation of that model is $\phi = -s\Delta/2$ (a delay of
//!   $\Delta/2$ on average); $\arctan(-s\sigma)$, $\sigma = \Delta/2$, is that of an
//!   exponential delay profile. The two agree to $10^{-3}$ rad at the separations of
//!   Fig. 7 (500 Hz and 1 kHz with $\Delta = 50$ µs).
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::correction::{sdr_gain_limited, sdr_optimum};
//! use com_math::C64;
//! // SIR 20 dB: limiting the correction near -10 dB costs little against the optimum
//! let best = sdr_gain_limited(100.0, 10f64.powf(-0.5), C64::new(1.0, 0.0));
//! assert!(10.0 * best.log10() > 13.0 && best < sdr_optimum(100.0));
//! ```
//!
//! # References
//!
//! 1. L. J. Cimini, Jr., "Analysis and simulation of a digital mobile channel using
//!    orthogonal frequency division multiplexing," *IEEE Trans. Commun.*, 33(7),
//!    665–675, 1985.

use com_math::{C64, erf, expint_e1};

/// Gain-limited correction of eq. (16): $\hat{D} = R / (\max(|Z_{p}|, \varepsilon)
/// e^{j \arg Z_{p}})$ for every sample of `r` and of the pilot envelope `pilot`
/// (`eps`: the limit, an amplitude relative to the RMS gain; 0 for unlimited
/// correction). A zero pilot with no limit gives a non-finite sample.
pub fn gain_limited(r: &[C64], pilot: &[C64], eps: f64) -> Vec<C64> {
    r.iter()
        .zip(pilot)
        .map(|(r, p)| {
            let a = p.norm();
            if a >= eps { r / p } else { r / (p * (eps / a)) }
        })
        .collect()
}

/// The SDR (linear) of gain-limited correction, eq. (25) of Cimini 1985 (see the
/// module): `sir` linear (infinite without interferer), `eps` the limit (amplitude
/// relative to the RMS gain, positive), `rho` the complex correlation
/// $\rho e^{j\phi} = \mathbb{E}[Z_{d} Z_{p}^{*}]$ of the data and pilot envelopes (1
/// in flat fading, eq. (20)).
pub fn sdr_gain_limited(sir: f64, eps: f64, rho: C64) -> f64 {
    let e2 = eps * eps;
    let gamma = 1.0 / sir;
    let d = (1.0 + gamma) * -(-e2).exp_m1() / e2 + 1.0 - rho.re * std::f64::consts::PI.sqrt() * erf(eps) / eps
        + (1.0 - rho.norm_sqr() + gamma) * expint_e1(e2);
    1.0 / d
}

/// The SDR (linear) of the optimum gain correction, eq. (21) of Cimini 1985:
/// $1 / \mathbb{E}\left[\gamma A_{i}^{2} / (A_{d}^{2} + \gamma A_{i}^{2})\right]$ for
/// independent Rayleigh amplitudes (`sir` linear; `sir` = 1 gives 2, the limit).
pub fn sdr_optimum(sir: f64) -> f64 {
    if (sir - 1.0).abs() < 1e-6 {
        // the expansion of eq. (21) at SIR = 1: 1/2 + (SIR - 1)/6 + …
        return 1.0 / (0.5 - (sir - 1.0) / 6.0);
    }
    let s = sir - 1.0;
    s / (sir / s * sir.ln() - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    /// The mean squared distortion of a correction rule by Monte Carlo over independent
    /// Rayleigh gains, the pilot of correlation `rho` with the data.
    fn distortion(sir: f64, rho: C64, correct: impl Fn(C64, C64, C64, f64) -> C64) -> (f64, f64) {
        let mut r = Rng::new(17);
        let n: usize = 400_000;
        let g = 1.0 / sir;
        let qpsk = |r: &mut Rng| {
            C64::new(if r.below(2) == 0 { 1.0 } else { -1.0 }, if r.below(2) == 0 { 1.0 } else { -1.0 }) / 2f64.sqrt()
        };
        let v: Vec<f64> = (0..n)
            .map(|_| {
                let (zd, zi, zx) = (r.complex_gaussian(), r.complex_gaussian(), r.complex_gaussian());
                // E[Zd Zp*] = rho: Zp = rho* Zd + sqrt(1 - |rho|²) Zx
                let zp = rho.conj() * zd + (1.0 - rho.norm_sqr()).sqrt() * zx;
                let (d, i) = (qpsk(&mut r), qpsk(&mut r));
                let rx = zd * d + g.sqrt() * zi * i;
                (correct(rx, zp, zd, zi.norm_sqr() * g) - d).norm_sqr()
            })
            .collect();
        let mean = v.iter().sum::<f64>() / n as f64;
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n * (n - 1)) as f64).sqrt();
        (mean, sd)
    }

    #[test]
    fn eq_25_is_the_distortion_of_gain_limited_correction() {
        for (sir, eps, rho) in [
            (100.0, 0.3, C64::new(1.0, 0.0)),
            (10.0, 0.6, C64::new(1.0, 0.0)),
            (f64::INFINITY, 0.2, C64::from_polar(0.95, -0.2)),
            (30.0, 0.4, C64::from_polar(0.8, 0.5)),
        ] {
            let (d, sd) = distortion(sir, rho, |rx, zp, _, _| gain_limited(&[rx], &[zp], eps)[0]);
            let want = 1.0 / sdr_gain_limited(sir, eps, rho);
            assert!((d - want).abs() < 4.0 * sd, "SIR {sir}, ε {eps}: {d} ± {sd} vs {want}");
        }
    }

    #[test]
    fn eq_21_is_the_distortion_of_the_optimum_correction() {
        for sir in [0.5, 1.0, 3.0, 100.0] {
            // Z_c = Z_d (1 + γ A_i²/A_d²)
            let (d, sd) =
                distortion(sir, C64::new(1.0, 0.0), |rx, _, zd, gai2| rx / (zd * (1.0 + gai2 / zd.norm_sqr())));
            let want = 1.0 / sdr_optimum(sir);
            assert!((d - want).abs() < 4.0 * sd, "SIR {sir}: {d} ± {sd} vs {want}");
        }
        assert!((sdr_optimum(1.0 + 1e-9) - sdr_optimum(1.0 + 2e-6)).abs() < 1e-5);
    }

    #[test]
    fn gain_limited_correction_follows_the_fades_down_to_the_limit() {
        let r = [C64::new(0.5, 0.0), C64::new(0.0, 0.01)];
        let p = [C64::new(0.0, 0.5), C64::new(0.0, 0.01)];
        let d = gain_limited(&r, &p, 0.1);
        assert!((d[0] - C64::new(0.0, -1.0)).norm() < 1e-15);
        // |Z_p| = 0.01 < 0.1: divided by 0.1 e^{jπ/2}
        assert!((d[1] - C64::new(0.1, 0.0)).norm() < 1e-15);
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    fn db(x: f64) -> f64 {
        10.0 * x.log10()
    }

    /// L. J. Cimini, Jr., IEEE Trans. Commun. 33(7), 1985, Fig. 3 (eq. (20), the SDR
    /// against the gain limit): the peaks read on the figure (17.9 dB at -16.7 dB for an
    /// SIR of 25 dB, 13.8 at -11.6 for 20 dB, 10.1 at -7.2 for 15 dB, 6.9 at -3.3 for 10
    /// dB) within 0.3 dB and 1 dB, the SDR at -30 dB (16.2, 11.2, 6.2, 1.2 dB) within
    /// 0.2 dB; the dashed levels of the figure are the SDR of the optimum correction,
    /// eq. (21), and Fig. 4 (eq. (21)) reads 3 dB at SIR 0 dB, 14.1 at 20 dB and 31 at 40
    /// dB.
    #[test]
    fn cimini_1985_figs_3_4() {
        let one = C64::new(1.0, 0.0);
        let curve =
            |sir_db: f64, e_db: f64| db(sdr_gain_limited(10f64.powf(sir_db / 10.0), 10f64.powf(e_db / 20.0), one));
        for (sir, peak, at, low) in
            [(25.0, 17.9, -16.7, 16.2), (20.0, 13.8, -11.6, 11.2), (15.0, 10.1, -7.2, 6.2), (10.0, 6.9, -3.3, 1.2)]
        {
            let (e, v) = (0..=400)
                .map(|k| -30.0 + 0.1 * k as f64)
                .map(|e| (e, curve(sir, e)))
                .fold((0.0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            assert!((v - peak).abs() < 0.3 && (e - at).abs() < 1.0, "SIR {sir}: {v} dB at {e} dB");
            assert!((curve(sir, -30.0) - low).abs() < 0.2, "SIR {sir}");
            // the optimum lies above every gain limit, by less than 1 dB at the peak
            let opt = db(sdr_optimum(10f64.powf(sir / 10.0)));
            assert!(opt > v && opt - v < 1.0, "SIR {sir}: {opt}");
        }
        for (sir, read) in [(0.0, 3.0), (20.0, 14.1), (40.0, 31.0)] {
            assert!((db(sdr_optimum(10f64.powf(sir / 10.0))) - read).abs() < 0.3, "SIR {sir}");
        }
    }
}
