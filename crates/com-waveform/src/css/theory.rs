//! The theory of chirp spread spectrum: error rates, cross-correlation, spectrum and
//! same-SF interference.
//!
//! # Error rates over AWGN
//!
//! After dechirping and the DFT, the bin of the transmitted symbol has the magnitude
//! $M$ and every bin the noise $\mathcal{CN}(0, M/\rho)$, $\rho$ the SNR per sample.
//! Normalised, the $M - 1$ other bins are Rayleigh and the bin of the symbol Rice with
//! the location $v = \sqrt{2M\rho}$, so the noncoherent decision errs with the
//! probability (O. Afisiadis, M. Cotting, A. Burg and A. Balatsoukas-Stimming, *IEEE
//! Trans. Wireless Commun.* 19(2), 2020, eq. (16))
//!
//! $$P_{s} = \int_{0}^{\infty} f_{\mathrm{Ri}}(y; v, 1)\\,\left[1 - \left(1 -
//! e^{-y^{2}/2}\right)^{M-1}\right] dy$$
//!
//! ([`ser`]), that of $M$ orthogonal signals detected noncoherently; every wrong
//! symbol being equally likely, the BER is $P_{s} M/(2(M - 1))$ ([`ber`]). [`ser_rice`]
//! and [`ser_gumbel`] are the approximations of their eqs. (17)–(18), after T.
//! Elshabrawy and J. Robert, *IEEE Commun. Lett.* 22(9), 2018.
//!
//! # Waveforms and spectrum
//!
//! [`cross_correlation`] is the correlation of two continuous-time LoRa waveforms, eq.
//! (7) of M. Chiani and A. Elzanaty, *IEEE Internet Things J.* 6(5), 2019: they are
//! orthogonal only at one sample per chip. [`continuous_spectrum`] and
//! [`spectral_lines`] are the two parts of the power spectrum of LoRa modulated by
//! random symbols, their eqs. (20)–(25) in Fresnel functions; the lines hold exactly
//! $1/M$ of the power (their Property 4) and [`occupied_bandwidth`] is the 99 % band of
//! their Table I.
//!
//! # Same-SF interference
//!
//! An interferer of power $P_{I}$ offset by $\tau$ chips leaves in bin $k$ the pattern
//! $R_{k}$ ([`interference_pattern`], the DFT of eq. (25) of Afisiadis et al.), of the
//! magnitude of their eqs. (38)–(43) ([`interference_magnitudes`]). [`ser_interference`]
//! and [`fer_interference`] are their approximations (61)–(63) and (64)–(67).

use super::Css;
use com_math::{C64, bessel_i0e, cis, fft, fresnel, qfunc};
use std::f64::consts::{PI, TAU};

/// The SNR per sample (linear) of `snr_db`.
fn rho(snr_db: f64) -> f64 {
    com_math::db2lin(snr_db)
}

/// `1 - (1 - e^{-y²/2})^n` without cancellation.
fn one_minus_power(y: f64, n: f64) -> f64 {
    let p = (-y * y / 2.0).exp();
    if p >= 1.0 {
        return 1.0;
    }
    -((n * (-p).ln_1p()).exp_m1())
}

/// The exact SER of the noncoherent receiver over AWGN at the SNR per sample `snr_db`
/// (dB), spreading factor `sf` (see the module).
///
/// # Examples
/// ```
/// use com_waveform::css::theory::ser;
/// // SF 7: 1.6e-3 at -8 dB and 1.4e-4 at -7 dB (Fig. 2 of Afisiadis et al. 2020)
/// assert!((ser(7, -8.0) / 1.61e-3 - 1.0).abs() < 0.01);
/// assert!((ser(7, -7.0) / 1.43e-4 - 1.0).abs() < 0.01);
/// ```
pub fn ser(sf: u32, snr_db: f64) -> f64 {
    let m = (1usize << sf) as f64;
    let v = (2.0 * m * rho(snr_db)).sqrt();
    let rice = |y: f64| y * (-(y - v) * (y - v) / 2.0).exp() * bessel_i0e(y * v);
    let (lo, hi) = ((v - 12.0).max(0.0), v + 12.0);
    let n = 4800;
    com_math::simpson(|y| rice(y) * one_minus_power(y, m - 1.0), lo, hi, n).clamp(0.0, 1.0)
}

/// The BER of the noncoherent receiver over AWGN: [`ser`]` M/(2(M - 1))`, every wrong
/// symbol being equally likely.
pub fn ber(sf: u32, snr_db: f64) -> f64 {
    let m = (1usize << sf) as f64;
    ser(sf, snr_db) * m / (2.0 * (m - 1.0))
}

/// The harmonic number `H_n`.
fn harmonic(n: usize) -> f64 {
    (1..=n).map(|k| 1.0 / k as f64).sum()
}

/// The approximation of the SER of eq. (17) of Afisiadis et al. (2020), after Elshabrawy
/// and Robert (2018), the magnitudes taken as Gaussian, at the SNR per sample `snr_db`.
/// As printed, the formula reads with the SNR per symbol $M\rho$ in place of its
/// "SNR"; with it, it follows the figure, whose abscissa is the SNR per sample.
pub fn ser_rice(sf: u32, snr_db: f64) -> f64 {
    let m = 1usize << sf;
    let h = harmonic(m - 1);
    let d = (h * h - PI * PI / 12.0).sqrt();
    let gamma = m as f64 * rho(snr_db);
    qfunc((gamma.sqrt() - d.sqrt()) / (h - d + 0.5).sqrt())
}

/// The concise approximation of eq. (18) of Afisiadis et al. (2020),
/// $Q(\sqrt{2M\rho} - \sqrt{2(\ln 2 \cdot \mathrm{SF} + \gamma_{\mathrm{EM}})})$, at the SNR
/// per sample $\rho$ (`snr_db`), $\gamma_{\mathrm{EM}}$ Euler's constant.
pub fn ser_gumbel(sf: u32, snr_db: f64) -> f64 {
    const EULER: f64 = 0.577_215_664_901_532_9;
    let gamma = (1usize << sf) as f64 * rho(snr_db);
    qfunc((2.0 * gamma).sqrt() - (2.0 * (std::f64::consts::LN_2 * f64::from(sf) + EULER)).sqrt())
}

/// The cross-correlation $C_{\ell, m} = \frac{1}{T_{s}}\int_{0}^{T_{s}} x(t; \ell)\\, x^{*}(t;
/// m)\\, dt$ of two continuous-time LoRa waveforms (Chiani and Elzanaty 2019, eq. (7)):
/// $1$ for $\ell = m$, else
/// $M \frac{e^{j2\pi\ell(m - \ell)/M} - e^{j2\pi m(m - \ell)/M}}{j2\pi(M - |m - \ell|)\\,|m - \ell|}$.
pub fn cross_correlation(sf: u32, l: usize, m_sym: usize) -> C64 {
    if l == m_sym {
        return C64::new(1.0, 0.0);
    }
    let m = (1usize << sf) as f64;
    let (l, q) = (l as f64, m_sym as f64);
    let d = (q - l).abs();
    let num = cis(TAU * l * (q - l) / m) - cis(TAU * q * (q - l) / m);
    num * m / (C64::new(0.0, TAU) * (m - d) * d)
}

/// The largest real cross-correlation $\max_{\ell \ne m} \Re C_{\ell, m}$ of the
/// continuous-time waveforms (Chiani and Elzanaty 2019, Table I), which costs the
/// coherent receiver at most $-10 \log_{10}(1 - \max \Re C)$ dB against orthogonal
/// signals.
pub fn max_cross_correlation(sf: u32) -> f64 {
    let m = 1usize << sf;
    // C_{l,q} depends on l through e^{j2πl(q-l)/M}: scan every pair
    let mut best = f64::NEG_INFINITY;
    for l in 0..m {
        for q in 0..m {
            if l != q {
                best = best.max(cross_correlation(sf, l, q).re);
            }
        }
    }
    best
}

/// $\int_{t_{1}}^{t_{2}} e^{j2\pi(a t + b t^{2})} dt$ for $b > 0$, in Fresnel functions
/// (Chiani and Elzanaty 2019, eq. (24)).
fn fresnel_integral(a: f64, b: f64, t1: f64, t2: f64) -> C64 {
    let k = |x: f64| {
        let (c, s) = fresnel(x);
        C64::new(c, s)
    };
    let r = b.sqrt();
    let shift = a / (2.0 * b);
    cis(-TAU * a * a / (4.0 * b)) / (2.0 * r) * (k(2.0 * r * (t2 + shift)) - k(2.0 * r * (t1 + shift)))
}

/// The Fourier transform $X(f; \ell)$ of the waveform of symbol `l` (Chiani and Elzanaty
/// 2019, eq. (25)), for $B = 1$ (`f` in units of $B$, time in chips).
fn waveform_spectrum(m: usize, l: usize, f: f64) -> C64 {
    let (mf, lf) = (m as f64, l as f64);
    let b = 1.0 / (2.0 * mf);
    let fold = mf - lf;
    fresnel_integral(lf / mf - 0.5 - f, b, 0.0, fold) + fresnel_integral(lf / mf - 1.5 - f, b, fold, mf)
}

/// The continuous part of the power spectrum of LoRa modulated by independent uniform
/// symbols, normalised as $G^{c}(f) B$, at the frequencies `f` in units of $B$ (Chiani
/// and Elzanaty 2019, eqs. (20) and (25)); its integral is $1 - 1/M$.
pub fn continuous_spectrum(sf: u32, f: &[f64]) -> Vec<f64> {
    let m = 1usize << sf;
    let mf = m as f64;
    f.iter()
        .map(|&fk| {
            let mut energy = 0.0;
            let mut sum = C64::new(0.0, 0.0);
            for l in 0..m {
                let x = waveform_spectrum(m, l, fk);
                energy += x.norm_sqr();
                sum += x;
            }
            // 1/(T_s M) (Σ|X|² - |ΣX|²/M) with T_s = M chips and B = 1
            (energy - sum.norm_sqr() / mf) / (mf * mf)
        })
        .collect()
}

/// The lines of the power spectrum, `(f, P)` at $f = nB/M$ for $|n| \le$ `n_max` (in
/// units of $B$), $P = |\sum_{\ell} X(nB/M; \ell)|^{2}/(T_{s}^{2} M^{2})$ (Chiani and
/// Elzanaty 2019, eq. (21)); they hold $1/M$ of the power (Property 4).
pub fn spectral_lines(sf: u32, n_max: usize) -> Vec<(f64, f64)> {
    let m = 1usize << sf;
    let mf = m as f64;
    let n_max = n_max as i64;
    (-n_max..=n_max)
        .map(|n| {
            let f = n as f64 / mf;
            let sum: C64 = (0..m).map(|l| waveform_spectrum(m, l, f)).sum();
            (f, sum.norm_sqr() / (mf * mf * mf * mf))
        })
        .collect()
}

/// The band $[-z, z]$ (in units of $B$, returned as $2z$) of the complex envelope that
/// holds the fraction `fraction` of the power, lines included (Chiani and Elzanaty 2019,
/// Table I, $B_{99}$ for 0.99), from the spectrum on a grid of $B/(8M)$ up to
/// `span` $B$ on each side.
pub fn occupied_bandwidth(sf: u32, fraction: f64, span: f64) -> f64 {
    let m = 1usize << sf;
    let step = 1.0 / (8.0 * m as f64);
    let n = (span / step).ceil() as usize;
    // the spectrum is even: integrate 0 ≤ f ≤ span by the midpoint rule
    let f: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) * step).collect();
    let g = continuous_spectrum(sf, &f);
    let lines = spectral_lines(sf, (span * m as f64).ceil() as usize);
    // P(|f| ≤ z) on the grid of the cells
    let mut inside = lines.iter().filter(|(fl, _)| *fl == 0.0).map(|l| l.1).sum::<f64>();
    let mut prev = (0.0, inside);
    for (i, gi) in g.iter().enumerate() {
        let z = (i + 1) as f64 * step;
        inside += 2.0 * gi * step;
        inside += lines.iter().filter(|(fl, _)| fl.abs() > z - step && fl.abs() <= z).map(|l| l.1).sum::<f64>();
        if inside >= fraction {
            let (z0, p0) = prev;
            return 2.0 * (z0 + (z - z0) * (fraction - p0) / (inside - p0));
        }
        prev = (z, inside);
    }
    2.0 * span
}

/// The pattern $R_{k} = \mathrm{DFT}(x_{I} \odot x_{\mathrm{ref}}^{*})\[k\]$, $k = 0 \dots M -
/// 1$, that an interferer of symbols `s1`, `s2` offset by `tau` chips leaves after the
/// dechirping (Afisiadis et al. 2020, eqs. (21) and (25)).
///
/// # Errors
/// As [`Css::interferer`].
pub fn interference_pattern(sf: u32, s1: usize, s2: usize, tau: f64) -> com_core::Result<Vec<C64>> {
    let css = Css::lora(sf);
    let x = css.interferer(s1, s2, tau)?;
    let reference = css.symbol(0)?;
    Ok(fft::fft(&x.iter().zip(&reference).map(|(a, b)| a * b.conj()).collect::<Vec<_>>()))
}

/// The magnitudes $|R_{k}|$ of the interference pattern in closed form, eqs. (39)–(43)
/// of Afisiadis et al. (2020): $A_{k,1}$, $A_{k,2}$ (with their limits $\lceil\tau\rceil$
/// and $M - \lceil\tau\rceil$ when the sine of the denominator vanishes) and the phases
/// $\theta_{k,1}$, $\theta_{k,2}$ as printed.
pub fn interference_magnitudes(sf: u32, s1: usize, s2: usize, tau: f64) -> Vec<f64> {
    let m = 1usize << sf;
    let mf = m as f64;
    let c = tau.ceil();
    let (l_int, lambda) = (tau.floor(), tau - tau.floor());
    let ratio = |x: f64, len: f64| {
        let den = (PI / mf * x).sin();
        if den.abs() < 1e-12 { len } else { (PI / mf * x * len).sin() / den }
    };
    let (s1f, s2f) = (s1 as f64, s2 as f64);
    (0..m)
        .map(|k| {
            let k = k as f64;
            let a1 = ratio(s1f - k - tau, c);
            let a2 = ratio(s2f - k - tau, mf - c);
            let t1 = PI / mf
                * (-tau * tau + (lambda - l_int) * mf + s1f * (2.0 * tau - c + 1.0) + k * (c - 1.0) + tau * (c - 1.0));
            let t2 = PI / mf * (-tau * tau + s2f * (2.0 * tau - c + 1.0 - mf) + k * (c - 1.0 + mf) + tau * (c - 1.0));
            (a1 * a1 + a2 * a2 + 2.0 * a1 * a2 * (t1 - t2).cos()).max(0.0).sqrt()
        })
        .collect()
}

/// The magnitude $|A_{k,1} + A_{k,2}|$ of the strongest interference bin for `s2 = 0`
/// and `s1 = si` (Afisiadis et al. 2020, eqs. (58)–(59)), at $k = [-\lfloor\tau\rceil]_{M}$:
/// the dechirped interferer delayed by $\tau$ is a tone at $-\tau$, where $A_{k,2}$ of
/// their eq. (40) peaks; eq. (58) prints $\lfloor\tau\rceil$, where it nearly vanishes.
fn strongest_bin(m: usize, si: usize, tau: f64) -> f64 {
    let mf = m as f64;
    let c = tau.ceil();
    let k = -tau.round();
    let ratio = |x: f64, len: f64| {
        let den = (PI / mf * x).sin();
        if den.abs() < 1e-12 { len } else { (PI / mf * x * len).sin() / den }
    };
    (ratio(si as f64 - k - tau, c) + ratio(-k - tau, mf - c)).abs()
}

/// The probability that the strongest interference bin beats the bin of the symbol,
/// $Q((M - \sqrt{P_{I}}|R_{k_{\max}}|)/\sqrt{M/\rho})$, for one interfering symbol `si` and
/// offset `tau` (Afisiadis et al. 2020, eqs. (60)–(61), in the normalisation of the
/// module: the difference of two magnitudes of noise variance $M/(2\rho)$ each).
fn interference_error(m: usize, si: usize, tau: f64, rho: f64, pi_lin: f64) -> f64 {
    let mf = m as f64;
    qfunc((mf - pi_lin.sqrt() * strongest_bin(m, si, tau)) / (mf / rho).sqrt())
}

/// The step (`eps` floored at 1e-3 chip) and the offsets `0, step, 2 step, …` below
/// `half`.
fn offsets(half: f64, eps: f64) -> (f64, Vec<f64>) {
    let eps = if eps.is_finite() { eps.max(1e-3) } else { 0.2 };
    let n = (half / eps).ceil().max(0.0) as usize;
    (eps, (0..n).map(|i| i as f64 * eps).filter(|t| *t < half).collect())
}

/// The approximate SER under AWGN and a same-SF interferer of power `pi_db` (dB, relative
/// to the signal), at the SNR per sample `snr_db`, the offset sampled every `eps` chips
/// over $[0, (M - 1)/2)$ (Afisiadis et al. 2020, eqs. (62)–(63); the paper takes
/// `eps` = 1/5).
pub fn ser_interference(sf: u32, snr_db: f64, pi_db: f64, eps: f64) -> f64 {
    let m = 1usize << sf;
    let (r, pi) = (rho(snr_db), com_math::db2lin(pi_db));
    let half = (m as f64 - 1.0) / 2.0;
    let (eps, taus) = offsets(half, eps);
    let mut acc = 0.0;
    for si in 0..m {
        for &tau in &taus {
            acc += interference_error(m, si, tau, r, pi);
        }
    }
    let p_int = acc * eps / (m as f64 * (m as f64 / 2.0));
    let p_noise = ser(sf, snr_db);
    p_noise + (1.0 - p_noise) * p_int
}

/// The approximate frame error rate of `frame` uncoded symbols under AWGN and a same-SF
/// interferer whose frame of the same length overlaps a uniform number of symbols
/// (Afisiadis et al. 2020, eqs. (64)–(67)), the offset sampled every `eps` chips.
pub fn fer_interference(sf: u32, snr_db: f64, pi_db: f64, frame: usize, eps: f64) -> f64 {
    let m = 1usize << sf;
    let (r, pi) = (rho(snr_db), com_math::db2lin(pi_db));
    let half = (m as f64 - 1.0) / 2.0;
    let (eps, taus) = offsets(half, eps);
    // eq. (65): the SER of an interfered symbol at the offset τ
    let p_tau: Vec<f64> = taus
        .iter()
        .map(|&tau| (0..m).map(|si| interference_error(m, si, tau, r, pi)).sum::<f64>() / m as f64)
        .collect();
    let p_noise = ser(sf, snr_db);
    let mut total = 0.0;
    for fi in 1..=frame {
        // eqs. (64) and (66)
        let ok: f64 = p_tau.iter().map(|p| (1.0 - p).powi(fi as i32) * (1.0 - p_noise).powi((frame - fi) as i32)).sum();
        total += 2.0 / m as f64 * eps * (taus.len() as f64 - ok);
    }
    (total / frame as f64).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::css::Waveform;
    use com_core::rng::Rng;

    #[test]
    fn the_exact_ser_is_that_of_a_simulated_receiver() {
        // SF 7 at -9 dB: SER 1e-2, about 300 errors
        let css = Css::lora(7);
        let mut rng = Rng::new(1);
        let n = 30_000;
        let s: Vec<usize> = (0..n).map(|_| rng.below(128)).collect();
        let sigma = (1.0 / com_math::db2lin(-9.0)).sqrt();
        let y: Vec<C64> =
            css.modulate(&s).unwrap().into_iter().map(|v| v * cis(1.3) + rng.complex_gaussian() * sigma).collect();
        let d = css.demodulate(&y).unwrap();
        let sim = d.iter().zip(&s).filter(|(a, b)| a != b).count() as f64 / n as f64;
        let want = ser(7, -9.0);
        assert!((sim / want - 1.0).abs() < 0.15, "{sim} vs {want}");
        assert!((ber(7, -9.0) / want - 128.0 / 254.0).abs() < 1e-12);
    }

    #[test]
    fn the_cross_correlation_is_that_of_the_sampled_waveforms() {
        // Chiani and Elzanaty 2019, eq. (7), against the inner product of the waveforms
        // sampled 64 times per chip (an O(1/J) error)
        let css = Css { sf: 4, oversampling: 64, waveform: Waveform::LoRa };
        let x: Vec<Vec<C64>> = (0..16).map(|a| css.symbol(a).unwrap()).collect();
        let n = x[0].len() as f64;
        for (l, q) in [(0usize, 1usize), (3, 7), (15, 2), (5, 13), (6, 6)] {
            let c: C64 = x[l].iter().zip(&x[q]).map(|(a, b)| a * b.conj()).sum::<C64>() / n;
            let want = cross_correlation(4, l, q);
            assert!((c - want).norm() < 0.02, "C({l}, {q}) = {c} vs {want}");
        }
    }

    #[test]
    fn the_spectrum_holds_the_power_and_the_lines_hold_one_over_m() {
        // Chiani and Elzanaty 2019, Property 4 and eq. (20): SF 5
        let lines: f64 = spectral_lines(5, 40 << 5).iter().map(|l| l.1).sum();
        assert!((lines - 1.0 / 32.0).abs() < 1e-9, "{lines}");
        let step = 1.0 / 256.0;
        let f: Vec<f64> = (0..(4.0 / step) as usize).map(|i| -2.0 + (i as f64 + 0.5) * step).collect();
        let cont: f64 = continuous_spectrum(5, &f).iter().sum::<f64>() * step;
        assert!((cont - (1.0 - 1.0 / 32.0)).abs() < 2e-3, "{cont}");
    }

    #[test]
    fn the_closed_form_interference_is_the_dft_and_has_its_symmetries() {
        // Afisiadis et al. 2020, eqs. (38)-(43) and Propositions 1-2
        let sorted = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v
        };
        let mags = |s1: usize, s2: usize, tau: f64| {
            interference_pattern(6, s1, s2, tau).unwrap().iter().map(|v| v.norm()).collect::<Vec<_>>()
        };
        for (s1, s2, tau) in [(5usize, 0usize, 3.0), (40, 10, 10.4), (63, 2, 30.7), (0, 0, 0.5), (17, 17, 62.9)] {
            let dft = mags(s1, s2, tau);
            let closed = interference_magnitudes(6, s1, s2, tau);
            let err = dft.iter().zip(&closed).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-9, "({s1}, {s2}, {tau}): {err}");
        }
        // Proposition 1: shifting both symbols by δ keeps the set of magnitudes while
        // s1 ≥ s2 stays true
        let base = sorted(mags(20, 5, 7.3));
        for delta in [1usize, 10, 43] {
            let shifted = sorted(mags(20 + delta, 5 + delta, 7.3));
            assert!(base.iter().zip(&shifted).all(|(a, b)| (a - b).abs() < 1e-9), "δ = {delta}");
        }
        // Proposition 2: τ and (M - 1) - τ give the same set
        let (a, b) = (sorted(mags(20, 5, 7.3)), sorted(mags(20, 5, 63.0 - 7.3)));
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-9));
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    /// M. Chiani and A. Elzanaty, "On the LoRa modulation for IoT: waveform properties and
    /// spectral analysis," IEEE Internet Things J. 6(5), 2019, Table I: the maximum real
    /// cross-correlation (0.212, 0.091, 0.045, 0.015 for M = 2^3, 2^5, 2^7, 2^10), the SNR
    /// penalty (1.04, 0.41, 0.20, 0.07 dB), the power of the lines (1/M) and the 99 %
    /// band (1.500 B, 1.185 B, 1.045 B for M = 2^3, 2^5, 2^7). SF 10 and 12 of the band
    /// are checked by `validation/tests/css.rs`.
    #[test]
    fn chiani_2019_table_1() {
        for (sf, corr, penalty) in [(3u32, 0.212, 1.04), (5, 0.091, 0.41), (7, 0.045, 0.20), (10, 0.015, 0.07)] {
            let c = max_cross_correlation(sf);
            assert!((c - corr).abs() < 5e-4, "SF {sf}: {c}");
            let p = -10.0 * (1.0 - c).log10();
            assert!((p - penalty).abs() < 0.005, "SF {sf}: {p} dB");
        }
        for (sf, b99) in [(3u32, 1.500), (5, 1.185), (7, 1.045)] {
            let b = occupied_bandwidth(sf, 0.99, 3.0);
            assert!((b / b99 - 1.0).abs() < 0.005, "SF {sf}: {b}");
            // the lines beyond 40 B hold less than 1e-4 of their power
            let lines: f64 = spectral_lines(sf, 40 << sf).iter().map(|l| l.1).sum();
            assert!((lines * (1 << sf) as f64 - 1.0).abs() < 1e-4, "SF {sf}: {lines}");
        }
    }

    /// O. Afisiadis, M. Cotting, A. Burg and A. Balatsoukas-Stimming, "On the error rate of
    /// the LoRa modulation with interference," IEEE Trans. Wireless Commun. 19(2), 2020,
    /// Fig. 2 (SF 7, read 1.8e-3 at -8 dB and 1.4e-4 at -7 dB) against the exact SER of eq.
    /// (16); eq. (17) follows it within 30 % from 1e-2 to 1e-6 when its "SNR" is the SNR
    /// per symbol $M\rho$ (the abscissa is the SNR per sample $\rho$, while eqs. (3)–(4)
    /// define the SNR as $1/N_{0}$ with $\sigma^{2} = N_{0}/(2N)$, which would make the
    /// SNR per sample $2N$ times larger); eq. (18) is optimistic, by a factor 1.05 to 5.2
    /// over that range (the most at SF 7 and low rates).
    #[test]
    fn afisiadis_2020_fig_2_and_eqs_16_to_18() {
        assert!((ser(7, -8.0) / 1.8e-3 - 1.0).abs() < 0.15);
        assert!((ser(7, -7.0) / 1.4e-4 - 1.0).abs() < 0.1);
        for sf in 7u32..=12 {
            for snr in (-30..=0).map(f64::from) {
                let exact = ser(sf, snr);
                if (1e-6..1e-2).contains(&exact) {
                    let (r, g) = (ser_rice(sf, snr), ser_gumbel(sf, snr));
                    assert!((r / exact - 1.0).abs() < 0.3, "SF {sf} {snr} dB: {r} vs {exact}");
                    assert!(g < exact && g > exact / 6.0, "SF {sf} {snr} dB: {g} vs {exact}");
                }
            }
        }
    }

    /// L. Vangelista, "Frequency shift chirp modulation: the LoRa modulation," IEEE Signal
    /// Process. Lett. 24(12), 2017, Fig. 1, flat channel: the BER of FSCM and FSK with
    /// SF 7, read 2e-2, 4.7e-3, 8e-4 and 7.5e-5 at -10, -9, -8 and -7 dB, against the BER
    /// of 128 orthogonal signals detected noncoherently.
    #[test]
    fn vangelista_2017_fig_1_awgn() {
        for (snr, read) in [(-10.0, 2e-2), (-9.0, 4.7e-3), (-8.0, 8e-4), (-7.0, 7.5e-5)] {
            let b = ber(7, snr);
            assert!((b / read - 1.0).abs() < 0.15, "{snr} dB: {b} vs {read}");
        }
    }

    /// Afisiadis et al. 2020, eq. (58): the strongest bin of an interferer offset by
    /// $\tau$ is $[-\lfloor\tau\rceil]_{M}$, where $A_{k,2}$ of their eq. (40) peaks; the
    /// printed $\lfloor\tau\rceil$ is a bin of almost no energy.
    #[test]
    fn afisiadis_2020_eq_58_strongest_bin() {
        let m = 128usize;
        for (si, tau) in [(0usize, 3.0), (40, 10.4), (100, 30.7), (5, 50.2)] {
            let r = interference_pattern(7, si, 0, tau).unwrap();
            let kmax =
                r.iter().enumerate().fold((0, 0.0), |b, (k, v)| if v.norm() > b.1 { (k, v.norm()) } else { b }).0;
            assert_eq!(kmax, (m - tau.round() as usize) % m, "τ = {tau}");
            assert!(r[tau.round() as usize].norm() < 0.05 * r[kmax].norm());
            // eq. (59) drops the phase term of eq. (43): within 0.5 %
            assert!((strongest_bin(m, si, tau) / r[kmax].norm() - 1.0).abs() < 5e-3);
        }
    }
}
