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
use com_core::Result;
use com_math::{C64, cis, fft, fresnel, qfunc};
use std::f64::consts::{PI, TAU};

/// The SNR per sample (linear) of `snr_db`.
fn rho(snr_db: f64) -> f64 {
    com_math::db2lin(snr_db)
}

/// The number of chirps $M = 2^{\mathrm{SF}}$ of the spreading factor `sf`.
///
/// # Errors
/// As [`Css::check`]: the spreading factor lies in 1..=16.
fn chips(sf: u32) -> Result<usize> {
    Css::lora(sf).check()?;
    Ok(1 << sf)
}

/// The exact SER of the noncoherent receiver over AWGN at the SNR per sample `snr_db`
/// (dB), spreading factor `sf` (see the module): the SER of $M$ orthogonal signals
/// ([`com_mapping::theory::ser_orthogonal_noncoherent`]) at $E_{s}/N_{0} = M\rho$.
///
/// # Errors
/// [`com_core::Error::Config`] unless the spreading factor lies in 1..=16.
///
/// # Examples
/// ```
/// use com_waveform::css::theory::ser;
/// // SF 7: 1.6e-3 at -8 dB and 1.4e-4 at -7 dB (Fig. 2 of Afisiadis et al. 2020)
/// assert!((ser(7, -8.0)? / 1.61e-3 - 1.0).abs() < 0.01);
/// assert!((ser(7, -7.0)? / 1.43e-4 - 1.0).abs() < 0.01);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn ser(sf: u32, snr_db: f64) -> Result<f64> {
    let m = chips(sf)?;
    com_mapping::theory::ser_orthogonal_noncoherent(m, snr_db + com_math::lin2db(m as f64))
}

/// The BER of the noncoherent receiver over AWGN: [`ser`]` M/(2(M - 1))`, every wrong
/// symbol being equally likely.
///
/// # Errors
/// As [`ser`].
pub fn ber(sf: u32, snr_db: f64) -> Result<f64> {
    let m = chips(sf)? as f64;
    Ok(ser(sf, snr_db)? * m / (2.0 * (m - 1.0)))
}

/// The harmonic number `H_n`.
fn harmonic(n: usize) -> f64 {
    (1..=n).map(|k| 1.0 / k as f64).sum()
}

/// The approximation of the SER of eq. (17) of Afisiadis et al. (2020), after Elshabrawy
/// and Robert (2018), the magnitudes taken as Gaussian, at the SNR per sample `snr_db`.
/// As printed, the formula reads with the SNR per symbol $M\rho$ in place of its
/// "SNR"; with it, it follows the figure, whose abscissa is the SNR per sample.
///
/// # Errors
/// As [`ser`].
pub fn ser_rice(sf: u32, snr_db: f64) -> Result<f64> {
    let m = chips(sf)?;
    let h = harmonic(m - 1);
    let d = (h * h - PI * PI / 12.0).sqrt();
    let gamma = m as f64 * rho(snr_db);
    Ok(qfunc((gamma.sqrt() - d.sqrt()) / (h - d + 0.5).sqrt()))
}

/// The concise approximation of eq. (18) of Afisiadis et al. (2020),
/// $Q(\sqrt{2M\rho} - \sqrt{2(\ln 2 \cdot \mathrm{SF} + \gamma_{\mathrm{EM}})})$, at the SNR
/// per sample $\rho$ (`snr_db`), $\gamma_{\mathrm{EM}}$ Euler's constant.
///
/// # Errors
/// As [`ser`].
pub fn ser_gumbel(sf: u32, snr_db: f64) -> Result<f64> {
    const EULER: f64 = 0.577_215_664_901_532_9;
    let gamma = chips(sf)? as f64 * rho(snr_db);
    Ok(qfunc((2.0 * gamma).sqrt() - (2.0 * (std::f64::consts::LN_2 * f64::from(sf) + EULER)).sqrt()))
}

/// The cross-correlation $C_{\ell, m} = \frac{1}{T_{s}}\int_{0}^{T_{s}} x(t; \ell)\\, x^{*}(t;
/// m)\\, dt$ of two continuous-time LoRa waveforms (Chiani and Elzanaty 2019, eq. (7)):
/// $1$ for $\ell = m$, else
/// $M \frac{e^{j2\pi\ell(m - \ell)/M} - e^{j2\pi m(m - \ell)/M}}{j2\pi(M - |m - \ell|)\\,|m - \ell|}$.
///
/// # Errors
/// As [`ser`], and [`com_core::Error::Config`] unless both symbols lie below $M$.
pub fn cross_correlation(sf: u32, l: usize, m_sym: usize) -> Result<C64> {
    let m = chips(sf)?;
    com_core::ensure(l < m && m_sym < m, || format!("the symbols {l} and {m_sym} must lie below M = {m}"))?;
    Ok(correlation(m as f64, l, m_sym))
}

/// [`cross_correlation`] for $M$ chips (`m` as a float) and valid symbols.
fn correlation(m: f64, l: usize, m_sym: usize) -> C64 {
    if l == m_sym {
        return C64::new(1.0, 0.0);
    }
    let (l, q) = (l as f64, m_sym as f64);
    let d = (q - l).abs();
    let num = cis(TAU * l * (q - l) / m) - cis(TAU * q * (q - l) / m);
    num * m / (C64::new(0.0, TAU) * (m - d) * d)
}

/// The largest real cross-correlation $\max_{\ell \ne m} \Re C_{\ell, m}$ of the
/// continuous-time waveforms (Chiani and Elzanaty 2019, Table I), which costs the
/// coherent receiver at most $-10 \log_{10}(1 - \max \Re C)$ dB against orthogonal
/// signals.
///
/// # Errors
/// As [`ser`].
pub fn max_cross_correlation(sf: u32) -> Result<f64> {
    let m = chips(sf)?;
    // C_{l,q} depends on l through e^{j2πl(q-l)/M}: scan every pair
    let mut best = f64::NEG_INFINITY;
    for l in 0..m {
        for q in 0..m {
            if l != q {
                best = best.max(correlation(m as f64, l, q).re);
            }
        }
    }
    Ok(best)
}

/// $K(x) = C(x) + j S(x)$, the Fresnel integrals.
fn fresnel_k(x: f64) -> C64 {
    let (c, s) = fresnel(x);
    C64::new(c, s)
}

/// The parts of the Fourier transform $X(f; \ell)$ of the waveform of symbol $\ell$
/// (Chiani and Elzanaty 2019, eq. (25)) at one frequency `f` (units of $B$, time in
/// chips) that every symbol shares. Each of the two chirp segments of eq. (25) is
/// $\int e^{j2\pi(a t + b t^{2})} dt$ in Fresnel functions (their eq. (24)), $b = 1/(2M)$;
/// two of their four arguments do not depend on $\ell$ and the third is common to both:
///
/// $$
/// X(f; \ell) = \sqrt{\frac{M}{2}}\\, e^{-j\pi(\ell - c)^{2}/M} \left[K(\beta_{+}) - w
/// K(\beta_{-}) + (w - 1) K\left(\sqrt{2/M}\\,(\ell - c)\right)\right],
/// $$
///
/// with $c = M/2 + fM$, $\beta_{\pm} = \sqrt{2/M}\\,(\pm M/2 - fM)$ and
/// $w = e^{-j 2\pi f M}$: one Fresnel evaluation per symbol instead of four.
struct Shared {
    /// $M$.
    m: f64,
    /// $c = M/2 + fM$.
    c: f64,
    /// $K(\beta_{+}) - w K(\beta_{-})$.
    d: C64,
    /// $w - 1$.
    w1: C64,
    /// $\sqrt{M/2}$.
    root: f64,
}

/// $e^{-j\pi x^{2}/M}$, the phase of $X(f; \ell)$ at $x = \ell - c$.
fn chirp_phase(m: usize, x: f64) -> C64 {
    cis(-PI * x * x / m as f64)
}

impl Shared {
    fn new(m: usize, f: f64) -> Shared {
        let m = m as f64;
        let s = (2.0 / m).sqrt();
        let w = cis(-TAU * f * m);
        let d = fresnel_k(s * (m / 2.0 - f * m)) - w * fresnel_k(s * (-m / 2.0 - f * m));
        Shared { m, c: m / 2.0 + f * m, d, w1: w - 1.0, root: (m / 2.0).sqrt() }
    }

    /// $X(f; \ell)$ from the phase $e^{-j\pi x^{2}/M}$ ([`chirp_phase`]) and
    /// $K(\sqrt{2/M}\\, x)$ of $x = \ell - c$.
    fn spectrum(&self, phase: C64, k: C64) -> C64 {
        phase * (self.d + self.w1 * k) * self.root
    }

    /// $G^{c}(f) B = \frac{1}{T_{s} M} (\sum_{\ell} |X(f; \ell)|^{2} - |\sum_{\ell} X(f;
    /// \ell)|^{2}/M)$ (eq. (20)) from the terms $X(f; \ell)$, with $T_{s} = M$ chips and
    /// $B = 1$.
    fn density(&self, terms: impl Iterator<Item = C64>) -> f64 {
        let (energy, sum) = terms.fold((0.0, C64::new(0.0, 0.0)), |(e, s), x| (e + x.norm_sqr(), s + x));
        (energy - sum.norm_sqr() / self.m) / (self.m * self.m)
    }
}

/// The continuous part of the power spectrum of LoRa modulated by independent uniform
/// symbols, normalised as $G^{c}(f) B$, at the frequencies `f` in units of $B$ (Chiani
/// and Elzanaty 2019, eqs. (20) and (25)); its integral is $1 - 1/M$.
///
/// Eq. (20) sums over the $M$ symbols: each frequency costs $M + 2$ Fresnel evaluations
/// (two of the four arguments of eq. (25) are shared by every symbol), $O(M)$.
///
/// # Errors
/// As [`ser`].
pub fn continuous_spectrum(sf: u32, f: &[f64]) -> Result<Vec<f64>> {
    let m = chips(sf)?;
    let s = (2.0 / m as f64).sqrt();
    Ok(f.iter()
        .map(|&fk| {
            let shared = Shared::new(m, fk);
            shared.density((0..m).map(|l| {
                let x = l as f64 - shared.c;
                shared.spectrum(chirp_phase(m, x), fresnel_k(s * x))
            }))
        })
        .collect())
}

/// The continuous spectrum at the midpoints $f_{i} = (i + 1/2)/(8M)$, $i < n$, of the
/// grid of [`occupied_bandwidth`]: there $\ell - c = (8\ell - 4M - i - 1/2)/8$ depends on
/// $8\ell - i$ only, so the Fresnel function and the phase of eq. (25) are tabulated
/// once on the $8(M - 1) + n$ values of $8\ell - i$, and each of the $nM$ terms costs
/// three complex products.
fn continuous_spectrum_grid(m: usize, n: usize) -> Vec<f64> {
    let s = (2.0 / m as f64).sqrt();
    let mf = m as f64;
    // index j = 8ℓ - i + n - 1 ∈ [0, 8(M - 1) + n - 1]
    let table: Vec<(C64, C64)> = (0..8 * (m - 1) + n)
        .map(|j| {
            let x = (j as f64 - (n - 1) as f64 - 4.0 * mf - 0.5) / 8.0;
            (chirp_phase(m, x), fresnel_k(s * x))
        })
        .collect();
    (0..n)
        .map(|i| {
            let shared = Shared::new(m, (i as f64 + 0.5) / (8.0 * mf));
            shared.density((0..m).map(|l| {
                let (phase, k) = table[8 * l + n - 1 - i];
                shared.spectrum(phase, k)
            }))
        })
        .collect()
}

/// The lines of the power spectrum, `(f, P)` at $f = nB/M$ for $|n| \le$ `n_max` (in
/// units of $B$), $P = |\sum_{\ell} X(nB/M; \ell)|^{2}/(T_{s}^{2} M^{2})$ (Chiani and
/// Elzanaty 2019, eq. (21)); they hold $1/M$ of the power (Property 4).
///
/// At $f = n/M$, $w = 1$ and the sum over the symbols is the quadratic Gauss sum
/// $\sum_{\ell = 0}^{M - 1} e^{-j\pi(\ell - c)^{2}/M} = \sqrt{M} e^{-j\pi/4}$ ($c$ an
/// integer, $M$ even), so that
///
/// $$
/// P_{n} = \frac{1}{2M^{2}} \left|K\left(\sqrt{2/M}\\,(M/2 - n)\right) +
/// K\left(\sqrt{2/M}\\,(M/2 + n)\right)\right|^{2},
/// $$
///
/// two Fresnel evaluations per line.
///
/// # Errors
/// As [`ser`].
pub fn spectral_lines(sf: u32, n_max: usize) -> Result<Vec<(f64, f64)>> {
    let m = chips(sf)?;
    let mf = m as f64;
    let s = (2.0 / mf).sqrt();
    let n_max = n_max as i64;
    Ok((-n_max..=n_max)
        .map(|n| {
            let n = n as f64;
            let d = fresnel_k(s * (mf / 2.0 - n)) + fresnel_k(s * (mf / 2.0 + n));
            (n / mf, d.norm_sqr() / (2.0 * mf * mf))
        })
        .collect())
}

/// The band $[-z, z]$ (in units of $B$, returned as $2z$) of the complex envelope that
/// holds the fraction `fraction` of the power, lines included (Chiani and Elzanaty 2019,
/// Table I, $B_{99}$ for 0.99), from the spectrum on a grid of $B/(8M)$ up to
/// `span` $B$ on each side.
///
/// The $8M \cdot$ `span` points of the grid cost $O(M^{2}\\,\mathrm{span})$ products
/// and $O(M\\,\mathrm{span})$ Fresnel evaluations (0.13 s at SF 12 and a span of $0.6B$).
///
/// # Errors
/// As [`ser`], and [`com_core::Error::Config`] unless `fraction` lies in (0, 1) and
/// `span` is positive.
pub fn occupied_bandwidth(sf: u32, fraction: f64, span: f64) -> Result<f64> {
    let m = chips(sf)?;
    com_core::ensure(fraction > 0.0 && fraction < 1.0 && span > 0.0, || {
        format!("a fraction in (0, 1) and a positive span, got {fraction} and {span}")
    })?;
    let step = 1.0 / (8.0 * m as f64);
    let n = (span / step).ceil() as usize;
    // the spectrum is even: integrate 0 ≤ f ≤ span by the midpoint rule
    let g = continuous_spectrum_grid(m, n);
    let mut lines = spectral_lines(sf, (span * m as f64).ceil() as usize)?;
    // P(|f| ≤ z) on the grid of the cells
    let mut inside = lines.iter().filter(|(fl, _)| *fl == 0.0).map(|l| l.1).sum::<f64>();
    // the lines of the cell (z - step, z]: a window of the lines sorted by |f|
    lines.sort_by(|a, b| a.0.abs().total_cmp(&b.0.abs()));
    let (mut lo, mut hi) = (0, 0);
    let mut prev = (0.0, inside);
    for (i, gi) in g.iter().enumerate() {
        let z = (i + 1) as f64 * step;
        inside += 2.0 * gi * step;
        while lo < lines.len() && lines[lo].0.abs() <= z - step {
            lo += 1;
        }
        hi = hi.max(lo);
        while hi < lines.len() && lines[hi].0.abs() <= z {
            hi += 1;
        }
        inside += lines[lo..hi].iter().map(|l| l.1).sum::<f64>();
        if inside >= fraction {
            let (z0, p0) = prev;
            return Ok(2.0 * (z0 + (z - z0) * (fraction - p0) / (inside - p0)));
        }
        prev = (z, inside);
    }
    Ok(2.0 * span)
}

/// The pattern $R_{k} = \mathrm{DFT}(x_{I} \odot x_{\mathrm{ref}}^{*})\[k\]$, $k = 0 \dots M -
/// 1$, that an interferer of symbols `s1`, `s2` offset by `tau` chips leaves after the
/// dechirping (Afisiadis et al. 2020, eqs. (21) and (25)).
///
/// # Errors
/// As [`Css::interferer`].
pub fn interference_pattern(sf: u32, s1: usize, s2: usize, tau: f64) -> Result<Vec<C64>> {
    let css = Css::lora(sf);
    let x = css.interferer(s1, s2, tau)?;
    let reference = css.symbol(0)?;
    Ok(fft::fft(&x.iter().zip(&reference).map(|(a, b)| a * b.conj()).collect::<Vec<_>>()))
}

/// The magnitudes $|R_{k}|$ of the interference pattern in closed form, eqs. (39)–(43)
/// of Afisiadis et al. (2020): $A_{k,1}$, $A_{k,2}$ (with their limits $\lceil\tau\rceil$
/// and $M - \lceil\tau\rceil$ when the sine of the denominator vanishes) and the phases
/// $\theta_{k,1}$, $\theta_{k,2}$ as printed.
///
/// # Errors
/// As [`Css::interferer`].
pub fn interference_magnitudes(sf: u32, s1: usize, s2: usize, tau: f64) -> Result<Vec<f64>> {
    let m = chips(sf)?;
    com_core::ensure(s1 < m && s2 < m, || format!("the symbols must lie below M = {m}"))?;
    com_core::ensure((0.0..m as f64).contains(&tau), || format!("the offset must lie in [0, {m}), got {tau}"))?;
    let mf = m as f64;
    let c = tau.ceil();
    let (l_int, lambda) = (tau.floor(), tau - tau.floor());
    // sin(π x len / M) / sin(π x / M), len a whole number of chips
    let ratio = |x: f64, len: f64| if len == 0.0 { 0.0 } else { len * com_math::diric(TAU * x / mf, len as usize) };
    let (s1f, s2f) = (s1 as f64, s2 as f64);
    Ok((0..m)
        .map(|k| {
            let k = k as f64;
            let a1 = ratio(s1f - k - tau, c);
            let a2 = ratio(s2f - k - tau, mf - c);
            let t1 = PI / mf
                * (-tau * tau + (lambda - l_int) * mf + s1f * (2.0 * tau - c + 1.0) + k * (c - 1.0) + tau * (c - 1.0));
            let t2 = PI / mf * (-tau * tau + s2f * (2.0 * tau - c + 1.0 - mf) + k * (c - 1.0 + mf) + tau * (c - 1.0));
            (a1 * a1 + a2 * a2 + 2.0 * a1 * a2 * (t1 - t2).cos()).max(0.0).sqrt()
        })
        .collect())
}

/// The magnitude $|A_{k,1} + A_{k,2}|$ of the strongest interference bin for `s2 = 0`
/// and `s1 = si` (Afisiadis et al. 2020, eqs. (58)–(59)), at $k = [-\lfloor\tau\rceil]_{M}$:
/// the dechirped interferer delayed by $\tau$ is a tone at $-\tau$, where $A_{k,2}$ of
/// their eq. (40) peaks; eq. (58) prints $\lfloor\tau\rceil$, where it nearly vanishes.
fn strongest_bin(m: usize, si: usize, tau: f64) -> f64 {
    let mf = m as f64;
    let c = tau.ceil();
    let k = -tau.round();
    // sin(π x len / M) / sin(π x / M), len a whole number of chips
    let ratio = |x: f64, len: f64| if len == 0.0 { 0.0 } else { len * com_math::diric(TAU * x / mf, len as usize) };
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
///
/// # Errors
/// As [`ser`].
pub fn ser_interference(sf: u32, snr_db: f64, pi_db: f64, eps: f64) -> Result<f64> {
    let m = chips(sf)?;
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
    let p_noise = ser(sf, snr_db)?;
    Ok(p_noise + (1.0 - p_noise) * p_int)
}

/// The approximate frame error rate of `frame` uncoded symbols under AWGN and a same-SF
/// interferer whose frame of the same length overlaps a uniform number of symbols
/// (Afisiadis et al. 2020, eqs. (64)–(67)), the offset sampled every `eps` chips.
///
/// # Errors
/// As [`ser`].
pub fn fer_interference(sf: u32, snr_db: f64, pi_db: f64, frame: usize, eps: f64) -> Result<f64> {
    let m = chips(sf)?;
    let (r, pi) = (rho(snr_db), com_math::db2lin(pi_db));
    let half = (m as f64 - 1.0) / 2.0;
    let (eps, taus) = offsets(half, eps);
    // eq. (65): the SER of an interfered symbol at the offset τ
    let p_tau: Vec<f64> = taus
        .iter()
        .map(|&tau| (0..m).map(|si| interference_error(m, si, tau, r, pi)).sum::<f64>() / m as f64)
        .collect();
    let p_noise = ser(sf, snr_db)?;
    let mut total = 0.0;
    for fi in 1..=frame {
        // eqs. (64) and (66)
        let ok: f64 = p_tau.iter().map(|p| (1.0 - p).powi(fi as i32) * (1.0 - p_noise).powi((frame - fi) as i32)).sum();
        total += 2.0 / m as f64 * eps * (taus.len() as f64 - ok);
    }
    Ok((total / frame as f64).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::css::Waveform;
    use com_core::rng::Rng;

    /// The former evaluation of eq. (25): both segments in Fresnel functions, four per
    /// symbol, $\int_{t_{1}}^{t_{2}} e^{j2\pi(a t + b t^{2})} dt$ by eq. (24).
    fn former_waveform_spectrum(m: usize, l: usize, f: f64) -> C64 {
        let segment = |a: f64, b: f64, t1: f64, t2: f64| {
            let r = b.sqrt();
            let shift = a / (2.0 * b);
            cis(-TAU * a * a / (4.0 * b)) / (2.0 * r)
                * (fresnel_k(2.0 * r * (t2 + shift)) - fresnel_k(2.0 * r * (t1 + shift)))
        };
        let (mf, lf) = (m as f64, l as f64);
        let b = 1.0 / (2.0 * mf);
        let fold = mf - lf;
        segment(lf / mf - 0.5 - f, b, 0.0, fold) + segment(lf / mf - 1.5 - f, b, fold, mf)
    }

    fn former_continuous_spectrum(m: usize, f: f64) -> f64 {
        let mf = m as f64;
        let x: Vec<C64> = (0..m).map(|l| former_waveform_spectrum(m, l, f)).collect();
        let sum: C64 = x.iter().sum();
        (x.iter().map(C64::norm_sqr).sum::<f64>() - sum.norm_sqr() / mf) / (mf * mf)
    }

    /// The former [`occupied_bandwidth`]: the spectrum by [`former_continuous_spectrum`],
    /// the lines by the sum over the symbols, every line tested in every cell.
    fn former_occupied_bandwidth(sf: u32, fraction: f64, span: f64) -> f64 {
        let m = 1usize << sf;
        let mf = m as f64;
        let step = 1.0 / (8.0 * mf);
        let n = (span / step).ceil() as usize;
        let g: Vec<f64> = (0..n).map(|i| former_continuous_spectrum(m, (i as f64 + 0.5) * step)).collect();
        let n_max = (span * mf).ceil() as i64;
        let lines: Vec<(f64, f64)> = (-n_max..=n_max)
            .map(|k| {
                let f = k as f64 / mf;
                let sum: C64 = (0..m).map(|l| former_waveform_spectrum(m, l, f)).sum();
                (f, sum.norm_sqr() / (mf * mf * mf * mf))
            })
            .collect();
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

    #[test]
    fn the_spectrum_is_the_former_evaluation_of_eq_25() {
        let mut rng = Rng::new(9);
        for sf in [1u32, 3, 5, 8] {
            let m = 1usize << sf;
            // generic frequencies, the midpoints of the grid and the lines
            let f: Vec<f64> = (0..40).map(|_| 3.0 * rng.uniform() - 1.5).collect();
            let peak = former_continuous_spectrum(m, 0.0).max(1e-3);
            for (fk, g) in f.iter().zip(continuous_spectrum(sf, &f).unwrap()) {
                let want = former_continuous_spectrum(m, *fk);
                assert!((g - want).abs() < 1e-12 * peak, "SF {sf}, f {fk}: {g} vs {want}");
            }
            let n = 8 * m + 5;
            let mid: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / (8.0 * m as f64)).collect();
            let direct = continuous_spectrum(sf, &mid).unwrap();
            for (a, b) in continuous_spectrum_grid(m, n).iter().zip(&direct) {
                assert!((a - b).abs() < 1e-12 * peak, "SF {sf}: {a} vs {b}");
            }
            for (fl, p) in spectral_lines(sf, 3 * m).unwrap() {
                let sum: C64 = (0..m).map(|l| former_waveform_spectrum(m, l, fl)).sum();
                let want = sum.norm_sqr() / (m as f64).powi(4);
                assert!((p - want).abs() < 1e-12 / m as f64, "SF {sf}, line {fl}: {p} vs {want}");
            }
        }
        for (sf, span) in [(3u32, 3.0), (5, 3.0), (7, 0.6)] {
            let (b, want) = (occupied_bandwidth(sf, 0.99, span).unwrap(), former_occupied_bandwidth(sf, 0.99, span));
            assert!((b - want).abs() < 1e-10, "SF {sf}: {b} vs {want}");
        }
    }

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
        let want = ser(7, -9.0).unwrap();
        assert!((sim / want - 1.0).abs() < 0.15, "{sim} vs {want}");
        assert!((ber(7, -9.0).unwrap() / want - 128.0 / 254.0).abs() < 1e-12);
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
            let want = cross_correlation(4, l, q).unwrap();
            assert!((c - want).norm() < 0.02, "C({l}, {q}) = {c} vs {want}");
        }
    }

    #[test]
    fn the_spectrum_holds_the_power_and_the_lines_hold_one_over_m() {
        // Chiani and Elzanaty 2019, Property 4 and eq. (20): SF 5
        let lines: f64 = spectral_lines(5, 40 << 5).unwrap().iter().map(|l| l.1).sum();
        assert!((lines - 1.0 / 32.0).abs() < 1e-9, "{lines}");
        let step = 1.0 / 256.0;
        let f: Vec<f64> = (0..(4.0 / step) as usize).map(|i| -2.0 + (i as f64 + 0.5) * step).collect();
        let cont: f64 = continuous_spectrum(5, &f).unwrap().iter().sum::<f64>() * step;
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
            let closed = interference_magnitudes(6, s1, s2, tau).unwrap();
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
            let c = max_cross_correlation(sf).unwrap();
            assert!((c - corr).abs() < 5e-4, "SF {sf}: {c}");
            let p = -com_math::lin2db(1.0 - c);
            assert!((p - penalty).abs() < 0.005, "SF {sf}: {p} dB");
        }
        for (sf, b99) in [(3u32, 1.500), (5, 1.185), (7, 1.045)] {
            let b = occupied_bandwidth(sf, 0.99, 3.0).unwrap();
            assert!((b / b99 - 1.0).abs() < 0.005, "SF {sf}: {b}");
            // the lines beyond 40 B hold less than 1e-4 of their power
            let lines: f64 = spectral_lines(sf, 40 << sf).unwrap().iter().map(|l| l.1).sum();
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
        assert!((ser(7, -8.0).unwrap() / 1.8e-3 - 1.0).abs() < 0.15);
        assert!((ser(7, -7.0).unwrap() / 1.4e-4 - 1.0).abs() < 0.1);
        for sf in 7u32..=12 {
            for snr in (-30..=0).map(f64::from) {
                let exact = ser(sf, snr).unwrap();
                if (1e-6..1e-2).contains(&exact) {
                    let (r, g) = (ser_rice(sf, snr).unwrap(), ser_gumbel(sf, snr).unwrap());
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
            let b = ber(7, snr).unwrap();
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
