//! A memoryless envelope nonlinearity driven by a complex Gaussian signal: the series of
//! its output autocorrelation and the spectrum of its distortion (Banelli & Cacopardi
//! 2000).
//!
//! # Model
//!
//! An OFDM signal of many subcarriers is close to a circular complex Gaussian process of
//! power $2\sigma^{2}$; an envelope nonlinearity $y = f(r) e^{j\theta}$ ($f$ the AM/AM and
//! AM/PM conversions, $f(r) = A(r) e^{j\Phi(r)}$) gives the output autocorrelation
//! (P. Banelli and S. Cacopardi, *IEEE Trans. Commun.* 48(3), 2000, eqs. 6–8)
//!
//! $$
//! R_{yy}(\tau) = \sum_{n=0}^{\infty} c_{n} \frac{R_{xx}(\tau)^{n+1}
//! R_{xx}^{*}(\tau)^{n}}{(2\sigma^{2})^{2n+1}}, \qquad
//! c_{n} = \frac{1}{2\sigma^{2}} \frac{1}{n+1} \left| \int_{0}^{\infty} f(r)
//! \frac{r^{2}}{\sigma^{2}} e^{-r^{2}/2\sigma^{2}} L_{n}^{(1)}\!\left(\frac{r^{2}}{2\sigma^{2}}
//! \right) dr \right|^{2},
//! $$
//!
//! with the generalised Laguerre polynomials $L_{n}^{(1)}$. The term $n = 0$ is the
//! attenuated signal, $c_{0} = |\alpha|^{2} 2\sigma^{2}$ with Bussgang's gain $\alpha$;
//! the term $n$ has the spectrum of the input convolved $2n + 1$ times with itself, so
//! the distortion spreads beyond the band and only part of it falls on the subcarriers
//! ([`distortion_spectrum`]). The coefficients sum to the output power $\mathbb{E}
//! |f(r)|^{2}$. [`output_series`] computes them for any $f$, by composite Simpson
//! quadrature with the Laguerre polynomials of every order at once, for an input of unit
//! power ($2\sigma^{2} = 1$).
//!
//! For the ideally predistorted amplifier, a soft envelope limiter $f(r) = \min(r, A)$ at
//! the input back-off $\gamma = A^{2}/2\sigma^{2}$, their eq. 28 gives
//! $c_{0} = 2\sigma^{2}\left[1 - e^{-\gamma} + \tfrac{1}{2}\sqrt{\pi\gamma}\,
//! \mathrm{erfc}\sqrt{\gamma}\right]^{2}$ and the output back-off is
//! $\gamma/(1 - e^{-\gamma})$ (their eq. 31 is printed $\gamma \cdot (1 - e^{-\gamma})$,
//! which would be below 0 dB for $\gamma < 1$). For $n \ge 2$ the integral equals their
//! eq. 28 with $+\sum_{i} d_{i,n}\gamma^{i+1}$ for the $d_{i,n}$ of their Table I (the
//! paper prints a minus: the signs of eq. 28 and Table I disagree).
//!
//! # Examples
//! ```
//! use com_hardware::gaussian_input::output_series;
//! use com_math::C64;
//! // a soft limiter at 3 dB of input back-off
//! let a = 2f64.sqrt();
//! let c = output_series(|r| C64::new(r.min(a), 0.0), 60);
//! let p_out: f64 = c.iter().sum();
//! assert!((p_out - (1.0 - (-2.0f64).exp())).abs() < 1e-4);
//! ```
//!
//! # References
//!
//! 1. P. Banelli and S. Cacopardi, "Theoretical analysis and performance of OFDM signals
//!    in nonlinear AWGN channels," *IEEE Trans. Commun.* 48(3), 430–441, 2000.

use com_math::C64;
use com_math::fft::{fft, ifft};

/// The coefficients $c_{0}, \dots, c_{N-1}$ of the output autocorrelation of the envelope
/// nonlinearity `f` (complex: AM/AM and AM/PM) for a circular complex Gaussian input of
/// unit power (see the module).
pub fn output_series(f: impl Fn(f64) -> C64, n_terms: usize) -> Vec<f64> {
    let values: Vec<C64> = series_grid().into_iter().map(f).collect();
    output_series_sampled(&values, n_terms)
}

/// The envelopes $r$ at which [`output_series_sampled`] takes the nonlinearity: the
/// 40 001 points of Simpson's rule on $[0, 7]$ ($e^{-49}$ beyond is negligible).
pub fn series_grid() -> Vec<f64> {
    (0..=SERIES_PANELS).map(|i| i as f64 * SERIES_MAX / SERIES_PANELS as f64).collect()
}

const SERIES_MAX: f64 = 7.0;
const SERIES_PANELS: usize = 40_000;

/// [`output_series`] from the values of $f$ on [`series_grid`] (values beyond the grid
/// are ignored, missing ones count as zero).
pub fn output_series_sampled(values: &[C64], n_terms: usize) -> Vec<f64> {
    let h = SERIES_MAX / SERIES_PANELS as f64;
    laguerre_projections(values, n_terms)
        .iter()
        .enumerate()
        .map(|(n, a)| (a * (h / 3.0)).norm_sqr() / (n as f64 + 1.0))
        .collect()
}

/// Bussgang's decomposition $y = K x + d$ of the envelope nonlinearity `f` (AM/AM and
/// AM/PM, $y = f(|x|) e^{j \arg x}$) driven by a circular complex Gaussian input of
/// power `power`: the complex gain $K = \mathbb{E}[y x^{*}] / \mathbb{E}|x|^{2}$ and the
/// power of the distortion $\sigma_{d}^{2} = \mathbb{E}|y|^{2} - |K|^{2}\, \mathbb{E}|x|^{2}$,
/// uncorrelated with the input, as `(K, σ_d²)`. $K$ is the projection of [`output_series`]
/// on $L_{0}^{(1)}$ (the term $c_{0} = |K|^{2}$ at unit power), equivalently
/// $K = \frac{1}{2}\mathbb{E}\left[f'(\rho) + f(\rho)/\rho\right]$ (Banelli & Cacopardi
/// 2000; Bouhadda et al. 2014, eqs. (20)–(21)); the same quadrature, to about
/// $10^{-8}$ (a kink of $f$, as that of a limiter, limits Simpson's rule).
///
/// # Examples
/// ```
/// use com_hardware::gaussian_input::bussgang;
/// use com_math::C64;
/// // a soft limiter at 3 dB of input back-off: K = 1 - e^{-γ} + ½ √(πγ) erfc √γ
/// let (k, d) = bussgang(|r| C64::new(r.min(1.0), 0.0), 0.5);
/// let g = 2.0f64;
/// let want = 1.0 - (-g).exp() + 0.5 * (std::f64::consts::PI * g).sqrt() * com_math::erfc(g.sqrt());
/// assert!((k.re - want).abs() < 1e-6 && k.im.abs() < 1e-15); // erfc: 1.2e-7
/// // σ_d² = P (1 - e^{-γ} - K²)
/// assert!((d - 0.5 * (1.0 - (-g).exp() - want * want)).abs() < 1e-6);
/// ```
pub fn bussgang(f: impl Fn(f64) -> C64, power: f64) -> (C64, f64) {
    let scale = power.sqrt();
    // the nonlinearity seen by an input of unit power
    let g = |u: f64| f(u * scale) / scale;
    let values: Vec<C64> = series_grid().into_iter().map(g).collect();
    let h = SERIES_MAX / SERIES_PANELS as f64;
    let gain = laguerre_projections(&values, 1)[0] * (h / 3.0);
    let out = com_math::simpson(|u| g(u).norm_sqr() * 2.0 * u * (-u * u).exp(), 0.0, SERIES_MAX, SERIES_PANELS);
    (gain, power * (out - gain.norm_sqr()))
}

/// The projections $\int_{0}^{\infty} f(r)\, 2r^{2} e^{-r^{2}} L_{n}^{(1)}(r^{2})\, dr$,
/// $n = 0, \dots, N - 1$, of the samples of $f$ on [`series_grid`], before the factor $h/3$
/// of Simpson's rule.
fn laguerre_projections(values: &[C64], n_terms: usize) -> Vec<C64> {
    let (panels, h) = (SERIES_PANELS, SERIES_MAX / SERIES_PANELS as f64);
    let mut acc = vec![C64::new(0.0, 0.0); n_terms];
    let mut lag = vec![0.0; n_terms];
    for (i, fv) in values.iter().take(panels + 1).enumerate() {
        let r = i as f64 * h;
        let x = r * r;
        let w = if i == 0 || i == panels {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let common = fv * (2.0 * x * (-x).exp() * w);
        // L_n^{(1)}(x): L_0 = 1, L_1 = 2 - x, (k+1) L_{k+1} = (2k + 2 - x) L_k - (k + 1) L_{k-1}
        if n_terms > 0 {
            lag[0] = 1.0;
        }
        if n_terms > 1 {
            lag[1] = 2.0 - x;
        }
        for k in 1..n_terms.saturating_sub(1) {
            let kf = k as f64;
            lag[k + 1] = ((2.0 * kf + 2.0 - x) * lag[k] - (kf + 1.0) * lag[k - 1]) / (kf + 1.0);
        }
        for (a, l) in acc.iter_mut().zip(&lag) {
            *a += common * *l;
        }
    }
    acc
}

/// The power on each of `bins` frequency bins (a circular grid) of the useful term and
/// of the distortion terms of `c` ([`output_series`]), for an input whose spectrum is flat
/// on the `occupied` bins centred on bin 0 (an OFDM signal of `occupied` subcarriers,
/// oversampled to `bins`): `(useful, distortion)`, each summing to its power.
pub fn distortion_spectrum(c: &[f64], occupied: usize, bins: usize) -> (Vec<f64>, Vec<f64>) {
    let mut p = vec![C64::new(0.0, 0.0); bins];
    for i in 0..occupied {
        let k = (i + bins - occupied / 2) % bins;
        p[k] = C64::new(1.0 / occupied as f64, 0.0);
    }
    let pf = fft(&p);
    let useful: Vec<f64> = p.iter().map(|v| c.first().copied().unwrap_or(0.0) * v.re).collect();
    let mut dist = vec![0.0; bins];
    for (n, cn) in c.iter().enumerate().skip(1) {
        // the (2n + 1)-fold convolution of the flat spectrum, by FFT
        let conv = ifft(&pf.iter().map(|v| v.powu(2 * n as u32 + 1)).collect::<Vec<_>>());
        for (d, v) in dist.iter_mut().zip(conv) {
            *d += cn * v.re.max(0.0);
        }
    }
    (useful, dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_linear_device_has_one_term() {
        let c = output_series(|r| C64::new(2.0 * r, 0.0), 6);
        assert!((c[0] - 4.0).abs() < 1e-9 && c[1..].iter().all(|v| v.abs() < 1e-9), "{c:?}");
        let (u, d) = distortion_spectrum(&[1.0, 0.0], 8, 32);
        assert!((u.iter().sum::<f64>() - 1.0).abs() < 1e-12 && d.iter().all(|v| v.abs() < 1e-15));
    }
}

#[cfg(test)]
mod literature {
    use super::*;
    use crate::amplifier::{Rapp, Saleh};

    /// H. Bouhadda, H. Shaiek, D. Roviras, R. Zayani, Y. Medjahdi and R. Bouallegue,
    /// "Theoretical analysis of BER performance of nonlinearly amplified FBMC/OQAM and
    /// OFDM signals," EURASIP J. Adv. Signal Process. 2014:60, Tables 2 and 3: Bussgang's
    /// gain and the distortion power of the SSPA (Rapp, $v = 1$) and of the TWTA (Saleh
    /// with $A_{\mathrm{sat}} = 1$, $\phi_{0} = 0$ and $\pi/6$) at input back-offs of 4, 6
    /// and 8 dB from the saturation, against their "simulation" columns (10⁶ symbols):
    /// the gains within 2·10⁻³, the distortion powers within 1.5 %. Their "theoretical"
    /// columns, from a polynomial fit of order 10 (eqs. (33)–(34)), are further from the
    /// exact values (0.5904 + 0.1068j against 0.5917 + 0.1106j at 4 dB).
    #[test]
    fn bouhadda_2014_tables_2_3() {
        let sspa = Rapp { gain: 1.0, saturation: 1.0, smoothness: 1.0 };
        let twta = |phi0: f64| Saleh { alpha_a: 1.0, beta_a: 1.0, alpha_p: phi0, beta_p: 1.0 };
        let rows: [(&str, f64, f64, C64, f64); 9] = [
            ("SSPA", 4.0, 0.0, C64::new(0.7699, 0.0), 4.4759e-3),
            ("SSPA", 6.0, 0.0, C64::new(0.8307, 0.0), 1.8950e-3),
            ("SSPA", 8.0, 0.0, C64::new(0.8798, 0.0), 7.0742e-4),
            ("TWTA", 4.0, 0.0, C64::new(0.6042, 0.0), 1.0330e-2),
            ("TWTA", 6.0, 0.0, C64::new(0.6976, 0.0), 4.9589e-3),
            ("TWTA", 8.0, 0.0, C64::new(0.7784, 0.0), 2.1228e-3),
            ("TWTA", 4.0, std::f64::consts::FRAC_PI_6, C64::new(0.5917, 0.1106), 1.1339e-2),
            ("TWTA", 6.0, std::f64::consts::FRAC_PI_6, C64::new(0.6887, 0.0995), 5.5901e-3),
            ("TWTA", 8.0, std::f64::consts::FRAC_PI_6, C64::new(0.7727, 0.0830), 2.4194e-3),
        ];
        for (model, ibo, phi0, k_table, d_table) in rows {
            // IBO = A_sat² / σ² with A_sat = 1 (eq. (14))
            let power = 10f64.powf(-ibo / 10.0);
            let (k, d) = if model == "SSPA" {
                bussgang(|r| C64::new(sspa.am_am(r), 0.0), power)
            } else {
                let t = twta(phi0);
                bussgang(|r| C64::from_polar(t.am_am(r), t.am_pm(r)), power)
            };
            assert!((k - k_table).norm() < 2e-3, "{model} {ibo} dB: K = {k}");
            assert!((d / d_table - 1.0).abs() < 0.015, "{model} {ibo} dB: σ_d² = {d}");
        }
    }
}
