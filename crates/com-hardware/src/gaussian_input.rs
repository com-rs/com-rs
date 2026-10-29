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
    acc.iter().enumerate().map(|(n, a)| (a * (h / 3.0)).norm_sqr() / (n as f64 + 1.0)).collect()
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
