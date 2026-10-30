//! Chromatic-dispersion compensation.
//!
//! The dispersion of a length $L$ of fibre is the all-pass response
//! $e^{j \beta_{2} \omega^{2} L / 2}$; its compensation multiplies by the conjugate.
//! Nothing here is specific to optics but the name of the parameter: the blocks take
//! the group-velocity dispersion `β₂` (s²/km) and the length (km).
//!
//! - [`cdc`]: exact compensation on a periodic block, `X(ω) e^{-j β₂ ω² L/2}`;
//! - [`cd_fir_taps`]: an FIR of `n` taps by frequency sampling of the same response
//!   (inverse DFT of `e^{-jβ₂ω_k²L/2}` on `n` bins, centred), applied by
//!   [`crate::filter::fir_filter_complex`]; [`cd_taps_needed`] gives the length rule
//!   of OptiCommPy's `edc`, `2⌈6.67 |β₂| L R_s² (f_s/R_s)⌉`;
//! - [`cd_savory_taps`]: the truncated impulse response of Savory (2008), of the length
//!   [`cd_savory_length`];
//! - [`cd_ls_taps`]: the optimal FIR in the least-squares sense of Eghbali et al.
//!   (2014), over the whole band or over the band of the signal, and [`cd_ls_error`],
//!   the energy of its error.
//!
//! # FIR design
//!
//! In the digital frequency $\omega \in [-\pi, \pi]$ (radians per sample), the
//! compensation is $H_{\mathrm{des}}(e^{j\omega}) = e^{jK\omega^{2}}$ with
//! $K = -\beta_{2} L f_{s}^{2}/2$ ([`cd_dispersion_factor`]; $K = D\lambda^{2}z/(4\pi c
//! T^{2})$ in the notation of the papers, positive for standard fibre). Savory's filter
//! samples the impulse response of the analog compensation,
//!
//! $$
//! h(n) = \sqrt{\frac{j}{4K\pi}}\, e^{-j n^{2}/(4K)}, \qquad |n| \le \lfloor N/2 \rfloor,
//! \qquad N = 2\lfloor 2\pi K \rfloor + 1,
//! $$
//!
//! the length beyond which the response would alias. The least-squares filter of
//! $N_{c}$ taps (odd), centred, $H(e^{j\omega}) = \sum_{n} h(n) e^{-jn\omega}$, minimises
//! the energy of the error over the band $[-\Omega, \Omega]$,
//!
//! $$
//! E = \frac{1}{2\pi} \int_{-\Omega}^{\Omega} \left| H(e^{j\omega}) - e^{jK\omega^{2}}
//! \right|^{2} d\omega,
//! $$
//!
//! by $\hat{\mathbf{h}} = (\mathbf{Q} + \epsilon \mathbf{I})^{-1} \mathbf{D}$ with the
//! Toeplitz $Q(n, m) = \sin((n - m)\Omega)/(\pi (n - m))$ ($\Omega/\pi$ on the diagonal)
//! and
//!
//! $$
//! D(n) = \frac{1}{2\pi} \int_{-\Omega}^{\Omega} e^{j(K\omega^{2} + n\omega)} d\omega
//! = \frac{e^{-jn^{2}/(4K)}}{2\pi\sqrt{K}} \left[ F\!\left(\sqrt{K}\left(\Omega +
//! \tfrac{n}{2K}\right)\right) - F\!\left(\sqrt{K}\left(-\Omega + \tfrac{n}{2K}\right)
//! \right) \right],
//! $$
//!
//! $F(u) = \int_{0}^{u} e^{jv^{2}} dv = \sqrt{\pi/2}\, [C + jS](u\sqrt{2/\pi})$ with the
//! Fresnel integrals [`com_math::fresnel`] (eq. 13 of the paper writes it with the error
//! function of a complex argument). Over the whole band ($\Omega = \pi$) $\mathbf{Q} =
//! \mathbf{I}$ and $\hat{h}(n) = D(n)$ (eq. 19). Over the band of a signal shaped by a
//! roll-off $\rho$ at $L$ samples per symbol, $\Omega = \pi(1 + \rho)/L$: $\mathbf{Q}$
//! is then ill-conditioned and the penalty $\epsilon$ (the paper uses $10^{-14}$) bounds
//! the energy of the taps. The response outside the band is not constrained: it reaches
//! $10^{5}$ for 119 taps at $K = 19.9$, so the receive filter must reject that band
//! (the paper assumes ideal pulse-shaping filters). Since $|H_{\mathrm{des}}| = 1$,
//! $E = \mathbf{h}^{H}\mathbf{Q}\mathbf{h} - 2\,\mathrm{Re}(\mathbf{h}^{H}\mathbf{D}) +
//! \Omega/\pi$ in closed form ([`cd_ls_error`]), accurate down to about $10^{-15}$
//! ($-150$ dB). For $K < 0$ every filter is the conjugate of the one of $|K|$.
//!
//! # Examples
//! ```
//! use com_dsp::compensation::cd::cdc;
//! use com_math::C64;
//! let b2 = -2.168e-23;                              // SSMF at 1550 nm (s²/km)
//! let x: Vec<C64> = (0..256).map(|n| C64::new((n % 5) as f64, 0.0)).collect();
//! // dispersion of 80 km (a compensation with the opposite sign), then compensation
//! let y = cdc(&cdc(&x, -b2, 80.0, 64e9), b2, 80.0, 64e9);
//! assert!(x.iter().zip(&y).all(|(a, b)| (a - b).norm() < 1e-9));
//! ```
//!
//! # References
//!
//! 1. S. J. Savory, "Digital filters for coherent optical receivers," *Opt. Express*
//!    16(2), 804–817, 2008 (eqs. 9–10).
//! 2. A. Eghbali, H. Johansson, O. Gustafsson, S. J. Savory, "Optimal least-squares FIR
//!    digital filters for compensation of chromatic dispersion in digital coherent
//!    optical receivers," *J. Lightw. Technol.* 32(8), 1449–1456, 2014.

use com_core::{Result, ensure};
use com_math::fft;
use com_math::linalg::solve_hpd;
use com_math::nlse::Medium;
use com_math::{C64, cis, fresnel};
use ndarray::Array2;
use std::f64::consts::PI;

/// Compensate the dispersion `beta2_s2_km` of `length_km` on a block sampled at `fs`.
pub fn cdc(x: &[C64], beta2_s2_km: f64, length_km: f64, fs: f64) -> Vec<C64> {
    fft::filter(x, inverse_dispersion(x.len(), beta2_s2_km, length_km, fs))
}

/// The inverse dispersion `e^{-jβ₂ω²L/2}` on the `n` angular frequencies of a block
/// sampled at `fs`: the linear response of a lossless medium of dispersion `-β₂`.
fn inverse_dispersion(n: usize, beta2_s2_km: f64, length_km: f64, fs: f64) -> Vec<C64> {
    let inverse = Medium { alpha_np_km: 0.0, beta2_s2_km: -beta2_s2_km, gamma_w_km: 0.0 };
    inverse.linear_response(&fft::angular_frequencies(n, fs), length_km)
}

/// Number of taps of the dispersion-compensating FIR (OptiCommPy `edc` rule).
pub fn cd_taps_needed(beta2_s2_km: f64, length_km: f64, symbol_rate: f64, fs: f64) -> usize {
    let b2 = beta2_s2_km.abs();
    (2.0 * (6.67 * b2 * length_km * symbol_rate * symbol_rate * (fs / symbol_rate)).ceil()) as usize
}

/// `n` centred taps of the dispersion compensation of `length_km` by frequency sampling.
pub fn cd_fir_taps(beta2_s2_km: f64, length_km: f64, fs: f64, n: usize) -> Vec<C64> {
    let t = fft::ifft(&inverse_dispersion(n, beta2_s2_km, length_km, fs));
    // centre: tap i of the FIR is the impulse response at time i - ⌊n/2⌋
    fft::fftshift(&t)
}

/// The dispersion factor $K = -\beta_{2} L f_{s}^{2}/2$ of the digital compensation
/// $e^{jK\omega^{2}}$ (see the module).
pub fn cd_dispersion_factor(beta2_s2_km: f64, length_km: f64, fs: f64) -> f64 {
    -beta2_s2_km * length_km * fs * fs / 2.0
}

/// The length $N = 2\lfloor 2\pi |K| \rfloor + 1$ of Savory's filter (see the module).
pub fn cd_savory_length(beta2_s2_km: f64, length_km: f64, fs: f64) -> usize {
    2 * (2.0 * PI * cd_dispersion_factor(beta2_s2_km, length_km, fs).abs()).floor() as usize + 1
}

/// The `n` centred taps of Savory's filter $\sqrt{j/(4K\pi)}\, e^{-jn^{2}/(4K)}$ (see the
/// module; `n` is usually [`cd_savory_length`]).
///
/// # Errors
/// [`com_core::Error::Config`] for an even `n` or no dispersion.
pub fn cd_savory_taps(beta2_s2_km: f64, length_km: f64, fs: f64, n: usize) -> Result<Vec<C64>> {
    let k = cd_dispersion_factor(beta2_s2_km, length_km, fs);
    ensure(n % 2 == 1, || format!("the filter needs an odd number of taps, got {n}"))?;
    ensure(k != 0.0, || "the filter needs a non-zero dispersion".to_string())?;
    let (ka, half) = (k.abs(), (n / 2) as f64);
    let gain = (C64::new(0.0, 1.0) / (4.0 * ka * PI)).sqrt();
    Ok((0..n).map(|i| i as f64 - half).map(|m| conj_if(gain * cis(-m * m / (4.0 * ka)), k < 0.0)).collect())
}

/// The `n` centred taps of the least-squares compensation over the band $|f| \le$
/// `band_hz` (half the bandwidth; `fs / 2` for the whole band), with the penalty
/// `epsilon` (Eghbali et al. 2014, see the module).
///
/// # Errors
/// [`com_core::Error::Config`] for an even `n`, no dispersion, a band outside
/// $(0, f_{s}/2]$ or a negative penalty; [`com_core::Error::Numerical`] when
/// $\mathbf{Q} + \epsilon\mathbf{I}$ is not positive definite to the machine precision
/// (raise `epsilon`).
pub fn cd_ls_taps(beta2_s2_km: f64, length_km: f64, fs: f64, n: usize, band_hz: f64, epsilon: f64) -> Result<Vec<C64>> {
    let k = cd_dispersion_factor(beta2_s2_km, length_km, fs);
    ensure(n % 2 == 1, || format!("the filter needs an odd number of taps, got {n}"))?;
    ensure(k != 0.0, || "the filter needs a non-zero dispersion".to_string())?;
    ensure(band_hz > 0.0 && band_hz <= fs / 2.0, || format!("the band {band_hz} Hz must lie in (0, fs/2]"))?;
    ensure(epsilon >= 0.0, || format!("the penalty must be non-negative, got {epsilon}"))?;
    let omega = 2.0 * PI * band_hz / fs;
    let d = ls_target(k.abs(), n, omega);
    let h = if band_hz == fs / 2.0 && epsilon == 0.0 {
        // Q = I (eq. 19)
        d
    } else {
        let q = toeplitz_band(n, omega);
        let a = Array2::from_shape_fn((n, n), |(i, j)| {
            C64::new(q[i.abs_diff(j)] + if i == j { epsilon } else { 0.0 }, 0.0)
        });
        let b = Array2::from_shape_fn((n, 1), |(i, _)| d[i]);
        solve_hpd(&a.view(), &b.view())?.column(0).to_vec()
    };
    Ok(h.into_iter().map(|v| conj_if(v, k < 0.0)).collect())
}

/// The energy of the error $E$ of the centred `taps` (odd in number) against the
/// compensation of `length_km`, over the band $|f| \le$ `band_hz` (see the module).
///
/// # Errors
/// [`com_core::Error::Config`] as [`cd_ls_taps`].
pub fn cd_ls_error(taps: &[C64], beta2_s2_km: f64, length_km: f64, fs: f64, band_hz: f64) -> Result<f64> {
    let (n, k) = (taps.len(), cd_dispersion_factor(beta2_s2_km, length_km, fs));
    ensure(n % 2 == 1, || format!("the filter needs an odd number of taps, got {n}"))?;
    ensure(k != 0.0, || "the filter needs a non-zero dispersion".to_string())?;
    ensure(band_hz > 0.0 && band_hz <= fs / 2.0, || format!("the band {band_hz} Hz must lie in (0, fs/2]"))?;
    let omega = 2.0 * PI * band_hz / fs;
    let h: Vec<C64> = taps.iter().map(|v| conj_if(*v, k < 0.0)).collect();
    let (q, d) = (toeplitz_band(n, omega), ls_target(k.abs(), n, omega));
    let mut e = omega / PI;
    for i in 0..n {
        let qh: C64 = (0..n).map(|j| h[j] * q[i.abs_diff(j)]).sum();
        e += (h[i].conj() * qh).re - 2.0 * (h[i].conj() * d[i]).re;
    }
    Ok(e.max(0.0))
}

fn conj_if(v: C64, c: bool) -> C64 {
    if c { v.conj() } else { v }
}

/// The first row of $\mathbf{Q}$: $\sin(m\Omega)/(\pi m)$, $\Omega/\pi$ at $m = 0$.
fn toeplitz_band(n: usize, omega: f64) -> Vec<f64> {
    (0..n).map(|m| if m == 0 { omega / PI } else { (m as f64 * omega).sin() / (PI * m as f64) }).collect()
}

/// $D(n)$ for $n = -\lfloor N/2 \rfloor, \dots, \lfloor N/2 \rfloor$ and $K > 0$ (see the
/// module).
fn ls_target(k: f64, n: usize, omega: f64) -> Vec<C64> {
    let f = |u: f64| {
        let (c, s) = fresnel(u * (2.0 / PI).sqrt());
        C64::new(c, s) * (PI / 2.0).sqrt()
    };
    let sk = k.sqrt();
    let half = (n / 2) as f64;
    (0..n)
        .map(|i| {
            let m = i as f64 - half;
            let b = m / (2.0 * k);
            cis(-m * m / (4.0 * k)) / (2.0 * PI * sk) * (f(sk * (omega + b)) - f(sk * (-omega + b)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::fir_filter_complex;
    use com_core::rng::Rng;

    #[test]
    fn cdc_and_its_fir_invert_the_dispersion() {
        // SSMF: D = 17 ps/(nm km) at 193.1 THz
        let b2 = -2.1749e-23;
        let mut r = Rng::new(1);
        let n = 4096;
        let fs = 64e9;
        let x: Vec<C64> = (0..n).map(|_| r.complex_gaussian()).collect();
        // band-limit to ±fs/4
        let mut xs = fft::fft(&x);
        for (v, fk) in xs.iter_mut().zip(fft::fftfreq(n, 1.0)) {
            if fk.abs() > 0.25 {
                *v = C64::new(0.0, 0.0);
            }
        }
        let x = fft::ifft(&xs);
        // the lossless linear channel of 20 km is the compensation of the opposite β₂
        let y = cdc(&x, -b2, 20.0, fs);
        let z = cdc(&y, b2, 20.0, fs);
        assert!(x.iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-9));
        let taps = cd_fir_taps(b2, 20.0, fs, 255);
        let zf = fir_filter_complex(&taps, &y);
        let mid = &zf[512..n - 512];
        let err: f64 = mid.iter().zip(&x[512..n - 512]).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>()
            / x[512..n - 512].iter().map(C64::norm_sqr).sum::<f64>();
        assert!(err < 1e-3, "relative error {err}");
        // 2⌈6.67 |β₂| L R_s² (f_s/R_s)⌉ = 2⌈5.92⌉ for 20 km of SSMF at 32 GBd, 2 sps
        assert_eq!(cd_taps_needed(b2, 20.0, 32e9, fs), 12);
    }

    /// $E$ by a fine trapezoidal rule.
    fn error_by_quadrature(taps: &[C64], k: f64, omega: f64) -> f64 {
        let m = 200_000;
        let half = (taps.len() / 2) as f64;
        (0..=m)
            .map(|i| {
                let w = -omega + 2.0 * omega * i as f64 / m as f64;
                let h: C64 = taps.iter().enumerate().map(|(t, v)| v * cis(-(t as f64 - half) * w)).sum();
                let wt = if i == 0 || i == m { 0.5 } else { 1.0 };
                wt * (h - cis(k * w * w)).norm_sqr()
            })
            .sum::<f64>()
            * (2.0 * omega / m as f64)
            / (2.0 * PI)
    }

    #[test]
    fn the_least_squares_filter_beats_savory_and_its_error_is_exact() {
        // K = 3 (b2 L fs² = -6), 37 taps, whole band and band of a roll-off 0.25 at 2 sps
        let (b2, len, fs) = (-6.0, 1.0, 1.0);
        let k = cd_dispersion_factor(b2, len, fs);
        assert_eq!(cd_savory_length(b2, len, fs), 37);
        let sav = cd_savory_taps(b2, len, fs, 37).unwrap();
        for (band, eps) in [(0.5, 0.0), (0.3125, 1e-12)] {
            let ls = cd_ls_taps(b2, len, fs, 37, band, eps).unwrap();
            let omega = 2.0 * PI * band;
            let (el, es) =
                (cd_ls_error(&ls, b2, len, fs, band).unwrap(), cd_ls_error(&sav, b2, len, fs, band).unwrap());
            assert!(el < es, "band {band}: LS {el}, Savory {es}");
            for (taps, e) in [(&ls, el), (&sav, es)] {
                let eq = error_by_quadrature(taps, k, omega);
                assert!((e - eq).abs() < 1e-9 + 1e-6 * eq, "band {band}: {e} vs {eq}");
            }
        }
        // the opposite dispersion: the conjugate filters
        let a = cd_ls_taps(b2, len, fs, 37, 0.3, 1e-12).unwrap();
        let b = cd_ls_taps(-b2, len, fs, 37, 0.3, 1e-12).unwrap();
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y.conj()).norm() < 1e-12));
        assert!(cd_ls_taps(b2, len, fs, 36, 0.5, 0.0).is_err());
        assert!(cd_ls_taps(b2, len, fs, 37, 0.6, 0.0).is_err());
    }
}
