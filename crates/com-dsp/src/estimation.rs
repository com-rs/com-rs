//! Waveform-agnostic channel estimation: least squares and linear MMSE.
//!
//! For known symbols $x_{i}$ received as $y_{i} = h_{i} x_{i} + w_{i}$,
//! $w_{i} \sim \mathcal{CN}(0, \sigma^{2})$:
//!
//! - least squares (LS): $\hat h_{i} = y_{i} / x_{i}$, unbiased, of error variance
//!   $\sigma^{2}/|x_{i}|^{2}$ ([`ls`]);
//! - linear MMSE: from the LS estimates $\hat h_{p}$ at the pilot positions $p$, the
//!   channel at the positions $q$ is
//!
//! $$
//! \hat h_{q} = R_{qp} \left(R_{pp} + \frac{\sigma^{2}}{E_{p}} I\right)^{-1} \hat h_{p},
//! $$
//!
//! with the channel correlations $R$ and the pilot energy $E_{p}$ ([`lmmse`]), the
//! estimator of minimum mean square error among the linear ones (van de Beek et al.
//! 1995; Edfors et al. 1998); its error covariance is
//! $R_{qq} - R_{qp} (R_{pp} + \frac{\sigma^{2}}{E_{p}} I)^{-1} R_{pq}$ ([`lmmse_mse`]);
//! - interpolation between known positions: [`linear_interpolation`] (constant beyond
//!   the ends), [`quadratic_interpolation`] (through the three nearest positions, the
//!   "second-order" interpolation of Coleri et al. 2002, eq. (19)),
//!   [`spline_interpolation`] (cubic spline with not-a-knot ends, MATLAB `spline`),
//!   [`lowpass_interpolation`] (the band-limited interpolator of MATLAB `interp`), and
//!   `com_dsp::resample::resample_fft` for the "time-domain" interpolation of
//!   uniformly spaced values of a periodic block (zero padding of the IDFT).
//!
//! # Examples
//! ```
//! use com_dsp::estimation::ls;
//! use com_math::C64;
//! let h = ls(&[C64::new(0.0, 2.0)], &[C64::new(1.0, 1.0)]);
//! assert!((h[0] - C64::new(1.0, 1.0)).norm() < 1e-12);
//! ```
//!
//! # References
//!
//! 1. J.-J. van de Beek, O. Edfors, M. Sandell, S. K. Wilson and P. O. Börjesson, "On
//!    channel estimation in OFDM systems," *Proc. IEEE VTC*, 815–819, 1995.
//! 2. O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson and P. O. Börjesson, "OFDM
//!    channel estimation by singular value decomposition," *IEEE Trans. Commun.*, 46(7),
//!    931–939, 1998.
//! 3. S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*,
//!    Prentice Hall, 1993, ch. 12.
//! 4. S. Coleri, M. Ergen, A. Puri and A. Bahai, "Channel estimation techniques based on
//!    pilot arrangement in OFDM systems," *IEEE Trans. Broadcast.*, 48(3), 223–229, 2002.
//! 5. G. Oetken, T. W. Parks and H. W. Schüssler, "New results in the design of digital
//!    interpolators," *IEEE Trans. Acoust., Speech, Signal Process.*, 23(3), 301–309,
//!    1975.
//! 6. C. de Boor, *A Practical Guide to Splines*, Springer, 1978, ch. IV (not-a-knot).

use com_core::{Error, Result, ensure, ensure_shape};
use com_math::C64;
use com_math::linalg::{CMat, hermitian, identity, solve_hpd};

/// Least-squares estimates $y_{i}/x_{i}$.
pub fn ls(y: &[C64], x: &[C64]) -> Vec<C64> {
    y.iter().zip(x).map(|(a, b)| a / b).collect()
}

fn regularized(r_pp: &CMat, noise_to_pilot: f64) -> CMat {
    r_pp + &identity(r_pp.nrows()).mapv(|v| v * noise_to_pilot)
}

/// The linear MMSE estimates at the positions of `r_qp` (rows) from the LS estimates
/// `h_ls` at the pilots, with `noise_to_pilot` = $\sigma^{2}/E_{p}$.
///
/// # Errors
/// [`Error::Shape`] for inconsistent sizes, [`Error::Numerical`] for a singular system.
pub fn lmmse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, h_ls: &[C64]) -> Result<Vec<C64>> {
    ensure(noise_to_pilot >= 0.0, || "negative noise".into())?;
    ensure_shape(r_pp.nrows() == h_ls.len() && r_qp.ncols() == h_ls.len(), || {
        format!("{} pilots for correlations of {} pilots", h_ls.len(), r_pp.nrows())
    })?;
    let b = ndarray::Array2::from_shape_vec((h_ls.len(), 1), h_ls.to_vec()).map_err(|e| Error::Shape(e.to_string()))?;
    let z = solve_hpd(&regularized(r_pp, noise_to_pilot).view(), &b.view())?;
    Ok(r_qp.dot(&z).column(0).to_vec())
}

/// The mean square error of every LMMSE estimate (the diagonal of its error covariance),
/// given the variances `r_qq_diag` of the channel at the estimated positions.
///
/// # Errors
/// As [`lmmse`].
pub fn lmmse_mse(r_qp: &CMat, r_pp: &CMat, noise_to_pilot: f64, r_qq_diag: &[f64]) -> Result<Vec<f64>> {
    let rpq = hermitian(&r_qp.view());
    let z = solve_hpd(&regularized(r_pp, noise_to_pilot).view(), &rpq.view())?;
    let g = r_qp.dot(&z);
    Ok(r_qq_diag.iter().enumerate().map(|(i, v)| (v - g[[i, i]].re).max(0.0)).collect())
}

/// Linear interpolation of `values` known at the increasing `positions` onto
/// `0..n` (constant beyond the first and last positions).
///
/// # Panics
/// If `positions` is empty or of another length than `values`.
pub fn linear_interpolation(positions: &[usize], values: &[C64], n: usize) -> Vec<C64> {
    assert!(!positions.is_empty() && positions.len() == values.len(), "one value per known position");
    let mut out = Vec::with_capacity(n);
    let mut j = 0;
    for i in 0..n {
        while j + 1 < positions.len() && positions[j + 1] <= i {
            j += 1;
        }
        let v = if i <= positions[0] {
            values[0]
        } else if j + 1 >= positions.len() {
            values[positions.len() - 1]
        } else {
            let (a, b) = (positions[j], positions[j + 1]);
            let t = (i - a) as f64 / (b - a) as f64;
            values[j] * (1.0 - t) + values[j + 1] * t
        };
        out.push(v);
    }
    out
}

/// The index of the interval of `positions` containing `x`: the largest `j` with
/// `positions[j] <= x`, clamped to `0..=len - 2` (the end intervals extend beyond the
/// ends).
fn interval(positions: &[usize], x: f64) -> usize {
    let j = positions.partition_point(|&p| p as f64 <= x);
    j.saturating_sub(1).min(positions.len().saturating_sub(2))
}

/// Quadratic (second-order) interpolation of `values` known at the increasing
/// `positions` onto `0..n`: the parabola through the three known values around each
/// point, the left neighbour `j` of the point with `j - 1` and `j + 1` (`0, 1, 2` and the
/// last three at the ends, extrapolated beyond them). With uniform spacing $L$ and the
/// point $jL + l$, $\alpha = l/L$, it is eq. (19) of Coleri et al. (2002), whose
/// "$\alpha = l/N$" is a misprint:
///
/// $$
/// \hat{H} = \frac{\alpha(\alpha - 1)}{2} H_{j-1} - (\alpha - 1)(\alpha + 1) H_{j} +
/// \frac{\alpha(\alpha + 1)}{2} H_{j+1} .
/// $$
///
/// Fewer than three positions fall back to [`linear_interpolation`].
///
/// # Panics
/// If `positions` is empty or of another length than `values`.
///
/// # Examples
/// ```
/// use com_dsp::estimation::quadratic_interpolation;
/// use com_math::C64;
/// // exact on a parabola
/// let v: Vec<C64> = [0usize, 4, 8, 12].iter().map(|&k| C64::new((k * k) as f64, 1.0)).collect();
/// let h = quadratic_interpolation(&[0, 4, 8, 12], &v, 16);
/// assert!(h.iter().enumerate().all(|(k, x)| (x - C64::new((k * k) as f64, 1.0)).norm() < 1e-9));
/// ```
pub fn quadratic_interpolation(positions: &[usize], values: &[C64], n: usize) -> Vec<C64> {
    assert!(!positions.is_empty() && positions.len() == values.len(), "one value per known position");
    if positions.len() < 3 {
        return linear_interpolation(positions, values, n);
    }
    (0..n)
        .map(|k| {
            let x = k as f64;
            let j = interval(positions, x).clamp(1, positions.len() - 2);
            let (a, b, c) = (positions[j - 1] as f64, positions[j] as f64, positions[j + 1] as f64);
            values[j - 1] * ((x - b) * (x - c) / ((a - b) * (a - c)))
                + values[j] * ((x - a) * (x - c) / ((b - a) * (b - c)))
                + values[j + 1] * ((x - a) * (x - b) / ((c - a) * (c - b)))
        })
        .collect()
}

/// Cubic-spline interpolation of `values` known at the increasing `positions` onto
/// `0..n`, with the not-a-knot end conditions (the third derivative is continuous at
/// the second and the next-to-last positions; de Boor 1978), as MATLAB `spline`: the
/// end polynomials extrapolate beyond the ends. Three positions give the parabola, two
/// the line, one a constant.
///
/// # Panics
/// If `positions` is empty or of another length than `values`.
///
/// # Examples
/// ```
/// use com_dsp::estimation::spline_interpolation;
/// use com_math::C64;
/// // not-a-knot reproduces cubics exactly
/// let f = |k: f64| C64::new(k * k * k - 2.0 * k, -k);
/// let pos = [0usize, 3, 5, 9, 12];
/// let v: Vec<C64> = pos.iter().map(|&k| f(k as f64)).collect();
/// let h = spline_interpolation(&pos, &v, 14);
/// assert!(h.iter().enumerate().all(|(k, x)| (x - f(k as f64)).norm() < 1e-9 * (1.0 + f(k as f64).norm())));
/// ```
pub fn spline_interpolation(positions: &[usize], values: &[C64], n: usize) -> Vec<C64> {
    assert!(!positions.is_empty() && positions.len() == values.len(), "one value per known position");
    let np = positions.len();
    if np < 4 {
        return if np == 3 {
            quadratic_interpolation(positions, values, n)
        } else {
            linear_interpolation(positions, values, n)
        };
    }
    let x: Vec<f64> = positions.iter().map(|&p| p as f64).collect();
    let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    // the second derivatives M_1 … M_{np-2}: h_{i-1} M_{i-1} + 2(h_{i-1} + h_i) M_i
    // + h_i M_{i+1} = 6 (slope_i - slope_{i-1}), with M_0 and M_{np-1} eliminated by the
    // not-a-knot conditions h_1 M_0 - (h_0 + h_1) M_1 + h_0 M_2 = 0 (and at the end)
    let slope: Vec<C64> = (0..np - 1).map(|i| (values[i + 1] - values[i]) / h[i]).collect();
    let m = np - 2;
    let (mut lower, mut diag, mut upper) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
    let mut rhs = vec![C64::new(0.0, 0.0); m];
    for r in 0..m {
        let i = r + 1;
        lower[r] = h[i - 1];
        diag[r] = 2.0 * (h[i - 1] + h[i]);
        upper[r] = h[i];
        rhs[r] = (slope[i] - slope[i - 1]) * 6.0;
    }
    // M_0 = ((h_0 + h_1) M_1 - h_0 M_2) / h_1
    diag[0] += h[0] * (h[0] + h[1]) / h[1];
    upper[0] -= h[0] * h[0] / h[1];
    // M_{np-1} = ((h_{np-3} + h_{np-2}) M_{np-2} - h_{np-2} M_{np-3}) / h_{np-3}
    let (a, b) = (h[np - 3], h[np - 2]);
    diag[m - 1] += b * (a + b) / a;
    lower[m - 1] -= b * b / a;
    let inner = thomas(&lower, &diag, &upper, &rhs);
    let mut mm = Vec::with_capacity(np);
    mm.push((inner[0] * (h[0] + h[1]) - inner[1] * h[0]) / h[1]);
    mm.extend_from_slice(&inner);
    mm.push((inner[m - 1] * (a + b) - inner[m.saturating_sub(2)] * b) / a);
    (0..n)
        .map(|k| {
            let t = k as f64;
            let i = interval(positions, t);
            let hi = h[i];
            let (l, r) = (x[i + 1] - t, t - x[i]);
            mm[i] * (l * l * l / (6.0 * hi))
                + mm[i + 1] * (r * r * r / (6.0 * hi))
                + (values[i] / hi - mm[i] * (hi / 6.0)) * l
                + (values[i + 1] / hi - mm[i + 1] * (hi / 6.0)) * r
        })
        .collect()
}

/// The tridiagonal system `lower[i] u[i-1] + diag[i] u[i] + upper[i] u[i+1] = rhs[i]`
/// (Thomas algorithm, no pivoting: for diagonally dominant systems).
fn thomas(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[C64]) -> Vec<C64> {
    let n = diag.len();
    let mut c = vec![0.0; n];
    let mut d = vec![C64::new(0.0, 0.0); n];
    for i in 0..n {
        let den = diag[i] - if i > 0 { lower[i] * c[i - 1] } else { 0.0 };
        c[i] = upper[i] / den;
        d[i] = (rhs[i] - if i > 0 { d[i - 1] * lower[i] } else { C64::new(0.0, 0.0) }) / den;
    }
    for i in (0..n.saturating_sub(1)).rev() {
        let next = d[i + 1];
        d[i] -= next * c[i];
    }
    d
}

/// Band-limited interpolation of `values` known at the increasing, uniformly spaced
/// `positions` (spacing $L$) onto `0..n`, as MATLAB `interp` (Oetken, Parks & Schüssler
/// 1975): every point is the combination of the `2 * half` nearest known values of least
/// mean square error for a signal of flat spectrum over the fraction `alpha` of the
/// band of the known samples, i.e. the noiseless [`lmmse`] with the correlation
/// $R(\Delta) = \operatorname{sinc}(\alpha \Delta / L)$. The known values pass
/// unchanged; near the ends the window is shifted inside. MATLAB's defaults are
/// `half = 4`, `alpha = 0.5`.
///
/// # Errors
/// [`Error::Config`] for unevenly spaced positions, `half = 0` or `alpha` outside
/// $(0, 1]$; [`Error::Numerical`] for a singular system.
///
/// # Examples
/// ```
/// use com_dsp::estimation::lowpass_interpolation;
/// use com_math::C64;
/// // a slow complex exponential is recovered between the samples
/// let f = |k: f64| C64::from_polar(1.0, 0.05 * k);
/// let pos: Vec<usize> = (0..16).map(|m| 8 * m).collect();
/// let v: Vec<C64> = pos.iter().map(|&k| f(k as f64)).collect();
/// let h = lowpass_interpolation(&pos, &v, 128, 4, 0.5)?;
/// assert!((40..80).all(|k| (h[k] - f(k as f64)).norm() < 1e-3));
/// assert_eq!(h[24], v[3]);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn lowpass_interpolation(
    positions: &[usize],
    values: &[C64],
    n: usize,
    half: usize,
    alpha: f64,
) -> Result<Vec<C64>> {
    ensure_shape(!positions.is_empty() && positions.len() == values.len(), || "one value per known position".into())?;
    ensure(half >= 1 && alpha > 0.0 && alpha <= 1.0, || {
        format!("need half ≥ 1 and 0 < alpha ≤ 1, got {half}, {alpha}")
    })?;
    let np = positions.len();
    if np == 1 {
        return Ok(vec![values[0]; n]);
    }
    let spacing = positions[1] - positions[0];
    ensure(spacing > 0 && positions.windows(2).all(|w| w[1] - w[0] == spacing), || "unevenly spaced positions".into())?;
    let corr = |d: f64| C64::new(com_math::sinc(alpha * d / spacing as f64), 0.0);
    let width = (2 * half).min(np);
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        if let Ok(i) = positions.binary_search(&k) {
            out.push(values[i]);
            continue;
        }
        let j = interval(positions, k as f64);
        let start = (j + 1).saturating_sub(half).min(np - width);
        let idx = start..start + width;
        let r_pp = CMat::from_shape_fn((width, width), |(a, b)| {
            corr(positions[start + a] as f64 - positions[start + b] as f64)
        });
        let r_qp = CMat::from_shape_fn((1, width), |(_, b)| corr(k as f64 - positions[start + b] as f64));
        out.push(lmmse(&r_qp, &r_pp, 1e-12, &values[idx])?[0]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use ndarray::Array2;

    #[test]
    #[allow(clippy::unreadable_literal, reason = "the values of SciPy, digit for digit")]
    fn the_spline_is_that_of_scipy_with_not_a_knot_ends() {
        // scipy.interpolate.CubicSpline(x, y, bc_type="not-a-knot") (SciPy 1.18)
        let pos = [0usize, 8, 16, 24, 32, 40];
        let v: Vec<C64> = [1.0, -0.5, 2.0, 0.3, -1.2, 0.8].iter().map(|&a| C64::new(a, -a)).collect();
        let h = spline_interpolation(&pos, &v, 45);
        for (k, want) in [
            (0, 1.0),
            (3, -0.76969401),
            (11, 0.59358724),
            (19, 1.77542318),
            (27, -0.5351237),
            (37, -0.4976888),
            (41, 1.40400391),
            (44, 3.78958333),
        ] {
            assert!((h[k] - C64::new(want, -want)).norm() < 1e-8, "{k}: {}", h[k]);
        }
    }

    #[test]
    fn coleri_eq_19_is_the_quadratic_with_alpha_l_over_l() {
        let mut r = Rng::new(4);
        let (spacing, np) = (8usize, 6usize);
        let pos: Vec<usize> = (0..np).map(|m| m * spacing).collect();
        let v: Vec<C64> = (0..np).map(|_| r.complex_gaussian()).collect();
        let h = quadratic_interpolation(&pos, &v, np * spacing);
        for m in 1..np - 1 {
            for l in 0..spacing {
                let a = l as f64 / spacing as f64;
                let eq19 = v[m - 1] * (a * (a - 1.0) / 2.0) - v[m] * ((a - 1.0) * (a + 1.0))
                    + v[m + 1] * (a * (a + 1.0) / 2.0);
                assert!((h[m * spacing + l] - eq19).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn the_lowpass_interpolator_is_exact_in_band_and_keeps_the_samples() {
        let mut r = Rng::new(8);
        let pos: Vec<usize> = (0..32).map(|m| 4 * m).collect();
        let v: Vec<C64> = (0..32).map(|_| r.complex_gaussian()).collect();
        let h = lowpass_interpolation(&pos, &v, 128, 4, 0.5).unwrap();
        assert!(pos.iter().zip(&v).all(|(&k, x)| h[k] == *x));
        // a tone in the band: better with a longer window
        let f = |k: f64| com_math::cis(0.3 * k / 4.0);
        let tone: Vec<C64> = pos.iter().map(|&k| f(k as f64)).collect();
        let err = |half: usize| {
            let h = lowpass_interpolation(&pos, &tone, 128, half, 0.5).unwrap();
            (40..88).map(|k| (h[k] - f(k as f64)).norm_sqr()).sum::<f64>()
        };
        assert!(err(8) < err(4) && err(4) < err(2) && err(4) < 1e-4, "{} {} {}", err(2), err(4), err(8));
        assert!(lowpass_interpolation(&[0, 2, 5], &v[..3], 6, 4, 0.5).is_err());
    }

    #[test]
    fn interpolation_is_exact_on_lines_and_lmmse_beats_ls() {
        let v: Vec<C64> = [0.0, 2.0, 4.0].iter().map(|x| C64::new(*x, -x)).collect();
        let out = linear_interpolation(&[2, 4, 6], &v, 9);
        assert_eq!(out[0], v[0]);
        assert!((out[3] - C64::new(1.0, -1.0)).norm() < 1e-12);
        assert_eq!(out[8], v[2]);
        // a smooth channel: R_ij = ρ^|i-j| with ρ = 0.95, 8 positions, SNR 0 dB
        let n = 8;
        let r = Array2::from_shape_fn((n, n), |(i, j)| C64::new(0.95f64.powi((i as i32 - j as i32).abs()), 0.0));
        let l = com_math::linalg::cholesky(&r.view()).unwrap();
        let mut rng = Rng::new(1);
        let (mut e_ls, mut e_mm) = (0.0, 0.0);
        for _ in 0..2000 {
            let g: Vec<C64> = (0..n).map(|_| rng.complex_gaussian()).collect();
            let h = com_math::linalg::matvec(&l.view(), &g);
            let y: Vec<C64> = h.iter().map(|v| v + rng.complex_gaussian()).collect();
            let x = vec![C64::new(1.0, 0.0); n];
            let hl = ls(&y, &x);
            let hm = lmmse(&r, &r, 1.0, &hl).unwrap();
            e_ls += hl.iter().zip(&h).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
            e_mm += hm.iter().zip(&h).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
        }
        assert!((e_ls / 2000.0 / n as f64 - 1.0).abs() < 0.05);
        let predicted = lmmse_mse(&r, &r, 1.0, &vec![1.0; n]).unwrap().iter().sum::<f64>() / n as f64;
        assert!((e_mm / 2000.0 / n as f64 / predicted - 1.0).abs() < 0.08);
        assert!(predicted < 0.4);
    }
}
