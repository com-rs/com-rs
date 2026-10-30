//! Interpolation of values known at some positions.
//!
//! - [`linear`] (constant beyond the ends);
//! - [`quadratic`], through the three nearest positions (the "second-order"
//!   interpolation of Coleri et al. 2002, eq. (19));
//! - [`spline`], the cubic spline with not-a-knot ends (MATLAB `spline`);
//! - [`lowpass`], the band-limited interpolator of MATLAB `interp` (Oetken, Parks &
//!   Schüssler 1975), the noiseless LMMSE of [`crate::estimation::lmmse`] for a flat
//!   spectrum, and [`LowPassInterpolator`], the same built once for positions shared by several sets
//!   of values;
//! - [`lagrange_cubic`], the cubic Lagrange (Farrow) interpolator of four samples at a
//!   fractional position, that of the timing recovery;
//! - and, for uniformly spaced values of a periodic block, the zero padding of the IDFT
//!   of [`crate::resample::resample_fft`] (the "time-domain" interpolation).
//!
//! The pilot interpolation of an OFDM grid (`com_waveform::ofdm::estimation`) chooses
//! among them.
//!
//! # Examples
//! ```
//! use com_dsp::interpolation::linear;
//! use com_math::C64;
//! let v = [C64::new(0.0, 0.0), C64::new(4.0, 0.0)];
//! let h = linear(&[0, 4], &v, 6)?;
//! assert_eq!(h[1], C64::new(1.0, 0.0));
//! assert_eq!(h[5], v[1]); // constant beyond the last position
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. S. Coleri, M. Ergen, A. Puri and A. Bahai, "Channel estimation techniques based on
//!    pilot arrangement in OFDM systems," *IEEE Trans. Broadcast.*, 48(3), 223–229, 2002.
//! 2. G. Oetken, T. W. Parks and H. W. Schüssler, "New results in the design of digital
//!    interpolators," *IEEE Trans. Acoust., Speech, Signal Process.*, 23(3), 301–309,
//!    1975.
//! 3. C. de Boor, *A Practical Guide to Splines*, Springer, 1978, ch. IV (not-a-knot).

use crate::estimation::regularized;
use com_core::{Result, ensure, ensure_shape};
use com_math::C64;
use com_math::linalg::{CMat, cholesky, solve_cholesky};

/// Linear interpolation of `values` known at the increasing `positions` onto
/// `0..n` (constant beyond the first and last positions).
///
/// # Errors
/// [`com_core::Error::Shape`] if `positions` is empty or of another length than `values`.
pub fn linear(positions: &[usize], values: &[C64], n: usize) -> Result<Vec<C64>> {
    ensure_known(positions, values)?;
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
    Ok(out)
}

/// One value per known position, at least one.
fn ensure_known(positions: &[usize], values: &[C64]) -> Result<()> {
    ensure_shape(!positions.is_empty() && positions.len() == values.len(), || {
        format!("{} values at {} known positions (at least one)", values.len(), positions.len())
    })
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
/// Fewer than three positions fall back to [`linear`].
///
/// # Errors
/// [`com_core::Error::Shape`] if `positions` is empty or of another length than `values`.
///
/// # Examples
/// ```
/// use com_dsp::interpolation::quadratic;
/// use com_math::C64;
/// // exact on a parabola
/// let v: Vec<C64> = [0usize, 4, 8, 12].iter().map(|&k| C64::new((k * k) as f64, 1.0)).collect();
/// let h = quadratic(&[0, 4, 8, 12], &v, 16)?;
/// assert!(h.iter().enumerate().all(|(k, x)| (x - C64::new((k * k) as f64, 1.0)).norm() < 1e-9));
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn quadratic(positions: &[usize], values: &[C64], n: usize) -> Result<Vec<C64>> {
    ensure_known(positions, values)?;
    if positions.len() < 3 {
        return linear(positions, values, n);
    }
    Ok((0..n)
        .map(|k| {
            let x = k as f64;
            let j = interval(positions, x).clamp(1, positions.len() - 2);
            let (a, b, c) = (positions[j - 1] as f64, positions[j] as f64, positions[j + 1] as f64);
            values[j - 1] * ((x - b) * (x - c) / ((a - b) * (a - c)))
                + values[j] * ((x - a) * (x - c) / ((b - a) * (b - c)))
                + values[j + 1] * ((x - a) * (x - b) / ((c - a) * (c - b)))
        })
        .collect())
}

/// Cubic-spline interpolation of `values` known at the increasing `positions` onto
/// `0..n`, with the not-a-knot end conditions (the third derivative is continuous at
/// the second and the next-to-last positions; de Boor 1978), as MATLAB `spline`: the
/// end polynomials extrapolate beyond the ends. Three positions give the parabola, two
/// the line, one a constant.
///
/// # Errors
/// [`com_core::Error::Shape`] if `positions` is empty or of another length than `values`.
///
/// # Examples
/// ```
/// use com_dsp::interpolation::spline;
/// use com_math::C64;
/// // not-a-knot reproduces cubics exactly
/// let f = |k: f64| C64::new(k * k * k - 2.0 * k, -k);
/// let pos = [0usize, 3, 5, 9, 12];
/// let v: Vec<C64> = pos.iter().map(|&k| f(k as f64)).collect();
/// let h = spline(&pos, &v, 14)?;
/// assert!(h.iter().enumerate().all(|(k, x)| (x - f(k as f64)).norm() < 1e-9 * (1.0 + f(k as f64).norm())));
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn spline(positions: &[usize], values: &[C64], n: usize) -> Result<Vec<C64>> {
    ensure_known(positions, values)?;
    let np = positions.len();
    if np < 4 {
        return if np == 3 { quadratic(positions, values, n) } else { linear(positions, values, n) };
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
    let inner = com_math::linalg::solve_tridiagonal(&lower, &diag, &upper, &rhs);
    let mut mm = Vec::with_capacity(np);
    mm.push((inner[0] * (h[0] + h[1]) - inner[1] * h[0]) / h[1]);
    mm.extend_from_slice(&inner);
    mm.push((inner[m - 1] * (a + b) - inner[m.saturating_sub(2)] * b) / a);
    Ok((0..n)
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
        .collect())
}

/// Band-limited interpolation of `values` known at the increasing, uniformly spaced
/// `positions` (spacing $L$) onto `0..n`, as MATLAB `interp` (Oetken, Parks & Schüssler
/// 1975): every point is the combination of the `2 * half` nearest known values of least
/// mean square error for a signal of flat spectrum over the fraction `alpha` of the
/// band of the known samples, i.e. the noiseless [`crate::estimation::lmmse`] with the
/// correlation $R(\Delta) = \operatorname{sinc}(\alpha \Delta / L)$. The known values
/// pass unchanged; near the ends the window is shifted inside ([`LowPassInterpolator`] builds the
/// interpolator once for several sets of values at the same positions). MATLAB's
/// defaults are `half = 4`, `alpha = 0.5`.
///
/// # Errors
/// [`com_core::Error::Config`] for unevenly spaced positions, `half = 0` or `alpha` outside
/// $(0, 1]$; [`com_core::Error::Numerical`] for a singular system.
///
/// # Examples
/// ```
/// use com_dsp::interpolation::lowpass;
/// use com_math::C64;
/// // a slow complex exponential is recovered between the samples
/// let f = |k: f64| C64::from_polar(1.0, 0.05 * k);
/// let pos: Vec<usize> = (0..16).map(|m| 8 * m).collect();
/// let v: Vec<C64> = pos.iter().map(|&k| f(k as f64)).collect();
/// let h = lowpass(&pos, &v, 128, 4, 0.5)?;
/// assert!((40..80).all(|k| (h[k] - f(k as f64)).norm() < 1e-3));
/// assert_eq!(h[24], v[3]);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn lowpass(positions: &[usize], values: &[C64], n: usize, half: usize, alpha: f64) -> Result<Vec<C64>> {
    ensure_known(positions, values)?;
    LowPassInterpolator::new(positions, n, half, alpha)?.apply(values)
}

/// The band-limited interpolator of [`lowpass`] for one set of positions, built once:
/// the positions being uniformly spaced, every window of `2 * half` known values has
/// the same correlation matrix, factored once; each window's system is solved once for
/// the points it serves, and each point is the product of its correlations with that
/// solution. [`LowPassInterpolator::apply`] costs $O(N_{p} W^{2} + n W)$ for $N_{p}$ positions and
/// windows of $W$ values, instead of $O(n W^{3})$ for a solve per point, with the same
/// operations as before (the same results, bit for bit).
///
/// # Examples
/// ```
/// use com_dsp::interpolation::{lowpass, LowPassInterpolator};
/// use com_math::C64;
/// let pos: Vec<usize> = (0..16).map(|m| 8 * m).collect();
/// let plan = LowPassInterpolator::new(&pos, 128, 4, 0.5)?;
/// for a in [0.05, 0.1] {
///     let v: Vec<C64> = pos.iter().map(|&k| C64::from_polar(1.0, a * k as f64)).collect();
///     assert_eq!(plan.apply(&v)?, lowpass(&pos, &v, 128, 4, 0.5)?);
/// }
/// # Ok::<(), com_core::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LowPassInterpolator {
    n_known: usize,
    width: usize,
    /// The Cholesky factor of the regularised correlation of a window.
    factor: CMat,
    /// Every point: a known position, or the start of its window and its correlations.
    points: Vec<Point>,
}

#[derive(Clone, Debug, PartialEq)]
enum Point {
    Known(usize),
    Window { start: usize, r_qp: CMat },
}

impl LowPassInterpolator {
    /// The interpolator of values known at the increasing, uniformly spaced `positions`
    /// onto `0..n` (see [`lowpass`]).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] without positions, [`com_core::Error::Config`] for
    /// unevenly spaced positions, `half = 0` or `alpha` outside $(0, 1]$;
    /// [`com_core::Error::Numerical`] for a singular system.
    pub fn new(positions: &[usize], n: usize, half: usize, alpha: f64) -> Result<LowPassInterpolator> {
        ensure_shape(!positions.is_empty(), || "no known position".into())?;
        ensure(half >= 1 && alpha > 0.0 && alpha <= 1.0, || {
            format!("need half ≥ 1 and 0 < alpha ≤ 1, got {half}, {alpha}")
        })?;
        let np = positions.len();
        if np == 1 {
            let points = vec![Point::Known(0); n];
            return Ok(LowPassInterpolator { n_known: 1, width: 1, factor: CMat::zeros((0, 0)), points });
        }
        let spacing = positions[1] - positions[0];
        ensure(spacing > 0 && positions.windows(2).all(|w| w[1] - w[0] == spacing), || {
            "unevenly spaced positions".into()
        })?;
        let corr = |d: f64| C64::new(com_math::sinc(alpha * d / spacing as f64), 0.0);
        let width = (2 * half).min(np);
        // the differences of positions in a window are multiples of the spacing, whatever
        // the window: one correlation matrix, that of the noiseless LMMSE (regularised)
        let r_pp = CMat::from_shape_fn((width, width), |(a, b)| corr(positions[a] as f64 - positions[b] as f64));
        let factor = cholesky(&regularized(&r_pp, LOWPASS_REGULARIZATION).view())?;
        let points = (0..n)
            .map(|k| {
                if let Ok(i) = positions.binary_search(&k) {
                    return Point::Known(i);
                }
                let j = interval(positions, k as f64);
                let start = (j + 1).saturating_sub(half).min(np - width);
                let r_qp = CMat::from_shape_fn((1, width), |(_, b)| corr(k as f64 - positions[start + b] as f64));
                Point::Window { start, r_qp }
            })
            .collect();
        Ok(LowPassInterpolator { n_known: np, width, factor, points })
    }

    /// The interpolation of `values`, one per known position.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] unless `values` has one value per known position.
    pub fn apply(&self, values: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(values.len() == self.n_known, || {
            format!("{} values at {} known positions", values.len(), self.n_known)
        })?;
        // the solution of each window, once
        let mut solved: Vec<Option<CMat>> = vec![None; self.n_known];
        let mut out = Vec::with_capacity(self.points.len());
        for point in &self.points {
            match point {
                Point::Known(i) => out.push(values[*i]),
                Point::Window { start, r_qp } => {
                    if solved[*start].is_none() {
                        let b = CMat::from_shape_fn((self.width, 1), |(a, _)| values[start + a]);
                        solved[*start] = Some(solve_cholesky(&self.factor.view(), &b.view())?);
                    }
                    if let Some(z) = &solved[*start] {
                        out.push(r_qp.dot(z)[[0, 0]]);
                    }
                }
            }
        }
        Ok(out)
    }
}

/// The regularisation $\sigma^{2}/E_{p}$ of the noiseless LMMSE of [`lowpass`].
const LOWPASS_REGULARIZATION: f64 = 1e-12;

/// The cubic Lagrange (Farrow) interpolator of four samples at the fractional position
/// `t` $\in [-1, 1]$: `t = 0` returns `x[2]`, `t = 1` returns `x[3]`, `t = -1` returns
/// `x[1]` (OptiCommPy `interpolator`, the interpolator of the Gardner loop of
/// [`crate::sync::timing`]).
///
/// # Examples
/// ```
/// use com_dsp::interpolation::lagrange_cubic;
/// use com_math::C64;
/// let x = [C64::new(1.0, 0.0), C64::new(2.0, 0.0), C64::new(3.0, 0.0), C64::new(4.0, 0.0)];
/// assert!((lagrange_cubic(&x, 0.0) - x[2]).norm() < 1e-15);
/// assert!((lagrange_cubic(&x, 0.5) - C64::new(3.5, 0.0)).norm() < 1e-15);
/// ```
pub fn lagrange_cubic(x: &[C64; 4], t: f64) -> C64 {
    let t2 = t * t;
    let t3 = t2 * t;
    x[0] * (-t3 / 6.0 + t / 6.0)
        + x[1] * (t3 / 2.0 + t2 / 2.0 - t)
        + x[2] * (-t3 / 2.0 - t2 + t / 2.0 + 1.0)
        + x[3] * (t3 / 6.0 + t2 / 2.0 + t / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn linear_is_exact_on_lines_and_constant_beyond_the_ends() {
        let v: Vec<C64> = [0.0, 2.0, 4.0].iter().map(|x| C64::new(*x, -x)).collect();
        let out = linear(&[2, 4, 6], &v, 9).unwrap();
        assert_eq!(out[0], v[0]);
        assert!((out[3] - C64::new(1.0, -1.0)).norm() < 1e-12);
        assert_eq!(out[8], v[2]);
    }

    #[test]
    #[allow(clippy::unreadable_literal, reason = "the values of SciPy, digit for digit")]
    fn the_spline_is_that_of_scipy_with_not_a_knot_ends() {
        // scipy.interpolate.CubicSpline(x, y, bc_type="not-a-knot") (SciPy 1.18)
        let pos = [0usize, 8, 16, 24, 32, 40];
        let v: Vec<C64> = [1.0, -0.5, 2.0, 0.3, -1.2, 0.8].iter().map(|&a| C64::new(a, -a)).collect();
        let h = spline(&pos, &v, 45).unwrap();
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
        let h = quadratic(&pos, &v, np * spacing).unwrap();
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
        let h = lowpass(&pos, &v, 128, 4, 0.5).unwrap();
        assert!(pos.iter().zip(&v).all(|(&k, x)| h[k] == *x));
        // a tone in the band: better with a longer window
        let f = |k: f64| com_math::cis(0.3 * k / 4.0);
        let tone: Vec<C64> = pos.iter().map(|&k| f(k as f64)).collect();
        let err = |half: usize| {
            let h = lowpass(&pos, &tone, 128, half, 0.5).unwrap();
            (40..88).map(|k| (h[k] - f(k as f64)).norm_sqr()).sum::<f64>()
        };
        assert!(err(8) < err(4) && err(4) < err(2) && err(4) < 1e-4, "{} {} {}", err(2), err(4), err(8));
        assert!(lowpass(&[0, 2, 5], &v[..3], 6, 4, 0.5).is_err());
    }

    /// The band-limited interpolation as it was computed before: an LMMSE system built
    /// and solved for every point.
    fn former_lowpass(positions: &[usize], values: &[C64], n: usize, half: usize, alpha: f64) -> Vec<C64> {
        let np = positions.len();
        if np == 1 {
            return vec![values[0]; n];
        }
        let spacing = positions[1] - positions[0];
        let corr = |d: f64| C64::new(com_math::sinc(alpha * d / spacing as f64), 0.0);
        let width = (2 * half).min(np);
        (0..n)
            .map(|k| {
                if let Ok(i) = positions.binary_search(&k) {
                    return values[i];
                }
                let j = interval(positions, k as f64);
                let start = (j + 1).saturating_sub(half).min(np - width);
                let r_pp = CMat::from_shape_fn((width, width), |(a, b)| {
                    corr(positions[start + a] as f64 - positions[start + b] as f64)
                });
                let r_qp = CMat::from_shape_fn((1, width), |(_, b)| corr(k as f64 - positions[start + b] as f64));
                crate::estimation::lmmse(&r_qp, &r_pp, 1e-12, &values[start..start + width]).unwrap()[0]
            })
            .collect()
    }

    #[test]
    fn the_lowpass_plan_is_the_former_solve_per_point() {
        let mut r = Rng::new(12);
        for (np, spacing, offset, n, half, alpha) in [
            (32usize, 4usize, 0usize, 128usize, 4usize, 0.5),
            (10, 6, 3, 70, 4, 0.8),
            (5, 3, 1, 20, 4, 0.5),
            (1, 1, 2, 6, 2, 1.0),
        ] {
            let pos: Vec<usize> = (0..np).map(|m| offset + m * spacing).collect();
            let plan = LowPassInterpolator::new(&pos, n, half, alpha).unwrap();
            for _ in 0..3 {
                let v: Vec<C64> = (0..np).map(|_| r.complex_gaussian()).collect();
                assert_eq!(plan.apply(&v).unwrap(), former_lowpass(&pos, &v, n, half, alpha), "{np} pilots");
            }
            assert!(plan.apply(&vec![C64::new(1.0, 0.0); np + 1]).is_err());
        }
    }

    #[test]
    fn interpolators_reject_unmatched_or_missing_pilots() {
        let v = [C64::new(1.0, 0.0); 3];
        for f in [linear, quadratic, spline] {
            assert!(f(&[], &[], 4).is_err());
            assert!(f(&[0, 2], &v, 4).is_err());
        }
    }

    #[test]
    fn lagrange_cubic_is_exact_on_cubics() {
        let f = |t: f64| 0.3 * t * t * t - t * t + 2.0 * t - 0.5;
        let x = [-2.0, -1.0, 0.0, 1.0].map(|t| C64::new(f(t), 0.0));
        for t in [-1.0, -0.4, 0.0, 0.25, 0.9, 1.0] {
            assert!((lagrange_cubic(&x, t).re - f(t)).abs() < 1e-12);
        }
    }
}
