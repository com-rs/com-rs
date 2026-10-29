//! Channel estimation on the OFDM resource grid.
//!
//! The receiver knows the pilots: LS estimates $\hat H = Y / X$ at the pilot REs, then an
//! interpolation over the grid — first across the subcarriers of every pilot-carrying
//! symbol, then across the symbols of every subcarrier (linear in time, constant beyond
//! the first and last pilot symbols).
//!
//! - [`FrequencyInterpolation::Linear`]: linear between pilot subcarriers;
//! - [`FrequencyInterpolation::SecondOrder`], [`FrequencyInterpolation::Spline`],
//!   [`FrequencyInterpolation::LowPass`] and [`FrequencyInterpolation::TimeDomain`]: the
//!   interpolations compared by Coleri et al. (2002), the quadratic through three
//!   pilots, the not-a-knot cubic spline, the band-limited interpolator of MATLAB
//!   `interp` and the zero padding of the IDFT of the pilot estimates (a comb of
//!   pilots on subcarriers $0, L, \dots$ of a grid of a multiple of $L$ subcarriers);
//! - [`FrequencyInterpolation::Lmmse`]: the linear MMSE estimator of
//!   `com_dsp::estimation` with the frequency correlation of a power-delay profile,
//!   $R(\Delta k) = \sum_{l} p_{l}\, e^{-j 2\pi \Delta k\, d_{l} / N}$ — the true profile, or the
//!   uniform profile over the cyclic prefix of the robust design (Edfors et al. 1998).
//!
//! With pilots on every subcarrier (block pattern) LMMSE still smooths the LS estimates
//! (van de Beek et al. 1995); with a comb it interpolates them.
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::estimation::{estimate, FrequencyInterpolation};
//! use com_waveform::ofdm::grid::{PilotPattern, ResourceGrid};
//! use com_math::C64;
//! let g = ResourceGrid { n_subcarriers: 16, n_symbols: 4, pattern: PilotPattern::Comb { spacing: 4 }, seed: 3 };
//! // a flat channel 0.5 + 0.5j, no noise: the estimate is exact everywhere
//! let h = C64::new(0.5, 0.5);
//! let tx = g.map(&vec![C64::new(1.0, 0.0); g.n_data()])?;
//! let rx: Vec<C64> = tx.iter().map(|x| x * h).collect();
//! let est = estimate(&g, &rx, 1e-6, &FrequencyInterpolation::Linear)?;
//! assert!(est.iter().all(|e| (e - h).norm() < 1e-9));
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J.-J. van de Beek, O. Edfors, M. Sandell, S. K. Wilson and P. O. Börjesson, "On
//!    channel estimation in OFDM systems," *Proc. IEEE VTC*, 815–819, 1995.
//! 2. O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson and P. O. Börjesson, "OFDM
//!    channel estimation by singular value decomposition," *IEEE Trans. Commun.*, 46(7),
//!    931–939, 1998.
//! 3. S. Coleri, M. Ergen, A. Puri and A. Bahai, "Channel estimation techniques based on
//!    pilot arrangement in OFDM systems," *IEEE Trans. Broadcast.*, 48(3), 223–229, 2002.

use super::grid::ResourceGrid;
use com_core::{Error, Result, ensure_shape};
use com_dsp::estimation::{frequency_correlation, lmmse};
use com_dsp::interpolation;
use com_dsp::resample::resample_fft;
use com_math::C64;
use ndarray::Array2;

/// A power-delay profile on the sampling grid of the OFDM symbol.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerDelayProfile {
    /// Delays (samples of the IFFT of size $N$).
    pub delays: Vec<f64>,
    /// Powers (linear, normalised to a unit sum by the constructors).
    pub powers: Vec<f64>,
    /// IFFT size $N$ the delays refer to.
    pub n_fft: usize,
}

impl PowerDelayProfile {
    /// A profile from delays (samples) and linear powers, normalised.
    pub fn new(delays: Vec<f64>, powers: &[f64], n_fft: usize) -> PowerDelayProfile {
        let t: f64 = powers.iter().sum::<f64>().max(1e-300);
        PowerDelayProfile { delays, powers: powers.iter().map(|p| p / t).collect(), n_fft }
    }

    /// The uniform profile over `max_delay` samples (the robust design: the cyclic
    /// prefix).
    pub fn uniform(max_delay: usize, n_fft: usize) -> PowerDelayProfile {
        let n = max_delay.max(1);
        PowerDelayProfile::new((0..n).map(|d| d as f64).collect(), &vec![1.0; n], n_fft)
    }

    /// $R(\Delta k)$ between subcarriers $\Delta k$ apart
    /// (`com_dsp::estimation::frequency_correlation`).
    pub fn correlation(&self, delta: f64) -> C64 {
        frequency_correlation(&self.delays, &self.powers, delta, self.n_fft)
    }
}

/// How the pilot estimates are interpolated across the subcarriers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum FrequencyInterpolation {
    /// Linear between pilot subcarriers.
    Linear,
    /// Linear MMSE with the frequency correlation of a power-delay profile.
    Lmmse(PowerDelayProfile),
    /// The parabola through the three nearest pilots (Coleri et al. 2002, eq. (19)).
    SecondOrder,
    /// The cubic spline through the pilots, not-a-knot ends (MATLAB `spline`).
    Spline,
    /// The band-limited interpolator of MATLAB `interp` over `2 * half` pilots, for a
    /// spectrum over the fraction `alpha` of the band of the pilots (`interp`: 4, 0.5).
    LowPass {
        /// Half the number of pilots combined.
        half: usize,
        /// The fraction of the band of the pilots the channel occupies.
        alpha: f64,
    },
    /// Zero padding of the IDFT of the pilot estimates ("time-domain" interpolation,
    /// Coleri et al. 2002, eqs. (20)–(22)): pilots on the subcarriers $0, L, \dots$ of a
    /// grid of $N_{p} L$ subcarriers.
    TimeDomain,
}

impl FrequencyInterpolation {
    /// The interpolation of a name: `linear`, `lmmse` (with the profile `pdp`),
    /// `second_order`, `spline`, `lowpass` (MATLAB `interp`: `half` = 4, `alpha` = 0.5)
    /// or `time_domain`.
    ///
    /// # Errors
    /// [`Error::Config`] for another name, or for `lmmse` without a profile.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::ofdm::estimation::FrequencyInterpolation;
    /// assert_eq!(FrequencyInterpolation::from_name("spline", None)?, FrequencyInterpolation::Spline);
    /// assert!(FrequencyInterpolation::from_name("lmmse", None).is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn from_name(name: &str, pdp: Option<PowerDelayProfile>) -> Result<FrequencyInterpolation> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "linear" => FrequencyInterpolation::Linear,
            "lmmse" => FrequencyInterpolation::Lmmse(
                pdp.ok_or_else(|| Error::Config("lmmse interpolation needs a power-delay profile".into()))?,
            ),
            "second_order" => FrequencyInterpolation::SecondOrder,
            "spline" => FrequencyInterpolation::Spline,
            "lowpass" => FrequencyInterpolation::LowPass { half: 4, alpha: 0.5 },
            "time_domain" => FrequencyInterpolation::TimeDomain,
            other => {
                return Err(Error::Config(format!(
                    "unknown interpolation {other:?}: linear, lmmse, second_order, spline, lowpass, time_domain"
                )));
            }
        })
    }
}

/// The channel at every RE of the grid (symbol after symbol) from the received grid
/// `rx` (the FFT outputs of the active subcarriers), with the noise variance per RE
/// `noise_var` (used by LMMSE).
///
/// # Errors
/// [`Error::Shape`] if `rx` is not a whole grid, [`Error::Config`] without pilots.
pub fn estimate(grid: &ResourceGrid, rx: &[C64], noise_var: f64, method: &FrequencyInterpolation) -> Result<Vec<C64>> {
    estimate_port(grid, rx, noise_var, method, 0, 1)
}

/// [`estimate`] for the channel of port `port` out of `n_ports`, from its pilots
/// ([`ResourceGrid::port_pilots`]): with one receive antenna's grid `rx`, the channel
/// between that antenna and the transmit port.
///
/// # Errors
/// As [`estimate`]; [`Error::Config`] if a pilot symbol has no pilot of the port.
pub fn estimate_port(
    grid: &ResourceGrid,
    rx: &[C64],
    noise_var: f64,
    method: &FrequencyInterpolation,
    port: usize,
    n_ports: usize,
) -> Result<Vec<C64>> {
    let (k_n, l_n) = (grid.n_subcarriers, grid.n_symbols);
    ensure_shape(rx.len() == k_n * l_n, || format!("{} REs for a grid of {k_n} × {l_n}", rx.len()))?;
    let symbols = grid.pilot_symbols();
    if symbols.is_empty() {
        return Err(Error::Config("the grid has no pilots".into()));
    }
    // the channel of every subcarrier at every pilot-carrying symbol
    let mut per_symbol = Vec::with_capacity(symbols.len());
    for &l in &symbols {
        let ks = grid.port_pilots(l, port, n_ports);
        if ks.is_empty() {
            return Err(Error::Config(format!("OFDM symbol {l} has no pilot of port {port} out of {n_ports}")));
        }
        let pilots: Vec<C64> = ks.iter().map(|&k| grid.pilot(k, l)).collect();
        let hp: Vec<C64> = ks.iter().zip(&pilots).map(|(&k, p)| rx[l * k_n + k] / p).collect();
        let full = match method {
            FrequencyInterpolation::Linear => interpolation::linear(&ks, &hp, k_n)?,
            FrequencyInterpolation::SecondOrder => interpolation::quadratic(&ks, &hp, k_n)?,
            FrequencyInterpolation::Spline => interpolation::spline(&ks, &hp, k_n)?,
            FrequencyInterpolation::LowPass { half, alpha } => interpolation::lowpass(&ks, &hp, k_n, *half, *alpha)?,
            FrequencyInterpolation::TimeDomain => {
                let spacing = k_n / ks.len();
                if ks.iter().enumerate().any(|(i, &k)| k != i * spacing) || spacing * ks.len() != k_n {
                    return Err(Error::Config(format!(
                        "time-domain interpolation needs pilots on 0, L, 2L, … of the {k_n} subcarriers"
                    )));
                }
                resample_fft(&hp, k_n)
            }
            FrequencyInterpolation::Lmmse(pdp) => {
                let r_pp =
                    Array2::from_shape_fn((ks.len(), ks.len()), |(i, j)| pdp.correlation(ks[i] as f64 - ks[j] as f64));
                let r_qp = Array2::from_shape_fn((k_n, ks.len()), |(q, j)| pdp.correlation(q as f64 - ks[j] as f64));
                // the LS estimate Y / X of a pilot of energy |X|² has the noise variance σ² / |X|²
                let energy = pilots.iter().map(C64::norm_sqr).sum::<f64>() / pilots.len() as f64;
                lmmse(&r_qp, &r_pp, noise_var / energy, &hp)?
            }
        };
        per_symbol.push(full);
    }
    // linear in time
    let mut out = vec![C64::new(0.0, 0.0); k_n * l_n];
    for k in 0..k_n {
        let vals: Vec<C64> = per_symbol.iter().map(|v| v[k]).collect();
        for (l, v) in interpolation::linear(&symbols, &vals, l_n)?.into_iter().enumerate() {
            out[l * k_n + k] = v;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::grid::PilotPattern;
    use com_core::rng::Rng;
    use com_math::cis;

    /// A random multipath channel on the grid (static), and the received grid.
    fn channel(grid: &ResourceGrid, pdp: &PowerDelayProfile, s2: f64, rng: &mut Rng) -> (Vec<C64>, Vec<C64>) {
        let taps: Vec<C64> = pdp.powers.iter().map(|p| rng.complex_gaussian() * p.sqrt()).collect();
        let hk: Vec<C64> = (0..grid.n_subcarriers)
            .map(|k| {
                pdp.delays
                    .iter()
                    .zip(&taps)
                    .map(|(d, t)| t * cis(-2.0 * std::f64::consts::PI * k as f64 * d / pdp.n_fft as f64))
                    .sum()
            })
            .collect();
        let tx = grid.map(&vec![C64::new(1.0, 0.0); grid.n_data()]).unwrap();
        let h: Vec<C64> = (0..tx.len()).map(|i| hk[i % grid.n_subcarriers]).collect();
        let rx = tx.iter().zip(&h).map(|(x, g)| x * g + rng.complex_gaussian() * s2.sqrt()).collect();
        (h, rx)
    }

    #[test]
    fn lmmse_beats_linear_interpolation_and_ls_has_the_noise_variance() {
        let grid =
            ResourceGrid { n_subcarriers: 64, n_symbols: 2, pattern: PilotPattern::Block { period: 1 }, seed: 1 };
        let pdp = PowerDelayProfile::new(vec![0.0, 2.0, 5.0], &[1.0, 0.5, 0.25], 64);
        let mut rng = Rng::new(2);
        let s2 = 0.1;
        let (mut e_ls, mut e_mm) = (0.0, 0.0);
        for _ in 0..200 {
            let (h, rx) = channel(&grid, &pdp, s2, &mut rng);
            let a = estimate(&grid, &rx, s2, &FrequencyInterpolation::Linear).unwrap();
            let b = estimate(&grid, &rx, s2, &FrequencyInterpolation::Lmmse(pdp.clone())).unwrap();
            e_ls += a.iter().zip(&h).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>() / a.len() as f64;
            e_mm += b.iter().zip(&h).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>() / b.len() as f64;
        }
        // pilots everywhere: linear interpolation is LS, of MSE σ²
        assert!((e_ls / 200.0 / s2 - 1.0).abs() < 0.05);
        // three taps: LMMSE keeps about 3/64 of the noise
        assert!(e_mm / 200.0 < 0.1 * s2, "{}", e_mm / 200.0);
    }

    #[test]
    fn boosted_pilots_lower_the_noise_of_the_estimates() {
        use crate::ofdm::grid::PilotTable;
        // pilots on every RE, of amplitude 2: the LS estimate Y / X has the variance σ² / 4
        let boosted = |a: f64| {
            let t = PilotTable::new(64, 2, (0..2).flat_map(|l| (0..64).map(move |k| ((k, l), C64::new(a, 0.0)))));
            ResourceGrid::from_table(t.unwrap())
        };
        let pdp = PowerDelayProfile::new(vec![0.0, 2.0, 5.0], &[1.0, 0.5, 0.25], 64);
        let mut rng = Rng::new(3);
        let s2 = 0.1;
        let mut mse = [[0.0; 2]; 2];
        for (i, a) in [1.0, 2.0].into_iter().enumerate() {
            let grid = boosted(a);
            for _ in 0..200 {
                let (h, rx) = channel(&grid, &pdp, s2, &mut rng);
                let ls = estimate(&grid, &rx, s2, &FrequencyInterpolation::Linear).unwrap();
                let mm = estimate(&grid, &rx, s2, &FrequencyInterpolation::Lmmse(pdp.clone())).unwrap();
                mse[i][0] += ls.iter().zip(&h).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>() / ls.len() as f64 / 200.0;
                mse[i][1] += mm.iter().zip(&h).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>() / mm.len() as f64 / 200.0;
            }
        }
        assert!((mse[0][0] / s2 - 1.0).abs() < 0.05, "{mse:?}");
        assert!((mse[1][0] / (s2 / 4.0) - 1.0).abs() < 0.05, "{mse:?}");
        // LMMSE weighs the boosted estimates with their own noise: about 4 times less error
        assert!((mse[0][1] / mse[1][1] / 4.0 - 1.0).abs() < 0.15, "{mse:?}");
    }
}
