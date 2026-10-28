//! Channel estimation on the OFDM resource grid.
//!
//! The receiver knows the pilots: LS estimates $\hat H = Y / X$ at the pilot REs, then an
//! interpolation over the grid — first across the subcarriers of every pilot-carrying
//! symbol, then across the symbols of every subcarrier (linear in time, constant beyond
//! the first and last pilot symbols).
//!
//! - [`FrequencyInterpolation::Linear`]: linear between pilot subcarriers;
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
use com_dsp::estimation::{linear_interpolation, lmmse};
use com_math::{C64, cis};
use ndarray::Array2;

/// A power-delay profile on the sampling grid of the OFDM symbol.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerDelayProfile {
    /// Delays (samples of the IFFT of size $N$).
    pub delays: Vec<f64>,
    /// Powers (linear, normalised to a unit sum by the constructors).
    pub powers: Vec<f64>,
    /// IFFT size $N$ the delays refer to.
    pub fft_len: usize,
}

impl PowerDelayProfile {
    /// A profile from delays (samples) and linear powers, normalised.
    pub fn new(delays: Vec<f64>, powers: &[f64], fft_len: usize) -> PowerDelayProfile {
        let t: f64 = powers.iter().sum::<f64>().max(1e-300);
        PowerDelayProfile { delays, powers: powers.iter().map(|p| p / t).collect(), fft_len }
    }

    /// The uniform profile over `max_delay` samples (the robust design: the cyclic
    /// prefix).
    pub fn uniform(max_delay: usize, fft_len: usize) -> PowerDelayProfile {
        let n = max_delay.max(1);
        PowerDelayProfile::new((0..n).map(|d| d as f64).collect(), &vec![1.0; n], fft_len)
    }

    /// $R(\Delta k)$ between subcarriers $\Delta k$ apart.
    pub fn correlation(&self, delta: f64) -> C64 {
        self.delays
            .iter()
            .zip(&self.powers)
            .map(|(&d, &p)| cis(-2.0 * std::f64::consts::PI * delta * d / self.fft_len as f64) * p)
            .sum()
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
        let hp: Vec<C64> = ks.iter().map(|&k| rx[l * k_n + k] / grid.pilot(k, l)).collect();
        let full = match method {
            FrequencyInterpolation::Linear => linear_interpolation(&ks, &hp, k_n),
            FrequencyInterpolation::Lmmse(pdp) => {
                let r_pp =
                    Array2::from_shape_fn((ks.len(), ks.len()), |(i, j)| pdp.correlation(ks[i] as f64 - ks[j] as f64));
                let r_qp = Array2::from_shape_fn((k_n, ks.len()), |(q, j)| pdp.correlation(q as f64 - ks[j] as f64));
                lmmse(&r_qp, &r_pp, noise_var, &hp)?
            }
        };
        per_symbol.push(full);
    }
    // linear in time
    let mut out = vec![C64::new(0.0, 0.0); k_n * l_n];
    for k in 0..k_n {
        let vals: Vec<C64> = per_symbol.iter().map(|v| v[k]).collect();
        for (l, v) in linear_interpolation(&symbols, &vals, l_n).into_iter().enumerate() {
            out[l * k_n + k] = v;
        }
    }
    Ok(out)
}

/// One-tap zero-forcing equalisation of every RE, and the noise variance of every
/// equalised symbol ($\sigma^{2}/|H|^{2}$).
pub fn equalize(rx: &[C64], h: &[C64], noise_var: f64) -> (Vec<C64>, Vec<f64>) {
    rx.iter().zip(h).map(|(y, g)| (y / g, noise_var / g.norm_sqr().max(1e-300))).unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ofdm::grid::PilotPattern;
    use com_core::rng::Rng;

    /// A random multipath channel on the grid (static), and the received grid.
    fn channel(grid: &ResourceGrid, pdp: &PowerDelayProfile, s2: f64, rng: &mut Rng) -> (Vec<C64>, Vec<C64>) {
        let taps: Vec<C64> = pdp.powers.iter().map(|p| rng.complex_gaussian() * p.sqrt()).collect();
        let hk: Vec<C64> = (0..grid.n_subcarriers)
            .map(|k| {
                pdp.delays
                    .iter()
                    .zip(&taps)
                    .map(|(d, t)| t * cis(-2.0 * std::f64::consts::PI * k as f64 * d / pdp.fft_len as f64))
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
}
