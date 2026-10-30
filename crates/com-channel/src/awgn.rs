//! Additive white Gaussian noise.
//!
//! - [`awgn`]: circular Gaussian noise at a given SNR over a bandwidth `B` within a
//!   simulation bandwidth `fs` (OptiCommPy `awgn`): `σ² = (fs/B) P(x)/SNR`;
//! - [`awgn_ebn0`]: the same at a given `E_b/N_0` for symbols carrying `b` bits sampled
//!   at `sps` samples per symbol: `σ² = P(x) sps / (b · E_b/N_0)`;
//! - [`snr_noise_variance`] and [`ebn0_noise_variance`]: the noise variances of these
//!   two conventions, for callers that add the noise themselves (several streams, a
//!   fractional number of information bits per symbol);
//! - [`Awgn`]: the [`Propagate`] block, at a given `E_s/N_0`, on every stream of a
//!   [`Signal`] independently, and [`Noise`], at a given variance;
//! - [`add_noise`]: the kernel of all of them, white noise of a given variance added in
//!   place.
//!
//! # Model
//!
//! $$
//! y_{k} = x_{k} + n_{k}, \qquad n_{k} \sim \mathcal{CN}(0, N_{0})
//! $$
//!
//! with $N_{0} = \bar{E}_{s} / (E_{s}/N_{0})$, where $\bar{E}_{s}$ is the average symbol
//! energy of the stream at one sample per symbol.
//!
//! # References
//!
//! 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.

use com_core::prelude::*;
use com_core::{ensure, ensure_shape};
use com_math::{db2lin, power};

/// Additive white Gaussian noise channel.
///
/// Adds circularly-symmetric complex Gaussian noise to every stream independently.
/// The noise power is set from `E_s/N_0` and the measured average power of each
/// stream (at one sample per symbol), so the input need not be normalised.
///
/// # Errors
///
/// [`Propagate::propagate_into`] returns [`Error::Shape`] if `x` has no samples.
///
/// # Examples
///
/// ```
/// use com_core::prelude::*;
/// use com_channel::awgn::Awgn;
///
/// let mut rng = batch_rng(42, 0);
/// let mut ch = Awgn::new(EsN0(Db(10.0)));
/// let x = Signal::from_stream(vec![C64::new(1.0, 0.0); 1024], Hertz(1.0))?;
/// let y = ch.propagate(&x, &mut rng)?;
/// assert_eq!(y.n_samples(), 1024);
/// # Ok::<(), com_core::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Awgn {
    /// Energy per symbol over the noise spectral density.
    pub es_n0: EsN0,
}

impl Awgn {
    /// The channel at `es_n0`.
    pub fn new(es_n0: EsN0) -> Awgn {
        Awgn { es_n0 }
    }

    /// Variance `N_0` of the noise added to a stream of average power `es`.
    pub fn noise_variance(&self, es: f64) -> f64 {
        es / self.es_n0.0.to_linear()
    }
}

impl Propagate for Awgn {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        ensure_shape(x.n_samples() > 0, || "empty signal".into())?;
        let power = x.power();
        x.map_streams(out, |s, xs| noisy(xs, self.noise_variance(power[s]), rng));
        Ok(())
    }
}

/// Add circular complex Gaussian noise of variance `noise_var` ($\mathcal{CN}(0,
/// \sigma^{2})$, $\sigma^{2}/2$ per dimension) to `x`, in place.
///
/// # Examples
/// ```
/// use com_channel::awgn::add_noise;
/// use com_core::rng::Rng;
/// use com_math::C64;
/// let mut x = vec![C64::new(0.0, 0.0); 100_000];
/// add_noise(&mut x, 0.5, &mut Rng::new(1));
/// assert!((com_math::power(&x) - 0.5).abs() < 0.01);
/// ```
pub fn add_noise(x: &mut [C64], noise_var: f64, rng: &mut Rng) {
    let sd = noise_var.sqrt();
    for v in x {
        *v += rng.complex_gaussian() * sd;
    }
}

/// `x` plus noise of variance `noise_var` ([`add_noise`] of a copy).
fn noisy(x: &[C64], noise_var: f64, rng: &mut Rng) -> Vec<C64> {
    let mut y = x.to_vec();
    add_noise(&mut y, noise_var, rng);
    y
}

/// Add circular complex Gaussian noise for an SNR of `snr_db` within the bandwidth `b`
/// of a signal sampled at `fs` (OptiCommPy `awgn`).
pub fn awgn(x: &[C64], snr_db: f64, fs: f64, b: f64, rng: &mut Rng) -> Vec<C64> {
    noisy(x, snr_noise_variance(power(x), snr_db, fs, b), rng)
}

/// Add noise for an `E_b/N_0` of `ebn0_db` (signal of `bits` information bits per
/// symbol, fractional for a coded or shaped signal, at `sps` samples per symbol, after a
/// unit-energy matched filter).
pub fn awgn_ebn0(x: &[C64], ebn0_db: f64, bits: f64, sps: f64, rng: &mut Rng) -> Vec<C64> {
    noisy(x, ebn0_noise_variance(power(x), ebn0_db, bits, sps), rng)
}

/// The noise variance of [`awgn`]: an SNR of `snr_db` within the bandwidth `b` of a
/// signal of power `p` sampled at `fs`,
///
/// $$
/// \sigma^{2} = \frac{f_{s}}{B} \frac{P}{\mathrm{SNR}}.
/// $$
pub fn snr_noise_variance(p: f64, snr_db: f64, fs: f64, b: f64) -> f64 {
    (fs / b) * p / db2lin(snr_db)
}

/// The noise variance of [`awgn_ebn0`]: an $E_{b}/N_{0}$ of `ebn0_db` for a signal of
/// power `p` at `sps` samples per symbol carrying `bits` information bits per symbol
/// (fractional for a coded or shaped signal: $R \log_{2} M$, or the entropy of the
/// source),
///
/// $$
/// \sigma^{2} = \frac{P\, n_{\mathrm{sps}}}{b\, E_{b}/N_{0}}.
/// $$
pub fn ebn0_noise_variance(p: f64, ebn0_db: f64, bits: f64, sps: f64) -> f64 {
    p * sps / (bits * db2lin(ebn0_db))
}

/// White noise of a given variance on every stream: $y_{k} = x_{k} + n_{k}$,
/// $n_{k} \sim \mathcal{CN}(0, \sigma^{2})$ — the receiver noise of a MIMO link, whose
/// SNR is set per receive antenna rather than measured on the signal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Noise {
    /// Variance $\sigma^{2} = \mathbb{E}|n_{k}|^{2}$.
    pub variance: f64,
}

impl Propagate for Noise {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        ensure(self.variance >= 0.0, || format!("negative noise variance {}", self.variance))?;
        x.map_streams(out, |_, xs| noisy(xs, self.variance, rng));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_levels() {
        let x = vec![C64::new(1.0, 0.0); 100_000];
        let y = awgn(&x, 10.0, 1.0, 1.0, &mut Rng::new(1));
        let p: f64 = y.iter().map(|v| (v - C64::new(1.0, 0.0)).norm_sqr()).sum::<f64>() / 1e5;
        assert!((p - 0.1).abs() < 0.003);
    }

    #[test]
    fn the_block_matches_the_function_and_is_the_identity_without_noise() {
        let x: Vec<C64> = (0..1000).map(|n| C64::new((n % 3) as f64, 1.0)).collect();
        let s = Signal::from_streams(&[x.clone(), x.clone()], Hertz(1.0)).unwrap();
        let y = Awgn::new(EsN0(Db(7.0))).propagate(&s, &mut Rng::new(4)).unwrap();
        // same draws as the slice function at Es/N0 = Eb/N0 for one bit per symbol
        let z = awgn_ebn0(&x, 7.0, 1.0, 1.0, &mut Rng::new(4));
        assert!(y.stream(0).iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-12));
        let quiet = Awgn::new(EsN0(Db(400.0))).propagate(&s, &mut Rng::new(4)).unwrap();
        assert!(quiet.stream(1).iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-12));
        assert!(Awgn::new(EsN0(Db(0.0))).propagate(&Signal::zeros(1, 0, Hertz(1.0)), &mut Rng::new(0)).is_err());
    }
}
