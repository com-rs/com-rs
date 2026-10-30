//! The theory of CE-OFDM: the error rates of the receivers, the spectrum, the optimum
//! asymptotic gain and the effect of a cycle slip.
//!
//! # Gaussian model
//!
//! With $K$ proper data symbols ($\mathbb{E}[d^{2}] = 0$: QAM, PSK of order $\ge 3$)
//! the OFDM signal $\mathbf{x}$ of a block is, by the central limit theorem, a circular
//! Gaussian process of unit variance with the normalised autocorrelation
//!
//! $$\rho\[m\] = \mathbb{E}[x_{n} x_{n+m}] = \frac{1}{K} \sum_{k=1}^{K}
//! \cos\frac{2\pi k m}{N}$$
//!
//! ([`phase_autocorrelation`]), and the phase $\varphi = 2\pi h x$ has the variance
//! $\varphi_{0}^{2} = (2\pi h)^{2}$. For jointly Gaussian phases of correlation $\rho$,
//! $\mathbb{E}[e^{j\varphi_{1}} e^{-j\varphi_{2}}] = e^{-\varphi_{0}^{2}(1 - \rho)}$ and
//! $\mathbb{E}[\sin a\varphi_{1} \sin b\varphi_{2}] = e^{-(a^{2} + b^{2})\varphi_{0}^{2}/2}
//! \sinh(ab\rho\varphi_{0}^{2})$, from which everything below follows in closed form.
//!
//! # Spectrum
//!
//! The expected periodogram of the transmitted block ([`psd`]) is the DFT of
//! $e^{-\varphi_{0}^{2}(1 - \rho\[m\])} = \sum_{i \ge 0} P_{i} \rho\[m\]^{i}$, with the powers
//! of the intermodulation products ([`imp_powers`])
//!
//! $$P_{i} = e^{-\varphi_{0}^{2}} \frac{\varphi_{0}^{2i}}{i!}\\,,$$
//!
//! the closed form of eqs. (12)–(14) of Guerreiro et al. (2015) and of eq. (2) of
//! Pacheco and Hatzinakos (2006) after Abramson (1963): $P_{0}$ is the carrier line,
//! $P_{1}$ the useful term and $P_{i}$ spreads over $i$ times the band of the message.
//! [`occupied_bandwidth`] gives the 99 % bandwidth of a flat message spectrum (Pacheco
//! and Hatzinakos, eqs. (2)–(5) and Fig. 1).
//!
//! # Receivers and their error rates
//!
//! - The arctangent receiver ([`super::Receiver::Arctangent`]) above its threshold sees
//!   the SNR $(2\pi h)^{2} E_{s}/N_{0} \cdot N/(N + N_{\mathrm{cp}})$ on every subcarrier
//!   ([`snr`], [`ser_esn0`], [`ber_esn0`]; eq. (15) of Thompson et al. 2008). For QPSK this is also
//!   the optimum asymptotic BER $Q(\sqrt{(2\pi h)^{2} \cdot 2E_{b}/N_{0}})$ of Guerreiro et
//!   al. (2016, eq. (8)), whose asymptotic gain is $G = (2\pi h)^{2}$ ([`optimum_gain`],
//!   [`ber_optimum`]).
//! - The linear receivers of Ahmed and Zeidler (2015) take a memoryless function $u$ of
//!   the received sample: $u = \operatorname{Im} r$ (basic linear receiver) or
//!   $u = \operatorname{Im} r \\, (2 - \operatorname{Re} r)$ (enhanced receiver). By
//!   Bussgang's theorem $u = \alpha\varphi + e$ with $e$ uncorrelated with $\varphi$:
//!   $\alpha = e^{-\varphi_{0}^{2}/2}$ for the first and
//!   $\alpha = 2e^{-\varphi_{0}^{2}/2} - e^{-2\varphi_{0}^{2}}$ for the second
//!   ([`receiver_gain`]). The receiver divides by $\alpha$, and subcarrier $k$ sees
//!
//!   $$\mathrm{SNR}_{k} = \frac{N \varphi_{0}^{2} \alpha^{2}}{2K S_{e}\[k\]}\\,,$$
//!
//!   $S_{e}\[k\]$ the DFT of the autocorrelation of $e$, which holds the noise and the
//!   distortion falling on subcarrier $k$ ([`subcarrier_snr`]). Ahmed and Zeidler's
//!   eqs. (17), (19), (26) and (29) are the first terms of the Taylor series of these
//!   expressions in $\varphi_{0}$.
//!
//! # Cycle slips
//!
//! A cycle slip of the phase demodulator, a step of $2\pi$ from sample $n_{0}$ to the
//! end of the block, adds to subcarrier $k$ ([`cycle_slip_response`])
//!
//! $$\Delta\hat{d}_{k} = \frac{\sqrt{2K}}{N h} \sum_{n = n_{0}}^{N - 1} e^{-j2\pi k n/N}
//! = \frac{\sqrt{2K}}{N h} \frac{e^{-j2\pi k n_{0}/N} - 1}{1 - e^{-j2\pi k/N}}\\,,$$
//!
//! which decreases as $1/k$: eqs. (11)–(14) of Ahmed et al. (2012) are its limit for a
//! large $N$.
//!
//! # References
//!
//! 1. S. C. Thompson, A. U. Ahmed, J. G. Proakis, J. R. Zeidler and M. J. Geile,
//!    "Constant envelope OFDM," *IEEE Trans. Commun.* 56(8), 1300–1312, 2008.
//! 2. R. A. Pacheco and D. Hatzinakos, "Error rate analysis of phase-modulated OFDM
//!    (OFDM-PM) in AWGN channels," *Proc. IEEE ICASSP*, 2006.
//! 3. A. U. Ahmed, S. C. Thompson, D. W. Chi and J. R. Zeidler, "Subcarrier based
//!    threshold performance enhancement in constant envelope OFDM," *Proc. IEEE MILCOM*,
//!    2012.
//! 4. A. U. Ahmed and J. R. Zeidler, "Novel low-complexity receivers for constant
//!    envelope OFDM," *IEEE Trans. Signal Process.* 63(17), 4572–4582, 2015.
//! 5. J. Guerreiro, R. Dinis and P. Montezuma, "Optimum performance and spectral
//!    characterization of CE-OFDM signals," *Proc. IEEE VTC Fall*, 2015.
//! 6. J. Guerreiro, R. Dinis and P. Montezuma, "On the detection of CE-OFDM signals,"
//!    *IEEE Commun. Lett.* 20(11), 2165–2168, 2016.

use super::{CeOfdm, Receiver, TAU};
use com_core::{Result, ensure};
use com_mapping::constellation::ConstellationKind;
use com_math::{C64, erf, fft, qfunc};

/// The SNR of every data subcarrier (linear scale) at `E_s/N_0` (dB) per data
/// symbol: `(2πh)² E_s/N_0 · N/(N + N_cp)`.
///
/// # Errors
/// [`com_core::Error::Config`] for invalid parameters.
pub fn snr(ce: &CeOfdm, esn0_db: f64) -> Result<f64> {
    ce.check()?;
    let (k, n) = (ce.n_subcarriers, ce.n_fft());
    Ok((TAU * ce.h).powi(2) * n as f64 / (k as f64 * ce.noise_variance(esn0_db)))
}

/// `E_b/N_0` (dB) of a constellation of `m` points at the SNR `snr` (linear).
fn ebn0_db(snr: f64, m: usize) -> f64 {
    com_math::lin2db((snr / (m as f64).log2()).max(1e-300))
}

/// The SER of a constellation of `m` points of `kind` over AWGN at the SNR of the
/// linear regime ([`com_mapping::theory::ser`]).
///
/// # Errors
/// As [`snr`] and [`com_mapping::theory::ser`].
pub fn ser_esn0(ce: &CeOfdm, kind: ConstellationKind, m: usize, esn0_db: f64) -> Result<f64> {
    com_mapping::theory::ser(kind, m, ebn0_db(snr(ce, esn0_db)?, m))
}

/// The BER of a constellation at the SNR `snr` (linear): the exact Gray BER where it
/// exists ([`com_mapping::theory::ber_exact`]), else the nearest-neighbour
/// approximation of [`com_mapping::theory::ber`].
fn ber_at(snr: f64, m: usize, kind: ConstellationKind) -> Result<f64> {
    let e = ebn0_db(snr, m);
    com_mapping::theory::ber_exact(kind, m, e).or_else(|_| com_mapping::theory::ber(kind, m, e))
}

/// The BER of a Gray-mapped constellation of `m` points of `kind` over AWGN at the
/// SNR of the linear regime: the exact Gray BER where it exists
/// ([`com_mapping::theory::ber_exact`]; for a square QAM, eq. (15) of Thompson et al.
/// with its further neighbours), else the nearest-neighbour approximation of
/// [`com_mapping::theory::ber`].
///
/// # Errors
/// As [`snr`] and [`com_mapping::theory::ber`].
pub fn ber_esn0(ce: &CeOfdm, kind: ConstellationKind, m: usize, esn0_db: f64) -> Result<f64> {
    ber_at(snr(ce, esn0_db)?, m, kind)
}

/// The normalised autocorrelation $\rho\[m\]$, $m = 0 \dots N - 1$, of the OFDM signal of
/// a block (see the module): $\rho\[0\] = 1$.
///
/// # Errors
/// [`com_core::Error::Config`] for invalid parameters.
pub fn phase_autocorrelation(ce: &CeOfdm) -> Result<Vec<f64>> {
    ce.check()?;
    let (k, n) = (ce.n_subcarriers, ce.n_fft());
    Ok((0..n).map(|m| (1..=k).map(|i| (TAU * (i * m % n) as f64 / n as f64).cos()).sum::<f64>() / k as f64).collect())
}

/// The real DFT of an even circular sequence.
fn circular_spectrum(r: &[f64]) -> Vec<f64> {
    fft::fft(&r.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>()).into_iter().map(|v| v.re).collect()
}

/// The expected periodogram $\mathbb{E}|S_{k}|^{2}/N$, $k = 0 \dots N - 1$, of the
/// transmitted block $s_{n} = e^{j2\pi h x_{n}}$ (no prefix; $S$ its DFT) under the
/// Gaussian model: the DFT of $e^{-\varphi_{0}^{2}(1 - \rho\[m\])}$ (see the module). The
/// bins average to 1, the unit envelope; bin 0 holds the carrier line
/// $N e^{-\varphi_{0}^{2}}$ ([`imp_powers`]) and the low-frequency distortion.
///
/// # Errors
/// [`com_core::Error::Config`] for invalid parameters.
///
/// # Examples
/// ```
/// use com_waveform::ce_ofdm::{CeOfdm, theory};
/// let ce = CeOfdm::new(16, 4, 0.5 / std::f64::consts::TAU).unwrap();
/// let s = theory::psd(&ce).unwrap();
/// assert!((s.iter().sum::<f64>() / s.len() as f64 - 1.0).abs() < 1e-12);
/// ```
pub fn psd(ce: &CeOfdm) -> Result<Vec<f64>> {
    let rho = phase_autocorrelation(ce)?;
    let v = (TAU * ce.h).powi(2);
    Ok(circular_spectrum(&rho.iter().map(|r| (-v * (1.0 - r)).exp()).collect::<Vec<_>>()))
}

/// The powers $P_{i} = e^{-\varphi_{0}^{2}} \varphi_{0}^{2i}/i!$, $i = 0 \dots n - 1$, of
/// the intermodulation products of the phase modulator at $\varphi_{0} = 2\pi h$
/// (`two_pi_h`): $P_{0}$ is the power of the carrier line, $P_{1}$ that of the useful
/// term, and the powers sum to 1 (eqs. (12)–(14) of Guerreiro et al. 2015 in closed
/// form; the weights of eq. (2) of Pacheco and Hatzinakos 2006).
///
/// # Examples
/// ```
/// use com_waveform::ce_ofdm::theory::imp_powers;
/// let p = imp_powers(1.0, 40);
/// assert!((p[0] - (-1f64).exp()).abs() < 1e-15 && (p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// ```
pub fn imp_powers(two_pi_h: f64, n: usize) -> Vec<f64> {
    let v = two_pi_h * two_pi_h;
    let mut p = Vec::with_capacity(n);
    let mut term = (-v).exp();
    for i in 0..n {
        p.push(term);
        term *= v / (i + 1) as f64;
    }
    p
}

/// The occupied bandwidth $B_{y}/(2B_{m})$ of a carrier phase-modulated by a Gaussian
/// message of standard deviation `sigma` (rad) with a flat spectrum on $[-B_{m}, B_{m}]$:
/// the band $[-z, z]$ that holds the fraction `fraction` of the power once the carrier
/// line is removed, divided by $2B_{m}$ (R. A. Pacheco and D. Hatzinakos, "Error rate
/// analysis of phase-modulated OFDM (OFDM-PM) in AWGN channels," *Proc. IEEE ICASSP*,
/// 2006, eqs. (2)–(5) and Fig. 1, where the band holds 99 %). The normalised spectrum
/// is $\sum_{i \ge 1} P_{i} S_{m}^{*i}/(1 - P_{0})$ ([`imp_powers`]); the $i$-fold
/// convolutions of the flat spectrum are computed on a grid of $B_{m}/400$, where the
/// paper approximates them by a Gaussian for $i \ge 3$. It tends to `fraction` for a
/// small `sigma` (narrowband phase modulation, $B_{y} = 2B_{m}$) and grows as `sigma`.
///
/// # Errors
/// [`com_core::Error::Config`] unless `sigma > 0` and `0 < fraction < 1`.
///
/// # Examples
/// ```
/// use com_waveform::ce_ofdm::theory::occupied_bandwidth;
/// // σ = 1.1 doubles the band of the message (Pacheco and Hatzinakos 2006, Sec. 2.1.1)
/// let b = occupied_bandwidth(1.1, 0.99).unwrap();
/// assert!((b - 2.0).abs() < 0.15, "{b}");
/// ```
pub fn occupied_bandwidth(sigma: f64, fraction: f64) -> Result<f64> {
    ensure(sigma > 0.0 && sigma.is_finite(), || format!("the standard deviation must be positive, got {sigma}"))?;
    ensure(fraction > 0.0 && fraction < 1.0, || format!("the fraction must lie in (0, 1), got {fraction}"))?;
    let v = sigma * sigma;
    // the Poisson weights of the products i ≥ 1, up to a negligible tail
    let n_terms = (v + 12.0 * sigma + 30.0).ceil() as usize;
    let p = imp_powers(sigma, n_terms + 1);
    let norm = 1.0 - p[0];
    // the flat spectrum on [-1, 1] as 2q + 1 equal atoms at the multiples of 1/q
    let q = 400usize;
    let width = 2 * q + 1;
    let mut density = vec![1.0 / width as f64; width];
    // cdf[j]: the fraction of the power within |f| ≤ j/q, accumulated over i
    let mut cdf = vec![0.0; n_terms * q + 1];
    for (i, &pi) in p.iter().enumerate().skip(1) {
        if i > 1 {
            // convolve with the flat spectrum: a moving sum of `width` atoms
            let mut next = vec![0.0; density.len() + width - 1];
            let mut prefix = vec![0.0; density.len() + 1];
            for (j, &d) in density.iter().enumerate() {
                prefix[j + 1] = prefix[j] + d;
            }
            for (j, out) in next.iter_mut().enumerate() {
                let hi = (j + 1).min(density.len());
                let lo = (j + 1).saturating_sub(width);
                *out = (prefix[hi] - prefix[lo]) / width as f64;
            }
            density = next;
        }
        // density[c] sits at (c - centre)/q, with centre = i q
        let centre = i * q;
        let mut inside = density[centre];
        for (j, slot) in cdf.iter_mut().enumerate() {
            if j > 0 && j <= centre {
                inside += density[centre - j] + density[centre + j];
            }
            *slot += pi / norm * inside;
        }
    }
    let j = cdf.iter().position(|&c| c >= fraction).unwrap_or(cdf.len() - 1);
    // interpolate between the grid points j - 1 and j
    let z = if j == 0 { 0.0 } else { (j - 1) as f64 + (fraction - cdf[j - 1]) / (cdf[j] - cdf[j - 1]) };
    Ok(z / q as f64)
}

/// The optimum asymptotic gain $G$ of CE-OFDM over linear OFDM in AWGN (J. Guerreiro, R.
/// Dinis and P. Montezuma, *Proc. IEEE VTC Fall*, 2015, eq. (20); *IEEE Commun. Lett.*
/// 20(11), 2016, eq. (7) with $\mu = 1$): the squared distance between two blocks that
/// differ in one bit, relative to linear OFDM,
///
/// $$G = \sigma^{2} \frac{\mathbb{E}|f'(s)|^{2}}{\mathbb{E}|f(s)|^{2}} = (2\pi h)^{2}$$
///
/// for $f(s) = e^{j2\pi h s/\sigma}$, and $(2\pi h)^{2}
/// \operatorname{erf}(\theta_{\max}/(\sqrt{2} \cdot 2\pi h))$ when the phase is clipped
/// to $[-\theta_{\max}, \theta_{\max}]$ (`theta_max`; Fig. 7 of the 2015 paper).
///
/// # Examples
/// ```
/// use com_waveform::ce_ofdm::theory::optimum_gain;
/// // 1.9 dB at 2πh = 1.25 and 3.5 dB at 1.5 (Guerreiro et al. 2016, Sec. III-A)
/// assert!((com_math::lin2db(optimum_gain(1.25, None)) - 1.94).abs() < 0.01);
/// assert!((com_math::lin2db(optimum_gain(1.5, None)) - 3.52).abs() < 0.01);
/// ```
pub fn optimum_gain(two_pi_h: f64, theta_max: Option<f64>) -> f64 {
    let g = two_pi_h * two_pi_h;
    match theta_max {
        None => g,
        Some(t) => g * erf(t / (std::f64::consts::SQRT_2 * two_pi_h)),
    }
}

/// The optimum asymptotic BER of QPSK CE-OFDM in AWGN,
/// $Q(\sqrt{G \cdot 2E_{b}/N_{0}})$ with $G$ = [`optimum_gain`] (Guerreiro et al. 2016,
/// eq. (8)): the linear theory of the arctangent receiver ([`ber_esn0`] for 4-QAM).
pub fn ber_optimum(two_pi_h: f64, ebn0_db: f64) -> f64 {
    qfunc((optimum_gain(two_pi_h, None) * 2.0 * com_math::db2lin(ebn0_db)).sqrt())
}

/// The change $\Delta\hat{d}_{k}$, $k = 1 \dots K$, of the estimates of the arctangent
/// receiver caused by a cycle slip, a step of $+2\pi$ of the demodulated phase from
/// sample `start` (counted from the end of the prefix) to the end of the block (see the
/// module). Its real part is the effect on the cosine subcarrier and minus its imaginary
/// part on the sine subcarrier, eqs. (13)–(14) of Ahmed et al. (2012) for a large $N$.
///
/// # Errors
/// [`com_core::Error::Config`] for invalid parameters or `start ≥ N`.
pub fn cycle_slip_response(ce: &CeOfdm, start: usize) -> Result<Vec<C64>> {
    ce.check()?;
    let (k, n) = (ce.n_subcarriers, ce.n_fft());
    ensure(start < n, || format!("the slip must start within the block of {n} samples, got {start}"))?;
    let scale = (2.0 * k as f64).sqrt() / (n as f64 * ce.h);
    Ok((1..=k)
        .map(|i| {
            let w = |m: usize| com_math::cis(-TAU * (i * m % n) as f64 / n as f64);
            scale * (w(start) - 1.0) / (1.0 - w(1))
        })
        .collect())
}

/// Bussgang's gain $\alpha$ of the receiver (see the module): 1 for the arctangent
/// receiver, $e^{-\varphi_{0}^{2}/2}$ for the basic linear receiver and
/// $2e^{-\varphi_{0}^{2}/2} - e^{-2\varphi_{0}^{2}}$ for the enhanced one.
pub fn receiver_gain(ce: &CeOfdm, receiver: Receiver) -> f64 {
    let v = (TAU * ce.h).powi(2);
    match receiver {
        Receiver::Arctangent(_) => 1.0,
        Receiver::Linear => (-v / 2.0).exp(),
        Receiver::Enhanced => 2.0 * (-v / 2.0).exp() - (-2.0 * v).exp(),
    }
}

/// $\mathbb{E}[\sin a\varphi_{1} \sin b\varphi_{2}]$ for jointly Gaussian phases of
/// variance `v` and correlation `rho`.
fn sin_sin(a: f64, b: f64, v: f64, rho: f64) -> f64 {
    (-(a * a + b * b) * v / 2.0).exp() * (a * b * rho * v).sinh()
}

/// The SNR (linear) of every data subcarrier $k = 1 \dots K$ at `E_s/N_0` (dB) per data
/// symbol, with the noise and the distortion of the receiver of `ce` (see the module): flat,
/// [`snr`], for the arctangent receiver above its threshold; Bussgang's analysis under
/// the Gaussian model for the linear receivers of Ahmed and Zeidler (2015), whose
/// distortion falls mostly on the middle of the band.
///
/// # Errors
/// [`com_core::Error::Config`] for invalid parameters.
pub fn subcarrier_snr(ce: &CeOfdm, esn0_db: f64) -> Result<Vec<f64>> {
    let receiver = ce.receiver;
    let flat = snr(ce, esn0_db)?;
    let (k, n) = (ce.n_subcarriers, ce.n_fft());
    let v = (TAU * ce.h).powi(2);
    let alpha = receiver_gain(ce, receiver);
    let sigma2 = ce.noise_variance(esn0_db);
    let rho = phase_autocorrelation(ce)?;
    let (noise, signal): (f64, Box<dyn Fn(f64) -> f64>) = match receiver {
        Receiver::Arctangent(_) => return Ok(vec![flat; k]),
        // u = sin φ + Im w
        Receiver::Linear => (sigma2 / 2.0, Box::new(move |r| sin_sin(1.0, 1.0, v, r))),
        // u = (sin φ + w_s)(2 - cos φ - w_c) = 2 sin φ - ½ sin 2φ + noise
        Receiver::Enhanced => (
            sigma2 / 2.0 * (5.0 - 4.0 * (-v / 2.0).exp()) + sigma2 * sigma2 / 4.0,
            Box::new(move |r| {
                4.0 * sin_sin(1.0, 1.0, v, r) - 2.0 * sin_sin(1.0, 2.0, v, r) + 0.25 * sin_sin(2.0, 2.0, v, r)
            }),
        ),
    };
    // the autocorrelation of e = u - α φ, then its spectrum
    let re: Vec<f64> = rho
        .iter()
        .enumerate()
        .map(|(m, &r)| signal(r) - alpha * alpha * v * r + if m == 0 { noise } else { 0.0 })
        .collect();
    let se = circular_spectrum(&re);
    Ok((1..=k).map(|i| n as f64 * v * alpha * alpha / (2.0 * k as f64 * se[i])).collect())
}

/// The SER of a constellation of `m` points of `kind` with the receiver of `ce`, averaged over
/// the subcarriers of [`subcarrier_snr`], the distortion taken as Gaussian; for the
/// arctangent receiver, [`ser_esn0`].
///
/// # Errors
/// As [`subcarrier_snr`] and [`com_mapping::theory::ser`].
pub fn ser_receiver(ce: &CeOfdm, kind: ConstellationKind, m: usize, esn0_db: f64) -> Result<f64> {
    let snrs = subcarrier_snr(ce, esn0_db)?;
    let mut total = 0.0;
    for &s in &snrs {
        total += com_mapping::theory::ser(kind, m, ebn0_db(s, m))?;
    }
    Ok(total / snrs.len() as f64)
}

/// The BER of a Gray-mapped constellation of `m` points of `kind` with the receiver of `ce`,
/// averaged over the subcarriers of [`subcarrier_snr`], the distortion taken as
/// Gaussian; for the arctangent receiver, [`ber_esn0`].
///
/// # Errors
/// As [`subcarrier_snr`] and [`com_mapping::theory::ber`].
pub fn ber_receiver(ce: &CeOfdm, kind: ConstellationKind, m: usize, esn0_db: f64) -> Result<f64> {
    let snrs = subcarrier_snr(ce, esn0_db)?;
    let mut total = 0.0;
    for &s in &snrs {
        total += ber_at(s, m, kind)?;
    }
    Ok(total / snrs.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use com_mapping::constellation::Constellation;

    #[test]
    fn the_psd_is_that_of_a_simulated_block() {
        // 2πh = 1.0, K = 32, J = 4: the mean periodogram of random QPSK blocks
        let ce = CeOfdm::new(32, 4, 1.0 / TAU).unwrap();
        let n = ce.n_fft();
        let theory = psd(&ce).unwrap();
        assert!((theory.iter().sum::<f64>() / n as f64 - 1.0).abs() < 1e-12);
        let c = Constellation::qam(4).unwrap().normalized();
        let blocks = 4000;
        let (_, d) = com_mapping::source::symbols(&c, 32 * blocks, None, &mut Rng::new(2)).unwrap();
        let s = ce.modulate(&d).unwrap();
        let mut sim = vec![0.0; n];
        for block in s.chunks(n) {
            for (acc, v) in sim.iter_mut().zip(fft::fft(block)) {
                *acc += v.norm_sqr() / (n * blocks) as f64;
            }
        }
        // the carrier line, the band and the rest: the Gaussian model holds to 1 %
        let band = |v: &[f64], f: &dyn Fn(usize) -> bool| (0..n).filter(|&k| f(k)).map(|k| v[k]).sum::<f64>();
        let inside = |k: usize| k >= 1 && (k <= 32 || k >= n - 32);
        for (name, f) in [
            ("line", &(|k: usize| k == 0) as &dyn Fn(usize) -> bool),
            ("band", &inside),
            ("outside", &|k| k != 0 && !inside(k)),
        ] {
            let (t, m) = (band(&theory, f), band(&sim, f));
            assert!((m / t - 1.0).abs() < 0.03, "{name}: {m} vs {t}");
        }
        // bin by bin above -20 dB of the band: within 4 % mostly, 10 % at the edges of
        // the second-order products, where 32 QPSK symbols are not quite Gaussian
        let top = theory[1];
        for (k, (t, m)) in theory.iter().zip(&sim).enumerate() {
            if *t > 1e-2 * top {
                assert!((m / t - 1.0).abs() < 0.15, "bin {k}: {m} vs {t}");
            }
        }
    }

    #[test]
    fn the_optimum_gain_is_the_distance_of_one_bit() {
        // the squared distance between two blocks that differ in one bit, relative to the
        // energy, against the same ratio for the OFDM signal itself
        let c = Constellation::qam(4).unwrap().normalized();
        let mut rng = Rng::new(7);
        for (two_pi_h, theta_max) in
            [(0.5, None), (1.25, None), (1.5, Some(std::f64::consts::PI)), (1.5, Some(0.9 * std::f64::consts::PI))]
        {
            let ce = CeOfdm::new(31, 4, two_pi_h / TAU).unwrap();
            let phase = |x: f64| {
                let p = two_pi_h * x;
                com_math::cis(theta_max.map_or(p, |t| p.clamp(-t, t)))
            };
            let mut acc = 0.0;
            let trials = 3000;
            for _ in 0..trials {
                let (_, d) = com_mapping::source::symbols(&c, 31, None, &mut rng).unwrap();
                let mut e = d.clone();
                let k = rng.below(31);
                e[k] = C64::new(-e[k].re, e[k].im);
                let (x1, x2) = (ce.ofdm_signal(&d).unwrap(), ce.ofdm_signal(&e).unwrap());
                let dy: f64 = x1.iter().zip(&x2).map(|(a, b)| (phase(*a) - phase(*b)).norm_sqr()).sum();
                let dx: f64 = x1.iter().zip(&x2).map(|(a, b)| (a - b).powi(2)).sum();
                let ex: f64 = x1.iter().map(|a| a * a).sum();
                // |y| = 1: the energy of the CE-OFDM block is N
                acc += (dy / ce.n_fft() as f64) / (dx / ex);
            }
            let g = acc / trials as f64;
            let want = optimum_gain(two_pi_h, theta_max);
            assert!((g / want - 1.0).abs() < 0.03, "2πh = {two_pi_h}, {theta_max:?}: {g} vs {want}");
        }
        // eq. (8) of the 2016 letter is the linear theory of QPSK
        let ce = CeOfdm::new(31, 4, 0.6 / TAU).unwrap();
        for eb in [4.0, 8.0] {
            let lin = ber_esn0(&ce, ConstellationKind::Qam, 4, eb + com_math::lin2db(2.0)).unwrap();
            assert!((ber_optimum(0.6, eb) / lin - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn the_occupied_bandwidth_tends_to_the_message_band_and_grows_as_the_index() {
        assert!((occupied_bandwidth(0.01, 0.99).unwrap() - 0.99).abs() < 3e-3);
        let b: Vec<f64> = [0.5, 1.0, 2.0, 3.0].iter().map(|&s| occupied_bandwidth(s, 0.99).unwrap()).collect();
        assert!(b.windows(2).all(|w| w[1] > w[0]), "{b:?}");
        // Woodward's limit: a Gaussian spectrum of standard deviation σ B_m/√3
        let wide = occupied_bandwidth(6.0, 0.99).unwrap();
        let woodward = 2.5758 * 6.0 / 3f64.sqrt();
        assert!((wide / woodward - 1.0).abs() < 0.03, "{wide} vs {woodward}");
        assert!(occupied_bandwidth(0.0, 0.99).is_err() && occupied_bandwidth(1.0, 1.0).is_err());
    }
}

#[cfg(test)]
mod literature {
    use super::*;
    use crate::ce_ofdm::Receiver;

    /// A. U. Ahmed and J. R. Zeidler, "Novel low-complexity receivers for constant
    /// envelope OFDM," IEEE Trans. Signal Process. 63(17), 2015, eq. (8) (p. 4574): the
    /// series of the enhanced receiver, sin x (2 - cos x) = 2 sin x - ½ sin 2x. Its terms
    /// x, x³/3, -7x⁵/60 and 31x⁷/2520 are right; the coefficient of x⁹ is
    /// (2 - 2⁸)/9! = -127/181440, not the -41/60480 = -123/181440 printed.
    #[test]
    fn ahmed_2015_eq_8_series_of_the_enhanced_receiver() {
        let x: f64 = 0.2;
        let exact = x.sin() * (2.0 - x.cos());
        let head = x + x.powi(3) / 3.0 - 7.0 * x.powi(5) / 60.0 + 31.0 * x.powi(7) / 2520.0;
        // the x¹¹ term, (-2 + 2¹⁰)/11!, removed
        let c11 = 1022.0 / 39_916_800.0;
        let c9 = (exact - head - c11 * x.powi(11)) / x.powi(9);
        assert!((c9 / (-127.0 / 181_440.0) - 1.0).abs() < 1e-3, "{c9}");
        assert!((c9 / (-41.0 / 60_480.0) - 1.0).abs() > 0.02);
    }

    /// Ahmed and Zeidler 2015, eq. (26): the noise of the enhanced receiver, the power of
    /// $n_{s}(2 - \cos x)$, is $N_{0}E_{b}N (1 + v + v^{2}/2 - 7v^{3}/12)$ with
    /// $v = (2\pi h)^{2}$: the series of $\mathbb{E}[(2 - \cos x)^{2}] = 4 - 4e^{-v/2} +
    /// (1 + e^{-2v})/2$, used by `subcarrier_snr`. Eq. (17): the signal of the linear
    /// receiver, $S_{k1} - S_{k2} = S_{k1}(1 - v(N - 1)/(2N))$, is the series of
    /// Bussgang's gain $e^{-v/2}$.
    #[test]
    fn ahmed_2015_eqs_17_and_26_are_the_series_of_the_bussgang_analysis() {
        for v in [1e-3, 1e-2, 0.04] {
            let exact = 4.0 - 4.0 * (-v / 2.0f64).exp() + (1.0 + (-2.0 * v).exp()) / 2.0;
            let series = 1.0 + v + v * v / 2.0 - 7.0 / 12.0 * v.powi(3);
            assert!((exact - series).abs() < v.powi(4), "{v}");
            let ce = CeOfdm::new(512, 2, v.sqrt() / TAU).unwrap();
            let n = 1024.0;
            let gain = receiver_gain(&ce, Receiver::Linear);
            assert!((gain - (1.0 - v * (n - 1.0) / (2.0 * n))).abs() < v * v, "{v}");
        }
    }

    /// A. U. Ahmed, S. C. Thompson, D. W. Chi and J. R. Zeidler, "Subcarrier based
    /// threshold performance enhancement in constant envelope OFDM," Proc. IEEE MILCOM
    /// 2012, eqs. (13)–(14) and Figs. 4–6 ($2\pi h = 0.8$, $C_{N} = \sqrt{2/N}$, $N = 64$
    /// real subcarriers): a positive cycle slip at $t_{l}$ moves the cosine subcarrier $k$
    /// by $-\sin(2\pi k t_{l}/T_{s})/(k\pi h C_{N})$ and the sine subcarrier by
    /// $(\cos(2\pi k t_{l}/T_{s}) - 1)/(k\pi h C_{N})$, whose extremes are
    /// $1/(k\pi h C_{N}) = 14.1/k$ and $-2/(k\pi h C_{N}) = -28.3/k$ (the scales of
    /// Fig. 6). The paper's real subcarriers are $\sqrt{2} \operatorname{Re} d$ and
    /// $-\sqrt{2} \operatorname{Im} d$ of the library's complex ones, with
    /// $C_{N} = 1/\sqrt{K}$.
    #[test]
    fn ahmed_2012_eqs_13_14_and_fig_6_cycle_slips() {
        let k = 32;
        let ce = CeOfdm::new(k, 32, 0.8 / TAU).unwrap();
        let n = ce.n_fft();
        let pi_h_cn = std::f64::consts::PI * ce.h / (k as f64).sqrt();
        let mut extremes = [(0.0f64, 0.0f64); 10];
        for start in 0..n {
            let r = cycle_slip_response(&ce, start).unwrap();
            let t = start as f64 / n as f64;
            for (i, ext) in extremes.iter_mut().enumerate() {
                let kk = (i + 1) as f64;
                let cos_sub = std::f64::consts::SQRT_2 * r[i].re;
                let sin_sub = -std::f64::consts::SQRT_2 * r[i].im;
                let want_cos = -(TAU * kk * t).sin() / (kk * pi_h_cn);
                let want_sin = ((TAU * kk * t).cos() - 1.0) / (kk * pi_h_cn);
                // the paper integrates: the sum differs by O(k/N) of the scale 1/(kπhC_N)
                let tol = 0.01 / (kk * pi_h_cn) * kk;
                assert!((cos_sub - want_cos).abs() < tol && (sin_sub - want_sin).abs() < tol, "k = {kk}, t = {t}");
                *ext = (ext.0.max(cos_sub), ext.1.min(sin_sub));
            }
        }
        // Fig. 6 with N = 64 (C_N = √(2/64)): 14.14 and -28.28 at k = 1
        let scale = 1.0 / (std::f64::consts::PI * 0.8 / TAU * (2.0f64 / 64.0).sqrt());
        assert!((scale - 14.14).abs() < 0.01);
        for (i, (hi, lo)) in extremes.iter().enumerate() {
            let kk = (i + 1) as f64;
            // the grid of t_l misses the extreme by O(k/N)
            assert!(
                (hi * kk * pi_h_cn - 1.0).abs() < 0.02 && (lo * kk * pi_h_cn + 2.0).abs() < 0.04,
                "k = {kk}: {hi} {lo}"
            );
        }
    }

    /// J. Guerreiro, R. Dinis and P. Montezuma, "Optimum performance and spectral
    /// characterization of CE-OFDM signals," Proc. IEEE VTC Fall 2015: eq. (13), the power
    /// of the intermodulation product of order $m$,
    /// $|\int e^{j\varphi_{0}s/\sigma} p(s) H_{m}(s/(\sqrt{2}\sigma)) ds|^{2}/(2^{m} m!)$,
    /// by Gauss–Hermite quadrature against the closed form
    /// $e^{-\varphi_{0}^{2}} \varphi_{0}^{2m}/m!$ of `imp_powers`; Fig. 4, the power
    /// $P_{0}$ of the DC component read at $2\pi h = 0.25 \dots 1.75$ (0.94, 0.785, 0.375,
    /// 0.22, 0.11, 0.05).
    #[test]
    fn guerreiro_2015_eq_13_and_fig_4() {
        let (t, w) = com_math::gauss_hermite(80);
        for phi in [0.5, 1.0, 2.0] {
            let p = imp_powers(phi, 8);
            for (m, pm) in p.iter().enumerate() {
                // ∫ e^{-t²} e^{j√2 φ₀ t} H_m(t) dt / √π
                let mut acc = C64::new(0.0, 0.0);
                for (ti, wi) in t.iter().zip(&w) {
                    let (mut h0, mut h1) = (1.0, 2.0 * ti);
                    let hm = if m == 0 {
                        1.0
                    } else {
                        for j in 1..m {
                            (h0, h1) = (h1, 2.0 * ti * h1 - 2.0 * j as f64 * h0);
                        }
                        h1
                    };
                    acc += wi * hm * com_math::cis(std::f64::consts::SQRT_2 * phi * ti);
                }
                let fact: f64 = (1..=m).map(|j| j as f64).product();
                let quad = acc.norm_sqr() / std::f64::consts::PI / (2f64.powi(m as i32) * fact);
                assert!((quad - pm).abs() < 1e-10, "φ₀ = {phi}, m = {m}: {quad} vs {pm}");
            }
        }
        for (two_pi_h, read) in [(0.25, 0.94), (0.5, 0.785), (1.0, 0.375), (1.25, 0.22), (1.5, 0.11), (1.75, 0.05)] {
            let p0 = imp_powers(two_pi_h, 1)[0];
            assert!((p0 - read).abs() < 0.015, "2πh = {two_pi_h}: {p0} vs {read}");
        }
    }

    /// J. Guerreiro, R. Dinis and P. Montezuma, "On the detection of CE-OFDM signals,"
    /// IEEE Commun. Lett. 20(11), 2016, Sec. III-A: G = (2πh)², "when 2πh = 1.25, G ≈ 1.9
    /// dB and when 2πh = 1.5, G ≈ 3.5 dB"; a loss below 2πh = 1.
    #[test]
    fn guerreiro_2016_optimum_gains() {
        let db = |x: f64| com_math::lin2db(optimum_gain(x, None));
        assert!((db(1.25) - 1.9).abs() < 0.05 && (db(1.5) - 3.5).abs() < 0.05);
        assert!(db(0.99) < 0.0 && db(1.01) > 0.0);
    }

    /// R. A. Pacheco and D. Hatzinakos, "Error rate analysis of phase-modulated OFDM
    /// (OFDM-PM) in AWGN channels," Proc. IEEE ICASSP 2006, Fig. 1 (the "Analysis" curve,
    /// read at $\sigma_{m} = 0.5 \dots 3$: 1.5, 1.88, 2.0, 2.55, 3.2, 3.9, 4.65) and Sec.
    /// 2.1.1 ("$\sigma_{m} = 1.1$ … gives $B_{y}/2B_{x} \approx 2$"). The paper
    /// approximates the $j$-fold convolutions by Gaussians for $j \ge 3$ (eq. (5)); the
    /// library convolves exactly, within 4 % of the figure. (The text gives the variance
    /// of those Gaussians as $jX^{2}/2$; eq. (5) uses $jX^{2}/3$, the right one for a flat
    /// spectrum.)
    #[test]
    fn pacheco_2006_fig_1_bandwidth() {
        for (sigma, read) in [(0.5, 1.5), (1.0, 1.88), (1.1, 2.0), (1.5, 2.55), (2.0, 3.2), (2.5, 3.9), (3.0, 4.65)] {
            let b = occupied_bandwidth(sigma, 0.99).unwrap();
            assert!((b / read - 1.0).abs() < 0.04, "σ = {sigma}: {b} vs {read}");
        }
    }
}
