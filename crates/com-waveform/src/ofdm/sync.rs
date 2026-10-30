//! Timing and carrier-frequency synchronisation of OFDM (Schmidl & Cox 1997).
//!
//! The preamble is an OFDM symbol whose odd subcarriers are empty: its useful part is
//! two identical halves of $L = LN/2$ samples. With the received samples $r$, the
//! correlation and the energy of the second half,
//!
//! $$
//! P(d) = \sum_{m=0}^{L-1} r^{*}[d + m]\, r[d + m + L], \qquad
//! R(d) = \sum_{m=0}^{L-1} |r[d + m + L]|^{2}, \qquad
//! M(d) = \frac{|P(d)|^{2}}{R(d)^{2}},
//! $$
//!
//! the timing metric $M(d)$ reaches a plateau of the length of the cyclic prefix at the
//! start of the preamble; the phase of $P$ there gives the fractional carrier frequency
//! offset, in units of the subcarrier spacing, $\hat\varepsilon = \angle P(\hat d) / \pi$
//! (an offset of $\varepsilon$ rotates the second half by $\pi\varepsilon$).
//!
//! [`schmidl_cox`] returns the metric; [`synchronize`] the timing (the middle of the
//! points above 90 % of the peak, which lies in the cyclic prefix when data follow the
//! preamble) and the offset.
//!
//! # A repeated symbol (Moose 1994)
//!
//! When a block of $N$ = [`Ofdm::fft_len`] samples is sent twice in a row
//! ([`repeated_preamble`]), the DFTs $Y_{1k}$, $Y_{2k}$ of the two received blocks differ by
//! the phase $2\pi\varepsilon$ of the offset alone, and the maximum-likelihood estimate is
//! (Moose, eq. (21))
//!
//! $$
//! \hat\varepsilon = \frac{1}{2\pi} \angle \sum_{k} Y_{2k} Y_{1k}^{*}, \qquad |\varepsilon| < \tfrac{1}{2},
//! $$
//!
//! over the active subcarriers ([`moose`]); for small errors it is unbiased, of variance
//! $1/((2\pi)^{2} E_{s}/N_{0})$ with $E_{s}$ the energy of one block (eq. (26),
//! [`moose_std`]). An offset leaves on subcarrier $k$ the fraction
//! $\operatorname{diric}(2\pi\varepsilon/N, N)^{2}$ of its power and adds the interference
//! of the others (eq. (10), [`ici_power`]):
//!
//! $$
//! \frac{E|I_{k}|^{2}}{|X|^{2}|H|^{2}} = \sum_{l \ne k}
//! \left(\frac{\sin \pi\varepsilon}{N \sin(\pi(l - k + \varepsilon)/N)}\right)^{2}
//! \le 0.5947 \sin^{2}\pi\varepsilon \quad (|\varepsilon| \le \tfrac{1}{2}),
//! $$
//!
//! hence the lower bound of the SNR after the DFT (eq. (15), [`cfo_snr_bound`]),
//! $\mathrm{SNR}(\varepsilon) \ge \mathrm{SNR}\, (\sin\pi\varepsilon/\pi\varepsilon)^{2} /
//! (1 + 0.5947\, \mathrm{SNR} \sin^{2}\pi\varepsilon)$.
//!
//! # The cyclic prefix (van de Beek, Sandell & Börjesson 1997)
//!
//! Without a preamble, the $L$ samples of the prefix and their copies $N$ samples later
//! carry the timing and the offset. Over a window of $2N + L$ samples that contains one
//! whole symbol, the joint maximum-likelihood estimates are (their eqs. (6)–(8),
//! (12)–(13))
//!
//! $$
//! \hat\theta = \arg\max_{\theta} \left\{ |\gamma(\theta)| - \rho\, \Phi(\theta) \right\},
//! \qquad \hat\varepsilon = -\frac{1}{2\pi} \angle \gamma(\hat\theta),
//! $$
//!
//! with $\gamma(\theta) = \sum_{k=\theta}^{\theta+L-1} r(k)\, r^{*}(k + N)$, the energy
//! $\Phi(\theta) = \frac{1}{2} \sum_{k=\theta}^{\theta+L-1} |r(k)|^{2} + |r(k+N)|^{2}$ and
//! $\rho = \mathrm{SNR}/(\mathrm{SNR} + 1)$ ([`cp_ml`]: $\hat\theta$ is the first
//! sample of the prefix).
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::Ofdm;
//! use com_waveform::ofdm::sync::{preamble, synchronize};
//! use com_core::rng::Rng;
//! use com_math::{cis, C64};
//! let ofdm = Ofdm { n_cp: 16, ..Ofdm::new(64, 52)? };
//! let p = preamble(&ofdm, 1)?;                               // with its prefix
//! // silence, the preamble, then a data symbol (random QPSK)
//! let mut r = Rng::new(4);
//! let data: Vec<C64> = (0..52).map(|_| cis(std::f64::consts::FRAC_PI_2 * r.below(4) as f64)).collect();
//! let mut rx = vec![C64::new(0.0, 0.0); 100];
//! rx.extend(&p);
//! rx.extend(ofdm.modulate(&data)?);
//! // an offset of 0.2 subcarrier spacing
//! let m = ofdm.fft_len() as f64;
//! let rx: Vec<C64> = rx.iter().enumerate().map(|(n, v)| v * cis(2.0 * std::f64::consts::PI * 0.2 * n as f64 / m)).collect();
//! let e = synchronize(&rx, &ofdm)?;
//! assert!((e.cfo - 0.2).abs() < 1e-6);
//! assert!(e.timing >= 100 && e.timing <= 100 + 4 * 16);       // within the prefix
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. T. M. Schmidl and D. C. Cox, "Robust frequency and timing synchronization for
//!    OFDM," *IEEE Trans. Commun.*, 45(12), 1613–1621, 1997.
//! 2. P. H. Moose, "A technique for orthogonal frequency division multiplexing frequency
//!    offset correction," *IEEE Trans. Commun.*, 42(10), 2908–2914, 1994.
//! 3. J.-J. van de Beek, M. Sandell and P. O. Börjesson, "ML estimation of time and
//!    frequency offset in OFDM systems," *IEEE Trans. Signal Process.*, 45(7),
//!    1800–1805, 1997.

use super::Ofdm;
use com_core::rng::Rng;
use com_core::{Result, ensure, ensure_shape};
use com_dsp::sync::frame::{delayed_correlation, windowed_energy};
use com_math::{C64, diric, sinc};
use std::f64::consts::TAU;

/// The preamble of Schmidl & Cox (with its cyclic prefix): QPSK on the even active
/// subcarriers (scaled by $\sqrt{2}$ to keep the unit power), nothing on the odd ones.
///
/// # Errors
/// [`com_core::Error::Config`] if the prefix is longer than the block.
pub fn preamble(ofdm: &Ofdm, seed: u64) -> Result<Vec<C64>> {
    let mut r = Rng::new(seed);
    let active: Vec<C64> = (0..ofdm.n_active)
        .map(|i| {
            let s = r.qpsk() * std::f64::consts::SQRT_2;
            if ofdm.subcarrier(i).rem_euclid(2) == 0 { s } else { C64::new(0.0, 0.0) }
        })
        .collect();
    ofdm.with_prefix(&ofdm.to_time(&active)?)
}

/// The timing metric $M(d)$ for every start $d$ (halves of `half` samples), on
/// [`delayed_correlation`] and [`windowed_energy`].
pub fn schmidl_cox(r: &[C64], half: usize) -> Vec<f64> {
    let p = delayed_correlation(r, half, half);
    let e = windowed_energy(r, half);
    p.iter()
        .enumerate()
        .map(|(d, p)| {
            let e = e[d + half];
            if e > 0.0 { p.norm_sqr() / (e * e) } else { 0.0 }
        })
        .collect()
}

/// A synchronisation estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// The timing (sample index): for [`synchronize`], the middle of the plateau of the
    /// metric, inside the cyclic prefix of the preamble; for [`cp_ml`], the first sample
    /// of the cyclic prefix.
    pub timing: usize,
    /// Fractional carrier frequency offset, in subcarrier spacings ([`synchronize`]:
    /// $|\varepsilon| < 1$; [`cp_ml`]: $|\varepsilon| < 1/2$).
    pub cfo: f64,
    /// Peak of the timing metric (for [`cp_ml`], the log-likelihood $|\gamma| - \rho\Phi$).
    pub peak: f64,
}

/// Estimate the timing and the frequency offset of the preamble of `ofdm` in `r`.
///
/// # Errors
/// [`com_core::Error::Shape`] if `r` is shorter than the preamble.
pub fn synchronize(r: &[C64], ofdm: &Ofdm) -> Result<Estimate> {
    let half = ofdm.fft_len() / 2;
    ensure_shape(r.len() >= 2 * half, || {
        format!("the received block of {} samples is shorter than the preamble of {}", r.len(), 2 * half)
    })?;
    let m = schmidl_cox(r, half);
    let (imax, peak) = m.iter().enumerate().fold((0, 0.0), |acc, (i, v)| if *v > acc.1 { (i, *v) } else { acc });
    // the plateau: the points above 90 % of the peak around it
    let thr = 0.9 * peak;
    let mut a = imax;
    while a > 0 && m[a - 1] >= thr {
        a -= 1;
    }
    let mut b = imax;
    while b + 1 < m.len() && m[b + 1] >= thr {
        b += 1;
    }
    let d = (a + b) / 2;
    let p: C64 = (0..half).map(|k| r[d + k].conj() * r[d + k + half]).sum();
    Ok(Estimate { timing: d, cfo: p.arg() / std::f64::consts::PI, peak })
}

/// The constant of Moose's bound of the interference, eq. (13): $E|I_{k}|^{2} \le 0.5947
/// |X|^{2}|H|^{2} \sin^{2}\pi\varepsilon$ for $|\varepsilon| \le 1/2$.
pub const MOOSE_ICI_BOUND: f64 = 0.5947;

/// The preamble of Moose (with its cyclic prefix): one block of QPSK on every active
/// subcarrier, sent twice in a row, $L(N_{\mathrm{cp}} + 2N)$ samples.
///
/// # Errors
/// [`com_core::Error::Config`] if the prefix is longer than the block.
pub fn repeated_preamble(ofdm: &Ofdm, seed: u64) -> Result<Vec<C64>> {
    let mut r = Rng::new(seed);
    let active: Vec<C64> = (0..ofdm.n_active).map(|_| r.qpsk()).collect();
    let block = ofdm.to_time(&active)?;
    let mut out = ofdm.with_prefix(&block)?;
    out.extend_from_slice(&block);
    Ok(out)
}

/// The frequency offset (in subcarrier spacings, $|\varepsilon| < 1/2$) of a block
/// repeated at `start` and `start + N` in `r`, Moose's eq. (21) over the active
/// subcarriers ([`Ofdm::to_active`]).
///
/// # Errors
/// [`com_core::Error::Shape`] if `r` ends before the second block.
pub fn moose(r: &[C64], start: usize, ofdm: &Ofdm) -> Result<f64> {
    let n = ofdm.fft_len();
    ensure_shape(start + 2 * n <= r.len(), || {
        format!("the two blocks of {n} samples from {start} end after the {} received samples", r.len())
    })?;
    let y1 = ofdm.to_active(&r[start..start + n])?;
    let y2 = ofdm.to_active(&r[start + n..start + 2 * n])?;
    let s: C64 = y2.iter().zip(&y1).map(|(b, a)| b * a.conj()).sum();
    Ok(s.arg() / TAU)
}

/// The standard deviation of Moose's estimate for small errors,
/// $1/(2\pi\sqrt{E_{s}/N_{0}})$, with $E_{s}/N_{0}$ (linear) the energy of one block over
/// the noise density (eq. (26)).
pub fn moose_std(es_n0: f64) -> f64 {
    1.0 / (TAU * es_n0.sqrt())
}

/// The power of the interference that a frequency offset `eps` (in subcarrier
/// spacings) brings onto the active subcarrier `i` from the other active subcarriers,
/// relative to the power of one subcarrier: Moose's eq. (10), each term
/// [`com_math::diric`]$(2\pi(l - k + \varepsilon)/N, N)^{2}$.
pub fn ici_power(ofdm: &Ofdm, i: usize, eps: f64) -> f64 {
    let n = ofdm.fft_len();
    let k = ofdm.subcarrier(i);
    (0..ofdm.n_active)
        .filter(|&l| l != i)
        .map(|l| {
            let p = (ofdm.subcarrier(l) - k) as f64;
            diric(TAU * (p + eps) / n as f64, n).powi(2)
        })
        .sum()
}

/// Moose's lower bound of the SNR of a subcarrier after the DFT under a frequency offset
/// `eps` (in subcarrier spacings), for an SNR `snr` (linear, $E_{c}/N_{0}$; infinite for
/// the interference alone), eq. (15):
/// $\mathrm{SNR}\,(\sin\pi\varepsilon/\pi\varepsilon)^{2} / (1 + 0.5947\, \mathrm{SNR}
/// \sin^{2}\pi\varepsilon)$ (linear).
pub fn cfo_snr_bound(snr: f64, eps: f64) -> f64 {
    let s = (std::f64::consts::PI * eps).sin();
    sinc(eps).powi(2) / (1.0 / snr + MOOSE_ICI_BOUND * s * s)
}

/// The joint maximum-likelihood timing and frequency offset of van de Beek, Sandell &
/// Börjesson (1997) from the cyclic prefix, for an SNR `snr` (linear) per sample: over
/// every start $\theta$ such that the prefix and its copy lie in `r` (a window of
/// $2N + L$ samples that contains one whole symbol), the maximum of $|\gamma(\theta)| -
/// \rho\Phi(\theta)$ ([`delayed_correlation`] and [`windowed_energy`]); the timing is the
/// first sample of the prefix.
///
/// # Errors
/// [`com_core::Error::Config`] without a cyclic prefix or for a non-positive SNR,
/// [`com_core::Error::Shape`] if `r` is shorter than one symbol.
pub fn cp_ml(r: &[C64], ofdm: &Ofdm, snr: f64) -> Result<Estimate> {
    let (n, l) = (ofdm.fft_len(), ofdm.oversampling * ofdm.n_cp);
    ensure(l > 0, || "the estimator needs a cyclic prefix".into())?;
    ensure(snr > 0.0, || format!("the SNR must be positive, got {snr}"))?;
    ensure_shape(r.len() >= n + l, || format!("{} samples are shorter than one symbol of {}", r.len(), n + l))?;
    let rho = snr / (snr + 1.0);
    // P(θ) = Σ r*(k) r(k + N) = γ(θ)*
    let p = delayed_correlation(r, n, l);
    let e = windowed_energy(r, l);
    let (theta, peak) = p.iter().enumerate().fold((0, f64::NEG_INFINITY), |best, (t, pt)| {
        let lambda = pt.norm() - rho * 0.5 * (e[t] + e[t + n]);
        if lambda > best.1 { (t, lambda) } else { best }
    });
    Ok(Estimate { timing: theta, cfo: p[theta].arg() / TAU, peak })
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_math::cis;

    /// `r` rotated by an offset of `eps` subcarrier spacings on a grid of `n`.
    fn offset(r: &[C64], eps: f64, n: usize) -> Vec<C64> {
        r.iter().enumerate().map(|(k, v)| v * cis(TAU * eps * k as f64 / n as f64)).collect()
    }

    #[test]
    fn schmidl_cox_on_the_shared_kernel_is_the_former_metric() {
        let mut rng = Rng::new(9);
        let r: Vec<C64> = (0..300).map(|_| rng.complex_gaussian()).collect();
        let former: Vec<f64> = (0..=r.len() - 2 * 64)
            .map(|d| {
                let p: C64 = (0..64).map(|m| r[d + m].conj() * r[d + m + 64]).sum();
                let e: f64 = (0..64).map(|m| r[d + m + 64].norm_sqr()).sum();
                if e > 0.0 { p.norm_sqr() / (e * e) } else { 0.0 }
            })
            .collect();
        assert_eq!(schmidl_cox(&r, 64), former);
    }

    #[test]
    fn moose_and_the_cyclic_prefix_are_exact_without_noise() {
        let ofdm = Ofdm { n_cp: 8, oversampling: 1, ..Ofdm::new(64, 52).unwrap() };
        let n = ofdm.fft_len();
        let p = repeated_preamble(&ofdm, 3).unwrap();
        assert_eq!(p.len(), 8 + 2 * n);
        for eps in [-0.45, -0.1, 0.0, 0.23, 0.49] {
            assert!((moose(&offset(&p, eps, n), 8, &ofdm).unwrap() - eps).abs() < 1e-12, "{eps}");
        }
        // a symbol inside random samples: the prefix is found exactly
        let mut rng = Rng::new(5);
        let data: Vec<C64> = (0..52).map(|_| rng.qpsk()).collect();
        let mut r: Vec<C64> = (0..30).map(|_| rng.complex_gaussian() * 0.01).collect();
        r.extend(ofdm.modulate(&data).unwrap());
        r.extend((0..n + 8 - 30).map(|_| rng.complex_gaussian() * 0.01));
        let e = cp_ml(&offset(&r, 0.2, n), &ofdm, 1e4).unwrap();
        assert_eq!(e.timing, 30);
        assert!((e.cfo - 0.2).abs() < 1e-3, "{}", e.cfo);
        assert!(cp_ml(&r, &Ofdm { n_cp: 0, ..ofdm }, 10.0).is_err() && cp_ml(&r[..60], &ofdm, 10.0).is_err());
        assert!(moose(&p, 9, &ofdm).is_err());
    }

    #[test]
    fn the_interference_stays_below_moose_bound() {
        // N = 256, 2K + 1 = 193 carriers (Moose, Sec. II)
        let ofdm = Ofdm { oversampling: 1, ..Ofdm::new(256, 193).unwrap() };
        for eps in [0.05, 0.25, 0.5] {
            let s2 = (std::f64::consts::PI * eps).sin().powi(2);
            let worst = (0..193).map(|i| ici_power(&ofdm, i, eps)).fold(0.0, f64::max);
            assert!(worst <= MOOSE_ICI_BOUND * s2 && worst > 0.0, "{eps}: {worst}");
        }
        assert!(ici_power(&ofdm, 7, 0.0) < 1e-25); // zero up to the rounding of sin(πp)
        assert!((cfo_snr_bound(100.0, 0.0) - 100.0).abs() < 1e-12);
        assert!(cfo_snr_bound(f64::INFINITY, 0.1).is_finite());
    }

    #[test]
    fn the_preamble_has_two_identical_halves_and_noise_hurts_gracefully() {
        let ofdm = Ofdm { n_cp: 16, ..Ofdm::new(128, 100).unwrap() };
        let p = preamble(&ofdm, 2).unwrap();
        let cp = 4 * 16;
        let half = ofdm.fft_len() / 2;
        for i in 0..half {
            assert!((p[cp + i] - p[cp + half + i]).norm() < 1e-12);
        }
        // at 10 dB the offset error is small
        let mut rng = Rng::new(3);
        let mut err = 0.0;
        for t in 0..100 {
            let mut rx: Vec<C64> = (0..200).map(|_| rng.complex_gaussian() * 0.1f64.sqrt()).collect();
            let eps = -0.3 + 0.006 * f64::from(t);
            rx.extend(p.iter().map(|v| v + rng.complex_gaussian() * 0.1f64.sqrt()));
            // a data symbol follows the preamble
            rx.extend((0..ofdm.block_len()).map(|_| rng.complex_gaussian() * 1.1f64.sqrt()));
            let m = ofdm.fft_len() as f64;
            let rx: Vec<C64> =
                rx.iter().enumerate().map(|(n, v)| v * cis(2.0 * std::f64::consts::PI * eps * n as f64 / m)).collect();
            let e = synchronize(&rx, &ofdm).unwrap();
            assert!(e.timing >= 200 && e.timing <= 200 + cp + 8, "timing {}", e.timing);
            err += (e.cfo - eps).powi(2);
        }
        assert!((err / 100.0).sqrt() < 0.01);
    }
}
