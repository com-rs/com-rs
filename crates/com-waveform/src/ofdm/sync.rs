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

use super::Ofdm;
use com_core::Result;
use com_core::ensure_shape;
use com_core::rng::Rng;
use com_math::C64;

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

/// The timing metric $M(d)$ for every start $d$ (halves of `half` samples).
pub fn schmidl_cox(r: &[C64], half: usize) -> Vec<f64> {
    (0..=r.len().saturating_sub(2 * half))
        .map(|d| {
            let p: C64 = (0..half).map(|m| r[d + m].conj() * r[d + m + half]).sum();
            let e: f64 = (0..half).map(|m| r[d + m + half].norm_sqr()).sum();
            if e > 0.0 { p.norm_sqr() / (e * e) } else { 0.0 }
        })
        .collect()
}

/// A synchronisation estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// The middle of the plateau of the metric (sample index): inside the cyclic prefix
    /// of the preamble.
    pub timing: usize,
    /// Fractional carrier frequency offset, in subcarrier spacings ($|\varepsilon| < 1$).
    pub cfo: f64,
    /// Peak of the timing metric.
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

#[cfg(test)]
mod tests {
    use super::*;
    use com_math::cis;

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
