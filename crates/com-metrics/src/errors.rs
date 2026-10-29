//! Error counts and rates, SNR, EVM and the least-squares gain of a received sequence.
//!
//! - [`bit_errors`], [`symbol_errors`], [`block_errors`] and [`ber`] compare two
//!   sequences directly;
//! - [`snr_db`], [`effective_snr_db`] and [`ls_gain`] relate a received sequence $z$ to
//!   the transmitted one $x$;
//! - [`fast_ber_calc`] and [`evm`] follow the pre-processing of OptiCommPy (see the
//!   crate documentation);
//! - [`ber_to_q_db`] is the Q factor of a bit error rate.
//!
//! # Model
//!
//! With $\rho^{2} = |\sum_{k} z_{k} x_{k}^{*}|^{2} / (\sum_{k} |z_{k}|^{2} \sum_{k} |x_{k}|^{2})$
//! the squared correlation coefficient of the two sequences, the effective SNR after the
//! least-squares gain of $x$ onto $z$ is
//!
//! $$
//! \mathrm{SNR}_{\mathrm{eff}} = \frac{\rho^{2}}{1 - \rho^{2}},
//! $$
//!
//! independent of any scaling of $z$ (Häger & Pfister 2021), and the least-squares gain
//! that maps $z$ onto $x$ is $g = \sum_{k} x_{k} z_{k}^{*} / \sum_{k} |z_{k}|^{2}$.
//!
//! # References
//!
//! 1. C. Häger and H. D. Pfister, "Physics-based deep learning for fiber-optic
//!    communication systems," *IEEE J. Sel. Areas Commun.*, 39(1), 280–294, 2021.

use crate::preprocess;
use com_core::{Result, ensure, same_len};
use com_mapping::constellation::{Constellation, ConstellationKind};
use com_math::{C64, erfcinv, power};

/// Number of positions where two bit sequences differ (over the shorter length).
pub fn bit_errors(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| (*x & 1) != (*y & 1)).count()
}

/// Number of positions where two label sequences differ.
pub fn symbol_errors(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Number of blocks of `block_len` values in error: the blocks of `tx` and `rx` (cut
/// at the same positions, the last one possibly shorter) that differ in at least one
/// value.
///
/// # Errors
/// [`com_core::Error::Shape`] if the sequences have different lengths,
/// [`com_core::Error::Config`] for a block length of zero.
///
/// # Examples
/// ```
/// use com_metrics::block_errors;
/// let tx = [0, 1, 1, 0, 1, 0];
/// let rx = [0, 1, 0, 0, 1, 0];
/// assert_eq!(block_errors(&tx, &rx, 2)?, 1);
/// assert_eq!(block_errors(&tx, &rx, 3)?, 1);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn block_errors(tx: &[u8], rx: &[u8], block_len: usize) -> Result<usize> {
    same_len(tx.len(), rx.len(), "block_errors")?;
    ensure(block_len >= 1, || "block_errors: the block length must be positive".into())?;
    Ok(tx.chunks(block_len).zip(rx.chunks(block_len)).filter(|(a, b)| a != b).count())
}

/// Bit error rate of two equal-length bit sequences (`NaN` if empty).
///
/// # Errors
/// [`com_core::Error::Shape`] if the sequences have different lengths.
pub fn ber(rx_bits: &[u8], tx_bits: &[u8]) -> Result<f64> {
    same_len(rx_bits.len(), tx_bits.len(), "ber")?;
    Ok(if tx_bits.is_empty() { f64::NAN } else { bit_errors(rx_bits, tx_bits) as f64 / tx_bits.len() as f64 })
}

/// Q factor in dB of a bit error rate, $20 \log_{10}\left(\sqrt{2}\, \operatorname{erfc}^{-1}(2 P_{b})\right)$
/// ($+\infty$ for 0).
///
/// # Examples
/// ```
/// assert!((com_metrics::ber_to_q_db(1e-3) - 9.80).abs() < 0.01);
/// assert_eq!(com_metrics::ber_to_q_db(0.0), f64::INFINITY);
/// ```
pub fn ber_to_q_db(ber: f64) -> f64 {
    if ber <= 0.0 {
        return f64::INFINITY;
    }
    20.0 * (std::f64::consts::SQRT_2 * erfcinv(2.0 * ber)).log10()
}

/// Estimated SNR in dB of a received sequence against the transmitted one,
/// $10 \log_{10}\left(P(x) / P(z - x)\right)$ (no normalisation).
///
/// # Errors
/// [`com_core::Error::Shape`] if the sequences have different lengths.
pub fn snr_db(rx: &[C64], tx: &[C64]) -> Result<f64> {
    same_len(rx.len(), tx.len(), "snr_db")?;
    let e: Vec<C64> = rx.iter().zip(tx).map(|(r, t)| r - t).collect();
    Ok(10.0 * (power(tx) / power(&e)).log10())
}

/// The effective SNR (dB) of `z` against `x` after the least-squares gain of `x` onto
/// `z`, $10 \log_{10}\left(\rho^{2} / (1 - \rho^{2})\right)$ (see the module; $\rho^{2} = 0$
/// when a sequence has no energy, and the denominator is floored at the smallest
/// positive number).
///
/// # Examples
/// ```
/// use com_math::C64;
/// let x = [C64::new(1.0, 0.0), C64::new(-1.0, 0.0)];
/// // a gain and a phase do not change it
/// let z: Vec<C64> = x.iter().map(|v| v * C64::new(0.0, 3.0) + C64::new(0.1, 0.0)).collect();
/// let s = com_metrics::effective_snr_db(&z, &x);
/// assert!((s - 10.0 * (9.0 / 0.01f64).log10()).abs() < 1e-9);
/// ```
pub fn effective_snr_db(z: &[C64], x: &[C64]) -> f64 {
    let c: C64 = z.iter().zip(x).map(|(a, b)| a * b.conj()).sum();
    let ez: f64 = z.iter().map(C64::norm_sqr).sum();
    let ex: f64 = x.iter().map(C64::norm_sqr).sum();
    let rho2 = if ez > 0.0 && ex > 0.0 { c.norm_sqr() / (ez * ex) } else { 0.0 };
    10.0 * (rho2 / (1.0 - rho2).max(f64::MIN_POSITIVE)).log10()
}

/// The least-squares complex gain $g = \sum_{k} x_{k} z_{k}^{*} / \sum_{k} |z_{k}|^{2}$
/// that maps `z` onto `x` (it removes a constant gain and phase; 1 when `z` has no
/// energy).
///
/// # Examples
/// ```
/// use com_math::C64;
/// let x = [C64::new(1.0, 1.0), C64::new(-1.0, 0.5)];
/// let z: Vec<C64> = x.iter().map(|v| v * C64::new(0.0, 2.0)).collect();
/// let g = com_metrics::ls_gain(&z, &x);
/// assert!((g - C64::new(0.0, -0.5)).norm() < 1e-15);
/// ```
pub fn ls_gain(z: &[C64], x: &[C64]) -> C64 {
    let num: C64 = x.iter().zip(z).map(|(a, b)| a * b.conj()).sum();
    let den: f64 = z.iter().map(C64::norm_sqr).sum();
    if den > 0.0 { num / den } else { C64::new(1.0, 0.0) }
}

/// The output of [`fast_ber_calc`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BerSerSnr {
    /// Bit error rate.
    pub ber: f64,
    /// Symbol error rate.
    pub ser: f64,
    /// SNR estimate (dB) of the received constellation, $E_{s}/N_{0}$ for a matched
    /// filter.
    pub snr_db: f64,
}

/// BER, SER and SNR of a received symbol sequence (OptiCommPy `fastBERcalc`).
///
/// `c` gives the family and the order; it does not need to be normalised.
///
/// # Errors
/// [`com_core::Error::Shape`] for sequences of different lengths,
/// [`com_core::Error::Config`] for empty ones, [`com_core::Error::Numerical`] for a
/// zero received symbol (phase ambiguity).
pub fn fast_ber_calc(rx: &[C64], tx: &[C64], c: &Constellation) -> Result<BerSerSnr> {
    let (r, t) = preprocess(rx, tx, c.kind())?;
    let snr_db = snr_db(&r, &t)?;
    let cn = c.normalized();
    let (ir, it) = (cn.decide(&r), cn.decide(&t));
    let b = cn.bits_per_symbol();
    let mut bit_err = 0usize;
    let mut sym_err = 0usize;
    for (a, bb) in ir.iter().zip(&it) {
        let d = (a ^ bb).count_ones() as usize;
        bit_err += d;
        sym_err += (d > 0) as usize;
    }
    Ok(BerSerSnr { ber: bit_err as f64 / (b * r.len()) as f64, ser: sym_err as f64 / r.len() as f64, snr_db })
}

/// Error vector magnitude, as a power ratio $E|y - \hat{x}|^{2} / E|\hat{x}|^{2}$
/// (OptiCommPy `calcEVM`).
///
/// The received sequence is normalised to unit power. With the transmitted sequence,
/// $\hat{x}$ is that sequence normalised to unit power and, for QAM and PSK, $y$ is then
/// multiplied by $\operatorname{mean}(\hat{x}/y)$ (OptiCommPy normalises before the
/// rotation here, after it in `fastBERcalc`; the order changes the value through the
/// gain of the rotation). Without it, $\hat{x}$ is the nearest point of the normalised
/// constellation. Multiply the square root by 100 for the EVM in percent.
///
/// # Errors
/// [`com_core::Error::Config`] for an empty sequence or a zero received symbol,
/// [`com_core::Error::Shape`] if `tx` has another length.
pub fn evm(symbols: &[C64], c: &Constellation, tx: Option<&[C64]>) -> Result<f64> {
    ensure(!symbols.is_empty(), || "empty symbol sequence".into())?;
    let mut y = com_dsp::signal::pnorm(symbols);
    let d = match tx {
        Some(t) => {
            same_len(symbols.len(), t.len(), "received and transmitted symbols")?;
            let d = com_dsp::signal::pnorm(t);
            if matches!(c.kind(), ConstellationKind::Qam | ConstellationKind::Psk) {
                let rot = d.iter().zip(&y).map(|(a, b)| a / b).sum::<C64>() / y.len() as f64;
                ensure(rot.is_finite(), || "phase ambiguity: a received symbol is zero".into())?;
                for v in &mut y {
                    *v *= rot;
                }
            }
            d
        }
        None => c.normalized().project(&y),
    };
    let num: f64 = y.iter().zip(&d).map(|(a, b)| (a - b).norm_sqr()).sum();
    Ok(num / d.iter().map(C64::norm_sqr).sum::<f64>())
}
