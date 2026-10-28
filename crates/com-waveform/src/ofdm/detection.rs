//! MIMO-OFDM: per-subcarrier detection.
//!
//! With a cyclic prefix longer than the channel, every subcarrier $k$ of every symbol
//! sees a flat MIMO channel, $y_{k} = H_{k} x_{k} + w_{k}$, with
//! $H_{k} = \sum_{l} H_{l}\, e^{-j 2\pi k d_{l}/N}$. [`detect`] applies any detector of
//! `com_mimo` to every resource element — a thin wrapper, as the detectors know nothing
//! of the waveform (ADR 0009).
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::detection::detect;
//! use com_mimo::detection::Linear;
//! use com_mapping::constellation::Constellation;
//! use com_math::{linalg::{identity, matvec}, C64};
//! let c = Constellation::qam(4)?.normalized();
//! let h = vec![identity(2); 3];                             // three REs, 2 × 2
//! let x: Vec<Vec<C64>> = (0..3).map(|i| vec![c.points()[i], c.points()[3 - i]]).collect();
//! let y: Vec<Vec<C64>> = h.iter().zip(&x).map(|(h, x)| matvec(&h.view(), x)).collect();
//! let d = detect(&h, &y, 0.01, &Linear::mmse(c))?;
//! assert_eq!(d[1], vec![1, 2]);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. G. L. Stüber, J. R. Barry, S. W. McLaughlin, Y. Li, M. A. Ingram and T. G. Pratt,
//!    "Broadband MIMO-OFDM wireless communications," *Proc. IEEE*, 92(2), 271–294, 2004.

use com_core::{Result, same_len};
use com_math::C64;
use com_math::linalg::CMat;
use com_mimo::detection::MimoDetector;

/// Detect every resource element: `h[i]` and `y[i]` are the channel ($n_{r} \times n_{t}$)
/// and the received vector of RE `i`. Returns the labels of every RE.
///
/// # Errors
/// [`com_core::Error::Shape`] for inconsistent lengths, and the errors of the detector.
pub fn detect(h: &[CMat], y: &[Vec<C64>], noise_var: f64, detector: &dyn MimoDetector) -> Result<Vec<Vec<usize>>> {
    same_len(h.len(), y.len(), "channels and received vectors")?;
    h.iter().zip(y).map(|(h, y)| detector.detect(&h.view(), y, noise_var)).collect()
}
