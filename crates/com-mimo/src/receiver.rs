//! The receiver of a link: one detector applied channel use after channel use, with
//! soft outputs.
//!
//! [`Receiver`] detects the layers of every channel use $y = H x + w$ (spatial
//! multiplexing) or every block of Alamouti's code, and accumulates a [`SoftOutput`]:
//!
//! - the estimate of every transmitted symbol: the unbiased linear estimate with ZF and
//!   MMSE, the decided point with the other detectors;
//! - its noise variance: that of the linear estimate, or the noise variance of the
//!   channel for a decision;
//! - optionally the bit LLRs of the detector (ZF, MMSE, ML, K-best, Alamouti).
//!
//! # Examples
//! ```
//! use com_mimo::receiver::{DetectorKind, Receiver, SoftOutput};
//! use com_mapping::constellation::Constellation;
//! use com_math::C64;
//! use ndarray::array;
//! let c = Constellation::qam(4)?.normalized();
//! let rx = Receiver::new(DetectorKind::Ml, c.clone(), true)?;
//! let h = array![[C64::new(1.0, 0.0), C64::new(0.2, 0.1)], [C64::new(-0.3, 0.0), C64::new(0.9, 0.0)]];
//! let x = [c.points()[1], c.points()[2]];
//! let y: Vec<C64> = (0..2).map(|r| h[[r, 0]] * x[0] + h[[r, 1]] * x[1]).collect();
//! let mut out = SoftOutput::default();
//! rx.layers(&h, &y, 1e-3, &mut out)?;
//! assert_eq!(out.symbols, x.to_vec());
//! assert_eq!(out.llr.len(), 4);
//! # Ok::<(), com_core::Error>(())
//! ```

use crate::detection::{KBest, Linear, MimoDetector, Ml, Sic, Sphere};
use crate::message_passing::{Amp, Oamp};
use crate::stbc::Ostbc;
use com_core::{Error, Result, ensure};
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_math::linalg::CMat;
use ndarray::Array2;

/// A detector of [`crate::detection`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectorKind {
    /// Zero forcing.
    Zf,
    /// Unbiased MMSE.
    Mmse,
    /// MMSE successive interference cancellation (V-BLAST).
    Sic,
    /// Zero-forcing successive interference cancellation.
    ZfSic,
    /// Exhaustive maximum likelihood.
    Ml,
    /// The sphere decoder (maximum likelihood).
    Sphere,
    /// K-best with this number of paths.
    KBest(usize),
    /// LAMA, the complex Bayesian AMP of large systems, with this number of iterations.
    Amp(usize),
    /// OAMP with the LMMSE estimator, with this number of iterations.
    Oamp(usize),
}

impl std::str::FromStr for DetectorKind {
    type Err = Error;

    /// The detector of a name: `zf`, `mmse`, `sic` (`mmse_sic`), `zf_sic`, `ml`,
    /// `sphere`, `kbest` (with [`DetectorKind::DEFAULT_PATHS`] paths), `amp` (`lama`)
    /// or `oamp` (with [`DetectorKind::DEFAULT_ITERATIONS`] iterations),
    /// case-insensitive.
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    fn from_str(name: &str) -> Result<DetectorKind> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "zf" => DetectorKind::Zf,
            "mmse" => DetectorKind::Mmse,
            "sic" | "mmse_sic" => DetectorKind::Sic,
            "zf_sic" => DetectorKind::ZfSic,
            "ml" => DetectorKind::Ml,
            "sphere" => DetectorKind::Sphere,
            "kbest" => DetectorKind::KBest(DetectorKind::DEFAULT_PATHS),
            "amp" | "lama" => DetectorKind::Amp(DetectorKind::DEFAULT_ITERATIONS),
            "oamp" => DetectorKind::Oamp(DetectorKind::DEFAULT_ITERATIONS),
            other => {
                return Err(Error::Config(format!(
                    "unknown detector {other:?}: zf, mmse, sic, zf_sic, ml, sphere, kbest, amp or oamp"
                )));
            }
        })
    }
}

impl DetectorKind {
    /// The paths of K-best when its name does not give them.
    pub const DEFAULT_PATHS: usize = 8;

    /// The iterations of AMP and OAMP when their name does not give them.
    pub const DEFAULT_ITERATIONS: usize = 10;

    /// The same detector with `iterations` iterations, for AMP and OAMP (the others are
    /// returned unchanged).
    ///
    /// # Errors
    /// [`Error::Config`] for zero iterations of AMP or OAMP.
    pub fn with_iterations(self, iterations: usize) -> Result<DetectorKind> {
        let needs = matches!(self, DetectorKind::Amp(_) | DetectorKind::Oamp(_));
        ensure(!needs || iterations >= 1, || format!("{self:?} needs at least one iteration"))?;
        Ok(match self {
            DetectorKind::Amp(_) => DetectorKind::Amp(iterations),
            DetectorKind::Oamp(_) => DetectorKind::Oamp(iterations),
            other => other,
        })
    }

    /// The detector of a name ([`DetectorKind::from_str`](std::str::FromStr::from_str)),
    /// K-best with `k` paths.
    ///
    /// # Errors
    /// [`Error::Config`] for another name or zero paths.
    pub fn from_name(name: &str, k: usize) -> Result<DetectorKind> {
        Ok(match name.parse()? {
            DetectorKind::KBest(_) => {
                ensure(k >= 1, || format!("kbest needs at least one path, got k = {k}"))?;
                DetectorKind::KBest(k)
            }
            other => other,
        })
    }

    /// Whether the detector computes bit LLRs.
    pub fn has_soft_output(self) -> bool {
        !matches!(self, DetectorKind::Sic | DetectorKind::ZfSic | DetectorKind::Sphere)
    }

    /// The detector for the alphabet `c`.
    pub fn detector(self, c: &Constellation) -> Box<dyn MimoDetector + Send + Sync> {
        let c = c.clone();
        match self {
            DetectorKind::Zf => Box::new(Linear::zf(c)),
            DetectorKind::Mmse => Box::new(Linear::mmse(c)),
            DetectorKind::Sic => Box::new(Sic::mmse(c)),
            DetectorKind::ZfSic => Box::new(Sic::zf(c)),
            DetectorKind::Ml => Box::new(Ml::new(c)),
            DetectorKind::Sphere => Box::new(Sphere::new(c)),
            DetectorKind::KBest(k) => Box::new(KBest::new(c, k)),
            DetectorKind::Amp(n) => Box::new(Amp::new(c, n)),
            DetectorKind::Oamp(n) => Box::new(Oamp::new(c, n)),
        }
    }
}

/// The accumulated output of a [`Receiver`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoftOutput {
    /// The estimate of every symbol.
    pub symbols: Vec<C64>,
    /// The noise variance of every estimate.
    pub noise_var: Vec<f64>,
    /// The bit LLRs (positive: bit 0), when asked.
    pub llr: Vec<f64>,
}

/// A detector applied channel use after channel use (see the module).
pub struct Receiver {
    kind: DetectorKind,
    constellation: Constellation,
    detector: Box<dyn MimoDetector + Send + Sync>,
    linear: Option<Linear>,
    llr: bool,
}

impl std::fmt::Debug for Receiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Receiver").field("kind", &self.kind).field("llr", &self.llr).finish_non_exhaustive()
    }
}

impl Receiver {
    /// The receiver of the detector `kind` for the alphabet `c`, with bit LLRs if `llr`.
    ///
    /// # Errors
    /// [`Error::Config`] when LLRs are asked from a detector without soft output.
    pub fn new(kind: DetectorKind, c: Constellation, llr: bool) -> Result<Receiver> {
        ensure(!llr || kind.has_soft_output(), || {
            "the sic and sphere detectors have no soft output: demap their decisions instead".into()
        })?;
        let linear = match kind {
            DetectorKind::Zf => Some(Linear::zf(c.clone())),
            DetectorKind::Mmse => Some(Linear::mmse(c.clone())),
            _ => None,
        };
        Ok(Receiver { kind, detector: kind.detector(&c), constellation: c, linear, llr })
    }

    /// Detect the layers of one channel use $y = H x + w$ and append them to `out`.
    ///
    /// # Errors
    /// The errors of the detector.
    pub fn layers(&self, h: &CMat, y: &[C64], noise_var: f64, out: &mut SoftOutput) -> Result<()> {
        let hv = h.view();
        if let Some(lin) = &self.linear {
            // the LLRs of a linear detector come from its own estimates: filter once
            let (z, var) = lin.estimate(&hv, y, noise_var)?;
            if self.llr {
                out.llr.extend(self.constellation.llr_with_variances(&z, &var, None, false)?);
            }
            out.symbols.extend(z);
            out.noise_var.extend(var);
        } else {
            let labels = self.detector.detect(&hv, y, noise_var)?;
            out.symbols.extend(labels.iter().map(|&i| self.constellation.points()[i]));
            out.noise_var.extend(std::iter::repeat_n(noise_var, labels.len()));
            if self.llr {
                out.llr.extend(self.detector.llr(&hv, y, noise_var)?);
            }
        }
        Ok(())
    }

    /// Detect the channel uses `uses` in turn and append them to `out`: use `i` has the
    /// channel `channel(i)` ($n_{r} \times n_{t}$) and the received vector
    /// `received(i)` ($n_{r}$); with `alamouti`, the pairs of consecutive uses carry the
    /// blocks of Alamouti's code, detected with the channel of the first use of the pair
    /// ([`Receiver::decode_alamouti`]), otherwise every use carries one vector of layers
    /// ([`Receiver::layers`]).
    ///
    /// # Errors
    /// The errors of `channel` and of the detector; [`Error::Shape`] for an odd number of
    /// uses with Alamouti's code.
    ///
    /// # Examples
    /// ```
    /// use com_mimo::receiver::{DetectorKind, Receiver, SoftOutput};
    /// use com_mapping::constellation::Constellation;
    /// use com_math::{C64, linalg::identity};
    /// let c = Constellation::qam(4)?.normalized();
    /// let rx = Receiver::new(DetectorKind::Zf, c.clone(), false)?;
    /// let x: Vec<Vec<C64>> = (0..3).map(|i| vec![c.points()[i], c.points()[3 - i]]).collect();
    /// let mut out = SoftOutput::default();
    /// rx.detect_uses(0..3, false, 1e-3, |_| Ok(identity(2)), |i| x[i].clone(), &mut out)?;
    /// assert!(out.symbols.iter().zip(x.concat()).all(|(a, b)| (a - b).norm() < 1e-12));
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn detect_uses<I, H, Y>(
        &self,
        uses: I,
        alamouti: bool,
        noise_var: f64,
        mut channel: H,
        mut received: Y,
        out: &mut SoftOutput,
    ) -> Result<()>
    where
        I: IntoIterator<Item = usize>,
        H: FnMut(usize) -> Result<CMat>,
        Y: FnMut(usize) -> Vec<C64>,
    {
        let mut uses = uses.into_iter();
        if !alamouti {
            for i in uses {
                self.layers(&channel(i)?, &received(i), noise_var, out)?;
            }
            return Ok(());
        }
        while let Some(i0) = uses.next() {
            let Some(i1) = uses.next() else {
                return Err(Error::Shape("Alamouti's code needs an even number of channel uses".into()));
            };
            let (y0, y1) = (received(i0), received(i1));
            let y = Array2::from_shape_fn((y0.len(), 2), |(r, j)| if j == 0 { y0[r] } else { y1[r] });
            self.decode_alamouti(&channel(i0)?, &y, noise_var, out)?;
        }
        Ok(())
    }

    /// Decode one block of Alamouti's code (the two channel uses are the columns of `y`,
    /// $n_{r} \times 2$) and append its two symbols to `out`.
    ///
    /// # Errors
    /// The errors of [`Ostbc::decode`].
    pub fn decode_alamouti(&self, h: &CMat, y: &Array2<C64>, noise_var: f64, out: &mut SoftOutput) -> Result<()> {
        let (z, var) = Ostbc::alamouti().decode(&h.view(), &y.view(), noise_var)?;
        if self.llr {
            out.llr.extend(self.constellation.llr_with_variances(&z, &var, None, false)?);
        }
        out.symbols.extend(z);
        out.noise_var.extend(var);
        Ok(())
    }
}
