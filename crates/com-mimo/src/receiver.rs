//! The receiver of a link: one detector applied channel use after channel use, with
//! soft outputs.
//!
//! [`Receiver`] detects the layers of every channel use $y = H x + w$ (spatial
//! multiplexing) or every block of Alamouti's code, and accumulates a [`SoftOutput`]:
//!
//! - the estimate of every transmitted symbol: the estimate of the detector when it forms
//!   one ([`MimoDetector::soft_estimate`]: the unbiased linear estimate of ZF and MMSE,
//!   the decoupled channel of AMP and OAMP), the decided point with the other detectors;
//! - its noise variance: that of the estimate, or the noise variance of the channel for
//!   a decision;
//! - optionally the bit LLRs of the detector (ZF, MMSE, ML, K-best, AMP, OAMP, Alamouti:
//!   [`MimoDetector::has_soft_output`]).
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
//! rx.layers(&h.view(), &y, 1e-3, &mut out)?;
//! assert_eq!(out.symbols, x.to_vec());
//! assert_eq!(out.llr.len(), 4);
//! # Ok::<(), com_core::Error>(())
//! ```

use crate::detection::{Estimates, KBest, Linear, MimoDetector, Ml, Sic, Sphere};
use crate::message_passing::{Amp, Oamp};
use crate::stbc::Ostbc;
use com_core::{Error, Result, ensure};
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_math::linalg::CMat;
use ndarray::{Array2, ArrayView2};

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
        let name = name.to_ascii_lowercase();
        let canonical = match name.as_str() {
            "mmse_sic" => "sic",
            "lama" => "amp",
            other => other,
        };
        DetectorKind::ALL
            .into_iter()
            .find(|d| d.name() == canonical)
            .ok_or_else(|| Error::Config(format!("unknown detector {name:?}: {}", DetectorKind::NAMES.join(", "))))
    }
}

impl DetectorKind {
    /// The paths of K-best when its name does not give them.
    pub const DEFAULT_PATHS: usize = 8;

    /// The iterations of AMP and OAMP when their name does not give them.
    pub const DEFAULT_ITERATIONS: usize = 10;

    /// Every detector, K-best with [`DetectorKind::DEFAULT_PATHS`] paths and AMP and
    /// OAMP with [`DetectorKind::DEFAULT_ITERATIONS`] iterations.
    pub const ALL: [DetectorKind; 9] = [
        DetectorKind::Zf,
        DetectorKind::Mmse,
        DetectorKind::Sic,
        DetectorKind::ZfSic,
        DetectorKind::Ml,
        DetectorKind::Sphere,
        DetectorKind::KBest(DetectorKind::DEFAULT_PATHS),
        DetectorKind::Amp(DetectorKind::DEFAULT_ITERATIONS),
        DetectorKind::Oamp(DetectorKind::DEFAULT_ITERATIONS),
    ];

    /// The names of [`DetectorKind::ALL`] ([`DetectorKind::name`]), in order: the list
    /// of the detector names that the specifications take theirs from.
    ///
    /// # Examples
    /// ```
    /// use com_mimo::receiver::DetectorKind;
    /// assert_eq!(DetectorKind::NAMES.len(), DetectorKind::ALL.len());
    /// for (name, d) in DetectorKind::NAMES.iter().zip(DetectorKind::ALL) {
    ///     assert_eq!((d.name(), name.parse::<DetectorKind>()?), (*name, d));
    /// }
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub const NAMES: [&'static str; 9] = {
        let mut names = [""; 9];
        let mut i = 0;
        while i < names.len() {
            names[i] = DetectorKind::ALL[i].name();
            i += 1;
        }
        names
    };

    /// The name of the detector, as [`DetectorKind::from_str`](std::str::FromStr::from_str)
    /// reads it.
    pub const fn name(self) -> &'static str {
        match self {
            DetectorKind::Zf => "zf",
            DetectorKind::Mmse => "mmse",
            DetectorKind::Sic => "sic",
            DetectorKind::ZfSic => "zf_sic",
            DetectorKind::Ml => "ml",
            DetectorKind::Sphere => "sphere",
            DetectorKind::KBest(_) => "kbest",
            DetectorKind::Amp(_) => "amp",
            DetectorKind::Oamp(_) => "oamp",
        }
    }

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

    /// Whether the detector computes bit LLRs ([`MimoDetector::has_soft_output`] of its
    /// detector; false for a detector that cannot be built, K-best with no path).
    pub fn has_soft_output(self) -> bool {
        Constellation::psk(2).and_then(|c| self.detector(&c)).is_ok_and(|d| d.has_soft_output())
    }

    /// The detector for the alphabet `c`.
    ///
    /// # Errors
    /// [`Error::Config`] for K-best with no path or AMP and OAMP with no iteration.
    pub fn detector(self, c: &Constellation) -> Result<Box<dyn MimoDetector + Send + Sync>> {
        self.build(c.clone())
    }

    /// [`DetectorKind::detector`] with the alphabet `c` moved into the detector.
    fn build(self, c: Constellation) -> Result<Box<dyn MimoDetector + Send + Sync>> {
        Ok(match self {
            DetectorKind::Zf => Box::new(Linear::zf(c)),
            DetectorKind::Mmse => Box::new(Linear::mmse(c)),
            DetectorKind::Sic => Box::new(Sic::mmse(c)),
            DetectorKind::ZfSic => Box::new(Sic::zf(c)),
            DetectorKind::Ml => Box::new(Ml::new(c)),
            DetectorKind::Sphere => Box::new(Sphere::new(c)),
            DetectorKind::KBest(k) => Box::new(KBest::new(c, k)?),
            DetectorKind::Amp(n) => Box::new(Amp::new(c, n)?),
            DetectorKind::Oamp(n) => Box::new(Oamp::new(c, n)?),
        })
    }
}

/// The accumulated output of a [`Receiver`]: the [`Estimates`] of the channel uses one
/// after the other (the same fields), and their bit LLRs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoftOutput {
    /// The estimate of every symbol.
    pub symbols: Vec<C64>,
    /// The noise variance of every estimate.
    pub noise_var: Vec<f64>,
    /// The bit LLRs (positive: bit 0), when asked.
    pub llr: Vec<f64>,
}

impl SoftOutput {
    /// Append the estimates `e` of one channel use.
    pub fn append(&mut self, e: Estimates) {
        self.symbols.extend(e.symbols);
        self.noise_var.extend(e.noise_var);
    }
}

/// A detector applied channel use after channel use (see the module).
pub struct Receiver {
    kind: DetectorKind,
    detector: Box<dyn MimoDetector + Send + Sync>,
    /// Alamouti's code, built (and checked) once.
    alamouti: Ostbc,
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
    /// [`Error::Config`] when LLRs are asked from a detector without soft output, or for
    /// a detector that cannot be built ([`DetectorKind::detector`]).
    pub fn new(kind: DetectorKind, c: Constellation, llr: bool) -> Result<Receiver> {
        let detector = kind.build(c)?;
        ensure(!llr || detector.has_soft_output(), || {
            format!("the {} detector has no soft output: demap its decisions instead", kind.name())
        })?;
        Ok(Receiver { kind, detector, alamouti: Ostbc::alamouti(), llr })
    }

    /// The alphabet of every stream.
    pub fn constellation(&self) -> &Constellation {
        self.detector.constellation()
    }

    /// Append the bit LLRs of the estimates `e` (one noise variance each) to `out`.
    fn append_llr(&self, e: &Estimates, out: &mut SoftOutput) -> Result<()> {
        let start = out.llr.len();
        out.llr.resize(start + e.symbols.len() * self.constellation().bits_per_symbol(), 0.0);
        self.constellation().llr_with_variances_into(&e.symbols, &e.noise_var, None, false, &mut out.llr[start..])
    }

    /// Detect the layers of one channel use $y = H x + w$ and append them to `out`.
    ///
    /// # Errors
    /// The errors of the detector.
    pub fn layers(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64, out: &mut SoftOutput) -> Result<()> {
        if let Some(e) = self.detector.soft_estimate(h, y, noise_var)? {
            // the LLRs of a detector that estimates come from its estimates: one pass
            if self.llr {
                self.append_llr(&e, out)?;
            }
            out.append(e);
        } else {
            // the decisions and the LLRs of one search
            let labels = if self.llr {
                let (labels, llr) = self.detector.detect_llr(h, y, noise_var)?;
                out.llr.extend(llr);
                labels
            } else {
                self.detector.detect(h, y, noise_var)?
            };
            let points = self.constellation().points();
            out.symbols.extend(labels.iter().map(|&i| points[i]));
            out.noise_var.extend(std::iter::repeat_n(noise_var, labels.len()));
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
                self.layers(&channel(i)?.view(), &received(i), noise_var, out)?;
            }
            return Ok(());
        }
        while let Some(i0) = uses.next() {
            let Some(i1) = uses.next() else {
                return Err(Error::Shape("Alamouti's code needs an even number of channel uses".into()));
            };
            let (y0, y1) = (received(i0), received(i1));
            let y = Array2::from_shape_fn((y0.len(), 2), |(r, j)| if j == 0 { y0[r] } else { y1[r] });
            self.decode_alamouti(&channel(i0)?.view(), &y.view(), noise_var, out)?;
        }
        Ok(())
    }

    /// Decode one block of Alamouti's code (the two channel uses are the columns of `y`,
    /// $n_{r} \times 2$) and append its two symbols to `out`.
    ///
    /// # Errors
    /// The errors of [`Ostbc::decode`].
    pub fn decode_alamouti(
        &self,
        h: &ArrayView2<'_, C64>,
        y: &ArrayView2<'_, C64>,
        noise_var: f64,
        out: &mut SoftOutput,
    ) -> Result<()> {
        let e = self.alamouti.decode(h, y, noise_var)?;
        if self.llr {
            self.append_llr(&e, out)?;
        }
        out.append(e);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use com_math::linalg::matvec;

    #[test]
    fn the_soft_outputs_are_those_of_the_detectors() {
        // the list that Receiver::new kept by hand before the trait method
        for kind in DetectorKind::ALL {
            let former = !matches!(kind, DetectorKind::Sic | DetectorKind::ZfSic | DetectorKind::Sphere);
            assert_eq!(kind.has_soft_output(), former, "{kind:?}");
        }
        assert!(!DetectorKind::KBest(0).has_soft_output() && DetectorKind::KBest(0).detector(&qpsk()).is_err());
        assert!(Receiver::new(DetectorKind::Sphere, qpsk(), true).is_err());
    }

    fn qpsk() -> Constellation {
        Constellation::qam(4).unwrap().normalized()
    }

    #[test]
    fn amp_and_oamp_give_their_decoupled_estimates_and_their_llrs() {
        let c = qpsk();
        let mut rng = Rng::new(31);
        let h = Array2::from_shape_fn((16, 8), |_| rng.complex_gaussian() * 0.25);
        let x: Vec<C64> = (0..8).map(|_| c.points()[rng.below(4)]).collect();
        let y: Vec<C64> = matvec(&h.view(), &x).iter().map(|v| v + rng.complex_gaussian() * 0.1).collect();
        for kind in [DetectorKind::Amp(10), DetectorKind::Oamp(10)] {
            let rx = Receiver::new(kind, c.clone(), true).unwrap();
            let mut out = SoftOutput::default();
            rx.layers(&h.view(), &y, 0.01, &mut out).unwrap();
            let d = kind.detector(&c).unwrap();
            let e = d.soft_estimate(&h.view(), &y, 0.01).unwrap().unwrap();
            assert_eq!((out.symbols, out.noise_var), (e.symbols, e.noise_var));
            // the LLRs of the former path: those of the joint detection and decision
            let (_, llr) = d.detect_llr(&h.view(), &y, 0.01).unwrap();
            assert!(out.llr.iter().zip(&llr).all(|(a, b)| a.to_bits() == b.to_bits()) && out.llr.len() == 16);
        }
    }

    #[test]
    fn the_stored_alamouti_code_decodes_as_a_new_one() {
        let c = qpsk();
        let mut rng = Rng::new(32);
        let rx = Receiver::new(DetectorKind::Mmse, c.clone(), true).unwrap();
        let code = Ostbc::alamouti();
        let s: Vec<C64> = (0..2).map(|_| c.points()[rng.below(4)]).collect();
        let h = Array2::from_shape_fn((2, 2), |_| rng.complex_gaussian());
        let y = h.dot(&code.encode(&s).unwrap()).mapv(|v| v + rng.complex_gaussian() * 0.1);
        let mut out = SoftOutput::default();
        rx.decode_alamouti(&h.view(), &y.view(), 0.01, &mut out).unwrap();
        let e = Ostbc::alamouti().decode_blocks(&[h], &y.view(), 0.01).unwrap();
        assert_eq!((out.symbols.clone(), out.noise_var.clone()), (e.symbols.clone(), e.noise_var.clone()));
        assert_eq!(out.llr, c.llr_with_variances(&e.symbols, &e.noise_var, None, false).unwrap());
    }
}
