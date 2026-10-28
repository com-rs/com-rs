//! Constellations, Gray mapping and hard decisions (OptiCommPy `optic.comm.modulation`).
//!
//! # Constellations
//!
//! A [`Constellation`] stores its `M = 2^b` points **indexed by their bit label**: the
//! point `points()[i]` carries the bits of `i` written on `b` bits, most significant
//! bit first. The points are those of OptiCommPy, not normalised:
//!
//! - PAM: `{-(M-1), …, -1, 1, …, M-1}` (minimum distance 2, `E_s = (M²-1)/3`);
//! - square QAM: the Cartesian product of two `√M`-PAM (`E_s = 2(M-1)/3`);
//! - PSK: `e^{j2πk/M}`, `k = 0..M-1` (`E_s = 1`);
//! - OOK: `{0, 1}`.
//!
//! The Gray labels are assigned as in OptiCommPy: the points are listed in their
//! construction order (PAM and PSK in increasing order; QAM row by row from the top,
//! every other row reversed, a "snake" that makes neighbouring points adjacent in the
//! list), and the `i`-th point of the list gets the Gray code `i ⊕ (i >> 1)`. Two
//! nearest neighbours then differ by one bit, and the bits of a symbol stream are those
//! of OptiCommPy's `modulateGray`.
//!
//! Use [`Constellation::normalized`] for unit average energy (`E_s = 1`), the usual
//! convention of the simulations.
//!
//! # Decisions
//!
//! [`Constellation::decide`] returns the nearest point (the maximum-likelihood decision
//! over AWGN); the soft demapper is in [`crate::demapper`].
//!
//! # Examples
//! ```
//! use com_mapping::constellation::{Constellation, ConstellationKind};
//! use com_math::C64;
//! let c = Constellation::qam(16).unwrap();
//! assert_eq!(c.points()[0], C64::new(-3.0, 3.0));            // label 0000
//! assert!((c.energy() - 10.0).abs() < 1e-12);                 // 2(M-1)/3
//! let x = c.modulate(&[0, 0, 0, 1, 1, 0, 1, 1]).unwrap();
//! assert_eq!(x, vec![c.points()[1], c.points()[11]]);
//! // Gray mapping: nearest neighbours differ by one bit
//! for i in 0..16 {
//!     for j in 0..16 {
//!         if (c.points()[i] - c.points()[j]).norm() == 2.0 {
//!             assert_eq!((i ^ j).count_ones(), 1);
//!         }
//!     }
//! }
//! assert_eq!(Constellation::new(ConstellationKind::Psk, 8).unwrap().order(), 8);
//! ```

use com_core::{Error, Result, ensure, same_len};
use com_math::nearest::Slicer;
use com_math::{C64, bits_to_index, cis, gray, index_to_bits};
use std::str::FromStr;

/// The family of a constellation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstellationKind {
    /// Pulse amplitude modulation (real).
    Pam,
    /// Square quadrature amplitude modulation.
    Qam,
    /// Phase shift keying.
    Psk,
    /// On-off keying `{0, 1}` (real).
    Ook,
    /// Points given by the user ([`Constellation::from_points`]).
    Custom,
}

impl FromStr for ConstellationKind {
    type Err = Error;

    /// Parse `"pam"`, `"qam"`, `"psk"` or `"ook"` (the `constType` of OptiCommPy),
    /// case-insensitive.
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown name.
    fn from_str(s: &str) -> Result<ConstellationKind> {
        match s.to_ascii_lowercase().as_str() {
            "pam" => Ok(ConstellationKind::Pam),
            "qam" => Ok(ConstellationKind::Qam),
            "psk" => Ok(ConstellationKind::Psk),
            "ook" => Ok(ConstellationKind::Ook),
            other => Err(Error::Config(format!("unknown constellation type {other:?}: pam, qam, psk or ook"))),
        }
    }
}

impl ConstellationKind {
    /// The name of the family: `"pam"`, `"qam"`, `"psk"`, `"ook"` or `"custom"` (the
    /// names that [`ConstellationKind::from_str`] reads, but the last).
    ///
    /// # Examples
    /// ```
    /// use com_mapping::constellation::ConstellationKind;
    /// let k: ConstellationKind = "QAM".parse()?;
    /// assert_eq!(k.name(), "qam");
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            ConstellationKind::Pam => "pam",
            ConstellationKind::Qam => "qam",
            ConstellationKind::Psk => "psk",
            ConstellationKind::Ook => "ook",
            ConstellationKind::Custom => "custom",
        }
    }

    /// Whether the constellation is real (one real dimension per symbol).
    pub fn is_real(self) -> bool {
        matches!(self, ConstellationKind::Pam | ConstellationKind::Ook)
    }
}

/// The Gray code of every `n`-bit integer: `code[i] = i ⊕ (i >> 1)` ([`com_math::gray`]).
pub fn gray_code(n_bits: u32) -> Vec<usize> {
    (0..1usize << n_bits).map(gray).collect()
}

/// The bits of every index, `bits_per_symbol` bits each, most significant bit first
/// ([`com_math::index_to_bits`]).
pub fn indices_to_bits(indices: &[usize], bits_per_symbol: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len() * bits_per_symbol);
    for &i in indices {
        out.extend(index_to_bits(i, bits_per_symbol));
    }
    out
}

/// A labelled constellation: `points()[i]` carries the bits of `i`.
#[derive(Clone, Debug, PartialEq)]
pub struct Constellation {
    kind: ConstellationKind,
    points: Vec<C64>,
    bits: usize,
    slicer: Slicer,
}

impl Constellation {
    /// The constellation of a family and an order `m` (power of two; a square for QAM;
    /// 2 for OOK).
    ///
    /// # Errors
    /// [`Error::Config`] unless `m` is a power of two ≥ 2 (a square for QAM).
    pub fn new(kind: ConstellationKind, m: usize) -> Result<Constellation> {
        ensure(m >= 2 && m.is_power_of_two(), || format!("the order must be a power of two ≥ 2, got {m}"))?;
        let natural: Vec<C64> = match kind {
            ConstellationKind::Pam => pam_levels(m).into_iter().map(|v| C64::new(v, 0.0)).collect(),
            ConstellationKind::Qam => {
                let l = qam_side(m).ok_or_else(|| Error::Config(format!("QAM needs a square order, got {m}")))?;
                let lv = pam_levels(l);
                let mut pts = Vec::with_capacity(m);
                for r in 0..l {
                    let im = lv[l - 1 - r];
                    let row: Vec<C64> = lv.iter().map(|&re| C64::new(re, im)).collect();
                    if r % 2 == 1 {
                        pts.extend(row.into_iter().rev());
                    } else {
                        pts.extend(row);
                    }
                }
                pts
            }
            ConstellationKind::Psk => (0..m).map(|k| cis(2.0 * std::f64::consts::PI * k as f64 / m as f64)).collect(),
            ConstellationKind::Ook => {
                ensure(m == 2, || format!("OOK has two points, got {m}"))?;
                vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)]
            }
            ConstellationKind::Custom => {
                return Err(Error::Config("use Constellation::from_points for a custom constellation".into()));
            }
        };
        let bits = m.trailing_zeros() as usize;
        let mut points = vec![C64::new(0.0, 0.0); m];
        for (i, p) in natural.into_iter().enumerate() {
            points[gray(i)] = p;
        }
        Ok(Constellation::assemble(kind, points, bits))
    }

    fn assemble(kind: ConstellationKind, points: Vec<C64>, bits: usize) -> Constellation {
        let slicer = Slicer::new(&points);
        Constellation { kind, points, bits, slicer }
    }

    /// Square `m`-QAM (`m` = 4, 16, 64, 256, …).
    ///
    /// # Errors
    /// [`Error::Config`] unless `m` is an even power of two ≥ 4.
    pub fn qam(m: usize) -> Result<Constellation> {
        Constellation::new(ConstellationKind::Qam, m)
    }

    /// `m`-PSK.
    ///
    /// # Errors
    /// [`Error::Config`] unless `m` is a power of two ≥ 2.
    pub fn psk(m: usize) -> Result<Constellation> {
        Constellation::new(ConstellationKind::Psk, m)
    }

    /// `m`-PAM.
    ///
    /// # Errors
    /// [`Error::Config`] unless `m` is a power of two ≥ 2.
    pub fn pam(m: usize) -> Result<Constellation> {
        Constellation::new(ConstellationKind::Pam, m)
    }

    /// On-off keying.
    #[allow(clippy::expect_used, reason = "OOK of order 2 is always a valid constellation")]
    pub fn ook() -> Constellation {
        Constellation::new(ConstellationKind::Ook, 2).expect("OOK is valid")
    }

    /// The constellation of a name: `16-QAM`, `qam16`, `QPSK` (4-QAM), `BPSK` (2-PSK),
    /// `8-PSK`, `4-PAM`, `OOK` (case-insensitive; `-`, `_` and spaces are ignored), not
    /// normalised (call [`Constellation::normalized`] for unit energy).
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown family, a missing order or an invalid one.
    ///
    /// # Examples
    /// ```
    /// use com_mapping::constellation::{Constellation, ConstellationKind};
    /// let c = Constellation::from_name("16-QAM")?;
    /// assert_eq!((c.kind(), c.order()), (ConstellationKind::Qam, 16));
    /// assert_eq!(Constellation::from_name("qpsk")?, Constellation::qam(4)?);
    /// assert!(Constellation::from_name("sixteen").is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn from_name(name: &str) -> Result<Constellation> {
        let t = name.to_ascii_lowercase().replace(['-', '_', ' '], "");
        let (kind, m) = match t.as_str() {
            "qpsk" => (ConstellationKind::Qam, 4),
            "bpsk" => (ConstellationKind::Psk, 2),
            "ook" => (ConstellationKind::Ook, 2),
            _ => {
                let digits: String = t.chars().filter(char::is_ascii_digit).collect();
                let letters: String = t.chars().filter(|c| !c.is_ascii_digit()).collect();
                let m = digits
                    .parse::<usize>()
                    .map_err(|_| Error::Config(format!("cannot read the constellation {name:?}")))?;
                (letters.parse()?, m)
            }
        };
        Constellation::new(kind, m)
    }

    /// A constellation from its points, `points[i]` labelled by the bits of `i`.
    ///
    /// # Errors
    /// [`Error::Config`] unless the number of points is a power of two ≥ 2.
    pub fn from_points(points: Vec<C64>) -> Result<Constellation> {
        let m = points.len();
        ensure(m >= 2 && m.is_power_of_two(), || format!("the number of points must be a power of two ≥ 2, got {m}"))?;
        Ok(Constellation::assemble(ConstellationKind::Custom, points, m.trailing_zeros() as usize))
    }

    /// The family.
    pub fn kind(&self) -> ConstellationKind {
        self.kind
    }

    /// The number of points `M`.
    pub fn order(&self) -> usize {
        self.points.len()
    }

    /// Bits per symbol `log2 M`.
    pub fn bits_per_symbol(&self) -> usize {
        self.bits
    }

    /// The points, indexed by their label.
    pub fn points(&self) -> &[C64] {
        &self.points
    }

    /// Average energy `Σ |x_i|² / M` (uniform probabilities).
    pub fn energy(&self) -> f64 {
        self.points.iter().map(C64::norm_sqr).sum::<f64>() / self.order() as f64
    }

    /// Average energy `Σ p_i |x_i|²` under the probabilities `px`.
    pub fn energy_with(&self, px: &[f64]) -> f64 {
        self.points.iter().zip(px).map(|(p, w)| w * p.norm_sqr()).sum()
    }

    /// The prior probabilities of the points: `px` checked (one non-negative value per
    /// point, not all zero) and normalised to a sum of 1, or uniform if `None`.
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of values, [`Error::Config`] for negative or
    /// all-zero values.
    pub fn priors(&self, px: Option<&[f64]>) -> Result<Vec<f64>> {
        let m = self.order();
        Ok(match px {
            None => vec![1.0 / m as f64; m],
            Some(p) => {
                same_len(p.len(), m, "probabilities and constellation points")?;
                let total: f64 = p.iter().sum();
                ensure(total > 0.0 && p.iter().all(|v| *v >= 0.0), || "the probabilities must be non-negative".into())?;
                p.iter().map(|v| v / total).collect()
            }
        })
    }

    /// The same constellation scaled to unit average energy under the probabilities
    /// `px` (probabilistic shaping).
    ///
    /// # Errors
    /// [`Error::Config`] if the energy under `px` is zero.
    pub fn normalized_under(&self, px: &[f64]) -> Result<Constellation> {
        let es = self.energy_with(px);
        ensure(es > 0.0, || "the constellation has no energy".into())?;
        Ok(self.scaled(1.0 / es.sqrt()))
    }

    /// The same constellation scaled to unit average energy.
    pub fn normalized(&self) -> Constellation {
        self.scaled(1.0 / self.energy().sqrt())
    }

    /// The same constellation with every point multiplied by `s`.
    pub fn scaled(&self, s: f64) -> Constellation {
        Constellation::assemble(self.kind, self.points.iter().map(|p| p * s).collect(), self.bits)
    }

    /// The label (index) of every symbol of a bit stream (most significant bit first).
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of bits is not a multiple of `log2 M`.
    pub fn indices(&self, bits: &[u8]) -> Result<Vec<usize>> {
        let b = self.bits;
        if bits.len() % b != 0 {
            return Err(Error::Shape(format!("{} bits is not a multiple of {b} bits per symbol", bits.len())));
        }
        Ok(bits.chunks(b).map(bits_to_index).collect())
    }

    /// Map a bit stream to symbols (OptiCommPy `modulateGray`).
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of bits is not a multiple of `log2 M`.
    pub fn modulate(&self, bits: &[u8]) -> Result<Vec<C64>> {
        Ok(self.indices(bits)?.into_iter().map(|i| self.points[i]).collect())
    }

    /// The symbols of a sequence of labels.
    pub fn symbols(&self, indices: &[usize]) -> Vec<C64> {
        indices.iter().map(|&i| self.points[i]).collect()
    }

    /// The label of the nearest point of every sample (minimum Euclidean distance).
    pub fn decide(&self, z: &[C64]) -> Vec<usize> {
        z.iter().map(|&v| self.nearest(v)).collect()
    }

    /// The label of the point nearest to `z`.
    #[inline]
    pub fn nearest(&self, z: C64) -> usize {
        self.slicer.nearest(z)
    }

    /// The nearest point of every sample.
    pub fn project(&self, z: &[C64]) -> Vec<C64> {
        z.iter().map(|&v| self.points[self.nearest(v)]).collect()
    }

    /// Hard-decision bits of every sample (OptiCommPy `demodulateGray`).
    pub fn demodulate(&self, z: &[C64]) -> Vec<u8> {
        indices_to_bits(&self.decide(z), self.bits)
    }
}

/// `{-(m-1), …, -1, 1, …, m-1}`.
fn pam_levels(m: usize) -> Vec<f64> {
    (0..m).map(|i| 2.0 * i as f64 - (m as f64 - 1.0)).collect()
}

/// The log of prior probabilities, floored at `1e-300` so that a zero probability gives
/// a finite log (the metric of an impossible point).
pub fn log_priors(px: &[f64]) -> Vec<f64> {
    px.iter().map(|v| v.max(1e-300).ln()).collect()
}

/// The side of a square QAM of order `m` (`None` unless `m` is a perfect square).
pub fn qam_side(m: usize) -> Option<usize> {
    let side = (m as f64).sqrt().round() as usize;
    (side * side == m).then_some(side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_slicer_agrees_with_the_exhaustive_search() {
        let mut rng = com_core::rng::Rng::new(11);
        let cs = [
            Constellation::qam(4).unwrap(),
            Constellation::qam(16).unwrap().normalized(),
            Constellation::qam(256).unwrap().scaled(0.37),
            Constellation::pam(8).unwrap(),
            Constellation::psk(8).unwrap(),
            Constellation::from_points(vec![
                C64::new(0.0, 0.0),
                C64::new(1.0, 0.2),
                C64::new(-0.4, 1.0),
                C64::new(0.3, -0.9),
            ])
            .unwrap(),
        ];
        let grids = [true, true, true, true, false, false];
        for (c, grid) in cs.iter().zip(grids) {
            assert_eq!(c.slicer.is_grid(), grid, "{:?} {}", c.kind(), c.order());
            let brute = Slicer::exhaustive(c.points());
            let scale = c.energy().sqrt() * 1.5;
            for _ in 0..2000 {
                let z = rng.complex_gaussian() * scale;
                assert_eq!(c.nearest(z), brute.nearest(z), "{:?} {} at {z}", c.kind(), c.order());
            }
        }
    }

    #[test]
    fn energies_and_labels() {
        assert!((Constellation::pam(4).unwrap().energy() - 5.0).abs() < 1e-12);
        assert!((Constellation::qam(64).unwrap().energy() - 42.0).abs() < 1e-12);
        assert!((Constellation::psk(8).unwrap().energy() - 1.0).abs() < 1e-12);
        assert!(Constellation::qam(32).is_err());
        let c = Constellation::psk(4).unwrap();
        // natural order 1, j, -1, -j gets the Gray codes 00, 01, 11, 10
        assert!((c.points()[0b01] - C64::new(0.0, 1.0)).norm() < 1e-12);
        assert!((c.points()[0b11] - C64::new(-1.0, 0.0)).norm() < 1e-12);
    }
}
