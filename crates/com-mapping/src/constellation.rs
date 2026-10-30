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
use com_math::bits::bits_to_indices;
use com_math::nearest::Slicer;
use com_math::{C64, cis, gray, index_to_bits};
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

/// The largest number of bits of [`gray_code`] (16 777 216 code words; the $2^{n}$ words
/// are allocated at once).
pub const MAX_GRAY_BITS: u32 = 24;

/// The Gray code of every `n`-bit integer: `code[i] = i ⊕ (i >> 1)` ([`com_math::gray`]).
///
/// # Errors
/// [`Error::Config`] for more than [`MAX_GRAY_BITS`] bits.
pub fn gray_code(n_bits: u32) -> Result<Vec<usize>> {
    ensure(n_bits <= MAX_GRAY_BITS, || format!("n_bits must be at most {MAX_GRAY_BITS}, got {n_bits}"))?;
    Ok((0..1usize << n_bits).map(gray).collect())
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

/// The real BPSK symbols $1 - 2b$ of bits ($0 \mapsto +1$, $1 \mapsto -1$), the
/// modulation of the code tests over a real channel ([`crate::demapper::bpsk_llr`] is
/// its demapper).
///
/// # Examples
/// ```
/// use com_mapping::constellation::bpsk;
/// assert_eq!(bpsk(&[0, 1, 1]), [1.0, -1.0, -1.0]);
/// ```
pub fn bpsk(bits: &[u8]) -> Vec<f64> {
    bits.iter().map(|b| 1.0 - 2.0 * f64::from(*b)).collect()
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
    pub fn ook() -> Constellation {
        Constellation::new(ConstellationKind::Ook, 2).unwrap_or_else(|_| unreachable!("OOK of order 2 is valid"))
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

    /// The same points as [`Constellation::new`] with a set-partitioning labelling
    /// (Ungerboeck 1982): fixing the last bits of the label leaves subsets whose minimum
    /// distance grows. PSK gets the natural labelling, the point $e^{j2\pi k/M}$ carrying
    /// the bits of $k$; 16-QAM the labelling of Caire, Taricco & Biglieri (*IEEE Trans.
    /// Inf. Theory* 44(3), 1998, Fig. 2(a)). The family of the result is
    /// [`ConstellationKind::Custom`].
    ///
    /// # Errors
    /// [`Error::Config`] for another family or order.
    ///
    /// # Examples
    /// ```
    /// use com_mapping::constellation::{Constellation, ConstellationKind};
    /// use com_math::C64;
    /// let c = Constellation::set_partitioning(ConstellationKind::Qam, 16)?;
    /// assert_eq!(c.points()[0b1001], C64::new(-3.0, 3.0));
    /// // the points ending in 00 form a 4-QAM of side 4
    /// let sub: Vec<C64> = (0..16).filter(|i| i & 3 == 0).map(|i| c.points()[i]).collect();
    /// assert!(sub.iter().all(|a| sub.iter().all(|b| a == b || (a - b).norm() >= 4.0)));
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn set_partitioning(kind: ConstellationKind, m: usize) -> Result<Constellation> {
        let points = match (kind, m) {
            (ConstellationKind::Psk, _) => {
                ensure(m >= 2 && m.is_power_of_two(), || format!("the order must be a power of two ≥ 2, got {m}"))?;
                (0..m).map(|k| cis(2.0 * std::f64::consts::PI * k as f64 / m as f64)).collect()
            }
            (ConstellationKind::Qam, 16) => {
                // Caire et al. 1998, Fig. 2(a), row by row from the top, left to right
                let rows: [[usize; 4]; 4] = [
                    [0b1001, 0b1100, 0b1101, 0b1000],
                    [0b1110, 0b1011, 0b1010, 0b1111],
                    [0b0101, 0b0000, 0b0001, 0b0100],
                    [0b0010, 0b0111, 0b0110, 0b0011],
                ];
                let mut pts = vec![C64::new(0.0, 0.0); 16];
                for (r, row) in rows.iter().enumerate() {
                    for (c, &label) in row.iter().enumerate() {
                        pts[label] = C64::new(2.0 * c as f64 - 3.0, 3.0 - 2.0 * r as f64);
                    }
                }
                pts
            }
            _ => return Err(Error::Config(format!("no set-partitioning labelling for {}-{}", m, kind.name()))),
        };
        Constellation::from_points(points)
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
        com_math::power(&self.points)
    }

    /// Average energy `Σ p_i |x_i|²` under the probabilities `px`.
    ///
    /// # Panics
    /// If `px` does not hold one probability per point.
    pub fn energy_with(&self, px: &[f64]) -> f64 {
        assert_eq!(px.len(), self.points.len(), "energy_with: one probability per point");
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
    /// `px` (probabilistic shaping), normalised to a sum of 1 first as by
    /// [`Constellation::priors`].
    ///
    /// # Errors
    /// As [`Constellation::priors`]; [`Error::Config`] if the energy under `px` is zero.
    pub fn normalized_under(&self, px: &[f64]) -> Result<Constellation> {
        let es = self.energy_with(&self.priors(Some(px))?);
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
        bits_to_indices(bits, self.bits)
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

/// The floor of a prior probability before its logarithm: $\ln 10^{-300} \approx -691$.
pub const PRIOR_FLOOR: f64 = 1e-300;

/// The log of prior probabilities, floored at [`PRIOR_FLOOR`] so that a zero
/// probability gives a finite log (the metric of an impossible point).
pub fn log_priors(px: &[f64]) -> Vec<f64> {
    px.iter().map(|v| v.max(PRIOR_FLOOR).ln()).collect()
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
    #[should_panic(expected = "energy_with: one probability per point")]
    fn energy_with_rejects_too_few_probabilities_instead_of_truncating() {
        let _ = Constellation::qam(4).unwrap().energy_with(&[0.5, 0.5]);
    }

    #[test]
    fn normalized_under_treats_unnormalised_priors_as_their_normalisation() {
        let c = Constellation::qam(16).unwrap();
        let p: Vec<f64> = c.points().iter().map(|x| (-0.1 * x.norm_sqr()).exp()).collect();
        let total: f64 = p.iter().sum();
        let unit: Vec<f64> = p.iter().map(|v| v / total).collect();
        let a = c.normalized_under(&p).unwrap();
        let b = c.normalized_under(&unit).unwrap();
        assert!(a.points().iter().zip(b.points()).all(|(x, y)| (x - y).norm() < 1e-14));
        assert!((a.energy_with(&unit) - 1.0).abs() < 1e-14);
        // three times the priors: the same constellation, not one of energy 1/3
        let triple: Vec<f64> = unit.iter().map(|v| 3.0 * v).collect();
        assert!((c.normalized_under(&triple).unwrap().energy_with(&unit) - 1.0).abs() < 1e-14);
        assert!(matches!(c.normalized_under(&unit[..15]), Err(Error::Shape(_))));
        assert!(matches!(c.normalized_under(&[-1.0; 16]), Err(Error::Config(_))));
    }

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
