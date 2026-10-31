//! Differential coding of M-PSK (DBPSK, DQPSK, D8PSK, …): encoding and decoding of the
//! phase indices, noncoherent detection, and the Gray-labelled link [`Dpsk`].
//!
//! # Model
//!
//! The information is carried by the **phase transitions**. With $a_{k} \in \\{0, \dots,
//! M - 1\\}$ the differential phase index of the symbol $k$ and $c_{-1} = 0$ the
//! reference,
//!
//! $$
//! c_{k} = (a_{k} + c_{k-1}) \bmod M, \qquad x_{k} = e^{j 2 \pi c_{k} / M},
//! $$
//!
//! so a sequence of $n$ information symbols is transmitted as $n + 1$ PSK symbols
//! ([`encode`]). Decoding is the difference $a_{k} = (c_{k} - c_{k-1}) \bmod M$
//! ([`decode`]). The noncoherent detector ([`detect`]) needs no carrier phase: it decides
//! the differential phase nearest to
//!
//! $$
//! \arg \left(r_{k} \\, r_{k-1}^{\ast}\right),
//! $$
//!
//! which a constant rotation $e^{j \phi}$ of the received symbols leaves unchanged. With
//! coherent decisions on every symbol followed by [`decode`], one wrong symbol spoils two
//! differences: the bit error rate of DBPSK becomes $2p(1 - p)$ with $p =
//! Q(\sqrt{2 E_{b}/N_{0}})$ ([`crate::theory::ber_dbpsk_coherent`]); the noncoherent
//! detector gives $\frac{1}{2} e^{-E_{b}/N_{0}}$ for DBPSK and the integral of Pawula,
//! Rice & Roberts for M-DPSK ([`crate::theory::ber_dpsk`]).
//!
//! [`Dpsk`] maps bits to differential phase indices with the Gray labelling of
//! [`crate::constellation`] (the bits of a symbol are the Gray label of its phase
//! transition), so that the nearest wrong transitions differ by one bit.
//!
//! # Square QAM with a differential quadrant
//!
//! A carrier recovery of square QAM, such as the blind phase search, leaves an ambiguity
//! of a multiple of $\pi/2$. [`QuadrantQam`] removes it as Pfau, Hoffmann and Noé (2009,
//! Sec. III-A, Fig. 5): the two bits that select the quadrant are the Gray label of a
//! differential quadrant index ([`encode`] with $M = 4$), and the other $\log_{2} M - 2$
//! bits select the point inside the quadrant, Gray-labelled along each axis, the pattern
//! of the first quadrant rotated with the quadrant. A rotation of the received symbols by
//! $j$ moves every symbol to the next quadrant with the same inner label, so the
//! differential decoding of the quadrants returns the same bits. The labelling is no
//! longer Gray across the axes, which costs about 3 dB of SNR for QPSK at a given bit
//! error rate (every symbol error spoils two quadrant differences), and less and less
//! for larger constellations.
//!
//! # Examples
//! ```
//! use com_mapping::differential::{Dpsk, decode, detect, encode};
//! use com_core::rng::Rng;
//! let a = vec![0, 1, 3, 2, 1];
//! let c = encode(&a, 4)?;
//! assert_eq!(c, vec![0, 0, 1, 0, 2, 3]);
//! assert_eq!(decode(&c, 4)?, a);
//! let dqpsk = Dpsk::new(4)?;
//! let bits = com_mapping::source::bits(200, &mut Rng::new(1));
//! let x = dqpsk.map_bits(&bits)?;
//! assert_eq!(x.len(), 101);
//! // a constant phase rotation does not change the decoded bits
//! let rotated: Vec<_> = x.iter().map(|v| v * com_math::cis(1.234)).collect();
//! assert_eq!(dqpsk.demap_bits(&rotated)?, bits);
//! assert_eq!(detect(&rotated, 4)?, detect(&x, 4)?);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008,
//!    §3.2-2 (differentially encoded PSK) and §4.5 (noncoherent detection, DPSK).
//! 2. R. F. Pawula, S. O. Rice and J. H. Roberts, "Distribution of the phase angle
//!    between two vectors perturbed by Gaussian noise," *IEEE Trans. Commun.*, 30(8),
//!    1828–1841, 1982.
//! 3. T. Pfau, S. Hoffmann and R. Noé, "Hardware-efficient coherent digital receiver
//!    concept with feedforward carrier recovery for M-QAM constellations," *J. Lightw.
//!    Technol.*, 27(8), 2009.

use crate::constellation::{indices_to_bits, qam_side};
use com_core::{Error, Result, ensure};
use com_math::bits::bits_to_indices;
use com_math::{C64, cis, gray, gray_inverse};
use std::f64::consts::TAU;

/// The detection of a differential PSK signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detection {
    /// Noncoherent: the phase difference of consecutive samples ([`detect`]).
    Noncoherent,
    /// Coherent detection, then differential decoding ([`detect_coherent`]).
    Coherent,
}

impl std::str::FromStr for Detection {
    type Err = Error;

    /// `noncoherent` or `coherent` (case-insensitive).
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    fn from_str(s: &str) -> Result<Detection> {
        match s.to_ascii_lowercase().as_str() {
            "noncoherent" => Ok(Detection::Noncoherent),
            "coherent" => Ok(Detection::Coherent),
            _ => Err(Error::Config(format!("detection must be noncoherent or coherent, got {s:?}"))),
        }
    }
}

/// Check the order and the indices of a differential code.
fn check(indices: &[usize], m: usize) -> Result<()> {
    ensure(m >= 2 && m.is_power_of_two(), || format!("the order must be a power of two ≥ 2, got {m}"))?;
    ensure(indices.iter().all(|&i| i < m), || format!("phase indices must be below {m}"))
}

/// Differential encoding: the `n + 1` phase indices $c_{-1} = 0, c_{0}, \dots, c_{n-1}$
/// with $c_{k} = (a_{k} + c_{k-1}) \bmod M$.
///
/// # Errors
/// [`Error::Config`] unless `m` is a power of two ≥ 2 and every index is below `m`.
///
/// # Examples
/// ```
/// use com_mapping::differential::encode;
/// assert_eq!(encode(&[1, 1, 1, 1], 2).unwrap(), vec![0, 1, 0, 1, 0]);
/// ```
pub fn encode(indices: &[usize], m: usize) -> Result<Vec<usize>> {
    check(indices, m)?;
    let mut out = Vec::with_capacity(indices.len() + 1);
    let mut c = 0;
    out.push(c);
    for &a in indices {
        c = (a + c) % m;
        out.push(c);
    }
    Ok(out)
}

/// Differential decoding: the `n` differences $a_{k} = (c_{k} - c_{k-1}) \bmod M$ of
/// `n + 1` phase indices.
///
/// # Errors
/// [`Error::Config`] unless `m` is a power of two ≥ 2 and every index is below `m`;
/// [`Error::Shape`] without a reference symbol (an empty input).
pub fn decode(coded: &[usize], m: usize) -> Result<Vec<usize>> {
    check(coded, m)?;
    ensure_reference(coded.len())?;
    Ok(coded.windows(2).map(|w| (w[1] + m - w[0]) % m).collect())
}

/// [`Error::Shape`] unless there is at least the reference symbol.
fn ensure_reference(n: usize) -> Result<()> {
    if n == 0 {
        Err(Error::Shape("a differential sequence holds at least its reference symbol".into()))
    } else {
        Ok(())
    }
}

/// The phase index nearest to the argument of `z`, $\lfloor M \arg z / 2 \pi \rceil \bmod M$.
fn phase_index(z: C64, m: usize) -> usize {
    let k = (z.arg() / TAU * m as f64).round() as i64;
    k.rem_euclid(m as i64) as usize
}

/// Coherent decisions: the phase index nearest to every received symbol (the hard
/// decision of M-PSK on the absolute phase); [`decode`] then removes the differential
/// encoding.
///
/// # Errors
/// [`Error::Config`] unless `m` is a power of two ≥ 2.
pub fn detect_coherent(r: &[C64], m: usize) -> Result<Vec<usize>> {
    check(&[], m)?;
    Ok(r.iter().map(|&z| phase_index(z, m)).collect())
}

/// Noncoherent detection: the differential phase index nearest to $\arg(r_{k}
/// r_{k-1}^{\ast})$ for every pair of consecutive received symbols (`n` decisions for
/// `n + 1` symbols).
///
/// # Errors
/// [`Error::Config`] unless `m` is a power of two ≥ 2; [`Error::Shape`] for an empty
/// input.
///
/// # Examples
/// ```
/// use com_mapping::differential::detect;
/// use com_math::cis;
/// let r = [cis(0.3), cis(0.3 + std::f64::consts::FRAC_PI_2), cis(0.3 - std::f64::consts::FRAC_PI_2)];
/// assert_eq!(detect(&r, 4).unwrap(), vec![1, 2]);
/// ```
pub fn detect(r: &[C64], m: usize) -> Result<Vec<usize>> {
    check(&[], m)?;
    ensure_reference(r.len())?;
    Ok(r.windows(2).map(|w| phase_index(w[1] * w[0].conj(), m)).collect())
}

/// Differential M-PSK with Gray-labelled phase transitions: the differential phase
/// index $k$ carries the label $g(k)$ ([`com_math::gray`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dpsk {
    m: usize,
    bits: usize,
}

impl Dpsk {
    /// Differential PSK of order `m`.
    ///
    /// # Errors
    /// [`Error::Config`] unless `m` is a power of two ≥ 2.
    pub fn new(m: usize) -> Result<Dpsk> {
        check(&[], m)?;
        Ok(Dpsk { m, bits: m.trailing_zeros() as usize })
    }

    /// The order $M$.
    pub fn order(&self) -> usize {
        self.m
    }

    /// Bits per symbol, $\log_{2} M$.
    pub fn bits_per_symbol(&self) -> usize {
        self.bits
    }

    /// The differential phase indices of a bit stream (most significant bit first).
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of bits is not a multiple of $\log_{2} M$.
    pub fn indices(&self, bits: &[u8]) -> Result<Vec<usize>> {
        Ok(bits_to_indices(bits, self.bits)?.into_iter().map(gray_inverse).collect())
    }

    /// The bits of a sequence of differential phase indices.
    ///
    /// # Errors
    /// [`Error::Config`] if an index is not below $M$.
    pub fn bits(&self, indices: &[usize]) -> Result<Vec<u8>> {
        check(indices, self.m)?;
        Ok(indices_to_bits(&indices.iter().map(|&k| gray(k)).collect::<Vec<_>>(), self.bits))
    }

    /// The `n/log2 M + 1` unit-energy PSK symbols carrying `n` bits (the reference first).
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of bits is not a multiple of $\log_{2} M$.
    pub fn map_bits(&self, bits: &[u8]) -> Result<Vec<C64>> {
        let c = encode(&self.indices(bits)?, self.m)?;
        Ok(c.into_iter().map(|k| cis(TAU * k as f64 / self.m as f64)).collect())
    }

    /// Noncoherent demodulation ([`detect`]) of `n + 1` received symbols into
    /// `n log2 M` bits.
    ///
    /// # Errors
    /// [`Error::Shape`] for an empty input.
    pub fn demap_bits(&self, r: &[C64]) -> Result<Vec<u8>> {
        self.bits(&detect(r, self.m)?)
    }

    /// Coherent demodulation with differential decoding ([`detect_coherent`] then
    /// [`decode`]) of `n + 1` received symbols into `n log2 M` bits.
    ///
    /// # Errors
    /// [`Error::Shape`] for an empty input.
    pub fn demodulate_coherent(&self, r: &[C64]) -> Result<Vec<u8>> {
        self.bits(&decode(&detect_coherent(r, self.m)?, self.m)?)
    }
}

/// Square M-QAM whose quadrant is differentially encoded and whose points inside a
/// quadrant are Gray-labelled (Pfau et al. 2009, Sec. III-A; see the module): the bits
/// are read by a coherent receiver after a carrier recovery with an ambiguity of a
/// multiple of $\pi/2$.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadrantQam {
    m: usize,
    side: usize,
    axis_bits: usize,
    scale: f64,
}

impl QuadrantQam {
    /// Square QAM of order `m` (4, 16, 64, …), of unit average energy.
    ///
    /// # Errors
    /// [`Error::Config`] unless `m` is an even power of two ≥ 4.
    pub fn new(m: usize) -> Result<QuadrantQam> {
        let side = qam_side(m).filter(|s| m >= 4 && s.is_power_of_two());
        let side = side.ok_or_else(|| Error::Config(format!("the order must be 4, 16, 64, …, got {m}")))?;
        let axis_bits = (side / 2).trailing_zeros() as usize;
        Ok(QuadrantQam { m, side, axis_bits, scale: (1.5 / (m as f64 - 1.0)).sqrt() })
    }

    /// The order $M$.
    pub fn order(&self) -> usize {
        self.m
    }

    /// Bits per symbol, $\log_{2} M$.
    pub fn bits_per_symbol(&self) -> usize {
        2 + 2 * self.axis_bits
    }

    /// The point of the quadrant `q` with the inner label `inner`: the point
    /// $(2i + 1) + j (2k + 1)$ of the first quadrant, $i$ and $k$ the Gray-decoded halves
    /// of the label, rotated by $j^{q}$.
    fn point(&self, q: usize, inner: usize) -> C64 {
        let i = gray_inverse(inner >> self.axis_bits);
        let k = gray_inverse(inner & ((1 << self.axis_bits) - 1));
        C64::new((2 * i + 1) as f64, (2 * k + 1) as f64) * C64::new(0.0, 1.0).powu(q as u32) * self.scale
    }

    /// The $M$ points (for the decisions of a carrier recovery).
    pub fn points(&self) -> Vec<C64> {
        let inner = self.m / 4;
        (0..self.m).map(|n| self.point(n / inner, n % inner)).collect()
    }

    /// The `n / log2 M + 1` symbols carrying `n` bits, the reference first (quadrant 0,
    /// inner label 0); the first two bits of every symbol select the quadrant transition.
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of bits is not a multiple of $\log_{2} M$.
    pub fn map_bits(&self, bits: &[u8]) -> Result<Vec<C64>> {
        let labels = bits_to_indices(bits, self.bits_per_symbol())?;
        let inner_mask = (1 << (2 * self.axis_bits)) - 1;
        let shift = 2 * self.axis_bits;
        let quadrants = encode(&labels.iter().map(|&l| gray_inverse(l >> shift)).collect::<Vec<_>>(), 4)?;
        let inner = std::iter::once(0).chain(labels.iter().map(|&l| l & inner_mask));
        Ok(quadrants.into_iter().zip(inner).map(|(q, i)| self.point(q, i)).collect())
    }

    /// The hard decision of a received symbol: its quadrant and its inner label.
    fn decide(&self, r: C64) -> (usize, usize) {
        let edge = (self.side - 1) as f64;
        let level = |x: f64| (2.0 * (x / 2.0).floor() + 1.0).clamp(-edge, edge);
        let z = C64::new(level(r.re / self.scale), level(r.im / self.scale));
        let q = match (z.re > 0.0, z.im > 0.0) {
            (true, true) => 0,
            (false, true) => 1,
            (false, false) => 2,
            (true, false) => 3,
        };
        // back to the first quadrant
        let w = z * C64::new(0.0, -1.0).powu(q as u32);
        let (i, k) = (((w.re.round() as usize) - 1) / 2, ((w.im.round() as usize) - 1) / 2);
        (q, (gray(i) << self.axis_bits) | gray(k))
    }

    /// Hard decisions on `n + 1` received symbols and differential decoding of their
    /// quadrants: `n log2 M` bits, whatever multiple of $\pi/2$ rotates the symbols.
    ///
    /// # Errors
    /// [`Error::Shape`] for an empty input.
    pub fn demap_bits(&self, r: &[C64]) -> Result<Vec<u8>> {
        ensure_reference(r.len())?;
        let (quadrants, inner): (Vec<usize>, Vec<usize>) = r.iter().map(|&z| self.decide(z)).unzip();
        let shift = 2 * self.axis_bits;
        let labels: Vec<usize> =
            decode(&quadrants, 4)?.into_iter().zip(&inner[1..]).map(|(a, &i)| (gray(a) << shift) | i).collect();
        Ok(indices_to_bits(&labels, self.bits_per_symbol()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn the_detections_are_read_by_name() {
        assert_eq!("NonCoherent".parse::<Detection>().unwrap(), Detection::Noncoherent);
        assert_eq!("coherent".parse::<Detection>().unwrap(), Detection::Coherent);
        assert!("differential".parse::<Detection>().is_err());
    }

    #[test]
    fn round_trips_and_errors() {
        let mut rng = Rng::new(4);
        for m in [2usize, 4, 8, 16] {
            let a: Vec<usize> = (0..500).map(|_| rng.below(m)).collect();
            let c = encode(&a, m).unwrap();
            assert_eq!(c.len(), a.len() + 1);
            assert_eq!(c[0], 0);
            assert_eq!(decode(&c, m).unwrap(), a);
            let d = Dpsk::new(m).unwrap();
            let bits = crate::source::bits(500 * d.bits_per_symbol(), &mut rng);
            let x = d.map_bits(&bits).unwrap();
            assert!(x.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
            assert_eq!(d.demap_bits(&x).unwrap(), bits);
            assert_eq!(d.demodulate_coherent(&x).unwrap(), bits);
            // Gray labelling of the transitions: neighbouring phases differ by one bit
            for k in 0..m {
                let (b1, b2) = (d.bits(&[k]).unwrap(), d.bits(&[(k + 1) % m]).unwrap());
                assert_eq!(b1.iter().zip(&b2).filter(|(x, y)| x != y).count(), 1);
            }
        }
        assert!(encode(&[0, 1], 3).is_err());
        assert!(encode(&[0, 4], 4).is_err());
        assert!(decode(&[], 4).is_err());
        assert!(detect(&[], 4).is_err());
        assert!(Dpsk::new(6).is_err());
        assert!(Dpsk::new(4).unwrap().map_bits(&[1, 0, 1]).is_err());
    }

    #[test]
    fn quadrant_qam_survives_every_rotation_by_a_multiple_of_pi_2() {
        let mut rng = Rng::new(9);
        for m in [4usize, 16, 64, 256] {
            let qq = QuadrantQam::new(m).unwrap();
            let pts = qq.points();
            assert_eq!(pts.len(), m);
            let energy: f64 = pts.iter().map(C64::norm_sqr).sum::<f64>() / m as f64;
            assert!((energy - 1.0).abs() < 1e-12);
            // the points are those of square QAM
            let qam = crate::constellation::Constellation::qam(m).unwrap().normalized();
            for p in &pts {
                assert!(qam.points().iter().any(|v| (v - p).norm() < 1e-12));
            }
            let bits = crate::source::bits(400 * qq.bits_per_symbol(), &mut rng);
            let x = qq.map_bits(&bits).unwrap();
            assert_eq!(x.len(), 401);
            for q in 0..4u32 {
                let rot: Vec<C64> = x.iter().map(|v| v * C64::new(0.0, 1.0).powu(q)).collect();
                assert_eq!(qq.demap_bits(&rot).unwrap(), bits, "M = {m}, rotation {q}");
            }
            // Gray inside a quadrant: horizontal and vertical neighbours differ by one bit
            for n in 0..m / 4 {
                let a = qq.point(0, n);
                for d in [C64::new(2.0, 0.0), C64::new(0.0, 2.0)] {
                    let b = a + d * qq.scale;
                    if let Some(j) = (0..m / 4).find(|&j| (qq.point(0, j) - b).norm() < 1e-9) {
                        assert_eq!((n ^ j).count_ones(), 1, "M = {m}");
                    }
                }
            }
        }
        assert!(QuadrantQam::new(8).is_err());
        assert!(QuadrantQam::new(2).is_err());
        assert!(QuadrantQam::new(16).unwrap().map_bits(&[1, 0, 1]).is_err());
        assert!(QuadrantQam::new(16).unwrap().demap_bits(&[]).is_err());
    }

    #[test]
    fn a_constant_rotation_does_not_change_the_decoded_bits() {
        let mut rng = Rng::new(8);
        for m in [2usize, 4, 8] {
            let d = Dpsk::new(m).unwrap();
            let bits = crate::source::bits(300 * d.bits_per_symbol(), &mut rng);
            let x = d.map_bits(&bits).unwrap();
            for phi in [0.1, 1.0, 2.5, -3.0, 4.2] {
                let r: Vec<C64> = x.iter().map(|v| v * cis(phi)).collect();
                assert_eq!(d.demap_bits(&r).unwrap(), bits, "M = {m}, φ = {phi}");
            }
        }
    }
}

#[cfg(test)]
mod literature {
    use super::*;
    use crate::theory::{ber_dbpsk_coherent, ber_dpsk, differential_phase_tail};
    use com_core::rng::Rng;
    use com_math::{db2lin, erfc};
    use std::f64::consts::PI;

    /// Unit-energy symbols plus noise at `ebn0_db` for `bits` bits per symbol:
    /// `N_0 = 1 / (bits · E_b/N_0)`.
    fn noisy(x: &[C64], bits: usize, ebn0_db: f64, rng: &mut Rng) -> Vec<C64> {
        let sd = (1.0 / (bits as f64 * db2lin(ebn0_db))).sqrt();
        x.iter().map(|v| v + rng.complex_gaussian() * sd).collect()
    }

    fn ber(a: &[u8], b: &[u8]) -> f64 {
        a.iter().zip(b).filter(|(x, y)| x != y).count() as f64 / a.len() as f64
    }

    /// J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., 2008, §4.5
    /// (binary DPSK): the noncoherent detector of DBPSK has the bit error probability
    /// `½ exp(-E_b/N_0)`. 400 000 bits: about 3 700 errors are expected at 6 dB
    /// (`9.3e-3`) and 364 at 8 dB (`9.1e-4`), so the relative standard errors are 1.6 %
    /// and 5.2 %; the test accepts three standard errors.
    #[test]
    fn proakis_dbpsk_noncoherent_ber() {
        let d = Dpsk::new(2).unwrap();
        let mut rng = Rng::new(31);
        let n = 400_000;
        for ebn0_db in [6.0, 8.0] {
            let bits = crate::source::bits(n, &mut rng);
            let r = noisy(&d.map_bits(&bits).unwrap(), 1, ebn0_db, &mut rng);
            let got = ber(&d.demap_bits(&r).unwrap(), &bits);
            let want = 0.5 * (-db2lin(ebn0_db)).exp();
            assert!((ber_dpsk(2, ebn0_db).unwrap() - want).abs() < 1e-12);
            let tol = 3.0 / (want * n as f64).sqrt();
            assert!((got / want - 1.0).abs() < tol, "{ebn0_db} dB: {got:e} vs {want:e}");
        }
    }

    /// Proakis & Salehi, 5th ed., §3.2-2 (differentially encoded PSK with coherent
    /// detection): an isolated symbol error spoils two differences, so the bit error rate
    /// of DBPSK is `2p(1 - p)` with `p = Q(√(2 E_b/N_0))`. 400 000 bits at 6 dB
    /// (`4.8e-3`, about 1 900 errors, standard error 2.3 %) and 8 dB (`3.8e-4`, about 150
    /// errors, 8 %); three standard errors accepted.
    #[test]
    fn proakis_coherent_detection_with_differential_decoding() {
        let d = Dpsk::new(2).unwrap();
        let mut rng = Rng::new(32);
        let n = 400_000;
        for ebn0_db in [6.0, 8.0] {
            let bits = crate::source::bits(n, &mut rng);
            let r = noisy(&d.map_bits(&bits).unwrap(), 1, ebn0_db, &mut rng);
            let got = ber(&d.demodulate_coherent(&r).unwrap(), &bits);
            let want = ber_dbpsk_coherent(ebn0_db);
            let tol = 3.0 / (want * n as f64).sqrt();
            assert!((got / want - 1.0).abs() < tol, "{ebn0_db} dB: {got:e} vs {want:e}");
        }
    }

    /// R. F. Pawula, S. O. Rice and J. H. Roberts, "Distribution of the phase angle
    /// between two vectors perturbed by Gaussian noise," IEEE Trans. Commun. 30(8), 1982
    /// (the integral form of the distribution of the differential phase, Section IV, for
    /// two vectors of the same SNR): the bit error rate of Gray-coded DQPSK and D8PSK
    /// with noncoherent detection ([`ber_dpsk`]). 399 996 bits at 8 dB for DQPSK
    /// (`3.6e-3`, about 1 460 errors, standard error 2.6 %) and 11 dB for D8PSK
    /// (`5.6e-3`, about 2 250 errors, 2.1 %); three standard errors accepted.
    #[test]
    fn pawula_rice_roberts_1982_mdpsk_ber() {
        let mut rng = Rng::new(33);
        let n = 399_996; // a multiple of 2 and 3 bits per symbol
        for (m, ebn0_db) in [(4usize, 8.0), (8, 11.0)] {
            let d = Dpsk::new(m).unwrap();
            let bits = crate::source::bits(n, &mut rng);
            let r = noisy(&d.map_bits(&bits).unwrap(), d.bits_per_symbol(), ebn0_db, &mut rng);
            let got = ber(&d.demap_bits(&r).unwrap(), &bits);
            let want = ber_dpsk(m, ebn0_db).unwrap();
            let tol = 3.0 / (want * n as f64).sqrt();
            assert!((got / want - 1.0).abs() < tol, "M = {m} at {ebn0_db} dB: {got:e} vs {want:e}");
        }
    }

    /// Pawula, Rice & Roberts (1982), Section IV-A: the symbol error rate of M-DPSK is
    /// their eq. (44), which [`differential_phase_tail`] computes at $\psi = \pi/M$; it
    /// tends to the asymptote (46), $\sqrt{(1 + \cos(\pi/M))/(2\cos(\pi/M))}
    /// \operatorname{erfc}\sqrt{\rho(1 - \cos(\pi/M))}$ (ratio within 0.3 % at $\rho = 100$,
    /// 0.03 % at 1000); the approximation (47) of Fleck & Trabka over it tends to the
    /// ratio $Q$ tabulated after eq. (50), 0.9186, 0.9768, 0.9987 and 0.9999 for $M = 3$,
    /// 4, 8 and 16 (within 3e-4 at $\rho = 1000$); and, footnote 11, the bit error rate of
    /// DQPSK is $\frac{1}{2} P_{E}(4) + P\{3\pi/4 < \psi \le \pi\}$, the last term
    /// $\sim e^{-\rho}/(2\pi\rho)$ (within 8 % at 10 dB, 4 % at 15 dB), so that
    /// $P_{B}(4) \approx \frac{1}{2} P_{E}(4)$, not the $\frac{2}{3} P_{E}(4)$ of orthogonal
    /// signals.
    #[test]
    fn pawula_rice_roberts_1982_section_iv_a() {
        let pe = |m: f64, rho: f64| differential_phase_tail(PI / m, rho);
        for (m, q) in [(3.0, 0.9186), (4.0, 0.9768), (8.0, 0.9987), (16.0, 0.9999)] {
            let c = (PI / m).cos();
            let asymptote = |rho: f64| ((1.0 + c) / (2.0 * c)).sqrt() * erfc((rho * (1.0 - c)).sqrt());
            let fleck_trabka = |rho: f64| {
                let x = (rho * (1.0 - c)).sqrt();
                erfc(x) + x * (-x * x).exp() / (4.0 * PI.sqrt() * (rho + 0.125))
            };
            for (rho, tol) in [(100.0, 3e-3), (1000.0, 3e-4)] {
                let r = pe(m, rho) / asymptote(rho);
                assert!((r - 1.0).abs() < tol, "M = {m}, rho = {rho}: (44)/(46) = {r}");
            }
            let ratio = fleck_trabka(1000.0) / pe(m, 1000.0);
            assert!((ratio - q).abs() < 3e-4, "M = {m}: Q = {ratio} vs {q}");
        }
        for (rho_db, tol) in [(10.0, 0.08), (15.0, 0.04)] {
            let rho = db2lin(rho_db);
            let tail = 0.5 * differential_phase_tail(3.0 * PI / 4.0, rho);
            let pb = ber_dpsk(4, rho_db - com_math::lin2db(2.0)).unwrap();
            assert!((pb / (0.5 * pe(4.0, rho) + tail) - 1.0).abs() < 1e-10, "{rho_db} dB: {pb:e}");
            let approx = (-rho).exp() / (2.0 * PI * rho);
            assert!((tail / approx - 1.0).abs() < tol, "{rho_db} dB: {tail:e} vs {approx:e}");
        }
    }
}
