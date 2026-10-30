//! Orthogonal space-time block codes (OSTBC).
//!
//! A block code sends $K$ symbols over $T$ channel uses from $n_{t}$ antennas: the
//! $T \times n_{t}$ codeword $X(s)$ is linear in the real and imaginary parts of the
//! symbols,
//!
//! $$
//! X(s) = c \sum_{k=1}^{K} \left(\Re(s_{k})\\, A_{k} + \Im(s_{k})\\, B_{k}\right),
//! $$
//!
//! with the dispersion matrices $A_{k} = X(e_{k})$ and $B_{k} = X(j e_{k})$ and the scale
//! $c$ that makes the average transmitted energy one per channel use (each antenna
//! sends $1/n_{t}$ of it on average). The code is *orthogonal* when
//! $X^{H} X \propto \lVert s \rVert^{2} I$: then, whatever the channel, the received block
//! $Y = H X^{T} + W$ ($n_{r} \times T$) decouples into one scalar channel per symbol, and
//! linear combining is maximum likelihood. Each estimate is
//! $\hat s_{k} = s_{k} + n_{k}$ with the noise variance
//! $\sigma^{2} / (c^{2} \lVert H \rVert_{F}^{2})$ for the codes of unit-norm columns, a
//! diversity of $n_{t} n_{r}$.
//!
//! | Code | $n_{t}$ | $T$ | $K$ | Rate |
//! |---|---|---|---|---|
//! | [`Ostbc::alamouti`] | 2 | 2 | 2 | 1 |
//! | [`Ostbc::rate_3_4`]`(3)` | 3 | 4 | 3 | 3/4 |
//! | [`Ostbc::rate_3_4`]`(4)` | 4 | 4 | 3 | 3/4 |
//!
//! The rows of a codeword are the channel uses, its columns the antennas:
//!
//! $$
//! X_{\mathrm{Alamouti}} = \begin{pmatrix} s_{1} & s_{2} \\\\ -s_{2}^{*} & s_{1}^{*} \end{pmatrix},
//! \qquad
//! X_{3/4} = \begin{pmatrix}
//!   s_{1} & s_{2} & s_{3} & 0 \\\\
//!   -s_{2}^{*} & s_{1}^{*} & 0 & s_{3} \\\\
//!   -s_{3}^{*} & 0 & s_{1}^{*} & -s_{2} \\\\
//!   0 & -s_{3}^{*} & s_{2}^{*} & s_{1}
//! \end{pmatrix}
//! $$
//!
//! (the three-antenna code keeps the first three columns).
//!
//! # Examples
//! ```
//! use com_mimo::stbc::Ostbc;
//! use com_math::C64;
//! use ndarray::array;
//! let code = Ostbc::alamouti();
//! let s = [C64::new(1.0, 1.0), C64::new(-1.0, 1.0)];
//! let x = code.encode(&s)?;                                  // 2 antennas × 2 uses
//! let h = array![[C64::new(0.8, -0.3), C64::new(-0.2, 1.1)]]; // one receive antenna
//! let (est, var) = code.decode(&h.view(), &h.dot(&x).view(), 0.01)?;
//! assert!(est.iter().zip(&s).all(|(a, b)| (a - b).norm() < 1e-12));
//! // the noise variance of every estimate is σ²/(c² ‖H‖²), with c² = 1/2 for Alamouti
//! let h2: f64 = h.iter().map(|v| v.norm_sqr()).sum();
//! assert!((var[0] - 0.01 / (0.5 * h2)).abs() < 1e-15);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. S. M. Alamouti, "A simple transmit diversity technique for wireless
//!    communications," *IEEE J. Sel. Areas Commun.*, 16(8), 1998.
//! 2. V. Tarokh, H. Jafarkhani and A. R. Calderbank, "Space-time block codes from
//!    orthogonal designs," *IEEE Trans. Inf. Theory*, 45(5), 1999.
//! 3. O. Tirkkonen and A. Hottinen, "Square-matrix embeddable space-time block codes
//!    for complex signal constellations," *IEEE Trans. Inf. Theory*, 48(2), 2002.

use com_core::rng::Rng;
use com_core::signal::Symbols;
use com_core::traits::Precode;
use com_core::{Error, Result, ensure};
use com_math::C64;
use com_math::linalg::CMat;
use ndarray::{Array2, ArrayView2};

/// An orthogonal space-time block code.
#[derive(Clone, Debug, PartialEq)]
pub struct Ostbc {
    name: String,
    n_tx: usize,
    n_slots: usize,
    a: Vec<CMat>,
    b: Vec<CMat>,
    scale: f64,
}

impl Ostbc {
    /// The code whose codeword is `generator(s)` ($T \times n_{t}$, linear in the real
    /// and imaginary parts of the $K$ symbols).
    ///
    /// # Errors
    /// [`Error::Config`] if the generator does not return `n_slots × n_tx` matrices or
    /// the code is not orthogonal.
    pub fn from_generator(
        name: &str,
        n_tx: usize,
        n_slots: usize,
        n_symbols: usize,
        generator: impl Fn(&[C64]) -> CMat,
    ) -> Result<Ostbc> {
        ensure(n_tx >= 1 && n_slots >= 1 && n_symbols >= 1, || "empty code".into())?;
        let basis = |k: usize, v: C64| {
            let mut s = vec![C64::new(0.0, 0.0); n_symbols];
            s[k] = v;
            generator(&s)
        };
        let a: Vec<CMat> = (0..n_symbols).map(|k| basis(k, C64::new(1.0, 0.0))).collect();
        let b: Vec<CMat> = (0..n_symbols).map(|k| basis(k, C64::new(0.0, 1.0))).collect();
        ensure(a.iter().chain(&b).all(|m| m.dim() == (n_slots, n_tx)), || {
            format!("the codewords must be {n_slots}×{n_tx}")
        })?;
        let energy: f64 = a.iter().chain(&b).map(|m| m.iter().map(C64::norm_sqr).sum::<f64>()).sum::<f64>() / 2.0;
        ensure(energy > 0.0, || "the code sends nothing".into())?;
        let code = Ostbc { name: name.into(), n_tx, n_slots, a, b, scale: (n_slots as f64 / energy).sqrt() };
        // orthogonality: the real equivalent channel has orthogonal columns for any H
        let mut rng = Rng::new(0x0057_bc00);
        for _ in 0..3 {
            let h = Array2::from_shape_fn((2, n_tx), |_| rng.complex_gaussian());
            let cols = code.equivalent(&h.view());
            let norms: Vec<f64> = cols.iter().map(|u| u.iter().map(|x| x * x).sum()).collect();
            for i in 0..cols.len() {
                for j in 0..i {
                    let dot: f64 = cols[i].iter().zip(&cols[j]).map(|(x, y)| x * y).sum();
                    ensure(dot.abs() <= 1e-9 * (norms[i] * norms[j]).sqrt(), || {
                        format!("the code {name} is not orthogonal")
                    })?;
                }
            }
        }
        Ok(code)
    }

    /// Alamouti's code: two antennas, rate one.
    #[allow(clippy::missing_panics_doc, clippy::expect_used, reason = "the code is orthogonal by construction")]
    pub fn alamouti() -> Ostbc {
        Ostbc::from_generator("Alamouti", 2, 2, 2, |s| ndarray::array![[s[0], s[1]], [-s[1].conj(), s[0].conj()]])
            .expect("Alamouti's code is orthogonal")
    }

    /// The rate-3/4 code for three or four antennas (three symbols over four channel
    /// uses).
    ///
    /// # Errors
    /// [`Error::Config`] for another number of antennas.
    pub fn rate_3_4(n_tx: usize) -> Result<Ostbc> {
        ensure(n_tx == 3 || n_tx == 4, || format!("the rate-3/4 code has 3 or 4 antennas, not {n_tx}"))?;
        Ostbc::from_generator(&format!("rate 3/4, {n_tx} antennas"), n_tx, 4, 3, move |s| {
            let z = C64::new(0.0, 0.0);
            let full = ndarray::array![
                [s[0], s[1], s[2], z],
                [-s[1].conj(), s[0].conj(), z, s[2]],
                [-s[2].conj(), z, s[0].conj(), -s[1]],
                [z, -s[2].conj(), s[1].conj(), s[0]],
            ];
            full.slice_axis(ndarray::Axis(1), ndarray::Slice::from(0..n_tx)).to_owned()
        })
    }

    /// Name of the code.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Transmit antennas $n_{t}$.
    pub fn n_tx(&self) -> usize {
        self.n_tx
    }

    /// Channel uses per block $T$.
    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    /// Symbols per block $K$.
    pub fn n_symbols(&self) -> usize {
        self.a.len()
    }

    /// Symbols per channel use $K/T$.
    pub fn rate(&self) -> f64 {
        self.n_symbols() as f64 / self.n_slots as f64
    }

    /// The codeword of one block, $T \times n_{t}$.
    fn codeword(&self, s: &[C64]) -> CMat {
        let mut x = Array2::<C64>::zeros((self.n_slots, self.n_tx));
        for (k, sk) in s.iter().enumerate() {
            x = x + self.a[k].mapv(|v| v * sk.re * self.scale) + self.b[k].mapv(|v| v * sk.im * self.scale);
        }
        x
    }

    /// Encode symbols (a multiple of $K$) into the signal of the antennas,
    /// $n_{t} \times T \cdot$ blocks (antennas along the rows, channel uses along the
    /// columns).
    ///
    /// # Errors
    /// [`Error::Shape`] if the number of symbols is not a multiple of $K$.
    pub fn encode(&self, s: &[C64]) -> Result<CMat> {
        let k = self.n_symbols();
        if s.len() % k != 0 {
            return Err(Error::Shape(format!("{} symbols is not a multiple of {k} per block", s.len())));
        }
        let blocks = s.len() / k;
        let mut out = Array2::<C64>::zeros((self.n_tx, blocks * self.n_slots));
        for (b, chunk) in s.chunks(k).enumerate() {
            let x = self.codeword(chunk);
            for t in 0..self.n_slots {
                for a in 0..self.n_tx {
                    out[[a, b * self.n_slots + t]] = x[[t, a]];
                }
            }
        }
        Ok(out)
    }

    /// The columns of the real equivalent channel: $\mathrm{rv}(H A_{k}^{T})$ then
    /// $\mathrm{rv}(H B_{k}^{T})$, scaled ($\mathrm{rv}$ stacks real and imaginary parts).
    fn equivalent(&self, h: &ArrayView2<'_, C64>) -> Vec<Vec<f64>> {
        self.a
            .iter()
            .chain(&self.b)
            .map(|m| {
                let v = h.dot(&m.t());
                v.iter().map(|x| x.re * self.scale).chain(v.iter().map(|x| x.im * self.scale)).collect()
            })
            .collect()
    }

    /// Decode received blocks that share the channel `h` ($n_{r} \times n_{t}$): `y` is
    /// $n_{r} \times T \cdot$ blocks. Returns the estimate of every symbol and its noise
    /// variance (the receiver noise has the variance `noise_var` per antenna).
    ///
    /// # Errors
    /// [`Error::Shape`] for inconsistent dimensions, [`Error::Numerical`] for a zero
    /// channel.
    pub fn decode(
        &self,
        h: &ArrayView2<'_, C64>,
        y: &ArrayView2<'_, C64>,
        noise_var: f64,
    ) -> Result<(Vec<C64>, Vec<f64>)> {
        if y.ncols() % self.n_slots != 0 {
            return Err(Error::Shape(format!("{} channel uses is not a multiple of {}", y.ncols(), self.n_slots)));
        }
        let hs = vec![h.to_owned(); y.ncols() / self.n_slots];
        self.decode_blocks(&hs, y, noise_var)
    }

    /// [`Ostbc::decode`] with one channel matrix per block.
    ///
    /// # Errors
    /// As [`Ostbc::decode`].
    pub fn decode_blocks(&self, hs: &[CMat], y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<(Vec<C64>, Vec<f64>)> {
        let t = self.n_slots;
        if y.ncols() != hs.len() * t {
            return Err(Error::Shape(format!("{} channel uses for {} blocks of {t}", y.ncols(), hs.len())));
        }
        let k = self.n_symbols();
        let (mut est, mut var) = (Vec::with_capacity(hs.len() * k), Vec::with_capacity(hs.len() * k));
        for (blk, h) in hs.iter().enumerate() {
            if h.dim() != (y.nrows(), self.n_tx) {
                return Err(Error::Shape(format!(
                    "a {}×{} channel for {} receive and {} transmit antennas",
                    h.nrows(),
                    h.ncols(),
                    y.nrows(),
                    self.n_tx
                )));
            }
            let cols = self.equivalent(&h.view());
            let block: Vec<C64> =
                (0..y.nrows()).flat_map(|r| (0..t).map(move |c| (r, c))).map(|(r, c)| y[[r, blk * t + c]]).collect();
            let yr: Vec<f64> = block.iter().map(|x| x.re).chain(block.iter().map(|x| x.im)).collect();
            let project = |u: &[f64]| -> Result<(f64, f64)> {
                let n2: f64 = u.iter().map(|x| x * x).sum();
                ensure(n2 > 1e-300, || "zero channel".into()).map_err(|_| Error::Numerical("zero channel".into()))?;
                Ok((u.iter().zip(&yr).map(|(a, b)| a * b).sum::<f64>() / n2, n2))
            };
            for i in 0..k {
                let (re, nu) = project(&cols[i])?;
                let (im, nw) = project(&cols[k + i])?;
                est.push(C64::new(re, im));
                var.push(noise_var / 2.0 * (1.0 / nu + 1.0 / nw));
            }
        }
        Ok((est, var))
    }
}

/// The code as a precoder: the first stream of `x` (a multiple of $K$ symbols) becomes
/// the $n_{t}$ streams of the antennas.
impl Precode for Ostbc {
    fn precode_into(&self, x: &Symbols, out: &mut Symbols) -> Result<()> {
        ensure(x.n_streams() == 1, || format!("a space-time code encodes one stream, not {}", x.n_streams()))?;
        out.assign(&self.encode(x.stream_slice(0))?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_math::linalg::gram;

    fn random_symbols(n: usize, rng: &mut Rng) -> Vec<C64> {
        (0..n).map(|_| rng.complex_gaussian()).collect()
    }

    #[test]
    fn codewords_are_orthogonal_with_unit_average_energy() {
        let mut rng = Rng::new(1);
        for code in [Ostbc::alamouti(), Ostbc::rate_3_4(3).unwrap(), Ostbc::rate_3_4(4).unwrap()] {
            let s = random_symbols(code.n_symbols(), &mut rng);
            let x = code.codeword(&s);
            let g = gram(&x.view());
            let e: f64 = s.iter().map(C64::norm_sqr).sum();
            for i in 0..code.n_tx() {
                for j in 0..code.n_tx() {
                    let want = if i == j { code.scale * code.scale * e } else { 0.0 };
                    assert!((g[[i, j]] - want).norm() < 1e-12, "{}", code.name());
                }
            }
            // E‖X‖²/T = 1 for unit-energy symbols
            let n = 20_000;
            let p: f64 = (0..n)
                .map(|_| {
                    code.codeword(&random_symbols(code.n_symbols(), &mut rng)).iter().map(C64::norm_sqr).sum::<f64>()
                })
                .sum::<f64>()
                / (f64::from(n) * code.n_slots() as f64);
            assert!((p - 1.0).abs() < 0.03, "{}: {p}", code.name());
        }
        assert!(Ostbc::rate_3_4(2).is_err());
        // a non-orthogonal design is rejected
        assert!(Ostbc::from_generator("bad", 2, 1, 2, |s| ndarray::array![[s[0], s[1]]]).is_err());
    }

    #[test]
    fn noiseless_decoding_recovers_the_symbols_over_any_channel() {
        let mut rng = Rng::new(2);
        for code in [Ostbc::alamouti(), Ostbc::rate_3_4(3).unwrap(), Ostbc::rate_3_4(4).unwrap()] {
            for nr in [1, 2, 3] {
                let s = random_symbols(code.n_symbols() * 5, &mut rng);
                let x = code.encode(&s).unwrap();
                let hs: Vec<CMat> =
                    (0..5).map(|_| Array2::from_shape_fn((nr, code.n_tx()), |_| rng.complex_gaussian())).collect();
                let t = code.n_slots();
                let mut y = Array2::<C64>::zeros((nr, x.ncols()));
                for (b, h) in hs.iter().enumerate() {
                    for c in 0..t {
                        y.column_mut(b * t + c).assign(&h.dot(&x.column(b * t + c)));
                    }
                }
                let (est, var) = code.decode_blocks(&hs, &y.view(), 0.1).unwrap();
                assert!(est.iter().zip(&s).all(|(a, b)| (a - b).norm() < 1e-10), "{} nr={nr}", code.name());
                // the noise variance is σ²/(c² ‖H‖²) for every symbol of a block
                let h0: f64 = hs[0].iter().map(C64::norm_sqr).sum();
                assert!((var[0] - 0.1 / (code.scale * code.scale * h0)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn alamouti_noise_matches_its_prediction() {
        let code = Ostbc::alamouti();
        let mut rng = Rng::new(3);
        let h = Array2::from_shape_fn((2, 2), |_| rng.complex_gaussian());
        let n = 20_000;
        let s = random_symbols(2 * n, &mut rng);
        let x = code.encode(&s).unwrap();
        let s2: f64 = 0.3;
        let y = h.dot(&x).mapv(|v| v + rng.complex_gaussian() * s2.sqrt());
        let (est, var) = code.decode(&h.view(), &y.view(), s2).unwrap();
        let mse: f64 = est.iter().zip(&s).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / s.len() as f64;
        assert!((mse / var[0] - 1.0).abs() < 0.03, "{mse} vs {}", var[0]);
    }
}

/// Checks against the published designs (`references/`).
#[cfg(test)]
mod literature {
    use super::*;
    use com_core::rng::Rng;

    /// V. Tarokh, H. Jafarkhani and A. R. Calderbank, "Space-time block codes from
    /// orthogonal designs," IEEE Trans. Inf. Theory 45(5), 1999, eqs. (39) and (40)
    /// (pp. 1464-1465, checked term by term against the paper): the rate-3/4 designs for
    /// three and four antennas, (39) being the first three columns of (40). The library's
    /// rate-3/4 code is another (sporadic) design; the paper's are accepted by the
    /// orthogonality check of `from_generator` and decoded exactly without noise.
    #[test]
    fn tarokh_1999_rate_3_4_designs_are_orthogonal() {
        let r2 = std::f64::consts::SQRT_2;
        let eq40 = |s: &[C64]| {
            let (x1, x2, x3) = (s[0], s[1], s[2]);
            let (c1, c2, c3) = (x1.conj(), x2.conj(), x3.conj());
            ndarray::array![
                [x1, x2, x3 / r2, x3 / r2],
                [-c2, c1, x3 / r2, -x3 / r2],
                [c3 / r2, c3 / r2, (-x1 - c1 + x2 - c2) / 2.0, (-x2 - c2 + x1 - c1) / 2.0],
                [c3 / r2, -c3 / r2, (x2 + c2 + x1 - c1) / 2.0, -(x1 + c1 + x2 - c2) / 2.0],
            ]
        };
        let eq39 = move |s: &[C64]| eq40(s).slice_axis(ndarray::Axis(1), ndarray::Slice::from(0..3)).to_owned();
        let mut rng = Rng::new(1999);
        for (n_tx, g) in [(4, Box::new(eq40) as Box<dyn Fn(&[C64]) -> CMat>), (3, Box::new(eq39))] {
            let code = Ostbc::from_generator("Tarokh 1999", n_tx, 4, 3, g).unwrap();
            assert!((code.rate() - 0.75).abs() < 1e-12);
            let s: Vec<C64> = (0..12).map(|_| rng.complex_gaussian()).collect();
            let x = code.encode(&s).unwrap();
            let h = Array2::from_shape_fn((2, n_tx), |_| rng.complex_gaussian());
            let y = h.dot(&x);
            let (decoded, _) = code.decode(&h.view(), &y.view(), 1e-12).unwrap();
            assert!(decoded.iter().zip(&s).all(|(a, b)| (a - b).norm() < 1e-9), "{n_tx} antennas");
        }
    }
}
