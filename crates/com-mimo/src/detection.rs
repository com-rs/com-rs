//! MIMO detection on the generic model $y = H x + w$.
//!
//! A detector recovers the $n_{t}$ symbols of $x$, drawn from a constellation, from
//! the $n_{r}$ received samples $y$, given the channel $H$ ($n_{r} \times n_{t}$) and the
//! noise variance $\sigma^{2} = \mathbb{E}|w_{i}|^{2}$:
//!
//! | Detector | Principle | Complexity | Diversity ($n_{r} \ge n_{t}$, Rayleigh) |
//! |---|---|---|---|
//! | [`Linear::zf`] | $\hat x = (H^{H} H)^{-1} H^{H} y$ | cubic in $n_{t}$ | $n_{r} - n_{t} + 1$ |
//! | [`Linear::mmse`] | $\hat x = (H^{H} H + \frac{\sigma^{2}}{E_{s}} I)^{-1} H^{H} y$, unbiased | cubic | $n_{r} - n_{t} + 1$ at high SNR, better at low SNR |
//! | [`Sic`] | ordered successive interference cancellation (V-BLAST) | $n_{t}$ linear filters | between linear and ML |
//! | [`Ml`] | $\arg\min_{x} \lVert y - H x\rVert^{2}$ by exhaustive search | $M^{n_{t}}$ | $n_{r}$ |
//! | [`Sphere`] | the same minimum by Schnorr–Euchner depth-first search | variable, exact ML | $n_{r}$ |
//! | [`KBest`] | breadth-first tree search keeping $K$ paths | $K M n_{t}$ | close to ML |
//!
//! Every detector returns the label (index) of the decided point of every transmit
//! stream ([`MimoDetector::detect`]); the linear detectors, [`Ml`] and [`KBest`] also
//! give bit log-likelihood ratios ([`MimoDetector::llr`], positive for a 0, most
//! significant bit first, stream after stream). [`Coherent`] binds a detector to a
//! channel matrix and implements the [`Detect`] trait on multi-stream containers.
//!
//! # Model
//!
//! The linear detectors filter with $W = (H^{H} H + \lambda I)^{-1} H^{H}$ ($\lambda = 0$
//! for ZF, $\sigma^{2}/E_{s}$ for MMSE). With $E = \sigma^{2} (H^{H} H + \lambda I)^{-1}$,
//! the ZF estimate of stream $k$ has the noise variance $E_{kk}$; the MMSE estimate is
//! divided by its bias $\beta_{k} = 1 - E_{kk}/E_{s}$, after which its noise variance is
//! $E_{s} E_{kk} / (E_{s} - E_{kk})$ (the SINR is $E_{s}/E_{kk} - 1$). The soft outputs
//! treat each estimate as one constellation symbol in Gaussian noise of that variance.
//!
//! The max-log LLR of a tree or exhaustive search is
//!
//! $$
//! L(b) = \frac{1}{\sigma^{2}}\Bigl(\min_{x:\\, b = 1} \lVert y - H x \rVert^{2}
//!        - \min_{x:\\, b = 0} \lVert y - H x \rVert^{2}\Bigr)
//! $$
//!
//! # Examples
//! ```
//! use com_mimo::detection::{Linear, MimoDetector, Sphere};
//! use com_mapping::constellation::Constellation;
//! use com_math::{linalg::matvec, C64};
//! use ndarray::array;
//! let c = Constellation::qam(4)?.normalized();
//! let h = array![[C64::new(1.0, 0.2), C64::new(0.3, -0.5)], [C64::new(-0.4, 0.1), C64::new(0.9, 0.0)]];
//! let x = [c.points()[2], c.points()[1]];
//! let y = matvec(&h.view(), &x);                         // noiseless
//! assert_eq!(Linear::mmse(c.clone()).detect(&h.view(), &y, 1e-3)?, vec![2, 1]);
//! assert_eq!(Sphere::new(c).detect(&h.view(), &y, 1e-3)?, vec![2, 1]);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. P. W. Wolniansky, G. J. Foschini, G. D. Golden and R. A. Valenzuela, "V-BLAST: an
//!    architecture for realizing very high data rates over the rich-scattering
//!    wireless channel," *Proc. ISSSE*, 1998.
//! 2. E. Agrell, T. Eriksson, A. Vardy and K. Zeger, "Closest point search in
//!    lattices," *IEEE Trans. Inf. Theory*, 48(8), 2002.
//! 3. Z. Guo and P. Nilsson, "Algorithm and implementation of the K-best sphere
//!    decoding for MIMO detection," *IEEE J. Sel. Areas Commun.*, 24(3), 2006.

use com_core::signal::Symbols;
use com_core::traits::Detect;
use com_core::{Error, Result, ensure};
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_math::linalg::{CMat, hermitian, identity, inverse_hpd, matvec, qr};
use ndarray::{Array2, ArrayView2};

/// Largest number of hypotheses $M^{n_{t}}$ of the exhaustive search.
pub const ML_MAX_HYPOTHESES: usize = 1 << 22;

/// A detector of the model $y = H x + w$.
pub trait MimoDetector {
    /// The label of the decided point of every transmit stream.
    ///
    /// # Errors
    /// [`Error::Shape`] if `y` does not have `n_rx` entries, [`Error::Config`] for a
    /// channel the detector cannot handle (fewer receive than transmit antennas for
    /// ZF and the tree searches, too many hypotheses for [`Ml`]).
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>>;

    /// Bit LLRs, $n_{t} \log_{2} M$ of them (positive for a 0).
    ///
    /// # Errors
    /// As [`MimoDetector::detect`]; [`Error::Config`] if the detector has no soft
    /// output.
    fn llr(&self, _h: &ArrayView2<'_, C64>, _y: &[C64], _noise_var: f64) -> Result<Vec<f64>> {
        Err(Error::Config("this detector has no soft output".into()))
    }

    /// The labels and the bit LLRs of one channel use, those of
    /// [`MimoDetector::detect`] and [`MimoDetector::llr`]; the detectors with soft
    /// output get both from one search.
    ///
    /// # Errors
    /// As [`MimoDetector::llr`].
    fn detect_llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<(Vec<usize>, Vec<f64>)> {
        Ok((self.detect(h, y, noise_var)?, self.llr(h, y, noise_var)?))
    }

    /// Detect every column of `y` (`n_rx × T`) with the same channel: the labels,
    /// `n_tx × T`. The detectors of this module factor the channel once for the block.
    ///
    /// # Errors
    /// As [`MimoDetector::detect`].
    fn detect_block(&self, h: &ArrayView2<'_, C64>, y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<Array2<usize>> {
        let mut out = Array2::zeros((h.ncols(), y.ncols()));
        for (t, col) in y.columns().into_iter().enumerate() {
            let d = self.detect(h, &col.to_vec(), noise_var)?;
            out.column_mut(t).assign(&ndarray::Array1::from(d));
        }
        Ok(out)
    }
}

fn check(h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<()> {
    if y.len() != h.nrows() {
        return Err(Error::Shape(format!("{} received samples for {} receive antennas", y.len(), h.nrows())));
    }
    ensure(noise_var > 0.0 && noise_var.is_finite(), || format!("the noise variance must be positive, got {noise_var}"))
}

fn tall(h: &ArrayView2<'_, C64>, what: &str) -> Result<()> {
    ensure(h.nrows() >= h.ncols(), || {
        format!("{what} needs at least as many receive ({}) as transmit ({}) antennas", h.nrows(), h.ncols())
    })
}

/// The checks of a block `y` (`n_rx × T`) detected with one channel.
fn check_block(h: &ArrayView2<'_, C64>, y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<()> {
    if y.nrows() != h.nrows() {
        return Err(Error::Shape(format!("{} received streams for {} receive antennas", y.nrows(), h.nrows())));
    }
    check(h, &vec![C64::new(0.0, 0.0); h.nrows()], noise_var)
}

/// Detect every column of `y` with `detect` into the `n_tx × T` labels.
fn by_column(
    nt: usize,
    y: &ArrayView2<'_, C64>,
    mut detect: impl FnMut(&mut [C64]) -> Result<Vec<usize>>,
) -> Result<Array2<usize>> {
    let mut out = Array2::zeros((nt, y.ncols()));
    let mut col = vec![C64::new(0.0, 0.0); y.nrows()];
    for (t, c) in y.columns().into_iter().enumerate() {
        col.iter_mut().zip(c).for_each(|(a, b)| *a = *b);
        for (o, l) in out.column_mut(t).iter_mut().zip(detect(&mut col)?) {
            *o = l;
        }
    }
    Ok(out)
}

/// Zero-forcing or MMSE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Criterion {
    /// Zero forcing: the pseudo-inverse of the channel.
    Zf,
    /// Minimum mean square error, unbiased.
    Mmse,
}

/// The filter of a linear detector and the noise variance of each unbiased estimate.
struct Filter {
    w: CMat,
    bias: Vec<f64>,
    var: Vec<f64>,
}

/// The regularisation $\lambda$ of the Gram matrix: 0 for zero forcing (which needs at
/// least as many receive as transmit antennas), $\sigma^{2}/E_{s}$ for MMSE.
fn regularisation(h: &ArrayView2<'_, C64>, criterion: Criterion, es: f64, noise_var: f64) -> Result<f64> {
    match criterion {
        Criterion::Zf => tall(h, "zero forcing").map(|()| 0.0),
        Criterion::Mmse => Ok(noise_var / es),
    }
}

fn filter(h: &ArrayView2<'_, C64>, criterion: Criterion, es: f64, noise_var: f64) -> Result<Filter> {
    let n = h.ncols();
    let lambda = regularisation(h, criterion, es, noise_var)?;
    if n == 1 {
        let s = SingleStream::new(h, criterion, es, noise_var, lambda)?;
        return Ok(Filter { w: h.t().mapv(|v| v.conj() / s.g), bias: vec![s.bias], var: vec![s.var] });
    }
    let hh = hermitian(h);
    let g = hh.dot(h) + identity(n).mapv(|x| x * lambda);
    let ginv = inverse_hpd(&g.view()).map_err(|_| Error::Numerical("the channel is rank deficient".into()))?;
    let w = ginv.dot(&hh);
    let (mut bias, mut var) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for k in 0..n {
        let ekk = noise_var * ginv[[k, k]].re;
        match criterion {
            Criterion::Zf => {
                bias.push(1.0);
                var.push(ekk);
            }
            Criterion::Mmse => {
                let b = (1.0 - ekk / es).max(1e-300);
                bias.push(b);
                var.push(es * ekk / (es - ekk).max(1e-300));
            }
        }
    }
    Ok(Filter { w, bias, var })
}

/// The filter of one stream (SISO, or SIMO combining) as scalars: $G = \|h\|^{2} +
/// \lambda$, so that $w = h^{H}/G$ (maximal-ratio combining for zero forcing), the bias
/// and the error variance $\sigma^{2}/G$. The same filter as the general case, whose
/// $1 \times 1$ matrices and their allocations cost more than the arithmetic.
struct SingleStream {
    g: f64,
    bias: f64,
    var: f64,
}

impl SingleStream {
    fn new(
        h: &ArrayView2<'_, C64>,
        criterion: Criterion,
        es: f64,
        noise_var: f64,
        lambda: f64,
    ) -> Result<SingleStream> {
        let g = h.iter().map(C64::norm_sqr).sum::<f64>() + lambda;
        if !(g > 0.0 && g.is_finite()) {
            return Err(Error::Numerical("the channel is rank deficient".into()));
        }
        let ekk = noise_var / g;
        let (bias, var) = match criterion {
            Criterion::Zf => (1.0, ekk),
            Criterion::Mmse => ((1.0 - ekk / es).max(1e-300), es * ekk / (es - ekk).max(1e-300)),
        };
        Ok(SingleStream { g, bias, var })
    }

    /// The unbiased estimate $w y / b$.
    fn estimate(&self, h: &ArrayView2<'_, C64>, y: &[C64]) -> C64 {
        h.column(0).iter().zip(y).map(|(hi, yi)| hi.conj() / self.g * yi).sum::<C64>() / self.bias
    }
}

/// A linear detector (ZF or MMSE) followed by symbol-by-symbol decisions.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    /// The alphabet of every stream.
    pub constellation: Constellation,
    /// ZF or MMSE.
    pub criterion: Criterion,
}

impl Linear {
    /// The zero-forcing detector.
    pub fn zf(constellation: Constellation) -> Linear {
        Linear { constellation, criterion: Criterion::Zf }
    }

    /// The (unbiased) MMSE detector.
    pub fn mmse(constellation: Constellation) -> Linear {
        Linear { constellation, criterion: Criterion::Mmse }
    }

    /// The unbiased estimates of the transmitted symbols and their noise variances.
    ///
    /// # Errors
    /// As [`MimoDetector::detect`]; [`Error::Numerical`] for a rank-deficient channel.
    pub fn estimate(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<(Vec<C64>, Vec<f64>)> {
        check(h, y, noise_var)?;
        let (z, var) = self.single(h, y, noise_var)?;
        if let Some(z) = z {
            return Ok((vec![z], vec![var]));
        }
        let f = filter(h, self.criterion, self.constellation.energy(), noise_var)?;
        let z = matvec(&f.w.view(), y).iter().zip(&f.bias).map(|(z, b)| z / b).collect();
        Ok((z, f.var))
    }

    /// The estimate and its variance for one stream, `None` for several streams.
    fn single(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<(Option<C64>, f64)> {
        if h.ncols() != 1 {
            return Ok((None, 0.0));
        }
        let es = self.constellation.energy();
        let s = SingleStream::new(h, self.criterion, es, noise_var, regularisation(h, self.criterion, es, noise_var)?)?;
        Ok((Some(s.estimate(h, y)), s.var))
    }
}

impl MimoDetector for Linear {
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
        let (z, _) = self.estimate(h, y, noise_var)?;
        Ok(self.constellation.decide(&z))
    }

    fn llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<f64>> {
        let (z, var) = self.estimate(h, y, noise_var)?;
        self.constellation.llr_with_variances(&z, &var, None, false)
    }

    fn detect_llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<(Vec<usize>, Vec<f64>)> {
        let (z, var) = self.estimate(h, y, noise_var)?;
        Ok((self.constellation.decide(&z), self.constellation.llr_with_variances(&z, &var, None, false)?))
    }

    fn detect_block(&self, h: &ArrayView2<'_, C64>, y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<Array2<usize>> {
        check_block(h, y, noise_var)?;
        let f = filter(h, self.criterion, self.constellation.energy(), noise_var)?;
        let z = f.w.dot(y);
        Ok(Array2::from_shape_fn(z.dim(), |(k, t)| self.constellation.nearest(z[[k, t]] / f.bias[k])))
    }
}

/// Successive interference cancellation (V-BLAST): detect the stream of largest
/// post-detection SINR with a linear filter, subtract it, repeat.
#[derive(Clone, Debug, PartialEq)]
pub struct Sic {
    /// The alphabet of every stream.
    pub constellation: Constellation,
    /// The filter of every stage.
    pub criterion: Criterion,
    /// Order the streams by SINR (V-BLAST); in their natural order otherwise.
    pub ordered: bool,
}

impl Sic {
    /// Ordered MMSE-SIC.
    pub fn mmse(constellation: Constellation) -> Sic {
        Sic { constellation, criterion: Criterion::Mmse, ordered: true }
    }

    /// Ordered ZF-SIC (the original V-BLAST).
    pub fn zf(constellation: Constellation) -> Sic {
        Sic { constellation, criterion: Criterion::Zf, ordered: true }
    }
}

/// One stage of [`Sic`]: the stream detected, the row of its filter, its bias and its
/// column of the channel (the filters depend on the channel only).
struct SicStage {
    stream: usize,
    w: Vec<C64>,
    bias: f64,
    column: Vec<C64>,
}

impl Sic {
    /// The stages of the cancellation for the channel `h`, in their order.
    fn stages(&self, h: &ArrayView2<'_, C64>, noise_var: f64) -> Result<Vec<SicStage>> {
        let es = self.constellation.energy();
        let mut remaining: Vec<usize> = (0..h.ncols()).collect();
        let mut stages = Vec::with_capacity(h.ncols());
        while !remaining.is_empty() {
            let hc = Array2::from_shape_fn((h.nrows(), remaining.len()), |(i, j)| h[[i, remaining[j]]]);
            let f = filter(&hc.view(), self.criterion, es, noise_var)?;
            let k = if self.ordered {
                (0..remaining.len()).min_by(|a, b| f.var[*a].total_cmp(&f.var[*b])).unwrap_or(0)
            } else {
                0
            };
            stages.push(SicStage {
                stream: remaining[k],
                w: f.w.row(k).to_vec(),
                bias: f.bias[k],
                column: hc.column(k).to_vec(),
            });
            remaining.remove(k);
        }
        Ok(stages)
    }

    /// Detect, subtract and repeat along the stages; `residual` starts as $y$.
    fn cancel(&self, stages: &[SicStage], residual: &mut [C64]) -> Vec<usize> {
        let mut out = vec![0; stages.len()];
        for s in stages {
            let z: C64 = s.w.iter().zip(residual.iter()).map(|(w, v)| w * v).sum::<C64>() / s.bias;
            let idx = self.constellation.nearest(z);
            let p = self.constellation.points()[idx];
            for (r, hk) in residual.iter_mut().zip(&s.column) {
                *r -= hk * p;
            }
            out[s.stream] = idx;
        }
        out
    }
}

impl MimoDetector for Sic {
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
        check(h, y, noise_var)?;
        let stages = self.stages(h, noise_var)?;
        Ok(self.cancel(&stages, &mut y.to_vec()))
    }

    fn detect_block(&self, h: &ArrayView2<'_, C64>, y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<Array2<usize>> {
        check_block(h, y, noise_var)?;
        let stages = self.stages(h, noise_var)?;
        by_column(h.ncols(), y, |col| Ok(self.cancel(&stages, col)))
    }
}

/// Maximum-likelihood detection by exhaustive search over the $M^{n_{t}}$ hypotheses.
#[derive(Clone, Debug, PartialEq)]
pub struct Ml {
    /// The alphabet of every stream.
    pub constellation: Constellation,
}

impl Ml {
    /// The exhaustive ML detector.
    pub fn new(constellation: Constellation) -> Ml {
        Ml { constellation }
    }

    /// The products $h_{rk} s_{i}$ of every channel coefficient with every point, at
    /// `[(k M + i) n_r + r]`: the terms of $H x$ for every hypothesis, computed once
    /// per channel.
    ///
    /// # Errors
    /// [`Error::Config`] when $M^{n_{t}}$ exceeds [`ML_MAX_HYPOTHESES`].
    fn products(&self, h: &ArrayView2<'_, C64>) -> Result<Vec<C64>> {
        let (m, nt, nr) = (self.constellation.order(), h.ncols(), h.nrows());
        let total = (0..nt).try_fold(1usize, |acc, _| acc.checked_mul(m)).filter(|t| *t <= ML_MAX_HYPOTHESES);
        ensure(total.is_some(), || format!("{m}^{nt} hypotheses exceed the limit of the exhaustive search"))?;
        let pts = self.constellation.points();
        let mut prod = Vec::with_capacity(nt * m * nr);
        for k in 0..nt {
            for p in pts {
                prod.extend((0..nr).map(|r| h[[r, k]] * p));
            }
        }
        Ok(prod)
    }

    /// Visit every hypothesis with its distance $\lVert y - H x \rVert^{2}$, with the
    /// products of [`Ml::products`] (the sums of [`matvec`], term by term: the same
    /// bits).
    fn search(&self, prod: &[C64], nt: usize, y: &[C64], mut visit: impl FnMut(&[usize], f64)) {
        let (m, nr) = (self.constellation.order(), y.len());
        let mut idx = vec![0usize; nt];
        loop {
            let d: f64 = (0..nr)
                .map(|r| {
                    let a: C64 = idx.iter().enumerate().map(|(k, &i)| prod[(k * m + i) * nr + r]).sum();
                    (y[r] - a).norm_sqr()
                })
                .sum();
            visit(&idx, d);
            // odometer over the hypotheses
            let mut k = 0;
            loop {
                if k == nt {
                    return;
                }
                idx[k] += 1;
                if idx[k] < m {
                    break;
                }
                idx[k] = 0;
                k += 1;
            }
        }
    }

    /// The labels of the smallest distance (the first one found among equals).
    fn best(&self, prod: &[C64], nt: usize, y: &[C64]) -> Vec<usize> {
        let mut best = (f64::INFINITY, vec![0; nt]);
        self.search(prod, nt, y, |idx, d| {
            if d < best.0 {
                best.0 = d;
                best.1.copy_from_slice(idx);
            }
        });
        best.1
    }
}

/// Max-log LLRs from the best distance found for every bit value.
fn max_log(best: &[[f64; 2]], noise_var: f64, clip: f64) -> Vec<f64> {
    best.iter()
        .map(|[d0, d1]| match (d0.is_finite(), d1.is_finite()) {
            (true, true) => ((d1 - d0) / noise_var).clamp(-clip, clip),
            (true, false) => clip,
            (false, true) => -clip,
            (false, false) => 0.0,
        })
        .collect()
}

/// Record a hypothesis in the per-bit minima.
fn update_bits(best: &mut [[f64; 2]], idx: &[usize], bits: usize, d: f64) {
    for (s, &i) in idx.iter().enumerate() {
        for p in 0..bits {
            let b = (i >> (bits - 1 - p)) & 1;
            let slot = &mut best[s * bits + p][b];
            if d < *slot {
                *slot = d;
            }
        }
    }
}

impl MimoDetector for Ml {
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
        check(h, y, noise_var)?;
        Ok(self.best(&self.products(h)?, h.ncols(), y))
    }

    fn llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<f64>> {
        Ok(self.detect_llr(h, y, noise_var)?.1)
    }

    fn detect_llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<(Vec<usize>, Vec<f64>)> {
        check(h, y, noise_var)?;
        let prod = self.products(h)?;
        let (nt, bits) = (h.ncols(), self.constellation.bits_per_symbol());
        let mut best = (f64::INFINITY, vec![0; nt]);
        let mut per_bit = vec![[f64::INFINITY; 2]; nt * bits];
        self.search(&prod, nt, y, |idx, d| {
            if d < best.0 {
                best.0 = d;
                best.1.copy_from_slice(idx);
            }
            update_bits(&mut per_bit, idx, bits, d);
        });
        Ok((best.1, max_log(&per_bit, noise_var, f64::INFINITY)))
    }

    fn detect_block(&self, h: &ArrayView2<'_, C64>, y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<Array2<usize>> {
        check_block(h, y, noise_var)?;
        let prod = self.products(h)?;
        by_column(h.ncols(), y, |col| Ok(self.best(&prod, h.ncols(), col)))
    }
}

/// The QR form of the search for one channel:
/// $\lVert y - H x\rVert^{2} = \lVert Q^{H} y - R x\rVert^{2} + c$.
struct Triangular {
    r: CMat,
    qh: CMat,
}

impl Triangular {
    /// The factors of `h`; `what` names the detector in the errors.
    fn new(h: &ArrayView2<'_, C64>, what: &str) -> Result<Triangular> {
        tall(h, what)?;
        let (q, r) = qr(h)?;
        if (0..r.nrows()).any(|i| r[[i, i]].re <= 0.0) {
            return Err(Error::Numerical("the channel is rank deficient".into()));
        }
        Ok(Triangular { r, qh: hermitian(&q.view()) })
    }

    /// The rotated observation $z = Q^{H} y$.
    fn rotate(&self, y: &[C64]) -> Vec<C64> {
        matvec(&self.qh.view(), y)
    }
}

/// One level of the depth-first search: the candidates sorted by partial distance,
/// the next one to try, and the distance of the path above.
type Frame = (Vec<(f64, usize)>, usize, f64);

/// Exact ML detection by the Schnorr–Euchner sphere decoder.
///
/// After $H = Q R$, the tree of the streams is searched depth first from the last one,
/// visiting the points of each level by increasing partial distance and pruning every
/// branch whose partial distance exceeds the best complete one found so far.
#[derive(Clone, Debug, PartialEq)]
pub struct Sphere {
    /// The alphabet of every stream.
    pub constellation: Constellation,
}

impl Sphere {
    /// The sphere decoder.
    pub fn new(constellation: Constellation) -> Sphere {
        Sphere { constellation }
    }

    /// The search of one rotated observation $z = Q^{H} y$.
    fn search(&self, r: &CMat, z: &[C64]) -> Vec<usize> {
        let n = r.ncols();
        let pts = self.constellation.points();
        let mut best = (f64::INFINITY, vec![0usize; n]);
        let mut x = vec![C64::new(0.0, 0.0); n];
        let mut idx = vec![0usize; n];
        // one frame per level: the candidates sorted by partial distance, the next to try;
        // the candidate lists of the frames left are reused
        let mut frames: Vec<Frame> = Vec::with_capacity(n);
        let mut spare: Vec<Vec<(f64, usize)>> = Vec::with_capacity(n);
        let candidates = |k: usize, x: &[C64], mut c: Vec<(f64, usize)>| -> Vec<(f64, usize)> {
            let mut b = z[k];
            for j in k + 1..n {
                b -= r[[k, j]] * x[j];
            }
            c.clear();
            c.extend(pts.iter().enumerate().map(|(i, p)| ((b - r[[k, k]] * p).norm_sqr(), i)));
            c.sort_by(|a, b| a.0.total_cmp(&b.0));
            c
        };
        frames.push((candidates(n - 1, &x, Vec::with_capacity(pts.len())), 0, 0.0));
        while !frames.is_empty() {
            let k = n - frames.len();
            let Some(f) = frames.last_mut() else { break };
            let candidate = f.0.get(f.1).copied();
            f.1 += 1;
            let base = f.2;
            let next = match candidate {
                Some((d, i)) if base + d < best.0 => Some((base + d, i)),
                // no candidate left, or (sorted) none of this level can do better
                _ => None,
            };
            let Some((partial, i)) = next else {
                if let Some(f) = frames.pop() {
                    spare.push(f.0);
                }
                continue;
            };
            x[k] = pts[i];
            idx[k] = i;
            if k == 0 {
                best.0 = partial;
                best.1.copy_from_slice(&idx);
            } else {
                let c = candidates(k - 1, &x, spare.pop().unwrap_or_default());
                frames.push((c, 0, partial));
            }
        }
        best.1
    }
}

impl MimoDetector for Sphere {
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
        check(h, y, noise_var)?;
        let t = Triangular::new(h, "the sphere decoder")?;
        Ok(self.search(&t.r, &t.rotate(y)))
    }

    fn detect_block(&self, h: &ArrayView2<'_, C64>, y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<Array2<usize>> {
        check_block(h, y, noise_var)?;
        let t = Triangular::new(h, "the sphere decoder")?;
        by_column(h.ncols(), y, |col| Ok(self.search(&t.r, &t.rotate(col))))
    }
}

/// The K-best tree search: breadth first from the last stream, keeping the `k` paths
/// of smallest partial distance at every level.
#[derive(Clone, Debug, PartialEq)]
pub struct KBest {
    /// The alphabet of every stream.
    pub constellation: Constellation,
    /// Paths kept at every level.
    pub k: usize,
    /// Magnitude of the LLR of a bit whose counter-hypothesis left the list.
    pub llr_clip: f64,
}

/// The surviving paths of [`KBest`], best first: the labels of path `p` (in the order
/// of the streams) are `labels[p n_t .. (p + 1) n_t]`, its distance `dist[p]`.
struct Paths {
    labels: Vec<usize>,
    dist: Vec<f64>,
}

impl Paths {
    fn iter(&self, nt: usize) -> impl Iterator<Item = (&[usize], f64)> {
        // `max(1)`: with no stream, the one path has no label
        self.labels.chunks_exact(nt.max(1)).zip(self.dist.iter().copied())
    }

    fn first(&self, nt: usize) -> Vec<usize> {
        self.labels.get(..nt).map(<[usize]>::to_vec).unwrap_or_default()
    }
}

impl KBest {
    /// The search keeping `k` paths.
    pub fn new(constellation: Constellation, k: usize) -> KBest {
        KBest { constellation, k, llr_clip: 30.0 }
    }

    /// The factors of the channel, after the checks of one channel use.
    fn prepare(&self, h: &ArrayView2<'_, C64>) -> Result<Triangular> {
        ensure(self.k >= 1, || "K-best needs k ≥ 1".into())?;
        Triangular::new(h, "K-best")
    }

    /// The surviving paths of one rotated observation $z = Q^{H} y$.
    fn search(&self, r: &CMat, z: &[C64]) -> Paths {
        let n = r.ncols();
        let pts = self.constellation.points();
        // during the search, a path holds the labels of the streams k..n-1, stored from
        // stream n-1 down, `depth` labels each
        let mut labels: Vec<usize> = Vec::new();
        let mut next_labels: Vec<usize> = Vec::new();
        let mut dist = vec![0.0];
        // the children (distance, parent, label), in the order of the parents then of
        // the points: the stable sort keeps the first of equal distances
        let mut children: Vec<(f64, usize, usize)> = Vec::new();
        for (depth, k) in (0..n).rev().enumerate() {
            children.clear();
            for (p, d) in dist.iter().enumerate() {
                let mut b = z[k];
                for (level, &l) in labels[p * depth..(p + 1) * depth].iter().enumerate() {
                    let j = n - 1 - level;
                    b -= r[[k, j]] * pts[l];
                }
                children.extend(pts.iter().enumerate().map(|(i, s)| (d + (b - r[[k, k]] * s).norm_sqr(), p, i)));
            }
            children.sort_by(|a, b| a.0.total_cmp(&b.0));
            children.truncate(self.k);
            next_labels.clear();
            dist.clear();
            for &(d, p, i) in &children {
                next_labels.extend_from_slice(&labels[p * depth..(p + 1) * depth]);
                next_labels.push(i);
                dist.push(d);
            }
            std::mem::swap(&mut labels, &mut next_labels);
        }
        labels.chunks_exact_mut(n.max(1)).for_each(<[usize]>::reverse);
        Paths { labels, dist }
    }

    /// The max-log LLRs of the surviving paths.
    fn soft(&self, paths: &Paths, nt: usize, noise_var: f64) -> Vec<f64> {
        let bits = self.constellation.bits_per_symbol();
        let mut best = vec![[f64::INFINITY; 2]; nt * bits];
        for (labels, d) in paths.iter(nt) {
            update_bits(&mut best, labels, bits, d);
        }
        max_log(&best, noise_var, self.llr_clip)
    }
}

impl MimoDetector for KBest {
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
        check(h, y, noise_var)?;
        let t = self.prepare(h)?;
        Ok(self.search(&t.r, &t.rotate(y)).first(h.ncols()))
    }

    fn llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<f64>> {
        Ok(self.detect_llr(h, y, noise_var)?.1)
    }

    fn detect_llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<(Vec<usize>, Vec<f64>)> {
        check(h, y, noise_var)?;
        let t = self.prepare(h)?;
        let paths = self.search(&t.r, &t.rotate(y));
        Ok((paths.first(h.ncols()), self.soft(&paths, h.ncols(), noise_var)))
    }

    fn detect_block(&self, h: &ArrayView2<'_, C64>, y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<Array2<usize>> {
        check_block(h, y, noise_var)?;
        let t = self.prepare(h)?;
        by_column(h.ncols(), y, |col| Ok(self.search(&t.r, &t.rotate(col)).first(h.ncols())))
    }
}

/// A detector bound to a channel matrix: the [`Detect`] block of a link whose
/// receiver knows the channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Coherent<D> {
    /// The detector.
    pub detector: D,
    /// The channel matrix ($n_{r} \times n_{t}$).
    pub h: CMat,
    /// The alphabet of the decided points (the detector's).
    pub constellation: Constellation,
}

impl<D: MimoDetector> Detect for Coherent<D> {
    /// Decide every column of `y` (`n_rx` streams) and write the decided points
    /// (`n_tx` streams).
    fn detect_into(&mut self, y: &Symbols, noise_var: f64, out: &mut Symbols) -> Result<()> {
        let labels = self.detector.detect_block(&self.h.view(), &y.as_array().view(), noise_var)?;
        let pts = self.constellation.points();
        out.assign(&labels.mapv(|i| pts[i]));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    fn channel(nr: usize, nt: usize, rng: &mut Rng) -> CMat {
        Array2::from_shape_fn((nr, nt), |_| rng.complex_gaussian())
    }

    fn noisy(h: &CMat, x: &[C64], s2: f64, rng: &mut Rng) -> Vec<C64> {
        matvec(&h.view(), x).into_iter().map(|v| v + rng.complex_gaussian() * s2.sqrt()).collect()
    }

    #[test]
    fn the_single_stream_filter_is_the_matrix_filter() {
        let mut rng = Rng::new(7);
        let es = 1.0;
        for nr in [1, 3] {
            let h = channel(nr, 1, &mut rng);
            for (criterion, lambda) in [(Criterion::Zf, 0.0), (Criterion::Mmse, 0.3 / es)] {
                let f = filter(&h.view(), criterion, es, 0.3).unwrap();
                // the general case: (HᴴH + λI)⁻¹ Hᴴ
                let hh = hermitian(&h.view());
                let ginv = inverse_hpd(&(hh.dot(&h) + identity(1).mapv(|x| x * lambda)).view()).unwrap();
                let w = ginv.dot(&hh);
                assert!(f.w.iter().zip(w.iter()).all(|(a, b)| (a - b).norm() < 1e-14 * b.norm().max(1.0)));
                assert!(
                    (f.var[0]
                        - match criterion {
                            Criterion::Zf => 0.3 * ginv[[0, 0]].re,
                            Criterion::Mmse => {
                                let e = 0.3 * ginv[[0, 0]].re;
                                es * e / (es - e)
                            }
                        })
                    .abs()
                        < 1e-14
                );
            }
        }
        // the scalar estimate is the product of the filter
        let qpsk = Constellation::psk(4).unwrap().normalized();
        let h = channel(3, 1, &mut rng);
        let y: Vec<C64> = (0..3).map(|_| rng.complex_gaussian()).collect();
        for d in [Linear::zf(qpsk.clone()), Linear::mmse(qpsk.clone())] {
            let f = filter(&h.view(), d.criterion, qpsk.energy(), 0.3).unwrap();
            let (z, var) = d.estimate(&h.view(), &y, 0.3).unwrap();
            assert!((z[0] - matvec(&f.w.view(), &y)[0] / f.bias[0]).norm() < 1e-14 && var == f.var);
        }
        let zero = Array2::from_elem((2, 1), C64::new(0.0, 0.0));
        assert!(filter(&zero.view(), Criterion::Zf, es, 0.3).is_err());
    }

    #[test]
    fn every_detector_is_exact_without_noise() {
        let c = Constellation::qam(16).unwrap().normalized();
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let h = channel(4, 3, &mut rng);
            let idx: Vec<usize> = (0..3).map(|_| rng.below(16)).collect();
            let x: Vec<C64> = idx.iter().map(|&i| c.points()[i]).collect();
            let y = matvec(&h.view(), &x);
            let dets: Vec<Box<dyn MimoDetector>> = vec![
                Box::new(Linear::zf(c.clone())),
                Box::new(Linear::mmse(c.clone())),
                Box::new(Sic::zf(c.clone())),
                Box::new(Sic::mmse(c.clone())),
                Box::new(Ml::new(c.clone())),
                Box::new(Sphere::new(c.clone())),
                Box::new(KBest::new(c.clone(), 4)),
            ];
            for d in &dets {
                assert_eq!(d.detect(&h.view(), &y, 1e-6).unwrap(), idx);
            }
        }
    }

    #[test]
    fn the_sphere_decoder_is_the_exhaustive_ml_detector() {
        let c = Constellation::qam(4).unwrap().normalized();
        let mut rng = Rng::new(2);
        for _ in 0..300 {
            let h = channel(3, 3, &mut rng);
            let x: Vec<C64> = (0..3).map(|_| c.points()[rng.below(4)]).collect();
            let y = noisy(&h, &x, 0.5, &mut rng);
            let ml = Ml::new(c.clone()).detect(&h.view(), &y, 0.5).unwrap();
            assert_eq!(Sphere::new(c.clone()).detect(&h.view(), &y, 0.5).unwrap(), ml);
            // K-best with every path kept is ML too
            assert_eq!(KBest::new(c.clone(), 64).detect(&h.view(), &y, 0.5).unwrap(), ml);
        }
    }

    #[test]
    fn soft_outputs_agree_in_sign_with_the_decisions() {
        let c = Constellation::qam(16).unwrap().normalized();
        let mut rng = Rng::new(3);
        let h = channel(4, 2, &mut rng);
        let x = [c.points()[5], c.points()[10]];
        let y = noisy(&h, &x, 1e-3, &mut rng);
        for d in [&Linear::mmse(c.clone()) as &dyn MimoDetector, &Ml::new(c.clone()), &KBest::new(c.clone(), 8)] {
            let l = d.llr(&h.view(), &y, 1e-3).unwrap();
            assert_eq!(l.len(), 8);
            let bits: Vec<usize> = [5usize, 10].iter().flat_map(|i| (0..4).map(move |p| (i >> (3 - p)) & 1)).collect();
            for (v, b) in l.iter().zip(bits) {
                assert!(if b == 0 { *v > 0.0 } else { *v < 0.0 }, "{l:?}");
            }
        }
        assert!(Sphere::new(c).llr(&h.view(), &y, 1e-3).is_err());
    }

    #[test]
    fn linear_block_detection_matches_vector_detection() {
        let c = Constellation::qam(4).unwrap().normalized();
        let mut rng = Rng::new(4);
        let h = channel(2, 2, &mut rng);
        let y = Array2::from_shape_fn((2, 20), |_| rng.complex_gaussian());
        for d in [Linear::zf(c.clone()), Linear::mmse(c.clone())] {
            let block = d.detect_block(&h.view(), &y.view(), 0.1).unwrap();
            for t in 0..20 {
                assert_eq!(block.column(t).to_vec(), d.detect(&h.view(), &y.column(t).to_vec(), 0.1).unwrap());
            }
        }
    }

    // the channel factored once for a block, and the decisions and LLRs of one search,
    // give the bits of the detection use after use
    #[test]
    fn block_and_joint_outputs_match_the_detection_of_every_use() {
        let c = Constellation::qam(16).unwrap().normalized();
        let mut rng = Rng::new(8);
        let dets: Vec<Box<dyn MimoDetector>> = vec![
            Box::new(Sic::zf(c.clone())),
            Box::new(Sic::mmse(c.clone())),
            Box::new(Sic { constellation: c.clone(), criterion: Criterion::Mmse, ordered: false }),
            Box::new(Ml::new(c.clone())),
            Box::new(Sphere::new(c.clone())),
            Box::new(KBest::new(c.clone(), 1)),
            Box::new(KBest::new(c.clone(), 12)),
        ];
        for _ in 0..10 {
            let h = channel(4, 3, &mut rng);
            let y = Array2::from_shape_fn((4, 8), |_| rng.complex_gaussian());
            for d in &dets {
                let block = d.detect_block(&h.view(), &y.view(), 0.2).unwrap();
                for t in 0..8 {
                    let col = y.column(t).to_vec();
                    let labels = d.detect(&h.view(), &col, 0.2).unwrap();
                    assert_eq!(block.column(t).to_vec(), labels);
                    if let Ok(llr) = d.llr(&h.view(), &col, 0.2) {
                        let (l, v) = d.detect_llr(&h.view(), &col, 0.2).unwrap();
                        assert_eq!(
                            (l, v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
                            (labels, llr.iter().map(|x| x.to_bits()).collect())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_inputs_are_errors() {
        let c = Constellation::qam(4).unwrap().normalized();
        let h = channel(2, 3, &mut Rng::new(5));
        let y = vec![C64::new(0.0, 0.0); 2];
        assert!(Linear::zf(c.clone()).detect(&h.view(), &y, 0.1).is_err());
        assert!(Sphere::new(c.clone()).detect(&h.view(), &y, 0.1).is_err());
        assert!(Linear::mmse(c.clone()).detect(&h.view(), &y, 0.1).is_ok()); // MMSE handles n_r < n_t
        assert!(Ml::new(c.clone()).detect(&h.view(), &y[..1], 0.1).is_err());
        assert!(Ml::new(c).detect(&h.view(), &y, 0.0).is_err());
    }

    #[test]
    fn the_coherent_block_implements_detect() {
        let c = Constellation::qam(4).unwrap().normalized();
        let mut rng = Rng::new(6);
        let h = channel(3, 2, &mut rng);
        let x = Array2::from_shape_fn((2, 8), |_| c.points()[rng.below(4)]);
        let y = Symbols::new(h.dot(&x));
        let mut d = Coherent { detector: Sphere::new(c.clone()), h, constellation: c };
        assert_eq!(d.detect_into_owned(&y).unwrap().as_array(), &x);
    }

    trait Owned {
        fn detect_into_owned(&mut self, y: &Symbols) -> Result<Symbols>;
    }

    impl<D: MimoDetector> Owned for Coherent<D> {
        fn detect_into_owned(&mut self, y: &Symbols) -> Result<Symbols> {
            let mut out = Symbols::default();
            self.detect_into(y, 1e-3, &mut out)?;
            Ok(out)
        }
    }
}
