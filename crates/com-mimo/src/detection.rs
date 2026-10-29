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

    /// Detect every column of `y` (`n_rx × T`) with the same channel: the labels,
    /// `n_tx × T`.
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

fn distance2(h: &ArrayView2<'_, C64>, y: &[C64], x: &[C64]) -> f64 {
    matvec(h, x).iter().zip(y).map(|(a, b)| (b - a).norm_sqr()).sum()
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

fn filter(h: &ArrayView2<'_, C64>, criterion: Criterion, es: f64, noise_var: f64) -> Result<Filter> {
    let n = h.ncols();
    let lambda = match criterion {
        Criterion::Zf => {
            tall(h, "zero forcing")?;
            0.0
        }
        Criterion::Mmse => noise_var / es,
    };
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
        let f = filter(h, self.criterion, self.constellation.energy(), noise_var)?;
        let z = matvec(&f.w.view(), y).iter().zip(&f.bias).map(|(z, b)| z / b).collect();
        Ok((z, f.var))
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

    fn detect_block(&self, h: &ArrayView2<'_, C64>, y: &ArrayView2<'_, C64>, noise_var: f64) -> Result<Array2<usize>> {
        if y.nrows() != h.nrows() {
            return Err(Error::Shape(format!("{} received streams for {} receive antennas", y.nrows(), h.nrows())));
        }
        check(h, &vec![C64::new(0.0, 0.0); h.nrows()], noise_var)?;
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

impl MimoDetector for Sic {
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
        check(h, y, noise_var)?;
        let es = self.constellation.energy();
        let mut remaining: Vec<usize> = (0..h.ncols()).collect();
        let mut residual = y.to_vec();
        let mut out = vec![0; h.ncols()];
        while !remaining.is_empty() {
            let hc = Array2::from_shape_fn((h.nrows(), remaining.len()), |(i, j)| h[[i, remaining[j]]]);
            let f = filter(&hc.view(), self.criterion, es, noise_var)?;
            let k = if self.ordered {
                (0..remaining.len()).min_by(|a, b| f.var[*a].total_cmp(&f.var[*b])).unwrap_or(0)
            } else {
                0
            };
            let z: C64 = f.w.row(k).iter().zip(&residual).map(|(w, v)| w * v).sum::<C64>() / f.bias[k];
            let idx = self.constellation.nearest(z);
            let s = self.constellation.points()[idx];
            for (r, hk) in residual.iter_mut().zip(hc.column(k)) {
                *r -= hk * s;
            }
            out[remaining[k]] = idx;
            remaining.remove(k);
        }
        Ok(out)
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

    /// Visit every hypothesis with its distance $\lVert y - H x \rVert^{2}$.
    fn search(&self, h: &ArrayView2<'_, C64>, y: &[C64], mut visit: impl FnMut(&[usize], f64)) -> Result<()> {
        let (m, nt) = (self.constellation.order(), h.ncols());
        let total = (0..nt).try_fold(1usize, |acc, _| acc.checked_mul(m)).filter(|t| *t <= ML_MAX_HYPOTHESES);
        ensure(total.is_some(), || format!("{m}^{nt} hypotheses exceed the limit of the exhaustive search"))?;
        let pts = self.constellation.points();
        let mut idx = vec![0usize; nt];
        let mut x = vec![pts[0]; nt];
        loop {
            visit(&idx, distance2(h, y, &x));
            // odometer over the hypotheses
            let mut k = 0;
            loop {
                if k == nt {
                    return Ok(());
                }
                idx[k] += 1;
                if idx[k] < m {
                    x[k] = pts[idx[k]];
                    break;
                }
                idx[k] = 0;
                x[k] = pts[0];
                k += 1;
            }
        }
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
        let mut best = (f64::INFINITY, vec![0; h.ncols()]);
        self.search(h, y, |idx, d| {
            if d < best.0 {
                best = (d, idx.to_vec());
            }
        })?;
        Ok(best.1)
    }

    fn llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<f64>> {
        check(h, y, noise_var)?;
        let bits = self.constellation.bits_per_symbol();
        let mut best = vec![[f64::INFINITY; 2]; h.ncols() * bits];
        self.search(h, y, |idx, d| update_bits(&mut best, idx, bits, d))?;
        Ok(max_log(&best, noise_var, f64::INFINITY))
    }
}

/// The QR form of the search: $\lVert y - H x\rVert^{2} = \lVert z - R x\rVert^{2} + c$.
fn triangular(h: &ArrayView2<'_, C64>, y: &[C64], what: &str) -> Result<(CMat, Vec<C64>)> {
    tall(h, what)?;
    let (q, r) = qr(h)?;
    if (0..r.nrows()).any(|i| r[[i, i]].re <= 0.0) {
        return Err(Error::Numerical("the channel is rank deficient".into()));
    }
    let z = matvec(&hermitian(&q.view()).view(), y);
    Ok((r, z))
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
}

impl MimoDetector for Sphere {
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
        check(h, y, noise_var)?;
        let (r, z) = triangular(h, y, "the sphere decoder")?;
        let n = r.ncols();
        let pts = self.constellation.points();
        let mut best = (f64::INFINITY, vec![0usize; n]);
        let mut x = vec![C64::new(0.0, 0.0); n];
        let mut idx = vec![0usize; n];
        // one frame per level: the candidates sorted by partial distance, the next to try
        let mut frames: Vec<Frame> = Vec::with_capacity(n);
        let candidates = |k: usize, x: &[C64]| -> Vec<(f64, usize)> {
            let mut b = z[k];
            for j in k + 1..n {
                b -= r[[k, j]] * x[j];
            }
            let mut c: Vec<(f64, usize)> =
                pts.iter().enumerate().map(|(i, p)| ((b - r[[k, k]] * p).norm_sqr(), i)).collect();
            c.sort_by(|a, b| a.0.total_cmp(&b.0));
            c
        };
        frames.push((candidates(n - 1, &x), 0, 0.0));
        while !frames.is_empty() {
            let k = n - frames.len();
            let Some(f) = frames.last_mut() else { break };
            let candidate = f.0.get(f.1).copied();
            f.1 += 1;
            let base = f.2;
            let Some((d, i)) = candidate else {
                frames.pop();
                continue;
            };
            let partial = base + d;
            if partial >= best.0 {
                // the candidates are sorted: no other point of this level can do better
                frames.pop();
                continue;
            }
            x[k] = pts[i];
            idx[k] = i;
            if k == 0 {
                best = (partial, idx.clone());
            } else {
                let c = candidates(k - 1, &x);
                frames.push((c, 0, partial));
            }
        }
        Ok(best.1)
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

impl KBest {
    /// The search keeping `k` paths.
    pub fn new(constellation: Constellation, k: usize) -> KBest {
        KBest { constellation, k, llr_clip: 30.0 }
    }

    /// The surviving paths (labels, distance), best first.
    fn list(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<(Vec<usize>, f64)>> {
        check(h, y, noise_var)?;
        ensure(self.k >= 1, || "K-best needs k ≥ 1".into())?;
        let (r, z) = triangular(h, y, "K-best")?;
        let n = r.ncols();
        let pts = self.constellation.points();
        // a path holds the labels of the streams k..n-1, stored from stream n-1 down
        let mut paths: Vec<(Vec<usize>, f64)> = vec![(Vec::new(), 0.0)];
        for k in (0..n).rev() {
            let mut next = Vec::with_capacity(paths.len() * pts.len());
            for (labels, d) in &paths {
                let mut b = z[k];
                for (depth, &l) in labels.iter().enumerate() {
                    let j = n - 1 - depth;
                    b -= r[[k, j]] * pts[l];
                }
                for (i, p) in pts.iter().enumerate() {
                    let mut l = labels.clone();
                    l.push(i);
                    next.push((l, d + (b - r[[k, k]] * p).norm_sqr()));
                }
            }
            next.sort_by(|a, b| a.1.total_cmp(&b.1));
            next.truncate(self.k);
            paths = next;
        }
        Ok(paths
            .into_iter()
            .map(|(mut l, d)| {
                l.reverse();
                (l, d)
            })
            .collect())
    }
}

impl MimoDetector for KBest {
    fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
        let list = self.list(h, y, noise_var)?;
        Ok(list.into_iter().next().map(|p| p.0).unwrap_or_default())
    }

    fn llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<f64>> {
        let bits = self.constellation.bits_per_symbol();
        let mut best = vec![[f64::INFINITY; 2]; h.ncols() * bits];
        for (labels, d) in self.list(h, y, noise_var)? {
            update_bits(&mut best, &labels, bits, d);
        }
        Ok(max_log(&best, noise_var, self.llr_clip))
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
        *out = Symbols::new(labels.mapv(|i| pts[i]));
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
