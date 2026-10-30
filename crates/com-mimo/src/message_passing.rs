//! Message-passing detection of large MIMO systems: LAMA (complex Bayesian AMP) and
//! orthogonal AMP (OAMP), with their state evolution.
//!
//! Both algorithms solve the linear model $y = H x + w$ ($H$ of $M$ rows and $N$
//! columns, $w$ of variance $\sigma^{2}$ per entry, i.i.d. entries $x_{j}$ of a known
//! prior) by alternating a linear step on the whole vector and a scalar MMSE denoiser
//! on every entry. In the large-system limit the input of the denoiser behaves as the
//! transmitted symbol in Gaussian noise of a variance that a scalar recursion, the
//! state evolution, predicts: the MIMO system decouples into parallel AWGN channels.
//!
//! # Denoiser
//!
//! With $r = x + n$, $n$ of variance $\tau$ (circularly-symmetric complex, or real),
//! the posterior mean and variance of $x$,
//!
//! $$
//! F(r, \tau) = \mathbb{E}\[X \mid r\], \qquad G(r, \tau) = \mathrm{Var}\[X \mid r\],
//! $$
//!
//! for a discrete alphabet $F = \sum_{a} w_{a} a$ with
//! $w_{a} \propto p_{a} e^{-|r - a|^{2}/\tau}$ (Jeon et al., eqs. 20 and 21), or for the
//! Bernoulli-Gaussian prior $(1 - \rho)\\, \delta(x) + \rho\\, \mathcal{N}(0, 1/\rho)$ of sparse
//! recovery (Ma & Ping, eq. 43). The MSE and variance functions of the state evolution
//! are
//!
//! $$
//! \Psi(\sigma^{2}, \gamma^{2}) = \mathbb{E}\left|F(S + \sigma Z, \gamma^{2}) - S\right|^{2}, \qquad
//! \Phi(\sigma^{2}, \gamma^{2}) = \mathbb{E}\\, G(S + \sigma Z, \gamma^{2}),
//! $$
//!
//! computed by Gauss–Hermite quadrature ([`Prior::mse`]); a separable square QAM reduces
//! to its real PAM, $\Psi(\sigma^{2}, \gamma^{2}) = 2 \Psi_{R}(\sigma^{2}/2, \gamma^{2}/2)$
//! (Jeon et al., Lemma 8).
//!
//! # LAMA
//!
//! With the columns of $H$ normalised to a unit norm and $\beta = N/M$, from
//! $\hat{s} = \mathbb{E}\[S\]$, $\gamma = \sigma^{2} + \beta \mathrm{Var}\[S\]$ and $r = y - H \hat{s}$
//! (Jeon et al., Algorithm 2),
//!
//! $$
//! z = \hat{s} + H^{H} r, \quad \hat{s}' = F(z, \gamma), \quad
//! \gamma' = \sigma^{2} + \beta \langle G(z, \gamma) \rangle, \quad
//! r' = y - H \hat{s}' + \frac{\beta \langle G(z, \gamma) \rangle}{\gamma}\\, r ,
//! $$
//!
//! the last term being the Onsager correction. Its state evolution (Theorem 1,
//! Corollary 2) is $\sigma^{2}_{t+1} = \sigma^{2} + \beta \Psi(\sigma^{2}_{t}, \sigma^{2}_{t})$
//! ([`amp_state_evolution`]), whose fixed points give the minimum and exact recovery
//! thresholds of Table II ([`amp_thresholds`]). [`Amp`] normalises the columns of any
//! channel itself.
//!
//! # OAMP
//!
//! With the SVD $H = U \Lambda V^{H}$, a de-correlated linear estimator and a
//! divergence-free denoiser (Ma & Ping, eqs. 19, 14 and 38):
//!
//! $$
//! r = s + W (y - H s), \quad W = \frac{N}{\mathrm{tr}(\hat{W} H)} \hat{W}, \quad
//! s' = C \left(F(r, \tau^{2}) - \frac{\overline{G}}{\tau^{2}} r\right), \quad
//! C = \frac{\tau^{2}}{\tau^{2} - \overline{G}},
//! $$
//!
//! with $\hat{W}$ the matched filter $H^{H}$, the pseudo-inverse or the LMMSE filter
//! $v^{2} H^{H} (v^{2} H H^{H} + \sigma^{2} I)^{-1}$, $v^{2}$ estimated from the residual
//! (eq. 30) and $\tau^{2}$ from the traces (eq. 31). The last iteration outputs
//! $F(r, \tau^{2})$. [`oamp_state_evolution`] is the empirical state evolution of eq. 32
//! with the optimal denoiser (eq. 39b).
//!
//! # Examples
//! ```
//! use com_mimo::message_passing::{amp_thresholds, Domain, Prior};
//! use com_mapping::constellation::Constellation;
//! // QPSK: LAMA is optimal at any SNR below β = 1.475 and recovers the noiseless
//! // signal below β = 2.086 (Jeon et al., Table II)
//! let qpsk = Prior::uniform(&Constellation::qam(4)?.normalized());
//! let t = amp_thresholds(&qpsk, Domain::Complex);
//! assert!((t.beta_min - 1.475).abs() < 2e-3 && (t.beta_max - 2.086).abs() < 2e-3);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. C. Jeon, R. Ghods, A. Maleki and C. Studer, "Optimal data detection in large
//!    MIMO," arXiv:1811.01917, 2018.
//! 2. J. Ma and L. Ping, "Orthogonal AMP," *IEEE Access*, 5, 2020–2033, 2017
//!    (arXiv:1602.06509).

use crate::detection::{Estimates, FLOOR, MimoDetector, check_shape};
use com_core::{Error, Result, ensure, ensure_shape};
use com_mapping::constellation::Constellation;
use com_math::Jacobian;
use com_math::linalg::svd;
use com_math::{C64, golden_max};
use ndarray::ArrayView2;

/// Nodes of the Gauss–Hermite rule per dimension of the expectations over the noise.
const NODES: usize = 96;

/// The number domain of the model: complex (circularly-symmetric noise) or real.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// Complex symbols, channel and circularly-symmetric noise: MIMO.
    Complex,
    /// Real symbols, channel and noise (the real parts of the complex values are used):
    /// the model of Ma & Ping and of real-valued compressed sensing.
    Real,
}

/// The prior of every entry of $x$.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Prior {
    /// Points of an alphabet and their probabilities.
    Discrete {
        /// The points.
        points: Vec<C64>,
        /// Their probabilities (summing to 1).
        probs: Vec<f64>,
    },
    /// $(1 - \rho)\\, \delta(x) + \rho\\, \mathcal{N}(0, 1/\rho)$: sparse, of unit variance
    /// (Ma & Ping, eq. 43).
    BernoulliGaussian {
        /// The probability $\rho$ of a non-zero entry.
        rho: f64,
    },
}

impl Prior {
    /// Equally likely points of a constellation.
    pub fn uniform(c: &Constellation) -> Prior {
        let m = c.order();
        Prior::Discrete { points: c.points().to_vec(), probs: vec![1.0 / m as f64; m] }
    }

    /// The points of a constellation with the probabilities `px` (normalised;
    /// uniform if `None`).
    ///
    /// # Errors
    /// Those of [`Constellation::priors`].
    pub fn with_probs(c: &Constellation, px: Option<&[f64]>) -> Result<Prior> {
        Ok(Prior::Discrete { points: c.points().to_vec(), probs: c.priors(px)? })
    }

    /// The Bernoulli-Gaussian prior of sparsity `rho`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless $0 < \rho \le 1$.
    pub fn bernoulli_gaussian(rho: f64) -> Result<Prior> {
        ensure(rho > 0.0 && rho <= 1.0, || format!("the sparsity must be in (0, 1], got {rho}"))?;
        Ok(Prior::BernoulliGaussian { rho })
    }

    /// $\mathbb{E}\[X\]$.
    pub fn mean(&self) -> C64 {
        match self {
            Prior::Discrete { points, probs } => points.iter().zip(probs).map(|(a, p)| a * *p).sum(),
            Prior::BernoulliGaussian { .. } => C64::new(0.0, 0.0),
        }
    }

    /// $\mathrm{Var}\[X\] = \mathbb{E}|X - \mathbb{E}X|^{2}$.
    pub fn variance(&self) -> f64 {
        match self {
            Prior::Discrete { points, probs } => {
                let m = self.mean();
                points.iter().zip(probs).map(|(a, p)| p * (a - m).norm_sqr()).sum()
            }
            Prior::BernoulliGaussian { .. } => 1.0,
        }
    }

    /// The posterior mean $F(r, \tau)$ and variance $G(r, \tau)$ of $x$ from
    /// $r = x + n$, $n$ of variance $\tau$ in `domain` (see the module).
    pub fn denoise(&self, r: C64, tau: f64, domain: Domain) -> (C64, f64) {
        let r = match domain {
            Domain::Complex => r,
            Domain::Real => C64::new(r.re, 0.0),
        };
        // the exponent of the Gaussian likelihood: |r - a|² / τ (complex), (r - a)² / (2τ) (real)
        let scale = match domain {
            Domain::Complex => 1.0 / tau,
            Domain::Real => 0.5 / tau,
        };
        match self {
            Prior::Discrete { points, probs } => {
                if tau <= 0.0 || !tau.is_finite() {
                    let best = points
                        .iter()
                        .zip(probs)
                        .filter(|(_, p)| **p > 0.0)
                        .min_by(|a, b| (r - a.0).norm_sqr().total_cmp(&(r - b.0).norm_sqr()))
                        .map_or(C64::new(0.0, 0.0), |(a, _)| *a);
                    return (best, 0.0);
                }
                let mut best = f64::NEG_INFINITY;
                for (a, p) in points.iter().zip(probs) {
                    if *p > 0.0 {
                        best = best.max(p.ln() - (r - a).norm_sqr() * scale);
                    }
                }
                let (mut sum, mut mean, mut second) = (0.0, C64::new(0.0, 0.0), 0.0);
                for (a, p) in points.iter().zip(probs) {
                    if *p > 0.0 {
                        let w = (p.ln() - (r - a).norm_sqr() * scale - best).exp();
                        sum += w;
                        mean += a * w;
                        second += w * a.norm_sqr();
                    }
                }
                let mean = mean / sum;
                (mean, (second / sum - mean.norm_sqr()).max(0.0))
            }
            Prior::BernoulliGaussian { rho } => {
                let rho = *rho;
                let s = 1.0 / rho;
                // log of the likelihood of r under N(0, v), up to the common constant
                let log_lik = |v: f64| match domain {
                    Domain::Complex => -r.norm_sqr() / v - v.ln(),
                    Domain::Real => -0.5 * r.norm_sqr() / v - 0.5 * v.ln(),
                };
                let tau = tau.max(f64::MIN_POSITIVE);
                let l1 = rho.ln() + log_lik(s + tau);
                let l0 = (1.0 - rho).ln() + log_lik(tau);
                // P(x ≠ 0 | r)
                let pi = if rho >= 1.0 { 1.0 } else { 1.0 / (1.0 + (l0 - l1).exp()) };
                let m = r * (s / (s + tau));
                let v = s * tau / (s + tau);
                let mean = m * pi;
                (mean, (pi * (v + m.norm_sqr()) - mean.norm_sqr()).max(0.0))
            }
        }
    }

    /// The real PAM of a separable square QAM (Jeon et al., Definition 10): the real
    /// parts with their marginal probabilities, if every point is the product of a real
    /// and an imaginary level of the same set with independent probabilities.
    fn separable(&self) -> Option<Prior> {
        let Prior::Discrete { points, probs } = self else { return None };
        let mut levels: Vec<f64> = Vec::new();
        for a in points {
            if !levels.iter().any(|l| (l - a.re).abs() <= 1e-9 * (1.0 + a.re.abs())) {
                levels.push(a.re);
            }
        }
        levels.sort_by(f64::total_cmp);
        let k = levels.len();
        if k < 2 || k * k != points.len() {
            return None;
        }
        let index = |x: f64| levels.iter().position(|l| (l - x).abs() <= 1e-9 * (1.0 + x.abs()));
        let mut joint = vec![f64::NAN; k * k];
        for (a, p) in points.iter().zip(probs) {
            let (i, j) = (index(a.re)?, index(a.im)?);
            joint[i * k + j] = *p;
        }
        if joint.iter().any(|p| p.is_nan()) {
            return None;
        }
        let marginal: Vec<f64> = (0..k).map(|i| (0..k).map(|j| joint[i * k + j]).sum()).collect();
        let independent = (0..k).all(|i| (0..k).all(|j| (joint[i * k + j] - marginal[i] * marginal[j]).abs() <= 1e-12));
        let symmetric = (0..k).all(|j| ((0..k).map(|i| joint[i * k + j]).sum::<f64>() - marginal[j]).abs() <= 1e-12);
        (independent && symmetric)
            .then(|| Prior::Discrete { points: levels.iter().map(|l| C64::new(*l, 0.0)).collect(), probs: marginal })
    }

    /// The MSE function $\Psi(\sigma^{2}, \gamma^{2})$ and the variance function
    /// $\Phi(\sigma^{2}, \gamma^{2})$ of the denoiser applied with the variance
    /// $\gamma^{2}$ to $S + \sigma Z$ (see the module), by Gauss–Hermite quadrature.
    pub fn mse(&self, noise_var: f64, gamma2: f64, domain: Domain) -> (f64, f64) {
        if domain == Domain::Complex {
            if let Some(pam) = self.separable() {
                let (psi, phi) = pam.mse(0.5 * noise_var, 0.5 * gamma2, Domain::Real);
                return (2.0 * psi, 2.0 * phi);
            }
        }
        let sigma = noise_var.max(0.0).sqrt();
        let nodes = Quadrature::new(domain);
        match self {
            Prior::Discrete { points, probs } => {
                let (mut psi, mut phi) = (0.0, 0.0);
                for (a, p) in points.iter().zip(probs).filter(|(_, p)| **p > 0.0) {
                    let a = match domain {
                        Domain::Complex => *a,
                        Domain::Real => C64::new(a.re, 0.0),
                    };
                    nodes.for_each(|z, w| {
                        let (f, g) = self.denoise(a + z * sigma, gamma2, domain);
                        psi += p * w * (f - a).norm_sqr();
                        phi += p * w * g;
                    });
                }
                (psi, phi)
            }
            Prior::BernoulliGaussian { rho } => {
                // R = S + σZ is a mixture of zero-mean Gaussians of variances σ² and
                // 1/ρ + σ²; given R, E|F(R, γ²) - S|² = |F(R, γ²) - F(R, σ²)|² + G(R, σ²)
                let (mut psi, mut phi) = (0.0, 0.0);
                for (weight, var) in [(1.0 - rho, noise_var), (*rho, 1.0 / rho + noise_var)] {
                    if weight <= 0.0 {
                        continue;
                    }
                    let sd = var.sqrt();
                    nodes.for_each(|z, w| {
                        let r = z * sd;
                        let (f, g) = self.denoise(r, gamma2, domain);
                        let (f0, g0) = self.denoise(r, noise_var, domain);
                        psi += weight * w * ((f - f0).norm_sqr() + g0);
                        phi += weight * w * g;
                    });
                }
                (psi, phi)
            }
        }
    }
}

/// The nodes $z$ and weights of $\mathbb{E} f(Z)$, $Z \sim \mathcal{CN}(0, 1)$ or
/// $\mathcal{N}(0, 1)$.
struct Quadrature {
    nodes: Vec<(C64, f64)>,
}

impl Quadrature {
    fn new(domain: Domain) -> Quadrature {
        Quadrature { nodes: com_math::gaussian_nodes(1.0, domain == Domain::Complex, NODES) }
    }

    fn for_each(&self, mut f: impl FnMut(C64, f64)) {
        for &(z, w) in &self.nodes {
            f(z, w);
        }
    }
}

/// One iteration of a state evolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateEvolutionStep {
    /// The variance of the noise of the decoupled channel at the input of the denoiser
    /// ($\sigma^{2}_{t}$ of LAMA, $\tau^{2}_{t}$ of OAMP).
    pub noise_var: f64,
    /// The predicted MSE of the MMSE estimate $F$ of this iteration.
    pub mse: f64,
}

/// The state evolution of LAMA without noise mismatch (Jeon et al., Corollary 2):
/// $\sigma^{2}_{1} = \sigma^{2} + \beta \mathrm{Var}\[S\]$,
/// $\sigma^{2}_{t+1} = \sigma^{2} + \beta \Psi(\sigma^{2}_{t}, \sigma^{2}_{t})$, for `iterations`
/// iterations, with $\beta = N/M$ and the columns of $H$ of unit norm.
pub fn amp_state_evolution(
    prior: &Prior,
    beta: f64,
    noise_var: f64,
    iterations: usize,
    domain: Domain,
) -> Vec<StateEvolutionStep> {
    let mut s2 = noise_var + beta * prior.variance();
    let mut out = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mse = prior.mse(s2, s2, domain).0;
        out.push(StateEvolutionStep { noise_var: s2, mse });
        s2 = noise_var + beta * mse;
    }
    out
}

/// The thresholds of LAMA for a prior (Jeon et al., Definitions 6 to 9, Table II).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmpThresholds {
    /// The minimum recovery threshold: below it the fixed point is unique (and LAMA
    /// optimal) at every noise variance.
    pub beta_min: f64,
    /// The exact recovery threshold: below it the noiseless signal is recovered.
    pub beta_max: f64,
    /// The minimum critical noise variance $N_{0}^{\min}(\beta_{\min})$.
    pub n0_min: f64,
    /// The maximum guaranteed noise variance $N_{0}^{\max}(\beta_{\max})$.
    pub n0_max: f64,
}

/// The minimum and exact recovery thresholds of LAMA and the critical noise variances
/// (Jeon et al., eqs. 33 and 34, Definitions 8 and 9):
/// $\beta_{\max} = \min_{\sigma^{2}} \sigma^{2} / \Psi(\sigma^{2}, \sigma^{2})$,
/// $\beta_{\min} = \min_{\sigma^{2}} 1/\Psi'(\sigma^{2})$; the noise variances are
/// $\sigma^{2} - \beta \Psi(\sigma^{2})$ where $\beta \Psi'(\sigma^{2}) = 1$.
pub fn amp_thresholds(prior: &Prior, domain: Domain) -> AmpThresholds {
    let var = prior.variance();
    let psi = |s2: f64| prior.mse(s2, s2, domain).0;
    // the derivative in σ², by central differences in log σ²
    let dpsi = |s2: f64| {
        let h = 1e-4;
        (psi(s2 * (1.0 + h)) - psi(s2 * (1.0 - h))) / (2.0 * h * s2)
    };
    // a coarse logarithmic scan, then a golden-section refinement in log σ²
    let grid: Vec<f64> = (0..=240).map(|i| var * 10f64.powf(-6.0 + 8.0 * f64::from(i) / 240.0)).collect();
    let refine = |f: &dyn Fn(f64) -> f64| -> f64 {
        let (i, _) = grid
            .iter()
            .enumerate()
            .map(|(i, s)| (i, f(*s)))
            .fold((0, f64::NEG_INFINITY), |b, (i, v)| if v > b.1 { (i, v) } else { b });
        let lo = grid[i.saturating_sub(1)].ln();
        let hi = grid[(i + 1).min(grid.len() - 1)].ln();
        golden_max(|x| f(x.exp()), lo, hi, 80).exp()
    };
    let s_max = refine(&|s2| psi(s2) / s2);
    let beta_max = s_max / psi(s_max);
    let s_min = refine(&dpsi);
    let beta_min = 1.0 / dpsi(s_min);
    let n0_min = s_min - beta_min * psi(s_min);
    // N0max(βmax): the largest σ² - βΨ(σ²) over the roots of βΨ'(σ²) = 1
    let h = |s2: f64| beta_max * dpsi(s2) - 1.0;
    let mut n0_max = 0.0f64;
    for w in grid.windows(2) {
        let (a, b) = (h(w[0]), h(w[1]));
        if a.signum() != b.signum() {
            let x = com_math::bisect(|x| h(x.exp()).signum() == a.signum(), w[0].ln(), w[1].ln(), 80).exp();
            n0_max = n0_max.max(x - beta_max * psi(x));
        }
    }
    AmpThresholds { beta_min, beta_max, n0_min, n0_max }
}

/// How LAMA estimates the variance of the decoupled channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmpVariance {
    /// The postulated variance $\sigma^{2} + \beta \langle G \rangle$ (Jeon et al., eq. 23).
    Postulated,
    /// The residual $\lVert r \rVert^{2} / M$ (Jeon et al., eq. 38), better in small
    /// systems.
    Residual,
}

impl std::str::FromStr for AmpVariance {
    type Err = Error;

    /// The variance of a name: `postulated` or `residual`, case-insensitive.
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    ///
    /// # Examples
    /// ```
    /// use com_mimo::message_passing::AmpVariance;
    /// assert_eq!("residual".parse::<AmpVariance>()?, AmpVariance::Residual);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    fn from_str(name: &str) -> Result<AmpVariance> {
        match name.to_ascii_lowercase().as_str() {
            "postulated" => Ok(AmpVariance::Postulated),
            "residual" => Ok(AmpVariance::Residual),
            other => Err(Error::Config(format!("unknown variance {other:?}: postulated or residual"))),
        }
    }
}

/// The iterations of a message-passing detector.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// The input of the denoiser at the last iteration: $x$ in Gaussian noise of
    /// variance [`Trace::noise_var`] in the large-system limit.
    pub z: Vec<C64>,
    /// The variance of that noise used by the denoiser.
    pub noise_var: f64,
    /// The MMSE estimate $F$ of every iteration.
    pub estimates: Vec<Vec<C64>>,
    /// The variance used by the denoiser at every iteration.
    pub noise_vars: Vec<f64>,
}

impl Trace {
    /// The final MMSE estimate.
    pub fn estimate(&self) -> &[C64] {
        self.estimates.last().map_or(&[], Vec::as_slice)
    }
}

fn check(h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64, iterations: usize) -> Result<()> {
    ensure_shape(h.nrows() == y.len(), || {
        format!("the channel has {} rows for {} received samples", h.nrows(), y.len())
    })?;
    ensure_shape(h.ncols() >= 1 && h.nrows() >= 1, || "the channel must not be empty".into())?;
    ensure(noise_var >= 0.0 && noise_var.is_finite(), || {
        format!("the noise variance must be finite and non-negative, got {noise_var}")
    })?;
    ensure(iterations >= 1, || "at least one iteration is needed".into())
}

fn residual(h: &ArrayView2<'_, C64>, y: &[C64], s: &[C64]) -> Vec<C64> {
    y.iter().enumerate().map(|(i, yi)| yi - h.row(i).iter().zip(s).map(|(a, b)| a * b).sum::<C64>()).collect()
}

fn adjoint(h: &ArrayView2<'_, C64>, r: &[C64]) -> Vec<C64> {
    (0..h.ncols()).map(|j| h.column(j).iter().zip(r).map(|(a, b)| a.conj() * b).sum()).collect()
}

/// LAMA (Jeon et al., Algorithm 2) on $y = H x + w$, $w$ of variance `noise_var` per
/// entry: the columns of $H$ are first scaled to a unit mean norm (with $y$ and the
/// noise variance), which leaves the problem unchanged.
///
/// # Errors
/// [`com_core::Error::Shape`] for an empty channel or one whose rows do not match $y$;
/// [`com_core::Error::Config`] for a negative noise variance, zero iterations or a zero
/// channel.
pub fn lama(
    h: &ArrayView2<'_, C64>,
    y: &[C64],
    noise_var: f64,
    prior: &Prior,
    domain: Domain,
    iterations: usize,
    variance: AmpVariance,
) -> Result<Trace> {
    check(h, y, noise_var, iterations)?;
    let (m, n) = h.dim();
    let c = h.iter().map(C64::norm_sqr).sum::<f64>() / n as f64;
    ensure(c > 0.0, || "the channel is zero".into())?;
    let k = 1.0 / c.sqrt();
    let h = h.mapv(|v| v * k);
    let y: Vec<C64> = y.iter().map(|v| v * k).collect();
    let n0 = noise_var / c;
    let beta = n as f64 / m as f64;
    let mut s = vec![prior.mean(); n];
    let mut gamma = n0 + beta * prior.variance();
    let mut r = residual(&h.view(), &y, &s);
    let mut trace =
        Trace { z: Vec::new(), noise_var: gamma, estimates: Vec::with_capacity(iterations), noise_vars: Vec::new() };
    for _ in 0..iterations {
        if variance == AmpVariance::Residual {
            gamma = (r.iter().map(C64::norm_sqr).sum::<f64>() / m as f64).max(FLOOR);
        }
        let back = adjoint(&h.view(), &r);
        let z: Vec<C64> = s.iter().zip(&back).map(|(a, b)| a + b).collect();
        let (next, g): (Vec<C64>, Vec<f64>) = z.iter().map(|zi| prior.denoise(*zi, gamma, domain)).unzip();
        let gbar = g.iter().sum::<f64>() / n as f64;
        let onsager = beta * gbar / gamma;
        let fit = residual(&h.view(), &y, &next);
        r = fit.iter().zip(&r).map(|(f, old)| f + old * onsager).collect();
        trace.z = z;
        trace.noise_var = gamma;
        trace.noise_vars.push(gamma);
        trace.estimates.push(next.clone());
        s = next;
        gamma = n0 + beta * gbar;
    }
    // the noise variance of the decoupled channel, back on the scale of the caller
    trace.noise_var *= c;
    for v in &mut trace.noise_vars {
        *v *= c;
    }
    Ok(trace)
}

/// The linear estimator $\hat{W}$ of OAMP (Ma & Ping, eq. 15).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OampLinear {
    /// The matched filter $H^{H}$.
    MatchedFilter,
    /// The pseudo-inverse.
    PseudoInverse,
    /// The LMMSE filter $v^{2} H^{H} (v^{2} H H^{H} + \sigma^{2} I)^{-1}$ (optimal, Lemma 1).
    Lmmse,
}

impl std::str::FromStr for OampLinear {
    type Err = Error;

    /// The estimator of a name: `lmmse`, `pinv` (the pseudo-inverse) or `mf` (the
    /// matched filter), case-insensitive.
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    ///
    /// # Examples
    /// ```
    /// use com_mimo::message_passing::OampLinear;
    /// assert_eq!("pinv".parse::<OampLinear>()?, OampLinear::PseudoInverse);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    fn from_str(name: &str) -> Result<OampLinear> {
        match name.to_ascii_lowercase().as_str() {
            "lmmse" => Ok(OampLinear::Lmmse),
            "pinv" => Ok(OampLinear::PseudoInverse),
            "mf" => Ok(OampLinear::MatchedFilter),
            other => Err(Error::Config(format!("unknown linear estimator {other:?}: lmmse, pinv or mf"))),
        }
    }
}

impl OampLinear {
    /// The singular values $\hat{g}_{i}$ of $\hat{W}$ for those $\lambda_{i}$ of $H$.
    fn gains(self, lambda: &[f64], v2: f64, noise_var: f64) -> Vec<f64> {
        let top = lambda.iter().copied().fold(0.0, f64::max);
        lambda
            .iter()
            .map(|&l| match self {
                OampLinear::MatchedFilter => l,
                OampLinear::PseudoInverse => {
                    if l > 1e-12 * top {
                        1.0 / l
                    } else {
                        0.0
                    }
                }
                OampLinear::Lmmse => v2 * l / (v2 * l * l + noise_var),
            })
            .collect()
    }
}

/// The de-correlated estimator $W = N \hat{W} / \mathrm{tr}(\hat{W} H)$ by its singular
/// values $g_{i}$, and the variance $\tau^{2}$ of eq. 31 for an input error $v^{2}$.
fn decorrelated(lambda: &[f64], ghat: &[f64], n: usize, v2: f64, noise_var: f64) -> (Vec<f64>, f64) {
    let trace: f64 = ghat.iter().zip(lambda).map(|(g, l)| g * l).sum();
    let c = if trace > 0.0 { n as f64 / trace } else { 0.0 };
    let g: Vec<f64> = ghat.iter().map(|x| x * c).collect();
    let bb = g.iter().zip(lambda).map(|(g, l)| (1.0 - g * l).powi(2)).sum::<f64>() + (n - lambda.len()) as f64;
    let ww = g.iter().map(|g| g * g).sum::<f64>();
    (g, (bb * v2 + ww * noise_var) / n as f64)
}

/// OAMP (Ma & Ping, eqs. 19, 25 and 38, Lemma 1) on $y = H x + w$, $w$ of variance
/// `noise_var` per entry, with the linear estimator `linear` and the optimal
/// divergence-free denoiser.
///
/// # Errors
/// As [`lama`].
pub fn oamp(
    h: &ArrayView2<'_, C64>,
    y: &[C64],
    noise_var: f64,
    prior: &Prior,
    domain: Domain,
    iterations: usize,
    linear: OampLinear,
) -> Result<Trace> {
    check(h, y, noise_var, iterations)?;
    let (m, n) = h.dim();
    // H = U Λ V^H (thin: k = min(m, n) singular values)
    let (u, lambda, v) = svd(h);
    let frob: f64 = lambda.iter().map(|l| l * l).sum();
    ensure(frob > 0.0, || "the channel is zero".into())?;
    let floor = 1e-12 * prior.variance().max(FLOOR);
    let mut s = vec![prior.mean(); n];
    let mut trace =
        Trace { z: Vec::new(), noise_var: 0.0, estimates: Vec::with_capacity(iterations), noise_vars: Vec::new() };
    for t in 0..iterations {
        let res = residual(h, y, &s);
        let energy: f64 = res.iter().map(C64::norm_sqr).sum();
        let v2 = ((energy - m as f64 * noise_var) / frob).max(floor);
        let (g, tau2) = decorrelated(&lambda, &linear.gains(&lambda, v2, noise_var), n, v2, noise_var);
        let tau2 = tau2.max(FLOOR);
        // r = s + V diag(g) U^H res
        let proj: Vec<C64> = (0..lambda.len())
            .map(|i| u.column(i).iter().zip(&res).map(|(a, b)| a.conj() * b).sum::<C64>() * g[i])
            .collect();
        let r: Vec<C64> = (0..n).map(|j| s[j] + (0..proj.len()).map(|i| v[[j, i]] * proj[i]).sum::<C64>()).collect();
        let (f, gv): (Vec<C64>, Vec<f64>) = r.iter().map(|ri| prior.denoise(*ri, tau2, domain)).unzip();
        let mmse = gv.iter().sum::<f64>() / n as f64;
        trace.noise_vars.push(tau2);
        trace.estimates.push(f.clone());
        trace.z.clone_from(&r);
        trace.noise_var = tau2;
        if t + 1 < iterations {
            let c = tau2 / (tau2 - mmse).max(1e-12 * tau2);
            s = f.iter().zip(&r).map(|(fi, ri)| (fi - ri * (mmse / tau2)) * c).collect();
        }
    }
    Ok(trace)
}

/// The empirical state evolution of OAMP with the optimal denoiser (Ma & Ping, eqs. 32
/// and 39): for the `singular_values` of $H$ ($N$ columns), from $v^{2}_{0} = \mathrm{Var}\[S\]$,
/// $\tau^{2}_{t} = \Phi_{t}(v^{2}_{t})$ (eq. 32), the MSE of $F$ is
/// $\mathrm{mmse}_{B}(\tau^{2}_{t})$ and $v^{2}_{t+1} = (1/\mathrm{mmse}_{B} - 1/\tau^{2}_{t})^{-1}$.
pub fn oamp_state_evolution(
    prior: &Prior,
    singular_values: &[f64],
    n: usize,
    noise_var: f64,
    iterations: usize,
    linear: OampLinear,
    domain: Domain,
) -> Vec<StateEvolutionStep> {
    let lambda: Vec<f64> = singular_values.iter().copied().take(n).collect();
    let mut v2 = prior.variance();
    let mut out = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (_, tau2) = decorrelated(&lambda, &linear.gains(&lambda, v2, noise_var), n, v2, noise_var);
        let mmse = prior.mse(tau2, tau2, domain).0;
        out.push(StateEvolutionStep { noise_var: tau2, mse: mmse });
        v2 = if mmse < tau2 { 1.0 / (1.0 / mmse.max(FLOOR) - 1.0 / tau2) } else { tau2 };
    }
    out
}

/// The MIMO detector of LAMA: the hard decisions of the decoupled channel, its bit
/// LLRs, and its estimates $z$ with the variance of the decoupled noise.
#[derive(Clone, Debug, PartialEq)]
pub struct Amp {
    /// The alphabet of every stream.
    constellation: Constellation,
    /// Iterations (at least 1).
    iterations: usize,
    /// The variance estimate of the decoupled channel.
    pub variance: AmpVariance,
}

impl Amp {
    /// LAMA with `iterations` iterations and the postulated variance.
    ///
    /// # Errors
    /// [`Error::Config`] for zero iterations.
    pub fn new(constellation: Constellation, iterations: usize) -> Result<Amp> {
        ensure(iterations >= 1, || "at least one iteration is needed".into())?;
        Ok(Amp { constellation, iterations, variance: AmpVariance::Postulated })
    }

    /// Iterations.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    fn trace(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Trace> {
        check_shape(h, y)?;
        let prior = Prior::uniform(&self.constellation);
        lama(h, y, noise_var, &prior, Domain::Complex, self.iterations, self.variance)
    }
}

/// The MIMO detector of OAMP with the LMMSE estimator (the outputs of [`Amp`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Oamp {
    /// The alphabet of every stream.
    constellation: Constellation,
    /// Iterations (at least 1).
    iterations: usize,
    /// The linear estimator.
    pub linear: OampLinear,
}

impl Oamp {
    /// OAMP-LMMSE with `iterations` iterations.
    ///
    /// # Errors
    /// [`Error::Config`] for zero iterations.
    pub fn new(constellation: Constellation, iterations: usize) -> Result<Oamp> {
        ensure(iterations >= 1, || "at least one iteration is needed".into())?;
        Ok(Oamp { constellation, iterations, linear: OampLinear::Lmmse })
    }

    /// Iterations.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    fn trace(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Trace> {
        check_shape(h, y)?;
        let prior = Prior::uniform(&self.constellation);
        oamp(h, y, noise_var, &prior, Domain::Complex, self.iterations, self.linear)
    }
}

/// The noise variance of the decoupled channel $z = x + n$ of a trace, floored: the state
/// evolution reaches 0 without noise.
fn decoupled_var(trace: &Trace) -> f64 {
    trace.noise_var.max(FLOOR)
}

/// The [`MimoDetector`] of a message-passing detector with a `trace` method: the
/// decisions, the LLRs and the estimates of its decoupled channel.
macro_rules! decoupled_detector {
    ($t:ty) => {
        impl MimoDetector for $t {
            fn constellation(&self) -> &Constellation {
                &self.constellation
            }

            fn has_soft_output(&self) -> bool {
                true
            }

            fn soft_estimate(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Option<Estimates>> {
                let t = self.trace(h, y, noise_var)?;
                let v = decoupled_var(&t);
                Ok(Some(Estimates { noise_var: vec![v; t.z.len()], symbols: t.z }))
            }

            fn detect(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<usize>> {
                Ok(self.constellation.decide(&self.trace(h, y, noise_var)?.z))
            }

            fn llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<Vec<f64>> {
                let t = self.trace(h, y, noise_var)?;
                self.constellation.llr(&t.z, decoupled_var(&t), None, Jacobian::Exact)
            }

            fn detect_llr(&self, h: &ArrayView2<'_, C64>, y: &[C64], noise_var: f64) -> Result<(Vec<usize>, Vec<f64>)> {
                let t = self.trace(h, y, noise_var)?;
                Ok((
                    self.constellation.decide(&t.z),
                    self.constellation.llr(&t.z, decoupled_var(&t), None, Jacobian::Exact)?,
                ))
            }
        }
    };
}

decoupled_detector!(Amp);
decoupled_detector!(Oamp);

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use ndarray::Array2;

    fn qpsk() -> Constellation {
        Constellation::qam(4).unwrap().normalized()
    }

    #[test]
    fn the_denoiser_is_the_posterior_of_the_alphabet() {
        let p = Prior::uniform(&qpsk());
        // far from the noise: the nearest point, no variance
        let a = qpsk().points()[2];
        let (f, g) = p.denoise(a * 1.01, 1e-4, Domain::Complex);
        assert!((f - a).norm() < 1e-9 && g < 1e-9);
        // no information: the mean and the energy of the alphabet
        let (f, g) = p.denoise(C64::new(0.0, 0.0), 1e9, Domain::Complex);
        assert!(f.norm() < 1e-9 && (g - 1.0).abs() < 1e-6);
        // BPSK in real noise: tanh(r / τ)
        let bpsk = Prior::Discrete { points: vec![C64::new(-1.0, 0.0), C64::new(1.0, 0.0)], probs: vec![0.5, 0.5] };
        let (f, g) = bpsk.denoise(C64::new(0.3, 7.0), 0.5, Domain::Real);
        assert!((f.re - (0.3f64 / 0.5).tanh()).abs() < 1e-12 && (g - (1.0 - f.re * f.re)).abs() < 1e-12);
    }

    #[test]
    fn the_mse_of_separable_qam_is_that_of_its_pam() {
        let p = Prior::uniform(&Constellation::qam(16).unwrap().normalized());
        let Prior::Discrete { points, probs } = &p else { unreachable!() };
        // the same prior written as a non-separable one: perturb nothing, compute in 2D
        let direct = |s2: f64| {
            let q = Quadrature::new(Domain::Complex);
            let mut psi = 0.0;
            for (a, pa) in points.iter().zip(probs) {
                q.for_each(|z, w| psi += pa * w * (p.denoise(a + z * s2.sqrt(), s2, Domain::Complex).0 - a).norm_sqr());
            }
            psi
        };
        for s2 in [0.01, 0.1, 1.0] {
            let fast = p.mse(s2, s2, Domain::Complex).0;
            assert!((fast / direct(s2) - 1.0).abs() < 1e-6, "{s2}: {fast} {}", direct(s2));
        }
    }

    #[test]
    fn lama_and_oamp_recover_a_noiseless_large_system() {
        let c = qpsk();
        let mut rng = Rng::new(4);
        let (m, n) = (96, 64);
        let h = Array2::from_shape_fn((m, n), |_| rng.complex_gaussian());
        let idx: Vec<usize> = (0..n).map(|_| rng.below(4)).collect();
        let x: Vec<C64> = idx.iter().map(|&i| c.points()[i]).collect();
        let y = com_math::linalg::matvec(&h.view(), &x);
        assert_eq!(Amp::new(c.clone(), 20).unwrap().detect(&h.view(), &y, 1e-6).unwrap(), idx);
        assert_eq!(Oamp::new(c.clone(), 10).unwrap().detect(&h.view(), &y, 1e-6).unwrap(), idx);
        assert!(Amp::new(c.clone(), 0).is_err() && Oamp::new(c.clone(), 0).is_err());
        // a length mismatch is a Shape error, as MimoDetector::detect documents
        for d in [&Amp::new(c.clone(), 5).unwrap() as &dyn MimoDetector, &Oamp::new(c.clone(), 5).unwrap()] {
            assert!(matches!(d.detect(&h.view(), &y[1..], 1e-6), Err(Error::Shape(_))));
            // the soft outputs are those of the decoupled channel
            let e = d.soft_estimate(&h.view(), &y, 1e-6).unwrap().unwrap();
            let (labels, llr) = d.detect_llr(&h.view(), &y, 1e-6).unwrap();
            assert_eq!(c.decide(&e.symbols), labels);
            let from = c.llr_with_variances(&e.symbols, &e.noise_var, None, Jacobian::Exact).unwrap();
            assert!(from.iter().zip(&llr).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn the_state_evolution_of_lama_decreases_to_its_fixed_point() {
        let p = Prior::uniform(&qpsk());
        let se = amp_state_evolution(&p, 0.5, 0.05, 30, Domain::Complex);
        assert!(se.windows(2).all(|w| w[1].noise_var <= w[0].noise_var + 1e-15));
        let last = se[se.len() - 1];
        assert!((last.noise_var - (0.05 + 0.5 * last.mse)).abs() < 1e-9);
    }
}
