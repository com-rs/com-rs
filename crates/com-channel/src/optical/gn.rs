//! The Gaussian-noise (GN) model of nonlinear interference in uncompensated coherent
//! links: the reference of the launch-power campaigns.
//!
//! # Model
//!
//! With dispersion uncompensated in the optical domain, the nonlinear interference
//! (NLI) of a WDM signal behaves as additive Gaussian noise whose power spectral density
//! at the end of $N_{s}$ identical spans is (Poggiolini 2012, the GN reference formula)
//!
//! $$
//! G_{\mathrm{NLI}}(f) = \frac{16}{27} \gamma^{2} \iint G(f_{1}) G(f_{2}) G(f_{1} + f_{2} - f)\\,
//!   |\rho(f_{1}, f_{2}, f)|^{2}\\, \chi(f_{1}, f_{2}, f)\\, df_{1}\\, df_{2},
//! $$
//!
//! with $G$ the power spectral density of the transmitted signal (both polarisations),
//! the span efficiency
//!
//! $$
//! \rho = \frac{1 - e^{-\alpha L_{s}} e^{j 4 \pi^{2} \beta_{2} L_{s} \Delta}}{\alpha - j 4 \pi^{2} \beta_{2} \Delta},
//! \qquad \Delta = (f_{1} - f)(f_{2} - f),
//! $$
//!
//! ($\alpha$ the power attenuation) and the phased-array factor of the spans
//!
//! $$
//! \chi = \frac{\sin^{2}(2 N_{s} \pi^{2} \beta_{2} L_{s} \Delta)}{\sin^{2}(2 \pi^{2} \beta_{2} L_{s} \Delta)}
//! $$
//!
//! (the coherent accumulation of the NLI of the spans), or $\chi = N_{s}$ for the
//! incoherent accumulation. [`GnModel::nli_psd`] integrates it numerically on a grid.
//! [`closed_form_nli_psd`] is Poggiolini's closed form for $N_{\mathrm{ch}}$ channels of
//! rectangular spectrum, incoherent accumulation:
//!
//! $$
//! G_{\mathrm{NLI}} \approx N_{s} \frac{8}{27} \gamma^{2} G_{\mathrm{ch}}^{3} L_{\mathrm{eff}}^{2}
//!   \frac{\operatorname{asinh}\left(\frac{\pi^{2}}{2} |\beta_{2}| L_{\mathrm{eff},a} R_{s}^{2}
//!   N_{\mathrm{ch}}^{2 R_{s} / \Delta f}\right)}{\pi |\beta_{2}| L_{\mathrm{eff},a}},
//! $$
//!
//! with $G_{\mathrm{ch}} = P_{\mathrm{ch}}/R_{s}$, $L_{\mathrm{eff}} = (1 - e^{-\alpha
//! L_{s}})/\alpha$ and $L_{\mathrm{eff},a} = 1/\alpha$.
//!
//! The SNR of a channel after a matched filter (noise bandwidth $R_{s}$) is then
//!
//! $$
//! \mathrm{SNR} = \frac{P_{\mathrm{ch}}}{(G_{\mathrm{ASE}} + G_{\mathrm{NLI}}) R_{s}},
//! \qquad G_{\mathrm{ASE}} = 2 N_{s} n_{sp} (G - 1) h f_{c},
//! $$
//!
//! the ASE of the EDFAs of [`super::edfa::Edfa`] in both polarisations. Since
//! $G_{\mathrm{NLI}} \propto P^{3}$, the SNR is largest at the launch power
//! $P_{\mathrm{opt}} = \left(G_{\mathrm{ASE}} / (2 \eta)\right)^{1/3}$ with $\eta = G_{\mathrm{NLI}}/P^{3}$,
//! where the NLI is half the ASE.
//!
//! # The EGN model
//!
//! The GN model treats the transmitted signal as Gaussian: for QAM formats it
//! overestimates the NLI (by 1 to 4 dB for few spans, about 1 dB for many). The EGN
//! model of Carena et al. (2014) adds the corrections of the fourth and sixth moments
//! $\Phi$, $\Psi$ of the symbols ([`FormatMoments`]): for the self-channel
//! interference of the channel under test (Eqs. (5)–(9))
//!
//! $$
//! G_{\mathrm{SCI}}^{\mathrm{EGN}} = P^{3} \left[\kappa_{1} + \Phi \kappa_{2} +
//!   \Psi \kappa_{3}\right],
//! $$
//!
//! and for the cross-channel interference of every other channel in the X1 domain
//! (Eqs. (14)–(17)), $P_{\mathrm{CUT}} P_{\mathrm{INT}}^{2} \Phi \kappa_{12}$, where
//! $\kappa_{1}$ is the GN model and $\kappa_{2}$, $\kappa_{3}$, $\kappa_{12}$ integrate
//! products of the link function $\mu(f_{1}, f_{2}, f)$ (with the power profiles of
//! Raman spans when there are some). With separable spectra these integrals over three
//! and four frequencies are squared magnitudes of single and double sums. The
//! corrections of the X2–X4 domains of the XCI and of the multi-channel interference
//! (Appendix B), smaller, are not included: their GN parts are. Validation:
//!
//! - Fig. 1(a) of Carena et al., the SCI of PM-QPSK over 1 to 50 spans of SMF: the GN
//!   curve within 0.2 dB and the EGN curve within 0.25 dB at 1, 10, 20 and 50 spans;
//!   between 2 and 5 spans the published EGN curve is up to 0.9 dB below its own
//!   formulas, which a Monte Carlo evaluation of the first-order perturbation reproduces
//!   within 0.1 dB (`validation/tests/egn.rs`);
//! - Manakov simulations of 16-QAM over five channels and five spans
//!   (`specs/wdm_gn.yaml`): within 0.3 dB from the ASE-limited regime to 4 dB above the
//!   optimum, where the GN model is 1.7 dB off.
//!
//! # References
//!
//! 1. P. Poggiolini, "The GN model of non-linear propagation in uncompensated coherent
//!    optical systems," *J. Lightw. Technol.*, 30(24), 3857–3879, 2012.
//! 2. P. Poggiolini, G. Bosco, A. Carena, V. Curri, Y. Jiang and F. Forghieri, "The GN
//!    model of fiber non-linear propagation and its applications," *J. Lightw.
//!    Technol.*, 32(4), 694–721, 2014.
//! 3. A. Carena, G. Bosco, V. Curri, Y. Jiang, P. Poggiolini and F. Forghieri, "EGN
//!    model of non-linear fiber propagation," *Opt. Express*, 22(13), 16335–16362, 2014.

use super::edfa::ase_density;
use super::fiber::FiberParams;
use super::raman::{Raman, RamanProfile};
use com_core::{Result, ensure, ensure_shape};
use com_math::{C64, db2lin};
use std::f64::consts::PI;

/// A WDM signal: the centre frequencies and powers of its channels, all at the same
/// symbol rate with raised-cosine spectra.
#[derive(Clone, Debug, PartialEq)]
pub struct WdmSpectrum {
    /// Centre frequencies relative to the carrier (Hz).
    pub freqs: Vec<f64>,
    /// Launch power of every channel, both polarisations (W).
    pub powers_w: Vec<f64>,
    /// Symbol rate $R_{s}$ (Bd).
    pub symbol_rate: f64,
    /// Roll-off of the raised-cosine spectra (0 for rectangular spectra).
    pub rolloff: f64,
}

impl WdmSpectrum {
    /// `n` channels of power `power_w` on the grid `freqs`.
    pub fn uniform(freqs: Vec<f64>, power_w: f64, symbol_rate: f64, rolloff: f64) -> WdmSpectrum {
        let n = freqs.len();
        WdmSpectrum { freqs, powers_w: vec![power_w; n], symbol_rate, rolloff }
    }

    /// The power spectral density $G(f)$ (W/Hz).
    pub fn psd(&self, f: f64) -> f64 {
        let rs = self.symbol_rate;
        self.freqs.iter().zip(&self.powers_w).map(|(fc, p)| p / rs * self.shape(f - fc)).sum()
    }

    /// The raised-cosine power shape of a channel at the offset `x` from its centre
    /// (1 in the flat band).
    fn shape(&self, x: f64) -> f64 {
        let (rs, b) = (self.symbol_rate, self.rolloff);
        let x = x.abs();
        let lo = (1.0 - b) * rs / 2.0;
        let hi = (1.0 + b) * rs / 2.0;
        if x <= lo {
            1.0
        } else if x >= hi {
            0.0
        } else {
            0.5 * (1.0 + (PI / (b * rs) * (x - lo)).cos())
        }
    }

    /// Check the spectrum before an integration on its band: at least one channel, one
    /// power per channel, finite frequencies and powers, a positive finite symbol rate
    /// and a roll-off in $[0, 1]$.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for no channel or another number of powers than of
    /// channels, [`com_core::Error::Config`] for the rest.
    fn check(&self) -> Result<()> {
        ensure_shape(!self.freqs.is_empty() && self.freqs.len() == self.powers_w.len(), || {
            format!(
                "the spectrum needs one power per channel, got {} channels and {} powers",
                self.freqs.len(),
                self.powers_w.len()
            )
        })?;
        let rs = self.symbol_rate;
        ensure(rs.is_finite() && rs > 0.0, || format!("the symbol rate must be positive and finite, got {rs}"))?;
        let b = self.rolloff;
        ensure((0.0..=1.0).contains(&b), || format!("the roll-off must lie in [0, 1], got {b}"))?;
        ensure(self.freqs.iter().chain(&self.powers_w).all(|v| v.is_finite()), || {
            "the frequencies and powers of the channels must be finite".into()
        })
    }

    /// The band occupied by the channels (Hz).
    fn band(&self) -> (f64, f64) {
        let half = (1.0 + self.rolloff) * self.symbol_rate / 2.0;
        let lo = self.freqs.iter().copied().fold(f64::INFINITY, f64::min) - half;
        let hi = self.freqs.iter().copied().fold(f64::NEG_INFINITY, f64::max) + half;
        (lo, hi)
    }
}

/// The moments of a modulation format that the EGN model needs (Carena et al. 2014,
/// Eqs. (6), (19)), for symbols $a$ of one polarisation:
///
/// $$
/// \Phi = \frac{\mathrm{E}|a|^{4}}{\mathrm{E}^{2}|a|^{2}} - 2, \qquad
/// \Psi = \frac{\mathrm{E}|a|^{6}}{\mathrm{E}^{3}|a|^{2}} -
///   9 \frac{\mathrm{E}|a|^{4}}{\mathrm{E}^{2}|a|^{2}} + 12;
/// $$
///
/// both vanish for Gaussian symbols. QPSK: $(-1, 4)$; 16-QAM: $(-17/25, 52/25)$;
/// 64-QAM: $(-13/21, 1161/646)$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FormatMoments {
    /// $\Phi$ (fourth moment).
    pub phi: f64,
    /// $\Psi$ (sixth moment).
    pub psi: f64,
}

impl FormatMoments {
    /// Gaussian symbols: the EGN model is the GN model.
    pub const GAUSSIAN: FormatMoments = FormatMoments { phi: 0.0, psi: 0.0 };

    /// The moments of the constellation `points` with the probabilities
    /// `probabilities` (uniform if `None`), for example of a shaped source.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of probabilities than of points,
    /// [`com_core::Error::Config`] for no point, a negative or non-finite probability,
    /// probabilities of zero sum, or a constellation of zero energy.
    pub fn of(points: &[C64], probabilities: Option<&[f64]>) -> Result<FormatMoments> {
        ensure(!points.is_empty(), || "the moments of a format need at least one point".into())?;
        if let Some(q) = probabilities {
            com_core::same_len(q.len(), points.len(), "the probabilities and the points")?;
            ensure(q.iter().all(|p| p.is_finite() && *p >= 0.0), || {
                "the probabilities must be finite and non-negative".into()
            })?;
        }
        let n = points.len() as f64;
        let (mut m2, mut m4, mut m6, mut total) = (0.0, 0.0, 0.0, 0.0);
        for (k, x) in points.iter().enumerate() {
            let p = probabilities.map_or(1.0 / n, |q| q[k]);
            let r = x.norm_sqr();
            m2 += p * r;
            m4 += p * r * r;
            m6 += p * r * r * r;
            total += p;
        }
        ensure(total > 0.0, || "the probabilities of the points sum to zero".into())?;
        let (m2, m4, m6) = (m2 / total, m4 / total, m6 / total);
        ensure(m2 > 0.0 && m6.is_finite(), || {
            format!("the constellation must have a finite non-zero energy, got {m2}")
        })?;
        let k4 = m4 / (m2 * m2);
        Ok(FormatMoments { phi: k4 - 2.0, psi: m6 / (m2 * m2 * m2) - 9.0 * k4 + 12.0 })
    }
}

/// The GN model of a link (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct GnModel {
    /// The link.
    pub fiber: FiberParams,
    /// Coherent accumulation of the NLI over the spans (the phased-array factor);
    /// incoherent ($N_{s}$ times one span) otherwise.
    pub coherent: bool,
    /// Grid step of the numerical integration (Hz); if `None`, the smaller of
    /// $R_{s}/256$ (coherent accumulation) or $R_{s}/64$ (incoherent) and
    /// $\alpha / (4\pi^{2} |\beta_{2}| D)$ (with the EGN correction and the coherent
    /// accumulation, $R_{s}/(256 \max(1, N_{s}/10))$ instead of $R_{s}/256$: the
    /// correction evaluates its phased-array factor on the nodes), with $D$ the largest distance from the
    /// frequency of interest to the edges of the band: the link function of a pair of
    /// frequencies $f_{1}, f_{2}$ depends on $(f_{1} - f)(f_{2} - f)$, so it varies the faster
    /// along $f_{2}$ the farther $f_{1}$ is from $f$. The phased-array factor of the
    /// coherent accumulation, whose peaks sharpen with the number of spans, is averaged
    /// exactly over every cell of the grid (see [`GnModel::nli_psd`]), so that it needs no
    /// finer grid.
    pub resolution_hz: Option<f64>,
    /// Distributed Raman amplification and ISRS in every span (see the module): the
    /// profiles of the channels replace the exponential decay, and the amplifiers
    /// restore every channel to its launch power.
    pub raman: Option<Raman>,
    /// The EGN model: the corrections of the non-Gaussian symbols of this format (see
    /// the module); the GN model if `None`.
    pub egn: Option<FormatMoments>,
}

/// The launch power per channel (W) of the Raman profiles from which [`GnModel::eta`]
/// and [`GnModel::optimal_power`] are computed: 1 mW (0 dBm).
pub const RAMAN_REFERENCE_POWER_W: f64 = 1e-3;

/// Segments of the span on which the power profiles are integrated piecewise
/// exponentially: exact for the loss alone, within 0.003 dB of a fine quadrature for a
/// counter-pumped span.
const PROFILE_SEGMENTS: usize = 50;

impl GnModel {
    /// The model of `fiber` with coherent accumulation and an automatic grid.
    pub fn new(fiber: FiberParams) -> GnModel {
        GnModel { fiber, coherent: true, resolution_hz: None, raman: None, egn: None }
    }

    /// The ASE power spectral density at the receiver, both polarisations (W/Hz), of
    /// the EDFAs alone (without [`GnModel::raman`]).
    pub fn ase_psd(&self) -> f64 {
        2.0 * self.fiber.n_spans as f64 * self.fiber.span_edfa().ase_psd()
    }

    /// The noise power spectral density at the receiver at channel `ch`, both
    /// polarisations (W/Hz): with [`GnModel::raman`], the spontaneous Raman noise of
    /// every span and the ASE of the EDFA that restores the channel, where the Raman
    /// gain does not compensate the loss of the span; [`GnModel::ase_psd`] otherwise.
    ///
    /// # Errors
    /// As [`GnModel::snr`].
    pub fn ase_psd_channel(&self, spectrum: &WdmSpectrum, ch: usize) -> Result<f64> {
        ensure(ch < spectrum.freqs.len(), || {
            format!("no channel {ch}: the spectrum has {} channels", spectrum.freqs.len())
        })?;
        let profile = self.profile(spectrum)?;
        Ok(self.ase_with(spectrum, ch, profile.as_ref()))
    }

    fn ase_with(&self, spectrum: &WdmSpectrum, ch: usize, profile: Option<&RamanProfile>) -> f64 {
        let Some(p) = profile else { return self.ase_psd() };
        let fib = &self.fiber;
        let g = db2lin(-p.net_gain_db(ch));
        // the EDFA restores the channel where the Raman gain does not, both polarisations
        let edfa = if g > 1.0 { 2.0 * ase_density(g, db2lin(fib.nf_db), fib.fc_hz + spectrum.freqs[ch]) } else { 0.0 };
        fib.n_spans as f64 * (p.ase_psd[ch] * g + edfa)
    }

    /// The Raman profile of the channels of `spectrum` over one span.
    ///
    /// # Errors
    /// As [`Raman::solve`].
    pub fn profile(&self, spectrum: &WdmSpectrum) -> Result<Option<RamanProfile>> {
        let Some(raman) = &self.raman else { return Ok(None) };
        spectrum.check()?;
        let signals: Vec<(f64, f64)> =
            spectrum.freqs.iter().zip(&spectrum.powers_w).map(|(f, p)| (self.fiber.fc_hz + f, *p)).collect();
        raman.solve(&self.fiber, &signals).map(Some)
    }

    /// The NLI power spectral density $G_{\mathrm{NLI}}(f)$ at the receiver (W/Hz).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for an empty spectrum or another number of powers than
    /// of channels, [`com_core::Error::Config`] for a non-positive resolution, a
    /// non-positive or non-finite symbol rate, a roll-off outside $[0, 1]$, a non-finite
    /// frequency or power, and as [`Raman::solve`].
    pub fn nli_psd(&self, spectrum: &WdmSpectrum, f: f64) -> Result<f64> {
        let profile = self.profile(spectrum)?;
        self.nli_with(spectrum, f, profile.as_ref())
    }

    fn nli_with(&self, spectrum: &WdmSpectrum, f: f64, profile: Option<&RamanProfile>) -> Result<f64> {
        spectrum.check()?;
        ensure(f.is_finite(), || format!("the frequency must be finite, got {f}"))?;
        let fib = &self.fiber;
        let (lo, hi) = spectrum.band();
        let df = self.resolution_hz.unwrap_or_else(|| self.default_resolution(spectrum, f, lo, hi));
        ensure(df > 0.0 && df.is_finite(), || format!("the resolution must be positive, got {df}"))?;
        // a grid with f on a node: f1 + f2 - f is then a node too
        let k_lo = ((lo - f) / df).floor() as i64;
        let k_hi = ((hi - f) / df).ceil() as i64;
        let grid = NodeGrid { f, df, k_lo, m: (k_hi - k_lo + 1) as usize };
        let span = SpanLink::new(fib, &grid, profile);
        let gn = self.gn_integral(spectrum, &grid, &span);
        let corr = match self.egn {
            Some(fm) if fm != FormatMoments::GAUSSIAN => {
                let per_span = if self.coherent { 1.0 } else { fib.n_spans as f64 };
                per_span * egn_correction(spectrum, f, df, fm, &|n1, n2| self.egn_mu(&grid, &span, n1, n2))
            }
            _ => 0.0,
        };
        Ok(gn + corr)
    }

    /// The default grid step of an integration at `f` over the band `lo..hi` (see
    /// [`GnModel::resolution_hz`]).
    fn default_resolution(&self, spectrum: &WdmSpectrum, f: f64, lo: f64, hi: f64) -> f64 {
        let fib = &self.fiber;
        let rs = spectrum.symbol_rate;
        let ns = fib.n_spans as f64;
        // the EGN correction evaluates its own phased-array factor on the nodes: its
        // grid keeps sharpening with the number of spans (at 50 spans, R_s/256
        // overestimates the correction of one channel by 45 %)
        let egn = matches!(self.egn, Some(fm) if fm != FormatMoments::GAUSSIAN);
        let base = match (self.coherent, egn) {
            (true, true) => rs / 256.0 / (ns / 10.0).max(1.0),
            (true, false) => rs / 256.0,
            (false, _) => rs / 64.0,
        };
        // the link function varies along f2 on the scale α / (4π² |β2| |f1 - f|)
        let far = (hi - f).abs().max((f - lo).abs());
        let band = fib.alpha_np_km() / (4.0 * PI * PI * fib.beta2_s2_km().abs() * far);
        if band.is_finite() && band > 0.0 { base.min(band) } else { base }
    }

    /// The GN reference formula (see the module) on the nodes of `grid`: the sum over
    /// the cells of $G(f_{1}) G(f_{2}) G(f_{1} + f_{2} - f) |\rho|^{2} \chi$ times
    /// $\frac{16}{27} \gamma^{2} \Delta f^{2}$.
    fn gn_integral(&self, spectrum: &WdmSpectrum, grid: &NodeGrid, span: &SpanLink) -> f64 {
        let fib = &self.fiber;
        let (f, df, k_lo, m) = (grid.f, grid.df, grid.k_lo, grid.m);
        let (a, e, c, ls) = (span.alpha, span.decay, span.phase, span.length_km);
        let ns = fib.n_spans as f64;
        let g: Vec<f64> = (0..m as i64).map(|i| spectrum.psd(grid.node(i))).collect();
        // the integrand is symmetric in (f1, f2), and so is its evaluation (the phase
        // c (u v), the Fejér factor and the link function are computed symmetrically):
        // the cells below the diagonal are those above it
        let mut acc = 0.0;
        for i in 0..m {
            if g[i] == 0.0 {
                continue;
            }
            let u = grid.node(i as i64) - f;
            for j in i..m {
                // f1 + f2 - f is node i + j + k_lo
                let k3 = i as i64 + j as i64 + k_lo;
                if g[j] == 0.0 || k3 < 0 || k3 >= m as i64 {
                    continue;
                }
                let g3 = g[k3 as usize];
                if g3 == 0.0 {
                    continue;
                }
                let ph = c * (u * (grid.node(j as i64) - f));
                let rho2 = match &span.half_log {
                    // |1 - e^{-αL} e^{jφ}|² / (α² + φ²/L²)
                    None => (1.0 - 2.0 * e * ph.cos() + e * e) / (a * a + (ph / ls) * (ph / ls)),
                    Some(hl) => link_function(&hl[i], &hl[j], &hl[k3 as usize], &hl[m], ph / ls, ls).norm_sqr(),
                };
                let chi = if self.coherent {
                    // the phase c u v changes by c v df along u and c u df along v over
                    // the cell
                    fejer_cell_average(fib.n_spans, ph, c * (grid.node(j as i64) - f) * df, c * u * df)
                } else {
                    ns
                };
                let cell = g[i] * g[j] * g3 * rho2 * chi;
                acc += if j == i { cell } else { 2.0 * cell };
            }
        }
        16.0 / 27.0 * fib.gamma_w_km * fib.gamma_w_km * acc * df * df
    }

    /// The link function $\gamma \mu(f_{1}, f_{2}, f)$ of the EGN model at the offsets
    /// $n_{1} \Delta f$, $n_{2} \Delta f$ from $f$ (Carena et al. 2014, Eqs. (10)–(12)):
    /// the span efficiency (of the power profiles with Raman spans) times the
    /// phased-array term $e^{j (N_{s} - 1) x} \sin(N_{s} x) / \sin x$ of the coherent
    /// accumulation, $x = \kappa L_{s} / 2$.
    fn egn_mu(&self, grid: &NodeGrid, span: &SpanLink, n1: i64, n2: i64) -> C64 {
        let fib = &self.fiber;
        let (a, e, c, ls) = (span.alpha, span.decay, span.phase, span.length_km);
        let ns = fib.n_spans as f64;
        let df = grid.df;
        let kappa = c / ls * (n1 as f64 * df) * (n2 as f64 * df);
        let link = match &span.half_log {
            None => {
                let w = C64::new(a, -kappa);
                if w.norm() * ls < 1e-12 {
                    C64::new(ls, 0.0)
                } else {
                    (C64::new(1.0, 0.0) - e * C64::new(0.0, kappa * ls).exp()) / w
                }
            }
            Some(hl) => {
                let at = |n: i64| &hl[(n - grid.k_lo) as usize];
                link_function(at(n1), at(n2), at(n1 + n2), &hl[grid.m], kappa, ls)
            }
        };
        let nu = if self.coherent {
            let x = kappa * ls / 2.0;
            // sin(N x) / sin x = N diric(2x, N), with its limits at x = kπ
            C64::new(0.0, (ns - 1.0) * x).exp() * (ns * com_math::diric(2.0 * x, fib.n_spans))
        } else {
            C64::new(1.0, 0.0)
        };
        link * nu * fib.gamma_w_km
    }

    /// The SNR of channel `ch` after a matched filter, ASE and NLI (linear).
    ///
    /// # Errors
    /// As [`GnModel::nli_psd`], [`com_core::Error::Config`] for a missing channel and
    /// [`com_core::Error::Shape`] for another number of powers than of channels.
    pub fn snr(&self, spectrum: &WdmSpectrum, ch: usize) -> Result<f64> {
        ensure(ch < spectrum.freqs.len(), || {
            format!("no channel {ch}: the spectrum has {} channels", spectrum.freqs.len())
        })?;
        com_core::same_len(spectrum.freqs.len(), spectrum.powers_w.len(), "the channels and their powers")?;
        let profile = self.profile(spectrum)?;
        let nli = self.nli_with(spectrum, spectrum.freqs[ch], profile.as_ref())?;
        let ase = self.ase_with(spectrum, ch, profile.as_ref());
        Ok(spectrum.powers_w[ch] / ((ase + nli) * spectrum.symbol_rate))
    }

    /// The NLI coefficient $\eta = G_{\mathrm{NLI}}/P^{3}$ of channel `ch` of the grid, symbol
    /// rate and roll-off of `spectrum`, for equal channel powers ((W/Hz)/W³; the powers
    /// of `spectrum` are not used). With [`GnModel::raman`], the profiles depend on the
    /// powers (ISRS, pump depletion): $\eta$ is that of [`RAMAN_REFERENCE_POWER_W`] per
    /// channel.
    ///
    /// # Errors
    /// As [`GnModel::snr`].
    pub fn eta(&self, spectrum: &WdmSpectrum, ch: usize) -> Result<f64> {
        let freqs = &spectrum.freqs;
        ensure(ch < freqs.len(), || format!("no channel {ch}: the grid has {} channels", freqs.len()))?;
        let p = if self.raman.is_some() { RAMAN_REFERENCE_POWER_W } else { 1.0 };
        let s = WdmSpectrum::uniform(freqs.clone(), p, spectrum.symbol_rate, spectrum.rolloff);
        Ok(self.nli_psd(&s, freqs[ch])? / (p * p * p))
    }

    /// The optimal launch power per channel $(G_{\mathrm{ASE}}/(2\eta))^{1/3}$ (W) on the
    /// grid, symbol rate and roll-off of `spectrum` (with [`GnModel::raman`], for the
    /// profiles of [`RAMAN_REFERENCE_POWER_W`] per channel).
    ///
    /// # Errors
    /// As [`GnModel::eta`].
    pub fn optimal_power(&self, spectrum: &WdmSpectrum, ch: usize) -> Result<f64> {
        let ase = if self.raman.is_some() {
            let reference = WdmSpectrum::uniform(
                spectrum.freqs.clone(),
                RAMAN_REFERENCE_POWER_W,
                spectrum.symbol_rate,
                spectrum.rolloff,
            );
            self.ase_psd_channel(&reference, ch)?
        } else {
            self.ase_psd()
        };
        Ok((ase / (2.0 * self.eta(spectrum, ch)?)).cbrt())
    }
}

/// The nodes $f + (k_{\mathrm{lo}} + i) \Delta f$, $i = 0, \dots, m - 1$, of an
/// integration at $f$.
struct NodeGrid {
    f: f64,
    df: f64,
    k_lo: i64,
    m: usize,
}

impl NodeGrid {
    /// The frequency of node `i`.
    fn node(&self, i: i64) -> f64 {
        self.f + (self.k_lo + i) as f64 * self.df
    }
}

/// The quantities of one span that the integrands need: the attenuation $\alpha$, the
/// decay $e^{-\alpha L_{s}}$, the phase constant $4\pi^{2} \beta_{2} L_{s}$, the length,
/// and with Raman amplification half of $\ln G(z, f)$ of every node and of $f$ (last) on
/// the segments of the span.
struct SpanLink {
    alpha: f64,
    decay: f64,
    phase: f64,
    length_km: f64,
    half_log: Option<Vec<HalfLog>>,
}

impl SpanLink {
    fn new(fib: &FiberParams, grid: &NodeGrid, profile: Option<&RamanProfile>) -> SpanLink {
        let a = fib.alpha_np_km();
        let ls = fib.span_km;
        let half_log = profile.map(|p| {
            let mut freqs: Vec<f64> = (0..grid.m as i64).map(|i| fib.fc_hz + grid.node(i)).collect();
            freqs.push(fib.fc_hz + grid.f);
            let w = p.interpolator(&freqs);
            let mut out: Vec<HalfLog> = (0..freqs.len())
                .map(|_| HalfLog { h: vec![0.0; PROFILE_SEGMENTS + 1], x: vec![0.0; PROFILE_SEGMENTS + 1] })
                .collect();
            let mut buf = vec![0.0; freqs.len()];
            for s in 0..=PROFILE_SEGMENTS {
                p.log_gain_at(&w, ls * s as f64 / PROFILE_SEGMENTS as f64, &mut buf);
                for (o, v) in out.iter_mut().zip(&buf) {
                    o.h[s] = v / 2.0;
                    o.x[s] = (v / 2.0).exp();
                }
            }
            out
        });
        SpanLink {
            alpha: a,
            decay: (-a * ls).exp(),
            phase: 4.0 * PI * PI * fib.beta2_s2_km() * ls,
            length_km: ls,
            half_log,
        }
    }
}

/// The phased-array factor $\chi(\varphi) = \sin^{2}(N \varphi / 2) / \sin^{2}(\varphi / 2)$
/// of `n` spans averaged over a cell of the grid where $\varphi$ is linear, of value `phi`
/// at the node and of total variations `du` and `dv` along the two sides of the cell:
/// with the Fejér expansion $\chi = N + 2 \sum_{k=1}^{N-1} (N - k) \cos k \varphi$, the
/// average of $\cos k\varphi$ over the cell is $\cos(k \varphi)\,
/// \mathrm{sinc}(k\, du/2)\, \mathrm{sinc}(k\, dv/2)$ (a Filon quadrature: the factor
/// is integrated exactly, the rest of the integrand at the node).
fn fejer_cell_average(n: usize, phi: f64, du: f64, dv: f64) -> f64 {
    // sin(k x) and cos(k x) by the recurrences of the Chebyshev polynomials
    struct Harmonics {
        two_cos: f64,
        prev: (f64, f64),
        cur: (f64, f64),
    }
    impl Harmonics {
        fn new(x: f64) -> Harmonics {
            Harmonics { two_cos: 2.0 * x.cos(), prev: (0.0, 1.0), cur: (x.sin(), x.cos()) }
        }
        /// (sin k x, cos k x) for k = 1, 2, ...
        fn next(&mut self) -> (f64, f64) {
            let out = self.cur;
            let s = self.two_cos * self.cur.0 - self.prev.0;
            let c = self.two_cos * self.cur.1 - self.prev.1;
            self.prev = self.cur;
            self.cur = (s, c);
            out
        }
    }
    let sinc = |s: f64, x: f64, k: f64| if (k * x).abs() < 1e-12 { 1.0 } else { s / (k * x) };
    let (mut h_phi, mut h_u, mut h_v) = (Harmonics::new(phi), Harmonics::new(du / 2.0), Harmonics::new(dv / 2.0));
    let mut sum = n as f64;
    for k in 1..n {
        let kf = k as f64;
        let (_, cos_k) = h_phi.next();
        let (su, _) = h_u.next();
        let (sv, _) = h_v.next();
        // symmetric in (du, dv)
        sum += 2.0 * (n - k) as f64 * cos_k * (sinc(su, du / 2.0, kf) * sinc(sv, dv / 2.0, kf));
    }
    sum
}

/// The EGN correction at `f` (W/Hz): that of the self-channel interference of the
/// channel under test (Carena et al. 2014, Eqs. (5)–(9)) and that of the cross-channel
/// interference of every other channel in its X1 domain (Eqs. (14)–(17)), on the grid
/// of step `df` with `f` on a node; `mu(n1, n2)` is the link function of the offsets
/// $n_{1} \Delta f$, $n_{2} \Delta f$. Every integral over three or four frequencies
/// factorises into a squared magnitude of a sum, since the spectra separate.
fn egn_correction(spectrum: &WdmSpectrum, f: f64, df: f64, fm: FormatMoments, mu: &dyn Fn(i64, i64) -> C64) -> f64 {
    let rs = spectrum.symbol_rate;
    let half = (1.0 + spectrum.rolloff) * rs / 2.0;
    let Some(cut) = (0..spectrum.freqs.len())
        .filter(|k| (spectrum.freqs[*k] - f).abs() < half)
        .min_by(|a, b| (spectrum.freqs[*a] - f).abs().total_cmp(&(spectrum.freqs[*b] - f).abs()))
    else {
        return 0.0;
    };
    // the field spectrum of channel k at the offset n df from f: √shape / R_s
    let s = |k: usize, n: i64| spectrum.shape(f + n as f64 * df - spectrum.freqs[k]).sqrt() / rs;
    let support = |k: usize| -> Vec<(i64, f64)> {
        let lo = ((spectrum.freqs[k] - f - half) / df).ceil() as i64;
        let hi = ((spectrum.freqs[k] - f + half) / df).floor() as i64;
        (lo..=hi).map(|n| (n, s(k, n))).filter(|(_, v)| *v > 0.0).collect()
    };
    let band = support(cut);
    let pc = spectrum.powers_w[cut];
    let df3 = df * df * df;
    // SCI: κ2 (two terms) and κ3, with one evaluation of μ per pair (n1, n2) of the
    // band whose sum n3 = n1 + n2 is in the band too: the second term of κ2 sums
    // μ(n1, n3 - n1) s1 s(n3 - n1) over n1 for every n3, in the same order
    let n0 = band.first().map_or(0, |v| v.0);
    let mut inner_b = vec![C64::new(0.0, 0.0); band.last().map_or(0, |v| (v.0 - n0 + 1) as usize)];
    let (mut a, mut b, mut t) = (0.0, 0.0, C64::new(0.0, 0.0));
    for &(n1, s1) in &band {
        let mut inner = C64::new(0.0, 0.0);
        for &(n2, s2) in &band {
            let w = s2 * s(cut, n1 + n2);
            if w != 0.0 {
                let m = mu(n1, n2);
                inner += m * w;
                t += m * (s1 * w);
                if let Some(slot) = usize::try_from(n1 + n2 - n0).ok().and_then(|k| inner_b.get_mut(k)) {
                    *slot += m * (s1 * s2);
                }
            }
        }
        a += s1 * s1 * inner.norm_sqr();
    }
    for &(n3, s3) in &band {
        b += s3 * s3 * inner_b[(n3 - n0) as usize].norm_sqr();
    }
    let kappa2 = rs * rs * (80.0 / 81.0 * a + 16.0 / 81.0 * b) * df3;
    let kappa3 = 16.0 / 81.0 * rs * t.norm_sqr() * df3 * df;
    let mut corr = pc * pc * pc * (fm.phi * kappa2 + fm.psi * kappa3);
    // XCI, the X1 domain of every interfering channel: κ12
    for k in (0..spectrum.freqs.len()).filter(|k| *k != cut) {
        let int = support(k);
        let mut x = 0.0;
        for &(n1, s1) in &band {
            let mut inner = C64::new(0.0, 0.0);
            for &(n2, s2) in &int {
                let w = s2 * s(k, n1 + n2);
                if w != 0.0 {
                    inner += mu(n1, n2) * w;
                }
            }
            x += s1 * s1 * inner.norm_sqr();
        }
        let pk = spectrum.powers_w[k];
        corr += pc * pk * pk * fm.phi * 80.0 / 81.0 * rs * rs * x * df3;
    }
    corr
}

/// Half of $\ln G(z, f)$ of a frequency on the uniform segments of a span, and its
/// exponential $\sqrt{G}$.
struct HalfLog {
    h: Vec<f64>,
    x: Vec<f64>,
}

/// $\int_{0}^{L} e^{E(z)} e^{j \kappa z} dz$ with $E = \frac{1}{2}(\ln G_{1} +
/// \ln G_{2} + \ln G_{3} - \ln G_{f})$ linear on every segment of the span `ls`: the
/// generalised $|\rho|^{2}$ of the GN model with power profiles (Semrau et al. 2019,
/// Eq. (2)), exact for profiles that are exponential on every segment.
fn link_function(p1: &HalfLog, p2: &HalfLog, p3: &HalfLog, pf: &HalfLog, kappa: f64, ls: f64) -> C64 {
    let n = p1.h.len() - 1;
    let step = ls / n as f64;
    let rot = C64::new(0.0, kappa * step).exp();
    let mut phase = C64::new(1.0, 0.0);
    let mut acc = C64::new(0.0, 0.0);
    let mut ea = p1.h[0] + p2.h[0] + p3.h[0] - pf.h[0];
    let mut xa = p1.x[0] * p2.x[0] * p3.x[0] / pf.x[0];
    for s in 1..=n {
        let eb = p1.h[s] + p2.h[s] + p3.h[s] - pf.h[s];
        let xb = p1.x[s] * p2.x[s] * p3.x[s] / pf.x[s];
        let w = C64::new((eb - ea) / step, kappa);
        // ∫ over the segment of e^{E_a + (s + jκ)(z - z_a)} = e^{E_a} (e^{(s + jκ)Δ} - 1)/(s + jκ)
        let term =
            if w.norm_sqr() * step * step < 1e-12 { xa * step * (1.0 + w * step / 2.0) } else { (rot * xb - xa) / w };
        acc += phase * term;
        phase *= rot;
        ea = eb;
        xa = xb;
    }
    acc
}

/// Poggiolini's closed form of the NLI power spectral density at the centre of a WDM
/// comb of `n_channels` rectangular channels spaced by `spacing` (see the module),
/// incoherent accumulation (W/Hz).
pub fn closed_form_nli_psd(
    fiber: &FiberParams,
    n_channels: usize,
    symbol_rate: f64,
    spacing: f64,
    power_w: f64,
) -> f64 {
    let a = fiber.alpha_np_km();
    let b2 = fiber.beta2_s2_km().abs();
    let l_eff = fiber.effective_length_km();
    let l_eff_a = 1.0 / a;
    let g = power_w / symbol_rate;
    let arg = PI * PI / 2.0
        * b2
        * l_eff_a
        * symbol_rate
        * symbol_rate
        * (n_channels as f64).powf(2.0 * symbol_rate / spacing);
    fiber.n_spans as f64 * 8.0 / 27.0 * fiber.gamma_w_km.powi(2) * g.powi(3) * l_eff * l_eff * arg.asinh()
        / (PI * b2 * l_eff_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize, spacing: f64) -> Vec<f64> {
        (0..n).map(|k| (k as f64 - (n / 2) as f64) * spacing).collect()
    }

    #[test]
    fn the_cell_average_of_the_phased_array_factor_is_exact() {
        // a point cell: the factor itself
        for (n, phi) in [(1usize, 0.3), (5, 0.7), (20, 2.0)] {
            let x = phi / 2.0;
            let chi = (n as f64 * com_math::diric(2.0 * x, n)).powi(2);
            assert!((fejer_cell_average(n, phi, 0.0, 0.0) / chi - 1.0).abs() < 1e-10, "{n} {phi}");
        }
        // a cell spanning one period of the phase along a side: the mean N
        assert!((fejer_cell_average(10, 0.4, 2.0 * PI, 0.3) - 10.0).abs() < 1e-9);
        // against a fine midpoint rule over the cell
        let (n, phi, du, dv) = (12usize, 0.25, 1.3, 0.8);
        let k = 400;
        let mut mean = 0.0;
        for i in 0..k {
            for j in 0..k {
                let p = phi + du * ((i as f64 + 0.5) / k as f64 - 0.5) + dv * ((j as f64 + 0.5) / k as f64 - 0.5);
                mean += (n as f64 * com_math::diric(p, n)).powi(2) / (k * k) as f64;
            }
        }
        assert!((fejer_cell_average(n, phi, du, dv) / mean - 1.0).abs() < 1e-4);
    }

    #[test]
    fn the_default_grid_is_converged_far_from_the_centre_and_over_many_spans() {
        // the edge channel of nine: the link function of the far pairs, and the coherent
        // phased-array factor of 20 spans (its default grid used to miss them by 0.2 to
        // 1 dB)
        for (spans, coherent) in [(1usize, false), (20, true)] {
            let fiber = FiberParams { n_spans: spans, d_ps_nm_km: 16.7, ..FiberParams::default() };
            let gn = GnModel { coherent, ..GnModel::new(fiber) };
            let s = WdmSpectrum::uniform(grid(9, 50e9), 1e-3, 32e9, 0.02);
            let f = s.freqs[0];
            let default = gn.nli_psd(&s, f).unwrap();
            let df = 2e8f64.min(32e9 / 256.0);
            let fine = GnModel { resolution_hz: Some(df / 4.0), ..gn.clone() }.nli_psd(&s, f).unwrap();
            let db = com_math::lin2db(default / fine);
            assert!(db.abs() < 0.02, "{spans} spans: {db} dB");
        }
    }

    // the half of the grid above the diagonal (the symmetry in f1, f2) against the whole
    // grid, as integrated before, coherent and incoherent
    #[test]
    fn the_symmetric_half_of_the_grid_gives_the_whole_integral() {
        let fib = FiberParams { n_spans: 5, ..FiberParams::default() };
        let s = WdmSpectrum::uniform(vec![-50e9, 0.0, 50e9], 1e-3, 32e9, 0.1);
        let df = 32e9 / 32.0;
        let (a, ls) = (fib.alpha_np_km(), fib.span_km);
        let (e, c) = ((-a * ls).exp(), 4.0 * PI * PI * fib.beta2_s2_km() * ls);
        for coherent in [false, true] {
            let model = GnModel { coherent, resolution_hz: Some(df), ..GnModel::new(fib.clone()) };
            for f in [0.0, 7e9] {
                let (lo, hi) = s.band();
                let k_lo = ((lo - f) / df).floor() as i64;
                let n = (((hi - f) / df).ceil() as i64 - k_lo + 1) as usize;
                let node = |i: usize| f + (k_lo + i as i64) as f64 * df;
                let g: Vec<f64> = (0..n).map(|i| s.psd(node(i))).collect();
                let mut acc = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        let Some(&g3) = usize::try_from(i as i64 + j as i64 + k_lo).ok().and_then(|k| g.get(k)) else {
                            continue;
                        };
                        let (u, v) = (node(i) - f, node(j) - f);
                        let ph = c * u * v;
                        let rho2 = (1.0 - 2.0 * e * ph.cos() + e * e) / (a * a + (ph / ls) * (ph / ls));
                        let chi = if coherent {
                            fejer_cell_average(fib.n_spans, ph, c * v * df, c * u * df)
                        } else {
                            fib.n_spans as f64
                        };
                        acc += g[i] * g[j] * g3 * rho2 * chi;
                    }
                }
                let whole = 16.0 / 27.0 * fib.gamma_w_km * fib.gamma_w_km * acc * df * df;
                let half = model.nli_psd(&s, f).unwrap();
                assert!((half / whole - 1.0).abs() < 1e-12, "coherent {coherent}, f = {f}: {half} vs {whole}");
            }
        }
    }

    #[test]
    fn the_integral_agrees_with_the_closed_form_for_nyquist_wdm() {
        let fiber = FiberParams { n_spans: 1, d_ps_nm_km: 16.0, ..FiberParams::default() };
        let gn = GnModel { coherent: false, ..GnModel::new(fiber.clone()) };
        for n in [1, 3, 5] {
            let s = WdmSpectrum::uniform(grid(n, 32e9), 1e-3, 32e9, 0.0);
            let num = gn.nli_psd(&s, 0.0).unwrap();
            let cf = closed_form_nli_psd(&fiber, n, 32e9, 32e9, 1e-3);
            assert!((num / cf - 1.0).abs() < 0.1, "{n} channels: {num} vs {cf}");
        }
    }

    #[test]
    fn the_profiles_of_the_loss_alone_give_the_exponential_link_function() {
        // Raman without pumps nor ISRS: the profiles are e^{-αz}, the NLI and the ASE of
        // the EDFAs those of the model without Raman
        let fiber = FiberParams { n_spans: 3, d_ps_nm_km: 16.0, ..FiberParams::default() };
        let plain = GnModel::new(fiber.clone());
        let raman = GnModel { raman: Some(Raman { isrs: false, ..Raman::default() }), ..plain.clone() };
        let s = WdmSpectrum::uniform(grid(3, 50e9), 1e-3, 32e9, 0.1);
        let (a, b) = (plain.nli_psd(&s, 0.0).unwrap(), raman.nli_psd(&s, 0.0).unwrap());
        assert!((a / b - 1.0).abs() < 1e-9, "{a} {b}");
        let (a, b) = (plain.ase_psd(), raman.ase_psd_channel(&s, 1).unwrap());
        assert!((a / b - 1.0).abs() < 1e-9, "{a} {b}");
    }

    #[test]
    fn the_moments_of_the_formats_are_those_of_carena() {
        let qam = |m: usize| -> Vec<C64> {
            let l = (m as f64).sqrt() as i64;
            (0..m as i64).map(|k| C64::new((2 * (k % l) - l + 1) as f64, (2 * (k / l) - l + 1) as f64)).collect()
        };
        for (m, phi, psi) in [(4, -1.0, 4.0), (16, -17.0 / 25.0, 52.0 / 25.0), (64, -13.0 / 21.0, 1161.0 / 646.0)] {
            let f = FormatMoments::of(&qam(m), None).unwrap();
            assert!((f.phi - phi).abs() < 1e-12 && (f.psi - psi).abs() < 1e-3, "{m}: {f:?}");
        }
    }

    #[test]
    fn format_moments_of_invalid_inputs_are_errors_not_panics() {
        let qpsk = [C64::new(1.0, 1.0), C64::new(-1.0, 1.0), C64::new(-1.0, -1.0), C64::new(1.0, -1.0)];
        let uniform = FormatMoments::of(&qpsk, Some(&[0.25; 4])).unwrap();
        assert!((uniform.phi + 1.0).abs() < 1e-12 && (uniform.psi - 4.0).abs() < 1e-12);
        assert!(matches!(FormatMoments::of(&qpsk, Some(&[0.5, 0.5])), Err(com_core::Error::Shape(_))));
        assert!(FormatMoments::of(&[], None).is_err());
        assert!(FormatMoments::of(&qpsk, Some(&[0.0; 4])).is_err());
        assert!(FormatMoments::of(&qpsk, Some(&[-0.5, 0.5, 0.5, 0.5])).is_err());
        assert!(FormatMoments::of(&[C64::new(0.0, 0.0)], None).is_err());
    }

    #[test]
    fn a_spectrum_of_invalid_rate_or_rolloff_is_an_error_not_an_abort() {
        let gn = GnModel::new(FiberParams::default());
        for (rs, b) in [(-32e9, 0.1), (0.0, 0.1), (f64::NAN, 0.1), (f64::INFINITY, 0.1), (32e9, -0.1), (32e9, 1.5)] {
            let s = WdmSpectrum::uniform(grid(3, 50e9), 1e-3, rs, b);
            assert!(matches!(gn.nli_psd(&s, 0.0), Err(com_core::Error::Config(_))), "{rs} {b}");
            assert!(gn.snr(&s, 1).is_err());
            assert!(gn.eta(&s, 1).is_err());
        }
        let s = WdmSpectrum::uniform(vec![0.0, f64::NAN], 1e-3, 32e9, 0.1);
        assert!(gn.nli_psd(&s, 0.0).is_err());
        let s = WdmSpectrum::uniform(grid(3, 50e9), 1e-3, 32e9, 0.1);
        assert!(gn.nli_psd(&s, f64::NAN).is_err());
        assert!(gn.nli_psd(&s, 0.0).unwrap() > 0.0);
    }

    #[test]
    fn egn_of_gaussian_symbols_is_the_gn_model_and_qpsk_gets_less_nli() {
        let fiber = FiberParams { n_spans: 2, d_ps_nm_km: 16.0, ..FiberParams::default() };
        let gn = GnModel::new(fiber);
        let s = WdmSpectrum::uniform(grid(3, 50e9), 1e-3, 32e9, 0.05);
        let egn = |fm| GnModel { egn: Some(fm), ..gn.clone() }.nli_psd(&s, 0.0).unwrap();
        assert_eq!(egn(FormatMoments::GAUSSIAN), gn.nli_psd(&s, 0.0).unwrap());
        let ratio = egn(FormatMoments { phi: -1.0, psi: 4.0 }) / gn.nli_psd(&s, 0.0).unwrap();
        assert!(ratio > 0.3 && ratio < 0.9, "{ratio}");
    }

    #[test]
    fn one_span_of_pm_qpsk_has_the_sci_of_carena() {
        // Carena et al. 2014, Fig. 1(a): one channel of PM-QPSK at 32 GBd (roll-off 0.05),
        // one span of 100 km of SMF: η_SCI = 22.3 dB(1/W²) by the GN model, 18.1 by the EGN
        let fiber = FiberParams {
            span_km: 100.0,
            d_ps_nm_km: 16.7,
            gamma_w_km: 1.3,
            alpha_db_km: 0.22,
            ..FiberParams::default()
        };
        let rs = 32e9;
        let s = WdmSpectrum::uniform(vec![0.0], 1.0, rs, 0.05);
        let eta = |egn| {
            let m = GnModel { egn, resolution_hz: Some(rs / 64.0), ..GnModel::new(fiber.clone()) };
            // Simpson over the band
            let p = com_math::simpson(|f| m.nli_psd(&s, f).unwrap(), -rs / 2.0, rs / 2.0, 8);
            com_math::lin2db(p)
        };
        let (gn, egn) = (eta(None), eta(Some(FormatMoments { phi: -1.0, psi: 4.0 })));
        assert!((gn - 22.3).abs() < 0.15 && (egn - 18.1).abs() < 0.25, "GN {gn} EGN {egn}");
    }

    #[test]
    fn nli_grows_as_the_cube_of_the_power_and_with_the_spans() {
        let fiber = FiberParams { n_spans: 2, d_ps_nm_km: 16.0, ..FiberParams::default() };
        let gn = GnModel::new(fiber);
        let f = grid(3, 50e9);
        let a = gn.nli_psd(&WdmSpectrum::uniform(f.clone(), 1e-3, 32e9, 0.1), 0.0).unwrap();
        let b = gn.nli_psd(&WdmSpectrum::uniform(f, 2e-3, 32e9, 0.1), 0.0).unwrap();
        assert!((b / a - 8.0).abs() < 1e-9);
        // coherent accumulation over two spans: more than twice one span
        let one = GnModel { fiber: FiberParams { n_spans: 1, ..gn.fiber.clone() }, ..gn.clone() };
        let s = WdmSpectrum::uniform(grid(3, 50e9), 1e-3, 32e9, 0.1);
        let ratio = gn.nli_psd(&s, 0.0).unwrap() / one.nli_psd(&s, 0.0).unwrap();
        assert!(ratio > 2.0 && ratio < 4.0, "{ratio}");
    }
}
