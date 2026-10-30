//! Continuous-phase modulation (CPM): CPFSK, MSK and GMSK, their coherent
//! maximum-likelihood sequence detector, two simple detectors for MSK and GMSK, and the
//! closed forms of the MSK spectrum and of the Gaussian pulse.
//!
//! # Model
//!
//! An $M$-ary symbol sequence $a_{k} \in \\{\pm 1, \pm 3, \dots, \pm (M - 1)\\}$ is
//! carried by the phase of a constant-envelope signal,
//!
//! $$
//! s(t) = e^{j \varphi(t)}, \qquad
//! \varphi(t) = 2 \pi h \sum_{k} a_{k} \\, q(t - kT),
//! $$
//!
//! where $h = p/q$ is the **modulation index** (a rational number in lowest terms) and
//! $q(t) = \int_{-\infty}^{t} g(\tau) \\, d\tau$ the **phase response**, the integral of a
//! frequency pulse $g$ of duration $LT$ and area $q(LT) = 1/2$ (Proakis & Salehi, ch. 3,
//! §3.3-3). The pulses of [`FrequencyPulse`] (Proakis & Salehi, Table 3.3-1):
//!
//! | Pulse | $g(t)$ on $0 \le t \le LT$ |
//! |---|---|
//! | $L$REC (rectangular; 1REC is CPFSK, MSK is 1REC with $h = 1/2$) | $1/(2LT)$ |
//! | $L$RC (raised cosine) | $\frac{1}{2LT}\left(1 - \cos \frac{2 \pi t}{LT}\right)$ |
//! | GMSK (Gaussian, $BT$ the normalised bandwidth of the Gaussian filter) | $\frac{1}{2T}\left[Q\left(2 \pi B \frac{t - LT/2 - T/2}{\sqrt{\ln 2}}\right) - Q\left(2 \pi B \frac{t - LT/2 + T/2}{\sqrt{\ln 2}}\right)\right]$, truncated to $L$ symbols and scaled to the area $1/2$ |
//!
//! The Gaussian pulse is the response of the Gaussian low-pass filter of Murota & Hirade
//! (1981, §II) to a rectangular pulse of one symbol ([`gmsk_pulse`]).
//!
//! # Phase state
//!
//! During the interval $nT \le t < (n + 1)T$ the phase splits into a **terminal phase**,
//! which the symbols that have left the pulse window fix, and the partial response of the
//! last $L$ symbols:
//!
//! $$
//! \varphi(t) = \underbrace{\pi h \sum_{k \le n - L} a_{k}}_{\theta_{n}} +
//! 2 \pi h \sum_{k = n - L + 1}^{n} a_{k} \\, q(t - kT).
//! $$
//!
//! The modulator keeps $\theta_{n}$ as an **integer multiple of $\pi / q$ modulo $2 \pi$**,
//! so a signal of millions of symbols has no numerical drift of its phase. The state of
//! the trellis is $(\theta_{n}, a_{n-1}, \dots, a_{n-L+1})$: $2q$ terminal phases
//! (only $q$ of them are visited when $p$ is even) times $M^{L-1}$ past symbols. The
//! signal starts in the state $\theta_{0} = 0$ with the past symbols equal to $+1$.
//!
//! # Receivers
//!
//! - [`Cpm::ml_detect`]: the coherent maximum-likelihood sequence detector, the Viterbi
//!   algorithm over the trellis above with the correlation
//!   $\operatorname{Re} \int_{nT}^{(n+1)T} r(t) \\, s^{\ast}(t; \text{branch}) \\, dt$ as
//!   branch metric (a constant envelope makes it equivalent to the Euclidean distance;
//!   Proakis & Salehi, ch. 4, "Optimum receiver for CPM signals");
//! - [`Cpm::msk_detect`]: the quadrature (OQPSK-type) detector of MSK, which decides the
//!   in-phase component at even symbol boundaries and the quadrature component at odd ones
//!   after a half-sinusoid matched filter (Proakis & Salehi, §3.3-2, MSK as OQPSK with
//!   sinusoidal pulses). It decides the **boundary signs** $d_{n}$ with
//!   $e^{j \theta_{n}} = j^{n} d_{n}$, which a differential precoder ([`msk_precode`])
//!   maps to the CPM symbols, $a_{n} = d_{n} d_{n+1}$; the bit error rate of the precoded
//!   bits is that of antipodal signalling, $Q(\sqrt{2 E_{b}/N_{0}})$, that of the raw CPM
//!   symbols twice as much;
//! - [`Cpm::differential_detect`]: the one-symbol differential phase detector of binary
//!   CPM (Simon & Wang, 1984), $\hat{a}_{n} = \operatorname{sign} \operatorname{Im}
//!   (r_{n} r_{n-1}^{\ast})$ on samples spaced by $T$, after an optional Gaussian
//!   predetection filter.
//!
//! # Closed forms
//!
//! - [`msk_psd`]: the power spectral density of MSK (Proakis & Salehi, §3.4-3),
//!   $S(f) = \frac{16 T}{\pi^{2}} \left[\frac{\cos 2 \pi f T}{1 - 16 f^{2} T^{2}}\right]^{2}$,
//!   whose first nulls are at $f T = 3/4$ and whose two-sided 3 dB bandwidth is $0.59/T$;
//! - [`gmsk_pulse`] and [`gmsk_phase`]: the Gaussian frequency pulse and its integral in
//!   closed form, $\int Q(x) \\, dx = x Q(x) - \phi(x)$ with $\phi$ the normal density;
//! - [`Cpm::min_distance2`]: the normalised minimum squared Euclidean distance over the
//!   merges of a few symbols, $d^{2} = \frac{\log_{2} M}{T} \int [1 - \cos \Delta
//!   \varphi(t)] \\, dt$ (Anderson, Aulin & Sundberg, 1986, ch. 3), 2 for MSK.
//!
//! # Examples
//! ```
//! use com_waveform::cpm::{Cpm, CpmConfig, FrequencyPulse, msk_precode};
//! let msk = CpmConfig::msk(8).build()?;
//! let bits = [1u8, 0, 0, 1, 1, 1, 0, 1, 0, 0];
//! let a = msk_precode(&bits);
//! let s = msk.modulate(&a)?;
//! assert_eq!(s.len(), 80);
//! assert!(s.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));        // constant envelope
//! assert_eq!(msk.ml_detect(&s)?, a);                                 // coherent ML
//! assert_eq!(msk.msk_detect(&s)?, bits);                             // quadrature detector
//! assert_eq!(msk.differential_detect(&s, None)?, a);                 // differential detector
//! let gmsk = CpmConfig::gmsk(0.3, 3, 8).build()?;
//! assert_eq!(gmsk.config().pulse, FrequencyPulse::Gaussian { bt: 0.3 });
//! assert_eq!(gmsk.ml_detect(&gmsk.modulate(&a)?)?, a);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008,
//!    §3.3-2 (CPFSK, MSK), §3.3-3 (CPM, Table 3.3-1), §3.4-3 (spectrum of MSK) and ch. 4
//!    (optimum receiver for CPM, minimum distance).
//! 2. K. Murota and K. Hirade, "GMSK modulation for digital mobile radio telephony,"
//!    *IEEE Trans. Commun.*, 29(7), 1044–1050, 1981.
//! 3. J. B. Anderson, T. Aulin and C.-E. Sundberg, *Digital Phase Modulation*, Plenum,
//!    1986, ch. 2–3.
//! 4. T. Aulin and C.-E. Sundberg, "Continuous phase modulation—Part I: Full response
//!    signaling," *IEEE Trans. Commun.*, 29(3), 196–209, 1981.
//! 5. M. K. Simon and C. C. Wang, "Differential detection of Gaussian MSK in a mobile
//!    radio environment," *IEEE Trans. Veh. Technol.*, 33(4), 307–320, 1984.

mod detection;
mod pulse;

pub use detection::{MlPath, msk_bits, msk_precode};
pub use pulse::{FrequencyPulse, gmsk_phase, gmsk_pulse, msk_psd};

use com_core::{Error, Result, ensure};
use com_math::{C64, cis};
use std::f64::consts::{PI, TAU};

/// A rational modulation index $h = p/q$ ([`CpmConfig::h_num`], [`CpmConfig::h_den`]),
/// read from the fraction `"p/q"`.
///
/// # Examples
/// ```
/// use com_waveform::cpm::ModulationIndex;
/// let h: ModulationIndex = " 1 / 2".parse()?;
/// assert_eq!((h.num, h.den), (1, 2));
/// # Ok::<(), com_core::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModulationIndex {
    /// The numerator $p$.
    pub num: u32,
    /// The denominator $q$.
    pub den: u32,
}

impl std::str::FromStr for ModulationIndex {
    type Err = Error;

    /// `"p/q"`, two non-negative integers (spaces around them are ignored); the values
    /// are checked by [`CpmConfig::build`].
    ///
    /// # Errors
    /// [`Error::Config`] for another text.
    fn from_str(s: &str) -> Result<ModulationIndex> {
        s.split_once('/')
            .and_then(|(p, q)| Some(ModulationIndex { num: p.trim().parse().ok()?, den: q.trim().parse().ok()? }))
            .ok_or_else(|| Error::Config(format!("h must be a fraction p/q, got {s:?}")))
    }
}

/// The parameters of a CPM signal, validated by [`CpmConfig::build`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpmConfig {
    /// The alphabet size $M$ (2 or 4).
    pub order: usize,
    /// The numerator $p$ of the modulation index $h = p/q$.
    pub h_num: u32,
    /// The denominator $q$ of the modulation index.
    pub h_den: u32,
    /// The frequency pulse.
    pub pulse: FrequencyPulse,
    /// The pulse length $L$ in symbols.
    pub length: usize,
    /// Samples per symbol.
    pub sps: usize,
}

impl CpmConfig {
    /// MSK: binary 1REC with $h = 1/2$.
    pub fn msk(sps: usize) -> CpmConfig {
        CpmConfig { order: 2, h_num: 1, h_den: 2, pulse: FrequencyPulse::Rectangular, length: 1, sps }
    }

    /// GMSK: binary Gaussian pulse of bandwidth `bt` over `length` symbols, $h = 1/2$.
    pub fn gmsk(bt: f64, length: usize, sps: usize) -> CpmConfig {
        CpmConfig { order: 2, h_num: 1, h_den: 2, pulse: FrequencyPulse::Gaussian { bt }, length, sps }
    }

    /// Build the modulator: tabulate the phase response and the trellis.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless the order is 2 or 4, $0 < h < 1$ with $q \le 64$, the
    /// length is between 1 and 8, `sps` is at least 2, and $BT > 0$ for a Gaussian pulse.
    pub fn build(self) -> Result<Cpm> {
        ensure(self.order == 2 || self.order == 4, || format!("the order must be 2 or 4, got {}", self.order))?;
        ensure(self.h_num >= 1 && self.h_den >= 2 && self.h_num < self.h_den && self.h_den <= 64, || {
            format!("the modulation index must be p/q with 0 < p < q ≤ 64, got {}/{}", self.h_num, self.h_den)
        })?;
        ensure((1..=8).contains(&self.length), || {
            format!("the pulse length must be 1 to 8 symbols, got {}", self.length)
        })?;
        ensure(self.sps >= 2, || format!("at least 2 samples per symbol, got {}", self.sps))?;
        if let FrequencyPulse::Gaussian { bt } = self.pulse {
            ensure(bt > 0.0 && bt.is_finite(), || format!("BT must be positive, got {bt}"))?;
        }
        let g = com_math::gcd(self.h_num as usize, self.h_den as usize) as u32;
        let cfg = CpmConfig { h_num: self.h_num / g, h_den: self.h_den / g, ..self };
        let (l, sps) = (cfg.length, cfg.sps);
        // q(t) at every sample of the L symbol intervals: index j·sps + i for the symbol
        // j intervals in the past at the sample i
        let q_table: Vec<f64> = (0..l * sps).map(|k| cfg.pulse.phase_response(k as f64 / sps as f64, l)).collect();
        let past_combos = cfg.order.pow((l - 1) as u32);
        Ok(Cpm { cfg, q_table, past_combos })
    }
}

/// A CPM modulator and its detectors (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct Cpm {
    cfg: CpmConfig,
    /// $q(t)$ at the sample `j·sps + i`: the response of the symbol `j` intervals in the
    /// past at the sample `i` of the current interval.
    q_table: Vec<f64>,
    /// $M^{L-1}$, the number of combinations of the past symbols of a state.
    past_combos: usize,
}

impl Cpm {
    /// The validated parameters (the modulation index in lowest terms).
    pub fn config(&self) -> &CpmConfig {
        &self.cfg
    }

    /// The modulation index $h = p/q$.
    pub fn h(&self) -> f64 {
        f64::from(self.cfg.h_num) / f64::from(self.cfg.h_den)
    }

    /// The number of terminal phases, $2q$ (the trellis visits $q$ of them when $p$ is
    /// even).
    pub fn n_phase_states(&self) -> usize {
        2 * self.cfg.h_den as usize
    }

    /// The number of trellis states, $2q M^{L-1}$.
    pub fn n_states(&self) -> usize {
        self.n_phase_states() * self.past_combos
    }

    /// The symbol value $a = 2i - (M - 1)$ of the index `i`.
    pub fn symbol_value(&self, index: usize) -> f64 {
        2.0 * index as f64 - (self.cfg.order as f64 - 1.0)
    }

    /// The tabulated phase response, $q(k T / \mathrm{sps})$ for $k < L \cdot \mathrm{sps}$.
    pub fn phase_response(&self) -> &[f64] {
        &self.q_table
    }

    /// The index of the symbol $+1$ (the past symbols of the initial state).
    fn one(&self) -> usize {
        self.cfg.order / 2
    }

    /// The terminal phase after the symbol of index `i` leaves the window.
    fn next_phase(&self, s: usize, i: usize) -> usize {
        let n = self.n_phase_states() as i64;
        let inc = i64::from(self.cfg.h_num) * (2 * i as i64 - (self.cfg.order as i64 - 1));
        ((s as i64 + inc).rem_euclid(n)) as usize
    }

    /// The partial response $2 \pi h \sum_{j=0}^{L-1} a_{n-j} q(t - (n - j)T)$ at the
    /// sample `t` of the interval, for the window `window` (`window[j]` is the index of
    /// the symbol `j` intervals in the past, `window[0]` the current one).
    fn partial_phase(&self, window: &[usize], t: usize) -> f64 {
        let sps = self.cfg.sps;
        let sum: f64 = window.iter().enumerate().map(|(j, &i)| self.symbol_value(i) * self.q_table[j * sps + t]).sum();
        TAU * self.h() * sum
    }

    /// The phase of the signal at every sample, with the terminal phase reduced modulo
    /// $2 \pi$ (the phase is continuous across the intervals up to these $2 \pi$ jumps).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if a symbol index is not below $M$.
    pub fn phase(&self, symbols: &[usize]) -> Result<Vec<f64>> {
        let m = self.cfg.order;
        ensure(symbols.iter().all(|&i| i < m), || format!("symbol indices must be below {m}"))?;
        let (l, sps) = (self.cfg.length, self.cfg.sps);
        let mut window = vec![self.one(); l];
        let mut state = 0usize;
        let unit = PI / f64::from(self.cfg.h_den);
        let mut out = Vec::with_capacity(symbols.len() * sps);
        for &i in symbols {
            window.rotate_right(1);
            window[0] = i;
            let theta = state as f64 * unit;
            for t in 0..sps {
                out.push(theta + self.partial_phase(&window, t));
            }
            state = self.next_phase(state, window[l - 1]);
        }
        Ok(out)
    }

    /// The complex baseband signal $e^{j \varphi(t)}$ at `sps` samples per symbol.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if a symbol index is not below $M$.
    pub fn modulate(&self, symbols: &[usize]) -> Result<Vec<C64>> {
        Ok(self.phase(symbols)?.into_iter().map(cis).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn the_modulation_index_is_read_from_a_fraction() {
        assert_eq!("3/8".parse::<ModulationIndex>().unwrap(), ModulationIndex { num: 3, den: 8 });
        for bad in ["0.5", "1/", "a/2", "-1/2", "1/2/3"] {
            assert!(bad.parse::<ModulationIndex>().is_err(), "{bad}");
        }
    }

    fn random_symbols(m: usize, n: usize, rng: &mut Rng) -> Vec<usize> {
        (0..n).map(|_| rng.below(m)).collect()
    }

    fn count_errors<T: PartialEq>(a: &[T], b: &[T]) -> usize {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).filter(|(x, y)| x != y).count()
    }

    #[test]
    fn config_is_validated_and_reduced() {
        assert!(CpmConfig { order: 8, ..CpmConfig::msk(4) }.build().is_err());
        assert!(CpmConfig { h_num: 2, h_den: 2, ..CpmConfig::msk(4) }.build().is_err());
        assert!(CpmConfig { sps: 1, ..CpmConfig::msk(4) }.build().is_err());
        assert!(CpmConfig { length: 0, ..CpmConfig::msk(4) }.build().is_err());
        assert!(CpmConfig::gmsk(0.0, 3, 4).build().is_err());
        let c = CpmConfig { h_num: 2, h_den: 8, ..CpmConfig::msk(4) }.build().unwrap();
        assert_eq!((c.config().h_num, c.config().h_den), (1, 4));
        assert_eq!(c.n_phase_states(), 8);
        let g = CpmConfig { order: 4, length: 3, ..CpmConfig::gmsk(0.3, 3, 4) }.build().unwrap();
        assert_eq!(g.n_states(), 4 * 16);
        assert!(g.ml_detect(&[C64::new(1.0, 0.0); 5]).is_err());
        assert!(g.msk_detect(&[C64::new(1.0, 0.0); 8]).is_err());
        assert!(g.differential_detect(&[C64::new(1.0, 0.0); 8], None).is_err());
        assert!(c.modulate(&[2]).is_err());
    }

    #[test]
    fn edge_cases_return_instead_of_panicking() {
        // one symbol with a pulse of three symbols: the reference sample is the last one
        let g = CpmConfig::gmsk(0.3, 3, 4).build().unwrap();
        assert_eq!(g.differential_detect(&[C64::new(1.0, 0.0); 4], None).unwrap().len(), 1);
        assert_eq!(CpmConfig::msk(4).build().unwrap().min_distance2(0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn constant_envelope_and_continuous_phase() {
        // the instantaneous frequency is 2πh Σ_k a_k g(t - kT), and the shifted pulses
        // sum to Σ_j g(t + jT) = 1/(2T) (LREC and LRC exactly, the truncated Gaussian
        // pulse up to the scaling of its missing tails, below 0.1 % at BT = 0.3 and
        // L = 4), so the phase increment per sample is at most πh (M-1) / sps
        let mut rng = Rng::new(3);
        let sps = 16;
        for cfg in [
            CpmConfig::msk(sps),
            CpmConfig { order: 4, h_num: 1, h_den: 4, pulse: FrequencyPulse::RaisedCosine, length: 3, sps },
            CpmConfig { order: 4, h_num: 3, h_den: 5, pulse: FrequencyPulse::Rectangular, length: 2, sps },
            CpmConfig::gmsk(0.3, 4, sps),
        ] {
            let cpm = cfg.build().unwrap();
            let a = random_symbols(cfg.order, 400, &mut rng);
            let s = cpm.modulate(&a).unwrap();
            assert!(s.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
            let bound = PI * cpm.h() * (cfg.order as f64 - 1.0) / sps as f64;
            let max_jump = s.windows(2).map(|w| (w[1] * w[0].conj()).arg().abs()).fold(0.0, f64::max);
            assert!(max_jump <= bound * (1.0 + 1e-3), "{cfg:?}: jump {max_jump} above {bound}");
        }
    }

    #[test]
    fn the_terminal_phase_never_drifts() {
        // one million MSK symbols: the terminal phase is an integer state, so the phase
        // at every boundary is exactly a multiple of π/2
        let msk = CpmConfig::msk(2).build().unwrap();
        let a = random_symbols(2, 1_000_000, &mut Rng::new(5));
        let phi = msk.phase(&a).unwrap();
        let worst =
            phi.iter().step_by(2).map(|p| (p / (PI / 2.0) - (p / (PI / 2.0)).round()).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-12, "{worst}");
        assert!(phi.iter().all(|p| *p > -PI && *p < TAU + PI));
    }

    #[test]
    fn ml_detector_recovers_noiseless_sequences() {
        let mut rng = Rng::new(11);
        for cfg in [
            CpmConfig::msk(4),
            CpmConfig { order: 2, h_num: 1, h_den: 4, pulse: FrequencyPulse::RaisedCosine, length: 2, sps: 4 },
            CpmConfig { order: 4, h_num: 1, h_den: 2, pulse: FrequencyPulse::Rectangular, length: 1, sps: 4 },
            CpmConfig { order: 4, h_num: 2, h_den: 3, pulse: FrequencyPulse::RaisedCosine, length: 3, sps: 4 },
            CpmConfig::gmsk(0.25, 4, 4),
        ] {
            let cpm = cfg.build().unwrap();
            let a = random_symbols(cfg.order, 300, &mut rng);
            let s = cpm.modulate(&a).unwrap();
            let path = cpm.ml_detect_path(&s).unwrap();
            assert_eq!(path.symbols, a, "{cfg:?}");
            // the phase states of the path are those of the modulator
            let unit = PI / f64::from(cpm.config().h_den);
            let phi = cpm.phase(&a).unwrap();
            for n in 0..a.len() {
                let theta = phi[n * cfg.sps] - cpm.partial_phase(&window_at(&cpm, &a, n), 0);
                let want = ((theta / unit).round() as i64).rem_euclid(cpm.n_phase_states() as i64) as usize;
                assert_eq!(path.phase_states[n], want, "{cfg:?} at {n}");
            }
        }
    }

    /// The window of the modulator at the interval `n` (`[a_n, a_{n-1}, …]`, +1 before
    /// the start).
    fn window_at(cpm: &Cpm, a: &[usize], n: usize) -> Vec<usize> {
        (0..cpm.config().length).map(|j| if j > n { cpm.one() } else { a[n - j] }).collect()
    }

    #[test]
    fn msk_precoder_and_quadrature_detector_round_trip() {
        let msk = CpmConfig::msk(8).build().unwrap();
        let bits: Vec<u8> = (0..500).map(|_| 0).chain(com_mapping::source::bits(500, &mut Rng::new(2))).collect();
        let a = msk_precode(&bits);
        let s = msk.modulate(&a).unwrap();
        assert_eq!(msk.msk_detect(&s).unwrap(), bits);
        assert_eq!(msk_bits(&msk.ml_detect_path(&s).unwrap().phase_states), bits);
        // a constant phase rotation of a quarter turn swaps the components: the detector
        // is coherent (sanity of the test, not a property of MSK)
        let rotated: Vec<C64> = s.iter().map(|v| v * C64::new(0.0, 1.0)).collect();
        assert!(count_errors(&msk.msk_detect(&rotated).unwrap(), &bits) > 100);
        // the differential detector ignores the rotation
        assert_eq!(msk.differential_detect(&rotated, None).unwrap(), a);
    }

    #[test]
    fn gaussian_phase_response_is_normalised() {
        for bt in [0.2, 0.3, 0.5] {
            for l in [2usize, 3, 4, 5] {
                let p = FrequencyPulse::Gaussian { bt };
                assert_eq!(p.phase_response(l as f64, l), 0.5);
                assert_eq!(p.phase_response(0.0, l), 0.0);
                assert!((p.phase_response(l as f64 / 2.0, l) - 0.25).abs() < 1e-12);
                // monotone
                let v: Vec<f64> = (0..=100).map(|k| p.phase_response(l as f64 * k as f64 / 100.0, l)).collect();
                assert!(v.windows(2).all(|w| w[1] >= w[0]));
            }
        }
        // the closed-form integral of Q matches a numerical one
        let bt = 0.3;
        let (lo, hi) = (-3.0, 1.2);
        let num = com_math::simpson(|t| gmsk_pulse(t, bt), lo, hi, 2000);
        assert!((num - (gmsk_phase(hi, bt) - gmsk_phase(lo, bt))).abs() < 1e-10);
    }

    #[test]
    fn parse_pulses() {
        assert_eq!(FrequencyPulse::from_name("REC", 0.0).unwrap(), FrequencyPulse::Rectangular);
        assert_eq!(FrequencyPulse::from_name("rc", 0.0).unwrap(), FrequencyPulse::RaisedCosine);
        assert_eq!(FrequencyPulse::from_name("gmsk", 0.5).unwrap(), FrequencyPulse::Gaussian { bt: 0.5 });
        assert!(FrequencyPulse::from_name("sinc", 0.5).is_err());
    }

    /// Averaged Hann-windowed periodogram of `x` over non-overlapping segments of `nfft`
    /// samples sampled at `sps` per symbol ([`com_dsp::spectrum::welch`]): frequencies
    /// `fT` (centred) and the density per unit `fT`, normalised to a unit total power.
    pub(super) fn welch(x: &[C64], sps: usize, nfft: usize) -> (Vec<f64>, Vec<f64>) {
        let hann = com_dsp::window::hann(nfft);
        let (f, d) = com_dsp::spectrum::welch(x, sps as f64, nfft, Some(&hann), nfft).unwrap();
        let total: f64 = d.iter().sum::<f64>() * (f[1] - f[0]);
        (f, d.into_iter().map(|v| v / total).collect())
    }

    /// The two-sided bandwidth holding the `fraction` of the power of a centred density.
    pub(super) fn power_bandwidth(freqs: &[f64], dens: &[f64], fraction: f64) -> f64 {
        let df = freqs[1] - freqs[0];
        // the bins by distance to the carrier, added as the band widens by `df`
        let mut bins: Vec<(f64, f64)> = freqs.iter().zip(dens).map(|(f, d)| (f.abs(), d * df)).collect();
        bins.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut inside, mut next, mut b) = (0.0, 0, 0.0);
        while inside < fraction && next < bins.len() {
            b += df;
            while next < bins.len() && bins[next].0 <= b / 2.0 {
                inside += bins[next].1;
                next += 1;
            }
        }
        b
    }
}

#[cfg(test)]
mod literature {
    use super::tests::{power_bandwidth, welch};
    use super::*;
    use com_core::rng::Rng;
    use com_math::{db2lin, qfunc};

    fn noisy(x: &[C64], sps: usize, ebn0_db: f64, rng: &mut Rng) -> Vec<C64> {
        let sd = (sps as f64 / db2lin(ebn0_db)).sqrt();
        x.iter().map(|v| v + rng.complex_gaussian() * sd).collect()
    }

    /// J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., 2008, §3.3-2:
    /// MSK is OQPSK with half-sinusoid pulses, so coherent detection of the precoded
    /// (OQPSK) bits has the error probability `Q(√(2 E_b/N_0))` of antipodal signalling,
    /// with the quadrature detector as with the maximum-likelihood sequence detector; the
    /// raw CPM symbols `a_n = d_n d_{n+1}` are then wrong with probability `2p(1 - p)`.
    ///
    /// 200 000 bits: at 4 dB (`p = 1.25e-2`) about 2 500 errors are expected, at 6 dB
    /// (`p = 2.39e-3`) about 477, so the relative standard errors are 2 % and 4.6 %; the
    /// test accepts three standard errors.
    #[test]
    fn proakis_msk_as_oqpsk_coherent_ber() {
        let sps = 8;
        let msk = CpmConfig::msk(sps).build().unwrap();
        let mut rng = Rng::new(2024);
        let n_bits = 200_000;
        for ebn0_db in [4.0, 6.0] {
            let p = qfunc((2.0 * db2lin(ebn0_db)).sqrt());
            let bits = com_mapping::source::bits(n_bits, &mut rng);
            let a = msk_precode(&bits);
            let r = noisy(&msk.modulate(&a).unwrap(), sps, ebn0_db, &mut rng);
            let tol = 3.0 / (p * n_bits as f64).sqrt();
            let quad = msk.msk_detect(&r).unwrap();
            let ber_quad = quad.iter().zip(&bits).filter(|(x, y)| x != y).count() as f64 / n_bits as f64;
            assert!((ber_quad / p - 1.0).abs() < tol, "quadrature at {ebn0_db} dB: {ber_quad:e} vs {p:e}");
            let path = msk.ml_detect_path(&r).unwrap();
            let ml = msk_bits(&path.phase_states);
            let ber_ml = ml.iter().zip(&bits).filter(|(x, y)| x != y).count() as f64 / n_bits as f64;
            assert!((ber_ml / p - 1.0).abs() < tol, "ML at {ebn0_db} dB: {ber_ml:e} vs {p:e}");
            let raw = path.symbols.iter().zip(&a).filter(|(x, y)| x != y).count() as f64 / n_bits as f64;
            let want = 2.0 * p * (1.0 - p);
            assert!((raw / want - 1.0).abs() < tol, "raw symbols at {ebn0_db} dB: {raw:e} vs {want:e}");
        }
    }

    /// Proakis & Salehi, 5th ed., §3.4-3 (the power spectrum of MSK): the closed form
    /// `S(f) = (16T/π²)[cos 2πfT / (1 - 16 f² T²)]²` has its first nulls at `fT = 3/4`
    /// and its two-sided 3 dB bandwidth at `0.59/T` (`S(0.295)/S(0) = 0.50`). The
    /// averaged periodogram of a simulated signal (Hann window, segments of 64 symbols,
    /// 400 segments, so about 5 % of standard deviation per bin, 0.2 dB) matches the
    /// closed form within 1 dB on every bin and 0.3 dB in RMS down to -20 dB, its minimum
    /// near `fT = 3/4` is 25 dB below the peak (the window fills the null), and the
    /// 3 dB bandwidth interpolated between the bins is `0.59/T ± 0.03/T` (the slope of
    /// the spectrum there turns 5 % of density into 0.017/T of bandwidth).
    #[test]
    fn proakis_msk_power_spectrum() {
        let sps = 8;
        let msk = CpmConfig::msk(sps).build().unwrap();
        let mut rng = Rng::new(77);
        let n_seg = 400;
        let nfft = 64 * sps;
        let a: Vec<usize> = (0..n_seg * 64).map(|_| rng.below(2)).collect();
        let s = msk.modulate(&a).unwrap();
        let (f, d) = welch(&s, sps, nfft);
        // the closed form integrates to one over all frequencies
        let n = 400_000;
        let integral: f64 = (0..n).map(|i| msk_psd(-20.0 + 40.0 * i as f64 / n as f64) * 40.0 / n as f64).sum();
        assert!((integral - 1.0).abs() < 1e-3, "{integral}");
        let peak = d[nfft / 2];
        let (mut sq, mut count) = (0.0, 0usize);
        for (fi, di) in f.iter().zip(&d) {
            let want = msk_psd(*fi);
            if want > 1e-2 * msk_psd(0.0) {
                let err_db = com_math::lin2db(di / want);
                assert!(err_db.abs() < 1.0, "fT = {fi}: {err_db} dB");
                sq += err_db * err_db;
                count += 1;
            }
        }
        assert!((sq / count as f64).sqrt() < 0.3, "RMS error {} dB", (sq / count as f64).sqrt());
        let (null_f, null_d) = f
            .iter()
            .zip(&d)
            .filter(|(fi, _)| **fi > 0.6 && **fi < 0.9)
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(fi, di)| (*fi, *di))
            .unwrap();
        assert!((null_f - 0.75).abs() < 0.02, "null at {null_f}");
        assert!(com_math::lin2db(null_d / peak) < -25.0, "null depth {}", com_math::lin2db(null_d / peak));
        let i0 = nfft / 2;
        let k = (i0..nfft - 1).find(|&i| d[i + 1] < peak / 2.0).unwrap();
        let half = f[k] + (f[k + 1] - f[k]) * (d[k] - peak / 2.0) / (d[k] - d[k + 1]);
        assert!((2.0 * half - 0.59).abs() < 0.03, "3 dB bandwidth {}", 2.0 * half);
        let x = 0.295;
        assert!((msk_psd(x) / msk_psd(0.0) - 0.5).abs() < 0.01);
    }

    /// K. Murota and K. Hirade, "GMSK modulation for digital mobile radio telephony,"
    /// IEEE Trans. Commun. 29(7), 1981: §II gives the Gaussian frequency pulse (area
    /// 1/2); Table I gives the two-sided bandwidths (`× 1/T`) holding 90, 99, 99.9 and
    /// 99.99 % of the power, for `BT = 0.2`, 0.25, 0.5 and MSK. The spectra of GMSK are
    /// estimated (pulse truncated to 5 symbols, Hann window, segments of 128 symbols, 400
    /// segments): the whole table within 3 %. MSK uses its closed form [`msk_psd`]: 0.777,
    /// 1.182 and 2.735 against 0.78, 1.20 and 2.76 (within 2 %); its 99.99 % bandwidth is
    /// 5.67, where the table prints 6.00 (6 % more; the estimated spectrum cannot settle
    /// it, its window leaks into the tails in `1/f^4`). `BT = 0.3`, absent from the
    /// table, falls between `BT = 0.25` and `BT = 0.5`.
    #[test]
    fn murota_hirade_1981_gmsk_bandwidths() {
        let sps = 8;
        let nfft = 128 * sps;
        let n_seg = 400;
        let fractions = [0.90, 0.99, 0.999, 0.9999];
        let bandwidths = |cfg: CpmConfig, seed: u64| {
            let cpm = cfg.build().unwrap();
            let mut rng = Rng::new(seed);
            let a: Vec<usize> = (0..n_seg * 128).map(|_| rng.below(2)).collect();
            let (f, d) = welch(&cpm.modulate(&a).unwrap(), sps, nfft);
            fractions.map(|q| power_bandwidth(&f, &d, q))
        };
        // Table I: BT, then 90, 99, 99.9 and 99.99 %
        let table_1 =
            [(0.2, [0.52, 0.79, 0.99, 1.22]), (0.25, [0.57, 0.86, 1.09, 1.37]), (0.5, [0.69, 1.04, 1.33, 2.08])];
        let mut b99 = Vec::new();
        for (seed, (bt, want)) in table_1.into_iter().enumerate() {
            let got = bandwidths(CpmConfig::gmsk(bt, 5, sps), seed as u64 + 1);
            for ((q, g), w) in fractions.iter().zip(got).zip(want) {
                assert!((g / w - 1.0).abs() < 0.03, "BT = {bt}, {q}: {g} vs {w}");
            }
            b99.push(got[1]);
        }
        let b03 = bandwidths(CpmConfig::gmsk(0.3, 5, sps), 9)[1];
        assert!(b03 > b99[1] && b03 < b99[2], "BT = 0.3: {b03} between {} and {}", b99[1], b99[2]);
        // MSK, from its closed form
        let df = 1e-4;
        let f: Vec<f64> = (-400_000..400_000).map(|i| i as f64 * df).collect();
        let d: Vec<f64> = f.iter().map(|v| msk_psd(*v)).collect();
        for (q, w) in [(0.90, 0.78), (0.99, 1.20), (0.999, 2.76)] {
            let g = power_bandwidth(&f, &d, q);
            assert!((g / w - 1.0).abs() < 0.02, "MSK, {q}: {g} vs {w}");
        }
        let g = power_bandwidth(&f, &d, 0.9999);
        assert!((g - 5.67).abs() < 0.03, "MSK, 99.99 %: {g} (printed 6.00)");
    }

    /// K. Murota and K. Hirade (1981), Fig. 5: the minimum distance of GMSK (normalised
    /// so that MSK has 2) against `BT`, read on the figure within 0.02: 1.53 at
    /// `BT = 0.2`, 1.89 at 0.4, 1.96 at 0.6 and 1.99 at 0.8.
    #[test]
    fn murota_hirade_1981_fig_5_minimum_distance() {
        for (bt, want) in [(0.2, 1.53), (0.4, 1.89), (0.6, 1.96), (0.8, 1.99)] {
            let got = CpmConfig::gmsk(bt, 5, 8).build().unwrap().min_distance2(3).unwrap();
            assert!((got - want).abs() < 0.02, "BT = {bt}: {got} vs {want}");
        }
    }

    /// M. K. Simon and C. C. Wang, "Differential detection of Gaussian MSK in a mobile
    /// radio environment," IEEE Trans. Veh. Technol. 33(4), 1984, Section II and III:
    /// one-symbol differential detection decides on the phase increment over a symbol,
    /// which the partial response of GMSK reduces for some patterns (with `BT = 0.3`
    /// the worst increment is 0.48 of the `π/2` of MSK), so the error rate over AWGN
    /// degrades as `BT` decreases, without an error floor, and the predetection filter
    /// trades noise against its own intersymbol interference. The test checks the
    /// ordering at `E_b/N_0 = 10 dB` with a Gaussian low-pass of 3 dB bandwidth `0.5/T`
    /// (the best found for MSK among 0.25 to 1; a band-pass bandwidth of `1/T` in the
    /// convention of the paper), and plausible ranges: MSK above the coherent
    /// `Q(√(2 E_b/N_0)) = 3.9e-6` and below 1e-2 (measured 1.3e-3: the filter is not a
    /// matched filter and its ISI closes the eye, against the ideal noncoherent DBPSK
    /// rate `½ e^{-E_b/N_0} = 2.3e-5`); GMSK `BT = 0.5` above MSK (measured 7.7e-3),
    /// `BT = 0.3` above `BT = 0.5` and below 0.1 (measured 4.1e-2). The published curves
    /// of the paper are not reproduced here: they use a different filter.
    ///
    /// 200 000 bits: about 260 errors are expected for MSK, so the ordering is not a
    /// fluctuation (the standard error of its rate is 6 %).
    #[test]
    fn simon_wang_1984_differential_detection_degrades_with_bt() {
        let sps = 8;
        let n_bits = 200_000;
        let ebn0_db = 10.0;
        let ber = |cfg: CpmConfig, seed: u64| {
            let cpm = cfg.build().unwrap();
            let mut rng = Rng::new(seed);
            let a: Vec<usize> = (0..n_bits).map(|_| rng.below(2)).collect();
            let r = noisy(&cpm.modulate(&a).unwrap(), sps, ebn0_db, &mut rng);
            let hat = cpm.differential_detect(&r, Some(0.5)).unwrap();
            hat.iter().zip(&a).filter(|(x, y)| x != y).count() as f64 / n_bits as f64
        };
        let msk = ber(CpmConfig::msk(sps), 10);
        let gmsk05 = ber(CpmConfig::gmsk(0.5, 4, sps), 11);
        let gmsk03 = ber(CpmConfig::gmsk(0.3, 4, sps), 12);
        let coherent = qfunc((2.0 * db2lin(ebn0_db)).sqrt());
        assert!(msk > coherent && msk < 1e-2, "MSK differential: {msk:e}");
        assert!(
            gmsk05 > msk && gmsk03 > gmsk05 && gmsk03 < 0.1,
            "GMSK: {gmsk03:e} (0.3), {gmsk05:e} (0.5), MSK {msk:e}"
        );
    }

    /// T. Aulin and C.-E. W. Sundberg, "Continuous phase modulation—Part I: Full response
    /// signaling," IEEE Trans. Commun. 29(3), 196–209, 1981, Table I: the upper bound
    /// $d_{B}^{2}$ of M-ary CPFSK reaches its maximum 2.434 at `h = 0.715` for `M = 2` and
    /// 4.232 at `h = 0.914` for `M = 4` (here at `h = 5/7` and `32/35`, within 1e-3); with
    /// `h = 1/2` binary CPFSK is MSK, `d² = 2`, and the gain of `h = 0.715` is 0.85 dB.
    #[test]
    fn aulin_sundberg_1981_table_i_optimum_cpfsk() {
        let bound = |order: usize, h_num: u32, h_den: u32| {
            CpmConfig { order, h_num, h_den, ..CpmConfig::msk(8) }.build().unwrap().min_distance2(2).unwrap()
        };
        let (b2, b4) = (bound(2, 5, 7), bound(4, 32, 35));
        assert!((b2 - 2.434).abs() < 1e-3 && (b4 - 4.232).abs() < 1e-3, "{b2} and {b4}");
        assert!((com_math::lin2db(b2 / bound(2, 1, 2)) - 0.85).abs() < 0.01);
    }

    /// C.-E. Sundberg, "Continuous phase modulation," IEEE Commun. Mag. 24(4), 25–38,
    /// 1986, Fig. 5: the upper bound $d_{B}^{2}$ of binary LRC against $h$, read on the
    /// figure (within 0.05): at `h = 1/2`, 2 for 1RC (the MSK point, where 2RC lies just
    /// below) and the dots 1.78, 1.53, 1.32 and 1.15 of 3RC to 6RC; the peaks 2.11 of 1RC
    /// near `h = 0.6`, 2.66 of 2RC near `h = 0.8` and 3.52 of 3RC near `h = 0.96`.
    #[test]
    fn sundberg_1986_fig_5_distance_of_binary_lrc() {
        let d2 = |length: usize, h_num: u32, h_den: u32| {
            let cfg = CpmConfig { h_num, h_den, pulse: FrequencyPulse::RaisedCosine, length, ..CpmConfig::msk(8) };
            cfg.build().unwrap().min_distance2(3).unwrap()
        };
        let read = [
            (1, 1, 2, 2.0),
            (3, 1, 2, 1.78),
            (4, 1, 2, 1.53),
            (5, 1, 2, 1.32),
            (6, 1, 2, 1.15),
            (1, 3, 5, 2.11),
            (2, 4, 5, 2.66),
            (3, 19, 20, 3.52),
        ];
        for (l, p, q, want) in read {
            let got = d2(l, p, q);
            assert!((got - want).abs() < 0.05, "{l}RC, h = {p}/{q}: {got} vs {want}");
        }
        let two = d2(2, 1, 2);
        assert!(two < 2.0 && two > 1.9, "2RC, h = 1/2: {two}");
    }

    /// J. B. Anderson, T. Aulin and C.-E. Sundberg, *Digital Phase Modulation*, Plenum,
    /// 1986, ch. 3 (also Aulin & Sundberg, IEEE Trans. Commun. 29(3), 1981, Section III):
    /// the minimum normalised squared Euclidean distance of binary 1REC CPM with
    /// `h = 1/2` (MSK) is `d²_min = 2`, that of antipodal signalling, reached by the
    /// merge `γ = (+2, -2)`; the upper bound of binary CPFSK from that merge,
    /// `d²_B(h) = 2(1 - sin(2πh)/(2πh))`, gives 0.727 at `h = 1/4` and reaches its
    /// maximum 2.43 near `h = 0.715` (Proakis & Salehi, 5th ed., ch. 4, the distance
    /// bound of CPFSK; the test uses `h = 5/7`, where the bound is 2.434).
    #[test]
    fn anderson_aulin_sundberg_minimum_distance() {
        let msk = CpmConfig::msk(4).build().unwrap();
        assert!((msk.min_distance2(4).unwrap() - 2.0).abs() < 1e-3, "{}", msk.min_distance2(4).unwrap());
        for (p, q) in [(1u32, 4u32), (5, 7)] {
            let h = f64::from(p) / f64::from(q);
            let want = 2.0 * (1.0 - com_math::sinc(2.0 * h));
            let cpm = CpmConfig { h_num: p, h_den: q, ..CpmConfig::msk(4) }.build().unwrap();
            let got = cpm.min_distance2(2).unwrap();
            // Simpson on 256 intervals per symbol: within 1e-9 of the closed form
            assert!((got - want).abs() < 1e-9, "h = {p}/{q}: {got} vs {want}");
        }
        // the ML detector of 4-ary 1REC with h = 1/4 decodes a noiseless sequence
        let cpm = CpmConfig { order: 4, h_num: 1, h_den: 4, ..CpmConfig::msk(4) }.build().unwrap();
        assert_eq!(cpm.n_states(), 8);
        let mut rng = Rng::new(9);
        let a: Vec<usize> = (0..2000).map(|_| rng.below(4)).collect();
        assert_eq!(cpm.ml_detect(&cpm.modulate(&a).unwrap()).unwrap(), a);
    }
}
