//! The receivers of CPM: the maximum-likelihood sequence detector, the quadrature
//! detector of MSK and its precoder, the differential detector, and the minimum
//! distance.

use super::{Cpm, FrequencyPulse};
use com_core::{Result, ensure, ensure_shape};
use com_dsp::filter::{LowPass, fir_filter, low_pass_fir};
use com_math::{C64, cis};
use std::f64::consts::{PI, TAU};

/// The survivor path of the maximum-likelihood sequence detector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlPath {
    /// The symbol indices $i_{n}$ ($a_{n} = 2 i_{n} - (M - 1)$).
    pub symbols: Vec<usize>,
    /// The terminal phase states $\theta_{n}$ in units of $\pi/q$, modulo $2q$, at the
    /// start of every interval and after the last one (`symbols.len() + 1` values).
    pub phase_states: Vec<usize>,
}

impl Cpm {
    /// The number of whole symbols of a received signal.
    fn n_symbols(&self, r: &[C64]) -> Result<usize> {
        let sps = self.cfg.sps;
        ensure_shape(r.len() >= sps && r.len() % sps == 0, || {
            format!("{} samples is not a positive multiple of {sps} samples per symbol", r.len())
        })?;
        Ok(r.len() / sps)
    }

    /// The conjugated partial-response waveforms of every combination of a window
    /// (`combo = i_new + M · past`).
    fn branch_waveforms(&self) -> Vec<Vec<C64>> {
        let (m, l, sps) = (self.cfg.order, self.cfg.length, self.cfg.sps);
        let mut window = vec![0usize; l];
        (0..m * self.past_combos)
            .map(|combo| {
                let mut c = combo;
                for w in &mut window {
                    *w = c % m;
                    c /= m;
                }
                (0..sps).map(|t| cis(-self.partial_phase(&window, t))).collect()
            })
            .collect()
    }

    /// The coherent maximum-likelihood sequence detector (Viterbi algorithm over the
    /// phase trellis): the symbol indices of the most likely sequence, starting from the
    /// initial state of the modulator and ending in any state.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] unless `r` holds a positive whole number of symbols.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::cpm::{CpmConfig, FrequencyPulse};
    /// let cpm = CpmConfig { order: 4, h_num: 1, h_den: 4, pulse: FrequencyPulse::RaisedCosine, length: 2, sps: 4 }.build()?;
    /// let a = vec![0, 3, 1, 2, 2, 0, 1, 3, 3, 0];
    /// assert_eq!(cpm.ml_detect(&cpm.modulate(&a)?)?, a);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn ml_detect(&self, r: &[C64]) -> Result<Vec<usize>> {
        Ok(self.ml_detect_path(r)?.symbols)
    }

    /// The maximum-likelihood sequence with its terminal phase states (see [`MlPath`]).
    ///
    /// # Errors
    /// As [`Cpm::ml_detect`].
    pub fn ml_detect_path(&self, r: &[C64]) -> Result<MlPath> {
        let n_sym = self.n_symbols(r)?;
        let (m, l, sps) = (self.cfg.order, self.cfg.length, self.cfg.sps);
        let n_phase = self.n_phase_states();
        let n_states = self.n_states();
        let pc = self.past_combos;
        let waves = self.branch_waveforms();
        let rot: Vec<C64> = (0..n_phase).map(|s| cis(-PI * s as f64 / f64::from(self.cfg.h_den))).collect();
        // the initial state: phase 0, past symbols +1
        let past0 = (0..l.saturating_sub(1)).fold(0usize, |acc, _| acc * m + self.one());
        let mut metric = vec![f64::NEG_INFINITY; n_states];
        metric[past0] = 0.0;
        let mut next = vec![f64::NEG_INFINITY; n_states];
        let mut prev = vec![0u32; n_sym * n_states];
        let mut sym = vec![0u8; n_sym * n_states];
        let mut corr = vec![C64::new(0.0, 0.0); waves.len()];
        for n in 0..n_sym {
            let seg = &r[n * sps..(n + 1) * sps];
            for (c, w) in corr.iter_mut().zip(&waves) {
                *c = seg.iter().zip(w).map(|(a, b)| a * b).sum();
            }
            next.fill(f64::NEG_INFINITY);
            for state in 0..n_states {
                let acc = metric[state];
                if acc == f64::NEG_INFINITY {
                    continue;
                }
                let (s, past) = (state / pc, state % pc);
                for i in 0..m {
                    let combo = i + m * past;
                    let branch = acc + (corr[combo] * rot[s]).re;
                    let leaving = if l == 1 { i } else { past / m.pow((l - 2) as u32) };
                    let s_next = self.next_phase(s, leaving);
                    let past_next = if l == 1 { 0 } else { (i + m * past) % pc };
                    let to = s_next * pc + past_next;
                    if branch > next[to] {
                        next[to] = branch;
                        prev[n * n_states + to] = state as u32;
                        sym[n * n_states + to] = i as u8;
                    }
                }
            }
            // keep the metrics bounded over long sequences
            let best = next.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for v in &mut next {
                *v -= best;
            }
            std::mem::swap(&mut metric, &mut next);
        }
        let mut state = (0..n_states).max_by(|&a, &b| metric[a].total_cmp(&metric[b])).unwrap_or(0);
        let mut symbols = vec![0usize; n_sym];
        let mut phase_states = vec![0usize; n_sym + 1];
        phase_states[n_sym] = state / pc;
        for n in (0..n_sym).rev() {
            symbols[n] = sym[n * n_states + state] as usize;
            state = prev[n * n_states + state] as usize;
            phase_states[n] = state / pc;
        }
        Ok(MlPath { symbols, phase_states })
    }

    /// Whether the signal is MSK (binary 1REC with $h = 1/2$).
    fn ensure_msk(&self) -> Result<()> {
        let c = &self.cfg;
        ensure(
            c.order == 2 && c.length == 1 && c.pulse == FrequencyPulse::Rectangular && c.h_num == 1 && c.h_den == 2,
            || "the quadrature detector needs MSK (binary 1REC, h = 1/2)".into(),
        )
    }

    /// The quadrature detector of MSK: the boundary bits $d_{n}$, $n = 1 \dots N$
    /// ($d_{n} = +1$ is the bit 1), decided from
    /// $\operatorname{Re}\left[(-j)^{n} \int_{(n-1)T}^{(n+1)T} r(t) \cos \frac{\pi (t - nT)}{2T} \\, dt\right]$
    /// (the last window is truncated to the signal). Over AWGN, each decision has the
    /// error probability $Q(\sqrt{2 E_{b}/N_{0}})$ of antipodal signalling.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless the signal is MSK, [`com_core::Error::Shape`]
    /// unless it holds a whole number of symbols.
    pub fn msk_detect(&self, r: &[C64]) -> Result<Vec<u8>> {
        self.ensure_msk()?;
        let n_sym = self.n_symbols(r)?;
        let sps = self.cfg.sps;
        let (mut rotate, step) = (C64::new(1.0, 0.0), C64::new(0.0, -1.0));
        let mut bits = Vec::with_capacity(n_sym);
        for n in 1..=n_sym {
            rotate *= step;
            let centre = n * sps;
            let lo = centre - sps;
            let hi = (centre + sps).min(r.len());
            let z: C64 = (lo..hi).map(|t| r[t] * (PI * (t as f64 - centre as f64) / (2.0 * sps as f64)).cos()).sum();
            bits.push(u8::from((z * rotate).re > 0.0));
        }
        Ok(bits)
    }

    /// Whether the signal is binary (the differential detector).
    fn ensure_binary(&self) -> Result<()> {
        ensure(self.cfg.order == 2, || "the differential detector needs a binary signal".into())
    }

    /// The one-symbol differential phase detector of binary CPM: after an optional
    /// low-pass predetection filter of normalised 3 dB bandwidth `rx_bt` (`None` for no
    /// filter; the band-pass bandwidth $B_{i}$ of Simon & Wang, 1984, is twice this
    /// value, and about $0.5/T$ is the best choice for MSK), the
    /// signal is sampled every symbol at $t_{n} = (n + (L + 1)/2) T$, where the phase
    /// increment $\varphi(t_{n}) - \varphi(t_{n-1})$ depends most on $a_{n}$ (for an even
    /// $L$ and an odd `sps` the instant is rounded down to a sample), and
    /// $\hat{a}_{n} = \operatorname{sign} \operatorname{Im} (r_{n} r_{n-1}^{\ast})$. A
    /// symbol whose instant falls after the end of the signal uses its last sample.
    ///
    /// For MSK without filter the increment is exactly $\pi a_{n}/2$; the partial
    /// response of GMSK closes the eye, so the detector degrades with $BT$.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless the signal is binary or if `rx_bt` is not
    /// positive, [`com_core::Error::Shape`] unless it holds a whole number of symbols.
    pub fn differential_detect(&self, r: &[C64], rx_bt: Option<f64>) -> Result<Vec<usize>> {
        self.ensure_binary()?;
        let n_sym = self.n_symbols(r)?;
        let (l, sps) = (self.cfg.length, self.cfg.sps);
        let filtered;
        let r = match rx_bt {
            Some(bt) => {
                ensure(bt > 0.0 && bt.is_finite(), || format!("the receive bandwidth must be positive, got {bt}"))?;
                let taps = low_pass_fir(bt, sps as f64, 4 * sps + 1, LowPass::Gauss);
                filtered = fir_filter(&taps, r);
                &filtered[..]
            }
            None => r,
        };
        let offset = (l + 1) * sps / 2;
        let sample = |n: usize| -> C64 {
            let k = n * sps + offset;
            if k < r.len() { r[k] } else { r[r.len() - 1] }
        };
        let mut prev = r[offset - sps];
        let mut out = Vec::with_capacity(n_sym);
        for n in 0..n_sym {
            let cur = sample(n);
            out.push(usize::from((cur * prev.conj()).im > 0.0));
            prev = cur;
        }
        Ok(out)
    }

    /// The normalised minimum squared Euclidean distance over the pairs of sequences
    /// that diverge at $t = 0$ and merge again within `max_len` symbols (the upper bound
    /// $d_{B}^{2}$ of Anderson, Aulin & Sundberg, ch. 3, which is the true $d_{\min}^{2}$
    /// when the shortest merge is the closest one, as for MSK):
    ///
    /// $$
    /// d^{2} = \frac{\log_{2} M}{T} \int_{0}^{(N + L - 1) T} \left[1 - \cos \Delta \varphi(t)\right] dt,
    /// \qquad \Delta \varphi(t) = 2 \pi h \sum_{k=0}^{N-1} \gamma_{k} \\, q(t - kT),
    /// $$
    ///
    /// over the difference sequences $\gamma_{k} = a_{k} - a_{k}^{\prime} \in \\{0, \pm 2,
    /// \dots, \pm 2(M-1)\\}$ with $\gamma_{0} \ne 0$ and $\pi h \sum_{k} \gamma_{k}
    /// \equiv 0 \pmod{2 \pi}$ (the merge). The integral is evaluated with 256 points per
    /// symbol; it is infinite if no sequence merges within `max_len` symbols. With this
    /// normalisation antipodal signalling has $d^{2} = 2$.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::cpm::CpmConfig;
    /// let msk = CpmConfig::msk(4).build()?;
    /// assert!((msk.min_distance2(3) - 2.0).abs() < 1e-3);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn min_distance2(&self, max_len: usize) -> f64 {
        let (m, l) = (self.cfg.order, self.cfg.length);
        let n_diff = 2 * m - 1;
        let res = 256usize;
        let h = self.h();
        let unit = i64::from(self.cfg.h_den) * 2;
        let mut best = f64::INFINITY;
        let mut gamma = vec![0i64; max_len];
        for code in 0..n_diff.pow(max_len as u32) {
            let mut c = code;
            for g in &mut gamma {
                *g = (c % n_diff) as i64 - (m as i64 - 1);
                c /= n_diff;
            }
            if gamma[0] == 0 {
                continue;
            }
            // the merge: p Σ γ ≡ 0 (mod 2q), γ in units of 2 (the differences are even)
            let total: i64 = gamma.iter().sum();
            if (i64::from(self.cfg.h_num) * 2 * total).rem_euclid(unit) != 0 {
                continue;
            }
            let n_samples = (max_len + l - 1) * res;
            let mut integral = 0.0;
            for k in 0..=n_samples {
                let t = k as f64 / res as f64;
                let dphi: f64 = gamma
                    .iter()
                    .enumerate()
                    .map(|(i, &g)| 2.0 * g as f64 * self.cfg.pulse.phase_response(t - i as f64, l))
                    .sum::<f64>()
                    * TAU
                    * h;
                let w = if k == 0 || k == n_samples { 0.5 } else { 1.0 };
                integral += w * (1.0 - dphi.cos());
            }
            let d2 = (m as f64).log2() * integral / res as f64;
            best = best.min(d2);
        }
        best
    }
}

/// The differential precoder of MSK: the information bits $b_{1} \dots b_{N}$ give the
/// boundary signs $d_{n} = 2 b_{n} - 1$ with $d_{0} = +1$, and the CPM symbols
/// $a_{n} = d_{n} d_{n+1}$ (as indices, $i_{n} = (a_{n} + 1)/2$), so that the signal
/// equals OQPSK with half-sinusoid pulses carrying $d_{n}$ on the in-phase component at
/// even $n$ and on the quadrature component at odd $n$ (Proakis & Salehi, §3.3-2).
///
/// # Examples
/// ```
/// use com_waveform::cpm::msk_precode;
/// assert_eq!(msk_precode(&[1, 1, 0, 0, 1]), vec![1, 1, 0, 1, 0]);
/// ```
pub fn msk_precode(bits: &[u8]) -> Vec<usize> {
    let mut d_prev = 1i8;
    bits.iter()
        .map(|&b| {
            let d = if b & 1 == 1 { 1i8 } else { -1 };
            let a = d_prev * d;
            d_prev = d;
            usize::from(a > 0)
        })
        .collect()
}

/// The information bits of MSK from the terminal phase states of a survivor path
/// ([`MlPath::phase_states`], in units of $\pi/2$): $d_{n} = \operatorname{Re}
/// (e^{j \theta_{n}} (-j)^{n})$ for $n = 1 \dots N$, the inverse of [`msk_precode`].
///
/// # Examples
/// ```
/// use com_waveform::cpm::{CpmConfig, msk_bits, msk_precode};
/// let msk = CpmConfig::msk(4).build()?;
/// let bits = [1u8, 0, 1, 1, 0, 0, 0, 1];
/// let path = msk.ml_detect_path(&msk.modulate(&msk_precode(&bits))?)?;
/// assert_eq!(msk_bits(&path.phase_states), bits);
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn msk_bits(phase_states: &[usize]) -> Vec<u8> {
    phase_states.iter().enumerate().skip(1).map(|(n, &s)| u8::from((s + 4 - n % 4) % 4 == 0)).collect()
}
