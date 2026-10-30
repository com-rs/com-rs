//! Filters: pulse shapes, FIR filtering, low-pass design, moving average.
//!
//! # Pulse shaping
//!
//! [`pulse_shape`] returns the taps of a transmit pulse at `sps` samples per symbol,
//! normalised to a unit sum (unit DC gain), on the symmetric time grid
//! `t_i = (-⌊L/2⌋ + i·2⌊L/2⌋/(L-1)) / sps`, `i = 0..L-1`. For an even `L` it is the grid
//! of OptiCommPy's `pulseShape` (a `numpy.linspace` of spacing `L/(L-1)` samples, so the
//! taps fall between the sampling instants); for an odd `L` the spacing is exactly one
//! sample and the taps sit on the sampling instants, the choice to make for matched
//! filtering (OptiCommPy writes `-L // 2`, which rounds to `-⌊L/2⌋ - 1` for an odd `L`
//! and shifts its grid by about half a sample):
//!
//! - raised cosine: `h(t) = sinc(t/T) cos(παt/T) / (1 - 4α²t²/T²)`;
//! - root-raised cosine: `h(t) = [sin(πt(1-α)/T) + 4αt/T cos(πt(1+α)/T)] / [πt/T (1-(4αt/T)²)]`,
//!   with its limits at `t = 0` and `|t| = T/(4α)`;
//! - rectangular: `sps` ones padded with `⌊sps/2⌋` zeros on each side;
//! - NRZ: `sps` ones convolved with a Gaussian `e^{-t²}` sampled on `[-2, 2]`.
//!
//! [`rrc_spectrum`] gives the exact root-raised-cosine *frequency* response on the bins
//! of a periodic block, the matched filter of the split-step simulations, from the
//! scalar [`raised_cosine_spectrum`].
//!
//! # Filtering
//!
//! - [`fir_filter`] convolves with the `same` alignment of `scipy.signal`: the output
//!   has the length of the input and `y[n] = Σ_k h_k x[n + ⌊(L-1)/2⌋ - k]`, so a
//!   symmetric filter introduces no delay; long filters use the FFT (directly up to 32
//!   taps, by overlap-add for a signal longer than 8 filters); a [`Fir`] keeps the
//!   converted taps and the spectrum of the overlap-add for many blocks;
//! - [`low_pass_fir`] designs a windowless sinc (`Rect`) or Gaussian low-pass FIR of
//!   cut-off `fc`;
//! - [`moving_average`] (centred, zeros outside);
//! - [`normalize_energy`] scales taps to unit energy, $\sum_{l} h_{l}^{2} = 1$;
//! - [`response_at`] evaluates the frequency response
//!   $H(f) = \sum_{l} h_{l}\, e^{-j 2\pi f l}$ at $f = k/n$ cycles per sample.
//!
//! # Examples
//! ```
//! use com_dsp::filter::fir_filter;
//! use com_math::C64;
//! // a symmetric filter keeps the alignment: an impulse comes out centred
//! let mut x = vec![C64::new(0.0, 0.0); 9];
//! x[4] = C64::new(1.0, 0.0);
//! let y = fir_filter(&[0.25, 0.5, 0.25], &x);
//! assert!((y[4].re - 0.5).abs() < 1e-15 && (y[3].re - 0.25).abs() < 1e-15);
//! ```

use com_core::{Error, Result};
use com_math::fft;
use com_math::{C64, sinc};
use std::f64::consts::PI;

/// A transmit pulse shape.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PulseShape {
    /// Rectangular pulse of one symbol.
    Rect,
    /// Non-return-to-zero pulse with Gaussian edges.
    Nrz,
    /// Raised cosine of roll-off `rolloff` (Nyquist pulse).
    Rc {
        /// Roll-off factor `α ∈ (0, 1]`.
        rolloff: f64,
    },
    /// Root-raised cosine of roll-off `rolloff` (half of a Nyquist pair).
    Rrc {
        /// Roll-off factor `α ∈ (0, 1]`.
        rolloff: f64,
    },
}

impl PulseShape {
    /// The pulse of a name, `rect`, `nrz`, `rc` or `rrc` (case-insensitive), with the
    /// roll-off of the raised cosines.
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    pub fn from_name(kind: &str, rolloff: f64) -> Result<PulseShape> {
        match kind.to_ascii_lowercase().as_str() {
            "rect" => Ok(PulseShape::Rect),
            "nrz" => Ok(PulseShape::Nrz),
            "rc" => Ok(PulseShape::Rc { rolloff }),
            "rrc" => Ok(PulseShape::Rrc { rolloff }),
            other => Err(Error::Config(format!("pulse kind must be rect, nrz, rc or rrc, got {other:?}"))),
        }
    }
}

/// The symmetric time grid of the raised-cosine pulses, in symbols (see the module
/// documentation for the comparison with OptiCommPy).
fn pulse_grid(sps: usize, n_taps: usize) -> Vec<f64> {
    let h = (n_taps / 2) as f64;
    if n_taps == 1 {
        return vec![0.0];
    }
    (0..n_taps).map(|i| (-h + i as f64 * 2.0 * h / (n_taps as f64 - 1.0)) / sps as f64).collect()
}

/// Root-raised-cosine impulse response at the times `t` (symbol period `ts`).
pub fn rrc_taps(t: &[f64], rolloff: f64, ts: f64) -> Vec<f64> {
    t.iter()
        .map(|&ti| {
            if ti == 0.0 {
                (1.0 / ts) * (1.0 + rolloff * (4.0 / PI - 1.0))
            } else if rolloff > 0.0 && (ti.abs() - ts / (4.0 * rolloff)).abs() < 1e-12 * ts {
                let a = (1.0 + 2.0 / PI) * (PI / (4.0 * rolloff)).sin();
                let b = (1.0 - 2.0 / PI) * (PI / (4.0 * rolloff)).cos();
                rolloff / (ts * 2f64.sqrt()) * (a + b)
            } else {
                let t1 = PI * ti / ts;
                let t2 = 4.0 * rolloff * ti / ts;
                (1.0 / ts) * ((t1 * (1.0 - rolloff)).sin() + t2 * (t1 * (1.0 + rolloff)).cos()) / (t1 * (1.0 - t2 * t2))
            }
        })
        .collect()
}

/// Raised-cosine impulse response at the times `t` (symbol period `ts`).
pub fn rc_taps(t: &[f64], rolloff: f64, ts: f64) -> Vec<f64> {
    t.iter()
        .map(|&ti| {
            if rolloff > 0.0 && (ti.abs() - ts / (2.0 * rolloff)).abs() < 1e-12 * ts {
                PI / (4.0 * ts) * sinc(1.0 / (2.0 * rolloff))
            } else {
                (1.0 / ts) * sinc(ti / ts) * (PI * rolloff * ti / ts).cos()
                    / (1.0 - 4.0 * rolloff * rolloff * ti * ti / (ts * ts))
            }
        })
        .collect()
}

/// Taps of a pulse at `sps` samples per symbol, normalised to a unit sum.
///
/// `n_taps` is the length of the raised-cosine pulses (ignored by `Rect` and `Nrz`).
///
/// # Errors
/// [`Error::Config`] for `sps = 0`, or `n_taps = 0` for a raised-cosine pulse.
pub fn pulse_shape(shape: PulseShape, sps: usize, n_taps: usize) -> Result<Vec<f64>> {
    let raised_cosine = matches!(shape, PulseShape::Rc { .. } | PulseShape::Rrc { .. });
    if sps == 0 || (raised_cosine && n_taps == 0) {
        return Err(Error::Config(format!("sps and n_taps must be positive, got {sps} and {n_taps}")));
    }
    let pulse: Vec<f64> = match shape {
        PulseShape::Rect => {
            let z = sps / 2;
            let mut p = vec![0.0; z];
            p.extend(std::iter::repeat_n(1.0, sps));
            p.extend(std::iter::repeat_n(0.0, z));
            p
        }
        PulseShape::Nrz => {
            let g: Vec<f64> = (0..sps)
                .map(|i| {
                    let t = if sps == 1 { 0.0 } else { -2.0 + 4.0 * i as f64 / (sps as f64 - 1.0) };
                    2.0 / PI.sqrt() * (-t * t).exp()
                })
                .collect();
            let mut p = vec![0.0; 2 * sps - 1];
            for i in 0..sps {
                for (j, gj) in g.iter().enumerate() {
                    p[i + j] += gj;
                }
            }
            p
        }
        PulseShape::Rc { rolloff } => rc_taps(&pulse_grid(sps, n_taps), rolloff, 1.0),
        PulseShape::Rrc { rolloff } => rrc_taps(&pulse_grid(sps, n_taps), rolloff, 1.0),
    };
    let s: f64 = pulse.iter().sum();
    Ok(pulse.iter().map(|v| v / s).collect())
}

/// The raised-cosine spectrum of roll-off $\beta$ = `rolloff` at the frequency `f` for
/// the symbol rate $R_{s}$ = `symbol_rate` (in the same unit), normalised to 1 in the
/// flat band:
///
/// $$
/// \mathrm{RC}(f) = \begin{cases} 1, & |f| \le f_{1}, \\\\
/// \frac{1}{2}\left[1 + \cos\left(\frac{\pi}{\beta R_{s}}\left(|f| - f_{1}\right)\right)
/// \right], & f_{1} < |f| < f_{2}, \\\\ 0, & |f| \ge f_{2}, \end{cases}
/// \qquad f_{1,2} = \frac{(1 \mp \beta) R_{s}}{2}.
/// $$
///
/// The shape of [`rrc_spectrum`] (its square root) and of the WDM channels of the GN
/// model (`com_channel::optical::gn::WdmSpectrum`); a rectangular spectrum for
/// $\beta = 0$.
///
/// # Examples
/// ```
/// use com_dsp::filter::raised_cosine_spectrum;
/// assert_eq!(raised_cosine_spectrum(10e9, 32e9, 0.1), 1.0);
/// assert!((raised_cosine_spectrum(16e9, 32e9, 0.1) - 0.5).abs() < 1e-15);
/// assert_eq!(raised_cosine_spectrum(-20e9, 32e9, 0.1), 0.0);
/// ```
pub fn raised_cosine_spectrum(f: f64, symbol_rate: f64, rolloff: f64) -> f64 {
    let x = f.abs();
    let lo = (1.0 - rolloff) * symbol_rate / 2.0;
    let hi = (1.0 + rolloff) * symbol_rate / 2.0;
    if x <= lo {
        1.0
    } else if x >= hi {
        0.0
    } else {
        0.5 * (1.0 + (PI / (rolloff * symbol_rate) * (x - lo)).cos())
    }
}

/// Root of the raised-cosine spectrum on the `n_sym·sps` bins of a periodic block.
///
/// `H_k = √RC(f_k)` ([`raised_cosine_spectrum`]) with `f_k` the frequency of bin `k` in
/// symbol-rate units; the value is 1 in the passband, so that shaping followed by
/// matched filtering with this response is exactly Nyquist on a periodic block.
pub fn rrc_spectrum(n_sym: usize, sps: usize, rolloff: f64) -> Vec<f64> {
    fft::fftfreq(n_sym * sps, 1.0)
        .into_iter()
        .map(|f| raised_cosine_spectrum(f * sps as f64, 1.0, rolloff).sqrt())
        .collect()
}

/// A filter shorter than this (or a signal), in taps, is convolved directly.
const DIRECT_MAX_TAPS: usize = 32;

/// A signal longer than this many filter lengths is convolved by overlap-add.
const OVERLAP_ADD_RATIO: usize = 8;

/// The smallest FFT of the overlap-add (blocks of a few filter lengths otherwise).
const OVERLAP_ADD_MIN_FFT: usize = 64;

/// The spectrum of a filter on the FFT of the overlap-add, `(4 L)` rounded up to a power
/// of two (at least [`OVERLAP_ADD_MIN_FFT`]).
#[derive(Clone, Debug)]
struct OverlapAdd {
    plan: fft::Fft,
    spectrum: Vec<C64>,
}

impl OverlapAdd {
    fn new(h: &[C64]) -> OverlapAdd {
        let nfft = (4 * h.len()).next_power_of_two().max(OVERLAP_ADD_MIN_FFT);
        let plan = fft::plan(nfft);
        let mut spectrum = vec![C64::new(0.0, 0.0); nfft];
        spectrum[..h.len()].copy_from_slice(h);
        plan.forward(&mut spectrum);
        OverlapAdd { plan, spectrum }
    }
}

/// The samples `start..start + out.len()` of the full linear convolution of `x` with the
/// non-empty taps `h`, added to `out` (zeros on entry): directly when the filter or the
/// signal is short, by overlap-add (with the spectrum of the filter, computed here if
/// `ola` does not hold it) when the signal is long, by one FFT otherwise. Every sample
/// adds its terms in the order of the full convolution.
fn convolve_window(h: &[C64], ola: Option<&OverlapAdd>, x: &[C64], start: usize, out: &mut [C64]) {
    let n = x.len() + h.len() - 1;
    let end = start + out.len();
    let mut add = |j: usize, v: C64| {
        if j >= start && j < end {
            out[j - start] += v;
        }
    };
    if h.len().min(x.len()) <= DIRECT_MAX_TAPS {
        for (i, &xi) in x.iter().enumerate() {
            for (k, &hk) in h.iter().enumerate() {
                add(i + k, hk * xi);
            }
        }
        return;
    }
    // a long signal and a shorter filter: overlap-add with blocks of a few filter lengths
    if x.len() > OVERLAP_ADD_RATIO * h.len() {
        let own;
        let ola = if let Some(o) = ola {
            o
        } else {
            own = OverlapAdd::new(h);
            &own
        };
        let (p, hf) = (&ola.plan, &ola.spectrum);
        let nfft = p.len();
        let block = nfft - h.len() + 1;
        let mut buf = vec![C64::new(0.0, 0.0); nfft];
        for first in (0..x.len()).step_by(block) {
            let seg = &x[first..(first + block).min(x.len())];
            buf.fill(C64::new(0.0, 0.0));
            buf[..seg.len()].copy_from_slice(seg);
            p.forward(&mut buf);
            for (u, v) in buf.iter_mut().zip(hf) {
                *u *= v;
            }
            p.inverse(&mut buf);
            let len = (seg.len() + h.len() - 1).min(n - first);
            for (m, b) in buf[..len].iter().enumerate() {
                add(first + m, *b);
            }
        }
        return;
    }
    let m = n.next_power_of_two();
    let p = fft::plan(m);
    let mut a = vec![C64::new(0.0, 0.0); m];
    let mut b = vec![C64::new(0.0, 0.0); m];
    a[..x.len()].copy_from_slice(x);
    b[..h.len()].copy_from_slice(h);
    p.forward(&mut a);
    p.forward(&mut b);
    for (u, v) in a.iter_mut().zip(&b) {
        *u *= v;
    }
    p.inverse(&mut a);
    out.copy_from_slice(&a[start..end]);
}

/// Full linear convolution of `x` with complex taps `h` (direct or by FFT).
pub fn convolve_full(h: &[C64], x: &[C64]) -> Vec<C64> {
    if h.is_empty() || x.is_empty() {
        return Vec::new();
    }
    let mut y = vec![C64::new(0.0, 0.0); x.len() + h.len() - 1];
    convolve_window(h, None, x, 0, &mut y);
    y
}

/// FIR filtering with complex taps and the `same` alignment: as many samples as `x`
/// (zeros for no tap).
pub fn fir_filter_complex(h: &[C64], x: &[C64]) -> Vec<C64> {
    let mut y = vec![C64::new(0.0, 0.0); x.len()];
    if !h.is_empty() && !x.is_empty() {
        convolve_window(h, None, x, (h.len() - 1) / 2, &mut y);
    }
    y
}

/// FIR filtering with real taps and the `same` alignment (OptiCommPy `firFilter`).
pub fn fir_filter(h: &[f64], x: &[C64]) -> Vec<C64> {
    let hc: Vec<C64> = h.iter().map(|&v| C64::new(v, 0.0)).collect();
    fir_filter_complex(&hc, x)
}

/// An FIR filter built once for many blocks: [`fir_filter_complex`] (the `same`
/// alignment, the same results bit for bit) with the taps converted once and, for a
/// filter of more than 32 taps, the spectrum of the overlap-add of long signals computed
/// once; [`Fir::filter_into`] reuses its output (ADR 0011).
///
/// # Examples
/// ```
/// use com_dsp::filter::{Fir, fir_filter};
/// use com_math::C64;
/// let h: Vec<f64> = (0..41).map(|k| 1.0 / (1.0 + (k as f64 - 20.0).abs())).collect();
/// let x: Vec<C64> = (0..1000).map(|n| C64::new((0.1 * n as f64).sin(), 0.0)).collect();
/// let fir = Fir::new(&h);
/// let mut y = Vec::new();
/// fir.filter_into(&x, &mut y);
/// assert_eq!(y, fir_filter(&h, &x));
/// ```
#[derive(Clone, Debug)]
pub struct Fir {
    taps: Vec<C64>,
    overlap_add: Option<OverlapAdd>,
}

impl Fir {
    /// The filter of the real taps `taps`.
    pub fn new(taps: &[f64]) -> Fir {
        Fir::complex(&taps.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>())
    }

    /// The filter of the complex taps `taps`.
    pub fn complex(taps: &[C64]) -> Fir {
        Fir { taps: taps.to_vec(), overlap_add: (taps.len() > DIRECT_MAX_TAPS).then(|| OverlapAdd::new(taps)) }
    }

    /// The taps.
    pub fn taps(&self) -> &[C64] {
        &self.taps
    }

    /// Filter `x` with the `same` alignment ([`fir_filter_complex`]).
    pub fn filter(&self, x: &[C64]) -> Vec<C64> {
        let mut y = Vec::new();
        self.filter_into(x, &mut y);
        y
    }

    /// [`Fir::filter`] into `out`, resized to the length of `x` and filled in place.
    pub fn filter_into(&self, x: &[C64], out: &mut Vec<C64>) {
        out.clear();
        out.resize(x.len(), C64::new(0.0, 0.0));
        if !self.taps.is_empty() && !x.is_empty() {
            convolve_window(&self.taps, self.overlap_add.as_ref(), x, (self.taps.len() - 1) / 2, out);
        }
    }
}

/// Upsample, filter and downsample: insert `up - 1` zeros after every sample of `x`,
/// filter with the real taps `h` with the `same` alignment ([`fir_filter`]), then keep
/// the samples $0, D, 2 D, \ldots$ ($D$ = `down`). The result is that of
/// `downsample(&fir_filter(h, &upsample(x, up)), down, 0)` of [`crate::resample`], bit
/// for bit for finite samples and taps, without its intermediate signals: when the
/// filter or the upsampled signal is short (the direct convolution of
/// [`fir_filter`]), only the kept samples are computed, from the polyphase components
/// of `h` (the zeros of the upsampled signal are skipped, every sample adds its terms
/// in the order of the full convolution); otherwise the upsampled signal is filtered
/// by FFT and decimated in place. As [`crate::resample::upsample`] and
/// [`crate::resample::downsample`], `up = 0` gives no sample and `down = 0` keeps
/// every sample.
///
/// Unlike `scipy.signal.upfirdn`, which returns the whole linear convolution, the
/// output has the $\lceil U N / D \rceil$ samples of the `same` alignment ($N$ samples
/// of `x`, $U$ = `up`).
///
/// # Examples
/// ```
/// use com_dsp::filter::{fir_filter, upfirdn};
/// use com_dsp::resample::{downsample, upsample};
/// use com_math::C64;
/// let h = [0.25, 0.5, 1.0, 0.5, 0.25];
/// let x: Vec<C64> = (0..50).map(|n| C64::new((0.3 * n as f64).cos(), (0.2 * n as f64).sin())).collect();
/// let y = upfirdn(&h, &x, 4, 3);
/// assert_eq!(y.len(), 67);
/// assert_eq!(y, downsample(&fir_filter(&h, &upsample(&x, 4)), 3, 0));
/// ```
pub fn upfirdn(h: &[f64], x: &[C64], up: usize, down: usize) -> Vec<C64> {
    let down = down.max(1);
    let len = x.len() * up;
    let n_out = len.div_ceil(down);
    if h.is_empty() || len == 0 {
        return vec![C64::new(0.0, 0.0); n_out];
    }
    let hc: Vec<C64> = h.iter().map(|&v| C64::new(v, 0.0)).collect();
    let start = (h.len() - 1) / 2;
    if h.len().min(len) > DIRECT_MAX_TAPS {
        let mut y = vec![C64::new(0.0, 0.0); len];
        convolve_window(&hc, None, &crate::resample::upsample(x, up), start, &mut y);
        for q in 1..n_out {
            y[q] = y[q * down];
        }
        y.truncate(n_out);
        return y;
    }
    // the sample `n` of the full convolution sums `h[n - m U] x[m]` over the input
    // samples `m` in increasing order, as the direct convolution adds them
    (0..len)
        .step_by(down)
        .map(|j| {
            let n = j + start;
            let first = n.saturating_sub(h.len() - 1).div_ceil(up);
            let last = (n / up).min(x.len() - 1);
            let mut acc = C64::new(0.0, 0.0);
            for (m, &xm) in x.iter().enumerate().take(last + 1).skip(first) {
                acc += hc[n - m * up] * xm;
            }
            acc
        })
        .collect()
}

/// Scale the taps `h` in place to unit energy, $h_{l} / \sqrt{\sum_{i} h_{i}^{2}}$ (taps
/// that are all zero are left unchanged).
///
/// # Examples
/// ```
/// use com_dsp::filter::normalize_energy;
/// let mut h = vec![3.0, 4.0];
/// normalize_energy(&mut h);
/// assert_eq!(h, vec![0.6, 0.8]);
/// ```
pub fn normalize_energy(h: &mut [f64]) {
    let norm = h.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for v in h {
            *v /= norm;
        }
    }
}

/// The frequency response of the FIR filter `h` at the frequency $f = k/n$ cycles per
/// sample (the subcarrier $k$ of a grid of $n$; $k$ need not be an integer):
///
/// $$
/// H(k/n) = \sum_{l} h_{l}\, e^{-j 2\pi k l / n}.
/// $$
///
/// # Examples
/// ```
/// use com_dsp::filter::response_at;
/// use com_math::C64;
/// let h = [C64::new(0.5, 0.0), C64::new(0.5, 0.0)];
/// assert!((response_at(&h, 0.0, 8.0) - C64::new(1.0, 0.0)).norm() < 1e-15);
/// assert!(response_at(&h, 4.0, 8.0).norm() < 1e-15);   // the null at half the rate
/// ```
pub fn response_at(h: &[C64], k: f64, n: f64) -> C64 {
    h.iter().enumerate().map(|(l, v)| v * com_math::cis(-std::f64::consts::TAU * k * l as f64 / n)).sum()
}

/// Low-pass FIR filter types of [`low_pass_fir`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowPass {
    /// Truncated sinc (ideal rectangular response).
    Rect,
    /// Gaussian response with 3 dB bandwidth `fc`.
    Gauss,
}

impl LowPass {
    /// The name of the filter, as `from_str` reads it: `rect` or `gauss`.
    pub fn name(self) -> &'static str {
        match self {
            LowPass::Rect => "rect",
            LowPass::Gauss => "gauss",
        }
    }
}

impl std::str::FromStr for LowPass {
    type Err = Error;

    /// `rect` or `gauss` (`gaussian`), case-insensitive.
    fn from_str(s: &str) -> Result<LowPass> {
        match s.to_ascii_lowercase().as_str() {
            "rect" => Ok(LowPass::Rect),
            "gauss" | "gaussian" => Ok(LowPass::Gauss),
            other => Err(Error::Config(format!("unknown filter {other:?} (rect, gauss)"))),
        }
    }
}

/// `n`-tap low-pass FIR of cut-off `fc` at the sampling rate `fs`, unit DC gain
/// (OptiCommPy `lowPassFIR`).
pub fn low_pass_fir(fc: f64, fs: f64, n: usize, kind: LowPass) -> Vec<f64> {
    let fu = fc / fs;
    let d = (n as f64 - 1.0) / 2.0;
    let h: Vec<f64> = (0..n)
        .map(|i| {
            let m = i as f64 - d;
            match kind {
                LowPass::Rect => 2.0 * fu * sinc(2.0 * fu * m),
                LowPass::Gauss => {
                    (2.0 * PI / 2f64.ln()).sqrt() * fu * (-(2.0 / 2f64.ln()) * (PI * fu * m).powi(2)).exp()
                }
            }
        })
        .collect();
    let s: f64 = h.iter().sum();
    h.iter().map(|v| v / s).collect()
}

/// Centred moving average over `n` samples, zeros outside the signal
/// (OptiCommPy `movingAverage`): `y[i] = (1/n) Σ_{m=i+c-n+1}^{i+c} x[m]`, `c = ⌊(n-1)/2⌋`.
pub fn moving_average(x: &[C64], n: usize) -> Vec<C64> {
    let len = x.len() as isize;
    let n = n.max(1) as isize;
    let c = (n - 1) / 2;
    // prefix sums for O(len)
    let mut pre = vec![C64::new(0.0, 0.0); x.len() + 1];
    for i in 0..x.len() {
        pre[i + 1] = pre[i] + x[i];
    }
    (0..len)
        .map(|i| {
            let hi = (i + c).min(len - 1);
            let lo = (i + c - n + 1).max(0);
            if hi < lo { C64::new(0.0, 0.0) } else { (pre[(hi + 1) as usize] - pre[lo as usize]) / n as f64 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::*;

    #[test]
    fn a_pulse_needs_samples_and_taps() {
        assert!(pulse_shape(PulseShape::Nrz, 0, 16).is_err());
        assert!(pulse_shape(PulseShape::Rrc { rolloff: 0.2 }, 4, 0).is_err());
        assert_eq!(pulse_shape(PulseShape::Rect, 2, 0).unwrap(), vec![0.0, 0.5, 0.5, 0.0]);
    }

    #[test]
    fn the_low_pass_names_are_read_back() {
        for f in [LowPass::Rect, LowPass::Gauss] {
            assert_eq!(f.name().parse::<LowPass>().unwrap(), f);
        }
    }

    #[test]
    fn rrc_pair_is_nyquist_on_a_periodic_block() {
        let n_sym = 64;
        let sps = 4;
        let h = rrc_spectrum(n_sym, sps, 0.25);
        // RC = |H|² vanishes on the alias pairs f and f ± Rs: zero ISI
        let s: Vec<C64> = (0..n_sym).map(|k| C64::new((k % 3) as f64 - 1.0, (k % 5) as f64 - 2.0)).collect();
        let mut up = fft::fft(&upsample(&s, sps));
        for (v, hk) in up.iter_mut().zip(&h) {
            *v *= hk * hk;
        }
        // Y_m = S_m (1/sps) Σ_aliases RC = S_m / sps
        let y = downsample(&fft::ifft(&up), sps, 0);
        for (a, b) in y.iter().zip(&s) {
            assert!((a * sps as f64 - b).norm() < 1e-9, "{a} vs {b}");
        }
        assert!((h[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn rc_taps_are_nyquist() {
        // samples of the RC pulse at the symbol instants: 1 at 0, 0 elsewhere
        let t: Vec<f64> = (-5..=5).map(|k| k as f64).collect();
        let h = rc_taps(&t, 0.3, 1.0);
        for (k, v) in t.iter().zip(&h) {
            if *k == 0.0 {
                assert!((v - 1.0).abs() < 1e-12);
            } else {
                assert!(v.abs() < 1e-12, "t = {k}: {v}");
            }
        }
    }

    #[test]
    fn moving_average_and_resampling() {
        let x: Vec<C64> = (0..10).map(|n| C64::new(n as f64, 0.0)).collect();
        let y = moving_average(&x, 3);
        assert!((y[5].re - 5.0).abs() < 1e-12 && (y[0].re - 1.0 / 3.0).abs() < 1e-12);
        let up = resample_fft(&x.iter().map(|v| v * 0.0 + C64::new(1.0, 0.0)).collect::<Vec<_>>(), 30);
        assert!(up.iter().all(|v| (v - C64::new(1.0, 0.0)).norm() < 1e-12));
        let d = delay_fft(&x, 0.0, 1.0);
        assert!(d.iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-12));
    }

    #[test]
    fn a_fir_built_once_filters_as_the_functions_bit_for_bit() {
        // the three paths: direct (17 taps), one FFT (101 taps, 500 samples) and
        // overlap-add (101 taps, 5000 samples), with a reused output
        let x: Vec<C64> = (0..5000).map(|k| C64::new((k as f64 * 0.37).sin(), (k as f64 * 0.11).cos())).collect();
        let h: Vec<f64> = (0..101).map(|k| ((k as f64 - 50.0) * 0.21).cos() / (1.0 + k as f64)).collect();
        let mut out = Vec::with_capacity(5000);
        for (taps, len) in [(&h[..17], 300), (&h[..], 500), (&h[..], 5000)] {
            let fir = Fir::new(taps);
            fir.filter_into(&x[..len], &mut out);
            assert_eq!(out, fir_filter(taps, &x[..len]));
            assert_eq!(fir.filter(&x[..len]), out);
        }
        assert!(out.capacity() >= 5000);
        assert_eq!(Fir::new(&[]).filter(&x[..3]), vec![C64::new(0.0, 0.0); 3]);
    }

    #[test]
    fn the_raised_cosine_spectrum_is_the_former_formula_of_rrc_spectrum() {
        for rolloff in [0.0, 0.1, 0.3, 1.0] {
            let (lo, hi) = (0.5 * (1.0 - rolloff), 0.5 * (1.0 + rolloff));
            for k in 0..200 {
                let f = (k as f64 - 100.0) * 0.0123;
                let a = f64::abs(f);
                let former = if a <= lo {
                    1.0
                } else if a < hi {
                    0.5 * (1.0 + (PI / rolloff * (a - lo)).cos())
                } else {
                    0.0
                };
                assert_eq!(raised_cosine_spectrum(f, 1.0, rolloff), former);
            }
        }
    }

    #[test]
    fn long_and_short_convolutions_agree() {
        let h: Vec<C64> = (0..40).map(|k| C64::new((k as f64 * 0.3).sin(), 0.1 * k as f64)).collect();
        let x: Vec<C64> = (0..100).map(|k| C64::new((k as f64 * 0.7).cos(), 0.0)).collect();
        let fast = convolve_full(&h, &x);
        let mut slow = vec![C64::new(0.0, 0.0); 139];
        for i in 0..100 {
            for k in 0..40 {
                slow[i + k] += h[k] * x[i];
            }
        }
        assert!(fast.iter().zip(&slow).all(|(a, b)| (a - b).norm() < 1e-9));
    }

    #[test]
    fn upfirdn_is_the_chain_of_upsample_fir_filter_and_downsample_bit_for_bit() {
        use crate::resample::{downsample, upsample};
        let bits = |v: &[C64]| v.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect::<Vec<_>>();
        // the direct path (short filter or signal), the overlap-add and the single FFT
        for n_taps in [0, 1, 4, 7, 33, 101] {
            let h: Vec<f64> = (0..n_taps).map(|k| (0.37 * k as f64).cos() / (1.0 + k as f64)).collect();
            for n in [0, 1, 5, 40, 300, 2000] {
                let x: Vec<C64> =
                    (0..n).map(|i| C64::new((0.11 * i as f64).sin(), (0.07 * i as f64).cos() - 0.5)).collect();
                for up in [0, 1, 2, 3, 4] {
                    for down in [0, 1, 2, 3, 5] {
                        let chain = downsample(&fir_filter(&h, &upsample(&x, up)), down, 0);
                        let y = upfirdn(&h, &x, up, down);
                        assert_eq!(bits(&y), bits(&chain), "{n_taps} taps, {n} samples, up {up}, down {down}");
                    }
                }
            }
        }
    }
}
