//! FFT plans with the NumPy conventions.
//!
//! [`Fft`] wraps the forward and inverse plans of `rustfft` for one length:
//! [`Fft::forward`] computes `X_k = Σ_n x_n e^{-j2πkn/N}` (unnormalised) and
//! [`Fft::inverse`] `x_n = (1/N) Σ_k X_k e^{+j2πkn/N}`, the conventions of `numpy.fft`
//! and of OptiCommPy. Plans are cached per thread by [`plan`], so creating an `Fft` for
//! a length already seen costs nothing, and the work space of the transforms is kept per
//! thread next to them: a transform allocates nothing once its length has been seen.
//! [`fftfreq`] and [`angular_frequencies`] give
//! the frequency of every bin in the same order, [`fftshift`] reorders a spectrum by
//! increasing frequency and [`centred_bin`] maps centred subcarriers to bins. The
//! unitary transforms are [`fft_unitary_in_place`] and [`ifft_unitary_in_place`].
//!
//! # Examples
//! ```
//! use com_math::fft::{fft, ifft, fftfreq};
//! use com_math::C64;
//! let x: Vec<C64> = (0..8).map(|n| C64::new(n as f64, 0.0)).collect();
//! let y = ifft(&fft(&x));
//! assert!(x.iter().zip(&y).all(|(a, b)| (a - b).norm() < 1e-12));
//! assert_eq!(fftfreq(4, 1.0), vec![0.0, 0.25, -0.5, -0.25]);
//! ```

use crate::C64;
use num_complex::Complex32 as C32;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Forward and inverse FFT plans of one length.
#[derive(Clone)]
pub struct Fft {
    n: usize,
    fwd: Arc<dyn rustfft::Fft<f64>>,
    inv: Arc<dyn rustfft::Fft<f64>>,
    // the in-place work space of each plan (`get_inplace_scratch_len`)
    fwd_scratch: usize,
    inv_scratch: usize,
}

impl std::fmt::Debug for Fft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fft(n = {})", self.n)
    }
}

thread_local! {
    static PLANS: RefCell<(FftPlanner<f64>, HashMap<usize, Fft>)> = RefCell::new((FftPlanner::new(), HashMap::new()));
    // the work space of the transforms of this thread, grown to the largest length seen
    static SCRATCH: RefCell<Vec<C64>> = const { RefCell::new(Vec::new()) };
    // the same in single precision, for the solver `nlse::SplitStepF32`
    static PLANS_F32: RefCell<(FftPlanner<f32>, HashMap<usize, FftF32>)> =
        RefCell::new((FftPlanner::new(), HashMap::new()));
    static SCRATCH_F32: RefCell<Vec<C32>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` on `len` values of the work space of this thread (grown, never shrunk).
fn with_scratch(len: usize, f: impl FnOnce(&mut [C64])) {
    SCRATCH.with(|c| {
        // a transform never calls back into another one, so the cell is free; a nested
        // borrow (impossible today) would fall back to a buffer of its own
        if let Ok(mut s) = c.try_borrow_mut() {
            if s.len() < len {
                s.resize(len, C64::new(0.0, 0.0));
            }
            f(&mut s[..len]);
        } else {
            f(&mut vec![C64::new(0.0, 0.0); len]);
        }
    });
}

/// The forward and inverse plans of one length in single precision, cached per thread
/// with their work space as [`Fft`]; both transforms are unnormalised (the solver
/// `nlse::SplitStepF32` folds the `1/N` into its linear step).
#[derive(Clone)]
pub(crate) struct FftF32 {
    fwd: Arc<dyn rustfft::Fft<f32>>,
    inv: Arc<dyn rustfft::Fft<f32>>,
    scratch: usize,
}

/// The (cached) single-precision plans of length `n ≥ 1`.
pub(crate) fn plan_f32(n: usize) -> FftF32 {
    PLANS_F32.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (planner, cache) = &mut *guard;
        if let Some(p) = cache.get(&n) {
            return p.clone();
        }
        let (fwd, inv) = (planner.plan_fft_forward(n), planner.plan_fft_inverse(n));
        let scratch = fwd.get_inplace_scratch_len().max(inv.get_inplace_scratch_len());
        let p = FftF32 { fwd, inv, scratch };
        cache.insert(n, p.clone());
        p
    })
}

impl FftF32 {
    /// Run `f` on the work space of this thread.
    fn with_scratch(&self, f: impl FnOnce(&mut [C32])) {
        SCRATCH_F32.with(|c| {
            if let Ok(mut s) = c.try_borrow_mut() {
                if s.len() < self.scratch {
                    s.resize(self.scratch, C32::new(0.0, 0.0));
                }
                f(&mut s[..self.scratch]);
            } else {
                f(&mut vec![C32::new(0.0, 0.0); self.scratch]);
            }
        });
    }

    /// In-place forward transform (unnormalised).
    pub(crate) fn forward(&self, x: &mut [C32]) {
        self.with_scratch(|s| self.fwd.process_with_scratch(x, s));
    }

    /// In-place inverse transform, not divided by `N`.
    pub(crate) fn inverse_unnormalized(&self, x: &mut [C32]) {
        self.with_scratch(|s| self.inv.process_with_scratch(x, s));
    }
}

/// The (cached) plans of length `n ≥ 1`.
pub fn plan(n: usize) -> Fft {
    assert!(n >= 1, "FFT length must be positive");
    PLANS.with(|cell| {
        let mut guard = cell.borrow_mut();
        let (planner, cache) = &mut *guard;
        if let Some(p) = cache.get(&n) {
            return p.clone();
        }
        let (fwd, inv) = (planner.plan_fft_forward(n), planner.plan_fft_inverse(n));
        let (fwd_scratch, inv_scratch) = (fwd.get_inplace_scratch_len(), inv.get_inplace_scratch_len());
        let p = Fft { n, fwd, inv, fwd_scratch, inv_scratch };
        cache.insert(n, p.clone());
        p
    })
}

impl Fft {
    /// The transform length.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the length is zero (never, a plan has `n ≥ 1`).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// In-place forward transform (unnormalised).
    ///
    /// # Panics
    /// If `x.len() != self.len()`.
    pub fn forward(&self, x: &mut [C64]) {
        assert_eq!(x.len(), self.n, "FFT length mismatch");
        with_scratch(self.fwd_scratch, |s| self.fwd.process_with_scratch(x, s));
    }

    /// In-place inverse transform, divided by `N`.
    ///
    /// # Panics
    /// If `x.len() != self.len()`.
    pub fn inverse(&self, x: &mut [C64]) {
        assert_eq!(x.len(), self.n, "FFT length mismatch");
        with_scratch(self.inv_scratch, |s| self.inv.process_with_scratch(x, s));
        let s = 1.0 / self.n as f64;
        for v in x.iter_mut() {
            *v *= s;
        }
    }
}

/// Forward FFT of a slice (allocates the output).
pub fn fft(x: &[C64]) -> Vec<C64> {
    let mut y = x.to_vec();
    if !y.is_empty() {
        plan(y.len()).forward(&mut y);
    }
    y
}

/// Inverse FFT of a slice, divided by `N` (allocates the output).
pub fn ifft(x: &[C64]) -> Vec<C64> {
    let mut y = x.to_vec();
    if !y.is_empty() {
        plan(y.len()).inverse(&mut y);
    }
    y
}

/// The unitary forward DFT in place, `X_k = Σ_n x_n e^{-j2πkn/N} / √N` (energy
/// preserving).
///
/// # Examples
/// ```
/// use com_math::{C64, energy, fft};
/// let mut x = vec![C64::new(1.0, 0.0), C64::new(0.0, 2.0), C64::new(-1.0, 1.0)];
/// let e = energy(&x);
/// fft::fft_unitary_in_place(&mut x);
/// assert!((energy(&x) - e).abs() < 1e-12);
/// fft::ifft_unitary_in_place(&mut x);
/// assert!((x[1] - C64::new(0.0, 2.0)).norm() < 1e-12);
/// ```
pub fn fft_unitary_in_place(x: &mut [C64]) {
    if x.is_empty() {
        return;
    }
    plan(x.len()).forward(x);
    let s = 1.0 / (x.len() as f64).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// The unitary inverse DFT in place, `x_n = Σ_k X_k e^{+j2πkn/N} / √N`.
pub fn ifft_unitary_in_place(x: &mut [C64]) {
    if x.is_empty() {
        return;
    }
    plan(x.len()).inverse(x);
    let s = (x.len() as f64).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// The DFT bin of the `i`-th of `k` contiguous subcarriers centred on bin 0 (indices
/// `-k/2 … k/2 - 1`, in increasing frequency) on a grid of `n` bins: `(i - ⌊k/2⌋) mod n`.
///
/// # Examples
/// ```
/// use com_math::fft::centred_bin;
/// // four subcarriers on a grid of eight: bins 6, 7, 0, 1
/// assert_eq!((0..4).map(|i| centred_bin(i, 4, 8)).collect::<Vec<_>>(), [6, 7, 0, 1]);
/// ```
pub fn centred_bin(i: usize, k: usize, n: usize) -> usize {
    (i as isize - (k / 2) as isize).rem_euclid(n as isize) as usize
}

/// Swap the two halves of a spectrum so that the zero frequency is in the middle
/// (`numpy.fft.fftshift`): the bin `(i + ⌊N/2⌋) mod N` moves to `i`.
///
/// # Examples
/// ```
/// use com_math::fft::fftshift;
/// assert_eq!(fftshift(&[0, 1, 2, -2, -1]), [-2, -1, 0, 1, 2]);
/// assert_eq!(fftshift(&[0, 1, -2, -1]), [-2, -1, 0, 1]);
/// ```
pub fn fftshift<T: Copy>(x: &[T]) -> Vec<T> {
    let n = x.len();
    (0..n).map(|i| x[(i + n.div_ceil(2)) % n]).collect()
}

/// Filter a periodic block by its frequency response: `IFFT(FFT(x) · H)`, `H` one
/// value per DFT bin in the order of [`fftfreq`] (a circular convolution).
///
/// # Panics
/// If `h` does not yield one value per sample of `x` (the callers build it on the
/// frequencies of `x`).
///
/// # Examples
/// ```
/// use com_math::{C64, fft};
/// // a delay of one sample: H_k = e^{-j2πk/N}
/// let x = vec![C64::new(1.0, 0.0), C64::new(2.0, 0.0), C64::new(3.0, 0.0), C64::new(4.0, 0.0)];
/// let h = (0..4).map(|k| com_math::cis(-std::f64::consts::TAU * k as f64 / 4.0));
/// let y = fft::filter(&x, h);
/// assert!((y[1] - x[0]).norm() < 1e-12 && (y[0] - x[3]).norm() < 1e-12);
/// ```
pub fn filter(x: &[C64], h: impl IntoIterator<Item = C64>) -> Vec<C64> {
    let mut y = fft(x);
    let mut h = h.into_iter();
    for v in &mut y {
        match h.next() {
            Some(hk) => *v *= hk,
            None => panic!("fft::filter: fewer response values than the {} samples", x.len()),
        }
    }
    assert!(h.next().is_none(), "fft::filter: more response values than the {} samples", x.len());
    if !y.is_empty() {
        plan(y.len()).inverse(&mut y);
    }
    y
}

/// Frequencies of the DFT bins for a sample spacing `d` (`numpy.fft.fftfreq`):
/// `[0, 1, …, ⌈N/2⌉-1, -⌊N/2⌋, …, -1] / (N d)`.
pub fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let s = if k < n.div_ceil(2) { k as f64 } else { k as f64 - n as f64 };
            s / (n as f64 * d)
        })
        .collect()
}

/// Angular frequencies `2π f` (rad/s) of the bins at the sampling rate `fs`.
pub fn angular_frequencies(n: usize, fs: f64) -> Vec<f64> {
    fftfreq(n, 1.0 / fs).into_iter().map(|f| 2.0 * std::f64::consts::PI * f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // the transforms with the work space of the thread give the bits of rustfft's own
    // `process` (which allocates its work space), for powers of two, primes (Bluestein,
    // Rader) and mixed radices, in any order of lengths
    #[test]
    fn the_shared_work_space_gives_the_bits_of_rustfft_process() {
        let mut planner = FftPlanner::<f64>::new();
        for &n in &[1, 2, 7, 64, 97, 360, 1024, 1031, 12, 4093, 3] {
            let x: Vec<C64> = (0..n).map(|k| C64::new((k as f64 * 0.37).sin(), (k as f64 * 1.3).cos())).collect();
            let p = plan(n);
            let (mut a, mut b) = (x.clone(), x.clone());
            p.forward(&mut a);
            planner.plan_fft_forward(n).process(&mut b);
            assert!(
                a.iter().zip(&b).all(|(u, v)| u.re.to_bits() == v.re.to_bits() && u.im.to_bits() == v.im.to_bits())
            );
            let (mut a, mut b) = (x.clone(), x.clone());
            p.inverse(&mut a);
            planner.plan_fft_inverse(n).process(&mut b);
            let s = 1.0 / n as f64;
            assert!(a.iter().zip(&b).all(|(u, v)| *u == v * s));
        }
    }

    #[test]
    #[should_panic(expected = "fewer response values")]
    fn filter_rejects_a_short_response_instead_of_leaving_bins_unfiltered() {
        let _ = filter(&[C64::new(1.0, 0.0); 4], vec![C64::new(0.0, 0.0); 3]);
    }

    #[test]
    #[should_panic(expected = "more response values")]
    fn filter_rejects_a_long_response() {
        let _ = filter(&[C64::new(1.0, 0.0); 4], vec![C64::new(0.0, 0.0); 5]);
    }
}
