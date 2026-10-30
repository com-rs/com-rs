//! Multi-stream containers: bits, log-likelihood ratios, symbols and waveforms.
//!
//! Every container is two-dimensional, streams × elements, in standard (row-major)
//! layout; SISO is `n_streams == 1`. MIMO, polarisation multiplexing and space-division
//! multiplexing therefore reuse the same blocks. [`DualPol`], the two polarisations of
//! an optical field as two vectors, is kept for the slice-based optical blocks.
//!
//! # Model
//!
//! The sign convention of the log-likelihood ratios is fixed library-wide: a positive
//! LLR favours the bit 0,
//!
//! $$
//! L(b) = \ln \frac{P(b = 0 \mid y)}{P(b = 1 \mid y)}
//! $$
//!
//! # Examples
//! ```
//! use com_core::signal::{Bits, Signal};
//! use com_core::units::Hertz;
//! let b = Bits::from_stream(vec![0, 1, 1, 0]).unwrap();
//! assert_eq!((b.n_streams(), b.len()), (1, 4));
//! let s = Signal::zeros(2, 1024, Hertz(64e9));
//! assert_eq!((s.n_streams(), s.n_samples()), (2, 1024));
//! assert_eq!(s.stream(1).len(), 1024);
//! ```

use crate::C64;
use crate::error::{Error, Result, expect_len, same_len};
use crate::units::Hertz;
use ndarray::{Array2, ArrayView1, ArrayViewMut1};

/// The hard decisions of log-likelihood ratios: a negative LLR is a 1, any other a 0
/// (see the module for the sign convention).
///
/// # Examples
/// ```
/// use com_core::signal::hard_decisions;
/// assert_eq!(hard_decisions(&[2.0, -0.5, 0.0]), vec![0, 1, 0]);
/// ```
pub fn hard_decisions(llr: &[f64]) -> Vec<u8> {
    llr.iter().map(|l| u8::from(*l < 0.0)).collect()
}

macro_rules! container {
    ($(#[$doc:meta])* $name:ident, $elem:ty, $zero:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug, PartialEq, Default)]
        pub struct $name(Array2<$elem>);

        impl $name {
            /// A container from a `[n_streams, n]` array.
            pub fn new(data: Array2<$elem>) -> $name {
                $name(data.as_standard_layout().into_owned())
            }

            /// `n_streams` streams of `n` zeros.
            pub fn zeros(n_streams: usize, n: usize) -> $name {
                $name(Array2::from_elem((n_streams, n), $zero))
            }

            /// One stream.
            ///
            /// # Errors
            /// Never in practice; the signature matches [`Self::from_streams`].
            pub fn from_stream(v: Vec<$elem>) -> Result<$name> {
                let n = v.len();
                Array2::from_shape_vec((1, n), v)
                    .map($name)
                    .map_err(|e| Error::Shape(e.to_string()))
            }

            /// Several streams of equal length.
            ///
            /// # Errors
            /// [`Error::Shape`] if the streams have different lengths.
            pub fn from_streams(streams: &[Vec<$elem>]) -> Result<$name> {
                let n = streams.first().map_or(0, Vec::len);
                for s in streams {
                    same_len(s.len(), n, "streams")?;
                }
                let flat: Vec<$elem> = streams.iter().flatten().copied().collect();
                Array2::from_shape_vec((streams.len(), n), flat)
                    .map($name)
                    .map_err(|e| Error::Shape(e.to_string()))
            }

            /// Number of streams.
            pub fn n_streams(&self) -> usize {
                self.0.nrows()
            }

            /// Number of elements per stream.
            pub fn len(&self) -> usize {
                self.0.ncols()
            }

            /// Whether the streams are empty.
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            /// Stream `i`.
            pub fn stream(&self, i: usize) -> ArrayView1<'_, $elem> {
                self.0.row(i)
            }

            /// Stream `i`, mutable.
            pub fn stream_mut(&mut self, i: usize) -> ArrayViewMut1<'_, $elem> {
                self.0.row_mut(i)
            }

            /// Stream `i` as a slice (the layout is always standard).
            pub fn stream_slice(&self, i: usize) -> &[$elem] {
                let n = self.len();
                &self.as_slice()[i * n..(i + 1) * n]
            }

            /// All the elements, stream after stream.
            pub fn as_slice(&self) -> &[$elem] {
                self.0.as_slice().unwrap_or(&[])
            }

            /// Make `self` the single stream `x`, reusing its storage when the length is
            /// unchanged (ADR 0011).
            pub fn set_stream(&mut self, x: &[$elem]) {
                self.resize(1, x.len());
                self.as_slice_mut().copy_from_slice(x);
            }

            /// Make `self` a copy of `a`, reusing its storage when the shape is
            /// unchanged (ADR 0011).
            pub fn assign(&mut self, a: &Array2<$elem>) {
                self.resize(a.nrows(), a.ncols());
                self.0.assign(a);
            }

            /// Stream `i` as a mutable slice.
            pub fn stream_slice_mut(&mut self, i: usize) -> &mut [$elem] {
                let n = self.len();
                &mut self.as_slice_mut()[i * n..(i + 1) * n]
            }

            /// All the elements, stream after stream, mutable.
            pub fn as_slice_mut(&mut self) -> &mut [$elem] {
                self.0.as_slice_mut().unwrap_or(&mut [])
            }

            /// The underlying `[n_streams, n]` array.
            pub fn as_array(&self) -> &Array2<$elem> {
                &self.0
            }

            /// The underlying array, mutable.
            pub fn as_array_mut(&mut self) -> &mut Array2<$elem> {
                &mut self.0
            }

            /// Resize to `n_streams × n` (the content is reset when the shape changes),
            /// so that an `_into` method can reuse a caller-owned buffer.
            pub fn resize(&mut self, n_streams: usize, n: usize) {
                if self.0.dim() != (n_streams, n) {
                    self.0 = Array2::from_elem((n_streams, n), $zero);
                }
            }
        }
    };
}

container!(
    /// Hard bits, one row per stream; the values are 0 or 1.
    Bits, u8, 0
);
container!(
    /// Log-likelihood ratios, one row per stream (positive favours 0, see the module
    /// documentation).
    Llr, f64, 0.0
);
container!(
    /// Complex symbols at one sample per symbol, one row per stream.
    Symbols, C64, C64::new(0.0, 0.0)
);

/// A complex baseband waveform, `[n_streams, n_samples]`, with its sample rate.
#[derive(Clone, Debug, PartialEq)]
pub struct Signal {
    data: Symbols,
    sample_rate: Hertz,
}

impl Signal {
    /// A waveform from its samples (`[n_streams, n_samples]`) and its sample rate.
    pub fn new(data: Array2<C64>, sample_rate: Hertz) -> Signal {
        Signal { data: Symbols::new(data), sample_rate }
    }

    /// `n_streams` streams of `n_samples` zeros.
    pub fn zeros(n_streams: usize, n_samples: usize, sample_rate: Hertz) -> Signal {
        Signal { data: Symbols::zeros(n_streams, n_samples), sample_rate }
    }

    /// A single-stream waveform.
    ///
    /// # Errors
    /// Never in practice (see [`Symbols::from_stream`]).
    pub fn from_stream(x: Vec<C64>, sample_rate: Hertz) -> Result<Signal> {
        Ok(Signal { data: Symbols::from_stream(x)?, sample_rate })
    }

    /// A waveform from several streams of equal length.
    ///
    /// # Errors
    /// [`Error::Shape`] if the streams have different lengths.
    pub fn from_streams(streams: &[Vec<C64>], sample_rate: Hertz) -> Result<Signal> {
        Ok(Signal { data: Symbols::from_streams(streams)?, sample_rate })
    }

    /// Sample rate.
    pub fn sample_rate(&self) -> Hertz {
        self.sample_rate
    }

    /// Number of streams.
    pub fn n_streams(&self) -> usize {
        self.data.n_streams()
    }

    /// Number of samples per stream.
    pub fn n_samples(&self) -> usize {
        self.data.len()
    }

    /// Stream `i` as a slice.
    pub fn stream(&self, i: usize) -> &[C64] {
        self.data.stream_slice(i)
    }

    /// The streams, one vector each (the inverse of [`Signal::from_streams`]).
    ///
    /// # Examples
    /// ```
    /// use com_core::signal::Signal;
    /// use com_core::units::Hertz;
    /// use num_complex::Complex64 as C64;
    /// let streams = vec![vec![C64::new(1.0, 0.0); 3], vec![C64::new(0.0, 1.0); 3]];
    /// let x = Signal::from_streams(&streams, Hertz(1.0))?;
    /// assert_eq!(x.to_streams(), streams);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn to_streams(&self) -> Vec<Vec<C64>> {
        (0..self.n_streams()).map(|i| self.stream(i).to_vec()).collect()
    }

    /// The samples, `[n_streams, n_samples]`.
    pub fn as_array(&self) -> &Array2<C64> {
        self.data.as_array()
    }

    /// The samples, mutable.
    pub fn as_array_mut(&mut self) -> &mut Array2<C64> {
        self.data.as_array_mut()
    }

    /// Reshape to `n_streams × n_samples` at `sample_rate` (the content is reset when
    /// the shape changes), for the `_into` methods.
    pub fn reshape(&mut self, n_streams: usize, n_samples: usize, sample_rate: Hertz) {
        self.data.resize(n_streams, n_samples);
        self.sample_rate = sample_rate;
    }

    /// Make `self` the single stream `x` at `sample_rate`, reusing its storage when the
    /// length is unchanged (ADR 0011).
    pub fn set_stream(&mut self, x: &[C64], sample_rate: Hertz) {
        self.reshape(1, x.len(), sample_rate);
        self.data.as_slice_mut().copy_from_slice(x);
    }

    /// Make `self` a copy of `other`, reusing its storage when the shape is unchanged
    /// (ADR 0011).
    pub fn assign(&mut self, other: &Signal) {
        self.data.assign(other.data.as_array());
        self.sample_rate = other.sample_rate;
    }

    /// Make `self` the streams `streams` at `sample_rate`, reusing its storage when the
    /// shape is unchanged (ADR 0011).
    ///
    /// # Errors
    /// [`Error::Shape`] if the streams have different lengths (`self` is then left
    /// unchanged).
    pub fn set_streams(&mut self, streams: &[Vec<C64>], sample_rate: Hertz) -> Result<()> {
        let n = streams.first().map_or(0, Vec::len);
        for s in streams {
            same_len(s.len(), n, "streams")?;
        }
        self.reshape(streams.len(), n, sample_rate);
        for (i, x) in streams.iter().enumerate() {
            self.data.stream_slice_mut(i).copy_from_slice(x);
        }
        Ok(())
    }

    /// Fill `out` with `f(stream index, stream)` for every stream of `self`, at the
    /// same shape and sampling rate: the body of the `_into` methods of the blocks that
    /// process the streams independently.
    ///
    /// # Errors
    /// The first error of `f`; [`Error::Shape`] if `f` returns another number of
    /// samples than it was given, rather than leaving the samples of an earlier use of
    /// `out` in place.
    pub fn try_map_streams(
        &self,
        out: &mut Signal,
        mut f: impl FnMut(usize, &[C64]) -> Result<Vec<C64>>,
    ) -> Result<()> {
        out.reshape(self.n_streams(), self.n_samples(), self.sample_rate());
        for s in 0..self.n_streams() {
            let y = f(s, self.stream(s))?;
            expect_len(y.len(), self.n_samples(), "samples returned")?;
            for (o, v) in out.as_array_mut().row_mut(s).iter_mut().zip(y) {
                *o = v;
            }
        }
        Ok(())
    }

    /// [`Signal::try_map_streams`] for an `f` that cannot fail.
    ///
    /// # Panics
    /// If `f` returns another number of samples than it was given (a broken invariant
    /// of the calling block).
    pub fn map_streams(&self, out: &mut Signal, mut f: impl FnMut(usize, &[C64]) -> Vec<C64>) {
        if let Err(e) = self.try_map_streams(out, |s, x| Ok(f(s, x))) {
            panic!("map_streams: {e}");
        }
    }

    /// Mean power of every stream, `mean(|x|²)`.
    pub fn power(&self) -> Vec<f64> {
        let n = self.n_samples().max(1) as f64;
        (0..self.n_streams()).map(|i| self.stream(i).iter().map(C64::norm_sqr).sum::<f64>() / n).collect()
    }
}

/// A field in two polarisations, `x` and `y`, of equal length.
#[derive(Clone, Debug, PartialEq)]
pub struct DualPol {
    /// Polarisation x.
    pub x: Vec<C64>,
    /// Polarisation y.
    pub y: Vec<C64>,
}

impl DualPol {
    /// Two polarisations of equal length.
    ///
    /// # Errors
    /// [`Error::Shape`] if the lengths differ.
    pub fn new(x: Vec<C64>, y: Vec<C64>) -> Result<DualPol> {
        same_len(x.len(), y.len(), "polarisations")?;
        Ok(DualPol { x, y })
    }

    /// Number of samples per polarisation.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the field is empty.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Mean total power `mean(|x|² + |y|²)`.
    pub fn power(&self) -> f64 {
        let n = self.len().max(1) as f64;
        self.x.iter().chain(&self.y).map(C64::norm_sqr).sum::<f64>() / n
    }

    /// Multiply both polarisations by `s`.
    pub fn scale(&mut self, s: f64) {
        for v in self.x.iter_mut().chain(self.y.iter_mut()) {
            *v *= s;
        }
    }

    /// The two polarisations as a two-stream [`Signal`].
    ///
    /// # Errors
    /// Never for a valid `DualPol`.
    pub fn to_signal(&self, sample_rate: Hertz) -> Result<Signal> {
        Signal::from_streams(&[self.x.clone(), self.y.clone()], sample_rate)
    }

    /// The field of a two-stream [`Signal`]: `x` its first stream, `y` its second (the
    /// inverse of [`DualPol::to_signal`]).
    ///
    /// # Errors
    /// [`Error::Shape`] unless the signal has exactly two streams.
    pub fn from_signal(s: &Signal) -> Result<DualPol> {
        same_len(s.n_streams(), 2, "streams of a dual-polarisation field")?;
        Ok(DualPol { x: s.stream(0).to_vec(), y: s.stream(1).to_vec() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_setters_reuse_the_storage_of_an_unchanged_shape() {
        let mut out = Signal::zeros(1, 4, Hertz(1.0));
        let p = out.as_array().as_ptr();
        out.set_stream(&[C64::new(1.0, 0.0); 4], Hertz(2.0));
        assert_eq!((out.as_array().as_ptr(), out.sample_rate()), (p, Hertz(2.0)));
        out.set_streams(&[vec![C64::new(0.0, 1.0); 2], vec![C64::new(0.0, 2.0); 2]], Hertz(2.0)).unwrap();
        assert_eq!(out.stream(1), &[C64::new(0.0, 2.0); 2]);
        assert!(out.set_streams(&[vec![C64::new(0.0, 1.0); 2], vec![]], Hertz(2.0)).is_err());
        let mut b = Bits::zeros(2, 3);
        let p = b.as_slice().as_ptr();
        b.assign(&Array2::from_elem((2, 3), 1));
        b.stream_slice_mut(1)[0] = 0;
        assert_eq!((b.as_slice().as_ptr(), b.as_slice()), (p, &[1, 1, 1, 0, 1, 1][..]));
    }

    #[test]
    fn streams_and_reshape() {
        let s = Signal::from_streams(&[vec![C64::new(1.0, 0.0); 3], vec![C64::new(0.0, 2.0); 3]], Hertz(1.0)).unwrap();
        assert_eq!(s.power(), vec![1.0, 4.0]);
        assert!(Symbols::from_streams(&[vec![C64::new(0.0, 0.0); 2], vec![]]).is_err());
        let mut b = Bits::zeros(1, 2);
        b.resize(2, 5);
        assert_eq!((b.n_streams(), b.len(), b.as_slice().len()), (2, 5, 10));
        let d = DualPol::from_signal(&s).unwrap();
        assert_eq!(d.to_signal(Hertz(1.0)).unwrap(), s);
        assert!((d.power() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn try_map_streams_rejects_a_short_stream_instead_of_keeping_stale_samples() {
        let x = Signal::from_streams(&[vec![C64::new(1.0, 0.0); 4]], Hertz(1.0)).unwrap();
        // a reused buffer of the same shape, full of samples of an earlier block
        let mut out = Signal::from_streams(&[vec![C64::new(9.0, 9.0); 4]], Hertz(1.0)).unwrap();
        let err = x.try_map_streams(&mut out, |_, s| Ok(s[..3].to_vec())).unwrap_err();
        assert!(matches!(err, Error::Shape(_)), "{err}");
        assert!(x.try_map_streams(&mut out, |_, s| Ok([s, s].concat())).is_err());
        x.try_map_streams(&mut out, |_, s| Ok(s.iter().map(|v| v * 2.0).collect())).unwrap();
        assert_eq!(out.stream(0), &[C64::new(2.0, 0.0); 4]);
    }

    #[test]
    #[should_panic(expected = "map_streams")]
    fn map_streams_panics_on_a_short_stream() {
        let x = Signal::from_streams(&[vec![C64::new(1.0, 0.0); 4]], Hertz(1.0)).unwrap();
        let mut out = x.clone();
        x.map_streams(&mut out, |_, s| s[..2].to_vec());
    }
}
