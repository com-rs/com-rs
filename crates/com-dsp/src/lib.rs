//! # com-dsp — waveform-agnostic signal processing
//!
//! Filtering, resampling, synchronisation, equalisation and compensation (OptiCommPy
//! `optic.dsp`). Nothing here assumes a specific waveform or standard: blocks that
//! need constellation points take them as a slice.
//!
//! | Module | Content |
//! |---|---|
//! | [`filter`] | pulse shapes (rectangular, NRZ, raised cosine, root-raised cosine), FIR filtering (`same` alignment), low-pass FIR design, moving average |
//! | [`nonlinear`] | a memoryless nonlinearity driven by a Gaussian signal: Bussgang's decomposition, clipping, output series and distortion spectrum (Banelli & Cacopardi 2000) |
//! | [`resample`] | up- and down-sampling, decimation, band-limited resampling, fractional delay |
//! | [`signal`] | normalisations, frequency shift and rotation, cyclic prefix |
//! | [`sync`] | carrier frequency and phase recovery, symbol timing (Gardner), frame synchronisation |
//! | [`estimation`] | least-squares and linear MMSE channel estimation, the frequency correlation of a power-delay profile |
//! | [`interpolation`] | interpolation between known positions: linear, quadratic, cubic spline, band-limited (MATLAB `interp`), cubic Lagrange (Farrow) |
//! | [`equalizer`] | the adaptive `n×n` butterfly equaliser (CMA, RDE, NLMS, DD-LMS) |
//! | [`cyclo`] | cyclic autocorrelation and blind symbol-rate estimation (cyclostationarity) |
//! | [`compensation`] | chromatic-dispersion compensation, digital back-propagation, digital predistortion |
//! | [`spectrum`] | spectral estimation: Welch's method, averaged periodogram |
//! | [`window`] | Hann and Dolph–Chebyshev windows |
//!
//! # Examples
//! ```
//! use com_dsp::filter::{pulse_shape, fir_filter, PulseShape};
//! use com_dsp::resample::upsample;
//! use com_math::C64;
//! let h = pulse_shape(PulseShape::Rrc { rolloff: 0.1 }, 4, 64)?;
//! assert!((h.iter().sum::<f64>() - 1.0).abs() < 1e-12);       // unit DC gain
//! let x = upsample(&[C64::new(1.0, 0.0); 8], 4);
//! assert_eq!(fir_filter(&h, &x).len(), x.len());
//! # Ok::<(), com_core::Error>(())
//! ```

pub mod compensation;
pub mod cyclo;
pub mod equalizer;
pub mod estimation;
pub mod filter;
pub mod interpolation;
pub mod nonlinear;
pub mod resample;
pub mod signal;
pub mod spectrum;
pub mod sync;
pub mod window;
