//! # com-mapping — constellations, mapping and demapping
//!
//! (OptiCommPy `optic.comm.modulation` and `optic.comm.sources`.)
//!
//! | Module | Content |
//! |---|---|
//! | [`bicm`] | capacity and cutoff rate of coded modulation and bit-interleaved coded modulation, over AWGN and Rayleigh fading (Caire, Taricco & Biglieri 1998) |
//! | [`constellation`] | PAM, QAM, PSK and OOK constellations with Gray (or set-partitioning) labelling, mapping of bits to symbols, hard decisions |
//! | [`demapper`] | log-likelihood ratios (exact and max-log); the [`Map`](com_core::traits::Map) and [`Demap`](com_core::traits::Demap) traits |
//! | [`differential`] | differential coding of M-PSK (DBPSK, DQPSK, D8PSK): encoding and decoding of phase indices, noncoherent detection, Gray-labelled [`Dpsk`](differential::Dpsk) |
//! | [`imperfect_csi`] | bit error rates over Rayleigh fading with an imperfect channel estimate (BPSK, QPSK, 16-QAM) and of differential detection over Jakes' fading (Chang & Su 2002) |
//! | [`source`] | random bits, pseudo-random binary sequences (PRBS), random and shaped symbols |
//! | [`theory`] | closed-form error rates and information rates over AWGN |
//!
//! # Examples
//! ```
//! use com_mapping::{constellation::Constellation, source};
//! use com_core::rng::Rng;
//! let c = Constellation::qam(16).unwrap().normalized();
//! let bits = source::bits(400, &mut Rng::new(1));
//! let x = c.modulate(&bits).unwrap();
//! assert_eq!(x.len(), 100);
//! assert_eq!(c.demodulate(&x), bits);
//! ```

pub mod bicm;
pub mod constellation;
pub mod demapper;
pub mod differential;
pub mod imperfect_csi;
pub mod source;
pub mod theory;
