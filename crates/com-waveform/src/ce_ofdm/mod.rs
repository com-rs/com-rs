//! Constant-envelope OFDM (CE-OFDM): the modulator, the phase demodulator and the
//! error rate of its linear regime, after S. C. Thompson, A. U. Ahmed, J. G. Proakis,
//! J. R. Zeidler and M. J. Geile, "Constant envelope OFDM," *IEEE Trans. Commun.*
//! 56(8), 2008.
//!
//! # Model (matrix form)
//!
//! A block carries $K$ data symbols $\mathbf{d} \in \mathbb{C}^{K}$ (unit mean energy)
//! on the subcarriers $1..K$ of a DFT of size $N = 2J(K + 1)$, with $J \ge 1$ the
//! **oversampling factor** ($J = 1$ is the smallest DFT that keeps the bins $0$ and
//! $K + 1$ empty). With $\mathbf{F} \in \mathbb{C}^{N \times K}$,
//! $F_{nk} = e^{j2\pi kn/N}$, the real OFDM signal and the transmitted samples are
//!
//! $$\mathbf{x} = \frac{\mathbf{F}\mathbf{d} + \mathbf{F}^{*}\mathbf{d}^{*}}{\sqrt{2K}}\,,
//! \qquad s_{n} = e^{j 2\pi h x_{n}}\,,$$
//!
//! so that $\mathbb{E}[\mathbf{x}\mathbf{x}^{T}]$ is circulant with a unit diagonal and
//! the phase $\varphi = 2\pi h x$ has the standard deviation $2\pi h$, $h$ being the
//! **modulation index** (Sec. II of the paper). An optional cyclic prefix of
//! $N_{\mathrm{cp}}$ samples is prepended. The channel adds
//! $\mathbf{w} \sim \mathcal{CN}(0, \sigma^{2}\mathbf{I})$.
//!
//! The receiver (Sec. III, Fig. 1 of the paper) takes the phase $\theta = \arg r$,
//! optionally unwraps it along the stream ([`Unwrap::Stream`], the phase unwrapper of
//! the paper), removes the prefix and projects on the data bins:
//!
//! $$\hat{\mathbf{d}} = \frac{\mathbf{G}\boldsymbol{\psi}}{2\pi h}\,, \qquad
//! \mathbf{G} = \frac{\sqrt{2K}}{N}\mathbf{F}^{H}$$
//!
//! ($\mathbf{G}\mathbf{x} = \mathbf{d}$ because $\mathbf{F}^{H}\mathbf{F} = N\mathbf{I}$
//! and $\mathbf{F}^{H}\mathbf{F}^{*} = \mathbf{0}$).
//!
//! # SNR convention
//!
//! The SNR is $E_{s}/N_{0}$ **per data symbol**: a block of energy $N + N_{\mathrm{cp}}$
//! (unit envelope) carries $K$ symbols, so
//! $\sigma^{2} = (N + N_{\mathrm{cp}})/(K \cdot E_{s}/N_{0})$. The results of the linear
//! regime are then independent of the oversampling factor.
//!
//! # Theory of the linear regime
//!
//! Above the threshold of the phase demodulator (a carrier-to-noise ratio of about
//! 10 dB, Sec. III-B of the paper), the phase error
//! $\varepsilon = \operatorname{Im}(w e^{-j\varphi}) \sim \mathcal{N}(0, \sigma^{2}/2)$
//! and $\hat{\mathbf{d}} = \mathbf{d} + \mathbf{G}\boldsymbol{\varepsilon}/(2\pi h)$, so
//! every data subcarrier sees the SNR
//!
//! $$\mathrm{SNR} = (2\pi h)^{2}\,\frac{E_{s}}{N_{0}}\,\frac{N}{N + N_{\mathrm{cp}}}\,:$$
//!
//! the error rates are those of the constellation over AWGN with the SNR multiplied by
//! $(2\pi h)^{2}$ ([`theory::snr`], [`theory::ser_esn0`], [`theory::ber_esn0`]). For $M$-PAM data
//! on every real subcarrier this is eq. (15) of the paper, which keeps the
//! nearest-neighbour term of the Gray BER; the library uses the exact Gray BER of the
//! square QAM whose real and imaginary parts are the two PAM symbols (checked against
//! eq. (15) by the test `literature`). Below the threshold the demodulator fails and
//! the error rate stays above this theory, as in Fig. 2 of the paper.
//!
//! # Receivers
//!
//! [`CeOfdm::demodulate_with`] takes a [`Receiver`]: the arctangent phase demodulator
//! above, or the low-complexity receivers of Ahmed and Zeidler (2015), which replace
//! $\arg r$ by $\operatorname{Im} r$ (basic linear receiver) or by
//! $\operatorname{Im} r \\, (2 - \operatorname{Re} r)$ (enhanced receiver) and avoid the
//! threshold and the wraps of the phase at the price of a distortion. [`gamp::Gamp`] is
//! the near-optimum receiver of López Morales et al. (2022), by message passing, which
//! decodes beyond $2\pi h \approx 0.7$ where the phase leaves $(-\pi, \pi]$.
//! [`theory`] gives the error rates of the receivers, the spectrum, the optimum
//! asymptotic gain and the effect of a cycle slip.
//!
//! # Examples
//! ```
//! use com_waveform::ce_ofdm::{CeOfdm, Receiver, Unwrap};
//! use com_mapping::constellation::Constellation;
//! use com_core::rng::Rng;
//! let ce = CeOfdm::new(16, 2, 0.1).unwrap();
//! let c = Constellation::qam(16).unwrap().normalized();
//! let (_, d) = com_mapping::source::symbols(&c, 64, None, &mut Rng::new(1)).unwrap();
//! let s = ce.modulate(&d).unwrap();
//! assert_eq!(s.len(), 4 * ce.n_fft());
//! assert!(s.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));          // constant envelope
//! let e = ce.demodulate_with_receiver(&s, Receiver::Arctangent(Unwrap::None)).unwrap();
//! assert!(e.iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-9));
//! ```

use crate::ofdm::{check_input, check_rate};
use com_core::signal::{Signal, Symbols};
use com_core::traits::{Demodulate, Modulate};
use com_core::units::Hertz;
use com_core::{Result, ensure, ensure_shape};
use com_dsp::signal::{extend_with_prefix, strip_prefix};
use com_dsp::sync::carrier;
use com_math::fft;
use com_math::{C64, cis, db2lin};
use std::f64::consts::TAU;

/// The phase processing of the receiver before the DFT.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unwrap {
    /// The principal phase `arg r ∈ (-π, π]` (phase wraps are errors).
    None,
    /// The phase unwrapped along the whole stream, the phase unwrapper of Thompson et
    /// al. (jumps of the increments above `π` are errors).
    Stream,
}

impl std::str::FromStr for Unwrap {
    type Err = com_core::Error;

    /// `none` or `stream` (case-insensitive).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for another name.
    fn from_str(s: &str) -> Result<Unwrap> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(Unwrap::None),
            "stream" => Ok(Unwrap::Stream),
            _ => Err(com_core::Error::Config(format!("unwrap must be none or stream, got {s:?}"))),
        }
    }
}

/// The receiver: how the samples become a proxy of the phase before the DFT.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Receiver {
    /// The arctangent phase demodulator, $\arg r$, with the given unwrapping (Thompson et
    /// al. 2008, Fig. 1; Ahmed et al. 2007, Fig. 6).
    Arctangent(Unwrap),
    /// The basic linear receiver of A. U. Ahmed and J. R. Zeidler, "Novel low-complexity
    /// receivers for constant envelope OFDM," *IEEE Trans. Signal Process.* 63(17), 2015,
    /// Fig. 4: the quadrature component $\operatorname{Im} r = \sin\varphi + \ldots$,
    /// the first term of its Taylor series in $\varphi$.
    Linear,
    /// The enhanced receiver of Ahmed and Zeidler (2015), Fig. 5 and eq. (8):
    /// $\operatorname{Im} r \\, (2 - \operatorname{Re} r) = \varphi + \varphi^{3}/3 + \ldots$,
    /// whose cubic term adds to the signal.
    Enhanced,
}

impl Default for Receiver {
    /// The arctangent demodulator of the principal phase.
    fn default() -> Receiver {
        Receiver::Arctangent(Unwrap::None)
    }
}

impl std::str::FromStr for Receiver {
    type Err = com_core::Error;

    /// `arctangent` (with the principal phase), `linear` or `enhanced`
    /// (case-insensitive).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for another name.
    fn from_str(s: &str) -> Result<Receiver> {
        match s.to_ascii_lowercase().as_str() {
            "arctangent" | "atan" => Ok(Receiver::Arctangent(Unwrap::None)),
            "linear" => Ok(Receiver::Linear),
            "enhanced" => Ok(Receiver::Enhanced),
            _ => {
                Err(com_core::Error::Config(format!("the receiver must be arctangent, linear or enhanced, got {s:?}")))
            }
        }
    }
}

/// A CE-OFDM link (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct CeOfdm {
    /// Data subcarriers `K` per block.
    pub n_subcarriers: usize,
    /// Oversampling factor `J`: the DFT has `N = 2J(K + 1)` points.
    pub oversampling: usize,
    /// Modulation index `h` (phase standard deviation `2πh`).
    pub h: f64,
    /// Cyclic prefix, in samples of the DFT of $N$ points (the oversampling is already in
    /// $N$, unlike [`crate::ofdm::Ofdm::n_cp`], which counts Nyquist-rate samples).
    pub n_cp: usize,
    /// The receiver of [`Demodulate`] ([`CeOfdm::demodulate_with`]); the arctangent
    /// demodulator of the principal phase by default.
    pub receiver: Receiver,
    /// The rate of the samples (one per point of the DFT of $N$): the rate of the
    /// [`Signal`] of [`Modulate`] and the rate [`Demodulate`] expects; 1 Hz by default
    /// (time in samples).
    pub sample_rate: Hertz,
}

impl CeOfdm {
    /// The DFT size `N = 2J(K + 1)`.
    pub fn n_fft(&self) -> usize {
        2 * self.oversampling * (self.n_subcarriers + 1)
    }

    /// Samples per block, `N + N_cp`.
    pub fn block_len(&self) -> usize {
        self.n_fft() + self.n_cp
    }

    /// Samples per data symbol, `(N + N_cp)/K`.
    pub fn sps(&self) -> f64 {
        self.block_len() as f64 / self.n_subcarriers as f64
    }

    /// Noise variance per sample for an `E_s/N_0` (dB) per data symbol and a unit
    /// envelope: `σ² = (N + N_cp)/(K E_s/N_0)`.
    pub fn noise_variance(&self, esn0_db: f64) -> f64 {
        self.sps() / db2lin(esn0_db)
    }

    /// `K` data subcarriers, oversampling `J`, modulation index `h`, no prefix, the default
    /// receiver, at the sample rate 1 Hz.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters (see [`CeOfdm::check`]).
    ///
    /// # Examples
    /// ```
    /// use com_waveform::ce_ofdm::CeOfdm;
    /// let ce = CeOfdm { n_cp: 8, ..CeOfdm::new(16, 2, 0.1)? };
    /// assert_eq!(ce.block_len(), 2 * 2 * 17 + 8);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn new(n_subcarriers: usize, oversampling: usize, h: f64) -> Result<CeOfdm> {
        let ce =
            CeOfdm { n_subcarriers, oversampling, h, n_cp: 0, receiver: Receiver::default(), sample_rate: Hertz(1.0) };
        ce.check()?;
        Ok(ce)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless `K ≥ 1`, `J ≥ 1`, `h > 0`, the prefix is at
    /// most one block of `N` samples and the sample rate is positive and finite.
    pub fn check(&self) -> Result<()> {
        ensure(self.n_subcarriers >= 1 && self.oversampling >= 1, || "CE-OFDM needs K ≥ 1 and J ≥ 1".into())?;
        ensure(self.h > 0.0 && self.h.is_finite(), || {
            format!("the modulation index must be positive, got {}", self.h)
        })?;
        ensure(self.n_cp <= self.n_fft(), || {
            format!("a prefix of {} samples is longer than the block of {}", self.n_cp, self.n_fft())
        })?;
        check_rate(self.sample_rate)
    }

    /// The real OFDM signal `x = (F d + F* d*)/√(2K)` of every block (no prefix).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters, [`com_core::Error::Shape`] for
    /// a number of symbols that is not a multiple of `K`.
    pub fn ofdm_signal(&self, d: &[C64]) -> Result<Vec<f64>> {
        self.check()?;
        let (k, n) = (self.n_subcarriers, self.n_fft());
        ensure_shape(d.len() % k == 0, || format!("{} symbols is not a multiple of K = {k}", d.len()))?;
        let mut out = Vec::with_capacity(d.len() / k * n);
        for block in d.chunks(k) {
            out.extend(self.synthesize(block));
        }
        Ok(out)
    }

    /// The real OFDM signal `x = (F d + F* d*)/√(2K)` of one block of `K` symbols.
    fn synthesize(&self, d: &[C64]) -> Vec<f64> {
        let (k, n) = (self.n_subcarriers, self.n_fft());
        let mut spec = vec![C64::new(0.0, 0.0); n];
        for (i, &v) in d.iter().enumerate() {
            spec[i + 1] = v;
            spec[n - 1 - i] = v.conj();
        }
        // ifft divides by N: x = N/√(2K) · ifft
        let scale = n as f64 / (2.0 * k as f64).sqrt();
        fft::ifft(&spec).into_iter().map(|v| v.re * scale).collect()
    }

    /// The DFT bins `1 … K` of one block of `N` real samples: `F^H x` on the data
    /// subcarriers.
    fn data_bins(&self, x: &[f64]) -> Vec<C64> {
        let spec = fft::fft(&x.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>());
        spec[1..=self.n_subcarriers].to_vec()
    }

    /// The transmitted samples `exp(j2πh x)`, with the cyclic prefix of every block.
    ///
    /// # Errors
    /// As [`CeOfdm::ofdm_signal`].
    pub fn modulate(&self, d: &[C64]) -> Result<Vec<C64>> {
        let x = self.ofdm_signal(d)?;
        let n = self.n_fft();
        let g = TAU * self.h;
        let mut out = Vec::with_capacity(x.len() / n * self.block_len());
        for block in x.chunks(n) {
            let s: Vec<C64> = block.iter().map(|&v| cis(g * v)).collect();
            extend_with_prefix(&mut out, &s, self.n_cp)?;
        }
        Ok(out)
    }

    /// The symbol estimates of every block with the receiver: the projection of its proxy
    /// of the phase on the data bins, divided by $2\pi h$ and by Bussgang's gain of the
    /// receiver ([`theory::receiver_gain`]), so that the estimates are unbiased. With
    /// [`Receiver::Arctangent`], the estimates `d̂ = G ψ/(2πh)` of the arctangent
    /// demodulator with its unwrapping.
    ///
    /// # Errors
    /// As [`CeOfdm::demodulate`].
    ///
    /// # Examples
    /// ```
    /// use com_waveform::ce_ofdm::{CeOfdm, Receiver};
    /// use com_mapping::constellation::Constellation;
    /// use com_core::rng::Rng;
    /// let ce = CeOfdm::new(64, 2, 0.02).unwrap();
    /// let c = Constellation::qam(16).unwrap().normalized();
    /// let (_, d) = com_mapping::source::symbols(&c, 64, None, &mut Rng::new(1)).unwrap();
    /// // without noise the linear receivers err only by their distortion, small at a small h
    /// let e = ce.demodulate_with_receiver(&ce.modulate(&d).unwrap(), Receiver::Enhanced).unwrap();
    /// assert_eq!(c.decide(&e), c.decide(&d));
    /// ```
    pub fn demodulate_with_receiver(&self, r: &[C64], receiver: Receiver) -> Result<Vec<C64>> {
        self.check_block(r.len())?;
        let (proxy, gain): (Vec<f64>, f64) = match receiver {
            Receiver::Arctangent(unwrap) => (self.received_phase(r, unwrap), 1.0),
            Receiver::Linear => (r.iter().map(|v| v.im).collect(), theory::receiver_gain(self, receiver)),
            Receiver::Enhanced => {
                (r.iter().map(|v| v.im * (2.0 - v.re)).collect(), theory::receiver_gain(self, receiver))
            }
        };
        self.project_checked(&proxy, gain)
    }

    /// `Ok` for a valid configuration and `len` samples of whole blocks.
    fn check_block(&self, len: usize) -> Result<()> {
        self.check()?;
        let l = self.block_len();
        ensure_shape(len % l == 0, || format!("{len} samples is not a multiple of N + N_cp = {l}"))
    }

    /// The projection `G ψ/(2πh g)` of the per-sample proxy `ψ` of every block.
    #[cfg(test)]
    fn project(&self, psi: &[f64], gain: f64) -> Result<Vec<C64>> {
        self.check_block(psi.len())?;
        self.project_checked(psi, gain)
    }

    /// [`CeOfdm::project`] of a configuration and a length already checked.
    fn project_checked(&self, psi: &[f64], gain: f64) -> Result<Vec<C64>> {
        let (k, n, l) = (self.n_subcarriers, self.n_fft(), self.block_len());
        let scale = (2.0 * k as f64).sqrt() / (n as f64 * TAU * self.h * gain);
        let mut out = Vec::with_capacity(psi.len() / l * k);
        for block in psi.chunks(l) {
            out.extend(self.data_bins(strip_prefix(block, self.n_cp)?).into_iter().map(|v| v * scale));
        }
        Ok(out)
    }

    /// The phase of the receiver before the DFT, prefix included: the principal phase
    /// `arg r` or the phase unwrapped along the stream.
    pub fn received_phase(&self, r: &[C64], unwrap: Unwrap) -> Vec<f64> {
        let theta: Vec<f64> = r.iter().map(|v| v.arg()).collect();
        match unwrap {
            Unwrap::None => theta,
            Unwrap::Stream => carrier::unwrap(&theta, TAU),
        }
    }
}

/// CE-OFDM as a [`Modulate`] block: one stream of data symbols into one stream of
/// samples at [`CeOfdm::sample_rate`].
impl Modulate for CeOfdm {
    fn modulate_into(&self, x: &Symbols, out: &mut Signal) -> Result<()> {
        ensure(x.n_streams() == 1, || format!("CE-OFDM modulates one stream, not {}", x.n_streams()))?;
        out.set_stream(&CeOfdm::modulate(self, x.stream(0))?, self.sample_rate);
        Ok(())
    }
}

/// CE-OFDM as a [`Demodulate`] block: one stream of samples at [`CeOfdm::sample_rate`],
/// demodulated by [`CeOfdm::receiver`].
impl Demodulate for CeOfdm {
    /// # Errors
    /// [`com_core::Error::Config`] for several streams or a signal at another sample rate,
    /// and the errors of [`CeOfdm::demodulate_with`].
    fn demodulate_into(&mut self, y: &Signal, out: &mut Symbols) -> Result<()> {
        check_input(y, self.sample_rate, "CE-OFDM")?;
        out.set_stream(&self.demodulate_with_receiver(y.stream(0), self.receiver)?);
        Ok(())
    }
}

pub mod gamp;
pub mod theory;

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use com_mapping::constellation::{Constellation, ConstellationKind};

    #[test]
    fn the_traits_use_the_receiver_and_the_rate_of_the_configuration() {
        let c = Constellation::qam(4).unwrap().normalized();
        let (_, d) = com_mapping::source::symbols(&c, 32 * 4, None, &mut Rng::new(3)).unwrap();
        let mut ce = CeOfdm {
            n_cp: 4,
            receiver: Receiver::Enhanced,
            sample_rate: Hertz(2e6),
            ..CeOfdm::new(32, 2, 0.02).unwrap()
        };
        let x = Modulate::modulate(&ce, &Symbols::from_stream(d.clone()).unwrap()).unwrap();
        assert_eq!((x.stream(0), x.sample_rate()), (&ce.modulate(&d).unwrap()[..], Hertz(2e6)));
        let back = Demodulate::demodulate(&mut ce, &x).unwrap();
        assert_eq!(back.as_slice(), &ce.demodulate_with_receiver(x.stream(0), Receiver::Enhanced).unwrap()[..]);
        let slow = Signal::from_stream(x.stream(0).to_vec(), Hertz(1e6)).unwrap();
        assert!(Demodulate::demodulate(&mut ce, &slow).is_err());
    }

    #[test]
    fn the_projection_inverts_the_ofdm_signal_and_the_phase_variance_is_2pi_h() {
        // h = 0.6: the phase (standard deviation 3.8 rad) wraps often, but at J = 16 its
        // increments stay far below π
        let ce = CeOfdm { n_cp: 5, ..CeOfdm::new(32, 16, 0.6).unwrap() };
        let c = Constellation::qam(16).unwrap().normalized();
        let (_, d) = com_mapping::source::symbols(&c, 32 * 200, None, &mut Rng::new(2)).unwrap();
        let x = ce.ofdm_signal(&d).unwrap();
        let var = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        assert!((var - 1.0).abs() < 0.03, "{var}");
        let s = ce.modulate(&d).unwrap();
        assert_eq!(s.len(), 200 * ce.block_len());
        // the prefix repeats the end of the block
        assert!((s[0] - s[ce.n_fft()]).norm() < 1e-12);
        // without noise, the unwrapped receiver recovers the symbols despite the wraps,
        // which the wrapped receiver does not
        let e = ce.demodulate_with_receiver(&s, Receiver::Arctangent(Unwrap::Stream)).unwrap();
        assert!(e.iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-8));
        let w = ce.demodulate_with_receiver(&s, Receiver::Arctangent(Unwrap::None)).unwrap();
        assert!(w.iter().zip(&d).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max) > 0.1);
    }

    #[test]
    fn the_linear_theory_is_proakis_with_the_snr_scaled_by_2pi_h_squared() {
        let ce = CeOfdm::new(32, 4, 0.05).unwrap();
        let g = (TAU * 0.05).powi(2) * 100.0;
        let s = theory::snr(&ce, 20.0).unwrap();
        assert!((s - g).abs() < 1e-9 * g);
        let ser_ce = theory::ser_esn0(&ce, ConstellationKind::Qam, 16, 20.0).unwrap();
        let proakis = com_mapping::theory::ser(ConstellationKind::Qam, 16, com_math::lin2db(g / 4.0)).unwrap();
        assert!((ser_ce - proakis).abs() < 1e-12);
        // the cyclic prefix costs its share of the energy
        let cp = CeOfdm { n_cp: ce.n_fft() / 4, ..ce.clone() };
        assert!((theory::snr(&cp, 20.0).unwrap() / s - 0.8).abs() < 1e-12);
        // a simulated link at high CNR follows the theory within the sampling error
        let c = Constellation::qam(16).unwrap().normalized();
        let (_, d) = com_mapping::source::symbols(&c, 32 * 2000, None, &mut Rng::new(5)).unwrap();
        let mut rng = Rng::new(6);
        // J = 2 keeps the CNR per sample 14 dB, above the threshold of the demodulator;
        // the SNR per subcarrier is 16 dB, a SER of 4e-3 for 16-QAM (about 250 errors)
        let ce = CeOfdm::new(32, 2, 0.1).unwrap();
        let esn0_db = 20.0;
        let sigma = ce.noise_variance(esn0_db).sqrt();
        let r: Vec<C64> = ce.modulate(&d).unwrap().into_iter().map(|v| v + rng.complex_gaussian() * sigma).collect();
        let e = ce.demodulate_with_receiver(&r, Receiver::Arctangent(Unwrap::None)).unwrap();
        let errors = c.decide(&e).iter().zip(c.decide(&d)).filter(|(a, b)| *a != b).count();
        let ser = errors as f64 / d.len() as f64;
        let want = theory::ser_esn0(&ce, ConstellationKind::Qam, 16, esn0_db).unwrap();
        assert!((ser / want - 1.0).abs() < 0.15, "{ser} vs {want}");
    }

    #[test]
    fn a_cycle_slip_adds_the_response_of_the_theory() {
        let ce = CeOfdm { n_cp: 6, ..CeOfdm::new(31, 2, 0.8 / TAU).unwrap() };
        let c = Constellation::qam(4).unwrap().normalized();
        let (_, d) = com_mapping::source::symbols(&c, 31, None, &mut Rng::new(4)).unwrap();
        let s = ce.modulate(&d).unwrap();
        let clean = ce.demodulate_with_receiver(&s, Receiver::Arctangent(Unwrap::Stream)).unwrap();
        for start in [0usize, 1, 40, 127] {
            // the phase gains 2π from `start`: the unwrapped phase of the samples rotated by a
            // full turn in small steps over one sample would do the same; add it to the phase
            let mut psi = ce.received_phase(&s, Unwrap::Stream);
            for v in &mut psi[ce.n_cp + start..] {
                *v += TAU;
            }
            let slipped = ce.project(&psi, 1.0).unwrap();
            let want = theory::cycle_slip_response(&ce, start).unwrap();
            for ((a, b), w) in slipped.iter().zip(&clean).zip(&want) {
                assert!((a - b - w).norm() < 1e-9, "{start}: {} vs {w}", a - b);
            }
        }
        assert!(theory::cycle_slip_response(&ce, ce.n_fft()).is_err());
    }

    #[test]
    fn the_linear_receivers_follow_the_bussgang_analysis() {
        // QPSK, K = 32 and J = 2 (the 2X sampling of Ahmed and Zeidler 2015)
        let c = Constellation::qam(4).unwrap().normalized();
        let mut rng = Rng::new(9);
        let (_, d) = com_mapping::source::symbols(&c, 32 * 3000, None, &mut rng).unwrap();
        for (two_pi_h, ebn0_db) in [(0.3, 15.0), (0.5, 10.5), (0.7, 8.0)] {
            let ce = CeOfdm::new(32, 2, two_pi_h / TAU).unwrap();
            let esn0_db = ebn0_db + com_math::lin2db(2.0);
            let sigma = ce.noise_variance(esn0_db).sqrt();
            let r: Vec<C64> =
                ce.modulate(&d).unwrap().into_iter().map(|v| v + rng.complex_gaussian() * sigma).collect();
            for receiver in [Receiver::Linear, Receiver::Enhanced] {
                let e = ce.demodulate_with_receiver(&r, receiver).unwrap();
                let bits: usize =
                    c.decide(&e).iter().zip(c.decide(&d)).map(|(a, b)| (a ^ b).count_ones() as usize).sum();
                let ber = bits as f64 / (2.0 * d.len() as f64);
                let want = theory::ber_receiver(&CeOfdm { receiver, ..ce.clone() }, ConstellationKind::Qam, 4, esn0_db)
                    .unwrap();
                // BER about 1e-2, 2000 errors: a 10 % band for the sampling error and the
                // Gaussian model of the distortion (simulation and theory agree to 4 %)
                assert!((ber / want - 1.0).abs() < 0.1, "2πh = {two_pi_h}, {receiver:?}: {ber:e} vs {want:e}");
            }
        }
        // at a small index the linear receiver is the arctangent one (eq. (20) of the paper)
        let ce = CeOfdm::new(32, 2, 0.01 / TAU).unwrap();
        let lin = theory::subcarrier_snr(&CeOfdm { receiver: Receiver::Linear, ..ce.clone() }, 30.0).unwrap();
        let flat = theory::snr(&ce, 30.0).unwrap();
        assert!(lin.iter().all(|s| (s / flat - 1.0).abs() < 1e-3), "{lin:?} vs {flat}");
    }
}

#[cfg(test)]
mod literature {
    use super::{CeOfdm, theory};
    use com_mapping::constellation::ConstellationKind;
    use com_math::{db2lin, qfunc};

    /// S. C. Thompson, A. U. Ahmed, J. G. Proakis, J. R. Zeidler and M. J. Geile,
    /// "Constant envelope OFDM," IEEE Trans. Commun. 56(8), 2008, eq. (15) (p. 1303):
    /// the BER of the linear regime, `2 (M-1)/(M log2 M) Q(2πh √(6 log2 M E_b/N_0 /
    /// (M²-1)))` with M-PAM per real subcarrier, which the paper calls a lower bound (it
    /// keeps the nearest-neighbour term of the Gray BER). The library's theory is the
    /// exact Gray BER of the QAM of order M² at `E_s/N_0 = 2 log2 M · E_b/N_0`: never
    /// below eq. (15), and within 1 % of it once the BER is under 1e-2.
    #[test]
    fn thompson_2008_eq_15_linear_ber() {
        for (m_pam, m_qam) in [(4usize, 16usize), (8, 64)] {
            let log2m = (m_pam as f64).log2();
            for h in [0.05, 0.1, 0.3] {
                for ebn0_db in [6.0, 10.0, 14.0] {
                    let ce = CeOfdm::new(64, 4, h).unwrap();
                    let esn0_db = ebn0_db + com_math::lin2db(2.0 * log2m);
                    let lib = theory::ber_esn0(&ce, ConstellationKind::Qam, m_qam, esn0_db).unwrap();
                    let arg = std::f64::consts::TAU
                        * h
                        * (6.0 * log2m / ((m_pam * m_pam - 1) as f64) * db2lin(ebn0_db)).sqrt();
                    let paper = 2.0 * (m_pam - 1) as f64 / (m_pam as f64 * log2m) * qfunc(arg);
                    assert!(lib >= paper * (1.0 - 1e-12), "M = {m_pam}, h = {h}: {lib} below the bound {paper}");
                    if paper < 1e-2 {
                        assert!((lib / paper - 1.0).abs() < 0.01, "M = {m_pam}, h = {h}: {lib} vs {paper}");
                    }
                }
            }
        }
    }
}
