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
//! $(2\pi h)^{2}$ ([`theory::snr`], [`theory::ser`], [`theory::ber`]). For $M$-PAM data
//! on every real subcarrier this is eq. (15) of the paper, which keeps the
//! nearest-neighbour term of the Gray BER; the library uses the exact Gray BER of the
//! square QAM whose real and imaginary parts are the two PAM symbols (checked against
//! eq. (15) by the test `literature`). Below the threshold the demodulator fails and
//! the error rate stays above this theory, as in Fig. 2 of the paper.
//!
//! # Examples
//! ```
//! use com_waveform::ce_ofdm::{CeOfdm, Unwrap};
//! use com_mapping::constellation::Constellation;
//! use com_core::rng::Rng;
//! let ce = CeOfdm { n_data: 16, oversampling: 2, h: 0.1, n_cp: 0 };
//! let c = Constellation::qam(16).unwrap().normalized();
//! let (_, d) = com_mapping::source::symbols(&c, 64, None, &mut Rng::new(1)).unwrap();
//! let s = ce.modulate(&d).unwrap();
//! assert_eq!(s.len(), 4 * ce.n_fft());
//! assert!(s.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));          // constant envelope
//! let e = ce.demodulate(&s, Unwrap::None).unwrap();
//! assert!(e.iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-9));
//! ```

use com_core::{Result, ensure, ensure_shape};
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

/// A CE-OFDM link (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct CeOfdm {
    /// Data subcarriers `K` per block.
    pub n_data: usize,
    /// Oversampling factor `J`: the DFT has `N = 2J(K + 1)` points.
    pub oversampling: usize,
    /// Modulation index `h` (phase standard deviation `2πh`).
    pub h: f64,
    /// Cyclic prefix (samples).
    pub n_cp: usize,
}

impl CeOfdm {
    /// The DFT size `N = 2J(K + 1)`.
    pub fn n_fft(&self) -> usize {
        2 * self.oversampling * (self.n_data + 1)
    }

    /// Samples per block, `N + N_cp`.
    pub fn block_len(&self) -> usize {
        self.n_fft() + self.n_cp
    }

    /// Samples per data symbol, `(N + N_cp)/K`.
    pub fn samples_per_symbol(&self) -> f64 {
        self.block_len() as f64 / self.n_data as f64
    }

    /// Noise variance per sample for an `E_s/N_0` (dB) per data symbol and a unit
    /// envelope: `σ² = (N + N_cp)/(K E_s/N_0)`.
    pub fn noise_variance(&self, esn0_db: f64) -> f64 {
        self.samples_per_symbol() / db2lin(esn0_db)
    }

    fn check(&self) -> Result<()> {
        ensure(self.n_data >= 1 && self.oversampling >= 1, || "CE-OFDM needs K ≥ 1 and J ≥ 1".into())?;
        ensure(self.h > 0.0 && self.h.is_finite(), || format!("the modulation index must be positive, got {}", self.h))
    }

    /// The real OFDM signal `x = (F d + F* d*)/√(2K)` of every block (no prefix).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters, [`com_core::Error::Shape`] for
    /// a number of symbols that is not a multiple of `K`.
    pub fn ofdm_signal(&self, d: &[C64]) -> Result<Vec<f64>> {
        self.check()?;
        let (k, n) = (self.n_data, self.n_fft());
        ensure_shape(d.len() % k == 0, || format!("{} symbols is not a multiple of K = {k}", d.len()))?;
        let scale = n as f64 / (2.0 * k as f64).sqrt();
        let mut out = Vec::with_capacity(d.len() / k * n);
        let mut spec = vec![C64::new(0.0, 0.0); n];
        for block in d.chunks(k) {
            for v in &mut spec {
                *v = C64::new(0.0, 0.0);
            }
            for (i, &v) in block.iter().enumerate() {
                spec[i + 1] = v;
                spec[n - 1 - i] = v.conj();
            }
            // ifft divides by N: x = N/√(2K) · ifft
            out.extend(fft::ifft(&spec).into_iter().map(|v| v.re * scale));
        }
        Ok(out)
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
            out.extend(block[n - self.n_cp..].iter().map(|&v| cis(g * v)));
            out.extend(block.iter().map(|&v| cis(g * v)));
        }
        Ok(out)
    }

    /// The symbol estimates `d̂ = G ψ/(2πh)` of every block.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters, [`com_core::Error::Shape`] for
    /// a number of samples that is not a multiple of `N + N_cp`.
    pub fn demodulate(&self, r: &[C64], unwrap: Unwrap) -> Result<Vec<C64>> {
        self.check()?;
        let (k, n, l) = (self.n_data, self.n_fft(), self.block_len());
        ensure_shape(r.len() % l == 0, || format!("{} samples is not a multiple of N + N_cp = {l}", r.len()))?;
        let psi = self.received_phase(r, unwrap);
        let scale = (2.0 * k as f64).sqrt() / (n as f64 * TAU * self.h);
        let mut out = Vec::with_capacity(r.len() / l * k);
        for block in psi.chunks(l) {
            let spec = fft::fft(&block[self.n_cp..].iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>());
            out.extend(spec[1..=k].iter().map(|v| v * scale));
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

/// The error rates of the linear regime (see the module documentation).
pub mod theory {
    use super::{CeOfdm, TAU};
    use com_core::Result;
    use com_mapping::constellation::ConstellationKind;

    /// The SNR of every data subcarrier (linear scale) at `E_s/N_0` (dB) per data
    /// symbol: `(2πh)² E_s/N_0 · N/(N + N_cp)`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters.
    pub fn snr(ce: &CeOfdm, esn0_db: f64) -> Result<f64> {
        ce.check()?;
        let (k, n) = (ce.n_data, ce.n_fft());
        Ok((TAU * ce.h).powi(2) * n as f64 / (k as f64 * ce.noise_variance(esn0_db)))
    }

    /// `E_b/N_0` (dB) of a constellation of `m` points at the SNR of the linear regime.
    fn ebn0_db(ce: &CeOfdm, m: usize, esn0_db: f64) -> Result<f64> {
        Ok(10.0 * (snr(ce, esn0_db)? / (m as f64).log2()).max(1e-300).log10())
    }

    /// The SER of a constellation of `m` points of `kind` over AWGN at the SNR of the
    /// linear regime ([`com_mapping::theory::ser`]).
    ///
    /// # Errors
    /// As [`snr`] and [`com_mapping::theory::ser`].
    pub fn ser(ce: &CeOfdm, m: usize, esn0_db: f64, kind: ConstellationKind) -> Result<f64> {
        com_mapping::theory::ser(kind, m, ebn0_db(ce, m, esn0_db)?)
    }

    /// The BER of a Gray-mapped constellation of `m` points of `kind` over AWGN at the
    /// SNR of the linear regime: the exact Gray BER of a square QAM
    /// ([`com_mapping::theory::ber_gray_qam`]), eq. (15) of Thompson et al. with its
    /// further neighbours, else the nearest-neighbour approximation of
    /// [`com_mapping::theory::ber`].
    ///
    /// # Errors
    /// As [`snr`] and [`com_mapping::theory::ber`].
    pub fn ber(ce: &CeOfdm, m: usize, esn0_db: f64, kind: ConstellationKind) -> Result<f64> {
        let e = ebn0_db(ce, m, esn0_db)?;
        if kind == ConstellationKind::Qam {
            if let Ok(exact) = com_mapping::theory::ber_gray_qam(m, e) {
                return Ok(exact);
            }
        }
        com_mapping::theory::ber(kind, m, e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;
    use com_mapping::constellation::{Constellation, ConstellationKind};

    #[test]
    fn the_projection_inverts_the_ofdm_signal_and_the_phase_variance_is_2pi_h() {
        // h = 0.6: the phase (standard deviation 3.8 rad) wraps often, but at J = 16 its
        // increments stay far below π
        let ce = CeOfdm { n_data: 32, oversampling: 16, h: 0.6, n_cp: 5 };
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
        let e = ce.demodulate(&s, Unwrap::Stream).unwrap();
        assert!(e.iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-8));
        let w = ce.demodulate(&s, Unwrap::None).unwrap();
        assert!(w.iter().zip(&d).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max) > 0.1);
    }

    #[test]
    fn the_linear_theory_is_proakis_with_the_snr_scaled_by_2pi_h_squared() {
        let ce = CeOfdm { n_data: 32, oversampling: 4, h: 0.05, n_cp: 0 };
        let g = (TAU * 0.05).powi(2) * 100.0;
        let s = theory::snr(&ce, 20.0).unwrap();
        assert!((s - g).abs() < 1e-9 * g);
        let ser_ce = theory::ser(&ce, 16, 20.0, ConstellationKind::Qam).unwrap();
        let proakis = com_mapping::theory::ser(ConstellationKind::Qam, 16, 10.0 * (g / 4.0).log10()).unwrap();
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
        let ce = CeOfdm { n_data: 32, oversampling: 2, h: 0.1, n_cp: 0 };
        let esn0_db = 20.0;
        let sigma = ce.noise_variance(esn0_db).sqrt();
        let r: Vec<C64> = ce
            .modulate(&d)
            .unwrap()
            .into_iter()
            .map(|v| v + C64::new(rng.gaussian(), rng.gaussian()) * sigma / 2f64.sqrt())
            .collect();
        let e = ce.demodulate(&r, Unwrap::None).unwrap();
        let errors = c.decide(&e).iter().zip(c.decide(&d)).filter(|(a, b)| *a != b).count();
        let ser = errors as f64 / d.len() as f64;
        let want = theory::ser(&ce, 16, esn0_db, ConstellationKind::Qam).unwrap();
        assert!((ser / want - 1.0).abs() < 0.15, "{ser} vs {want}");
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
                    let ce = CeOfdm { n_data: 64, oversampling: 4, h, n_cp: 0 };
                    let esn0_db = ebn0_db + 10.0 * (2.0 * log2m).log10();
                    let lib = theory::ber(&ce, m_qam, esn0_db, ConstellationKind::Qam).unwrap();
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
