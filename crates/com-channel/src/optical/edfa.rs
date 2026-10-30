//! Optical amplification and its ASE noise (OptiCommPy `optic.models.devices.edfa`).
//!
//! An ideal erbium-doped fibre amplifier of gain `G` and noise figure `NF` multiplies the
//! field by `√G` and adds amplified spontaneous emission (ASE), a white circular complex
//! Gaussian noise of power spectral density
//!
//! `N_ASE = (G - 1) n_sp h f_c`, with `n_sp = (G·NF - 1) / (2 (G - 1))`
//!
//! (Essiambre et al., *Capacity limits of optical fiber networks*, JLT 28(4), 2010,
//! Eq. (54)). Over a simulation bandwidth equal to the sampling rate `f_s`, the noise
//! variance per sample is `N_ASE f_s`. The noise is added independently to each
//! polarisation.
//!
//! # Examples
//! ```
//! use com_channel::optical::edfa::Edfa;
//! use com_core::rng::Rng;
//! use com_math::C64;
//! let amp = Edfa { gain_db: 20.0, ..Edfa::default() };
//! let x = vec![C64::new(1e-3, 0.0); 100_000];
//! let y = amp.apply(&x, 32e9, &mut Rng::new(1)).unwrap();
//! let noise: f64 = y.iter().map(|v| (v - C64::new(1e-2, 0.0)).norm_sqr()).sum::<f64>() / y.len() as f64;
//! assert!((noise / amp.noise_variance(32e9) - 1.0).abs() < 0.02);
//! ```

use com_core::rng::Rng;
use com_core::{Result, ensure};
use com_math::{C64, PLANCK, db2lin};

/// The ASE power spectral density of one polarisation (W/Hz) of an amplifier of power
/// gain `gain` and noise figure `nf` (both linear) at the frequency `freq_hz`,
///
/// $$
/// N_{\mathrm{ASE}} = (G - 1)\\, n_{sp}\\, h f, \qquad n_{sp} = \frac{G F - 1}{2 (G - 1)}
/// $$
///
/// (Essiambre et al. 2010, Eq. (54)), in the order of operations of OptiCommPy's `edfa`.
/// The one formula of [`Edfa::ase_psd`], of the GN model with Raman amplification
/// (`gn::GnModel::ase_psd_channel`) and of the amplifier after a Raman-pumped span; a
/// gain of 1 has no ASE only in the limit, so the callers skip it.
///
/// # Examples
/// ```
/// use com_channel::optical::edfa::{Edfa, ase_density};
/// use com_math::db2lin;
/// let amp = Edfa::default();
/// assert_eq!(ase_density(db2lin(amp.gain_db), db2lin(amp.nf_db), amp.fc_hz), amp.ase_psd());
/// ```
pub fn ase_density(gain: f64, nf: f64, freq_hz: f64) -> f64 {
    (gain - 1.0) * ((gain * nf - 1.0) / (2.0 * (gain - 1.0))) * PLANCK * freq_hz
}

/// An ideal EDFA (flat gain, white ASE).
#[derive(Clone, Debug, PartialEq)]
pub struct Edfa {
    /// Gain (dB, > 0).
    pub gain_db: f64,
    /// Noise figure (dB, at least 3 dB, the quantum limit).
    pub nf_db: f64,
    /// Carrier frequency (Hz).
    pub fc_hz: f64,
}

impl Default for Edfa {
    /// 20 dB gain, 4.5 dB noise figure at 193.1 THz.
    fn default() -> Self {
        Edfa { gain_db: 20.0, nf_db: 4.5, fc_hz: 193.1e12 }
    }
}

impl Edfa {
    /// Spontaneous-emission factor `n_sp = (G·NF - 1)/(2(G - 1))`.
    pub fn n_sp(&self) -> f64 {
        let g = db2lin(self.gain_db);
        (g * db2lin(self.nf_db) - 1.0) / (2.0 * (g - 1.0))
    }

    /// ASE power spectral density `(G - 1) n_sp h f_c` (W/Hz), one polarisation
    /// ([`ase_density`]).
    pub fn ase_psd(&self) -> f64 {
        ase_density(db2lin(self.gain_db), db2lin(self.nf_db), self.fc_hz)
    }

    /// ASE variance per sample `E|w|²` at the sampling rate `fs` (W).
    pub fn noise_variance(&self, fs: f64) -> f64 {
        self.ase_psd() * fs
    }

    fn check(&self) -> Result<()> {
        ensure(self.gain_db > 0.0, || format!("EDFA gain must be positive, got {} dB", self.gain_db))?;
        ensure(self.nf_db >= 3.0, || format!("EDFA noise figure must be at least 3 dB, got {}", self.nf_db))
    }

    /// Amplify one polarisation sampled at `fs`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a non-positive gain or a noise figure below 3 dB.
    pub fn apply(&self, x: &[C64], fs: f64, rng: &mut Rng) -> Result<Vec<C64>> {
        self.check()?;
        let g = db2lin(self.gain_db).sqrt();
        let s = self.noise_variance(fs).sqrt();
        Ok(x.iter().map(|&v| v * g + rng.complex_gaussian() * s).collect())
    }

    /// Amplify in place without noise (an ideal, noiseless gain).
    pub fn apply_noiseless(&self, x: &mut [C64]) {
        let g = db2lin(self.gain_db).sqrt();
        for v in x.iter_mut() {
            *v *= g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The EDFA, the GN model with Raman amplification and the amplifier after a
    /// Raman-pumped span wrote the formula three times: the EDFA is bit-identical, the
    /// two others, `(G F - 1) h f` for both polarisations and `(G F - 1)/2 · h f` for one,
    /// differ by the rounding of the factor `(G - 1)/(G - 1)`.
    #[test]
    fn ase_density_is_the_former_formulas() {
        let mut worst: f64 = 0.0;
        for gain_db in [0.01, 0.5, 3.0, 10.0, 16.0, 20.0, 25.3, 40.0] {
            for nf_db in [3.0, 4.5, 5.0, 7.2] {
                for f in [186e12, 193.1e12, 196.7e12] {
                    let amp = Edfa { gain_db, nf_db, fc_hz: f };
                    let (g, nf) = (db2lin(gain_db), db2lin(nf_db));
                    assert_eq!(ase_density(g, nf, f), (g - 1.0) * amp.n_sp() * PLANCK * f);
                    let gn = (g * nf - 1.0) * PLANCK * f;
                    let raman = (g * nf - 1.0) / 2.0 * PLANCK * f;
                    worst = worst.max((2.0 * ase_density(g, nf, f) / gn - 1.0).abs());
                    worst = worst.max((ase_density(g, nf, f) / raman - 1.0).abs());
                }
            }
        }
        assert!(worst < 1e-15, "{worst}");
    }
}
