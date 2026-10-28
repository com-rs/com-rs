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

    /// ASE power spectral density `(G - 1) n_sp h f_c` (W/Hz).
    pub fn ase_psd(&self) -> f64 {
        (db2lin(self.gain_db) - 1.0) * self.n_sp() * PLANCK * self.fc_hz
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
