//! A wavelength-division multiplexed transmitter: the assembly of the pulse shaping and
//! upsampling of `com-dsp`, the lasers and the IQ modulator of `com-hardware`
//! (OptiCommPy's `simpleWDMTx`).
//!
//! # Model
//!
//! Channel $c$ of the grid, at the frequency $f_{c}$ relative to the carrier
//! (`com_dsp::signal::wdm_grid`), of power $P_{c}$ over $N_{\mathrm{pol}}$ polarisations,
//! adds to polarisation $p$ of the multiplex sampled at $f_{s} = R_{s} \cdot
//! \mathrm{sps}$
//!
//! $$
//! x_{p}\[k\] \mathrel{+}= \sqrt{P_{c} / N_{\mathrm{pol}}}\; w_{c,p}\[k\]\;
//! e^{j(\phi_{c,k} + 2\pi f_{c} k / f_{s})},
//! $$
//!
//! with $w_{c,p}$ the shaped symbols normalised to unit power (or the field of the IQ
//! modulator they drive) and the carrier of the laser of the channel,
//! `com_hardware::optical::Laser::carrier`, whose Wiener phase $\phi_{c,k}$ has the
//! linewidth of the transmitter. The receiver of a channel is
//! `com_dsp::signal::wdm_demultiplex`.
//!
//! # Examples
//! ```
//! use com_systems::wdm::WdmTransmitter;
//! use com_dsp::signal::wdm_grid;
//! use com_core::rng::Rng;
//! use com_math::C64;
//! let taps = com_dsp::filter::pulse_shape(com_dsp::filter::PulseShape::Rrc { rolloff: 0.1 }, 4, 129);
//! let tx = WdmTransmitter { freqs: wdm_grid(3, 50e9), powers_w: vec![1e-3; 3], taps, sps: 4, linewidth_hz: 0.0, mzm_scale: None };
//! let x = tx.multiplex(2, 64, 32e9, &mut Rng::new(1), |_, _, _| vec![C64::new(1.0, 0.0); 64])?;
//! assert_eq!((x.len(), x[0].len()), (2, 64 * 4));
//! # Ok::<(), com_core::Error>(())
//! ```

use com_core::rng::Rng;
use com_core::{Result, ensure, ensure_shape, same_len};
use com_dsp::filter::fir_filter;
use com_hardware::optical::{Iqm, Laser};
use com_math::C64;

/// A WDM transmitter: channels of the same symbol rate on the frequencies `freqs`
/// (relative to the carrier), each shaped by the pulse `taps` at `sps` samples per
/// symbol, of power `powers_w` (W, both polarisations), on a laser of linewidth
/// `linewidth_hz` (OptiCommPy's `simpleWDMTx`; see the module).
///
/// A polarisation of a channel is the shaped symbols normalised to unit power, or,
/// with `mzm_scale`, the field of OptiCommPy's IQ modulator driven by the shaped
/// symbols normalised to the peak amplitude `mzm_scale · V_π`.
#[derive(Clone, Debug, PartialEq)]
pub struct WdmTransmitter {
    /// Centre frequencies of the channels relative to the carrier (Hz).
    pub freqs: Vec<f64>,
    /// Launch power of every channel, both polarisations (W).
    pub powers_w: Vec<f64>,
    /// Taps of the pulse.
    pub taps: Vec<f64>,
    /// Samples per symbol of the multiplex.
    pub sps: usize,
    /// Linewidth of the transmit lasers (Hz).
    pub linewidth_hz: f64,
    /// Peak drive of the IQ modulator relative to `V_π`; a linear modulator if `None`.
    pub mzm_scale: Option<f64>,
}

impl WdmTransmitter {
    /// Validate the configuration.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of powers than of frequencies;
    /// [`com_core::Error::Config`] for no channel, no tap, zero samples per symbol, or a
    /// negative or non-finite power.
    pub fn check(&self) -> Result<()> {
        ensure(!self.freqs.is_empty(), || "a WDM transmitter needs at least one channel".into())?;
        same_len(self.powers_w.len(), self.freqs.len(), "powers and frequencies of the channels")?;
        ensure(!self.taps.is_empty() && self.sps >= 1, || {
            "the pulse needs taps and one sample per symbol or more".into()
        })?;
        ensure(self.powers_w.iter().all(|p| *p >= 0.0 && p.is_finite()), || {
            format!("the channel powers must be non-negative, got {:?}", self.powers_w)
        })
    }

    /// The unit-power waveform of one polarisation of a channel.
    ///
    /// # Errors
    /// The errors of the IQ modulator.
    pub fn shape(&self, symbols: &[C64]) -> Result<Vec<C64>> {
        let x = fir_filter(&self.taps, &com_dsp::resample::upsample(symbols, self.sps));
        Ok(match self.mzm_scale {
            None => com_dsp::signal::pnorm(&x),
            Some(scale) => {
                // OptiCommPy: the drive normalised to the peak amplitude scale·V_π
                let iqm = Iqm::default();
                let peak = x.iter().map(|v| v.norm()).fold(0.0, f64::max).max(1e-300);
                let u: Vec<C64> = x.iter().map(|v| v * (scale * iqm.vpi / peak)).collect();
                com_dsp::signal::pnorm(&iqm.modulate(&vec![C64::new(1.0, 0.0); u.len()], &u)?)
            }
        })
    }

    /// The multiplex of `pols` polarisations of `n_symbols` symbols per channel at the
    /// symbol rate `symbol_rate`: `symbols(channel, polarisation, rng)` gives the
    /// symbols of every polarisation of every channel, drawn after the phase noise of
    /// the channel's laser.
    ///
    /// # Errors
    /// The errors of [`WdmTransmitter::check`], of the laser
    /// (`com_hardware::optical::Laser::carrier`) and of [`WdmTransmitter::shape`];
    /// [`com_core::Error::Config`] for no polarisation; [`com_core::Error::Shape`] if
    /// `symbols` gives another number of symbols than `n_symbols`.
    pub fn multiplex<F>(
        &self,
        pols: usize,
        n_symbols: usize,
        symbol_rate: f64,
        rng: &mut Rng,
        mut symbols: F,
    ) -> Result<Vec<Vec<C64>>>
    where
        F: FnMut(usize, usize, &mut Rng) -> Vec<C64>,
    {
        self.check()?;
        ensure(pols >= 1, || "a WDM multiplex needs at least one polarisation".into())?;
        let n = n_symbols * self.sps;
        let fs = symbol_rate * self.sps as f64;
        let mut out = vec![vec![C64::new(0.0, 0.0); n]; pols];
        for (ch, (&f, &p)) in self.freqs.iter().zip(&self.powers_w).enumerate() {
            // the carrier of the channel, with the phase noise of its laser
            let carrier =
                Laser { linewidth_hz: self.linewidth_hz, freq_shift_hz: f, ..Laser::default() }.carrier(n, fs, rng)?;
            let a = (p / pols as f64).sqrt();
            for (pol, o) in out.iter_mut().enumerate() {
                let s = symbols(ch, pol, rng);
                ensure_shape(s.len() == n_symbols, || {
                    format!("{} symbols on channel {ch}, polarisation {pol}, for {n_symbols}", s.len())
                })?;
                for ((o, w), e) in o.iter_mut().zip(self.shape(&s)?).zip(&carrier) {
                    *o += w * e * a;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_dsp::signal::{wdm_demultiplex, wdm_grid};
    use com_mapping::constellation::Constellation;

    #[test]
    fn the_demultiplexer_recovers_a_channel_of_the_multiplex() {
        // three channels of 1 mW; the centre one, demultiplexed, is its shaped symbols
        let taps = com_dsp::filter::pulse_shape(com_dsp::filter::PulseShape::Rrc { rolloff: 0.1 }, 8, 1025).unwrap();
        let tx = WdmTransmitter {
            freqs: wdm_grid(3, 50e9),
            powers_w: vec![1e-3; 3],
            taps,
            sps: 8,
            linewidth_hz: 0.0,
            mzm_scale: None,
        };
        let qpsk = Constellation::qam(4).unwrap().normalized();
        let mut rng = Rng::new(2);
        let draw = |rng: &mut Rng| com_mapping::source::symbols(&qpsk, 512, None, rng).unwrap().1;
        let centre = draw(&mut rng);
        let x = tx
            .multiplex(1, 512, 32e9, &mut rng, |ch, _, rng| if ch == 1 { centre.clone() } else { draw(rng) })
            .unwrap();
        let p = x[0].iter().map(C64::norm_sqr).sum::<f64>() / x[0].len() as f64;
        assert!((p / 3e-3 - 1.0).abs() < 0.05, "{p}");
        // at 2 samples per symbol the band is ±32 GHz: the neighbours (50 ± 17.6 GHz)
        // are out of it
        let y = wdm_demultiplex(&x[0], 0.0, 256e9, 512 * 2);
        let reference = com_dsp::resample::resample_fft(&tx.shape(&centre).unwrap(), 512 * 2);
        let err: f64 = y.iter().zip(&reference).map(|(a, b)| (a / 1e-3f64.sqrt() - b).norm_sqr()).sum::<f64>();
        let sig: f64 = reference.iter().map(C64::norm_sqr).sum();
        assert!(err / sig < 1e-3, "{}", err / sig);
        // a short block of symbols or mismatched powers are refused, not truncated
        assert!(tx.multiplex(1, 512, 32e9, &mut rng, |_, _, _| vec![C64::new(1.0, 0.0); 511]).is_err());
        assert!(WdmTransmitter { powers_w: vec![1e-3; 2], ..tx }.check().is_err());
    }
}
