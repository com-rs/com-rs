//! The fading channels: `rayleigh`, `tdl` and `cdl`.

use super::Csi;
use crate::blocks::{Block, Frame, NameOr, Named};
use com_channel::cdl::{Array, Cdl, CdlParams, Profile as CdlProfile};
use com_channel::fading::Mimo;
use com_channel::multipath::{Profile, Tdl, TdlParams};
use com_channel::spatial::{Kronecker, exponential_correlation};
use com_core::Result;
use com_core::rng::Rng;
use com_math::C64;

// ----------------------------------------------------------------------- rayleigh

block_params! {
    pub(super) struct RayleighSpec {
        n_rx: usize [1] >= 1, "receive antennas";
        coherence_samples: usize [0] "samples", "samples per channel matrix (0: one matrix per trial, 1: fast fading)";
        k_factor: f64 [0.0] >= 0, "Rician factor K (linear), 0 for Rayleigh fading";
        rx_correlation: f64 [0.0] > -1 < 1, "exponential correlation ρ of the receive antennas";
        tx_correlation: f64 [0.0] > -1 < 1, "exponential correlation ρ of the transmit antennas";
    }
}

pub(super) struct Rayleigh {
    pub(super) p: RayleighSpec,
}

/// A fading channel on every stream of the frame: the transmitted power, summed over
/// the antennas, becomes the reference of the noise blocks, and `channel(x, fs, rng)`
/// gives the received streams and the channel that a genie-aided receiver knows.
fn fade(
    s: &mut Frame,
    rng: &mut Rng,
    channel: impl FnOnce(&[Vec<C64>], f64, &mut Rng) -> Result<(Vec<Vec<C64>>, Csi)>,
) -> Result<()> {
    s.channel.faded_power = Some(s.mean_power() * s.n_streams() as f64);
    let fs = s.fs();
    let x = s.take_streams();
    let (y, csi) = channel(&x, fs, rng)?;
    s.set_streams(y);
    s.channel.csi = Some(csi);
    Ok(())
}

impl Block for Rayleigh {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        fade(s, rng, |x, _, rng| {
            let (nt, nr, n) = (x.len(), self.p.n_rx, x[0].len());
            let correlation = if self.p.rx_correlation != 0.0 || self.p.tx_correlation != 0.0 {
                Some(Kronecker::new(
                    exponential_correlation(nr, C64::new(self.p.rx_correlation, 0.0)),
                    exponential_correlation(nt, C64::new(self.p.tx_correlation, 0.0)),
                )?)
            } else {
                None
            };
            let coh = if self.p.coherence_samples == 0 { n.max(1) } else { self.p.coherence_samples };
            let model = Mimo { n_rx: nr, n_tx: nt, k_factor: self.p.k_factor, los: None, correlation, coherence: coh };
            let (y, hs) = model.apply(x, rng)?;
            Ok((y, Csi::Flat { h: hs, coherence: coh }))
        })
    }
}

// ---------------------------------------------------------------------------- tdl

block_params! {
    pub(super) struct TdlSpec {
        profile: NameOr<ExponentialSpec> ["EPA"],
            "EPA, EVA, ETU (3GPP TS 36.101), pedestrian_a, pedestrian_b, vehicular_a, vehicular_b (ITU-R M.1225, Tables 4 and 5), or {rms_delay_s, resolution_s} (exponential)";
        doppler_hz: f64 [0.0] "Hz" >= 0, "maximum Doppler shift; 0 for a channel constant over the trial";
        n_rx: usize [1] >= 1, "receive antennas";
    }
}

block_params! {
    /// The exponential power-delay profile of `tdl`.
    struct ExponentialSpec {
        rms_delay_s: f64 [required] "s" > 0, "RMS delay spread";
        resolution_s: f64 [required] "s" > 0, "spacing of the taps";
    }
}

pub(super) struct TdlBlock {
    taps: Vec<(f64, f64)>,
    doppler: f64,
    n_rx: usize,
}

impl TdlBlock {
    pub(super) fn new(p: &TdlSpec) -> Result<TdlBlock> {
        let profile = match p.profile.get("tdl profile")? {
            Named::Name(name) => name.parse()?,
            Named::Params(e) => Profile::Exponential { rms_delay_s: e.rms_delay_s, resolution_s: e.resolution_s },
        };
        Ok(TdlBlock { taps: profile.taps(), doppler: p.doppler_hz, n_rx: p.n_rx })
    }
}

impl Block for TdlBlock {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        fade(s, rng, |x, fs, rng| {
            let tdl = Tdl::new(&TdlParams {
                taps: self.taps.clone(),
                doppler_hz: self.doppler,
                sample_rate: fs,
                n_rx: self.n_rx,
                n_tx: x.len(),
            })?;
            let taps = tdl.draw(x[0].len(), rng);
            Ok((taps.apply(x)?, Csi::Taps(taps)))
        })
    }
}

// ---------------------------------------------------------------------------- cdl

block_params! {
    pub(super) struct CdlSpec {
        profile: String ["CDL-A"] one_of ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"],
            "the profile of TR 38.901 Tables 7.7.1-1 to 7.7.1-5 (A, B, C: NLOS; D, E: LOS)";
        delay_spread_s: f64 [100e-9] "s" >= 0, "desired RMS delay spread (TR 38.901 §7.7.3)";
        n_rx: usize [1] >= 1, "receive antennas, a vertical uniform linear array";
        spacing_wavelengths: f64 [0.5] "wavelengths" > 0, "element spacing of both arrays";
        velocity_mps: f64 [0.0] "m/s" >= 0, "speed of the receiver along the array axis; 0 for a channel constant over the trial";
        fc_hz: f64 [3.5e9] "Hz" > 0, "carrier frequency (the wavelength of the Doppler shifts)";
    }
}

pub(super) struct CdlBlock {
    profile: CdlProfile,
    delay_spread: f64,
    n_rx: usize,
    spacing: f64,
    velocity: f64,
    carrier: f64,
}

impl CdlBlock {
    pub(super) fn new(p: &CdlSpec) -> Result<CdlBlock> {
        Ok(CdlBlock {
            profile: p.profile.parse()?,
            delay_spread: p.delay_spread_s,
            n_rx: p.n_rx,
            spacing: p.spacing_wavelengths,
            velocity: p.velocity_mps,
            carrier: p.fc_hz,
        })
    }
}

impl Block for CdlBlock {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        fade(s, rng, |x, fs, rng| {
            let tx = Array::linear(x.len(), self.spacing);
            let params = CdlParams {
                delay_spread_s: self.delay_spread,
                sample_rate: fs,
                tx,
                rx: Array::linear(self.n_rx, self.spacing),
                velocity_mps: [0.0, self.velocity, 0.0],
                fc_hz: self.carrier,
            };
            let cdl = Cdl::new(self.profile, params)?;
            let taps = cdl.draw(x[0].len(), rng)?;
            Ok((taps.apply(x)?, Csi::Taps(taps)))
        })
    }
}
