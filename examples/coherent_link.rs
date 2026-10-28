//! A single-polarisation coherent link with the standard receiver DSP chain.
//!
//! Transmitter: Gray-mapped 16-QAM at 32 GBd, root-raised-cosine shaping (roll-off 0.2)
//! at 2 samples per symbol, a transmit laser of 100 kHz linewidth.
//!
//! Channel: `length_km` of standard single-mode fibre (loss and chromatic dispersion,
//! the loss compensated by an ideal gain), then white noise for an SNR of `snr_db` in
//! the symbol-rate bandwidth (the ASE of an optically amplified link).
//!
//! Receiver: a coherent front end with a local oscillator of 100 kHz linewidth detuned
//! by 500 MHz, then
//!
//! 1. chromatic-dispersion compensation in the frequency domain;
//! 2. the matched root-raised-cosine filter;
//! 3. Gardner clock recovery at 2 samples per symbol (`k_p = 3·10⁻³`; the default
//!    `10⁻³` takes about 6000 symbols to converge after 3000 km). It is needed even
//!    with ideal clocks: the dispersion compensation is applied to a signal shifted by
//!    the LO offset `Δf`, which leaves a group delay `β₂ 2πΔf L` (0.2 symbol after
//!    80 km, 7 symbols after 3000 km);
//! 4. decimation to 1 sample per symbol at the phase of maximum variance;
//! 5. fourth-power frequency-offset estimation (refined periodogram peak);
//! 6. blind phase search (32 test phases, 2·32 + 1 symbols) and unwrapping;
//! 7. frame synchronisation on a preamble of 512 known symbols placed after the
//!    convergence of the clock loop: the delay and the multiple of `π/2` left by the
//!    blind phase search maximise the correlation with the preamble;
//! 8. hard decisions on the payload that follows.
//!
//! Every condition reports the BER, the SNR after the DSP and the error of the
//! frequency-offset estimate; the table compares the BER with the theory of 16-QAM at
//! the nominal SNR (the gap is the implementation penalty of the chain). With seed 7 the
//! SNR after the DSP is 0.2 to 0.3 dB below the nominal SNR at 14 dB and 0.9 to 1.1 dB
//! below it at 20 dB, at every length, and the frequency offset is found within 0.3 MHz.
//!
//! ```text
//! cargo run --release -p comrs --example coherent_link [output directory]
//! ```

use comrs::dsp::sync::{carrier, timing::GardnerLoop};
use comrs::dsp::{filter, resample, signal};
use comrs::hardware::optical::{Laser, Photodiode, coherent_receiver};
use comrs::prelude::*;
use std::f64::consts::FRAC_PI_2;

/// Symbol rate (Bd).
const RS: f64 = 32e9;
/// Samples per symbol.
const SPS: usize = 2;
/// Frequency offset of the local oscillator (Hz).
const CFO: f64 = 500e6;
/// First symbol of the preamble (the clock loop has converged by then).
const PREAMBLE_START: usize = 2048;
/// Length of the preamble.
const PREAMBLE: usize = 512;
/// Largest delay searched by the frame synchronisation (symbols).
const MAX_DELAY: isize = 64;
/// Symbols excluded at the end of the block (edges of the filters and of the BPS window).
const GUARD: usize = 256;

/// The link of the module documentation.
struct CoherentLink {
    /// Symbols per trial.
    n_symbols: usize,
}

/// What does not change within a condition.
struct Link {
    /// The unit-energy 16-QAM constellation.
    constellation: Constellation,
    /// Root-raised-cosine taps (transmit and matched filter).
    rrc: Vec<f64>,
    /// The fibre.
    fiber: FiberParams,
}

/// Delay `d` and rotation of `z` that maximise `|Σ_n z[n + d] x*[n]|` over the preamble.
fn frame_sync(z: &[C64], x: &[C64]) -> (isize, C64) {
    let mut best = (0, C64::new(0.0, 0.0));
    for d in -MAX_DELAY..=MAX_DELAY {
        let c: C64 =
            (PREAMBLE_START..PREAMBLE_START + PREAMBLE).map(|n| z[(n as isize + d) as usize] * x[n].conj()).sum();
        if c.norm() > best.1.norm() {
            best = (d, c);
        }
    }
    // the rotation is the multiple of π/2 closest to the phase of the correlation
    let k = (best.1.arg() / FRAC_PI_2).round();
    (best.0, C64::from_polar(1.0, -k * FRAC_PI_2))
}

impl Scenario for CoherentLink {
    type State = Link;

    fn name(&self) -> &'static str {
        "coherent-link"
    }

    fn setup(&self, _: &Condition) -> Result<Link> {
        Ok(Link {
            constellation: Constellation::qam(16)?.normalized(),
            // an odd length puts the taps on the sampling grid
            rrc: filter::pulse_shape(filter::PulseShape::Rrc { rolloff: 0.2 }, SPS, 1025),
            fiber: FiberParams::default(),
        })
    }

    fn trial(&self, link: &Link, cond: &Condition, rng: &mut Rng, rec: &mut Record) -> Result<()> {
        let c = &link.constellation;
        let fs = RS * SPS as f64;
        let length_km = cond.f64("length_km")?;

        // transmitter
        let bits = source::bits(self.n_symbols * c.bits_per_symbol(), rng);
        let x = c.modulate(&bits)?;
        let shaped = signal::pnorm(&filter::fir_filter(&link.rrc, &resample::upsample(&x, SPS)));
        let tx_laser = Laser { power_dbm: 0.0, ..Laser::default() };
        let carrier_tx = tx_laser.field(shaped.len(), fs, rng);
        let tx: Vec<C64> = shaped.iter().zip(&carrier_tx).map(|(s, l)| s * l).collect();

        // channel: fibre, ideal loss compensation, ASE in the signal bandwidth
        let y = fiber::linear_fiber_channel(&tx, &link.fiber, length_km, fs);
        let y = awgn::awgn(&signal::pnorm(&y), cond.f64("snr_db")?, fs, RS, rng);

        // coherent front end
        let lo = Laser { power_dbm: 10.0, freq_shift_hz: CFO, ..Laser::default() }.field(y.len(), fs, rng);
        let r = coherent_receiver(&y, &lo, &Photodiode::ideal(), fs, rng)?;

        // DSP
        let r = comrs::dsp::compensation::cd::cdc(&r, link.fiber.beta2_s2_km(), length_km, fs);
        let r = filter::fir_filter(&link.rrc, &r);
        let mut r = GardnerLoop { kp: 3e-3, ki: 1e-5, ..GardnerLoop::default() }.run(&signal::pnorm(&r)).signal;
        r.truncate(r.len() / SPS * SPS);
        let z = signal::pnorm(&resample::decimate(&r, SPS, 1)?);
        let (z, fo) = carrier::fourth_power_foe_refined(&z, RS, 4);
        let theta = carrier::unwrap(&carrier::bps(&z, 32, c.points(), 32)?, FRAC_PI_2);
        let z = carrier::derotate(&z, &theta, 1.0);
        let (delay, rotation) = frame_sync(&z, &x);

        // metrics on the payload, aligned on the transmitted symbols
        let (a, b) = (PREAMBLE_START + PREAMBLE, self.n_symbols - GUARD);
        let payload: Vec<C64> = (a..b).map(|n| z[(n as isize + delay) as usize] * rotation).collect();
        let k = c.bits_per_symbol();
        let decided = c.demodulate(&payload);
        rec.errors("ber", metrics::bit_errors(&decided, &bits[a * k..b * k]) as u64, ((b - a) * k) as u64);
        rec.value("snr_dsp_db", metrics::snr_db(&payload, &x[a..b])?);
        // the LO is above the carrier: the receiver sees an offset of -CFO
        rec.value("foe_error_mhz", (fo + CFO) / 1e6);
        Ok(())
    }
}

fn main() -> Result<()> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "results/coherent_link".into());
    let out = std::path::Path::new(&out);
    let grid = Grid::new().axis("length_km", [80.0, 1000.0, 3000.0]).axis("snr_db", [14.0, 16.0, 18.0, 20.0]);
    let results = Runner::new()
        .seed(7)
        .batch(8)
        .stop(Stop::min_errors("ber", 200).and(Stop::min_trials(16)).or(Stop::max_trials(256)))
        .callback(ProgressBar::default())
        .logger(CsvLogger::new(out.join("coherent_link.csv")))
        .logger(JsonLogger::new(out.join("coherent_link.json")))
        .run(&CoherentLink { n_symbols: 1 << 14 }, &grid)?;

    println!("{:>8} {:>7} {:>11} {:>11} {:>10} {:>12}", "L (km)", "SNR", "BER", "theory", "SNR DSP", "FOE err MHz");
    for r in &results {
        let snr = r.condition.f64("snr_db")?;
        let value = |m: &str| r.metric(m).map_or(f64::NAN, Estimate::value);
        // Es/N0 = SNR in the symbol-rate bandwidth; Eb/N0 = Es/N0 - 10 log10(4)
        let theory = metrics::theory_ber(ConstellationKind::Qam, 16, snr - lin2db(4.0))?;
        println!(
            "{:>8.0} {:>7.1} {:>11.3e} {:>11.3e} {:>10.2} {:>12.3}",
            r.condition.f64("length_km")?,
            snr,
            value("ber"),
            theory,
            value("snr_dsp_db"),
            value("foe_error_mhz")
        );
    }
    println!("results in {}", out.display());
    Ok(())
}
