//! Nonlinear propagation over 10 × 80 km and digital back-propagation.
//!
//! The classical launch-power sweep of a single-channel long-haul link: Gray-mapped
//! 16-QAM at 32 GBd with root-raised-cosine shaping (roll-off 0.1, 4 samples per
//! symbol), propagated by the symmetric split-step Fourier method over 10 spans of
//! 80 km of standard single-mode fibre (0.2 dB/km, 17 ps/(nm·km), 1.3 /(W·km), 1 km
//! steps), with an EDFA of 4.5 dB noise figure after every span.
//!
//! Every trial propagates one block and evaluates four receivers on it:
//!
//! - `cdc`: chromatic-dispersion compensation only;
//! - `dbp1`, `dbp4`, `dbp20`: digital back-propagation with 1, 4 and 20 steps per span.
//!
//! each followed by the matched filter, sampling, and the constant phase rotation of
//! `fast_ber_calc` (which removes the average nonlinear phase). The effective SNR rises
//! with the launch power while the ASE dominates, then falls when the nonlinear
//! interference takes over; back-propagation moves the optimum to a higher power and a
//! higher SNR, all the more with more steps (single channel and no polarisation effects:
//! the only limit left is the signal-ASE interaction). With seed 11, dispersion
//! compensation peaks at about 21.6 dB near -2 dBm, while 20 steps per span stay within
//! 2 dB of the ASE limit up to 8 dBm (32.4 dB). With one step per span the full nonlinear
//! phase over-compensates at 32 GBd (the `nonlinear_scale` ξ of the `dbp` block tunes it).
//!
//! ```text
//! cargo run --release -p comrs --example ssfm_dbp [output directory]
//! ```

use comrs::channel::optical::fiber::Ssfm;
use comrs::dsp::{filter, resample, signal};
use comrs::prelude::*;

/// Symbol rate (Bd).
const RS: f64 = 32e9;
/// Samples per symbol (the nonlinear spectral broadening needs some oversampling).
const SPS: usize = 4;
/// The receivers and their steps per span (0: dispersion compensation only).
const RECEIVERS: [(&str, usize); 4] = [("cdc", 0), ("dbp1", 1), ("dbp4", 4), ("dbp20", 20)];

/// The long-haul link of the module documentation.
struct LongHaul {
    /// Symbols per trial.
    n_symbols: usize,
}

/// What does not change within a condition.
struct Link {
    /// The unit-energy 16-QAM constellation.
    constellation: Constellation,
    /// Root-raised-cosine taps (transmit and matched filter).
    rrc: Vec<f64>,
    /// The split-step simulation of the link.
    ssfm: Ssfm,
}

impl Scenario for LongHaul {
    type State = Link;

    fn name(&self) -> &'static str {
        "ssfm-dbp"
    }

    fn setup(&self, _: &Condition) -> Result<Link> {
        let fiber = FiberParams { n_spans: 10, ..FiberParams::default() };
        Ok(Link {
            constellation: Constellation::qam(16)?.normalized(),
            rrc: filter::pulse_shape(filter::PulseShape::Rrc { rolloff: 0.1 }, SPS, 4097)?,
            ssfm: Ssfm::new(fiber, 1.0),
        })
    }

    fn trial(&self, link: &Link, cond: &Condition, rng: &mut Rng, rec: &mut Record) -> Result<()> {
        let c = &link.constellation;
        let fs = RS * SPS as f64;
        let (_, x) = source::symbols(c, self.n_symbols, None, rng)?;
        let amplitude = dbm2w(cond.f64("power_dbm")?).sqrt();
        let tx: Vec<C64> = signal::pnorm(&filter::fir_filter(&link.rrc, &resample::upsample(&x, SPS)))
            .into_iter()
            .map(|v| v * amplitude)
            .collect();
        let mut y = tx;
        link.ssfm.propagate(&mut y, fs, Some(rng))?;
        let fiber = &link.ssfm.fiber;
        for (name, steps) in RECEIVERS {
            let r = if steps == 0 {
                comrs::dsp::compensation::cd::cdc(&y, fiber.beta2_s2_km(), fiber.total_km(), fs)
            } else {
                link.ssfm.backpropagate(&y, fs, steps)?
            };
            let z = resample::downsample(&filter::fir_filter(&link.rrc, &r), SPS, 0);
            // the first and last 64 symbols see the edges of the filters
            let n = self.n_symbols;
            let m = metrics::fast_ber_calc(&z[64..n - 64], &x[64..n - 64], c)?;
            rec.value(&format!("snr_{name}"), m.snr_db);
            rec.value(&format!("ber_{name}"), m.ber);
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "results/ssfm_dbp".into());
    let out = std::path::Path::new(&out);
    let grid = Grid::new().axis("power_dbm", [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]);
    let results = Runner::new()
        .seed(11)
        .batch(8)
        .stop(Stop::max_trials(16))
        .callback(ProgressBar::new(10.0))
        .logger(CsvLogger::new(out.join("ssfm_dbp.csv")))
        .logger(JsonLogger::new(out.join("ssfm_dbp.json")))
        .run(&LongHaul { n_symbols: 1 << 12 }, &grid)?;

    print!("{:>9}", "P (dBm)");
    for (name, _) in RECEIVERS {
        print!(" {:>10}", format!("SNR {name}"));
    }
    println!();
    for r in &results {
        print!("{:>9.1}", r.condition.f64("power_dbm")?);
        for (name, _) in RECEIVERS {
            print!(" {:>10.2}", r.metric(&format!("snr_{name}")).map_or(f64::NAN, Estimate::value));
        }
        println!();
    }
    println!("effective SNR in dB after the matched filter; results in {}", out.display());
    Ok(())
}
