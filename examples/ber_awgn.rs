//! Bit error rate of Gray-mapped QAM over AWGN against the theory.
//!
//! A minimal Monte Carlo campaign with the [`Runner`]: the grid sweeps the modulation
//! order and `E_b/N_0`; every trial sends 4096 symbols; a condition stops after 200 bit
//! errors (at least 2 batches, at most 4000 trials); [`SkipAfter`] skips the rest of a
//! sweep once the BER falls below 1e-6; the results go to the console, to a CSV file
//! written as the conditions complete, and to a JSON report.
//!
//! ```text
//! cargo run --release -p comrs --example ber_awgn [output directory]
//! ```

use comrs::prelude::*;

/// Gray-mapped `M`-QAM over AWGN with hard decisions.
struct QamAwgn {
    /// Symbols per trial.
    n_symbols: usize,
}

impl Scenario for QamAwgn {
    /// The unit-energy constellation of the condition, built once.
    type State = Constellation;

    fn name(&self) -> &'static str {
        "qam-awgn"
    }

    fn setup(&self, cond: &Condition) -> Result<Constellation> {
        Ok(Constellation::qam(cond.usize("m")?)?.normalized())
    }

    fn trial(&self, c: &Constellation, cond: &Condition, rng: &mut Rng, rec: &mut Record) -> Result<()> {
        let k = c.bits_per_symbol();
        let bits = source::bits(self.n_symbols * k, rng);
        let x = c.modulate(&bits)?;
        let y = awgn::awgn_ebn0(&x, cond.f64("ebn0_db")?, k, 1.0, rng);
        let decided = c.demodulate(&y);
        rec.errors("ber", metrics::bit_errors(&decided, &bits) as u64, bits.len() as u64);
        Ok(())
    }
}

fn main() -> Result<()> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "results/ber_awgn".into());
    let out = std::path::Path::new(&out);
    let grid = Grid::new().axis("m", [4usize, 16, 64]).axis("ebn0_db", (0..=20).step_by(2).map(f64::from));
    let results = Runner::new()
        .seed(2024)
        .batch(64)
        .stop(Stop::min_errors("ber", 200).and(Stop::min_trials(128)).or(Stop::max_trials(4000)))
        .callback(ProgressBar::default())
        .callback(SkipAfter::below("ber", 1e-6, "ebn0_db"))
        .logger(CsvLogger::new(out.join("ber.csv")))
        .logger(JsonLogger::new(out.join("ber.json")))
        .run(&QamAwgn { n_symbols: 4096 }, &grid)?;

    // the ratio is meaningful only with enough errors: the points that reached the trial
    // cap with a handful of errors have wide intervals (see the CSV)
    println!("{:>4} {:>8} {:>8} {:>12} {:>12} {:>8}", "M", "Eb/N0", "errors", "BER", "theory", "ratio");
    for r in results.iter().filter(|r| r.trials > 0) {
        let m = r.condition.usize("m")?;
        let ebn0 = r.condition.f64("ebn0_db")?;
        let (ber, errors) = r.metric("ber").map_or((f64::NAN, 0), |e| (e.value(), e.count()));
        let theory = metrics::theory_ber(ConstellationKind::Qam, m, ebn0)?;
        println!("{m:>4} {ebn0:>8.1} {errors:>8} {ber:>12.3e} {theory:>12.3e} {:>8.3}", ber / theory);
    }
    println!("results in {}", out.display());
    Ok(())
}
