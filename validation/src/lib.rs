//! # com-validation — simulated curves against theory
//!
//! The statistical tests of `tests/` compare Monte Carlo estimates with closed forms,
//! published figures and tables, and, for the channel codes, the AFF3CT runs recorded in
//! `aff3ct/`. The quick ones run with `cargo test`; the slow ones are `#[ignore]`d and run
//! by hand, before a release (or by the workflow `validation.yml`, started by hand):
//!
//! ```text
//! cargo test --release -p com-validation -- --ignored
//! ```
//!
//! # Acceptance rule
//!
//! The rule of every comparison is stated in its test (overlapping Wilson intervals for
//! AFF3CT, slopes for the diversity orders, the tolerance of a published table). Against
//! a closed form, each simulated point runs until at least [`MIN_ERRORS`] errors. The relative
//! standard error of the estimate is then about $1/\sqrt{N_{e}}$ (10 % for 100 errors),
//! and a point passes when it lies within three standard errors of theory
//! ([`within_three_sigma`]).
//!
//! # The loops of the tests
//!
//! | Helper | Loop |
//! |---|---|
//! | [`until_errors`] | trials until a number of errors or of trials, trial $t$ drawing from `batch_rng(seed, t)`, on [`Runner`] (in parallel, the count independent of the threads) |
//! | [`until_errors_with`] | the same loop, sequential, the trials sharing one generator |
//! | [`coded_frame`], [`fer`] | one frame of a [`BlockCode`] over BPSK and AWGN ([`bpsk_awgn_llr`] at the variance of `bpsk_noise_variance`), and its frame error rate |
//! | [`sweep_crossing`] | an error rate on a grid of SNRs, until it falls below a target, and the SNR of the crossing ([`log_interp`]) |
//! | [`pmd::pmd_vectors`], [`pmd::moments`] | the PMD vectors of a `PmdLink` (Gordon & Kogelnik 2000) and the moments of a pulse, for the PMD tests |
//! | [`wdm::NyquistGrid`] | the fields of Nyquist WDM channels and the matched filter of the centre channel, for the fibre tests |

#[cfg(feature = "spec")]
pub mod card;
pub mod pmd;
pub mod shaping;
pub mod wdm;

use comrs::core::rng::Rng;
use comrs::core::{Error, Result};
use comrs::fec::block::BlockCode;
use comrs::fec::bounds::bpsk_noise_variance;
use comrs::mapping::constellation::bpsk;
use comrs::mapping::demapper::bpsk_llr;
use comrs::mapping::source;
use comrs::metrics::errors::bit_errors;
use comrs::sim::grid::Grid;
use comrs::sim::runner::Runner;
use comrs::sim::scenario::from_fn;
use comrs::sim::stats::Estimate;
use comrs::sim::stopping::Stop;

/// Errors to collect at every simulated point.
pub const MIN_ERRORS: u64 = 100;

/// Whether `errors` out of `total` trials is within three standard errors of the
/// probability `theory` (binomial standard error $\sqrt{p(1-p)/n}$).
pub fn within_three_sigma(errors: u64, total: u64, theory: f64) -> bool {
    let n = total as f64;
    let p = errors as f64 / n;
    let sigma = (theory * (1.0 - theory) / n).sqrt();
    (p - theory).abs() <= 3.0 * sigma
}

/// The channel LLRs $2 y/\sigma^{2}$ of the bits `c` sent in BPSK ($0 \mapsto +1$) over a
/// real AWGN channel of variance `noise_var`, the link of the code tests: the symbols of
/// [`bpsk`], one Gaussian draw per bit in order, the LLRs of [`bpsk_llr`].
pub fn bpsk_awgn_llr(c: &[u8], noise_var: f64, rng: &mut comrs::core::rng::Rng) -> Vec<f64> {
    let s = noise_var.sqrt();
    let y: Vec<f64> = bpsk(c).iter().map(|x| x + s * rng.gaussian()).collect();
    bpsk_llr(&y, noise_var)
}

/// The errors counted by a Monte Carlo loop: `errors` among `total` events over `trials`
/// trials.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Count {
    /// Errors.
    pub errors: u64,
    /// Events (bits, symbols, frames) among which the errors are counted.
    pub total: u64,
    /// Trials run.
    pub trials: u64,
}

impl Count {
    /// The error rate `errors / total`.
    pub fn rate(&self) -> f64 {
        self.errors as f64 / self.total as f64
    }
}

/// Trials $t = 0, 1, \dots$ of `trial`, which returns its errors and events and draws from
/// `batch_rng(seed, t)`, until at least `min_errors` errors or `max_trials` trials, the
/// rule checked after every batch of `batch` trials (`u64::MAX` for no bound). The
/// trials run on the threads of [`Runner`] (the scenario of no name and one condition,
/// whose trial $t$ draws from `stream(seed, "", 0, t)`, which is `batch_rng(seed, t)`):
/// the count depends on `batch`, not on the threads, and with `batch = 1` it is that of
/// the sequential loop.
///
/// # Errors
/// The errors of `trial` and of the runner.
pub fn until_errors(
    seed: u64,
    min_errors: u64,
    max_trials: u64,
    batch: u64,
    trial: impl Fn(&mut Rng) -> Result<(u64, u64)> + Sync,
) -> Result<Count> {
    let scenario = from_fn("", |_, rng, rec| {
        let (errors, total) = trial(rng)?;
        rec.errors("errors", errors, total);
        Ok(())
    });
    let results = Runner::new()
        .seed(seed)
        .batch(batch)
        .stop(Stop::min_errors("errors", min_errors).or(Stop::max_trials(max_trials)))
        .run(&scenario, &Grid::new())?;
    let r = results.first().ok_or_else(|| Error::Numerical("the runner returned no condition".into()))?;
    let (errors, total) = match r.metric("errors") {
        Some(Estimate::Rate(e)) => (e.errors, e.total),
        _ => (0, 0),
    };
    Ok(Count { errors, total, trials: r.trials })
}

/// [`until_errors`] run sequentially, the trials drawing in turn from the generator `rng`
/// (the rule checked after every trial).
///
/// # Errors
/// The errors of `trial`.
pub fn until_errors_with(
    rng: &mut Rng,
    min_errors: u64,
    max_trials: u64,
    mut trial: impl FnMut(&mut Rng) -> Result<(u64, u64)>,
) -> Result<Count> {
    let mut c = Count::default();
    while c.errors < min_errors && c.trials < max_trials {
        let (errors, total) = trial(rng)?;
        c.errors += errors;
        c.total += total;
        c.trials += 1;
    }
    Ok(c)
}

/// The errors of one frame of a code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameErrors {
    /// Information bits in error.
    pub bit_errors: u64,
    /// Whether the decoded information differs from the sent one.
    pub frame_error: bool,
}

/// One frame of `code` over BPSK and a real AWGN channel of variance `noise_var`: $k$
/// fair bits ([`source::bits`]), their codeword, the channel LLRs of [`bpsk_awgn_llr`]
/// and the decoded bits.
///
/// # Errors
/// The errors of the encoder and the decoder.
pub fn coded_frame(code: &dyn BlockCode, noise_var: f64, rng: &mut Rng) -> Result<FrameErrors> {
    let info = source::bits(code.k(), rng);
    let c = code.encode_block(&info)?;
    let decided = code.decode_block(&bpsk_awgn_llr(&c, noise_var, rng))?;
    let bit_errors = bit_errors(&decided, &info)? as u64;
    Ok(FrameErrors { bit_errors, frame_error: bit_errors > 0 })
}

/// The frame errors of `code` over BPSK and AWGN at `ebn0_db` (the noise variance of
/// `bpsk_noise_variance` at the rate $k/n$), frames of [`coded_frame`] drawn in turn from
/// `rng` until `min_errors` frame errors or `max_frames` frames ([`until_errors_with`]).
///
/// # Errors
/// The errors of the encoder and the decoder.
pub fn fer(code: &dyn BlockCode, ebn0_db: f64, min_errors: u64, max_frames: u64, rng: &mut Rng) -> Result<Count> {
    let noise_var = bpsk_noise_variance(ebn0_db, code.k() as f64 / code.n() as f64);
    until_errors_with(rng, min_errors, max_frames, |rng| {
        Ok((u64::from(coded_frame(code, noise_var, rng)?.frame_error), 1))
    })
}

/// The SNR (dB) at which an error rate falls below `target`: `rate(k, x)` is evaluated on
/// the grid $x_{k} = $ `start` $+ k \cdot$ `step`, $k = 0, 1, \dots$, `n - 1`, until the
/// first point $k \ge 1$ below the target, and the crossing is the [`log_interp`] of that
/// point and the previous one; `None` if no point of the grid is below it.
pub fn sweep_crossing(
    start: f64,
    step: f64,
    n: usize,
    target: f64,
    mut rate: impl FnMut(usize, f64) -> f64,
) -> Option<f64> {
    let mut prev = (start, rate(0, start));
    for k in 1..n {
        let x = start + k as f64 * step;
        let b = rate(k, x);
        if b < target {
            return Some(log_interp(prev, (x, b), target));
        }
        prev = (x, b);
    }
    None
}

/// The abscissa where the line through the points `a` and `b` in $(x, \ln y)$ reaches
/// `target`: the log-linear interpolation of an error rate $y$ between two simulated
/// points.
pub fn log_interp(a: (f64, f64), b: (f64, f64), target: f64) -> f64 {
    a.0 + (b.0 - a.0) * (a.1.ln() - target.ln()) / (a.1.ln() - b.1.ln())
}

/// The abscissa (an SNR in dB) where the error rates `points` (`(x, rate)`, in increasing
/// `x`) first cross `target` downwards, by [`log_interp`] between the two points that
/// bracket it; `None` when no two consecutive points bracket it.
pub fn crossing(points: &[(f64, f64)], target: f64) -> Option<f64> {
    points.windows(2).find(|w| w[0].1 >= target && w[1].1 < target).map(|w| log_interp(w[0], w[1], target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_acceptance_rule_is_three_standard_errors() {
        // p = 1e-2 over 10⁴ trials: σ ≈ 9.95e-4
        assert!(within_three_sigma(100, 10_000, 1e-2));
        assert!(within_three_sigma(129, 10_000, 1e-2));
        assert!(!within_three_sigma(131, 10_000, 1e-2));
    }

    #[test]
    fn the_sequential_and_the_parallel_loops_count_the_same_trials() {
        // a trial of 1000 bits with an error probability of 0.01: until 30 errors
        let trial = |rng: &mut Rng| -> Result<(u64, u64)> {
            Ok(((0..1000).filter(|_| rng.uniform() < 0.01).count() as u64, 1000))
        };
        let seq = {
            let mut c = Count::default();
            while c.errors < 30 {
                let (e, n) = trial(&mut comrs::core::rng::batch_rng(3, c.trials)).unwrap();
                c = Count { errors: c.errors + e, total: c.total + n, trials: c.trials + 1 };
            }
            c
        };
        assert_eq!(until_errors(3, 30, u64::MAX, 1, trial).unwrap(), seq);
        // in batches of 8: whole batches, the same trials
        let b = until_errors(3, 30, u64::MAX, 8, trial).unwrap();
        assert_eq!(b.trials, seq.trials.div_ceil(8) * 8);
        assert_eq!(until_errors(3, u64::MAX, 5, 2, trial).unwrap().trials, 5);
        let mut rng = Rng::new(1);
        assert_eq!(until_errors_with(&mut rng, u64::MAX, 7, trial).unwrap().trials, 7);
    }

    #[test]
    fn a_sweep_stops_at_the_first_point_below_the_target() {
        let mut calls = 0;
        let x = sweep_crossing(0.0, 1.0, 10, 1e-3, |_, x| {
            calls += 1;
            10f64.powf(-x)
        })
        .unwrap();
        assert!((x - 3.0).abs() < 1e-12 && calls == 5);
        assert_eq!(sweep_crossing(0.0, 1.0, 3, 1e-9, |_, x| 10f64.powf(-x)), None);
    }

    #[test]
    fn the_crossing_is_log_linear_between_the_bracketing_points() {
        // 1e-2 at 0 dB, 1e-4 at 2 dB: 1e-3 halfway
        let points = [(-2.0, 1e-1), (0.0, 1e-2), (2.0, 1e-4)];
        assert!((crossing(&points, 1e-3).unwrap() - 1.0).abs() < 1e-12);
        assert!(crossing(&points, 1e-6).is_none());
    }
}
