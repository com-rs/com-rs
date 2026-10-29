//! # com-sim — Monte Carlo simulation campaigns
//!
//! A campaign evaluates a **scenario** (one random trial of a link) on every
//! **condition** of a **grid** (an SNR, a launch power, a modulation order, …), as many
//! times as a **stopping rule** requires, in parallel on every core, and reports every
//! metric with a confidence interval. The organisation follows PyTorch Lightning, so
//! that the experiment code stays short and the engineering (parallelism, seeds,
//! stopping, logging, progress) is written once:
//!
//! | PyTorch Lightning | `com_sim` | role |
//! |---|---|---|
//! | `LightningModule` | [`scenario::Scenario`] | what one trial does: `setup` once per condition, `trial` many times |
//! | `self.log(...)` | [`scenario::Record`] | the metrics a trial reports: error counts or values |
//! | `Trainer` | [`runner::Runner`] | runs the trials, in batches, on a thread pool |
//! | `seed_everything` | [`runner::Runner::seed`] | trial `t` of condition `c` uses `stream(seed, name, c, t)` |
//! | `max_epochs`, `EarlyStopping` | [`stopping::Stop`] | when a condition has enough trials (errors, precision, time) |
//! | `Callback` | [`callbacks::Callback`] | hooks at the start and end of the run, of every condition and batch |
//! | `Logger` | [`loggers::Logger`] | CSV (one row per condition, written as it completes) and JSON reports |
//! | `fast_dev_run` | [`runner::Runner::fast_dev_run`] | one batch per condition, to check a scenario quickly |
//!
//! | Module | Content |
//! |---|---|
//! | [`scenario`] | the [`Scenario`](scenario::Scenario) trait, [`from_fn`](scenario::from_fn), the [`Record`](scenario::Record) of a trial |
//! | [`grid`] | the conditions of a campaign: axes, zips, fixed values, explicit lists |
//! | [`runner`] | the [`Runner`](runner::Runner): batches of trials on a thread pool, seeds, results |
//! | [`stopping`] | the stopping rules [`Stop`](stopping::Stop) and their combinations |
//! | [`callbacks`] | the hooks of a run: progress bar, skipping the rest of a sweep |
//! | [`loggers`] | the CSV and JSON reports |
//! | [`stats`] | error rates with Wilson intervals, means with normal intervals |
//! | [`json`] | a minimal JSON writer for the reports |
//!
//! # Reproducibility
//!
//! Every trial draws its random numbers from its own stream
//! `stream(seed, scenario name, condition index, trial index)` ([`com_core::rng::stream`]),
//! the trials of a batch run in parallel, their records are merged **in trial order**,
//! and the stopping rule is evaluated after every batch. The results are therefore
//! identical whatever the number of threads (only a wall-time rule,
//! [`stopping::Stop::max_time`], depends on the machine).
//!
//! # Statistics
//!
//! An error count is an [`stats::ErrorRate`] (`errors / total`, Wilson score interval);
//! any other value is a [`stats::Mean`] (Welford's running mean and variance, normal
//! interval of the mean). Both merge exactly across trials.
//!
//! # Example: BER of QPSK over AWGN against the theory
//!
//! ```
//! use com_sim::prelude::*;
//! use com_mapping::{constellation::{Constellation, ConstellationKind}, source as sources};
//! use com_channel::awgn as channels;
//! use com_metrics as metrics;
//! use com_sim::scenario::from_fn;
//!
//! // a scenario written as a closure: no per-condition state
//! let qpsk = from_fn("qpsk", |cond, rng, rec| {
//!     let c = Constellation::qam(4)?.normalized();
//!     let bits = sources::bits(2000, rng);
//!     let rx = channels::awgn_ebn0(&c.modulate(&bits)?, cond.f64("ebn0_db")?, 2.0, 1.0, rng);
//!     rec.errors("ber", metrics::bit_errors(&c.demodulate(&rx), &bits) as u64, 2000);
//!     Ok(())
//! });
//! let grid = Grid::new().axis("ebn0_db", [0.0, 2.0, 4.0]);
//! let res = Runner::new()
//!     .seed(7)
//!     .batch(16)
//!     .stop(Stop::min_errors("ber", 500).or(Stop::max_trials(2000)))
//!     .run(&qpsk, &grid)
//!     .unwrap();
//! for r in &res {
//!     let eb = r.condition.f64("ebn0_db").unwrap();
//!     let ber = r.metric("ber").unwrap();
//!     let (lo, hi) = ber.interval(0.99);
//!     let theory = metrics::theory_ber(ConstellationKind::Qam, 4, eb).unwrap();
//!     assert!(lo <= theory && theory <= hi, "{eb} dB: {theory} not in [{lo}, {hi}]");
//! }
//! ```

/// Version of the library, recorded in every run.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Commit of the library at build time (`<hash>-dirty` with uncommitted changes,
/// `unknown` outside a Git checkout), recorded in every run.
pub const COMMIT: &str = env!("COM_GIT_COMMIT");

pub mod callbacks;
pub mod grid;
pub mod json;
pub mod loggers;
pub mod runner;
pub mod scenario;
pub mod stats;
pub mod stopping;

/// The items used by most campaigns.
pub mod prelude {
    pub use crate::{
        callbacks::{Callback, ProgressBar, SkipAfter},
        grid::{Condition, Grid, Value},
        loggers::{CsvLogger, JsonLogger, Logger},
        runner::{ConditionResult, Runner},
        scenario::{Record, Scenario},
        stats::{ErrorRate, Estimate, Mean},
        stopping::Stop,
    };
    pub use com_core::rng::{Rng, stream};
    pub use com_core::{Error, Result};
}
