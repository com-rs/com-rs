//! Declarative simulation specifications.
//!
//! A specification describes a whole Monte Carlo campaign in a YAML, JSON or TOML file:
//! the grid of conditions, the link as a list of blocks, the metrics, the stopping
//! rule and the outputs. Every number of a study then comes from a file that can be
//! read, versioned and rerun (`comrs run spec.yaml`, [`run`] in Rust,
//! `comrs.run_spec` in Python), and a simulation that cannot be written as a
//! specification shows that a block is missing from the [registry](blocks::registry).
//!
//! # Format
//!
//! ```yaml
//! name: qam16_awgn                 # the name of the campaign (it enters the seeds)
//! description: BER of 16-QAM over AWGN
//! seed: 1                          # base seed (default 0)
//! batch: 64                        # trials between two checks of the stopping rule
//! threads: 0                       # worker threads (0: all cores)
//! confidence: 0.95                 # level of the reported intervals
//!
//! grid:                            # axes, the last one varying fastest
//!   ebn0_db: {from: 0, to: 12, step: 2}   # also: a list, or {linspace: [a, b, n]}
//! cases:                           # optional explicit conditions, crossed with the axes
//!   - {m: 4}
//!   - {m: 16}
//!
//! fiber: {n_spans: 10}             # defaults of every fibre block (FiberParams)
//!
//! link:                            # blocks, in order; `$name` is a grid variable
//!   - {type: source, constellation: {kind: qam, m: $m}, n_symbols: 4096}
//!   - {type: awgn_ebn0, ebn0_db: $ebn0_db}
//!
//! metrics: {report: [ber, ser, snr_db]}
//!
//! stop:                            # (targets and min_trials) or max_trials or max_time_s
//!   min_errors: {ber: 200}
//!   precision: {snr_db: 0.01}      # relative half-width of the interval
//!   min_trials: 8
//!   max_trials: 10000
//!   max_time_s: 600
//!
//! callbacks:
//!   progress: true
//!   skip_below: {metric: ber, threshold: 1e-6, along: ebn0_db}
//!
//! profile: true                    # the time of every block, as the metrics <label>.time_ms
//! output: {csv: results/qam16_awgn.csv, json: results/qam16_awgn.json}
//!
//! search:                          # instead of the sweep: the operating point
//!   {along: ebn0_db, range: [4, 16], metric: ber, target: 1.0e-3}
//!
//! expect:                          # expected results, checked by `comrs verify`
//!   - {metric: ber, theory: {function: ber, kind: qam, m: $m, ebn0_db: $ebn0_db}, rtol: 0.05}
//! ```
//!
//! - The variables of the grid are substituted in the link and the metrics before the
//!   blocks are built: a string `$name` (or `${name}`) becomes the value with its type,
//!   and `${name}` inside a longer string is replaced by its text.
//! - The stopping rule is `(all targets and min_trials) or max_trials or max_time_s`;
//!   without any field it is `max_trials: 100`, and a rule with targets but no cap is
//!   rejected (it could run forever). A target on a metric that the link does not
//!   record, or `min_errors` on a mean (`snr_db`: use `precision`), is an error at the
//!   first batch (`comrs check` reports it); `max_time_s` is a finite number of seconds
//!   at least 0, a `precision` a positive number. A run of `max_trials` runs exactly
//!   that many trials, whatever the batch.
//! - The accepted values of a parameter (`reference: signal`, `kind: rrc`) are compared
//!   without case, and the block receives the declared spelling.
//! - The blocks are listed with their parameters by [`blocks::registry`] (and
//!   `comrs blocks`); the metrics are those of [`metrics`].
//! - `search` replaces the sweep by the search of the value of a variable at which a
//!   metric reaches a target, for every combination of the other axes ([`search`]).
//! - `expect` lists results that the campaign must reproduce (closed forms, published
//!   curves), with their sources and tolerances; see [`verify`].
//! - A block trained once per condition ([`Block::needs_training`](blocks::Block::needs_training),
//!   a learned receiver of another crate, say) is trained on an independent block drawn
//!   from the stream `"<name>/train"` before the trials.
//! - Blocks defined in other crates join the registry by [`blocks::register`].
//!
//! # Modules
//!
//! | Module | Content |
//! |---|---|
//! | [`blocks`] | the blocks of a link, the frame they act on, their registry and its extension point |
//! | [`metrics`] | the metrics of a declarative link |
//! | [`verify`] | the expected results of a specification and their checks |
//! | [`search`] | the search of operating points |
//! | [`io`] | recordings: SigMF, raw IQ files, NPY arrays |
//! | [`cli`] | the command line `comrs` (`run`, `check`, `verify`, `conditions`, `blocks`, `schema`) |
//!
//! The specification itself ([`Spec`]), the link ([`Link`]) and the runs ([`run`],
//! [`profile`]) are at the root.
//!
//! # Examples
//! ```
//! use com_spec::{run, RunOptions, Spec};
//! let spec = Spec::from_yaml(r#"
//! name: qpsk_doc
//! seed: 3
//! grid: {ebn0_db: [2, 4]}
//! link:
//!   - {type: source, constellation: QPSK, n_symbols: 2000}
//!   - {type: awgn_ebn0, ebn0_db: $ebn0_db}
//! metrics: {report: [ber]}
//! stop: {min_errors: {ber: 300}, max_trials: 500}
//! "#).unwrap();
//! let results = run(&spec, &RunOptions::default()).unwrap();
//! let ber = results[1].metric("ber").unwrap();
//! let theory = com_metrics::theory_ber(com_mapping::constellation::ConstellationKind::Qam, 4, 4.0).unwrap();
//! let (lo, hi) = ber.interval(0.999);
//! assert!(lo < theory && theory < hi);
//! ```

pub mod blocks;
pub mod cli;
pub mod io;
mod link;
pub mod metrics;
mod params;
mod run;
pub mod search;
mod spec;
pub mod verify;

pub use link::{Link, SpecScenario, link_labels, number_labels};
pub use params::{substitute, variable};
pub use run::{ProfileRow, RunOptions, profile, profile_of, run};
pub use spec::{Format, Spec};

/// The paths the exported macros expand to, so that a crate that uses them needs no
/// dependency of its own on `com-core` or `serde_json`.
#[doc(hidden)]
pub mod __macro {
    pub use com_core::Result;
    pub use serde_json::{Value as Json, json};
}
