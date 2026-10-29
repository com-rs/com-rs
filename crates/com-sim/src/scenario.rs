//! Scenarios and the metrics of a trial.
//!
//! A [`Scenario`] is what one Monte Carlo trial does, the counterpart of a
//! `LightningModule`:
//!
//! - [`Scenario::setup`] runs once per condition, before its trials, and returns a
//!   **state** shared (read-only) by all of them: a constellation, a filter, a designed
//!   or trained receiver, a fibre plan… (`setup` hook of Lightning);
//! - [`Scenario::trial`] runs one trial with its own random generator and records its
//!   metrics in a [`Record`] (`training_step` and `self.log`).
//!
//! Trials run in parallel, so the scenario and its state are `Sync` and a trial must
//! not mutate shared data. [`from_fn`] makes a scenario from a closure when no state
//! is needed.
//!
//! A [`Record`] maps metric names to [`Estimate`]s: [`Record::errors`] adds error counts
//! (an error rate with a Wilson interval), [`Record::value`] adds a real value (a mean
//! with a normal interval). The runner merges the records of all the trials of a
//! condition.
//!
//! # Examples
//! ```
//! use com_sim::prelude::*;
//! use com_mapping::{constellation::{Constellation, ConstellationKind}, source as sources};
//! use com_channel::awgn as channels;
//! use com_metrics as metrics;
//!
//! /// SER of M-PSK over AWGN; the constellation is built once per condition.
//! struct Psk;
//! impl Scenario for Psk {
//!     type State = Constellation;
//!     fn name(&self) -> &str { "psk-ser" }
//!     fn setup(&self, cond: &Condition) -> Result<Constellation> {
//!         Constellation::psk(cond.usize("m")?)
//!     }
//!     fn trial(&self, c: &Constellation, cond: &Condition, rng: &mut Rng, rec: &mut Record)
//!         -> Result<()> {
//!         let (idx, x) = sources::symbols(c, 1000, None, rng)?;
//!         let b = c.bits_per_symbol();
//!         let y = channels::awgn_ebn0(&x, cond.f64("ebn0_db")?, b as f64, 1.0, rng);
//!         let dec = c.decide(&y);
//!         rec.errors("ser", metrics::symbol_errors(&dec, &idx) as u64, idx.len() as u64);
//!         rec.value("snr_db", metrics::snr_db(&y, &x)?);
//!         Ok(())
//!     }
//! }
//! let grid = Grid::new().axis("m", [8usize]).axis("ebn0_db", [10.0]);
//! let r = Runner::new().batch(10).stop(Stop::max_trials(20)).run(&Psk, &grid).unwrap();
//! assert_eq!(r[0].trials, 20);   // the rule is checked after every batch of 10 trials
//! assert!((r[0].metric("snr_db").unwrap().value() - (10.0 + com_math::lin2db(3.0))).abs() < 0.1);
//! ```

use crate::grid::Condition;
use crate::stats::{ErrorRate, Estimate, Mean};
use com_core::Result;
use com_core::rng::Rng;
use std::collections::BTreeMap;

/// The metrics reported by one trial, or merged over many.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Record {
    metrics: BTreeMap<String, Estimate>,
    trial: u64,
}

impl Record {
    /// An empty record.
    pub fn new() -> Record {
        Record::default()
    }

    /// An empty record of trial `trial` of its condition (the runner numbers the
    /// trials; blocks that save or label a trial read it).
    pub fn for_trial(trial: u64) -> Record {
        Record { trial, ..Record::default() }
    }

    /// The index of the trial of the condition this record belongs to (0 when merged).
    pub fn trial(&self) -> u64 {
        self.trial
    }

    /// Add `errors` errors among `total` events to the error rate `name`.
    pub fn errors(&mut self, name: &str, errors: u64, total: u64) {
        let e = self.metrics.entry(name.to_string()).or_insert(Estimate::Rate(ErrorRate::default()));
        if let Estimate::Rate(r) = e {
            r.add(errors, total);
        }
    }

    /// Add a value to the mean `name` (non-finite values are ignored).
    pub fn value(&mut self, name: &str, v: f64) {
        if !v.is_finite() {
            return;
        }
        let e = self.metrics.entry(name.to_string()).or_insert(Estimate::Mean(Mean::default()));
        if let Estimate::Mean(m) = e {
            m.add(v);
        }
    }

    /// Add several values to the mean `name`.
    pub fn values(&mut self, name: &str, vs: impl IntoIterator<Item = f64>) {
        for v in vs {
            self.value(name, v);
        }
    }

    /// Merge another record (metrics of the same name and kind are combined).
    pub fn merge(&mut self, o: &Record) {
        for (k, v) in &o.metrics {
            match self.metrics.get_mut(k) {
                Some(e) => {
                    e.merge(v);
                }
                None => {
                    self.metrics.insert(k.clone(), *v);
                }
            }
        }
    }

    /// The estimate of `name`.
    pub fn get(&self, name: &str) -> Option<&Estimate> {
        self.metrics.get(name)
    }

    /// The metrics, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Estimate)> {
        self.metrics.iter()
    }

    /// Whether the record holds no metric.
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }
}

/// One random trial of a link, evaluated under many conditions.
pub trait Scenario: Sync {
    /// The per-condition state built by [`Scenario::setup`] and shared by the trials.
    type State: Sync;

    /// The name of the scenario (it enters the seeds of the trials and the reports).
    fn name(&self) -> &str;

    /// Build the state of a condition (run once per condition, before its trials).
    ///
    /// # Errors
    /// Any error of the blocks built for the condition; the run stops with it.
    fn setup(&self, condition: &Condition) -> Result<Self::State>;

    /// Run one trial: draw from `rng` only, record the metrics in `record`.
    ///
    /// # Errors
    /// Any error of the blocks of the trial; the run stops with it.
    fn trial(&self, state: &Self::State, condition: &Condition, rng: &mut Rng, record: &mut Record) -> Result<()>;
}

/// A scenario made of a closure (no per-condition state); see [`from_fn`].
pub struct FnScenario<F> {
    name: String,
    f: F,
}

/// A scenario from a name and a closure `|condition, rng, record| -> Result<()>`.
pub fn from_fn<F>(name: &str, f: F) -> FnScenario<F>
where
    F: Fn(&Condition, &mut Rng, &mut Record) -> Result<()> + Sync,
{
    FnScenario { name: name.to_string(), f }
}

impl<F> Scenario for FnScenario<F>
where
    F: Fn(&Condition, &mut Rng, &mut Record) -> Result<()> + Sync,
{
    type State = ();

    fn name(&self) -> &str {
        &self.name
    }

    fn setup(&self, _: &Condition) -> Result<()> {
        Ok(())
    }

    fn trial(&self, (): &(), condition: &Condition, rng: &mut Rng, record: &mut Record) -> Result<()> {
        (self.f)(condition, rng, record)
    }
}
