//! The link of a condition ([`Link`]) and a specification as a scenario of the runner
//! ([`SpecScenario`]).

use crate::blocks::{self, Block, BuildContext, Frame};
use crate::metrics::Metrics;
use crate::{Spec, params};
use com_core::rng::{Rng, fnv1a64, stream};
use com_core::{Error, Result};
use com_sim::grid::Condition;
use com_sim::scenario::{Record, Scenario};
use std::time::Instant;

impl Spec {
    /// Build the link of a condition: substitute the variables, build the blocks, train
    /// the blocks that are trained once, parse the metrics.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid block or metric; the errors of a training.
    pub fn link_for(&self, cond: &Condition) -> Result<Link> {
        let ctx = BuildContext {
            fiber: self.fiber.clone(),
            base_dir: self.base_dir.clone(),
            name: Some(self.name.clone()),
            condition: Some(cond.label()),
        };
        let mut blocks: Vec<Box<dyn Block>> = Vec::with_capacity(self.link.len());
        let mut kinds = Vec::with_capacity(self.link.len());
        let mut labels = Vec::with_capacity(self.link.len());
        for (i, b) in self.link.iter().enumerate() {
            let (kind, label, p) = params::split_block(&params::substitute(b, cond)?)?;
            labels.push(label.clone().unwrap_or_else(|| kind.clone()));
            let block = blocks::build_parts(kind.clone(), label, p, &ctx)
                .map_err(|e| e.context(format!("link block {}", i + 1)))?;
            blocks.push(block);
            kinds.push(kind);
        }
        let labels = number_labels(&labels);
        blocks::with_registry(|registry| {
            if registry.iter().any(|b| b.starts && b.name == kinds[0]) {
                return Ok(());
            }
            let starts: Vec<&str> = registry.iter().filter(|b| b.starts).map(|b| b.name).collect();
            Err(Error::Config(format!("the first block of the link must be one of {starts:?}")))
        })?;
        blocks::check_link(&kinds.iter().map(String::as_str).collect::<Vec<_>>())?;
        for i in 0..blocks.len() {
            if blocks[i].needs_training() {
                let mut rng = self.training_rng(cond, i);
                let mut s = Frame::default();
                for b in &blocks[..i] {
                    b.apply(&mut s, &mut rng)?;
                }
                blocks[i].train(&s, &mut rng)?;
            }
        }
        let metrics = Metrics::from_json(&params::substitute(&self.metrics, cond)?)?;
        Ok(Link { blocks, kinds, labels, metrics, profile: self.profile })
    }

    fn training_rng(&self, cond: &Condition, block: usize) -> Rng {
        stream(self.seed, &format!("{}/train", self.name), fnv1a64(cond.label().as_bytes()), block as u64)
    }
}

/// The labels of the blocks of a link, as the profile names them (and the Python chains
/// their blocks): the `label` of a block, else its type, numbered by [`number_labels`].
///
/// # Errors
/// [`Error::Config`] for a block that is neither a name nor a mapping with a type.
pub fn link_labels(link: &[serde_json::Value]) -> Result<Vec<String>> {
    let labels = link
        .iter()
        .map(|b| params::split_block(b).map(|(kind, label, _)| label.unwrap_or(kind)))
        .collect::<Result<Vec<_>>>()?;
    Ok(number_labels(&labels))
}

/// Unique labels: a repeated label takes the position of its block, counted from 1
/// (`[source, awgn, awgn]` gives `[source, awgn_2, awgn_3]`).
pub fn number_labels(labels: &[String]) -> Vec<String> {
    labels
        .iter()
        .enumerate()
        .map(|(i, l)| if labels.iter().filter(|g| *g == l).count() > 1 { format!("{l}_{}", i + 1) } else { l.clone() })
        .collect()
}

/// The link of one condition: its blocks and its metrics.
pub struct Link {
    /// The blocks, in order.
    pub blocks: Vec<Box<dyn Block>>,
    /// The type of every block (the context of its errors).
    pub kinds: Vec<String>,
    /// The label of every block (its `label`, else its type; a repeated label takes
    /// the position of the block, `awgn_3`), the names of the profile.
    pub labels: Vec<String>,
    /// The metrics.
    pub metrics: Metrics,
    /// Time every block: the trial records `<label>.time_ms` for every block and
    /// `link.time_ms` for the whole link (the metrics excluded), reported as means.
    pub profile: bool,
}

impl Link {
    /// Run one trial: the blocks in order, then the metrics and the recorded values.
    ///
    /// # Errors
    /// The first error of a block or of the metrics.
    pub fn trial(&self, rng: &mut Rng, rec: &mut Record) -> Result<Frame> {
        let mut s = Frame { trial: rec.trial(), ..Frame::default() };
        let t_link = Instant::now();
        for (i, (b, kind)) in self.blocks.iter().zip(&self.kinds).enumerate() {
            let t = Instant::now();
            b.apply(&mut s, rng).map_err(|e| e.context(format!("link block {} ({kind})", i + 1)))?;
            if self.profile {
                rec.value(&format!("{}.time_ms", self.labels[i]), t.elapsed().as_secs_f64() * 1e3);
            }
        }
        if self.profile {
            rec.value("link.time_ms", t_link.elapsed().as_secs_f64() * 1e3);
        }
        self.metrics.record_frame(&s, rec)?;
        Ok(s)
    }
}

/// A specification as a [`Scenario`] of the runner.
pub struct SpecScenario<'a> {
    spec: &'a Spec,
    profile: bool,
}

impl<'a> SpecScenario<'a> {
    /// The scenario of a specification.
    pub fn new(spec: &'a Spec) -> SpecScenario<'a> {
        SpecScenario { spec, profile: spec.profile }
    }

    /// Time every block, whatever the specification says.
    #[must_use]
    pub fn profile(mut self, profile: bool) -> Self {
        self.profile = profile;
        self
    }
}

impl Scenario for SpecScenario<'_> {
    type State = Link;

    fn name(&self) -> &str {
        &self.spec.name
    }

    fn setup(&self, condition: &Condition) -> Result<Link> {
        let mut link = self.spec.link_for(condition)?;
        link.profile = self.profile;
        Ok(link)
    }

    fn trial(&self, link: &Link, _: &Condition, rng: &mut Rng, record: &mut Record) -> Result<()> {
        link.trial(rng, record).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_labels_take_the_position_of_the_block() {
        let spec = Spec::from_yaml(
            "name: twice\nlink: [{type: source, n_symbols: 64}, {type: awgn, snr_db: 10}, {type: awgn, snr_db: 10}, {type: awgn, snr_db: 10, label: last}]\nstop: {max_trials: 1}",
        )
        .unwrap();
        let link = spec.link_for(&spec.conditions()[0]).unwrap();
        assert_eq!(link.labels, ["source", "awgn_2", "awgn_3", "last"]);
    }

    #[test]
    fn errors_name_their_place() {
        let e = Spec::from_yaml("name: t\nlink: [source]\nseeds: 3").unwrap_err().to_string();
        assert!(e.contains("seeds"), "{e}");
        let spec = Spec::from_yaml("name: t\nlink: [awgn_ebn0]").unwrap();
        let e = spec.link_for(&Condition::new()).err().unwrap().to_string();
        assert!(e.contains("link block 1"), "{e}");
        let spec = Spec::from_yaml("name: t\nlink: [{type: awgn_ebn0, ebn0_db: 3}, source]").unwrap();
        assert!(spec.link_for(&Condition::new()).err().unwrap().to_string().contains("first block"));
        let spec = Spec::from_yaml("name: t\nlink: [source]\nmetrics: {report: [fer]}").unwrap();
        assert!(spec.link_for(&Condition::new()).err().unwrap().to_string().contains("fer"));
    }
}
