//! A block of another crate joins the registry (ADR 0006): it is described with the
//! macros of the built-in blocks, listed, checked and run in a specification.
#![allow(clippy::unwrap_used, reason = "a test")]

use com_core::rng::Rng;
use com_spec::blocks::{self, Block, BuildContext, Extension, Frame, Params};
use com_spec::{RunOptions, Spec, block_info, block_params, run};
use serde_json::{Value as Json, json};

block_params! {
    /// The parameters of `scale_probe`.
    struct ScaleSpec {
        factor: f64 [2.0] > 0, "amplitude factor";
        record: String ["power"] one_of ["power", "none"], "what the block records";
    }
}

/// Scale the samples and record their mean power.
struct Scale {
    p: ScaleSpec,
    label: String,
}

impl Block for Scale {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> com_core::Result<()> {
        s.streams_mut().flatten().for_each(|v| *v *= self.p.factor);
        if self.p.record == "power" {
            let power = s.mean_power();
            s.record(&self.label, "power", power);
        }
        Ok(())
    }
}

fn build(p: Json, label: String, _: &BuildContext) -> com_core::Result<Box<dyn Block>> {
    Ok(Box::new(Scale { p: ScaleSpec::parse("scale_probe", p)?, label }))
}

block_params! {
    /// The parameters of `learned_gain` (none).
    struct LearnedSpec {}
}

/// A block trained once per condition: the gain that brings the training samples to
/// unit power, applied to every trial, which records the gain and the number of
/// symbols it was trained on.
struct Learned {
    gain: f64,
    trained_on: usize,
    label: String,
}

impl Block for Learned {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> com_core::Result<()> {
        s.streams_mut().flatten().for_each(|v| *v *= self.gain);
        s.record(&self.label, "gain", self.gain);
        s.record(&self.label, "trained_on", self.trained_on as f64);
        Ok(())
    }

    fn needs_training(&self) -> bool {
        true
    }

    fn train(&mut self, s: &Frame, _: &mut Rng) -> com_core::Result<()> {
        assert!(s.antennas.is_empty(), "a per-polarisation block is trained on one polarisation");
        self.gain = 1.0 / s.mean_power().sqrt();
        self.trained_on = s.symbols.len();
        Ok(())
    }
}

fn build_learned(p: Json, label: String, _: &BuildContext) -> com_core::Result<Box<dyn Block>> {
    LearnedSpec::parse("learned_gain", p)?;
    Ok(Box::new(Learned { gain: 0.0, trained_on: 0, label }))
}

fn register() {
    // the tests of this file share the registry: the first one registers the blocks
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        blocks::register(Extension {
            info: block_info!(
                per_polarization "learned_gain",
                "Bring the samples to unit power with a gain trained once per condition.",
                LearnedSpec,
                needs [Samples],
                gives [],
            ),
            build: build_learned,
        })
        .unwrap();
        blocks::register(Extension {
            info: block_info!(
                per_polarization "scale_probe",
                "Scale the samples and record their power.",
                ScaleSpec,
                needs [Samples],
                gives [],
            ),
            build,
        })
        .unwrap();
    });
}

#[test]
fn an_external_block_is_described_like_the_others() {
    register();
    let info = blocks::registry().into_iter().find(|b| b.name == "scale_probe").unwrap();
    assert!(info.per_polarization && !info.starts);
    assert_eq!(info.params[0].default_text(), "2.0");
    assert_eq!(info.params[1].choices, ["power", "none"]);
    assert!(info.describe().contains("amplitude factor"));
    let schema = blocks::json_schema();
    let alt = schema["oneOf"].as_array().unwrap().iter().find(|a| a["title"] == json!("scale_probe")).unwrap();
    assert_eq!(alt["properties"]["factor"]["anyOf"][0]["default"], json!(2.0));
    assert!(blocks::check_link(&["source", "scale_probe", "awgn"]).is_ok());
    let e = blocks::check_link(&["scale_probe"]).unwrap_err().to_string();
    assert!(e.contains("(scale_probe) needs samples"), "{e}");
    // its parameters are checked as those of the built-in blocks
    let ctx = BuildContext::default();
    let e = blocks::build(&json!({"type": "scale_probe", "factor": 0}), &ctx).err().unwrap().to_string();
    assert!(e.contains("factor must be > 0"), "{e}");
    let e = blocks::build(&json!({"type": "scale_probe", "gain": 1}), &ctx).err().unwrap().to_string();
    assert!(e.contains("unknown field `gain`"), "{e}");
    // a name is registered once
    let again = Extension { info: blocks::registry().into_iter().find(|b| b.name == "awgn").unwrap(), build };
    assert!(blocks::register(again).is_err());
}

#[test]
fn a_specification_runs_an_external_block() {
    register();
    let spec = Spec::from_yaml(
        "name: ext\nseed: 1\nlink: [{type: source, constellation: QPSK, n_symbols: 256}, \
         {type: scale_probe, factor: 3, label: gain}]\nmetrics: {report: [ser]}\nstop: {max_trials: 4}",
    )
    .unwrap();
    let r = run(&spec, &RunOptions::default()).unwrap();
    assert!((r[0].metric("gain.power").unwrap().value() - 9.0).abs() < 1e-9);
    assert_eq!(r[0].metric("ser").unwrap().value(), 0.0);
}

#[test]
fn a_per_polarisation_block_is_trained_on_one_polarisation() {
    register();
    // two polarisations of 256 symbols scaled by 3: the gain trained on polarisation x
    // brings them back to unit power
    let spec = Spec::from_yaml(
        "name: trained\nseed: 2\nlink: [{type: source, constellation: QPSK, n_symbols: 256, pols: 2}, \
         {type: scale_probe, factor: 3, record: none}, {type: learned_gain, label: rx}]\n\
         metrics: {report: [ser]}\nstop: {max_trials: 3}",
    )
    .unwrap();
    let r = run(&spec, &RunOptions::default()).unwrap();
    assert!((r[0].metric("rx.gain").unwrap().value() - 1.0 / 3.0).abs() < 1e-9);
    assert_eq!(r[0].metric("rx.trained_on").unwrap().value(), 256.0);
    assert_eq!(r[0].metric("ser").unwrap().value(), 0.0);
}
