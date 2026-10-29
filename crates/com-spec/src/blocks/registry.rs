//! The registry of the blocks: their documentation ([`BlockInfo`]), the families of
//! this crate, the blocks of other crates ([`register`]) and the builder ([`build`]).

use super::per_pol::PerPol;
use super::schema::{self, Fact, ParamDefault, ParamInfo};
use super::{Block, BuildContext};
use super::{analysis, ce_ofdm, channel, coding, cpm, dsp, fiber, optical, source, systems, wireless};
use crate::params::split_block;
use com_core::{Error, Result};
use serde_json::{Map, Value as Json};
use std::fmt::Write;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

/// The documentation of a block ([`BlockInfo`](crate::blocks::BlockInfo)) from its
/// parameters, declared with [`block_params!`](crate::block_params), and its ports.
///
/// ```text
/// block_info!([start | per_polarization] NAME, SUMMARY, PARAMS, needs [FACT, …], gives [FACT, …])
/// ```
///
/// `start` marks a block that starts a link (it draws or reads the transmitted signal),
/// `per_polarization` a block written for one polarisation, applied to each
/// polarisation of a dual-polarisation channel with its own symbols; the facts are
/// variants of [`Fact`](crate::blocks::Fact). See [`register`](crate::blocks::register)
/// for a block of another crate.
#[macro_export]
macro_rules! block_info {
    (@info $starts:expr, $per_pol:expr, $name:expr, $summary:expr, $params:ty,
        needs [$($n:ident),* $(,)?], gives [$($g:ident),* $(,)?] $(,)?) => {
        $crate::blocks::BlockInfo {
            name: $name,
            summary: $summary,
            params: <$params as $crate::blocks::Params>::params(),
            needs: &[$($crate::blocks::Fact::$n),*],
            gives: &[$($crate::blocks::Fact::$g),*],
            starts: $starts,
            per_polarization: $per_pol,
        }
    };
    (start $($rest:tt)+) => { $crate::block_info!(@info true, false, $($rest)+) };
    (per_polarization $($rest:tt)+) => { $crate::block_info!(@info false, true, $($rest)+) };
    ($($rest:tt)+) => { $crate::block_info!(@info false, false, $($rest)+) };
}

/// The documentation of a block: its parameters and its ports.
#[derive(Clone, Debug)]
pub struct BlockInfo {
    /// The type of the block.
    pub name: &'static str,
    /// What it does.
    pub summary: &'static str,
    /// Its parameters.
    pub params: Vec<ParamInfo>,
    /// The facts of the frame it reads.
    pub needs: &'static [Fact],
    /// The facts it may add to the frame.
    pub gives: &'static [Fact],
    /// Whether it starts a link (it draws or reads the transmitted signal).
    pub starts: bool,
    /// Whether it is written for one polarisation: with a dual-polarisation channel
    /// ([`Layout::polarisations`](super::Layout::polarisations)) each polarisation goes
    /// through it with its own symbols.
    pub per_polarization: bool,
}

impl BlockInfo {
    /// The block as text (name, summary, ports, one line per parameter).
    pub fn describe(&self) -> String {
        let names = |f: &[Fact]| {
            if f.is_empty() { "nothing".to_string() } else { f.iter().map(|f| f.name()).collect::<Vec<_>>().join(", ") }
        };
        let mut s = format!("{}\n    {}\n", self.name, self.summary);
        let start = if self.starts { "starts a link; " } else { "" };
        let _ = writeln!(s, "    [{start}needs: {}; gives: {}]", names(self.needs), names(self.gives));
        for p in &self.params {
            s.push_str(&p.describe(4));
        }
        s
    }

    /// The block as JSON: name, summary, ports and the JSON Schema of every parameter.
    pub fn to_json(&self) -> Json {
        let params: Map<String, Json> = self.params.iter().map(|p| (p.name.to_string(), p.json_schema())).collect();
        serde_json::json!({
            "name": self.name,
            "summary": self.summary,
            "starts": self.starts,
            "per_polarization": self.per_polarization,
            "needs": self.needs.iter().map(|f| f.name()).collect::<Vec<_>>(),
            "gives": self.gives.iter().map(|f| f.name()).collect::<Vec<_>>(),
            "required": self.params.iter().filter(|p| matches!(p.default, ParamDefault::Required)).map(|p| p.name).collect::<Vec<_>>(),
            "params": params,
        })
    }
}

/// The builder of a family of blocks: the block of type `kind` with its label, its
/// parameters (without `type` and `label`) and the context, or `None` for a type the
/// family does not define.
type BuildFn = fn(kind: &str, label: &str, p: &Json, ctx: &BuildContext) -> Option<Result<Box<dyn Block>>>;

/// The documentation of the blocks of a family.
type InfosFn = fn() -> Vec<BlockInfo>;

/// The families of blocks of this crate, in the order of the registry: the
/// documentation of their blocks and their builder.
const FAMILIES: &[(InfosFn, BuildFn)] = &[
    (source::infos, source::build),
    (channel::infos, channel::build),
    (fiber::infos, fiber::build),
    (dsp::infos, dsp::build),
    (ce_ofdm::infos, ce_ofdm::build),
    (cpm::infos, cpm::build),
    (analysis::infos, analysis::build),
    (optical::infos, optical::build),
    (wireless::infos, wireless::build),
    (coding::infos, coding::build),
    (systems::infos, systems::build),
];

/// The blocks of this crate with the builder of their family, built once.
fn builtins() -> &'static [(BlockInfo, BuildFn)] {
    static BUILTINS: OnceLock<Vec<(BlockInfo, BuildFn)>> = OnceLock::new();
    BUILTINS.get_or_init(|| {
        FAMILIES.iter().flat_map(|&(infos, build)| infos().into_iter().map(move |info| (info, build))).collect()
    })
}

/// A block defined outside this crate, added to the registry by [`register`].
///
/// `build` receives the parameters of the block (without `type` and `label`), its
/// label and the context, as the builders of this crate do.
#[derive(Clone)]
pub struct Extension {
    /// The documentation of the block.
    pub info: BlockInfo,
    /// Its builder.
    pub build: fn(Json, String, &BuildContext) -> Result<Box<dyn Block>>,
}

static EXTENSIONS: Mutex<Vec<Extension>> = Mutex::new(Vec::new());

/// The blocks added by [`register`], locked.
fn extensions() -> MutexGuard<'static, Vec<Extension>> {
    EXTENSIONS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Add a block to the registry: it is listed by [`registry`], built by [`build`],
/// described by [`json_schema`] and checked by [`check_link`] like the others. A crate
/// that defines blocks registers them before it runs a specification, and may ship its
/// own `comrs` binary around [`cli::main`](crate::cli::main).
///
/// # Errors
/// [`Error::Config`] when a block of that name exists.
///
/// # Examples
///
/// A block of another crate: its parameters are declared with
/// [`block_params!`](crate::block_params) and its ports with
/// [`block_info!`](crate::block_info), as those of the blocks of this crate, so that
/// `comrs blocks`, the JSON Schema and the checks of the links describe it too.
///
/// ```
/// use com_core::rng::Rng;
/// use com_spec::blocks::{self, Block, BuildContext, Extension, Frame, Params};
/// use com_spec::{RunOptions, Spec, block_info, block_params, run};
/// use serde_json::Value as Json;
///
/// block_params! {
///     struct GainSpec {
///         gain_db: f64 [0.0] "dB", "power gain";
///     }
/// }
///
/// /// Multiply every sample by an amplitude gain.
/// struct Gain(f64);
///
/// impl Block for Gain {
///     fn apply(&self, s: &mut Frame, _: &mut Rng) -> com_core::Result<()> {
///         s.streams_mut().flatten().for_each(|v| *v *= self.0);
///         Ok(())
///     }
/// }
///
/// fn build(p: Json, _label: String, _: &BuildContext) -> com_core::Result<Box<dyn Block>> {
///     let q = GainSpec::parse("gain", p)?;
///     Ok(Box::new(Gain(com_math::db2amp(q.gain_db))))
/// }
///
/// blocks::register(Extension {
///     info: block_info!("gain", "Amplify the samples.", GainSpec, needs [Samples], gives []),
///     build,
/// })?;
/// let spec = Spec::from_yaml(
///     "name: amplified\nlink: [source, {type: gain, gain_db: 6}]\n\
///      metrics: {report: [ser]}\nstop: {max_trials: 2}",
/// )?;
/// let results = run(&spec, &RunOptions::default())?;
/// assert_eq!(results[0].metric("ser").unwrap().value(), 0.0);
/// // listed with its parameters, and checked as the blocks of this crate
/// let info = blocks::registry().into_iter().find(|b| b.name == "gain").unwrap();
/// assert_eq!((info.params[0].name, info.params[0].unit), ("gain_db", "dB"));
/// let unknown = serde_json::json!({"type": "gain", "gain": 6});
/// assert!(blocks::build(&unknown, &BuildContext::default()).is_err());
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn register(ext: Extension) -> Result<()> {
    let name = ext.info.name;
    let mut extensions = extensions();
    if builtins().iter().any(|(b, _)| b.name == name) || extensions.iter().any(|e| e.info.name == name) {
        return Err(Error::Config(format!("a block {name:?} is already registered")));
    }
    extensions.push(ext);
    Ok(())
}

/// Every block of the registry, in the order of a link: the blocks of this crate, then
/// those added by [`register`].
pub fn registry() -> Vec<BlockInfo> {
    let mut all: Vec<BlockInfo> = builtins().iter().map(|(info, _)| info.clone()).collect();
    all.extend(extensions().iter().map(|e| e.info.clone()));
    all
}

/// Build a block from its JSON description.
///
/// # Errors
/// [`Error::Config`] for an unknown type (the message lists the known ones), an unknown
/// or ill-typed parameter, or an invalid value.
pub fn build(v: &Json, ctx: &BuildContext) -> Result<Box<dyn Block>> {
    let (kind, label, p) = split_block(v)?;
    let label = label.unwrap_or_else(|| kind.clone());
    let unknown = || {
        let known: Vec<&str> = registry().iter().map(|b| b.name).collect();
        Error::Config(format!("unknown block {kind:?}; the blocks are {known:?}"))
    };
    let builtin = builtins().iter().find(|(info, _)| info.name == kind);
    let (block, gives, per_polarization) = if let Some((info, family)) = builtin {
        (family(&kind, &label, &p, ctx).ok_or_else(unknown)??, info.gives, info.per_polarization)
    } else {
        // the lock is released before the builder runs
        let found =
            extensions().iter().find(|e| e.info.name == kind).map(|e| (e.build, e.info.gives, e.info.per_polarization));
        let (builder, gives, per_polarization) = found.ok_or_else(unknown)?;
        (builder(p, label, ctx)?, gives, per_polarization)
    };
    let block: Box<dyn Block> = if per_polarization { Box::new(PerPol(block)) } else { block };
    // debug builds check the declared ports of every block while it runs
    #[cfg(debug_assertions)]
    let block: Box<dyn Block> = Box::new(schema::Checked { inner: block, kind, gives });
    #[cfg(not(debug_assertions))]
    let _ = gives;
    Ok(block)
}

/// Check the ports of a link of block types (see [`schema::check_link`]).
///
/// # Errors
/// [`Error::Config`] naming the first block that misses a fact.
pub fn check_link(kinds: &[&str]) -> Result<()> {
    schema::check_link(kinds, &registry())
}

/// The JSON Schema of a block of a link (see [`schema::json_schema`]).
pub fn json_schema() -> Json {
    schema::json_schema(&registry())
}
