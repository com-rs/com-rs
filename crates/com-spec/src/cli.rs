//! The command-line tool `comrs`: run declarative Monte Carlo campaigns.
//!
//! ```text
//! comrs run SPEC... [--out DIR] [--threads N] [--fast-dev-run] [--profile] [--quiet]
//! comrs check SPEC...
//! comrs verify SPEC... [--out DIR] [--threads N] [--quiet]
//! comrs conditions SPEC
//! comrs blocks [NAME...] [--json]
//! comrs schema
//! ```
//!
//! `run` writes the CSV and JSON reports given by the specification, or
//! `DIR/<name>.csv` and `DIR/<name>.json` (`DIR` = `results` by default), and prints
//! every metric with its confidence interval (for a specification with a `search`, the
//! operating point of every combination of the other axes); `--profile` adds the mean
//! time of every block ([`profile`](crate::profile)). `check` builds the link of every
//! condition and runs one trial of each, without writing anything (for a specification
//! with a `search`, at the lower end of its range for every combination of the other
//! axes). `verify` runs the specifications and checks their expected results (closed
//! forms, published curves), prints every check, writes the reports of the run and
//! `DIR/<name>.verify.json`, and fails if a check fails. `conditions` lists the
//! conditions of the grid; `blocks` documents the blocks of the links (their parameters
//! with types, units and accepted values, and their ports; as JSON with `--json`);
//! `schema` prints the JSON Schema of a block of a link, for editors and graphical
//! front ends.
//!
//! The binary `comrs` of this crate is [`main`] on the arguments of the process; a crate
//! that adds blocks ([`blocks::register`]) wraps the same function in its own binary,
//! so that its command lists, checks and runs them with the others.

use crate::{RunOptions, Spec, blocks, run, search, verify};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  comrs run SPEC... [--out DIR] [--threads N] [--fast-dev-run] [--profile] [--quiet]
  comrs check SPEC...
  comrs verify SPEC... [--out DIR] [--threads N] [--quiet]
  comrs conditions SPEC
  comrs blocks [NAME...] [--json]
  comrs schema

SPEC is a YAML, JSON or TOML specification (see the documentation of the crate com-spec).";

#[allow(clippy::struct_excessive_bools, reason = "the flags of the command line")]
struct Args {
    command: String,
    files: Vec<PathBuf>,
    out: PathBuf,
    threads: Option<usize>,
    fast_dev_run: bool,
    profile: bool,
    quiet: bool,
    json: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut it = args.into_iter();
    let command = it.next().ok_or_else(|| USAGE.to_string())?;
    let mut a = Args {
        command,
        files: Vec::new(),
        out: PathBuf::from("results"),
        threads: None,
        fast_dev_run: false,
        profile: false,
        quiet: false,
        json: false,
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => a.out = PathBuf::from(it.next().ok_or("--out needs a directory")?),
            "--threads" => {
                let n = it.next().ok_or("--threads needs a number")?;
                a.threads = Some(n.parse().map_err(|_| format!("--threads: {n:?} is not a number"))?);
            }
            "--fast-dev-run" => a.fast_dev_run = true,
            "--profile" => a.profile = true,
            "--quiet" => a.quiet = true,
            "--json" => a.json = true,
            "-h" | "--help" => return Err(USAGE.to_string()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n\n{USAGE}")),
            s => a.files.push(PathBuf::from(s)),
        }
    }
    Ok(a)
}

/// The options of `run` and `verify`: the threads, the progress and the reports, those
/// of the specification or `DIR/<name>.csv` and `DIR/<name>.json`.
fn report_options(spec: &Spec, a: &Args) -> RunOptions {
    RunOptions {
        threads: a.threads,
        progress: Some(!a.quiet),
        csv: Some(spec.csv.clone().unwrap_or_else(|| a.out.join(format!("{}.csv", spec.name)))),
        json: Some(spec.json.clone().unwrap_or_else(|| a.out.join(format!("{}.json", spec.name)))),
        ..RunOptions::default()
    }
}

/// Say where the reports went, unless `--quiet`.
fn written(opts: &RunOptions, a: &Args) {
    if let (false, Some(csv), Some(json)) = (a.quiet, &opts.csv, &opts.json) {
        eprintln!("written {} and {}", csv.display(), json.display());
    }
}

fn run_one(path: &PathBuf, a: &Args) -> com_core::Result<()> {
    let spec = Spec::from_path(path)?;
    let opts =
        RunOptions { fast_dev_run: a.fast_dev_run, profile: a.profile.then_some(true), ..report_options(&spec, a) };
    if let Some(sp) = &spec.search {
        if !a.quiet {
            eprintln!(
                "{}: {} (search of {} for {} = {:e})",
                spec.name, spec.description, sp.along, sp.metric, sp.target
            );
        }
        for r in search::search(&spec, &opts)? {
            println!(
                "{} | {} = {:.4} [{:.4}, {:.4}] | {} evaluations{}",
                r.condition.label(),
                r.along,
                r.value,
                r.lo,
                r.hi,
                r.evaluations.len(),
                if r.converged { "" } else { " (not converged)" }
            );
        }
        written(&opts, a);
        return Ok(());
    }
    if !a.quiet {
        eprintln!("{}: {} ({} conditions)", spec.name, spec.description, spec.conditions().len());
    }
    let results = run(&spec, &opts)?;
    for r in &results {
        let metrics: Vec<String> = r
            .record
            .iter()
            .map(|(k, e)| {
                let (lo, hi) = e.interval(spec.confidence);
                format!("{k}={:.4e} [{lo:.3e}, {hi:.3e}]", e.value())
            })
            .collect();
        println!("{} | {} | {} trials | {}", r.condition.label(), r.status.as_str(), r.trials, metrics.join("  "));
    }
    let profile = crate::profile_of(&spec, &results)?;
    if !profile.is_empty() {
        println!("\nprofile: mean time per trial over {} conditions", results.len());
        for row in &profile {
            println!("{:>24}  {:>10.3} ms  {:>5.1} %", row.label, row.time_ms, 100.0 * row.share);
        }
    }
    written(&opts, a);
    Ok(())
}

/// Run a specification and check its expectations; returns the number of failures.
fn verify_one(path: &PathBuf, a: &Args) -> com_core::Result<usize> {
    let spec = Spec::from_path(path)?;
    // the reports of the run, as with `run`
    let opts = report_options(&spec, a);
    let (_, checks) = verify::verify_spec(&spec, &opts)?;
    let failed = checks.iter().filter(|c| !c.passed).count();
    println!("{}: {} checks, {} failed", spec.name, checks.len(), failed);
    for c in &checks {
        println!(
            "  {} {} | {} | {}: expected {:.4e}, measured {:.4e} [{:.3e}, {:.3e}]",
            if c.passed { "ok  " } else { "FAIL" },
            c.name,
            c.condition,
            c.metric,
            c.expected,
            c.measured.value,
            c.measured.lo,
            c.measured.hi
        );
    }
    verify::write_report(&a.out.join(format!("{}.verify.json", spec.name)), &spec, &opts, &checks)?;
    Ok(failed)
}

fn check_one(path: &PathBuf) -> com_core::Result<()> {
    let spec = Spec::from_path(path)?;
    // a search: the first point it evaluates, for every combination of the other axes
    let mut spec = search::first_points(&spec).unwrap_or(spec);
    // nothing is written
    (spec.csv, spec.json) = (None, None);
    let opts = RunOptions { fast_dev_run: true, batch: Some(1), progress: Some(false), ..RunOptions::default() };
    let r = run(&spec, &opts)?;
    let names: Vec<&String> = r.first().map(|c| c.record.iter().map(|(k, _)| k).collect()).unwrap_or_default();
    println!("{}: ok, {} conditions, {} blocks, metrics {:?}", path.display(), r.len(), spec.link.len(), names);
    Ok(())
}

/// Pretty JSON, or the error as a configuration error.
fn to_json(v: &impl serde::Serialize) -> com_core::Result<String> {
    serde_json::to_string_pretty(v).map_err(|e| com_core::Error::Config(format!("cannot serialise the report: {e}")))
}

/// Run the command line on its arguments (without the program name); the exit code is
/// 0 on success, 1 on a failure and 2 on a usage error.
pub fn main(args: impl IntoIterator<Item = String>) -> ExitCode {
    let a = match parse_args(args) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    if ["run", "check", "verify", "conditions"].contains(&a.command.as_str()) && a.files.is_empty() {
        eprintln!("no specification given\n\n{USAGE}");
        return ExitCode::from(2);
    }
    let result: com_core::Result<()> = match a.command.as_str() {
        "blocks" => {
            let all = blocks::registry();
            let wanted: Vec<String> = a.files.iter().map(|f| f.display().to_string()).collect();
            let shown: Vec<_> =
                all.iter().filter(|b| wanted.is_empty() || wanted.iter().any(|w| w == b.name)).collect();
            if shown.is_empty() {
                Err(com_core::Error::Config(format!("no block {wanted:?}")))
            } else if a.json {
                let all: Vec<_> = shown.iter().map(|b| b.to_json()).collect();
                to_json(&all).map(|j| println!("{j}"))
            } else {
                for b in shown {
                    println!("{}", b.describe());
                }
                Ok(())
            }
        }
        "schema" => to_json(&blocks::json_schema()).map(|j| println!("{j}")),
        "conditions" => a.files.iter().try_for_each(|f| {
            let spec = Spec::from_path(f)?;
            for (i, c) in spec.conditions().iter().enumerate() {
                println!("{i}: {}", c.label());
            }
            Ok(())
        }),
        "check" => a.files.iter().try_for_each(check_one),
        "run" => a.files.iter().try_for_each(|f| run_one(f, &a)),
        "verify" => {
            let mut failed = 0;
            let r = a.files.iter().try_for_each(|f| verify_one(f, &a).map(|n| failed += n));
            if r.is_ok() && failed > 0 {
                eprintln!("{failed} checks failed");
                return ExitCode::FAILURE;
            }
            r
        }
        other => {
            eprintln!("unknown command {other:?}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
