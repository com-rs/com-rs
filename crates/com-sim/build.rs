//! Records the commit of the working tree, so that every result carries it.
//!
//! The commit is `<hash>-dirty` when the sources of the library (the tracked files of
//! `crates/`, `bindings/`, `Cargo.toml` and `Cargo.lock`) differ from it, `unknown`
//! outside a Git checkout. The script reruns when `HEAD` moves (a checkout, or a new
//! commit on the branch it points to), when the index changes (a staged file) and when
//! one of these sources changes (an unstaged edit), so that the recorded commit is never
//! stale.

use std::path::Path;
use std::process::Command;

/// The sources that make the library; the dirty flag and the reruns follow them.
const SOURCES: [&str; 4] = ["crates", "bindings", "Cargo.toml", "Cargo.lock"];

fn main() {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let Some(top) = git(&["rev-parse", "--show-toplevel"]) else {
        println!("cargo:rustc-env=COM_GIT_COMMIT=unknown");
        println!("cargo:rerun-if-changed=build.rs");
        return;
    };
    let in_top = |args: &[&str]| {
        let mut all = vec!["-C", top.as_str()];
        all.extend_from_slice(args);
        git(&all)
    };
    // no optional lock: the status does not rewrite the index it watches
    let mut status = vec!["--no-optional-locks", "status", "--porcelain", "--untracked-files=no", "--"];
    status.extend_from_slice(&SOURCES);
    let commit = match in_top(&["rev-parse", "HEAD"]) {
        Some(c) if in_top(&status).is_some_and(|s| !s.is_empty()) => format!("{c}-dirty"),
        Some(c) => c,
        None => "unknown".into(),
    };
    println!("cargo:rustc-env=COM_GIT_COMMIT={commit}");
    println!("cargo:rerun-if-changed=build.rs");
    // a path that does not exist would rerun the script at every build
    let watch = |path: &str| {
        let p = Path::new(&top).join(path);
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    };
    // HEAD, the index and the branch HEAD points to (a loose ref, or the packed refs);
    // `--git-path` resolves them in a linked worktree too
    let mut paths = vec!["HEAD".to_string(), "index".to_string(), "packed-refs".to_string()];
    paths.extend(in_top(&["symbolic-ref", "-q", "HEAD"]));
    for p in paths {
        if let Some(full) = in_top(&["rev-parse", "--git-path", &p]) {
            watch(&full);
        }
    }
    // the tracked sources: an unstaged edit changes no file of the Git directory
    let mut files = vec!["ls-files", "--"];
    files.extend_from_slice(&SOURCES);
    for f in in_top(&files).unwrap_or_default().lines() {
        watch(f);
    }
}
