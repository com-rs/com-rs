//! Records the commit of the working tree, so that every result carries it.

use std::process::Command;

fn main() {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let commit = match git(&["rev-parse", "HEAD"]) {
        Some(c) if git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty()) => {
            format!("{c}-dirty")
        }
        Some(c) => c,
        None => "unknown".into(),
    };
    println!("cargo:rustc-env=COM_GIT_COMMIT={commit}");
    if let Some(dir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={dir}/HEAD");
        println!("cargo:rerun-if-changed={dir}/index");
    }
}
