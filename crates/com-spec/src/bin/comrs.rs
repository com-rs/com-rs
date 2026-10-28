//! `comrs`: run declarative Monte Carlo campaigns (see `com_spec::cli`).

fn main() -> std::process::ExitCode {
    com_spec::cli::main(std::env::args().skip(1))
}
