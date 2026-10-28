# com-sim

[![crates.io](https://img.shields.io/crates/v/com-sim.svg)](https://crates.io/crates/com-sim)
[![docs.rs](https://img.shields.io/docsrs/com-sim)](https://docs.rs/com-sim)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 3 (simulation) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

The Monte Carlo engine: a scenario (one random trial of a link) run on every condition of a parameter grid, as many times as a stopping rule requires, on a thread pool, with every metric reported with a confidence interval. The results do not depend on the number of threads.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::sim`. The API is documented on
[docs.rs/com-sim](https://docs.rs/com-sim), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
