# com-spec

[![crates.io](https://img.shields.io/crates/v/com-spec.svg)](https://crates.io/crates/com-spec)
[![docs.rs](https://img.shields.io/docsrs/com-spec)](https://docs.rs/com-spec)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 4 (integration) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

Declarative campaigns: a YAML, JSON or TOML file describes the grid, the link as a list of blocks, the metrics, the stopping rule and the results expected from the literature. The crate also reads and writes recordings (SigMF, raw IQ, NPY) and provides the `comrs` command:

```bash
cargo install com-spec
comrs blocks              # the blocks of a link and their parameters
comrs run campaign.yaml   # reports under results/
comrs verify check.yaml   # fails if a result leaves its reference
```

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::spec`. The API is documented on
[docs.rs/com-spec](https://docs.rs/com-spec), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
