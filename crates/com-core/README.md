# com-core

[![crates.io](https://img.shields.io/crates/v/com-core.svg)](https://crates.io/crates/com-core)
[![docs.rs](https://img.shields.io/docsrs/com-core)](https://docs.rs/com-core)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 0 (the foundation) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

The vocabulary every block of com-rs shares: the multi-stream containers (`Bits`, `Llr`, `Symbols`, `Signal`, `DualPol`), the verb traits (`Map`, `Demap`, `Propagate`, `Impair`, …), the unit newtypes (`Db`, `EbN0`, `EsN0`, `Hertz`), the seeded generator `SimRng` (xoshiro256\*\*, one stream per trial) and the error type.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::core`. The API is documented on
[docs.rs/com-core](https://docs.rs/com-core), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
