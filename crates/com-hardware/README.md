# com-hardware

[![crates.io](https://img.shields.io/crates/v/com-hardware.svg)](https://crates.io/crates/com-hardware)
[![docs.rs](https://img.shields.io/docsrs/com-hardware)](https://docs.rs/com-hardware)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

Transmitter and receiver non-idealities: power amplifiers (Rapp, Saleh, memory polynomial) and their predistortion, Wiener phase noise and frequency offset, transmitter and receiver IQ imbalance; with the feature `optical`, lasers, phase, Mach–Zehnder and IQ modulators, photodiodes with shot and thermal noise, the 90° hybrid and the coherent receiver.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::hardware`. The API is documented on
[docs.rs/com-hardware](https://docs.rs/com-hardware), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
