# com-systems

[![crates.io](https://img.shields.io/crates/v/com-systems.svg)](https://crates.io/crates/com-systems)
[![docs.rs](https://img.shields.io/docsrs/com-systems)](https://docs.rs/com-systems)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 4 (integration) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

Reference chains of standards and assemblies of several link-block crates: the physical layer of DVB-S2 (ETSI EN 302 307-1), from the BBFRAME to the scrambled PLFRAME and back, with its BCH and LDPC codes, the 28 MODCODs, normal and short frames and pilots; probabilistic amplitude shaping (Böcherer, Steiner & Schulte 2015) with the CCDM and the LDPC codes of `com-fec`.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::systems`. The API is documented on
[docs.rs/com-systems](https://docs.rs/com-systems), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
