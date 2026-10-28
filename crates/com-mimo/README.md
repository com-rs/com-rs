# com-mimo

[![crates.io](https://img.shields.io/crates/v/com-mimo.svg)](https://crates.io/crates/com-mimo)
[![docs.rs](https://img.shields.io/docsrs/com-mimo)](https://docs.rs/com-mimo)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

MIMO detection (ZF, MMSE, V-BLAST SIC, ML, sphere decoder, K-best) with soft outputs, the receiver of a link (spatial layers or Alamouti blocks), orthogonal space-time block codes (Alamouti, rate 3/4), eigenmode and zero-forcing precoding, water filling and capacity.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::mimo`. The API is documented on
[docs.rs/com-mimo](https://docs.rs/com-mimo), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
