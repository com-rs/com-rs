# com-mapping

[![crates.io](https://img.shields.io/crates/v/com-mapping.svg)](https://crates.io/crates/com-mapping)
[![docs.rs](https://img.shields.io/docsrs/com-mapping)](https://docs.rs/com-mapping)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

QAM, PSK, PAM, OOK and custom constellations with Gray labels, hard decisions and LLRs (exact or max-log), differential M-PSK, the sources (PRBS of ITU-T O.150, shaped symbols) and the closed-form error and information rates.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::mapping`. The API is documented on
[docs.rs/com-mapping](https://docs.rs/com-mapping), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
