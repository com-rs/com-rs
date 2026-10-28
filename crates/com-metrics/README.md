# com-metrics

[![crates.io](https://img.shields.io/crates/v/com-metrics.svg)](https://crates.io/crates/com-metrics)
[![docs.rs](https://img.shields.io/docsrs/com-metrics)](https://docs.rs/com-metrics)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

Error rates, SNR, EVM, the mutual information and the generalised mutual information, PAPR and CCDF, power spectral density (Welch, aligned periodogram) and ACLR.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::metrics`. The API is documented on
[docs.rs/com-metrics](https://docs.rs/com-metrics), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
