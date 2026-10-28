# com-fec

[![crates.io](https://img.shields.io/crates/v/com-fec.svg)](https://crates.io/crates/com-fec)
[![docs.rs](https://img.shields.io/docsrs/com-fec)](https://docs.rs/com-fec)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

Channel coding: the CRCs and the scrambler of 5G NR, block, random and QPP interleavers, convolutional codes (Viterbi, BCJR) and their puncturing, the LTE turbo code, LDPC codes (PEG, repeat-accumulate, alist files; sum-product, min-sum, layered decoding), polar codes (SC, CRC-aided SCL), Reed–Solomon and binary BCH codes, constant-composition distribution matching, and the BI-AWGN capacity and normal approximation.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::fec`. The API is documented on
[docs.rs/com-fec](https://docs.rs/com-fec), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
