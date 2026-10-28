# com-waveform

[![crates.io](https://img.shields.io/crates/v/com-waveform.svg)](https://crates.io/crates/com-waveform)
[![docs.rs](https://img.shields.io/docsrs/com-waveform)](https://docs.rs/com-waveform)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 2 (link blocks) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

OFDM and DFT-spread OFDM with resource grids, pilots, LS and LMMSE channel estimation, Schmidl–Cox synchronisation and MIMO-OFDM detection; continuous-phase modulation (CPFSK, MSK, GMSK) with its detectors; constant-envelope OFDM and the theory of its error rates; the distribution of the PAPR and its reduction (clipping, SLM, PTS, tone reservation, ACE, companding).

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::waveform`. The API is documented on
[docs.rs/com-waveform](https://docs.rs/com-waveform), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
