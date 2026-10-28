# com-dsp

[![crates.io](https://img.shields.io/crates/v/com-dsp.svg)](https://crates.io/crates/com-dsp)
[![docs.rs](https://img.shields.io/docsrs/com-dsp)](https://docs.rs/com-dsp)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 1 (generic signal processing) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

Waveform-agnostic signal processing: pulse shapes and FIR filtering, resampling, carrier, timing and frame synchronisation, the adaptive n×n butterfly equaliser (CMA, RDE, NLMS, DD-LMS), LS and LMMSE estimation, chromatic-dispersion compensation, digital back-propagation and digital predistortion.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::dsp`. The API is documented on
[docs.rs/com-dsp](https://docs.rs/com-dsp), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
