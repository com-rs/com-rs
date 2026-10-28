# com-math

[![crates.io](https://img.shields.io/crates/v/com-math.svg)](https://crates.io/crates/com-math)
[![docs.rs](https://img.shields.io/docsrs/com-math)](https://docs.rs/com-math)

Part of [com-rs](https://github.com/com-rs/com-rs), link-level simulation of digital communication systems,
wireless and optical. Layer 0 (the foundation) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)).

The numerical methods without a communication meaning: decibel conversions, special functions, Gauss–Hermite quadrature, FFT plans, complex linear algebra (QR, Cholesky, SVD), nearest-point search, the split-step solvers of the nonlinear Schrödinger and Manakov equations, arithmetic over GF(2) and GF(2^m), exact integers, the Gray code, bisection and golden-section search.

Most users depend on the facade [`comrs`](https://crates.io/crates/comrs), which
re-exports this crate as `comrs::math`. The API is documented on
[docs.rs/com-math](https://docs.rs/com-math), with the formula that every item implements.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE), shipped in the package).
