# 0001 — Split `comrs` into the layered `com-*` workspace

Status: accepted (2026-10-26)

## Context

Up to 0.1, the library was one crate, `comrs`, organised like OptiCommPy (`comm`,
`dsp`, `models`, `sim`, `spec`). `docs/ARCHITECTURE.md` organises the code by function
in five layers of crates, with dependencies pointing downward only, multi-stream
containers and one verb trait per role.

## Decision

The code moves to the crates of the guide; the migration is staged:

| 0.1 (`comrs::…`) | 0.2 |
| --- | --- |
| `rng`, `error` | `com_core::{rng, error}` (+ `signal`, `traits`, `units`) |
| `math`, `dsp::fft` | `com_math`, `com_math::fft` |
| nearest-point search of `Constellation`, `log_sum_exp` | `com_math::nearest::Slicer`, `com_math::log_sum_exp` |
| split-step kernels of `Ssfm` | `com_math::nlse` |
| `dsp::core` | `com_dsp::{filter, resample, signal}` |
| `dsp::{carrier, clock, sync}` | `com_dsp::sync::{carrier, timing, frame}` |
| `dsp::equalization::{cdc, cd_*}` | `com_dsp::compensation::cd` (takes `β₂` instead of `FiberParams`) |
| `dsp::equalization::AdaptiveEqualizer` | `com_dsp::equalizer` (takes the constellation points) |
| `Ssfm::backpropagate*` | `com_dsp::compensation::dbp::Dbp` (the methods remain, as shortcuts) |
| `comm::modulation` | `com_mapping::{constellation, demapper}` |
| `comm::sources` | `com_mapping::source` |
| `comm::metrics::theory_*` | `com_mapping::theory::{ber, ser, mi, gmi}` (re-exported by `com_metrics`) |
| `comm::metrics` | `com_metrics` |
| `models::ce_ofdm` | `com_waveform::ce_ofdm` |
| `models::channels::{awgn, awgn_ebn0}` | `com_channel::awgn` |
| `models::{fiber, channels, amplification}` | `com_channel::optical::{fiber, pmd, edfa}` |
| `models::channels::phase_noise` | `com_hardware::oscillator::phase_noise` |
| `models::devices` | `com_hardware::optical` |
| `models::channels::DualPol` | `com_core::signal::DualPol` |
| `sim` | `com_sim` |
| `spec`, `io`, the `comrs` binary | `com_spec` (see ADR 0004) |
| `comrs` (facade, prelude) | `com` (renamed `comrs` before release) |

Stage 1 (this change) keeps the slice-based functions of 0.1 and adds the containers
(`Bits`, `Llr`, `Symbols`, `Signal`), the verb traits and the unit newtypes in
`com-core`, implemented by `Constellation` (`Map`, `Demap`), `Awgn`, `Ssfm` and
`fading::Rayleigh` (`Propagate`), `PhaseNoise` and `Cfo` (`Impair`). `com-fec`,
`com-mimo` and `com-systems` exist with their scope and no code.

Stage 2, not done here: port the remaining blocks to the traits with `Config` builders
and unit newtypes in their signatures, remove the module stutter of some types
(`fiber::FiberParams`, `pmd::PmdLink`), split `com_hardware::optical` into `laser`,
modulators and receiver, add `#[non_exhaustive]` to the configuration structs (today
plain structs with public fields, built by struct update), `ofdm` and
`single_carrier` in `com-waveform`, and `com-wasm`.

## Consequences

- Version 0.2.0, a breaking change for Rust users (the table above is the migration
  note). The Python package keeps its name and API (`import comrs`).
- Numerical results are unchanged: the moved kernels keep their order of operations
  and the generator is the same (ADR 0002), so the published results are reproduced
  bit for bit.

## Later status (2026-10-28)

Version 0.1 was never published: 0.2.0 is the first public release, and the table
above records the reorganisation rather than a migration. Of stage 2, OFDM, the verb
traits on more blocks (`docs/ARCHITECTURE.md`, "Traits") and `#[non_exhaustive]` on
the enums that will grow were done, while `com_hardware::optical` stayed one module
and `fiber::FiberParams` and `pmd::PmdLink` kept their names. The configuration
structs stay plain structs with public fields, changed by struct update and checked
when used (`docs/ARCHITECTURE.md`, "Configuration"); `single_carrier` and `com-wasm`
were not created.
