# com-rs — Architecture & Engineering Guide

Vincent Choqueuse · revised 30 October 2026

This guide describes the workspace as it is: its layers, where code goes, how it is
named, documented, tested and released. Its exceptions are recorded in the
architecture decision records of [`adr/`](adr/); when the code and the guide disagree,
the guide is wrong and is fixed.

## Purpose and design principles

com-rs is a Rust workspace for link-level simulation of digital communication systems,
wireless and optical, with Python bindings (the package `comrs`) and a command line
that runs declarative campaigns (`comrs`). It targets research-grade correctness and
reproducibility first, then performance, then ergonomics.

Six principles govern every design decision. When a new feature does not fit cleanly,
revisit the principles before bending the structure.

1. **Organise by function, never by domain.** A module answers "what does this block
   do in the chain?" (encode, map, shape, propagate, impair, estimate, detect,
   measure). Domains such as optics or a standard only appear as submodules
   (`channel::optical`) or as assembled systems (`com-systems`).
2. **Multi-stream containers.** The containers of `com-core` (`Bits`, `Llr`,
   `Symbols`, `Signal`) carry an explicit stream dimension, SISO being one stream, and
   the verb traits work on them, so MIMO and polarisation multiplexing reuse the same
   blocks. Most kernels are written on one stream (`&[C64]`, `&[u8]`, `&[f64]`); the
   multi-stream blocks apply them stream by stream (`Signal::map_streams`).
3. **Dependencies point downward only.** Foundation crates know nothing about the
   crates above them. Two crates of the same layer do not depend on each other unless
   an ADR records the edge (four do, ADRs 0007–0010). If a block needs a sibling, the
   shared part moves one layer down.
4. **Propagation and hardware are separate concerns.** `com-channel` models the
   medium; `com-hardware` models component non-idealities. Any impairment combines
   with any channel.
5. **Physics is explicit.** A quantity carries its unit in its type or in its name.
   The unit newtypes of `com-core` (`Db`, `EbN0`, `EsN0`, `Hertz`, `Seconds`) are used
   by the verb-trait blocks; the functions and configuration fields of the numerical
   kernels take `f64` named with their unit (`ebn0_db`, `alpha_db_km`, `fs`,
   `linewidth_hz`, `beta2_s2_km`).
6. **Every result is reproducible.** Randomness is seeded and derived from the trial,
   never from the worker; every JSON report records the library version, the commit
   and the configuration (with its seed) that produced it.

## Workspace layout

The workspace has five layers: foundation, signal processing, link blocks, simulation
and integration. Every crate is published under the `com-` prefix and re-exported by
the facade `comrs`, so users write `comrs::fec::ldpc` rather than `com_fec::ldpc`.

```text
com-rs/
├── Cargo.toml                 # [workspace]: members, shared lints, dependency versions
├── crates/
│   │  ── Layer 0: foundation ──────────────────────────────────────────────
│   ├── com-core/              # signal, traits, units, rng, error
│   ├── com-math/              # special functions, quadrature (lib), fft, linalg, nearest,
│   │                          # geometry, nlse, gf2, gf2m, bigint, bits, search, sequences
│   │  ── Layer 1: generic signal processing ───────────────────────────────
│   ├── com-dsp/
│   │   ├── filter  resample  signal  window  interpolation
│   │   ├── sync/              # carrier, timing, frame
│   │   ├── equalizer          # the n×n butterfly: CMA, RDE, NLMS, DD-LMS
│   │   ├── estimation         # LS, LMMSE (waveform-agnostic)
│   │   ├── nonlinear          # Bussgang's decomposition of memoryless nonlinearities
│   │   ├── spectrum  cyclo    # Welch's PSD; cyclic autocorrelation, symbol rate
│   │   └── compensation/      # cd, dbp, dpd
│   │  ── Layer 2: link blocks ─────────────────────────────────────────────
│   ├── com-fec/               # crc, scrambling, interleaving, block, conv, puncture,
│   │   │                      # turbo, ldpc, polar, rs, bch, algebraic, ccdm, bounds
│   │   └── nr/                # 5G NR LDPC and polar codes (TS 38.212: ldpc, polar, tables)
│   ├── com-mapping/           # constellation, demapper, differential, source, theory,
│   │                          # bicm, imperfect_csi
│   ├── com-waveform/
│   │   ├── ofdm/              # grid, estimation, sync, detection, link, correction, nr
│   │   ├── papr               # distribution and reduction of the PAPR
│   │   ├── ce_ofdm/           # constant-envelope OFDM (gamp, theory)
│   │   ├── cpm/               # continuous-phase modulation (pulse, detection)
│   │   ├── css/               # chirp spread spectrum, LoRa (theory)
│   │   ├── fbmc/              # FBMC/OQAM
│   │   └── ufmc  otfs         # universal filtered multicarrier; OTFS
│   ├── com-mimo/              # detection, message_passing, receiver, stbc, precoding,
│   │                          # capacity
│   ├── com-channel/
│   │   ├── awgn  fading  spatial
│   │   ├── multipath          # TDL profiles, the delay–Doppler channel and its model
│   │   ├── cdl/               # 3GPP TR 38.901 (tables, array)
│   │   ├── owc/               # optical wireless: indoor (rooms, VLC), atmospheric (FSO)
│   │   └── optical/           # fiber, manakov, pmd, edfa, gn, raman/   [feature: optical]
│   ├── com-hardware/
│   │   ├── amplifier  oscillator  iq  converter
│   │   └── optical            # laser, modulators, photodiode, receiver [feature: optical]
│   ├── com-metrics/           # errors, information, spectrum, blind, stability
│   │  ── Layer 3: simulation ──────────────────────────────────────────────
│   ├── com-sim/               # scenario, grid, runner, stopping, callbacks, loggers, stats,
│   │                          # json
│   │  ── Layer 4: integration ─────────────────────────────────────────────
│   ├── com-spec/              # spec, params, link, metrics, run, blocks/, verify, search,
│   │                          # io, cli, the `comrs` binary
│   ├── com-systems/           # dvbs2/ (with fec), pas
│   └── comrs/                 # facade: re-exports, prelude, examples, benchmarks
├── bindings/com-py/           # PyO3 extension of the Python package (built by maturin)
├── python/comrs/              # the Python package: chains, results, plots, generated blocks
├── validation/                # statistical validation (crate com-validation): tests/, cards/
│                              # (validation cards), aff3ct/ fixtures, literature.yaml,
│                              # maturity.yaml
├── specs/                     # specifications; specs/validation/: checked against the literature
├── tests/                     # Python tests; tests/oracle/: recorded OptiCommPy, GNPy, py3gpp
│                              # and Sionna outputs
├── examples/  benches/        # runnable examples and criterion benchmarks of the facade
├── benchmarks/                # speed against OptiCommPy (bench_opticommpy.py, results.md);
│                              # libraries/: results and speed against komm, OptiCommPy, Sionna
├── data/                      # a synthetic SigMF capture (`just make-capture`)
├── tools/                     # generators and checks: API.md, the stub, comrs.blocks,
│                              # LITERATURE.md, MATURITY.md, the cards, the benchmarks page
├── references/                # the list of the documents the code is checked against
└── docs/                      # this guide, adr/, API.md, LITERATURE.md, MATURITY.md, audit/,
                               # katex-header.html, python/ (Sphinx)
```

## Crate reference

Each crate owns one responsibility and states what it must not contain. The "Must not
contain" column is as binding as the first one.

| Crate | Layer | Owns | Must not contain | Depends on |
| --- | --- | --- | --- | --- |
| `com-core` | 0 | Signal containers, verb traits, unit newtypes, the generator and its streams, the error type | Any algorithm; any heavy dependency | `num-complex`, `ndarray`, `thiserror` |
| `com-math` | 0 | Special functions (`erfc`, `erfcinv`, `qfunc`, `qfuncinv`, `diric`), Simpson, Gauss–Hermite and Gauss–Legendre quadrature, FFT plans and their unitary and centred forms, complex linear algebra (tridiagonal systems included), vectors of 3-D space, nearest-point search, the NLSE split-step solvers, GF(2) and GF(2^m), exact integers, Gray code, LFSR sequences, bisection and golden-section search, decibels, entropy | Communication semantics (no "symbol", no "channel") | `com-core`, `rustfft` |
| `com-dsp` | 1 | Waveform-agnostic signal processing: filters, resampling, cyclic prefixes and frequency shifts, synchronisation, equalisation, estimation, interpolation, spectral estimation, cyclostationary analysis, memoryless nonlinearities driven by Gaussian signals (Bussgang), compensation, predistortion | Code that assumes a specific waveform or standard | 0 |
| `com-fec` | 2 | Channel codes (the 5G NR LDPC and polar codes included), CRC, scrambling, interleaving, puncturing, distribution matching, finite-length bounds | Modulation or channel knowledge beyond LLR input | 0 |
| `com-mapping` | 2 | Constellations, labelling, hard and soft demapping, differential PSK, sources, the closed forms of error and information rates (BICM cutoff rates, imperfect channel estimates included) | Pulse shaping or framing | 0 |
| `com-waveform` | 2 | Time–frequency structure: OFDM and its grid, PAPR, CE-OFDM, CPM, CSS, FBMC/OQAM, UFMC, OTFS | Channel models; MIMO detection algorithms | 0, 1, `com-mapping`, `com-mimo` (ADR 0009) |
| `com-mimo` | 2 | Detection (message passing included), the receiver of a link, space-time codes, precoding, capacity | Waveform-specific resource mapping | 0, `com-mapping` (ADR 0008) |
| `com-channel` | 2 | Propagation media: AWGN, fading, spatial correlation, TDL and CDL multipath, delay–Doppler channels, optical fibre, optical wireless (indoor rooms, free-space optics) | Component non-idealities | 0, 1 |
| `com-hardware` | 2 | Transmitter and receiver non-idealities, electrical and optical: amplifiers, oscillators, IQ imbalance, converters, lasers, modulators, photodiodes | Propagation effects | 0, 1 |
| `com-metrics` | 2 | Error counts and rates, SNR, EVM, information rates, metrics of spectra (ACLR), blind estimates (cumulants, modulation classification, the SNR), the frequency stability of oscillators (Allan variances) | Simulation control flow | 0, 1, `com-mapping` (ADR 0007) |
| `com-sim` | 3 | Monte Carlo engine, grids, stopping rules, callbacks, loggers, statistics | Physical-layer algorithms | 0 (`rayon`) |
| `com-spec` | 4 | Specifications, their blocks and registry, verification, operating-point search, recordings I/O, the `comrs` command (ADR 0004) | Algorithms of the blocks (they live in layers 0–2) | 0–3, `com-systems` (ADR 0010); `serde`, `serde_yaml`, `toml` |
| `com-systems` | 4 | Chains of standards and assemblies of several layer-2 crates: `dvbs2` (with its FEC), `pas` | New algorithms (they belong in layers 1–2) | 0–2 |
| `comrs` | 4 | The facade: re-exports, prelude, features `optical` and `spec` | Any code of its own | all |

`com-math` depends on `com-core` for `C64` only; both are layer 0. Layer 2 has four
sibling edges, all recorded: `com-metrics → com-mapping` (ADR 0007), `com-mimo →
com-mapping` (ADR 0008), `com-waveform → com-mapping` and `com-waveform → com-mimo`
(ADR 0009). In layer 4, `com-spec → com-systems` (ADR 0010). Any new edge between two
crates of one layer requires an ADR.

The Python extension (`bindings/com-py`) depends on the facade and adds no algorithm:
it converts arrays and calls the crates. `validation/` (the crate `com-validation`,
not published) depends on the facade.

## Placement rules

One rule decides where new code goes: **place it by what it does in the chain; if it
depends on a specific waveform, it lives under that waveform; if something else could
reuse it, it moves one layer down.**

Apply the questions in order and stop at the first "yes":

1. Is it a numerical method with no communication meaning? → `com-math`.
2. Is it a type or trait every block needs? → `com-core`.
3. Does it only make sense for one waveform or standard, or does it assemble several
   crates of layer 2? → that waveform's submodule, or `com-systems`.
4. Does it model the medium, or a component? → `com-channel` or `com-hardware`.
5. Otherwise → the functional crate matching its role in the chain.

Before adding a function, search [`API.md`](API.md), the generated index of every
public item: reuse the prototype that exists, or move it one layer down.

The following cases are settled. Treat them as precedent.

| Item | Location | Rationale |
| --- | --- | --- |
| NLSE split-step solvers (scalar, Manakov) | `com_math::nlse` | Shared by forward propagation and back-propagation; avoids a `dsp` → `channel` edge |
| Fibre channel | `com_channel::optical::fiber` | Thin wrapper supplying physical parameters (β₂, γ, α) to the solver |
| Digital back-propagation | `com_dsp::compensation::dbp` | Receiver processing; runs the same solver with negated parameters |
| Chromatic dispersion compensation | `com_dsp::compensation::cd` | An all-pass frequency-domain filter, not an optical concept |
| Digital predistortion | `com_dsp::compensation::dpd` | Receiver-side estimation of an inverse; the amplifier models stay in `com_hardware::amplifier` |
| Laser phase noise | `com_hardware::optical::Laser` | Reuses the Wiener model of `com_hardware::oscillator` |
| IQ imbalance | `com_hardware::iq` | A component of the transmitter or receiver |
| CE-OFDM | `com_waveform::ce_ofdm` | A waveform: a real OFDM signal driving a phase modulator |
| LS / LMMSE estimators | `com_dsp::estimation` | Waveform-agnostic |
| Interpolation between known positions (linear, quadratic, spline, band-limited, cubic Lagrange) | `com_dsp::interpolation` | Waveform-agnostic; shared by the pilot interpolation of OFDM and the timing recovery |
| Pilot interpolation on a resource grid | `com_waveform::ofdm::estimation` | Depends on the grid layout |
| MIMO detectors (ZF, MMSE, SIC, ML, sphere, K-best) | `com_mimo::detection` | Operate on a generic $y = H x + w$ model |
| The receiver of a link (layers, Alamouti, soft outputs) | `com_mimo::receiver` | Waveform-agnostic; used by the OFDM link and the `mimo_rx` block |
| Per-resource-element MIMO detection | `com_waveform::ofdm::{detection, link}` | Thin wrappers applying `com_mimo` across the grid (ADR 0009) |
| Polarisation demultiplexing | `com_dsp::equalizer` (2×2 CMA/RDE/LMS) | Polarisation multiplexing is 2×2 MIMO; no optical-specific equaliser |
| Soft demapper (LLR) | `com_mapping::demapper` | Inverse of the mapper, owned by the same crate |
| Closed forms of error and information rates | `com_mapping::theory` | Properties of a constellation and its labels (ADR 0007), re-exported by `com_metrics` |
| Gray code, bit ↔ index conversions | `com_math::bits` | No communication meaning; shared by mapping, coding and shaping |
| GF(2^m) arithmetic | `com_math::gf2m` | Shared by the BCH and Reed–Solomon codes |
| DVB-S2 BCH and LDPC codes and their tables | `com_systems::dvbs2::fec` | Only make sense for one standard |
| Probabilistic amplitude shaping | `com_systems::pas` | Assembles the CCDM and the LDPC code of `com-fec` and a PAM of `com-mapping` |
| `ber_to_q_db` | `com_metrics::errors` | A way of reporting a measured error rate |
| Bussgang's decomposition, the output series of a nonlinearity driven by a Gaussian signal | `com_dsp::nonlinear` | Shared by the clipping of OFDM (`com_waveform::papr`) and the amplifier models (`com_hardware::amplifier`), which are siblings of layer 2 |
| Welch's PSD and the averaged periodogram | `com_dsp::spectrum` | Spectral estimation, used by tests of the waveforms; `com_metrics::spectrum` re-exports them next to the ACLR |
| Cyclic prefix, frequency shift, unitary and centred FFT | `com_dsp::signal`, `com_math::fft` | Shared by OFDM, CE-OFDM, NR, OTFS and the oscillator impairments |
| Delay–Doppler channel and its random model | `com_channel::multipath::{DelayDoppler, DelayDopplerModel}` | A medium; OTFS only takes its paths (`Otfs::effective_channel`) |
| SER of $M$ orthogonal signals detected noncoherently | `com_mapping::theory::ser_orthogonal_noncoherent` | Shared by noncoherent FSK and chirp spread spectrum |
| Mutual information over fading (CM and BICM capacities) | `com_mapping::theory::{constellation_mi, constellation_gmi}` | One kernel for the closed forms over AWGN and Rayleigh fading; `com_mapping::bicm` keeps the cutoff rates |
| Hard decisions of LLRs | `com_core::signal::hard_decisions` | The sign convention of LLRs is fixed in `com-core`; used by codes, demappers and blocks |
| Indoor optical channel (room, reflections, illuminance) and free-space optics (turbulence, pointing errors, fog) | `com_channel::owc` | Media: the waveforms and receivers take a realisation (an impulse response, a gain) |
| PIN-FET receiver noise (shot, thermal) | `com_hardware::optical::PinFetReceiver` | A component of the receiver, next to the photodiodes |
| Gamma variates | `com_core::rng::Rng::gamma` | Random variates are drawn by the generator every block takes |
| Modified Bessel function $K_{\nu}$ | `com_math::{bessel_k, ln_bessel_k}` | A special function without communication meaning |
| BPSK-over-AWGN LLRs of the code tests | `com_validation::bpsk_awgn_llr` | A test link, not a channel model of the library |
| BPSK symbols $1 - 2b$ and their LLRs $2y/\sigma^{2}$ | `com_mapping::{constellation::bpsk, demapper::bpsk_llr}` | A modulation: `com-fec` holds none |
| The log-sums of the label halves of a bit (the bitwise demapper) | `com_mapping::demapper::LabelHalves` | One kernel for the demapper, the bitwise closed forms and the bit-metric rates of `com_metrics` (ADR 0007) |
| Soft output of a code (a posteriori LLRs of the information bits) | `com_fec::block::SoftDecode` | Implemented by the LDPC, convolutional (BCJR), turbo and NR LDPC codes and passed through the punctured codes |
| The bits a rate matching sends (puncturing, circular buffers) | `com_fec::puncture::BitSelection` | `select` on the encoder side, `accumulate_into` (LLRs summed over the copies) on the decoder side |

## Naming conventions

Names follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/naming.html),
plus the domain rules below.

| Element | Rule | Good | Avoid |
| --- | --- | --- | --- |
| Crate | `com-` prefix, kebab-case, singular | `com-channel` | `com-channels`, `comchannel` |
| Module | snake\_case, singular noun, no prefix | `fec::ldpc`, `hardware::amplifier` | `fec::ldpc_codes`, `hw::amp` |
| Trait | Verb for the action the block performs | `Encode`, `Map`, `Propagate`, `Impair`, `Detect` | `Encoder` (clashes with concrete types), `ChannelTrait` |
| Concrete type | Noun, no module stutter | `fading::Rayleigh`, `awgn::Awgn`, `stbc::Ostbc` | `fading::RayleighFading` |
| Configuration | A struct with public fields and a `Default` or a constructor, changed by struct update | `FiberParams { n_spans: 10, ..FiberParams::default() }` | positional constructors with 6 `f64` |
| Parser of a name | `FromStr` when the name alone gives the value; `from_name(name, …)` when it needs other arguments or builds a structure | `"eva".parse::<multipath::Profile>()`, `DetectorKind::from_name("kbest", 8)`, `Crc::from_name("crc24a")` | ad-hoc `match` on strings in the bindings or the specifications |
| Unit | In the type (`Db`, `EbN0`, `EsN0`, `Hertz`, `Seconds`) or in the name | `ebn0_db`, `fc_hz`, `alpha_db_km` | bare `f64` named `snr` |
| Function | snake\_case verb or noun of the quantity; `_db` / `_lin` suffix on numeric helpers | `theory_ber`, `qfunc`, `snr_noise_variance` | `BerBpsk`, `calc_ber` |
| Feature flag | Lowercase domain or capability | `optical`, `spec` | `enable-optical`, `OPTICAL` |
| Error | The shared `com_core::Error` (ADR 0003) | `Error::Config`, `Error::Shape` | `String` errors, `Box<dyn Error>` in public API |

Some types kept the names of the first API and stutter (`ldpc::LdpcCode`,
`fiber::FiberParams`, `pmd::PmdLink`, `cpm::CpmConfig`); new types do not.

Use the notation of the reference literature in parameter names when it is unambiguous
and documented: `beta2`, `gamma`, `n_fft`, `n_cp`, `rolloff`. Never use single letters
in public signatures except for mathematical objects named that way in the docs (`h`
for a channel matrix is acceptable inside a function body, not as a field name). The
Python API shares one vocabulary with the block parameters of the specifications
(`fs`, `fc_hz`, `doppler_hz`, `noise_var`, `esn0_db`, `n_rx`, `n_tx`).

## Core types and traits

`com-core` defines the vocabulary every other crate speaks: typed containers, verb
traits, unit newtypes, a single generator type and the error type. It contains no
algorithm.

### Containers

All containers are two-dimensional, streams × elements. SISO is `n_streams == 1`.

```rust
use ndarray::Array2;
use num_complex::Complex64;

/// Hard bits, one row per stream. Values are 0 or 1.
pub struct Bits(Array2<u8>);

/// Log-likelihood ratios, one row per stream (sign convention below).
pub struct Llr(Array2<f64>);

/// Complex symbols at one sample per symbol.
pub struct Symbols(Array2<Complex64>);

/// Complex baseband waveform with its sample rate.
pub struct Signal {
    data: Array2<Complex64>, // [n_streams, n_samples]
    sample_rate: Hertz,
}
```

`DualPol` holds the two polarisations of an optical field.

The LLR sign convention is fixed library-wide. A positive LLR favours bit 0:

```latex
L(b) = \ln \frac{P(b = 0 \mid y)}{P(b = 1 \mid y)}
```

### Traits

One verb trait per role: `Encode`, `Decode`, `Map`, `Demap`, `Modulate`, `Demodulate`,
`Precode`, `Detect`, `Propagate`, `Impair`. Methods taking `&self` are pure; `&mut self`
signals internal state (memory, adaptation). Every trait has an `_into` method writing
into a caller-owned buffer, which it resizes and fills in place (never replaces), so a
loop that passes the same buffer allocates no output once its shape is stable; scratch
buffers may allocate unless a profile shows them (ADR 0011). All but `Precode` and
`Detect` also have an allocating method that calls it.

```rust
pub trait Encode {
    /// Code rate k/n.
    fn rate(&self) -> f64;
    fn encode_into(&self, info: &Bits, out: &mut Bits) -> Result<()>;
    fn encode(&self, info: &Bits) -> Result<Bits> { /* allocates, calls encode_into */ }
}

pub trait Decode {
    fn decode_into(&mut self, llr: &Llr, out: &mut Bits) -> Result<()>;
    fn decode(&mut self, llr: &Llr) -> Result<Bits> { /* … */ }
}

pub trait Map    { fn map_into(&self, bits: &Bits, out: &mut Symbols) -> Result<()>; /* map */ }
pub trait Demap  { fn demap_into(&self, y: &Symbols, noise_var: f64, out: &mut Llr) -> Result<()>; /* demap */ }
pub trait Detect { fn detect_into(&mut self, y: &Symbols, noise_var: f64, out: &mut Symbols) -> Result<()>; }

/// Propagation through a medium. Randomness is always explicit.
pub trait Propagate {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()>;
}

/// A component non-ideality (amplifier, oscillator, IQ imbalance…).
pub trait Impair {
    fn impair_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()>;
}
```

Implementations: every `BlockCode` of `com-fec` (`Encode`, `Decode`, by the macro
`container_traits!`): `ConvolutionalCode`, `TurboCode`, `PuncturedTurbo`,
`Punctured<C>`, `LdpcCode`, `PolarCode`, `ReedSolomon`, `Bch`, `NrLdpc`, `NrPolar`
(`Dvbs2Fec` of `com-systems` is a `BlockCode` only); `Constellation` (`Map`, `Demap`);
`Ofdm`, `Ufmc`, `Otfs` (`Modulate`, `Demodulate`); `Eigenmode`, `LinearPrecoder`,
`Ostbc` (`Precode`); `detection::Coherent` (`Detect`); `Awgn`, `Noise`, `Rayleigh`,
`Mimo` (`fading`), `Tdl`, `DelayDopplerModel` (`multipath`), `Cdl`, `Fso`
(`owc::atmospheric`), `Ssfm` (`optical::fiber`) (`Propagate`); `Rapp`, `Saleh`,
`MemoryPolynomial`, `PhaseNoise`, `Cfo`, `TxIq`, `RxIq`, `Adc`, `Dac` (`Impair`). The
other blocks expose slice-based functions and methods.

### Randomness

There is no global or thread-local generator anywhere in the library. Every random
block receives `&mut SimRng`, a concrete type alias, which keeps traits object-safe and
calls monomorphic. `SimRng` is xoshiro256\*\* seeded through SplitMix64 (ADR 0002).

```rust
pub type SimRng = com_core::rng::Rng; // xoshiro256**

/// The stream of one trial: independent of every other (seed, experiment, condition, trial).
pub fn stream(base_seed: u64, experiment: &str, condition: u64, trial: u64) -> Rng { /* … */ }

/// Independent, reproducible stream for one batch of frames: stream(master_seed, "", 0, batch).
pub fn batch_rng(master_seed: u64, batch: u64) -> SimRng { /* … */ }
```

The runner of `com-sim` gives trial `t` of condition `c` of the scenario `name` the
stream `stream(seed, name, c, t)`, so results do not depend on the thread count, nor on
the batch size when the stopping rule bounds the trials (the last batch ends at
`max_trials`); an error or precision target is checked after every batch.

### Units

Quantities that are routinely confused get their own type. Conversions are explicit
methods, never `From` impls that hide a formula.

```rust
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Db(pub f64);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct EbN0(pub Db);

impl EbN0 {
    /// Converts to Es/N0 for `bits_per_symbol` coded bits at code rate `rate`.
    pub fn to_es_n0(self, bits_per_symbol: u32, rate: f64) -> EsN0 { /* … */ }
}
```

The conversion it implements, with k bits per symbol and code rate R:

```latex
\left.\frac{E_s}{N_0}\right|_{\mathrm{dB}} = \left.\frac{E_b}{N_0}\right|_{\mathrm{dB}} + 10 \log_{10}(k R)
```

### Configuration

A block is configured by a struct with public fields, built by a constructor or
`Default` and adjusted by struct-update syntax; its fields are checked when the block
is built or used, and an invalid value returns `Error::Config`:

```rust
let fiber = FiberParams { span_km: 100.0, n_spans: 10, ..FiberParams::default() };
let ofdm = Ofdm { n_cp: 16, dft_spread: true, ..Ofdm::new(256, 200)? };
```

A modem that samples a filter once at construction (`Fbmc`, its prototype) keeps its
fields private behind getters, so that the cached taps cannot disagree with them; so do
the codes whose tables derive from their parameters (`Trellis`, `Interleaver`,
`TurboCode`, `Punctured`, `NrLdpc`, `NrPolar`), whose decoder settings are a struct of
public fields (`LdpcDecoder`, `TurboDecoder`, reached by `decoder_mut()` on a wrapping
code).
`CpmConfig::build` is the one configuration validated by a separate `build` step,
since it tabulates the phase response. The enums that will gain variants are
`#[non_exhaustive]` (`com_core::Error`, `equalizer::Algorithm`, `filter::{PulseShape,
LowPass}`, `ConstellationKind`, `CheckRule`, `DetectorKind`, `multipath::Profile`,
`cdl::{Profile, Pattern}`, `fiber::{Amplification, Precision}`, `PhaseOrigin`,
`StepRule`, `PilotPattern`, `FrequencyInterpolation`, `ce_ofdm::{Unwrap, Receiver}`,
`papr::{Partition, Companding}`, `cpm::FrequencyPulse`, `fbmc::Prototype`,
`css::Waveform`, `message_passing::Prior`, `atmospheric::Turbulence`, the DVB-S2
`FrameSize`, `CodeRate` and `Modulation`); the configuration structs are not.

## Documentation standards

Every public item is documented, every formula is written in LaTeX, and every example
compiles and runs as a test. `missing_docs` and `cargo doc -D warnings` enforce the
first; the doctests the last.

### Tooling

- **API reference:** rustdoc, with KaTeX injected through `--html-in-header`
  (`.cargo/config.toml` for local builds). docs.rs builds each crate in isolation, so
  each crate ships its own copy of `docs/katex-header.html` (`just katex` refreshes
  them, CI checks they are identical).
- **Index of the public API:** [`API.md`](API.md), every public item of the crates,
  of the bindings and of the Python package with its signature and the first line of
  its documentation, generated by `tools/api_index.py` (`just api-index`).
- **Python API:** Google-style docstrings written on the PyO3 bindings and in
  `python/comrs`, rendered with Sphinx (PyData theme, `sphinx.ext.napoleon`,
  autosummary) in `docs/python`, with tutorials and a gallery of classical results run
  when the documentation is built (`sphinx-gallery`). Python docstrings are written for
  Python users, not copied from rustdoc. The stub `python/comrs/_comrs.pyi` is
  generated from them (`tools/gen_stubs.py`).

```toml
# .cargo/config.toml (local builds)
[build]
rustdocflags = ["--html-in-header", "docs/katex-header.html"]

# crates/<crate>/Cargo.toml (docs.rs builds)
[package.metadata.docs.rs]
rustdoc-args = ["--html-in-header", "katex-header.html"]
all-features = true
```

### Item template

Sections appear in this order and are omitted only when empty: summary line,
description, `# Model`, `# Errors`, `# Panics`, `# Examples`, `# References`.

````rust
/// Additive white Gaussian noise channel.
///
/// Adds circularly-symmetric complex Gaussian noise to every stream independently.
/// The noise power is set from `E_s/N_0` and the measured average power of each
/// stream (at one sample per symbol), so the input need not be normalised.
///
/// # Model
///
/// $$
/// y_{k} = x_{k} + n_{k}, \qquad n_{k} \sim \mathcal{CN}(0, N_{0})
/// $$
///
/// with $N_{0} = \bar{E}_{s} / (E_{s}/N_{0})$, where $\bar{E}_{s}$ is the average
/// symbol energy of the stream at one sample per symbol.
///
/// # Errors
///
/// [`Propagate::propagate_into`] returns [`Error::Config`] if `x` has no samples.
///
/// # Examples
///
/// ```
/// use com_core::prelude::*;
/// use com_channel::awgn::Awgn;
///
/// let mut rng = batch_rng(42, 0);
/// let mut ch = Awgn::new(EsN0(Db(10.0)));
/// let x = Signal::from_stream(vec![C64::new(1.0, 0.0); 1024], Hertz(1.0))?;
/// let y = ch.propagate(&x, &mut rng)?;
/// assert_eq!(y.n_samples(), 1024);
/// # Ok::<(), com_core::Error>(())
/// ```
///
/// # References
///
/// 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed.,
///    McGraw-Hill, 2008.
pub struct Awgn { /* … */ }
````

A comparison with a publication or a standard cites its authors, venue, year and the
figure, table or equation it reproduces; a standard is cited with its number, version
and section. The documents are listed in
[`references/README.md`](../references/README.md).

### Module documentation

Each module's `//!` header gives an overview, the signal model with its notation, a
table of the types it exports, and references. The crate root has a table of its
public modules (`| [`module`] | content |`), which lists all of them.

### LaTeX in rustdoc

Markdown is parsed before KaTeX runs, so a few rules prevent silent corruption:

- Put display math in `$$` on its own lines; inline math in single `$`.
- Brace every subscript (`E_{b}`, `x_{k}`) so pairs of underscores are never read as
  emphasis.
- Double the backslash before punctuation, which Markdown would otherwise consume:
  write `\\\\` in the source for a LaTeX line break `\\`, and `\\{` for a literal brace
  `\{`. Backslashes before letters (`\frac`, `\mathcal`) need no doubling.
- Check every new formula in the rendered page (`just doc`) before merging.

### Prose

British spelling (identifiers keep their spelling: `normalized`, `Equalizer`).

## Code quality standards

Lints are declared once at the workspace root and inherited by every crate (`[lints]
workspace = true`); the exceptions are listed and justified in ADR 0005. CI runs clippy
with `-D warnings`, so a warning blocks a merge.

```toml
# Cargo.toml (workspace root)
[workspace.package]
version = "0.2.0"              # one version for every crate
edition = "2024"
rust-version = "1.85"          # MSRV, tested in CI
license = "MIT OR Apache-2.0"

[workspace.lints.rust]
missing_docs = "warn"
unsafe_code = "forbid"
unreachable_pub = "warn"

[workspace.lints.clippy]
pedantic = { level = "warn", priority = -1 }
cast_precision_loss = "allow"  # usize -> f64 is pervasive in numeric code
unwrap_used = "deny"
expect_used = "warn"
allow_attributes_without_reason = "warn"
# … the numeric exceptions of ADR 0005 (suboptimal_flops, manual_midpoint, casts, …)
```

### Errors and panics

- One error type, `com_core::Error` (`thiserror`, `#[non_exhaustive]`: `Config`,
  `Shape`, `Numerical`, `Condition`, `Io`), shared by every crate and re-exported as
  `comrs::Error` (ADR 0003). No `String` errors, no `Box<dyn Error>` in public
  signatures. The length checks return `Error::Shape` with the lengths received
  (`com_core::ensure_shape`).
- Invalid user input returns an error; it never panics. Panics are reserved for broken
  internal invariants and are documented under `# Panics`.
- A configuration is checked when the block is built or used, once; hot loops never
  re-validate. The Python bindings map `Config`, `Shape` and `Condition` to `ValueError`,
  `Numerical` to `ArithmeticError`, `Io` to `OSError`.

### Numerics

- `f64` throughout. A lower-precision path is added only with a measured need and a
  check proving it holds (`Ssfm` with `Precision::F32`, compared with `f64` in its
  tests).
- Probabilities and LLR combinations are computed in the log domain (`log_sum_exp`,
  `max_star`, `ln_1p_exp`) to avoid underflow at high SNR.
- Results do not depend on the number of threads. Random streams are derived from the
  trial index, never from the worker id, so a run on 1 core and on 16 cores is
  bit-identical (`crates/comrs/tests/simulation.rs`).
- Kernels that produce published results keep their order of operations: no
  `mul_add`, no `f64::midpoint` (ADR 0005).
- Floating-point comparisons in tests use explicit tolerances written in the assertion,
  with the reason (sampling error, precision of a published value) when it is not
  obvious.

### Performance

- No allocation inside the Monte Carlo loop where it matters: the verb traits have
  `_into` methods that reuse their output, callers reuse buffers, and blocks whose
  scratch shows in a profile keep a workspace (ADR 0011).
- Parallelism lives in `com-sim` only (rayon over the trials of a batch). Blocks are
  single-threaded, which avoids nested parallelism.
- FFT plans and code matrices are built once at construction and cached in the block.
- Profile before optimising (`samply`, `cargo flamegraph`, or `profile: true` in a
  specification for the time of every block). An optimisation states its gain,
  measured with the criterion benchmarks (`benches/blocks.rs`) or
  `benchmarks/bench_opticommpy.py`.

### API stability

- Semantic versioning; one version for every crate; the changelog lists the changes of
  every release, with a migration note for a breaking change.
- The enums that will grow are `#[non_exhaustive]` (see Configuration).
- `unsafe` is forbidden workspace-wide. The Python bindings do not inherit the
  workspace lints (PyO3's macros expand to `unsafe` code, ADR 0005); a future exception
  in the library requires its own module, a `// SAFETY:` comment on every block, and
  Miri in CI.

### Automation

`just check` runs the checks of CI, in this order (run it before declaring a task
done):

1. `cargo fmt --all --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test` (the default members: every crate but the Python extension, doctests
   and the quick tests of `com-validation` included)
4. `cargo run --release -p com-spec --bin comrs -- verify` of the validation
   specifications listed in `ci_specs` of the `justfile` (`VALIDATION_SPECS` of
   `ci.yml`, the same list): `proakis_awgn`, `information_rates`, `hager2021_fig5`,
   `msk_bpsk`, `thompson2008_eq15`
5. `cargo doc --workspace --no-deps --exclude com-py` with
   `RUSTDOCFLAGS="--html-in-header $PWD/docs/katex-header.html -D warnings"`
6. every crate ships the same `katex-header.html` as `docs/katex-header.html` (`cmp`)
7. `ruff check` and `ruff format --check` of `python tests docs/python tools
   benchmarks/libraries`
8. `mypy` (`[tool.mypy]` of `pyproject.toml`): the types of the package and of the
   generated stub, the tests checked as callers of the stub only (a method stubbed
   `-> None` whose value is used fails)
9. `python3 tools/api_index.py --check` (`docs/API.md` is up to date) and `--duplicates
   --check` (no public name defined in several modules that `docs/api-duplicates.txt`
   does not accept: factor it, or add it with the reason it is not a copy)
10. `python tools/literature_table.py --check` and `python tools/maturity_table.py
    --check` (`docs/LITERATURE.md`, `docs/MATURITY.md` and the maturity tables are up to
    date, and every test they cite exists)
11. `python tools/cards.py --check` (every validation card agrees with the repository)
12. `python3 tools/benchmarks_page.py --check` (`docs/python/benchmarks.rst` is up to
    date with `benchmarks/libraries/results/`) and `python tools/highlights.py --check`
    (the highlights of `README.md` and `docs/python/index.rst` and the page
    `docs/python/why.rst`, computed from the manifests, the benchmark and the
    validation files)
13. `python tools/gen_stubs.py --check` and `python tools/gen_blocks.py --check` (the
    generated stub and `comrs.blocks` are up to date; they need the extension built)

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests, with read-only
permissions; it runs the quick tests only:

| Job | Runs on | Steps |
| --- | --- | --- |
| `rust` | Linux | `cargo fmt --all --check`; clippy with `-D warnings`; `cargo doc` with the KaTeX header and `-D warnings`; the KaTeX headers are identical |
| `test` | Linux, macOS, Windows | `cargo test`; `comrs verify $VALIDATION_SPECS` |
| `msrv` | Linux, Rust 1.85 | `cargo build` |
| `deny` | Linux | `cargo deny check` (licences, advisories, `deny.toml`) |
| `python` | Linux, macOS, Windows × Python 3.11, 3.12, 3.13 | `maturin develop --release` with the dependency groups `test` and `lint`; on Linux with 3.12 only: `ruff check`, `ruff format --check`, `mypy`, `api_index.py --check` and `--duplicates --check`, `literature_table.py --check`, `maturity_table.py --check`, `cards.py --check`, `benchmarks_page.py --check`, `highlights.py --check`, `gen_stubs.py --check`, `gen_blocks.py --check`; then `python -m pytest` |
| `docs-python` | Linux | `sphinx-build -W --keep-going` of `docs/python`; the site is uploaded on pushes to `main` |
| `deploy-docs` | Linux, pushes to `main` | publishes the site to GitHub Pages |

`.github/workflows/validation.yml` runs the long validations and the benchmarks, and
only when started by hand (`workflow_dispatch`): `cargo test --release -p
com-validation -- --ignored`, with an optional filter on the test names, and, when its
switch is set, `cargo bench -p comrs`. The ignored tests never run automatically, and
the benchmarks are run, not compared: no regression threshold is enforced.

`.github/workflows/release.yml` runs on a tag `vX.Y.Z` and on demand:

1. `version`: the tag equals `workspace.package.version`, `CITATION.cff` gives the
   same version, and the section of `CHANGELOG.md` whose heading contains `[X.Y.Z]`
   becomes the release notes;
2. `crates` (tag only): `cargo publish --workspace`, in the order of the dependencies;
3. `wheels`: one abi3 wheel (CPython 3.11 and later) per platform: manylinux and
   musllinux on x86_64 and aarch64, macOS on x86_64 and aarch64, Windows x64; `sdist`;
4. `install`: the wheels install from the built files alone on Linux, macOS and
   Windows (Python 3.11 and 3.13) and run a link;
5. `testpypi` (on demand only, the dry run to do before the first tag): the wheels and
   the sdist go to TestPyPI, and nothing else is published;
6. `pypi` (tag only, trusted publishing) and `github-release` (the notes and the
   wheels).

Generated files are committed and checked:

| File | Generator | Check |
| --- | --- | --- |
| `docs/API.md` | `tools/api_index.py` (`just api-index`; `--duplicates` lists the names defined in several modules) | `--check` |
| `python/comrs/_comrs.pyi` | `tools/gen_stubs.py`, from the built extension | `--check` |
| `python/comrs/blocks.py` | `tools/gen_blocks.py`, from the block registry (`just gen-python` runs both) | `--check` |
| `docs/LITERATURE.md` | `tools/literature_table.py`, from `validation/literature.yaml` (`just literature`) | `--check` |
| `docs/MATURITY.md`, the maturity tables of `README.md` and `docs/python/maturity.rst` | `tools/maturity_table.py`, from `validation/maturity.yaml` (`just maturity`) | `--check` |
| `docs/python/benchmarks.rst` | `tools/benchmarks_page.py`, from `benchmarks/libraries/results/` (`just bench-libraries`) | `--check` |
| the highlights of `README.md` and `docs/python/index.rst`, `docs/python/why.rst` | `tools/highlights.py`, from the manifests, `benchmarks/libraries/results/` (with `footprint.json`) and the validation files | `--check` |
| `crates/*/katex-header.html` | `just katex` | `cmp` |
| `tests/oracle/*` | `tests/oracle.py` with `COMRS_RECORD_ORACLE=1` (dependency group `oracle`) | the tests read them |
| `validation/aff3ct/*.json` | `tools/record_aff3ct.py`, with AFF3CT installed | the tests read them |
| `benchmarks/results.md` | `benchmarks/bench_opticommpy.py` (`just bench`) | none |

## Testing and validation

A block is merged when it is correct by construction (unit tests), correct by physics
(invariants) and correct against an independent reference. The evidence for every
module is summarised in [`MATURITY.md`](MATURITY.md).

| Level | Where | Checks | Runs |
| --- | --- | --- | --- |
| Unit | `#[cfg(test)] mod tests` of every module | Edge cases, error paths, round trips and identities over seeded random inputs, known small vectors | `cargo test`, every push |
| Literature | `#[cfg(test)] mod literature`, one test named after each reference | The tables, small examples and closed forms of a document of `references/`, verbatim | `cargo test`, every push |
| Doctest | rustdoc | Every documented example compiles and runs | `cargo test`, every push |
| Integration | `crates/comrs/tests/` (`links`, `simulation`, `specs`), `crates/com-spec/tests/extension.rs` | End-to-end links against closed forms, the runner (thread independence, stopping rules), every specification runs one trial per condition | `cargo test`, every push |
| Specifications | `specs/validation/*.yaml` and `specs/*.yaml` with an `expect` section | Simulated results against closed forms, tables of standards and published curves | `comrs verify`: five in CI; the others with `just validate`, `validation/tests/specs.rs` or `tests/test_performance.py` |
| Statistical | `validation/tests/*.rs` (crate `com-validation`) | Monte Carlo curves against closed forms, published figures and tables, and recorded AFF3CT runs | quick tests with `cargo test`, every push; `#[ignore]` ones on demand (`cargo test --release -p com-validation -- --ignored`, or `validation.yml` started by hand) |
| Oracle | `tests/test_oracle.py`, `tests/test_gnpy_oracle.py` | The blocks against the recorded outputs of OptiCommPy 0.11 and GNPy 3.0 on the same inputs | `python -m pytest`, every push |
| Classical results | `tests/test_performance.py` and the gallery of `docs/python` | Textbook results end to end from Python | `python -m pytest`; the gallery when the documentation is built |
| Benchmark | `benches/blocks.rs` (criterion), `benchmarks/bench_opticommpy.py`, `benchmarks/libraries/` | Throughput; speed and agreement against OptiCommPy; results and speed against komm, OptiCommPy and Sionna | on demand (`validation.yml` with its benchmarks switch, `just bench-rust`, `just bench`, `just bench-libraries`); not compared with a threshold |

There are no property-based tests (`proptest`): the round trips and identities are
unit tests on inputs drawn from seeded generators.

### Properties every block family must satisfy

- **FEC:** noiseless encode → decode returns the input; the codeword satisfies the
  parity checks (H c = 0).
- **Mapping:** hard demapping of noiseless symbols returns the bits; constellations
  have unit average energy once normalised.
- **Waveforms:** demodulate(modulate(x)) = x for an ideal channel; OFDM keeps unit
  power.
- **Channels:** a noiseless, distortionless configuration is the identity; lossless
  SSFM conserves energy; a fundamental soliton propagates without changing shape.
- **Hardware:** zero-impairment configurations are the identity.

### Classes of evidence

A result is checked against one of four kinds of reference, from the strongest:

1. **The recorded output of a reference implementation** on the same inputs or the
   same configuration: OptiCommPy and GNPy (`tests/oracle/`), AFF3CT
   (`validation/aff3ct/`). The fixtures record the tool, its version and the date.
2. **A table of a standard, reproduced verbatim**, or the result it specifies: the CDL
   and TDL profiles, the CRC polynomials, the puncturing patterns, Table 13 of
   EN 302 307-1.
3. **A published figure or table**: the points read on the figure, or the values of
   the table, with the tolerance the reading allows.
4. **A closed form**: the error rate, capacity or distribution that theory gives for
   the configuration.

`MATURITY.md` grades every module by the strongest evidence it has: **A** for 1–3,
**B** for 4 only, **C** for self-consistency only (round trips, invariants). It is
generated, with the maturity tables of the README and of the Python documentation,
from `validation/maturity.yaml` (`just maturity`); `LITERATURE.md` is generated from
`validation/literature.yaml` (`just literature`). Both group their rows by the themes
of the documentation (foundations, coding and shaping, waveforms, MIMO, wireless
channels, optical, hardware impairments, receiver DSP, metrics and analysis, systems
and standards, simulation and specifications), as the Python API reference and the
gallery do.

### Specifications with expectations

An `expect` entry of a specification gives the expected result as a value, a curve
read from a publication (`curve: {along, x, y}`) or a closed form (`theory:`), with
its `source` and a tolerance (`atol`, `rtol`, or `interval`: the theory must lie in
the confidence interval of the estimate); `reference` compares a metric with another
metric of the same condition within a `factor`. `comrs verify` runs the campaign,
prints every check and fails if one fails; a specification with a `search` section
checks its operating points.

### The oracle fixtures

`tests/test_oracle.py` compares every deterministic block shared with OptiCommPy, and
`tests/test_gnpy_oracle.py` the Raman amplification and the generalised GN model with
GNPy. The outputs of the tools are recorded once into `tests/oracle/` (`<name>.json`
with the tool, its version, the versions of Python and NumPy and the date; the arrays of
more than 64 values in `<name>.npz`) by `tests/oracle.py`; the tests read them and skip
when a file is missing, so neither tool is needed. Re-recording
(`pip install --group oracle`, `COMRS_RECORD_ORACLE=1 python -m pytest
tests/test_oracle.py tests/test_gnpy_oracle.py`) is a deliberate act, named in the
commit message with the version of the tool; re-recording an unchanged oracle
produces no diff.

### The AFF3CT fixtures

`tools/record_aff3ct.py` runs AFF3CT (Cassagne et al., *SoftwareX* 10, 2019) on 17
configurations over BPSK and AWGN (Reed–Solomon, BCH, the (133, 171) convolutional
code, LDPC codes with flooding and layered decoders and the DVB-S2 short code, polar
SC and CA-SCL, LTE turbo) and records, for every $E_b/N_0$, its frames, bit and frame
errors with the command line in `validation/aff3ct/<name>.json`. The LDPC matrices and
polar frozen sets given to AFF3CT are written by com-rs
(`COMRS_RECORD_AFF3CT=1 cargo test -p com-validation --test aff3ct`).
`validation/tests/aff3ct.rs` simulates the same configurations: a point passes when
the 99 % Wilson intervals of the two frame error rates overlap and the bit error rates
are within the burst-aware bound of the test. The quick configurations run with
`cargo test` (with at most 10 000 frames and 100 frame errors per point;
`COMRS_VALIDATION_FULL=1` lifts the limit); the long ones are `#[ignore]`d.

### Statistical validation

`validation/tests/` holds the Monte Carlo checks, one file per reference document
(`raviteja2019.rs`, `tal2015.rs`, `loyka2001.rs`, … named after its first author and
year, listed with what they reproduce in `validation/literature.yaml` and
[`LITERATURE.md`](LITERATURE.md)) and the files below. The helpers they share live in
`validation/src/lib.rs`: the acceptance rule (`within_three_sigma`), the BPSK link of
the code tests (`bpsk_awgn_llr`), the crossing of an error rate by log-linear
interpolation (`crossing`, `log_interp`) and the Doppler shift of a speed
(`max_doppler_hz`). A comparison written as data is a validation card of
`validation/cards/` (`validation/cards/README.md`), checked by `tools/cards.py` and
rerun by `just card <id>`.

- `closed_forms.rs`: BPSK over AWGN and flat Rayleigh fading; each point runs until
  100 errors and passes within three standard errors of the closed form
  (`com_validation::within_three_sigma`);
- `mimo.rs`: zero forcing and Alamouti against maximal-ratio combining, and the
  diversity orders of the detectors against Table X of Yang & Hanzo (2015);
- `shaping.rs`: the CCDM, the information rates of shaped ASK and QAM, and PAS on the
  DVB-S2 codes, against Schulte & Böcherer (2016), Böcherer, Steiner & Schulte (2015)
  and Liang & Kahn (2025);
- `egn.rs`: the EGN model against Fig. 1(a) of Carena et al. (2014) and the
  first-order perturbation;
- `aff3ct.rs`: the channel codes against AFF3CT;
- `specs.rs`: the quick specifications of `specs/validation/`.

## Versioning and workflow

All crates share one version, set in `[workspace.package]`, and are released together
with a Git tag `vX.Y.Z` (`release.yml`). Papers depend on a tag, never on a branch.

### Records

- **Changelog:** `CHANGELOG.md` in [Keep a Changelog](https://keepachangelog.com/)
  format, one section per release (`## [X.Y.Z] — date`), which the release workflow
  publishes as the release notes.
- **Architecture decision records:** `docs/adr/NNNN-title.md`, one page each (status,
  context, decision, consequences). An ADR is required for a new crate, a new
  dependency between crates of one layer, a change to a core trait, or an exception to
  this guide; it is never deleted, a later one supersedes it.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) scoped by
  crate or area, for example `feat(fec): layered min-sum LDPC decoder`. `main` is
  always green.

### Adding a block

1. Locate it with the placement rules (and [`API.md`](API.md)); record an ADR if none
   applies.
2. Implement it: a configuration struct checked when used, the role trait with its
   `_into` method where one applies, `&mut SimRng` for randomness.
3. Document it with the item template: model in LaTeX, errors, example, references.
4. Expose it to the specifications with a wrapper in `com_spec::blocks`: its
   parameters declared once with `block_params!` (type, default, unit, accepted values,
   bounds, meaning) and its ports with `block_info!` (the facts of the frame it needs
   and gives). The parser, `comrs blocks`, the Python classes of `comrs.blocks`
   (`tools/gen_blocks.py`) and the JSON Schema of `comrs schema` all come from this
   declaration; debug builds check the ports while the block runs. A block of another
   crate joins the registry with `com_spec::blocks::register` (ADR 0006).
5. Add unit and invariant tests.
6. Check it against its reference: a `mod literature` test or a specification of
   `specs/validation/` with `expect`; its entry in `validation/maturity.yaml` and, for a
   document, in `validation/literature.yaml` (`just maturity`, `just literature`).
7. Add the Python binding if users need it (`bindings/com-py`), then `just gen-python`
   and `just api-index`, and list it in its theme of `docs/python/api/index.rst` (a test
   fails otherwise).
8. Update the changelog.

### Working with a coding agent

- This guide is `docs/ARCHITECTURE.md`. The root `CLAUDE.md` (read by every agent,
  `AGENTS.md` points to it) stays short: it points to the guide and to `docs/API.md`,
  restates the dependency rule and the placement rule, and lists what `just check`
  runs.
- A personal skill teaches the public API to sessions working in other repositories;
  it points to `examples/` as the canonical usage and is updated with each release.
- Paper repositories read the library and need explicit approval to edit it. Library
  changes made from a paper session must be generic, tested and committed in the
  library repository.
