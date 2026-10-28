# Changelog

All notable changes to com-rs are documented here, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crates share one version
and follow [Semantic Versioning](https://semver.org/).

## [0.2.0] — unreleased

The first public release: the Rust crates on crates.io, the Python package `comrs` on
PyPI and the command line `comrs`.

### Crates and layers

- Fourteen crates in five layers whose dependencies point downward only
  (`docs/ARCHITECTURE.md`, decisions in `docs/adr/`): `com-core` and `com-math` (0),
  `com-dsp` (1), `com-fec`, `com-mapping`, `com-waveform`, `com-mimo`, `com-channel`,
  `com-hardware`, `com-metrics` (2), `com-sim` (3), `com-spec`, `com-systems` and the
  facade `comrs` (4), with the features `optical` and `spec`.
- `com-core`: the multi-stream containers (`Bits`, `Llr`, `Symbols`, `Signal`,
  `DualPol`), the verb traits (`Encode`, `Decode`, `Map`, `Demap`, `Modulate`,
  `Demodulate`, `Precode`, `Detect`, `Propagate`, `Impair`), the unit newtypes, one
  error type for every crate, and the generator `SimRng` (xoshiro256\*\*) with one
  stream per `(seed, experiment, condition, trial)`.
- `com-math`: special functions, Gauss–Hermite quadrature, FFT plans, complex linear
  algebra (QR, Cholesky, SVD), nearest-point search, the split-step solvers of the NLSE
  and the Manakov equation, GF(2) and GF(2^m), exact integers, the Gray code,
  bisection and golden-section search.
- MSRV 1.85, edition 2024, licence MIT OR Apache-2.0; no `unsafe` in the library.

### Blocks

- Mapping: QAM, PSK, PAM, OOK and custom constellations with Gray labels, decisions,
  exact and max-log LLRs; sources (fair bits, PRBS, uniform, Maxwell–Boltzmann and
  Gaussian symbols); differential M-PSK; the closed forms of the error rates and of
  the information rates (MI, GMI, with priors).
- Coding: CRCs and the Gold scrambler of 5G NR; block, random and QPP interleavers;
  convolutional codes (Viterbi, BCJR) and their puncturing (IEEE 802.11, DVB-S); LTE
  turbo codes; LDPC codes (PEG, repeat-accumulate, alist files; sum-product,
  normalised and offset min-sum, flooding or layered); polar codes (SC, CRC-aided
  SCL); Reed–Solomon codes over any field with erasures; shortened binary BCH codes;
  constant-composition distribution matching; BI-AWGN capacity and the normal
  approximation.
- Waveforms: OFDM and DFT-spread OFDM with resource grids, pilots, LS and LMMSE
  channel estimation, Schmidl–Cox synchronisation and MIMO-OFDM detection; the
  distribution of the PAPR and its reduction (clipping and filtering, SLM, PTS, tone
  reservation, ACE, companding); constant-envelope OFDM (Thompson et al. 2008);
  continuous-phase modulation (CPFSK, MSK, GMSK, raised-cosine pulses) with sequence,
  quadrature and differential detection.
- MIMO: ZF, MMSE, SIC, ML, sphere and K-best detection with soft outputs; message
  passing for large systems, LAMA (Jeon et al. 2018) and OAMP (Ma & Ping 2017), with
  their state evolution and the recovery thresholds of LAMA (detectors `amp` and
  `oamp`); the receiver of a link, orthogonal space-time block codes, eigenmode and zero-forcing precoding,
  capacity.
- Channels: AWGN; flat Rayleigh and Rician fading, SISO and MIMO, with Kronecker
  correlation; tapped delay lines (EPA, EVA, ETU, ITU-R M.1225) with Doppler; the CDL
  profiles of 3GPP TR 38.901 between antenna arrays; optical fibre (linear, split-step
  Fourier in double or single precision, Manakov with PMD), EDFA, distributed Raman
  amplification and inter-channel stimulated Raman scattering, the GN and EGN models
  (their integration grid follows the width of the band, and the phased-array factor
  of the coherent accumulation is integrated exactly over every cell).
- Blind estimation: fourth-order cumulants, modulation classification by cumulants
  (Swami & Sadler 2000) and by likelihood (Wei & Mendel 2000), the M2M4 SNR (Pauluzzi &
  Beaulieu 2000), the symbol rate by cyclostationarity (Gardner 1991); the `classify`
  block.
- Hardware: power amplifiers (Rapp, Saleh, memory polynomial), Wiener phase noise,
  frequency offset, IQ imbalance; DAC and ADC (uniform quantisation with clipping,
  effective number of bits, clock jitter, bandwidth, zero-order hold) with the closed
  forms of their noise; lasers, phase, Mach–Zehnder and IQ modulators,
  photodiodes with shot and thermal noise, the 90° hybrid, the coherent receiver, the
  WDM transmitter.
- Receiver DSP: pulse shaping and FIR filtering, resampling, chromatic-dispersion
  compensation (exact, frequency sampling, Savory's FIR, the least-squares FIR of Eghbali
  et al. 2014 over the whole band or the band of the signal), digital back-propagation (one and two polarisations), the n×n
  adaptive equaliser (CMA, RDE, NLMS, DD-LMS, Kikuchi's initialisation), carrier
  recovery (fourth-power FOE, Viterbi–Viterbi, blind phase search, DD-PLL), Gardner
  clock recovery, frame synchronisation, digital predistortion.
- Metrics: error counts and rates, SNR, EVM, MI and GMI estimators, the rates of a
  receiver's posterior, PAPR and CCDF, Welch PSD, ACLR.
- Systems: the DVB-S2 physical layer (ETSI EN 302 307-1) from the BBFRAME to the
  scrambled PLFRAME, with its BCH and LDPC codes and the 28 MODCODs; probabilistic
  amplitude shaping (Böcherer, Steiner & Schulte 2015) with the bit mapper of the
  paper.

### Simulation

- `com-sim`: scenarios run on every condition of a grid by a parallel runner whose
  results do not depend on the thread count; stopping rules on errors, precision,
  trials and time; Wilson and normal intervals; callbacks, CSV and JSON reports with
  the version, the commit and the configuration.

### Specifications and command line

- `com-spec`: campaigns written as YAML, JSON or TOML files (grid, link of blocks,
  metrics, stopping rule, outputs), 54 blocks with typed parameters and ports,
  expectations checked against closed forms and published curves, the search of
  operating points, the time of every block (`profile`), recordings (SigMF, raw IQ,
  NPY) as sources and as dumps, and a JSON Schema of the blocks.
- Blocks of other crates join the registry with `com_spec::blocks::register` (ADR 0006).
- The command `comrs` (`cargo install com-spec`): `run`, `check`, `verify`,
  `conditions`, `blocks`, `schema`.
- `specs/`: the example campaigns, and `specs/validation/` checked against the
  literature.

### Validation

- `literature` unit tests check the tables, examples and closed forms of the documents
  of `references/` (ITU-R M.1225, 3GPP TS 38.212 and TR 38.901, IEEE 802.11-2020, ETSI
  EN 300 421 and EN 300 744, CCSDS 131.0-B, and the papers they cite).
- `specs/validation/`: Proakis & Salehi over AWGN, information rates by quadrature,
  Häger & Pfister 2021 Fig. 5, the required SNR, MSK against BPSK, Thompson et al.
  2008 eq. (15), and Table 13 of EN 302 307-1 for the 28 DVB-S2 MODCODs.
- `com-validation`: the channel codes against 17 recorded AFF3CT configurations; the
  diversity orders of the MIMO detectors against Yang & Hanzo 2015, Table X; the CCDM,
  the information rates of shaping and PAS on the DVB-S2 codes against Schulte &
  Böcherer 2016, Böcherer, Steiner & Schulte 2015 and Liang & Kahn 2025; the EGN model
  against Carena et al. 2014, Fig. 1(a).
- The Python tests compare the blocks with the recorded outputs of OptiCommPy 0.11 and
  GNPy 3.0 (`tests/oracle/`) and reproduce the classical results of the textbooks.
- `docs/MATURITY.md` grades the evidence of every module.

### Python package

- `comrs`: NumPy bindings of the blocks with explicit generators, one vocabulary for
  the arguments (`fs`, `fc_hz`, `noise_var`, `esn0_db`, …), the closed forms
  `theory_*`, the codes as classes that encode and decode whole blocks, typed with a
  generated stub.
- `comrs.Chain`: links of native and Python blocks, traced block by block, run as
  campaigns on the Rust runner, converted to and from specifications; the generated
  classes of `comrs.blocks`; `comrs.sim` for Python trials on the same runner;
  `Results`, `compare`, probes, profiles and `comrs.plot`.
- abi3 wheels for CPython 3.11 and later on Linux (manylinux, musllinux), macOS and
  Windows.

### Documentation

- The Python documentation (Sphinx): getting started, six tutorials, a user guide, a
  gallery of classical results with a section that reproduces named figures of the
  literature, the API reference and the maturity table.
- The Rust API on docs.rs, every formula in LaTeX; `docs/API.md`, the generated index
  of every public item; `docs/ARCHITECTURE.md` and the ADRs; `references/README.md`,
  the documents the code is checked against.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`,
  `THIRD_PARTY.md`.
- CI on Linux, macOS and Windows (`just check` runs its checks); a release workflow
  with a TestPyPI dry run; the long validations and the benchmarks in a workflow
  started by hand.
