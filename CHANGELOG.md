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
  turbo codes with the 188 QPP interleavers of TS 36.212; the LDPC code of 5G NR (both base
  graphs, lifting, filler bits, the circular buffer and the bit interleaver) and its
  polar code (the polar sequence, puncturing, shortening, repetition, parity-check bits,
  the input and coded-bit interleavers), from the tables of TS 38.212; LDPC codes (PEG, repeat-accumulate, alist files; sum-product,
  normalised and offset min-sum, flooding or layered); polar codes (SC, CRC-aided
  SCL); Reed–Solomon codes over any field with erasures; shortened binary BCH codes;
  constant-composition distribution matching; BI-AWGN capacity and the normal
  approximation.
- Waveforms: OFDM and DFT-spread OFDM with resource grids, pilots, LS and LMMSE
  channel estimation, Schmidl–Cox synchronisation and MIMO-OFDM detection; the
  distribution of the PAPR and its reduction (clipping and filtering, SLM, PTS, tone
  reservation, ACE, companding); constant-envelope OFDM (Thompson et al. 2008); 5G NR
  (and LTE) numerologies and slot modulation, the synchronisation signal block (PSS,
  SSS, PBCH DMRS), the cell search and the DMRS of the PDSCH over a slot, checked
  against py3gpp;
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
  metrics, stopping rule, outputs), 57 blocks with typed parameters and ports,
  expectations checked against closed forms and published curves, the search of
  operating points, the time of every block (`profile`), recordings (SigMF, raw IQ,
  NPY) as sources and as dumps, and a JSON Schema of the blocks. SigMF 1.x: every
  sample format (little and big endian; unsigned types centred by 2^(b-1), as
  sigmf-python), several channels, capture headers, trailing bytes, non-conforming
  datasets, `.sigmf` archives, centre frequencies and annotations; checked against
  sigmf-python.
- Blocks of other crates join the registry with `com_spec::blocks::register` (ADR 0006).
- Link theory: the bit error rates of equalisation with an imperfect channel estimate
  and of DPSK over Jakes' fading (Chang & Su 2002); the output series and in-band
  distortion of a nonlinear amplifier driven by OFDM (Banelli & Cacopardi 2000); the
  Bessel function J0; the exact Gray BER of PAM and rectangular QAM (Cho & Yoon 2002)
  and the density of the phase of a carrier in noise.
- CE-OFDM receivers and theory: the basic linear and enhanced receivers of Ahmed &
  Zeidler (2015) and the near-optimum GAMP receiver of López Morales et al. (2022); the
  spectrum and the powers of the intermodulation products (Guerreiro et al. 2015), the
  99 % bandwidth (Pacheco & Hatzinakos 2006), the optimum asymptotic gain (Guerreiro et
  al. 2016), the Bussgang analysis of the linear receivers per subcarrier and the effect
  of a cycle slip (Ahmed et al. 2012); the block `ce_ofdm_rx` takes a `receiver`.
  Reproduced against Figs. 1 of Guerreiro et al. 2016 and Pacheco & Hatzinakos 2006,
  Fig. 3 of López Morales et al. 2022 and Fig. 6 of Ahmed & Zeidler 2015
  (`validation/tests/ce_ofdm.rs`); a misprint in eq. (8) of Ahmed & Zeidler 2015.
- Chirp spread spectrum, the modulation of LoRa (`com_waveform::css`, Python `Css`):
  the chirps of Chiani & Elzanaty (2019) at any oversampling, with the waveform of
  Vangelista (2017) and FSK for comparison, the noncoherent DFT receiver, a same-SF
  interferer; the exact SER and its approximations, the cross-correlation, the spectrum
  in Fresnel functions and its lines, the interference pattern in closed form and the
  SER and FER under interference (Afisiadis et al. 2020). Reproduces Table I of Chiani
  & Elzanaty, Figs. 2 and 7 of Afisiadis et al. and Fig. 1 of Vangelista; a sign
  misprint in eq. (58) of Afisiadis et al. Adds `com_math::bessel_i0e`.
- Ergodic and outage capacity of Rayleigh MIMO channels in closed form
  (`com_mimo::capacity::{ergodic_rayleigh, ergodic_rayleigh_limit, outage_rayleigh,
  outage_capacity_rayleigh, wishart_eigenvalue_density}`, Python `ergodic_capacity`,
  `outage_probability`, `outage_capacity`), after Telatar (1999): Theorem 2, eqs. (13)
  and (16)–(18), Figs. 1–4. Adds `com_math::{erf, ln_gamma, gamma_p, expint_e1}`.
- Capacity and cutoff rate of CM and BICM over AWGN and Rayleigh fading
  (`com_mapping::bicm`, Python `cm_capacity`, `bicm_capacity`, `cm_cutoff_rate`,
  `bicm_cutoff_rate`) and set-partitioning labels (`Constellation::set_partitioning`),
  after Caire, Taricco & Biglieri (1998): Figs. 4–7; the "more than 3 dB" of Sec. III-C
  is 2.55 dB. The kernels of `theory::mi` and `theory::gmi` are shared with it.
- Pilot-based gain correction of OFDM against a cochannel interferer
  (`com_waveform::ofdm::correction`, Python `gain_limited_correction`,
  `sdr_gain_limited`, `sdr_optimum_correction`), after Cimini (1985): eqs. (20), (21),
  (25), Figs. 3–7 by simulation of his system; eq. (19) lacks a $-1$, the coherence
  bandwidth of eq. (23) is $1/(4\Delta)$, and the "6–7 dB" of the text is 4.5 dB on
  Fig. 5.
- The comb interpolations of Coleri et al. (2002): second order (eq. (19), with
  $\alpha = l/L$), not-a-knot cubic spline, the band-limited interpolator of MATLAB
  `interp` and the time-domain interpolation (`com_dsp::estimation::{
  quadratic_interpolation, spline_interpolation, lowpass_interpolation}`,
  `FrequencyInterpolation::{SecondOrder, Spline, LowPass, TimeDomain}`, Python
  `ofdm_estimate(method=...)`); their ranking of Sec. VI-B holds only when the comb
  oversamples the channel.
- Filtered multicarrier: FBMC/OQAM with the PHYDYAS prototype, polyphase modulator and
  demodulator, OQAM staggering and the intrinsic interference (`com_waveform::fbmc`,
  Python `Fbmc`, `oqam_stagger`, `oqam_destagger`), and UFMC with Dolph–Chebyshev
  sub-band filters (`com_waveform::ufmc`, `com_dsp::window::chebyshev`, Python `Ufmc`,
  `chebyshev_window`), after Farhang-Boroujeny (2011) and Schaich & Wild (2014).
  Bussgang's complex gain and distortion power of any memoryless amplifier
  (`com_hardware::gaussian_input::bussgang`, Python `amplifier_bussgang`). Reproduces
  Tables 1–3 and Figs. 7, 10, 11 of Bouhadda et al. (2014) (the curves of their Fig. 7
  are at an input back-off of 8 dB, not 4 dB) and Figs. 2 and 4 of Nissel & Rupp
  (2017) for OFDM and CP-OFDM; their eqs. (16)–(17) are the Rayleigh closed forms
  already in `com_mapping`.
- OTFS with rectangular pulses and one prefix per frame (`com_waveform::otfs`: ISFFT
  and Heisenberg transform, the sparse effective channel of Theorem 1 of Raviteja et al.
  2019 and the general one of their eq. (13)), and the delay–Doppler channel of
  discrete paths with one Doppler shift each (`com_channel::multipath::DelayDoppler`,
  realised as `Taps`); detection with the MMSE detector of `com-mimo`. Python `Otfs`,
  `delay_doppler_paths`.
- Optical wireless channels (`com_channel::owc`): the indoor channel of a rectangular
  room with Lambertian emitters and reflections of every order, computed recursively on
  the elements of the surfaces (Barry et al. 1993), with the illuminance of LED lights,
  the −3 dB bandwidth and the split of on–off keying into signal and interference
  (`owc::indoor`); free-space optics with log-normal, gamma–gamma and negative
  exponential turbulence, pointing errors and the attenuation of fog (`owc::atmospheric`,
  `Fso` implements `Propagate`); the PIN-FET receiver of Komine & Nakagawa (2004)
  (`com_hardware::optical::PinFetReceiver`), gamma variates (`SimRng::gamma`) and the
  modified Bessel function $K_{\nu}$ (`com_math::{bessel_k, ln_bessel_k}`). Reproduces
  the powers of the four configurations of Barry et al. (1993) and the bandwidths of
  three of them, Figs. 3, 4, 6, 8 and 9 of Komine & Nakagawa (2004), and Figs. 3 and 6–8
  of Al-Habash, Andrews & Phillips (2001), whose exponent 7/6 of eqs. (19) and (26) is
  5/6. Python `OwcRoom`, `OwcIllumination`, `OwcImpulseResponse`, `PinFetReceiver`,
  `Fso`, `GammaGamma`, `PointingError` and helpers.
- Validation cards (`validation/cards/`): one file per algorithm with its inputs, the
  expected outputs and their tolerance, and the exact source of the expectation, run by
  `tests/test_cards.py` and checked by a JSON Schema.
- DRY pass: `com_math::{gaussian_nodes, log_sum_exp_iter, normal_pdf, simpson}`,
  `com_math::fft::filter`, `com_dsp::signal::map_envelope` and
  `com_dsp::compensation::dpd::polynomial_term` replace copies across the crates.
- Comparisons with other libraries: the reference vectors of Sionna (CRC, convolutional
  and turbo codes, bit for bit), py3gpp (5G NR), sigmf-python, next to OptiCommPy,
  GNPy and AFF3CT; a documentation page gathers them.
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

### Consistency of the API

A review of the whole library before the release settled its conventions
(`docs/ARCHITECTURE.md`) and removed the duplicated kernels:

- invalid input returns an error instead of panicking: the interpolators of
  `com_dsp::estimation`, `Ofdm::to_time` and `to_active`, `oqam_stagger` and
  `oqam_destagger`, the LLRs of `Constellation::llr` (and `llr_with_variances` for one
  noise variance per symbol), `theory::ber` and `ser` outside the orders they cover,
  `ber_rayleigh_mrc` without a branch, the closed forms of chirp spread spectrum, CE-OFDM
  and OTFS prefixes longer than a block, FBMC demodulation of a signal of another length;
- one kernel per method, one layer down: decibels (`db2amp`, `amp2db`), `qfuncinv`,
  `diric`, `try_simpson`, `gauss_legendre`, `linalg::solve_tridiagonal` and the unitary
  and centred FFT in `com-math`; cyclic prefixes, rotations, interpolation
  (`com_dsp::interpolation`, gathered from the estimators and the timing recovery),
  spectral estimation (`com_dsp::spectrum`) and Bussgang's decomposition
  (`com_dsp::nonlinear`, shared by clipping and the amplifiers) in `com-dsp`;
  `hard_decisions` and the seeded families of sequences (`rng::member`) in `com-core`,
  their QPSK draw in `com_mapping::source::qpsk`; `add_noise` in `com-channel`; the frequency correlation of a power-delay
  profile; the PRBS on the shared LFSR;
- the channel stays in `com-channel`: the delay–Doppler channel of OTFS
  (`DelayDoppler`, `DelayDopplerModel`), the validation tests draw their channels there;
- one API for the mutual information over AWGN and Rayleigh fading
  (`theory::constellation_mi`, `constellation_gmi`, the CM and BICM capacities),
  `theory::ber_exact`, and `ser_orthogonal_noncoherent` shared by noncoherent FSK and CSS;
- the modems share their vocabulary (`n_fft`, `n_active`, `n_cp`, `block_len`, public
  configurations checked by `check`) and the `Modulate`/`Demodulate` traits (UFMC, OTFS);
  `BlockCode` for BCH and the DVB-S2 FEC;
- names: `_lin` on linear SNRs, `noise_var`, `from_name` for the parsers that take
  arguments, `water_filling_powers`, `Domain` (real or complex message passing),
  `decode_alamouti`, the argument order `(rx, tx)` of the metrics and `(kind, m, …)` of
  the closed forms; Python `bandwidth_hz`, `smoothness`, `n_fft`, keyword-only options;
- Python bindings of OTFS (`Otfs`, `delay_doppler_paths`), `fading` for `theory_mi` and
  `theory_gmi`.
- the documentation follows one taxonomy, in the order of a link (foundations, coding and
  shaping, waveforms, MIMO, wireless channels, optical, hardware impairments, receiver
  DSP, metrics and analysis, systems and standards, simulation and specifications): the
  Python API reference, now complete (a test fails if an exported name is missing), the
  gallery, one sub-gallery per theme with the reproduced figures in their theme,
  `docs/LITERATURE.md` and `docs/MATURITY.md`; the maturity tables of `MATURITY.md`, of
  the README and of the Python documentation are generated from
  `validation/maturity.yaml` (`just maturity`, checked by `just check`).

### Waveforms and hardware (review of 30 October 2026)

- Verb traits: CE-OFDM (its receiver a field) and FBMC (the OQAM staggering inside)
  implement `Modulate` and `Demodulate`; every modem (`Ofdm`, `Ufmc`, `Otfs`, `CeOfdm`,
  `Fbmc`) carries its `sample_rate` (1 Hz by default) and `demodulate_into` refuses an
  input at another rate. ADR 0012 proposes a container of indices for CSS and CPM.
- Placement: `WdmTransmitter` moves to `com_systems::wdm`; `wdm_grid` and
  `wdm_demultiplex` to `com_dsp::signal`; `Path` to `com_core::channel` (re-exported by
  `com_channel::multipath`); the envelope limiter to `com_dsp::nonlinear::limit_envelope`
  and the static predistorter to `com_dsp::compensation::dpd::static_predistort`
  (`Rapp::predistort` calls it).
- One kernel each: `Laser::carrier` (the laser and the WDM multiplex), `phase_noise` on
  `increment_sd`, `shot_noise_variance` and `thermal_noise_variance`,
  `ResourceGrid::data_indices`, the OFDM link on `Ofdm::modulate`/`demodulate`,
  `com_math::argmax`, `com_dsp::filter::{normalize_energy, response_at}`,
  `Signal::map_streams_in_place` for the in-place `Impair` blocks.
- `Otfs::effective_channel(&[Path])` returns a sparse `EffectiveChannel` (P entries per
  row; `to_dense` for the detectors that take a matrix).
- New: `nr::Carrier` with `demodulate`; `Photodiode::detect_signal` and
  `coherent_receiver_signal` on a `Signal`; `Rapp::bussgang`, `Saleh::bussgang`.
- Checks instead of NaN, panics or silent changes: `ResourceGrid::check` (a spacing of
  zero), `PowerDelayProfile::new`, `Css::lora`, `Rapp`/`Saleh`/`Mzm::check`, the
  laser, the phase noise, `ResourceGrid::extract`, `Cpm::min_distance2` (bounded search);
  the length mismatches of NR return `Error::Shape`.
- Numerics: the chirps of CSS are tabulated (within 10⁻¹¹ of the direct evaluation,
  closer to the exact phase; the error counts of the validation links are unchanged);
  `Cpm::min_distance2` integrates by Simpson's rule (within 10⁻¹¹ of the closed form of
  CPFSK, 4·10⁻⁶ before); `Laser::field` at zero linewidth no longer draws its null
  phase increments (its random stream changes there only).
- Migration: `Css::lora`, `PowerDelayProfile::{new, uniform}`, `Laser::field`,
  `phase_noise`, `increment_sd`, `Rapp::{amplify, predistort}`, `Saleh::amplify`,
  `ResourceGrid::extract` and `Cpm::min_distance2` return `Result`;
  `nr::modulate(nr, grid, k, n_fft, first)` is `nr::Carrier { numerology: nr,
  n_subcarriers: k, n_fft, first_symbol: first }.modulate(grid)`;
  `Otfs::effective_channel(&tuples, fs)` is `Otfs { sample_rate: Hertz(fs), .. }
  .effective_channel(&paths)?.to_dense()`; a full literal of `Ofdm`, `Ufmc`, `Otfs` or
  `CeOfdm` adds `sample_rate` (and `receiver` for `CeOfdm`), or starts from `new`; the
  fields of `PowerDelayProfile` are getters; `com_hardware::optical::{WdmTransmitter,
  wdm_grid, wdm_demultiplex}` are `com_systems::wdm::WdmTransmitter` and
  `com_dsp::signal::{wdm_grid, wdm_demultiplex}`.

### Documentation

- The Python documentation (Sphinx): getting started, six tutorials, a user guide (with
  pages on the waveforms beyond OFDM and on optical wireless links), a
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
