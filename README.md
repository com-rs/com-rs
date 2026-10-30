# com-rs

[![CI](https://github.com/com-rs/com-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/com-rs/com-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/comrs.svg)](https://crates.io/crates/comrs)
[![Documentation](https://img.shields.io/badge/docs-com--rs.github.io-blue.svg)](https://com-rs.github.io/com-rs/)
[![docs.rs](https://img.shields.io/docsrs/comrs)](https://docs.rs/comrs)
[![PyPI](https://img.shields.io/pypi/v/comrs.svg)](https://pypi.org/project/comrs/)
[![Licence](https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/com-rs/com-rs#licence)

Link-level simulation of digital communication systems, wireless and optical, in Rust
with Python bindings: the blocks of a link, a parallel Monte Carlo runner whose results
do not depend on the thread count, and declarative campaigns checked against the
literature.

**Documentation:** <https://com-rs.github.io/com-rs/> (the Python package: getting
started, tutorials, a gallery of classical results and of reproduced figures of the
literature, the API) and <https://docs.rs/comrs> (the Rust API).

## Install

```bash
cargo add comrs          # the Rust library
pip install comrs        # the Python package (NumPy arrays, Python 3.11 or later)
cargo install com-spec   # the `comrs` command, which runs campaigns written as YAML files
```

## Example

The BER of QPSK over AWGN, on every core, until 200 errors are counted at each Eb/N0:

```rust
use comrs::prelude::*;
use comrs::sim::scenario::from_fn;

let qpsk = from_fn("qpsk", |cond, rng, rec| {
    let c = Constellation::qam(4)?.normalized();
    let bits = source::bits(2000, rng);
    let rx = awgn::awgn_ebn0(&c.modulate(&bits)?, cond.f64("ebn0_db")?, 2.0, 1.0, rng);
    rec.errors("ber", metrics::bit_errors(&c.demodulate(&rx), &bits) as u64, 2000);
    Ok(())
});
let stop = Stop::min_errors("ber", 200).or(Stop::max_trials(10_000));
let grid = Grid::new().axis("ebn0_db", [0.0, 2.0, 4.0, 6.0, 8.0]);
let results = Runner::new().seed(7).stop(stop).run(&qpsk, &grid)?;
```

The same link in Python, run on the Rust runner:

```python
import comrs
from comrs import blocks as B

link = comrs.Chain(tx=B.Source(constellation="QPSK", n_symbols=4096),
                   channel=B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")))
res = link.run(grid={"ebn0_db": [0, 2, 4, 6, 8]}, metrics=["ber"],
               stop={"min_errors": {"ber": 200}, "max_trials": 10_000})
```

**Documentation:** the Python package, with tutorials and a gallery of classical
results, at <https://com-rs.github.io/com-rs/>; the Rust API at
<https://docs.rs/comrs>; the [changelog](https://github.com/com-rs/com-rs/blob/main/CHANGELOG.md).

## Positioning

`com-rs` aims to be a reference link-level simulator: fast, reproducible bit for bit
(seeded streams independent of the thread count), and validated block by block against
the literature. It is meant to complement, not replace, the existing tools. Sionna
targets differentiable, GPU-accelerated end-to-end learning. The MATLAB toolboxes and
AFF3CT cover wider or more specialised ground. `com-rs` offers a single, documented
chain from the waveform to the metrics, wireless and optical, whose results can be
traced to a closed form, a standard or a published curve.

The blocks follow the layout, names and conventions of
[OptiCommPy](https://github.com/edsonportosilva/OptiCommPy), so that a user of one
finds the same blocks in the other. The algorithms are written from their publications.
OptiCommPy (GPL-3.0) is used only as a test oracle, never imported by the package, which
is released under the MIT or Apache-2.0 licence; the tests read its recorded outputs
(`tests/oracle/`), so it need not be installed.

A disagreement with a reference or a missing check is a bug: please
[report it](https://github.com/com-rs/com-rs/issues/new/choose) with the
specification that reproduces it. [`CONTRIBUTING.md`](https://github.com/com-rs/com-rs/blob/main/CONTRIBUTING.md) explains how
a block is validated.

## Scope and non-goals

**In scope:** the physical layer of a point-to-point link, simulated sample by sample
and symbol by symbol, wireless and optical: sources and constellations, coding and
shaping, waveforms, channels and propagation, hardware impairments, receiver DSP,
detection and decoding, metrics; Monte Carlo campaigns with confidence intervals and
reproducible draws; and, for every block, the evidence that it is right (a closed form,
a table of a standard, a published figure or the recorded output of a reference tool).

**Not in scope**, because other tools do it well:

- **System and network level**: routing, multiple links and nodes, scheduling, traffic,
  link budgets of whole networks (GNPy for optical networks, ns-3, system-level
  simulators).
- **Differentiable, GPU-accelerated learning**: training end-to-end systems or neural
  receivers (Sionna, PyTorch). com-rs evaluates trained blocks at scale, from their
  parameters, but does not train them.
- **Real-time and hardware**: software-defined radio processing, FPGA or embedded
  targets (GNU Radio and vendor toolchains); com-rs reads and writes recordings
  (SigMF) but does not stream.
- **Graphical tools**: com-rs is a library, a command line and declarative files; a user
  interface belongs in a separate project built on them.
- **Unvalidated models**: a block enters the library with the reference it is checked
  against; a model without one stays in the project that needs it, which can add it
  from its own crate (`com_spec::blocks::register`, see
  [Blocks from other crates](https://github.com/com-rs/com-rs#blocks-from-other-crates)).

## How this library was written

com-rs is developed with an AI coding assistant, and says so plainly:

- **Claude (Anthropic), through Claude Code, wrote most of the code, tests and
  documentation**, under the direction of the author, Vincent Choqueuse.
- **The author decided**: the scope and the architecture, the models and the
  algorithms, the publications and standards each block is checked against, the
  tolerances, and what counts as validated. Every change was reviewed and accepted by
  the author.
- **Correctness rests on evidence, not on who wrote a line.** Every block is checked
  against an independent reference: closed-form error rates and capacities, the tables
  of the standards (ETSI EN 302 307-1 for DVB-S2, 3GPP for the channel models and the
  codes), published figures (each cited with its authors, venue, year and figure or
  table), and the recorded outputs of reference implementations (OptiCommPy, GNPy,
  AFF3CT) used as test oracles; the evidence of every module is graded in
  [`docs/MATURITY.md`](https://github.com/com-rs/com-rs/blob/main/docs/MATURITY.md). The checks are reproducible: see
  [Validation against the literature](https://github.com/com-rs/com-rs#validation-against-the-literature).
- **The rules are in the repository.**
  [`docs/ARCHITECTURE.md`](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)
  and the decisions of [`docs/adr/`](https://github.com/com-rs/com-rs/tree/main/docs/adr)
  apply to every contribution, human or generated;
  [`CLAUDE.md`](https://github.com/com-rs/com-rs/blob/main/CLAUDE.md) is the
  file of instructions that the assistant reads (the same checks a human contributor
  runs), and [`docs/API.md`](https://github.com/com-rs/com-rs/blob/main/docs/API.md)
  the index it consults before adding a function.
- Contributions written with an assistant are welcome on the same terms: they cite
  their references and come with the checks that validate them
  ([`CONTRIBUTING.md`](https://github.com/com-rs/com-rs/blob/main/CONTRIBUTING.md)).

## Contents

The workspace is organised by function, in five layers whose dependencies point
downward only (see
[docs/ARCHITECTURE.md](https://github.com/com-rs/com-rs/blob/main/docs/ARCHITECTURE.md)
and the decisions of
[docs/adr/](https://github.com/com-rs/com-rs/tree/main/docs/adr)). Every crate
is published on crates.io and re-exported by the facade `comrs`, so Rust users write
`comrs::channel::awgn`.

| Layer | Crate (`comrs::…`) | Python names | Content |
|---|---|---|---|
| 0 | `com-core` (`core`) | `Rng` | multi-stream containers (`Bits`, `Llr`, `Symbols`, `Signal`, `DualPol`), the verb traits (`Map`, `Demap`, `Propagate`, `Impair`, …), unit newtypes (`Db`, `EbN0`, `EsN0`, `Hertz`), the seeded generator (xoshiro256\*\*, one stream per trial), the error type |
| 0 | `com-math` (`math`) | `db2lin`, `lin2db`, `dbm2w`, `w2dbm`, `qfunc`, `entropy` | conversions, special functions, Gauss–Hermite quadrature, FFT plans, complex linear algebra (QR, Cholesky, SVD), nearest-point search, the split-step solvers of the NLSE and the Manakov equation, GF(2) and GF(2^m), exact integers, the Gray code, bisection and golden-section search |
| 1 | `com-dsp` (`dsp`) | `pulse_shape`, `rrc_spectrum`, `fir_filter`, `convolve`, `upsample`, `downsample`, `decimate`, `resample_fft`, `low_pass_fir`, `pnorm`, `anorm`, `freq_shift`, `moving_average`, `delay_fft`, `cdc`, `cd_taps_needed`, `cd_fir_taps`, `cd_dispersion_factor`, `cd_savory_taps`, `cd_ls_taps`, `cd_ls_error`, `adaptive_equalizer`, `fourth_power_foe`, `viterbi_viterbi`, `bps`, `ddpll`, `derotate`, `unwrap`, `gardner_ted`, `gardner_ted_nyquist`, `interpolate_cubic`, `gardner_clock_recovery`, `MemoryPolynomialDpd` | `filter`, `resample`, `signal`; `sync::{carrier, timing, frame}`; `equalizer` (the n×n butterfly: CMA, RDE, NLMS, DD-LMS, schedules); `compensation::{cd, dbp, dpd}` (single- and dual-polarisation back-propagation; digital predistortion by memory polynomials, indirect learning); `window`; `estimation` (LS, LMMSE); `interpolation` (linear, quadratic, spline, band-limited, cubic Lagrange); `spectrum` (Welch, periodogram); `nonlinear` (Bussgang's decomposition, Banelli's series) |
| 2 | `com-mapping` (`mapping`) | `Constellation`, `gray_code`, `bits`, `prbs`, `symbols`, `maxwell_boltzmann`, `theory_ber`, `theory_ser`, `theory_mi`, `theory_gmi`, `Dpsk`, `theory_ber_dpsk`, `differential_phase_tail` | QAM, PSK, PAM, OOK and custom constellations with OptiCommPy's Gray labels, decisions, LLRs (exact or max-log); sources (PRBS of ITU-T O.150, shaped symbols); closed-form error and information rates; differential M-PSK (noncoherent and coherent detection, closed-form BER) |
| 2 | `com-waveform` (`waveform`) | `Ofdm`, `ResourceGrid`, `ofdm_estimate`, `ofdm_preamble`, `schmidl_cox`, `ofdm_synchronize`, `PaprReduction`, `theory_papr_ccdf`, `clipping_bussgang`, `Cpm`, `msk_precode`, `msk_bits`, `msk_psd`, `CeOfdm`, `Css`, `Fbmc`, `Ufmc`, `Otfs` | OFDM and DFT-spread OFDM; continuous-phase modulation (CPFSK, MSK, GMSK, raised-cosine pulses, maximum-likelihood sequence and differential detection); resource grids with block, comb or lattice pilots, LS and LMMSE channel estimation, Schmidl–Cox synchronisation, per-subcarrier MIMO-OFDM detection; PAPR: its distribution, clipping (Bussgang), clipping and filtering, SLM, PTS, tone reservation, ACE, μ-law and exponential companding; constant-envelope OFDM, its phase demodulator with or without unwrapping, and the error rates of its linear regime; chirp spread spectrum (the modulation of LoRa); FBMC/OQAM with the PHYDYAS prototype; UFMC; OTFS and its effective delay–Doppler channel |
| 2 | `com-channel` (`channel`) | `awgn`, `awgn_ebn0`, `mimo_channel`, `exponential_correlation`, `TdlChannel`, `CdlChannel`, `ChannelTaps`, `delay_doppler_paths`, `cdl_profile`, `cdl_k_factor_db`, `power_delay_profile`, `rms_delay_spread`, `FiberParams`, `linear_fiber_channel`, `Ssfm`, `Manakov`, `ManakovSsfm`, `Edfa`, `raman_profile`, `gn_model`, `gn_closed_form`, `format_moments` | AWGN, flat fading (SISO and MIMO, Rayleigh and Rician, Kronecker antenna correlation), the clustered delay lines CDL-A to CDL-E of 3GPP TR 38.901 between antenna arrays, tapped-delay-line multipath (3GPP EPA, EVA, ETU, ITU Pedestrian and Vehicular A and B) with sum-of-sinusoids Doppler, delay–Doppler channels of discrete paths; optical: fibre parameters, linear fibre, split-step Fourier (scalar, Manakov with PMD, and the Manakov scheme of OptiCommPy's `manakovSSF` with adaptive steps) and their back-propagation, the EDFA with ASE, distributed Raman amplification with co- and counter-propagating pumps and inter-channel stimulated Raman scattering, the GN and EGN models of the nonlinear interference with the power profiles of the spans |
| 2 | `com-hardware` (`hardware`) | `pa_rapp`, `pa_saleh`, `pa_memory_polynomial`, `rapp_predistort`, `phase_noise`, `laser`, `phase_modulator`, `mzm`, `iqm`, `Photodiode`, `balanced_photodiode`, `hybrid_2x4`, `coherent_receiver`, `wdm_grid` | power amplifiers (Rapp, Saleh, memory polynomial of Ding et al. 2004); Wiener phase noise and frequency offset; transmitter and receiver IQ imbalance; transmitter and receiver optical front ends, the WDM transmitter |
| 2 | `com-metrics` (`metrics`) | `ber`, `snr_db`, `fast_ber_calc`, `evm`, `ber_to_q_db`, `monte_carlo_gmi`, `monte_carlo_mi`, `information_rates`, `posterior_rates`, `papr_db`, `ccdf`, `psd`, `periodogram`, `aclr_db` | error rates, SNR, EVM, generalised and symbol-wise mutual information, the rates of a receiver's posterior, PAPR and CCDF, power spectral density (Welch, aligned periodogram), ACLR |
| 2 | `com-mimo` (`mimo`) | `mimo_detect`, `mimo_llr`, `mimo_estimate`, `Ostbc`, `mimo_capacity`, `water_filling`, `eigenmode`, `zf_precoder` | MIMO detection (ZF, MMSE, V-BLAST SIC, ML, sphere decoder, K-best) with soft outputs, the receiver of a link, orthogonal space-time block codes (Alamouti, rate 3/4), eigenmode and zero-forcing precoding, capacity |
| 2 | `com-fec` (`fec`) | `Crc`, `Scrambler`, `Interleaver`, `ConvolutionalCode`, `TurboCode`, `LdpcCode`, `PolarCode`, `NrLdpc`, `NrPolar`, `ReedSolomonCode`, `BchCode`, `Ccdm`, `free_distance`, `theory_rs_wer`, `theory_rs_ser`, `theory_rs_ber`, `pack_symbols`, `unpack_symbols`, `biawgn_capacity`, `normal_approximation`, `shannon_limit_ebn0_db` | CRC and scrambling of 5G NR, block, random and QPP interleavers, convolutional codes (Viterbi, BCJR) and their puncturing (802.11, DVB-S patterns), LTE turbo codes, Reed–Solomon codes (any field, shortening, erasures, closed-form error rates), LDPC codes (PEG construction, repeat-accumulate codes, alist files, sum-product, min-sum, layered), shortened binary BCH codes, polar codes (SC, CRC-aided SCL), the LDPC and polar codes of 5G NR with their rate matching, constant-composition distribution matching, BI-AWGN capacity and the normal approximation |
| 3 | `com-sim` (`sim`) | `comrs.sim` | the Monte Carlo framework (below) |
| 4 | `com-spec` (`spec`, `io`) | `run_spec`, `verify_spec`, `search_spec`, `spec_blocks`, `spec_conditions`, `spec_schema`, `spec_check_link`, `read_sigmf`, `write_sigmf` | declarative campaigns from YAML, JSON or TOML files and their validation against the literature, SigMF recordings, raw IQ files (`cf32_le`, `ci16_le`, …) and NPY arrays, the `comrs` command |
| 4 | `com-systems` (`systems`) | `Dvbs2`, `dvbs2_bbframe`, `dvbs2_bbframe_data`, `dvbs2_decode_plheader` | reference chains of standards: DVB-S2 (ETSI EN 302 307-1), from the BBFRAME to the scrambled PLFRAME and back, its BCH and LDPC codes, the 28 MODCODs, normal and short frames, pilots; probabilistic amplitude shaping (Böcherer, Steiner & Schulte 2015) with the DVB-S2 codes and the bit mapper of the paper |

The Python extension is `bindings/com-py` (package `comrs`); `validation/` holds the statistical validation, `examples/` the
runnable examples, `benches/` the criterion benchmarks.

Every public item is documented, with the formula it implements and an example when
the use is not obvious (`just doc`, or `help(comrs.bps)` in Python).

## Monte Carlo campaigns

The `com-sim` crate runs a **scenario** (one random trial of a link) on every **condition**
of a **grid**, as many times as a **stopping rule** requires, in parallel on every core,
and reports every metric with a confidence interval. Its organisation follows PyTorch
Lightning:

| PyTorch Lightning | `comrs::sim` | Role |
|---|---|---|
| `LightningModule` | `Scenario` | `setup` once per condition, `trial` many times |
| `self.log(...)` | `Record` | error counts (Wilson intervals) or values (normal intervals) |
| `Trainer` | `Runner` | batches of trials on a thread pool |
| `seed_everything` | `Runner::seed` | trial `t` of condition `c` draws from `stream(seed, name, c, t)` |
| `max_epochs`, `EarlyStopping` | `Stop` | `max_trials`, `min_errors`, `precision`, `max_time`, combined with `or`/`and` |
| `Callback` | `Callback` | `ProgressBar`, `SkipAfter` (skip the rest of a sweep below a BER floor) |
| `Logger` | `Logger` | `CsvLogger` (a row per condition as it completes), `JsonLogger` |
| `fast_dev_run` | `Runner::fast_dev_run` | one batch per condition |

The records of a batch are merged in trial order, so the results are identical whatever
the number of threads.

```rust
use comrs::prelude::*;
use comrs::sim::scenario::from_fn;

let qpsk = from_fn("qpsk", |cond, rng, rec| {
    let c = Constellation::qam(4)?.normalized();
    let bits = source::bits(2000, rng);
    let rx = awgn::awgn_ebn0(&c.modulate(&bits)?, cond.f64("ebn0_db")?, 2.0, 1.0, rng);
    rec.errors("ber", metrics::bit_errors(&c.demodulate(&rx), &bits) as u64, 2000);
    Ok(())
});
let results = Runner::new()
    .seed(7)
    .stop(Stop::min_errors("ber", 500).or(Stop::max_trials(10_000)))
    .callback(ProgressBar::default())
    .callback(SkipAfter::below("ber", 1e-6, "ebn0_db"))
    .logger(CsvLogger::new("results/qpsk.csv"))
    .run(&qpsk, &Grid::new().axis("ebn0_db", [0.0, 2.0, 4.0, 6.0, 8.0]))?;
```

`comrs.sim` runs trials written in Python on the same Rust runner, with the same
streams, statistics, stopping rules and loggers (on one thread, since the trials hold
the GIL):

```python
import comrs
from comrs import sim

c = comrs.Constellation("qam", 16).normalized()

def trial(cond, rng, rec):
    b = comrs.bits(4 * 1000, rng)
    y = comrs.awgn_ebn0(c.modulate(b), cond["ebn0_db"], 4, rng)
    rec.errors("ber", int((c.demodulate(y) != b).sum()), b.size)

res = sim.run(trial, sim.grid(ebn0_db=[6.0, 8.0, 10.0]), name="qam16",
              stop=sim.Stop.min_errors("ber", 200) | sim.Stop.max_trials(5000))
```

## Declarative campaigns

Every number of a study should come from a specification file: the grid, the link as a
list of blocks, the metrics and the stopping rule, run by the same parallel runner. A
simulation that cannot be written as a specification shows that a block is missing.

```yaml
name: qam_awgn
seed: 2024
grid:
  m: [4, 16, 64]
  ebn0_db: {from: 0, to: 20, step: 2}
link:
  - {type: source, constellation: {kind: qam, m: $m}, n_symbols: 4096}
  - {type: awgn_ebn0, ebn0_db: $ebn0_db}
metrics: {report: [ber, ser], scale: none}
stop: {min_errors: {ber: 200}, min_trials: 128, max_trials: 4000}
callbacks:
  skip_below: {metric: ber, threshold: 1e-6, along: ebn0_db}
```

```bash
comrs run specs/qam_awgn.yaml                              # after `cargo install com-spec`
cargo run --release -p com-spec -- run specs/qam_awgn.yaml # from a clone of the repository
```

- `$name` in a block is the value of the grid variable `name`, with its type. `${name}`
  inside a longer string, such as a file name, is replaced by its text.
- The stopping rule is `(targets and min_trials) or max_trials or max_time_s`.
- `comrs blocks` lists the 57 blocks with their parameters, defaults and units:
  sources and channels, optical propagation (`fiber`, `ssfm`, `manakov`, WDM
  multiplexing), receiver DSP, constant-envelope waveforms (`ce_ofdm_tx`, `cpm_tx` and
  their receivers), wireless blocks (antennas, `rayleigh`, `tdl`, `cdl`, OFDM with
  pilots and channel estimation, MIMO detection) and channel coding (`fec_encode`,
  `map`, `demap`, `fec_decode`, with convolutional, punctured, turbo, LDPC, polar and
  Reed–Solomon codes), and the analysis blocks (`probe` measures metrics inside the
  link, `dump` saves the samples at a point as SigMF recordings). `source` with
  `pols: 2` gives a dual-polarisation channel whose polarisations go through the
  single-polarisation blocks separately.
- `profile: true` (or `comrs run --profile`) times every block of every trial and
  prints where the time goes, block by block.
  Unknown blocks and parameters are errors that name the block.
- `comrs check` builds every condition and runs one trial of each. It writes nothing.
- `comrs.run_spec(path | text | dict)` runs a specification from Python on the
  Rust runner. It returns `comrs.Results`, one entry per condition.

| Specification | Content |
|---|---|
| `specs/qam_awgn.yaml` | BER of QAM over AWGN, the example `ber_awgn` |
| `specs/coherent_link.yaml` | the coherent link and its DSP chain, the example `coherent_link` |
| `specs/ssfm_dbp.yaml` | launch-power sweep, CDC against DBP, the example `ssfm_dbp` |
| `specs/capture_16qam.yaml` | the receiver DSP chain on a recorded capture (below) |
| `specs/mimo_diversity.yaml` | Alamouti and zero forcing over 2 × 2 Rayleigh fading, checked against maximal-ratio combining |
| `specs/ofdm_eva.yaml` | OFDM over EVA with known, LMMSE and LS channels, checked against Rayleigh fading per subcarrier |
| `specs/bicm_ldpc.yaml` | LDPC (1024, 512) BICM with 16-QAM over AWGN: BER, coded BER and BLER |
| `specs/shaping_awgn.yaml` | MI and GMI of Maxwell-Boltzmann 64-QAM over AWGN, checked against the closed forms with the priors |
| `specs/pas_64qam.yaml` | probabilistic amplitude shaping of 64-QAM (CCDM, LDPC of rate 2/3) at 3 bits per symbol |
| `specs/wdm_gn.yaml` | five DP channels over 5 × 80 km by the Manakov equation, checked against the GN model (Gaussian symbols) and the EGN model (16-QAM) |
| `specs/raman_wdm.yaml` | the same channels over 3 × 100 km with a counter-propagating Raman pump and gain-flattened EDFAs, checked against the GN model with the Raman power profiles |
| `specs/polmux_cma.yaml`, `specs/polmux_cma_45deg.yaml` | blind 2 × 2 polarisation demultiplexing (CMA then RDE) under random rotation and PMD, with and without Kikuchi's initialisation against the singularity |
| `specs/wdm_dp16qam.yaml` | the same link with the practical receiver (BPS), BER and GMI against the launch power |
| `specs/validation/dvbs2_table13.yaml` | the 28 MODCODs of DVB-S2 (BCH, LDPC, APSK, PL framing), searched Es/N0 at a FER of 10 % against Table 13 of the standard |
| `specs/validation/*.yaml` | results checked against the literature (below) |

## Validation against the literature

A specification can state what its results must be: a closed form, or a curve read from
a publication, with its source and a tolerance. `comrs verify` runs it, prints every
check and fails if one fails.

An expectation can also compare a metric with another metric of the same condition
(`reference`), for example a theory that a block records, within a multiplicative
`factor`.

```yaml
expect:
  - name: DBP, 1 step per span
    source: Häger & Pfister, IEEE JSAC 39(1), 2021, Fig. 5
    where: {receiver: dbp1}
    metric: snr_eff_db
    curve: {along: power_dbm, x: [-10, -8, -6], y: [16.65, 18.64, 20.60]}
    atol: 0.3
```

| Specification | Reference | Agreement |
|---|---|---|
| `specs/validation/proakis_awgn.yaml` | Proakis & Salehi: BER and SER of QAM, PSK and PAM over AWGN | 47 checks, exact SER inside the 99.9 % intervals |
| `specs/validation/information_rates.yaml` | MI and GMI by Gauss–Hermite quadrature (Alvarado et al. 2018, OptiCommPy `theoryMI`) | within 0.02 bit |
| `specs/validation/hager2021_fig5.yaml` | Häger & Pfister 2021, Fig. 5: linear equalisation and DBP over 25 × 80 km | within 0.13 dB of the published curves |
| `specs/validation/required_snr.yaml` | Eb/N0 of Gray QAM at a BER of 10⁻³, by search, against the inverted closed forms | within 0.02 dB |
| `specs/validation/msk_bpsk.yaml` | Proakis & Salehi, Sec. 3.3-2: precoded MSK with maximum-likelihood sequence detection has the BER of BPSK | within 4 % at 2, 4 and 6 dB |
| `specs/validation/thompson2008_eq15.yaml` | Thompson et al., IEEE Trans. Commun. 56(8), 2008, eq. (15): the BER and SER of CE-OFDM in the linear regime, 16-QAM (two 4-PAM) at 2πh = 0.5 and 0.7 above the threshold of the phase demodulator | within 8 % at 2πh = 0.5 (the heavier tails of the phase error), 14 % at 0.7 and 18 dB |
| `specs/mimo_diversity.yaml` | MRC of Rayleigh branches (Proakis & Salehi, eq. 13.4-15): Alamouti 2 × 2, zero forcing 2 × 2 | within 5 % |
| `specs/ofdm_eva.yaml` | flat Rayleigh fading per subcarrier, OFDM over EVA with the channel known | within 4 % |
| `specs/wdm_gn.yaml` | the GN model (Poggiolini 2012) for Gaussian symbols and the EGN model (Carena et al. 2014) for 16-QAM, −8 to +4 dBm per channel | within 0.3 dB (14 checks; the GN model is 1.7 dB off for 16-QAM) |
| `specs/raman_wdm.yaml` | the GN model with the power profiles of the channels (Semrau et al. 2019), Gaussian symbols, without and with a 500 mW counter-pump, −10 to +2 dBm per channel | within 0.21 dB (8 checks) |
| `specs/shaping_awgn.yaml` | the MI and the GMI of bit-metric decoding with the priors (Böcherer 2014), Gauss–Hermite quadrature | within 0.02 bit (30 checks) |
| `specs/pas_64qam.yaml` | the GMI of Maxwell–Boltzmann 64-QAM (λ = 2.55) with the priors (Böcherer 2014) | within 0.03 bit |
| `specs/wdm_dp16qam.yaml` | the ASE-limited SNR of the link (Essiambre et al., J. Lightw. Technol. 28(4), 2010, eq. (54)), at −8 dBm per channel | within 0.3 dB |
| `specs/validation/dvbs2_table13.yaml` | ETSI EN 302 307-1, Table 13: the Es/N0 of the 28 DVB-S2 MODCODs at PER 1e-7, normal frames | FER of 10 % 0.17 to 0.38 dB below the table (0.41 to 0.72 dB for QPSK 1/4 to 2/5), all 28 within [threshold − 0.8 dB, threshold] |
| `specs/polmux_cma*.yaml` | Kikuchi (2008): the singularity of the blind 2 × 2 equaliser | 48 % of singular trials at 45° from the identity, none with the initialisation; SNR within 1 dB of the channel |

The unit tests named `literature` check the tables and closed forms of the documents of
`references/` (the list and the files are described in
[`references/README.md`](https://github.com/com-rs/com-rs/blob/main/references/README.md)), every test naming its
figure, table or equation:

| Test | Reference | Checks |
|---|---|---|
| `com_channel::multipath` | ITU-R M.1225 (1997), Tables 2, 4 and 5 | the taps of the pedestrian and vehicular channels A and B, verbatim; their RMS delay spreads |
| `com_fec::crc` | 3GPP TS 38.212 V18.5.0, Sec. 5.1 | the six generator polynomials of NR |
| `com_fec::ldpc` | Hu, Eleftheriou & Arnold, IEEE Trans. Inf. Theory 51(1), 2005, Sec. VI | the (504, 252) PEG code has girth 8 and the check degrees of the paper |
| `com_mimo::stbc` | Tarokh, Jafarkhani & Calderbank, IEEE Trans. Inf. Theory 45(5), 1999, eqs. (39)–(40) | the rate-3/4 designs of the paper are orthogonal and decoded exactly |
| `com_waveform::papr` | Ochiai & Imai, IEEE Trans. Commun. 49(2), 2001, eqs. (11) and (27) | the Nyquist-rate and asymptotic CCDFs of the PAPR |
| `com_waveform::ce_ofdm` | Thompson et al., IEEE Trans. Commun. 56(8), 2008, eq. (15); Ahmed & Zeidler, IEEE Trans. Signal Process. 63(17), 2015, eqs. (8), (17), (26); Ahmed et al., MILCOM 2012, eqs. (13)–(14), Fig. 6; Guerreiro et al., VTC Fall 2015, eq. (13), Fig. 4, and IEEE Commun. Lett. 20(11), 2016; Pacheco & Hatzinakos, ICASSP 2006, Fig. 1 | the BER of the linear regime; the series of the receivers of Ahmed & Zeidler (a misprint in eq. (8)); the effect of a cycle slip on each subcarrier; the powers of the intermodulation products and the carrier line; the optimum gains of 1.9 and 3.5 dB; the 99 % bandwidth |
| `com_waveform::css` | Chiani & Elzanaty, IEEE Internet Things J. 6(5), 2019, Table I, eq. (7), Property 4; Afisiadis et al., IEEE Trans. Wireless Commun. 19(2), 2020, Fig. 2, eqs. (16)–(18), (38)–(43), (58); Vangelista, IEEE Signal Process. Lett. 24(12), 2017, Fig. 1 | the cross-correlation, the 99 % band and the lines of the LoRa spectrum; the exact SER and its approximations; the interference pattern in closed form (a sign misprint in eq. (58)); the AWGN curves of FSCM and FSK |
| `com_mapping::theory` | Cho & Yoon, IEEE Trans. Commun. 50(7), 2002, Table I, eqs. (9)–(22), Fig. 2 | the coefficients of Table I verbatim; the exact Gray BER of PAM, square and rectangular QAM against an enumeration of the decision regions; 4–5 dB per extra bit |
| `com_channel::optical::raman` | Semrau & Bayvel, J. Lightw. Technol. 36(14), 2018, eq. (10) | the ISRS power transfer of a 10 THz band over 100 km, within 5 % |
| `com_channel::cdl` | 3GPP TR 38.901 V17.0.0, Tables 7.7.1-1 to 7.7.1-5, 7.5-3, Sec. 7.7.3 | the CDL-A to CDL-E clusters verbatim, the unit delay spread of the normalised profiles (1.000 for A, B, C, E; 0.994 for D), the K-factors 13.3 and 22 dB of D and E within 0.2 dB |
| `com_waveform::cpm` | Proakis & Salehi, Digital Communications, 5th ed., Secs. 3.3-2 and 3.4-3; Murota & Hirade, IEEE Trans. Commun. 29(7), 1981, Table I; Aulin & Sundberg, IEEE Trans. Commun. 29(3), 1981 | MSK detected coherently has the BER of BPSK (4 and 6 dB, 2·10⁵ bits), its spectrum follows the closed form (null at 0.75/T, 3 dB bandwidth 0.60/T), the 99 % bandwidths of GMSK (0.86/T at BT = 0.25, 1.03/T at 0.5, 1.20/T for MSK), the minimum distances d²_min = 2 of MSK and 2(1 − sinc 2h) of CPFSK |
| `com_mapping::differential` | Proakis & Salehi, 5th ed., Sec. 4.5; Pawula, Rice & Roberts, IEEE Trans. Commun. 30(8), 1982, Sec. IV | noncoherent DBPSK against ½e^{−Eb/N0}, DQPSK and D8PSK against the distribution of the differential phase, coherent detection with differential decoding against 2p(1 − p) |
| `com_fec::rs` | Lin & Costello, Error Control Coding, 2nd ed., 2004, Secs. 7.3–7.4; ETSI EN 300 744 Sec. 4.3.2; CCSDS 131.0-B Sec. 4.2 | the generator polynomial and the decoding example of the RS(15, 9) code, the RS(204, 188) generator of DVB and the roots of the CCSDS RS(255, 223) code; RS(255, 223) corrects 16 errors and detects 17; the word error rate against the closed form |
| `com_fec::puncture` | Yasuda, Kashiki & Hirata, IEEE Trans. Commun. 32(3), 1984, Table I; IEEE 802.11-2020 Figs. 17-9 and 19-11; ETSI EN 300 421 Table 1 | the free distances 6, 5, 4, 3 of the punctured (133, 171) code at rates 2/3 to 7/8, the patterns of 802.11 and DVB-S |

The tests of `validation/tests/` (crate `com-validation`) compare Monte Carlo
simulations with published figures and tables and with recorded runs of AFF3CT; the
quick ones run with `cargo test`, the ignored ones by hand
(`cargo test --release -p com-validation -- --ignored`):

| Test | Reference | Checks |
|---|---|---|
| `validation/tests/aff3ct.rs` | AFF3CT v4.7.1 (Cassagne et al., SoftwareX 10, 2019), recorded by `tools/record_aff3ct.py` into `validation/aff3ct/` | 17 configurations over BPSK and AWGN (RS (15, 11), (63, 55), (255, 239); BCH (63, 45), (127, 92), (255, 223); the (133, 171) code with soft Viterbi; PEG LDPC codes with flooding and layered sum-product, normalised and offset min-sum, and the DVB-S2 short code; polar SC and CA-SCL; LTE turbo K = 40 log-MAP and K = 1024 max-log-MAP): at every point the 99 % Wilson intervals of the two frame error rates overlap and the bit error rates are within a burst-aware bound; the fixtures reach frame error rates of 4·10⁻⁵ to 2·10⁻³ |
| `validation/tests/mimo.rs` | Yang & Hanzo, IEEE Commun. Surveys Tuts. 17(4), 2015, Table X | the diversity orders of ML, sphere, ZF, MMSE, ZF/MMSE-(O)SIC and K-best (K = 1, 2, 4) on 2 × 2 and 2 × 4 QPSK links over i.i.d. Rayleigh fading: the high-SNR slopes within ±0.3 of the slope of maximal-ratio combining of the order of the table; on 2 × 4 the OSIC slopes (3.25 to 3.38) lie between orders 3 and 4 and are bounded by them (ignored) |
| `validation/tests/shaping.rs` (CCDM) | Schulte & Böcherer, IEEE Trans. Inf. Theory 62(1), 2016, Fig. 2 and eqs. (10), (17), (22) | the 50 points of the figure: the CCDM rates k/n exactly (within 6e-7), the normalised divergences within 2e-4 relative, the plotted bound (22) (ignored) |
| `validation/tests/shaping.rs` (information rates) | Böcherer, Steiner & Schulte, IEEE Trans. Commun. 63(12), 2015, Tables I and III; Liang & Kahn, J. Lightw. Technol. 43(4), 2025, eq. (27) and Fig. 6 | the SNRs of optimised MB, uniform and BMD 4- to 64-ASK within 2e-4 dB of the 4-decimal tables; πe/6 = 1.53 dB; the largest shaping gains of 256- to 65536-QAM 1.08 to 1.48 dB (paper ∼1 to ∼1.4 dB) |
| `validation/tests/shaping.rs` (PAS, ignored) | Böcherer, Steiner & Schulte, IEEE Trans. Commun. 63(12), 2015, Tables VI–X | PAS on the DVB-S2 LDPC codes (64 800 bits, bit mappers of the tables, 100 sum-product iterations): the SNR at the FER of the table within 0.05 dB of the paper at 6 points (4- to 64-ASK) |
| `validation/tests/message_passing.rs` | Jeon, Ghods, Maleki & Studer, arXiv:1811.01917, 2018, Table II, Sec. V and Fig. 6(a); Ma & Ping, IEEE Access 5, 2017, eq. (42), Figs. 2 and 3 | the recovery thresholds and critical noise variances of LAMA for BPSK, QPSK, 16- to 256-QAM, 8- and 16-PSK within a unit of the last digit; the state evolution against simulated systems of 1024 streams; OAMP-MF, -PINV and -LMMSE and AMP on i.i.d. and ill-conditioned matrices. Not reproduced from the text of Jeon et al.: 13.5 dB at βmid after 90 iterations (13.17 dB) and the SER floor of 0.08 at βmax (0.208) |
| `validation/tests/egn.rs` | Carena et al., Opt. Express 22(13), 2014, Fig. 1(a) | the self-channel interference of PM-QPSK by the GN and EGN models at 1, 10, 20 and 50 spans within 0.2 and 0.25 dB of the figure; the EGN correction against a Monte Carlo evaluation of the first-order perturbation (ignored) |

The validation found two problems, both fixed:

- **Coarse back-propagation** applied the nonlinear phase over the step length instead
  of the effective length. With one step per span it applied 60 % of the phase.
- **OptiCommPy's estimators of MI and GMI** normalise the power of the noisy signal.
  This underestimates the rates by 0.06 bit at 0 dB. The declarative metrics now use
  the least-squares channel gain (`information_rates`), and the OptiCommPy-compatible
  functions are kept for the oracle tests.

### Validation cards

A validation card states, for one algorithm, its inputs, the outputs it must give with
their tolerance, and the exact source of the expectation (a paper with its equation,
table or figure; a standard with its section; a reference implementation with its
version). The cards of [`validation/cards/`](validation/cards/) are data, not code:
`tests/test_cards.py` runs every case against the Python interface, a JSON Schema checks
their structure, and another library can run the same cards by mapping the function
names to its own.

```yaml
id: chang2002_fig4_floors
function: comrs.theory_ber_imperfect_csi
source: {kind: paper, reference: "Chang & Su, IEEE TWC 1(4), 2002", locator: "Fig. 4, read at 50 dB"}
tolerance: {rel: 0.3}
cases:
  - inputs: {modulation: 16qam, ebn0_db: 50.0, rho1: 0.999}
    expected: 3.5e-3
```

## Maturity

The strongest evidence of every module: **A**, validated against an external reference
(a published figure or table, a table of a standard reproduced verbatim, the recorded
output of a reference implementation); **B**, against a closed form only; **C**,
self-consistency only (round trips, invariants). The tests behind every grade and
the regimes not covered yet are in
[`docs/MATURITY.md`](https://github.com/com-rs/com-rs/blob/main/docs/MATURITY.md); the
comparisons with the literature, one row per paper with what was reproduced, a
green, orange or red badge and the inconsistencies found in the papers, are in
[`docs/LITERATURE.md`](https://github.com/com-rs/com-rs/blob/main/docs/LITERATURE.md).

<!-- generated by tools/maturity_table.py from validation/maturity.yaml: do not edit -->

Summary: 60 modules: 52 at A, 4 at B, 4 at C.

**Foundations**

| Module | Level | Strongest evidence |
|---|---|---|
| Constellations, mapping, demapping and closed-form error rates | A | OptiCommPy outputs; closed forms of Proakis & Salehi |
| Capacity and cutoff rate of CM and BICM | A | Caire, Taricco & Biglieri 1998 Figs. 4–7 |
| Sources (bits, PRBS, uniform and shaped symbols) | C | periods and distributions |
| Differential PSK | A | Pawula, Rice & Roberts 1982 Sec. IV-A; closed forms of Proakis & Salehi |
| Error rates with an imperfect channel estimate and over Jakes' fading | A | Chang & Su 2002 eqs. 31–53, Figs. 3–4 |

**Coding and shaping**

| Module | Level | Strongest evidence |
|---|---|---|
| Convolutional codes (Viterbi, BCJR) | A | AFF3CT; Yasuda et al. Table I |
| Puncturing | A | Yasuda et al. Table I; patterns of IEEE 802.11 and EN 300 421 |
| Turbo codes (LTE) | A | AFF3CT, K = 40 and 1024 |
| LDPC codes | A | AFF3CT, PEG and DVB-S2 short code; Hu et al. 2005 |
| Polar codes (SC, CA-SCL) | A | AFF3CT, SC and CA-SCL |
| 5G NR channel codes: LDPC and polar with rate matching | A | Sionna vectors bit for bit; TS 38.212 tables |
| Reed–Solomon codes | A | AFF3CT; Lin & Costello; ETSI EN 300 744, CCSDS 131.0-B |
| BCH codes | A | AFF3CT; generators of EN 302 307-1 Table 6a |
| CRC | A | CRC polynomials of TS 38.212 |
| Scrambling, interleaving | C | balanced sequences, invertible permutations |
| Finite-length bounds | A | Polyanskiy, Poor & Verdú 2010 Table I; the Shannon limits of BPSK |
| Distribution matching and probabilistic amplitude shaping | A | Schulte & Böcherer 2016 Fig. 2; Böcherer et al. 2015 Tables VI–X |

**Waveforms**

| Module | Level | Strongest evidence |
|---|---|---|
| OFDM, channel estimation and gain correction | A | Cimini 1985 Figs. 3–7; Coleri et al. 2002 Sec. VI-B (with an oversampled comb) |
| OFDM synchronisation (Schmidl–Cox) | A | Schmidl & Cox 1997 eqs. (19), (20), (28) and (43) |
| PAPR: distribution and reduction | B | Ochiai & Imai 2001 eqs. (11), (27) |
| Constant-envelope OFDM | A | Thompson et al. 2008 eq. (15); Guerreiro et al. 2016 Fig. 1; López Morales et al. 2022 Fig. 3; Pacheco & Hatzinakos 2006 Fig. 1 |
| Chirp spread spectrum (LoRa modulation) | A | Chiani & Elzanaty 2019 Table I; Afisiadis et al. 2020 Figs. 2, 7; Vangelista 2017 Fig. 1 |
| Continuous-phase modulation | A | Murota & Hirade 1981 Table I and Fig. 5; Sundberg 1986 Fig. 5; MSK of Proakis & Salehi |
| FBMC/OQAM | A | Bouhadda et al. 2014 Table 1, Figs. 7, 10, 11; Nissel & Rupp 2017 eq. (17) |
| UFMC | A | Vakilian et al. 2013 Fig. 3(b) |
| OTFS and the delay–Doppler channel | A | Raviteja et al. 2019 Theorem 1 and its example; Hadani et al. 2017 Fig. 5 qualitatively |

**MIMO**

| Module | Level | Strongest evidence |
|---|---|---|
| MIMO detection | A | Yang & Hanzo 2015 Table X; MRC closed forms |
| Message-passing detection of large MIMO (LAMA, OAMP) | A | Jeon et al. 2018 Table II; Ma & Ping 2017 eq. (42), Figs. 2–3 |
| Space-time block codes | A | Tarokh et al. 1999 eqs. (39)–(40); MRC closed forms |
| Capacity and precoding | A | Telatar 1999 Theorem 2, eqs. (13), (16)–(18), Figs. 1–4 |

**Wireless channels**

| Module | Level | Strongest evidence |
|---|---|---|
| AWGN | B | closed forms of the error rates |
| Flat fading and antenna correlation | A | Loyka 2001 (exponential correlation, Fig. 1); BPSK over Rayleigh fading; MRC |
| Tapped and clustered delay lines (TDL, CDL) | A | ITU-R M.1225 and 3GPP TR 38.901 tables, verbatim |

**Optical**

| Module | Level | Strongest evidence |
|---|---|---|
| Fibre: linear, split-step Fourier, Manakov, back-propagation | A | OptiCommPy outputs; Häger & Pfister 2021 Fig. 5 |
| GN and EGN models | A | Carena et al. 2014 Fig. 1(a); GNPy outputs |
| Raman amplification and ISRS | A | Bromage 2004 eqs. (3)–(5), Figs. 7–8; Hu et al. 2004 eq. (2); GNPy outputs; Semrau & Bayvel 2018 eq. (10) |
| EDFA and PMD | B | the accumulated ASE |
| Optical front ends (laser, modulators, photodiodes, coherent receiver) | A | OptiCommPy outputs |
| Indoor optical wireless | A | Barry et al. 1993 Table II; Komine & Nakagawa 2004 Figs. 3–9 |
| Free-space optics | A | Al-Habash et al. 2001 Figs. 3, 6–8 |

**Hardware impairments**

| Module | Level | Strongest evidence |
|---|---|---|
| Nonlinear amplifier driven by OFDM | A | Banelli & Cacopardi 2000 eqs. 7–31, Fig. 5; Bouhadda et al. 2014, Tables 2–3 |
| Power amplifiers, IQ imbalance and digital predistortion | A | Ding et al. 2004 Fig. 4 (memory-polynomial predistortion) |
| Phase noise and frequency offset | A | IEEE Std 1139-2022 Table B.2 (Allan variances of white FM) |
| DAC and ADC | A | Max 1960 Table II; Bennett 1948; the exact error of a Gaussian input; IEEE Std 1241; Walden 1999 |

**Receiver DSP**

| Module | Level | Strongest evidence |
|---|---|---|
| Filters and resampling | A | OptiCommPy outputs |
| Carrier recovery | A | OptiCommPy outputs |
| Clock recovery | A | OptiCommPy outputs |
| Frame synchronisation | A | Massey 1972 Tables I–II (optimum rule) |
| Adaptive equalisers | A | OptiCommPy outputs; Kikuchi's singularity |
| Chromatic dispersion compensation | A | Eghbali et al. 2014 Table I, Figs. 5 and 9 |
| Blind estimation: modulation recognition, M2M4 SNR, symbol rate | A | Pauluzzi & Beaulieu 2000 Figs. 9–10 (M2M4, CRB); exact cumulants of the alphabets |

**Metrics and analysis**

| Module | Level | Strongest evidence |
|---|---|---|
| Error rates, SNR and EVM | A | OptiCommPy outputs |
| Information rates and shaping gains | A | Böcherer et al. 2015 Tables I, III; OptiCommPy outputs |
| Receiver posteriors (`air` metrics) | A | Alvarado et al. 2018 Remarks 4–5, eq. (35) |
| Spectral metrics (PSD, ACLR, PAPR, CCDF) | C | a tone lands in its bin |

**Systems and standards**

| Module | Level | Strongest evidence |
|---|---|---|
| DVB-S2 | A | EN 302 307-1 Table 13 and the tables of the standard |
| 5G NR numerology, SSB, cell search and PDSCH DMRS | A | py3gpp outputs; Lin et al. 2019 |

**Simulation and specifications**

| Module | Level | Strongest evidence |
|---|---|---|
| Monte Carlo runner | C | thread independence, stopping rules |
| Specifications, verification and operating-point search | B | the search against the inverted closed forms |
| Recordings (SigMF, raw IQ, NPY) | A | sigmf-python, the reference implementation, both ways |

<!-- end of the generated table -->

## Operating points

A sweep answers "what is the BER at 12 dB?". A link budget asks the converse: which
SNR gives a BER of 10⁻³? The `search` section of a specification answers it directly,
for every combination of the other grid axes.

```yaml
search: {along: ebn0_db, range: [2, 20], metric: ber, target: 1.0e-3, tolerance: 0.02}
```

- The bracket is halved until it is narrower than the tolerance.
- Every evaluation runs the stopping rule of the specification on the same random
  streams. These common random numbers keep the comparison of two SNRs free of draw
  noise.
- The crossing is interpolated in log10 of the metric. Its confidence interval comes
  from the intervals of the two bracketing estimates.
- `comrs run` prints the operating points. `search_spec` returns them in Python, with
  every evaluation. Expectations can check them against inverted closed forms
  (`ebn0_for_ber`, `snr_for_ser`, …).

The QPSK, 16-QAM and 64-QAM points for a BER of 10⁻³ come out within 0.02 dB of the
theory, in about 10 evaluations each.

## Receiver posteriors

The information rates of the metrics `mi` and `gmi` use a genie: the least-squares
gain and the true noise variance. A real decoder uses the receiver's own beliefs,
its output `z` and the noise variance `σ̂²` it estimated. The `air` metrics measure
the rate of that decoder, `AIR = H(X) - E[-log2 q(x|z)]` with
`q(x|z) ∝ e^{-|z - x|²/σ̂²}`. This is the rate under mismatched decoding, a lower bound
on the information rate.

| Metric | Meaning |
|---|---|
| `air`, `air_bmd` | symbol-wise and bit-wise rates with the receiver's posterior |
| `air_opt`, `air_bmd_opt` | the same with the best scaling of the posterior, per block |
| `posterior_calibration` | `σ̂²/mean|z - x|²`: 1 when calibrated, below 1 when over-confident |

A receiver block declares its posterior: its output and the σ̂² it believes in.
After a classic receiver, `noise_estimate` declares one, from the decisions or from
the known symbols; a learned receiver added from another crate (see
[Blocks from other crates](https://github.com/com-rs/com-rs#blocks-from-other-crates))
declares its own.

## Measured data

The source of a link can be a recording instead of a simulation. The `capture` block
reads a SigMF file, a raw IQ file or an NPY array, and one period of the transmitted
symbols. Every trial takes a segment of the recording. `resample` handles a
non-integer number of samples per symbol, such as an 80 GS/s oscilloscope at 32 GBd.
`reference_sync` finds where the periodic reference starts, by circular correlation.

```yaml
link:
  - {type: capture, file: ../data/synthetic_16qam.sigmf-meta, symbol_rate: 32e9,
     reference: ../data/synthetic_16qam_reference.npy, segment: 16384}
  - {type: resample, sps: 2}
  - {type: cdc, length_km: 200}
  - {type: matched_filter, rolloff: 0.2}
  - {type: gardner, kp: 3e-3, ki: 1e-5}
  - decimate
  - foe
  - {type: bps, n: 32, b: 32}
  - reference_sync
```

`specs/capture_16qam.yaml` processes a synthetic recording written by
`cargo run --example make_capture`, which stands in for a real measurement: 16-bit
samples, a 300 MHz offset and a fractional delay. It finds 17.5 dB of SNR for 18 dB
nominal. Every receiver block and metric applies to recordings in the same way.

## Speed

`benchmarks/bench_opticommpy.py` times the same operations on the same inputs and
checks that the two implementations agree. The latest results are in
[`benchmarks/results.md`](https://github.com/com-rs/com-rs/blob/main/benchmarks/results.md),
measured on an Apple M6 (12 cores) with comrs on one thread unless stated.

| Operation | OptiCommPy | comrs | Speed-up |
|---|---:|---:|---:|
| split-step Fourier, 5 × 80 km, 800 steps, `f64` / `f32` | 1.27 s | 559 ms / 333 ms | 2.3 / 3.8 |
| Gardner clock recovery, 65 536 samples | 20.4 ms | 0.4 ms | 47 |
| adaptive equaliser (CMA, 15 taps), 32 768 symbols | 21.2 ms | 1.5 ms | 14 |
| BER, SER and SNR, 64-QAM, 10⁶ symbols | 64.1 ms | 7.0 ms | 9.1 |
| Viterbi–Viterbi, 32 768 symbols | 1.0 ms | 0.3 ms | 3.4 |
| FIR filter, blind phase search, DD-PLL | | | 1.7 to 1.9 |
| GMI by Monte Carlo, 16-QAM, 32 768 symbols | 6.8 ms | 7.2 ms | 0.9 |
| Monte Carlo BER campaign, 1 thread / 12 cores | 332 ms | 74 ms / 12 ms | 4.5 / 28 |

The GMI estimator is as fast as OptiCommPy's, not faster: both spend their time in the
same exponentials. The first version of the benchmark showed comrs slower on four
cases (the FIR filter, the blind phase search, the DD-PLL and the GMI); the fixes were
an overlap-add convolution, an O(1) slicer for square QAM and PAM, and allocation-free
LLRs.

## Blocks from other crates

A project built on com-rs (a learned receiver, a model that depends on unpublished
code) adds its own blocks to the declarative links without com-rs depending on it
([ADR 0006](https://github.com/com-rs/com-rs/blob/main/docs/adr/0006-external-blocks.md)).
`com_spec::blocks::register` adds a block to the registry: it is then listed by
`comrs blocks`, described by the JSON Schema, checked by `comrs check` and built like
the others; `com_spec::cli::main` is the command line as a function, so the crate ships
its own binary with its blocks.

```rust
use com_spec::blocks::{register, Extension};

fn main() -> std::process::ExitCode {
    register(Extension { info: my_block_info(), build: |params, label, ctx| build_my_block(params, label, ctx) })
        .expect("a block of that name exists already");
    com_spec::cli::main(std::env::args().skip(1))
}
```

## Examples

```bash
cargo run --release -p comrs --example ber_awgn
```

| Example | What it shows | Result (default seeds) |
|---|---|---|
| `ber_awgn` | BER of 4-, 16- and 64-QAM over AWGN against the theory, with `SkipAfter`, CSV and JSON | within 2.5 % of the theory wherever more than 10⁴ errors are counted, except 64-QAM below 4 dB, where the Gray formula of the theory is optimistic |
| `coherent_link` | 16-QAM at 32 GBd over 80 to 3000 km: lasers, coherent front end, dispersion compensation, Gardner clock recovery, fourth-power FOE, blind phase search, frame synchronisation | 0.2 to 1.1 dB of SNR penalty after the DSP, offset found within 0.3 MHz |
| `ssfm_dbp` | launch-power sweep over 10 × 80 km with EDFAs: dispersion compensation against back-propagation with 1, 4 and 20 steps per span | the optimum moves from 21.6 dB near −2 dBm to above 32 dB at 8 dBm |

## Python

```bash
pip install comrs                      # the wheel of PyPI
pip install "comrs[plot]"              # with matplotlib, for comrs.plot and Results.plot
```

From a clone of the repository (needs a Rust toolchain):

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin && maturin develop --release
```

The [documentation of the package](https://com-rs.github.io/com-rs/) (Sphinx,
PyData theme) is built from `docs/python`: six
tutorials (a first link, a classical single-carrier chain, a block written in Python
and checked against the native one, a campaign as a YAML file and the command line,
the profile of a chain block by block, the signals along a chain), a
user guide, a gallery of classical results (error rates over AWGN and fading,
information rates, pulse shaping, dispersion, launch-power sweep with back-propagation,
MIMO, PAPR and predistortion, channel coding, OFDM channel estimation and
synchronisation, chains) with a section that reproduces named figures of the
literature (Alamouti 1998 Fig. 4, Hu et al. 2005 Fig. 5, Thompson et al. 2008 Fig. 2,
Häger & Pfister 2021 Fig. 5 with the published points, MSK against BPSK), and the API
reference. `just docs-py` builds it in `docs/python/_build/html`, and CI publishes it
to GitHub Pages.

Signals are NumPy `complex128` arrays, bits `uint8`, indices `int64`. Every function that
draws takes an explicit `comrs.Rng`. Invalid parameters raise `ValueError`.

`comrs.Chain` builds a link from named blocks: the blocks of the declarative links
(`comrs.blocks`) and Python blocks, run once, traced block by block, or run as a
campaign on the parallel Rust runner, and converted to and from a specification:

```python
import comrs
from comrs import blocks as B

link = comrs.Chain(
    tx=B.Source(constellation="QPSK", n_symbols=4096),
    channel=B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")),
)
frames = link.trace({"ebn0_db": 6}, comrs.Rng(1))          # the frame after each block
res = link.run(grid={"ebn0_db": [0, 2, 4, 6]}, metrics=["ber"],
               stop={"min_errors": {"ber": 200}, "max_trials": 1000})
spec = link.to_spec()                                       # the YAML specification
```

The analysis afterwards: `run` returns `comrs.Results` (series, selections, a pandas
table, plots with the confidence intervals and a closed form, a JSON file with the
specification), `comrs.compare` gathers variants of a chain, probes measure metrics
inside the link during the campaign (`link.probe("channel", ["snr_db"])` reports `channel.snr_db`),
`link.profile()` tabulates the time of every block, a `Dump` block records the signal
at a point of the link,
and `comrs.plot` draws the constellation, the eye diagram, the time signal and the
spectrum of a frame, or of every stage of `link.trace()`:

```python
res = comrs.compare({"QPSK": link, "16-QAM": link.replace(tx=B.Source(constellation="16-QAM", n_symbols=4096))},
                    grid={"ebn0_db": [0, 4, 8]}, metrics=["ber"])
res.plot("ebn0_db", "ber", by="variant")
link.trace({"ebn0_db": 6}, comrs.Rng(2)).plot("constellation")
```

## Tests

```bash
cargo test                                   # every crate but the extension, doctests included
cargo test --release -p com-validation -- --ignored   # statistical validation (by hand)
pip install --group test && python -m pytest        # pip >= 25.1; the oracles are recorded: no OptiCommPy, no GNPy
pip install --group oracle && COMRS_RECORD_ORACLE=1 python -m pytest tests/test_oracle.py tests/test_gnpy_oracle.py   # re-record them
just check                                   # every check of CI
```

- The Rust tests cover the unit behaviour of every block. They also include end-to-end
  checks: Monte Carlo error rates against the theory, information rates against their
  bounds, split-step against the linear channel, accumulated ASE, and the nonlinear
  phase of a constant field. The documentation examples run as doctests.
- `tests/test_oracle.py` compares every deterministic block with OptiCommPy 0.11, to
  machine precision in most cases. This covers Gray mapping, metrics, pulses, filters,
  the linear channel, the split-step method, modulators, the hybrid and the receiver,
  FOE, Viterbi–Viterbi, BPS, DD-PLL and the Gardner loop.
- `tests/test_gnpy_oracle.py` compares the Raman amplification with GNPy 3.0: the power
  profiles of the channels and of a co-pump and two counter-pumps, and the spontaneous
  noise (GNPy's Euler steps converge to them, 0.004 dB at 5 m), and the NLI of the
  generalised GN model with counter-pumps (0.02 dB with a fine frequency grid in GNPy).
  At its default tolerances GNPy underestimates that NLI by about 1 dB.
- The outputs of both oracles are recorded fixtures in `tests/oracle/` (`<name>.json`,
  with the tool, its version, the versions of Python and NumPy and the date; the arrays
  of more than 64 values in `<name>.npz`), written by `tests/oracle.py`. The tests read
  them and skip when a file is missing; the inputs are generated by the tests from seeded
  generators. The tools are only needed to re-record, with the dependency group `oracle` and
  `COMRS_RECORD_ORACLE=1`: a deliberate act, whose commit message says why and names
  the version of the tool. Re-recording an unchanged oracle produces no diff.
- `tests/test_performance.py` reproduces the classical results of the textbooks (error
  rates of QAM, PSK and PAM, Rayleigh fading, information rates, Nyquist pulses,
  dispersion compensation, back-propagation), the curves of the documentation gallery.
- `tests/test_chain.py` checks the chains: blocks, trace, specifications, and the
  identity of the Rust and Python runners on the same chain.
- `tests/test_api.py` checks the docstring examples, the conversions and errors, the
  generator parity with Rust, and the Python runner.
- `validation/tests/` compares Monte Carlo simulations with closed forms, published
  figures and tables and recorded AFF3CT runs (above); its quick tests run with
  `cargo test`, the long ones (`#[ignore]`) only by hand, before a release.
- `crates/comrs/tests/specs.rs` runs one trial per condition of every specification.
  It also checks that a specification draws exactly what its hand-written scenario
  draws. `tests/test_spec.py` covers `run_spec`.

## Deviations from OptiCommPy

They are deliberate and tested as such:

- The declarative metrics compute MI and GMI with the least-squares channel gain.
  `monte_carlo_mi` and `monte_carlo_gmi` keep OptiCommPy's power normalisation.
- `evm` follows `calcEVM`, which normalises before the rotation, unlike `fastBERcalc`.

- `theory_ber` of BPSK is the exact `Q(√(2E_b/N_0))`. OptiCommPy's general PSK formula
  doubles it.
- `monte_carlo_mi` snaps the normalised transmitted symbols to the constellation, so
  the estimate never exceeds `log2 M`.
- `monte_carlo_mi` and `monte_carlo_gmi` return `log2 M` without noise instead of 0/0.
- `pulse_shape` uses a symmetric time grid. It equals OptiCommPy's for an even length.
  An odd length puts the taps on the sampling instants, while OptiCommPy's odd grid is
  shifted by about half a sample.
- The clock recovery computes in double precision. OptiCommPy stores its output in
  `complex64`.
- With shaped symbols, `theory_gmi` is `H(X) - Σ_k H(B_k|Y)`, the rate of bit-metric
  decoding and the limit of `monte_carlo_gmi` (and of OptiCommPy's `monteCarloGMI`);
  OptiCommPy's `theoryGMI` sums the bit-wise informations `Σ_k I(B_k;Y)`, which can
  exceed the entropy of the source. Both agree for uniform symbols.
- With a `seed`, OptiCommPy's `edfa` re-seeds its generator on every call, so that
  `manakovSSF` adds the same ASE to both polarisations and after every span. The
  comparisons of noisy links leave its seed unset; `ManakovSsfm` draws independent
  noise.
- OptiCommPy's `edc` limits the SNR of a linear, noiseless link to 23 dB; the oracle
  tests compensate dispersion with its linear fibre of opposite dispersion.

## Licence

MIT or Apache-2.0, at your option (see [`LICENSE-MIT`](https://github.com/com-rs/com-rs/blob/main/LICENSE-MIT) and
[`LICENSE-APACHE`](https://github.com/com-rs/com-rs/blob/main/LICENSE-APACHE)). The data of third parties shipped in the
crates (the Raman gain spectrum of GNPy, BSD-3-Clause) and the status of the recorded
oracle fixtures are listed in [`THIRD_PARTY.md`](https://github.com/com-rs/com-rs/blob/main/THIRD_PARTY.md).

To cite com-rs, use the metadata of [`CITATION.cff`](https://github.com/com-rs/com-rs/blob/main/CITATION.cff).
