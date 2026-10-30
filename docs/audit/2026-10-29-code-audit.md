# Code audit — 29 October 2026

Seven independent reviews, read-only, of the whole repository (about 85 000 lines of Rust
and Python), each finding checked against the code by its reviewer; three were reproduced
by running the code (`comrs verify`, `max_trials`, case of accepted values). About 160
findings in all; this report groups them by theme and by priority, then lists them by slice.

**Verdict.** The code is well above the average of scientific code bases: no `unsafe`, no
layer violation, formulas in the rustdoc with their references, tests named after the
papers, explicit generators, shared conversion helpers in the bindings, Python docstrings
enforced by tests. Its weaknesses are of four kinds, in order of importance:

1. **Silent wrong results at the edges of the API** — a validation that checks nothing
   passes, a channel trusts a sample rate the signal does not have, public fields
   bypass the constructors, lengths that differ are truncated without a word.
2. **Panics and endless loops on invalid input** where a `Result` is expected.
3. **Performance traps** — a few algorithms of the wrong complexity (polar decoding,
   μ-law companding, the OTFS channel), and a general contract (`_into` allocates nothing)
   that the implementations do not keep.
4. **Repetitions and drift** — the same kernel or the same parameter list written in
   several places, which have started to disagree; API conventions (argument order,
   units in names, keyword-only, scalar-or-array) applied unevenly.

## 1. Silent wrong results (fix first)

| # | Where | Problem |
|---|---|---|
| 1 | `com-spec/src/verify.rs:347-365` | an expectation that matches no condition (typo in `where`, curve outside the grid) gives 0 checks, which pass: `comrs verify` exits 0 (reproduced) |
| 2 | `com-spec/src/spec.rs:104-129`, `com-sim/src/stopping.rs:124` | a stopping target on a metric that is not recorded is never met: the run goes to `max_trials` silently (reproduced); `min_errors` on a mean counts values |
| 3 | `com-channel/src/multipath.rs:435`, `cdl/mod.rs:345` | `Tdl`, `Cdl` and the delay–Doppler channel use the sample rate given at construction, never the one of the `Signal`: an oversampled signal gets wrong delays and Doppler |
| 4 | `com-channel/src/multipath.rs:331`, `cdl/mod.rs:129` | public `sample_rate`/`delay_spread` fields out of sync with the cached delays |
| 5 | `com-channel/src/optical/fiber.rs:297` | `Ssfm` simulates ⌊L/h⌋·h km per span while the EDFA restores α·L: +0.4 dB per span with 80 km and h = 3 km (4 dB over 10 spans) |
| 6 | `com-fec/src/polar.rs:147-250` | public `crc` and `with_crc` after `with_positions` break the code silently (the CRC is dropped, positions overlap) |
| 7 | `com-fec/src/rs.rs:178` | `ReedSolomon::k()` counts symbols, `BlockCode::k()` bits |
| 8 | `com-mapping/src/constellation.rs:335` | `normalized_under` with priors that do not sum to one scales wrongly |
| 9 | `com-core/src/signal.rs:252`, `com-math/src/linalg.rs:65`, `fft.rs:188`, `estimation.rs:67`, `carrier.rs:60`, `metrics::bit_errors` | lengths that differ are truncated by `zip` (stale samples left in a reused buffer for `try_map_streams`) |
| 10 | `python/comrs/spec.py:32-73` | `spec.load` and `$var` substitution re-implemented in Python, already different from Rust: a misspelt path returns the string, no `base_dir`, no `${a}_${b}` interpolation |
| 11 | `tests/test_cards.py:59` | the regex meant to catch numbers written as strings never matches (doubled backslashes) |
| 12 | `tools/literature_table.py:51`, `maturity_table.py:106` | "the cited test exists" is a substring test (a comment or a longer name passes) |
| 13 | `python/comrs/results.py:316` | `_same` is absolute below 1: `where(noise_var=…)` mixes 1e-12 and 2e-12 |
| 14 | `com-sim/src/runner.rs:324` | `max_trials: 1000` with batches of 32 runs 1024 trials (the guide says the batch does not matter) |
| 15 | `com-sim/src/scenario.rs:84-118` | `Record` drops values of the wrong kind or non-finite without a word (biases the mean of `snr_db`) |
| 16 | `com-spec/src/blocks/*` (schema.rs:283) | accepted values are checked without case, parsed with case: `check` passes, `run` fails (reproduced) |
| 17 | `com-sim/build.rs:21` | the commit recorded in every report can be stale (unstaged edits, a new commit) |

## 2. Panics and endless loops on invalid input

`ManakovSplitStep` with a zero or negative step loops forever (`nlse.rs:357`);
`ln_bessel_k(ν, ∞)` loops (`lib.rs:382`); `gf2m::Field::new` with a wrong polynomial
panics (`gf2m.rs:84`); `Profile::Exponential` with a zero resolution aborts on memory
(`multipath.rs:162`); `WdmSpectrum` with a negative rate (`gn.rs:320`);
`FormatMoments::of` (`gn.rs:183`); `Mimo::apply` on unequal streams (`fading.rs:187`);
`bbframe` underflow (`dvbs2/mod.rs:429`); `differential_detect` with L ≥ 3 and one symbol
(`cpm/detection.rs:207`); `min_distance2(0)` (`:240`); `Fbmc` with a large negative
offset (`fbmc/mod.rs:209`); `fourth_power_foe` on empty input (`carrier.rs:88`);
`gray_code(n ≥ 40)` from Python aborts the process (`comm.rs:300`); `max_time_s: .inf`
(`spec.rs:108`); `untar` and NPY reading on corrupt files (`io.rs:402, 647`).

## 3. Performance

| Where | Problem | Order of the gain |
|---|---|---|
| `com-fec/src/polar.rs:119-325` | SC and SCL compute every leaf from the channel: $O(L n^{2} \log n)$ instead of $O(L n \log n)$ (Arıkan 2009; Tal & Vardy 2015, Sec. IV); 7.6 ms per frame for SC at n = 2048, 250 ms for L = 32 | ×1000 at n = 1024 |
| `com-waveform/src/papr.rs:478-503` | μ-law companding computes a 6000-point integral per sample | ×6000 |
| `com-waveform/src/otfs.rs:211` | effective channel in $O((MN)^{3})$, separable in $O((MN)^{2}\log N)$ | ×1000 at 32 × 32 |
| `com-math/src/fft.rs:76` | every f64 FFT allocates its scratch (rustfft `process`) | split-step, OFDM |
| the `_into` methods (`block.rs:83`, `demapper.rs:165`, `detection.rs:536`, `stbc.rs:293`, every `Propagate`/`Impair`/`Modulate`) | they build a new container and replace `out`: the guide's "hot loops allocate nothing" is not kept; `BlockCode` (`&self`, returns a `Vec`) cannot keep a workspace | needs a design decision (ADR) |
| `com-fec`: LDPC (`ldpc.rs:561`), Viterbi/BCJR (`conv.rs:168`), turbo | allocations per iteration and per step, survivors of 16 bytes per state | constant factor |
| `com-mimo`: `receiver.rs:211`, `detection.rs:98, 121, 476` | detection run twice for the LLRs, QR redone per column with a fixed H, one allocation per ML hypothesis, K-best clones | ×2 to ×n |
| `com-dsp/src/interpolation.rs:238`, `estimation.rs:80`, `com-waveform/src/ofdm/estimation.rs:203` | LMMSE and band-limited interpolation re-solve the same system per symbol | ×symbols |
| `com-channel/src/multipath.rs:395` | `Taps` dense per sample and antenna pair even when static (2.3 GB for EVA 4×4 over 10⁶ samples) | memory |
| `com-waveform/src/css/theory.rs:236`, `com-channel/src/optical/gn.rs:346`, `com-hardware/src/converter.rs:172`, `com-metrics/src/information.rs:290` | quadratic or worse in their parameters (CSS spectrum, GN integral, ADC MSE at 24 bits, `posterior_rates` storing N×M) | |

## 4. Repetitions (DRY)

- **Kernels written twice or more**: the memory polynomial (`com-hardware/amplifier.rs:187`
  and `com-dsp/compensation/dpd.rs:99`, which builds an N×KM matrix); the symbol-posterior
  log-sum-exp (four times in `com-metrics/information.rs` and `blind.rs`); the EDFA ASE
  (`edfa.rs:50`, `gn.rs:262`, `raman/mod.rs:386`); the PAS demapper
  (`com-systems/pas.rs:238` against `com-mapping/demapper.rs:100`); NR modulation
  against `Ofdm::to_time` (`ofdm/nr.rs:176`); `centred_bin`, `fftshift` and frequency
  shifts hand-rolled (`ofdm/mod.rs:118`, `ufmc.rs`, `cd.rs:114`, `spectrum.rs:99`,
  `nonlinear.rs:229`, `optical.rs:83, 494, 511`); GF(2) elimination (`ldpc.rs:265`);
  `Vec3` helpers (`owc/indoor.rs`, `cdl/array.rs`); the PMD split-step in com-channel
  instead of `com_math::nlse`; the BPSK LLR test helper three times; `snr_db`/`evm`/`mse`
  of com-spec against com-metrics (different edge cases).
- **Lists maintained by hand that have drifted**: the detector names of com-spec (no
  `zf_sic`), parameter lists copied between blocks (`DetectorSpec`/`OfdmRxSpec`,
  `PasSpec`/`LdpcSpec`), the four lists of Python names, the theme list in three tools.
- **Logic in the wrong layer**: the trial recording of `Link::trial` copied twice in the
  bindings, with three Python shapes for one estimate; the quantiser, the SLM CCDF,
  guards and option-name parsing in the bindings; the equaliser diagnostic in com-spec.
- **Python**: `tools/` (extension loading, formatting, write-or-check duplicated), the
  MRC closed form four times (not bound), the repository-root lookup seven times in the
  gallery, 158 colour literals, the oracle helpers twice.
- **Validation tests**: `golden` against `com_math::golden_max`, the BER-crossing
  interpolation twice, the speed-to-Doppler conversion three times.

## 5. Consistency of the API

- **Verb traits**: `Modulate`/`Demodulate` for OFDM, UFMC and OTFS only, not for CE-OFDM,
  FBMC, CSS and CPM; no shared soft-decoding trait (`decode_llr`, `bcjr`, `decode_list`
  are three different shapes).
- **Units in names** (the guide's rule 5): `Cdl.carrier_frequency`, `delay_spread`,
  `Ray.doppler`, `WdmSpectrum.freqs`, `ddpll(ts, kv, …)`, `low_pass_fir(fc, fs)`;
  positional constructors of 5 to 7 arguments.
- **Python**: `NrLdpc(k, n)` and `NrPolar(k, n)` against `PolarCode(n, k)`; `draw(rng, n)`
  against `draw(n, rng)`; `(n, n_rx)` against `(n_tx, n)` for antennas; tuning
  parameters not keyword-only; `theory_*` closed forms that do not take arrays or lack
  the prefix; return types that depend on the arguments (`Ofdm.modulate`, `papr_db`);
  22 classes without `__repr__`, none picklable.

## 6. Documentation and tests

Docs that contradict the code (DVB-S2 errors documented as `Shape`, `gn.rs` 0.15 dB
against 0.2 dB tested, the Hadani factor 10 against 5 tested, `traits.rs:4`,
`DualPol::from_signal`, `schema.rs:41`); the guide's layout and `Propagate` list miss the
new modules; CLAUDE.md omits the `gen_stubs`/`gen_blocks` checks; vacuous or weak tests
(`bch.rs:263`, `loyka2001.rs:58`, `test_api.py` `zip(strict=False)`, `assert comrs`);
`erfc` accurate to 1.2·10⁻⁷ only while `erf` reaches 10⁻¹⁵; `py.typed` without a type
checker in CI (fixed today: the stubs of the OWC methods said `-> None`).

## 7. Recommended order

1. **Correctness** — section 1 and section 2: small, local fixes with a regression test
   each (about one day).
2. **The three complexity traps** — polar SC/SCL (rewrite on per-level LLR and
   partial-sum arrays with lazy copy), μ-law, OTFS; plus the FFT scratch.
3. **The `_into` contract** — decide in an ADR: either keep it and add workspaces
   (`decode_block_into(&mut self, …)`, scratch in the plan cache), or relax the guide.
4. **DRY** — one kernel per method (memory polynomial, posterior, ASE, PAS demapper,
   NR on `Ofdm`), single sources for the lists (detector names, Python names, themes),
   the bindings and Python calling Rust (`Link::trial`, `spec.load`, substitution).
5. **API consistency** — before the public release of 0.2.0, since it breaks Python
   signatures: argument orders, keyword-only, units in names, verb traits, `__repr__`.

## Appendix — the seven slices

| Slice | Lines | Findings | Overall assessment of the reviewer |
|---|---|---|---|
| com-core, com-math, com-dsp, com-metrics | 8 600 | 24 | well above average for numerical Rust; robustness at the edges, the FFT allocation, `erfc`, the posterior kernel written four times |
| com-fec, com-mapping, com-mimo | 11 400 | 25 | well documented and referenced; performance of the decoders (polar quadratic, allocations everywhere) and the `BlockCode` API |
| com-waveform, com-hardware | 8 900 | 24 | clean and layered; two high performance traps (μ-law, OTFS), uneven verb traits |
| com-channel, com-systems, comrs, validation | 13 700 | 21 | carefully validated; silent physics errors at the API edges (sample rate, span truncation), dense `Taps` |
| com-sim, com-spec | 11 700 | 22 | com-sim correct on determinism; com-spec repeats parameter lists and metrics, `verify` can pass with nothing checked |
| bindings/com-py | 13 000 | 22 | disciplined (helpers, error mapping, GIL release); API surface uneven, logic leaked from the crates |
| python/comrs, tests, tools, gallery | 16 000 | 22 | well above average; logic copied from Rust has diverged, guard checks that check nothing, duplicated tools |

## Status — 30 October 2026

Sections 1 and 2 are fixed, each item with a regression test, except two that stay open:
a nested `Chain.from_spec` loses its `base_dir` (the outer chain flattens it), and the
Python bindings do not expose the drop counts of `Record` yet. Choices made on the way:

- `Tdl`, `Cdl` and the delay–Doppler channel return `Error::Config` when the signal has
  another sample rate (their delays are rounded to samples at construction); their
  rate and delay-spread fields are private, with getters.
- `Ssfm` divides each span into $\lceil L/h \rceil$ equal steps: unchanged results when
  $h$ divides $L$, the whole span otherwise; `dbp` stays its exact inverse.
- `ReedSolomon::n()`/`k()` count bits (as `BlockCode`); the symbol counts are
  `n_symbols()`/`k_symbols()`. `PolarCode::with_crc` refuses the fixed positions of a
  table.
- Infallible kernels (`matvec`, `fft::filter`, `map_streams`, `energy_with`,
  `frequency_correlation`) panic on a length mismatch, documented; the fallible ones
  (`ls`, `derotate`, `bit_errors`, `symbol_errors`, `fourth_power_foe`, the Manakov
  solver) return `Result`.
- `max_trials` is met exactly whatever the batch; a stopping target on a metric that is
  not recorded, `min_errors` on a mean, an expectation that checks nothing and an empty
  `expect` list are errors; values dropped by `Record` are counted in the JSON report.
- `spec.load` and `$var` substitution of Python call the Rust loader; `gray_code`
  refuses more than 24 bits; `Stop.max_time` refuses an infinite or negative time.

Section 3 is fixed (30 October 2026), every change measured and checked against the
former computation (bit-identical, or within a stated tolerance):

| Item | Before | After |
|---|---|---|
| Polar SC, n = 2048 / SCL L = 32 (Tal & Vardy, Sec. IV, lazy copy) | 7.66 / 222 ms per frame | 0.08 / 2.4 ms |
| `validation/tests/tal2015.rs` | 1072 s | 10 s (130 s with the L = 32 points restored) |
| μ-law companding (64-point Gauss–Legendre once per block; the former Simpson rule was off by up to 2.6·10⁻⁴) | 4.15 s | 4–5 ms |
| OTFS `transform_channel`, M = N = 32 | 561 ms | 15–18 ms |
| CSS `occupied_bandwidth`, SF 12 | 46.4 s | 0.12 s |
| OFDM LMMSE estimate (factored once per pilot set) | 110 ms | 8.8 ms |
| MIMO: ML with LLRs / K-best with LLRs / SIC blocks | 70 / 654 / 13.9 ms | 15 / 85 / 0.6 ms |
| Static `Taps`, EVA 4×4 over 10⁶ samples | 2.3 GB | 2.3 kB |
| ADC MSE at 24 bits (closed-form granular sum) | 140 ms | 0.4 µs |
| EGN, 50 spans (half grid by symmetry) | 742 ms | 265 ms |
| LDPC / Viterbi / BCJR / turbo (no allocation per iteration or step) | | ×1.2 to ×2 |
| FFT scratch kept with the plans | | 2–6 % |

The `_into` contract is decided by ADR 0011: the output is resized and filled in place,
scratch may allocate unless a profile shows it; `BlockCode` has slice methods
(`encode_block_into`, `decode_block_into`) and every `_into` method of the workspace now
reuses its output. Sections 4 and 6 are fixed (30 October 2026); section 5 (API consistency) is not
started. One kernel per method now: the memory polynomial (`com_dsp::compensation::dpd`),
the EDFA ASE (`com_channel::optical::edfa::ase_density`), the symbol posterior
(`com-metrics`, private), the PAS demapper on `com_mapping::demapper`, NR on
`Ofdm::to_time`, `centred_bin`/`fftshift`/`freq_shift` everywhere, GF(2) elimination on
`gf2::Matrix`, `com_math::geometry`, the Manakov rotation, the quantiser of a signal
(`Quantizer::for_signal`); the detector names, the parameter groups of the blocks
(`DetectorSpec`, `LdpcSettings`), the theme list and the palette (`comrs.plot`) have one
source; the trial recording, the metrics and the guards of the bindings call the
crates; the tools share `tools/_common.py`. `erfc` is accurate to 8·10⁻¹⁶ (Cody 1969):
the closed forms move by at most 1.1·10⁻⁷ relative, no verdict changes. The guide, the
drifted docs and the weak tests are fixed, and mypy checks the stubs and the package
in CI. Every result of the benchmark (92 cases), of the validation cards and of the CI
specifications is unchanged, apart from `posterior_rates` (10⁻¹⁴), the PAS LLRs
(10⁻¹²) and the closed forms of `erfc`.
