# Review of com-channel, com-systems, validation — 30 October 2026

Read-only review; each finding was checked against the code by the reviewer. Paths and line numbers refer to commit 21af3f7.

Findings (category · severity · fix · effort), most important first:

1. **Redundancy · high.** The Monte Carlo loop "until N errors" is written by hand 7 times: `closed_forms.rs:26-38`, `mimo.rs:34-48` and `61-74`, `tal2015.rs:21-36`, `converters.rs:182`, `message_passing.rs:232`, and `shaping.rs:411-452`. The last one re-implements com-sim's parallel runner with `thread::scope` and atomics, which breaks "parallelism lives in com-sim only". Meanwhile `mimo.rs:115` and `aff3ct.rs:290` already use `Runner` + `Stop::min_errors(..).or(Stop::max_trials(..))`. Fix: use `com_sim::Runner` (`scenario::from_fn`), or add a sequential `com_validation::until_errors(seed, min, max, |rng| (e, n))`. **M**

2. **Redundancy · high.** The frame error rate of a `BlockCode` over BPSK/AWGN is written three times: `tal2015.rs:21` `wer`, `nr_codes.rs:19` `bler`, `aff3ct.rs:132` `block`. All three also re-derive the noise variance `1/(2R·Eb/N0)` (`tal2015.rs:23`, `nr_codes.rs:21`, `aff3ct.rs:191, 274`), although `com_fec::block::bpsk_noise_variance` exists. Fix: add `com_validation::fer(code, ebn0_db, stop, rng)` on top of `bpsk_awgn_llr` and `bpsk_noise_variance`. **S**

3. **Redundancy · medium.** `validation/tests/specs.rs:12-17` re-implements `com_spec::verify::verify_spec` (run or search, then verify). Three of its four specs (proakis_awgn, information_rates, thompson2008_eq15) are also in `ci_specs` (`justfile:7`) and `VALIDATION_SPECS` (`ci.yml:15`), so CI runs them twice. Those two lists are kept equal only by a comment. Fix: call `verify_spec`, drop the specs already verified by CI from the test (or the reverse), and read one list from a file. **S**

4. **Redundancy (cards) · medium.** Two tests assert values that cards already hold:
 - `chang2002.rs:115-131` asserts the same Fig. 3/4 values as `cards/chang2002_fig3_dbpsk_floors.yaml` and `chang2002_fig4_floors.yaml`.
 - `cd_filters.rs:49-56` asserts the K values of `cards/eghbali2014_table1.yaml`.

 Fix: delete the inline asserts, or move these cases to `Card::check`. **S**

5. **Consistency (validation) · medium.** Only 17 of the 48 test files go through `Card`; the other 31 assert inline with tolerances in the code. File names follow the "author+year" rule unevenly: `bicm.rs` (Caire 1998), `cd_filters.rs` (Eghbali 2014), `nr.rs` (Lin 2019), `egn.rs` (Carena 2014). Test names follow references unevenly too (e.g. `converters.rs::sqnr_of_a_full_scale_sine`, `blind.rs::sample_cumulants_converge_whatever_the_snr`). Fix: rename the single-reference files, write cards for the published-figure tests, and say in ARCHITECTURE which tests may stay card-less (closed forms, invariants). **M**

6. **Redundancy · medium.** The "sweep a grid until the rate drops below a target, then `log_interp`" helper is written three times: `cimini1985.rs:104-113`, `bouhadda2014.rs:86-96` (named `crossing`, which shadows the lib's `crossing` with different semantics) and `fatadin2009.rs:117-126`. Fix: `com_validation::sweep_crossing(f, start, step, n, target)`. **S**

7. **Redundancy · medium.** Tests re-implement library helpers:

 | Test | Library helper |
 |---|---|
 | bit errors (`aff3ct.rs:127`, `css.rs:44`, `mimo.rs:43`, `nissel2017.rs:119,162`, `bouhadda2014.rs:67`, `raviteja2019.rs:127`, `converters.rs:190`) | `metrics::errors::bit_errors` / `symbol_errors` |
 | `fatadin2009.rs:60` | `source::bits` |
 | `blind.rs:34`, `eun1997.rs:42` | `source::symbols` |
 | `message_passing.rs:165` | `metrics::mse` |
 | BPSK `1-2b` (`closed_forms.rs:20`, `mimo.rs:65`) | `fec::block::bpsk` |
 | QPSK (`egn.rs:107`) | `Rng::qpsk` |
 | about 30 inline `complex_gaussian()*σ` noise draws | `awgn::add_noise` / `Rng::complex_noise` |

 **M**

8. **Redundancy · low.** Averaged periodograms are hand-rolled in `vakilian2013.rs:25-44` and `ce_ofdm.rs:124-130`. Fix: `com_dsp::spectrum::periodogram` with `fft::fftfreq`/`fftshift` (`egn.rs:113` also hand-rolls fftshift). **S**

9. **Consistency (units, audit §5, still open) · medium.** Units are missing from these names:
 - `Cdl.carrier_frequency` (`cdl/mod.rs:143`, hidden default 3.5 GHz at `:210`), `Cdl.velocity` (`:141`), `delay_spread` (`:165, :229`, `tables::Profile::pdp`), `Cdl::max_doppler()` (`:255`), `Ray.doppler` (`:394`), `WdmSpectrum.freqs` (`gn.rs:108`).
 - The spec blocks and Python already use `delay_spread_s`, `fc_hz`, `velocity_mps` (`com-spec/src/blocks/wireless.rs:334-338`).
 - Inside multipath itself, one Doppler is named `Tdl.doppler_hz` (`multipath.rs:440`) and the other `DelayDopplerModel.max_doppler_hz` (`:682`), next to `Profile::Exponential{rms_delay_s}`.

 Fix: rename to `delay_spread_s`, `fc_hz`, `velocity_mps`, `doppler_hz`, `max_doppler_hz`, `freqs_hz`. **M**

10. **Reusability · medium.** The channel constructors and verbs have no common shape:
  - `Tdl::new(taps, doppler_hz, sample_rate, n_rx, n_tx)`, 5 positional arguments plus public fields.
  - `Cdl::new` takes 5 and `Cdl::custom` 7, and Doppler is set through fields after construction.
  - `DelayDopplerModel::new(taps, max_doppler_hz, sample_rate)` is SISO only.
  - `Mimo` is built from a struct.
  - `draw` returns `Taps` (`multipath.rs:503`), `Result<Taps>` (`cdl/mod.rs:348`), `DelayDoppler` without a length (`multipath.rs:704`), or `Result<CMat>` (`fading.rs:127`). `Fso::gain` is a draw under another name.

  Fix: one config-struct shape (`TdlParams{..}`, `CdlParams{..}`) and a `Fading` trait (`draw(n, rng) -> Result<Taps>`, `n_tx`, `sample_rate`). `too_many_arguments` is allowed workspace-wide, so clippy will not flag this. **L**

11. **Redundancy · medium.** The bodies of `fade` are copied (`multipath.rs:555`, `cdl/mod.rs:358`). The `Propagate` impl `out.assign(&self.fade(x, rng)?.0)` is copied four times (`fading.rs:93, 201`, `cdl/mod.rs:370`, `multipath.rs:732`). It builds a temporary `Signal` and copies it, while `DelayDopplerModel` uses `set_streams` (`:724`). Fix: a generic `fade_with(taps)` once the trait of #10 exists, and fill `out` with `set_streams`. **S**

12. **Redundancy · medium.** The optical span loop (propagate, then amplify every span, with or without Raman) is written five times: `fiber.rs:329` (propagate), `:357` (propagate_raman), `:418` (f32), `manakov.rs:86` (Raman and EDFA branches), `pmd.rs:246`. Fix: one span driver in `optical::fiber` taking a closure for the propagation step. **M**

13. **Consistency · medium.**
  - There are two Manakov propagators: `manakov::ManakovSsfm` (`propagate -> Result<usize>`, chosen `Amplification`, Raman) and `pmd::Manakov` (`-> Result<()>`, `Amplification::Edfa` hard-coded at `pmd.rs:267-268`, no Raman).
  - The other optical signatures disagree: `Ssfm::propagate(&[C64], fs, Option<&mut Rng>) -> Vec`, `Edfa::apply(&[C64], fs, &mut Rng) -> Result<Vec>`, `PmdLink::apply(&mut Vec, &mut Vec, fs)` with no `Result`.

  Fix: give `pmd::Manakov` an `amplification` field, name it `ManakovPmd`, and pick one in-place, `Result`-returning shape. **M**

14. **Maintainability · medium.**
  - `GnModel::nli_with` (`gn.rs:331-456`) is 125 lines, nested 5 deep, with the EGN `mu` closure inline. Fix: extract `gn_integral` and `egn_mu`.
  - `gn.rs:442` hand-rolls the Dirichlet kernel `sin(Nx)/sin x`, while the tests of the same file (`:687`) and `validation/tests/egn.rs:93` use `com_math::diric`. Fix: use `diric`.
  - Before switching to `diric`, check that results are unchanged under ADR 0005. **M**

15. **Reusability · medium.** `GnModel::eta`/`optimal_power(freqs, symbol_rate, rolloff, ch)` (`gn.rs:479, 491`) take loose parameters while `snr(&WdmSpectrum, ch)` takes the spectrum. They also use an unnamed 1 mW reference (`:481`). Fix: take `&WdmSpectrum` and name the constant. **S**

16. **Separation / redundancy · low.** The spherical-to-Cartesian conversion is written twice in com-channel, with opposite conventions: `owc/indoor.rs:95` `direction(elevation_rad, azimuth_rad)` versus `cdl/array.rs:106` `unit_vector(zenith°, azimuth°)`. The CDL angles are in degrees without a suffix. Fix: `com_math::geometry::from_spherical`. **S**

17. **Maintainability · low.** `Mimo::draw` re-validates antennas, correlation and LOS on every draw (`fading.rs:127-139`). `apply` calls it once per sample when `coherence = 1` (`:184`), which breaks "hot loops never re-validate". Fix: validate once in `apply`/`fade` and add a private `draw_unchecked`. **S**

18. **Redundancy · low.**
  - Phase draws `rng.uniform()*2π-π`: 7 times (`multipath.rs:524, 528, 710`, `cdl/mod.rs:294-297`). Fix: `Rng::phase()`.
  - `Edfa::apply` (`edfa.rs:98-103`) re-implements `awgn::add_noise`.
  - `c/(nm·1e-9)` conversions (`raman/mod.rs:140`, `com-spec/blocks/context.rs:155`, `bromage2004.rs:40, 56, 83`). Fix: one `com_math` helper.

  **S**

19. **Consistency · low.** Validation re-implements a channel formula: `com_validation::max_doppler_hz` (`validation/src/lib.rs:53`) is `Cdl::max_doppler`'s formula. Fix: move it to `com_channel::multipath` and use it in `Cdl` and the tests. **S**

20. **Docs contradicting code · low.**
  - `com-systems/src/lib.rs:3` says "layers 0 to 3", but the guide and the `Cargo.toml` say 0–2.
  - `com-systems/src/lib.rs:11` says the guide places NR and 400ZR there. The guide does not mention 400ZR, and NR lives in `com-waveform`/`com-fec`.
  - The facade table (`comrs/src/lib.rs:14, 18`) leaves out FBMC/UFMC/OTFS/CSS and the blind and stability metrics.
  - The `comrs/Cargo.toml` feature comment says `com::spec`.
  - The `# Errors` doc of `Tdl::new` (`multipath.rs:452`) omits the negative-Doppler error.

  **S**

21. **Maintainability · low.** DVB-S2:
  - Magic numbers 90/80/36/16 (`dvbs2/mod.rs:331-584`). Fix: name them `PLHEADER_LEN`, `BBHEADER_LEN`, `PILOT_BLOCK`, `SLOTS_PER_PILOT`.
  - `data_positions()` is rebuilt per frame (`:526`, called at `:548, 572`). Fix: cache it like `scrambler`.
  - `Dvbs2::new(.., n)` and `scrambling_sequence(n, len)` use a single-letter name for the scrambling code. Fix: `gold_code`.
  - Booleans everywhere: `plheader_bits(u8, bool, bool)`, `decode_plheader -> (u8, bool, bool)`. Fix: a `PlsCode` struct.

  **S**

22. **Consistency · low.** `Modcod` has public fields, so `number()` is fallible and `modulate` calls `number()?` (`mod.rs:114-147, 554`). Fix: private fields with getters (the `Fbmc` rule). **S**

23. **Reusability · low.**
  - `Pas::llr` takes only per-symbol variances (`pas.rs:261`), while `decode` has a scalar form and `decode_with` a slice form. As a result, `shaping.rs:437` builds `vec![1e-6; n]` just to recover the sent codeword. Fix: `llr(y, f64)` + `llr_with`, plus `encode_codeword` returning the codeword.
  - `Pas::with_bit_mapper` takes 5 positional arguments.

  **S**

24. **Redundancy · low.** `card.rs` repeats the input-matching predicate three times (`:167`, `:183-186`, `:199-202`). Fix: `fn matches(&Case, keys, exact_len)`. **S**

25. **Maintainability · low.**
  - Long multi-reference test files: `shaping.rs` (640 lines, 3 papers), `aff3ct.rs` (552), `message_passing.rs` (399). Fix: split `shaping.rs` per paper.
  - Test helpers with 7-8 positional arguments and booleans: `bouhadda2014.rs:35`, `cimini1985.rs:51`. Fix: small config structs.
  - The hand-written regula falsi in `shaping.rs:172-211` is a numerical method. Fix: move it to `com_math::search`.

  **M**

26. **Redundancy · low.** `comrs/tests/links.rs:136` rebuilds `fiber.span_edfa()` by hand, and `:121` recomputes `Amplification::Ideal.field_gain`. **S**

**What is good:**
- No layer violation in the slice.
- Every `#[allow]` has a reason.
- Every remaining public positional constructor validates and returns `Result`.
- The audit fixes hold: the sample-rate check in `same_rate` is shared, rate fields are private, `Ssfm` uses ⌈L/h⌉ steps, `ase_density` is the one ASE kernel.
- Rustdoc is exemplary: LaTeX models, tables cited verbatim (TR 38.901, M.1225), `literature` tests.
- Carded validation tests are clean: 0 inline asserts, the tolerances and their reasons live in YAML, card tiers match `#[ignore]`, and `Card` fails on unmeasured cases.
- Module docs of every test file cite the reference with its file in `references/`.
