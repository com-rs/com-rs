# Review of com-sim, com-spec — 30 October 2026

Read-only review; each finding was checked against the code by the reviewer. Paths and line numbers refer to commit 21af3f7.

## com-sim / com-spec review (read-only; each finding checked against the code)

**Largest files (non-test lines), and how to split them.** `io.rs` 970 (~800): split into `io/{datatype,raw,sigmf,npy}.rs`. `blocks/wireless.rs` 830 (~675): split into `wireless/{fading,ofdm,mimo}.rs`. `blocks/schema.rs` 738: split into `schema/{fact.rs (Fact, check_link, Checked), param.rs (ParamInfo, ParamType, take, mapping), macros.rs, json_schema.rs}`. `blocks/coding.rs` 716 (~560): split into `coding/{codes,fec,pas,map}.rs`. `verify.rs` 673 (~590): move the theory catalogue to `verify/theory.rs`. Functions over ~75 lines: `Metrics::record` (129), `search` (82), `Expectation::expected` (77), `Runner::run` (75).

### Findings (most important first)

1. **Redundancy · high · S.** A `Runner` is built from a `Spec` in three places, and the copies already differ:
 - `run.rs:35-55`;
 - `search.rs:196-203`, which leaves out `.config()`;
 - `bindings/com-py/src/sim.rs:867-884`, which ignores `spec.threads` and `RunOptions`.
 Fix: add `Spec::runner(&self, &RunOptions) -> Runner` (with or without loggers) and use it in all three places.

2. **Reusability/correctness · high · S.** `PerPol` does not forward `needs_training`/`train` (`per_pol.rs:22-29`). An extension block declared `per_polarization` that needs training is therefore never trained, and nothing reports it. The training path has no test anywhere: `link.rs:44-53` is the only caller, and `rg` finds no test that uses it. Fix: forward both methods, and add a trainable block to `tests/extension.rs`.

3. **Silent wrong result · medium · S.** `confidence` is never validated: not by `Spec` (`spec.rs:70,285`), not by `Runner::confidence` (`runner.rs:211`), and `z_score` clamps its input (`stats.rs:37-39`). A typo such as `confidence: 95` silently gives z ≈ 8.2. The only guard is in the bindings (`sim.rs:910-913`). Separately, `Stop::precision` always uses 0.95 (`stopping.rs:110`, called at `spec.rs:123`), whatever the spec's `confidence` says. Fix: return `Error::Config` in `Spec::from_json` and `Runner::run`, and pass `spec.confidence` to the precision targets.

4. **Consistency/redundancy · medium · M.** There are three report writers with three different schemas:
 - `JsonLogger` (`loggers.rs:256-300`) records the full `RunInfo`;
 - search `write_json` (`search.rs:310-367`) has no batch, threads, stop or wall time;
 - the CLI verify report (`cli.rs:182-191`, serde_json) has the version of com-spec and nothing else: no commit, no seed.

 The conversion `Value` → JSON is written three times: `loggers.rs:224`, `search.rs:318-330` and `params.rs:9`. The verify report is written by the CLI, so Python `verify_spec` cannot write it. Fix: make the run-info header and `value_json` public in `com_sim::loggers`, and move the verify report into `verify.rs`.

5. **Redundancy · medium · M.** Parameter groups are declared several times even though `[flatten]` exists:
 - the pulse group (`kind`/`rolloff`/`n_taps`) at `dsp.rs:110-113`, `dsp.rs:165-167` and `optical.rs:157-159`; their docs have already drifted;
 - `amplification` + `noise` at `fiber.rs:117-119,181` and `optical.rs:236-238,332`;
 - `probe` redeclares the metrics window (`analysis.rs:63-70`) and rebuilds `Metrics` through a `json!` round trip (`analysis.rs:91-94`);
 - `fec_encode`/`fec_decode` repeat `interleave`/`seed` (`coding.rs:325-339`), and the decoder builds the code a second time (PEG construction twice, `coding.rs:122-124`).

 Fix: `PulseSettings` and `AmplifierSettings` groups, and `MetricsSpec` written with `block_params!` and flattened into `ProbeSpec`.

6. **Consistency/SoC · medium · M.** Nested mappings bypass the "declared once" pipeline. These use hand-written serde structs:
 - `ExponentialSpec` (`wireless.rs:286`), `PilotSpec` (`:406`);
 - `FiberSpec`, `RamanSpec`, `PumpSpec`, `GainSpec` (`context.rs:81-139`).

 These parameters are typed `Json`: `source.rs:61,134,138`, `dsp.rs:269,271`, `optical.rs:156`, `wireless.rs:277,396`.

 As a result their keys are missing from `comrs blocks`, the JSON Schema (`{}`) and the Python classes, and their choices are case-sensitive (`context.rs:149`, `wireless.rs:427`). The spec-level `metrics.scale` is case-sensitive too (`metrics.rs:148`), against `lib.rs:65`. Fix: declare them with `block_params!` and expose them through `ParamType::variants`, as `Code` does (`coding.rs:229-237`).

7. **Redundancy · medium · S.** `FiberSpec` copies `FiberParams` field by field three times (`context.rs:36-44`, `81-94`, `180-193`), so a new fibre parameter needs four edits. Fix: a `block_params!` group whose defaults come from the `FiberParams` constants, as `LdpcSettings` does.

8. **Maintainability · medium · M.** `Metrics::record` (`metrics.rs:183-312`) matches metric names as strings at every trial and ends in `unreachable!`. The names live in four hand-kept lists (`:78`, `:99`, `:237-309`, and the inner matches at `:266` and `:298`). The `raw`/`tx` vectors are built twice (`:257-258`, `:284-285`), and `Scale::Power` hand-rolls `com_math::energy` (`:225-226`). Fix: an `enum Metric` (FromStr, NAMES) parsed in `from_json`, and helpers `window`, `scale`, `record_coded`.

9. **Maintainability · medium · M.** `TheorySpec` is one struct of 22 fields shared by 11 functions (`verify.rs:172-214`):
 - parameters a function does not use are accepted without a word (for example `branches` on `ber`);
 - the function names are retyped in the error message (`:394`);
 - `ber_rayleigh` compares `t.kind == "qam"` with case (`:387`) after `kind` was parsed without case (`:322`), so `kind: QAM` fails.

 Fix: `#[serde(tag = "function")] enum Theory`.

10. **Correctness · medium · S.** `gn_snr_db` uses `BuildContext::default()` (`verify.rs:225`), so the theory ignores the spec-level `fiber` defaults that every fibre block of the same spec merges (`context.rs:59-68`). Fix: pass the spec's context to `Expectation::check`.

11. **Reusability · medium · M.** Built-in and external builders have two different signatures: `BuildFn(&str, &str, &Json, &BuildContext) -> Option<Result<…>>` (`registry.rs:100`) against `Extension::build(Json, String, &BuildContext)` (`:138`). Each family matches on the name again after the registry has already found its entry (`:237-239`). The families use three idioms (`channel.rs:60-76`, `dsp.rs:86-104`, `wireless.rs:150-162`, `coding.rs:117-132`, `optical.rs:134-146`). Fix: one builder per `BlockInfo`, with the `Extension` signature, so that the built-ins become `Extension`s too. Consider `Arc<dyn Fn>`, so that a builder can capture state.

12. **Consistency · medium · S.** Public items return unnamed tuples:
  - `Spec::skip_below: Option<(String, f64, String)>` (`spec.rs:41`, destructured at `run.rs:46` and in the bindings at `sim.rs:876`);
  - `Check::measured: (f64, f64, f64)` (`verify.rs:277`, read as `.0`/`.1`/`.2` at `cli.rs:186`);
  - `io::read_samples` returns `(Vec<C64>, Option<f64>)` (`io.rs:771`).

  Fix: store a `SkipAfter` or a named struct, and use an `Interval { value, lo, hi }`.

13. **Consistency (audit section 5) · medium · S.** Some block parameters lack the unit in their name:
  - cfo `hz` and `omega` (`channel.rs:202-203`);
  - phase_noise `sigma` (radians, `:162`);
  - `tau1`/`tau2` (seconds, `dsp.rs:409-410`);
  - `max_nl_phase` (radians, `optical.rs:232`);
  - cdl `spacing` is in wavelengths (`wireless.rs:336`), while wdm has `spacing_hz`;
  - `coherence` (samples, `wireless.rs:230`).

  Fix: rename (`offset_hz`, `omega_rad`, `sigma_rad`, …) and keep the old names as aliases for one release.

14. **Consistency · low · S.** Bounds are checked by hand instead of declared: `coding.rs:255` (conv `k`), `:264` (rs `m` in 2..16), `:271` (turbo `k`), `wireless.rs:243` (correlations; the macro has no exclusive upper bound). These bounds are invisible to the schema, and the messages differ ("must be positive", "must be 2 to 16", "at least 8"). Fix: declare `>= 1` and `>= 2 <= 16`, and add `<` to the macro.

15. **Redundancy · low · S.** `profile_of` builds the whole link, training included, just to get its labels (`run.rs:95-98`), although `link_labels` exists (`link.rs:68`). `profile` searches the record linearly instead of calling `record.get` (`run.rs:76`).

16. **Redundancy · low · S.** The three fading blocks repeat the same preamble and postamble (`wireless.rs:252-269`, `316-327`, `371-390`). Fix: a `fade(s, |x, rng| (y, csi))` helper.

17. **Redundancy · low · S.** Small duplications:
  - `JsonLogger` rewrites `create_parent_dir` (`loggers.rs:293-297` against `:91-96`);
  - `Normalize` has two code paths for one operation (`dsp.rs:150-158`);
  - `Frame::mean_power` re-implements `com_math::power` (`frame.rs:255-259`);
  - the bounds text is written twice, with different symbols (`schema.rs:203-206` uses ≥, `:317-326` uses >=);
  - the per-object schema loop is written twice (`:365-382`, `:676-699`).

18. **Maintainability · low · S.** `Stop::MaxTrials` and `MinTrials` behave the same in `is_met`, `trial_limit` and `Display` (`stopping.rs:132,170,230`), so a printed rule does not say which one is the cap (see `spec.rs:361`). `Action::Skip` returned after a batch means "stop this condition" (`callbacks.rs:54`, `runner.rs:360`). Fix: print `min_trials(n)`/`max_trials(n)`, and add `Action::StopCondition`.

19. **Consistency · low · S.** `Grid::zip` panics on ragged rows (`grid.rs:250-257`). A repeated axis name silently overwrites the earlier one (`grid.rs:134-140`), which gives duplicated conditions. Fix: return a `Result`, and reject duplicate keys in `Spec::from_json`.

20. **Performance · low · S.** In `link_for`, every block is split twice (`link.rs:30-31` and `registry.rs:231`), and `registry()` clones every `BlockInfo` twice per condition (`link.rs:38`, `registry.rs:261`).

21. **Dead code / odd code · low · S.**
  - `ParamInfo::text` is public and used nowhere; it takes a magic `"required"` string (`schema.rs:174-191`).
  - `per_pol` has a useless inner block (`per_pol.rs:34-71`).
  - `part()` returns an empty vector, without a word, when the length does not divide (`per_pol.rs:14-20`).
  - The CLI checks "no specification given" only after the command has run (`cli.rs:265`).

22. **Docs · low · S.** Formulas in the rustdoc are plain text in backticks, against the LaTeX rule of CLAUDE.md: `stats.rs:3-13,35,143`, `metrics.rs:28-31`, `frame.rs:93`. ADR 0004 and the crate table of ARCHITECTURE name `serde_yaml`, but the code uses `serde_norway` (`spec.rs:199`).

23. **Consistency · low · S.** The wording of "unknown" errors varies: `registry.rs:235` ("…; the blocks are"), `coding.rs:291` ("…: the codes are"), `verify.rs:394` (a list in parentheses), `metrics.rs:146`. Fix: one helper that formats the message.

### What is good
- **Determinism is done right:** one stream per trial, records merged in trial order, `max_trials` met exactly whatever the batch, and dropped values counted and reported.
- **Parameters are declared once:** `block_params!`/`block_info!` give the parser, the docs, the schema and the Python classes from one declaration, and a test checks that what a block accepts matches what it documents.
- **Ports are checked:** `Fact` checks the links before they run (`check_link`) and debug builds check them while they run, with messages that name the block to add.
- **`verify` refuses vacuous passes:** its messages show how many conditions each step left.
- **The extension API (ADR 0006) is small**, and is covered by a doc test and an integration test.
