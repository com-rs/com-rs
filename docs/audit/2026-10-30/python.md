# Review of Python package, tests, tools and benchmarks — 30 October 2026

Read-only review; each finding was checked against the code by the reviewer. Paths and line numbers refer to commit 21af3f7.

Findings, most important first. Format: category · severity · effort. I checked each one in the code. I could not run the tests (`.venv/bin/python` points to a missing interpreter), so I did not measure test durations.

1. **Redundancy · high · S.** The Rayleigh BPSK closed form `0.5*(1-sqrt(g/(1+g)))` is written by hand in four places: `tests/test_performance.py:109`, `:568`, `docs/python/examples/foundations/plot_04_rayleigh.py:53` and `mimo/plot_01_mimo_detection.py:60`. The same test file already calls `comrs.theory_ber_rayleigh_mrc(..., 1)` at `:711`. Fix: call `theory_ber_rayleigh_mrc(eb, 1)`, which takes arrays.

2. **Redundancy · high · M.** The coded BPSK-over-AWGN trial is copied five times. Each copy re-derives σ², the ±1 mapping and the LLR `2y/σ²`:
 - `tests/test_performance.py:468` `coded_errors` is identical to `examples/coding/plot_01_channel_coding.py:28` `simulate`;
 - `coding/plot_02_hu2005_fig5.py:44` and `coding/plot_03_nr_codes.py:54`;
 - `tests/test_fec_codes.py:321`, which also takes uncoded BPSK from `scipy.special.erfc` (`:331`).
 Fix: one helper using `Constellation("psk",2).llr` and `awgn_ebn0`, or the native chain `FecEncode/Map/AwgnEbn0/Demap/FecDecode`; use `theory_ber(2, 4.5, "psk")` instead of `erfc`.

3. **Redundancy · medium · M.** The gallery bypasses `Results` and reads `r["metrics"][m]["value"/"lo"/"hi"]` by hand:
 - `foundations/plot_02_ber_qam_awgn.py:47-51` and `waveforms/plot_04_proakis_msk.py:72-74`;
 - `waveforms/plot_05_thompson2008_fig2.py:73-74`, which is `min_count`;
 - `simulation/plot_02_wireless_chains.py:38` `rate()`, which is `series(min_count=20, **cond)`;
 - `simulation/plot_01_chains.py:76` and `optical/plot_03/04/05`.
 Fix: `res.series(..., interval=True, min_count=...)` or `comrs.plot.curves`.

4. **Redundancy · medium · S/M.** About 45 inline `10 ** (x / 10)` / `10 * np.log10` in the gallery and tests, although `db2lin`/`lin2db` exist. Some expressions mix both, e.g. `mimo/plot_02_space_time_diversity.py:70`: `comrs.lin2db(share * 10 ** (eb / 10))`. Closed forms that take arrays are looped or vectorised: `waveforms/plot_04_proakis_msk.py:62` (`np.vectorize(comrs.qfunc)`), `foundations/plot_02:53`, `coding/plot_01:53`. Fix: use `db2lin`/`lin2db` and pass arrays.

5. **Maintainability/bug · medium · S.** `tools/api_index.py:109-146` parses Python with regexes and a 12-line window, so long signatures are truncated in `docs/API.md`. `sim.run` (`API.md:2376`) loses `json_path` and its return type; `plot.constellation` (`:2347`) and `plot.curves` (`:2351`) end mid-signature. Fix: use `ast` (`ast.unparse` of args/returns, `ast.get_docstring`).

6. **API · medium · S.** Positional orders differ: `Chain.to_spec(name, grid, metrics, stop, seed)` (`chain.py:313`) against `Chain.run(grid, metrics, stop, seed, name)` (`chain.py:381`). `sim.run` makes `name`/`seed` keyword-only. Fix: make them keyword-only, as part of the section-5 plan before 0.2.0.

7. **API/perf · medium · S/M.** In `chain.py:417-433`, a native chain given a `sim.Stop` object silently runs on the one-thread Python path and ignores `threads=`. Fix: convert the `Stop` to its spec mapping in Rust, or raise when `threads` is given.

8. **Silent wrong result · medium · S.** `Results.where`/`series` (`results.py:80`, `124-134`) return empty arrays for a misspelt metric or variable. They also interleave several curves when the other variables are not fixed. `plot.curves` (`plot.py:352`) then draws nothing and its `kind` defaults to `"mean"`. Fix: `KeyError` for unknown names, `ValueError` when x repeats.

9. **Bug · medium · S.** `plot.psd` (`plot.py:295-308`): for a sampling rate in Hz below 1 MHz (e.g. 48 kHz) the unit is `""`. The labels then say "frequency / sampling rate" and "PSD (dB)" while the axis is in Hz. Fix: add Hz/kHz branches and choose the label from whether `fs` is known.

10. **Tests that cannot fail · medium · S.** `tests/oracle.py:58-66` and `tests/test_sionna_vectors.py:37` skip when a fixture is missing, so a deleted fixture turns the oracle tests into skips in CI. Fix: fail when `CI` is set, or add a test that every referenced fixture exists.

11. **Benchmark SoC · medium · M.** `run_spec_lib` is copied in `s1_awgn.py:224`, `s2_rayleigh.py:168`, `s3_mimo.py:202` and `s4_coding.py:380`. `threads = 1 if args.threads == "1" else 0` appears six times. `rust_binary`/`run_rust_cli` live in `s1_awgn.py:252-276`, and `s4_coding.py:410` imports them from that sibling scenario. Fix: move them to `common.py`.

12. **Benchmarks · medium · S.** The committed results contain absolute local paths: `"binary": str(rust_binary())` (`s1_awgn.py:290`, `s4_coding.py:421`) and `NUMBA_CACHE_DIR` (`common.py:67`, `:277`). `results/s1_awgn.json:951` holds `/Users/<user>/…`, which exposes the user name and makes diffs noisy. Fix: `relative_to(ROOT)`, and drop the cache directory from the recorded environment.

13. **Tools DRY · medium · S/M.**
  - Three list-table renderers: `benchmarks_page.py:158`, `highlights.py:284`, inline in `maturity_table.py:139-151`.
  - Two `rst()` with different regexes: `highlights.py:258` mangles text that is already in double backticks, `maturity_table.py:127` does not.
  - Two `splice()` with different semantics: `highlights.py:457` keeps the marker lines and raises `ValueError`; `maturity_table.py:156` replaces them and raises `SystemExit`.
  - `maturity_table.main` (`:179-195`) re-implements `write_or_check`, and `md()` (`highlights.py:254`) does nothing.
  Fix: move `table`/`rst`/`splice` into `_common.py`.

14. **SoC (tests) · medium · S.** `test_api.py:394` `test_invalid_parameters_raise_value_errors` makes 15 unrelated checks in one test. `:433` `test_defaults_are_those_of_the_blocks` mixes pulse shape, K-best, FOE, SSFM and SigMF. The first failure hides the rest. Fix: parametrise, with ids.

15. **Redundancy · medium · M.** `tests/test_cards.py:111` `accepted()` re-implements "the rule of the cards (and of `comrs verify`)" in Python. It can drift from Rust; for example `factor` only accepts `want > 0`. Fix: bind the Rust tolerance check and call it.

16. **Redundancy · low · M.** Tests copy gallery code: `test_performance.py:606-628` (MIMO-OFDM SER loop) matches `receiver/plot_03_sync_mimo_ofdm.py:78-92`, and the test's BICM chain matches `simulation/plot_02_wireless_chains.py:120`. Fix: one shared builder imported by both.

17. **DRY · low · S.** `conftest.py:21-23` exports `ROOT`/`SPECS`, yet `test_api.py:198`, `test_cards.py:27`, `test_io_rates.py:13`, `test_posterior_search.py:11`, `test_performance.py:740` and `test_spec.py:9` recompute them. `tools/record_aff3ct.py:46` recomputes `ROOT` despite `_common`. The card loader exists twice (`tools/cards.py:31`, `tests/test_cards.py:30`). `test_chain.py:14` `qpsk_link` is a trivial wrapper.

18. **API · low · S.** `Block.bind` (`block.py:125`) has no `base_dir`, while `Native.bind` and `Chain.bind` do, so `chain.py:474` `_paths` dispatches by `isinstance`. `Chain.trace` (`:303-309`) re-implements the defaults of `bind` and drops `name`. Fix: add `base_dir` to `Block.bind`, and share a `_steps()` helper.

19. **Typing · low · S.** `pyproject.toml` says "the public functions are annotated", but `plot.constellation/eye/time/psd/curves`, `Results.plot`, `Results.to_frame` and `Trace.plot` have no return type, and `ax=None` is untyped. `Results(list)` and `Trace(dict)` are not parametrised. Fix: `Axes` under `TYPE_CHECKING`, `class Results(list[dict[str, Any]])`.

20. **Consistency · low · S.** `ax` is positional in `plot.constellation/eye/time/psd` but keyword-only in `plot.curves` and `Results.plot`. Pick one.

21. **Consistency · low · S.** `Chain(*blocks)` names unnamed blocks `block0`, `block1` (`chain.py:79`), and `to_spec` writes `label: block0` (`block.py:181`). The Rust runner and `from_spec` name blocks by type (`source`, `awgn_2`). Fix: name them with `_link_labels`.

22. **SoC · low · S.** `sim.grid` (`sim.py:88-89`) expands the axes by building a fake spec with `"link": ["normalize"]`. The axis normalisation is duplicated in `spec.settings` (`spec.py:134`). Fix: bind the Rust grid expander directly.

23. **Silent overwrite · low · S.** `Trace.save` (`results.py:413-419`): the keys `<name>_1`, `<name>_sps` and `tx_symbols` collide with stages of those names. Fix: namespace the keys, e.g. `stage/<name>/…`.

24. **Encoding · low · S.** Text is read and written without `encoding` in `results.py:273,287`, `tests/oracle.py:97,110,196`, `test_cards.py:31` and `api_index.py:112`, while the other tools use utf-8. Windows CI runs pytest with the locale default, and 3 cards contain non-ASCII.

25. **Maintainability · low · S.** `test_api.py:173` asserts `__version__ == "0.2.0"`, so it breaks on every release. Compare with the `pyproject.toml`/Cargo version instead.

26. **Reusability · low · S.** The figure style lives only in `docs/python/conf.py:60` `reset_style`. An example run directly, as `_gallery_utils.py:8` recommends, gets the matplotlib defaults. Fix: a `comrs.plot.style()` next to the palette, called by `conf.py`.

27. **Reusability · low · M.** `receiver/plot_04_modulation_recognition.py:207-222` re-implements the RRC pulse at arbitrary times, because `pulse_shape` takes an integer `sps` only (stub `:1517`). Fix: bind the continuous-time RRC of com-dsp.

28. **Consistency · low · S.** Near-duplicate palette entries: `BLUE` #2a78d6 / `COBALT` #2a6fdb, `GREY` #8a8984 / `GREY_WARM` #8a8780, `GREEN` / `TEAL` (`plot.py:55-82`). `STREAM_COLORS[i % 4]` (`:165`, `:300`) should be `len(STREAM_COLORS)`.

29. **Consistency · low · S.** Gallery titles of reproductions: most read "Author Year, Fig. N:" (hu2005, alamouti1998, hager2021, thompson2008). `plot_04_proakis_msk.py` reads "Proakis & Salehi, Sec. 3.3-2" with no year, and `plot_06`/`plot_07` read "against five/three papers". The file names follow the same split.

30. **Maintainability · low · M.** `tools/highlights.py:292-455` `page()` builds the whole page in about 165 lines. Split it per section, as `benchmarks_page.py` does.

**What is good**
- The package is thin. Spec loading, `$var` substitution, link labels, statistics, stopping rules and the trial runner all call Rust (`spec.py`, `sim.py`, `block.py:_resolve`), and the section-1/4 fixes of the previous audit have held.
- Google-style docstrings are consistent and enforced by tests (Args sections, doctests in every module, the four name lists cross-checked in `test_api.py:203`).
- Tests are named after their references, use explicit seeds and four-sigma binomial acceptance, and the oracle fixtures are recorded with their tool versions.
- `tools/_common.py` and the generators are clean (write-or-check, `--check` everywhere); `benchmarks/libraries/theory.py` is deliberately independent of every library it compares.
- The gallery cites authors, venue, year and figure; it has one palette with no colour literals left.
