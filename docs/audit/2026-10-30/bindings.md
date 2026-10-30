# Review of Python bindings (bindings/com-py) — 30 October 2026

Read-only review; each finding was checked against the code by the reviewer. Paths and line numbers refer to commit 21af3f7.

Findings, most important first. Format: category · severity · effort. Paths are relative to `bindings/com-py/src/` unless stated otherwise.

1. **Consistency · high · breaking · S.** `ReedSolomonCode.n`/`.k` return symbol counts (`fec_codes.rs:127-133`, `n_symbols()`/`k_symbols()`). In Rust, `n()`/`k()` count bits since the audit fix. Every other Python code uses bits, and `encode` expects multiples of `k·m` bits. Fix: make `.n`/`.k` bits and add `.n_symbols`/`.k_symbols`.

2. **API · high · breaking · M.** Antenna layout differs within one file.
 - `mimo_detect`/`mimo_llr`/`mimo_estimate` take `y` as `(n, n_rx)` (`mimo.rs:54` `received()`).
 - `Ostbc.encode`/`decode` use `(n_tx, slots)` and `(n_rx, slots)` (`mimo.rs:638`).
 - `TdlChannel`/`CdlChannel.fade` use `(n_tx, n)` (`streams_in`).
 Fix: one stream-major `(n_ant, n)` convention, like `Signal`, used everywhere through `streams_in`/`streams_out`.

3. **Redundancy · medium · S.** Crate defaults are copied as literals in the `#[pyo3(signature)]` lines:
 - `Photodiode` (`devices.rs:284`), `FiberParams` (`fiber.rs:90`), `Edfa` (`fiber.rs:208`)
 - Saleh coefficients (`ofdm.rs:595`), `MemoryPolynomialDpd` `[1,3,5]`/3 (`ofdm.rs:646`)

 Fix: public `DEFAULT_*` consts or `T::default()` fields in the crates, used in the signatures. `ldpc::DEFAULT_*` and `DetectorKind::DEFAULT_PATHS` already do this.

4. **Redundancy · medium · M.** Several classes mirror their Rust struct field by field, with hand-written `to_rust`/`from_rust`:
 - `fiber.rs:62-85`, `fiber.rs:200`, `devices.rs:263`
 - `PinFetReceiver`, copied twice (`owc.rs:657`, `owc.rs:705`)

 `Photodiode.filter_kind` stays a string that is parsed on every `detect` (`devices.rs:264`), so a typo only fails at use. Fix: wrap `inner: T` with getters and setters (a small macro), and parse at assignment.

5. **SoC · medium · S.** `amplifier_bussgang` (`multicarrier.rs:239-264`) builds the AM/AM–AM/PM envelope closures in the bindings. Its defaults also disagree with `pa_saleh`/`pa_rapp`: Saleh 1/1/0/1 vs 2.1587/…, Rapp smoothness 1.0 vs 2.0. Fix: add `nonlinear::bussgang_of(&impl Amplifier, power)` in the crate and reuse the crate defaults.

6. **API · medium · S.** `cm_capacity`/`bicm_capacity`/`bicm_cutoff_rate` default to `points=16` (`comm.rs:754/779/826`). `theory_mi`/`theory_gmi` default to `points=40` (`comm.rs:689/718`), and their docstrings say they are "`theory_mi` of a constellation object". Same kernel, different numbers. Fix: one shared constant.

7. **Redundancy · medium · S.** `Sim.run_chain` rebuilds the runner from a `Spec` (`sim.rs:867-884`), duplicating `com_spec::run` (`crates/com-spec/src/run.rs:35-53`). It has already drifted: it lacks `fast_dev_run` and the `opts` overrides. Fix: expose `com_spec::runner(&Spec, &RunOptions) -> Runner`.

8. **Redundancy · medium · S.** The flattened result row (`m`, `m_lo`, `m_hi`, `m_count`, `m_total`) is written three times:
 - `ConditionResult.row` (`sim.rs:578`)
 - `CsvLogger` (`crates/com-sim/src/loggers.rs:125-160`)
 - `python/comrs/results.py:150`, which already omits `_total` and `index`

 Fix: add `ConditionResult::row()` in com-sim and call it from both bindings.

9. **SoC · medium · M.** Ad-hoc `match` on strings in the bindings, which the guide forbids:
 - `NrPolar` link (`fec.rs:959`), `"lte"` QPP (`fec.rs:477`)
 - `Wave` (`owc.rs:411`), turbulence (`owc.rs:578`), `SsbElement` (`nr.rs:143`)
 - `theory_ber_gray` (`comm.rs:587`, lower-cases its input, unlike every other parser)
 - `theory_ber_imperfect_csi` (`link_theory.rs:46`), `theory_ber_dpsk` (`differential.rs:269`)
 - amplifier model (`multicarrier.rs:250`), classifier method (`blind.rs:109`)

 Fix: `FromStr`/`from_name` in the crates.

10. **SoC · medium · M.** Checks exist only in Python, so Rust callers get silent wrong results or panics:
  - `ddpll` pilots and `fs` (`dsp.rs:604,612`); the crate zips and skips bad indices silently
  - `phase_noise` `fs` (`devices.rs:99`), `pulse_shape` (`dsp.rs:50`)
  - `Cdl` spacing and carrier frequency (`cdl.rs:126,133`)
  - `Constellation.symbols` index (`comm.rs:203`), `gray_code` limit (`comm.rs:303`)
  - `ResourceGrid.extract` (`ofdm_link.rs:334`), `distortion_spectrum`, `snr_crb_nmse`, `nr_detect_cell_id`

  Also, `ofdm_estimate` silently ignores partial combinations of `delays`/`powers`/`max_delay`/`n_fft` (`ofdm_link.rs:388`). Fix: the crate functions return `Result`.

11. **Maintainability · medium · S.** The `Ssfm` docstring says each span gets `⌊L_span/h⌋` steps (`fiber.rs:299`). The code now uses `⌈L/h⌉` equal steps (`crates/com-channel/src/optical/fiber.rs:298-316`). Fix: update the docstring.

12. **API · medium · breaking · M.** Return types depend on the arguments:
  - `Ofdm.modulate` (`ofdm.rs:155`) and `papr_db` (`ofdm.rs:411`, which also drops the tail through `chunks_exact`)
  - `mimo_channel(n=…)`, `mimo_capacity` (`mimo.rs:706`)
  - `ChannelTaps.frequency_response` (`ofdm_link.rs:204`), `read_sigmf` (`spec.rs:464`)
  - `ManakovSsfm.propagate` returns 3 values, `Manakov.propagate` returns 2

  Fix: separate methods (`modulate_with`, `papr_db_blocks`) and always-array returns.

13. **API · medium · breaking · M.** The `theory_*` closed forms are uneven:
  - Scalar-only, contradicting the `lib.rs` header: `converter.rs:120,142,179,218`, `mimo_capacity`/`outage_probability` `snr_db`, `css_ser_interference` (`waveform.rs:1068`).
  - Missing the `theory_` prefix: `cm_*`/`bicm_*`, `biawgn_capacity`, `normal_approximation`, `shannon_limit_ebn0_db`, `ergodic_capacity`, `outage_*`, `css_ser`/`css_ber`, `msk_psd`, `sdr_*`, `differential_phase_tail`.
  - Linear SNRs with no `_lin`: `theory_phase_pdf(snr)` (`comm.rs:642`), `differential_phase_tail(esn0)` (`differential.rs:297`).

14. **API · medium · breaking · S.** Argument order:
  - `NrLdpc(k, n)`/`NrPolar(k, n)` vs `PolarCode(n, k)`/`LdpcCode.peg(n, k)`/`normal_approximation(n, k)`.
  - `Constellation(kind, m)` vs `theory_ber(m, ebn0, kind)` vs `theory_ber_imperfect_csi("16qam")` vs `theory_ber_dpsk_jakes(m=2|4)`.

  Fix: `(n, k)` everywhere, and `(m, ebn0_db, kind=…)` for every modulation closed form.

15. **API · medium · S.** Iteration settings are named differently: `LdpcCode.set_decoder(max_iterations)`, `NrLdpc(iterations)`, `TurboCode.iterations`, `Dvbs2.iterations`. `NrLdpc` hard-codes `DEFAULT_ALPHA` (`fec.rs:871`), so normalized or offset min-sum cannot be tuned. Its `"sum_product"`/`50` literals (`fec.rs:858`) bypass `ldpc::DEFAULT_*`.

16. **Redundancy · medium · S.** The `"p/q"` parser for the modulation index exists twice, identical: `waveform.rs:16-25` and `crates/com-spec/src/blocks/cpm.rs:68-72`. Fix: move it to com-waveform (`FromStr` for the index).

17. **SoC · low · S.** Validation by side effect:
  - `GammaGamma`/`PointingError` construct a dummy `Fso` and call `check()` (`owc.rs:395,484`).
  - `CeOfdm` calls `theory::snr(&inner, 0.0)` (`waveform.rs:464`) although `CeOfdm::check` exists.

18. **Redundancy · low · S.** The convolutional constraint check `2..=16` is duplicated (`fec.rs:328`, `fec_codes.rs:539`) and contradicts the crate, which accepts 1..=24. Fix: delete both and rely on `Trellis::feedforward`.

19. **SoC · low · M.** Arithmetic in the bindings:
  - ln-power to dBm in `raman_profile` (`gn.rs:209`)
  - NR DMRS indices with a magic `12` (`nr.rs:223`)
  - `BchCode.rate` computed by hand (`fec_codes.rs:450`) instead of `code_rate()`
  - RS/BCH default polynomial and `n` with `m.min(16)`, duplicated (`fec_codes.rs:118,422`)
  - `gn_model` orchestrating five calls with a default channel of `n/2` (`gn.rs:78`)

20. **SoC · medium · M.** Large or mixed files:
  - `fec.rs` (1172 lines): split into crc/interleaver, conv/turbo, ldpc/polar, nr; move CCDM next to the shaping code.
  - `waveform.rs` (1098): split into cpm, ce_ofdm, css.
  - `comm.rs` (953): split into constellation, sources, metrics, theory.
  - `ofdm.rs` (790): mixes OFDM, PAPR, amplifiers, DPD, spectra and SDR.
  - The amplifier bindings are split between `ofdm.rs` and `multicarrier.rs`. `chebyshev_window` is in `multicarrier.rs` and `bessel_j0` in `link_theory.rs`; gather them in `amplifier.rs`, `dsp.rs` and `math.rs`.

21. **API · low · breaking · S.** Grid layouts differ: `ResourceGrid.map`/`extract` use flat grids, while `nr_*` use 2-D `(K, L)` arrays. NR indices are Fortran-order (`l·240 + k`) into C-order arrays, so `grid.ravel()[idx]` gives wrong elements.

22. **API · low · breaking · S.** The generator argument sits in different places: `GammaGamma.draw(rng, n)`, `PointingError.draw(rng, n)`, `Fso.gains(rng, n)` (`owc.rs:457,534,609`) vs `CdlChannel.draw(n, rng)` and `bits(n, rng)`. Also `phase_noise(linewidth_hz, n, fs, rng)` vs `laser(n, fs, rng)`.

23. **API · low · breaking · S.** Tuning parameters that are not keyword-only:
  - `rng=None` of `Ssfm`/`Manakov`/`ManakovSsfm.propagate`
  - `message_passing(method, iterations)`, `cm_*(fading)`, `theory_ber_rayleigh_mrc(branches)`
  - `theory_ber_imperfect_csi(rho2, sigma_ratio)`, `cyclic_autocorrelation(lag, fs)`, `psd`/`periodogram(fs)`
  - `cd_savory_taps(n_taps)`, `Cpm(pulse)`, `css_ser(approximation)`, `nr_ssb_grid(pbch)`

24. **API · low · breaking · S.** Units missing from names beyond the audit's list: `aclr_db(bandwidth, offset)`, `delay_fft(delay)`, `freq_shift(delta_f)`, `gn_model(symbol_rate)`, `gn_closed_form`/`wdm_grid(spacing)`, `estimate_symbol_rate(f_min, f_max)`, `theory_jitter_snr_db(f)`, `zero_order_hold_gain(f)`, the `nr_numerology` keys, `ofdm_synchronize["cfo"]`, `Photodiode.dark_current` (vs `PinFetReceiver.background_current_a`).

25. **Consistency · low · M.** Rust and Python names differ for the same item:
  - `Ofdm.n_subcarriers` (Rust `n_active`), while `Ufmc.n_active`
  - `Cdl` `carrier_frequency`/`velocity`/`max_doppler()` vs `fc_hz`/`velocity_mps`/`max_doppler_hz`
  - `ergodic_rayleigh`→`ergodic_capacity`, `lagrange_cubic`→`interpolate_cubic`, `attenuation_db_per_km`→`fog_attenuation_db_per_km`, `Fso::gain`→`gains`, `closed_form_nli_psd`→`gn_closed_form`
  - `samples_per_symbol` in `snr_crb_nmse` vs `sps` everywhere else

  Fix: state the rule in the guide and have `api_index.py` flag differences.

26. **Redundancy · low · S.** Small repeats:
  - 1-D/2-D handling through `getattr("ndim")` in `adaptive_equalizer` (`dsp.rs:433,444`) instead of `streams_in`
  - "profile name or taps" twice (`ofdm_link.rs:108`, `otfs.rs:136`)
  - `TurboCode.encode`/`decode` instead of the shared `encode`/`decode` helpers (relax them to `?Sized`)
  - `search_result` rebuilds `condition_dict` (`spec.rs:359`)
  - about 20 hand-built result dicts; derive `Serialize` and use `from_json`, or result pyclasses

27. **API · low · M.** 22 classes lack `__repr__`: LinkBlock, Photodiode, Crc, Scrambler, Interleaver, ConvolutionalCode, TurboCode, LdpcCode, PolarCode, NrLdpc, NrPolar, Ccdm, Ssfm, Manakov, ManakovSsfm, Ufmc, MemoryPolynomialDpd, OwcIllumination, PointingError, Fso, PinFetReceiver, Dvbs2. Nothing is picklable; most important is `Rng`, for multiprocessing: add `__getstate__`/`__setstate__`, and `__getnewargs_ex__` on value classes. `Frame.copy()` should be `__copy__`.

28. **Maintainability · low · M.** About 222 of 237 fallible documented bindings have no `Raises:` section (for example `Crc`, `PolarCode`, `NrPolar`). Fix: add the sections, and a check in `tests/test_api.py`.

29. **API · low · S.** Exceptions:
  - Metric-kind mismatches raise `TypeError` (`sim.rs:468,484,501`), where `ValueError` is expected.
  - Type errors are rewrapped as `ValueError` (`fec.rs:483`, `puncturer`).
  - `count()` is used about 10 times, but about 40 `n: usize` arguments still raise `OverflowError` on negative input (for example `Scrambler.sequence`, `GammaGamma.draw`).

30. **Maintainability · low · S.** 14 of 26 `#[allow(clippy::…)]` have no `reason`: `cdl.rs:114`, `devices.rs:122,201,285,396`, `fiber.rs:565`, `dsp.rs:419,587`, `mimo.rs:155`, `gn.rs:59`, `ofdm_link.rs:262,376`, `sim.rs:780`, `spec.rs:152`. Magic defaults: `max_trials(100)` (`sim.rs:805`), `ddpll` `fs=32e9`, `mu=1e-3` (`dsp.rs:449`).

**What is good**
- The shared helpers in `lib.rs` (`cvec`/`fvec`/`bvec`/`uvec`, `map_real`, `streams_in`/`streams_out`, `err`/`.py()`) are used almost everywhere; only `owc.rs:26` converts by hand.
- The GIL is released consistently on long kernels. Python exceptions from trials and from the DPD amplifier callback pass through Rust unchanged.
- Docstrings are rich, cite their references and carry doctests. The stub is generated and matches the code.
- Most audit section 4/6 fixes held: trial recording, metrics and guards call the crates, and `spec.load` calls Rust.
- Several parameters already reuse crate constants (`ldpc::DEFAULT_*`, `DetectorKind::DEFAULT_*`), which is the model for finding 3.
