# Review of com-core, com-math, com-dsp, com-metrics — 30 October 2026

Read-only review; each finding was checked against the code by the reviewer. Paths and line numbers refer to commit 21af3f7.

Read-only review of com-core, com-math, com-dsp and com-metrics. Every item was checked in the code. Items 1 and 24 were reproduced with a probe crate built in the scratchpad; no repository file was touched. Items the audit already lists (`ddpll`, `low_pass_fir`) are not repeated.

Format: **#. Title** (category, severity, effort): where; fix.

1. **`Dbp` panics on an empty field** (consistency, high, S): `Dbp::run` and `run_dual` (compensation/dbp.rs:106,125) call `SplitStep::new` (nlse.rs:129), which calls `fft::plan(0)`, which asserts (fft.rs:72). Reproduced: `Dbp::new(m,80.,1,1.).run(&[],64e9)` panics, although the function returns `Result`. Fix: return early for `n == 0` in `Dbp::steps`, and guard or document `SplitStep::new`.

2. **DBP's exact inverse depends on a float floor test** (maintainability, high, M): `Dbp::exact` (dbp.rs:77-85) compares `steps_per_span == ⌊span/h⌋`. To make that test pass, com-channel lowers the forward step by ULPs (optical/fiber.rs:298-319). This is a hidden float protocol between two crates. Fix: replace it with an explicit `forward: Option<ForwardSteps { n_steps, step_km }>` and compare integers.

3. **`ln_1p_exp` written twice** (redundancy, medium, S): the private copy at information.rs:327 (it branches at x > 30) duplicates `com_math::ln_1p_exp` (lib.rs:901) with another formula. Fix: call com_math, then re-check the GMI cards for last-bit changes.

4. **Bitwise demapper re-implemented** (redundancy, medium, M): `posterior_rates::bmd_at` (information.rs:296-317) is `Constellation::llr` at variance σ̂²/s with priors (com-mapping demapper.rs:99-136), followed by `gmi_from_llr`. Fix: a per-symbol `llr_into` in com-mapping, reused here.

5. **`SplitStep::run_with_gain` does twice the FFTs** (redundancy/performance, medium, S): nlse.rs:196-211 goes back to the time domain after every step, then transforms forward again at the next one. That is 4 FFTs per step against 2 in `run` (142-156), on every Raman span (fiber.rs:365). Fix: stay in the frequency domain between steps; rounding changes, so compare against the current output.

6. **The scalar solvers' constructors differ** (reusability, medium, M): `SplitStep::new(medium, step_km, nl_length_km, n_steps, n, fs)` and `SplitStepF32::new(medium, step_km, n_steps, n, fs)` (nlse.rs:129, 247). The f32 version cannot take a nonlinear length (so no f32 DBP), plans per construction and allocates scratch per run (254, 268). Fix: one shared step-plan struct, and cached f32 plans.

7. **`ManakovSplitStep::run_inner` is 91 lines** (maintainability, medium, M): nlse.rs:348-438 holds step choice, gain half-steps and the trapezoidal iteration. It allocates `power`, `phi`, `lin`, `xf` and `yf` per step or iteration (371, 379, 393, 416). Fix: split into helpers and reuse buffers.

8. **`AdaptiveEqualizer::run` is 136 lines, 6 levels deep** (maintainability/reusability, medium, M): equalizer.rs:200-335.
 - Validation, alphabet statistics, the Kikuchi initialisation with literal `h[1]..h[3]`, filtering, four error laws and the update are all in one body.
 - It allocates `y` and `norm2` per output sample (280, 289).
 - Inputs are `Vec<Vec<C64>>`, and taps use the implicit index `mo + mi·n` (152).
 - Fix: one error law per algorithm, a taps struct with `tap(mo, mi)`, preallocated buffers, and `&[&[C64]]` or `Symbols` input.

9. **Normalised correlation written four times** (redundancy, medium, S): `|Σab*|/√(EₐE_b)` appears at frame.rs:118-126 and 211-214, errors.rs:173-178 and equalizer.rs:183-187. Fix: `com_math::normalized_correlation`.

10. **frame.rs has twin result types and a copied rounding** (redundancy/consistency, medium, S):
  - The rounding of the phase to 2π/M is copied (138-145, 216-223).
  - `FrameSync` and `PeriodicOffset` (82, 160) are the same struct.
  - "Window outside z" is `Config` in `frame_sync` (128) but `Shape` in `periodic_offset` (199).
  - Fix: one `quantize_phase`, one result type, `Shape` in both.

11. **cd.rs repeats its parameters and checks** (redundancy/reusability, medium, M): seven `cd_*` functions take `(beta2_s2_km, length_km, fs)`. The checks are pasted three times (135-136, 153-155, 180-182). `cd_` stutters inside module `cd`, and `cd_taps_needed` and `cd_savory_length` name one notion two ways. Fix: a `Dispersion { beta2_s2_km, length_km, fs_hz }` with methods and one validator.

12. **`MemoryPolynomialDpd` panics on its own public fields** (consistency/reusability, medium, M): `apply` asserts the coefficient count from public fields (dpd.rs:150-158), so invalid user input panics. The amplifier also stores the same model differently: `terms: Vec<(u32, Vec<C64>)>` against flat `orders×memory`. Fix: private fields with a `Result`, and one coefficient type shared with com-hardware.

13. **The unit newtypes are unused, their formulas duplicated** (consistency/redundancy, medium, M):
  - `Db::to_linear`/`from_linear`, `EbN0::to_es_n0` and `EsN0::to_eb_n0` (units.rs:31-62) have no caller outside units.rs; `Awgn::new(EsN0(Db(..)))` is the only use of the newtypes.
  - `com_math::db2lin`/`lin2db` (lib.rs:91-99) are used 109 times.
  - Eb/N0 conversions are rewritten inline (ce_ofdm/mod.rs:439,482; ce_ofdm/theory.rs:100; awgn.rs:139).
  - `to_es_n0` takes `u32` bits, so it cannot express shaped or coded `f64` bits.
  - Fix: one source (dB helpers in `com_core::units`, re-exported by com-math, and `bits: f64`), or retire the newtypes with an ADR.

14. **The OptiCommPy rotation is written twice** (redundancy, low, S): `evm` (errors.rs:258-268) re-derives the `mean(t/x)` rotation of `preprocess` (lib.rs:90-104), in calcEVM order. Fix: a shared `qam_psk_rotation`, with the order kept and commented.

15. **Alphabet moments computed three times** (redundancy, low, S): `Cumulants::estimate` and `of_constellation` paste the same accumulation (blind.rs:110-150). `m2m4` recomputes E|x|² and E|x|⁴ (244-250). The CMA radius recomputes them, and also hand-rolls `pnorm` (equalizer.rs:235-240). Fix: one weighted-moments helper.

16. **Raised-cosine spectrum written twice** (redundancy, low, S): `rrc_spectrum` (filter.rs:167-184) and com-channel `WdmSpectrum::shape` (optical/gn.rs:132-145). Fix: a scalar `com_dsp::filter::raised_cosine(f, rs, rolloff)` used by both.

17. **`fourth_power_foe` both estimates and corrects** (separation of concerns, low, S): it returns `(corrected, fo)` (carrier.rs:92-101). The refined variant throws the correction away and recomputes x^M (110-112). Fix: an estimator that returns `f64`; the caller applies `freq_shift`.

18. **Carrier-loop conventions** (consistency, low, S): `viterbi_viterbi` subtracts π/4 for every M (carrier.rs:189), which is the wrong offset for BPSK, and has no `Result` (m = 0 divides by zero). `ddpll` zips pilot indices with symbols, so a length mismatch silently drops pilots (246-251). Fix: offset π/M, return `Result`, check pilot lengths with `same_len`.

19. **com-math's lib.rs is a 1130-line grab bag** (maintainability, medium, M):
  - The modified Lentz loop is written three times (420-440, 666-683, 723-736); the last copy has no `tiny` guard.
  - The series loop is written three times and the midpoint rule twice (451, 546).
  - `1e-17` and `1e-300` are repeated as literals.
  - Fix: modules `special`, `quadrature`, `db`, `logdomain` re-exported at the root, a generic `lentz`, and named tolerances.

20. **`Slicer` grid is an anonymous 5-tuple** (maintainability, low, S): nearest.rs:25. PAM is told apart from QAM by `lookup.len() == side` (108). Fix: a private `Grid` struct.

21. **Type name `LowPass` used twice, and the duplicate check misses it** (consistency, low, S): `filter::LowPass` (filter.rs:263) and `interpolation::LowPass` (interpolation.rs:252). `tools/api_index.py:199-233` compares functions only; it also misses `Estimate`, `Profile` and `Receiver`. Fix: rename one, and extend `--duplicates` to types.

22. **Container accessors disagree** (consistency, low, M, breaking): `Symbols::stream` returns an `ArrayView1`, `Signal::stream` a slice. The resize method is `resize` on the containers but `reshape` on `Signal`, and the length is `len` against `n_samples` (signal.rs:94-165, 229-269). Fix: `stream` returns a slice everywhere, `stream_view` gives the view, and one name for resizing.

23. **Error kinds** (consistency, low, S): `stability::check` returns `Shape` for `ts ≤ 0` or `m = 0` (stability.rs:47-50). An empty input is `Config` in `welch`, `preprocess`, `Cumulants::estimate` and `fourth_power_foe`, but `Shape` in interpolation (72-77, 276). Fix: domain errors → `Config`, length errors → `Shape`.

24. **Silent values at zero** (consistency, low, S): reproduced `Field::inv(0) = 1` and `div(5,0) = 5` (gf2m.rs:134-146), and `Rng::below(0) = 0` (rng.rs:104). `gf2::poly_mul(&[],&[])` underflows (gf2.rs:18). Fix: document under `# Panics` with a `debug_assert`, or return `Option`.

25. **`delayed_correlation` and `windowed_energy` are O(n·W)** (performance, low, S): frame.rs:242-258, called with W = N/2 by com-waveform ofdm/sync.rs:128 and 275. Fix: running sums; this changes rounding, so recheck the cards (ADR 0005).

26. **Hot-path free functions have no workspace** (reusability, low, M):
  - `fir_filter` converts the taps and re-FFTs them on every call (filter.rs:187-260).
  - `fft::filter` and `resample_fft` copy again for the inverse (fft.rs:218-228).
  - com-spec dsp.rs:137 chains three allocations; the thresholds 32/8/64 (filter.rs:192-203) have no names.
  - Fix: a `Fir` with cached spectrum and `filter_into`, plus `upfirdn`, after a profile.

27. **`posterior_rates` normalises differently from its siblings** (consistency, low, S): it uses `c.normalized()` (information.rs:275), while the other functions use `normalized_under(p)` (36-38). Its doc compares it with `information_rates`. With shaped priors the two use different scales. Fix: document the choice or align.

28. **Unit and vocabulary drift** (consistency, low, S):
  - No units: `freq_shift(.., delta_f, fs)`, `delay_fft(.., delay, fs)`, `aclr_db(.., bandwidth, offset)`.
  - Linear `snr` with no marker: `tone_frequency_crb` and `SyncRule::Optimum`.
  - Mixed names: `samples_per_symbol` (`snr_crb_nmse`) against `sps`, `alpha` (`rrc_taps`) against `rolloff`, `sigma2` against `noise_var`.
  - `fftfreq` takes a spacing, `angular_frequencies` a rate.

29. **Unnamed pairs** (reusability, low, S): `monte_carlo_gmi` returns `(gmi, ngmi)` (information.rs:68) right next to the `InformationRates` struct; `m2m4` (blind.rs:244) and `features` (170) also return bare pairs. Fix: small structs.

30. **`Rng::qpsk` sits in com-core** (placement, low, S): rng.rs:169-174 draws a constellation point inside the crate that must contain no algorithm. Only OFDM and the validation tests use it. Fix: move it to `com_mapping::source`.

**What is good:**
- No layer violation, and shared kernels are really reused: lfsr and gold in four crates, `Medium::linear_response` in cd, `regularized` with Cholesky in estimation and interpolation, and the private `SymbolMetric` posterior.
- All 315 public functions that return `Result` have a `# Errors` section.
- Formulas carry their references, tests compare to SciPy and mpmath digit for digit, and tests named after the former computation guard bit-identity.
- Careful numerics: Cody's erfc, the log domain, the operation order of ADR 0005, and an FFT scratch proven bit-identical to rustfft.
- The containers follow ADR 0011 with tests of storage reuse, and `try_map_streams` refuses stale samples.
