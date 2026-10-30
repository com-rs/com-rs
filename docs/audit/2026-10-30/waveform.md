# Review of com-waveform, com-hardware — 30 October 2026

Read-only review; each finding was checked against the code by the reviewer. Paths and line numbers refer to commit 21af3f7.

## Findings: com-waveform and com-hardware (most important first)

1. **Verb traits only partly implemented.** Reusability, high, M. CE-OFDM (`ce_ofdm/mod.rs:266,310`) and FBMC (`fbmc/mod.rs:227,255`) map complex symbols to samples, yet neither implements `Modulate`/`Demodulate`. CSS and CPM work on `&[usize]` indices, which `Symbols` cannot carry. The existing impls invent sample rates: `Hertz(L)` for OFDM (`ofdm/mod.rs:320`), `Hertz(1.0)` for UFMC and OTFS (`ufmc.rs:220`, `otfs.rs:241`). No `demodulate_into` checks `y.sample_rate()`. Fix:
 - Implement the traits for CeOfdm (receiver as a field) and for Fbmc (OQAM stagger inside the trait).
 - Write an ADR for index-symbol waveforms.
 - Give the modems a sample rate, or check the input rate.
2. **WDM transmitter in the wrong crate.** Placement, medium, M. `WdmTransmitter`, `wdm_demultiplex` and `wdm_grid` (`optical.rs:436-521`) assemble pulse shaping, upsampling, laser and IQ modulator. Fix:
 - Move the transmitter to `com-systems`.
 - `wdm_demultiplex` is only `freq_shift` plus `resample_fft`, so it belongs in `com_dsp::signal`.
 - The test helper `grid()` of `com-channel/src/optical/gn.rs:678` re-implements `wdm_grid` without the half-spacing shift. Keep one source.
3. **The laser model is written twice.** Redundancy, medium, S. `WdmTransmitter::multiplex` (`optical.rs:491-497`) rebuilds the carrier of `Laser::field` (`:78-87`): Wiener phase plus frequency offset. Both hand-code `cis(2π f k/fs)` instead of `com_dsp::signal::rotate`. `phase_noise` (`oscillator.rs:41`) recomputes the formula of `increment_sd` (`:47`), one taking `ts` and the other `fs`. Fix: add `Laser::carrier(n, fs, rng)` and make `phase_noise` call `increment_sd`. Caution: `phase_noise` draws n−1 Gaussians even at zero linewidth while `multiplex` skips them, so the merge changes random streams.
4. **The OFDM link re-implements the modem.** Redundancy, medium, S. `link.rs:113-117` (`with_prefix(to_time(sym))` per symbol) equals `Ofdm::modulate(&g)`, since DFT spreading is refused at `:95`. `received_grid` (`:156-166`) strips prefixes and calls `to_active` by hand, which is `Ofdm::demodulate` on the grid's samples. Fix: call `modulate`/`demodulate` on the slice and scale afterwards.
5. **OTFS path type, and a dense matrix documented as sparse.** Reusability/docs, medium, M. `Otfs::effective_channel(&[(C64, usize, f64)], sample_rate: f64)` (`otfs.rs:179`) is an anonymous copy of `com_channel::multipath::Path` (`multipath.rs:579`). The bindings and `validation/tests/raviteja2019.rs` convert by hand. Fix: move `Path` to `com-core` and take `&[Path]` and `Hertz`. The function also returns a dense MN×MN `CMat` (`:183`) while `lib.rs:10` and the module call it "the sparse effective channel". That is 16 MB at 32×32 and 268 MB at 64×64. Return the P entries of each row, or an operator.
6. **Three SNR conventions under the names `ser`/`ber`.** Consistency, medium, S/M.
 - `com_mapping::theory::ser(kind, m, ebn0_db)`, `ce_ofdm::theory::ser(ce, kind, m, esn0_db)` (per data symbol) and `css::theory::ser(sf, snr_db)` (per sample).
 - CE-OFDM converts back through a private `ebn0_db` (`ce_ofdm/theory.rs:100`).
 - Fix: add an Es/N0 entry point in `com_mapping::theory`. Move `ber_orthogonal_noncoherent` (SER·M/(2(M−1)), `css/theory.rs:84`) next to `ser_orthogonal_noncoherent`.
7. **Units missing from names.** Consistency, medium, S; breaking, so batch it with audit section 5.
 - Linear SNRs without `_lin`: `cp_ml(…, snr)`, `moose_std(es_n0)`, `cfo_snr_bound(snr, …)` (`sync.rs:217,253,268`), while `correction.rs` in the same module uses `sir_lin`.
 - Bare f64 in Hz or s: `Numerology::subcarrier_spacing()`/`sample_rate()` (`nr.rs:124,144`); `phase_noise(…, ts)`; `jitter_snr_db(f, …)`, `zero_order_hold_gain(f, fs, …)` (`converter.rs:161,167`); `wdm_grid(n, spacing)`.
 - `PhaseNoise.linewidth: Hertz` and `Cfo.offset: Hertz` against `Laser.linewidth_hz`/`freq_shift_hz: f64`.
 - `Photodiode.dark_current` (A, no suffix) and `temperature_c`, against `PinFetReceiver.background_current_a` and `temperature_k`.
 - CPM uses `sps` where the crate's vocabulary is `oversampling`. `n_data` means subcarriers (CeOfdm), symbols per antenna (`OfdmFrame`) and data REs (`ResourceGrid::n_data`).
8. **User input that panics or is silently altered.** Result vs panic, medium, S. `PowerDelayProfile::new` (`estimation.rs:73`) cannot fail and has public fields. Delays and powers of different lengths hit the `assert_eq!` of `com_dsp::estimation::frequency_correlation` during `estimate`. A zero sum is clamped to 1e-300. `ResourceGrid` has no `check`: `Comb { spacing: 0 }` silently becomes 1 through `.max(1)` (`grid.rs:187-189`), although `from_name` refuses 0. Fix: `new -> Result`, and `ResourceGrid::check` called by `map` and `estimate_port`.
9. **Wrong error variants, and docs that contradict the code.** Consistency, low-medium, S. Length mismatches return `Error::Config` in `nr::modulate` (`nr.rs:180`), `ssb_grid` (`:302`) and `detect_cell_id` (`:317`); `Ofdm` uses `ensure_shape` for the same case. `preamble` and `repeated_preamble` document `Config` (`sync.rs:113,187`), but `extend_with_prefix` returns `Shape`.
10. **Validation and recomputation inside loops.** Performance/guide rule, medium, S. `Css::modulate` calls `symbol(a)` for every symbol, which runs `check()` and M·J `cis` each time (`css/mod.rs:148-165`). `energies` rebuilds `symbol(0)` on every call. `CeOfdm::demodulate` checks the configuration and the shape, then `project` checks both again (`ce_ofdm/mod.rs:284-287, 321-323`). Fix: tabulate the base chirp once and validate once at entry.
11. **Optical devices are not `Impair` blocks.** Reusability, medium, M. `Laser`, `Mzm`, `Iqm`, `Photodiode` and `coherent_receiver` take slices, `fs: f64` and `&mut Rng` (`optical.rs:78-425`); the amplifiers, oscillator, IQ and converter blocks are `Impair` on `Signal`. Fix: at least the receivers on `Signal`/`DualPol`, with the rate read from the signal.
12. **NR has no inverse of `modulate`.** Reusability, low-medium, S. `nr::modulate(nr, grid, k, n_fft, first_symbol)` (`nr.rs:178`) has no demodulator. A receiver must re-derive the variable prefixes before `detect_cell_id` or `estimate`. Fix: add `nr::demodulate`, and a config struct instead of 5 positional arguments.
13. **Constructors take different shapes.** Consistency, low, M.
  - `Ofdm`, `CeOfdm` and `Otfs` have `new -> Result` plus struct update.
  - `Ufmc::new` takes 5 positional arguments (`ufmc.rs:79`).
  - `Css` has no `new`, and `Css::lora` is unchecked (`css/mod.rs:102`).
  - `Rapp`, `Saleh` and `Mzm` are never validated: saturation 0 gives NaN.
  - Long positional lists: `PilotPattern::from_name(name, period, spacing, freq, time)`, where three of the four numbers are ignored (`grid.rs:144`); `estimate_port` (6); `OfdmFrame::detect` (6, with an `alamouti: bool`, `link.rs:199`); `multiplex`, `ser_receiver`, `ber_receiver`, `fer_interference` (5 each).
14. **Resource-grid enumeration written five times.** Redundancy, low, S. `pilot_positions`, `map`, `extract`, `port_pilots` (`grid.rs:198-284`) and `OfdmFrame::data_res` (`link.rs:127`) each walk the grid. `n_data()` allocates `pilot_positions` on every call, and `map` calls it twice. Fix: `ResourceGrid::data_indices()`.
15. **Two CE-OFDM entry points for one thing.** Redundancy/naming, low, S. `CeOfdm::demodulate(r, unwrap)` is `demodulate_with(r, Receiver::Arctangent(unwrap))`. `demodulate_with` also means "with a PAPR technique" in `Ofdm` (`ofdm/mod.rs:259`). Keep one and rename the other.
16. **Argmax by `fold` written four times.** Redundancy, low, S. `sync.rs:163,277`, `css/mod.rs:200`, `css/theory.rs:706`, plus loops in `nr.rs:326-340`. Fix: `com_math::argmax`. `synchronize` also recomputes P(d̂) by hand (`sync.rs:175`) after `schmidl_cox` discarded `delayed_correlation`.
17. **Small copies and aliases.** Redundancy, low, S. The `ratio` closure is copied in `css/theory.rs:380` and `:404`. `rho()` (`:44`) and `papr::amplitude()` (`papr.rs:143`) are bare aliases of `db2lin`/`db2amp`.
18. **Filter helpers written inline.** Redundancy, low, S.
  - Unit-energy normalisation of the taps: `fbmc/mod.rs:114` and `ufmc.rs:124`.
  - The OQAM phase j^(m+n): three times in FBMC (`:189,237,270`).
  - The FIR response at a frequency, by hand in UFMC (`ufmc.rs:152`).
  - Fix: `com_dsp::filter::{normalize_energy, response_at}`.
19. **Shot and thermal noise written twice.** Redundancy, low, S. `Photodiode::detect_power` (`optical.rs:265-277`) and `PinFetReceiver` (`:360-372`), with a bare `273.15`. Fix: two shared variance functions and a named constant.
20. **Compensation kernels in com-hardware and com-waveform.** Placement, low, S. `Rapp::predistort` (`amplifier.rs:93`) is a predistorter, while the guide puts DPD in `com_dsp::compensation::dpd`. The com-hardware Cargo description still says "predistortion". `papr::limit` (`papr.rs:139`), the ideal envelope limiter, belongs in `com_dsp::nonlinear` next to `clipping_bussgang`.
21. **The same `Impair` body five times.** Maintainability, low, S. `amplifier.rs:104-201` and `iq.rs:54-91` each do copy, apply, and a second allocation. Fix: an in-place `map_streams` helper in `com-core`.
22. **Rustdoc carries history; one module rule is not followed.** Docs, low, S. `papr.rs:500-503` and `:571-574` describe "the former 6000-point Simpson rule". Move that to the changelog and state the accuracy only. The rule in `lib.rs:21` ("built by `new` and checked by `check`") is not followed by Css, Cpm and Fbmc.
23. **Tests re-implement library helpers.** Maintainability, low, S.
  - Frequency offsets (`sync.rs:291,363`, doctest `:82`) instead of `com_dsp::signal::rotate`.
  - QPSK closures instead of `Rng::qpsk`: `correction.rs:115`, `optical.rs:541`, and in library code the padding of `link.rs:107`.
  - Trapezoid rules (`cpm/mod.rs:415`, the `gmsk_pulse` doctest, `cpm/detection.rs:266`) instead of `com_math::simpson`.
  - Two AWGN styles in one file (`ce_ofdm/mod.rs:399` vs `:442`).
24. **Magic numbers and unbounded searches.** Maintainability, low, S.
  - The 0.9 plateau threshold (`sync.rs:165`).
  - GAMP's 8, 7, 14 and 4096 (`gamp.rs:113,136-138`).
  - `offsets` falls back to 0.2 for a non-finite `eps`, undocumented (`css/theory.rs:420`).
  - Rapp's 1e-12 and 1e-6 (`amplifier.rs:86,94`).
  - `min_distance2` has no bound on (2M−1)^N: `max_len` = 20 means 3.5·10⁹ sequences (`cpm/detection.rs:251`).

## What is good

- No function in the library code exceeds about 70 lines (`estimate_port` is the longest). There is no `unsafe`, and the one `#[allow]` (`optical.rs:194`) carries a reason.
- The fixes of the previous audit hold: shared prefix and `centred_bin` kernels, NR on `Ofdm::to_time`, the memory polynomial on `com_dsp::dpd`, LMMSE and band-limited interpolators cached per pilot set, and μ-law on Gauss–Legendre. Each has a bit-exact regression test.
- Every formula is in LaTeX with its paper, equation and figure. `literature` test modules check them against published values, and they document where a paper's printed equation is wrong (Cimini 1985, Afisiadis 2020).
- Errors use `ensure`/`ensure_shape` almost everywhere, enums are `#[non_exhaustive]`, and the CPM phase state is kept exact as an integer.
