# API consistency before the first release — 30 October 2026

Section 5 of the review of 30 October ([`docs/audit/2026-10-30-review.md`](../audit/2026-10-30-review.md))
and the items the seven slices deferred to it (the slice reports in
[`docs/audit/2026-10-30/`](../audit/2026-10-30/), cited as `core #28`, `bindings #2`…).
Version 0.2.0 is not released yet, so the breaking changes are made directly, without
deprecated aliases; the CHANGELOG of 0.2.0 records the final names only.

## Rules

These rules are added to `docs/ARCHITECTURE.md` (Naming conventions, Errors) by the
first phase, so that later code follows them.

1. **Units in names.** A public `f64` that has a unit carries it, in its type
   (`Hertz`, `Db`, `EbN0`) or in a suffix:

   | Suffix | Unit |
   |---|---|
   | `_hz` | Hz |
   | `_s` | s |
   | `_m` | m |
   | `_km` | km |
   | `_mps` | m/s |
   | `_rad` | rad |
   | `_db`, `_dbm` | dB, dBm |
   | `_w` | W |
   | `_a` | A |
   | `_k` | K |
   | `_lin` | a linear ratio that could be mistaken for dB, such as a linear SNR |

   This covers struct fields, methods that return a quantity, block parameters, Python
   parameters, and Rust parameters (renaming those is free).

   Names that stay bare:
   - the vocabulary already documented: `fs` (Hz), `rolloff`, `beta2`, `gamma`,
     `noise_var` (linear by its name);
   - counts and dimensionless quantities.
2. **One word per notion.**

   | Notion | Word | Replaces |
   |---|---|---|
   | samples per symbol of a pulse | `sps` | `samples_per_symbol` |
   | FFT oversampling factor (OFDM, CE-OFDM) | `oversampling` | — |
   | roll-off | `rolloff` | `alpha` |
   | noise variance | `noise_var` | `sigma2` |
   | fixed iteration count | `iterations` | — |
   | iteration cap with early stopping | `max_iterations` | — |
   | subcarriers | `n_subcarriers` | — |
   | data resource elements | `n_data` | — |

   `iterations` and `max_iterations` are chosen by meaning: the NR LDPC code is a cap,
   so it gets `max_iterations`, as `LdpcCode` has.
3. **Code dimensions.** Arguments keep the order of the reference: (n, k) for an
   (n, k) block code; (K, E) for the 3GPP rate-matched codes, named `k` and `e`, never
   `n`. Swapping two integers would break callers silently, so no order is changed.
   Instead, every name says what it is.
4. **One choice, one enum.** `com_math::Jacobian { Exact, MaxLog }` replaces the
   booleans `exact` (`max_star`, BCJR, turbo) and `max_log` (`Constellation::llr`).
   The YAML key `max_log` of the specifications stays a boolean, mapped to the enum.
5. **Errors.**
   - `Error::Shape` for lengths: two inputs that disagree, or an input empty where
     samples are needed.
   - `Error::Config` for values: outside their domain, not finite, or an unknown name.
   - Every function follows this rule, and its `# Errors` documentation says the same.
6. **Constructors.**
   - At most three positional arguments.
   - Beyond that, a parameter struct with public fields, a `Default`, and a
     `check -> Result`. Building is `new(params) -> Result` or struct update followed
     by `check`.
   - A boolean argument becomes an enum or a field.
7. **Containers.**
   - `stream(i)` returns a slice on every container, and `stream_view(i)` returns the
     `ndarray` view.
   - The length of a stream is `len()`, and the resize is `resize(..)`, everywhere.
8. **A verb names one operation.** The names of the verb traits (`modulate`,
   `demodulate`, `map`, `demap`…) are reserved for their meaning:
   - Bits to symbols is `map_bits`; symbols to bits by hard decision is
     `demap_bits`.
   - A second meaning of a name gets another name.
9. **Python.**
   - Signals and multi-antenna arrays are stream-major, `(n_streams, n)`, everywhere;
     a channel matrix is `(n_rx, n_tx)`.
   - The random generator is the last parameter, named `rng`.
   - Tuning parameters are keyword-only.
   - A function returns one type, whatever its arguments.
   - A closed form is named `theory_*` and accepts arrays for its SNR argument.
   - Each item has the name of its Rust item, unless the Python idiom requires
     otherwise (`__repr__`, `n_streams` properties).

## Phase A — Rust (one agent; bindings call sites updated, the Python API unchanged)

### 1. Rules

The rules above go into `docs/ARCHITECTURE.md`, and the consolidated items of
`core #22/#23/#28`.

### 2. Units

| Where | From | To |
|---|---|---|
| `Cdl` | `carrier_frequency` | `fc_hz` |
| `Cdl` | `velocity` | `velocity_mps` |
| `Cdl` | `delay_spread` | `delay_spread_s` |
| `Cdl` | `max_doppler()` | `max_doppler_hz()` |
| `Ray` | `doppler` | `doppler_hz` |
| `WdmSpectrum` | `freqs` | `freqs_hz` |
| `Numerology` | `subcarrier_spacing()`, `sample_rate()` | `subcarrier_spacing_hz()`, `sample_rate_hz()` |
| `Photodiode` | `dark_current` | `dark_current_a` |
| `Photodiode` | `temperature_c` | `temperature_k` (default 298.15 K; same results) |

- **Parameter names:** units added in the sync functions (`snr_lin`, `es_n0_lin`), the
  converters, `phase_noise(ts_s)`, `wdm_grid(spacing_hz)`, `freq_shift(offset_hz)`,
  `delay_fft(delay_s)`, `aclr_db(bandwidth_hz, offset_hz)` and `tone_frequency_crb`.
  Sources: `core #28`, `waveform #7`, `channel #9`.
- **`fftfreq` and `angular_frequencies`:** `fftfreq` takes a spacing, and
  `angular_frequencies` a rate. Document both, and name their parameters.

### 3. Vocabulary

Rule 2 applied:
- `samples_per_symbol` → `sps`;
- `rrc_taps(alpha)` → `rolloff`;
- `sigma2` → `noise_var`;
- `n_data` of `CeOfdm` → `n_subcarriers`;
- `n_data` of `OfdmFrame` (symbols per antenna) → `n_symbols`;
- CPM `sps` stays: it is the shaping rate.

### 4. Jacobian

The `com_math::Jacobian` enum of rule 4, used by `max_star`, `bcjr`, `TurboDecoder`
and `Constellation::llr`, and through the com-spec coding blocks (`fec #11`).

### 5. Errors

Rule 5 applied across the crates:
- `lama`/`oamp` lengths → `Shape`;
- an empty input to `welch`, `preprocess`, `Cumulants::estimate` and
  `fourth_power_foe` → `Shape`, as interpolation already does;
- domain errors of `stability::check` → `Config`;
- the documentation updated with each change.

Sources: `core #23`, `fec #9`, `waveform #9`.

### 6. Constructors

Rule 6 applied:

| Item | Change |
|---|---|
| `Ufmc::new` (5 arguments) | parameter struct |
| `Tdl::new` (5) | `TdlParams` |
| `Cdl::new` (5) | `CdlParams`, which also holds the Doppler, now set after construction |
| `Cdl::custom` (7) | `CdlParams` |
| `DelayDopplerModel::new` | parameter struct |
| `PilotPattern::from_name` | one constructor per pattern, taking only what it reads |
| `OfdmFrame::detect` | options struct; the `alamouti: bool` becomes a field or an enum |
| `estimate_port` | options struct |
| `multiplex`, `ser_receiver`, `ber_receiver`, `fer_interference` | grouped arguments |
| `lama`/`oamp` and their state evolutions | one `MessagePassing` configuration; `domain` at one place |
| `Pas` | check that `with_bit_mapper` follows the rule |
| `Rapp`, `Saleh`, `Mzm` | keep struct update followed by `check` |

Sources: `waveform #13`, `channel #10`, `fec #15`.

### 7. Containers

Rule 7 applied (`core #22`); every call site is updated.

### 8. Names

- **Bits and symbols (rule 8):**
  - `Constellation::modulate`/`demodulate` → `map_bits`/`demap_bits`, and the same for
    `Dpsk`;
  - `Ccdm::matched` → `match_bits` (it pairs with `dematch`).
- **CE-OFDM:** `demodulate_with(r, receiver)` → `demodulate_with_receiver`, and the
  duplicate `demodulate(r, unwrap)` is removed. `Ofdm::demodulate_with` keeps its PAPR
  meaning (`waveform #15`).
- **Other renames:**
  - `pmd::Manakov` → `ManakovPmd` (`channel #13`);
  - `com_mapping::theory::Fading` → `RateChannel`, so the name is left to the trait
    `com_channel::fading::Fading`.
- **Puncturers:** `dvbs_*` and `ieee80211_*` (`fec #24`). A pattern used by both
  standards is defined once; the second name becomes a documented alias function, or
  the patterns are renamed by rate. The agent decides and records why.
- **SNR conventions (`waveform #6`):**
  - add `com_mapping::theory::ser_esn0`, the Es/N0 entry point;
  - CE-OFDM `theory::ser` → `ser_esn0`;
  - the CSS closed forms name their per-sample SNR `snr_db`, in the docs;
  - `ber_orthogonal_noncoherent` moves next to `ser_orthogonal_noncoherent` in
    com-mapping.

### 9. Optical signatures

One in-place, `Result`-returning shape for the propagators (`channel #13`):
- `Ssfm::propagate`;
- `Edfa::apply`;
- `PmdLink::apply`;
- `ManakovPmd`.

### 10. Specification blocks

These keys are renamed, and the specs in `specs/` updated (`spec #13`):

| Block | From | To |
|---|---|---|
| cfo | `hz`, `omega` | `offset_hz`, `omega_rad` |
| phase_noise | `sigma` | `sigma_rad` |
| — | `tau1`, `tau2` | `tau1_s`, `tau2_s` |
| — | `max_nl_phase` | `max_nl_phase_rad` |
| cdl | `spacing` | `spacing_wavelengths` |
| — | `coherence` | `coherence_samples` |

Then `python/comrs/blocks.py` is regenerated.

### 11. Results

The results stay unchanged:
- every CI spec gives the same records;
- the validation tests touched are rerun in release and give the same values.

## Phase B — Python (one agent, after A: bindings, `python/`, `tests/`, `docs/python/`)

1. Rule 9 applied:
   - the antenna layout of `mimo_detect`/`mimo_llr`/`mimo_estimate` (`bindings #2`);
   - return types that no longer depend on the arguments (`bindings #12`):
     `Ofdm.modulate`, `papr_db`, `mimo_channel`, `mimo_capacity`,
     `ChannelTaps.frequency_response`, `read_sigmf`, the two Manakov propagators;
   - `theory_*` names and arrays (`bindings #13`);
   - the generator position (`bindings #22`);
   - keyword-only tuning parameters (`bindings #23`, `python #6/#20`);
   - units (`bindings #24`), the Rust/Python name alignment (`bindings #25`) and the
     iteration names (`bindings #15`).
2. **NR grid layout (`bindings #21`).** A 2-D `(n_subcarriers, n_symbols)` array
   everywhere, with indices given as `(k, l)` pairs, so that `grid.ravel()` is never
   needed. The same holds for `ResourceGrid`.
3. **The Phase A renames, reflected in Python** (`blocks.py`, the stubs, the docs, the
   gallery and the notebooks).
4. **Leftovers of the slices:**
   - `Stop.to_spec()`;
   - a public grid expander, `com_spec::grid_conditions`, with its binding, so that
     `sim.grid` and `spec.settings` no longer build a fake spec;
   - the `Raises:` sections of `sim.rs` and `spec.rs`, removed from `RAISES_PENDING`;
   - the Python-side checks dropped where the crates now check (`ddpll` pilots,
     `phase_noise` fs, `Cdl` spacing and frequency, `ResourceGrid.extract`);
   - the palette aliases `COBALT`/`GREY_WARM` removed;
   - the ~20 hand-built result dicts (`bindings #26`), on `Serialize` where the crate
     types allow it.
5. **Results.** The gallery and the notebooks give the same numbers.

## Phase C — leftovers in the crates (one agent, in parallel with B; no bindings, no `python/`)

1. **com-spec.**
   - Split `blocks/wireless.rs`, `blocks/coding.rs` and `blocks/schema.rs` into
     sub-modules (`spec`, "not done").
   - Split `search` and `Runner::run`.
   - Add `com_dsp::filter::upfirdn` and use it in `blocks/dsp.rs` (`core #26`).
   - Do not touch `run.rs`, `spec.rs` or the grid code, which belong to phase B.
2. **`viterbi_viterbi` for BPSK ±1.** The offset of a constellation whose M-th power
   is positive (`core #18`, the documented caveat). Either the offset follows from the
   constellation, or BPSK gets a correct path; with a test against OptiCommPy's
   convention.
3. **Validation cards** for the tests that read published figure values (`channel #5`):
   - cimini1985, bouhadda2014, nissel2017, raviteja2019;
   - alhabash2001, komine2004, carena2014;
   - bocherer2015 Tables VI–X, liang2025.

   Each card states the same tolerances as the current asserts, and the asserts move
   to `Card::check`.
4. **The 1e-300 floors** left unnamed in blind, linalg and nearest (`core`): one named
   constant per meaning.

Not in this batch: two decisions for the author, which change core traits.
- ADR 0012, proposed: symbols given as indices for CSS and CPM.
- `Demap` taking priors and the max-log choice (`fec #13`).
