# com-rs

Link-level simulation of digital communication systems (wireless and optical) in Rust,
with Python bindings. The architecture, naming, documentation, quality and testing
rules are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); its exceptions are recorded
in [docs/adr/](docs/adr/). Read the guide before adding a block or a crate.

## The two rules to keep in mind

- **Dependencies point downward only.** Layers: 0 `com-core`, `com-math`; 1 `com-dsp`;
  2 `com-fec`, `com-mapping`, `com-waveform`, `com-mimo`, `com-channel`,
  `com-hardware`, `com-metrics`; 3 `com-sim`; 4 `com-spec`, `com-systems`, `comrs`.
  Crates of one layer never depend on each other, except `com-waveform`,
  `com-metrics` (ADR 0007) and `com-mimo` (ADR 0008) towards `com-mapping`,
  `com-waveform → com-mimo` (ADR 0009) and `com-spec → com-systems` (ADR 0010). A new
  edge needs an ADR.
- **Placement:** put code by what it does in the chain; if it depends on a specific
  waveform, it lives under that waveform; if something else could reuse it, it moves
  one layer down. Numerical method without communication meaning → `com-math`; type or
  trait every block needs → `com-core`; medium → `com-channel`; component →
  `com-hardware`.

## Before adding a function or a block

Read [docs/API.md](docs/API.md), the generated index of every public item (signature
and first line of documentation): reuse the prototype that exists, or move it one
layer down, instead of writing it again. `just api-index` regenerates the index (CI
checks it is up to date); `just api-duplicates` lists the names defined in several
modules. When a feature comes from a paper or a standard, name the reference in the
reply so that it can be downloaded into `references/` and the code checked against
its figures and tables (`specs/validation/`, tests named after the reference).

## Before declaring a task done

Run `just check`: the checks of CI, in its order.

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test` (the default members: every crate but the Python extension)
- `comrs verify` of the validation specifications that CI checks (`ci_specs` of the
  `justfile`, `VALIDATION_SPECS` of `.github/workflows/ci.yml`)
- `cargo doc --workspace --no-deps --exclude com-py` with the KaTeX header and
  `-D warnings`
- every crate ships the same `katex-header.html` as `docs/katex-header.html`
- `ruff check` and `ruff format --check` of `python tests docs/python tools`
- `python3 tools/api_index.py --check` (docs/API.md is up to date)
- `python tools/literature_table.py --check` (docs/LITERATURE.md is up to date and every
  test named in `validation/literature.yaml` exists)
- `python tools/maturity_table.py --check` (docs/MATURITY.md and the maturity tables of
  README.md and docs/python/maturity.rst are generated from `validation/maturity.yaml`,
  and every test it cites exists)

CI also runs the Python tests (`just test`: `maturin develop`, then `python -m pytest`;
`just setup-py` installs the dependency groups), `just deny` and the MSRV build.
Statistical validation: `cargo test --release -p com-validation -- --ignored`.

## Notes

- com-rs contains no block of another project: an external crate adds its blocks with
  `com_spec::blocks::register` and ships its own binary (ADR 0006).
- Random blocks take `&mut SimRng` (xoshiro256**, ADR 0002); results must not depend on
  the thread count. Kernels that produce published results keep their order of
  operations: no `mul_add` or `f64::midpoint` (ADR 0005).
- Every formula is LaTeX in the rustdoc (`$…$`, `$$…$$`, braced subscripts, `\\` before
  punctuation); every crate ships `katex-header.html` (`just katex` refreshes them).
