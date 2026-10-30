# Development tasks of com-rs (https://github.com/casey/just).
# `cargo` without `-p` works on the default members: every crate but the Python
# extension, which maturin builds.

# The specifications of specs/validation that CI verifies (VALIDATION_SPECS of
# .github/workflows/ci.yml: keep both lists equal)
ci_specs := "specs/validation/proakis_awgn.yaml specs/validation/information_rates.yaml specs/validation/hager2021_fig5.yaml specs/validation/msk_bpsk.yaml specs/validation/thompson2008_eq15.yaml"

# The development dependencies of the Python package, in the virtual environment
# (dependency groups of pyproject.toml: pip >= 25.1)
setup-py:
    python -m pip install --upgrade "pip>=25.1"
    python -m pip install maturin --group test --group lint --group docs

# Build the Python extension in the virtual environment
develop:
    maturin develop --release

# Rust and Python tests (doctests included)
test: develop
    cargo test
    python -m pytest

# The statistical validation (slow): simulated curves against closed forms
validation:
    cargo test --release -p com-validation -- --ignored

# The checks of CI, in its order: run them before declaring a task done (CI also runs
# the Python tests, `just test`, on three systems, plus `just deny` and the MSRV build)
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test
    cargo run --release -p com-spec --bin comrs -- verify {{ci_specs}}
    RUSTDOCFLAGS="--html-in-header {{justfile_directory()}}/docs/katex-header.html -D warnings" cargo doc --workspace --no-deps --exclude com-py
    for c in crates/*/; do cmp docs/katex-header.html "$c/katex-header.html"; done
    ruff check python tests docs/python tools
    ruff format --check python tests docs/python tools
    python3 tools/api_index.py --check
    python tools/literature_table.py --check
    python tools/maturity_table.py --check
    python tools/cards.py --check
    python tools/gen_stubs.py --check
    python tools/gen_blocks.py --check

# Licences and advisories of the dependencies (needs cargo-deny)
deny:
    cargo deny check

# Rerun the comparison of one validation card (validation/cards/<id>.yaml): its Python
# function, Rust test or specification
card id:
    python tools/cards.py run {{id}}

# Regenerate docs/API.md, the index of every public item (read it before adding one)
api-index:
    python3 tools/api_index.py

# Regenerate docs/LITERATURE.md, the comparisons with the literature, from
# validation/literature.yaml
literature:
    python tools/literature_table.py

# Regenerate docs/MATURITY.md and the maturity tables of README.md and
# docs/python/maturity.rst from validation/maturity.yaml
maturity:
    python tools/maturity_table.py

# Regenerate the stub of the extension and the classes of comrs.blocks (after changing
# a binding or a block; the extension built)
gen-python: develop
    python tools/gen_stubs.py
    python tools/gen_blocks.py

# The names defined in several modules: candidates for a shared helper
api-duplicates:
    python3 tools/api_index.py --duplicates

# Refresh the KaTeX header shipped by every crate (docs.rs builds crates in isolation)
katex:
    for c in crates/*/; do cp docs/katex-header.html "$c/katex-header.html"; done

# The examples (results under results/)
examples:
    cargo run --release -p comrs --example ber_awgn
    cargo run --release -p comrs --example coherent_link
    cargo run --release -p comrs --example ssfm_dbp

# Criterion benchmarks of the main blocks
bench-rust:
    cargo bench -p comrs

# The documentation of the Python package, with the gallery (docs/python/_build/html)
docs-py: develop
    sphinx-build -W --keep-going -b html docs/python docs/python/_build/html

# Open the Rust documentation
doc:
    cargo doc --workspace --no-deps --exclude com-py --open

# Run every specification of specs/ (reports under results/)
specs:
    cargo run --release -p com-spec -- run specs/*.yaml

# Check every specification (one trial per condition)
check-specs:
    cargo run --release -p com-spec -- check specs/*.yaml

# Check the results against the literature (specs/validation)
validate:
    cargo run --release -p com-spec -- verify specs/validation/*.yaml

# Speed against OptiCommPy (writes benchmarks/results.md)
bench: develop
    python benchmarks/bench_opticommpy.py

# Rewrite the synthetic capture of data/
make-capture:
    cargo run --release -p comrs --example make_capture -- data
