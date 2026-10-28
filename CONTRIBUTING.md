# Contributing to com-rs

Thank you for helping. com-rs aims to be a reference link-level simulator: a
contribution is judged first on the evidence that it is right. This page says how to
work on the code, how a block is validated, and how decisions are recorded.

## Before writing code

1. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the layers of the workspace,
   the placement rules (where a function belongs), the naming conventions, the
   documentation template and the tests every block family must pass. Its exceptions
   are the decisions of [`docs/adr/`](docs/adr/).
2. Search [`docs/API.md`](docs/API.md), the generated index of every public item,
   before adding a function or a block: it may exist under another name, or a shared
   helper may already do part of the work (`just api-duplicates` lists the names
   defined in several modules).
3. For a change of some size, open an issue first, so that its place and its
   reference can be agreed on.

Two rules matter most: dependencies point downward only (layer 0 `com-core`,
`com-math`; 1 `com-dsp`; 2 the link blocks; 3 `com-sim`; 4 `com-spec`, `com-systems`,
`comrs`), and code is placed by what it does in the chain.

## Setting up

```bash
git clone https://github.com/com-rs/com-rs && cd com-rs
cargo test                                   # the Rust crates
python -m venv .venv && source .venv/bin/activate
just setup-py                                # maturin and the dependency groups (pip >= 25.1)
just test                                    # builds the extension, then Rust and Python tests
```

[`just`](https://github.com/casey/just) runs the tasks of the [`justfile`](justfile).
## Before opening a pull request

Run `just check`. It runs what CI runs, in its order: `cargo fmt --check`, clippy with
`-D warnings`, the tests, `comrs verify` of the validation specifications, the
documentation with `-D warnings` and its KaTeX header, `ruff`, and the check that
`docs/API.md` is up to date (`just api-index` regenerates it). CI also runs the Python
tests on Linux, macOS and Windows.

- Every public item is documented with the formula it implements, in LaTeX
  (`$…$`, `$$…$$`), and its reference.
- Random blocks take `&mut SimRng`; results must not depend on the number of threads
  (ADR 0002). Kernels that produce published results keep their order of operations:
  no `mul_add` or `f64::midpoint` (ADR 0005).
- Add an entry to [`CHANGELOG.md`](CHANGELOG.md) (Keep a Changelog), with a migration
  note for a breaking change.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), scoped by crate or
area, as in `git log`: `feat(fec): layered min-sum LDPC decoder`,
`fix(spec): number the trials so that dump saves them by index`,
`test(ce_ofdm): …`, `docs(python): …`, `refactor: …`, `ci: …`. The subject says what
the change does; the body says why, and names the reference a result was checked
against. `main` is always green.

## Validating a block against a reference

A block is merged when it is correct by construction (unit and property tests), by
physics (invariants) and by statistics (reference results). The last step ties it to
the literature:

1. **Cite the reference** in the rustdoc of the item: authors, title, venue, year, and
   the equation, figure or table that the block reproduces. A standard is cited with
   its number, version and section. Add the document to
   [`references/README.md`](references/README.md) with what it is needed for (the
   files themselves are not redistributed).
2. **Check it**, in one of two ways:
   - a specification under [`specs/validation/`](specs/validation/) with an `expect`
     section: the closed form or the curve read from the publication, its `source`
     and a tolerance justified in the header comment (sampling error of the published
     points, confidence intervals of ours). `comrs verify` runs it and fails if a
     check fails; add it to `VALIDATION_SPECS` of `.github/workflows/ci.yml` and to
     `ci_specs` of the `justfile` if it runs in a few seconds;
   - or a unit test in a `mod literature` of the module, named after the reference,
     that checks the tables, the small examples or the closed forms of the document
     verbatim, and names the figure, table or equation it checks.
3. **Report the agreement** in the tables of the README ("Validation against the
   literature"). A disagreement is a finding: explain it, or report it as an issue.

Coded systems without a closed form are compared with reference implementations
(AFF3CT) run with the same parameters; slow statistical checks go to the `validation/`
crate (`cargo test --release -p com-validation -- --ignored`, run by hand or by the
workflow `validation.yml`, started by hand).

## The test oracles

`tests/test_oracle.py` compares the deterministic blocks with OptiCommPy and
`tests/test_gnpy_oracle.py` the Raman amplification with GNPy. Their outputs are
recorded fixtures of `tests/oracle/` (`<name>.json`, with the tool, its version and
the date, and `<name>.npz` for the arrays), so the tests need neither tool: they are
never imported by the package (OptiCommPy is GPL-3.0). To re-record them after a
change of the tool or of the inputs:

```bash
pip install --group oracle
COMRS_RECORD_ORACLE=1 python -m pytest tests/test_oracle.py tests/test_gnpy_oracle.py
```

Re-recording is a deliberate act: the commit message says why and names the version
of the tool. Re-recording an unchanged oracle produces no diff.

## Architecture decision records

An ADR is required for a new crate, a new dependency between crates of one layer, a
change to a core trait, or an exception to `docs/ARCHITECTURE.md`. Add
`docs/adr/NNNN-short-title.md` (the next number), one page: a status line (accepted,
with the date; superseded by a later ADR when it is), the context, the decision and
its consequences. An ADR is never deleted: a later one supersedes it, and both say
so.

## Reporting problems

A disagreement with a reference is a bug: open a "Validation discrepancy" issue with
the specification that reproduces it. Security issues are reported privately (see
[`SECURITY.md`](SECURITY.md)). Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## Licence

Contributions are released under the licence of the project, MIT or Apache-2.0 at
the user's option, without additional terms (Apache-2.0, section 5).
