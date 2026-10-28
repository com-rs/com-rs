## What and why

<!-- What the change does, and why. Link the issue it closes. -->

## Reference

<!-- For a new block or a changed result: the paper or standard (authors, venue, year,
equation, figure or table) and how it is checked: a specification of specs/validation
with `expect`, or a `literature` test. "None" for a change without a numerical result. -->

## Checklist

- [ ] `just check` passes (fmt, clippy, tests, `comrs verify`, docs, ruff, API index)
- [ ] the Python tests pass (`just test`) if the bindings or `python/` changed
- [ ] every new public item is documented, with its formula in LaTeX and its reference
- [ ] `docs/API.md` regenerated (`just api-index`) if the public API changed
- [ ] an ADR in `docs/adr/` for a new crate, a new dependency between crates of a layer, a change to a core trait, or an exception to `docs/ARCHITECTURE.md`
- [ ] `CHANGELOG.md` updated, with a migration note for a breaking change
