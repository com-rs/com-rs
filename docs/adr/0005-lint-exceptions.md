# 0005 — Exceptions to the workspace lints

Status: accepted (2026-10-26) — exception to `docs/ARCHITECTURE.md` ("Code quality standards")

## Context

The guide enables `clippy::pedantic` with `-D warnings`. On numerical code some of its
lints flag the idiom rather than a defect, and two of them change results.

## Decision

Besides `cast_precision_loss` (in the guide), the workspace allows:

- `suboptimal_flops` and `manual_midpoint`: `mul_add` and `f64::midpoint` change the
  rounding, hence published results;
- `cast_possible_truncation`, `cast_sign_loss`, `cast_possible_wrap`, `cast_lossless`:
  the index ↔ float conversions of numerical kernels;
- `similar_names`, `many_single_char_names`, `needless_range_loop`: the notation of the
  literature (`h`, `w`, `x`, `y`, indices shared by several arrays);
- `float_cmp`: exact comparisons are deliberate (zero tests, bit-identity tests);
- `too_many_lines`, `too_many_arguments`, `module_name_repetitions`,
  `must_use_candidate`, `return_self_not_must_use`, `missing_panics_doc`,
  `naive_bytecount`.

`clippy.toml` allows `unwrap`/`expect` in tests (`allow-unwrap-in-tests`) and lists the
proper names of the documentation (`doc-valid-idents`); integration tests, the
validation crate and the benchmarks allow them at the top of the file. Library code
keeps `unwrap_used = "deny"`; the few justified `expect`/`allow` carry a `reason`.

The Python bindings do not inherit the workspace lints: PyO3's macros expand to
`unsafe` code, so `unsafe_code = "forbid"` cannot apply; they keep `missing_docs`,
`unreachable_pub` and `unwrap_used`.

## Consequences

`missing_errors_doc`, `doc_markdown` and the other pedantic lints stay enforced.
