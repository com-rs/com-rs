# 0010 — `com-spec` depends on `com-systems`

Status: accepted (2026-10-27) — new edge between two crates of layer 4

## Context

`com-systems` holds the standard-compliant chains (`com_systems::dvbs2`), as the guide
requires: what only makes sense for one standard goes there, not to the functional
crates. The declarative campaigns of `com-spec` must run these chains as blocks
(`dvbs2_tx`, `dvbs2_rx`), and both crates sit in layer 4, whose crates do not depend on
each other.

## Decision

`com-spec` depends on `com-systems`. The blocks of `crates/com-spec/src/blocks/systems.rs`
are thin wrappers that parse their parameters and call the chains; the chains keep no
knowledge of the specifications, frames or metrics of `com-spec`.

## Consequences

`com-systems` must never depend on `com-spec`: it depends on layers 0 to 3 only. The
facade `comrs` re-exports both. A standard chain becomes available to specifications by
a block of `blocks/systems.rs`, like any other block (ADR 0004).
