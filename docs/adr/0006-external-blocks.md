# 0006 — Blocks of other crates join the registry through `blocks::register`

Status: accepted (2026-10-28)

## Context

A research project built on com-rs (a learned receiver, a proprietary model, a block
that depends on code not published on crates.io) needs to run in the declarative
campaigns, the command line and the checks of ports, without com-rs depending on it.
A dependency of com-rs on such a crate would make the workspace unbuildable for
everyone else, and crates.io refuses path dependencies in published crates.

## Decision

`com-spec` exposes an extension point: `com_spec::blocks::register(Extension { info,
build })` adds a block to the registry, listed by `comrs blocks`, described by the JSON
Schema, checked by `check_link` and built like the others. `com_spec::cli::main` is the
command line as a function, so that the crate of the extension ships its own binary
(the `comrs` command with its blocks). The dependency points from the extension to
com-rs, never the reverse: com-rs contains no block of another project.

## Consequences

The workspace builds and tests from a clean checkout, with nothing outside it.
An extension is versioned with com-rs's `com-spec` and follows its `Block` trait and
`BlockInfo` description; an unknown block type in a specification is an error that
names the known blocks.
