# 0002 — Keep xoshiro256** as `SimRng`

Status: accepted (2026-10-26) — exception to the first version of `docs/ARCHITECTURE.md`
("Randomness"), which now states this decision

## Context

The first version of the guide set `SimRng = rand_chacha::ChaCha12Rng` with
`batch_rng` built by `set_stream`. com-rs uses xoshiro256** seeded through SplitMix64, with streams derived
per `(seed, experiment, condition, trial)`. It is the generator of the earlier projects
of the author (a stream drawn here is the same as one drawn there), and every stored
result and validation of com-rs depends on its draws.

## Decision

`com_core::rng::SimRng` is an alias of `com_core::rng::Rng` (xoshiro256**).
`batch_rng(master_seed, batch)` is provided with the guide's signature, as the stream
`(master_seed, "", 0, batch)`. Everything else in the guide holds: no global or
thread-local generator, `&mut SimRng` passed to every random block, streams keyed by
batch or trial index and never by worker.

## Consequences

- Results stay reproducible across versions and identical to the streams of those projects.
- No dependency on `rand`. xoshiro256** is not cryptographic, which simulation does not
  need.
- Moving to ChaCha12 later changes every random result: it needs its own ADR and a
  regeneration of the stored results.
