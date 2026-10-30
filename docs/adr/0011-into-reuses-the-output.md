# 0011 — `_into` reuses the output; workspaces where profiling asks for them

Status: accepted (2026-10-30) — refines `docs/ARCHITECTURE.md` ("Traits", "Performance")

## Context

The guide said that every verb trait has an `_into` method "so hot loops allocate
nothing". The audit of 29 October 2026 found that no implementation keeps that promise:
the `_into` methods build a new container (`Bits::from_streams`, a new `Signal`) and
replace `out`, and `BlockCode` (`encode_block(&self, …) -> Vec<u8>`,
`decode_block(&self, …) -> Vec<u8>`) cannot keep a workspace between frames. Keeping the
promise literally would mean a workspace in every block and a breaking change of
`BlockCode`; the same audit found that the cost of a trial is dominated by algorithms of
the wrong complexity (polar decoding, μ-law companding, the OTFS channel), not by
allocations.

## Decision

1. **The output is reused.** An `_into` method writes into `out` in place: it resizes
   `out` (`Bits::resize`, `Signal::reshape`) and fills its storage, so a caller that
   passes the same buffer at every trial allocates no output once the shape is stable.
   It never replaces `out` by a new container.
2. **Scratch may allocate, unless profiling says otherwise.** Temporary buffers inside a
   block are allowed. A block whose allocations show in a profile (`profile: true` of a
   specification, `samply`, the criterion benchmarks) keeps them in a workspace: in the
   block when the method takes `&mut self`, in a thread-local or the plan cache
   (`com_math::fft`) when it takes `&self`.
3. **`BlockCode` gains slice methods with defaults.** `encode_block_into(&self, info,
   out: &mut [u8])` and `decode_block_into(&self, llr, out: &mut [u8])` default to the
   allocating methods and a copy; the container traits call them on the stream slices
   of `out`. A code overrides them when a profile shows the copy: none does yet, since
   the decoders allocate their buffers once per frame (never per iteration, step or
   path) and a workspace kept between frames of the polar decoder measured no gain. No
   existing signature changes.
4. **The allocating methods stay** the convenient form, implemented by the `_into` ones.

## Consequences

The promise of the guide becomes precise and checkable: a test can pass the same `out`
twice and compare its storage address (`as_ptr`) when the shape is unchanged. Blocks are
fixed one at a time, the hot ones first, with the gain measured as the guide requires
for every optimisation. A block that keeps a workspace behind `&self` must not depend on
its previous call (results are a pure function of the inputs, ADR 0002).
