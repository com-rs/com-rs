# 0007 — `com-metrics` depends on `com-mapping`

Status: accepted (2026-10-26) — new sibling dependency in layer 2

## Context

The guide allows one sibling edge in layer 2, `com-waveform → com-mapping`. The metrics
of com-rs measure symbol streams against a constellation: decisions for the SER, the
rotation of QAM/PSK, the bit labels and LLRs of the GMI, the posterior rates.

## Decision

`com-metrics` depends on `com-mapping`. The closed forms of the error and information
rates are properties of a constellation and its labelling: they live in
`com_mapping::theory`, and `com-metrics` re-exports them (`theory_ber`, `theory_ser`,
`theory_mi`, `theory_gmi`), so that `com-waveform` (CE-OFDM theory) uses them without a
`waveform → metrics` edge.

## Consequences

Layer 2 has two sibling edges, both towards `com-mapping`, which stays a leaf of the
layer (it depends on layers 0 and 1 only).
