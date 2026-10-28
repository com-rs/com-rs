# 0009 — `com-waveform` depends on `com-mimo`

Status: accepted (2026-10-26) — new sibling dependency in layer 2

## Context

The precedent table of `docs/ARCHITECTURE.md` places per-subcarrier MIMO detection in
`com_waveform::ofdm::detection`, as a thin wrapper applying the detectors of
`com_mimo` across the resource grid. That wrapper needs a `com-waveform → com-mimo`
edge, which the crate table does not list.

## Decision

`com-waveform` depends on `com-mimo` for the per-resource-element detection of OFDM only:
`com_waveform::ofdm::detection` (the detectors of `com_mimo::detection` across the grid)
and `com_waveform::ofdm::link` (the receiver of `com_mimo::receiver`, whose
`Receiver::detect_uses` detects the data resource elements of a frame). The detectors
and the receiver stay waveform-agnostic in `com-mimo`; the resource mapping stays in
`com-waveform`.

## Consequences

`com-mimo` must never depend on `com-waveform` (it does not: it depends on layer 0 and
on `com-mapping`, ADR 0008). Layer 2 now has four sibling edges, all
pointing to `com-mapping` or `com-mimo`, both free of waveform knowledge.
