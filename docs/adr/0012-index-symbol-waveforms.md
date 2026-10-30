# 0012 — Waveforms of index symbols (CSS, CPM) and the verb traits

Status: proposed (2026-10-30) — would change a core trait (`docs/ARCHITECTURE.md`,
"Traits")

## Context

`Modulate` and `Demodulate` of `com-core` map `Symbols` (complex values, streams ×
symbols) to a `Signal` and back. Five modems of `com-waveform` map complex symbols and
implement them: OFDM, UFMC, OTFS, CE-OFDM and FBMC (the OQAM staggering inside the
trait), each at its `sample_rate`, the input rate checked by `demodulate_into`.

Two waveforms carry the index of a symbol instead of a point of a constellation:

- chirp spread spectrum (`css::Css`): the symbol $a \in \{0, \dots, M - 1\}$ selects a
  chirp, and the noncoherent receiver returns the index of the strongest bin
  (`modulate(&[usize])`, `demodulate -> Vec<usize>`, `energies` for soft values);
- continuous-phase modulation (`cpm::Cpm`): the symbol is an index of the $M$-ary
  alphabet $\{\pm 1, \pm 3, \dots\}$, the phase state an exact integer, and the
  detectors return indices (`ml_detect`, `differential_detect`).

`Symbols` cannot carry these indices without a convention (an index as the real part
of a `C64`) that every caller would have to know, and the soft output of these
receivers (the $M$ energies of CSS, the path metrics of CPM) is not a symbol either.
The specification blocks call the slice methods, so nothing is lost today, but a chain
of verb-trait blocks cannot contain CSS or CPM.

## Options

1. **Keep the slice methods only** (the status quo): the waveforms stay outside the
   trait chains; documented in the crate rule of `com-waveform`.
2. **Indices in `Symbols`**, the index in the real part: no new type, but a silent
   convention, a loss of type safety, and a float round trip of an integer.
3. **A container of indices and a generic input of the traits.** A container `Labels`
   (streams × indices, `usize`) made by the `container!` macro of `com_core::signal`,
   next to `Bits`, `Llr` and `Symbols`, and the traits generic over their input or output
   with the present container as default:

   ```rust
   pub trait Modulate<In = Symbols> {
       fn modulate_into(&self, x: &In, out: &mut Signal) -> Result<()>;
       fn modulate(&self, x: &In) -> Result<Signal> { /* … */ }
   }
   pub trait Demodulate<Out = Symbols> {
       fn demodulate_into(&mut self, y: &Signal, out: &mut Out) -> Result<()>;
       fn demodulate(&mut self, y: &Signal) -> Result<Out> { /* … */ }
   }
   impl Modulate<Labels> for Css { /* … */ }
   impl Demodulate<Labels> for Css { /* the noncoherent decisions */ }
   ```

   The existing implementations and callers compile unchanged (the default parameter);
   `Labels` is also the natural output of hard decisions (`Constellation::decide`) and
   the input of `com_mapping::constellation::indices_to_bits`.
4. **Separate traits** `ModulateIndices` and `DemodulateIndices`: no change to the
   existing traits, but two more verbs for one role, against the rule "one verb trait per
   role".

## Decision (proposed)

Option 3: `com_core::signal::Labels` and the default type parameters on `Modulate` and
`Demodulate`; `Css` and `Cpm` implement `Modulate<Labels>` and `Demodulate<Labels>` at a
`sample_rate` of their configuration (one sample per chip, or `sps` per symbol, at 1 Hz
by default), the input rate checked as for the other modems. Their soft outputs
(`Css::energies`, the metrics of CPM) stay methods.

It is not implemented with this record: it changes a trait of `com-core`, whose owner
and every downstream crate must agree, and the bindings and the specifications would
gain a `Labels` conversion at the same time.

## Consequences

- One verb per role is kept, and a chain of trait blocks can hold every waveform of the
  crate.
- `Labels` adds a container to `com-core` (no algorithm) and a conversion in the
  bindings (`numpy` integer arrays).
- Until the decision is taken, the crate rule of `com-waveform` states that CSS and CPM
  expose slice methods only.
