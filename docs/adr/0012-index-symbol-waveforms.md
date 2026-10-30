# 0012 — Waveforms of index symbols (CSS, CPM) and the verb traits

Status: accepted (2026-10-30) — changes the core traits `Modulate`, `Demodulate` and
`Demap` (`docs/ARCHITECTURE.md`, "Traits")

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

## Decision

Option 3, implemented as proposed:

- `com_core::signal::Labels`, streams × indices (`usize`), made by the `container!`
  macro next to `Bits`, `Llr` and `Symbols` (in the prelude);
- default type parameters on the waveform traits, `Modulate<In = Symbols>` and
  `Demodulate<Out = Symbols>` (`demodulate` requires `Out: Default`); the five modems
  of complex symbols and their callers are unchanged;
- `Css` implements `Modulate<Labels>` (`Css::modulate`) and `Demodulate<Labels>` (the
  noncoherent decisions of `Css::demodulate`) at a new public field `sample_rate` (1 Hz
  in `Css::lora`), checked by `Css::check`;
- `Cpm` implements `Modulate<Labels>` (`Cpm::modulate`) and `Demodulate<Labels>` (the
  coherent maximum-likelihood sequence detector `Cpm::ml_detect`) at
  `Cpm::sample_rate`, 1 Hz from `CpmConfig::build` and set by `Cpm::with_sample_rate`
  (as `Fbmc`, whose fields are private too; `CpmConfig` keeps its fields, so that its
  struct literals are unchanged);
- both take one stream and refuse a signal at another rate (`check_input` of the other
  modems). The slice methods stay, and the trait methods call them: the samples and the
  decisions are those of the slice methods bit for bit (tests
  `the_verb_traits_are_the_slice_methods` of `css` and `cpm`; the CSS validation of
  `validation/tests/css.rs` runs through the traits with the same values). The soft
  outputs (`Css::energies`, the path metrics and phase states of CPM) and the other
  detectors (`msk_detect`, `differential_detect`) stay methods.

The same change of the core traits lets `Demap` take the a priori log-probabilities of
the points (probabilistic shaping) and the choice of the Jacobian logarithm, which the
specification block `demap` needed and which made it bypass the trait (review of the
FEC crates, item 13):

- `Jacobian { Exact, MaxLog }` moves from `com-math` to `com_core::traits`, since the
  trait names it; `com_math::Jacobian` re-exports it, so every path compiles unchanged;
- `com_core::traits::Demapping<'a> { log_priors: Option<&'a [f64]>, jacobian }`, uniform
  and exact by default; the log-priors are borrowed, so a caller computes them once and
  the demapper allocates no table;
- `Demap::demap_with_into(&self, y, noise_var, demapping, out)` is the required method;
  `demap_into` (the default `Demapping`), `demap_with` and `demap` are provided;
- `Constellation` implements it on the kernel of its `llr_into` paths, the metric row of
  a symbol on the stack up to `demapper::STACK_POINTS` = 1024 points, so that
  `demap_into` and `demap_with_into` allocate nothing once the shape of `out` is stable;
  with the log-priors of `log_priors(&c.priors(Some(px))?)` its LLRs equal those of
  `Constellation::llr(.., Some(px), ..)` bit for bit, exact and max-log (test
  `demap_with_equals_llr_with_and_without_priors`);
- the block `demap` of `com-spec` calls the trait when the noise variance is common to
  the symbols; one variance per symbol (after a linear MIMO detector) keeps
  `Constellation::llr_with_variances`, which the trait, with one `noise_var`, does not
  express.

A companion trait (`DemapWithPriors`) was rejected for the rule "one verb trait per
role"; the priors and the Jacobian are grouped in one `Copy` struct rather than two
more arguments, so that another option of a demapper does not change the signature.

## Consequences

- One verb per role is kept, and a chain of trait blocks can hold every waveform of the
  crate.
- `Labels` adds a container to `com-core` (no algorithm). The Python API is unchanged:
  the bindings of `Css` and `Cpm` call the slice methods, and `Css` is built at 1 Hz.
- A struct literal of `Css` names `sample_rate` (or updates `Css::lora`); an
  implementation of `Demap` outside the workspace implements `demap_with_into` instead
  of `demap_into`. Both are changes of the unreleased 0.2.0.
