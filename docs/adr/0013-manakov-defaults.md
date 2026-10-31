# 0013 — The Manakov solver departs from OptiCommPy's defaults

Status: accepted (2026-10-31) — changes the default results of `ManakovSsfm`, of the
block `manakov` and of `comrs.ManakovSsfm`

## Context

`com_math::nlse::ManakovSplitStep` was a transcription of OptiCommPy's `manakovSSF`,
checked against it to $10^{-8}$ on the same field (`tests/test_oracle.py`): the nonlinear
phase of a step is the trapezoidal mean of the power at its two ends, iterated until
the field converges, and the steps keep the largest nonlinear phase of a step at
0.02 rad. Comparing the solver with references that do not come from a simulator
showed that both choices make the results of WDM links depend on the step:

- the trapezoidal mean smooths the intensity that drives the cross-phase modulation
  when channels walk off within a step: a continuous probe 125 GHz from a modulated
  pump receives 0.447 of its XPM with steps of 0.5 km (eq. (9) of Chiang et al. 1996,
  `validation/tests/chiang1996.rs`), a five-channel link 0.9 dB too little
  interference (`validation/cards/dar2014_manakov.yaml`), and the error on the exact
  collision of two solitons 800 GHz apart is 6.4 times that of the scalar solver
  (`validation/cards/sinkin2003_exact.yaml`);
- the rule of 0.02 rad depends on the power and ignores the bandwidth: it leaves 0.45 dB
  of error on the SNR of `specs/wdm_gn.yaml` and `specs/raman_wdm.yaml` against
  converged steps, with either nonlinear step.

## Options

1. **Keep OptiCommPy's defaults** and document their limits: the results stay those of
   OptiCommPy, and a user who keeps the defaults gets up to 0.45 dB of error without a
   warning.
2. **Fixed short steps by default** (0.1 km): accurate, but 3 to 5 times slower than
   needed on long links, and still a choice the user cannot relate to an accuracy.
3. **The scheme of the scalar solver and a step rule with a stated accuracy**, keeping
   OptiCommPy's choices available by name.

## Decision

Option 3:

- `NonlinearStep::Midpoint`, the nonlinear phase of the power at the middle of the step
  (the symmetric scheme of eq. (3) of Sinkin et al. 2003, as `SplitStep`), without
  iterations, is the default; `NonlinearStep::Trapezoidal` is OptiCommPy's;
- `StepRule::FWM_DEFAULT` is the default: the rule of Musetti, Serena & Bononi (2018,
  eq. (13)), whose first step of every span bounds the four-wave-mixing phase across the
  signal bandwidth to 20 rad (about 0.01 dB of error on the SNR in WDM, Sec. IV-B), the
  next ones growing with the loss (constant local error, eq. (10)), the bandwidth
  measured once on the launched field; the steps are also kept below 0.02 rad of
  nonlinear phase, without which a single channel at high power gets 0.3 to 0.7 dB too
  much SNR (`validation/cards/musetti2018_fig7.yaml`);
- `StepRule::NonlinearPhase(2e-2)` with `NonlinearStep::Trapezoidal` reproduces
  `manakovSSF`, and the recorded comparison with OptiCommPy selects them.

## Evidence

| Reference | What it checks | Result |
|---|---|---|
| Exact two-soliton solution (Hirota 1973), collision of Sinkin et al. 2003 | the scheme, walk-off | scalar and midpoint exact to $3 \cdot 10^{-9}$, second order |
| Exact Manakov collision (Radhakrishnan et al. 1997, Fig. 1) | the coupling of the polarisations | the switching of eq. (16), 0.40 % against 0.39 % |
| Chiang et al. 1996, eq. (9) | XPM with walk-off | midpoint within 0.25 % at 0.5 km |
| Dar et al. 2014, Sec. 3 | the simulated interference of a WDM link | 9.95 and 5.78 dB against 10 and 6 dB |
| Musetti et al. 2018, Fig. 7 | the accuracy of the step rule | within 2.4 dB from −72 to −7 dB |
| GN model (Gaussian symbols) | the level of the interference | 0.16 dB (one polarisation), 0.17 dB (two) |

## Consequences

- The results of the Manakov solver with its defaults change: about 0.9 dB more
  interference than before on a five-channel link with long steps, now within 0.02 dB
  of converged steps; the WDM specifications use the defaults.
- The defaults cost 1.5 to 3 times the time of OptiCommPy's choices on the WDM
  specifications, which leave 0.45 dB of error; at equal accuracy (0.02 dB, which
  OptiCommPy's scheme reaches with steps of 0.1 km) they are 11 times faster on
  `specs/wdm_gn.yaml` (27 s against 291 s) and 6.4 times on `specs/raman_wdm.yaml`
  (21 s against 135 s); on one channel they are faster and as accurate
  (`docs/python/user_guide/optical.rst`).
- The step rule needs the bandwidth of the signal: measured on the launched field
  (`signal_bandwidth`, before any amplifier noise) or given (`bandwidth_hz`).
- The scalar `Ssfm` keeps OptiCommPy's constant steps (`ssfm`), its scheme being already
  the midpoint one.
