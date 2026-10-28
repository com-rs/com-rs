User guide
==========

A link is a chain of blocks applied to NumPy arrays; a *campaign* evaluates that link
on a grid of conditions (an SNR, a launch power, a modulation order…), as many times
as a stopping rule requires, and reports every metric with a confidence interval.
``comrs`` offers two ways to run one:

- :doc:`monte_carlo`: the trial is a Python function, run by :mod:`comrs.sim` —
  the quickest way to prototype;
- :doc:`specifications`: the whole campaign is a YAML file run by the parallel Rust
  runner — the way to produce the numbers of a study.

A :doc:`chain <chains>` of named blocks bridges the two: built and inspected in
Python, it runs on the Rust runner and converts to and from a specification. The
blocks cover :doc:`wireless and coded links <wireless>` (antennas, fading, OFDM, MIMO
detection, channel codes) and :doc:`optical links <optical>` (WDM, dual polarisation,
nonlinear propagation, back-propagation).

Both draw trial ``t`` of condition ``i`` from the same stream
``Rng.stream(seed, name, i, t)``, so a scenario gives the same draws in both, and the
results never depend on the number of threads.

.. toctree::
   :maxdepth: 1

   monte_carlo
   chains
   analysis
   wireless
   optical
   specifications
   recordings
