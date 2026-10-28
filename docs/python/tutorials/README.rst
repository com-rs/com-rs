Tutorials
=========

Six guided scripts, each a complete program that runs as is (the figures of these
pages are produced when the documentation is built). They go from the first call to
a campaign run from the command line, then to the tools that look inside a chain:

1. **A first link** — bits, constellation, noise, decisions, the closed form, and the
   same link as a chain on the parallel runner.
2. **A classical single-carrier chain** — pulse shaping, matched filter, decimation;
   the eye diagram and the spectrum; phase noise and carrier recovery as variants.
3. **Writing a block in Python** — a transmitter IQ imbalance and its blind
   compensation as Python blocks, checked against the native blocks trial for trial.
4. **Specifications and the command line** — the chain as a YAML file, expected
   results, the operating point of a metric, and the ``comrs`` command.
5. **Where the time goes** — the profile of a chain, block by block, on both
   runners; two settings compared; the price of a Python block.
6. **Looking at the signals** — the trace of one trial, probes during a campaign,
   ``dump`` recordings at a point of the link, and a Python block as a tap.

The :doc:`user guide <../user_guide/index>` is the reference for what the tutorials
touch; the :doc:`gallery <../auto_examples/index>` has the classical curves.
