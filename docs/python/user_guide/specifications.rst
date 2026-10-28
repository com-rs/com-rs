Declarative campaigns
=====================

A specification describes a whole campaign — the grid, the link as a list of blocks,
the metrics, the stopping rule, the outputs and the expected results — in a YAML,
JSON or TOML file (or a Python ``dict``). :func:`comrs.run_spec` runs it on the
parallel Rust runner; the same file runs from the command line with ``comrs run``.

.. code-block:: python

   import comrs

   results = comrs.run_spec("""
   name: qam_awgn
   seed: 2024
   grid:
     m: [4, 16, 64]
     ebn0_db: {from: 0, to: 20, step: 2}
   link:
     - {type: source, constellation: {kind: qam, m: $m}, n_symbols: 4096}
     - {type: awgn_ebn0, ebn0_db: $ebn0_db}
   metrics: {report: [ber, ser]}
   stop: {min_errors: {ber: 200}, min_trials: 128, max_trials: 4000}
   callbacks:
     skip_below: {metric: ber, threshold: 1e-6, along: ebn0_db}
   """)
   results[0]["condition"], results[0]["metrics"]["ber"]

Every result is a ``dict`` with the condition, the status, the number of trials and
every metric with its value and confidence interval (``value``, ``lo``, ``hi``).

- ``profile: true`` (or ``run_spec(spec, profile=True)``, ``comrs run --profile``)
  times every block, reported as the metrics ``<label>.time_ms`` (see
  :doc:`analysis`);
- ``$name`` in a block is the value of the grid variable ``name``;
- :func:`comrs.spec_blocks` lists the blocks of the links and their parameters, with
  units and defaults; :func:`comrs.spec_conditions` the conditions of a grid;
- the JSON report records the library version, its commit, the seed and the
  specification itself, so that every number can be traced back to what produced it.

Expected results
----------------

An ``expect`` section lists results the campaign must reproduce — a value, a published
curve, or a closed form — with its source and a tolerance. :func:`comrs.verify_spec`
runs the campaign and checks them:

.. code-block:: yaml

   expect:
     - name: SER of square QAM (exact)
       source: Proakis & Salehi, Digital Communications, 5th ed., Sec. 4.3
       metric: ser
       theory: {function: ser, kind: qam, m: $m, ebn0_db: $ebn0_db}
       interval: 0.999          # the closed form lies in the 99.9 % interval

The specifications of ``specs/validation/`` reproduce Proakis' error rates, the
information rates of QAM by quadrature and figures of the literature; they run in CI.
The closed forms available to ``theory`` are the error rates over AWGN (``ber``,
``ser``), the information rates (``mi``, ``gmi``), the BER of BPSK and QPSK with
maximal-ratio combining of Rayleigh branches (``ber_rayleigh``, with ``branches`` and
``offset_db``) and the SNR of a WDM link by the GN model (``gn_snr_db``): see
``specs/mimo_diversity.yaml``, ``specs/ofdm_eva.yaml`` and ``specs/wdm_gn.yaml``.

Operating points
----------------

A ``search`` section replaces the sweep by the search of the value of a variable at
which a metric reaches a target — the :math:`E_b/N_0` required for a BER of
:math:`10^{-3}`, say — for every combination of the other axes
(:func:`comrs.search_spec`).
