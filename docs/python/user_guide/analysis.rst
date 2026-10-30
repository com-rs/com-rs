Analysing a campaign
====================

A campaign answers a question with numbers; the analysis afterwards looks at the
numbers (:class:`~comrs.Results`), compares variants (:func:`~comrs.compare`), measures
inside the link (probes) and looks at the signals (:class:`~comrs.Trace` and
:mod:`comrs.plot`).

Results
-------

:meth:`Chain.run <comrs.Chain.run>` returns :class:`~comrs.Results`: the list of one
``dict`` per condition of :func:`comrs.run_spec`, with the specification of the
campaign attached.

.. code-block:: python

   res = link.run(grid={"snr_db": range(0, 21, 2), "m": [4, 16, 64]},
                  metrics=["ber", "snr_db"],
                  stop={"min_errors": {"ber": 200}, "max_trials": 1000})

   res.variables, res.metrics                 # ['snr_db', 'm'], ['ber', 'snr_db']
   x, ber, lo, hi = res.series("snr_db", "ber", interval=True, m=16)
   res.where(m=64)                            # the conditions of 64-QAM
   res.to_frame()                             # a pandas DataFrame, one row per condition
   res.plot("snr_db", "ber", by="m")          # one curve per order, with the intervals
   res.save("results/qam.json")               # the numbers, the specification, the version
   res = comrs.Results.load("results/qam.json")

``Results.plot`` takes a closed form, ``theory=lambda x, m: …``, drawn as a dotted line
on every curve; ``min_count`` leaves out the points of too few errors.

Variants
--------

:func:`comrs.compare` runs several chains with the same grid, metrics and stopping rule
and gathers them, the name of every chain as the variable ``variant``:

.. code-block:: python

   res = comrs.compare(
       {"CDC": link, "DBP 1 step": link.replace(comp=B.ManakovDbp(steps_per_span=1)),
        "DBP 4 steps": link.replace(comp=B.ManakovDbp(steps_per_span=4))},
       grid={"power_dbm": range(-4, 9, 2)}, metrics=["snr_db"], stop={"max_trials": 4},
   )
   res.plot("power_dbm", "snr_db", by="variant")

Probes
------

A probe measures metrics at a point of the link during the whole campaign, on the
parallel runner: :meth:`Chain.probe <comrs.Chain.probe>` inserts a ``probe`` block after
a named block, and its metrics are reported as ``<block>.<metric>``.

.. code-block:: python

   probed = bicm.probes({"rx": ["snr_db", "ser"], "tx": ["power_dbm"]})
   res = probed.run(grid={"ebn0_db": [6, 8, 10]}, metrics=["coded_ber"])
   res.series("ebn0_db", "rx.ser")            # the symbol errors before decoding

The metrics of the specifications (``snr_db``, ``ber``, ``ser``, ``evm``, ``gmi``…)
need the samples aligned on the transmitted symbols at one sample per symbol: after the
detection of a wireless receiver, after frame synchronisation of an optical one.
``power_dbm`` and ``papr_db`` are measured anywhere.

Signals
-------

:meth:`Chain.trace <comrs.Chain.trace>` runs one trial and keeps the frame after every
block: a :class:`~comrs.Trace`, which plots every stage and saves them to a ``.npz``
file (``stage/<name>/stream0``, ``stage/<name>/sps``…). The functions of
:mod:`comrs.plot` draw a frame or an array — :func:`~comrs.plot.constellation`,
:func:`~comrs.plot.eye`, :func:`~comrs.plot.time`, :func:`~comrs.plot.psd` — and take its
samples per symbol, sampling rate, constellation and polarisations from the frame;
:func:`comrs.plot.style` sets the figure style of this documentation:

.. code-block:: python

   frames = link.trace({"snr_db": 18}, comrs.Rng(1))
   frames.plot("constellation", stages=["decimate", "foe", "bps"], skip=512)
   comrs.plot.eye(frames["matched_filter"])
   comrs.plot.psd(frames["wdm"])
   frames.save("results/trace.npz")

A ``dump`` block saves the samples at its position during a campaign, for the first
``trials`` trials of every condition, as SigMF recordings
``<dir>/<campaign>/<condition>/<label>_<trial>.sigmf-*`` (one per stream, with the
samples per symbol and the symbol rate in the metadata), which :func:`comrs.read_sigmf`
and any SigMF tool read back; the first trial of a condition is the frame that
``trace`` gives with ``Rng.stream(seed, name, i, 0)``:

.. code-block:: python

   tapped = link.insert_after("cdc", tap=B.Dump(dir="results/signals", trials=2))
   tapped.run(grid={"snr_db": [14, 18]}, name="taps")
   x, fs, meta = comrs.read_sigmf("results/signals/taps/snr_db=18/tap_0.sigmf-meta")

Profiling
---------

``profile=True`` (``profile: true`` in a specification, ``--profile`` on the command
line) times every block of every trial and reports the mean time per trial as the
metrics ``<block>.time_ms``, with ``link.time_ms`` for the whole link.
:meth:`Results.profile <comrs.Results.profile>` tabulates them in the order of the
link with the share of every block, and :meth:`Chain.profile <comrs.Chain.profile>`
runs a short campaign for that purpose; both runners time their blocks the same way,
so a Python block and its native version compare directly:

.. code-block:: python

   for r in link.profile(grid={"snr_db": [18]}, stop={"max_trials": 16}):
       print(f"{r['block']:>16} {r['time_ms']:8.3f} ms {100 * r['share']:5.1f} %")

The times are those of one trial on one core; the throughput of the parallel runner
is about the number of cores divided by ``link.time_ms``.
