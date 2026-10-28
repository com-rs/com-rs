Chains of blocks
================

A :class:`~comrs.Chain` describes a link as an ordered sequence of **named blocks**,
in the spirit of ``torch.nn.Sequential``: the same object runs once to inspect the
signals, runs a whole campaign on the parallel Rust runner, and converts to and from
a declarative specification.

Building a chain
----------------

The blocks of the declarative links are classes of :mod:`comrs.blocks`, one per entry
of :func:`comrs.spec_blocks` in CamelCase (``awgn_ebn0`` → ``AwgnEbn0``), their
parameters as keyword arguments (an unknown name raises ``TypeError``; ``help(B.Bps)``
shows them). A parameter given as :func:`comrs.var` takes the value of a grid
variable, as ``$name`` does in a specification:

.. code-block:: python

   import comrs
   from comrs import blocks as B

   link = comrs.Chain(
       tx=B.Source(constellation="16-QAM", n_symbols=8192, symbol_rate=32e9),
       shaping=B.PulseShape(sps=2, rolloff=0.2),
       channel=B.Awgn(snr_db=comrs.var("snr_db")),
       laser=B.PhaseNoise(linewidth_hz=100e3),
       mf=B.MatchedFilter(rolloff=0.2),
       down=B.Decimate(),
       cpr=B.Bps(n=32, b=32),
   )

A chain behaves as an ordered mapping: ``link["cpr"]``, ``link.names``, ``len(link)``;
``link.replace(cpr=B.Ddpll(...))`` and ``link.insert_after("channel", cfo=B.Cfo(hz=1e6))``
return modified copies, so variants of a link stay one line apart. A chain is itself a
block: ``comrs.Chain(tx=transmitter, channel=..., rx=receiver)`` nests sub-chains,
whose blocks are named ``rx.cpr`` and so on.

Running once and tracing
------------------------

``link(condition, rng)`` runs the chain once and returns the final
:class:`~comrs.Frame`: the samples, the samples per symbol, and what the metrics need
(the transmitted bits, indices and symbols, the constellation). ``link.trace(condition,
rng)`` returns a copy of the frame after every block — the quickest way to look at a
constellation after each stage of the receiver:

.. code-block:: python

   frames = link.trace({"snr_db": 18}, comrs.Rng(1))
   frames["channel"].samples   # before the receiver
   frames["cpr"].samples       # after carrier recovery

Python blocks
-------------

Any callable ``f(frame, rng)`` is a block, and so is any subclass of
:class:`~comrs.Block`. A Python block reads and writes ``frame.samples`` (and
``frame.sps``), and ``frame.record(name, value)`` reports a value, estimated as a mean
by the runners:

.. code-block:: python

   class Clipper(comrs.Block):
       def __init__(self, level):
           self.level = level

       def __call__(self, frame, rng):
           x = frame.samples
           frame.samples = x * np.minimum(1, self.level / np.maximum(np.abs(x), 1e-12))

   clipped = link.insert_after("shaping", clip=Clipper(1.2))

Campaigns
---------

:meth:`Chain.run <comrs.Chain.run>` evaluates the chain on a grid, with the metrics
and the stopping rule of the specifications:

.. code-block:: python

   results = link.run(
       grid={"snr_db": [14, 16, 18, 20]},
       metrics={"report": ["ber", "snr_db"], "skip_start": 512},
       stop={"min_errors": {"ber": 200}, "max_trials": 500},
       seed=1,
   )
   results[0]["metrics"]["ber"]   # {value, lo, hi, kind, count, total}

Both kinds of chain run on the Rust runner: a chain of native blocks on all cores
(:func:`comrs.run_spec`), a chain with a Python block on one thread, its Python blocks
called under the GIL. Both draw trial ``t`` of condition ``i`` from
``Rng.stream(seed, name, i, t)`` and share the stopping rules, the statistics and the
reports: inserting a Python block that leaves the frame unchanged gives the same error
counts, trial for trial, so a Python prototype of a block can be checked against the
native chain before it is ported to Rust.

Specifications
--------------

:meth:`Chain.to_spec <comrs.Chain.to_spec>` returns the declarative specification of a
native chain (the block names become labels, the variables ``$name``), ready to be
saved as YAML and run by the ``comrs`` command (``comrs run spec.yaml``); :meth:`Chain.from_spec <comrs.Chain.from_spec>`
reads one back, from a file, a YAML text or a dict, keeping its grid, metrics and
stopping rule as :attr:`~comrs.Chain.settings`, so that

.. code-block:: python

   link = comrs.Chain.from_spec("specs/coherent_link.yaml")
   results = link.replace(bps=B.Bps(n=64, b=32)).run()

reruns the campaign of the file with a single block changed.
