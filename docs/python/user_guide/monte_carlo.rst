Monte Carlo campaigns in Python
===============================

:mod:`comrs.sim` runs Python trials on the Rust runner (the crate ``com-sim``), with
the vocabulary of PyTorch Lightning: a **trial** is what one random experiment does, a
**record** collects its metrics, a **stopping rule** says when a condition has enough
trials. The runner calls the Python code under the GIL, with one thread; its
statistics, stopping rules, statuses, condition labels and CSV and JSON reports are
those of the Rust campaigns.

The trial
---------

A trial receives the condition (a ``dict``), its own generator and a
:class:`~comrs.sim.Record`. It reports error counts, estimated as rates with Wilson
score intervals, or values, estimated as means with normal intervals:

.. code-block:: python

   import comrs
   from comrs import sim

   c = comrs.Constellation("qam", 16).normalized()

   def trial(cond, rng, rec):
       bits = comrs.bits(4 * 4096, rng)
       y = comrs.awgn_ebn0(c.map_bits(bits), cond["ebn0_db"], 4, rng)
       rec.errors("ber", int((c.demap_bits(y) != bits).sum()), bits.size)
       rec.value("evm", comrs.evm(y, c))

The grid and the stopping rule
------------------------------

:func:`~comrs.sim.grid` builds the Cartesian product of its axes (the last one varies
fastest). A :class:`~comrs.sim.Stop` is checked after every batch of trials; rules
combine with ``|`` (or) and ``&`` (and), and a rule must be bounded by
``max_trials`` or ``max_time``. The ``stop`` mapping of a specification works too
(``stop={"min_errors": {"ber": 200}, "max_trials": 2000}``, parsed by
:meth:`Stop.from_spec <comrs.sim.Stop.from_spec>` as the Rust runner parses it):

.. code-block:: python

   stop = sim.Stop.min_errors("ber", 200) | sim.Stop.max_trials(2_000)
   results = sim.run(trial, sim.grid(ebn0_db=range(0, 15, 2)), name="qam16",
                     seed=1, stop=stop)

   for r in results:
       lo, hi = r.metrics["ber"].interval(0.95)
       print(r.condition["ebn0_db"], r.metrics["ber"].value, (lo, hi), r.trials)

``csv_path=`` and ``json_path=`` write the reports of the Rust loggers. An exception of
a trial is raised again unchanged, with a note naming the trial and the condition
(``trial 3 of (ebn0_db=4) failed``).

Statistics
----------

An error count of :math:`k` errors over :math:`n` trials is reported with the Wilson
score interval

.. math::

   \frac{\hat p + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat p (1 - \hat p)}{n} + \frac{z^2}{4n^2}}}
        {1 + \frac{z^2}{n}}, \qquad \hat p = \frac{k}{n},

which stays inside :math:`[0, 1]` and is accurate with few errors; the relative
standard error of a rate is about :math:`1/\sqrt{k}`, so 100 errors give about 10 %
and 400 errors about 5 %. ``Stop.precision(metric, relative)`` stops when the relative
half-width of the interval falls below a target instead.

Reproducibility
---------------

Trial ``t`` of condition ``i`` of the scenario ``name`` draws from
``comrs.Rng.stream(seed, name, i, t)``: a result is reproducible from the seed, a
single trial can be replayed alone, and the Rust runner draws exactly the same numbers
for the same scenario. For long campaigns, write the link with the blocks of the
registry (a native :class:`~comrs.Chain` or a :doc:`specification <specifications>`
runs on all cores) or in Rust: Python trials run on one thread.
