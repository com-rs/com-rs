"""Monte Carlo campaigns of Python trials, on the Rust runner.

The trials of a Python scenario run on the runner of the Rust crate ``com-sim``, with
one thread (the Python code is called under the GIL): the vocabulary (conditions,
trials, records, stopping rules), the per-trial random streams, the statistics, the
statuses, the labels of the conditions and the CSV and JSON loggers are those of the
Rust campaigns, so a scenario written in Python draws exactly what its Rust port
draws and reports the same numbers. For long campaigns, write the link with the
blocks of the registry (a native :class:`comrs.Chain` runs on all cores).

Vocabulary:

- a **condition** is a ``dict`` of named values; :func:`grid` builds the Cartesian
  product of axes (the last axis varies fastest);
- a **trial** is a function ``trial(condition, rng, record[, state])`` that draws only
  from ``rng`` and reports metrics in the :class:`Record`: error counts
  (:meth:`Record.errors`, estimated as an :class:`ErrorRate` with a Wilson interval)
  or values (:meth:`Record.value`, estimated as a :class:`Mean` with a normal
  interval);
- trial ``t`` of condition ``i`` draws from ``Rng.stream(seed, name, i, t)``, so the
  results depend neither on the batch size nor on the order of execution;
- a **stopping rule** (:class:`Stop`) is checked after every batch of trials:
  ``Stop.min_errors("ber", 200) | Stop.max_trials(10_000)``, or the ``stop`` mapping
  of a specification.

Example:
    >>> import comrs
    >>> from comrs import sim
    >>> c = comrs.Constellation("qam", 4).normalized()
    >>> def trial(cond, rng, rec):
    ...     b = comrs.bits(2 * 1000, rng)
    ...     y = comrs.awgn_ebn0(c.modulate(b), cond["ebn0_db"], 2, rng)
    ...     rec.errors("ber", int((c.demodulate(y) != b).sum()), b.size)
    >>> res = sim.run(trial, sim.grid(ebn0_db=[4.0, 6.0]), name="qpsk",
    ...               stop=sim.Stop.min_errors("ber", 100) | sim.Stop.max_trials(500))
    >>> [r.status for r in res]
    ['completed', 'completed']
    >>> lo, hi = res[0].metrics["ber"].interval(0.999)
    >>> bool(lo < comrs.theory_ber(4, 4.0, "qam") < hi)
    True
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from comrs._comrs import (
    ConditionResult,
    ErrorRate,
    Mean,
    Record,
    Stop,
    _sim_run,
    spec_conditions,
    z_score,
)

__all__ = [
    "ConditionResult",
    "ErrorRate",
    "Mean",
    "Record",
    "Stop",
    "grid",
    "run",
    "z_score",
]


def grid(**axes: Any) -> list[dict[str, Any]]:
    """The Cartesian product of the axes, the last one varying fastest (the ``grid`` of
    a specification).

    Args:
        **axes: the values of every variable: a list, ``{"from": a, "to": b, "step":
            s}`` or ``{"linspace": [start, stop, count]}``.

    Returns:
        list[dict]: the conditions.

    Example:
        >>> from comrs.sim import grid
        >>> grid(m=[4, 16], snr_db=[0.0, 10.0])
        [{'m': 4, 'snr_db': 0.0}, {'m': 4, 'snr_db': 10.0}, {'m': 16, 'snr_db': 0.0}, {'m': 16, 'snr_db': 10.0}]
    """
    from comrs.spec import _grid_axes

    # the conditions of a specification with this grid, expanded by the Rust parser
    return spec_conditions({"name": "grid", "grid": _grid_axes(axes), "link": ["normalize"]})


def run(
    trial: Callable[..., None],
    conditions: Iterable[Mapping[str, Any]],
    *,
    name: str,
    seed: int = 0,
    batch: int = 32,
    stop: Stop | Mapping[str, Any] | None = None,
    setup: Callable[[Mapping[str, Any]], Any] | None = None,
    confidence: float = 0.95,
    progress: bool = False,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> list[ConditionResult]:
    """Run a scenario on every condition, on the Rust runner (one thread).

    Args:
        trial (callable): ``trial(condition, rng, record)``, or ``trial(condition, rng,
            record, state)`` when ``setup`` is given.
        conditions (list[dict]): the conditions, for instance from :func:`grid`.
        name (str): the scenario name (it enters the seeds).
        seed (int): the base seed.
        batch (int): trials between two checks of the stopping rule.
        stop (Stop | dict | None): the stopping rule, or the ``stop`` mapping of a
            specification (default: 100 trials); it must be bounded.
        setup (callable | None): ``setup(condition) -> state``, run once per condition.
        confidence (float): level of the intervals of the reports.
        progress (bool): print one line per condition on the standard error.
        csv_path (str | os.PathLike | None): write one row per condition to this CSV
            file (the columns of the Rust logger).
        json_path (str | os.PathLike | None): write the report of the run to this JSON
            file.

    Returns:
        list[ConditionResult]: one result per condition, in order.

    Raises:
        ValueError: for a stopping rule that can run forever. An exception of a trial
            or of the setup is raised again with a note naming the trial and the
            condition.
    """
    if stop is not None and not isinstance(stop, Stop):
        # a dict: the extension serialises it with `json.dumps`, which takes no other mapping
        stop = Stop.from_spec(dict(stop))
    return _sim_run(
        trial,
        # the extension copies every mapping of any iterable (its stub says list[dict])
        cast("list[dict[str, Any]]", conditions),
        name=name,
        seed=seed,
        batch=batch,
        stop=stop,
        setup=setup,
        confidence=confidence,
        progress=progress,
        csv_path=csv_path,
        json_path=json_path,
    )
