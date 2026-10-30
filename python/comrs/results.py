"""The results of a campaign and the trace of a trial, for the analysis after the run.

:class:`Results` is what :meth:`comrs.Chain.run` returns: a list of one ``dict`` per
condition (as :func:`comrs.run_spec`), with a tabular view, selections, plots and a
JSON file that keeps the specification next to the numbers. :func:`compare` runs
several chains on the same grid and gathers them, the variant as one more variable.
:class:`Trace` is what :meth:`comrs.Chain.trace` returns: the frame after every block
of one trial.

Example:
    >>> import comrs
    >>> from comrs import blocks as B
    >>> link = comrs.Chain(
    ...     tx=B.Source(constellation="QPSK", n_symbols=1000),
    ...     ch=B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")),
    ... )
    >>> res = link.run(grid={"ebn0_db": [2.0, 4.0]}, metrics=["ber"], stop={"max_trials": 16})
    >>> res.variables, res.metrics
    (['ebn0_db'], ['ber'])
    >>> x, ber = res.series("ebn0_db", "ber")
    >>> bool(ber[0] > ber[1])
    True
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

__all__ = ["Results", "Trace", "compare"]


class Results(list):
    """The results of a campaign: one ``dict`` per condition with ``condition``,
    ``status``, ``trials``, ``wall_s`` and ``metrics`` (``{name: {value, lo, hi, kind,
    count, total}}``).

    It is a ``list``; ``spec`` keeps the specification of the campaign when it is
    known (:func:`comrs.run_spec` and :meth:`comrs.Chain.run` return one).

    Args:
        items (list[dict]): the conditions.
        spec (dict | None): the specification.
    """

    def __init__(
        self, items: Iterable[Mapping[str, Any]] = (), spec: Mapping[str, Any] | None = None
    ):
        super().__init__(dict(r) for r in items)
        self.spec = dict(spec) if spec is not None else None

    # --- contents -------------------------------------------------------------------

    @property
    def variables(self) -> list[str]:
        """list[str]: The variables of the conditions, in order."""
        return list(dict.fromkeys(k for r in self for k in r["condition"]))

    @property
    def metrics(self) -> list[str]:
        """list[str]: The metrics reported, sorted (the probes as
        ``<label>.<metric>``)."""
        return sorted({k for r in self for k in r["metrics"]})

    def where(self, **condition: Any) -> Results:
        """The conditions whose variables have the given values.

        Args:
            **condition: the values (reals compared to a relative tolerance of 1e-9,
                zero to an absolute tolerance of 1e-12).

        Returns:
            Results: the selection, with the specification.
        """
        keep = [
            r for r in self if all(_same(r["condition"].get(k), v) for k, v in condition.items())
        ]
        return Results(keep, self.spec)

    def values(self, variable: str) -> list[Any]:
        """The distinct values of a variable, in order.

        Args:
            variable (str): the variable.

        Returns:
            list: its values.
        """
        return list(
            dict.fromkeys(r["condition"][variable] for r in self if variable in r["condition"])
        )

    def series(
        self, x: str, metric: str, *, interval: bool = False, min_count: int = 0, **condition: Any
    ) -> tuple[np.ndarray, ...]:
        """A metric against a variable, for the conditions that match ``condition``.

        Args:
            x (str): the variable of the abscissa.
            metric (str): the metric.
            interval (bool): also return the bounds of the confidence intervals.
            min_count (int): values of fewer errors (or values) are NaN.
            **condition: the values of the other variables.

        Returns:
            tuple[numpy.ndarray, ...]: ``(x, value)``, or ``(x, value, lo, hi)`` with
            ``interval``.

        Example:
            >>> import comrs
            >>> res = comrs.Results([
            ...     {"condition": {"snr_db": 0.0}, "metrics": {"ser": {"value": 0.1, "lo": 0.09, "hi": 0.11, "count": 10}}},
            ...     {"condition": {"snr_db": 5.0}, "metrics": {"ser": {"value": 0.01, "lo": 0.008, "hi": 0.012, "count": 1}}},
            ... ])
            >>> x, ser = res.series("snr_db", "ser", min_count=5)
            >>> x.tolist(), ser.tolist()
            ([0.0, 5.0], [0.1, nan])
        """
        sel = self.where(**condition)
        xs, ys, lo, hi = [], [], [], []
        for r in sel:
            m = r["metrics"].get(metric)
            if m is None or x not in r["condition"]:
                continue
            ok = m.get("count", 0) >= min_count
            xs.append(r["condition"][x])
            ys.append(m["value"] if ok else np.nan)
            lo.append(m["lo"] if ok else np.nan)
            hi.append(m["hi"] if ok else np.nan)
        out: tuple[np.ndarray, ...] = (np.asarray(xs), np.asarray(ys, dtype=float))
        if interval:
            out += (np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
        return out

    def rows(self) -> list[dict[str, Any]]:
        """One flat row per condition, with the columns of the CSV report.

        Returns:
            list[dict]: ``index``, the variables, ``status``, ``trials``, ``wall_s``, and
            for every metric ``m``: ``m``, ``m_lo``, ``m_hi``, ``m_count``, ``m_total``
            (the names of ``ConditionResult.row``).
        """
        from comrs._comrs import _metric_columns

        out = []
        for r in self:
            row = {
                "index": r.get("index"),
                **r["condition"],
                "status": r.get("status"),
                "trials": r.get("trials"),
                "wall_s": r.get("wall_s"),
            }
            for k, m in sorted(r["metrics"].items()):
                values = (m["value"], m["lo"], m["hi"], m.get("count"), m.get("total"))
                row.update(zip(_metric_columns(k), values, strict=True))
            out.append(row)
        return out

    def profile(self, **condition: Any) -> list[dict[str, Any]]:
        """Where the time goes, for a campaign run with ``profile=True``.

        Args:
            **condition: average over the conditions with these values only.

        Returns:
            list[dict]: one row per block of the link, in order, with ``block``,
            ``time_ms`` (the mean time per trial, averaged over the conditions) and
            ``share`` (of the time of the whole link, the last row, ``link``); empty
            when the campaign was not profiled.

        Example:
            >>> import comrs
            >>> from comrs import blocks as B
            >>> link = comrs.Chain(tx=B.Source(n_symbols=256), ch=B.Awgn(snr_db=10.0))
            >>> rows = link.run(stop={"max_trials": 2}, profile=True).profile()
            >>> [r["block"] for r in rows], rows[-1]["share"]
            (['tx', 'ch', 'link'], 1.0)
        """
        rows = self.where(**condition) if condition else self
        names: list[str] = []
        for r in rows:
            for k in r["metrics"]:
                if k.endswith(".time_ms") and k not in names:
                    names.append(k)
        if "link.time_ms" not in names:
            return []
        names.remove("link.time_ms")
        # the order of the link when the specification is known, else alphabetical
        order = {f"{b}.time_ms": i for i, b in enumerate(_block_labels(self.spec))}
        names.sort(key=lambda n: (order.get(n, len(order)), n))
        names.append("link.time_ms")

        def mean(name):
            v = [r["metrics"][name]["value"] for r in rows if name in r["metrics"]]
            return sum(v) / len(v)

        total = mean("link.time_ms")
        return [
            {
                "block": n.removesuffix(".time_ms"),
                "time_ms": mean(n),
                "share": mean(n) / total if total > 0 else 0.0,
            }
            for n in names
        ]

    def to_frame(self):
        """The rows as a table (pandas is needed).

        Returns:
            pandas.DataFrame: :meth:`rows`.
        """
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("Results.to_frame needs pandas: pip install pandas") from e
        return pd.DataFrame(self.rows())

    # --- plots ----------------------------------------------------------------------

    def plot(
        self,
        x: str,
        metric: str,
        *,
        by: str | None = None,
        ax=None,
        theory: Callable[..., Any] | None = None,
        **kwargs: Any,
    ):
        """Plot a metric against a variable, one curve per value of ``by``.

        Args:
            x (str): the variable of the abscissa.
            metric (str): the metric.
            by (str | None): one curve per value of this variable.
            ax (matplotlib.axes.Axes | None): the axes (new ones by default).
            theory (callable | None): a closed form of ``x``, drawn as a line.
            **kwargs: passed to :func:`comrs.plot.curves`.

        Returns:
            matplotlib.axes.Axes: the axes.
        """
        from comrs import plot

        return plot.curves(self, x, metric, by=by, ax=ax, theory=theory, **kwargs)

    # --- persistence ----------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the results, the specification and the version of comrs to a JSON
        file.

        Args:
            path (str | os.PathLike): the file.

        Returns:
            pathlib.Path: the path written.
        """
        from comrs import __version__

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"comrs_version": __version__, "spec": self.spec, "results": list(self)}
        path.write_text(json.dumps(doc, indent=1, default=_json_default))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Results:
        """Read results written by :meth:`save` (or the JSON report of a
        specification).

        Args:
            path (str | os.PathLike): the file.

        Returns:
            Results: the results and their specification.
        """
        doc = json.loads(Path(path).read_text())
        if isinstance(doc, list):
            return cls(doc)
        items = doc.get("results", doc.get("conditions", []))
        return cls(items, doc.get("spec", doc.get("config")))

    def __repr__(self) -> str:
        if not self:
            return "Results([])"
        lines = [f"Results: {len(self)} conditions, metrics {self.metrics}"]
        for r in self[:12]:
            cond = ", ".join(f"{k}={v}" for k, v in r["condition"].items())
            ms = ", ".join(f"{k}={m['value']:.4g}" for k, m in sorted(r["metrics"].items()))
            lines.append(f"  ({cond}) {r.get('trials')} trials: {ms}")
        if len(self) > 12:
            lines.append(f"  … {len(self) - 12} more")
        return "\n".join(lines)


def _block_labels(spec: Mapping[str, Any] | None) -> list[str]:
    """The labels of the blocks of a specification, as the profile names them (the
    Rust numbering of the labels)."""
    if not spec:
        return []
    if "block_order" in spec:
        return list(spec["block_order"])
    from comrs._comrs import _link_labels

    return _link_labels(spec.get("link", []))


def _same(a: Any, b: Any) -> bool:
    """Whether two values of a variable are the same: reals within a relative tolerance
    of 1e-9 (1e-12 and 2e-12 differ), a real and zero within 1e-12 (a grid step that
    lands near zero)."""
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12 if 0 in (a, b) else 0.0)
    return a == b


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def compare(chains: Mapping[str, Any], *, variable: str = "variant", **run: Any) -> Results:
    """Run several chains with the same arguments of :meth:`comrs.Chain.run` and gather
    their results, the name of every chain as the variable ``variable``.

    Args:
        chains (dict[str, Chain]): the chains, by name.
        variable (str): the name of the variable of the variants.
        **run: the arguments of :meth:`comrs.Chain.run`.

    Returns:
        Results: the results of every chain, the variant first in the conditions.

    Example:
        >>> import comrs
        >>> from comrs import blocks as B
        >>> base = comrs.Chain(tx=B.Source(constellation="QPSK", n_symbols=500),
        ...                    ch=B.Awgn(snr_db=8.0))
        >>> res = comrs.compare({"8 dB": base, "4 dB": base.replace(ch=B.Awgn(snr_db=4.0))},
        ...                     metrics=["ser"], stop={"max_trials": 8})
        >>> res.values("variant")
        ['8 dB', '4 dB']
    """
    variants: dict[str, Any] = {}
    out = Results(spec={"variable": variable, "variants": variants})
    for name, chain in chains.items():
        res = chain.run(**run)
        out.extend({**r, "condition": {variable: name, **r["condition"]}} for r in res)
        variants[name] = getattr(res, "spec", None)
    return out


class Trace(dict):
    """The frame after every block of one trial (``{name: Frame}``, in the order of the
    chain), from :meth:`comrs.Chain.trace`."""

    def plot(self, kind: str = "constellation", stages: Iterable[str] | None = None, **kwargs: Any):
        """One panel per stage.

        Args:
            kind (str): a function of :mod:`comrs.plot` (``constellation``, ``time``,
                ``psd``, ``eye``).
            stages (list[str] | None): the stages (all by default).
            **kwargs: passed to the function.

        Returns:
            matplotlib.figure.Figure: the figure.
        """
        import matplotlib.pyplot as plt

        from comrs import plot

        names = list(stages) if stages is not None else list(self)
        n = len(names)
        cols = min(n, 4)
        rows = -(-n // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
        draw = getattr(plot, kind)
        for ax, name in zip(axes.flat, names, strict=False):
            draw(self[name], ax=ax, **kwargs)
            ax.set_title(name)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        return fig

    def save(self, path: str | Path) -> Path:
        """Write the samples of every stage to a NumPy ``.npz`` file: ``<name>`` (the
        first stream), ``<name>_1``, ``<name>_2``… (the other polarisations or
        antennas), ``<name>_sps`` (the samples per symbol) and ``tx_symbols`` (the
        transmitted symbols).

        Args:
            path (str | os.PathLike): the file.

        Returns:
            pathlib.Path: the path written.
        """
        arrays: dict[str, np.ndarray] = {}
        for name, f in self.items():
            for i, s in enumerate(f.streams):
                arrays[name if i == 0 else f"{name}_{i}"] = np.asarray(s)
            arrays[f"{name}_sps"] = np.asarray(f.sps)
        last = next(reversed(self.values()), None)
        if last is not None:
            arrays["tx_symbols"] = np.asarray(last.tx_symbols)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # every key is a name of an array (none is the keyword `allow_pickle`)
        np.savez(path, **cast("dict[str, Any]", arrays))
        return path

    def __repr__(self) -> str:
        return "Trace(" + ", ".join(f"{k}: {v!r}" for k, v in self.items()) + ")"
