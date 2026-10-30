"""Communication chains as Python objects.

A :class:`Chain` is an ordered, named sequence of blocks, in the spirit of
``torch.nn.Sequential``: every block transforms a :class:`~comrs.Frame` (the samples
and what the metrics need) in place, drawing its randomness from an explicit
:class:`~comrs.Rng`.

- The blocks of the registry of declarative links (``comrs.spec_blocks()``) are
  available as classes of :mod:`comrs.blocks` (``blocks.Source``, ``blocks.AwgnEbn0``,
  ``blocks.MatchedFilter``…), with their parameters as keyword arguments; a parameter
  given as ``comrs.var("name")`` takes the value of the grid variable ``name``.
- A Python block is any :class:`~comrs.Block` subclass (or callable) ``f(frame, rng)``.
- A chain runs on the Rust runner (:meth:`Chain.run`): a chain of native blocks on all
  cores, and converts to and from a declarative specification (:meth:`Chain.to_spec`,
  :meth:`Chain.from_spec`); a chain with a Python block on one thread, the Python
  blocks called under the GIL, with the same random streams, stopping rules and
  statistics.

Example:
    >>> import comrs
    >>> from comrs import blocks as B
    >>> link = comrs.Chain(
    ...     tx=B.Source(constellation="QPSK", n_symbols=2000),
    ...     channel=B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")),
    ... )
    >>> frame = link({"ebn0_db": 6.0}, comrs.Rng(1))
    >>> frame.samples.shape
    (2000,)
    >>> res = link.run(grid={"ebn0_db": [4.0]}, metrics=["ber"], stop={"max_trials": 64})
    >>> lo, hi = res[0]["metrics"]["ber"]["lo"], res[0]["metrics"]["ber"]["hi"]
    >>> bool(lo < comrs.theory_ber(4, 4.0, "qam") < hi)
    True
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from comrs import spec as _spec
from comrs._comrs import Frame, Rng, Stop, _run_chain, run_spec
from comrs.block import Block, Native, Var, _as_block, var
from comrs.results import Results, Trace

__all__ = ["Block", "Chain", "Native", "Var", "var"]


class Chain(Block):
    """An ordered, named sequence of blocks; a chain is itself a block.

    Args:
        *blocks: unnamed blocks, named as the Rust runner names the blocks of a link:
            by their type (``source``; a Python block by its class or function,
            ``gain``), a repeated name followed by the position of its block
            (``awgn_2``, ``awgn_3``).
        **named: named blocks, in order, after the unnamed ones.

    Attributes:
        settings (dict): the campaign settings of the specification the chain comes
            from (``name``, ``seed``, ``grid``, ``metrics``, ``stop``, ``fiber``…),
            the defaults of :meth:`to_spec` and :meth:`run`.
        base_dir (pathlib.Path | None): the directory of the specification file the
            chain comes from, against which the relative paths of its blocks (the
            ``file`` of a ``capture``) are resolved, as the ``comrs`` command does;
            None: the working directory.

    Example:
        >>> import comrs
        >>> from comrs import blocks as B
        >>> link = comrs.Chain(tx=B.Source(n_symbols=64), norm=B.Normalize())
        >>> link.names, len(link)
        (['tx', 'norm'], 2)
        >>> comrs.Chain(B.Source(), B.Awgn(snr_db=3.0), B.Awgn(snr_db=6.0)).names
        ['source', 'awgn_2', 'awgn_3']
    """

    def __init__(self, *blocks: Any, **named: Any) -> None:
        from comrs._comrs import _link_labels

        self.settings: dict[str, Any] = {}
        self.base_dir: Path | None = None
        unnamed = [_as_block(b) for b in blocks]
        labels = _link_labels([_kind(b) for b in unnamed]) if unnamed else []
        clash = sorted(set(labels) & set(named))
        if clash:
            raise ValueError(f"the names {clash} are given to a named and an unnamed block")
        self._blocks: dict[str, Block] = dict(zip(labels, unnamed, strict=True))
        for name, b in named.items():
            self._blocks[name] = _as_block(b)

    # --- containers --------------------------------------------------------------

    def __getitem__(self, name: str) -> Block:
        return self._blocks[name]

    def __iter__(self) -> Iterator[tuple[str, Block]]:
        return iter(self._blocks.items())

    def __len__(self) -> int:
        return len(self._blocks)

    @property
    def names(self) -> list[str]:
        """list[str]: The names of the blocks, in order."""
        return list(self._blocks)

    def _copy(self) -> Chain:
        out = Chain()
        out.settings = dict(self.settings)
        out.base_dir = self.base_dir
        return out

    def replace(self, **blocks: Any) -> Chain:
        """A copy of the chain with some blocks replaced.

        Args:
            **blocks: the new blocks, by name.

        Returns:
            Chain: the copy.

        Raises:
            KeyError: for a name the chain does not have.
        """
        unknown = set(blocks) - set(self._blocks)
        if unknown:
            raise KeyError(f"no block named {sorted(unknown)}")
        out = self._copy()
        out._blocks = {
            k: _as_block(blocks[k]) if k in blocks else v for k, v in self._blocks.items()
        }
        return out

    def insert_after(self, after: str, **blocks: Any) -> Chain:
        """A copy of the chain with blocks inserted after a block.

        Args:
            after (str): the name of the block.
            **blocks: the new blocks, by name, in order.

        Returns:
            Chain: the copy.
        """
        if after not in self._blocks:
            raise KeyError(f"no block named {after!r}")
        out = self._copy()
        for k, v in self._blocks.items():
            out._blocks[k] = v
            if k == after:
                out._blocks.update({n: _as_block(b) for n, b in blocks.items()})
        return out

    def probe(
        self,
        after: str,
        metrics: Any = ("snr_db",),
        *,
        label: str | None = None,
        **params: Any,
    ) -> Chain:
        """A copy of the chain with a ``probe`` block after a block: the metrics
        measured at that point of every trial of a campaign, reported as
        ``<label>.<metric>``. The metrics of the specifications need symbols aligned at
        one sample per symbol; ``power_dbm`` and ``papr_db`` are measured anywhere.

        Args:
            after (str): the name of the block.
            metrics (list[str]): the metrics.
            label (str | None): the prefix of the metrics (``after`` by default).
            **params: the other parameters of the block (``skip_start``,
                ``skip_end``, ``scale``).

        Returns:
            Chain: the copy.
        """
        label = label or after
        return self.insert_after(
            after,
            **{f"{label}_probe": Native("probe", metrics=list(metrics), label=label, **params)},
        )

    def probes(self, points: Mapping[str, Any], **params: Any) -> Chain:
        """A copy of the chain with a probe after every block named in ``points``.

        Args:
            points (dict): ``{block: metrics}``.
            **params: the parameters of every probe.

        Returns:
            Chain: the copy.
        """
        out = self
        for after, metrics in points.items():
            out = out.probe(after, metrics, **params)
        return out

    def flatten(self) -> list[tuple[str, Block]]:
        """The blocks of the chain and of its sub-chains, with dotted names.

        Returns:
            list[tuple[str, Block]]: the names and the blocks, in order.
        """
        out: list[tuple[str, Block]] = []
        for name, b in self._blocks.items():
            if isinstance(b, Chain):
                out.extend((f"{name}.{n}", s) for n, s in b.flatten())
            else:
                out.append((name, b))
        return out

    @property
    def is_native(self) -> bool:
        """bool: Whether every block belongs to the registry (the chain then runs on
        all cores)."""
        return all(isinstance(b, Native) for _, b in self.flatten())

    # --- execution ---------------------------------------------------------------

    def bind(
        self,
        condition: Mapping[str, Any] | None,
        *,
        fiber: Mapping[str, Any] | None = None,
        name: str | None = None,
        label: str | None = None,
        profile: bool = False,
        base_dir: str | os.PathLike[str] | None = None,
    ) -> Callable[[Frame, Rng], None]:
        """The blocks of a condition as one callable (see :meth:`Block.bind`).

        Args:
            condition (dict | None): the values of the grid variables.
            fiber (dict | None): the ``fiber`` of the specification (the one of
                :attr:`settings` by default).
            name (str | None): the name of the campaign, for the blocks that write
                files.
            label (str | None): unused (the blocks are named by the chain).
            profile (bool): record the time of every block as ``<name>.time_ms`` and of
                the link as ``link.time_ms``.
            base_dir (str | os.PathLike | None): the directory the relative paths of
                the native blocks are resolved against (:attr:`base_dir` by default).

        Returns:
            Callable[[Frame, Rng], None]: ``apply(frame, rng)``.

        Example:
            >>> import comrs
            >>> from comrs import blocks as B
            >>> link = comrs.Chain(tx=B.Source(n_symbols=8), ch=B.Awgn(snr_db=comrs.var("snr")))
            >>> apply = link.bind({"snr": 10.0})
            >>> f = comrs.Frame()
            >>> apply(f, comrs.Rng(1))
            >>> f.samples.size
            8
        """
        bound = self._steps(condition, fiber, name, base_dir)
        names, steps = [n for n, _ in bound], [s for _, s in bound]

        def apply(frame: Frame, rng: Rng) -> None:
            if not profile:
                for s in steps:
                    s(frame, rng)
                return
            t_link = time.perf_counter()
            for n, s in zip(names, steps, strict=True):
                t = time.perf_counter()
                s(frame, rng)
                frame.record(f"{n}.time_ms", 1e3 * (time.perf_counter() - t))
            frame.record("link.time_ms", 1e3 * (time.perf_counter() - t_link))

        return apply

    def _steps(
        self,
        condition: Mapping[str, Any] | None,
        fiber: Mapping[str, Any] | None,
        name: str | None,
        base_dir: str | os.PathLike[str] | None,
    ) -> list[tuple[str, Callable[[Frame, Rng], None]]]:
        """The blocks of a condition, with their dotted names: the arguments of
        :meth:`bind`, the ``fiber`` and the ``base_dir`` of the chain by default."""
        fiber = fiber if fiber is not None else self.settings.get("fiber")
        base_dir = base_dir if base_dir is not None else self.base_dir
        return [
            (n, b.bind(condition, fiber=fiber, name=name, label=n, base_dir=base_dir))
            for n, b in self.flatten()
        ]

    # a chain is a block that also returns its frame: callers of a block ignore the value
    def __call__(  # type: ignore[override]
        self, condition: Mapping[str, Any] | Frame | None, rng: Rng
    ) -> Frame:
        """Run the chain once.

        Args:
            condition (dict | Frame | None): the values of the grid variables, or, as a
                block inside another chain, the frame to transform.
            rng (Rng): the generator (randomness is explicit).

        Returns:
            Frame: the final frame.
        """
        if isinstance(condition, Frame):
            self.bind(None)(condition, rng)
            return condition
        frame = Frame()
        self.bind(condition)(frame, rng)
        return frame

    def trace(
        self, condition: Mapping[str, Any] | None, rng: Rng, *, name: str | None = None
    ) -> Trace:
        """Run the chain once and keep a copy of the frame after every block.

        Args:
            condition (dict | None): the values of the grid variables.
            rng (Rng): the generator.
            name (str | None): the name of the campaign, for the blocks that write
                files (as :meth:`bind`).

        Returns:
            Trace: ``{name: Frame}`` in the order of the chain (``trace.plot()``,
            ``trace.save(path)``).
        """
        frame = Frame()
        out = Trace()
        for label, step in self._steps(condition, None, name, None):
            step(frame, rng)
            out[label] = frame.copy()
        return out

    # --- specifications -----------------------------------------------------------

    def to_spec(
        self,
        name: str | None = None,
        grid: Mapping[str, Any] | None = None,
        metrics: list[str] | Mapping[str, Any] | None = None,
        stop: Mapping[str, Any] | None = None,
        seed: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """The declarative specification of the chain (native blocks only): the
        :attr:`settings`, updated by the arguments given.

        Args:
            name (str | None): the campaign name (it enters the seeds; default
                ``"chain"``).
            grid (dict | None): the grid, ``{variable: values}``.
            metrics (list[str] | dict | None): the reported metrics (``["ber",
                "snr_db"]``) or the full ``metrics`` mapping (``report``,
                ``skip_start``…).
            stop (dict | None): the stopping rule (``min_errors``, ``min_trials``,
                ``max_trials``, ``max_time_s``, ``precision``).
            seed (int | None): the base seed.
            **extra: other fields (``batch``, ``fiber``, ``confidence``…).

        Returns:
            dict: the specification.

        Raises:
            TypeError: for a chain with Python blocks.
        """
        if not self.is_native:
            raise TypeError("a chain with Python blocks has no declarative specification")
        spec = _spec.settings(
            self.settings, name=name, grid=grid, metrics=metrics, stop=stop, seed=seed, **extra
        )
        spec["link"] = _link(self.flatten())
        return spec

    @classmethod
    def from_spec(cls, spec: str | Path | Mapping[str, Any]) -> Chain:
        """The chain of a specification: the blocks of its link, named by their labels
        (or types; a repeated name takes the position of its block, as the Rust runner
        numbers them), and its other fields as :attr:`settings`.

        The specification is read by the Rust parser (:func:`comrs.spec.load`); the
        directory of a file is kept as :attr:`base_dir`, so that the relative paths of
        the blocks are those of the ``comrs`` command.

        Args:
            spec (str | os.PathLike | dict): a mapping, a YAML or JSON text, or a file.

        Returns:
            Chain: the chain.

        Raises:
            FileNotFoundError: for a string that is neither a file nor a specification.
            ValueError: for an invalid specification.
        """
        d, base_dir = _spec._load(spec)
        blocks, settings = _spec.chain_blocks(d)
        chain = cls()
        chain._blocks = dict(blocks)
        chain.settings = settings
        chain.base_dir = base_dir
        return chain

    # --- campaigns ----------------------------------------------------------------

    def run(
        self,
        grid: Mapping[str, Any] | None = None,
        metrics: list[str] | Mapping[str, Any] | None = None,
        stop: Mapping[str, Any] | Stop | None = None,
        seed: int | None = None,
        name: str | None = None,
        *,
        threads: int | None = None,
        progress: bool = False,
        **extra: Any,
    ) -> Results:
        """A Monte Carlo campaign over the grid, the :attr:`settings` updated by the
        arguments given (as :meth:`to_spec`; the metrics default to ``["ber"]``).

        Both kinds of chain run on the Rust runner, trial ``t`` of condition ``i``
        drawing from ``Rng.stream(seed, name, i, t)``: a native chain on all cores
        (:func:`comrs.run_spec`), a chain with Python blocks on one thread; the
        stopping rule (a mapping, or a :class:`comrs.sim.Stop`, which runs a native
        chain on the one-thread path too), the statistics and the outputs are the
        same, so a native chain gives the same results on both.

        Args:
            grid (dict | None): the grid.
            metrics (list[str] | dict | None): the metrics.
            stop (dict | Stop | None): the stopping rule.
            seed (int | None): the base seed.
            name (str | None): the campaign name.
            threads (int | None): worker threads of a native chain with a stopping rule
                given as a mapping (0: all cores).
            progress (bool): print one line per condition on the standard error.
            **extra: other fields of the specification (``batch``, ``profile``,
                ``fiber``, ``output``…).

        Returns:
            Results: one dict per condition, as :func:`comrs.run_spec`, with tabular
            access, plots and persistence.

        Raises:
            ValueError: for ``threads`` other than 1 on the one-thread path (Python
                blocks, or a :class:`comrs.sim.Stop`), which would ignore it.
        """
        rule, mapping = (stop, None) if isinstance(stop, Stop) else (None, stop)
        spec = _spec.settings(
            self.settings,
            name=name,
            grid=grid,
            metrics=metrics,
            stop=mapping,
            seed=seed,
            **extra,
        )
        spec.setdefault("metrics", {"report": ["ber"]})
        if self.is_native and rule is None:
            spec["link"] = self.to_spec()["link"]
            res = run_spec(spec, threads=threads, progress=progress, base_dir=self.base_dir)
            return Results(res, spec)
        if threads not in (None, 1):
            why = "a comrs.sim.Stop" if rule is not None else "Python blocks"
            raise ValueError(
                f"threads={threads}: a chain with {why} runs on one thread (give the stopping "
                "rule as a mapping to run a native chain on several)"
            )
        results = self._run_python(spec, rule, progress)
        return Results(results, {**spec, "block_order": [n for n, _ in self.flatten()]})

    def profile(self, **run: Any) -> list[dict[str, Any]]:
        """Where the time goes: run the chain with ``profile=True`` and tabulate the
        mean time per trial of every block (see :meth:`comrs.Results.profile`).

        Args:
            **run: the arguments of :meth:`run` (by default one condition and 8
                trials).

        Returns:
            list[dict]: one row per block, ``block``, ``time_ms``, ``share``, then the
            whole link.

        Example:
            >>> import comrs
            >>> from comrs import blocks as B
            >>> link = comrs.Chain(tx=B.Source(n_symbols=256), ch=B.Awgn(snr_db=10.0))
            >>> [r["block"] for r in link.profile(stop={"max_trials": 2})]
            ['tx', 'ch', 'link']
        """
        run.setdefault("stop", {"max_trials": 8})
        run.setdefault("metrics", ["ber"])
        return self.run(profile=True, **run).profile()

    def _run_python(self, spec: dict[str, Any], rule: Any, progress: bool) -> list[dict[str, Any]]:
        """The campaign of a chain with Python blocks, on the Rust runner."""
        profile = bool(spec.get("profile", False))
        fiber = spec.get("fiber")
        name = spec["name"]

        def setup(cond: dict[str, Any]) -> Callable[[Frame, Rng], None]:
            return self.bind(cond, fiber=fiber, name=name, profile=profile)

        return _run_chain(spec, setup, progress=progress, stop=rule)

    def __repr__(self) -> str:
        inner = ",\n".join(f"    {k}={v!r}" for k, v in self._blocks.items())
        return f"Chain(\n{inner}\n)"


def _link(blocks: list[tuple[str, Block]]) -> list[dict[str, Any]]:
    """The link of a specification of native blocks, every block labelled by its name in
    the chain, the label left out where the Rust runner names the block so anyway (its
    type, or the type and the position of a repeated one: ``awgn_2``)."""
    from comrs._comrs import _link_labels

    names = [n.split(".")[-1] for n, _ in blocks]
    natives = [cast("Native", b) for _, b in blocks]
    link = [b.spec(n) for n, b in zip(names, natives, strict=True)]
    # the blocks named by the chain, not by a `label` parameter (a probe's)
    bare = [
        {k: v for k, v in d.items() if k != "label" or "label" in b.params}
        for d, b in zip(link, natives, strict=True)
    ]
    trimmed = [
        lean if auto == n else d
        for d, lean, n, auto in zip(link, bare, names, _link_labels(bare), strict=True)
    ]
    return trimmed if _link_labels(trimmed) == names else link


def _kind(block: Block) -> str:
    """The type of a block, as an unnamed block is named: the registry type of a
    native block, ``chain``, or the class (or function) name of a Python block in
    snake case (``block`` when it has none)."""
    if isinstance(block, Native):
        return block.kind
    if isinstance(block, Chain):
        return "chain"
    raw = getattr(getattr(block, "f", None), "__name__", None) or type(block).__name__
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
    return snake if snake.isidentifier() else "block"
