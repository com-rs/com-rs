"""The blocks of a chain: Python blocks, native blocks of the registry, grid variables.

- :class:`Block` is the interface: ``block(frame, rng)`` transforms a
  :class:`~comrs.Frame` in place, drawing from an explicit :class:`~comrs.Rng`;
  subclass it to write a block in Python (a plain callable works too).
- :class:`Native` is a block of the registry of declarative links
  (``comrs.spec_blocks()``), run by the Rust code; the classes of :mod:`comrs.blocks`
  build them with their parameters as keyword arguments.
- :func:`var` makes a parameter that takes the value of a grid variable (``$name`` in
  a specification).

Example:
    >>> import comrs
    >>> from comrs import blocks as B
    >>> b = B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db"))
    >>> b.spec()
    {'type': 'awgn_ebn0', 'ebn0_db': '$ebn0_db'}
    >>> frame = comrs.Frame()
    >>> B.Source(constellation="QPSK", n_symbols=4)(frame, comrs.Rng(1))
    >>> frame.samples.size
    4
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from comrs._comrs import Frame, LinkBlock, Rng, spec_blocks

__all__ = ["Block", "Native", "Var", "var"]


class Var:
    """A reference to a grid variable, substituted when the chain is built for a
    condition (``$name`` in a specification).

    Args:
        name (str): the name of the variable of the grid.

    Example:
        >>> import comrs
        >>> comrs.var("ebn0_db")
        var('ebn0_db')
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and other.name == self.name

    def __hash__(self) -> int:
        return hash(("Var", self.name))

    def __repr__(self) -> str:
        return f"var({self.name!r})"


def var(name: str) -> Var:
    """The grid variable ``name`` as a block parameter.

    Args:
        name (str): the name of the variable of the grid.

    Returns:
        Var: the reference, substituted for every condition.
    """
    return Var(name)


def _resolve(value: Any, condition: Mapping[str, Any] | None) -> Any:
    """Substitute the variables of a parameter value."""
    if isinstance(value, Var):
        if condition is None or value.name not in condition:
            raise KeyError(f"the condition has no variable {value.name!r}")
        return condition[value.name]
    if isinstance(value, dict):
        return {k: _resolve(v, condition) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_resolve(v, condition) for v in value]
    return value


def _to_spec_value(value: Any) -> Any:
    """A parameter value as it appears in a specification."""
    if isinstance(value, Var):
        return f"${value.name}"
    if isinstance(value, dict):
        return {k: _to_spec_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_spec_value(v) for v in value]
    return value


class Block:
    """A block of a chain: ``block(frame, rng)`` transforms the frame in place.

    Subclass it and implement :meth:`__call__` to write a block in Python; the frame
    exposes ``samples`` (read and write), ``sps``, ``tx_bits``, ``tx_symbols`` and
    ``constellation``. A block that depends on the condition overrides :meth:`bind`.

    Example:
        >>> import comrs
        >>> class Gain(comrs.Block):
        ...     def __call__(self, frame, rng):
        ...         frame.samples = 2 * frame.samples
        >>> f = comrs.Frame()
        >>> f.samples = [1.0]
        >>> Gain()(f, comrs.Rng(0))
        >>> f.samples.tolist()
        [(2+0j)]
    """

    def __call__(self, frame: Frame, rng: Rng) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def bind(
        self,
        condition: Mapping[str, Any] | None,
        *,
        fiber: Mapping[str, Any] | None = None,
        name: str | None = None,
        label: str | None = None,
        profile: bool = False,
    ) -> Callable[[Frame, Rng], None]:
        """The callable that applies the block for a condition (itself by default).

        Args:
            condition (dict | None): the values of the grid variables.
            fiber (dict | None): the ``fiber`` of the specification, for the optical
                blocks.
            name (str | None): the name of the campaign, for the blocks that write
                files.
            label (str | None): the name of the block in the chain.
            profile (bool): time the blocks (chains only).

        Returns:
            Callable[[Frame, Rng], None]: the block of the condition.
        """
        return self


class Native(Block):
    """A block of the registry of declarative links, ``Native("awgn_ebn0",
    ebn0_db=10)``; the classes of :mod:`comrs.blocks` build them.

    Args:
        kind (str): the type of the block (positional only: a block may have a
            parameter of that name, as ``pulse_shape``).
        **params: its parameters; :func:`var` makes one a grid variable.

    Example:
        >>> import comrs
        >>> comrs.Native("awgn_ebn0", ebn0_db=10).spec("channel")
        {'type': 'awgn_ebn0', 'label': 'channel', 'ebn0_db': 10}
    """

    def __init__(self, kind: str, /, **params: Any) -> None:
        self.kind = kind
        self.params = params

    def spec(self, label: str | None = None) -> dict[str, Any]:
        """The block as it appears in a specification.

        Args:
            label (str | None): its name in the chain (kept when it differs from the
                type; an explicit ``label`` parameter, a probe's, wins).

        Returns:
            dict: ``{"type": …, "label": …, <parameters>}``.
        """
        d: dict[str, Any] = {"type": self.kind}
        if label is not None and label != self.kind:
            d["label"] = label
        d.update({k: _to_spec_value(v) for k, v in self.params.items()})
        return d

    def bind(
        self,
        condition: Mapping[str, Any] | None,
        *,
        fiber: Mapping[str, Any] | None = None,
        name: str | None = None,
        label: str | None = None,
        profile: bool = False,
    ) -> Callable[[Frame, Rng], None]:
        """The block of a condition, built by the Rust registry (see
        :meth:`Block.bind`): ``label`` is its name in the chain, ``name`` the
        campaign, and the condition is labelled as the Rust runner does, for the blocks
        that report or write files under their label (``probe``, ``dump``).

        Args:
            condition (dict | None): the values of the grid variables.
            fiber (dict | None): the ``fiber`` of the specification.
            name (str | None): the name of the campaign.
            label (str | None): the name of the block in the chain.
            profile (bool): unused (a single block).

        Returns:
            Callable[[Frame, Rng], None]: the ``apply`` method of a :class:`LinkBlock`.

        Example:
            >>> import comrs
            >>> from comrs import blocks as B
            >>> apply = B.AwgnEbn0(ebn0_db=comrs.var("x")).bind({"x": 5.0})
            >>> callable(apply)
            True
        """
        params = {k: _resolve(v, condition) for k, v in self.params.items()}
        spec = {"type": self.kind, **params}
        if label is not None:
            spec.setdefault("label", label)
        cond = dict(condition) if name is not None and condition else None
        block = LinkBlock(spec, fiber=fiber, name=name, condition=cond)
        return block.apply

    def __call__(self, frame: Frame, rng: Rng) -> None:
        self.bind(None)(frame, rng)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{_class_name(self.kind)}({args})"


class _Function(Block):
    """A plain callable used as a block."""

    def __init__(self, f: Callable[[Frame, Rng], None]) -> None:
        self.f = f

    def __call__(self, frame: Frame, rng: Rng) -> None:
        self.f(frame, rng)

    def __repr__(self) -> str:
        return getattr(self.f, "__name__", "function")


def _as_block(b: Any) -> Block:
    if isinstance(b, Block):
        return b
    if callable(b):
        return _Function(b)
    raise TypeError(f"{b!r} is not a block")


def _class_name(kind: str) -> str:
    """The class of :mod:`comrs.blocks` of a block type (``awgn_ebn0`` → ``AwgnEbn0``)."""
    return "".join(p.capitalize() for p in kind.split("_"))


def _native_init(block: Native, kind: str, /, **params: Any) -> None:
    """The constructor of the classes of :mod:`comrs.blocks`: the parameters equal to
    their default are not given to the block (the block applies the same default), so
    that its specification lists the other parameters only."""
    defaults = type(block).__init__.__kwdefaults__ or {}

    def is_default(k: str, v: Any) -> bool:
        if k not in defaults:
            return False
        d = defaults[k]
        return v is d or (type(v) is type(d) and isinstance(v, str | int | float | bool) and v == d)

    Native.__init__(block, kind, **{k: v for k, v in params.items() if not is_default(k, v)})


def _check_registry(classes: Mapping[str, Any]) -> None:
    """Warn when the classes of :mod:`comrs.blocks` differ from the registry of the
    extension (``tools/gen_blocks.py`` regenerates them)."""
    registry = spec_blocks()
    expected = {_class_name(k): set(v["params"]) for k, v in registry.items()}
    actual = {
        n: set(getattr(c, "param_names", ()))
        for n, c in classes.items()
        if isinstance(c, type) and issubclass(c, Native) and c is not Native
    }
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(n for n in set(expected) & set(actual) if expected[n] != actual[n])
        warnings.warn(
            "comrs.blocks is out of date with the registry of the extension "
            f"(missing {missing}, extra {extra}, other parameters {changed}): "
            "run python tools/gen_blocks.py",
            RuntimeWarning,
            stacklevel=3,
        )
