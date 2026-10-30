"""The classical views of a signal and of the results of a campaign (matplotlib).

Every function draws on ``ax`` (a new figure if ``None``) and returns it. A signal is a
NumPy array, or a :class:`comrs.Frame` from :meth:`comrs.Chain.trace`: the functions
then take its samples per symbol, its sampling rate and its constellation, and draw
every antenna or polarisation.

=====================  ===============================================================
Function               View
=====================  ===============================================================
:func:`constellation`  the samples in the complex plane, over the reference points
:func:`eye`            the eye diagram, traces of ``n_symbols`` symbols
:func:`time`           I and Q, the envelope, the power or the phase against time
:func:`psd`            the power spectral density (Welch)
:func:`curves`         a metric of :class:`comrs.Results` against a variable, with
                       confidence intervals and an optional closed form
=====================  ===============================================================

matplotlib is needed (``pip install matplotlib``); it is imported when a function is
called.

Example:
    >>> import comrs
    >>> from comrs import blocks as B
    >>> link = comrs.Chain(tx=B.Source(constellation="16-QAM", n_symbols=2000),
    ...                    ch=B.Awgn(snr_db=18.0))
    >>> frames = link.trace({}, comrs.Rng(1))
    >>> ax = comrs.plot.constellation(frames["ch"])        # doctest: +SKIP
    >>> fig = frames.plot("constellation")                 # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from comrs._comrs import Frame
from comrs._comrs import psd as _psd

__all__ = ["constellation", "curves", "eye", "psd", "time"]

#: Colours of the antennas or polarisations, in order.
STREAM_COLORS = ["#2a6fdb", "#d95f02", "#1b9e77", "#7570b3"]


def _axes(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return ax


def _streams(x: Any) -> tuple[list[np.ndarray], float, float | None, np.ndarray | None]:
    """The streams, the samples per symbol, the sampling rate and the reference points
    of a frame or an array."""
    if isinstance(x, Frame):
        c = x.constellation
        fs = x.symbol_rate * x.sps if x.symbol_rate else None
        return (
            [np.asarray(s) for s in x.streams],
            x.sps,
            fs,
            None if c is None else np.asarray(c.points),
        )
    a = np.asarray(x)
    streams = [a] if a.ndim == 1 else [row for row in a]
    return streams, 1.0, None, None


def _label(i: int, n: int, names: tuple[str, ...]) -> str | None:
    if n == 1:
        return None
    return names[i] if i < len(names) else f"stream {i}"


def constellation(
    x: Any,
    ax=None,
    *,
    stream: int | None = None,
    skip: int = 0,
    max_points: int = 20_000,
    density: bool = False,
    reference: Any = True,
    size: float = 2.0,
    alpha: float = 0.35,
    names: tuple[str, ...] = ("x", "y"),
):
    """The samples in the complex plane.

    Args:
        x (Frame | array_like): a frame or a complex array (rows: streams).
        ax (matplotlib.axes.Axes | None): the axes (a new figure by default).
        stream (int | None): draw this stream only (all by default, one colour each).
        skip (int): samples left out at the start of every stream (filter transients).
        max_points (int): largest number of points drawn per stream (evenly spaced).
        density (bool): a 2D histogram (log scale) instead of points.
        reference (bool | array_like | Constellation): the reference points: ``True``
            for the constellation of the frame, an array or a constellation, ``False``
            for none.
        size (float): the size of the markers.
        alpha (float): the opacity of the markers.
        names (tuple[str, ...]): the labels of the streams (the polarisations by
            default).

    Returns:
        matplotlib.axes.Axes: the axes.
    """
    ax = _axes(ax)
    streams, _, _, points = _streams(x)
    idx = range(len(streams)) if stream is None else [stream]
    for i in idx:
        z = streams[i][skip:]
        if z.size > max_points:
            z = z[:: -(-z.size // max_points)]
        if density:
            lim = 1.3 * np.max(np.abs(np.concatenate([z.real, z.imag]))) if z.size else 1.0
            ax.hist2d(
                z.real, z.imag, bins=200, range=[[-lim, lim], [-lim, lim]], cmap="Blues", norm="log"
            )
        else:
            ax.plot(z.real, z.imag, ".", markersize=size, alpha=alpha, color=STREAM_COLORS[i % 4],
                    label=_label(i, len(streams), names))  # fmt: skip
    if reference is not True and reference is not False:
        points = np.asarray(getattr(reference, "points", reference))
    if reference is not False and points is not None:
        ax.plot(points.real, points.imag, "+", color="#1f1f1d", markersize=7, mew=1.2)
    ax.set_aspect("equal")
    ax.set(xlabel="in phase", ylabel="quadrature")
    if stream is None and len(streams) > 1 and not density:
        ax.legend(loc="upper right", markerscale=6)
    return ax


def eye(
    x: Any,
    ax=None,
    *,
    sps: int | None = None,
    n_symbols: int = 2,
    traces: int = 300,
    skip: int = 0,
    part: str = "real",
    stream: int = 0,
):
    """The eye diagram: ``traces`` segments of ``n_symbols`` symbols overlaid.

    Args:
        x (Frame | array_like): a frame or a complex array.
        ax (matplotlib.axes.Axes | None): the axes (a new figure by default).
        sps (int | None): samples per symbol (that of the frame by default; at least 2).
        n_symbols (int): symbols per trace.
        traces (int): number of traces.
        skip (int): samples left out at the start.
        part (str): ``"real"`` (I), ``"imag"`` (Q) or ``"abs"``.
        stream (int): the stream drawn.

    Returns:
        matplotlib.axes.Axes: the axes.

    Raises:
        ValueError: below 2 samples per symbol.
    """
    ax = _axes(ax)
    streams, f_sps, _, _ = _streams(x)
    k = round(sps or f_sps)
    if k < 2:
        raise ValueError("an eye diagram needs at least 2 samples per symbol")
    z = streams[stream][skip:]
    v = {"real": z.real, "imag": z.imag, "abs": np.abs(z)}[part]
    span = n_symbols * k
    starts = np.arange(0, v.size - span - 1, k)[:traces]
    t = np.arange(span + 1) / k
    for s in starts:
        ax.plot(t, v[s : s + span + 1], color=STREAM_COLORS[0], alpha=0.08, linewidth=0.8)
    ax.set(xlabel="time (symbols)", ylabel={"real": "I", "imag": "Q", "abs": "envelope"}[part])
    return ax


def time(
    x: Any,
    ax=None,
    *,
    part: str = "iq",
    fs: float | None = None,
    start: int = 0,
    n: int | None = 400,
    stream: int = 0,
):
    """The signal against time.

    Args:
        x (Frame | array_like): a frame or a complex array.
        ax (matplotlib.axes.Axes | None): the axes (a new figure by default).
        part (str): ``"iq"`` (I and Q), ``"abs"`` (envelope), ``"power"`` or
            ``"phase"`` (unwrapped).
        fs (float | None): sampling rate (Hz); the time axis is in samples without it
            (that of the frame by default).
        start (int): the first sample drawn.
        n (int | None): the number of samples drawn (all with ``None``).
        stream (int): the stream drawn.

    Returns:
        matplotlib.axes.Axes: the axes.
    """
    ax = _axes(ax)
    streams, _, f_fs, _ = _streams(x)
    fs = fs or f_fs
    z = streams[stream][start : None if n is None else start + n]
    t: np.ndarray = np.arange(start, start + z.size)
    if fs:
        t = t / fs
    if part == "iq":
        ax.plot(t, z.real, color=STREAM_COLORS[0], label="I")
        ax.plot(t, z.imag, color=STREAM_COLORS[1], label="Q")
        ax.legend(loc="upper right")
    else:
        v = {"abs": np.abs(z), "power": np.abs(z) ** 2, "phase": np.unwrap(np.angle(z))}[part]
        ax.plot(t, v, color=STREAM_COLORS[0])
    ax.set(xlabel="time (s)" if fs else "sample", ylabel={"iq": "amplitude", "abs": "envelope",
           "power": "power", "phase": "phase (rad)"}[part])  # fmt: skip
    return ax


def psd(
    x: Any,
    ax=None,
    *,
    fs: float | None = None,
    nfft: int = 1024,
    names: tuple[str, ...] = ("x", "y"),
):
    """The power spectral density (Welch, Hann window, 50 % overlap) of every stream,
    in dB, against the frequency: in GHz or MHz for a sampling rate in hertz, relative
    to the symbol rate for a frame on a normalised time axis, to the sampling rate for
    an array without ``fs``.

    Args:
        x (Frame | array_like): a frame or a complex array (rows: streams).
        ax (matplotlib.axes.Axes | None): the axes (a new figure by default).
        fs (float | None): sampling rate (Hz; that of the frame by default).
        nfft (int): length of the segments.
        names (tuple[str, ...]): the labels of the streams.

    Returns:
        matplotlib.axes.Axes: the axes.
    """
    ax = _axes(ax)
    streams, _, f_fs, _ = _streams(x)
    # a frame on a normalised time axis (symbol rate 1): frequencies in symbol rates
    per_symbol = fs is None and isinstance(x, Frame) and x.symbol_rate == 1.0
    fs = fs or f_fs or 1.0
    scale, unit = (1e9, "GHz") if fs >= 1e9 else (1e6, "MHz") if fs >= 1e6 else (1.0, "")
    for i, z in enumerate(streams):
        f, p = _psd(z, fs, nfft=min(nfft, max(16, z.size)))
        ax.plot(np.asarray(f) / scale, 10 * np.log10(np.maximum(np.asarray(p), 1e-300)),
                color=STREAM_COLORS[i % 4], label=_label(i, len(streams), names))  # fmt: skip
    xlabel = (
        f"frequency ({unit})"
        if unit
        else "frequency / symbol rate"
        if per_symbol
        else "frequency / sampling rate"
    )
    ax.set(xlabel=xlabel, ylabel="PSD (dB/Hz)" if unit else "PSD (dB)")
    if len(streams) > 1:
        ax.legend(loc="upper right")
    return ax


def curves(
    results: Any,
    x: str,
    metric: str,
    *,
    by: str | None = None,
    ax=None,
    theory: Callable[..., Any] | None = None,
    errorbars: bool = True,
    logy: bool | None = None,
    min_count: int = 0,
    **condition: Any,
):
    """A metric of :class:`comrs.Results` against the variable ``x``, one curve per
    value of ``by``, for the conditions that match ``condition``.

    Args:
        results (Results | list[dict]): the results of a campaign.
        x (str): the variable of the abscissa.
        metric (str): the metric.
        by (str | None): one curve per value of this variable.
        ax (matplotlib.axes.Axes | None): the axes (a new figure by default).
        theory (callable | None): a closed form drawn as a dotted line: ``theory(x)``,
            or ``theory(x, value)`` with ``by``, on a fine grid of ``x`` (an array: the
            ``theory_*`` functions of the package take one).
        errorbars (bool): draw the confidence intervals.
        logy (bool | None): a logarithmic axis (by default for error rates).
        min_count (int): points with fewer errors are left out.
        **condition: the values of the other variables.

    Returns:
        matplotlib.axes.Axes: the axes.
    """
    from comrs.results import Results

    ax = _axes(ax)
    res = results if isinstance(results, Results) else Results(results)
    groups = res.values(by) if by is not None else [None]
    kind = next((r["metrics"][metric].get("kind") for r in res if metric in r["metrics"]), "mean")
    logy = (kind == "rate") if logy is None else logy
    markers = "osD^v<>"
    for i, g in enumerate(groups):
        cond = dict(condition, **({by: g} if by is not None else {}))
        xs, ys, lo, hi = res.series(x, metric, interval=True, min_count=min_count, **cond)
        if logy:
            keep = ys > 0
            xs, ys, lo, hi = xs[keep], ys[keep], lo[keep], hi[keep]
        label = None if by is None else f"{by} = {g}" if not isinstance(g, str) else g
        if errorbars:
            line = ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker=markers[i % len(markers)],
                               markeredgecolor="#fcfcfb", capsize=2, label=label)[0]  # fmt: skip
        else:
            (line,) = ax.plot(
                xs, ys, marker=markers[i % len(markers)], markeredgecolor="#fcfcfb", label=label
            )
        if theory is not None and xs.size:
            fine = np.linspace(np.min(xs), np.max(xs), 200)
            yt = theory(fine) if by is None else theory(fine, g)
            ax.plot(fine, yt, ":", color=line.get_color())
    if logy:
        ax.set_yscale("log")
    ax.set(xlabel=x, ylabel=metric)
    if by is not None:
        ax.legend()
    return ax
