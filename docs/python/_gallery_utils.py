"""What the examples of the gallery and the tests that reproduce them share: the folders
of the repository, for the examples that read its specifications and recordings, and
the trials and chains that an example draws and a test checks at a few points (the
colours are those of :mod:`comrs.plot`).

sphinx-gallery runs every example with ``docs/python`` on the import path (``conf.py``
puts it there), so that an example imports this module as ``from _gallery_utils import
SPECS``; the tests put it there too (``tests/conftest.py``). To run an example directly,
put the folder on the path::

    PYTHONPATH=docs/python python docs/python/examples/simulation/plot_01_chains.py

The gallery draws every example in the style of :func:`comrs.plot.style` (``conf.py``
sets it before each one); an example run directly gets it with ``comrs.plot.style()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import comrs
from comrs import blocks as B

# --- the repository -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "specs"

# --- coded BPSK over AWGN (coding/plot_01, plot_02, plot_03; tests/test_performance.py,
# tests/test_fec_codes.py) ------------------------------------------------------------

BPSK = comrs.Constellation("psk", 2)


def bpsk_awgn_decisions(code: Any, bits: np.ndarray, ebn0_db: float, rng: comrs.Rng) -> np.ndarray:
    """The decoded bits of ``code`` for ``bits`` sent in BPSK over a real AWGN channel at
    :math:`E_b/N_0` = ``ebn0_db`` (dB), from the soft LLRs of the BPSK constellation;
    the noise is drawn from ``rng``.

    The noise variance per real dimension is :math:`\\sigma^2 = 1 / (2 R\\, E_b/N_0)` for a
    code of rate :math:`R = k/n`, and the LLRs are those of :meth:`comrs.Constellation.llr` for
    the complex noise variance :math:`N_0 = 2\\sigma^2`."""
    c = code.encode(bits)
    s2 = 1 / (2 * (code.k / code.n) * comrs.db2lin(ebn0_db))
    y = BPSK.modulate(c).real + np.sqrt(s2) * rng.gaussian(c.size)
    return code.decode(BPSK.llr(y, 2 * s2))


def coded_error_rates(code: Any, ebn0_db: float, blocks: int, seed: int) -> tuple[float, float]:
    """The bit and block error rates of ``blocks`` codewords of ``code`` over BPSK and
    AWGN (:func:`bpsk_awgn_decisions`): the bits drawn from ``Rng(seed)``, the noise from
    ``Rng(seed + 1)``."""
    b = comrs.bits(code.k * blocks, comrs.Rng(seed))
    e = (bpsk_awgn_decisions(code, b, ebn0_db, comrs.Rng(seed + 1)) != b).reshape(blocks, -1)
    return float(e.mean()), float(e.any(axis=1).mean())


# --- 2 × 2 MIMO-OFDM (receiver/plot_03; tests/test_performance.py) ------------------


def mimo_ofdm_ser(
    ofdm: comrs.Ofdm,
    channel: comrs.TdlChannel,
    constellation: comrs.Constellation,
    snr_db: float,
    method: str,
    frames: int,
    *,
    seed: int,
    condition: int,
) -> float:
    """The symbol error rate of spatial multiplexing over a multipath channel with OFDM
    and perfect channel knowledge, detected subcarrier by subcarrier
    (:func:`comrs.mimo_detect`, ``method``), at an :math:`E_s/N_0` of ``snr_db`` per
    receive antenna; frame ``f`` draws from ``Rng.stream(seed, "mimo-ofdm", condition,
    f)``. One realisation of the channel per frame, static over its four OFDM symbols."""
    n_tx, n, k = channel.n_tx, ofdm.n_fft, ofdm.n_subcarriers
    err = tot = 0
    for f in range(frames):
        rng = comrs.Rng.stream(seed, "mimo-ofdm", condition, f)
        idx = rng.integers(constellation.order, n_tx * k * 4).reshape(n_tx, -1)
        x = np.stack([ofdm.modulate(constellation.points[i]) for i in idx])
        y, taps = channel.fade(x, rng)
        s2 = n_tx * n / (k * comrs.db2lin(snr_db))
        y = y + rng.complex_noise(y.size, s2).reshape(y.shape)
        rx = np.stack([ofdm.demodulate(r) for r in y])  # (n_rx, 4 · k)
        h = taps.frequency_response(0, ofdm.bins, n)  # (k, n_rx, n_tx), static
        d = comrs.mimo_detect(np.tile(h, (4, 1, 1)), rx.T, constellation, s2 * k / n, method)
        err += int((d != idx.T).sum())
        tot += idx.size
    return err / tot


# --- BICM with an LDPC code (simulation/plot_02; tests/test_performance.py) ---------

LDPC_1024 = {"type": "ldpc", "n": 1024, "k": 512, "seed": 1}


def bicm_chain(fading: bool, code: dict[str, Any] = LDPC_1024) -> comrs.Chain:
    """Bit-interleaved coded modulation: ``code`` (four codewords per trial, interleaved)
    on 16-QAM over AWGN at the grid variable ``ebn0_db``, with fast Rayleigh fading (a
    new coefficient per symbol, equalised by ``mimo_rx``) when ``fading``."""
    blocks = [B.FecEncode(code=code, blocks=4, interleave=True), B.Map(constellation="16-QAM")]
    if fading:
        blocks.append(B.Rayleigh(coherence_samples=1))
    blocks.append(B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")))
    if fading:
        blocks.append(B.MimoRx())
    blocks += [B.Demap(), B.FecDecode(code=code, interleave=True)]
    return comrs.Chain(*blocks)
