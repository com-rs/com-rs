"""The closed forms the simulated error rates are compared with, written here from the
literature so that the reference belongs to none of the libraries compared.

- Gray-mapped square M-QAM over AWGN, exact bit error rate: K. Cho and D. Yoon, "On
  the general BER expression of one- and two-dimensional amplitude modulations", IEEE
  Trans. Commun. 50(7), 2002, eqs. (14)-(16). M = 4 gives Q(sqrt(2 Eb/N0)), the BER of
  BPSK.
- The same over flat Rayleigh fading: every term erfc(a sqrt(g)) averaged over an
  exponential SNR g of mean G is 1 - sqrt(a^2 G / (1 + a^2 G)) (Proakis & Salehi,
  Digital Communications, 5th ed., Sec. 13.3); BPSK gives (1 - sqrt(G / (1 + G))) / 2.
- Zero forcing over n_t x n_r i.i.d. Rayleigh fading: the SNR of every stream is that
  of n_r - n_t + 1 branches of maximal-ratio combining (J. H. Winters, J. Salz and
  R. D. Gitlin, IEEE Trans. Commun. 42(2/3/4), 1994); 4 x 4 is one branch.
- Soft-decision Viterbi decoding: the union (transfer-function) bound
  Pb <= sum_d c_d Q(sqrt(2 d R Eb/N0)) (A. J. Viterbi, "Convolutional codes and their
  performance in communication systems", IEEE Trans. Commun. Technol. 19(5), 1971),
  with the spectrum c_d enumerated from the trellis below.
"""

from __future__ import annotations

import math

import numpy as np


def qfunc(x):
    return 0.5 * np.vectorize(math.erfc)(np.asarray(x, dtype=float) / math.sqrt(2.0))


def _qam_terms(m: int) -> list[tuple[float, float]]:
    """The pairs (weight w, factor a) of the BER of Gray M-QAM, Pb = sum w erfc(a sqrt(Eb/N0))
    (Cho & Yoon 2002, eqs. (14)-(16))."""
    if m == 2:
        return [(0.5, 1.0)]
    sm = math.isqrt(m)
    if sm * sm != m:
        raise ValueError("square M-QAM only")
    nb = int(math.log2(sm))
    terms = []
    for k in range(1, nb + 1):
        for i in range(int((1 - 2.0**-k) * sm)):
            f = math.floor(i * 2 ** (k - 1) / sm)
            w = (-1) ** f * (2 ** (k - 1) - math.floor(i * 2 ** (k - 1) / sm + 0.5)) / sm / nb
            a = (2 * i + 1) * math.sqrt(3 * math.log2(m) / (2 * (m - 1)))
            terms.append((w, a))
    return terms


def ber_qam_awgn(m: int, ebn0_db):
    """Exact BER of Gray square M-QAM (m = 2: BPSK) over AWGN."""
    g = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    erfc = np.vectorize(math.erfc)
    return sum(w * erfc(a * np.sqrt(g)) for w, a in _qam_terms(m))


def ber_qam_rayleigh(m: int, ebn0_db):
    """BER of Gray square M-QAM (m = 2: BPSK) over flat Rayleigh fading with perfect
    channel knowledge, mean Eb/N0 ``ebn0_db``."""
    g = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    return sum(w * (1 - np.sqrt(a * a * g / (1 + a * a * g))) for w, a in _qam_terms(m))


def conv_spectrum(generators: list[int], constraint: int, d_max: int) -> dict[int, int]:
    """The bit-error weights c_d (total information weight of the paths of output
    weight d that leave the zero state once and merge back) of a rate-1/n feedforward
    convolutional code, for d <= d_max. The generators are octal in the usual
    convention: the most significant tap multiplies the current input bit."""
    nu = constraint - 1
    n_states = 1 << nu

    def step(state: int, u: int) -> tuple[int, int]:
        reg = (u << nu) | state  # bit nu: current input, bits nu-1..0: past inputs
        w = sum(bin(reg & g).count("1") % 2 for g in generators)
        return reg >> 1, w

    # paths[state][d] = (number of paths, total information weight)
    first, w0 = step(0, 1)
    frontier = {(first, w0): (1, 1)}
    spectrum: dict[int, int] = {}
    while frontier:
        nxt: dict[tuple[int, int], tuple[int, int]] = {}
        for (s, d), (cnt, iw) in frontier.items():
            for u in (0, 1):
                s2, w = step(s, u)
                d2 = d + w
                if d2 > d_max:
                    continue
                if s2 == 0:
                    spectrum[d2] = spectrum.get(d2, 0) + iw + u * cnt
                    continue
                c0, i0 = nxt.get((s2, d2), (0, 0))
                nxt[(s2, d2)] = (c0 + cnt, i0 + iw + u * cnt)
        frontier = nxt
        assert all(s < n_states for s, _ in frontier)
    return dict(sorted(spectrum.items()))


def conv_union_bound(generators: list[int], constraint: int, rate: float, ebn0_db, d_max: int = 20):
    """Union bound on the BER of soft-decision Viterbi decoding over BPSK and AWGN."""
    spec = conv_spectrum(generators, constraint, d_max)
    g = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    return sum(c * qfunc(np.sqrt(2 * d * rate * g)) for d, c in spec.items())
