"""
5G NR channel codes: LDPC and polar
===================================

The data channels of 5G NR use a quasi-cyclic LDPC code, the control channels a polar
code (3GPP TS 38.212; T. Richardson and S. Kudekar, "Design of low-density parity check
codes for 5G new radio," *IEEE Commun. Mag.* 56(3), 2018; V. Bioglio, C. Condo and
I. Land, "Design of polar codes in 5G New Radio," *IEEE Commun. Surveys Tuts.* 23(1),
2021). :class:`comrs.NrLdpc` and :class:`comrs.NrPolar` build them from the tables of
the standard — the two LDPC base graphs, the polar sequence of 1024 positions, the
interleavers — with their rate matching to any number of transmitted bits.

Their encoders give the same bits as the reference vectors of Sionna
(``tests/test_sionna_vectors.py``): the LDPC codewords of 28 sizes from 64 to 8448
information bits, both base graphs, and the polar codewords of the uplink control
information with puncturing, shortening and repetition.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs

# %%
# The LDPC code: base graphs and lifting
# --------------------------------------
#
# Base graph 1 (46 × 68) serves large blocks and high rates, base graph 2 (42 × 52)
# small blocks and low rates (TS 38.212, Sec. 7.2.2); every nonzero entry becomes a
# :math:`Z \times Z` cyclically shifted identity, :math:`Z` the smallest lifting size
# that holds the block. The first :math:`2Z` systematic bits are never sent, and the
# circular buffer gives any rate.

for k, n in [(100, 300), (1000, 1500), (4000, 4800), (8448, 25344)]:
    code = comrs.NrLdpc(k, n)
    print(f"K' = {k:5d}, E = {n:5d}: base graph {code.base_graph}, Z = {code.lifting_size}")

# %%
# Block error rates over BPSK and AWGN
# ------------------------------------
#
# Belief propagation (50 iterations of sum-product) for the LDPC code, CRC-aided SC
# list (L = 8) for the polar code, against the normal approximation of the best code of
# the same length and rate (Y. Polyanskiy, H. V. Poor and S. Verdú, *IEEE Trans. Inf.
# Theory* 56(5), 2010; :func:`comrs.normal_approximation`, dashed). At a block error
# rate of 10⁻², the LDPC code of 500 bits at rate 1/2 sits about 0.6 dB from it, the
# polar code of 40 bits in 128 about 1.1 dB, part of which is its CRC
# (``validation/tests/nr_codes.rs``).

rng = comrs.Rng(31)


def bler(code, ebn0_db, blocks):
    k, n = code.k, code.n
    b = comrs.bits(k * blocks, rng)
    c = code.encode(b).astype(float)
    s2 = 1 / (2 * k / n * 10 ** (ebn0_db / 10))
    y = 1 - 2 * c + np.sqrt(s2) * rng.gaussian(c.size)
    return np.mean(np.any((code.decode(2 * y / s2) != b).reshape(blocks, k), axis=1))


cases = [
    ("LDPC, K' = 500, E = 1000", comrs.NrLdpc(500, 1000), np.arange(0.5, 2.01, 0.5), "#2a78d6"),
    ("LDPC, K' = 500, E = 2000", comrs.NrLdpc(500, 2000), np.arange(-0.5, 1.01, 0.5), "#1b9e77"),
    (
        "polar uplink, A = 40, E = 128",
        comrs.NrPolar(40, 128, list_size=8),
        np.arange(0.5, 3.51, 0.5),
        "#c0392b",
    ),
]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for name, code, grid, color in cases:
    p = [bler(code, e, 250) for e in grid]
    ax.semilogy(grid, np.maximum(p, 1e-3), "o-", color=color, markersize=4, label=name)
    fine = np.linspace(grid[0] - 1, grid[-1], 60)
    ax.semilogy(
        fine, comrs.normal_approximation(code.n, code.k, fine), "--", color=color, linewidth=1
    )
ax.set(xlabel="$E_b/N_0$ (dB)", ylabel="block error rate", ylim=(3e-3, 1.1))
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()

# %%
# Rate matching of the polar code
# -------------------------------
#
# The mother code length is a power of two; the sub-block interleaver and the bit
# selection reach any :math:`E`: the last :math:`E` bits (puncturing) at low rate, the
# first :math:`E` (shortening) at high rate, the word repeated when :math:`E \ge N`.

for a, e in [(30, 45), (30, 100), (30, 200), (100, 1500)]:
    code = comrs.NrPolar(a, e)
    n = code.mother_length
    kind = "repetition" if e >= n else ("puncturing" if 16 * (a + 11) <= 7 * e else "shortening")
    print(f"A = {a:3d}, E = {e:4d}: N = {n:4d}, {kind}")
