"""
Hu, Eleftheriou & Arnold 2005, Fig. 5: a regular PEG code of length 504
========================================================================

Fig. 5 of X.-Y. Hu, E. Eleftheriou and D. M. Arnold, "Regular and irregular
progressive edge-growth Tanner graphs", *IEEE Trans. Inf. Theory* 51(1), 2005,
gives the bit and block error rates of a regular LDPC code built by progressive edge
growth, with :math:`n = 504`, :math:`m = 252` parity checks, symbol-node degree
:math:`d_s = 3` and check-node degree :math:`d_c = 6`, over the AWGN channel with BPSK
and sum-product decoding, next to MacKay's code and a random graph of the same
parameters.

:meth:`comrs.LdpcCode.peg` builds a graph by the same algorithm; the unit test
``literature`` of ``com_fec::ldpc`` checks that it has the girth 8 and the check
degrees of the paper's Sec. VI. The graph is not the paper's graph (the algorithm
breaks its ties at random), so the curves are those of a PEG code of these
parameters, decoded by the flooding sum-product algorithm with at most 100
iterations: open Fig. 5 of the paper next to this figure and compare the PEG curves,
solid for the bit error rate and dashed for the block error rate. The dotted line is
uncoded BPSK.
"""

import matplotlib.pyplot as plt
import numpy as np
from _gallery_utils import bpsk_awgn_decisions

import comrs
from comrs.plot import BACKGROUND, GREY

code = comrs.LdpcCode.peg(504, 252, var_degree=3, seed=1)
code.set_decoder("sum_product", layered=False, max_iterations=100)
print(code, "rate", code.rate)


def error_rates(ebn0_db, seed, min_block_errors=100, max_blocks=40_000, batch=500):
    """Bit and block error rates over AWGN with BPSK, until `min_block_errors` block
    errors or `max_blocks` codewords."""
    bit_errors = block_errors = blocks = 0
    t = 0
    while block_errors < min_block_errors and blocks < max_blocks:
        rng = comrs.Rng.stream(seed, "peg504", int(10 * ebn0_db), t)
        b = comrs.bits(code.k * batch, rng)
        e = (bpsk_awgn_decisions(code, b, ebn0_db, rng) != b).reshape(batch, -1)
        bit_errors += int(e.sum())
        block_errors += int(e.any(axis=1).sum())
        blocks += batch
        t += 1
    return bit_errors / (blocks * code.k), block_errors / blocks


points = np.arange(1.0, 3.76, 0.25)
rates = np.array([error_rates(e, seed=5) for e in points])
rates[rates == 0] = np.nan  # no error observed: nothing to draw

fig, ax = plt.subplots()
fine = np.linspace(0, 4, 81)
ax.semilogy(fine, comrs.theory_ber(2, fine, "psk"), ":", color=GREY, linewidth=1.4,
            label="uncoded BPSK")  # fmt: skip
ax.semilogy(
    points, rates[:, 0], "o-", markeredgecolor=BACKGROUND, label="PEG (504, 252), bit error rate"
)
ax.semilogy(
    points, rates[:, 1], "s--", markeredgecolor=BACKGROUND, label="PEG (504, 252), block error rate"
)
ax.set(
    xlim=(0, 4),
    ylim=(1e-6, 1),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="error rate",
    title="Regular (3, 6) PEG code, n = 504 (Hu et al. 2005, Fig. 5)",
)
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# The waterfall of a regular (3, 6) code of this length lies about 1.5 dB from the
# Shannon limit of rate 1/2 (0.19 dB); the paper's Fig. 5 shows the PEG code below
# MacKay's code and the random graph at every SNR.
print(f"Shannon limit at rate 1/2: {comrs.theory_shannon_limit_ebn0_db(0.5):.2f} dB")
for e, (ber, bler) in zip(points, rates, strict=True):
    print(f"Eb/N0 = {e:.2f} dB: BER {ber:.2e}, BLER {bler:.2e}")
