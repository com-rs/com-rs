"""
Channel coding: from convolutional codes to the finite-length limit
===================================================================

BPSK over AWGN, decoded from channel LLRs :math:`2y/\\sigma^2`.

**Bit error rate.** The constraint-length-7 convolutional code (133, 171) with soft
Viterbi decoding gains about 5 dB over uncoded BPSK at :math:`10^{-5}`, and hard
decisions lose about 2 dB of that (Viterbi 1967). Turbo codes (Berrou, Glavieux &
Thitimajshima 1993) and LDPC codes (Gallager 1962; here a regular (3, 6) code built by
progressive edge growth, Hu, Eleftheriou & Arnold 2005, with layered min-sum decoding)
work within about 1 to 2 dB of the Shannon limit of their rate (dotted lines).

**Short blocks.** At :math:`n = 256` the reference is no longer the Shannon limit but
the normal approximation of Polyanskiy, Poor & Verdú (2010), the smallest block error
rate any code of that length and rate can reach. Polar codes (Arıkan 2009) decoded by
successive cancellation are far from it; CRC-aided list decoding (Tal & Vardy 2015)
closes most of the gap — the reason why 5G NR uses it for its control channels.
"""

import matplotlib.pyplot as plt
import numpy as np

# the trial of BPSK over AWGN with a code (the LLRs of comrs.Constellation("psk", 2)),
# shared with the tests that check these curves
from _gallery_utils import coded_error_rates as simulate

import comrs
from comrs.plot import BACKGROUND, GREY, MUTED


def curve(code, points, bits=400_000):
    """BER at every point, an independent draw per point; points of fewer than 20 bit
    errors are left out (an error burst of a single block would be all they show)."""
    blocks = max(bits // code.k, 20)
    out = []
    for i, e in enumerate(points):
        ber, _ = simulate(code, e, blocks, seed=10 * i + 1)
        out.append(ber if ber * blocks * code.k >= 20 else np.nan)
    return np.array(out)


fig, ax = plt.subplots()
eb = np.linspace(-1, 10, 221)
ax.semilogy(
    eb,
    comrs.theory_ber(2, eb, "psk"),
    color=GREY,
    linewidth=1.4,
    label="uncoded BPSK",
)
# (code, points, simulated bits per point)
codes = {
    "conv. (133, 171), hard": (
        comrs.ConvolutionalCode(1000, hard_decision=True),
        np.arange(2.0, 7.6, 0.5),
        3_000_000,
    ),
    "conv. (133, 171), soft": (comrs.ConvolutionalCode(1000), np.arange(1.0, 5.1, 0.5), 3_000_000),
    "turbo, K = 1024, R ≈ 1/3": (comrs.TurboCode(1024), np.arange(-0.25, 1.6, 0.25), 600_000),
    "LDPC (1024, 512)": (
        comrs.LdpcCode.peg(1024, 512, var_degree=3, seed=1),
        np.arange(0.5, 2.6, 0.25),
        1_000_000,
    ),
}
for (name, (code, pts, bits)), marker in zip(codes.items(), ["^", "o", "s", "D"], strict=True):
    ber = curve(code, pts, bits)
    keep = ber > 0
    ax.semilogy(pts[keep], ber[keep], marker=marker, markeredgecolor=BACKGROUND, label=name)
for rate, label in [(1 / 3, "R = 1/3"), (1 / 2, "R = 1/2")]:
    lim = comrs.theory_shannon_limit_ebn0_db(rate)
    ax.axvline(lim, color=GREY, linestyle=":", linewidth=1.0)
    ax.text(lim + 0.1, 2e-6, f"Shannon, {label}", rotation=90, fontsize=8, color=MUTED, va="bottom")
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(-1, 10),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="BPSK over AWGN: coded bit error rates",
)
ax.legend(loc="upper right")
fig.tight_layout()

# %%
# Short blocks: n = 256, k = 128, against the normal approximation.
pts = np.arange(1.0, 3.6, 0.25)
fig, ax = plt.subplots()
na = comrs.theory_normal_approximation(256, 128, np.linspace(0, 4, 81))
ax.semilogy(np.linspace(0, 4, 81), na, color=GREY, linewidth=1.4, label="normal approximation")
short = {
    "polar, SC": comrs.PolarCode(256, 128, design_ebn0_db=2.0),
    "polar, CRC6-aided SCL, L = 8": comrs.PolarCode(
        256, 128, design_ebn0_db=2.0, crc="crc6", list_size=8
    ),
    "LDPC (256, 128)": comrs.LdpcCode.peg(256, 128, var_degree=3, seed=1),
}
for (name, code), marker in zip(short.items(), ["o", "s", "D"], strict=True):
    bler = np.array([simulate(code, e, 1000, seed=1)[1] for e in pts])
    keep = bler > 0
    ax.semilogy(pts[keep], bler[keep], marker=marker, markeredgecolor=BACKGROUND, label=name)
ax.set(
    ylim=(1e-4, 1),
    xlim=(0, 4),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="block error rate",
    title="Short codes, n = 256, R = 1/2",
)
ax.legend(loc="lower left")
fig.tight_layout()
