"""
MIMO detection: linear, successive cancellation and maximum likelihood
======================================================================

Four QPSK streams sent from four antennas and received on four, over i.i.d. Rayleigh
fading with a new channel for every vector: :math:`y = H x + w`. The detectors trade
complexity for diversity:

- **ZF** inverts the channel and enhances the noise: diversity
  :math:`n_r - n_t + 1 = 1`, the slope of a single Rayleigh branch;
- **MMSE** balances interference and noise — better at low SNR, the same slope;
- **MMSE-SIC** (V-BLAST) detects the strongest stream first and subtracts it;
- **ML**, here by the sphere decoder (exact), reaches the full receive diversity
  :math:`n_r = 4`.

The ZF curve follows the closed form of maximal-ratio combining with one branch,
:math:`\\frac{1}{2}\\bigl(1 - \\sqrt{\\gamma/(1 + \\gamma)}\\bigr)` per bit (Gray QPSK is
two BPSK).
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs

NR = NT = 4
c = comrs.Constellation("qam", 4).normalized()
k = c.bits_per_symbol
points = np.arange(0.0, 26.0, 3.0)
methods = {"ZF": "zf", "MMSE": "mmse", "MMSE-SIC": "mmse_sic", "ML (sphere)": "sphere"}
ber = {name: [] for name in methods}
for i, eb in enumerate(points):
    s2 = 1 / (k * comrs.db2lin(eb))  # unit energy per stream: Es/N0 = k Eb/N0
    for name, m in methods.items():
        errors = total = t = 0
        while errors < 200 and total < 4e6:
            rng = comrs.Rng.stream(9, m, i, t)
            n = 20_000
            h = comrs.mimo_channel(NR, NT, rng, n=n)
            idx = rng.integers(4, n * NT).reshape(n, NT)
            y = np.einsum("nij,nj->ni", h, c.points[idx]) + rng.complex_noise(n * NR, s2).reshape(
                n, NR
            )
            d = comrs.mimo_detect(h, y, c, s2, m)
            errors += int(np.unpackbits((d ^ idx).astype(np.uint8)[..., None], axis=-1).sum())
            total += idx.size * k
            t += 1
        # a point needs enough errors to be drawn
        ber[name].append(errors / total if errors >= 50 else np.nan)

# %%
fig, ax = plt.subplots()
eb = np.linspace(0, 25, 251)
g = 10 ** (eb / 10)
ax.semilogy(
    eb,
    0.5 * (1 - np.sqrt(g / (1 + g))),
    color="#8a8984",
    linewidth=1.2,
    linestyle=":",
    label="1 Rayleigh branch (theory)",
)
for (name, values), marker in zip(ber.items(), ["o", "s", "D", "^"], strict=True):
    v = np.array(values)
    keep = v > 0  # NaN (too few errors) compares false
    ax.semilogy(points[keep], v[keep], marker=marker, markeredgecolor="#fcfcfb", label=name)
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 25),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title=f"QPSK, {NT} × {NR} i.i.d. Rayleigh",
)
ax.legend()
fig.tight_layout()

# %%
# Soft outputs feed a channel decoder: the max-log LLRs of the ML detector, and the
# per-stream LLRs after the unbiased MMSE filter.
rng = comrs.Rng(1)
h = comrs.mimo_channel(NR, NT, rng)
x = c.points[[0, 1, 2, 3]]
y = h @ x + rng.complex_noise(NR, 0.05)
print("ML   LLRs:", np.round(comrs.mimo_llr(h, y, c, 0.05, "ml"), 1))
print("MMSE LLRs:", np.round(comrs.mimo_llr(h, y, c, 0.05, "mmse"), 1))
