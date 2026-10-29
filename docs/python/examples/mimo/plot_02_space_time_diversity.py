"""
Transmit diversity: Alamouti and space-time block codes
=======================================================

Diversity is the slope of the error rate at high SNR: with :math:`L` independent
fading branches, :math:`P_b \\propto \\bar\\gamma^{-L}`. Receive diversity is obtained
by maximal-ratio combining (MRC); Alamouti's code obtains the same order at the
transmitter, without channel knowledge there, at the price of splitting the power over
the two antennas (3 dB). BPSK, i.i.d. Rayleigh fading constant over a code block; the
lines are the closed form of MRC with :math:`L` branches of mean SNR :math:`\\gamma`
(Proakis & Salehi, eq. 13.4-15):

.. math::

   P_b = \\left(\\frac{1 - \\mu}{2}\\right)^{L} \\sum_{k=0}^{L-1}
   \\binom{L - 1 + k}{k} \\left(\\frac{1 + \\mu}{2}\\right)^{k},
   \\qquad \\mu = \\sqrt{\\frac{\\gamma}{1 + \\gamma}}.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

import comrs


def mrc(gamma, branches):
    mu = np.sqrt(gamma / (1 + gamma))
    return ((1 - mu) / 2) ** branches * sum(
        math.comb(branches - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(branches)
    )


code = comrs.Ostbc.alamouti()
points = np.arange(0.0, 31.0, 3.0)


def simulate(n_tx, n_rx, eb_db, seed):
    """BER of BPSK: Alamouti for two transmit antennas, MRC otherwise."""
    g = comrs.db2lin(eb_db)
    errors = total = t = 0
    blocks = 20_000
    while errors < 200 and total < 4e6:
        rng = comrs.Rng.stream(seed, "diversity", int(eb_db), t)
        bits = comrs.bits(2 * blocks, rng)
        s = 1.0 - 2.0 * bits
        h = comrs.mimo_channel(n_rx, n_tx, rng, n=blocks)
        if n_tx == 2:
            x = code.encode(s)  # 2 antennas x 2 uses per block
            y = np.einsum("nij,njt->nit", h, x.reshape(2, blocks, 2).transpose(1, 0, 2))
            y = y + rng.complex_noise(y.size, 1 / g).reshape(y.shape)
            est, _ = code.decode(h, y.transpose(1, 0, 2).reshape(n_rx, 2 * blocks), 1 / g)
        else:  # one antenna: two symbols per block over the same channel, MRC
            x = s.reshape(blocks, 2)
            y = h[:, :, :1] * x[:, None, :] + rng.complex_noise(blocks * n_rx * 2, 1 / g).reshape(
                blocks, n_rx, 2
            )
            est = (np.conj(h[:, :, :1]) * y).sum(axis=1).reshape(-1)
        errors += int(((est.real < 0) != (bits == 1)).sum())
        total += bits.size
        t += 1
    return errors / total


configs = {
    "1 × 1": (1, 1, 1, 1.0),
    "1 × 2, MRC": (1, 2, 2, 1.0),
    "2 × 1, Alamouti": (2, 1, 2, 0.5),
    "2 × 2, Alamouti": (2, 2, 4, 0.5),
}
fig, ax = plt.subplots()
eb = np.linspace(0, 30, 301)
for (name, (nt, nr, branches, share)), marker in zip(
    configs.items(), ["o", "s", "D", "^"], strict=True
):
    (line,) = ax.semilogy(eb, mrc(share * 10 ** (eb / 10), branches), linewidth=1.4)
    sim = np.array([simulate(nt, nr, e, seed=nt * 10 + nr) for e in points])
    keep = sim > 0
    ax.semilogy(
        points[keep],
        sim[keep],
        marker,
        color=line.get_color(),
        markeredgecolor="#fcfcfb",
        label=name,
    )
ax.plot([], [], color="#52514e", linewidth=1.4, label="theory")
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 30),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="BPSK over Rayleigh fading: diversity",
)
ax.legend()
fig.tight_layout()

# %%
# The rate-3/4 codes extend transmit diversity to three and four antennas.
for code34 in [comrs.Ostbc.rate_3_4(3), comrs.Ostbc.rate_3_4(4)]:
    print(code34, f"{code34.n_symbols} symbols over {code34.n_slots} channel uses")
