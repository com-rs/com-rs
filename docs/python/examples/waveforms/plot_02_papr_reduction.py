"""
PAPR reduction techniques
=========================

The classical techniques (Han & Lee, *IEEE Wireless Commun.* 12(2), 2005), applied to
16-QAM OFDM with 128 subcarriers (oversampling by 4):

**Without distortion** — the data are sent unchanged, at the price of side
information or rate:

- *selected mapping* (SLM): the best of :math:`U` phase-rotated copies (Bäuml, Fischer
  & Huber, *Electron. Lett.* 32(22), 1996); with independent candidates the CCDF is that
  of OFDM raised to the power :math:`U` (dashed);
- *partial transmit sequences* (PTS): the best phase combination of :math:`V`
  subblocks (Müller & Huber, *Electron. Lett.* 33(5), 1997);
- *tone reservation* (TR): a peak-cancelling signal on unused subcarriers (Tellado,
  *Multicarrier Modulation with Low PAR*, 2000).

**With distortion** — the signal itself is modified:

- *clipping and filtering* (ICF): repeated clipping and removal of the out-of-band
  components (Armstrong, *Electron. Lett.* 38(5), 2002);
- *active constellation extension* (ACE): outer points pushed outwards, where they do
  not harm the decisions (Krongold & Jones, *IEEE Trans. Broadcast.* 49(3), 2003);
- *companding*: μ-law (Wang, Tjhung & Ng, *IEEE Trans. Broadcast.* 45(3), 1999) or
  exponential (Jiang, Yang & Song, *IEEE Trans. Broadcast.* 51(2), 2005), expanded at
  the receiver.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs.plot import BACKGROUND, GREY

N, BLOCKS = 128, 1500
ofdm = comrs.Ofdm(N, N)
c = comrs.Constellation("qam", 16).normalized()
_, d = comrs.symbols(c, N * BLOCKS, comrs.Rng(1))
t = np.arange(2.0, 12.1, 0.1)
original = comrs.ccdf(ofdm.papr_db(ofdm.modulate(d)), t)


def ccdf_of(reduction):
    k = ofdm.data_len(reduction)
    x, _ = ofdm.modulate(d[: (d.size // k) * k], reduction)
    return comrs.ccdf(ofdm.papr_db(x), t)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
for ax in (ax1, ax2):
    ax.semilogy(t, original, color=GREY, linewidth=1.6, label="OFDM")
lossless = {
    "SLM, U = 16": comrs.PaprReduction.slm(16),
    "PTS, V = 4, W = 4": comrs.PaprReduction.pts(4, phases=4),
    "TR, 12 tones": comrs.PaprReduction.tone_reservation(N, 12, 4.0),
}
for name, r in lossless.items():
    (line,) = ax1.semilogy(t, ccdf_of(r), label=name)
    if name.startswith("SLM"):
        ax1.semilogy(
            t,
            comrs.theory_papr_ccdf(N, t, candidates=16),
            color=line.get_color(),
            linestyle="--",
            linewidth=1.2,
            label="SLM, theory",
        )
ax1.set(
    xlabel=r"$\gamma$ (dB)",
    ylabel=r"$\Pr(\mathrm{PAPR} > \gamma)$",
    ylim=(1e-3, 1.1),
    xlim=(2, 12),
    title="without distortion",
)
ax1.legend(loc="lower left")
distorting = {
    "ICF, CR = 4 dB, 4 it.": comrs.PaprReduction.clip_filter(4.0, iterations=4),
    "ACE, 10 it.": comrs.PaprReduction.ace(c, 4.0, iterations=10),
    "exponential companding": comrs.PaprReduction.exponential(2.0),
}
for name, r in distorting.items():
    ax2.semilogy(t, ccdf_of(r), label=name)
ax2.set(xlabel=r"$\gamma$ (dB)", xlim=(2, 12), title="with distortion")
ax2.legend(loc="lower left")
fig.tight_layout()

# %%
# The price of the distortion: the bit error rate over AWGN at a fixed average
# transmitted power (the power added by ACE is paid for). ACE keeps the performance of
# OFDM — its points only move away from the decision boundaries — while clipping and
# filtering leaves an in-band distortion floor. μ-law companding is expanded at the
# receiver. The exponential law is not: it squeezes every amplitude above about twice
# the RMS into a narrow range, where the expansion would amplify the noise 20 to 150
# times; treated as a distortion (Bussgang gain), it leaves an error floor instead.


def ber_curve(reduction, ebn0s, blocks=400, seed=2, expand=True):
    k = ofdm.data_len(reduction)
    bits = comrs.bits(4 * k * blocks, comrs.Rng(seed))
    s = c.modulate(bits)
    x, side = ofdm.modulate(s, reduction)
    p = np.mean(np.abs(x) ** 2)
    out = []
    for i, eb in enumerate(ebn0s):
        # Eb/N0 = (block energy / bits) / σ²: the block energy p·L·N carries 4 k bits
        s2 = p * ofdm.fft_len / (4 * k * comrs.db2lin(eb))
        y = x + comrs.Rng.stream(seed, "awgn", i, 0).complex_noise(x.size, s2)
        if expand:
            z = ofdm.demodulate(y, reduction, side=side)
        else:  # no expansion: one-tap equalisation by the Bussgang gain
            z = ofdm.demodulate(y)
            z = z * np.vdot(s, s) / np.vdot(s, z)
        out.append(comrs.ber(c.demodulate(z), bits))
    return np.array(out)


ebn0 = np.arange(0.0, 17.0, 2.0)
fig, ax = plt.subplots()
eb = np.linspace(0, 16, 161)
ax.semilogy(
    eb,
    comrs.theory_ber(16, eb, "qam"),
    color=GREY,
    linewidth=1.4,
    label="16-QAM, AWGN",
)
cases = {
    "ICF, CR = 4 dB": comrs.PaprReduction.clip_filter(4.0, iterations=4),
    "ACE": comrs.PaprReduction.ace(c, 4.0, iterations=10),
    "μ-law, μ = 4": comrs.PaprReduction.mu_law(4.0),
    "exponential, not expanded": comrs.PaprReduction.exponential(2.0),
}
for (name, r), marker in zip(cases.items(), ["o", "s", "D", "^"], strict=True):
    b = ber_curve(r, ebn0, expand="not expanded" not in name)
    keep = b > 0
    ax.semilogy(ebn0[keep], b[keep], marker=marker, markeredgecolor=BACKGROUND, label=name)
ax.set(
    ylim=(1e-5, 0.5),
    xlim=(0, 16),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="16-QAM OFDM over AWGN: the cost of the distortion",
)
ax.legend()
fig.tight_layout()
