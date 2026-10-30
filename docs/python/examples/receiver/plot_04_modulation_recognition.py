"""
Blind receiver: modulation recognition, SNR and symbol rate
===========================================================

A receiver that does not know what it receives (spectrum monitoring, cognitive radio,
an adaptive link before its first header) must find the modulation, the SNR and the
symbol rate from the samples alone. comrs implements the classical methods:

- **cumulants** (`comrs.classify_modulation(..., "cumulants")`): the fourth-order
  cumulants :math:`C_{40}` and :math:`C_{42}`, normalised by the power :math:`C_{21}`,
  characterise the alphabet and ignore Gaussian noise (Swami & Sadler, *IEEE Trans.
  Commun.* 48(3), 2000);
- **likelihood** (`"ml_known"` with the noise known, Wei & Mendel, *IEEE Trans.
  Commun.* 48(2), 2000; `"ml"` with the scale and the noise estimated under every
  candidate): the optimal classifier;
- **M2M4** (`comrs.snr_m2m4_db`): the SNR from the second and fourth moments (Pauluzzi &
  Beaulieu, *IEEE Trans. Commun.* 48(10), 2000);
- **cyclostationarity** (`comrs.estimate_symbol_rate`): the line of the instantaneous
  power at the symbol rate (Gardner, *IEEE Signal Process. Mag.* 8(2), 1991).

The survey of Dobre et al. (*IET Commun.* 1(2), 2007) puts them in perspective.
``validation/tests/blind.rs`` checks them; the classification happens here at one
sample per symbol, with the timing and the carrier recovered.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs.plot import BACKGROUND, GREY, LABEL, ORANGE

rng = comrs.Rng(1)
NAMES = ["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"]
CANDS = [
    comrs.Constellation(k, m).normalized()
    for k, m in [("psk", 2), ("qam", 4), ("psk", 8), ("qam", 16), ("qam", 64)]
]


def received(c, n, snr_db, rotate=False):
    """`n` symbols of `c` in complex Gaussian noise at `snr_db`; with `rotate`, turned by
    a random carrier phase."""
    s2 = 1 / comrs.db2lin(snr_db)
    phase = np.exp(2j * np.pi * rng.uniform(1)[0]) if rotate else 1.0
    return c.points[rng.integers(c.order, n)] * phase + rng.complex_noise(n, s2), s2


# %%
# The cumulant features
# ---------------------
#
# Each alphabet has its point :math:`(|\tilde C_{40}|, \tilde C_{42})`: -2 for BPSK, 1 and
# -1 for QPSK, 0 and -1 for every higher PSK, -0.68 for 16-QAM, -0.62 for 64-QAM. The
# clouds are the estimates from 1000 symbols at 10 dB: normalised by the received power
# :math:`C_{21} = S + N`, they shrink towards the origin by :math:`(1 + 1/\mathrm{SNR})^{-2}`;
# with the noise variance known (filled markers), they are centred on the theory. Every
# trial has a random carrier phase: the features do not depend on it.

fig, ax = plt.subplots()
for c, name, marker in zip(CANDS, NAMES, ["o", "s", "^", "D", "v"], strict=True):
    t = comrs.theory_cumulants(c)
    (pt,) = ax.plot(abs(t["c40_normalized"]), t["c42_normalized"], marker, markersize=13, markerfacecolor="none",
                    markeredgewidth=2, label=name)  # fmt: skip
    raw, known = [], []
    for _ in range(40):
        x, s2 = received(c, 1000, 10.0, rotate=True)
        k = comrs.cumulants(x)
        raw.append((abs(k["c40_normalized"]), k["c42_normalized"]))
        s = k["c21"] - s2
        known.append((abs(k["c40"]) / s**2, k["c42"] / s**2))
    raw, known = np.array(raw), np.array(known)
    ax.plot(raw[:, 0], raw[:, 1], ".", color=pt.get_color(), alpha=0.35, markersize=4)
    ax.plot(known[:, 0], known[:, 1], marker, color=pt.get_color(), alpha=0.35, markersize=4)
ax.set(xlabel=r"$|\tilde C_{40}|$", ylabel=r"$\tilde C_{42}$",
       title="Fourth-order cumulants at 10 dB, 1000 symbols")  # fmt: skip
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()

# %%
# Probability of correct classification
# -------------------------------------
#
# The five alphabets equally likely, 512 symbols per decision, the carrier recovered
# (the likelihood needs it; the cumulants do not). The likelihood, even with the scale
# and the noise estimated (the scale by M2M4, the noise by the larger of M2M4 and the
# residual of the decisions, since M2M4 alone fails for QAM at high SNR), needs the
# lowest SNR; the cumulants need more, all the more
# with the noise unknown, since the features shrink: 16-QAM is then taken for 64-QAM, and
# BPSK for QPSK, until the SNR is high.

snrs = np.arange(-4.0, 21.0, 3.0)
trials = 60


def pcc(method, snr_db, n=512, known=False):
    """The rate of correct decisions over the five alphabets."""
    ok = 0
    for i, c in enumerate(CANDS):
        for _ in range(trials):
            x, s2 = received(c, n, snr_db)
            d = comrs.classify_modulation(x, CANDS, method, noise_var=s2 if known else None)
            ok += d["index"] == i
    return ok / (trials * len(CANDS))


curves = {
    "cumulants, noise unknown": [pcc("cumulants", s) for s in snrs],
    "cumulants, noise known": [pcc("cumulants", s, known=True) for s in snrs],
    "likelihood, scale and noise estimated": [pcc("ml", s) for s in snrs],
    "likelihood, noise known": [pcc("ml_known", s, known=True) for s in snrs],
}
fig, ax = plt.subplots()
for (name, values), marker in zip(curves.items(), ["o", "s", "D", "^"], strict=True):
    ax.plot(snrs, values, marker=marker, markeredgecolor=BACKGROUND, label=name)
ax.axhline(1 / len(CANDS), color=GREY, linestyle=":", linewidth=1.0, label="chance")
ax.set(xlabel="SNR (dB)", ylabel="probability of correct classification", ylim=(0, 1.02),
       title="BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM; 512 symbols")  # fmt: skip
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()

# %%
# The confusions at 8 dB: rows are the transmitted alphabets, columns the decisions.

fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
for ax, method in zip(axes, ["cumulants", "ml"], strict=True):
    m = np.zeros((5, 5))
    for i, c in enumerate(CANDS):
        for _ in range(trials):
            x, _ = received(c, 512, 8.0)
            m[i, comrs.classify_modulation(x, CANDS, method)["index"]] += 1
    m /= trials
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.6 else LABEL)  # fmt: skip
    ax.set(xticks=range(5), yticks=range(5), xticklabels=NAMES, yticklabels=NAMES,
           title="cumulants" if method == "cumulants" else "likelihood")  # fmt: skip
    ax.tick_params(axis="x", labelrotation=45)
fig.suptitle("Confusion matrices at 8 dB, 512 symbols", fontsize=11)
fig.tight_layout()

# %%
# More symbols reduce the variance of the estimates, not their bias: at 10 dB, the
# cumulants with the noise known and the likelihood improve with the length, while the
# cumulants with the noise unknown stay at the level their shrunken features allow.

lengths = [32, 64, 128, 256, 512, 1024, 2048]
fig, ax = plt.subplots()
for (name, method, known), marker in zip(
    [("cumulants, noise unknown", "cumulants", False), ("cumulants, noise known", "cumulants", True),
     ("likelihood, scale and noise estimated", "ml", False)],
    ["o", "s", "D"],
    strict=True,
):  # fmt: skip
    ax.semilogx(lengths, [pcc(method, 10.0, n, known) for n in lengths], marker=marker,
                markeredgecolor=BACKGROUND, label=name)  # fmt: skip
ax.set(xlabel="symbols per decision", ylabel="probability of correct classification", ylim=(0, 1.02),
       title="Classification against the observation length")  # fmt: skip
ax.legend()
fig.tight_layout()

# %%
# The SNR without the symbols: M2M4
# ---------------------------------
#
# With :math:`M_2 = S + N` and :math:`M_4 = k_a S^2 + 4SN + 2N^2` (:math:`k_a` the
# kurtosis of the alphabet, 1 for PSK), the signal power is
# :math:`\sqrt{(2M_2^2 - M_4)/(2 - k_a)}`. For 8-PSK the estimate follows the SNR; for
# 16-QAM it degrades at high SNR, where the noise is a small difference of moments that
# the symbols themselves make fluctuate (median and 10–90 % range over 200 trials of 2000
# symbols).

true_snr = np.arange(0.0, 31.0, 3.0)
fig, ax = plt.subplots()
ax.plot(true_snr, true_snr, color=GREY, linewidth=1.0, linestyle=":", label="ideal")
for c, name, marker in [(CANDS[2], "8-PSK", "o"), (CANDS[3], "16-QAM", "s")]:
    med, lo, hi = [], [], []
    for s in true_snr:
        est = [comrs.snr_m2m4_db(received(c, 2000, s)[0], c) for _ in range(200)]
        est = np.clip(est, -10, 60)
        med.append(np.median(est))
        lo.append(np.percentile(est, 10))
        hi.append(np.percentile(est, 90))
    (line,) = ax.plot(
        true_snr, med, marker=marker, markeredgecolor=BACKGROUND, label=f"{name}, M2M4"
    )
    ax.fill_between(true_snr, lo, hi, color=line.get_color(), alpha=0.15)
ax.set(
    xlabel="true SNR (dB)",
    ylabel="estimated SNR (dB)",
    ylim=(-2, 45),
    title="Blind SNR estimation, 2000 symbols",
)
ax.legend()
fig.tight_layout()

# %%
# The symbol rate by cyclostationarity
# ------------------------------------
#
# A root-raised-cosine signal of roll-off :math:`\beta` has a power that oscillates at the
# symbol rate: a spectral line at :math:`1/T` whose height grows with :math:`\beta`. Here
# 16-QAM at 4.37 samples per symbol (an unknown, non-integer ratio) and 10 dB.


def rrc(t, beta):
    """The root-raised-cosine pulse of roll-off `beta` at the times `t` (in symbols)."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    zero = np.isclose(t, 0.0)
    edge = np.isclose(np.abs(t), 1 / (4 * beta))
    rest = ~(zero | edge)
    tr = t[rest]
    out[rest] = (
        np.sin(np.pi * tr * (1 - beta)) + 4 * beta * tr * np.cos(np.pi * tr * (1 + beta))
    ) / (np.pi * tr * (1 - (4 * beta * tr) ** 2))
    out[zero] = 1 - beta + 4 * beta / np.pi
    a = np.pi / (4 * beta)
    out[edge] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(a) + (1 - 2 / np.pi) * np.cos(a))
    return out


sps, n_sym, beta = 4.37, 4000, 0.3
sym = CANDS[3].points[rng.integers(16, n_sym)]
t = np.arange(int(n_sym * sps)) / sps
x = np.zeros(t.size, dtype=complex)
for k, a in enumerate(sym):
    lo, hi = np.searchsorted(t, k - 8.0), np.searchsorted(t, k + 8.0)
    x[lo:hi] += a * rrc(t[lo:hi] - k, beta)
x += rng.complex_noise(x.size, 0.1)

p = np.abs(x) ** 2
# the spectrum of |x|² averaged over 16 segments, so that the line stands out of the
# fluctuations of the data
seg = p.size // 16
spec = np.mean(
    [np.abs(np.fft.rfft(q - q.mean())) ** 2 for q in p[: 16 * seg].reshape(16, seg)], axis=0
)
freqs = np.fft.rfftfreq(seg, d=1 / sps)
rs = comrs.estimate_symbol_rate(x, sps, 0.3, sps / 2)
fig, ax = plt.subplots()
ax.semilogy(freqs[1:], spec[1:], linewidth=0.8)
ax.axvline(rs, color=ORANGE, linestyle="--", linewidth=1.2, label=f"estimate: {rs:.5f} (true 1)")
ax.set(xlim=(0, sps / 2), xlabel="frequency (in units of the symbol rate)", ylabel="averaged spectrum of |x|²",
       title=f"16-QAM, RRC β = {beta}, {sps} samples per symbol")  # fmt: skip
ax.legend()
fig.tight_layout()
