"""
DVB-S2: MODCODs, constellations and frame error rates
=====================================================

:class:`comrs.Dvbs2` implements the physical layer of ETSI EN 302 307-1: the BBFRAME,
the BCH and LDPC codes of the standard (the 21 parity-check tables of its annexes),
the bit interleaver, the QPSK, 8PSK, 16APSK and 32APSK constellations, and the PLFRAME
with its π/2-BPSK header, its pilots and its Gold scrambling.

**The 28 MODCODs.** Table 13 of the standard gives the ideal Es/N0 of every MODCOD at a
quasi-error-free packet error rate of 1e-7 (normal frames of 64 800 bits). The MODCODs
cover 0.49 to 4.45 bit/s/Hz within 1 to 2.5 dB of the AWGN capacity
:math:`\\log_2(1 + E_s/N_0)`. ``specs/validation/dvbs2_table13.yaml`` checks the
simulated waterfall of every one of them against this table.

**The APSK constellations.** 16APSK has rings of 4 and 12 points and 32APSK rings of
4, 12 and 16 points, with the ring ratios of Tables 9 and 10 for each code rate. Their
labels (Figures 11 and 12) are close to Gray between neighbours of a ring.

**Frame error rates.** Short frames of 16 200 bits for four MODCODs, against the
thresholds of Table 13 (normal frames, quasi error free). The waterfalls are steep:
the frame error rate falls from 50 % to 1 % within 0.3 dB. At a frame error rate of
10 %, 8PSK 3/5, 16APSK 3/4 and 32APSK 4/5 lie 0.1 to 0.2 dB below the thresholds of the
normal frames, 0.1 dB above the normal frames themselves
(``specs/validation/dvbs2_table13.yaml``). QPSK 1/2 lies 0.8 dB below: its short
code has an effective rate of 4/9 (Table 5b).
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs import blocks as B
from comrs.plot import BACKGROUND, BLUE, GREY, INK, PURPLE, RULE, RUST, TEAL

# %%
# The spectral efficiency of the 28 MODCODs at the thresholds of Table 13.
families = [
    (
        "QPSK",
        ["1/4", "1/3", "2/5", "1/2", "3/5", "2/3", "3/4", "4/5", "5/6", "8/9", "9/10"],
        "o",
        BLUE,
    ),
    ("8PSK", ["3/5", "2/3", "3/4", "5/6", "8/9", "9/10"], "s", RUST),
    ("16APSK", ["2/3", "3/4", "4/5", "5/6", "8/9", "9/10"], "D", TEAL),
    ("32APSK", ["3/4", "4/5", "5/6", "8/9", "9/10"], "^", PURPLE),
]
fig, ax = plt.subplots()
esn0 = np.linspace(-4, 18, 200)
ax.plot(esn0, np.log2(1 + comrs.db2lin(esn0)), color=GREY, linewidth=1.2, label="AWGN capacity")
for name, rates, marker, color in families:
    pts = []
    for r in rates:
        link = comrs.Dvbs2(name, r, frame="normal", pilots=False)
        # spectral efficiency: data bits of a BBFRAME per symbol of the PLFRAME (no pilots)
        pts.append((link.threshold_db, (link.k - 80) / link.plframe_len))
    x, y = zip(*pts, strict=True)
    ax.plot(
        x, y, marker=marker, linestyle="none", color=color, markeredgecolor=BACKGROUND, label=name
    )
ax.set(xlabel="Es/N0 (dB)", ylabel="spectral efficiency (bit/symbol)", xlim=(-4, 18), ylim=(0, 5),
       title="DVB-S2 MODCODs at PER 1e-7 (Table 13)")  # fmt: skip
ax.legend(loc="upper left")
fig.tight_layout()

# %%
# The 16APSK (rate 3/4) and 32APSK (rate 4/5) constellations with their labels.
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (name, rate) in zip(axes, [("16APSK", "3/4"), ("32APSK", "4/5")], strict=True):
    c = comrs.Dvbs2(name, rate, frame="short").constellation
    p = np.asarray(c.points)
    b = c.bits_per_symbol
    ax.plot(p.real, p.imag, "o", color=BLUE, markeredgecolor=BACKGROUND)
    for label, z in enumerate(p):
        ax.annotate(format(label, f"0{b}b"), (z.real, z.imag), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color=INK)  # fmt: skip
    ax.set(aspect="equal", title=f"{name} {rate}", xlim=(-1.45, 1.45), ylim=(-1.45, 1.45))
    ax.axhline(0, color=RULE, linewidth=0.6)
    ax.axvline(0, color=RULE, linewidth=0.6)
fig.tight_layout()

# %%
# Frame error rate of short frames over AWGN, with the thresholds of Table 13.
cases = [("QPSK", "1/2", "o", BLUE), ("8PSK", "3/5", "s", RUST),
         ("16APSK", "3/4", "D", TEAL), ("32APSK", "4/5", "^", PURPLE)]  # fmt: skip
fig, ax = plt.subplots()
for name, rate, marker, color in cases:
    thr = comrs.Dvbs2(name, rate).threshold_db
    link = comrs.Chain(
        B.Dvbs2Tx(modulation=name, rate=rate, frame="short"),
        B.Awgn(snr_db=comrs.var("esn0_db"), reference="unit"),
        B.Dvbs2Rx(modulation=name, rate=rate, frame="short"),
    )
    grid = list(np.round(np.arange(thr - 1.3, thr + 0.21, 0.1), 2))
    res = link.run(grid={"esn0_db": grid}, metrics={"report": ["bler"], "scale": "none"},
                   stop={"min_errors": {"bler": 30}, "max_trials": 400})  # fmt: skip
    x, y = res.series("esn0_db", "bler", min_count=1)
    keep = (y > 0) & (y < 1)
    ax.semilogy(
        x[keep],
        y[keep],
        marker=marker,
        color=color,
        markeredgecolor=BACKGROUND,
        label=f"{name} {rate}",
    )
    ax.axvline(thr, color=color, linestyle=":", linewidth=1.0)
ax.set(xlabel="Es/N0 (dB)", ylabel="frame error rate", ylim=(1e-3, 1),
       title="DVB-S2 short frames over AWGN; dotted: Table 13 (normal frames)")  # fmt: skip
ax.legend(loc="lower left")
fig.tight_layout()
