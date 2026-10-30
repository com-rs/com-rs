"""
Proakis & Salehi, Sec. 3.3-2: MSK has the error rate of BPSK
============================================================

Minimum-shift keying is continuous-phase FSK with a modulation index of 1/2. Section
3.3-2 of J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., shows that
its signal is an offset QPSK whose quadrature pulses are half sinusoids, so that
coherent detection over AWGN decides the bits with the BER of BPSK,
:math:`Q(\\sqrt{2 E_b/N_0})`, once the bits are precoded so that each decides a sign
of the equivalent offset QPSK signal.

Without precoding, the information bits are the phase increments, and a symbol-by-
symbol decision on the phase states of the maximum-likelihood path decides each bit
from two consecutive phases: an error in one phase affects two bits, and the BER is
that of differentially decoded BPSK, :math:`2p(1 - p)` with :math:`p` the BER of
BPSK. The blocks ``cpm_tx`` and ``cpm_rx`` run the maximum-likelihood sequence
detector on the phase trellis in both cases; ``specs/validation/msk_bpsk.yaml``
checks the precoded case in CI within 4 % at 2, 4 and 6 dB.
"""

import matplotlib.pyplot as plt
import numpy as np
from _gallery_utils import BACKGROUND, MUTED

import comrs

EBN0 = np.arange(0.0, 9.0, 1.0)
results = comrs.run_spec(
    {
        "name": "msk_bpsk",
        "seed": 3,
        "cases": [{"precode": True}, {"precode": False}],
        "grid": {"ebn0_db": list(EBN0)},
        "link": [
            {"type": "source", "constellation": "BPSK", "n_symbols": 4000},
            {
                "type": "cpm_tx",
                "h": "1/2",
                "pulse": "rec",
                "length": 1,
                "sps": 4,
                "precode": "$precode",
            },
            {"type": "awgn_ebn0", "ebn0_db": "$ebn0_db"},
            {
                "type": "cpm_rx",
                "h": "1/2",
                "pulse": "rec",
                "length": 1,
                "sps": 4,
                "detector": "ml",
                "precode": "$precode",
            },
        ],
        "metrics": {"report": ["ber"], "scale": "none"},
        "stop": {"min_errors": {"ber": 400}, "min_trials": 2, "max_trials": 2000},
    }
)


def bpsk(ebn0_db):
    return np.vectorize(comrs.qfunc)(np.sqrt(2 * 10 ** (np.asarray(ebn0_db) / 10)))


fig, ax = plt.subplots()
fine = np.linspace(0, 8, 81)
ax.semilogy(fine, bpsk(fine), color=MUTED, linewidth=1.4, label="BPSK, $Q(\\sqrt{2E_b/N_0})$")
ax.semilogy(fine, 2 * bpsk(fine) * (1 - bpsk(fine)), ":", color=MUTED, linewidth=1.4,
            label="differentially decoded BPSK, $2p(1-p)$")  # fmt: skip
for precode, marker, label in [(True, "o", "MSK, precoded"), (False, "s", "MSK, not precoded")]:
    rows = [r for r in results if r["condition"]["precode"] is precode]
    ber = np.array([r["metrics"]["ber"]["value"] for r in rows])
    lo = np.array([r["metrics"]["ber"]["lo"] for r in rows])
    hi = np.array([r["metrics"]["ber"]["hi"] for r in rows])
    ax.errorbar(EBN0, ber, yerr=[ber - lo, hi - ber], fmt=marker, markeredgecolor=BACKGROUND,
                capsize=0, label=label)  # fmt: skip
ax.set(
    xlim=(0, 8),
    ylim=(1e-4, 0.5),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="MSK with maximum-likelihood sequence detection over AWGN",
)
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# The precoded points lie on the BPSK curve within their intervals.
for r in results:
    if r["condition"]["precode"]:
        e = r["condition"]["ebn0_db"]
        print(f"Eb/N0 = {e:.0f} dB: MSK {r['metrics']['ber']['value']:.3e}, BPSK {bpsk(e):.3e}")
