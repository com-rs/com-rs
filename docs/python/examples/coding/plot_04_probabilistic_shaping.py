"""
Probabilistic shaping: the rate gain and probabilistic amplitude shaping
=========================================================================

**Rate.** A uniform QAM wastes energy on its outer points: at a given SNR its rate stays
below the capacity :math:`\\log_2(1 + \\mathrm{SNR})` of the AWGN channel, by up to the
1.53 dB of the ultimate shaping gain. Maxwell-Boltzmann probabilities
:math:`p(x) \\propto e^{-\\lambda |x|^2}` (Kschischang & Pasupathy 1993) close most of
that gap: the curve is the GMI of bit-metric decoding of 64-QAM with the priors,
:math:`H(X) - \\sum_k H(B_k \\mid Y)` (Böcherer 2014), with the best :math:`\\lambda` at
every SNR, and the points are the estimates of simulated links (``specs/shaping_awgn.yaml``
checks them against the closed forms).

**Probabilistic amplitude shaping** (Böcherer, Steiner & Schulte 2015) realises it with
a binary code: per dimension, a constant-composition distribution matcher (Schulte &
Böcherer 2016) shapes the amplitudes, a systematic LDPC code of rate 2/3 protects their
bits, and its parity bits, uniform, become the signs. At 3 bits per QAM symbol, the
shaped 64-QAM decodes at a lower SNR than the uniform 64-QAM with a code of rate 1/2
and than the uniform 16-QAM with a code of rate 3/4; the gain over 16-QAM, smaller than
the gain of rate, is limited by the regular LDPC codes of this example, which are not
optimised for the shaped bit channels.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs import blocks as B

# %%
# The rates of uniform and shaped 64-QAM against the capacity.
snr = np.arange(0.0, 24.1, 2.0)
c = comrs.Constellation("qam", 64).normalized()
lambdas = np.linspace(0.0, 6.0, 25)


def mb(lam):
    return np.asarray(comrs.maxwell_boltzmann(c, lam))


uniform = [comrs.theory_gmi(64, s, "qam", points=20) for s in snr]
best = []
for s in snr:
    rates = [comrs.theory_gmi(64, s, "qam", px=mb(lam), points=20) for lam in lambdas]
    best.append((max(rates), lambdas[int(np.argmax(rates))]))
fig, ax = plt.subplots()
ax.plot(snr, np.log2(1 + 10 ** (snr / 10)), color="#8a8984", linewidth=1.4, label="AWGN capacity")
ax.plot(snr, [b[0] for b in best], color="#2a6fdb", label="shaped 64-QAM, best λ")
ax.plot(snr, uniform, color="#d95f02", label="uniform 64-QAM")
# simulated links at the best λ of every SNR
link = comrs.Chain(
    tx=B.Source(constellation="64-QAM", n_symbols=16384, mb_lambda=comrs.var("lam")),
    ch=B.Awgn(snr_db=comrs.var("snr")),
)
for s, (_, lam) in zip(snr[::2], best[::2], strict=True):
    r = link.run(grid={"snr": [s], "lam": [lam]}, metrics=["gmi"], stop={"max_trials": 8})
    ax.plot(s, r[0]["metrics"]["gmi"]["value"], "o", color="#2a6fdb", markeredgecolor="#fcfcfb")
ax.set(xlabel="SNR (dB)", ylabel="GMI (bits per QAM symbol)", xlim=(0, 24), ylim=(0, 6.5),
       title="Bit-metric decoding of 64-QAM over AWGN")  # fmt: skip
ax.legend(loc="upper left")
fig.tight_layout()

# %%
# Coded links at 3 information bits per QAM symbol.
pas = {"constellation": "64-QAM", "mb_lambda": 2.55, "n": 1536}


def bicm(constellation, k):
    code = {"type": "ldpc", "n": 1536, "k": k}
    return comrs.Chain(
        B.FecEncode(code=code, blocks=4, interleave=True),
        B.Map(constellation=constellation),
        B.Awgn(snr_db=comrs.var("snr")),
        B.Demap(),
        B.FecDecode(code=code, interleave=True),
    )


links = {
    "PAS 64-QAM, LDPC rate 2/3": comrs.Chain(
        B.PasEncode(**pas, blocks=4), B.Awgn(snr_db=comrs.var("snr")), B.PasDecode(**pas)
    ),
    "uniform 16-QAM, LDPC rate 3/4": bicm("16-QAM", 1152),
    "uniform 64-QAM, LDPC rate 1/2": bicm("64-QAM", 768),
}
grid = list(np.arange(9.0, 13.1, 0.5))
fig, ax = plt.subplots()
for (name, chain), marker in zip(links.items(), "osD", strict=True):
    res = chain.run(grid={"snr": grid}, metrics=["bler"],
                    stop={"min_errors": {"bler": 50}, "max_trials": 400})  # fmt: skip
    x, y = res.series("snr", "bler", min_count=3)
    keep = y > 0
    ax.semilogy(x[keep], y[keep], marker=marker, markeredgecolor="#fcfcfb", label=name)
ax.axvline(10 * np.log10(2**3 - 1), color="#8a8984", linestyle=":", linewidth=1.0)
ax.text(10 * np.log10(7) + 0.05, 0.12, "capacity, 3 bits", rotation=90, fontsize=8, color="#52514e")
ax.set(xlabel="SNR (dB)", ylabel="block error rate", ylim=(1e-3, 1),
       title="3 bits per QAM symbol over AWGN, codewords of 1536 bits")  # fmt: skip
ax.legend(loc="upper right")
fig.tight_layout()
