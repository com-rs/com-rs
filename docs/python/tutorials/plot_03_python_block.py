"""
Writing a block in Python
=========================

A block of a chain is any callable ``f(frame, rng)`` that reads and writes the
samples of a :class:`comrs.Frame`, or any subclass of :class:`comrs.Block`. Python
blocks prototype an impairment or an algorithm with NumPy, run inside the same chains
as the native blocks, and are checked against them trial for trial. This tutorial
writes a transmitter IQ imbalance and its blind compensation, the Gram–Schmidt
orthogonalisation procedure (GSOP) of Fatadin, Savory & Ives, *IEEE Photon. Technol.
Lett.* 20(20), 2008, and compares both with the native blocks.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs import blocks as B

# %%
# **The frame.** A Python block sees the current samples (``frame.samples``, a complex
# array, or ``frame.streams`` with several antennas or polarisations) and the samples
# per symbol (``frame.sps``), and may read what the source recorded: the transmitted
# bits, symbols and constellation. ``frame.record(name, value)`` reports a value that
# the runners estimate as a mean. The simplest block is a function:


def invert(frame, rng):
    """A sign flip: a deterministic block that draws nothing."""
    frame.samples = -frame.samples


# %%
# **An impairment as a block.** A gain and phase mismatch between the I and Q paths of
# a transmitter turns :math:`x` into :math:`y = x + \beta x^*` (the image of the
# signal leaks into it, at :math:`-20\log_{10}|\beta|` dB). A class carries the
# parameter; the frame keeps everything else.


class TxIqImbalance(comrs.Block):
    """y = x + β x*."""

    def __init__(self, beta):
        self.beta = complex(beta)

    def __call__(self, frame, rng):
        x = frame.samples
        frame.samples = x + self.beta * np.conj(x)


# %%
# **Checked against the native block.** ``TxIq`` is the same model in Rust. The two
# chains below draw the same symbols (same seed), so the samples after the Python
# block and after the native block must agree to machine precision.
BETA = 0.1 + 0.15j
source = B.Source(constellation="16-QAM", n_symbols=4096)
python_chain = comrs.Chain(tx=source, iq=TxIqImbalance(BETA))
native_chain = comrs.Chain(tx=source, iq=B.TxIq(beta=[BETA.real, BETA.imag]))
a = python_chain({}, comrs.Rng(3)).samples
b = native_chain({}, comrs.Rng(3)).samples
print("Python and native IQ imbalance agree:", np.allclose(a, b))

# %%
# **An algorithm as a block.** GSOP makes the I and Q components orthogonal and of
# equal power: :math:`I' = I/\sqrt{P_I}`, :math:`Q'' = Q - \rho I / P_I` with
# :math:`\rho = \mathbb{E}[IQ]`, then :math:`Q' = Q''/\sqrt{P_{Q''}}`. It needs no
# training symbols, only the circularity of the transmitted constellation. The block
# records the normalised correlation it measured, which the campaign reports.


class Gsop(comrs.Block):
    """Gram–Schmidt orthogonalisation of I and Q (Fatadin, Savory & Ives 2008)."""

    def __call__(self, frame, rng):
        x = frame.samples
        i, q = x.real, x.imag
        p_i = np.mean(i**2)
        rho = np.mean(i * q)
        q = q - rho / p_i * i
        i = i / np.sqrt(p_i)
        q = q / np.sqrt(np.mean(q**2))
        frame.record("iq_correlation", rho / p_i)
        frame.samples = (i + 1j * q) / np.sqrt(2)


# %%
# The constellations after the impairment and after the compensation, at 25 dB.
link = comrs.Chain(
    tx=source,
    iq=TxIqImbalance(BETA),
    channel=B.Awgn(snr_db=comrs.var("snr_db")),
    gsop=Gsop(),
)
frames = link.trace({"snr_db": 25.0}, comrs.Rng(1))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, (name, title) in zip(
    axes,
    {"tx": "transmitted", "channel": "with IQ imbalance", "gsop": "after GSOP"}.items(),
    strict=True,
):
    comrs.plot.constellation(frames[name], ax=ax, max_points=4000)
    ax.set_title(title)
fig.tight_layout()

# %%
# **The campaign.** A chain with a Python block runs on the Python runner
# (:func:`comrs.sim.run`, sequential), with the same grid, metrics, stopping rule and
# random streams as the Rust runner. :func:`comrs.compare` gathers the variants; the
# recorded value appears as a metric.
campaign = {
    "grid": {"snr_db": [10.0, 14.0, 18.0, 22.0, 26.0]},
    "metrics": ["ber"],
    "stop": {"min_errors": {"ber": 200}, "max_trials": 100},
    "seed": 1,
}
variants = {
    "no imbalance": comrs.Chain(tx=source, channel=B.Awgn(snr_db=comrs.var("snr_db"))),
    "imbalance": link.replace(gsop=invert).replace(gsop=lambda f, r: None),
    "imbalance + GSOP": link,
}
res = comrs.compare(variants, **campaign)
rho = res.where(variant="imbalance + GSOP", snr_db=26.0)[0]["metrics"]["iq_correlation"]
expected = 2 * BETA.imag / ((1 + BETA.real) ** 2 + BETA.imag**2)
print(f"measured I/Q correlation at 26 dB: {rho['value']:.3f}, expected {expected:.3f}")

fig, ax = plt.subplots()
res.plot("snr_db", "ber", by="variant", ax=ax,
         theory=lambda s, _: comrs.theory_ber(16, s - comrs.lin2db(4), "qam"))  # fmt: skip
ax.set(xlabel="SNR per symbol (dB)", ylabel="bit error rate", ylim=(1e-5, 0.5),
       title="16-QAM, β = 0.1 + 0.15j; dotted: closed form")  # fmt: skip
fig.tight_layout()

# %%
# **Parity with the native runner.** The two runners draw trial ``t`` of condition
# ``i`` from the same stream, so a Python block that leaves the frame unchanged gives
# the native campaign back, error for error: the way to check a Python prototype
# against the native block that replaces it.
native = comrs.Chain(tx=source, channel=B.Awgn(snr_db=comrs.var("snr_db")))
python = native.insert_after("tx", noop=lambda frame, rng: None)
counts = {
    "Rust runner": [r["metrics"]["ber"]["count"] for r in native.run(**campaign)],
    "Python runner": [r["metrics"]["ber"]["count"] for r in python.run(**campaign)],
}
print(counts)
assert counts["Rust runner"] == counts["Python runner"]

# %%
# **Beyond the prototype.** The Python runner is sequential: once a block is settled,
# port it to Rust (a ``Block`` of ``com-spec``, or an extension registered with
# ``com_spec::blocks::register``, see ``docs/ARCHITECTURE.md``), keep the Python
# version as its test oracle, and the chain runs on every core.
