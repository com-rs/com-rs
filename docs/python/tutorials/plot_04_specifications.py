"""
Specifications and the command line
===================================

A campaign is a document: the grid, the link as a list of blocks, the metrics, the
stopping rule and the expected results, in a YAML file. The file runs from Python
(:func:`comrs.run_spec`) and from the command line (``comrs run``), on the parallel
Rust runner, and its report records the version, the seed and the specification, so
that every number can be traced back to what produced it. This tutorial writes the
chain of the first tutorial as a file, states what its results must be, searches the
operating point of a metric, and shows the command line.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import comrs
from comrs import blocks as B

# %%
# **From a chain to a file.** ``to_spec`` returns the specification of a native chain
# as a ``dict``: the block names become labels, the variables ``$name``.
link = comrs.Chain(
    tx=B.Source(constellation={"kind": "qam", "m": comrs.var("m")}, n_symbols=4096),
    channel=B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")),
)
spec = link.to_spec(
    name="qam_awgn",
    grid={"m": [4, 16, 64], "ebn0_db": {"from": 0, "to": 20, "step": 2}},
    metrics=["ber", "ser"],
    stop={"min_errors": {"ber": 200}, "min_trials": 8, "max_trials": 4000},
    seed=2024,
)
spec["callbacks"] = {"skip_below": {"metric": "ber", "threshold": 1e-6, "along": "ebn0_db"}}
text = yaml.safe_dump(spec, sort_keys=False)
print(text)

# %%
# **Running it.** :func:`comrs.run_spec` takes a path, a YAML text or a ``dict``, and
# returns one ``dict`` per condition (``condition``, ``status``, ``trials``, and every
# metric with its value and interval). The callback stops a sweep of ``ebn0_db`` once
# the BER falls below :math:`10^{-6}`: those conditions are reported as skipped.
folder = Path(tempfile.mkdtemp())
path = folder / "qam_awgn.yaml"
path.write_text(text, encoding="utf-8")
results = comrs.Results(comrs.run_spec(path), spec)
print({r["status"] for r in results}, len(results), "conditions")

fig, ax = plt.subplots()
results.plot("ebn0_db", "ber", by="m", ax=ax,
             theory=lambda e, m: comrs.theory_ber(m, e, "qam"))  # fmt: skip
ax.set(xlabel="$E_b/N_0$ (dB)", ylabel="bit error rate", ylim=(1e-6, 0.5),
       title="Gray-mapped QAM over AWGN; dotted: closed form")  # fmt: skip
fig.tight_layout()

# %%
# **Expected results.** An ``expect`` entry names a source and what the campaign must
# reproduce: a closed form (``theory``), a published curve (``curve``) or a value. The
# check passes when the expectation lies in the confidence interval of the metric (here
# the 99.9 % interval, plus a relative tolerance). :func:`comrs.verify_spec` runs the
# campaign and reports every check; the specifications of ``specs/validation/`` do
# this in CI for the error rates of Proakis & Salehi, the information rates and figures
# of the literature.
spec["expect"] = [
    {
        "name": "SER of square QAM (exact)",
        "source": "Proakis & Salehi, Digital Communications, 5th ed., Sec. 4.3-2",
        "metric": "ser",
        "theory": {"function": "ser", "kind": "qam", "m": "$m", "ebn0_db": "$ebn0_db"},
        "interval": 0.999,
        "rtol": 0.02,
    }
]
report = comrs.verify_spec(spec)
print("all checks passed:", report["passed"])
for c in report["checks"][:6]:
    print(f"  {c['condition']}: {c['metric']} = {c['value']:.3e}, expected {c['expected']:.3e}")

# %%
# **Operating points.** A ``search`` section replaces the sweep of a variable by the
# search of the value at which a metric reaches a target, for every combination of
# the other axes: here the :math:`E_b/N_0` required for a BER of :math:`10^{-3}`. The
# bracket is halved on the same random streams (common random numbers) until it is
# narrower than the tolerance. The result is compared with the inverted closed form.
search = dict(spec)
search.pop("expect")
search.pop("callbacks")
search["grid"] = {"m": [4, 16, 64]}
search["search"] = {"along": "ebn0_db", "range": [0, 20], "metric": "ber", "target": 1e-3,
                    "tolerance": 0.05}  # fmt: skip
for r in comrs.search_spec(search):
    m = r["condition"]["m"]
    lo, hi = 0.0, 20.0
    for _ in range(50):  # the closed form, inverted by bisection
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if comrs.theory_ber(m, mid, "qam") > 1e-3 else (lo, mid)
    print(
        f"{m:>2}-QAM: BER 1e-3 at Eb/N0 = {r['value']:.2f} dB "
        f"[{r['lo']:.2f}, {r['hi']:.2f}], closed form {lo:.2f} dB, {len(r['evaluations'])} evaluations"
    )

# %%
# **The command line.** The binary ``comrs`` (``cargo install com-spec``, or
# ``cargo build --release``) runs the same files without Python, writes the JSON
# and CSV reports to ``results/``, and documents the blocks:
#
# .. code-block:: bash
#
#    comrs check qam_awgn.yaml            # the link, the grid and the ports
#    comrs conditions qam_awgn.yaml       # the conditions of the grid
#    comrs run qam_awgn.yaml --out results --threads 8
#    comrs verify qam_awgn.yaml           # the expected results; exit code 1 on a failure
#    comrs blocks awgn_ebn0 source        # the parameters of a block, with units and defaults
#    comrs schema                         # the JSON Schema of the blocks, for editors
#
# ``comrs run --fast-dev-run`` runs one trial per condition to check a file before a
# long campaign. The reports are read back with :meth:`comrs.Results.load`.

# %%
# **Reading a file back.** ``Chain.from_spec`` turns a file into a chain again, with
# its grid, metrics and stopping rule as ``settings``: a published campaign is rerun
# with one block changed in a line.
again = comrs.Chain.from_spec(path)
print(again.names, again.settings["grid"])
harder = again.replace(channel=B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")))
print(np.array([r["status"] for r in harder.run(grid={"m": [16], "ebn0_db": [8, 10]})]))
