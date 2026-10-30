"""The Python chain API: named blocks, Python blocks, trace, specifications and both
runners."""

import inspect

import numpy as np
import pytest
from conftest import SPECS, qpsk_chain

import comrs
from comrs import blocks as B


def qpsk_link(n_symbols=1000):
    return qpsk_chain(n_symbols)


def test_blocks_from_registry():
    assert set(B.__all__) == {comrs.block._class_name(k) for k in comrs.spec_blocks()}
    assert "ebn0_db" in B.AwgnEbn0.__doc__
    assert list(inspect.signature(B.AwgnEbn0).parameters) == list(
        comrs.spec_blocks()["awgn_ebn0"]["params"]
    )
    with pytest.raises(TypeError, match="snr"):
        B.AwgnEbn0(snr=3)
    with pytest.raises(TypeError):
        B.AwgnEbn0(3.0)  # keyword-only
    # a parameter equal to its default is left to the block
    assert B.PulseShape(sps=4, n_taps=1025).spec() == {"type": "pulse_shape", "sps": 4}


def test_containers():
    link = qpsk_link()
    assert link.names == ["tx", "channel"]
    assert len(link) == 2
    assert link["channel"].params["ebn0_db"].name == "ebn0_db"
    assert link.is_native
    louder = link.replace(channel=B.AwgnEbn0(ebn0_db=20))
    assert louder["channel"].params == {"ebn0_db": 20}
    assert link["channel"].params["ebn0_db"].name == "ebn0_db"
    longer = link.insert_after("tx", norm=B.Normalize())
    assert longer.names == ["tx", "norm", "channel"]
    nested = comrs.Chain(link=link, rx=B.Normalize())
    assert [n for n, _ in nested.flatten()] == ["link.tx", "link.channel", "rx"]


def test_call_and_trace_are_reproducible():
    link = qpsk_link()
    a = link({"ebn0_db": 6.0}, comrs.Rng(3))
    b = link({"ebn0_db": 6.0}, comrs.Rng(3))
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.tx_bits.size == 2000
    t = link.trace({"ebn0_db": 6.0}, comrs.Rng(3))
    assert list(t) == ["tx", "channel"]
    np.testing.assert_array_equal(t["tx"].samples, t["tx"].tx_symbols)
    np.testing.assert_array_equal(t["channel"].samples, a.samples)
    with pytest.raises(KeyError, match="ebn0_db"):
        link({}, comrs.Rng(0))


def test_native_run_matches_run_spec():
    link = qpsk_link()
    kw = {"grid": {"ebn0_db": [2.0, 6.0]}, "metrics": ["ber", "snr_db"], "seed": 5}
    res = link.run(stop={"max_trials": 64}, **kw)
    ref = comrs.run_spec(link.to_spec(stop={"max_trials": 64}, **kw))
    assert [r["metrics"]["ber"]["count"] for r in res] == [
        r["metrics"]["ber"]["count"] for r in ref
    ]
    for r in res:
        theory = comrs.theory_ber(4, r["condition"]["ebn0_db"], "qam")
        assert r["metrics"]["ber"]["lo"] < theory < r["metrics"]["ber"]["hi"]


def test_python_runner_matches_rust_runner():
    """The same streams on both runners: a Python identity block switches a native
    chain to the Python runner without changing a single error."""
    link = qpsk_link()
    kw = {
        "grid": {"ebn0_db": [3.0, 5.0]},
        "metrics": ["ber", "ser"],
        "stop": {"min_errors": {"ber": 100}, "max_trials": 96},
        "seed": 11,
        "name": "parity",
    }
    rust = link.run(**kw)
    python = link.insert_after("channel", noop=lambda frame, rng: None)
    assert not python.is_native
    py = python.run(**kw)
    for a, b in zip(rust, py, strict=True):
        assert a["trials"] == b["trials"]
        for m in ("ber", "ser"):
            assert a["metrics"][m]["count"] == b["metrics"][m]["count"]
            assert a["metrics"][m]["total"] == b["metrics"][m]["total"]


class Attenuator(comrs.Block):
    """A Python block: a fixed gain on the samples."""

    def __init__(self, gain):
        self.gain = gain

    def __call__(self, frame, rng):
        frame.samples = self.gain * frame.samples


def test_python_block_and_recorded_values():
    def power(frame, rng):
        frame.record("power", float(np.mean(np.abs(frame.samples) ** 2)))

    link = comrs.Chain(
        tx=B.Source(constellation="QPSK", n_symbols=500),
        att=Attenuator(0.5),
        meter=power,
        rx=B.Normalize(),
    )
    res = link.run(metrics=["ber"], stop={"max_trials": 8})
    assert res[0]["metrics"]["power"]["value"] == pytest.approx(0.25)
    assert res[0]["metrics"]["ber"]["count"] == 0
    with pytest.raises(TypeError, match="Python blocks"):
        link.to_spec()


def test_python_stop_rule():
    res = qpsk_link().run(grid={"ebn0_db": [4.0]}, stop=comrs.sim.Stop.max_trials(10), seed=1)
    assert res[0]["trials"] == 32  # one batch


def test_from_spec_round_trip():
    chain = comrs.Chain.from_spec(SPECS / "coherent_link.yaml")
    assert chain.names[:4] == ["source", "pulse_shape", "tx_laser", "fiber"]
    assert chain["lo_laser"].kind == "phase_noise"
    assert chain["fiber"].params["length_km"].name == "length_km"
    assert chain.settings["seed"] == 7
    spec = chain.to_spec()
    assert spec["name"] == "coherent_link"
    assert spec["link"][2] == {"type": "phase_noise", "label": "tx_laser", "linewidth_hz": 100e3}
    assert spec["link"][3]["length_km"] == "$length_km"
    again = comrs.Chain.from_spec(spec)
    assert again.to_spec() == spec
    assert comrs.spec_conditions(spec) == comrs.spec_conditions(SPECS / "coherent_link.yaml")


def test_from_yaml_text_runs_like_the_specification():
    text = """
name: yaml
seed: 3
grid: {ebn0_db: [5]}
link: [{type: source, constellation: QPSK, n_symbols: 500},
       {type: awgn_ebn0, ebn0_db: $ebn0_db}]
metrics: {report: [ber]}
stop: {max_trials: 32}
"""
    chain = comrs.Chain.from_spec(text)
    assert (
        chain.run()[0]["metrics"]["ber"]["count"]
        == (comrs.run_spec(text)[0]["metrics"]["ber"]["count"])
    )


# --- results, probes, traces and plots -------------------------------------------------


def awgn_link():
    """QPSK over AWGN at the grid variable ``snr_db``, the channel named ``ch``."""
    link = qpsk_chain(2000, snr="snr_db")
    return comrs.Chain(tx=link["tx"], ch=link["channel"])


def test_results_are_tables_that_round_trip(tmp_path):
    res = awgn_link().run(
        grid={"snr_db": [4.0, 8.0]}, metrics=["ser", "snr_db"], stop={"max_trials": 8}
    )
    assert isinstance(res, comrs.Results) and isinstance(res, list)
    assert res.variables == ["snr_db"] and res.metrics == ["ser", "snr_db"]
    x, ser, lo, hi = res.series("snr_db", "ser", interval=True)
    assert list(x) == [4.0, 8.0] and ser[0] > ser[1] and np.all(lo <= ser) and np.all(ser <= hi)
    assert res.where(snr_db=8.0)[0]["condition"] == {"snr_db": 8.0}
    rows = res.rows()
    assert {"snr_db", "ser", "ser_lo", "ser_hi", "ser_count", "trials"} <= set(rows[0])
    back = comrs.Results.load(res.save(tmp_path / "r.json"))
    assert back == res and back.spec["link"][1]["type"] == "awgn"
    # the JSON report of a specification reads back as well
    assert len(comrs.Results(comrs.run_spec(res.spec))) == 2


def test_where_compares_reals_to_a_relative_tolerance():
    res = comrs.Results(
        [
            {"condition": {"noise_var": v, "x": x}, "metrics": {}}
            for v, x in [(1e-12, 2.2e-16), (2e-12, 1.0)]
        ]
    )
    assert [r["condition"]["noise_var"] for r in res.where(noise_var=1e-12)] == [1e-12]
    assert [r["condition"]["noise_var"] for r in res.where(noise_var=2e-12 * (1 + 1e-12))] == [
        2e-12
    ]
    assert len(res.where(x=0)) == 1  # a grid step that lands near zero
    assert len(res.where(x=1.0 + 1e-6)) == 0


def test_compare_gathers_variants():
    base = awgn_link()
    res = comrs.compare(
        {"awgn": base, "phase noise": base.insert_after("ch", pn=B.PhaseNoise(sigma=0.05))},
        grid={"snr_db": [10.0]},
        metrics=["ser"],
        stop={"max_trials": 8},
    )
    assert res.values("variant") == ["awgn", "phase noise"]
    ser = {r["condition"]["variant"]: r["metrics"]["ser"]["value"] for r in res}
    assert ser["phase noise"] > ser["awgn"]


def test_probes_measure_inside_the_chain_on_both_runners():
    link = (
        awgn_link()
        .insert_after("ch", rot=B.Cfo(omega=0.0))
        .probes({"tx": ["power_dbm"], "ch": ["snr_db", "ser"]}, scale="none")
    )
    kw = {"grid": {"snr_db": [10.0]}, "metrics": ["ser"], "stop": {"max_trials": 16}}
    res = link.run(**kw)[0]["metrics"]
    assert res["tx.power_dbm"]["value"] == pytest.approx(30.0)
    assert res["ch.snr_db"]["value"] == pytest.approx(10.0, abs=0.1)
    assert res["ch.ser"]["count"] == res["ser"]["count"]
    # the same probes on the Python runner
    py = link.insert_after("ch", noop=lambda frame, rng: None).run(**kw)[0]["metrics"]
    assert py["ch.ser"]["count"] == res["ch.ser"]["count"]
    # a probe is a block of the specification
    assert {
        "type": "probe",
        "label": "ch",
        "metrics": ["snr_db", "ser"],
        "scale": "none",
    } in link.to_spec()["link"]


def test_trace_saves_every_stage(tmp_path):
    t = awgn_link().trace({"snr_db": 10.0}, comrs.Rng(1))
    assert isinstance(t, comrs.Trace) and list(t) == ["tx", "ch"]
    data = np.load(t.save(tmp_path / "t.npz"))
    np.testing.assert_array_equal(data["ch"], t["ch"].samples)
    np.testing.assert_array_equal(data["tx_symbols"], t["tx"].tx_symbols)


def test_plots_draw_frames_and_results():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    link = comrs.Chain(
        tx=B.Source(constellation="16-QAM", n_symbols=1000, pols=2),
        shape=B.PulseShape(sps=4, rolloff=0.2),
        ch=B.Awgn(snr_db=20.0),
    )
    t = link.trace({}, comrs.Rng(2))
    ax = comrs.plot.constellation(t["tx"])
    assert len(ax.lines) == 3  # two polarisations and the reference points
    comrs.plot.eye(t["ch"])
    comrs.plot.time(t["ch"], part="power")
    comrs.plot.psd(t["ch"])
    comrs.plot.constellation(t["ch"].samples, density=True)
    fig = t.plot("constellation")
    assert len([a for a in fig.axes if a.get_visible()]) == 3
    res = awgn_link().run(grid={"snr_db": [4.0, 6.0, 8.0]}, metrics=["ser"], stop={"max_trials": 8})
    ax = res.plot(
        "snr_db", "ser", theory=lambda x: [comrs.theory_ser(4, v - 3.0103, "qam") for v in x]
    )
    assert ax.get_yscale() == "log"
    with pytest.raises(ValueError, match="2 samples per symbol"):
        comrs.plot.eye(t["tx"])
    plt.close("all")


def test_pmd_rotates_and_frame_sync_restores_the_order_of_the_polarisations():
    # a 90° rotation swaps the polarisations (and the sign of one): frame_sync finds the
    # swap from the known symbols
    link = comrs.Chain(
        tx=B.Source(constellation="QPSK", n_symbols=2000, pols=2),
        pmd=B.Pmd(theta_deg=90.0),
        sync=B.FrameSync(start=256, length=256, max_delay=4, symmetry=0),
    )
    f = link({}, comrs.Rng(1))
    assert (
        dict(f.values)["sync.pol_swap"] == 1.0
    )  # recorded under the name of the block in the chain
    x = f.tx_symbols
    np.testing.assert_allclose(np.concatenate(f.streams), x, atol=1e-9)
    # a random realisation conserves the power of the two polarisations
    g = comrs.Chain(
        tx=B.Source(constellation="QPSK", n_symbols=2000, pols=2), pmd=B.Pmd(mean_dgd_ps=10.0)
    )
    t = g.trace({"x": 0}, comrs.Rng(2))
    p0 = sum(np.sum(abs(s) ** 2) for s in t["tx"].streams)
    p1 = sum(np.sum(abs(s) ** 2) for s in t["pmd"].streams)
    assert p1 == pytest.approx(p0, rel=1e-9)


def test_block_with_a_parameter_named_kind():
    """`kind` of `Native` is positional-only: the blocks whose parameter is `kind`
    (pulse_shape, matched_filter) take it as a keyword."""
    b = B.PulseShape(kind="rc", sps=4, rolloff=0.25)
    assert b.kind == "pulse_shape"
    assert b.spec() == {"type": "pulse_shape", "kind": "rc", "sps": 4, "rolloff": 0.25}
    assert B.MatchedFilter(kind="rc").params == {"kind": "rc"}


def test_profile_on_both_runners(tmp_path):
    """`profile=True` times every block on both runners, in the order of the link."""
    link = comrs.Chain(
        tx=B.Source(constellation="QPSK", n_symbols=2048),
        shaping=B.PulseShape(sps=2),
        channel=B.AwgnEbn0(ebn0_db=6.0),
        mf=B.MatchedFilter(),
        down=B.Decimate(),
    )
    rows = link.profile(stop={"max_trials": 4})
    assert [r["block"] for r in rows] == ["tx", "shaping", "channel", "mf", "down", "link"]
    assert all(r["time_ms"] >= 0 for r in rows) and rows[-1]["share"] == 1.0
    assert abs(sum(r["share"] for r in rows[:-1]) - 1.0) < 0.5
    assert link.run(stop={"max_trials": 2}).profile() == []
    py = link.insert_after("channel", noop=lambda f, r: None)
    rows = py.run(stop={"max_trials": 2}, profile=True).profile()
    assert [r["block"] for r in rows] == ["tx", "shaping", "channel", "noop", "mf", "down", "link"]


def test_dump_block_saves_the_first_trials(tmp_path):
    """`dump` writes the samples of the first trials of every condition as SigMF."""
    link = comrs.Chain(
        tx=B.Source(constellation="QPSK", n_symbols=256, symbol_rate=1e6),
        shaping=B.PulseShape(sps=4),
        tap=B.Dump(dir=str(tmp_path), trials=2),
        channel=B.Awgn(snr_db=comrs.var("snr_db")),
        mf=B.MatchedFilter(),
        down=B.Decimate(),
    )
    link.run(grid={"snr_db": [5, 10]}, stop={"max_trials": 8}, batch=8, name="dumps")
    files = sorted(p.name for p in (tmp_path / "dumps" / "snr_db=5").iterdir())
    assert files == ["tap_0.sigmf-data", "tap_0.sigmf-meta", "tap_1.sigmf-data", "tap_1.sigmf-meta"]
    x, fs, meta = comrs.read_sigmf(tmp_path / "dumps" / "snr_db=10" / "tap_0.sigmf-meta")
    assert x.shape == (1024,) and fs == 4e6
    assert meta["global"]["comrs:sps"] == 4.0 and meta["global"]["comrs:block"] == "tap"
    # trial 0 (by index, whatever the order the threads ran them in) is reproducible
    # from its stream, on both runners
    frame = link.trace({"snr_db": 10}, comrs.Rng.stream(0, "dumps", 1, 0))["tap"]
    np.testing.assert_allclose(frame.samples, x)
    py = link.insert_after("tx", noop=lambda f, r: None)
    py.run(grid={"snr_db": [10]}, stop={"max_trials": 3}, name="pydumps")
    y, _, meta = comrs.read_sigmf(tmp_path / "pydumps" / "snr_db=10" / "tap_1.sigmf-meta")
    assert meta["global"]["comrs:trial"] == 1
    np.testing.assert_allclose(
        y, link.trace({"snr_db": 10}, comrs.Rng.stream(0, "pydumps", 0, 1))["tap"].samples
    )
