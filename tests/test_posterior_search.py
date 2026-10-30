"""Receiver posteriors (achievable rates with the receiver's own beliefs) and the search
of operating points, from Python."""

import pytest
from conftest import ROOT, qpsk_spec

import comrs


def test_posterior_rates_of_calibrated_and_over_confident_receivers():
    c = comrs.Constellation("qam", 16).normalized()
    rng = comrs.Rng(5)
    idx, x = comrs.symbols(c, 20_000, rng)
    z = comrs.awgn(x, 12.0, rng)
    s2 = 10**-1.2
    good = comrs.posterior_rates(z, idx, c, s2)
    genie = comrs.information_rates(z, idx, c)
    assert good["air"] == pytest.approx(genie["mi"], abs=0.01)
    assert good["calibration"] == pytest.approx(1.0, abs=0.05)
    over = comrs.posterior_rates(z, idx, c, s2 / 4)
    assert over["air"] < good["air"] - 0.05
    assert over["scale"] == pytest.approx(0.25, abs=0.05)
    assert over["air_opt"] == pytest.approx(good["air"], abs=0.01)
    assert over["air_bmd"] <= over["air"] + 1e-9
    with pytest.raises(ValueError, match="variance"):
        comrs.posterior_rates(z, idx, c, 0.0)


def test_noise_estimate_declares_the_posterior_of_a_classic_receiver():
    spec = {
        "name": "post",
        "link": [
            {"type": "source", "constellation": "16-QAM", "n_symbols": 4000},
            {"type": "awgn", "snr_db": 14},
            {"type": "noise_estimate", "mode": "known"},
        ],
        "metrics": {"report": ["mi", "air", "posterior_calibration"], "scale": "none"},
        "stop": {"max_trials": 8},
    }
    m = comrs.run_spec(spec)[0]["metrics"]
    assert m["air"]["value"] == pytest.approx(m["mi"]["value"], abs=0.01)
    assert m["posterior_calibration"]["value"] == pytest.approx(1.0)
    spec["link"] = spec["link"][:2]
    with pytest.raises(ValueError, match="posterior"):
        comrs.run_spec(spec)


def test_search_finds_the_operating_point_and_its_interval(tmp_path):
    spec = qpsk_spec(
        4000,
        name="qpsk_point",
        metrics={"report": ["ber"], "scale": "none"},
        stop={"min_errors": {"ber": 400}, "max_trials": 400},
    )
    del spec["grid"]
    spec["search"] = {
        "along": "ebn0_db",
        "range": [2, 10],
        "metric": "ber",
        "target": 1e-3,
        "tolerance": 0.05,
    }
    r = comrs.search_spec(spec, csv_path=tmp_path / "p.csv", json_path=tmp_path / "p.json")
    assert len(r) == 1 and r[0]["converged"]
    assert r[0]["lo"] <= r[0]["value"] <= r[0]["hi"]
    assert r[0]["value"] == pytest.approx(6.79, abs=0.15)
    xs = [e["x"] for e in r[0]["evaluations"]]
    assert xs[:2] == [2.0, 10.0] and len(xs) <= 10
    assert (tmp_path / "p.csv").read_text(encoding="utf-8").startswith("metric,target,ebn0_db,")
    with pytest.raises(ValueError, match="not between"):
        comrs.search_spec({**spec, "search": {**spec["search"], "range": [9, 10]}})


def test_verify_spec_checks_a_search():
    out = comrs.verify_spec(ROOT / "specs" / "validation" / "required_snr.yaml")
    assert out["passed"] and len(out["checks"]) == 3
    for c in out["checks"]:
        assert c["metric"] == "ebn0_db" and abs(c["value"] - c["expected"]) < 0.1
