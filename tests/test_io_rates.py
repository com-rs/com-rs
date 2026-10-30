"""Recordings (SigMF), the least-squares information rates, single-precision split step
and the verification of specifications from Python."""

import json

import numpy as np
import pytest
from conftest import ROOT, qpsk_spec

import comrs


def test_sigmf_round_trip_with_metadata(tmp_path):
    rng = comrs.Rng(1)
    x = np.round(rng.complex_noise(1000, 1e6))
    comrs.write_sigmf(
        tmp_path / "rec", x, 80e9, datatype="ci16_le", global_meta={"comrs:symbol_rate": 32e9}
    )
    y, fs, meta = comrs.read_sigmf(tmp_path / "rec.sigmf-meta")
    np.testing.assert_array_equal(
        y, np.clip(x.real, -32768, 32767) + 1j * np.clip(x.imag, -32768, 32767)
    )
    assert fs == 80e9 and meta["global"]["comrs:symbol_rate"] == 32e9
    assert meta["global"]["core:datatype"] == "ci16_le"
    raw = np.fromfile(tmp_path / "rec.sigmf-data", dtype="<i2")
    assert raw.size == 2 * x.size
    with pytest.raises(ValueError, match="data type"):
        comrs.write_sigmf(tmp_path / "bad", x, 1.0, datatype="cf16_be")


def test_the_synthetic_capture_is_readable():
    y, fs, meta = comrs.read_sigmf(ROOT / "data" / "synthetic_16qam.sigmf-meta")
    ref = np.load(ROOT / "data" / "synthetic_16qam_reference.npy")
    assert (y.size, fs, ref.size) == (32768, 80e9, 4096)
    assert meta["global"]["comrs:symbol_rate"] == 32e9 and ref.max() < 16


def test_least_squares_rates_are_consistent_where_the_opticommpy_estimators_are_biased():
    c = comrs.Constellation("qam", 16).normalized()
    rng = comrs.Rng(2)
    idx, x = comrs.symbols(c, 60_000, rng)
    y = comrs.awgn(x, 0.0, rng) * 0.3 * np.exp(0.7j)  # any complex gain
    r = comrs.information_rates(y, idx, c)
    assert r["mi"] == pytest.approx(comrs.theory_mi(16, 0.0), abs=0.01)
    assert r["gmi"] == pytest.approx(comrs.theory_gmi(16, 0.0), abs=0.01)
    assert r["snr_db"] == pytest.approx(0.0, abs=0.05)
    # the power normalisation of OptiCommPy's estimator loses about 0.06 bit at 0 dB
    assert comrs.monte_carlo_mi(y, x, c) < r["mi"] - 0.04


def test_single_precision_split_step():
    link = comrs.Ssfm(comrs.FiberParams(n_spans=3), step_km=0.5, amplification="ideal")
    x = comrs.Rng(3).complex_noise(4096, 2e-3)
    a = link.propagate(x, 64e9)
    b = link.propagate(x, 64e9, precision="f32")
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-4
    with pytest.raises(ValueError, match="precision"):
        link.propagate(x, 64e9, precision="f16")


def test_verify_spec_reports_every_check():
    out = comrs.verify_spec(ROOT / "specs" / "validation" / "proakis_awgn.yaml")
    assert out["passed"] and len(out["checks"]) > 20
    c = out["checks"][0]
    assert c["source"].startswith("Proakis") and c["lo"] <= c["value"] <= c["hi"]
    bad = qpsk_spec(name="wrong", expect=[{"metric": "ber", "value": 0.1, "rtol": 0.1}])
    out = comrs.verify_spec(bad)
    assert not out["passed"] and json.dumps(out["checks"])
    with pytest.raises(ValueError, match="expect"):
        comrs.verify_spec({**bad, "expect": []})
