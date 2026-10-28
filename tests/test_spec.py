"""Declarative campaigns from Python: the specifications of ``specs/``."""

from pathlib import Path

import pytest

import comrs

SPECS = Path(__file__).resolve().parents[1] / "specs"


def test_a_specification_as_text_dict_and_file_gives_the_same_results(tmp_path):
    text = """
name: forms
seed: 4
grid: {ebn0_db: [3, 6]}
link:
  - {type: source, constellation: 16-QAM, n_symbols: 1000}
  - {type: awgn_ebn0, ebn0_db: $ebn0_db}
metrics: {report: [ber, snr_db]}
stop: {max_trials: 32}
"""
    as_dict = {
        "name": "forms",
        "seed": 4,
        "grid": {"ebn0_db": [3, 6]},
        "link": [
            {"type": "source", "constellation": "16-QAM", "n_symbols": 1000},
            {"type": "awgn_ebn0", "ebn0_db": "$ebn0_db"},
        ],
        "metrics": {"report": ["ber", "snr_db"]},
        "stop": {"max_trials": 32},
    }
    path = tmp_path / "forms.yaml"
    path.write_text(text)
    results = [comrs.run_spec(s) for s in (text, as_dict, path, str(path))]
    for r in results[1:]:
        assert [c["metrics"] for c in r] == [c["metrics"] for c in results[0]]
    r = results[0]
    assert [c["condition"] for c in r] == [{"ebn0_db": 3}, {"ebn0_db": 6}]
    ber = r[1]["metrics"]["ber"]
    assert ber["kind"] == "rate" and ber["lo"] <= comrs.theory_ber(16, 6, "qam") <= ber["hi"]


def test_results_do_not_depend_on_the_threads():
    spec = SPECS / "qam_awgn.yaml"
    a = comrs.run_spec(spec, threads=1, fast_dev_run=True)
    b = comrs.run_spec(spec, threads=4, fast_dev_run=True)
    assert [c["metrics"] for c in a] == [c["metrics"] for c in b]


def test_reports_are_written(tmp_path):
    comrs.run_spec(
        SPECS / "qam_awgn.yaml",
        fast_dev_run=True,
        csv_path=tmp_path / "r.csv",
        json_path=tmp_path / "r.json",
    )
    assert (tmp_path / "r.csv").read_text().startswith("index,m,ebn0_db,status")
    assert '"scenario": "qam_awgn"' in (tmp_path / "r.json").read_text()


def test_conditions_and_blocks():
    conds = comrs.spec_conditions(SPECS / "ssfm_dbp.yaml")
    assert len(conds) == 4 * 7
    assert conds[0] == {"receiver": "cdc", "steps": 1, "nonlinear": False, "power_dbm": -4}
    blocks = comrs.spec_blocks()
    assert "awgn" in blocks and "snr_db" in blocks["awgn"]["params"]


def test_invalid_specifications_raise_value_errors():
    with pytest.raises(ValueError, match="unknown block"):
        comrs.run_spec({"name": "x", "link": [{"type": "source"}, {"type": "awgnn"}]})
    with pytest.raises(ValueError, match="snr"):
        comrs.run_spec("name: x\nlink: [source, {type: awgn, snr: 3}]")
    with pytest.raises(ValueError, match="1 sample per symbol"):
        comrs.run_spec(
            "name: x\nlink: [source, {type: pulse_shape, sps: 2}]\nstop: {max_trials: 1}"
        )
