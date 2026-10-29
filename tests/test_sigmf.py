"""SigMF against its reference implementation, sigmf-python (skipped when absent).

Recordings written by comrs pass the validation of sigmf-python and read back the same
samples; recordings written by sigmf-python (every data type, big endian, several
channels, captures, annotations, archives) read back the same in comrs. sigmf-python
scales 32-bit integers in float32 (an error of 3e-8 at full scale), comrs in float64:
the comparison allows that error for those types.
"""

import numpy as np
import pytest

import comrs

sigmf = pytest.importorskip("sigmf")
from sigmf import SigMFFile  # noqa: E402
from sigmf.sigmffile import fromfile  # noqa: E402

TYPES = [
    f"{c}{t}{e}"
    for c in "cr"
    for t in ["f32", "f64", "i32", "i16", "u32", "u16"]
    for e in ["_le", "_be"]
] + [f"{c}{t}" for c in "cr" for t in ["i8", "u8"]]


def atol(datatype):
    return 1e-7 if "32" in datatype and datatype[1] in "iu" else 1e-12


def ramp(n=64):
    k = np.arange(n)
    return (k - 32) + 1j * np.round(7 - k / 3)


@pytest.mark.parametrize("datatype", TYPES)
def test_comrs_recordings_read_by_sigmf_python(tmp_path, datatype):
    x = ramp()
    if datatype.startswith("r"):
        x = x.real.astype(complex)
    base = tmp_path / "rec"
    comrs.write_sigmf(
        base,
        x,
        1e6,
        datatype=datatype,
        frequency=2.4e9,
        annotations=[{"sample_start": 3, "sample_count": 5, "label": "QPSK"}],
    )
    f = fromfile(str(base) + ".sigmf-meta", skip_checksum=True)
    f.validate()
    ours, fs, _ = comrs.read_sigmf(base, autoscale=True)
    theirs = f.read_samples()
    assert fs == 1e6 and f.get_captures()[0]["core:frequency"] == 2.4e9
    assert f.get_annotations()[0]["core:label"] == "QPSK"
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=atol(datatype))


@pytest.mark.parametrize("datatype", TYPES)
def test_sigmf_python_recordings_read_by_comrs(tmp_path, datatype):
    info = sigmf.sigmffile.dtype_info(datatype)
    n, channels = 30, 2
    rng = np.random.default_rng(1)
    raw = rng.integers(-100, 100, size=n * channels * (2 if info["is_complex"] else 1))
    if info["is_unsigned"]:
        raw = raw + 128
    raw = raw.astype(info["sample_dtype"] if not info["is_complex"] else info["component_dtype"])
    path = tmp_path / "x.sigmf-data"
    raw.tofile(path)
    meta = SigMFFile(
        data_file=str(path),
        global_info={
            sigmf.DATATYPE_KEY: datatype,
            sigmf.SAMPLE_RATE_KEY: 2e6,
            sigmf.NUM_CHANNELS_KEY: channels,
        },
    )
    meta.add_capture(0, metadata={sigmf.FREQUENCY_KEY: 915e6})
    meta.add_capture(10, metadata={sigmf.FREQUENCY_KEY: 916e6})
    meta.add_annotation(4, 6, metadata={sigmf.LABEL_KEY: "16-QAM"})
    meta.tofile(tmp_path / "x")
    meta.tofile(tmp_path / "arch", toarchive=True)
    theirs = fromfile(str(tmp_path / "x.sigmf-meta"), skip_checksum=True).read_samples()
    for name in ["x.sigmf-meta", "arch.sigmf"]:
        ours, fs, m = comrs.read_sigmf(tmp_path / name, autoscale=True)
        assert fs == 2e6 and ours.shape == (channels, n)
        assert [c["core:frequency"] for c in m["captures"]] == [915e6, 916e6]
        np.testing.assert_allclose(ours.T, theirs, rtol=0, atol=atol(datatype))
        one = comrs.read_sigmf(tmp_path / name, channel=1, autoscale=True)[0]
        np.testing.assert_allclose(one, theirs[:, 1], rtol=0, atol=atol(datatype))
