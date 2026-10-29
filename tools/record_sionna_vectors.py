#!/usr/bin/env python3
"""Record the reference vectors of Sionna that com-rs is compared with.

Sionna (NVIDIA, Apache-2.0, <https://github.com/NVlabs/sionna>) ships test vectors of
its channel codes, most of them generated with a reference implementation (MATLAB):
the CRCs of 3GPP, convolutional codes encoded and Viterbi-decoded, the turbo code of
LTE encoded and decoded at Eb/N0 = 0 dB. This script downloads them at the pinned
commit ``COMMIT``, keeps the first ``ROWS`` blocks of every set, and writes
``tests/oracle/sionna_vectors.npz`` and ``tests/oracle/sionna_vectors.json`` (the
provenance: the repository, the commit, the files, the licence), read by
``tests/test_sionna_vectors.py``. The vectors are redistributed under the terms of the
Apache License 2.0 (see ``THIRD_PARTY.md``)::

    python3 tools/record_sionna_vectors.py

The output is deterministic (sorted keys, a fixed timestamp in the archive): recording
again at the same commit produces no diff.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

REPO = "NVlabs/sionna"
COMMIT = "2f0a3820269b53fd022780a6f2f5f720f7ceb9b4"
ROWS = 4
OUT = Path(__file__).resolve().parent.parent / "tests" / "oracle"
CRCS = ["CRC6", "CRC11", "CRC16", "CRC24A", "CRC24B", "CRC24C"]
CONV = [
    "conv_rate_half_57_",
    "conv_rate_half_6474_",
    "conv_rate_onethird_577_",
    "conv_rate_onefourth_5777_",
]
TURBO = [40, 112, 168, 432]


def files() -> dict[str, str]:
    """The arrays to record, by key, and their paths in the repository."""
    out = {}
    for c in CRCS:
        out[f"crc/{c}/u"] = f"test/codes/crc/crc_u_{c}.npy"
        out[f"crc/{c}/crc"] = f"test/codes/crc/crc_x_ref_np_{c}.npy"
    for c in CONV:
        for part in ("u", "x", "y", "uhat"):
            out[f"conv/{c.strip('_')}/{part}"] = f"test/codes/conv/{c}ref_{part}.npy"
    for k in TURBO:
        for part in ("u", "x", "y", "uhat"):
            out[f"turbo/k{k}/{part}"] = f"test/codes/turbo/ref_k{k}_{part}.npy"
    return out


def fetch(path: str) -> np.ndarray:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return np.load(io.BytesIO(r.read()), allow_pickle=False)


def main() -> None:
    arrays = {}
    for key, path in sorted(files().items()):
        a = fetch(path)
        if a.ndim == 2:
            a = a[:ROWS]
        # bits as uint8, soft values as float64
        arrays[key] = a.astype(np.float64) if key.endswith("/y") else np.rint(a).astype(np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key in sorted(arrays):
            b = io.BytesIO()
            np.save(b, arrays[key], allow_pickle=False)
            info = zipfile.ZipInfo(key.replace("/", ".") + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, b.getvalue())
    (OUT / "sionna_vectors.npz").write_bytes(buf.getvalue())
    doc = {
        "tool": "Sionna (test vectors)",
        "repository": f"https://github.com/{REPO}",
        "commit": COMMIT,
        "license": "Apache-2.0",
        "rows": ROWS,
        "files": dict(sorted(files().items())),
    }
    (OUT / "sionna_vectors.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
