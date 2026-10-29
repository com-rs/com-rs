#!/usr/bin/env python3
"""Record the reference vectors of Sionna that com-rs is compared with.

Sionna (NVIDIA, Apache-2.0, <https://github.com/NVlabs/sionna>) ships test vectors of
its channel codes, most of them generated with a reference implementation (MATLAB):
the CRCs of 3GPP, convolutional codes encoded and Viterbi-decoded, the turbo code of
LTE encoded and decoded at Eb/N0 = 0 dB, the polar code of the 5G uplink control
information (CRC, rate matching), and the generator matrices of the 5G LDPC code with
its rate matching. The generator matrices (19 MB) are not kept: two random messages
per matrix (seed ``SEED``) and their codewords ``u G`` are. This script downloads them at the pinned
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
POLAR = ["E45_k30_K41", "E70_k32_K43", "E127_k29_K40", "E1023_k400_K411", "E70_k28_K39"]
LDPC = [
    (64, 128),
    (64, 160),
    (75, 210),
    (200, 220),
    (256, 412),
    (256, 480),
    (257, 871),
    (275, 510),
    (364, 547),
    (562, 871),
    (698, 878),
    (784, 985),
    (948, 1024),
    (1347, 1554),
    (1487, 3554),
    (2657, 3754),
    (5478, 6045),
    (5478, 7525),
    (5781, 8962),
    (5874, 10255),
    (6479, 13235),
    (7854, 9754),
    (7854, 12354),
    (8445, 18665),
    (8446, 15874),
    (8448, 12354),
    (8448, 23540),
    (8448, 25344),
]
SEED = 5


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
    for name in POLAR:
        for part in ("u", "c"):
            out[f"polar/{name}/{part}"] = f"test/codes/polar/{name}_{part}.npy"
    return out


def generator_files() -> dict[str, str]:
    return {f"ldpc/k{k}_n{n}": f"test/codes/ldpc/k{k}_n{n}_G.npy" for k, n in LDPC}


def fetch(path: str) -> np.ndarray:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return np.load(io.BytesIO(r.read()), allow_pickle=False)


def fetch_pickled(path: str) -> np.ndarray:
    # the generator matrices are object arrays: loaded once here, never redistributed
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
    with urllib.request.urlopen(url, timeout=120) as r:
        return np.array(np.load(io.BytesIO(r.read()), allow_pickle=True))


def main() -> None:
    arrays = {}
    for key, path in sorted(files().items()):
        a = fetch(path)
        if a.ndim == 2:
            a = a[:ROWS]
        # bits as uint8, soft values as float64
        arrays[key] = a.astype(np.float64) if key.endswith("/y") else np.rint(a).astype(np.uint8)
    rng = np.random.default_rng(SEED)
    for key, path in sorted(generator_files().items()):
        k, n = (int(x) for x in key.split("/k")[1].split("_n"))
        sparse = fetch_pickled(path)
        g = np.zeros((k, n), np.uint8)
        g[sparse[0] - 1, sparse[1] - 1] = 1
        u = rng.integers(0, 2, (2, k), dtype=np.uint8)
        arrays[f"{key}/u"] = u
        arrays[f"{key}/c"] = (u.astype(np.int64) @ g % 2).astype(np.uint8)
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
        "generator_matrices": dict(sorted(generator_files().items())),
        "generator_seed": SEED,
    }
    (OUT / "sionna_vectors.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
