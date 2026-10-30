"""comrs against the reference vectors of Sionna (Apache-2.0).

Sionna (NVIDIA) ships test vectors of its channel codes, generated with a reference
implementation; ``tools/record_sionna_vectors.py`` keeps four blocks of every set at a
pinned commit in ``tests/oracle/sionna_vectors.npz`` (provenance in the ``.json``):

- the CRCs of 3GPP TS 38.212 (CRC6 to CRC24C): the parity bits of a message;
- convolutional codes of rates 1/2, 1/3 and 1/4 (generators (5, 7), (15, 17), (5, 7, 7),
  (5, 7, 7, 7) in octal), not terminated: the codewords, and the Viterbi decisions from
  noisy observations;
- the turbo code of LTE (TS 36.212), K = 40, 112, 168 and 432, terminated: the
  codewords, and the decisions of 10 iterations of log-MAP at Eb/N0 = 0 dB, errors
  included;
- the polar code of the 5G uplink control information (TS 38.212, CRC11, rate matching
  by puncturing, shortening and repetition, the triangular interleaver): the codewords
  of five (A, E);
- the LDPC code of 5G NR with its rate matching (TS 38.212): the codewords of two
  messages through the generator matrices of 28 (K', E), from 64 to 8448 bits, both
  base graphs.

Conventions: Sionna writes the turbo codeword symbol by symbol (systematic, parity 1,
parity 2 at every index, then the 12 tail bits), comrs stream by stream; its
observations are logits (positive for a 1) of BPSK at noise variance 3 (the LLR
``4 y / N0``), comrs's LLRs are positive for a 0. comrs terminates its convolutional
codes: the codewords of Sionna are the first ``2K`` (or ``3K``, ``4K``) bits, and the
tail is decoded with zero LLRs.
"""

import json

import numpy as np
import pytest
from oracle import DIR, missing

import comrs

if not (DIR / "sionna_vectors.npz").exists():
    missing(
        "tests/oracle/sionna_vectors.npz is missing: run tools/record_sionna_vectors.py",
        module=True,
    )
V = dict(np.load(DIR / "sionna_vectors.npz"))
META = json.loads((DIR / "sionna_vectors.json").read_text(encoding="utf-8"))


def get(key):
    return V[key.replace("/", ".")]


@pytest.mark.parametrize("name", ["CRC6", "CRC11", "CRC16", "CRC24A", "CRC24B", "CRC24C"])
def test_crc(name):
    u = get(f"crc/{name}/u")[0]
    np.testing.assert_array_equal(comrs.Crc(name.lower()).checksum(u), get(f"crc/{name}/crc"))


CONV = {
    "conv_rate_half_57": ([0o5, 0o7], 3),
    "conv_rate_half_6474": ([0o15, 0o17], 4),
    "conv_rate_onethird_577": ([0o5, 0o7, 0o7], 3),
    "conv_rate_onefourth_5777": ([0o5, 0o7, 0o7, 0o7], 3),
}


@pytest.mark.parametrize("name", sorted(CONV))
def test_convolutional_codes(name):
    gens, constraint = CONV[name]
    u, x, y, uhat = (get(f"conv/{name}/{p}") for p in ("u", "x", "y", "uhat"))
    code = comrs.ConvolutionalCode(u.shape[1], generators=gens, constraint=constraint)
    tail = code.n - x.shape[1]
    for i in range(len(u)):
        np.testing.assert_array_equal(code.encode(u[i])[: x.shape[1]], x[i])
        np.testing.assert_array_equal(code.decode(np.concatenate([-y[i], np.zeros(tail)])), uhat[i])


def streams(v, k):
    """Sionna's symbol-by-symbol order to the stream-by-stream order of comrs."""
    return np.concatenate([v[0 : 3 * k : 3], v[1 : 3 * k : 3], v[2 : 3 * k : 3], v[3 * k :]])


@pytest.mark.parametrize("k", [40, 112, 168, 432])
def test_lte_turbo_code(k):
    u, x, y, uhat = (get(f"turbo/k{k}/{p}") for p in ("u", "x", "y", "uhat"))
    code = comrs.TurboCode(k, qpp="lte", iterations=10, exact=True)
    for i in range(len(u)):
        np.testing.assert_array_equal(code.encode(u[i]), streams(x[i], k))
        # the same decisions, the same errors
        np.testing.assert_array_equal(code.decode(streams(4.0 * y[i] / 3.0, k)), uhat[i])
    assert META["license"] == "Apache-2.0"


POLAR = ["E45_k30_K41", "E70_k32_K43", "E127_k29_K40", "E1023_k400_K411", "E70_k28_K39"]


@pytest.mark.parametrize("name", POLAR)
def test_nr_polar_uplink(name):
    u, c = get(f"polar/{name}/u"), get(f"polar/{name}/c")
    code = comrs.NrPolar(u.shape[1], c.shape[1], list_size=8)
    for i in range(len(u)):
        np.testing.assert_array_equal(code.encode(u[i]), c[i])
    # and back from the noiseless codeword
    np.testing.assert_array_equal(code.decode(3.0 - 6.0 * c[0]), u[0])


LDPC = sorted(
    {tuple(int(x) for x in k.split(".")[1][1:].split("_n")) for k in V if k.startswith("ldpc.")}
)


@pytest.mark.parametrize("k,n", LDPC)
def test_nr_ldpc(k, n):
    u, c = get(f"ldpc/k{k}_n{n}/u"), get(f"ldpc/k{k}_n{n}/c")
    code = comrs.NrLdpc(k, n)
    np.testing.assert_array_equal(code.encode(u.ravel()).reshape(c.shape), c)
    if k <= 1024:
        np.testing.assert_array_equal(code.decode(4.0 - 8.0 * c[0]), u[0])
