"""comrs against py3gpp, the oracle of the 5G NR physical layer.

py3gpp (AGPL-3.0, <https://github.com/catkira/py3gpp>) reimplements in Python the 5G
functions of MATLAB from TS 38.211. Its outputs are recorded fixtures
(``tests/oracle/nr_*.json``, see ``tests/oracle.py``): py3gpp is imported only inside
the functions that compute them, when recording::

    COMRS_RECORD_ORACLE=1 python -m pytest tests/test_nr_oracle.py

Compared: the PSS and the SSS of seven cells, the PBCH DMRS of three SSB indices, the
indices of the PBCH and of its DMRS in the SSB (TS 38.211, Sec. 7.4.2, 7.4.3, 7.4.1.4);
the DMRS of the PDSCH of mapping type A over a slot of 14 symbols, configuration types
1 and 2, ``dmrs-TypeA-Position`` 2 and 3, the four additional positions (Sec.
7.4.1.1); the OFDM modulation of a subframe at 15 and 30 kHz (Sec. 5.3.1).

Documented difference: py3gpp lengthens the cyclic prefix of the first symbol of every
slot; TS 38.211 lengthens that of the first symbol of every half subframe
(``l = 0`` and ``l = 7·2^mu`` in the subframe), so that a subframe lasts 1 ms. The two
agree up to 30 kHz (at most one slot per half subframe) and differ from 60 kHz: the waveforms are compared at 15 and 30 kHz, and the
prefixes of comrs at 60 and 120 kHz are checked against the 1 ms of the subframe.
"""

import numpy as np
from oracle import oracle

import comrs

TOOL = "py3gpp"
CELLS = [0, 1, 2, 3, 417, 1000, 1007]


def _ssb():
    from py3gpp.nrPBCHDMRS import nrPBCHDMRS
    from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices
    from py3gpp.nrPBCHIndices import nrPBCHIndices
    from py3gpp.nrPSS import nrPSS
    from py3gpp.nrSSS import nrSSS

    return {
        str(c): {
            "pss": np.asarray(nrPSS(c), float),
            "sss": np.asarray(nrSSS(c), float),
            "pbch_dmrs": {str(i): np.asarray(nrPBCHDMRS(c, i)) for i in (0, 3, 7)},
            "pbch_dmrs_indices": np.asarray(nrPBCHDMRSIndices(c), np.int64),
            # nrPBCHIndices takes the cell identity (its argument is named ibar)
            "pbch_indices": np.asarray(nrPBCHIndices(c), np.int64),
        }
        for c in CELLS
    }


def test_ssb_against_py3gpp():
    ref = oracle("nr_ssb", _ssb, tool=TOOL)
    for c in CELLS:
        r = ref[str(c)]
        np.testing.assert_array_equal(comrs.nr_pss(c % 3), r["pss"])
        np.testing.assert_array_equal(comrs.nr_sss(c // 3, c % 3), r["sss"])
        for i in (0, 3, 7):
            np.testing.assert_allclose(comrs.nr_pbch_dmrs(c, i), r["pbch_dmrs"][str(i)], atol=1e-15)
        # the (k, l) pairs as the linear indices l · 240 + k of MATLAB (minus one)
        for kind in ("pbch_dmrs", "pbch"):
            kl = comrs.nr_ssb_indices(c, kind)
            np.testing.assert_array_equal(kl[:, 1] * 240 + kl[:, 0], r[f"{kind}_indices"])


DMRS_CASES = [
    (ctype, pos, add, nid, slot)
    for ctype in (1, 2)
    for pos in (2, 3)
    for add in range(4)
    if not (pos == 3 and add == 3)
    for nid, slot in [(0, 0), (417, 5), (1007, 19)]
]
N_RB = 24


def _pdsch_dmrs():
    from py3gpp.configs.nrCarrierConfig import nrCarrierConfig
    from py3gpp.configs.nrPDSCHConfig import nrPDSCHConfig
    from py3gpp.nrPDSCHDMRS import nrPDSCHDMRS
    from py3gpp.nrPDSCHDMRSIndices import nrPDSCHDMRSIndices

    out = {}
    for ctype, pos, add, nid, slot in DMRS_CASES:
        car = nrCarrierConfig(NCellID=nid, NSizeGrid=N_RB, NSlot=slot, SubcarrierSpacing=30)
        cfg = nrPDSCHConfig()
        cfg.NSizeBWP, cfg.NStartBWP, cfg.PRBSet = N_RB, 0, list(range(N_RB))
        cfg.SymbolAllocation = [0, 14]
        cfg.DMRS.DMRSConfigurationType = ctype
        cfg.DMRS.DMRSTypeAPosition = pos
        cfg.DMRS.DMRSAdditionalPosition = add
        cfg.DMRS.NIDNSCID = nid
        out[f"{ctype}_{pos}_{add}_{nid}_{slot}"] = {
            "values": np.asarray(nrPDSCHDMRS(cfg, car)),
            "indices": np.asarray(nrPDSCHDMRSIndices(car, cfg), np.int64),
        }
    return out


def test_pdsch_dmrs_against_py3gpp():
    ref = oracle("nr_pdsch_dmrs", _pdsch_dmrs, tool=TOOL)
    for ctype, pos, add, nid, slot in DMRS_CASES:
        r = ref[f"{ctype}_{pos}_{add}_{nid}_{slot}"]
        d = comrs.nr_pdsch_dmrs(
            N_RB,
            config_type=ctype,
            type_a_position=pos,
            additional_position=add,
            n_id=nid,
            slot=slot,
        )
        np.testing.assert_allclose(d["values"].T.ravel(), r["values"], atol=1e-15)
        kl = d["indices"]
        np.testing.assert_array_equal(kl[:, 1] * 12 * N_RB + kl[:, 0], r["indices"])


OFDM_CASES = [(0, 6, 128), (1, 8, 256)]


def _grid(mu, n_rb):
    rng = np.random.default_rng(10 + mu)
    k, n = 12 * n_rb, 14 * 2**mu
    return (
        (2 * rng.integers(0, 2, (k, n)) - 1) + 1j * (2 * rng.integers(0, 2, (k, n)) - 1)
    ) / np.sqrt(2)


def _ofdm():
    from py3gpp.configs.nrCarrierConfig import nrCarrierConfig
    from py3gpp.nrOFDMModulate import nrOFDMModulate

    out = {}
    for mu, n_rb, n_fft in OFDM_CASES:
        scs = 15 * 2**mu
        car = nrCarrierConfig(NSizeGrid=n_rb, SubcarrierSpacing=scs)
        w, info = nrOFDMModulate(
            carrier=car,
            grid=_grid(mu, n_rb),
            scs=scs,
            initialNSlot=0,
            SampleRate=n_fft * scs * 1000,
        )
        out[str(mu)] = {
            "waveform": np.asarray(w),
            "cp": np.asarray(info["CyclicPrefixLengths"], np.int64),
        }
    return out


def test_ofdm_modulation_against_py3gpp():
    ref = oracle("nr_ofdm", _ofdm, tool=TOOL)
    for mu, n_rb, n_fft in OFDM_CASES:
        r = ref[str(mu)]
        np.testing.assert_array_equal(comrs.nr_numerology(mu, n_fft=n_fft)["cp_lengths"], r["cp"])
        # py3gpp's IFFT is scaled by 1/N_FFT, that of comrs by 1/sqrt(K)
        ours = comrs.nr_ofdm_modulate(_grid(mu, n_rb), mu, n_fft) * np.sqrt(12 * n_rb) / n_fft
        np.testing.assert_allclose(ours, r["waveform"], atol=1e-14)


def test_the_subframe_lasts_one_millisecond():
    for mu in range(5):
        for n_fft in (256, 1024, 4096):
            info = comrs.nr_numerology(mu, n_fft=n_fft)
            assert np.sum(info["cp_lengths"] + n_fft) == round(1e-3 * info["sample_rate"])
