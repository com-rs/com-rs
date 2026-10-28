"""comrs against GNPy, the oracle of the Raman amplification.

GNPy (Telecom Infra Project, BSD-3-Clause) solves the same coupled power equations of
stimulated Raman scattering (explicit Euler steps, forward-backward sweeps for the
counter-propagating pumps), the same spontaneous noise of the pumps and the generalised
GN model with power profiles (spectrally separated: SPM and XPM). Its Raman coefficient
matrix is replaced by that of comrs, written again here from its definition: GNPy scales
the effective area with the frequency, and its pumps lose the power that the signals
gain where comrs conserves the photons (the pump loses ``f_p/f_s`` times that power).
The documented differences are tested as such:

- GNPy integrates by Euler's method: its profiles converge to those of comrs (RK4) at
  first order in its step;
- the NLI of GNPy converges to that of comrs as its frequency grid is refined (its
  ``dispersion_tolerance`` and ``phase_shift_tolerance``); at its default tolerances it
  underestimates the NLI of counter-pumped spans by about 1 dB;
- comrs, as GNPy, takes the pumps alone as the sources of spontaneous scattering.

The outputs of GNPy are recorded fixtures (``tests/oracle/<name>.json``, see
``tests/oracle.py``): every ``oracle(...)`` call below names the fixture and the
function that computes it, and GNPy is imported only inside that function, when
recording. The inputs (channel grids, pumps, fibre) are fixed here. The tests run from
the files alone and skip when a file is missing. To re-record, with GNPy installed
(``pip install --group oracle``)::

    COMRS_RECORD_ORACLE=1 python -m pytest tests/test_gnpy_oracle.py

Re-recording is a deliberate act: the commit that changes the fixtures says why and
names the version of GNPy (the ``version`` field of the files).
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from oracle import oracle

import comrs

TOOL = "GNPy"
C0 = 299_792_458.0
FC = 193.1e12


def coupling(freqs, aeff=80e-12):
    """The Raman coefficients of comrs (1/(W m)): the gain efficiency of the lower wave,
    the loss of the higher one scaled by the ratio of the frequencies."""
    from gnpy.core.parameters import DEFAULT_RAMAN_COEFFICIENT as d

    def eff(fs, fp):
        g = np.interp(fp - fs, d["frequency_offset"], d["gamma_raman"], right=0.0)
        return g * fp / d["reference_frequency"] / aeff

    n = len(freqs)
    cr = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            if freqs[j] > freqs[k]:
                cr[k, j] = eff(freqs[k], freqs[j])
            elif freqs[j] < freqs[k]:
                cr[k, j] = -(freqs[k] / freqs[j]) * eff(freqs[j], freqs[k])
    return cr


def pump_frequency(p):
    return p["frequency_thz"] * 1e12 if "frequency_thz" in p else C0 / (p["wavelength_nm"] * 1e-9)


def gnpy_pumps(pumps):
    return [
        {
            "power": p["power_mw"] * 1e-3,
            "frequency": pump_frequency(p),
            "propagation_direction": "coprop" if p["direction"] == "co" else "counterprop",
        }
        for p in pumps
    ]


class Span:
    """The fibre GNPy's Raman solver reads: the coupling of comrs for the waves of
    ``freqs``, a uniform attenuation, the pumps."""

    def __init__(self, length_km, freqs, pumps):
        self.params = NS(length=length_km * 1e3)
        self.lumped_losses = np.array([])
        self.z_lumped_losses = np.array([])
        self.temperature = 300.0
        self.raman_pumps = [
            NS(
                power=q["power"],
                frequency=q["frequency"],
                propagation_direction=q["propagation_direction"],
            )
            for q in gnpy_pumps(pumps)
        ]
        self._cr = coupling(freqs)
        self._index = {f: i for i, f in enumerate(freqs)}

    def cr(self, f):
        i = [self._index[x] for x in f]
        return self._cr[np.ix_(i, i)]

    def alpha(self, f):
        return np.full(len(f), 0.2 / (10 * np.log10(np.e)) / 1e3)


def test_power_profiles_and_spontaneous_noise_of_multi_pump_spans():
    """A co-pump and two counter-pumps on 40 channels over 80 km."""
    length, nch = 80.0, 40
    rel = (np.arange(nch) - nch / 2) * 100e9
    pumps = [
        {"frequency_thz": 205.0, "power_mw": 150, "direction": "co"},
        {"wavelength_nm": 1425, "power_mw": 300, "direction": "counter"},
        {"wavelength_nm": 1455, "power_mw": 250, "direction": "counter"},
    ]
    fiber = comrs.FiberParams(span_km=length, fc_hz=FC)
    for p_dbm, step_m in [(0.0, 20.0), (0.0, 5.0), (-30.0, 5.0)]:
        ours = comrs.raman_profile(fiber, list(rel), p_dbm, {"pumps": pumps, "step_km": 0.05})
        freqs = ours["freqs_hz"]

        def compute(p_dbm=p_dbm, step_m=step_m, freqs=freqs):
            from gnpy.core.parameters import SimParams
            from gnpy.core.science_utils import RamanSolver

            span = Span(length, freqs, pumps)
            SimParams.set_params(
                {
                    "raman_params": {
                        "flag": True,
                        "method": "numerical",
                        "result_spatial_resolution": 1e3,
                        "solver_spatial_resolution": step_m,
                    }
                }
            )
            si = NS(
                pch=np.full(nch, 1e-3 * 10 ** (p_dbm / 10)),
                frequency=freqs[:nch],
                baud_rate=np.ones(nch),
                number_of_channels=nch,
            )
            srs = RamanSolver.calculate_stimulated_raman_scattering(si, span)
            out = {
                "net_gain_db": 10 * np.log10(srs.loss_profile[:nch, -1]),
                "pumps_dbm": 10 * np.log10(srs.power_profile[nch:][:, [0, -1]]) + 30,
            }
            if p_dbm < -20:
                out["ase_psd"] = (
                    RamanSolver.calculate_spontaneous_raman_scattering(si, srs, span)
                    * srs.loss_profile[:nch, -1]
                )
            return out

        ref = oracle(f"raman_profile_{p_dbm}dbm_{step_m}m", compute, tool=TOOL)
        err = np.abs(ours["net_gain_db"] - ref["net_gain_db"]).max()
        # Euler converges at first order: 0.016 dB at 20 m, 0.004 dB at 5 m
        assert err < 1e-3 * step_m, (step_m, err)
        pumps_ours = ours["power_dbm"][nch:][:, [0, -1]]
        assert np.abs(pumps_ours - ref["pumps_dbm"]).max() < 1e-3 * step_m
        if p_dbm < -20:
            np.testing.assert_allclose(ours["ase_psd"], ref["ase_psd"], rtol=2e-3)


@pytest.mark.parametrize("with_pumps", [False, True])
def test_generalized_gn_model_with_counter_pumps(with_pumps):
    """The NLI of the centre of three channels, GNPy with a fine frequency grid."""
    nch, rs, spacing, roll, length = 3, 32e9, 50e9, 0.02, 80.0
    rel = (np.arange(nch) - nch // 2) * spacing
    pumps = [
        {"wavelength_nm": 1425, "power_mw": 300, "direction": "counter"},
        {"wavelength_nm": 1455, "power_mw": 250, "direction": "counter"},
    ]
    raman = {"pumps": pumps if with_pumps else []}
    fiber = comrs.FiberParams(span_km=length, d_ps_nm_km=16.7, gamma_w_km=1.3, n_spans=1, fc_hz=FC)
    ours = (
        comrs.gn_model(fiber, list(rel), 0.0, rs, rolloff=roll, coherent=False, raman=raman)[
            "nli_psd"
        ]
        * rs
    )

    def compute():
        from gnpy.core.elements import Fiber, RamanFiber
        from gnpy.core.info import create_arbitrary_spectral_information
        from gnpy.core.parameters import SimParams
        from gnpy.core.science_utils import NliSolver, RamanSolver

        SimParams.set_params(
            {
                "raman_params": {
                    "flag": True,
                    "method": "numerical",
                    "result_spatial_resolution": 50,
                    "solver_spatial_resolution": 5,
                },
                "nli_params": {
                    "method": "ggn_spectrally_separated",
                    "computed_channels": [nch // 2 + 1],
                    "dispersion_tolerance": 0.5,
                    "phase_shift_tolerance": 0.02,
                },
            }
        )
        params = {
            "length": length,
            "length_units": "km",
            "loss_coef": 0.2,
            "dispersion": 16.7e-6,
            "gamma": 1.3e-3,
            "pmd_coef": 0.0,
            "con_in": 0,
            "con_out": 0,
            "att_in": 0,
            "ref_frequency": FC,
        }
        si = create_arbitrary_spectral_information(
            FC + rel,
            slot_width=spacing,
            pch=1e-3,
            baud_rate=rs,
            roll_off=roll,
            tx_osnr=40,
            tx_power=1e-3,
        )
        if with_pumps:
            f = RamanFiber(
                uid="span",
                params=params,
                operational={"temperature": 300, "raman_pumps": gnpy_pumps(pumps)},
            )
        else:
            f = Fiber(uid="span", params=params)
        f.cr = coupling
        srs = RamanSolver.calculate_stimulated_raman_scattering(si, f)
        return float(NliSolver.compute_nli(si, srs, f)[nch // 2])

    theirs = oracle(f"ggn_nli_{'counter_pumps' if with_pumps else 'no_pumps'}", compute, tool=TOOL)
    assert abs(10 * np.log10(ours / theirs)) < 0.03
