Optical links: WDM and nonlinearity
===================================

A WDM link is a chain of the blocks ``WdmTx`` (the channel under test and its
neighbours), ``Manakov`` (dual-polarisation nonlinear propagation over the spans, with
the EDFAs), ``WdmRx`` (the channel under test at baseband) and the receiver DSP. With
``Source(pols=2)`` the two polarisations carry independent symbols; the blocks written
for one polarisation process each of them, ``AdaptiveEqualizer`` is the 2 × 2
butterfly, and the metrics pool both.

.. code-block:: python

   import comrs
   from comrs import blocks as B

   fiber = {"n_spans": 10, "span_km": 80, "d_ps_nm_km": 16, "gamma_w_km": 1.3}
   link = comrs.Chain(
       tx=B.Source(constellation="16-QAM", n_symbols=8192, symbol_rate=32e9, pols=2),
       wdm=B.WdmTx(n_channels=5, spacing_hz=50e9, sps=8,
                   power_dbm=comrs.var("power_dbm"), rolloff=0.1),
       fiber=B.Manakov(),
       rx=B.WdmRx(sps=2),
       comp=B.Cdc(),
       mf=B.MatchedFilter(rolloff=0.1),
       dec=B.Decimate(),
       cpr=B.Bps(n=32, b=32),
       sync=B.FrameSync(start=2048, length=512),
   )
   results = link.run(grid={"power_dbm": range(-6, 7, 2)},
                      metrics={"report": ["snr_db", "ber", "gmi"], "skip_start": 512,
                               "skip_end": 512},
                      stop={"max_trials": 8}, fiber=fiber)

   # dual-polarisation digital back-propagation instead of dispersion compensation
   dbp = link.replace(comp=B.ManakovDbp(steps_per_span=4))

The propagation
---------------

``Manakov`` solves the Manakov equation by the symmetric split-step method of
OptiCommPy's ``manakovSSF`` (the nonlinear phase integrated by the trapezoidal rule,
iterated to convergence, and steps adapted to a nonlinear phase rotation of 0.02 rad):
on the same field, the two implementations agree to :math:`10^{-13}`
(``tests/test_oracle.py``), and the same WDM link simulated end to end by both gives the
same SNR. :class:`comrs.ManakovSsfm` runs it on arrays.

The GN model
------------

:func:`comrs.gn_model` computes the nonlinear interference of a WDM signal by the
Gaussian-noise model of Poggiolini (the double integral with the coherent accumulation
of the spans), the ASE of the amplifiers, the SNR of a channel and the optimal launch
power; :func:`comrs.gn_closed_form` is Poggiolini's closed form. The model assumes a
Gaussian signal: ``specs/wdm_gn.yaml`` checks that the Manakov simulation of Gaussian
symbols follows it within 0.4 dB from the ASE-limited regime to 4 dB above the optimum.
Specifications can use it as an expected result (``theory: {function: gn_snr_db, …}``).

QAM symbols generate less interference than Gaussian ones. With ``format`` (a
constellation, ``"16-QAM"``, with ``px`` for a shaped source), :func:`comrs.gn_model`
returns the EGN model of Carena et al. (2014): the GN model corrected for the fourth and
sixth moments of the symbols (:func:`comrs.format_moments`), for the self-channel
interference and the cross-channel interference of every channel. In
``specs/wdm_gn.yaml`` it follows the simulated 16-QAM within 0.3 dB, where the GN model
is up to 1.7 dB off. ``validation/tests/carena2014.rs`` checks it against Fig. 1(a) of the
paper and against a Monte Carlo evaluation of the first-order perturbation.
Specifications use it with ``theory: {function: gn_snr_db, format: 16-QAM, …}``.

Raman amplification
-------------------

A ``raman`` mapping in the fibre parameters pumps every span: any number of pumps,
co-propagating (launched with the signals) or counter-propagating (launched at the end
of the span), given by ``wavelength_nm`` or ``frequency_thz`` and ``power_mw``, with
their own attenuation ``alpha_db_km`` if needed. The channels also exchange power
among themselves (inter-channel stimulated Raman scattering, ``isrs: true`` by
default), which tilts a wide WDM spectrum towards low frequencies even without pumps.

.. code-block:: python

   fiber = {
       "n_spans": 3, "span_km": 100, "nf_db": 5,
       "raman": {"pumps": [
           {"wavelength_nm": 1425, "power_mw": 300, "direction": "counter"},
           {"wavelength_nm": 1455, "power_mw": 250, "direction": "counter"},
           {"frequency_thz": 205.0, "power_mw": 100, "direction": "co"},
       ]},
   }
   results = link.run(grid={"power_dbm": range(-10, 3, 2)},
                      metrics={"report": ["snr_db"]}, fiber=fiber)

   # the power profiles of one span (channels, then pumps) and the gains
   p = comrs.raman_profile(comrs.FiberParams(span_km=100), comrs.wdm_grid(40, 100e9),
                           0.0, fiber["raman"])
   p["on_off_gain_db"], p["net_gain_db"], p["ase_psd"]

:func:`comrs.raman_profile` solves the coupled power equations of stimulated Raman
scattering (Bromage 2004) along the span: RK4 integration, forward for the waves that
travel with the signals and backward for the counter-propagating pumps, repeated until
the profiles converge. The coefficients are those of the measured gain spectrum of
standard single-mode fibre (``gain`` replaces it), scaled with the pump frequency, and
conserve the number of photons. The spontaneous Raman scattering of the pumps adds a
noise whose thermal factor grows near the pumps.

``Ssfm`` and ``Manakov`` apply the profile of every span to the field as a gain that
depends on the position and on the frequency. The profile is solved for slices of the
spectrum of the field (``slice_hz``, 25 GHz) at the input of the span. After the span
they add the spontaneous noise; the amplifier (``amplification``) then restores every
frequency to its launch power, as an EDFA with an ideal gain-flattening filter that adds
its ASE only where the Raman gain did not compensate the loss. :func:`comrs.gn_model`
takes the same ``raman`` mapping. It integrates the power profiles of the channels in
the link function of the GN model, the generalised GN model of Semrau, Killey & Bayvel
(2019), and counts the noise of the hybrid amplifiers. ``specs/raman_wdm.yaml`` checks
the Manakov simulation against it, without and with a counter-pump, within 0.21 dB.
``tests/test_gnpy_oracle.py`` checks the profiles, the noise and the NLI against GNPy.
The digital back-propagation keeps the profile of lumped amplifiers.

Polarisation demultiplexing
---------------------------

``Pmd`` rotates the polarisations at random (a uniform SU(2) rotation) and adds
polarisation-mode dispersion of a mean differential group delay ``mean_dgd_ps`` (the
coarse-step model of Marcuse, Menyuk and Wai), a new realisation every trial;
``theta_deg`` gives a fixed real rotation instead. ``AdaptiveEqualizer`` with two
polarisations is the 2 × 2 butterfly of OptiCommPy's ``mimoAdaptEqualizer`` (equal to it
to :math:`10^{-6}` on the same input), with one step size per stage of the schedule.
Blind criteria such as the CMA admit a singular solution, both outputs on the same
polarisation; ``kikuchi=n`` is Kikuchi's two-stage initialisation (output x alone for
``n`` symbols, then output y as its unitary complement), which removes it. The block
records ``<label>.correlation`` (of the two outputs) and ``<label>.singular`` (above
0.5); ``FrameSync`` restores the order of the outputs from the known symbols.

.. code-block:: python

   link = comrs.Chain(
       tx=B.Source(constellation="16-QAM", n_symbols=16384, symbol_rate=32e9, pols=2),
       shape=B.PulseShape(sps=2, rolloff=0.2),
       pmd=B.Pmd(mean_dgd_ps=20.0),
       noise=B.Awgn(snr_db=20.0),
       mf=B.MatchedFilter(rolloff=0.2),
       eq=B.AdaptiveEqualizer(algorithm=[["cma", 6000], ["rde", 10**9]],
                              mu=[2e-3, 1e-4], kikuchi=3000),
       sync=B.FrameSync(start=12384, length=512, symmetry=0),
   )

``specs/polmux_cma.yaml`` and ``specs/polmux_cma_45deg.yaml`` check it: from the
identity, half of the trials of a 45° rotation end singular; with Kikuchi's
initialisation none does, and the SNR after the equaliser is within 1 dB of the
channel's under rotation and DGD up to 20 ps.
