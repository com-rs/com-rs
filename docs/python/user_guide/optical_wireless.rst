Optical wireless links
======================

Light also carries data through the air: LED lights that communicate while they
illuminate a room (visible-light communication) or infrared emitters indoors, and laser
links between buildings (free-space optics). Both use intensity modulation and direct
detection: the transmitted quantity is an optical power, and the channel is a real,
non-negative impulse response for indoor links and a random gain of the intensity for
outdoor links. The gallery page
:doc:`../auto_examples/optical/plot_07_optical_wireless` reproduces the figures of the
three papers cited below.

Indoor channel
--------------

An :class:`~comrs.OwcRoom` is an empty rectangular room whose walls, ceiling and floor
are ideal diffuse reflectors. :meth:`~comrs.OwcRoom.illuminate` carries the light of
Lambertian emitters (positions, axes, mode :func:`~comrs.lambertian_mode` of their
semi-angle, power) to the elements of the surfaces, once, with as many orders of
reflection as partitions given (Barry et al. 1993). An
:class:`~comrs.OwcIllumination` then gives the impulse response at any detector, by
order of reflection:

.. code-block:: python

   import comrs, numpy as np

   room = comrs.OwcRoom([5.0, 5.0, 3.0], {"north": 0.7, "south": 0.7, "east": 0.7, "west": 0.7})
   down, up = [0, 0, -1], [0, 0, 1]
   lit = room.illuminate([2.5, 2.5, 2.5], down, comrs.lambertian_mode(np.radians(70)), 1.0,
                         bin_s=0.01e-9, partitions=[[50, 50, 30]])
   h = lit.impulse_responses([1.0, 1.0, 0.85], up, 1e-4, np.radians(60),
                             concentrator_index=1.5)
   both = h[0] + h[1]                 # line of sight and one reflection
   both.total_w(), both.bandwidth_3db_hz()
   signal, isi = both.bit_split(10e-9, h[0].first_arrival_s())

:meth:`~comrs.OwcImpulseResponse.bit_split` splits the power of on–off keying into the
part received within the bit and the part spilled onto the next ones;
:class:`~comrs.PinFetReceiver` turns them into an SNR with the shot and thermal noise of
a PIN photodiode and its FET preamplifier (Komine & Nakagawa 2004), and
:func:`~comrs.owc_illuminance` gives the illuminance of the same lights. The time bins
must be small against the delay spread: a line of sight spreads over a few hundred
picoseconds, hence 10 ps bins for an SNR at 100 Mb/s. ``validation/tests/barry1993.rs``
and ``komine2004.rs`` check the received powers, bandwidths and SNR against the papers.

Free-space optics
-----------------

An :class:`~comrs.Fso` link multiplies the intensity by a gain drawn once per frame
(the fading lasts milliseconds): a deterministic loss
(:func:`~comrs.fog_attenuation_db_per_km` for haze and fog), the turbulence and the
pointing errors (:class:`~comrs.PointingError`). The turbulence is log-normal for weak
fluctuations, negative exponential at saturation, or gamma–gamma across all regimes,
with parameters predicted from the Rytov variance (:func:`~comrs.rytov_variance`,
:meth:`~comrs.GammaGamma.from_rytov`, Al-Habash, Andrews & Phillips 2001):

.. code-block:: python

   g = comrs.GammaGamma.from_rytov(comrs.rytov_variance(5e-14, 1550e-9, 2000.0))
   link = comrs.Fso(10.0, turbulence=g, pointing=comrs.PointingError(2.0, 0.1, 0.3))
   h = link.gains(10_000, comrs.Rng(1))   # one gain per frame
   g.alpha, g.beta, g.scintillation_index()

References: J. R. Barry et al., *IEEE J. Sel. Areas Commun.* 11(3), 1993; T. Komine and
M. Nakagawa, *IEEE Trans. Consum. Electron.* 50(1), 2004; M. A. Al-Habash, L. C.
Andrews and R. L. Phillips, *Opt. Eng.* 40(8), 2001; M. A. Khalighi and M. Uysal,
*IEEE Commun. Surveys Tuts.* 16(4), 2014.
