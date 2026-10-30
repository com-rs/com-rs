Wireless and coded links
========================

The blocks of :mod:`comrs.blocks` include multi-antenna transmission, fading channels,
OFDM with pilots and channel estimation, MIMO detection, channel codes and soft
demapping: a whole wireless link is a :class:`~comrs.Chain` (or a specification) that
runs on the parallel Rust runner.

Antennas and fading
-------------------

``MimoTx`` spreads the symbols over the transmit antennas, by spatial multiplexing
(symbol ``i`` on antenna ``i mod n_t``, the power divided among the antennas) or by
Alamouti's code. ``Rayleigh`` (flat Rayleigh or Rician fading, constant over coherence
blocks, with exponential antenna correlation) and ``Tdl`` (the tapped delay lines EPA,
EVA, ETU of 3GPP TS 36.101 and the ITU profiles, with Clarke/Jakes Doppler) act on
every antenna and record the channel for a genie-aided receiver. ``MimoRx`` detects a
single-carrier link with the known channel (ZF, MMSE, MMSE-SIC, ML, sphere decoder,
K-best):

.. code-block:: python

   import comrs
   from comrs import blocks as B

   alamouti = comrs.Chain(
       B.Source(constellation="QPSK", n_symbols=8192),
       B.MimoTx(scheme="alamouti"),
       B.Rayleigh(n_rx=2, coherence_samples=2),
       B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")),
       B.MimoRx(),
   )

The fading blocks have unit mean gain and record the power that enters them: the noise
blocks that follow take it as their reference, so that ``ebn0_db`` and ``snr_db`` are
averages over the fading, as in the closed forms of fading channels. ``specs/
mimo_diversity.yaml`` checks the chain above against the BER of maximal-ratio combining.

OFDM
----

``OfdmTx`` maps the symbols of every antenna on resource grids (block, comb or lattice
pilots, dealt to the antennas in turn so that they are orthogonal) and modulates them;
``OfdmRx`` demodulates, estimates the channel (``perfect``, ``ls`` with linear
interpolation, or ``lmmse``) and detects every data resource element. The modulator
scales the samples so that ``Awgn(snr_db=…)`` sets the SNR per resource element:

.. code-block:: python

   ofdm = comrs.Chain(
       B.Source(constellation="16-QAM", n_symbols=4800),
       B.MimoTx(n_tx=2),
       B.OfdmTx(n_fft=256, n_subcarriers=200, n_cp=18),
       B.Tdl(profile="EPA", doppler_hz=5, n_rx=2),
       B.Awgn(snr_db=comrs.var("snr_db")),
       B.OfdmRx(estimation="lmmse", detector="mmse"),
   )

Channel codes
-------------

``FecEncode`` draws the information bits of ``blocks`` codewords and encodes them
(``conv``, ``turbo``, ``ldpc`` or ``polar``, optionally through a random bit
interleaver); ``Map`` maps the coded bits on a constellation; the receiver ends with
``Demap`` (exact or max-log LLRs with the noise variance of every symbol, which the
OFDM and MIMO receivers provide) and ``FecDecode``. The metrics ``coded_ber`` and
``bler`` compare the decoded bits with the transmitted ones, ``ber`` stays the error
rate of the coded bits:

.. code-block:: python

   code = {"type": "ldpc", "n": 1024, "k": 512}
   bicm = comrs.Chain(
       B.FecEncode(code=code, blocks=4, interleave=True),
       B.Map(constellation="16-QAM"),
       B.Rayleigh(coherence_samples=1),
       B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")),
       B.MimoRx(),
       B.Demap(),
       B.FecDecode(code=code, interleave=True),
   )
   bicm.run(grid={"ebn0_db": [6, 8, 10]}, metrics=["ber", "coded_ber", "bler"],
            stop={"min_errors": {"bler": 100}, "max_trials": 500})

After ``FecEncode``, ``AwgnEbn0`` counts the energy per information bit.

Probabilistic shaping
---------------------

``Source(mb_lambda=…)`` draws the points with the Maxwell-Boltzmann probabilities
:math:`p(x) \propto e^{-\lambda |x|^2}` on the unit-energy uniform constellation, and the
frame carries them: the metrics ``mi``, ``gmi`` and ``ngmi`` use them (the GMI of
bit-metric decoding :math:`H(X) - \sum_k H(B_k \mid Y)`), ``entropy`` reports
:math:`H(X)`, ``Demap`` computes the LLRs with the priors and ``AwgnEbn0`` counts the
energy per bit of information. :func:`comrs.theory_mi` and :func:`comrs.theory_gmi`
take the same priors (``px``), as :func:`comrs.monte_carlo_gmi` and
:func:`comrs.information_rates`.

``PasEncode`` and ``PasDecode`` are probabilistic amplitude shaping of a square QAM
(Böcherer, Steiner & Schulte 2015): per dimension, a constant-composition distribution
matcher (:class:`comrs.Ccdm`, Schulte & Böcherer 2016) shapes the amplitudes, a
systematic LDPC code protects their bits, and its parity bits become the signs:

.. code-block:: python

   p = {"constellation": "64-QAM", "mb_lambda": 2.55, "n": 1536}
   pas = comrs.Chain(
       B.PasEncode(**p, blocks=4),
       B.Awgn(snr_db=comrs.var("snr_db")),
       B.PasDecode(**p),
   )
   pas.run(grid={"snr_db": [10, 10.5, 11]}, metrics=["coded_ber", "bler", "gmi", "entropy"])

``specs/shaping_awgn.yaml`` checks the rates of shaped 64-QAM against the closed forms,
``specs/pas_64qam.yaml`` runs PAS at 3 bits per QAM symbol.

Satellite: DVB-S2
-----------------

:class:`comrs.Dvbs2` is the physical layer of DVB-S2 (ETSI EN 302 307-1). It covers the
28 MODCODs of Table 12, normal (64 800 bits) and short (16 200 bits) frames, with or
without pilots. A BBFRAME of ``k`` bits (:func:`comrs.dvbs2_bbframe`: BBHEADER with its
CRC-8, data field, PRBS randomisation) is encoded by the BCH and LDPC codes of the
standard. The bits are then interleaved, mapped on QPSK, 8PSK, 16APSK or 32APSK
(Figures 9 to 12, ring ratios of Tables 9 and 10), and framed into a PLFRAME:
π/2-BPSK header, pilots, Gold scrambling. The receiver descrambles, computes the
exact (or max-log) LLRs, deinterleaves and decodes with the layered sum-product
algorithm (50 iterations); :func:`comrs.dvbs2_decode_plheader` recovers the MODCOD of a
header by maximum likelihood.

.. code-block:: python

   import comrs
   from comrs import blocks as B

   link = comrs.Chain(
       B.Dvbs2Tx(modulation="16APSK", rate="3/4", frame="short"),
       B.Awgn(snr_db=comrs.var("esn0_db"), reference="unit"),   # Es/N0
       B.Dvbs2Rx(modulation="16APSK", rate="3/4", frame="short"),
   )
   res = link.run(grid={"esn0_db": [10.2, 10.4, 10.6]},
                  metrics={"report": ["bler", "coded_ber"], "scale": "none"})

Every symbol of a PLFRAME has unit energy, so that ``reference: unit`` sets the Es/N0
of Table 13. ``specs/validation/dvbs2_table13.yaml`` checks the waterfall of all 28
MODCODs against that table. The LDPC tables, the BCH polynomials, the constellations,
the interleaver and the framing follow the text of the standard; the LDPC tables are
also identical to those of AFF3CT.
