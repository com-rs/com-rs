Comparisons with other libraries
================================

com-rs gathers blocks that usually live in separate tools: channel codes, wireless
waveforms and channels, fibre propagation, optical devices, receiver DSP, recordings.
Each domain has its reference implementations, and com-rs is compared with them on the
same inputs. The outputs of the tools are recorded once (``tests/oracle/``,
``validation/aff3ct/``), with the version that produced them, so that the tests run
without the tools; none of their code is copied (see ``THIRD_PARTY.md``). The
:doc:`maturity <maturity>` page lists, module by module, the tests named below.

Agreeing with a tool is evidence, not proof: two implementations of a standard can
share a misreading. Where a tool and com-rs differ, the difference is traced to the
standard or the paper, and documented.

.. list-table::
   :header-rows: 1
   :widths: 18 34 22 26

   * - Library
     - What is compared
     - Agreement
     - Differences found
   * - `OptiCommPy <https://github.com/edsonportosilva/OptiCommPy>`_ 0.11
       (GPL-3.0)
     - Gray mapping and demapping, closed-form error rates, pulse shaping and
       filters, resampling, linear fibre, split-step Fourier and Manakov propagation,
       WDM grid and SNR, Mach–Zehnder and IQ modulators, photodiodes, optical hybrid
       and coherent receiver, adaptive equalisers (CMA, RDE, NLMS, DD-LMS, 2×2),
       carrier recovery (fourth power, Viterbi–Viterbi, blind phase search, DD-PLL),
       Gardner clock recovery, BER, SNR, EVM, MI and GMI
       (``tests/test_oracle.py``)
     - to machine precision for the deterministic blocks (10⁻⁸ relative on the
       propagated fields); statistically for the Monte Carlo estimators
     - the BER of BPSK from ``theoryBER`` is twice the exact value; its ``edc``
       limits the SNR of a noiseless link to 23 dB; its ``edfa`` re-seeds its
       generator on every call
   * - `GNPy <https://github.com/Telecominfraproject/oopt-gnpy>`_ 3.0
       (BSD-3-Clause)
     - Raman power profiles with co- and counter-propagating pumps, their spontaneous
       noise, the generalised GN model with power profiles
       (``tests/test_gnpy_oracle.py``)
     - 0.02 dB on the NLI; 0.004 dB on the profiles at a 5 m step
     - GNPy integrates by Euler's method (first order in its step); at its default
       tolerances it underestimates the NLI of counter-pumped spans by about 1 dB
   * - `AFF3CT <https://github.com/aff3ct/aff3ct>`_ v4.7.1 (MIT)
     - bit and frame error rates of convolutional (Viterbi), LTE turbo (log-MAP and
       max-log-MAP), LDPC (sum-product, min-sum variants, flooding and layered),
       polar (SC, CA-SCL), Reed–Solomon and BCH codes, the short LDPC code of DVB-S2
       (``validation/tests/aff3ct.rs``)
     - within the confidence intervals of both simulations
     - none on the decoders; the polar construction is that of com-rs on both sides
   * - `Sionna <https://github.com/NVlabs/sionna>`_ (Apache-2.0), its test vectors
     - the CRCs of 3GPP (CRC6 to CRC24C); convolutional codes of rates 1/2, 1/3, 1/4,
       encoded and Viterbi-decoded; the LTE turbo code at K = 40, 112, 168, 432,
       encoded and decoded at 0 dB; the 5G LDPC code with rate matching (28 sizes, 64
       to 8448 bits, both base graphs) and the 5G polar code of the uplink control
       information (``tests/test_sionna_vectors.py``)
     - bit for bit, the decisions included (errors and all)
     - none; the vectors brought the full QPP table of LTE (188 block sizes) to
       ``TurboCode.lte``; its tables of the LDPC base graphs and of the polar
       sequence are identical to those extracted from TS 38.212
   * - `py3gpp <https://github.com/catkira/py3gpp>`_ 0.6 (AGPL-3.0)
     - 5G NR: PSS, SSS, PBCH DMRS and their positions, the PDSCH DMRS of 42
       configurations, the OFDM modulation of a subframe
       (``tests/test_nr_oracle.py``)
     - bit for bit for sequences and positions; 10⁻¹⁴ for the waveforms
     - py3gpp lengthens the cyclic prefix of every slot instead of every half
       subframe (its subframes no longer last 1 ms from 60 kHz); its DMRS table adds
       a symbol to 8- and 9-symbol allocations without additional position
   * - `sigmf-python <https://github.com/sigmf/sigmf-python>`_ 1.x (LGPL-3.0)
     - SigMF recordings both ways: the 28 sample formats, several channels, captures,
       annotations, archives (``tests/test_sigmf.py``)
     - identical samples and metadata; the recordings of com-rs pass its validation
     - sigmf-python scales 32-bit integers in float32 (an error of 3·10⁻⁸)

Other open implementations of the standards can serve the same way, through their
outputs: srsRAN and OpenAirInterface (LTE, 5G NR), the ``gr-dtv`` module of GNU Radio
(DVB-T, DVB-T2) and ``gr-ieee802-11`` (Wi-Fi). Their licences (GPL, AGPL and others)
forbid copying their code into com-rs, which is MIT/Apache-2.0, but not comparing with
their results.
