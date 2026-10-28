API reference
=============

.. currentmodule:: comrs

Every name is importable from the package: ``import comrs; comrs.awgn_ebn0(...)``.
The formulas, the conventions of OptiCommPy that a function follows and its
deviations are in its page.

One vocabulary runs through the package: ``fs`` is a sampling rate (Hz), ``fc_hz`` a
carrier, ``noise_var`` a noise variance ``E|w|²``, ``ebn0_db`` and ``esn0_db`` the SNRs
per bit and per symbol, MIMO matrices are ``(n_rx, n_tx)``; the tuning parameters are
keyword-only; the functions return NumPy arrays; the closed forms (``theory_*``,
:func:`qfunc`, :func:`db2lin`…) take a scalar (and return a float) or an array-like
(and return an array). The package is typed: ``comrs/_comrs.pyi`` describes the
extension.

Random numbers and conversions
------------------------------

Every function that draws takes a :class:`Rng`; the conversions use the power ratio convention (``10 log10``).

.. autosummary::
   :toctree: generated

   Rng
   db2lin
   lin2db
   dbm2w
   w2dbm
   qfunc
   ber_to_q_db

Sources
-------

.. autosummary::
   :toctree: generated

   bits
   prbs
   symbols
   maxwell_boltzmann

Constellations
--------------

.. autosummary::
   :toctree: generated

   Constellation
   gray_code
   Dpsk

Metrics
-------

Estimates from received and transmitted sequences.

.. autosummary::
   :toctree: generated

   ber
   snr_db
   fast_ber_calc
   evm
   monte_carlo_gmi
   monte_carlo_mi
   entropy
   information_rates
   posterior_rates

Closed forms
------------

Error rates, rates and limits, on scalars or arrays.

.. autosummary::
   :toctree: generated

   theory_ber
   theory_ser
   theory_mi
   theory_gmi
   theory_ber_dpsk
   differential_phase_tail
   theory_papr_ccdf
   theory_rs_wer
   theory_rs_ser
   theory_rs_ber
   biawgn_capacity
   normal_approximation
   shannon_limit_ebn0_db

Filters and sampling
--------------------

.. autosummary::
   :toctree: generated

   pulse_shape
   rrc_spectrum
   fir_filter
   convolve
   upsample
   downsample
   decimate
   resample_fft
   low_pass_fir
   pnorm
   anorm
   freq_shift
   moving_average
   delay_fft

Equalisation and compensation
-----------------------------

.. autosummary::
   :toctree: generated

   cdc
   cd_taps_needed
   cd_fir_taps
   adaptive_equalizer

Carrier recovery
----------------

.. autosummary::
   :toctree: generated

   fourth_power_foe
   viterbi_viterbi
   bps
   ddpll
   derotate
   unwrap

Clock recovery
--------------

.. autosummary::
   :toctree: generated

   gardner_ted
   gardner_ted_nyquist
   interpolate_cubic
   gardner_clock_recovery

Channels
--------

.. autosummary::
   :toctree: generated

   awgn
   awgn_ebn0
   phase_noise
   linear_fiber_channel
   FiberParams
   Edfa
   Ssfm
   Manakov
   ManakovSsfm
   gn_model
   gn_closed_form
   format_moments
   raman_profile
   wdm_grid

Optical devices
---------------

.. autosummary::
   :toctree: generated

   laser
   phase_modulator
   mzm
   iqm
   Photodiode
   balanced_photodiode
   hybrid_2x4
   coherent_receiver

Channel coding
--------------

Codes encode bits into bits and decode LLRs (positive for a 0) into bits, whole blocks
at a time (``encode``, ``decode``); the algebraic codes also decode one word of hard
bits with the number of corrections (``decode_bits``).

.. autosummary::
   :toctree: generated

   Crc
   Scrambler
   Interleaver
   ConvolutionalCode
   TurboCode
   LdpcCode
   PolarCode
   ReedSolomonCode
   BchCode
   Ccdm
   free_distance
   pack_symbols
   unpack_symbols

Standards
---------

Reference chains of standards (``com-systems``).

.. autosummary::
   :toctree: generated

   Dvbs2
   dvbs2_bbframe
   dvbs2_bbframe_data
   dvbs2_decode_plheader

OFDM and PAPR
-------------

.. autosummary::
   :toctree: generated

   Ofdm
   PaprReduction
   papr_db
   ccdf
   clipping_bussgang

Constant-envelope waveforms
---------------------------

Continuous-phase modulation (CPFSK, MSK, GMSK) and constant-envelope OFDM.

.. autosummary::
   :toctree: generated

   Cpm
   msk_precode
   msk_bits
   msk_psd
   CeOfdm

OFDM links
----------

.. autosummary::
   :toctree: generated

   ResourceGrid
   ofdm_estimate
   ofdm_preamble
   schmidl_cox
   ofdm_synchronize
   TdlChannel
   ChannelTaps
   power_delay_profile
   rms_delay_spread
   CdlChannel
   cdl_profile
   cdl_k_factor_db

Amplifiers and predistortion
----------------------------

.. autosummary::
   :toctree: generated

   pa_rapp
   pa_saleh
   pa_memory_polynomial
   rapp_predistort
   MemoryPolynomialDpd

Spectrum
--------

.. autosummary::
   :toctree: generated

   psd
   periodogram
   aclr_db

MIMO
----

Channels :math:`H` are ``(n_rx, n_tx)`` arrays, or ``(n, n_rx, n_tx)`` stacks with one
matrix per received vector; received vectors are ``(n, n_rx)`` arrays.

.. autosummary::
   :toctree: generated

   mimo_channel
   exponential_correlation
   mimo_detect
   mimo_llr
   mimo_estimate
   Ostbc
   mimo_capacity
   water_filling
   eigenmode
   zf_precoder

Chains
------

.. autosummary::
   :toctree: generated

   Chain
   Block
   Native
   var
   Frame
   LinkBlock
   frame_metrics
   blocks

Analysis
--------

.. autosummary::
   :toctree: generated

   Results
   Trace
   compare

.. autosummary::
   :toctree: generated
   :recursive:

   plot

Declarative campaigns
---------------------

.. autosummary::
   :toctree: generated

   run_spec
   verify_spec
   search_spec
   spec_conditions
   spec_blocks
   spec_schema
   spec_check_link

Recordings
----------

.. autosummary::
   :toctree: generated

   read_sigmf
   write_sigmf

Monte Carlo runner
------------------

Python trials on the Rust runner (one thread), with its statistics, stopping rules and
reports.

.. autosummary::
   :toctree: generated
   :recursive:

   sim
