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

The names are grouped by theme, in the order of a link: the foundations, coding,
waveforms, MIMO, the channels (wireless and optical), the hardware, the receiver,
the metrics, the standards and the simulation. The same themes order the gallery, the
maturity table and the comparisons with the literature.

Foundations
-----------

Random numbers, decibels, sources, constellations and the closed forms of their error
and information rates.

Random numbers and conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every function that draws takes a :class:`Rng`; the conversions use the power ratio
convention (``10 log10``).

.. autosummary::
   :toctree: generated

   Rng
   db2lin
   lin2db
   dbm2w
   w2dbm
   qfunc
   bessel_j0
   entropy

Sources
~~~~~~~

.. autosummary::
   :toctree: generated

   bits
   prbs
   symbols
   maxwell_boltzmann

Constellations
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Constellation
   gray_code
   Dpsk

Error rates in closed form
~~~~~~~~~~~~~~~~~~~~~~~~~~

On scalars or arrays.

.. autosummary::
   :toctree: generated

   theory_ber
   theory_ser
   theory_ber_gray
   theory_ber_dpsk
   differential_phase_tail
   theory_phase_pdf
   theory_ber_imperfect_csi
   theory_ber_dpsk_jakes

Information rates and capacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mutual information of constellations over AWGN or Rayleigh fading, the capacities
and cutoff rates of coded modulation and BICM.

.. autosummary::
   :toctree: generated

   theory_mi
   theory_gmi
   cm_capacity
   bicm_capacity
   cm_cutoff_rate
   bicm_cutoff_rate

Coding and shaping
------------------

Codes encode bits into bits and decode LLRs (positive for a 0) into bits, whole blocks
at a time (``encode``, ``decode``); the algebraic codes also decode one word of hard
decisions (``decode_bits``).

Codes
~~~~~

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
   NrLdpc
   NrPolar
   free_distance
   pack_symbols
   unpack_symbols

Distribution matching
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Ccdm

Limits and bounds
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   shannon_limit_ebn0_db
   biawgn_capacity
   normal_approximation
   theory_rs_wer
   theory_rs_ser
   theory_rs_ber

Waveforms
---------

OFDM
~~~~

The modulator, the resource grid and its pilots, the channel estimation, the
synchronisation and the gain correction of OFDM.

.. autosummary::
   :toctree: generated

   Ofdm
   ResourceGrid
   ofdm_estimate
   ofdm_preamble
   schmidl_cox
   ofdm_synchronize
   gain_limited_correction
   sdr_gain_limited
   sdr_optimum_correction

PAPR
~~~~

.. autosummary::
   :toctree: generated

   PaprReduction
   papr_db
   ccdf
   theory_papr_ccdf
   clipping_bussgang

Constant-envelope waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuous-phase modulation (CPFSK, MSK, GMSK) and constant-envelope OFDM.

.. autosummary::
   :toctree: generated

   Cpm
   msk_precode
   msk_bits
   msk_psd
   CeOfdm
   ce_ofdm_imp_powers
   ce_ofdm_optimum_gain
   ce_ofdm_ber_optimum
   pm_occupied_bandwidth

Chirp spread spectrum
~~~~~~~~~~~~~~~~~~~~~

The modulation of LoRa and its theory.

.. autosummary::
   :toctree: generated

   Css
   css_ser
   css_ber
   css_cross_correlation
   css_max_cross_correlation
   css_continuous_spectrum
   css_spectral_lines
   css_occupied_bandwidth
   css_interference_pattern
   css_ser_interference

Filtered multicarrier and OTFS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Fbmc
   oqam_stagger
   oqam_destagger
   Ufmc
   chebyshev_window
   Otfs

MIMO
----

One channel matrix per received vector; received vectors are ``(n, n_rx)`` arrays.

Detection
~~~~~~~~~

.. autosummary::
   :toctree: generated

   mimo_detect
   mimo_llr
   mimo_estimate

Message passing for large MIMO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   message_passing
   amp_state_evolution
   amp_thresholds
   oamp_state_evolution

Space-time codes, precoding and capacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Ostbc
   eigenmode
   zf_precoder
   water_filling_powers
   mimo_capacity
   ergodic_capacity
   outage_probability
   outage_capacity

Wireless channels
-----------------

Noise and flat fading
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   awgn
   awgn_ebn0
   mimo_channel
   exponential_correlation

Multipath
~~~~~~~~~

Tapped and clustered delay lines, delay–Doppler channels.

.. autosummary::
   :toctree: generated

   TdlChannel
   ChannelTaps
   power_delay_profile
   rms_delay_spread
   CdlChannel
   cdl_profile
   cdl_k_factor_db
   delay_doppler_paths

Optical
-------

Fibre and amplification
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   FiberParams
   linear_fiber_channel
   Ssfm
   Manakov
   ManakovSsfm
   Edfa
   raman_profile
   gn_model
   gn_closed_form
   format_moments
   wdm_grid

Optical devices
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   laser
   phase_noise
   phase_modulator
   mzm
   iqm
   Photodiode
   balanced_photodiode
   hybrid_2x4
   coherent_receiver

Optical wireless
~~~~~~~~~~~~~~~~

Indoor visible-light and infrared links (Lambertian emitters, the reflections of a room,
the PIN-FET receiver) and free-space optics (turbulence, pointing errors, fog).

.. autosummary::
   :toctree: generated

   OwcRoom
   OwcIllumination
   OwcImpulseResponse
   lambertian_mode
   owc_direction
   owc_illuminance
   PinFetReceiver
   Fso
   GammaGamma
   PointingError
   rytov_variance
   fog_attenuation_db_per_km

Hardware impairments
--------------------

Amplifiers and predistortion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   pa_rapp
   pa_saleh
   pa_memory_polynomial
   rapp_predistort
   MemoryPolynomialDpd
   amplifier_bussgang
   envelope_series
   distortion_spectrum

Converters
~~~~~~~~~~

.. autosummary::
   :toctree: generated

   adc
   dac
   theory_quantization_mse
   optimal_loading
   theory_sqnr_sine_db
   enob_from_sinad_db
   theory_jitter_snr_db
   zero_order_hold_gain

Receiver DSP
------------

Filters and sampling
~~~~~~~~~~~~~~~~~~~~

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

Carrier recovery
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   fourth_power_foe
   viterbi_viterbi
   bps
   ddpll
   derotate
   unwrap

Clock recovery
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   gardner_ted
   gardner_ted_nyquist
   interpolate_cubic
   gardner_clock_recovery

Equalisation and compensation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   adaptive_equalizer
   cdc
   cd_taps_needed
   cd_fir_taps
   cd_dispersion_factor
   cd_savory_taps
   cd_ls_taps
   cd_ls_error

Blind estimation
~~~~~~~~~~~~~~~~

Modulation recognition, SNR and symbol rate without a reference.

.. autosummary::
   :toctree: generated

   cumulants
   theory_cumulants
   classify_modulation
   snr_m2m4_db
   snr_crb_nmse
   cyclic_autocorrelation
   estimate_symbol_rate

Metrics and analysis
--------------------

Error rates, SNR and information rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimates from received and transmitted sequences.

.. autosummary::
   :toctree: generated

   ber
   snr_db
   fast_ber_calc
   evm
   ber_to_q_db
   monte_carlo_gmi
   monte_carlo_mi
   information_rates
   posterior_rates

Spectrum
~~~~~~~~

.. autosummary::
   :toctree: generated

   psd
   periodogram
   aclr_db

Results and plots
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Results
   Trace
   compare
   plot

Systems and standards
---------------------

Reference chains of standards (``com-systems``) and the physical layer of 5G NR.

DVB-S2
~~~~~~

.. autosummary::
   :toctree: generated

   Dvbs2
   dvbs2_bbframe
   dvbs2_bbframe_data
   dvbs2_decode_plheader

5G NR
~~~~~

.. autosummary::
   :toctree: generated

   nr_numerology
   nr_ofdm_modulate
   nr_pss
   nr_sss
   nr_detect_cell_id
   nr_ssb_grid
   nr_ssb_indices
   nr_pbch_dmrs
   nr_pdsch_dmrs

Simulation and specifications
-----------------------------

Chains
~~~~~~

.. autosummary::
   :toctree: generated

   Chain
   Block
   Native
   Var
   Frame
   LinkBlock
   frame_metrics
   blocks

``var`` (the function that makes a :class:`Var`) is documented here rather than on a
page of its own, whose file name would clash with that of :class:`Var` on
case-insensitive file systems:

.. autosummary::

   var

.. autofunction:: var

Monte Carlo runner
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   sim

Declarative campaigns
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~

.. autosummary::
   :toctree: generated

   read_sigmf
   write_sigmf
