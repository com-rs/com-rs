Classical performance
=====================

``tests/test_performance.py`` checks that the simulated links reproduce the textbook
results. It runs with the rest of the test suite (``pytest``, about one second) and
every check below is drawn as a full curve in the :doc:`gallery <auto_examples/index>`.

A simulated error rate passes when it lies within four binomial standard errors of
its closed form (every point collects at least 400 errors, so the relative standard
error is below 5 %); the Gray bit error rates, which count one bit per symbol error,
are allowed a few percent of model error.

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - Result
     - Reference
     - Gallery
   * - SER and BER of 4-, 16-, 64-QAM, 8-PSK and 4-PAM over AWGN
     - Proakis & Salehi, *Digital Communications*, 5th ed., Sec. 4.3 (exact SER of
       square QAM and PAM, nearest-neighbour bound for PSK)
     - :doc:`auto_examples/plot_02_ber_qam_awgn`, :doc:`auto_examples/plot_03_ber_psk`
   * - Gray QPSK has the BER of BPSK, :math:`Q(\sqrt{2E_b/N_0})`
     - Proakis & Salehi, Sec. 4.3
     - :doc:`auto_examples/plot_03_ber_psk`
   * - :math:`E_b/N_0 = 9.6` dB for a BER of :math:`10^{-5}` with BPSK
     - the textbook operating point
     - :doc:`auto_examples/plot_02_ber_qam_awgn`
   * - BPSK over flat Rayleigh fading,
       :math:`\frac{1}{2}\bigl(1 - \sqrt{\bar\gamma/(1 + \bar\gamma)}\bigr)`
     - Proakis & Salehi, Sec. 13.3
     - :doc:`auto_examples/plot_04_rayleigh`
   * - MI and GMI of 16-QAM: Monte Carlo against quadrature,
       GMI ≤ MI ≤ :math:`\min(\log_2 M, \log_2(1 + \mathrm{SNR}))`
     - Shannon; Gauss–Hermite quadrature
     - :doc:`auto_examples/plot_05_information_rates`
   * - An RRC pair has no intersymbol interference (residual ISI 40 dB below the signal)
     - Nyquist's criterion
     - :doc:`auto_examples/plot_06_pulse_shaping`
   * - 80 km of SSMF close the eye of 16-QAM at 32 GBd; frequency-domain compensation
       restores the back-to-back performance
     - :math:`e^{j\beta_2\omega^2 L/2}` is all-pass
     - :doc:`auto_examples/plot_07_chromatic_dispersion`
   * - Above the optimum launch power, back-propagation with 20 steps per span beats
       dispersion compensation by more than 3 dB (3 × 80 km, 6 dBm)
     - Ip & Kahn, *J. Lightw. Technol.* 26(20), 2008
     - :doc:`auto_examples/plot_08_launch_power_dbp`
   * - Zero forcing over :math:`n_t \times n_r` Rayleigh has the error rate of MRC with
       :math:`n_r - n_t + 1` branches (2 × 2, 2 × 3, 2 × 4)
     - Winters, Salz & Gitlin, *IEEE Trans. Commun.* 42(2), 1994
     - :doc:`auto_examples/plot_09_mimo_detection`
   * - Alamouti over :math:`2 \times n_r` is MRC of :math:`2 n_r` branches at half
       power (2 × 1, 2 × 2)
     - Alamouti, *IEEE J. Sel. Areas Commun.* 16(8), 1998
     - :doc:`auto_examples/plot_10_space_time_diversity`
   * - 16-QAM over 4 × 4: ML (sphere) < MMSE-SIC < MMSE < ZF; the sphere decoder agrees
       with an exhaustive K-best
     - Agrell et al., *IEEE Trans. Inf. Theory* 48(8), 2002
     - :doc:`auto_examples/plot_09_mimo_detection`
   * - Ergodic capacity of SISO Rayleigh, :math:`\log_2(e)\, e^{1/\rho} E_1(1/\rho)`;
       4 × 4 gains 4 bits per 3 dB at high SNR; water-filling never loses
     - Telatar, *Eur. Trans. Telecommun.* 10(6), 1999
     - :doc:`auto_examples/plot_11_mimo_capacity`
   * - CCDF of the PAPR of OFDM: :math:`1 - \exp(-N\sqrt{\pi\gamma/3}\, e^{-\gamma})` with
       oversampling, :math:`1 - (1 - e^{-\gamma})^N` at the Nyquist rate
     - Ochiai & Imai, *IEEE Trans. Commun.* 49(2), 2001; van Nee & Prasad 2000
     - :doc:`auto_examples/plot_12_papr_ccdf`
   * - SLM with :math:`U` candidates: the CCDF raised to the power :math:`U`
     - Bäuml, Fischer & Huber, *Electron. Lett.* 32(22), 1996
     - :doc:`auto_examples/plot_13_papr_reduction`
   * - DFT-s-OFDM lowers the PAPR at 1 % by more than 2.5 dB (localised, QPSK)
     - Myung, Lim & Goodman, *IEEE Veh. Technol. Mag.* 1(3), 2006
     - :doc:`auto_examples/plot_12_papr_ccdf`
   * - Clipping at the Nyquist rate: gain and signal-to-distortion ratio of the Bussgang
       decomposition
     - Ochiai & Imai, *IEEE Trans. Commun.* 50(1), 2002
     - :doc:`auto_examples/plot_13_papr_reduction`
   * - Clipping alone regrows the spectrum (ACLR < 35 dB); repeated filtering removes it
       and keeps the PAPR near the target
     - Li & Cimini, *IEEE Commun. Lett.* 2(5), 1998; Armstrong, *Electron. Lett.* 38(5),
       2002
     - :doc:`auto_examples/plot_13_papr_reduction`
   * - SLM, PTS, tone reservation and exponential companding reduce the PAPR tail and
       recover the data exactly; ACE changes no decision
     - Müller & Huber 1997; Tellado 2000; Krongold & Jones 2003; Jiang, Yang & Song 2005
     - :doc:`auto_examples/plot_13_papr_reduction`
   * - Digital predistortion of the memory-polynomial amplifier: ACLR and EVM improved
       by more than 15 dB; static inverse of Rapp's amplifier: ACLR +10 dB at 9 dB of
       back-off
     - Ding et al., *IEEE Trans. Commun.* 52(1), 2004; Morgan et al., *IEEE Trans.
       Signal Process.* 54(10), 2006
     - :doc:`auto_examples/plot_14_amplifier_dpd`
   * - Shannon limits of BPSK: 0.187 dB at rate 1/2, −0.495 dB at rate 1/3
     - Shannon 1948
     - :doc:`auto_examples/plot_15_channel_coding`
   * - Soft-decision Viterbi decoding of the (133, 171) code gains about 2 dB over hard
       decisions
     - Viterbi 1967; Forney 1973
     - :doc:`auto_examples/plot_15_channel_coding`
   * - Turbo code, K = 1024, R ≈ 1/3: block error rate below 5 % at 1.0 dB
     - Berrou, Glavieux & Thitimajshima 1993
     - :doc:`auto_examples/plot_15_channel_coding`
   * - LDPC (1024, 512) and CRC-aided list-decoded polar (256, 128) codes: never below the
       normal approximation, within 1.3 and 1 dB of it; list decoding beats SC by more
       than 3×
     - Polyanskiy, Poor & Verdú 2010; Tal & Vardy 2015
     - :doc:`auto_examples/plot_15_channel_coding`
   * - OFDM over EVA with perfect CSI: every subcarrier has the BER of BPSK over flat
       Rayleigh fading
     - Proakis & Salehi, Sec. 13.3
     - :doc:`auto_examples/plot_16_ofdm_channel_estimation`
   * - Channel estimation: LMMSE with the true profile < robust LMMSE < linear
       interpolation; LMMSE within 20 % of the BER of perfect CSI
     - van de Beek et al. 1995; Edfors et al. 1998
     - :doc:`auto_examples/plot_16_ofdm_channel_estimation`
   * - Schmidl & Cox: timing inside the cyclic prefix, offset error falling with the SNR
     - Schmidl & Cox 1997
     - :doc:`auto_examples/plot_17_sync_mimo_ofdm`
   * - 2 × 2 MIMO-OFDM over EPA: per-subcarrier ML beats MMSE
     - Stüber et al. 2004
     - :doc:`auto_examples/plot_17_sync_mimo_ofdm`

The Rust crates carry their own checks of the same kind: the specifications of
``specs/validation/`` (Proakis' error rates, information rates by quadrature, Fig. 5
of Häger & Pfister 2021) run in CI with ``comrs verify``, and the crate
``com-validation`` compares BPSK over AWGN and Rayleigh fading, zero forcing and
Alamouti with their closed forms every night. The deterministic blocks are compared with OptiCommPy to machine precision
by ``tests/test_oracle.py``.
