Gallery of classical results
============================

The textbook curves of digital and optical communications, simulated with ``comrs``
and compared with their closed forms. Every script runs as is (the figures of this
page are produced when the documentation is built), and the same results are checked
at a few points by ``tests/test_performance.py``.

Against the literature
----------------------

The scripts are grouped by theme, in the order of a link. Some reproduce one named
figure or table of a paper or a standard, with the same system parameters, and say
what the library's curve is expected to match: where the published points are
available as numbers, they are drawn on the figure; otherwise the caption names the
figure to open next to the plot. The others compare with closed forms and cite the
equations they use. The table below gathers every comparison of this gallery.

.. list-table::
   :header-rows: 1
   :widths: 30 42 28

   * - Reference
     - What is reproduced
     - Where
   * - S. M. Alamouti, *IEEE J. Sel. Areas Commun.* 16(8), 1998, Fig. 4
     - BER of coherent BPSK over Rayleigh fading: no diversity, MRRC with 2 and 4
       receive antennas, the two-branch transmit diversity scheme with 1 and 2
       receive antennas
     - :ref:`sphx_glr_auto_examples_mimo_plot_03_alamouti1998_fig4.py`
   * - X.-Y. Hu, E. Eleftheriou, D. M. Arnold, *IEEE Trans. Inf. Theory* 51(1), 2005,
       Fig. 5
     - Bit and block error rates of a regular (3, 6) PEG code of length 504 over
       AWGN with BPSK and sum-product decoding
     - :ref:`sphx_glr_auto_examples_coding_plot_02_hu2005_fig5.py`
   * - S. C. Thompson et al., *IEEE Trans. Commun.* 56(8), 2008, Fig. 2 and eq. (15)
     - BER of the CE-OFDM phase demodulator receiver for 8-PAM data and several
       modulation indices, against the high-CNR approximation
     - :ref:`sphx_glr_auto_examples_waveforms_plot_05_thompson2008_fig2.py`
   * - C. Häger, H. D. Pfister, *IEEE J. Sel. Areas Commun.* 39(1), 2021, Fig. 5
     - Effective SNR versus launch power over 25 × 80 km, linear equalisation and
       digital back-propagation with one step per span; the published points are
       drawn
     - :ref:`sphx_glr_auto_examples_optical_plot_03_hager2021_fig5.py`
   * - J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., 2008, Sec. 3.3-2
     - MSK detected by maximum-likelihood sequence detection has the BER of BPSK
       once precoded; without precoding, that of differentially decoded BPSK
     - :ref:`sphx_glr_auto_examples_waveforms_plot_04_proakis2008_msk.py`
   * - J. Guerreiro et al., *IEEE Commun. Lett.* 20(11), 2016, Fig. 1 and eq. (8);
       M. J. López Morales et al., *Globecom Workshops* 2022, Fig. 3; A. U. Ahmed,
       J. R. Zeidler, *IEEE Trans. Signal Process.* 63(17), 2015, Fig. 6; R. A. Pacheco,
       D. Hatzinakos, *ICASSP* 2006, Fig. 1; J. Guerreiro et al., *VTC Fall* 2015, Fig. 3
     - CE-OFDM: the phase detector, GAMP and the optimum BER at :math:`2\pi h = 1.4`;
       the arctangent, basic linear and enhanced receivers at 0.7; the 99 % bandwidth;
       the spectrum at :math:`2\pi h = 2`
     - :ref:`sphx_glr_auto_examples_waveforms_plot_06_ce_ofdm_receivers.py`
   * - O. Afisiadis et al., *IEEE Trans. Wireless Commun.* 19(2), 2020, Figs. 2 and 7;
       L. Vangelista, *IEEE Signal Process. Lett.* 24(12), 2017, Fig. 1; M. Chiani,
       A. Elzanaty, *IEEE Internet Things J.* 6(5), 2019, Fig. 3a
     - The LoRa modulation: the SER over AWGN for SF 7 to 12, FSCM against FSK over a
       frequency-selective channel, the continuous spectrum, a same-SF interferer
     - :ref:`sphx_glr_auto_examples_waveforms_plot_07_lora.py`
   * - ETSI EN 302 307-1 V1.4.1, Table 13
     - The :math:`E_s/N_0` thresholds of the 28 DVB-S2 MODCODs against the simulated
       frame error rates
     - :ref:`sphx_glr_auto_examples_systems_plot_01_dvbs2.py`
   * - Proakis & Salehi, 5th ed., Secs. 4.3 and 13.3; eq. (13.4-15)
     - Error rates of QAM, PSK and PAM over AWGN, BPSK over Rayleigh fading,
       maximal-ratio combining
     - :ref:`sphx_glr_auto_examples_foundations_plot_02_ber_qam_awgn.py`,
       :ref:`sphx_glr_auto_examples_foundations_plot_03_ber_psk.py`,
       :ref:`sphx_glr_auto_examples_foundations_plot_04_rayleigh.py`,
       :ref:`sphx_glr_auto_examples_mimo_plot_02_space_time_diversity.py`
   * - H. Ochiai, H. Imai, *IEEE Trans. Commun.* 49(2), 2001, eqs. (11) and (27)
     - The distribution of the PAPR of OFDM at the Nyquist rate and oversampled
     - :ref:`sphx_glr_auto_examples_waveforms_plot_01_papr_ccdf.py`
   * - Y. Polyanskiy, H. V. Poor, S. Verdú, *IEEE Trans. Inf. Theory* 56(5), 2010
     - The normal approximation of the finite-length limit, against polar and LDPC
       codes of length 256
     - :ref:`sphx_glr_auto_examples_coding_plot_01_channel_coding.py`
   * - P. Poggiolini, *J. Lightw. Technol.* 30(24), 2012; A. Carena et al., *Opt.
       Express* 22(13), 2014
     - The GN and EGN models of nonlinear interference against split-step
       simulations of a WDM link
     - :ref:`sphx_glr_auto_examples_optical_plot_04_wdm_nonlinearity.py`
   * - G. Böcherer, F. Steiner, P. Schulte, *IEEE Trans. Commun.* 63(12), 2015
     - Probabilistic amplitude shaping of 64-QAM at 3 bits per symbol
     - :ref:`sphx_glr_auto_examples_coding_plot_04_probabilistic_shaping.py`

The README of ``references/`` in the repository lists every document the library is
validated against, with its status, and ``specs/validation/`` holds the campaigns that
CI checks against these sources.
