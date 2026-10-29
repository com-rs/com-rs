# References to download

The standards and papers that the blocks of com-rs are validated against. Put the PDFs
in this folder: `.gitignore` keeps every file of `references/` but this list out of the
repository, since most of them are not redistributable. The list says what every
document is needed for, so that a validation campaign can cite the exact table or
figure it reproduces.

File names: `year_firstauthor_short-title_venue.pdf` for a paper (`_arxiv` when the
file is the preprint), `body_number_version_name.pdf` for a standard. The **File**
column names the file when it is in the folder.

Status: **have** (already used to validate the code), **needed** (the code exists, the
comparison is still to be made), **next** (for a feature not yet implemented).

## Standards

| Document | Where to get it | Needed for | Status | File |
|---|---|---|---|---|
| ETSI EN 302 307-1 V1.4.1, *DVB-S2* | etsi.org (free) | BCH/LDPC tables, constellations, PL framing, Table 13 (`specs/validation/dvbs2_table13.yaml`) | have | |
| ETSI EN 302 307-2 V1.3.1, *DVB-S2X* | etsi.org (free) | the S2X MODCODs, constellations and super-frames | next | `etsi_en-302-307-2_v1.3.1_dvb-s2x.pdf` |
| 3GPP TS 38.212 V18.5.0, *NR; Multiplexing and channel coding* | 3gpp.org (free) | CRCs (Sec. 5.1, checked by `com_fec::crc`), the LDPC base graphs and lifting sizes (5.3.2), polar codes and rate matching (5.3.1, 5.4.1) | have (CRC); next (NR polar sequence and LDPC) | `3gpp_ts-38.212_v18.5.0_nr-channel-coding.pdf` |
| 3GPP TS 38.211, *NR; Physical channels and modulation* | 3gpp.org (free) | the Gold scrambler (5.2.1), the numerologies and prefixes (5.3.1), the PSS, SSS and PBCH DMRS (7.4.1.4, 7.4.2, 7.4.3) and the PDSCH DMRS (7.4.1.1) of `com_waveform::ofdm::nr`, checked against py3gpp; the DMRS tables of the shorter allocations, two-symbol DMRS, PTRS and CSI-RS | needed (to check the transcriptions); next (the remaining tables) | |
| 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz* | 3gpp.org (free) | the CDL profiles (Tables 7.7.1-1 to 7.7.1-5, ray offsets of Table 7.5-3, `com_channel::cdl`, transcribed from V17.0.0) and the TDL profiles of Table 7.7.2 | have (CDL, checked by `com_channel::cdl`); needed (TDL) | |
| 3GPP TS 36.212, *E-UTRA; Multiplexing and channel coding* | 3gpp.org (free) | the LTE turbo code and its QPP interleaver (Table 5.1.3-3) | needed (to check the QPP table, transcribed from Sionna and checked by its vectors at K = 40, 112, 168, 432) | |
| 3GPP TS 36.101, *E-UTRA; UE radio transmission and reception*, Annex B.2 | 3gpp.org (free) | the EPA, EVA and ETU delay profiles of `tdl` | needed | |
| ITU-R M.1225 (1997), *Guidelines for evaluation of radio transmission technologies for IMT-2000* | itu.int (free) | the pedestrian and vehicular profiles A and B (Tables 2, 4 and 5, checked by `com_channel::multipath`) | have | `itu-r_m.1225_1997_imt-2000-channel-models.pdf` |
| ITU-T O.150, *General requirements for instrumentation for performance measurements on digital transmission equipment* | itu.int (free) | the PRBS polynomials of `prbs` | needed | |
| IEEE 802.11-2020, Sec. 19.3.11 and Annex F | ieee.org (free through IEEE GET) | the puncturing patterns of Figs. 17-9 and 19-11 (checked by `com_fec::puncture`); the LDPC codes of Wi-Fi (an `accumulate` code family) | have (puncturing, Figs. 17-9 and 19-11); next (LDPC) | `ieee_802.11-2020.pdf` |
| ETSI EN 300 744, *DVB-T* | etsi.org (free) | the RS(204, 188) generator of Sec. 4.3.2 (checked by `com_fec::rs`) | needed | |
| ETSI EN 300 421, *DVB-S* | etsi.org (free) | the puncturing patterns of Table 1 (checked by `com_fec::puncture`) | needed | |
| CCSDS 131.0-B, *TM synchronization and channel coding* | ccsds.org (free) | the roots of the RS(255, 223) code, Sec. 4.2 (checked by `com_fec::rs`) | needed | |
| SigMF specification 1.x | github.com/sigmf (free) | the recordings read by `capture` | have | |

## Papers used as references of the models

Optical (channel and amplification):

| Paper | Needed for | Status | File |
|---|---|---|---|
| P. Poggiolini, "The GN model of non-linear propagation in uncompensated coherent optical systems," *JLT* 30(24), 2012 | the GN integral and closed form (`specs/wdm_gn.yaml`) | have | `2012_poggiolini_gn-model_jlt.pdf` |
| A. Carena, G. Bosco, V. Curri, Y. Jiang, P. Poggiolini, F. Forghieri, "EGN model of non-linear fiber propagation," *Opt. Express* 22(13), 2014 | the EGN corrections, Fig. 1 | have | |
| P. Serena, C. Lasagni, S. Musetti, A. Bononi, "On numerical simulations of ultra-wideband long-haul optical communication systems," *JLT* 38(5), 2020 | split-step accuracy and ISRS in simulations | have | |
| J. Bromage, "Raman amplification for fiber communications systems," *JLT* 22(1), 2004 | the Raman gain spectrum and power equations | have | |
| V. E. Perlin, H. G. Winful, "Optimal design of flat-gain wide-band fiber Raman amplifiers," *JLT* 20(2), 2002 | multi-pump gain flattening; the gain spectrum of the paper is a figure only (Fig. 1), so the designs cannot be reproduced exactly | needed | `2002_perlin_flat-gain-raman-amplifiers_jlt.pdf` |
| D. Semrau, P. Bayvel, "The Gaussian noise model in the presence of inter-channel stimulated Raman scattering," *JLT* 36(14), 2018 (arXiv 1801.02460) | the ISRS power transfer, eq. (10) (checked by `com_channel::optical::raman`); the ISRS GN model of `gn_snr_db` against its Fig. 2 remains | have | `2018_semrau_gn-model-isrs_jlt_arxiv.pdf` |
| D. Semrau, R. I. Killey, P. Bayvel, "A closed-form approximation of the Gaussian noise model in the presence of inter-channel stimulated Raman scattering," *JLT* 37(9), 2019 | the closed form of the ISRS GN model | needed | |
| D. Marcuse, C. R. Menyuk, P. K. A. Wai, "Application of the Manakov-PMD equation…," *JLT* 15(9), 1997 | the coarse-step PMD model | have | `1997_marcuse_manakov-pmd_jlt.pdf` |
| E. Ip, J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital backpropagation," *JLT* 26(20), 2008 | DBP with few steps, the `nonlinear_scale` | have | `2008_ip_digital-backpropagation_jlt.pdf` |
| O. V. Sinkin, R. Holzlöhner, J. Zweck, C. R. Menyuk, "Optimization of the split-step Fourier method…," *JLT* 21(1), 2003 | the step rules of the split step | have | |
| C. Häger, H. D. Pfister, "Physics-based deep learning for fiber-optic communication systems," *JSAC* 39(1), 2021 | `specs/validation/hager2021_fig5.yaml` | have | |

Coding and shaping:

| Paper | Needed for | Status | File |
|---|---|---|---|
| X.-Y. Hu, E. Eleftheriou, D. M. Arnold, "Regular and irregular progressive edge-growth Tanner graphs," *IT* 51(1), 2005 | the PEG construction, Sec. VI (girth and degrees checked by `com_fec::ldpc`) | have | `2005_hu_peg-ldpc_it.pdf` |
| E. Arıkan, "Channel polarization," *IT* 55(7), 2009; I. Tal, A. Vardy, "List decoding of polar codes," *IT* 61(5), 2015 | SC and CA-SCL decoding, to compare block error rates with Fig. 1 of Tal & Vardy | needed | |
| C. Berrou, A. Glavieux, P. Thitimajshima, "Near Shannon limit error-correcting coding and decoding: turbo-codes," *ICC* 1993 | the turbo decoder; the LTE curves are in TS 36.212 evaluations | have | `1993_berrou_turbo-codes_icc.pdf` |
| G. Böcherer, F. Steiner, P. Schulte, "Bandwidth efficient and rate-matched low-density parity-check coded modulation," *TCOM* 63(12), 2015 (arXiv 1502.02733) | PAS, Fig. 8 (`specs/pas_64qam.yaml`); Tables I and III (shaping gaps of ASK, bit-metric decoding) and Tables VI–X (PAS on the DVB-S2 LDPC codes, bit mappers of Sec. VII-B), checked by `validation/tests/shaping.rs` | have | `2015_bocherer_pas-ldpc_tcom_arxiv.pdf` |
| P. Schulte, G. Böcherer, "Constant composition distribution matching," *IT* 62(1), 2016 (arXiv 1503.05133) | the CCDM: rate, normalised divergence (eq. 17) and bound (22) of Fig. 2 with the composition of eq. (10), checked by `validation/tests/shaping.rs` | have | `2016_schulte_ccdm_it_arxiv.pdf` |
| E. M. Liang, J. M. Kahn, "Probabilistic shaping distributions for optical communications," *JLT* 43(4), 2025 | the ultimate shaping gain πe/6 (eq. 27) and the finite-constellation shaping gains of MB square QAM over the SC channel, Fig. 6, checked by `validation/tests/shaping.rs`; the direct-detection channels (SD, SV, KK) are not implemented | have | `2025_liang_shaping-distributions_jlt.pdf` |
| Y. Polyanskiy, H. V. Poor, S. Verdú, "Channel coding rate in the finite blocklength regime," *IT* 56(5), 2010 | the normal approximation | have | |
| A. Cassagne, O. Hartmann, M. Léonardon, K. He, C. Leroux, R. Tajan, O. Aumage, D. Barthou, T. Tonnellier, V. Pignoly, B. Le Gal, C. Jégo, "AFF3CT: A Fast Forward Error Correction Toolbox!," *SoftwareX* 10, 100345, 2019 | the reference implementation of the error rates of 17 configurations over BPSK and AWGN: RS and BCH algebraic decoding, (133, 171) Viterbi, LDPC (flooding and layered sum-product, normalised and offset min-sum, the DVB-S2 short code), polar SC and CA-SCL, LTE turbo (log-MAP, max-log-MAP) (`validation/tests/aff3ct.rs`) | have (recorded outputs, `validation/aff3ct/`) | |
| S. Lin, D. J. Costello, *Error Control Coding*, 2nd ed., Pearson, 2004 | the generator polynomial and the decoding example of the RS(15, 9) code, Secs. 7.3–7.4 (checked by `com_fec::rs`) | needed | |
| Y. Yasuda, K. Kashiki, Y. Hirata, "High-rate punctured convolutional codes for soft decision Viterbi decoding," *TCOM* 32(3), 1984 | the free distances of the punctured (133, 171) code at rates 2/3 to 7/8, Table I (checked by `com_fec::puncture`) | needed | |
| A. Morello, V. Mignone, "DVB-S2: the second generation standard for satellite broad-band services," *Proc. IEEE* 94(1), 2006 | the DVB-S2 performance curves beyond Table 13 | needed | |

Waveforms, synchronisation and MIMO:

| Paper | Needed for | Status | File |
|---|---|---|---|
| J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008 | the error rates of QAM, PSK and PAM (`specs/validation/proakis_awgn.yaml`), MSK against BPSK, Sec. 3.3-2 (`specs/validation/msk_bpsk.yaml`, `com_waveform::cpm`), the spectrum of MSK, Sec. 3.4-3, differential PSK, Sec. 4.5 (`com_mapping::differential`), maximal-ratio combining, eq. 13.4-15 (`specs/mimo_diversity.yaml`) | needed | |
| K. Murota, K. Hirade, "GMSK modulation for digital mobile radio telephony," *TCOM* 29(7), 1981 | the 99 % bandwidths of GMSK, Table I (checked by `com_waveform::cpm`) | needed | |
| T. Aulin, C.-E. W. Sundberg, "Continuous phase modulation — Part I: Full response signaling," *TCOM* 29(3), 1981 | the minimum distances of MSK and CPFSK (checked by `com_waveform::cpm`) | needed | |
| R. F. Pawula, S. O. Rice, J. H. Roberts, "Distribution of the phase angle between two vectors perturbed by Gaussian noise," *TCOM* 30(8), 1982 | the error rates of DQPSK and D8PSK, Sec. IV (checked by `com_mapping::differential`) | needed | |
| K. Cho, D. Yoon, "On the general BER expression of one- and two-dimensional amplitude modulations," *TCOM* 50(7), 2002 | the exact BER of Gray-mapped square QAM, eq. (16) (`com_mapping::theory`) | needed | |
| S. M. Alamouti, "A simple transmit diversity technique for wireless communications," *JSAC* 16(8), 1998 | `mimo_tx` with `scheme: alamouti`, Fig. 4 | have | `1998_alamouti_transmit-diversity_jsac.pdf` |
| V. Tarokh, H. Jafarkhani, A. R. Calderbank, "Space-time block codes from orthogonal designs," *IT* 45(5), 1999 | the rate-3/4 OSTBC, eqs. (39)–(40) (checked by `com_mimo::stbc`) | have | `1999_tarokh_space-time-block-codes_it.pdf` |
| S. Yang, L. Hanzo, "Fifty years of MIMO detection: the road to large-scale MIMOs," *IEEE Commun. Surveys Tuts.* 17(4), 2015 (arXiv 1507.05138) | the receive diversity orders of the hard-decision detectors, Table X: the high-SNR slopes of ML, sphere, ZF, MMSE, ZF/MMSE-(O)SIC and K-best on 2 × 2 and 2 × 4 QPSK links (`validation/tests/mimo.rs`) | have | `2015_yang_fifty-years-mimo-detection_comst_arxiv.pdf` |
| C. Jeon, R. Ghods, A. Maleki, C. Studer, "Optimal data detection in large MIMO," arXiv 1811.01917, 2018 | LAMA (`com_mimo::message_passing`): Table II (recovery thresholds and critical noise variances), the operating points of Sec. V, Fig. 6(a), checked by `validation/tests/message_passing.rs` | have | `2018_jeon_optimal-detection-large-mimo_arxiv.pdf` |
| J. Ma, L. Ping, "Orthogonal AMP," *IEEE Access* 5, 2020–2033, 2017 (arXiv 1602.06509) | OAMP (`com_mimo::message_passing`): eq. (42) and Figs. 2 and 3 (state evolution against simulation on i.i.d. and ill-conditioned matrices), checked by `validation/tests/message_passing.rs` | have | `2017_ma_orthogonal-amp_access_arxiv.pdf` |
| O. Edfors et al., "OFDM channel estimation by singular value decomposition," *TCOM* 46(7), 1998 (the file is the VTC 1996 conference version, DOI 10.1109/VETEC.1996.501446) | LMMSE estimation | have | `1996_edfors_ofdm-channel-estimation-svd_vtc.pdf` |
| S. Coleri, M. Ergen, A. Puri, A. Bahai, "Channel estimation techniques based on pilot arrangement in OFDM systems," *TBC* 48(3), 2002 | LS estimation, pilot patterns | have | `2002_coleri_ofdm-pilot-channel-estimation_tbc.pdf` |
| X. Lin et al., "5G New Radio: Unveiling the essentials of the next generation wireless access technology," *IEEE Commun. Standards Mag.* 3(3), 30–37, 2019, arXiv:1806.06898 | the NR numerology (Sec. II) and the SSB (Sec. III-A) of `com_waveform::ofdm::nr` (`validation/tests/nr.rs`); an overview of DMRS, PTRS, CSI-RS (Sec. VI) | have | `2019_lin_5g-new-radio-essentials_arxiv.pdf` |
| D. Astély, E. Dahlman, A. Furuskär, Y. Jading, M. Lindström, S. Parkvall, "LTE: the evolution of mobile broadband," *IEEE Commun. Mag.* 47(4), 44–51, 2009 | the LTE numerology (15 kHz, 4.7 and 16.7 µs prefixes, 14 or 12 symbols per subframe), the case $\mu = 0$ of `ofdm::nr` | have | `2009_astely_lte-evolution-mobile-broadband_commag.pdf` |
| I. Eizmendi et al., "DVB-T2: the second generation of terrestrial digital video broadcasting system," *IEEE Trans. Broadcast.* 60(2), 258–271, 2014 (author version) | the DVB-T2 OFDM options (FFT sizes, guard intervals, PP1–PP8), the P1 symbol (C-A-B structure of 542, 1024 and 482 samples) and the number of P2 symbols; the pilot patterns and the P1 sequences need EN 302 755 | have (overview); next (P1 structure) | `2014_eizmendi_dvb-t2-second-generation_tbc.pdf` |
| T. M. Schmidl, D. C. Cox, "Robust frequency and timing synchronization for OFDM," *TCOM* 45(12), 1997 | `schmidl_cox` | have | |
| A. B. Awoseyila, C. Kasparis, B. G. Evans, "Robust time-domain timing and frequency synchronization for OFDM systems," *IEEE Trans. Consumer Electron.* 55(2), 2009 | a time-domain alternative to Schmidl–Cox, to compare `ofdm_synchronize` with its Figs. 4–6 | needed | `2009_awoseyila_ofdm-timing-frequency-sync_tce.pdf` |
| F. M. Gardner, "A BPSK/QPSK timing-error detector for sampled receivers," *TCOM* 34(5), 1986 | `gardner` | needed | |
| T. Pfau, S. Hoffmann, R. Noé, "Hardware-efficient coherent digital receiver concept with feedforward carrier recovery for M-QAM constellations," *JLT* 27(8), 2009 | `bps`, Table I (the 1 dB-penalty linewidths include the differential decoding of the paper, which the library does not have: the comparison needs it) | needed | `2009_pfau_blind-phase-search_jlt.pdf` |
| K. Kikuchi, "Performance analyses of polarization demultiplexing based on constant-modulus algorithm in digital coherent optical receivers," *Opt. Express* 19(10), 2011 | the singularity and the `kikuchi` initialisation | have | |
| H. Ochiai, H. Imai, "On the distribution of the peak-to-average power ratio in OFDM signals," *TCOM* 49(2), 2001 | the PAPR CCDF, eqs. (11) and (27) (checked by `com_waveform::papr`) | have | `2001_ochiai_papr-distribution_tcom.pdf` |
| H. Ochiai, H. Imai, "Performance analysis of deliberately clipped OFDM signals," *TCOM* 50(1), 2002 | the Bussgang decomposition | have | |
| L. Ding et al., "A robust digital baseband predistorter constructed using memory polynomials," *TCOM* 52(1), 2004 | `MemoryPolynomialDpd` | have | |
| A. Swami, B. M. Sadler, "Hierarchical digital modulation classification using cumulants," *TCOM* 48(3), 416–429, 2000, doi:10.1109/26.837045 | the normalised cumulants of the alphabets: check the values of `validation/tests/blind.rs::normalised_cumulants_of_swami_and_sadler` against its table; its classification rates | needed | |
| W. Wei, J. M. Mendel, "Maximum-likelihood classification for digital amplitude-phase modulations," *TCOM* 48(2), 189–193, 2000, doi:10.1109/26.823550 | the likelihood classifier of `com_metrics::blind` | needed | |
| D. R. Pauluzzi, N. C. Beaulieu, "A comparison of SNR estimation techniques for the AWGN channel," *TCOM* 48(10), 1681–1691, 2000, doi:10.1109/26.871393 | the bias and variance of the M2M4 estimator | needed | |
| O. A. Dobre, A. Abdi, Y. Bar-Ness, W. Su, "Survey of automatic modulation classification techniques: classical approaches and new trends," *IET Commun.* 1(2), 137–156, 2007, doi:10.1049/iet-com:20050176 | the classification methods and their reported performance | needed | |
| W. A. Gardner, "Exploitation of spectral redundancy in cyclostationary signals," *IEEE Signal Process. Mag.* 8(2), 14–36, 1991, doi:10.1109/79.81007 | the cyclic autocorrelation and the symbol-rate line of `com_dsp::cyclo` | needed | |
| F. Koyama, K. Iga, "Frequency chirping in external modulators," *J. Lightw. Technol.*, 1988 | the chirp of an unbalanced Mach–Zehnder drive, for a chirped model of `com_hardware::optical::Mzm` | next | |
| W. R. Bennett, "Spectra of quantized signals," *Bell Syst. Tech. J.* 27(3), 446–472, 1948 | the granular noise Δ²/12 and the SQNR of a full-scale sine of `com_hardware::converter` (`validation/tests/converters.rs`) | needed | |
| J. Max, "Quantizing for minimum distortion," *IRE Trans. Inf. Theory* 6(1), 7–12, 1960 | the optimal uniform quantiser of a Gaussian input: check the steps and errors of `validation/tests/converters.rs::optimal_uniform_quantizers_of_max_1960` against its table | needed | |
| R. H. Walden, "Analog-to-digital converter survey and analysis," *JSAC* 17(4), 539–550, 1999 | the jitter-limited SNR of `com_hardware::converter` | needed | |
| IEEE Std 1241-2010, *Standard for terminology and test methods for analog-to-digital converters* | the definition of the ENOB from the SINAD | needed | |
| A. Eghbali, H. Johansson, O. Gustafsson, S. J. Savory, "Optimal least-squares FIR digital filters for compensation of chromatic dispersion in digital coherent optical receivers," *JLT* 32(8), 2014 | the least-squares CD filters of `com_dsp::compensation::cd` (`cd_ls_taps`): Table I, eqs. 12–19, Figs. 5, 8 and 9 in `validation/tests/cd_filters.rs` | have | `2014_eghbali_ls-fir-cd-compensation_jlt.pdf` |
| S. J. Savory, "Digital filters for coherent optical receivers," *Opt. Express* 16(2), 804–817, 2008, doi:10.1364/OE.16.000804 (open access) | the truncated CD filter and its length rule (`cd_savory_taps`, eqs. 9–10) | needed | |
| S. C. Thompson et al., "Constant envelope OFDM," *TCOM* 56(8), 2008 | CE-OFDM: the modulator and the phase demodulator of Secs. II–III, eq. (15) (checked by `com_waveform::ce_ofdm` and `specs/validation/thompson2008_eq15.yaml`), Fig. 2 (the gallery) | have | `2008_thompson_constant-envelope-ofdm_tcom.pdf` |

## Reference implementations (free, used as test oracles only)

| Software | Used for |
|---|---|
| OptiCommPy (GPL-3.0) | the conventions of the blocks, `tests/test_oracle.py` |
| GNPy (BSD-3) | the Raman solver and the GN/EGN models |
| AFF3CT (MIT) | the error rates of the Reed–Solomon, BCH, convolutional, LDPC, polar and LTE turbo codes, recorded by `tools/record_aff3ct.py` into `validation/aff3ct/` and compared by `validation/tests/aff3ct.rs`; its reference database (github.com/aff3ct/error_rate_references, no licence) is only read for its command lines |
