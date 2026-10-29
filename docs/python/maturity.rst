Maturity
========

For every module, the strongest evidence that it is right. The levels:

- **A**: validated against an external reference (a published figure or table, a
  table of a standard reproduced verbatim, or the recorded output of a reference
  implementation: OptiCommPy 0.11, GNPy 3.0, AFF3CT v4.7.1);
- **B**: against a closed form only;
- **C**: self-consistency only (round trips, invariants);
- **—**: not validated.

The tests that give the evidence and the regimes not covered yet are listed module by
module in `docs/MATURITY.md <https://github.com/com-rs/com-rs/blob/main/docs/MATURITY.md>`_.

.. list-table::
   :header-rows: 1
   :widths: 40 8 52

   * - Module
     - Level
     - Strongest evidence
   * - Constellations, mapping, demapping and closed-form error rates
     - A
     - OptiCommPy outputs; closed forms of Proakis & Salehi
   * - Sources (bits, PRBS, uniform and shaped symbols)
     - C
     - periods and distributions
   * - Differential PSK
     - B
     - closed forms of Proakis & Salehi and Pawula, Rice & Roberts
   * - Convolutional codes (Viterbi, BCJR)
     - A
     - AFF3CT; Yasuda et al. Table I
   * - Puncturing
     - A
     - Yasuda et al. Table I; patterns of IEEE 802.11 and EN 300 421
   * - Turbo codes (LTE)
     - A
     - AFF3CT, K = 40 and 1024
   * - LDPC codes
     - A
     - AFF3CT, PEG and DVB-S2 short code; Hu et al. 2005
   * - Polar codes (SC, CA-SCL)
     - A
     - AFF3CT, SC and CA-SCL
   * - 5G NR LDPC and polar codes with rate matching
     - A
     - Sionna vectors bit for bit; TS 38.212 tables
   * - Reed–Solomon codes
     - A
     - AFF3CT; Lin & Costello; ETSI EN 300 744, CCSDS 131.0-B
   * - BCH codes
     - A
     - AFF3CT; generators of EN 302 307-1 Table 6a
   * - CRC
     - A
     - CRC polynomials of TS 38.212
   * - Scrambling, interleaving
     - C
     - balanced sequences, invertible permutations
   * - Finite-length bounds
     - B
     - the Shannon limits of BPSK
   * - Distribution matching and probabilistic amplitude shaping
     - A
     - Schulte & Böcherer 2016 Fig. 2; Böcherer et al. 2015 Tables VI–X
   * - DVB-S2
     - A
     - EN 302 307-1 Table 13 and the tables of the standard
   * - OFDM and channel estimation
     - B
     - flat Rayleigh fading per subcarrier
   * - Error rates with an imperfect channel estimate, DPSK over Jakes' fading
     - A
     - Chang & Su 2002 eqs. 31–53, Figs. 3–4
   * - Nonlinear amplifier driven by OFDM
     - A
     - Banelli & Cacopardi 2000 eqs. 7–31, Fig. 5
   * - 5G NR numerology, SSB, cell search, PDSCH DMRS
     - A
     - py3gpp outputs; Lin et al. 2019
   * - OFDM synchronisation (Schmidl–Cox)
     - C
     - the error falls with the SNR
   * - PAPR: distribution and reduction
     - B
     - Ochiai & Imai 2001 eqs. (11), (27)
   * - Constant-envelope OFDM
     - B
     - Thompson et al. 2008 eq. (15)
   * - Continuous-phase modulation
     - A
     - Murota & Hirade 1981 Table I; MSK of Proakis & Salehi
   * - MIMO detection
     - A
     - Yang & Hanzo 2015 Table X; MRC closed forms
   * - Message-passing detection of large MIMO (LAMA, OAMP)
     - A
     - Jeon et al. 2018 Table II; Ma & Ping 2017 eq. (42), Figs. 2–3
   * - Space-time block codes
     - A
     - Tarokh et al. 1999 eqs. (39)–(40); MRC closed forms
   * - Blind estimation: modulation recognition, M2M4 SNR, symbol rate
     - B
     - exact cumulants of the alphabets; consistency of the estimators
   * - DAC and ADC
     - B
     - Bennett 1948; the exact error of a Gaussian input; IEEE Std 1241; Walden 1999
   * - Capacity and precoding
     - B
     - the ergodic capacity of Rayleigh fading
   * - AWGN
     - B
     - closed forms of the error rates
   * - Flat fading and antenna correlation
     - B
     - BPSK over Rayleigh fading; MRC
   * - Tapped and clustered delay lines (TDL, CDL)
     - A
     - ITU-R M.1225 and 3GPP TR 38.901 tables, verbatim
   * - Fibre: linear, split-step Fourier, Manakov, back-propagation
     - A
     - OptiCommPy outputs; Häger & Pfister 2021 Fig. 5
   * - GN and EGN models
     - A
     - Carena et al. 2014 Fig. 1(a); GNPy outputs
   * - Raman amplification and ISRS
     - A
     - GNPy outputs; Semrau & Bayvel 2018 eq. (10)
   * - EDFA and PMD
     - B
     - the accumulated ASE
   * - Phase noise and frequency offset
     - B
     - the variance of the Wiener increments
   * - Power amplifiers, IQ imbalance and digital predistortion
     - C
     - properties of the models
   * - Optical front ends (laser, modulators, photodiodes, coherent receiver)
     - A
     - OptiCommPy outputs
   * - Filters and resampling
     - A
     - OptiCommPy outputs
   * - Chromatic dispersion compensation
     - A
     - Eghbali et al. 2014 Table I, Figs. 5 and 9
   * - Adaptive equalisers
     - A
     - OptiCommPy outputs; Kikuchi's singularity
   * - Carrier recovery
     - A
     - OptiCommPy outputs
   * - Clock recovery
     - A
     - OptiCommPy outputs
   * - Frame synchronisation
     - C
     - the blocks find their offset
   * - Error rates, SNR and EVM
     - A
     - OptiCommPy outputs
   * - Information rates and shaping gains
     - A
     - Böcherer et al. 2015 Tables I, III; OptiCommPy outputs
   * - Receiver posteriors (``air`` metrics)
     - C
     - consistency of the rates
   * - Spectral metrics (PSD, ACLR, PAPR, CCDF)
     - C
     - a tone lands in its bin
   * - Monte Carlo runner
     - C
     - thread independence, stopping rules
   * - Specifications, verification and operating-point search
     - B
     - the search against the inverted closed forms
   * - Recordings (SigMF, raw IQ, NPY)
     - A
     - sigmf-python, the reference implementation, both ways
