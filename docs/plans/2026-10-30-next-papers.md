# The papers marked "next": what to test, what to add — 30 October 2026

The papers of `references/` whose comparison needs a feature com-rs does not have yet,
read for the cases of their validation cards and for the smallest feature that computes
them, following the steps of `CLAUDE.md` ("Before adding a function or a block"): the
existing prototypes each one reuses are named. None needs a new crate edge or an ADR.
DOIs are checked on Crossref.

## A. OFDM synchronisation: Moose 1994, van de Beek et al. 1997

**Shared kernel** (`com_dsp::sync::frame`), on which `com_waveform::ofdm::sync::schmidl_cox`
is rebuilt: `delay_correlation(r, lag, window) -> Vec<C64>` ($\sum_{m} r^{*}[d+m]\,
r[d+m+\text{lag}]$, direct sums to keep the order of operations, ADR 0005) and
`moving_energy(r, window) -> Vec<f64>` (`com_dsp::filter::moving_average` is centred and
zero-padded, so it does not fit).

**Moose, TCOM 42(10), 1994, doi:10.1109/26.328961.** Cases: eq. (15), the SNR bound under
a frequency offset, and Fig. 1 (N = 256, K = 96, 8PSK, Ec/N0 = 11–29 dB, ε ≤ 0.5, ±1 dB);
eq. (21), the estimator, and eq. (26), its variance $1/((2\pi)^{2} E_{s}/N_{0})$; Fig. 3
(193 carriers, ε = 0 and 0.45, 20–60 dB, a factor 1.3); the example of Sec. IV (σ =
0.0063, SIR 24.3 dB, loss 0.2 dB, all reproduced); the constant 0.5947 (the exact sum of
eq. (10) gives 0.5937). Remark: the text cites "(25)" for the variance, which is eq. (26).
Feature (`com_waveform::ofdm::sync`): `repeated_preamble(ofdm, seed)` (reuses
`Ofdm::to_time`, `Ofdm::with_prefix`), `moose(r, start, ofdm) -> Result<f64>` (reuses
`Ofdm::to_active`), `snr_cfo_bound(snr, eps)` (eq. 15), `ici_power(ofdm, i, eps)` (eq.
10); the tests use `com_hardware::oscillator::Cfo` and `com_channel::awgn::add_noise`.
Small to medium; mostly CI.

**van de Beek, Sandell, Börjesson, TSP 45(7), 1997, doi:10.1109/78.599949.** Cases: eq.
(2) (|ε| ≤ 1.3·10⁻² at 30 dB, exact: 0.01306); the estimator of eqs. (5)–(13); Fig. 5
(MSE against L = 2…15 at 4, 10, 16 dB); Fig. 6 (N = 256, L = 4, 8, 15, AWGN 0–24 dB, and
the dispersive floors) and Fig. 7 (SNR loss), within a factor 1.5; the dispersive channel
(15 Rayleigh taps, exponential, 2 samples RMS) is a `com_channel::multipath::Tdl`.
Feature: `cp_ml(r, ofdm, snr) -> Result<Estimate>` on `delay_correlation` (lag
`fft_len()`, window the prefix) and `moving_energy`. Small; campaign tier.

## B. Spectrum sensing: Yücek & Arslan 2009, Axell et al. 2012, Zhao & Sadler 2007

Placement: `com_metrics::sensing`, next to the cumulant classification of
`com_metrics::blind`. `energy_threshold(n, noise_var, pfa)` (by `com_math::search::bisect`),
`energy_pfa`, `energy_pd_gaussian` (on `com_math::gamma_p`), `snr_wall(rho)`,
`energy_samples_required(snr, rho, pfa, pd)`; the statistic is `com_math::energy`. New in
`com_math`: `gamma_q` (1 − `gamma_p` loses the small false-alarm probabilities), later
`marcum_q` (deterministic signals, fading).

- **Yücek & Arslan, COMST 11(1), 2009:** eqs. (7)–(9) and Fig. 3 (N = 15, SNR −2.5, 0,
  2.5 dB). Remark: Fig. 3 puts P_D on x and P_F on y and its curves decrease, which is not
  an ROC; the values are taken from eqs. (8)–(9).
- **Axell, Leus, Larsson, Poor, SPM 29(3), 2012:** eq. (5) and P_D(P_FA); the SNR wall
  $(\rho^{2}-1)/\rho$ (−3.33, −0.20, +4.54 dB for ρ = 1, 2, 5 dB); Fig. 1 (samples for
  P_FA = 0.05, P_D = 0.9); Fig. 4 (prefix-autocorrelation detectors) on
  `delay_correlation` and `com_dsp::cyclo::cyclic_autocorrelation`.
- **Zhao & Sadler, SPM 24(3), 2007:** a survey without values; its claim that the energy
  detector needs $O(1/\text{SNR}^{2})$ samples is a range case of the energy-detector card.

Originals to fetch: Urkowitz 1967 (doi:10.1109/PROC.1967.5573), Digham, Alouini & Simon
2007 (doi:10.1109/TCOMM.2006.887483), Tandra & Sahai 2008 (doi:10.1109/JSTSP.2007.914879).

## C. Armstrong, JLT 27(3), 2009: a card, no new code

Eq. (20) is the Bussgang gain of the envelope limiter, `com_dsp::nonlinear::clipping_bussgang`
(α = 0.9217 at a clipping ratio of 3 dB). Fig. 13 (BER of 4- and 16-QAM at clipping ratios
of 3, 6, 9 dB, Eb/N0 0–15 dB, within 20 %; the number of subcarriers is not stated) with
`com_waveform::papr::Clip` and `com_mapping::theory::ber`; "1 sample in 1000 more than 8 dB
above the mean" with `com_waveform::papr::ccdf_nyquist`. Figs. 15–16 (phase noise, offset)
once `ici_power` of A exists. Small.

## D. Dissanayake & Armstrong, JLT 31(7), 2013: IM/DD OFDM (ACO, DCO, ADO)

Fills the gap "no IM/DD modulation" of `validation/maturity.yaml`. Cases: eqs. (5), (7)
(DCO optical and electrical power), (10)–(13) (ACO), (30) (ADO, by quadrature); the DCO
optima at biases of 4.42, 7.16, 8.99, 9.95 dB for 4- to 256-QAM (Fig. 4); points G (21.85
dB, 1.9 dB below 64-QAM DCO) and E (3.68 dB below 1024-QAM ACO); Table II; the optimum at
0.7 ACO power and 3.9 dB bias (Fig. 9); Figs. 3–4 at BER 10⁻³ within ±0.5 dB. Feature: a
real Hermitian-symmetric OFDM kernel factored out of
`com_waveform::ce_ofdm::CeOfdm::ofdm_signal` (both use it), a module
`com_waveform::ofdm::imdd` with `Aco`, `Dco { bias_db }`, `Ado` (`modulate`,
`demodulate`, `optical_power`, `electrical_power`), reusing
`com_dsp::signal::{extend_with_prefix, strip_prefix}`, `com_math::{qfunc, simpson}`,
`com_mapping::theory`; a real-Gaussian Bussgang gain
`com_dsp::nonlinear::clipping_bussgang_real(lower, upper)`. Fetch: Armstrong & Schmidt
2008 (doi:10.1109/LCOMM.2008.080193), Armstrong & Lowery 2006 (doi:10.1049/el:20063636).
Medium to large.

## E. Vannucci, Serena, Bononi, JLT 20(7), 2002, doi:10.1109/JLT.2002.800376: regular perturbation

Cases: the zero-dispersion closed forms (28), (29), (35) (exact, CI); Table I; the NSD of
eq. (27) printed on Fig. 4 (first order, 100 ps/nm/span, 8 dBm: 1.2e-2, 4.0e-2, 3.3e-1,
8.5e-2) and Fig. 6 (enhanced RP, 11 dBm: 3.9e-3, 6.9e-3, 7.7e-2, 2.1e-2), within a factor
2 (the paper's reference solver is commercial and its filter and compensation placement
are not given); Table II as ranges. Feature: `com_math::nlse::RegularPerturbation`
(`new(medium, step_km, n, fs)`, `run(x, length_km, p0) -> (A0, A1)`, `p0` for the enhanced
form) on `Medium::linear_response`, `com_math::fft::{plan, angular_frequencies}`,
`com_math::effective_length`; the NSD is `com_metrics::errors::mse` over `com_math::power`.
Medium.

## F. Falconer et al., Commun. Mag. 40(4), 2002: SC-FDE

Cases: the overheads 2P/(D+2P), 3P/(D+3P) (exact); Figs. 10–11 (SUI-5, the (133, 171)
code at rates 1/2 to 7/8, blocks of 512, ±1 dB; the symbol rate is not stated, so placing
the 5 and 10 µs taps needs an assumption); the claims of the text as ranges. Feature
(`com_dsp::equalizer`): `fde_weights(h, noise_var, mmse)`, the output SNRs `fde_snr_zf`,
`fde_snr_mmse`, `dfe_snr_ideal`, `matched_filter_bound`; the link reuses
`Ofdm { dft_spread: true }` (refused today by `ofdm/link.rs`), `Taps::frequency_response`,
`ConvolutionalCode::k7`, the IEEE 802.11 puncturers, `Interleaver::block`,
`com_mapping::demapper`. Fetch: Sari, Karam & Jeanclaude 1995 (doi:10.1109/35.350382), Salz
1973 (doi:10.1002/j.1538-7305.1973.tb02023.x). Medium to large.

## Recommended order

1. A (the shared kernel, Moose, van de Beek): the most value for the least work, next to
   the existing Schmidl & Cox code; most cases are exact and run in CI.
2. B (energy detector, SNR wall; Axell Fig. 4 reuses A's kernel).
3. C (Armstrong 2009): a card on existing code, plus `ici_power` of A.
4. D (IM/DD OFDM): a new waveform with strong closed forms; removes a recorded gap.
5. E (regular perturbation): an analysis tool next to the split-step and GN models.
6. F (SC-FDE): the largest, with the least precise cases.
