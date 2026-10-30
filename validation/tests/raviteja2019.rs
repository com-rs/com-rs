//! OTFS (`comrs::waveform::otfs`) through the delay–Doppler channel of
//! `comrs::channel::multipath::DelayDoppler` against P. Raviteja, Y. Hong, E. Viterbo and
//! E. Biglieri, "Practical pulse-shaping waveforms for reduced-cyclic-prefix OTFS," *IEEE
//! Trans. Veh. Technol.* 68(1), 2019
//! (`references/multicarrier/2019_raviteja_pulse-shaping-reduced-cp-otfs_tvt.pdf`) and R. Hadani et
//! al., "Orthogonal time frequency space modulation," *Proc. IEEE WCNC*, 2017
//! (`references/multicarrier/2017_hadani_orthogonal-time-frequency-space-modulation_wcnc_arxiv.pdf`):
//!
//! - Theorem 1 (eq. (12)): a frame with one prefix through a channel of integer delays and
//!   Doppler shifts, demodulated, is $H_{\mathrm{eff}} \mathbf{x}$ to $10^{-10}$; with
//!   fractional Doppler shifts the effective channel is that of eq. (13), from the
//!   time-domain matrix of the realisation (`Taps::cyclic_matrix`);
//! - Sec. V of Hadani et al. (Fig. 5), qualitatively (their TDL-C channel and frame of
//!   1024 subcarriers are not simulated): with a linear MMSE equaliser on the whole
//!   frame, uncoded 4QAM over EVA at 120 km/h (4 GHz, 15 kHz, one path per tap with
//!   Jakes' Doppler) has a BER falling much faster with the SNR than one-tap CP-OFDM.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
use comrs::channel::multipath::{DelayDoppler, DelayDopplerModel, Path, Profile, max_doppler_hz};
use comrs::mimo::detection::{Linear, MimoDetector};
use comrs::prelude::*;
use comrs::waveform::otfs::Otfs;

#[test]
fn raviteja_2019_theorem_1_through_the_channel() {
    let (m, n, cp) = (16usize, 8usize, 4usize);
    let o = Otfs { n_cp: cp, ..Otfs::new(m, n).unwrap() };
    let fs = 1.0e6;
    let bin = fs / (m * n) as f64; // one Doppler bin, 1/(NT)
    let paths = vec![
        Path { gain: C64::new(0.8, 0.1), delay: 0, doppler_hz: 0.0 },
        Path { gain: C64::new(-0.3, 0.4), delay: 2, doppler_hz: bin },
        Path { gain: C64::new(0.2, -0.2), delay: 3, doppler_hz: -2.0 * bin },
    ];
    let ch = DelayDoppler::new(paths.clone(), fs).unwrap();
    let mut rng = Rng::new(1);
    let x: Vec<C64> = (0..m * n).map(|_| rng.complex_gaussian()).collect();
    let s = o.modulate(&x).unwrap();
    let r = ch.taps(s.len()).apply(std::slice::from_ref(&s)).unwrap().remove(0);
    let y = o.demodulate(&r).unwrap();
    let spec: Vec<(C64, usize, f64)> = paths.iter().map(|p| (p.gain, p.delay, p.doppler_hz)).collect();
    let h = o.effective_channel(&spec, fs).unwrap();
    for p in 0..m * n {
        let hx: C64 = (0..m * n).map(|q| h[[p, q]] * x[q]).sum();
        assert!((hx - y[p]).norm() < 1e-10, "{p}: {hx} against {}", y[p]);
    }
}

#[test]
fn fractional_doppler_follows_eq_13() {
    let (m, n, cp) = (8usize, 4usize, 3usize);
    let o = Otfs { n_cp: cp, ..Otfs::new(m, n).unwrap() };
    let ch = DelayDoppler::new(
        vec![
            Path { gain: C64::new(1.0, 0.0), delay: 0, doppler_hz: 777.0 },
            Path { gain: C64::new(0.3, -0.5), delay: 2, doppler_hz: -1234.5 },
        ],
        1.0e5,
    )
    .unwrap();
    let mn = m * n;
    let taps = ch.taps(mn + cp);
    let heff = o.transform_channel(&taps.cyclic_matrix(cp, mn).unwrap()).unwrap();
    let mut rng = Rng::new(2);
    let x: Vec<C64> = (0..mn).map(|_| rng.complex_gaussian()).collect();
    let y = o.demodulate(&taps.apply(&[o.modulate(&x).unwrap()]).unwrap().remove(0)).unwrap();
    for p in 0..mn {
        let hx: C64 = (0..mn).map(|q| heff[[p, q]] * x[q]).sum();
        assert!((hx - y[p]).norm() < 1e-10);
    }
}

/// The BER of uncoded 4QAM over EVA at `kmh` (4 GHz, 15 kHz, 32 subcarriers of
/// 480 kHz, frames of 16 symbols): OTFS with the linear MMSE detector of `com-mimo` on
/// the exact effective channel, or CP-OFDM with one-tap equalisers on the response at the
/// middle of every symbol.
fn ber(otfs: bool, kmh: f64, snr_db: f64, frames: usize, seed: u64) -> f64 {
    let (m, n, cp) = (32usize, 16usize, 4usize);
    let fs = m as f64 * 15e3;
    let fd = max_doppler_hz(kmh / 3.6, 4e9);
    let c = Constellation::qam(4).unwrap().normalized();
    let sigma2 = 1.0 / db2lin(snr_db);
    let mut rng = Rng::new(seed);
    let (mut errors, mut total) = (0usize, 0usize);
    let o = Otfs { n_cp: cp, ..Otfs::new(m, n).unwrap() };
    let model = DelayDopplerModel::new(&Profile::Eva.taps(), fd, fs).unwrap();
    let ofdm =
        comrs::waveform::ofdm::Ofdm { oversampling: 1, n_cp: cp, ..comrs::waveform::ofdm::Ofdm::new(m, m).unwrap() };
    for _ in 0..frames {
        let ch = model.draw(&mut rng);
        let bits: Vec<u8> = (0..m * n * 2).map(|_| rng.below(2) as u8).collect();
        let x = c.modulate(&bits).unwrap();
        let labels: Vec<usize> = if otfs {
            let s = o.modulate(&x).unwrap();
            let taps = ch.taps(s.len());
            let noisy: Vec<C64> = taps
                .apply(&[s])
                .unwrap()
                .remove(0)
                .iter()
                .map(|v| v + rng.complex_gaussian() * sigma2.sqrt())
                .collect();
            let y = o.demodulate(&noisy).unwrap();
            let h = o.transform_channel(&taps.cyclic_matrix(cp, m * n).unwrap()).unwrap();
            Linear::mmse(c.clone()).detect(&h.view(), &y, sigma2).unwrap()
        } else {
            let s = ofdm.modulate(&x).unwrap();
            let taps = ch.taps(s.len());
            let noisy: Vec<C64> = taps
                .apply(&[s])
                .unwrap()
                .remove(0)
                .iter()
                .map(|v| v + rng.complex_gaussian() * sigma2.sqrt())
                .collect();
            let y = ofdm.demodulate(&noisy).unwrap();
            let bins: Vec<usize> = (0..m).map(|i| ofdm.bin(i)).collect();
            let mut z = Vec::with_capacity(y.len());
            for (k, sym) in y.chunks_exact(m).enumerate() {
                let mid = k * (m + cp) + cp + m / 2;
                let resp = taps.frequency_response(mid, &bins, m);
                z.extend(sym.iter().zip(&resp).map(|(v, h)| v / h));
            }
            c.decide(&z)
        };
        let rx = comrs::mapping::constellation::indices_to_bits(&labels, 2);
        errors += rx.iter().zip(&bits).filter(|(a, b)| a != b).count();
        total += bits.len();
    }
    errors as f64 / total as f64
}

/// Hadani et al. 2017, Sec. V and Fig. 5: OTFS spreads every symbol over the whole frame
/// and collects the diversity of the paths; one-tap OFDM sees one Rayleigh gain per
/// symbol. From 10 to 20 dB the BER of OFDM falls about eightfold (4.2e-2 to 5.5e-3,
/// diversity 1), that of OTFS ninetyfold (3.4e-2 to 3.7e-4), fifteen times below OFDM
/// at 20 dB. The test checks these factors with the margin of the frames simulated: OFDM
/// between 4 and 15, OTFS above 30, OTFS at least five times below OFDM at 20 dB.
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn hadani_2017_fig_5_diversity_of_otfs() {
    let (o10, o20) = (ber(true, 120.0, 10.0, 60, 1), ber(true, 120.0, 20.0, 120, 2));
    let (f10, f20) = (ber(false, 120.0, 10.0, 200, 3), ber(false, 120.0, 20.0, 400, 4));
    assert!(f10 / f20 < 15.0 && f10 / f20 > 4.0, "OFDM {f10} → {f20}");
    assert!(o10 / o20 > 30.0, "OTFS {o10} → {o20}");
    assert!(o20 < f20 / 5.0, "at 20 dB: OTFS {o20}, OFDM {f20}");
}
