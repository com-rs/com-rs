use crate::blocks::Frame;
use crate::blocks::tests::run;
use serde_json::json;

fn max_error(s: &Frame) -> f64 {
    assert_eq!(s.samples.len(), s.symbols.len());
    s.samples.iter().zip(&s.symbols).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max)
}

#[test]
fn noiseless_siso_ofdm_over_multipath_is_exact() {
    for est in ["perfect", "ls", "lmmse"] {
        let s = run(
            &[
                json!({"type": "source", "constellation": "16-QAM", "n_symbols": 1000}),
                json!({"type": "ofdm_tx", "n_fft": 64, "n_subcarriers": 48, "n_cp": 8, "pilots": {"pattern": "block", "period": 2}, "n_symbols": 4}),
                json!({"type": "tdl", "profile": "EPA"}),
                json!({"type": "ofdm_rx", "estimation": est, "detector": "zf"}),
            ],
            1,
        );
        // pilot symbols on every subcarrier and a static channel: LS is exact; LMMSE smooths a little
        let tol = if est == "lmmse" { 1e-3 } else { 1e-9 };
        assert!(max_error(&s) < tol, "{est}: {}", max_error(&s));
    }
}

#[test]
fn noiseless_mimo_ofdm_with_orthogonal_pilots_is_exact() {
    let s = run(
        &[
            json!({"type": "source", "constellation": "QPSK", "n_symbols": 2000}),
            json!({"type": "mimo_tx", "n_tx": 2}),
            json!({"type": "ofdm_tx", "n_fft": 64, "n_subcarriers": 48, "n_cp": 16, "pilots": {"pattern": "block", "period": 2}, "n_symbols": 4}),
            json!({"type": "tdl", "profile": {"rms_delay_s": 2e-7, "resolution_s": 1e-6}, "n_rx": 2}),
            json!({"type": "ofdm_rx", "estimation": "perfect", "detector": "ml"}),
        ],
        2,
    );
    assert!(max_error(&s) < 1e-9);
}

#[test]
fn the_snr_is_set_per_resource_element() {
    let s = run(
        &[
            json!({"type": "source", "constellation": "QPSK", "n_symbols": 40_000}),
            json!({"type": "ofdm_tx", "n_fft": 256, "n_subcarriers": 160, "oversampling": 2, "pilots": "none"}),
            json!({"type": "awgn", "snr_db": 10.0}),
            json!({"type": "ofdm_rx", "detector": "zf"}),
        ],
        3,
    );
    let mse: f64 =
        s.samples.iter().zip(&s.symbols).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / s.samples.len() as f64;
    assert!((mse / 0.1 - 1.0).abs() < 0.05, "{mse}");
    assert!((s.receiver.symbol_noise[0] / 0.1 - 1.0).abs() < 0.02);
}

#[test]
fn alamouti_over_flat_fading_is_exact_without_noise() {
    let s = run(
        &[
            json!({"type": "source", "constellation": "16-QAM", "n_symbols": 400}),
            json!({"type": "mimo_tx", "scheme": "alamouti"}),
            json!({"type": "rayleigh", "n_rx": 1, "coherence_samples": 2}),
            json!({"type": "mimo_rx"}),
        ],
        4,
    );
    assert!(max_error(&s) < 1e-9);
}

#[test]
fn spatial_multiplexing_with_soft_outputs() {
    let s = run(
        &[
            json!({"type": "source", "constellation": "QPSK", "n_symbols": 400}),
            json!({"type": "mimo_tx", "n_tx": 2}),
            json!({"type": "rayleigh", "n_rx": 2, "coherence_samples": 1}),
            json!({"type": "awgn", "snr_db": 40.0, "reference": "unit"}),
            json!({"type": "mimo_rx", "detector": "ml", "llr": true}),
        ],
        5,
    );
    assert_eq!(s.coding.llr.len(), 800);
    let hard: Vec<u8> = com_core::signal::hard_decisions(&s.coding.llr);
    assert_eq!(hard, s.bits);
}

#[test]
fn blind_classification_in_a_link() {
    // the likelihood at moderate SNR, the cumulants (normalised by the received power)
    // at high SNR; M2M4 degrades for QAM at high SNR (com_metrics::blind)
    for (name, snr_db, method) in
        [("QPSK", 12.0, "ml"), ("16-QAM", 12.0, "ml"), ("QPSK", 20.0, "cumulants"), ("16-QAM", 30.0, "cumulants")]
    {
        {
            let s = run(
                &[
                    json!({"type": "source", "constellation": name, "n_symbols": 2000}),
                    json!({"type": "awgn", "snr_db": snr_db, "reference": "unit"}),
                    json!({"type": "classify", "method": method}),
                ],
                8,
            );
            let value = |k: &str| s.values.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
            assert_eq!(value("classify.correct"), Some(1.0), "{name} by {method}");
            if snr_db < 20.0 {
                let snr = value("classify.snr_m2m4_db").unwrap();
                assert!((snr - snr_db).abs() < 1.5, "{name} by {method}: {snr} dB");
            }
        }
    }
}

#[test]
fn converters_in_a_link() {
    // 16-QAM through an 8-bit DAC and a 10-bit ADC at a loading of 4: every sample
    // within half a step of each converter
    let s = run(
        &[
            json!({"type": "source", "constellation": "16-QAM", "n_symbols": 400}),
            json!({"type": "dac", "bits": 8, "loading": 4.0}),
            json!({"type": "adc", "bits": 10, "loading": 4.0}),
        ],
        7,
    );
    let rms = (0.5f64).sqrt();
    let tol = 4.0 * rms * (1.0 / 256.0 + 1.0 / 1024.0) * 1.01;
    for (y, x) in s.samples.iter().zip(&s.symbols) {
        assert!((y.re - x.re).abs() <= tol && (y.im - x.im).abs() <= tol, "{y} vs {x}");
    }
}

#[test]
fn message_passing_detectors_in_a_link() {
    // 16 streams on 32 antennas, noiseless: LAMA and OAMP recover every bit
    for detector in ["amp", "oamp"] {
        let s = run(
            &[
                json!({"type": "source", "constellation": "QPSK", "n_symbols": 640}),
                json!({"type": "mimo_tx", "n_tx": 16}),
                json!({"type": "rayleigh", "n_rx": 32, "coherence_samples": 1}),
                json!({"type": "awgn", "snr_db": 60.0, "reference": "unit"}),
                json!({"type": "mimo_rx", "detector": detector, "iterations": 20, "llr": true}),
            ],
            6,
        );
        let hard: Vec<u8> = com_core::signal::hard_decisions(&s.coding.llr);
        assert_eq!(hard, s.bits, "{detector}");
    }
}
