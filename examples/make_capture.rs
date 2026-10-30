//! Write a synthetic "measured" capture in SigMF, to exercise the processing of real
//! recordings (`specs/capture_16qam.yaml`).
//!
//! An arbitrary waveform generator repeats one period of 4096 random 16-QAM symbols at
//! 32 GBd (root-raised cosine, roll-off 0.2); the signal crosses 200 km of standard
//! fibre, the lasers have 100 kHz linewidths and the local oscillator is detuned by
//! 300 MHz; the received SNR is 18 dB; an oscilloscope samples it at 80 GS/s (2.5
//! samples per symbol, not an integer) with 16-bit integers, starting at an arbitrary
//! point of the period with a fractional delay. The capture holds 32768 samples.
//!
//! ```text
//! cargo run --release -p comrs --example make_capture [output directory]
//! ```
//!
//! writes `synthetic_16qam.sigmf-meta`, `synthetic_16qam.sigmf-data` and the period of
//! the transmitted indices `synthetic_16qam_reference.npy` (default directory `data`).

use comrs::dsp::{filter, resample, signal};
use comrs::io::{Datatype, write_npy_i64, write_sigmf};
use comrs::prelude::*;
use serde_json::json;

fn main() -> Result<()> {
    let out = std::path::PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "data".into()));
    let mut rng = Rng::new(2026);
    let (rs, period, sps_sim) = (32e9, 4096usize, 5usize);
    let fs_sim = rs * sps_sim as f64;
    let c = Constellation::qam(16)?.normalized();
    let reference: Vec<usize> = (0..period).map(|_| rng.below(16)).collect();
    // 10 periods of the waveform, cut later into the capture window
    let idx: Vec<usize> = (0..10 * period).map(|k| reference[k % period]).collect();
    let rrc = filter::pulse_shape(filter::PulseShape::Rrc { rolloff: 0.2 }, sps_sim, 4097)?;
    let mut x = signal::pnorm(&filter::fir_filter(&rrc, &resample::upsample(&c.symbols(&idx), sps_sim)));
    let fiber = FiberParams::default();
    let phi = comrs::hardware::oscillator::phase_noise(100e3, x.len(), 1.0 / fs_sim, &mut rng);
    x.iter_mut().zip(&phi).for_each(|(v, p)| *v *= comrs::math::cis(*p));
    let x = signal::pnorm(&fiber::linear_fiber_channel(&x, &fiber, 200.0, fs_sim));
    let x = awgn::awgn(&x, 18.0, fs_sim, rs, &mut rng);
    let lo = comrs::hardware::oscillator::phase_noise(100e3, x.len(), 1.0 / fs_sim, &mut rng);
    let x: Vec<C64> = x
        .iter()
        .enumerate()
        .map(|(n, v)| v * comrs::math::cis(lo[n] - 2.0 * std::f64::consts::PI * 300e6 * n as f64 / fs_sim))
        .collect();
    // the oscilloscope: a fractional delay, 80 GS/s, a window of 32768 samples
    let x = resample::delay_fft(&x, 0.37 / fs_sim, fs_sim);
    let start = 1234 * sps_sim + 3 * period * sps_sim;
    let window = &x[start..start + 65536];
    let scope = resample::resample_fft(window, 32768);
    let peak = scope.iter().map(|v| v.re.abs().max(v.im.abs())).fold(0.0, f64::max);
    let adc: Vec<C64> = scope.iter().map(|v| v * (12000.0 / peak)).collect();
    write_sigmf(
        out.join("synthetic_16qam"),
        &adc,
        Datatype::CI16,
        80e9,
        Some(&json!({
            "core:description": "synthetic capture: 32 GBd 16-QAM, 200 km SSMF, 18 dB SNR, 100 kHz lasers, 300 MHz LO offset, 80 GS/s",
            "comrs:symbol_rate": rs,
            "comrs:reference": "synthetic_16qam_reference.npy",
        })),
    )?;
    write_npy_i64(out.join("synthetic_16qam_reference.npy"), &reference.iter().map(|&i| i as i64).collect::<Vec<_>>())?;
    println!("written {} samples and a reference of {period} symbols to {}", adc.len(), out.display());
    Ok(())
}
