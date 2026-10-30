//! Memory-polynomial predistortion by indirect learning (`comrs::dsp::compensation::dpd`)
//! of the memory-polynomial amplifier of `comrs::hardware::amplifier::MemoryPolynomial::ding2004`
//! against L. Ding, G. T. Zhou, D. R. Morgan, Z. Ma, J. S. Kenney, J. Kim and C. R.
//! Giardina, "A robust digital baseband predistorter constructed using memory
//! polynomials," *IEEE Trans. Commun.* 52(1), 159–165, 2004, Example 2: the amplifier of
//! eqs. (11)–(12) (the nine coefficients of the library), a three-carrier WCDMA-like input
//! (three RRC QPSK carriers of roll-off 0.22, 5/3.84 chip rates apart, ±0.24 of the
//! normalised band as in the figures), the predistorter with $Q = 2$ (three taps) and
//! $K = 5$. The paper gives no drive level: the input is set so that the spectral regrowth
//! of the amplifier alone is that of Fig. 4(a), about −43 dB between the normalised
//! frequencies 0.27 and 0.43; then
//!
//! - the odd-order predistorter brings it down to about −72 dB, Fig. 4(c) (here −70.3
//!   and −72.1 dB from −42.3 and −43.0; a gain of at least 25 dB checked);
//! - with the even orders too, "an additional 3–5 dB suppression can be achieved"
//!   (Fig. 4(d)): here 3.1 dB near the band and 6.2 dB further out (2 to 8 dB checked);
//! - a memoryless predistorter ($Q = 0$) stops near −49 dB, Fig. 4(b) (more than 10 dB
//!   above the memory polynomial checked).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::dsp::compensation::dpd::MemoryPolynomialDpd;
use comrs::dsp::filter::{PulseShape, convolve_full, pulse_shape};
use comrs::dsp::spectrum::welch;
use comrs::dsp::window::hann;
use comrs::hardware::amplifier::MemoryPolynomial;
use comrs::prelude::*;
use std::f64::consts::PI;

const SPS: usize = 16;

/// Three RRC QPSK carriers, unit power, then scaled to the rms `rms`.
fn three_carriers(n_sym: usize, rms: f64, rng: &mut Rng) -> Vec<C64> {
    let h: Vec<C64> =
        pulse_shape(PulseShape::Rrc { rolloff: 0.22 }, SPS, SPS * 40 + 1).unwrap().into_iter().map(C64::from).collect();
    let spacing = 5.0 / 3.84 / SPS as f64;
    let qpsk = Constellation::psk(4).unwrap().normalized();
    let mut x = vec![C64::new(0.0, 0.0); n_sym * SPS];
    for k in [-1.0, 0.0, 1.0] {
        let mut up = vec![C64::new(0.0, 0.0); n_sym * SPS];
        for i in 0..n_sym {
            up[i * SPS] = qpsk.points()[rng.below(4)];
        }
        for (n, v) in convolve_full(&h, &up).into_iter().take(n_sym * SPS).enumerate() {
            x[n] += v * C64::from_polar(1.0, 2.0 * PI * k * spacing * n as f64);
        }
    }
    let p = x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64;
    x.iter().map(|v| v * (rms / p.sqrt())).collect()
}

/// The regrowth (dB below the peak of the density) between the normalised frequencies
/// `lo` and `hi` (Nyquist = 1), on both sides.
fn regrowth_db(y: &[C64], lo: f64, hi: f64) -> f64 {
    let (f, d) = welch(y, 2.0, 1024, Some(&hann(1024)), 512).unwrap();
    let peak = d.iter().copied().fold(0.0, f64::max);
    let band: Vec<f64> = f.iter().zip(&d).filter(|(f, _)| (lo..hi).contains(&f.abs())).map(|(_, d)| *d).collect();
    lin2db(band.iter().sum::<f64>() / band.len() as f64 / peak)
}

#[test]
fn ding_2004_example_2_fig_4() {
    let mut rng = Rng::new(2004);
    let x = three_carriers(4000, 0.2, &mut rng);
    let pa = MemoryPolynomial::ding2004();
    let amplify = |z: &[C64]| Ok(pa.amplify(z));
    let near = |y: &[C64]| regrowth_db(y, 0.27, 0.33);
    let far = |y: &[C64]| regrowth_db(y, 0.37, 0.43);
    let linearised = |orders: Vec<u32>, memory: usize| {
        let mut dpd = MemoryPolynomialDpd::new(orders, memory);
        dpd.train_indirect(&x, amplify, pa.linear_gain(), 3, 1e-9).unwrap();
        pa.amplify(&dpd.apply(&x))
    };
    let alone = pa.amplify(&x);
    let (a_near, a_far) = (near(&alone), far(&alone));
    assert!((-46.0..-40.0).contains(&a_near) && (-46.0..-40.0).contains(&a_far), "(a): {a_near} {a_far}");
    let odd = linearised(vec![1, 3, 5], 3);
    let (c_near, c_far) = (near(&odd), far(&odd));
    assert!(c_near < a_near - 25.0 && c_far < a_far - 25.0, "(c): {c_near} {c_far}");
    let all = linearised(vec![1, 2, 3, 4, 5], 3);
    let (d_near, d_far) = (near(&all), far(&all));
    for gain in [c_near - d_near, c_far - d_far] {
        assert!((2.0..8.0).contains(&gain), "(d): {d_near} {d_far} against {c_near} {c_far}");
    }
    let memoryless = linearised(vec![1, 3, 5], 1);
    assert!(near(&memoryless) > c_near + 10.0, "(b): {}", near(&memoryless));
}
