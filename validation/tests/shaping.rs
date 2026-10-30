//! Probabilistic shaping against three publications: the constant-composition
//! distribution matcher (Schulte & Böcherer 2016), probabilistic amplitude shaping on
//! the DVB-S2 LDPC codes (Böcherer, Steiner & Schulte 2015) and the mutual information
//! of shaped square QAM (Liang & Kahn 2025).
//!
//! 1. P. Schulte and G. Böcherer, "Constant composition distribution matching," *IEEE
//!    Trans. Inf. Theory*, 62(1), 430–434, 2016 (arXiv:1503.05133v1).
//! 2. G. Böcherer, F. Steiner and P. Schulte, "Bandwidth efficient and rate-matched
//!    low-density parity-check coded modulation," *IEEE Trans. Commun.*, 63(12),
//!    4651–4665, 2015 (arXiv:1502.02733v2, the numbering of its tables).
//! 3. E. M. Liang and J. M. Kahn, "Probabilistic shaping distributions for optical
//!    communications," *J. Lightw. Technol.*, 43(4), 1504–1521, 2025.
//!
//! The rates and divergences of the matcher and the information rates are
//! deterministic and always checked; the frame error rates of PAS are `#[ignore]`d
//! (`cargo test --release -p com-validation --test shaping -- --ignored --nocapture`).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::core::rng::batch_rng;
use comrs::fec::block::BlockCode;
use comrs::fec::ccdm::Ccdm;
use comrs::mapping::constellation::{Constellation, ConstellationKind};
use comrs::mapping::source::maxwell_boltzmann;
use comrs::mapping::theory;
use comrs::math::{db2lin, entropy, golden_max, lin2db};
use comrs::systems::dvbs2::fec::{CodeRate, Dvbs2Fec, FrameSize};
use comrs::systems::pas::Pas;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------------
// Schulte & Böcherer 2016: the CCDM
// ---------------------------------------------------------------------------------

/// The target distribution of Example 1 and Fig. 2 of Schulte & Böcherer (2016).
const P_FIG2: [f64; 4] = [0.0722, 0.1654, 0.3209, 0.4415];

/// The points of Fig. 2 of Schulte & Böcherer (2016): the output length $n$ (the 50
/// lengths `round(logspace(1, 4, 50))`), the normalised divergence of the ccdm (bits
/// per symbol), its rate $k/n$ and the lower bound (22) as plotted, read from the vector
/// data of the figure (arXiv:1503.05133v1, p. 3) to 5 significant digits.
const FIG2: [(usize, f64, f64, f64); 50] = [
    (10, 5.6211e-01, 1.30000, 0.12717),
    (12, 4.5337e-01, 1.33333, 0.33345),
    (13, 4.3231e-01, 1.30769, 0.41551),
    (15, 3.6260e-01, 1.33333, 0.55067),
    (18, 3.2523e-01, 1.38889, 0.70357),
    (20, 3.3367e-01, 1.35000, 0.78312),
    (23, 3.1483e-01, 1.47826, 0.87970),
    (27, 2.5482e-01, 1.48148, 0.97912),
    (31, 2.4112e-01, 1.48387, 1.05601),
    (36, 2.1837e-01, 1.55556, 1.13101),
    (41, 1.9743e-01, 1.56098, 1.18993),
    (47, 1.8105e-01, 1.55319, 1.24608),
    (54, 1.6140e-01, 1.59259, 1.29767),
    (63, 1.3875e-01, 1.61905, 1.34909),
    (72, 1.3292e-01, 1.61111, 1.38912),
    (83, 1.1846e-01, 1.63855, 1.42756),
    (95, 1.0328e-01, 1.65263, 1.46046),
    (110, 9.3195e-02, 1.65454, 1.49255),
    (126, 8.1641e-02, 1.66667, 1.51925),
    (146, 7.2417e-02, 1.68493, 1.54521),
    (168, 6.5609e-02, 1.68452, 1.56735),
    (193, 5.7426e-02, 1.69430, 1.58696),
    (222, 5.2878e-02, 1.69820, 1.60471),
    (256, 4.6190e-02, 1.70703, 1.62086),
    (295, 4.0507e-02, 1.70847, 1.63521),
    (339, 3.5527e-02, 1.71091, 1.64779),
    (391, 3.3096e-02, 1.71611, 1.65931),
    (450, 2.9084e-02, 1.72222, 1.66943),
    (518, 2.6773e-02, 1.72201, 1.67847),
    (596, 2.2593e-02, 1.72819, 1.68650),
    (687, 1.9913e-02, 1.73217, 1.69374),
    (791, 1.8162e-02, 1.73198, 1.70013),
    (910, 1.6093e-02, 1.73407, 1.70578),
    (1048, 1.4668e-02, 1.73569, 1.71084),
    (1207, 1.2734e-02, 1.73736, 1.71534),
    (1389, 1.1270e-02, 1.73866, 1.71932),
    (1600, 9.8399e-03, 1.74125, 1.72287),
    (1842, 8.7714e-03, 1.74158, 1.72601),
    (2121, 7.5769e-03, 1.74257, 1.72880),
    (2442, 6.6587e-03, 1.74324, 1.73127),
    (2812, 6.0215e-03, 1.74395, 1.73346),
    (3237, 5.3816e-03, 1.74482, 1.73539),
    (3728, 4.6698e-03, 1.74544, 1.73711),
    (4292, 4.1481e-03, 1.74604, 1.73863),
    (4942, 3.6480e-03, 1.74646, 1.73998),
    (5690, 3.2484e-03, 1.74692, 1.74117),
    (6551, 2.8398e-03, 1.74737, 1.74222),
    (7543, 2.5774e-03, 1.74771, 1.74315),
    (8685, 2.2216e-03, 1.74796, 1.74397),
    (10000, 2.0142e-03, 1.74810, 1.74470),
];

/// The $n$-type of eq. (10) of Schulte & Böcherer (2016): the composition $(n_{a})$ that
/// minimises $D(P_{\bar A} \| P_{A})$. The objective $\sum_{a} n_{a} \ln(n_{a}/p_{a})$ is
/// a sum of convex functions of the integers $n_{a}$, so adding one symbol at a time
/// where it increases it least is optimal (the solution of Böcherer & Geiger, "Optimal
/// quantization for distribution synthesis," arXiv:1307.6843, Algorithm 2).
fn paper_composition(p: &[f64], n: usize) -> Vec<usize> {
    let f = |a: usize, c: usize| if c == 0 { 0.0 } else { c as f64 * (c as f64 / p[a]).ln() };
    let mut c = vec![0usize; p.len()];
    for _ in 0..n {
        let a = (0..p.len())
            .min_by(|&a, &b| (f(a, c[a] + 1) - f(a, c[a])).total_cmp(&(f(b, c[b] + 1) - f(b, c[b]))))
            .unwrap();
        c[a] += 1;
    }
    c
}

/// $D(P \| Q)$ in bits.
fn divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q).log2()).sum()
}

/// Fig. 2 of Schulte & Böcherer (2016), arXiv:1503.05133v1: the rate $k/n$ and the
/// normalised divergence $D(P_{\tilde A^{n}} \| P_{A}^{n})/n = H(\bar A) - k/n +
/// D(P_{\bar A} \| P_{A})$ (eq. 17) of the ccdm for $P_{A} = (0.0722, 0.1654, 0.3209,
/// 0.4415)$ (Example 1) at the 50 output lengths of the figure, and the lower bound
/// (22) as plotted (with $H(A)$ in place of $H(\bar A)$). The composition is that of
/// eq. (10), $k = \lfloor \log_{2} |\mathcal{T}^{n}_{\bar P_{A}}| \rfloor$ that of
/// [`Ccdm`] (eqs. 12 and 14).
#[test]
fn schulte_bocherer_2016_fig2_ccdm_rate_and_divergence() {
    let h = entropy(&P_FIG2);
    // the horizontal line H(A) of the figure, plotted at 1.75012 bits
    assert!((h - 1.750_12).abs() < 1e-5, "{h}");
    for &(n, d_paper, r_paper, bound_paper) in &FIG2 {
        let c = paper_composition(&P_FIG2, n);
        let dm = Ccdm::new(c.clone()).unwrap();
        let pbar = dm.distribution();
        let rate = dm.rate();
        let d = entropy(&pbar) - rate + divergence(&pbar, &P_FIG2);
        // the rates are k/n: one bit more or less would move them by 1/n ≥ 1e-4
        assert!((rate - r_paper).abs() < 2e-5, "n = {n} {c:?}: rate {rate} vs {r_paper}");
        assert!((d / d_paper - 1.0).abs() < 1e-3, "n = {n} {c:?}: divergence {d} vs {d_paper}");
        let k = P_FIG2.len() as f64;
        let nf = n as f64;
        let bound = -k * (nf + k).log2() / nf + h - 1.0 / nf;
        assert!((bound - bound_paper).abs() < 2e-5, "n = {n}: bound {bound} vs {bound_paper}");
        // the matcher of com-rs indexes the set of the composition: a round trip
        if n <= 100 {
            let bits: Vec<u8> = (0..dm.k()).map(|i| u8::from(i % 3 == 1)).collect();
            let a = dm.matched(&bits).unwrap();
            assert_eq!(dm.dematch(&a).unwrap(), bits);
        }
    }
}

// ---------------------------------------------------------------------------------
// Information rates of ASK and QAM: Böcherer, Steiner & Schulte 2015 (Tables I, III);
// Liang & Kahn 2025 (Fig. 6)
// ---------------------------------------------------------------------------------

/// Gauss–Hermite nodes of the information rates.
const POINTS: usize = 40;

/// The Maxwell–Boltzmann probabilities $\propto e^{-\nu x^{2}}$ of the $M$-ASK points
/// $\pm 1, \pm 3, \dots$ (eq. 8 of Böcherer et al.), in the order of the points of
/// [`Constellation::pam`].
fn mb(m: usize, nu: f64) -> Vec<f64> {
    maxwell_boltzmann(&Constellation::pam(m).unwrap(), nu)
}

/// The rate (bits per real dimension) of `rate_fn` at an SNR in dB.
type RateFn<'a> = dyn Fn(f64) -> f64 + 'a;

/// The SNR (dB) at which the increasing `rate` reaches `target`, by regula falsi
/// (Illinois) from the capacity $10 \log_{10}(2^{2R} - 1)$, which no input beats; `None`
/// when the rate stays below the target (an entropy smaller than the target).
fn snr_for(rate: &RateFn<'_>, target: f64) -> Option<f64> {
    let mut a = lin2db(2f64.powf(2.0 * target) - 1.0);
    let mut fa = rate(a) - target;
    if fa >= 0.0 {
        return Some(a);
    }
    let mut b = a + 1.0;
    let mut fb = rate(b) - target;
    while fb < 0.0 {
        if b - a > 20.0 {
            return None;
        }
        (a, fa) = (b, fb);
        b += 2.0;
        fb = rate(b) - target;
    }
    let mut side = 0;
    for _ in 0..100 {
        let c = (a * fb - b * fa) / (fb - fa);
        let fc = rate(c) - target;
        if fc.abs() < 1e-12 || (b - a).abs() < 1e-9 {
            return Some(c);
        }
        if fc < 0.0 {
            (a, fa) = (c, fc);
            if side == -1 {
                fb /= 2.0;
            }
            side = -1;
        } else {
            (b, fb) = (c, fc);
            if side == 1 {
                fa /= 2.0;
            }
            side = 1;
        }
    }
    Some((a + b) / 2.0)
}

/// The mutual information (bits per real dimension) of $M$-ASK with the MB parameter
/// `nu` (0: uniform) at an SNR in dB.
fn mi_ask(m: usize, nu: f64, snr_db: f64) -> f64 {
    theory::mi(ConstellationKind::Pam, m, snr_db, Some(&mb(m, nu)), POINTS).unwrap()
}

/// The rate of bit-metric decoding (bits per real dimension) of $M$-ASK with the BRGC
/// labels and the MB parameter `nu` at an SNR in dB (eq. 50 of Böcherer et al.).
fn bmd_ask(m: usize, nu: f64, snr_db: f64) -> f64 {
    theory::gmi(ConstellationKind::Pam, m, snr_db, Some(&mb(m, nu)), POINTS).unwrap()
}

/// The SNR (dB) at which the MB input of the best parameter reaches `target` with the
/// rate `rate(nu, snr_db)` (eqs. 9–10 of Böcherer et al.: the best scaling, the power
/// fixed): `(SNR, nu)`.
fn snr_mb(m: usize, target: f64, rate: &dyn Fn(f64, f64) -> f64) -> (f64, f64) {
    // nu E_s(uniform) of the optimum is about 1 at every order
    let es = ((m * m - 1) as f64) / 3.0;
    let f = |ln_nu: f64| snr_for(&|s| rate(ln_nu.exp(), s), target).unwrap_or(f64::INFINITY);
    let x = golden_max(|ln_nu| -f(ln_nu), (0.05 / es).ln(), (4.0 / es).ln(), 40);
    (f(x), x.exp())
}

/// Table I of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: the SNR (dB) at
/// which the mutual information of the optimised Maxwell–Boltzmann input $X^{\clubsuit}$
/// (eqs. 8–10), of the uniform input and the capacity $\frac{1}{2}\log_{2}(1 + P)$
/// (eq. 1) reach $m - 1$ bits per channel use for $2^{m}$-ASK, and the shaping gaps
/// (printed to 4 decimals). Table III: the SNR at which the rate of bit-metric decoding
/// with the BRGC labelling (eq. 50, the labels of Table II) of $X^{\clubsuit}$ reaches
/// the same rates.
#[test]
fn bocherer_2015_table_i_and_iii_shaping_gaps_of_ask() {
    // (M, rate, SNR(X♣), SNR(uniform), gap, SNR(capacity), gap, SNR(R_BMD) of Table III)
    let table = [
        (4, 1.0, 4.8180, 5.1181, -0.3001, 4.7712, 0.0468, 4.8313),
        (8, 2.0, 11.8425, 12.6187, -0.7761, 11.7609, 0.0816, 11.8481),
        (16, 3.0, 18.0910, 19.1681, -1.0772, 17.9934, 0.0975, 18.0951),
        (32, 4.0, 24.1706, 25.4140, -1.2434, 24.0654, 0.1052, 24.1742),
        (64, 5.0, 30.2078, 31.5384, -1.3307, 30.0988, 0.1090, 30.2110),
    ];
    for (m, r, s_mb, s_unif, gap_unif, s_cap, gap_cap, s_bmd) in table {
        let (snr, nu) = snr_mb(m, r, &|nu, s| mi_ask(m, nu, s));
        let unif = snr_for(&|s| mi_ask(m, 0.0, s), r).unwrap();
        let cap = lin2db(2f64.powf(2.0 * r) - 1.0);
        let bmd = snr_for(&|s| bmd_ask(m, nu, s), r).unwrap();
        println!(
            "{m}-ASK, {r} bit: X♣ {snr:.4} dB (paper {s_mb}), uniform {unif:.4} ({s_unif}), capacity {cap:.4} \
             ({s_cap}), BMD {bmd:.4} ({s_bmd}), nu = {nu:.6}"
        );
        assert!((cap - s_cap).abs() < 6e-5, "{m}-ASK: capacity {cap} vs {s_cap}");
        assert!((snr - s_mb).abs() < 5e-4, "{m}-ASK: X♣ {snr} vs {s_mb}");
        assert!((unif - s_unif).abs() < 5e-4, "{m}-ASK: uniform {unif} vs {s_unif}");
        assert!((snr - unif - gap_unif).abs() < 1e-3, "{m}-ASK: shaping gap {} vs {gap_unif}", snr - unif);
        assert!((snr - cap - gap_cap).abs() < 1e-3, "{m}-ASK: gap to capacity {} vs {gap_cap}", snr - cap);
        assert!((bmd - s_bmd).abs() < 5e-4, "{m}-ASK: BMD {bmd} vs {s_bmd}");
    }
}

/// The finite-constellation shaping gain of square $M$-QAM (Liang & Kahn 2025,
/// Sec. III-C3): the largest difference, at a fixed mutual information, between the SNR
/// of the uniform input and that of the optimised Maxwell–Boltzmann input:
/// `(gain in dB, MI in bits per QAM symbol, SNR of the MB input in dB)`. The MB law
/// $e^{-\beta |x|^{2}}$ of square QAM is the product of the MB laws of its two
/// $\sqrt{M}$-ASK quadratures, and the SNR $E[|X|^{2}]/\sigma^{2}$ of the complex channel
/// is that of each real dimension, so the MI is twice that of $\sqrt{M}$-ASK at the same
/// SNR.
fn qam_shaping_gain(m_qam: usize) -> (f64, f64, f64) {
    let m = (m_qam as f64).sqrt().round() as usize;
    // the SNRs of the uniform and of the shaped input at the MI `mi`
    let snrs = |mi: f64| {
        let unif = snr_for(&|s| mi_ask(m, 0.0, s), mi / 2.0).unwrap();
        let (shaped, _) = snr_mb(m, mi / 2.0, &|nu, s| mi_ask(m, nu, s));
        (unif, shaped)
    };
    let bits = (m_qam as f64).log2();
    let mi = golden_max(
        |mi| {
            let (unif, shaped) = snrs(mi);
            unif - shaped
        },
        2.0,
        bits - 1.0,
        30,
    );
    let (unif, shaped) = snrs(mi);
    (unif - shaped, mi, shaped)
}

/// Eq. (27) of Liang & Kahn (2025): the ultimate shaping gain of the standard coherent
/// (SC) channel, $\gamma_{s}^{(\mathrm{SC})} = \pi e / 6 \approx 1.53$ dB (also Böcherer
/// et al. 2015, Sec. III-D, after Forney et al. 1984), and the Shannon limit of Fig. 6,
/// $\log_{2}(1 + \mathrm{SNR})$, that the MI of every input stays below.
#[test]
fn liang_kahn_2025_eq27_ultimate_shaping_gain() {
    let ultimate = lin2db(std::f64::consts::PI * std::f64::consts::E / 6.0);
    assert!((ultimate - 1.53).abs() < 5e-3, "{ultimate}");
    assert!((ultimate - 1.5329).abs() < 1e-4, "{ultimate}");
    // MI of MB and uniform 256-QAM below the Shannon limit and saturating at 8 bits
    for snr_db in [10.0, 20.0, 30.0, 50.0] {
        let shannon = (1.0 + db2lin(snr_db)).log2();
        let unif = 2.0 * mi_ask(16, 0.0, snr_db);
        assert!(unif < shannon, "{snr_db} dB: {unif} vs {shannon}");
        let shaped = 2.0 * mi_ask(16, 0.02, snr_db);
        assert!(shaped < shannon, "{snr_db} dB: {shaped} vs {shannon}");
    }
    assert!((2.0 * mi_ask(16, 0.0, 50.0) - 8.0).abs() < 1e-9);
}

/// Fig. 6 of Liang & Kahn (2025): the finite-constellation shaping gain of 256-QAM over
/// the SC (AWGN) channel, "∼ 1 dB" (red arrow and text, at a mutual information of about
/// 6 bits per symbol, Sec. III-C3).
#[test]
fn liang_kahn_2025_fig6_shaping_gain_of_256qam() {
    let (gain, mi, snr) = qam_shaping_gain(256);
    println!("256-QAM: largest shaping gain {gain:.3} dB at {mi:.2} bits ({snr:.2} dB)");
    assert!((gain - 1.0).abs() < 0.1, "{gain}");
    assert!((mi - 6.0).abs() < 0.5, "{mi}");
}

/// Fig. 6 of Liang & Kahn (2025), Sec. III-C3: the finite-constellation shaping gain
/// grows from ∼ 1 dB to ∼ 1.4 dB as $M$ grows from 256 to 65536, by smaller steps at
/// every factor of 4, and that of 65536-QAM is "∼ 0.13 dB less than the ultimate shaping
/// gain" (eq. 27), i.e. ∼ 1.40 dB (blue arrow, at about 11.5 bits per symbol).
#[test]
#[ignore = "the 128- and 256-ASK quadratures take minutes: run with --ignored"]
fn liang_kahn_2025_fig6_shaping_gains_256_to_65536_qam() {
    let start = Instant::now();
    let ultimate = lin2db(std::f64::consts::PI * std::f64::consts::E / 6.0);
    let mut gains = Vec::new();
    for m in [256, 1024, 4096, 16384, 65536] {
        let (gain, mi, snr) = qam_shaping_gain(m);
        println!("{m}-QAM: largest shaping gain {gain:.3} dB at {mi:.2} bits ({snr:.2} dB)");
        gains.push(gain);
    }
    let steps: Vec<f64> = gains.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(steps.iter().all(|s| *s > 0.0), "{gains:?}");
    assert!(steps.windows(2).all(|w| w[1] < w[0]), "{steps:?}");
    let g = gains[4];
    assert!((g - 1.4).abs() < 0.1, "{g}");
    assert!(g < ultimate, "{g}");
    println!("{:.1} s", start.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------------------------
// Böcherer, Steiner & Schulte 2015: PAS on the DVB-S2 LDPC codes (Tables VI–X)
// ---------------------------------------------------------------------------------

/// One operating point of Tables VI–X of Böcherer, Steiner & Schulte (2015),
/// arXiv:1502.02733v2: $2^{m}$-ASK, the DVB-S2 code (64 800 bits), the bit mapper $\pi$
/// without its final 1 (the signs), the spectral efficiency (bits per channel use), the
/// SNR (dB) and the frame error rate with its 95 % confidence interval; the seed of the
/// simulation.
struct Point {
    table: &'static str,
    m: usize,
    code: CodeRate,
    levels: &'static [usize],
    rate: f64,
    snr_db: f64,
    fer: f64,
    ci: f64,
    seed: u64,
}

/// The Maxwell–Boltzmann amplitude distribution $\propto e^{-\nu a^{2}}$ of $2^{m}$-ASK,
/// $a = 1, 3, \dots, 2^{m} - 1$ (eqs. 8 and 57).
fn amplitudes(m: usize, nu: f64) -> Vec<f64> {
    (0..1usize << (m - 1)).map(|a| (-nu * ((2 * a + 1) as f64).powi(2)).exp()).collect()
}

/// The PAS transceiver of a point: the DVB-S2 LDPC code of 64 800 bits (sum-product,
/// 100 iterations, Sec. IX), the bit mapper of the table, and the MB amplitudes whose
/// matcher gives the spectral efficiency of the table, $R = k_{c}/n_{c}$ (the
/// effective rate of Sec. IX-3; the rate adaption of eqs. 58–60 keeps the amplitudes
/// in the MB family).
fn pas_of(p: &Point) -> Pas {
    let mut code = Dvbs2Fec::new(FrameSize::Normal, p.code).unwrap().ldpc().clone();
    code.max_iterations = 100;
    let n_c = code.n() / p.m;
    let extra = code.k() - n_c * (p.m - 1);
    let rate = |nu: f64| (Ccdm::from_distribution(&amplitudes(p.m, nu), n_c).unwrap().k() + extra) as f64 / n_c as f64;
    // the rate decreases with nu
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if rate(mid) > p.rate { lo = mid } else { hi = mid }
    }
    let nu = if (rate(lo) - p.rate).abs() <= (rate(hi) - p.rate).abs() { lo } else { hi };
    Pas::new(code, p.m, &amplitudes(p.m, nu), 1.0).unwrap().with_bit_mapper(p.levels).unwrap()
}

/// The frame errors of PAS over frames $0, 1, \dots$ at an SNR $E[X^{2}]/\sigma^{2}$ (dB,
/// one real dimension), until `min_errors` errors or `max_frames` frames, by rounds of
/// frames shared by the threads (the count does not depend on the number of threads).
/// A frame is in error when the decoded data bits (the dematcher's and the signs') differ
/// from the sent ones; when the LDPC decoder returns the sent information bits the
/// dematcher returns the sent bits, and it runs only on the other frames. Frame `f`
/// draws its bits and its noise from `batch_rng(seed, f)`: the SNRs share their random
/// numbers.
fn frame_errors(pas: &Pas, snr_db: f64, seed: u64, min_errors: u64, max_frames: u64) -> (u64, u64) {
    let half = pas.matcher().composition().len();
    let levels: Vec<f64> = (0..2 * half).map(|i| (2 * i) as f64 - (2 * half - 1) as f64).collect();
    let es: f64 = pas.level_distribution().iter().zip(&levels).map(|(p, x)| p * x * x).sum();
    let noise_var = es / db2lin(snr_db);
    let sigma = noise_var.sqrt();
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get) as u64;
    let round = 4 * threads;
    let (mut errors, mut frames) = (0u64, 0u64);
    while errors < min_errors && frames < max_frames {
        let next = AtomicU64::new(frames);
        let wrong = AtomicU64::new(0);
        let end = frames + round;
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    loop {
                        let f = next.fetch_add(1, Ordering::Relaxed);
                        if f >= end {
                            break;
                        }
                        let mut rng = batch_rng(seed, f);
                        let bits: Vec<u8> = (0..pas.info_bits()).map(|_| rng.bit()).collect();
                        let x = pas.encode(&bits).unwrap();
                        let y: Vec<f64> = x.iter().map(|v| v + sigma * rng.gaussian()).collect();
                        let sent = pas.encode_codeword(&bits).unwrap();
                        let (post, _) = pas.code().decode_llr(&pas.llr(&y, noise_var).unwrap()).unwrap();
                        let info_ok = pas.code().info_positions().iter().all(|&i| (post[i] < 0.0) == (sent[i] == 1));
                        if !info_ok && pas.decode(&y, noise_var).unwrap() != bits {
                            wrong.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        errors += wrong.load(Ordering::Relaxed);
        frames = end;
    }
    (errors, frames)
}

/// The SNR (dB) at which the line through the points `(snr, ln FER)` reaches `target`
/// (least squares; the points without errors are left out).
fn snr_at_fer(points: &[(f64, u64, u64)], target: f64) -> Option<f64> {
    let pts: Vec<(f64, f64)> =
        points.iter().filter(|(_, e, _)| *e > 0).map(|&(s, e, n)| (s, (e as f64 / n as f64).ln())).collect();
    if pts.len() < 2 {
        return None;
    }
    let k = pts.len() as f64;
    let (ms, mf) = (pts.iter().map(|p| p.0).sum::<f64>() / k, pts.iter().map(|p| p.1).sum::<f64>() / k);
    let sxy: f64 = pts.iter().map(|p| (p.0 - ms) * (p.1 - mf)).sum();
    let sxx: f64 = pts.iter().map(|p| (p.0 - ms) * (p.0 - ms)).sum();
    let slope = sxy / sxx;
    (slope < 0.0).then(|| ms + (target.ln() - mf) / slope)
}

/// PAS on the DVB-S2 LDPC codes against one point of Tables VI–X of Böcherer, Steiner &
/// Schulte (2015): the FER of com-rs at the SNR of the table and at ±0.05 dB (and
/// between them while fewer than two points have 5 errors), and the SNR at which it
/// reaches the FER of the table (log-linear fit). Returns the difference in dB from the
/// SNR of the table.
///
/// The system is that of the paper: the DVB-S2 LDPC code alone (no BCH), sum-product
/// decoding with 100 iterations (layered here, Sec. IX does not give the schedule),
/// bit-metric decoding with the priors of the amplitudes, the bit mapper $\pi$ of the
/// table (Sec. VII-B) and the BRGC labelling of Table II. The labels of com-rs differ
/// from those of the paper by the first amplitude bit (BRGC of the amplitude index
/// instead of the descending order of Table II) and the sign bit ($0$: $+$, the
/// paper's $b(1) = 1$): both give the same code up to a fixed codeword and a reflection
/// of the channel, which leave the FER of a sum-product decoder unchanged. The
/// compositions of the matcher are those of `Ccdm::from_distribution` (largest
/// remainders), which can differ from eq. (10) of Schulte & Böcherer (2016) by one
/// amplitude.
fn check_pas_point(p: &Point) -> f64 {
    let start = Instant::now();
    let pas = pas_of(p);
    let rate = pas.rate();
    assert!((rate - p.rate).abs() < 0.005, "Table {}: rate {rate} vs {}", p.table, p.rate);
    let label = format!("Table {} ({}-ASK, {}, R = {rate:.4})", p.table, 1 << p.m, p.code.name());
    let mut points: Vec<(f64, u64, u64)> = Vec::new();
    let run = |s: f64, points: &mut Vec<(f64, u64, u64)>| {
        let (e, n) = frame_errors(&pas, s, p.seed, 30, 6000);
        println!("{label}: {s:.3} dB, FER {e}/{n} = {:.2e}", e as f64 / n as f64);
        points.push((s, e, n));
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
    };
    for ds in [-0.05, 0.0, 0.05] {
        run(p.snr_db + ds, &mut points);
    }
    // a waterfall steeper than a decade per 0.05 dB: halve the step after the last point
    // with 5 errors
    while points.iter().filter(|q| q.1 >= 5).count() < 2 && points.len() < 6 {
        let i = points.iter().rposition(|q| q.1 >= 5).unwrap_or(0);
        if i + 1 >= points.len() {
            break;
        }
        run((points[i].0 + points[i + 1].0) / 2.0, &mut points);
    }
    let at = points.iter().find(|q| (q.0 - p.snr_db).abs() < 1e-9).unwrap();
    let snr = snr_at_fer(&points, p.fer);
    let delta = snr.map_or(f64::NAN, |s| s - p.snr_db);
    println!(
        "{label}: paper FER {:.1e} ± {:.1e} at {} dB; com-rs {:.2e} there, FER {:.1e} at {:.3} dB: {delta:+.3} dB \
         ({:.0} s)",
        p.fer,
        p.ci,
        p.snr_db,
        at.1 as f64 / at.2 as f64,
        p.fer,
        snr.unwrap_or(f64::NAN),
        start.elapsed().as_secs_f64()
    );
    delta
}

/// Table X of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 4-ASK, the
/// DVB-S2 code of rate 2/3, $\pi = (2, 1)$, the practical operating point (first row):
/// 1.13 bits per channel use, FER $5.2 \cdot 10^{-3} \pm 2.6 \cdot 10^{-3}$ at 6.70 dB.
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at a FER of 5e-3: minutes in release, run with --ignored"]
fn bocherer_2015_table_x_pas_dvbs2() {
    let p = Point {
        table: "X",
        m: 2,
        code: CodeRate::R2_3,
        levels: &[2],
        rate: 1.13,
        snr_db: 6.70,
        fer: 5.2e-3,
        ci: 2.6e-3,
        seed: 100,
    };
    let delta = check_pas_point(&p);
    assert!(delta.abs() < 0.2, "{delta} dB");
}

/// Table IX of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 8-ASK, the
/// DVB-S2 code of rate 3/4, $\pi = (3, 2, 1)$, the practical operating point (first
/// row): 1.85 bits per channel use, FER $1.5 \cdot 10^{-3} \pm 8 \cdot 10^{-4}$ at
/// 11.45 dB.
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at a FER of 1.5e-3: minutes in release, run with --ignored"]
fn bocherer_2015_table_ix_pas_dvbs2() {
    let p = Point {
        table: "IX",
        m: 3,
        code: CodeRate::R3_4,
        levels: &[3, 2],
        rate: 1.85,
        snr_db: 11.45,
        fer: 1.5e-3,
        ci: 8.0e-4,
        seed: 101,
    };
    let delta = check_pas_point(&p);
    assert!(delta.abs() < 0.2, "{delta} dB");
}

/// Table VIII of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 16-ASK, the
/// DVB-S2 code of rate 5/6, $\pi = (4, 3, 2, 1)$: the practical operating point (first
/// row), 2.96 bits per channel use, FER $2.0 \cdot 10^{-2} \pm 9.8 \cdot 10^{-3}$ at
/// 18.40 dB, and the adapted point of 2.50 bits per channel use, FER $7.2 \cdot
/// 10^{-3} \pm 3.6 \cdot 10^{-3}$ at 15.51 dB (rate adaption, eq. 62).
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at FERs of 1e-2: minutes in release, run with --ignored"]
fn bocherer_2015_table_viii_pas_dvbs2() {
    let first = Point {
        table: "VIII",
        m: 4,
        code: CodeRate::R5_6,
        levels: &[4, 3, 2],
        rate: 2.96,
        snr_db: 18.40,
        fer: 2.0e-2,
        ci: 9.8e-3,
        seed: 102,
    };
    let adapted = Point { rate: 2.50, snr_db: 15.51, fer: 7.2e-3, ci: 3.6e-3, seed: 103, ..first };
    for p in [first, adapted] {
        let delta = check_pas_point(&p);
        assert!(delta.abs() < 0.2, "{} bits: {delta} dB", p.rate);
    }
}

/// Table VII of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 32-ASK, the
/// DVB-S2 code of rate 5/6, $\pi = (4, 5, 2, 3, 1)$, the practical operating point
/// (first row): 3.62 bits per channel use, FER $2.1 \cdot 10^{-3} \pm 1.1 \cdot
/// 10^{-3}$ at 22.60 dB.
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at a FER of 2e-3: a quarter of an hour in release, run with --ignored"]
fn bocherer_2015_table_vii_pas_dvbs2() {
    let p = Point {
        table: "VII",
        m: 5,
        code: CodeRate::R5_6,
        levels: &[4, 5, 2, 3],
        rate: 3.62,
        snr_db: 22.60,
        fer: 2.1e-3,
        ci: 1.1e-3,
        seed: 104,
    };
    let delta = check_pas_point(&p);
    assert!(delta.abs() < 0.2, "{delta} dB");
}

/// Table VI of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 64-ASK, the
/// DVB-S2 code of rate 9/10, $\pi = (4, 2, 5, 3, 6, 1)$, the practical operating point
/// (first row): 5.09 bits per channel use, FER $4.1 \cdot 10^{-3} \pm 2.1 \cdot
/// 10^{-3}$ at 31.80 dB.
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at a FER of 4e-3: a quarter of an hour in release, run with --ignored"]
fn bocherer_2015_table_vi_pas_dvbs2() {
    let p = Point {
        table: "VI",
        m: 6,
        code: CodeRate::R9_10,
        levels: &[4, 2, 5, 3, 6],
        rate: 5.09,
        snr_db: 31.80,
        fer: 4.1e-3,
        ci: 2.1e-3,
        seed: 105,
    };
    let delta = check_pas_point(&p);
    assert!(delta.abs() < 0.2, "{delta} dB");
}
