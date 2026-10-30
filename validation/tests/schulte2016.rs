//! The constant-composition distribution matcher (`comrs::fec::ccdm`) against P. Schulte
//! and G. Böcherer, "Constant composition distribution matching," *IEEE Trans. Inf.
//! Theory*, 62(1), 430–434, 2016 (arXiv:1503.05133v1), Fig. 2: the rates and the
//! normalised divergences of the matcher, deterministic.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::fec::ccdm::Ccdm;
use comrs::math::entropy;

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
            let a = dm.match_bits(&bits).unwrap();
            assert_eq!(dm.dematch(&a).unwrap(), bits);
        }
    }
}
