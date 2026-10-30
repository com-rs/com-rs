# Review of com-fec, com-mapping, com-mimo — 30 October 2026

Read-only review; each finding was checked against the code by the reviewer. Paths and line numbers refer to commit 21af3f7.

**Slice review: com-fec, com-mapping, com-mimo.** Read-only; no file was edited. Every finding below was checked against the code (file:line). Most important first.

1. **Reusability, high, M–L. There is no shared soft decoder.**
 - `LdpcCode::decode_llr` (ldpc.rs:535) returns the a posteriori LLRs of all n coded bits plus the iteration count.
 - `conv::bcjr` (conv.rs:237) is a free function with positional `terminated, exact` and returns the input LLRs, tail included.
 - `PolarCode::decode_list` (polar.rs:595) returns hard candidates only.
 - `TurboCode` (turbo.rs:196-200) and `NrLdpc` (nr/ldpc.rs:206) compute a posteriori LLRs and throw them away. `Punctured<C>` cannot pass soft output through.
 - Fix: add `trait SoftDecode: BlockCode { fn app_into(&self, llr, apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize> }` in `com_fec::block`. Implement it for LDPC, conv (BCJR), turbo, `NrLdpc` and `Punctured<C: SoftDecode>`. Keep a separate `ListDecode`.

2. **Correctness, high, S–M. Public fields bypass the constructors.**
 - Replacing `NrLdpc.code` (nr/ldpc.rs:74) or `NrPolar.code` (nr/polar.rs:81) leaves `map`/`shortened` pointing out of range, which panics.
 - `TurboCode.trellis` (turbo.rs:46): `constituents` (132-148) hardwires 2 outputs per step.
 - `Trellis` fields (conv.rs:54-65): `n_out = 0` divides by zero in `viterbi` (188).
 - `Interleaver.perm` (interleaving.rs:29) is never checked to be a permutation.
 - `Punctured`/`PuncturedTurbo` fields (puncture.rs:270-313): `n()` hides the error with `unwrap_or(0)` (292, 352).
 - Fix: private fields with getters, and a `decoder_mut()` that exposes only the settings (rule, layered, iterations, list).

3. **Redundancy, medium, M. The bit-wise log-sum-exp over label halves is written four times.**
 - demapper.rs:99-136 (`llr_kernel`).
 - theory.rs:377-400 (`bit_expectations`).
 - com-metrics information.rs:296-312 (`bmd_at`'s `lse`, which is `Constellation::llr` at variance nv/s).
 - The max-log form in `update_bits`/`max_log` of detection.rs.
 - Fix: one public kernel in `com_mapping::demapper` (per-bit L0/L1 from a metric row, exact or max). com-metrics calls it, which ADR 0007 allows.

4. **Performance / separation of concerns, medium, S. Alamouti decoding rebuilds the code every block.**
 - `Receiver::decode_alamouti` (receiver.rs:333) calls `Ostbc::alamouti()` per block, and `from_generator` reruns three random orthogonality checks each time.
 - `Ostbc::decode` (stbc.rs:246) clones H per block, and `decode_blocks` recomputes `equivalent(h)` per block.
 - Both run per resource-element pair (ofdm/link.rs:219, com-spec wireless.rs:662).
 - Fix: store the `Ostbc` in `Receiver`, and compute the equivalent channel once when H is constant.

5. **Consistency / correctness, medium, S. `Coherent<D>` keeps its own copy of the constellation.**
 - Its `pub constellation` (detection.rs:774) duplicates the detector's. A mismatch gives wrong points or an index panic in `detect_into` (780).
 - Fix: add `MimoDetector::constellation()` and drop the field.

6. **Separation of concerns, medium, M. `Receiver` special-cases the linear detectors.**
 - `Receiver::new` (receiver.rs:234) repeats the Zf/Mmse→`Linear` mapping of `DetectorKind::detector` just to get soft estimates.
 - AMP/OAMP's decoupled estimate `z` and its variance (`Trace`) are discarded: the receiver outputs decided points with the channel noise variance.
 - `has_soft_output` (receiver.rs:178) is a hand-kept list next to the trait's default `llr` error.
 - The module doc (receiver.rs:11) omits AMP/OAMP.
 - Fix: add `MimoDetector::estimate() -> Option<…>` and a `has_soft_output()` method on the trait.

7. **Redundancy, medium, S–M. Rate-matching selection is written several times.**
 - LLR accumulation over `map` appears in nr/ldpc.rs:199-202 and nr/polar.rs:274-277. Selection by `map` appears twice on the encode side. `Puncturer::depuncture` is a third variant.
 - The NR bit interleaver (nr/ldpc.rs:158-160) is exactly `Interleaver::block(qm, e/qm)` (interleaving.rs:35).
 - `KNOWN = 30.0` is defined twice (nr/ldpc.rs:58, nr/polar.rs:65), and `KBest` has its own `llr_clip: 30.0` (detection.rs:683).
 - Fix: a `BitSelection` type (`select`, `accumulate_into`) and one known-bit LLR constant in `block.rs`.

8. **Placement, medium, S. BPSK lives in com-fec.**
 - `com_fec::block::{bpsk, bpsk_llr}` (block.rs:156-163) contradicts the crate's "Must not contain modulation" and the settled row "BPSK-over-AWGN LLRs → `com_validation`".
 - validation/src/lib.rs:24 and com-systems dvbs2/fec.rs:359 import them from com-fec.
 - lib.rs:23 still says `block` holds "hard decisions"; they moved to com-core.

9. **Consistency, medium, S. The same error gets different variants.**
 - A y/H length mismatch is `Error::Config` in message_passing.rs:478 but `Shape` in detection.rs:118. The `MimoDetector::detect` docs promise `Shape`, so Amp/Oamp contradict them.
 - Variants are retargeted with `ensure(..).map_err(..)` at conv.rs:188-189 (should use `ensure_shape`) and stbc.rs:277.

10. **Consistency, medium, M. Argument order (n, k) versus (k, n) also differs in Rust.**
  - (n, k): `PolarCode::new`, `LdpcCode::peg`, `normal_approximation`.
  - (k, n): `NrLdpc::new(k_info, e)`, `NrPolar::uplink(a, e)`.
  - Inside rs.rs, the constructor is `shortened(m, t, n)` while the closed forms are `word_error_probability(n, t, p_s)` and `bit_error_probability(n, t, m, p_s)` (467+).

11. **Consistency, medium, S. One choice, opposite boolean polarity.**
  - `exact: bool` in `max_star` (com-math lib.rs:908), `bcjr` and `TurboCode`; `max_log: bool` in `Constellation::llr` (demapper.rs:55). com-spec converts with `exact: !max_log` (coding.rs:275).
  - Fix: a `Jacobian { Exact, MaxLog }` enum in com-math.

12. **Silent wrong result, medium, S. Invalid probabilities give a zero error rate.**
  - `word_error_probability` and `symbol_error_probability` return 0.0 for `p_s` outside [0, 1] (rs.rs:468, 488).
  - Sibling closed forms disagree on invalid input: `Result` (`ber_rayleigh_mrc`, theory.rs:186) versus a bare f64 (imperfect_csi.rs, `ber_dbpsk_coherent`).

13. **Reusability / performance, medium, S. The demapper cannot write in place.**
  - `Demap::demap_into` (demapper.rs:164-173) allocates through `llr()` and copies.
  - The trait cannot carry priors or max-log, so com-spec's `Demap` block bypasses it.
  - `log_priors_of` allocates per call; it runs for every channel use of `Linear` and `Receiver`.
  - Fix: `llr_into(.., out: &mut [f64])`.

14. **Maintainability, medium, M. Long functions.**
  - `NrPolar::build` (nr/polar.rs:117-235): 118 lines, 8 parameters with an `#[allow(too_many_arguments)]`.
  - `ListDecoder::decode` (polar.rs:319-433): 115 lines.
  - `decode_erasures` (rs.rs:272-359): 88 lines.
  - `parse_alist` (ldpc.rs:619-704): 86 lines.
  - `peg` (ldpc.rs:296-347): eight nesting levels.
  - Fix: split into sub-block interleaver, frozen set, PC positions and triangular interleaver; pull out a `farthest_checks` BFS.

15. **Redundancy / consistency, low–medium, S. AMP and OAMP repeat each other.**
  - Their `MimoDetector` impls are copy-pasted (message_passing.rs:748-783).
  - `domain` changes position between `lama`/`oamp` (5th of 7 arguments), `oamp_state_evolution` (7th) and `amp_state_evolution` (5th).
  - Fix: a config struct shared by all four.

16. **Redundancy, low, S. Small repeated formulas and copies.**
  - The MMSE bias/variance formula and its 1e-300 floors appear twice (detection.rs:201-203, 235).
  - `Eigenmode::precode_into` clones the precoder on every call (precoding.rs:125).

17. **Redundancy, low, S. GF(2) and GF(2^m) polynomial work is duplicated.**
  - Remainder by an LFSR is written twice: `Crc::checksum` (crc.rs:125-138, up to 64 bits) and `Bch::encode` (bch.rs:114-146, multi-word). Fix: `com_math::gf2::poly_rem`.
  - The product of linear factors is written twice in rs.rs (163-172, 310-319).

18. **Redundancy, low, S. Bit packing and Gray tables are re-implemented.**
  - Packing is written four times: `rs::pack_symbols`/`unpack_symbols` (rs.rs:414-427), `indices_to_bits` (constellation.rs:123), `Constellation::indices` (388) and `Dpsk::indices` (differential.rs:196).
  - `Dpsk` rebuilds Gray tables that are `com_math::gray` and `gray_inverse`.

19. **Redundancy / performance, low, S. Quadrature and polar construction.**
  - `bounds::moments` (bounds.rs:66-80) scales Gauss–Hermite nodes by hand; `com_math::gaussian_nodes` already does this.
  - `polar::reliability` (93-101) runs a 100-step bisection n log n times for 2n−1 distinct values.

20. **Maintainability, low, S. Unnamed tuples.**
  - `TheorySetup` is a 4-tuple (theory.rs:202), and `table_i` holds 5-tuples (imperfect_csi.rs).
  - (estimates, variances) comes in three shapes: `Linear::estimate`, `Ostbc::decode` and `SoftOutput`. Fix: one `Estimates` struct.

21. **Consistency, low, S. Counts are validated unevenly.**
  - `PolarCode::with_list(0)` silently becomes 1 (polar.rs:533); turbo applies `.max(1)` to its iterations (186).
  - `KBest::new(c, 0)` is accepted, then `k` is re-checked on every call (detection.rs:687), against "hot loops never re-validate". Amp/Oamp re-check their iterations on every call too.
  - `ConvolutionalCode::k7(0)` and `TurboCode::lte(0)` are accepted; com-spec re-checks (coding.rs:257, 274).

22. **Maintainability, low, S. Magic numbers and inconsistent invariant panics.**
  - Magic numbers: the tanh clamp 0.999_999_999_999 (ldpc.rs:500), the turbo interleaver seed `0x7` (turbo.rs:114), and 1e-300 floors scattered across the slice.
  - Invariant panics come in two styles: `unwrap_or_else(unreachable!)` (conv.rs:347, turbo.rs:121, puncture.rs) and `#[allow] .expect` (stbc.rs:134, constellation.rs).

23. **Maintainability, low, S. LDPC per-frame waste.**
  - `decode_llr` rebuilds the edge offsets every frame (ldpc.rs:543-547).
  - `encode_block` re-verifies Hc = 0 on every encode and returns `Error::Numerical` for what is an internal invariant (750).

24. **Consistency, low, S. Naming and signatures.**
  - `Constellation::modulate`/`demodulate` and `Dpsk::modulate` map bits to symbols, which the `Modulate` trait reserves for symbols to signal.
  - `Ccdm::matched` pairs with `dematch`.
  - `dvbs_*` puncturers duplicate the `ieee80211_*` ones.
  - `Receiver::layers` takes `&CMat` and `decode_alamouti` takes `&Array2`, while detectors take `ArrayView2`.
  - com-mimo lib.rs's capacity row omits the ergodic, outage and Wishart functions.

**What is good:**
- No layer violation; the ADR 0008 edge is the only cross-crate use.
- Every formula and reference is in the rustdoc, and literature tests carry the paper's name (`hu_2005_peg`, `cho_2002_table_1`, Polyanskiy Table I).
- Some kernels are already shared well: Berlekamp–Massey and Chien between BCH and RS (`algebraic.rs`), the `container_traits!` macro, GF(2) elimination on `gf2::Matrix`, and the BCJR workspace reused across turbo iterations.
- The polar decoder is a clean Tal–Vardy lazy-copy implementation, and the detectors factor the channel once per block.
- `DetectorKind` takes its names from one source.
