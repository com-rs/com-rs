//! # com-fec — channel coding
//!
//! Codes take information bits and return coded bits; decoders take log-likelihood
//! ratios (positive for a 0, the convention of the whole library) and return the
//! decided information bits. Every code implements [`block::BlockCode`] and the verb
//! traits [`Encode`](com_core::traits::Encode) and [`Decode`](com_core::traits::Decode).
//!
//! | Module | Content |
//! |---|---|
//! | [`crc`] | cyclic redundancy checks of 5G NR (CRC24A/B/C, CRC16, CRC11, CRC6) |
//! | [`scrambling`] | the Gold sequence of 3GPP TS 38.211; binary LFSR sequences ([`scrambling::lfsr`]) |
//! | [`interleaving`] | block, random and QPP interleavers |
//! | [`conv`] | convolutional codes: trellis, Viterbi (soft and hard), BCJR (log-MAP, max-log) |
//! | [`turbo`] | turbo codes (two $(13, 15)_{8}$ recursive codes, LTE) with iterative decoding |
//! | [`ldpc`] | LDPC codes: progressive edge growth, repeat-accumulate codes, `alist` files, belief propagation, (normalised and offset) min-sum, layered scheduling |
//! | [`ccdm`] | constant-composition distribution matching (uniform bits to a shaped symbol sequence) |
//! | [`bch`] | shortened binary BCH codes (the outer code of DVB-S2) with Berlekamp–Massey decoding, over the fields of [`com_math::gf2m`] |
//! | [`rs`] | Reed–Solomon codes: systematic encoding, errors-and-erasures decoding, shortening, closed-form error rates |
//! | [`puncture`] | puncturing of the convolutional and turbo codes (802.11, DVB-S patterns), free distance of punctured codes |
//! | [`polar`] | polar codes: Gaussian-approximation construction, SC and CRC-aided SC-list decoding |
//! | [`bounds`] | the capacity of the binary-input AWGN channel and the normal approximation of the error probability at finite length |
//! | [`block`] | the [`BlockCode`](block::BlockCode) interface, BPSK LLRs, hard decisions |

mod algebraic;
pub mod bch;
pub mod block;
pub mod bounds;
pub mod ccdm;
pub mod conv;
pub mod crc;
pub mod interleaving;
pub mod ldpc;
pub mod polar;
pub mod puncture;
pub mod rs;
pub mod scrambling;
pub mod turbo;
