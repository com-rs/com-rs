//! The blocks `fec_encode` and `fec_decode`: a code with its bit interleaver.

use super::codes::{Code, build_code};
use crate::blocks::{Block, Coding, Frame};
use com_core::rng::Rng;
use com_core::{Result, ensure};
use com_fec::block::BlockCode;
use com_fec::interleaving::Interleaver;

block_params! {
    /// The bit interleaver of a coded link: the parameters shared by `fec_encode` and
    /// `fec_decode`, which must give the same.
    #[derive(Clone, Copy)]
    struct InterleaverSettings {
        interleave: bool [false], "random bit interleaver over the coded bits of the trial (BICM), the same in fec_encode and fec_decode";
        seed: u64 [0], "seed of the interleaver";
    }
}

block_params! {
    pub(super) struct EncodeSpec {
        code: Code [required], "the code: {type: conv|rs|turbo|ldpc|polar, …} and its parameters";
        blocks: usize [1] >= 1, "codewords per trial";
        interleaver: InterleaverSettings [flatten], "the bit interleaver";
    }
}

block_params! {
    pub(super) struct DecodeSpec {
        code: Code [required], "the code of fec_encode, with the decoder settings";
        interleaver: InterleaverSettings [flatten], "the bit interleaver of fec_encode";
    }
}

/// A code with its interleaver.
struct Coder {
    code: Box<dyn BlockCode + Send + Sync>,
    blocks: usize,
    interleaver: InterleaverSettings,
}

impl Coder {
    fn new(code: &Code, blocks: usize, interleaver: InterleaverSettings) -> Result<Coder> {
        Ok(Coder { code: build_code(&code.0)?, blocks, interleaver })
    }

    fn interleaver(&self, len: usize) -> Option<Interleaver> {
        self.interleaver.interleave.then(|| Interleaver::random(len, self.interleaver.seed))
    }
}

pub(super) struct Encode(Coder);

impl Encode {
    pub(super) fn new(q: &EncodeSpec) -> Result<Encode> {
        Ok(Encode(Coder::new(&q.code, q.blocks, q.interleaver)?))
    }
}

impl Block for Encode {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let c = &self.0;
        let payload: Vec<u8> = (0..c.code.k() * c.blocks).map(|_| rng.bit()).collect();
        let mut bits = c.code.encode_stream(&payload)?;
        if let Some(il) = c.interleaver(bits.len()) {
            bits = il.interleave(&bits);
        }
        *s = Frame {
            trial: s.trial,
            bits,
            coding: Coding { payload, n_blocks: c.blocks, code_rate: Some(c.code.code_rate()), ..Coding::default() },
            sps: 1.0,
            symbol_rate: 1.0,
            ..Frame::default()
        };
        Ok(())
    }
}

pub(super) struct Decode(Coder);

impl Decode {
    pub(super) fn new(q: &DecodeSpec) -> Result<Decode> {
        Ok(Decode(Coder::new(&q.code, 1, q.interleaver)?))
    }
}

impl Block for Decode {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let c = &self.0;
        let n = c.code.n() * s.coding.n_blocks.max(1);
        ensure(s.coding.llr.len() >= n, || {
            format!("fec_decode needs {n} LLRs (a demap block before it), the signal has {}", s.coding.llr.len())
        })?;
        let llr = &s.coding.llr[..n];
        let llr = match c.interleaver(n) {
            Some(il) => il.deinterleave(llr),
            None => llr.to_vec(),
        };
        s.coding.decoded = c.code.decode_stream(&llr)?;
        Ok(())
    }
}
