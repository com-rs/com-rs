//! Blocks of the standard-compliant chains of `com_systems` (ADR 0010).
//!
//! | Block | Does |
//! |---|---|
//! | `dvbs2_tx` | DVB-S2: draws the data of `frames` BBFRAMEs (generic continuous stream), encodes them (BCH, LDPC), interleaves, maps and frames them (PLHEADER, pilots, scrambling): the PLFRAME symbols at 1 sample per symbol |
//! | `dvbs2_rx` | DVB-S2: descrambles the frame-aligned PLFRAMEs, removes the headers and the pilots, demaps with the noise variance of the channel, decodes, checks the BBHEADER |
//!
//! Every symbol of a PLFRAME has unit energy: `{type: awgn, snr_db: …, reference: unit}`
//! sets $E_{s}/N_{0}$, the abscissa of Table 13 of the standard. The receiver records the
//! MODCOD errors of the PLHEADER decoder (`dvbs2_rx.plheader_errors`) and the CRC-8
//! failures of the BBHEADERs (`dvbs2_rx.crc_failures`).
//!
//! # Example
//!
//! ```yaml
//! link:
//!   - {type: dvbs2_tx, modulation: 8PSK, rate: 3/5, frame: short}
//!   - {type: awgn, snr_db: $esn0_db, reference: unit}
//!   - {type: dvbs2_rx, modulation: 8PSK, rate: 3/5, frame: short}
//! metrics: {report: [coded_ber, bler]}
//! ```
//!
//! # References
//!
//! 1. ETSI EN 302 307-1 V1.4.1, *DVB-S2*, 2014 (see `com_systems::dvbs2`).

use super::{Block, BlockInfo, BuildContext, Coding, Frame, Params};
use com_core::rng::Rng;
use com_core::{Error, Result, ensure};
use com_systems::dvbs2::{Dvbs2, FrameSize, Modcod, bbframe, bbframe_data, decode_plheader};
use serde_json::Value as Json;

/// The documentation of the blocks of this module.
pub(super) fn infos() -> Vec<BlockInfo> {
    vec![
        block_info!(
            start "dvbs2_tx",
            "DVB-S2 transmitter: BBFRAME, BCH and LDPC, bit interleaving, mapping, PL framing.",
            Dvbs2Spec,
            needs [],
            gives [Samples, Payload],
        ),
        block_info!(
            "dvbs2_rx",
            "DVB-S2 receiver of frame-aligned PLFRAMEs: descrambling, demapping, decoding.",
            Dvbs2Spec,
            needs [Samples, Noise],
            gives [Decoded],
        ),
    ]
}

/// Build a block of this module, or `None` for another type.
pub(super) fn build(kind: &str, label: &str, p: &Json, _ctx: &BuildContext) -> Option<Result<Box<dyn Block>>> {
    let label = label.to_string();
    let tx = match kind {
        "dvbs2_tx" => true,
        "dvbs2_rx" => false,
        _ => return None,
    };
    Some(Dvbs2Spec::parse(kind, p.clone()).and_then(|q| {
        let link = Dvbs2Link::new(&q)?;
        Ok(if tx { Box::new(Dvbs2Tx(link)) as Box<dyn Block> } else { Box::new(Dvbs2Rx { link, label }) })
    }))
}

block_params! {
    struct Dvbs2Spec {
        modulation: String ["QPSK"] one_of ["QPSK", "8PSK", "16APSK", "32APSK"], "modulation";
        rate: String ["1/2"]
            one_of ["1/4", "1/3", "2/5", "1/2", "3/5", "2/3", "3/4", "4/5", "5/6", "8/9", "9/10"],
            "code rate (Table 12)";
        frame: String ["normal"] one_of ["normal", "short", "64800", "16200"],
            "FECFRAME of 64 800 bits (normal) or 16 200 bits (short)";
        pilots: bool [true], "pilot blocks of 36 symbols every 16 slots";
        scrambling_code: usize [0] <= 262_141, "number n of the Gold scrambling sequence";
        frames: usize [1] >= 1, "PLFRAMEs per trial (dvbs2_tx; the receiver reads it from the frame)";
        symbol_rate: f64 [1.0] "Bd" > 0, "symbol rate";
        iterations: usize [50] >= 1, "LDPC iterations (sum-product, layered)";
        max_log: bool [false], "max-log demapping instead of the exact LLRs";
    }
}

struct Dvbs2Link {
    link: Dvbs2,
    frames: usize,
    symbol_rate: f64,
    max_log: bool,
}

impl Dvbs2Link {
    fn new(p: &Dvbs2Spec) -> Result<Dvbs2Link> {
        let frame: FrameSize = p.frame.parse().map_err(|e: Error| e.context("dvbs2"))?;
        let mut link = Dvbs2::new(Modcod::from_name(&p.modulation, &p.rate)?, frame, p.pilots, p.scrambling_code)?;
        link.fec_mut().ldpc_mut().decoder.max_iterations = p.iterations;
        Ok(Dvbs2Link { link, frames: p.frames, symbol_rate: p.symbol_rate, max_log: p.max_log })
    }
}

struct Dvbs2Tx(Dvbs2Link);

impl Block for Dvbs2Tx {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let l = &self.0;
        let k = l.link.k();
        let data: Vec<u8> = (0..(k - 80) * l.frames).map(|_| rng.bit()).collect();
        let mut samples = Vec::with_capacity(l.link.plframe_len() * l.frames);
        for chunk in data.chunks(k - 80) {
            samples.extend(l.link.modulate(&bbframe(chunk, k, 0)?)?);
        }
        let bits_per_symbol = l.link.modcod().modulation.bits_per_symbol();
        *s = Frame {
            trial: s.trial,
            coding: Coding {
                code_rate: Some(data.len() as f64 / (samples.len() * bits_per_symbol) as f64),
                payload: data,
                n_blocks: l.frames,
                ..Coding::default()
            },
            sps: 1.0,
            symbol_rate: l.symbol_rate,
            ..Frame::default()
        };
        s.samples = samples;
        Ok(())
    }
}

struct Dvbs2Rx {
    link: Dvbs2Link,
    label: String,
}

impl Block for Dvbs2Rx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let l = &self.link;
        // the variance the receiver declares, the channel's, or the mean of the
        // per-symbol variances of a detector
        let per_symbol = &s.receiver.symbol_noise;
        let var = s
            .receiver
            .sigma2
            .or(s.channel.noise_var)
            .or((!per_symbol.is_empty()).then(|| per_symbol.iter().sum::<f64>() / per_symbol.len() as f64))
            .ok_or_else(|| {
                Error::Config("dvbs2_rx needs the noise variance: add a noise block (awgn) before it".into())
            })?;
        let len = l.link.plframe_len();
        let frames = s.coding.n_blocks.max(1);
        ensure(s.samples.len() >= len * frames, || {
            format!("dvbs2_rx: {} samples for {frames} PLFRAMEs of {len} symbols", s.samples.len())
        })?;
        let number = l.link.modcod().number()?;
        let (mut decoded, mut header_errors, mut crc_failures) = (Vec::new(), 0usize, 0usize);
        for f in 0..frames {
            let rx = &s.samples[f * len..(f + 1) * len];
            if decode_plheader(&rx[..90])?.0 != number {
                header_errors += 1;
            }
            let bb = l.link.fec().decode(&l.link.llr(rx, var, l.max_log)?)?;
            let (data, crc_ok) = bbframe_data(&bb)?;
            crc_failures += usize::from(!crc_ok);
            decoded.extend(data);
        }
        s.record(&self.label, "plheader_errors", header_errors as f64);
        s.record(&self.label, "crc_failures", crc_failures as f64);
        s.coding.decoded = decoded;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::build;
    use serde_json::json;

    #[test]
    fn a_clean_channel_decodes_every_frame() {
        let ctx = crate::blocks::BuildContext::default();
        let p = json!({"modulation": "16APSK", "rate": "3/4", "frame": "short", "frames": 2});
        let mut q = p.clone();
        q["type"] = json!("dvbs2_tx");
        let tx = build(&q, &ctx).unwrap();
        q["type"] = json!("dvbs2_rx");
        let rx = build(&q, &ctx).unwrap();
        let mut s = Frame::default();
        let mut rng = Rng::new(1);
        tx.apply(&mut s, &mut rng).unwrap();
        s.channel.noise_var = Some(1e-3);
        rx.apply(&mut s, &mut rng).unwrap();
        assert_eq!(s.coding.decoded, s.coding.payload);
        assert_eq!(s.coding.payload.len(), 2 * (11712 - 80));
    }
}
