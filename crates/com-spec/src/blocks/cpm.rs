//! Continuous-phase modulation: the modulator and its detectors.
//!
//! `cpm_tx` replaces the symbols of a real constellation (PAM of order $M$, or BPSK)
//! by the constant-envelope signal $e^{j \phi(t)}$ of [`com_waveform::cpm`]: the
//! levels of the constellation, in increasing order, are the CPM symbols
//! $\pm 1, \pm 3, \dots$. `cpm_rx` decides the symbols back, by the maximum-likelihood
//! sequence detector or the one-symbol differential detector, and hands the metrics
//! the decided constellation points.

use super::schema::ParamType;
use super::{Block, Extension, Frame};
use com_core::rng::Rng;
use com_core::{Error, Result, ensure};
use com_mapping::constellation::Constellation;
use com_waveform::cpm::{Cpm, CpmConfig, FrequencyPulse, ModulationIndex, msk_bits, msk_precode};
use serde::Deserialize;
use serde_json::Value as Json;

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            "cpm_tx",
            "Continuous-phase modulator of the symbols of a real constellation (CPFSK, MSK, GMSK, raised-cosine CPM): a unit-envelope signal at sps samples per symbol.",
            CpmSpec,
            needs [Samples, Constellation],
            gives [],
            |q, _, _| Ok(Box::new(CpmTx(q))),
        ),
        builtin!(
            "cpm_rx",
            "CPM detector (maximum-likelihood sequence detection over the phase trellis, or one-symbol differential detection) back to the symbols.",
            CpmRxSpec,
            needs [Samples, Constellation],
            gives [],
            |q, _, _| Ok(Box::new(CpmRx::new(q))),
        ),
    ]
}

block_params! {
    struct CpmSpec {
        h: String ["1/2"], "modulation index p/q, as a fraction (1/2 with a rectangular pulse of length 1 is MSK)";
        pulse: String ["rec"] one_of ["rec", "rc", "gmsk"],
            "frequency pulse: rectangular (1REC is CPFSK), raised cosine, or Gaussian (GMSK, with bt)";
        length: usize [1] >= 1, "length L of the pulse in symbols";
        bt: f64 [0.3] > 0, "bandwidth–time product of the Gaussian pulse";
        sps: usize [4] >= 2, "samples per symbol";
        precode: bool [false],
            "MSK precoding (binary, h 1/2, rec, length 1): the bits are the boundary signs of the equivalent OQPSK signal, and cpm_rx reads them from the phase states of the ML path, with the BER of BPSK";
    }
}

impl CpmSpec {
    /// The modulator of order `order`.
    fn cpm(&self, order: usize) -> Result<Cpm> {
        let h: ModulationIndex = self.h.parse().map_err(|e: Error| e.context("cpm"))?;
        CpmConfig {
            order,
            h_num: h.num,
            h_den: h.den,
            pulse: FrequencyPulse::from_name(&self.pulse, self.bt)?,
            length: self.length,
            sps: self.sps,
        }
        .build()
    }

    /// The modulator, checked for MSK when `precode` is set.
    fn checked(&self, order: usize, precode: bool, block: &str) -> Result<Cpm> {
        let cpm = self.cpm(order)?;
        let c = cpm.config();
        let msk = order == 2 && c.h_num == 1 && c.h_den == 2 && c.pulse == FrequencyPulse::Rectangular && c.length == 1;
        ensure(!precode || msk, || format!("{block}: precode needs MSK (BPSK, h: 1/2, pulse: rec, length: 1)"))?;
        Ok(cpm)
    }
}

/// The CPM symbol of every constellation index (the rank of its level), and the
/// constellation index of every CPM symbol.
fn levels(c: &Constellation, block: &str) -> Result<(Vec<usize>, Vec<usize>)> {
    let points = c.points();
    ensure(points.iter().all(|p| p.im.abs() < 1e-9), || {
        format!("{block}: the constellation must be real (PAM or BPSK), got {:?}", c.kind())
    })?;
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| points[a].re.total_cmp(&points[b].re));
    let mut rank = vec![0; points.len()];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }
    Ok((rank, order))
}

struct CpmTx(CpmSpec);

impl Block for CpmTx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        s.require_one_sps("cpm_tx")?;
        let c = s.constellation("cpm_tx")?;
        let (rank, _) = levels(c, "cpm_tx")?;
        let cpm = self.0.checked(c.order(), self.0.precode, "cpm_tx")?;
        let mut symbols: Vec<usize> = c.decide(&s.samples).into_iter().map(|i| rank[i]).collect();
        if self.0.precode {
            let bits: Vec<u8> = symbols.iter().map(|&r| r as u8).collect();
            symbols = msk_precode(&bits);
        }
        s.samples = cpm.modulate(&symbols)?;
        s.sps = self.0.sps as f64;
        Ok(())
    }
}

block_params! {
    struct CpmRxSpec {
        h: String ["1/2"], "as cpm_tx";
        pulse: String ["rec"] one_of ["rec", "rc", "gmsk"], "as cpm_tx";
        length: usize [1] >= 1, "as cpm_tx";
        bt: f64 [0.3] > 0, "as cpm_tx";
        sps: usize [4] >= 2, "as cpm_tx";
        detector: CpmDetector ["ml"] one_of ["ml", "differential"],
            "ml (Viterbi over the phase trellis, coherent) or differential (sign of the one-symbol phase increment, after a Gaussian low-pass of rx_bt)";
        rx_bt: Option<f64> [absent] > 0, "bandwidth–time product of the receive low-pass of the differential detector (none: no filter)";
        precode: bool [false], "as cpm_tx (with the ml detector)";
    }
}

/// The detector of `cpm_rx`.
#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum CpmDetector {
    /// Viterbi over the phase trellis.
    Ml,
    /// Sign of the one-symbol phase increment.
    Differential,
}

impl ParamType for CpmDetector {
    fn schema() -> Json {
        String::schema()
    }
}

struct CpmRx {
    spec: CpmRxSpec,
    differential: bool,
}

impl CpmRx {
    fn new(spec: CpmRxSpec) -> CpmRx {
        let differential = spec.detector == CpmDetector::Differential;
        CpmRx { spec, differential }
    }

    /// The MSK precoding is read from the ML path.
    fn precode(&self) -> bool {
        self.spec.precode && !self.differential
    }
}

impl Block for CpmRx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let c = s.constellation("cpm_rx")?;
        let (_, index_of) = levels(c, "cpm_rx")?;
        ensure(s.sps == self.spec.sps as f64, || {
            format!("cpm_rx: expects {} samples per symbol (as cpm_tx), the signal has {}", self.spec.sps, s.sps)
        })?;
        let tx = CpmSpec {
            h: self.spec.h.clone(),
            pulse: self.spec.pulse.clone(),
            length: self.spec.length,
            bt: self.spec.bt,
            sps: self.spec.sps,
            precode: self.spec.precode,
        };
        let cpm = tx.checked(c.order(), self.precode(), "cpm_rx")?;
        let decided = if self.differential {
            cpm.differential_detect(&s.samples, self.spec.rx_bt)?
        } else if self.precode() {
            msk_bits(&cpm.ml_detect_path(&s.samples)?.phase_states).into_iter().map(usize::from).collect()
        } else {
            cpm.ml_detect(&s.samples)?
        };
        s.samples = decided.iter().map(|&r| c.points()[index_of[r]]).collect();
        s.sps = 1.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BuildContext, build};
    use com_core::rng::Rng;
    use serde_json::json;

    #[test]
    fn msk_round_trip_is_exact_without_noise() {
        let ctx = BuildContext::default();
        let mut rng = Rng::new(3);
        let mut s = super::Frame::default();
        let source = build(&json!({"type": "source", "constellation": "BPSK", "n_symbols": 500}), &ctx).unwrap();
        source.apply(&mut s, &mut rng).unwrap();
        let tx_symbols = s.samples.clone();
        build(&json!({"type": "cpm_tx", "sps": 4}), &ctx).unwrap().apply(&mut s, &mut rng).unwrap();
        assert_eq!(s.samples.len(), 2000);
        assert!(s.samples.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12), "constant envelope");
        build(&json!({"type": "cpm_rx", "sps": 4}), &ctx).unwrap().apply(&mut s, &mut rng).unwrap();
        assert_eq!(s.samples, tx_symbols);
    }

    #[test]
    fn a_complex_constellation_is_refused() {
        let ctx = BuildContext::default();
        let mut rng = Rng::new(3);
        let mut s = super::Frame::default();
        build(&json!({"type": "source", "constellation": "QPSK", "n_symbols": 100}), &ctx)
            .unwrap()
            .apply(&mut s, &mut rng)
            .unwrap();
        let e = build(&json!({"type": "cpm_tx"}), &ctx).unwrap().apply(&mut s, &mut rng).unwrap_err();
        assert!(e.to_string().contains("real"), "{e}");
        assert!(build(&json!({"type": "cpm_tx", "h": "half"}), &ctx).unwrap().apply(&mut s, &mut rng).is_err());
    }
}
