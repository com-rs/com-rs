//! Constant-envelope OFDM (Thompson et al. 2008): the modulator and the phase
//! demodulator, which records the error rates of the linear regime.

use super::{Block, Extension, Frame};
use com_core::Result;
use com_core::rng::Rng;
use com_waveform::ce_ofdm::CeOfdm;
use com_waveform::ce_ofdm::gamp::Gamp;
use com_waveform::ce_ofdm::theory as ce_theory;
use com_waveform::ce_ofdm::{Receiver, Unwrap};

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            "ce_ofdm_tx",
            "Constant-envelope OFDM modulator (Thompson et al. 2008): K symbols per block on the bins 1..K of an N = 2J(K + 1) DFT, exp(j2πh x).",
            CeOfdmSpec,
            needs [Samples],
            gives [],
            |q, _, _| Ok(Box::new(CeOfdmTx(q.link()))),
        ),
        builtin!(
            "ce_ofdm_rx",
            "CE-OFDM receiver back to the symbols: the arctangent phase demodulator, the linear and enhanced receivers of Ahmed and Zeidler (2015) or GAMP (López Morales et al. 2022); records ser_theory and ber_theory when esn0_db is given.",
            CeOfdmRxSpec,
            needs [Samples, Constellation],
            gives [],
            |q, label, _| Ok(Box::new(CeOfdmRx::new(&q, label)?)),
        ),
    ]
}

// --------------------------------------------------------------------------- CE-OFDM

block_params! {
    struct CeOfdmSpec {
        n_data: usize [64] >= 1, "data subcarriers K per block";
        oversampling: usize [4] >= 1, "oversampling factor J";
        h: f64 [required] > 0, "modulation index (phase standard deviation 2πh)";
        n_cp: usize [0] "samples", "cyclic prefix";
    }
}

impl CeOfdmSpec {
    fn link(&self) -> CeOfdm {
        CeOfdm {
            n_subcarriers: self.n_data,
            oversampling: self.oversampling,
            h: self.h,
            n_cp: self.n_cp,
            receiver: Receiver::default(),
            sample_rate: com_core::units::Hertz(1.0),
        }
    }
}

struct CeOfdmTx(CeOfdm);

impl Block for CeOfdmTx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        s.require_one_sps("ce_ofdm_tx")?;
        s.samples = self.0.modulate(&s.samples)?;
        s.sps = self.0.sps();
        Ok(())
    }
}

block_params! {
    struct CeOfdmRxSpec {
        n_data: usize [64] >= 1, "data subcarriers K (as the modulator)";
        oversampling: usize [4] >= 1, "oversampling factor J (as the modulator)";
        h: f64 [required] > 0, "modulation index (as the modulator)";
        n_cp: usize [0] "samples", "cyclic prefix (as the modulator)";
        unwrap: String ["none"] one_of ["none", "stream"],
            "none (principal phase) or stream (unwrapped along the stream, the phase unwrapper of the paper)";
        receiver: String ["arctangent"] one_of ["arctangent", "linear", "enhanced", "gamp"],
            "arctangent (phase demodulator), linear (Im r) or enhanced (Im r (2 - Re r)) of Ahmed and Zeidler 2015, or gamp (near-optimum message passing, needs esn0_db)";
        esn0_db: Option<f64> [absent] "dB",
            "E_s/N_0 of the channel per data symbol: records ser_theory and ber_theory (the linear regime for arctangent and gamp, where it is the optimum asymptotic rate of QPSK; Bussgang's analysis for linear and enhanced), and sets the noise variance of gamp";
    }
}

struct CeOfdmRx {
    ce: CeOfdm,
    /// `None` for GAMP.
    receiver: Option<Receiver>,
    label: String,
    esn0_db: Option<f64>,
}

impl CeOfdmRx {
    fn new(p: &CeOfdmRxSpec, label: String) -> Result<CeOfdmRx> {
        let ce = CeOfdmSpec { n_data: p.n_data, oversampling: p.oversampling, h: p.h, n_cp: p.n_cp }.link();
        let receiver = match p.receiver.as_str() {
            "gamp" => {
                if p.esn0_db.is_none() {
                    return Err(com_core::Error::Config("ce_ofdm_rx: the gamp receiver needs esn0_db".into()));
                }
                None
            }
            "arctangent" => Some(Receiver::Arctangent(p.unwrap.parse::<Unwrap>()?)),
            other => Some(other.parse::<Receiver>()?),
        };
        Ok(CeOfdmRx { ce, receiver, label, esn0_db: p.esn0_db })
    }
}

impl Block for CeOfdmRx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let c = s.constellation("ce_ofdm_rx")?.clone();
        if let Some(es) = self.esn0_db {
            let (m, kind) = (c.order(), c.kind());
            let ce =
                CeOfdm { receiver: self.receiver.unwrap_or(Receiver::Arctangent(Unwrap::None)), ..self.ce.clone() };
            s.record(&self.label, "ser_theory", ce_theory::ser_receiver(&ce, kind, m, es)?);
            s.record(&self.label, "ber_theory", ce_theory::ber_receiver(&ce, kind, m, es)?);
        }
        s.samples = match (self.receiver, self.esn0_db) {
            (Some(r), _) => self.ce.demodulate_with_receiver(&s.samples, r)?,
            (None, Some(es)) => Gamp::default().detect(&self.ce, &s.samples, &c, self.ce.noise_variance(es))?,
            (None, None) => unreachable!("checked by new"),
        };
        s.sps = 1.0;
        Ok(())
    }
}
