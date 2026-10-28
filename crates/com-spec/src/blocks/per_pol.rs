//! The single-polarisation blocks applied to every polarisation of a
//! dual-polarisation channel.

use super::{Block, ChannelState, Frame, ReceiverState};
use com_core::rng::Rng;
use com_core::{Result, ensure};

/// A single-polarisation block applied to every polarisation of a dual-polarisation
/// channel (see [`BlockInfo::per_polarization`](super::BlockInfo::per_polarization)); the
/// block itself otherwise.
pub(super) struct PerPol(pub(super) Box<dyn Block>);

/// The `i`-th of `n` equal parts of `v` (empty if `v` does not split).
pub(super) fn part<T: Clone>(v: &[T], i: usize, n: usize) -> Vec<T> {
    if v.len() % n != 0 {
        return Vec::new();
    }
    let m = v.len() / n;
    v[i * m..(i + 1) * m].to_vec()
}

impl Block for PerPol {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        if !s.layout.polarisations || s.antennas.is_empty() {
            return self.0.apply(s, rng);
        }
        per_pol(s, rng, &|sub, r| self.0.apply(sub, r))
    }
}

/// Apply `f` to every polarisation of a dual-polarisation signal, each with its own
/// symbols, and reassemble the signal.
pub(super) fn per_pol(s: &mut Frame, rng: &mut Rng, f: &dyn Fn(&mut Frame, &mut Rng) -> Result<()>) -> Result<()> {
    {
        let n = s.n_streams();
        ensure(s.symbols.len() % n == 0, || "the polarisations carry different numbers of symbols".into())?;
        let streams = s.take_streams();
        let mut subs = Vec::with_capacity(n);
        for (i, x) in streams.into_iter().enumerate() {
            let mut sub = Frame {
                constellation: s.constellation.clone(),
                bits: part(&s.bits, i, n),
                indices: part(&s.indices, i, n),
                symbols: part(&s.symbols, i, n),
                samples: x,
                sps: s.sps,
                symbol_rate: s.symbol_rate,
                excluded: part(&s.excluded, i, n),
                receiver: ReceiverState { sigma2: s.receiver.sigma2, ..ReceiverState::default() },
                channel: ChannelState { noise_var: s.channel.noise_var, ..ChannelState::default() },
                ..Frame::default()
            };
            f(&mut sub, rng)?;
            subs.push(sub);
        }
        s.sps = subs[0].sps;
        s.symbol_rate = subs[0].symbol_rate;
        s.bits = subs.iter().flat_map(|u| u.bits.iter().copied()).collect();
        s.indices = subs.iter().flat_map(|u| u.indices.iter().copied()).collect();
        s.symbols = subs.iter().flat_map(|u| u.symbols.iter().copied()).collect();
        s.excluded = subs.iter().flat_map(|u| u.excluded.iter().copied()).collect();
        for u in &mut subs {
            s.values.append(&mut u.values);
        }
        let sig: Vec<f64> = subs.iter().filter_map(|u| u.receiver.sigma2).collect();
        if !sig.is_empty() {
            s.receiver.sigma2 = Some(sig.iter().sum::<f64>() / sig.len() as f64);
        }
        s.set_streams(subs.into_iter().map(|u| u.samples).collect());
        Ok(())
    }
}
