//! The single-polarisation blocks applied to every polarisation of a
//! dual-polarisation channel.

use super::{Block, ChannelState, Frame, ReceiverState};
use com_core::rng::Rng;
use com_core::{Result, ensure_shape};
use com_math::C64;

/// A single-polarisation block applied to every polarisation of a dual-polarisation
/// channel (see [`BlockInfo::per_polarization`](super::BlockInfo::per_polarization)); the
/// block itself otherwise. A block trained once per condition is trained on
/// polarisation x, with its own symbols, and then applied to each polarisation.
pub(super) struct PerPol(pub(super) Box<dyn Block>);

/// The `i`-th of `n` equal parts of `v`, `what` naming `v` in the error.
///
/// # Errors
/// [`com_core::Error::Shape`] when the length of `v` is not a multiple of `n`.
pub(super) fn part<T: Clone>(v: &[T], i: usize, n: usize, what: &str) -> Result<Vec<T>> {
    ensure_shape(n > 0 && v.len() % n == 0, || format!("the {} {what} do not split into {n} streams", v.len()))?;
    let m = v.len() / n;
    Ok(v[i * m..(i + 1) * m].to_vec())
}

/// Whether the frame carries the polarisations of a dual-polarisation channel.
fn dual(s: &Frame) -> bool {
    s.layout.polarisations && !s.antennas.is_empty()
}

/// The frame of polarisation `i` of the `n` of `s`, with its own symbols and the
/// samples `samples`.
fn polarisation(s: &Frame, i: usize, n: usize, samples: Vec<C64>) -> Result<Frame> {
    Ok(Frame {
        constellation: s.constellation.clone(),
        bits: part(&s.bits, i, n, "bits")?,
        indices: part(&s.indices, i, n, "symbol indices")?,
        symbols: part(&s.symbols, i, n, "symbols")?,
        samples,
        sps: s.sps,
        symbol_rate: s.symbol_rate,
        excluded: part(&s.excluded, i, n, "exclusion flags")?,
        receiver: ReceiverState { noise_var: s.receiver.noise_var, ..ReceiverState::default() },
        channel: ChannelState { noise_var: s.channel.noise_var, ..ChannelState::default() },
        trial: s.trial,
        ..Frame::default()
    })
}

impl Block for PerPol {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        if !dual(s) {
            return self.0.apply(s, rng);
        }
        per_pol(s, rng, &|sub, r| self.0.apply(sub, r))
    }

    fn needs_training(&self) -> bool {
        self.0.needs_training()
    }

    fn train(&mut self, s: &Frame, rng: &mut Rng) -> Result<()> {
        if !dual(s) {
            return self.0.train(s, rng);
        }
        let x = polarisation(s, 0, s.n_streams(), s.samples.clone())?;
        self.0.train(&x, rng)
    }
}

/// Apply `f` to every polarisation of a dual-polarisation signal, each with its own
/// symbols, and reassemble the signal.
///
/// # Errors
/// [`com_core::Error::Shape`] when the symbols do not split between the polarisations;
/// the first error of `f`.
pub(super) fn per_pol(s: &mut Frame, rng: &mut Rng, f: &dyn Fn(&mut Frame, &mut Rng) -> Result<()>) -> Result<()> {
    let n = s.n_streams();
    let streams = s.take_streams();
    let mut subs = Vec::with_capacity(n);
    for (i, x) in streams.into_iter().enumerate() {
        let mut sub = polarisation(s, i, n, x)?;
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
    let sig: Vec<f64> = subs.iter().filter_map(|u| u.receiver.noise_var).collect();
    if !sig.is_empty() {
        s.receiver.noise_var = Some(sig.iter().sum::<f64>() / sig.len() as f64);
    }
    s.set_streams(subs.into_iter().map(|u| u.samples).collect());
    Ok(())
}
