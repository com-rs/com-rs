//! Optical fibre channels (feature `optical`).
//!
//! - [`fiber`]: the parameters of a fibre link, the linear channel, the split-step
//!   Fourier method;
//! - [`gn`]: the Gaussian-noise model of the nonlinear interference (the reference of
//!   the launch-power campaigns);
//! - [`manakov`]: dual-polarisation nonlinear propagation (the Manakov equation) over
//!   a multi-span link;
//! - [`pmd`]: Jones matrices, polarisation-mode dispersion, the Manakov-PMD equation;
//! - [`edfa`]: the erbium-doped fibre amplifier (gain and ASE noise), the amplifier
//!   of the multi-span links;
//! - [`raman`]: distributed Raman amplification with co- and counter-propagating
//!   pumps and inter-channel stimulated Raman scattering (the power profiles of the
//!   spans and the spontaneous Raman noise), used by the split-step solvers and the GN
//!   model.

pub mod edfa;
pub mod fiber;
pub mod gn;
pub mod manakov;
pub mod pmd;
pub mod raman;
