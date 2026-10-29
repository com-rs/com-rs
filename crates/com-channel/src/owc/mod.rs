//! Optical wireless communication: light through free space, intensity-modulated and
//! directly detected (IM/DD), indoors and outdoors.
//!
//! | Module | Content |
//! |---|---|
//! | [`indoor`] | the room channel of visible-light (VLC) and indoor infrared links: Lambertian emitters (LED lights), a detector with a field of view, filter and concentrator, the line of sight and the reflections of every order on diffuse walls (Barry et al. 1993), the illuminance of the lights and the split of on–off keying into signal and intersymbol interference (Komine & Nakagawa 2004) |
//! | [`atmospheric`] | the free-space optical (FSO) channel: the Rytov variance, gamma–gamma (Al-Habash et al. 2001), log-normal and negative exponential turbulence, pointing errors, the attenuation of haze and fog, and the link as a [`Propagate`](com_core::traits::Propagate) block |
//!
//! The two share the intensity channel $y(t) = \gamma\, x(t) \otimes h(t) + n(t)$: the
//! transmitted optical power $x(t) \ge 0$, the photodetector of responsivity $\gamma$
//! and its noise (`com_hardware::optical`). Indoors $h(t)$ is deterministic and
//! dispersive (the geometry of the room); outdoors it is a flat gain that fades
//! randomly with the atmosphere.
//!
//! # References
//!
//! 1. M. A. Khalighi and M. Uysal, "Survey on free space optical communication: a
//!    communication theory perspective," *IEEE Commun. Surveys Tuts.*, 16(4),
//!    2231–2258, 2014.
//! 2. L. E. M. Matheus, A. B. Vieira, L. F. M. Vieira, M. A. M. Vieira and O. Gnawali,
//!    "Visible light communication: concepts, applications and challenges," *IEEE
//!    Commun. Surveys Tuts.*, 21(4), 3204–3237, 2019.

pub mod atmospheric;
pub mod indoor;
