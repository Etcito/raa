//! Sky state shared by the `sky.env` pass, the `sky.background` contributor and the sky
//! handle setters.

use std::cell::RefCell;
use std::rc::Rc;

use crate::desc::SkyDesc;

/// The shared sky state; clones of the `Rc` see the same description and flags.
#[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background)
pub(crate) type SharedSky = Rc<RefCell<SkyState>>;

/// Current sky description and the work it implies.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkyState {
    /// The current description (default: preset `day`).
    pub desc: SkyDesc,
    /// The env cube must be re-rendered (and re-filtered).
    pub env_dirty: bool,
    /// The `EnvGpu` sky fields must be rewritten from `desc` (`SkyDesc::write_env`).
    pub gpu_dirty: bool,
    /// Set by the `sky.background` contributor in each of its prepares (it only prepares
    /// while enabled).
    pub background_on: bool,
}

impl SkyState {
    /// A new shared state over [`SkyState::default`].
    #[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background)
    pub(crate) fn shared() -> SharedSky {
        Rc::new(RefCell::new(SkyState::default()))
    }
}

impl Default for SkyState {
    /// Preset `day`, everything dirty (cube and uniform not written yet), background off.
    fn default() -> Self {
        SkyState {
            // Invariant: `day` is a built-in preset (desc.rs tests check every preset).
            desc: SkyDesc::preset("day").expect("the `day` sky preset exists"),
            env_dirty: true,
            gpu_dirty: true,
            background_on: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_day_and_dirty() {
        let s = SkyState::shared();
        let s = s.borrow();
        assert_eq!(Some(s.desc.clone()), SkyDesc::preset("day"));
        assert!(s.env_dirty && s.gpu_dirty && !s.background_on);
    }
}
