//! Display transform of the lit renderer (S12 `tonemap`): exposure, AgX / ACES /
//! linear tone mapping, the sRGB OETF and dither. This module holds the settings and
//! their shared control; [`curves`] holds CPU ports of the curves the shader runs.

use std::cell::Cell;
use std::rc::Rc;

use raa_core::RaaError;

pub mod curves;
mod pass;

pub use pass::{TONEMAP_NAME, TONEMAP_ORDER, TONEMAP_WGSL, TonemapPass};

/// Refusal code of invalid tone mapping settings.
pub const POST_INVALID: &str = "post.invalid";

/// Tone mapping curve (FrameGpu.exposure.z: 0 agx, 1 aces, 2 linear).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ToneMapper {
    /// Sobotka's AgX base.
    Agx,
    /// Hill's fitted ACES RRT+ODT.
    Aces,
    /// Clamp to [0, 1].
    Linear,
}

impl ToneMapper {
    /// Index written to FrameGpu.exposure.z (S10).
    pub fn gpu_index(self) -> u32 {
        match self {
            ToneMapper::Agx => 0,
            ToneMapper::Aces => 1,
            ToneMapper::Linear => 2,
        }
    }
}

/// AgX look preset (slope / power / saturation); ignored by ACES and linear
/// (FrameGpu.exposure.w: 0 none, 1 punchy, 2 golden).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Look {
    None,
    Punchy,
    Golden,
}

impl Look {
    /// Index written to FrameGpu.exposure.w (S10).
    pub fn gpu_index(self) -> u32 {
        match self {
            Look::None => 0,
            Look::Punchy => 1,
            Look::Golden => 2,
        }
    }
}

/// Tone mapping settings; `Default` is TONEMAP_DEFAULT (agx, look none, exposure 0 EV,
/// dither on).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ToneMapping {
    pub mapper: ToneMapper,
    /// Exposure in EV: the HDR colour is multiplied by 2^exposure_ev.
    pub exposure_ev: f32,
    pub look: Look,
    /// Dither +-0.5/255 with an 8x8 Bayer matrix.
    pub dither: bool,
}

impl Default for ToneMapping {
    fn default() -> ToneMapping {
        ToneMapping {
            mapper: ToneMapper::Agx,
            exposure_ev: 0.0,
            look: Look::None,
            dither: true,
        }
    }
}

impl ToneMapping {
    /// Refuses settings the pass cannot honour: a non-finite `exposure_ev` gives
    /// `post.invalid`.
    pub fn validate(&self) -> Result<(), RaaError> {
        if !self.exposure_ev.is_finite() {
            return Err(RaaError::FrameGraph {
                code: POST_INVALID,
                message: format!(
                    "tonemap exposureEv must be finite, got {}",
                    self.exposure_ev
                ),
            });
        }
        Ok(())
    }

    /// Linear multiplier 2^exposure_ev applied to the HDR colour.
    pub fn exposure_scale(&self) -> f32 {
        self.exposure_ev.exp2()
    }
}

/// Shared handle on the settings the `tonemap` pass reads each frame.
#[derive(Clone, Debug, Default)]
pub struct TonemapControl {
    settings: Rc<Cell<ToneMapping>>,
}

impl TonemapControl {
    /// A control holding `settings` (validated).
    pub fn new(settings: ToneMapping) -> Result<TonemapControl, RaaError> {
        settings.validate()?;
        Ok(TonemapControl {
            settings: Rc::new(Cell::new(settings)),
        })
    }

    /// Current settings.
    pub fn get(&self) -> ToneMapping {
        self.settings.get()
    }

    /// Replaces the settings; invalid settings are refused and the current ones kept.
    pub fn set(&self, settings: ToneMapping) -> Result<(), RaaError> {
        settings.validate()?;
        self.settings.set(settings);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_tonemap_default() {
        let t = ToneMapping::default();
        assert_eq!(t.mapper, ToneMapper::Agx);
        assert_eq!(t.look, Look::None);
        assert_eq!(t.exposure_ev, 0.0);
        assert!(t.dither);
        assert_eq!(t.exposure_scale(), 1.0);
        assert!(t.validate().is_ok());
    }

    #[test]
    fn non_finite_exposure_is_refused_with_post_invalid() {
        for ev in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let t = ToneMapping {
                exposure_ev: ev,
                ..ToneMapping::default()
            };
            let err = t.validate().unwrap_err();
            assert_eq!(err.code(), "post.invalid");
            assert!(!err.message().is_empty());
        }
    }

    #[test]
    fn control_clones_share_settings_and_keep_them_on_refusal() {
        let a = TonemapControl::default();
        let b = a.clone();
        let aces = ToneMapping {
            mapper: ToneMapper::Aces,
            exposure_ev: 1.5,
            ..ToneMapping::default()
        };
        b.set(aces).unwrap();
        assert_eq!(a.get(), aces);
        let bad = ToneMapping {
            exposure_ev: f32::NAN,
            ..aces
        };
        assert_eq!(a.set(bad).unwrap_err().code(), "post.invalid");
        assert_eq!(b.get(), aces);
        assert!(TonemapControl::new(bad).is_err());
    }

    #[test]
    fn gpu_indices_follow_s10() {
        assert_eq!(ToneMapper::Agx.gpu_index(), 0);
        assert_eq!(ToneMapper::Aces.gpu_index(), 1);
        assert_eq!(ToneMapper::Linear.gpu_index(), 2);
        assert_eq!(Look::None.gpu_index(), 0);
        assert_eq!(Look::Punchy.gpu_index(), 1);
        assert_eq!(Look::Golden.gpu_index(), 2);
    }
}
