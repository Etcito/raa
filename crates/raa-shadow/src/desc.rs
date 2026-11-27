//! Shadow description and validation (`shadow.invalid`).

use raa_core::RaaError;

/// Sun shadow settings.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ShadowDesc {
    /// Multiplies the sun angular radius; in (0, 16].
    pub softness: f32,
    /// Shadow distance in metres (> 0).
    pub distance_m: f32,
    /// Cascade resolution: 1024, 2048 or 4096.
    pub resolution: u32,
    /// Keep the sun shadowed in the whole scene: the last cascade reaches the farthest corner
    /// of the scene bounds (capped by the camera far plane) when that is past `distance_m`,
    /// and a cascade is fitted to the scene's light-space bounds when they are smaller than
    /// its slice. `false` restores the plain range `min(far, distance_m)`, past which the
    /// sun is unshadowed.
    pub cover: bool,
    /// Number of cascades actually used and rendered: 1..=4 (the splits are the practical
    /// scheme for this count; the array keeps 4 layers, the unused ones are not rendered).
    pub cascades: u32,
    /// Minimum PCF filter radius in metres (>= 0, <= 0.5): the same world size in every
    /// cascade (at least 1.5 texels of each), so the shadow edge stays smooth where a cascade
    /// is coarse and does not change at a cascade boundary. The sun penumbra widens it.
    pub filter_m: f32,
    /// Practical split weight of the cascades in [0, 1]: 0 uniform, 1 logarithmic. Higher
    /// values give the near cascades more texels per metre (sharper close shadows).
    pub split_lambda: f32,
    /// With `cover`: spread the cascade splits over the scene the camera can see
    /// (`[near, min(far, cover distance, 4 x distance)]`) instead of `[near, distance]`, so the
    /// middle cascades reach the far side of the scene instead of leaving it to the coarse
    /// scene-wide last cascade, and no split is spent past the scene.
    pub spread: bool,
    /// Sun shadow technique: cascades (default) or the virtual shadow map.
    pub mode: ShadowMode,
    /// VSM physical page pool, in 128 x 128 pages: 1024, 2048 or 4096 (64, 128 or 256 MB).
    pub vsm_pages: u32,
    /// VSM pages rendered per frame at most (16..=768): new pages refine progressively and
    /// a camera cut never costs more than this many page renders in one frame.
    pub vsm_budget: u32,
    /// VSM level bias in [-2, 2]: negative values pick finer levels (more texels per pixel,
    /// more pages), positive coarser ones.
    pub vsm_bias: f32,
}

/// Sun shadow technique of `shadow.csm`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ShadowMode {
    /// Cascaded shadow maps.
    #[default]
    Csm,
    /// Virtual shadow map: a 16k x 16k clipmap per level around the camera, paged in 128 x 128
    /// pages from a physical pool, only the pages the visible pixels need, cached while the
    /// sun and the casters stay still. The cascade layers become its page render targets.
    Vsm,
}

impl ShadowMode {
    /// The catalog / JS name.
    pub fn name(self) -> &'static str {
        match self {
            ShadowMode::Csm => "csm",
            ShadowMode::Vsm => "vsm",
        }
    }

    /// Parses a catalog / JS name.
    pub fn parse(name: &str) -> Option<ShadowMode> {
        match name {
            "csm" => Some(ShadowMode::Csm),
            "vsm" => Some(ShadowMode::Vsm),
            _ => None,
        }
    }
}

/// Default VSM pool (pages).
pub const VSM_PAGES_DEFAULT: u32 = 2048;
/// Default VSM pages rendered per frame.
pub const VSM_BUDGET_DEFAULT: u32 = 256;

/// Default practical split weight: more logarithmic than the plain 0.75, the near cascades
/// keep their density once `spread` widens the split range.
pub const SHADOW_SPLIT_DEFAULT: f32 = 0.875;

/// Default minimum filter radius (m).
pub const SHADOW_FILTER_DEFAULT_M: f32 = 0.015625;

impl Default for ShadowDesc {
    fn default() -> Self {
        ShadowDesc {
            softness: 1.0,
            distance_m: 60.0,
            resolution: 2048,
            cover: true,
            cascades: 4,
            filter_m: SHADOW_FILTER_DEFAULT_M,
            split_lambda: SHADOW_SPLIT_DEFAULT,
            spread: true,
            mode: ShadowMode::Csm,
            vsm_pages: VSM_PAGES_DEFAULT,
            vsm_budget: VSM_BUDGET_DEFAULT,
            vsm_bias: 0.0,
        }
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "shadow.invalid",
        message,
    }
}

impl ShadowDesc {
    /// Refuses with `shadow.invalid`, the message naming the offending field.
    pub fn validate(&self) -> Result<(), RaaError> {
        if !(self.softness.is_finite() && self.softness > 0.0 && self.softness <= 16.0) {
            return Err(invalid(format!(
                "softness must be finite and in (0, 16], got {}",
                self.softness
            )));
        }
        if !(self.distance_m.is_finite() && self.distance_m > 0.0) {
            return Err(invalid(format!(
                "distance_m must be finite and > 0, got {}",
                self.distance_m
            )));
        }
        if !(self.filter_m.is_finite() && (0.0..=0.5).contains(&self.filter_m)) {
            return Err(invalid(format!(
                "filter_m must be finite and in [0, 0.5], got {}",
                self.filter_m
            )));
        }
        if !(self.split_lambda.is_finite() && (0.0..=1.0).contains(&self.split_lambda)) {
            return Err(invalid(format!(
                "split_lambda must be finite and in [0, 1], got {}",
                self.split_lambda
            )));
        }
        if !matches!(self.vsm_pages, 1024 | 2048 | 4096) {
            return Err(invalid(format!(
                "vsm_pages must be 1024, 2048 or 4096, got {}",
                self.vsm_pages
            )));
        }
        if !(16..=768).contains(&self.vsm_budget) {
            return Err(invalid(format!(
                "vsm_budget must be 16 to 768, got {}",
                self.vsm_budget
            )));
        }
        if !(self.vsm_bias.is_finite() && (-2.0..=2.0).contains(&self.vsm_bias)) {
            return Err(invalid(format!(
                "vsm_bias must be finite and in [-2, 2], got {}",
                self.vsm_bias
            )));
        }
        if !(1..=4).contains(&self.cascades) {
            return Err(invalid(format!(
                "cascades must be 1 to 4, got {}",
                self.cascades
            )));
        }
        if !matches!(self.resolution, 1024 | 2048 | 4096) {
            return Err(invalid(format!(
                "resolution must be 1024, 2048 or 4096, got {}",
                self.resolution
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(d: ShadowDesc, field: &str) {
        match d.validate() {
            Err(RaaError::FrameGraph { code, message }) => {
                assert_eq!(code, "shadow.invalid");
                assert!(!message.is_empty());
                assert!(message.contains(field), "{message}");
            }
            other => panic!("expected refusal, got {other:?}"),
        }
    }

    #[test]
    fn default_is_valid() {
        assert!(ShadowDesc::default().validate().is_ok());
    }

    #[test]
    fn softness_refused() {
        for v in [0.0, -1.0, 16.5, f32::NAN, f32::INFINITY] {
            refused(
                ShadowDesc {
                    softness: v,
                    ..Default::default()
                },
                "softness",
            );
        }
    }

    #[test]
    fn distance_refused() {
        for v in [0.0, -5.0, f32::NAN, f32::INFINITY] {
            refused(
                ShadowDesc {
                    distance_m: v,
                    ..Default::default()
                },
                "distance_m",
            );
        }
    }

    #[test]
    fn filter_refused() {
        for v in [-0.01, 0.6, f32::NAN, f32::INFINITY] {
            refused(
                ShadowDesc {
                    filter_m: v,
                    ..Default::default()
                },
                "filter_m",
            );
        }
    }

    #[test]
    fn vsm_fields_refused() {
        refused(
            ShadowDesc {
                vsm_pages: 3000,
                ..Default::default()
            },
            "vsm_pages",
        );
        refused(
            ShadowDesc {
                vsm_budget: 8,
                ..Default::default()
            },
            "vsm_budget",
        );
        refused(
            ShadowDesc {
                vsm_bias: 3.0,
                ..Default::default()
            },
            "vsm_bias",
        );
        assert_eq!(ShadowMode::parse("vsm"), Some(ShadowMode::Vsm));
        assert_eq!(ShadowMode::Vsm.name(), "vsm");
        assert_eq!(ShadowMode::parse("x"), None);
    }

    #[test]
    fn resolution_refused() {
        for v in [0, 512, 3000, 8192] {
            refused(
                ShadowDesc {
                    resolution: v,
                    ..Default::default()
                },
                "resolution",
            );
        }
    }
}
