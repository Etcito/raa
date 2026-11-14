//! CPU base of the M4 area-light soft-shadow tiers: the settings record, its validation,
//! the per-tier visibility weights and the hero shadow-map projection.
//!
//! Every area light is shaded as `U x V x fade`; the tiers differ only in `V`. This module
//! supplies the settings ([`AreaShadowSettings`]), the tier selector ([`AreaShadowTier`]) and
//! the pure maths ([`tier_weights`], [`rect_distance`], [`hero_view_proj`]) that the renderer,
//! the hero selector and the shadow pass share. It allocates nothing and touches no GPU state.

use raa_core::RaaError;
use raa_math::{Mat4, Vec3};

use crate::light_math;

/// Largest number of hero shadow-map lights in one frame.
pub const MAX_HERO_LIGHTS: usize = 4;

/// First shadow-array layer used by a hero light: hero slot `h` renders layer `4 + h`.
pub const HERO_LAYER_BASE: u32 = 4;

/// Vertical field of view of a hero shadow map, 140 degrees in radians (square map, aspect 1).
///
/// The tangent of the half angle is 2.7474774.
pub const HERO_FOV_Y_RAD: f32 = 2.443_461;

/// Near plane of a hero shadow map, in metres.
pub const HERO_NEAR: f32 = 0.05;

/// Which area-light shadow tier the settings select.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum AreaShadowTier {
    /// Hero shadow maps for the nearer lights, the cone further out and the far fade.
    #[default]
    Auto,
    /// Only the nearest lights get a hero shadow map; the rest keep the cone.
    Hero,
    /// The cone visibility only, with no hero map and no fade.
    Cone,
    /// The direct term fades to zero over the fade band; the cone stays.
    Fade,
}

impl AreaShadowTier {
    /// The stable lower-case name of the tier.
    pub fn as_str(self) -> &'static str {
        match self {
            AreaShadowTier::Auto => "auto",
            AreaShadowTier::Hero => "hero",
            AreaShadowTier::Cone => "cone",
            AreaShadowTier::Fade => "fade",
        }
    }

    /// Parses one of `auto`, `hero`, `cone` or `fade`; any other string is `None`.
    pub fn from_name(name: &str) -> Option<AreaShadowTier> {
        match name {
            "auto" => Some(AreaShadowTier::Auto),
            "hero" => Some(AreaShadowTier::Hero),
            "cone" => Some(AreaShadowTier::Cone),
            "fade" => Some(AreaShadowTier::Fade),
            _ => None,
        }
    }
}

/// Area-light soft-shadow settings shared by the renderer and the hero selector.
///
/// Distances are metres and `hero_count` is the number of hero shadow layers requested.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AreaShadowSettings {
    /// The tier selector.
    pub tier: AreaShadowTier,
    /// Number of hero shadow-map lights, at most [`MAX_HERO_LIGHTS`].
    pub hero_count: u32,
    /// Distance from the light at which the hero weight starts to fade, metres.
    pub hero_distance: f32,
    /// Distance at which the direct term has fully faded, metres.
    pub fade_distance: f32,
    /// Width of the smoothstep band at each distance, metres.
    pub band: f32,
}

impl Default for AreaShadowSettings {
    fn default() -> Self {
        AreaShadowSettings {
            tier: AreaShadowTier::Auto,
            hero_count: 0,
            hero_distance: 25.0,
            fade_distance: 80.0,
            band: 5.0,
        }
    }
}

impl AreaShadowSettings {
    /// Validates the settings in contract order, returning the first refusal.
    ///
    /// The refusals use `RaaError::FrameGraph` with code `light.invalid_area_shadows` and a
    /// `<key>: <reason>` message.
    pub fn validate(self) -> Result<AreaShadowSettings, RaaError> {
        let refuse = |message: String| RaaError::FrameGraph {
            code: "light.invalid_area_shadows",
            message,
        };
        if self.hero_count as usize > MAX_HERO_LIGHTS {
            return Err(refuse(format!(
                "heroCount: must be an integer in [0, 4] (got {})",
                self.hero_count
            )));
        }
        if !self.hero_distance.is_finite() || self.hero_distance <= 0.0 {
            return Err(refuse(format!(
                "heroDistance: must be finite and > 0 (got {})",
                self.hero_distance
            )));
        }
        if !self.fade_distance.is_finite() || self.fade_distance <= self.hero_distance {
            return Err(refuse(format!(
                "fadeDistance: must be finite and > heroDistance (got {})",
                self.fade_distance
            )));
        }
        if !self.band.is_finite() || self.band <= 0.0 || self.band > self.hero_distance {
            return Err(refuse(format!(
                "band: must be finite, > 0 and <= heroDistance (got {})",
                self.band
            )));
        }
        Ok(self)
    }
}

/// Smoothstep `s(e0, e1, x) = t * t * (3 - 2 t)` with `t = clamp((x - e0) / (e1 - e0), 0, 1)`.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The visibility weights of one area light: `(w_cone, w_direct)`, each in `[0, 1]`.
///
/// `w_cone` blends the hero shadow map with the cone visibility and `w_direct` scales the
/// direct term. `hero` says whether the light holds a hero slot; `distance` is metres.
pub fn tier_weights(settings: &AreaShadowSettings, hero: bool, distance: f32) -> (f32, f32) {
    match settings.tier {
        AreaShadowTier::Auto => {
            let w_cone = if hero {
                smoothstep(
                    settings.hero_distance - settings.band,
                    settings.hero_distance,
                    distance,
                )
            } else {
                1.0
            };
            let w_direct = 1.0
                - smoothstep(
                    settings.fade_distance - settings.band,
                    settings.fade_distance,
                    distance,
                );
            (w_cone, w_direct)
        }
        AreaShadowTier::Hero => (if hero { 0.0 } else { 1.0 }, 1.0),
        AreaShadowTier::Cone => (1.0, 1.0),
        AreaShadowTier::Fade => (1.0, 0.0),
    }
}

/// Distance from `eye` to the light's representative point, in metres.
///
/// The representative point is [`light_math::representative_point`]: the eye clamped onto the
/// light rectangle along its basis.
pub fn rect_distance(eye: Vec3, position: Vec3, direction: Vec3, size: [f32; 2]) -> f32 {
    (light_math::representative_point(position, direction, size, eye) - eye).length()
}

/// The hero shadow-map view-projection of one area light.
///
/// It is `perspective_rh_zo(HERO_FOV_Y_RAD, 1, HERO_NEAR, far) * look_at_rh(position,
/// position + direction, bitangent)` with `far = range.max(2 * HERO_NEAR)` and
/// `(_, bitangent) = rectangle_basis(direction)`. A receiver at `dz = HERO_NEAR` maps to depth
/// 0 and one at `dz = far` maps to depth 1.
///
/// Handoff: `pub fn hero_view_proj(position: Vec3, direction: Vec3, range: f32) -> Mat4` lives in
/// `crates/raa-lit/src/area_shadow.rs`, re-exported by `crates/raa-lit/src/lib.rs`. The card
/// 'raa-lit: hero selection by screen coverage with hysteresis and the area-light lanes' calls
/// [`tier_weights`], [`rect_distance`] and [`hero_view_proj`] from `HeroSelector` and the
/// area-light lanes; this module passes no registry state or keys, only the settings values and
/// the per-light distance and range.
pub fn hero_view_proj(position: Vec3, direction: Vec3, range: f32) -> Mat4 {
    let far = range.max(2.0 * HERO_NEAR);
    let (_, bitangent) = light_math::rectangle_basis(direction);
    let projection = Mat4::perspective_rh_zo(HERO_FOV_Y_RAD, 1.0, HERO_NEAR, far);
    let view = Mat4::look_at_rh(position, position + direction, bitangent);
    projection.mul(&view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_the_contract_literal() {
        let expected = AreaShadowSettings {
            tier: AreaShadowTier::Auto,
            hero_count: 0,
            hero_distance: 25.0,
            fade_distance: 80.0,
            band: 5.0,
        };
        assert_eq!(AreaShadowSettings::default(), expected);
        assert_eq!(AreaShadowSettings::default().validate(), Ok(expected));
    }
}
