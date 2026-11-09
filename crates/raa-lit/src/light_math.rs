//! S11 CPU local-light mathematics: point window/falloff, spot cone and the directional
//! no-distance rule. Descriptor validation keeps every input finite; `distance` is
//! nonnegative, `range` positive and a cone cosine is a dot product of unit vectors.
//!
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-render/shaders/lighting/deferred.wgsl`, `fn local_light`, revision 7e465b5.
//!
//! Handoff: the rectangle share adds `pub(crate) fn rectangle_basis(direction: Vec3) -> (Vec3, Vec3)`
//! and `pub(crate) fn representative_point(position: Vec3, direction: Vec3, size: [f32; 2], surface: Vec3) -> Vec3`
//! to `crates/raa-lit/src/light_math.rs`, below the functions here. `crates/raa-lit/src/lib.rs`
//! attaches this module with `mod light_math;`; these pure functions pass no registry state and
//! no keys.

#![allow(dead_code)] // Staged before tile selection and shading consumers land; see Handoff.

use crate::local_light::LocalLightKind;

/// Windowed inverse-square falloff shared by point, spot and area lights.
///
/// A light at or beyond `range` contributes nothing. Otherwise the window is
/// `(clamp(1 - (distance / range)^4, 0, 1))^2` and it is divided by
/// `max(distance^2, 0.04)`, so a coincident or near-coincident position stays finite.
pub(crate) fn windowed_falloff(distance: f32, range: f32) -> f32 {
    if distance >= range {
        return 0.0;
    }
    let ratio = distance / range;
    let ratio2 = ratio * ratio;
    let window = (1.0 - ratio2 * ratio2).clamp(0.0, 1.0);
    window * window / (distance * distance).max(0.04)
}

/// Spot cone factor: 0 outside the outer cone, 1 inside the inner cone and smoothstep
/// between their cosines. Equal cosines use a hard boundary at `outer`, without a
/// division by zero.
pub(crate) fn spot_factor(cosine: f32, outer: f32, inner: f32) -> f32 {
    if inner <= outer {
        return if cosine >= outer { 1.0 } else { 0.0 };
    }
    if cosine >= inner {
        return 1.0;
    }
    if cosine <= outer {
        return 0.0;
    }
    let t = (cosine - outer) / (inner - outer);
    t * t * (3.0 - 2.0 * t)
}

/// Distance attenuation by kind: directional lights ignore position and range.
pub(crate) fn kind_falloff(kind: LocalLightKind, distance: f32, range: f32) -> f32 {
    match kind {
        LocalLightKind::Directional => 1.0,
        LocalLightKind::Point | LocalLightKind::Spot | LocalLightKind::Area => {
            windowed_falloff(distance, range)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_light::LocalLight;

    const F: &str = include_str!("../tests/fixtures/local-light-math.txt");

    fn close(actual: f32, expected: f32) -> bool {
        (actual - expected).abs() <= (expected.abs() * 1e-4).max(1e-9)
    }

    fn check(actual: f32, expected: f32, name: &str) {
        assert!(close(actual, expected), "{name}: {actual} != {expected}");
    }

    fn cone_cosines(angle_deg: f32, inner_deg: f32) -> (f32, f32) {
        let outer = (angle_deg * 0.5).to_radians().cos();
        let inner = (inner_deg * 0.5).to_radians().cos();
        (outer, inner)
    }

    #[test]
    fn windowed_falloff_matches_fixture() {
        let mut cases = 0;
        for line in F.lines() {
            let mut fields = line.split_whitespace();
            if fields.next() != Some("falloff") {
                continue;
            }
            let name = fields.next().unwrap();
            let distance: f32 = fields.next().unwrap().parse().unwrap();
            let range: f32 = fields.next().unwrap().parse().unwrap();
            let expected: f32 = fields.next().unwrap().parse().unwrap();
            assert!(fields.next().is_none(), "{name}: extra tokens");
            check(windowed_falloff(distance, range), expected, name);
            cases += 1;
        }
        assert_eq!(cases, 11, "falloff fixture cases");
        assert_eq!(windowed_falloff(0.0, 6.0), 25.0);
        assert_eq!(windowed_falloff(6.0, 6.0), 0.0);
    }

    #[test]
    fn spot_factor_matches_fixture() {
        let mut cases = 0;
        for line in F.lines() {
            let mut fields = line.split_whitespace();
            if fields.next() != Some("spot") {
                continue;
            }
            let name = fields.next().unwrap();
            let cosine: f32 = fields.next().unwrap().parse().unwrap();
            let outer: f32 = fields.next().unwrap().parse().unwrap();
            let inner: f32 = fields.next().unwrap().parse().unwrap();
            let expected: f32 = fields.next().unwrap().parse().unwrap();
            assert!(fields.next().is_none(), "{name}: extra tokens");
            check(spot_factor(cosine, outer, inner), expected, name);
            cases += 1;
        }
        assert_eq!(cases, 8, "spot fixture cases");
    }

    #[test]
    fn hard_boundary_and_default_softness_match_fixture() {
        let (mut hard, mut cone) = (0, 0);
        for line in F.lines() {
            let mut fields = line.split_whitespace();
            let Some(tag) = fields.next() else {
                continue;
            };
            match tag {
                "hard" => {
                    let name = fields.next().unwrap();
                    let cosine: f32 = fields.next().unwrap().parse().unwrap();
                    let boundary: f32 = fields.next().unwrap().parse().unwrap();
                    let expected: f32 = fields.next().unwrap().parse().unwrap();
                    check(spot_factor(cosine, boundary, boundary), expected, name);
                    hard += 1;
                }
                "cone" => {
                    let name = fields.next().unwrap();
                    let angle: f32 = fields.next().unwrap().parse().unwrap();
                    let inner_deg: f32 = fields.next().unwrap().parse().unwrap();
                    let t: f32 = fields.next().unwrap().parse().unwrap();
                    let expected: f32 = fields.next().unwrap().parse().unwrap();
                    let (outer, inner) = cone_cosines(angle, inner_deg);
                    check(
                        spot_factor(outer + t * (inner - outer), outer, inner),
                        expected,
                        name,
                    );
                    cone += 1;
                }
                _ => {}
            }
        }
        assert_eq!((hard, cone), (3, 6), "hard/cone fixture cases");
    }

    #[test]
    fn default_spot_softness_uses_eighty_percent_inner_angle() {
        let light = LocalLight::default();
        assert_eq!((light.angle_deg, light.inner_angle_deg), (35.0, None));
        assert_eq!(light.effective_inner_angle_deg(), 28.0);
        let (outer, inner) = cone_cosines(light.angle_deg, light.effective_inner_angle_deg());
        assert_eq!(spot_factor(inner, outer, inner), 1.0);
        assert_eq!(spot_factor(outer, outer, inner), 0.0);
        assert!(close(spot_factor(0.5 * (outer + inner), outer, inner), 0.5));
        assert_eq!(spot_factor(outer - 0.01, outer, inner), 0.0);
        let authored = LocalLight {
            angle_deg: 60.0,
            inner_angle_deg: Some(45.0),
            ..LocalLight::default()
        };
        assert_eq!(authored.effective_inner_angle_deg(), 45.0);
    }

    #[test]
    fn directional_lights_ignore_distance_and_range() {
        let near = kind_falloff(LocalLightKind::Directional, 0.0, 0.5);
        let far = kind_falloff(LocalLightKind::Directional, 1_000.0, 0.5);
        assert_eq!((near, far), (1.0, 1.0));
        assert_eq!(kind_falloff(LocalLightKind::Point, 1_000.0, 0.5), 0.0);
        assert_eq!(
            kind_falloff(LocalLightKind::Spot, 0.25, 6.0),
            windowed_falloff(0.25, 6.0)
        );
        assert_eq!(
            kind_falloff(LocalLightKind::Area, 3.0, 6.0),
            windowed_falloff(3.0, 6.0)
        );
    }
}
