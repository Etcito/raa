//! S11 CPU local-light mathematics: point window/falloff, spot cone and the directional
//! no-distance rule. Descriptor validation keeps every input finite; `distance` is
//! nonnegative, `range` positive and a cone cosine is a dot product of unit vectors.
//!
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-render/shaders/lighting/deferred.wgsl`, `fn local_light`, revision 7e465b5.
//!
//! Handoff: this rectangle share adds `pub(crate) fn rectangle_basis(direction: Vec3) -> (Vec3, Vec3)`
//! and `pub(crate) fn representative_point(position: Vec3, direction: Vec3, size: [f32; 2], surface: Vec3) -> Vec3`
//! to `crates/raa-lit/src/light_math.rs`, below the functions here. `crates/raa-lit/src/lib.rs`
//! attaches this module with `mod light_math;`; these pure functions pass no registry state and
//! no keys. Shader work (`raa-shader`, `LOCAL_LIGHTING_WGSL`, `raa_local_lighting`) evaluates the
//! same S11 basis, representative point and one-sided emission rule from the light's normalized
//! direction and full metre extents.

#![allow(dead_code)] // Staged before tile selection and shading consumers land; see Handoff.

use crate::local_light::LocalLightKind;
use raa_math::Vec3;

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

/// Deterministic S11 rectangle basis of an area light's unit forward axis `direction`.
///
/// `up0` is `+Y` when `abs(dot(direction, Y)) < 0.99` and `+X` otherwise, the tangent is
/// `normalize(cross(up0, direction))` and the bitangent is `cross(direction, tangent)`, so
/// `(tangent, bitangent, direction)` is orthonormal and right-handed
/// (`tangent.cross(bitangent) == direction`). The rectangle spans `size[0]` along `tangent`
/// and `size[1]` along `bitangent`.
///
/// Handoff: `pub(crate) fn rectangle_basis(direction: Vec3) -> (Vec3, Vec3)` lives in
/// `crates/raa-lit/src/light_math.rs`, attached by `mod light_math;` in
/// `crates/raa-lit/src/lib.rs`. Shader work (`raa-shader`, `LOCAL_LIGHTING_WGSL`) evaluates the
/// same basis from the light's normalized direction and passes no registry state or keys.
pub(crate) fn rectangle_basis(direction: Vec3) -> (Vec3, Vec3) {
    let d = direction.normalize_or_zero();
    let up0 = if d.dot(Vec3::Y).abs() < 0.99 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = up0.cross(d).normalize_or_zero();
    let bitangent = d.cross(tangent);
    (tangent, bitangent)
}

/// S11 representative point of an area light's oriented rectangle for `surface`.
///
/// With `v = surface - position` and `(t, b) = rectangle_basis(direction)`, the point is
/// `position + t * clamp(dot(v, t), -size[0]/2, size[0]/2) + b * clamp(dot(v, b), -size[1]/2, size[1]/2)`.
/// The point stays on the rectangle plane, so a mathematical `size` of `[0, 0]` degenerates to
/// `position`; the S17 public descriptor still refuses non-positive extents. The caller applies
/// the one-sided rule ([`area_light_emits`]) before evaluating direction and distance from it.
///
/// Handoff: `pub(crate) fn representative_point(position: Vec3, direction: Vec3, size: [f32; 2], surface: Vec3) -> Vec3`
/// lives in `crates/raa-lit/src/light_math.rs`, attached by `mod light_math;` in
/// `crates/raa-lit/src/lib.rs`. Shader work (`raa-shader`, `LOCAL_LIGHTING_WGSL`) evaluates the
/// same point from the light's position, normalized direction and full metre extents and passes
/// no registry state or keys.
pub(crate) fn representative_point(
    position: Vec3,
    direction: Vec3,
    size: [f32; 2],
    surface: Vec3,
) -> Vec3 {
    let (tangent, bitangent) = rectangle_basis(direction);
    let v = surface - position;
    let half_t = size[0] * 0.5;
    let half_b = size[1] * 0.5;
    let along_t = v.dot(tangent).clamp(-half_t, half_t);
    let along_b = v.dot(bitangent).clamp(-half_b, half_b);
    position + tangent * along_t + bitangent * along_b
}

/// S11 one-sided emission: the rectangle emits along `+direction`, so a `surface` with
/// `dot(surface - position, direction) <= 0` is behind the emitter and contributes nothing.
///
/// Handoff: `pub(crate) fn area_light_emits(position: Vec3, direction: Vec3, surface: Vec3) -> bool`
/// lives in `crates/raa-lit/src/light_math.rs`, attached by `mod light_math;` in
/// `crates/raa-lit/src/lib.rs`. Shader work (`raa-shader`, `LOCAL_LIGHTING_WGSL`) applies the
/// same predicate with the light's normalized direction and no registry state or keys.
pub(crate) fn area_light_emits(position: Vec3, direction: Vec3, surface: Vec3) -> bool {
    (surface - position).dot(direction.normalize_or_zero()) > 0.0
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

    fn check3(actual: Vec3, expected: Vec3, name: &str) {
        check(actual.x, expected.x, name);
        check(actual.y, expected.y, name);
        check(actual.z, expected.z, name);
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

    #[test]
    fn rectangle_basis_matches_fixture_and_is_orthonormal() {
        let mut cases = 0;
        for line in F.lines() {
            let mut fields = line.split_whitespace();
            if fields.next() != Some("basis") {
                continue;
            }
            let name = fields.next().unwrap();
            let values: Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
            assert_eq!(values.len(), 9, "{name}: field count");
            let direction = Vec3::new(values[0], values[1], values[2]);
            let expected_t = Vec3::new(values[3], values[4], values[5]);
            let expected_b = Vec3::new(values[6], values[7], values[8]);
            let (t, b) = rectangle_basis(direction);
            check3(t, expected_t, name);
            check3(b, expected_b, name);
            let d = direction.normalize_or_zero();
            assert!((t.length() - 1.0).abs() < 1e-5, "{name}: |tangent| != 1");
            assert!((b.length() - 1.0).abs() < 1e-5, "{name}: |bitangent| != 1");
            assert!(t.dot(b).abs() < 1e-5, "{name}: t not orthogonal to b");
            assert!(t.dot(d).abs() < 1e-5, "{name}: t not orthogonal to d");
            assert!(b.dot(d).abs() < 1e-5, "{name}: b not orthogonal to d");
            assert!(
                (t.cross(b) - d).length() < 1e-5,
                "{name}: (t, b, d) not right-handed"
            );
            cases += 1;
        }
        assert_eq!(cases, 6, "basis fixture cases");
    }

    #[test]
    fn representative_point_matches_fixture() {
        let mut cases = 0;
        for line in F.lines() {
            let mut fields = line.split_whitespace();
            if fields.next() != Some("point") {
                continue;
            }
            let name = fields.next().unwrap();
            let values: Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
            assert_eq!(values.len(), 14, "{name}: field count");
            let position = Vec3::new(values[0], values[1], values[2]);
            let direction = Vec3::new(values[3], values[4], values[5]);
            let size = [values[6], values[7]];
            let surface = Vec3::new(values[8], values[9], values[10]);
            let expected = Vec3::new(values[11], values[12], values[13]);
            check3(
                representative_point(position, direction, size, surface),
                expected,
                name,
            );
            cases += 1;
        }
        assert_eq!(cases, 12, "representative point fixture cases");
    }

    #[test]
    fn area_emission_is_one_sided_matching_fixture() {
        let mut cases = 0;
        for line in F.lines() {
            let mut fields = line.split_whitespace();
            if fields.next() != Some("emit") {
                continue;
            }
            let name = fields.next().unwrap();
            let values: Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
            assert_eq!(values.len(), 10, "{name}: field count");
            let position = Vec3::new(values[0], values[1], values[2]);
            let direction = Vec3::new(values[3], values[4], values[5]);
            let surface = Vec3::new(values[6], values[7], values[8]);
            let expected = values[9] != 0.0;
            assert_eq!(
                area_light_emits(position, direction, surface),
                expected,
                "{name}"
            );
            cases += 1;
        }
        assert_eq!(cases, 5, "emit fixture cases");
    }

    #[test]
    fn area_near_field_is_finite_matching_fixture() {
        let mut cases = 0;
        for line in F.lines() {
            let mut fields = line.split_whitespace();
            if fields.next() != Some("near") {
                continue;
            }
            let name = fields.next().unwrap();
            let values: Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
            assert_eq!(values.len(), 13, "{name}: field count");
            let position = Vec3::new(values[0], values[1], values[2]);
            let direction = Vec3::new(values[3], values[4], values[5]);
            let size = [values[6], values[7]];
            let surface = Vec3::new(values[8], values[9], values[10]);
            let range = values[11];
            let expected = values[12];
            let point = representative_point(position, direction, size, surface);
            let distance = (surface - point).length();
            assert!(distance.is_finite(), "{name}: distance is not finite");
            check(windowed_falloff(distance, range), expected, name);
            cases += 1;
        }
        assert_eq!(cases, 3, "near fixture cases");
    }

    #[test]
    fn full_extents_clamp_at_half_size() {
        let position = Vec3::ZERO;
        let direction = Vec3::new(0.0, 0.0, 1.0);
        let surface = Vec3::new(100.0, -100.0, 5.0);
        let point = representative_point(position, direction, [2.0, 4.0], surface);
        check3(point, Vec3::new(1.0, -2.0, 0.0), "full_extents");
    }

    #[test]
    fn zero_size_degenerates_while_public_extents_stay_positive() {
        let position = Vec3::new(1.0, 2.0, 3.0);
        let direction = Vec3::new(0.0, 0.0, 1.0);
        let surface = Vec3::new(9.0, -4.0, 12.0);
        let point = representative_point(position, direction, [0.0, 0.0], surface);
        check3(point, position, "zero_size_degenerates");
        assert!(area_light_emits(position, direction, surface));
        for size in [[0.0, 1.0], [0.0, 0.0]] {
            let light = LocalLight {
                size,
                ..LocalLight::default()
            };
            assert_eq!(light.validate().unwrap_err().code(), "light.invalid_desc");
        }
        assert!(LocalLight::default().validate().is_ok());
    }
}
