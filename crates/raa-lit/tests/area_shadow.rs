//! Contract tests of the CPU area-light soft-shadow base: settings validation, tier names,
//! tier weights, the rectangle distance and the hero projection. Expected values are derived
//! analytically from the contract.

use raa_lit::{
    AreaShadowSettings, AreaShadowTier, HERO_LAYER_BASE, HERO_NEAR, MAX_HERO_LIGHTS,
    hero_view_proj, rect_distance, tier_weights,
};
use raa_math::Vec3;

fn defaults() -> AreaShadowSettings {
    AreaShadowSettings {
        tier: AreaShadowTier::Auto,
        hero_count: 0,
        hero_distance: 25.0,
        fade_distance: 80.0,
        band: 5.0,
    }
}

fn refusal(settings: AreaShadowSettings) -> raa_core::RaaError {
    match settings.validate() {
        Err(error) => error,
        Ok(_) => panic!("expected a refusal"),
    }
}

fn assert_refused(settings: AreaShadowSettings, message: &str) {
    let error = refusal(settings);
    assert_eq!(error.code(), "light.invalid_area_shadows");
    assert_eq!(error.message(), message);
}

#[test]
fn constants_and_defaults_validate() {
    assert_eq!(MAX_HERO_LIGHTS, 4);
    assert_eq!(HERO_LAYER_BASE, 4);
    assert_eq!(HERO_NEAR, 0.05);
    let settings = defaults();
    assert_eq!(settings.validate(), Ok(settings));
}

#[test]
fn refusals_use_the_contract_code_and_messages() {
    assert_refused(
        AreaShadowSettings {
            hero_count: 5,
            ..defaults()
        },
        "heroCount: must be an integer in [0, 4] (got 5)",
    );
    assert_refused(
        AreaShadowSettings {
            hero_distance: 0.0,
            ..defaults()
        },
        "heroDistance: must be finite and > 0 (got 0)",
    );
    assert_refused(
        AreaShadowSettings {
            hero_distance: f32::NAN,
            ..defaults()
        },
        "heroDistance: must be finite and > 0 (got NaN)",
    );
    assert_refused(
        AreaShadowSettings {
            hero_distance: 25.0,
            fade_distance: 25.0,
            ..defaults()
        },
        "fadeDistance: must be finite and > heroDistance (got 25)",
    );
    assert_refused(
        AreaShadowSettings {
            band: 0.0,
            ..defaults()
        },
        "band: must be finite, > 0 and <= heroDistance (got 0)",
    );
    assert_refused(
        AreaShadowSettings {
            band: 26.0,
            ..defaults()
        },
        "band: must be finite, > 0 and <= heroDistance (got 26)",
    );
}

#[test]
fn tier_names_round_trip() {
    for tier in [
        AreaShadowTier::Auto,
        AreaShadowTier::Hero,
        AreaShadowTier::Cone,
        AreaShadowTier::Fade,
    ] {
        assert_eq!(AreaShadowTier::from_name(tier.as_str()), Some(tier));
    }
    assert_eq!(AreaShadowTier::from_name("x"), None);
}

fn weights(tier: AreaShadowTier, hero: bool, distance: f32) -> (f32, f32) {
    let settings = AreaShadowSettings { tier, ..defaults() };
    tier_weights(&settings, hero, distance)
}

#[test]
fn auto_weights_follow_the_hero_and_fade_bands() {
    assert_eq!(weights(AreaShadowTier::Auto, true, 10.0), (0.0, 1.0));
    assert_eq!(weights(AreaShadowTier::Auto, true, 22.5), (0.5, 1.0));
    assert_eq!(weights(AreaShadowTier::Auto, true, 30.0), (1.0, 1.0));
    assert_eq!(weights(AreaShadowTier::Auto, false, 10.0), (1.0, 1.0));
    assert_eq!(weights(AreaShadowTier::Auto, true, 77.5), (1.0, 0.5));
    assert_eq!(weights(AreaShadowTier::Auto, true, 90.0), (1.0, 0.0));
}

#[test]
fn fixed_tiers_ignore_distance() {
    assert_eq!(weights(AreaShadowTier::Hero, true, 10.0), (0.0, 1.0));
    assert_eq!(weights(AreaShadowTier::Hero, false, 10.0), (1.0, 1.0));
    assert_eq!(weights(AreaShadowTier::Cone, true, 10.0), (1.0, 1.0));
    assert_eq!(weights(AreaShadowTier::Fade, true, 10.0), (1.0, 0.0));
}

fn assert_close(got: f32, want: f32) {
    let tolerance = 1e-5 * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tolerance,
        "got {got}, want {want} (tolerance {tolerance})"
    );
}

fn basis(direction: Vec3) -> (Vec3, Vec3) {
    let up0 = if direction.dot(Vec3::Y).abs() < 0.99 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = up0.cross(direction).normalize_or_zero();
    (tangent, direction.cross(tangent))
}

#[test]
fn hero_projection_matches_the_explicit_form() {
    let position = Vec3::new(1.0, 5.0, 2.0);
    let direction = Vec3::new(0.0, -1.0, 0.0);
    let range: f32 = 30.0;
    let far = range.max(2.0 * HERO_NEAR);
    let (tangent, bitangent) = basis(direction);
    let f = 1.0 / 2.7474774;
    let matrix = hero_view_proj(position, direction, range);

    for point in [
        Vec3::new(1.0, 4.0, 2.0),
        Vec3::new(2.0, 3.0, 3.0),
        Vec3::new(0.0, 2.0, 1.0),
    ] {
        let d = point - position;
        let dz = d.dot(direction);
        let want = Vec3::new(
            -d.dot(tangent) * f / dz,
            d.dot(bitangent) * f / dz,
            far * (dz - HERO_NEAR) / ((far - HERO_NEAR) * dz),
        );
        let got = matrix.transform_point(point);
        assert_close(got.x, want.x);
        assert_close(got.y, want.y);
        assert_close(got.z, want.z);
    }
}

#[test]
fn hero_projection_depth_endpoints() {
    let position = Vec3::new(1.0, 5.0, 2.0);
    let direction = Vec3::new(0.0, -1.0, 0.0);
    let matrix = hero_view_proj(position, direction, 30.0);
    let near_point = position + direction * HERO_NEAR;
    let far_point = position + direction * 30.0;
    assert_close(matrix.transform_point(near_point).z, 0.0);
    assert_close(matrix.transform_point(far_point).z, 1.0);
}

#[test]
fn rectangle_distance_clamps_the_eye_onto_the_light() {
    let position = Vec3::new(0.0, 0.0, 0.0);
    let direction = Vec3::new(0.0, -1.0, 0.0);
    let eye = Vec3::new(0.0, -3.0, 0.0);
    assert_close(rect_distance(eye, position, direction, [1.0, 1.0]), 3.0);

    let (tangent, _) = basis(direction);
    let along = eye + tangent * 2.0;
    let expected = (3.0f32 * 3.0 + 1.5 * 1.5).sqrt();
    assert_close(
        rect_distance(along, position, direction, [1.0, 1.0]),
        expected,
    );
}
