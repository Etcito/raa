//! Contract tests of the CPU area-light soft-shadow base: settings validation, tier names,
//! tier weights, the rectangle distance and the hero projection. Expected values are derived
//! analytically from the contract.

use raa_lit::{
    AreaShadowCamera, AreaShadowSettings, AreaShadowTier, HERO_LAYER_BASE, HERO_NEAR, HeroSelector,
    LocalLight, LocalLightKind, LocalLights, MAX_HERO_LIGHTS, MAX_LOCAL_LIGHTS, hero_view_proj,
    rect_distance, screen_coverage, tier_weights,
};
use raa_math::{Mat4, Vec3};
use raa_shader::LIGHT_GPU_SIZE;

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

const RECORDS: usize = MAX_LOCAL_LIGHTS * LIGHT_GPU_SIZE;
const DOWN_Z: Vec3 = Vec3::new(0.0, 0.0, -1.0);

fn origin() -> Vec3 {
    Vec3::new(0.0, 0.0, 0.0)
}

fn camera(eye: Vec3) -> AreaShadowCamera {
    AreaShadowCamera {
        eye,
        forward: DOWN_Z,
        fov_y_rad: std::f32::consts::FRAC_PI_2,
        viewport: (100, 100),
    }
}

fn area_at(position: Vec3, range: f32) -> LocalLight {
    LocalLight {
        kind: LocalLightKind::Area,
        position: position.to_array(),
        direction: DOWN_Z.to_array(),
        range,
        size: [1.0, 1.0],
        ..LocalLight::default()
    }
}

fn registry(lights: &[LocalLight]) -> LocalLights {
    let mut registry = LocalLights::new();
    for light in lights {
        registry.add(*light).unwrap();
    }
    registry
}

fn tiered(tier: AreaShadowTier, hero_count: u32) -> AreaShadowSettings {
    AreaShadowSettings {
        tier,
        hero_count,
        ..defaults()
    }
}

fn lane(records: &[u8], record: usize, offset: usize) -> f32 {
    let start = record * LIGHT_GPU_SIZE + offset;
    f32::from_le_bytes(records[start..start + 4].try_into().unwrap())
}

fn hero_at(position: Vec3, range: f32) -> Option<Mat4> {
    Some(hero_view_proj(position, DOWN_Z, range))
}

fn run(
    selector: &mut HeroSelector,
    settings: AreaShadowSettings,
    eye: Vec3,
    lights: &LocalLights,
    records: &mut [u8; RECORDS],
    heroes: &mut [Option<Mat4>; MAX_HERO_LIGHTS],
) -> bool {
    lights.write_gpu(records);
    selector.update(&settings, &camera(eye), lights, records, heroes)
}

#[test]
fn zero_heroes_give_every_area_record_the_cone_lanes() {
    let near = Vec3::new(0.0, 0.0, -10.0);
    let far = Vec3::new(0.0, 0.0, -30.0);
    let lights = registry(&[area_at(near, 1.0), area_at(far, 1.0)]);
    let settings = tiered(AreaShadowTier::Auto, 0);
    let mut records = [0u8; RECORDS];
    let mut heroes = [None; MAX_HERO_LIGHTS];
    let mut selector = HeroSelector::new();
    run(
        &mut selector,
        settings,
        origin(),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert_eq!(selector.active_heroes(), 0);
    assert!(heroes.iter().all(Option::is_none));
    for (record, position) in [(0, near), (1, far)] {
        let distance = rect_distance(origin(), position, DOWN_Z, [1.0, 1.0]);
        let (_, w_direct) = tier_weights(&settings, false, distance);
        assert_eq!(lane(&records, record, 28), w_direct);
        assert_eq!(lane(&records, record, 44), -1.0);
        assert_eq!(lane(&records, record, 48), 1.0);
    }
}

#[test]
fn hero_tier_puts_the_stronger_light_in_slot_zero() {
    let first = Vec3::new(0.0, 0.0, -10.0);
    let second = Vec3::new(0.0, 0.0, -20.0);
    let lights = registry(&[area_at(first, 1.0), area_at(second, 1.0)]);
    let mut records = [0u8; RECORDS];
    let mut heroes = [None; MAX_HERO_LIGHTS];
    let mut selector = HeroSelector::new();
    let changed = run(
        &mut selector,
        tiered(AreaShadowTier::Hero, 1),
        origin(),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert!(changed);
    assert_eq!(selector.active_heroes(), 1);
    assert_eq!(lane(&records, 0, 44), 4.0);
    assert_eq!(lane(&records, 0, 48), 0.0);
    assert_eq!(lane(&records, 1, 44), -1.0);
    assert_eq!(lane(&records, 1, 48), 1.0);
    assert_eq!(heroes[0], hero_at(first, 1.0));
    assert!(heroes[1..].iter().all(Option::is_none));
}

#[test]
fn hero_slot_swaps_only_past_the_hysteresis_ratio() {
    let near = Vec3::new(0.0, 0.0, -10.0);
    let side = Vec3::new(5.0, 0.0, -10.0);
    let lights = registry(&[area_at(near, 1.0), area_at(side, 1.0)]);
    let settings = tiered(AreaShadowTier::Hero, 1);
    let mut selector = HeroSelector::new();
    let mut records = [0u8; RECORDS];
    let mut heroes = [None; MAX_HERO_LIGHTS];

    // The near light has the larger coverage, so it takes slot 0 in the first call.
    run(
        &mut selector,
        settings,
        origin(),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert_eq!(lane(&records, 0, 44), 4.0);

    // The side light is closer and its ratio is 1.05x, not above 1.25x: no swap.
    run(
        &mut selector,
        settings,
        Vec3::new(3.0, 0.0, 0.0),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert_eq!(lane(&records, 0, 44), 4.0);
    assert_eq!(lane(&records, 1, 44), -1.0);

    // Ratio 1.51x, above 1.25x: the side light takes slot 0.
    let changed = run(
        &mut selector,
        settings,
        Vec3::new(8.0, 0.0, 0.0),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert!(changed);
    assert_eq!(lane(&records, 1, 44), 4.0);
    assert_eq!(lane(&records, 1, 48), 0.0);
    assert_eq!(lane(&records, 0, 44), -1.0);
    assert_eq!(lane(&records, 0, 48), 1.0);
    assert_eq!(heroes[0], hero_at(side, 1.0));
    assert!(heroes[1..].iter().all(Option::is_none));
}

#[test]
fn cone_tier_releases_every_slot() {
    let lights = registry(&[area_at(Vec3::new(0.0, 0.0, -10.0), 1.0)]);
    let mut records = [0u8; RECORDS];
    let mut heroes = [None; MAX_HERO_LIGHTS];
    let mut selector = HeroSelector::new();
    run(
        &mut selector,
        tiered(AreaShadowTier::Hero, 1),
        origin(),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert_eq!(selector.active_heroes(), 1);
    let changed = run(
        &mut selector,
        tiered(AreaShadowTier::Cone, 1),
        origin(),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert!(changed);
    assert_eq!(selector.active_heroes(), 0);
    assert!(heroes.iter().all(Option::is_none));
    assert_eq!(lane(&records, 0, 44), -1.0);
}

#[test]
fn identical_second_update_changes_no_byte() {
    let lights = registry(&[area_at(Vec3::new(0.0, 0.0, -10.0), 1.0)]);
    let settings = tiered(AreaShadowTier::Hero, 1);
    let mut records = [0u8; RECORDS];
    let mut heroes = [None; MAX_HERO_LIGHTS];
    let mut selector = HeroSelector::new();
    run(
        &mut selector,
        settings,
        origin(),
        &lights,
        &mut records,
        &mut heroes,
    );
    let changed = selector.update(
        &settings,
        &camera(origin()),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert!(!changed);
}

#[test]
fn point_records_stay_as_write_gpu_wrote_them() {
    let point = LocalLight {
        position: [1.0, 2.0, 3.0],
        ..LocalLight::default()
    };
    let lights = registry(&[point, area_at(Vec3::new(0.0, 0.0, -10.0), 1.0)]);
    let mut expected = [0u8; RECORDS];
    lights.write_gpu(&mut expected);
    let mut records = [0u8; RECORDS];
    let mut heroes = [None; MAX_HERO_LIGHTS];
    let mut selector = HeroSelector::new();
    run(
        &mut selector,
        tiered(AreaShadowTier::Hero, 1),
        origin(),
        &lights,
        &mut records,
        &mut heroes,
    );
    assert_eq!(records[..LIGHT_GPU_SIZE], expected[..LIGHT_GPU_SIZE]);
    assert_eq!(lane(&records, 1, 44), 4.0);
}

#[test]
fn screen_coverage_follows_the_contract_cases() {
    assert_eq!(
        screen_coverage(&camera(origin()), Vec3::new(0.0, 0.0, 0.5), 1.0),
        1.0
    );
    assert_eq!(
        screen_coverage(&camera(origin()), Vec3::new(0.0, 0.0, 10.0), 1.0),
        0.0
    );
    let value = screen_coverage(&camera(origin()), Vec3::new(0.0, 0.0, -10.0), 1.0);
    assert!((value - 0.00793329).abs() <= 1e-5, "got {value}");
}

#[test]
fn auto_tier_never_heroes_a_light_beyond_hero_distance() {
    let far = area_at(Vec3::new(0.0, 0.0, -30.0), 100.0);
    let near = area_at(Vec3::new(0.0, 0.0, -10.0), 1.0);
    let settings = tiered(AreaShadowTier::Auto, 2);
    let mut records = [0u8; RECORDS];
    let mut heroes = [None; MAX_HERO_LIGHTS];
    let mut selector = HeroSelector::new();

    // Alone, the far light (coverage 1, distance 30 >= 25) gets no slot.
    let alone = registry(&[far]);
    run(
        &mut selector,
        settings,
        origin(),
        &alone,
        &mut records,
        &mut heroes,
    );
    assert_eq!(selector.active_heroes(), 0);
    assert!(heroes.iter().all(Option::is_none));
    assert_eq!(lane(&records, 0, 44), -1.0);

    // With the near light present, only the near light (distance 10) becomes a hero.
    let both = registry(&[far, near]);
    run(
        &mut selector,
        settings,
        origin(),
        &both,
        &mut records,
        &mut heroes,
    );
    assert_eq!(selector.active_heroes(), 1);
    assert_eq!(lane(&records, 0, 44), -1.0);
    assert_eq!(lane(&records, 1, 44), 4.0);
    assert_eq!(heroes[0], hero_at(Vec3::new(0.0, 0.0, -10.0), 1.0));
}
