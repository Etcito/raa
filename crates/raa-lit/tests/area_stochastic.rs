//! Contract tests of the CPU mid-tier base: the stochastic settings and their refusals, the
//! distance weight, and the 80-byte slot table codec. Expected values are derived from the
//! contract (`area-stochastic-settings` and `area-stochastic-gpu`), never from a GPU run.

use raa_core::RaaError;
use raa_lit::{
    AreaShadowCamera, AreaShadowSettings, AreaShadowTier, AreaStochasticFrame, LocalLight,
    LocalLightKind, LocalLights, MAX_LOCAL_LIGHTS, StochasticSelector, StochasticShadowSettings,
    stochastic_weight, table_cells, table_limit, table_slot, write_table,
};
use raa_math::Vec3;
use raa_pipeline::{AREA_STOCHASTIC_GPU_SIZE, MAX_STOCHASTIC_LIGHTS};
use raa_shader::LIGHT_GPU_SIZE;

fn settings(count: u32, distance: f32, cells: u32) -> StochasticShadowSettings {
    StochasticShadowSettings {
        count,
        distance,
        cells,
    }
}

fn refusal(value: StochasticShadowSettings) -> RaaError {
    match value.validate() {
        Err(error) => error,
        Ok(_) => panic!("expected a refusal for {value:?}"),
    }
}

fn assert_refused(value: StochasticShadowSettings, message: &str) {
    let error = refusal(value);
    assert_eq!(error.code(), "light.invalid_area_shadows");
    assert_eq!(error.message(), message);
}

#[test]
fn standard_settings_are_the_contract_literal() {
    let standard = StochasticShadowSettings::standard();
    assert_eq!(standard, settings(4, 40.0, 4));
    assert_eq!(standard.validate().ok(), Some(standard));
    assert_eq!(StochasticShadowSettings::default(), standard);
}

#[test]
fn refusals_name_the_first_bad_field_in_order() {
    assert_refused(
        settings(9, 40.0, 4),
        "stochasticCount: must be an integer in [0, 8] (got 9)",
    );
    assert_refused(
        settings(4, 0.0, 4),
        "stochasticDistance: must be finite and > 0 (got 0)",
    );
    assert_refused(
        settings(4, f32::NAN, 4),
        "stochasticDistance: must be finite and > 0 (got NaN)",
    );
    assert_refused(
        settings(4, 40.0, 5),
        "stochasticCells: must be 4 or 9 (got 5)",
    );
    assert_refused(
        settings(9, 40.0, 5),
        "stochasticCount: must be an integer in [0, 8] (got 9)",
    );
}

#[test]
fn count_and_cells_bounds_are_accepted() {
    for count in [0, 8] {
        assert!(settings(count, 40.0, 4).validate().is_ok(), "count {count}");
    }
    for cells in [4, 9] {
        assert!(settings(4, 40.0, cells).validate().is_ok(), "cells {cells}");
    }
}

#[test]
fn stochastic_weight_blends_from_the_camera_to_the_distance() {
    let standard = StochasticShadowSettings::standard();
    let band = 5.0;
    let cases = [(10.0, 1.0), (40.0, 0.0), (50.0, 0.0), (37.5, 0.5)];
    for (distance, expected) in cases {
        let weight = stochastic_weight(&standard, band, distance);
        assert!(
            (weight - expected).abs() < 1e-6,
            "distance {distance}: weight {weight}, expected {expected}"
        );
    }
}

#[test]
fn write_table_encodes_slots_limit_and_cells() {
    let slots: [Option<(u32, f32)>; MAX_STOCHASTIC_LIGHTS] = [
        Some((7, 0.25)),
        None,
        None,
        Some((2, 1.0)),
        None,
        None,
        None,
        None,
    ];
    let mut table = [0xAB_u8; AREA_STOCHASTIC_GPU_SIZE];
    write_table(&mut table, &slots, 4, 9);

    assert_eq!(&table[0..4], &7u32.to_le_bytes());
    assert_eq!(&table[4..8], &0xFFFF_FFFFu32.to_le_bytes());
    assert_eq!(&table[12..16], &2u32.to_le_bytes());
    assert_eq!(&table[32..36], &0.25f32.to_le_bytes());
    assert_eq!(&table[44..48], &1.0f32.to_le_bytes());
    assert_eq!(&table[64..68], &4u32.to_le_bytes());
    assert_eq!(&table[68..72], &9u32.to_le_bytes());
    assert_eq!(&table[72..80], &[0u8; 8]);

    assert_eq!(table_slot(&table, 0), Some((7, 0.25)));
    assert_eq!(table_slot(&table, 1), None);
    assert_eq!(table_slot(&table, 3), Some((2, 1.0)));
    assert_eq!(table_slot(&table, 4), None);
    assert_eq!(table_limit(&table), 4);
    assert_eq!(table_cells(&table), 9);
}

#[test]
fn write_table_clamps_weight_and_limit() {
    let mut slots: [Option<(u32, f32)>; MAX_STOCHASTIC_LIGHTS] = [None; MAX_STOCHASTIC_LIGHTS];
    slots[0] = Some((5, 1.5));
    let mut table = [0_u8; AREA_STOCHASTIC_GPU_SIZE];
    write_table(&mut table, &slots, 12, 4);

    assert_eq!(&table[32..36], &1.0f32.to_le_bytes());
    assert_eq!(table_limit(&table), 8);
    assert_eq!(table_cells(&table), 4);
    assert_eq!(table_slot(&table, 0), Some((5, 1.0)));
}

#[test]
fn empty_frame_has_an_all_zero_table() {
    let frame = AreaStochasticFrame::empty();
    assert_eq!(frame.table, [0_u8; AREA_STOCHASTIC_GPU_SIZE]);
    assert_eq!(frame.generations, [0_u32; MAX_STOCHASTIC_LIGHTS]);
    assert_eq!(frame.active, 0);
    assert_eq!(table_limit(&frame.table), 0);
    assert_eq!(table_slot(&frame.table, 0), None);
    assert_eq!(AreaStochasticFrame::default(), frame);
}

const RECORDS: usize = MAX_LOCAL_LIGHTS * LIGHT_GPU_SIZE;
const ORIGIN: Vec3 = Vec3::new(0.0, 0.0, 0.0);
const DOWN_Z: Vec3 = Vec3::new(0.0, 0.0, -1.0);

fn camera(eye: Vec3) -> AreaShadowCamera {
    AreaShadowCamera {
        eye,
        forward: DOWN_Z,
        fov_y_rad: std::f32::consts::FRAC_PI_3,
        viewport: (1280, 720),
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

fn defaults() -> AreaShadowSettings {
    AreaShadowSettings {
        tier: AreaShadowTier::Auto,
        hero_count: 0,
        hero_distance: 25.0,
        fade_distance: 80.0,
        band: 5.0,
    }
}

fn one() -> StochasticShadowSettings {
    StochasticShadowSettings {
        count: 1,
        ..StochasticShadowSettings::standard()
    }
}

fn records_of(lights: &LocalLights) -> [u8; RECORDS] {
    let mut records = [0u8; RECORDS];
    lights.write_gpu(&mut records);
    records
}

fn set_lane(records: &mut [u8; RECORDS], record: usize, offset: usize, value: f32) {
    let start = record * LIGHT_GPU_SIZE + offset;
    records[start..start + 4].copy_from_slice(&value.to_le_bytes());
}

fn select(
    selector: &mut StochasticSelector,
    area: AreaShadowSettings,
    eye: Vec3,
    lights: &LocalLights,
    records: &[u8; RECORDS],
    out: &mut AreaStochasticFrame,
) {
    selector.update(&area, &one(), &camera(eye), lights, records, out);
}

#[test]
fn nearest_light_takes_slot_zero_with_count_one() {
    let lights = registry(&[
        area_at(Vec3::new(0.0, 0.0, -10.0), 1.0),
        area_at(Vec3::new(0.0, 0.0, -20.0), 1.0),
    ]);
    let records = records_of(&lights);
    let mut selector = StochasticSelector::new();
    let mut out = AreaStochasticFrame::empty();
    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );

    assert_eq!(table_slot(&out.table, 0), Some((0, 1.0)));
    assert_eq!(out.active, 1);
    assert_eq!(selector.active(), 1);
    assert_eq!(table_limit(&out.table), 1);
    assert_eq!(table_cells(&out.table), 4);
    assert_eq!(out.generations[0], 1);
}

#[test]
fn identical_second_update_leaves_the_frame_unchanged() {
    let lights = registry(&[
        area_at(Vec3::new(0.0, 0.0, -10.0), 1.0),
        area_at(Vec3::new(0.0, 0.0, -20.0), 1.0),
    ]);
    let records = records_of(&lights);
    let mut selector = StochasticSelector::new();
    let mut out = AreaStochasticFrame::empty();
    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );
    let before = out;
    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );
    assert_eq!(out, before);
}

#[test]
fn light_beyond_the_stochastic_distance_gets_no_slot() {
    let lights = registry(&[area_at(Vec3::new(0.0, 0.0, -60.0), 1.0)]);
    let records = records_of(&lights);
    let mut selector = StochasticSelector::new();
    let mut out = AreaStochasticFrame::empty();
    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );

    assert_eq!(table_slot(&out.table, 0), None);
    assert_eq!(out.active, 0);
    assert_eq!(out.generations[0], 0);
}

#[test]
fn hero_shadowed_light_is_excluded_unless_it_keeps_a_cone_weight() {
    let lights = registry(&[area_at(Vec3::new(0.0, 0.0, -10.0), 1.0)]);
    let mut records = records_of(&lights);
    let mut selector = StochasticSelector::new();
    let mut out = AreaStochasticFrame::empty();

    set_lane(&mut records, 0, 44, 4.0);
    set_lane(&mut records, 0, 48, 0.0);
    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );
    assert_eq!(table_slot(&out.table, 0), None);
    assert_eq!(out.active, 0);

    set_lane(&mut records, 0, 48, 0.5);
    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );
    assert_eq!(table_slot(&out.table, 0), Some((0, 1.0)));
    assert_eq!(out.active, 1);
}

#[test]
fn stochastic_slot_swaps_only_past_the_hysteresis_ratio() {
    let lights = registry(&[
        area_at(Vec3::new(0.0, 0.0, -10.0), 1.0),
        area_at(Vec3::new(5.0, 0.0, -10.0), 1.0),
    ]);
    let records = records_of(&lights);
    let mut selector = StochasticSelector::new();
    let mut out = AreaStochasticFrame::empty();

    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );
    assert_eq!(table_slot(&out.table, 0), Some((0, 1.0)));

    // Ratio 1.05x, not above 1.25x: the near light keeps slot 0.
    let eye = Vec3::new(3.0, 0.0, 0.0);
    select(&mut selector, defaults(), eye, &lights, &records, &mut out);
    assert_eq!(table_slot(&out.table, 0), Some((0, 1.0)));
    assert_eq!(out.generations[0], 1);

    // Ratio 1.51x, above 1.25x: the side light (index 1) takes slot 0.
    let eye = Vec3::new(8.0, 0.0, 0.0);
    select(&mut selector, defaults(), eye, &lights, &records, &mut out);
    assert_eq!(table_slot(&out.table, 0), Some((1, 1.0)));
    assert_eq!(out.generations[0], 2);
}

#[test]
fn cone_tier_releases_the_slot_and_bumps_its_generation() {
    let lights = registry(&[area_at(Vec3::new(0.0, 0.0, -10.0), 1.0)]);
    let records = records_of(&lights);
    let mut selector = StochasticSelector::new();
    let mut out = AreaStochasticFrame::empty();
    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );
    assert_eq!(out.generations[0], 1);

    let cone = AreaShadowSettings {
        tier: AreaShadowTier::Cone,
        ..defaults()
    };
    select(&mut selector, cone, ORIGIN, &lights, &records, &mut out);
    assert_eq!(table_limit(&out.table), 0);
    assert_eq!(out.active, 0);
    assert_eq!(table_slot(&out.table, 0), None);
    assert_eq!(out.generations[0], 2);
}

#[test]
fn point_lights_are_never_mid_tier_candidates() {
    let point = LocalLight {
        kind: LocalLightKind::Point,
        position: [0.0, 0.0, -10.0],
        range: 1.0,
        ..LocalLight::default()
    };
    let lights = registry(&[point, area_at(Vec3::new(0.0, 0.0, -20.0), 1.0)]);
    let records = records_of(&lights);
    let mut selector = StochasticSelector::new();
    let mut out = AreaStochasticFrame::empty();
    select(
        &mut selector,
        defaults(),
        ORIGIN,
        &lights,
        &records,
        &mut out,
    );

    assert_eq!(table_slot(&out.table, 0), Some((1, 1.0)));
    assert_eq!(out.active, 1);
}
