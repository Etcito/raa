//! Contract tests of the CPU mid-tier base: the stochastic settings and their refusals, the
//! distance weight, and the 80-byte slot table codec. Expected values are derived from the
//! contract (`area-stochastic-settings` and `area-stochastic-gpu`), never from a GPU run.

use raa_core::RaaError;
use raa_lit::{
    AreaStochasticFrame, StochasticShadowSettings, stochastic_weight, table_cells, table_limit,
    table_slot, write_table,
};
use raa_pipeline::{AREA_STOCHASTIC_GPU_SIZE, MAX_STOCHASTIC_LIGHTS};

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
