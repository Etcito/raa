//! Subject tests of the native foliage volume accumulator (M7, contract `m7-volume`): the
//! closed-form weighted mean of a uniform coloured patch, an untouched volume, saturated
//! occlusion, the boundary voxel mapping at both resolutions, the deterministic sequence
//! and the refusals. The module is path-compiled until production registration.

#![cfg(not(target_arch = "wasm32"))]

#[path = "../src/foliage_volume.rs"]
mod foliage_volume;

use foliage_volume::FoliageVolumeAccumulator;
use raa_core::RaaError;

const UNIT_BOX: ([f32; 3], [f32; 3]) = ([0.0; 3], [1.0; 3]);

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

fn quantize(value: f64) -> u8 {
    (value * 255.0).round().clamp(0.0, 255.0) as u8
}

fn alpha(area: f64, mean_face_area: f64) -> u8 {
    quantize(1.0 - (-area / mean_face_area).exp())
}

fn base(x: usize, y: usize, z: usize, resolution: usize) -> usize {
    4 * (x + resolution * (y + resolution * z))
}

fn accumulator(resolution: u32) -> FoliageVolumeAccumulator {
    FoliageVolumeAccumulator::new(UNIT_BOX.0, UNIT_BOX.1, resolution).expect("valid volume box")
}

#[test]
fn uniform_patch_matches_the_closed_form_weighted_mean() {
    let mut volume = accumulator(16);
    volume
        .add([0.01, 0.01, 0.01], [1.0, 0.0, 0.0], 1.0)
        .expect("red contribution");
    volume
        .add([0.02, 0.02, 0.02], [0.0, 0.0, 1.0], 3.0)
        .expect("blue contribution");
    let bytes = volume.finish();
    let cell = &bytes[base(0, 0, 0, 16)..base(0, 0, 0, 16) + 4];
    assert_eq!(cell[0], 64, "red is 1 / 4 of the area");
    assert_eq!(cell[1], 0);
    assert_eq!(cell[2], 191, "blue is 3 / 4 of the area");
    let mean = 1.0 / 256.0;
    assert_eq!(
        cell[3],
        alpha(4.0, mean),
        "optical depth of 4 square metres"
    );
}

#[test]
fn a_uniform_colour_repeats_in_every_occupied_cell() {
    let mut volume = accumulator(16);
    let albedo = [0.25, 0.5, 0.75];
    for step in 0..8 {
        let offset = step as f32 * 0.125 + 0.0625;
        volume
            .add([offset, offset, offset], albedo, 0.5)
            .expect("uniform contribution");
    }
    let bytes = volume.finish();
    let mean = 1.0 / 256.0;
    for step in 0..8 {
        let voxel = (step * 2) + 1;
        let cell = &bytes[base(voxel, voxel, voxel, 16)..base(voxel, voxel, voxel, 16) + 4];
        assert_eq!(cell[0], quantize(0.25));
        assert_eq!(cell[1], quantize(0.5));
        assert_eq!(cell[2], quantize(0.75));
        assert_eq!(cell[3], alpha(0.5, mean));
    }
}

#[test]
fn zero_occupancy_leaves_every_byte_zero_at_both_resolutions() {
    for resolution in [16u32, 32] {
        let mut volume = accumulator(resolution);
        volume
            .add([0.5, 0.5, 0.5], [1.0, 1.0, 1.0], 0.0)
            .expect("zero area is accepted");
        let bytes = volume.finish();
        let resolution = resolution as usize;
        assert_eq!(bytes.len(), 4 * resolution * resolution * resolution);
        assert!(
            bytes.iter().all(|byte| *byte == 0),
            "resolution {resolution}"
        );
    }
}

#[test]
fn occlusion_saturates_and_halves_at_the_logarithmic_midpoint() {
    let mean = 1.0 / 256.0;
    let mut half = accumulator(16);
    half.add(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        (mean * 2.0f64.ln()) as f32,
    )
    .expect("half occlusion");
    let half_bytes = half.finish();
    assert_eq!(half_bytes[base(8, 8, 8, 16) + 3], 128);

    let mut saturated = accumulator(16);
    saturated
        .add([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1.0e9)
        .expect("saturated occlusion");
    let bytes = saturated.finish();
    assert_eq!(
        bytes[base(8, 8, 8, 16) + 3],
        255,
        "the canopy becomes fully opaque"
    );
}

#[test]
fn coordinates_clamp_into_the_boundary_voxels() {
    let mut volume = accumulator(16);
    volume
        .add([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0)
        .expect("minimum corner");
    volume
        .add([1.0, 1.0, 1.0], [0.0, 1.0, 0.0], 1.0)
        .expect("maximum corner");
    volume
        .add([2.0, -1.0, 0.5], [0.0, 0.0, 1.0], 1.0)
        .expect("clamped corner");
    let bytes = volume.finish();
    assert_ne!(
        bytes[base(0, 0, 0, 16) + 3],
        0,
        "minimum maps to voxel zero"
    );
    assert_ne!(
        bytes[base(15, 15, 15, 16) + 3],
        0,
        "the maximum maps to the last voxel, not past it"
    );
    assert_ne!(
        bytes[base(15, 0, 8, 16) + 3],
        0,
        "out-of-box coordinates clamp on every axis"
    );
}

#[test]
fn the_maximum_boundary_lands_in_the_last_voxel_at_32() {
    let mut volume = accumulator(32);
    volume
        .add([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0)
        .expect("maximum corner");
    let bytes = volume.finish();
    assert_eq!(bytes.len(), 4 * 32 * 32 * 32);
    assert_ne!(bytes[base(31, 31, 31, 32) + 3], 0);
    assert_eq!(bytes[base(31, 31, 31, 32)], 255);
}

#[test]
fn a_repeated_sequence_finishes_byte_identically() {
    let samples = [
        ([0.1, 0.2, 0.3], [0.4, 0.6, 0.8], 0.25),
        ([0.7, 0.2, 0.3], [0.1, 0.2, 0.3], 0.75),
        ([0.1, 0.9, 0.3], [0.9, 0.1, 0.5], 1.5),
        ([0.1, 0.2, 0.95], [0.3, 0.3, 0.3], 0.5),
    ];
    let mut first = accumulator(32);
    let mut second = accumulator(32);
    for (position, albedo, area) in samples {
        first.add(position, albedo, area).expect("first pass");
        second.add(position, albedo, area).expect("second pass");
    }
    assert_eq!(first.finish(), second.finish());
}

#[test]
fn degenerate_bounds_and_unsupported_resolutions_are_refused() {
    let cases = [
        ([0.0, 0.0, f32::NAN], [1.0; 3], 16),
        ([0.0; 3], [0.0, 1.0, 1.0], 16),
        ([2.0; 3], [1.0; 3], 16),
        ([0.0; 3], [1.0; 3], 8),
    ];
    for (min, max, resolution) in cases {
        let err = FoliageVolumeAccumulator::new(min, max, resolution)
            .expect_err("invalid volume box is refused");
        assert_eq!(
            err.code(),
            invalid_code(),
            "{min:?} to {max:?} at {resolution}"
        );
    }
}

#[test]
fn non_finite_or_out_of_range_inputs_are_refused() {
    let mut volume = accumulator(16);
    let positions = [([f32::INFINITY, 0.0, 0.0], [0.5; 3], 1.0)];
    for (position, albedo, area) in positions {
        let err = volume
            .add(position, albedo, area)
            .expect_err("non-finite position is refused");
        assert_eq!(err.code(), invalid_code());
    }
    let albedos = [([0.5, f32::NAN, 0.5], 1.0), ([0.5, 1.5, 0.5], 1.0)];
    for (albedo, area) in albedos {
        let err = volume
            .add([0.5; 3], albedo, area)
            .expect_err("invalid albedo is refused");
        assert_eq!(err.code(), invalid_code());
    }
    let areas = [-0.5f32, f32::INFINITY];
    for area in areas {
        let err = volume
            .add([0.5; 3], [0.5; 3], area)
            .expect_err("invalid area is refused");
        assert_eq!(err.code(), invalid_code());
    }
}
