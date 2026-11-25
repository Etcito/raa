//! Subject tests of the differentiable foliage optimizer (M7, contracts `m7-compute` and
//! `m7-optimize`): the settings and reference-range refusals, the shared 64-byte params
//! layout, the coordinator's ordered transitions with a bin-overflow retry, cancellation,
//! the refusal of an incomplete submission and of a non-finite or non-opaque terminal
//! generation, and the assembly that keeps the source wood while publishing only opaque
//! terminal triangles with their independent normals and the native volume. The modules are
//! path-compiled until production registration; no test constructs GPU work or calls the
//! optimizer allocation.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::RaaError;
use raa_lit::LitHandle;

#[allow(dead_code)]
#[path = "../src/foliage_appearance.rs"]
mod foliage_appearance;
#[allow(dead_code)]
#[path = "../src/foliage_bins.rs"]
mod foliage_bins;
#[allow(dead_code)]
#[path = "../src/foliage_gradients.rs"]
mod foliage_gradients;
#[allow(dead_code)]
#[path = "../src/foliage_optimize.rs"]
mod foliage_optimize;
#[allow(dead_code)]
#[path = "../src/foliage_poisson.rs"]
mod foliage_poisson;
#[allow(dead_code)]
#[path = "../src/foliage_raster.rs"]
mod foliage_raster;
#[allow(dead_code)]
#[allow(clippy::wrong_self_convention)]
#[path = "../src/foliage_resources.rs"]
mod foliage_resources;
#[allow(dead_code)]
#[path = "../src/foliage_sampling.rs"]
mod foliage_sampling;
#[allow(dead_code)]
#[path = "../src/foliage_stochastic.rs"]
mod foliage_stochastic;
#[allow(dead_code)]
#[path = "../src/foliage_update.rs"]
mod foliage_update;
#[allow(dead_code)]
#[path = "../src/foliage_volume.rs"]
mod foliage_volume;

use foliage_optimize::{
    FOLIAGE_OPTIMIZE_PARAMS_SIZE, FoliageBinOutcome, FoliageOptimizeCoordinator,
    FoliageOptimizeParams, FoliageOptimizeSettings, FoliageOptimizeState, INITIAL_BIN_CAPACITY,
    assemble_optimized_foliage, initial_bin_capacity, validate_reference_count,
};
use foliage_resources::{FoliageSource, MicroTriangle, SourceTriangle};
use foliage_sampling::sample_foliage;

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

fn bake_failed_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.bake_failed",
        message: String::new(),
    }
    .code()
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn unit_triangle() -> MicroTriangle {
    MicroTriangle {
        p0: [0.0, 0.0, 0.0],
        p1: [1.0, 0.0, 0.0],
        p2: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        opacity: 1.0,
    }
}

fn source_triangle(positions: [[f32; 3]; 3], foliage: bool) -> SourceTriangle {
    SourceTriangle {
        positions,
        normals: [[0.0, 0.0, 1.0]; 3],
        uv: [[0.0; 2]; 3],
        material: 1,
        source_tag: 2,
        foliage,
    }
}

fn source() -> FoliageSource {
    FoliageSource {
        triangles: vec![
            source_triangle([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], true),
            source_triangle([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], false),
            source_triangle([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 1.0, 1.0]], true),
        ],
        min: [-1.0, -1.0, -1.0],
        max: [3.0, 2.0, 2.0],
        root_world: raa_math::Mat4::IDENTITY,
        generation: 5,
    }
}

fn drive_to_finalizing(iterations: u32) -> FoliageOptimizeCoordinator {
    let mut coordinator = FoliageOptimizeCoordinator::new(iterations, 4).unwrap();
    coordinator.begin().unwrap();
    for _ in 0..iterations {
        coordinator.bins_encoded().unwrap();
        assert_eq!(
            coordinator.bins_completed(&[0, 1, 2]).unwrap(),
            FoliageBinOutcome::Raster
        );
        coordinator.raster_encoded().unwrap();
        coordinator.gradients_encoded().unwrap();
        coordinator.update_encoded().unwrap();
    }
    assert_eq!(coordinator.state(), FoliageOptimizeState::Finalizing);
    coordinator
}

#[test]
fn settings_follow_the_material_defaults_and_refuse_impossible_values() {
    let settings = FoliageOptimizeSettings::DEFAULT;
    assert_eq!(
        (
            settings.count,
            settings.seed,
            settings.resolution,
            settings.iterations,
        ),
        (4096, 0, 32, 64)
    );
    assert_eq!(
        (
            settings.silhouette_weight,
            settings.normal_weight,
            settings.edge_weight,
        ),
        (1.0, 1.0, 0.01)
    );
    settings.validate().unwrap();

    let mut invalid = settings;
    invalid.count = 0;
    assert_eq!(invalid.validate().unwrap_err().code(), invalid_code());
    let mut invalid = settings;
    invalid.resolution = 8;
    assert_eq!(invalid.validate().unwrap_err().code(), invalid_code());
    let mut invalid = settings;
    invalid.iterations = 0;
    assert_eq!(invalid.validate().unwrap_err().code(), invalid_code());
    let mut invalid = settings;
    invalid.silhouette_weight = -1.0;
    assert_eq!(invalid.validate().unwrap_err().code(), invalid_code());
    let mut invalid = settings;
    invalid.normal_weight = f32::NAN;
    assert_eq!(invalid.validate().unwrap_err().code(), invalid_code());
    let mut zeroed = settings;
    zeroed.silhouette_weight = 0.0;
    zeroed.normal_weight = 0.0;
    zeroed.edge_weight = 0.0;
    assert_eq!(zeroed.validate().unwrap_err().code(), invalid_code());
    let mut edge_only = settings;
    edge_only.silhouette_weight = 0.0;
    edge_only.normal_weight = 0.0;
    edge_only.validate().unwrap();
}

#[test]
fn the_reference_range_and_initial_bin_capacity_are_explicit() {
    for refused in [0, 1, 8, 63, 129, 256] {
        assert_eq!(
            validate_reference_count(refused).unwrap_err().code(),
            invalid_code()
        );
    }
    for accepted in [64, 65, 96, 127, 128] {
        validate_reference_count(accepted).unwrap();
    }
    assert_eq!(INITIAL_BIN_CAPACITY, 32);
    assert_eq!(
        [0, 1, 16, 32, 4096].map(initial_bin_capacity),
        [1, 1, 16, 32, 32]
    );
}

#[test]
fn the_params_record_matches_the_shared_compute_layout() {
    let params = FoliageOptimizeParams {
        count: 7,
        views: 96,
        resolution: 128,
        iteration: 3,
        silhouette_weight: 1.5,
        normal_weight: 2.5,
        edge_weight: 0.25,
        temperature: 0.5,
        learning_rate: 0.125,
        opacity_target: 1.0,
        capacity: 64,
        generation: 9,
    };
    let bytes = params.to_bytes();
    assert_eq!(bytes.len(), FOLIAGE_OPTIMIZE_PARAMS_SIZE);
    assert_eq!(FOLIAGE_OPTIMIZE_PARAMS_SIZE, 64);
    assert_eq!(u32_at(&bytes, 0), 7);
    assert_eq!(u32_at(&bytes, 4), 96);
    assert_eq!(u32_at(&bytes, 8), 128);
    assert_eq!(u32_at(&bytes, 12), 3);
    assert_eq!(f32_at(&bytes, 16), 1.5);
    assert_eq!(f32_at(&bytes, 20), 2.5);
    assert_eq!(f32_at(&bytes, 24), 0.25);
    assert_eq!(f32_at(&bytes, 28), 0.5);
    assert_eq!(f32_at(&bytes, 32), 0.125);
    assert_eq!(f32_at(&bytes, 36), 1.0);
    assert_eq!(u32_at(&bytes, 40), 64);
    assert_eq!(u32_at(&bytes, 44), 9);
    for offset in [48, 52, 56, 60] {
        assert_eq!(u32_at(&bytes, offset), 0);
    }
}

#[test]
fn the_coordinator_runs_every_iteration_in_order_and_publishes_once() {
    let mut coordinator = FoliageOptimizeCoordinator::new(2, 4).unwrap();
    assert_eq!(coordinator.state(), FoliageOptimizeState::Prepared);
    assert_eq!(coordinator.iteration(), 0);
    assert_eq!(coordinator.capacity(), 4);
    coordinator.begin().unwrap();
    assert_eq!(coordinator.state(), FoliageOptimizeState::Bins);
    for iteration in 0..2 {
        assert_eq!(coordinator.iteration(), iteration);
        coordinator.bins_encoded().unwrap();
        assert_eq!(coordinator.state(), FoliageOptimizeState::AwaitingBins);
        assert_eq!(
            coordinator.bins_completed(&[0, 1, 2]).unwrap(),
            FoliageBinOutcome::Raster
        );
        assert_eq!(coordinator.state(), FoliageOptimizeState::Raster);
        coordinator.raster_encoded().unwrap();
        assert_eq!(coordinator.state(), FoliageOptimizeState::Gradients);
        coordinator.gradients_encoded().unwrap();
        assert_eq!(coordinator.state(), FoliageOptimizeState::Update);
        coordinator.update_encoded().unwrap();
    }
    assert_eq!(coordinator.iteration(), 1);
    assert_eq!(coordinator.state(), FoliageOptimizeState::Finalizing);
    coordinator.publish(&[unit_triangle()]).unwrap();
    assert_eq!(coordinator.state(), FoliageOptimizeState::Published);
    assert_eq!(
        coordinator.publish(&[unit_triangle()]).unwrap_err().code(),
        invalid_code()
    );
    assert_eq!(coordinator.cancel().unwrap_err().code(), invalid_code());
}

#[test]
fn an_overflowing_bin_grows_and_retries_before_the_raster_consumer() {
    let mut coordinator = FoliageOptimizeCoordinator::new(1, 4).unwrap();
    coordinator.begin().unwrap();
    coordinator.bins_encoded().unwrap();
    assert_eq!(
        coordinator.bins_completed(&[0, 5, 1, 0]).unwrap(),
        FoliageBinOutcome::Retry
    );
    assert_eq!(coordinator.capacity(), 8);
    assert_eq!(coordinator.state(), FoliageOptimizeState::Bins);
    assert_eq!(
        coordinator.raster_encoded().unwrap_err().code(),
        invalid_code(),
        "the raster consumer cannot read an overflowing bin"
    );
    coordinator.bins_encoded().unwrap();
    assert_eq!(
        coordinator.bins_completed(&[0, 5, 1, 0]).unwrap(),
        FoliageBinOutcome::Raster
    );
    assert_eq!(coordinator.state(), FoliageOptimizeState::Raster);
}

#[test]
fn an_incomplete_submission_never_reaches_a_consumer() {
    let mut coordinator = FoliageOptimizeCoordinator::new(1, 4).unwrap();
    assert_eq!(
        coordinator.raster_encoded().unwrap_err().code(),
        invalid_code()
    );
    assert_eq!(
        coordinator.publish(&[unit_triangle()]).unwrap_err().code(),
        invalid_code()
    );
    coordinator.begin().unwrap();
    assert_eq!(
        coordinator.gradients_encoded().unwrap_err().code(),
        invalid_code()
    );
    assert_eq!(
        coordinator.update_encoded().unwrap_err().code(),
        invalid_code()
    );
    coordinator.bins_encoded().unwrap();
    assert_eq!(
        coordinator.update_encoded().unwrap_err().code(),
        invalid_code(),
        "the update reads gradients the raster has not produced yet"
    );
}

#[test]
fn cancellation_stops_the_run_before_publication() {
    let mut coordinator = FoliageOptimizeCoordinator::new(1, 4).unwrap();
    coordinator.begin().unwrap();
    coordinator.cancel().unwrap();
    assert_eq!(coordinator.state(), FoliageOptimizeState::Cancelled);
    assert_eq!(
        coordinator.bins_encoded().unwrap_err().code(),
        invalid_code()
    );
    assert_eq!(
        coordinator.publish(&[unit_triangle()]).unwrap_err().code(),
        invalid_code()
    );
    coordinator.cancel().unwrap();
    assert_eq!(coordinator.state(), FoliageOptimizeState::Cancelled);
}

#[test]
fn terminal_publication_refuses_non_finite_and_non_opaque_triangles() {
    let mut coordinator = drive_to_finalizing(1);
    let mut translucent = unit_triangle();
    translucent.opacity = 0.5;
    assert_eq!(
        coordinator.publish(&[translucent]).unwrap_err().code(),
        bake_failed_code()
    );
    let mut nonfinite = unit_triangle();
    nonfinite.p0[0] = f32::NAN;
    assert_eq!(
        coordinator.publish(&[nonfinite]).unwrap_err().code(),
        bake_failed_code()
    );
    let mut degenerate = unit_triangle();
    degenerate.normal = [0.0; 3];
    assert_eq!(
        coordinator.publish(&[degenerate]).unwrap_err().code(),
        bake_failed_code()
    );
    assert_eq!(coordinator.state(), FoliageOptimizeState::Finalizing);
    coordinator.publish(&[unit_triangle()]).unwrap();
    assert_eq!(coordinator.state(), FoliageOptimizeState::Published);
}

#[test]
fn optimized_assembly_keeps_the_wood_and_publishes_opaque_proxy_normals() {
    let source = source();
    let samples = sample_foliage(&source, 2, 11).unwrap();
    assert_eq!(samples.len(), 2);
    let albedo = vec![[0.5f32, 0.25, 0.75]; samples.len()];
    // The stored normal is independent of the winding: the geometric normal is -Z.
    let terminal = [
        MicroTriangle {
            p0: [0.0, 0.0, 0.0],
            p1: [0.0, 1.0, 0.0],
            p2: [1.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            opacity: 1.0,
        },
        unit_triangle(),
    ];
    let bake = assemble_optimized_foliage(&source, &terminal, &samples, &albedo, 16).unwrap();
    assert_eq!(bake.wood_triangles, 1);
    assert_eq!(bake.foliage_triangles, 2);
    assert_eq!(bake.mesh.positions.len(), 9);
    assert_eq!(bake.mesh.indices.len(), 9);
    assert_eq!(
        bake.mesh.positions[..3],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        "wood keeps its source positions and order"
    );
    assert_eq!(
        bake.mesh.positions[6..9],
        [terminal[1].p0, terminal[1].p1, terminal[1].p2]
    );
    for normal in &bake.mesh.normals[6..9] {
        assert_eq!(*normal, terminal[1].normal);
    }
    assert!(
        bake.mesh.normals[3..6]
            .iter()
            .all(|normal| *normal == [0.0, 0.0, 1.0])
    );
    assert_eq!(bake.volume.len(), 4 * 16 * 16 * 16);
    assert!(
        bake.volume.iter().any(|byte| *byte != 0),
        "the sampled source volume is occupied"
    );
    assert_eq!(
        assemble_optimized_foliage(&source, &terminal, &samples, &albedo[..1], 16)
            .unwrap_err()
            .code(),
        invalid_code()
    );
}

#[test]
fn empty_eligible_foliage_keeps_the_wood_and_a_zero_volume() {
    let source = source();
    let bake = assemble_optimized_foliage(&source, &[], &[], &[], 32).unwrap();
    assert_eq!(bake.wood_triangles, 1);
    assert_eq!(bake.foliage_triangles, 0);
    assert_eq!(bake.mesh.positions.len(), 3);
    assert_eq!(bake.volume.len(), 4 * 32 * 32 * 32);
    assert!(bake.volume.iter().all(|byte| *byte == 0));
    let translucent = [MicroTriangle {
        opacity: 0.25,
        ..unit_triangle()
    }];
    assert_eq!(
        assemble_optimized_foliage(&source, &translucent, &[], &[], 16)
            .unwrap_err()
            .code(),
        bake_failed_code()
    );
    assert_eq!(
        assemble_optimized_foliage(&source, &[], &[], &[], 8)
            .unwrap_err()
            .code(),
        invalid_code()
    );
}
