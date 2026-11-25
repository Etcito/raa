use raa_math::{Mat4, Vec3};

use super::*;
use crate::mesh::MeshInput;

fn grid_config() -> SphericalGridConfig {
    SphericalGridConfig {
        angular_resolution: 1,
        radial_shells: 3,
        clip_start: 1.0,
        z_max: 8.0,
        radial_growth: 2.0,
    }
}

fn config() -> SphericalSchedulerConfig {
    SphericalSchedulerConfig {
        grid: grid_config(),
        near_shells: 1,
        max_cells: 128,
        center_probe: false,
        origin_tolerance: 0.25,
        ..SphericalSchedulerConfig::default()
    }
}

fn triangle(key: u64, positions: &[[f32; 3]; 3]) -> MeshInput<'_> {
    MeshInput {
        key,
        mesh: key as u32,
        positions,
        normals: &[[1.0, 0.0, 0.0]; 3],
        colors: None,
        uvs: None,
        indices: &[0, 1, 2],
        material: 0,
        world: Mat4::IDENTITY,
    }
}

fn facing_x(x: f32) -> [[f32; 3]; 3] {
    [[x, -0.2, -0.1], [x, 0.2, -0.1], [x, 0.0, 0.2]]
}

fn update_geometry(scene: &mut MeshTraceScene, inputs: &[MeshInput<'_>]) {
    scene.geometry.update(inputs, 128 << 20, 0).unwrap();
}

fn key(shell: u16, face: CubeFace) -> CellKey {
    CellKey {
        shell,
        face,
        x: 0,
        y: 0,
    }
}

#[test]
fn radial_widths_are_geometric_with_z_max_as_the_outer_boundary() {
    let grid = SphericalGrid::new(grid_config()).unwrap();
    for (i, (inner, center, outer)) in [(1., 1.5, 2.), (2., 3., 4.), (4., 6., 8.)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            grid.radial_shell(i as u16),
            Some(RadialShell {
                inner,
                center,
                outer
            })
        );
    }
    assert_eq!(grid.key_at([2.0, 0.0, 0.0]), Some(key(1, CubeFace::PosX)));
    assert_eq!(grid.key_at([8.0, 0.0, 0.0]), Some(key(2, CubeFace::PosX)));
    assert_eq!(grid.key_at([8.001, 0.0, 0.0]), None);
    assert_eq!(grid.key_at([0.999, 0.0, 0.0]), None);
    let last = grid.radial_shell(2).unwrap();
    assert_eq!(last.center, 8.0 - (last.outer - last.inner) * 0.5);

    let uniform = SphericalGrid::new(SphericalGridConfig {
        radial_growth: 1.0,
        ..grid_config()
    })
    .unwrap();
    assert!((uniform.radial_shell(0).unwrap().outer - (1.0 + 7.0 / 3.0)).abs() < 1e-6);
    assert_eq!(uniform.radial_shell(2).unwrap().outer, 8.0);
    assert!(
        SphericalGrid::new(SphericalGridConfig {
            z_max: 1.0,
            ..grid_config()
        })
        .is_err()
    );
}

#[test]
fn cubemap_center_and_lookup_round_trip_with_world_anchor() {
    let mut grid = SphericalGrid::new(SphericalGridConfig {
        angular_resolution: 5,
        ..grid_config()
    })
    .unwrap();
    grid.origin = [12.0, -2.0, 1.0];
    for i in 0..u32::from(grid.shell_count()) * grid.cells_per_shell() {
        let key = grid.key(i);
        assert_eq!(grid.ordinal(key), i);
        assert_eq!(grid.key_at(grid.center(key).unwrap()), Some(key));
    }
    let seam = grid.key_at([13.0, -1.0, 2.0]).unwrap();
    assert_eq!(seam.face, CubeFace::PosX);
}

#[test]
fn bvh_candidates_need_exact_angular_and_radial_overlap() {
    let grid_settings = SphericalGridConfig {
        radial_shells: 1,
        z_max: 3.0,
        ..grid_config()
    };
    let grid = SphericalGrid::new(grid_settings).unwrap();
    let cell = grid.cell(key(0, CubeFace::PosX));
    let misses_angle = [[1.7, 1.9, 0.0], [1.7, 2.0, 0.1], [1.7, 2.1, -0.1]];
    let misses_radius = [[2.1, 1.8, 1.8], [2.1, 1.9, 1.8], [2.1, 1.8, 1.9]];
    let crossing = [[2.0, -5.0, -5.0], [2.0, 5.0, -5.0], [2.0, 0.0, 5.0]];
    let mut scene = MeshTraceScene::default();
    update_geometry(
        &mut scene,
        &[triangle(1, &misses_angle), triangle(2, &misses_radius)],
    );
    assert!(
        scene
            .geometry
            .find_triangle(cell.min, cell.max, Some)
            .is_some()
    );
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        grid: grid_settings,
        ..config()
    })
    .unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    assert!(scheduler.progress(key(0, CubeFace::PosX)).is_none());

    update_geometry(
        &mut scene,
        &[
            triangle(1, &misses_angle),
            triangle(2, &misses_radius),
            triangle(3, &crossing),
        ],
    );
    scheduler.update(&scene, [0.0; 3], 0);
    let occupied = scheduler.progress(key(0, CubeFace::PosX)).unwrap();
    assert_eq!(occupied.surface_position, [2.0, 0.0, 0.0]);
    assert_eq!(occupied.normal, [1.0, 0.0, 0.0]);
}

#[test]
fn near_and_far_budgets_track_each_face_and_retry_only_the_aborted_face() {
    let mut scene = MeshTraceScene::default();
    update_geometry(
        &mut scene,
        &[triangle(1, &facing_x(1.5)), triangle(2, &facing_x(-5.0))],
    );
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    scheduler.update(&scene, [0.0; 3], 1);
    assert_eq!(scheduler.stats().cells, 2);
    assert_eq!(scheduler.stats().pending_faces, 12);
    let jobs = scheduler.take_jobs(FaceBudget { near: 2, far: 1 });
    assert_eq!(
        jobs.iter()
            .map(|job| (job.key.shell, job.face))
            .collect::<Vec<_>>(),
        [
            (0, CubeFace::PosX),
            (0, CubeFace::NegX),
            (2, CubeFace::PosX)
        ]
    );
    assert_eq!(scheduler.stats().in_flight_faces, 3);
    assert!(scheduler.complete(jobs[0]));
    assert!(scheduler.cancel(jobs[1]));
    let retry = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert_eq!(retry.face, jobs[1].face);
    assert!(!scheduler.complete(jobs[1]));
    assert!(scheduler.complete(retry));
    assert!(scheduler.complete(jobs[2]));
    assert!(!scheduler.complete(jobs[2]));
    assert_eq!(scheduler.stats().ready_faces, 3);
    assert_eq!(scheduler.stats().ready_cells, 0);

    for job in scheduler.take_jobs(FaceBudget { near: 20, far: 20 }) {
        assert!(scheduler.complete(job));
    }
    assert_eq!(scheduler.stats().ready_cells, 2);
    assert_eq!(scheduler.stats().pending_faces, 0);
}

#[test]
fn light_change_resets_faces_and_recenter_reissues_in_flight_faces_in_place() {
    let mut scene = MeshTraceScene::default();
    update_geometry(&mut scene, &[triangle(1, &facing_x(1.7))]);
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    scheduler.update(&scene, [0.0; 3], 9);
    let mut jobs = scheduler.take_jobs(FaceBudget { near: 2, far: 0 });
    assert!(scheduler.complete(jobs[0]));
    let stale_light_job = jobs.pop().unwrap();
    let slot = stale_light_job.slot;
    scheduler.update(&scene, [0.0; 3], 10);
    assert_eq!(scheduler.stats().light_revision, 10);
    assert_eq!(scheduler.stats().ready_faces, 0);
    assert!(!scheduler.complete(stale_light_job));
    let first = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert!(scheduler.complete(first));
    let old_origin_job = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert_eq!(old_origin_job.slot, slot);
    // The world-space probe keeps its cell, slot, epoch and completed face; only the
    // in-flight face is reissued because its job named the retired anchor.
    scheduler.update(&scene, [0.3, 0.0, 0.0], 10);
    assert_eq!(scheduler.grid().origin(), [0.25, 0.0, 0.0]);
    assert_eq!(scheduler.stats().ready_faces, 1);
    assert!(!scheduler.take_released_slots().contains(&slot));
    let new_job = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert_eq!(
        (
            new_job.key,
            new_job.slot,
            new_job.epoch,
            new_job.face,
            new_job.probe
        ),
        (
            old_origin_job.key,
            old_origin_job.slot,
            old_origin_job.epoch,
            old_origin_job.face,
            old_origin_job.probe
        )
    );
    assert_ne!(new_job.center, old_origin_job.center);
    assert!(!scheduler.complete(old_origin_job));
    assert!(scheduler.complete(new_job));
}

#[test]
fn sustained_walk_keeps_ready_world_space_probes_across_origin_epochs() {
    let mut scene = MeshTraceScene::default();
    // Walking away from the wall keeps its probes outside the inner sphere.
    update_geometry(&mut scene, &[triangle(1, &facing_x(-1.9))]);
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    let mut epochs = BTreeSet::new();
    let mut first = None;
    let mut first_survived_epochs = BTreeSet::new();
    for step in 0..48 {
        let camera = [step as f32 * 0.02, 0.0, 0.0];
        scheduler.update(&scene, camera, 0);
        let anchor = scheduler.grid().origin();
        assert!((camera[0] - anchor[0]).abs() <= config().origin_tolerance);
        let epoch = scheduler.stats().origin_epoch;
        epochs.insert(epoch);
        for job in scheduler.take_jobs(FaceBudget { near: 1, far: 1 }) {
            assert!(scheduler.complete(job));
        }
        if first.is_some() {
            // Recentering never leaves the walk without a published probe.
            assert!(
                scheduler
                    .cells()
                    .any(|cell| cell.is_ready() && cell.published)
            );
        } else if let Some(cell) = scheduler.cells().find(|cell| cell.is_ready()) {
            first = Some(cell.probe);
        }
        if let Some(probe) = first
            && scheduler
                .cells()
                .any(|cell| cell.probe == probe && cell.published)
        {
            first_survived_epochs.insert(epoch);
        }
    }
    assert!(epochs.len() >= 3);
    // The first published probe was reused, not recaptured, after a recenter.
    assert!(first_survived_epochs.len() >= 2);
}

#[test]
fn geometry_refit_dirties_old_and_new_regions_and_preserves_an_unrelated_cell() {
    let mut scene = MeshTraceScene::default();
    let positions = facing_x(1.5);
    let static_positions = [[-0.1, 1.5, -0.1], [0.1, 1.5, -0.1], [0.0, 1.5, 0.1]];
    let extra = facing_x(20.0);
    let mut inputs = [
        triangle(1, &positions),
        triangle(2, &static_positions),
        triangle(3, &extra),
        triangle(4, &extra),
        triangle(5, &extra),
    ];
    update_geometry(&mut scene, &inputs);
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    let old = key(0, CubeFace::PosX);
    let static_key = key(0, CubeFace::PosY);
    let stale = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert_eq!(stale.key, old);
    let unchanged = *scheduler.progress(static_key).unwrap();

    inputs[0].world = Mat4::from_translation(Vec3::new(-3.0, 0.0, 0.0));
    update_geometry(&mut scene, &inputs);
    scheduler.update(&scene, [0.0; 3], 0);
    assert!(scheduler.progress(old).is_none());
    assert!(scheduler.progress(key(0, CubeFace::NegX)).is_some());
    assert_eq!(scheduler.progress(static_key), Some(&unchanged));
    assert!(!scheduler.complete(stale));
    assert_eq!(scheduler.stats().cells, 2);

    update_geometry(&mut scene, &inputs[1..]);
    scheduler.update(&scene, [0.0; 3], 0);
    assert!(scheduler.progress(key(0, CubeFace::NegX)).is_none());
    assert_eq!(scheduler.progress(static_key), Some(&unchanged));
}

#[test]
fn capacity_and_lookup_work_are_bounded_and_evicted_far_cells_can_return() {
    let mut scene = MeshTraceScene::default();
    let near = facing_x(1.5);
    let far = facing_x(-5.0);
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        max_cells: 1,
        cell_queries_per_update: 1,
        ..config()
    })
    .unwrap();
    update_geometry(&mut scene, &[triangle(2, &far)]);
    let check = |scheduler: &mut SphericalScheduler, scene: &MeshTraceScene| {
        for _ in 0..80 {
            scheduler.update(scene, [0.0; 3], 0);
            assert!(scheduler.stats().last_cell_queries <= 1);
            assert!(scheduler.stats().cells <= 1);
        }
    };
    let far_key = key(2, CubeFace::NegX);
    check(&mut scheduler, &scene);
    let initial = *scheduler.progress(far_key).unwrap();
    update_geometry(&mut scene, &[triangle(1, &near), triangle(2, &far)]);
    check(&mut scheduler, &scene);
    assert!(scheduler.progress(far_key).is_none());
    assert_eq!(
        scheduler.progress(key(0, CubeFace::PosX)).unwrap().slot,
        initial.slot
    );
    update_geometry(&mut scene, &[triangle(2, &far)]);
    check(&mut scheduler, &scene);
    let restored = scheduler.progress(far_key).unwrap();
    assert_eq!(restored.slot, initial.slot);
    assert_ne!(restored.epoch, initial.epoch);
    update_geometry(&mut scene, &[]);
    scheduler.update(&scene, [0.0; 3], 0);
    assert_eq!(scheduler.stats().cells, 0);
    assert_eq!(scheduler.stats().last_cell_queries, 0);
}

#[test]
fn recenter_rechecks_far_cells_without_waiting_for_a_full_scan() {
    let mut scene = MeshTraceScene::default();
    let far = [[-0.1, -0.1, -6.0], [0.1, -0.1, -6.0], [0.0, 0.1, -6.0]];
    update_geometry(&mut scene, &[triangle(1, &far)]);
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        cell_queries_per_update: 1,
        ..config()
    })
    .unwrap();
    for _ in 0..18 {
        scheduler.update(&scene, [0.0; 3], 0);
    }
    let key = key(2, CubeFace::NegZ);
    let old = scheduler.take_jobs(FaceBudget { near: 0, far: 1 })[0];
    assert_eq!(old.key, key);

    // Reaching the final grid cell by discovery alone takes 18 frames.
    // A walk at this speed recenters every 13 frames.
    let mut ready_epochs = BTreeSet::new();
    for step in 1..50 {
        scheduler.update(&scene, [step as f32 * 0.02, 0.0, 0.0], 0);
        for job in scheduler.take_jobs(FaceBudget { near: 0, far: 1 }) {
            assert!(scheduler.complete(job));
        }
        if scheduler.stats().origin_epoch > 1 {
            assert!(!scheduler.complete(old));
            assert!(scheduler.progress(key).is_some());
            if scheduler.stats().ready_cells == 1 {
                ready_epochs.insert(scheduler.stats().origin_epoch);
            }
        }
    }
    assert!(ready_epochs.len() >= 3);
}

#[test]
fn materials_and_skipped_geometry_revisions_invalidate_current_faces() {
    let mut scene = MeshTraceScene::default();
    let points = facing_x(1.5);
    let mut input = triangle(1, &points);
    update_geometry(&mut scene, std::slice::from_ref(&input));
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    let budget = FaceBudget { near: 1, far: 1 };
    let old_material = scheduler.take_jobs(budget)[0];
    scene.material_revision += 1;
    scheduler.update(&scene, [0.0; 3], 0);
    assert!(!scheduler.complete(old_material));
    let old_geometry = scheduler.take_jobs(budget)[0];
    assert_eq!(old_material.slot, old_geometry.slot);

    input.world = Mat4::from_translation(Vec3::new(-3.0, 0.0, 0.0));
    update_geometry(&mut scene, std::slice::from_ref(&input));
    input.world = Mat4::from_translation(Vec3::new(-4.0, 0.0, 0.0));
    update_geometry(&mut scene, &[input]);
    scheduler.update(&scene, [0.0; 3], 0);
    assert!(!scheduler.complete(old_geometry));
    assert!(scheduler.progress(key(0, CubeFace::PosX)).is_none());
    assert!(scheduler.progress(key(1, CubeFace::NegX)).is_some());
}

fn light(id: u64, position: [f32; 3], range: f32, intensity: f32) -> SphericalLight {
    let lanes = [
        position[0],
        position[1],
        position[2],
        range,
        intensity,
        0.,
        0.,
        0.,
        0.,
        0.,
        -1.,
        0.9,
        0.95,
        0.,
        1.,
        1.,
    ];
    let record: Vec<_> = lanes.into_iter().flat_map(f32::to_le_bytes).collect();
    SphericalLight::from_packed(id, &record).unwrap()
}

fn capture_light_metadata(scheduler: &mut SphericalScheduler, ids: impl Fn(CellKey) -> Vec<u64>) {
    for job in scheduler.take_jobs(FaceBudget { near: 32, far: 32 }) {
        let hit = if job.key.shell == 0 {
            [3.0, 0.0, 0.0]
        } else {
            [-6.0, 0.0, 0.0]
        };
        assert!(scheduler.complete(job));
        assert!(scheduler.record_face_lighting(job, Some((hit, hit)), &ids(job.key)));
    }
}

#[test]
fn local_light_mutations_preserve_unaffected_ready_cells_and_hashes() {
    let mut scene = MeshTraceScene::default();
    update_geometry(
        &mut scene,
        &[triangle(1, &facing_x(1.5)), triangle(2, &facing_x(-5.0))],
    );
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    let near = key(0, CubeFace::PosX);
    let far = key(2, CubeFace::NegX);
    let contributor = light(10, [3.1, 0.0, 0.0], 0.5, 1.0);
    let removed_irrelevant = light(20, [3.1, 0.0, 0.0], 0.5, 1.0);
    let far_contributor = light(30, [-6.1, 0.0, 0.0], 0.5, 1.0);
    scheduler.update_with_lights(
        &scene,
        [0.0; 3],
        5,
        &[contributor, removed_irrelevant, far_contributor],
    );
    capture_light_metadata(&mut scheduler, |key| {
        if key == near {
            vec![10, SUN_LIGHT_ID]
        } else {
            vec![30, SKY_LIGHT_ID]
        }
    });
    let initial_near = *scheduler.progress(near).unwrap();
    let initial_far = *scheduler.progress(far).unwrap();
    assert_eq!(
        scheduler.contributing_light_ids(near),
        Some([10, SUN_LIGHT_ID].as_slice())
    );
    assert_eq!(scheduler.stats().lighting_metadata_ready_cells, 2);
    let remote = light(40, [60.0, 0.0, 0.0], 0.5, 1.0);
    // The remaining records can be reordered or shifted by removal of a dense GPU index.
    assert!(!scheduler.update_with_lights(
        &scene,
        [0.0; 3],
        5,
        &[remote, far_contributor, contributor]
    ));
    assert_eq!(scheduler.progress(near), Some(&initial_near));
    assert_eq!(scheduler.progress(far), Some(&initial_far));
    assert_eq!(scheduler.stats().light_invalidated_cells, 0);

    let edited = light(10, [3.1, 0.0, 0.0], 0.5, 2.0);
    assert!(scheduler.update_with_lights(&scene, [0.0; 3], 5, &[edited, far_contributor, remote]));
    assert!(!scheduler.progress(near).unwrap().is_ready());
    assert_eq!(scheduler.progress(far), Some(&initial_far));
    assert_eq!(scheduler.stats().light_invalidated_cells, 1);
    assert_eq!(scheduler.stats().light_invalidated_faces, 6);
    capture_light_metadata(&mut scheduler, |_| vec![10, SUN_LIGHT_ID]);
    assert_ne!(
        scheduler.progress(near).unwrap().light_hash,
        initial_near.light_hash
    );

    assert!(scheduler.update_with_lights(&scene, [0.0; 3], 5, &[far_contributor, remote]));
    assert!(!scheduler.progress(near).unwrap().is_ready());
    assert_eq!(scheduler.progress(far), Some(&initial_far));
}

#[test]
fn new_or_moved_lights_cover_previously_occluded_hits() {
    let mut scene = MeshTraceScene::default();
    update_geometry(
        &mut scene,
        &[triangle(1, &facing_x(1.5)), triangle(2, &facing_x(-5.0))],
    );
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    let near = key(0, CubeFace::PosX);
    let far = key(2, CubeFace::NegX);
    scheduler.update(&scene, [0.0; 3], 0);
    capture_light_metadata(&mut scheduler, |_| vec![]);
    let initial_far = *scheduler.progress(far).unwrap();

    let occluded = light(70, [3.1, 0.0, 0.0], 0.4, 1.0);
    assert!(scheduler.update_with_lights(&scene, [0.0; 3], 0, &[occluded]));
    assert!(!scheduler.progress(near).unwrap().is_ready());
    assert_eq!(scheduler.progress(far), Some(&initial_far));
    capture_light_metadata(&mut scheduler, |_| vec![]);
    let initial_near = *scheduler.progress(near).unwrap();

    let brighter = light(70, [3.1, 0.0, 0.0], 0.4, 5.0);
    assert!(!scheduler.update_with_lights(&scene, [0.0; 3], 0, &[brighter]));
    assert_eq!(scheduler.progress(near), Some(&initial_near));
    let moved = light(70, [-6.1, 0.0, 0.0], 0.4, 5.0);
    assert!(scheduler.update_with_lights(&scene, [0.0; 3], 0, &[moved]));
    assert_eq!(scheduler.progress(near), Some(&initial_near));
    assert!(!scheduler.progress(far).unwrap().is_ready());
}

#[test]
fn pending_light_metadata_is_conservative_and_stale_readback_is_rejected() {
    let mut scene = MeshTraceScene::default();
    update_geometry(&mut scene, &[triangle(1, &facing_x(1.5))]);
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    let old = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert!(scheduler.complete(old));
    let remote = light(10, [60.0, 0.0, 0.0], 0.5, 1.0);
    assert!(!scheduler.update_with_lights(&scene, [0.0; 3], 0, &[remote]));
    let overlapping = light(20, [1.5, 0.0, 0.0], 0.5, 1.0);
    assert!(scheduler.update_with_lights(&scene, [0.0; 3], 0, &[remote, overlapping]));
    assert!(!scheduler.record_face_lighting(old, Some(([1.5, 0.0, 0.0], [1.5, 0.0, 0.0])), &[20]));
    assert_eq!(scheduler.stats().light_invalidated_faces, 1);
    let retried = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert_eq!(retried.slot, old.slot);
    assert_ne!(retried.epoch, old.epoch);
    assert!(scheduler.complete(retried));
    assert!(scheduler.record_face_lighting(retried, None, &[SKY_LIGHT_ID]));
    assert!(!scheduler.record_face_lighting(retried, None, &[20]));
}

#[test]
fn rotating_view_keeps_near_cells_behind_the_camera() {
    let mut scene = MeshTraceScene::default();
    let visible = [[-0.1, -0.1, -5.0], [0.1, -0.1, -5.0], [0.0, 0.1, -5.0]];
    update_geometry(
        &mut scene,
        &[
            triangle(1, &facing_x(1.5)),
            triangle(2, &facing_x(-1.5)),
            triangle(3, &visible),
        ],
    );
    let make = |max_cells| {
        SphericalScheduler::new(SphericalSchedulerConfig {
            max_cells,
            cell_queries_per_update: 3,
            ..config()
        })
        .unwrap()
    };
    let mut scheduler = make(2);
    for _ in 0..12 {
        scheduler.update(&scene, [0.0; 3], 0);
    }
    capture_light_metadata(&mut scheduler, |_| vec![]);
    let retained = key(0, CubeFace::PosX);
    let behind = key(0, CubeFace::NegX);
    let visible = key(2, CubeFace::NegZ);
    let original = *scheduler.progress(retained).unwrap();
    let behind_cell = *scheduler.progress(behind).unwrap();
    let anchor_epoch = scheduler.stats().origin_epoch;
    assert!(scheduler.progress(visible).is_none());

    // Shell-major priority: a far visible cell never displaces a near one.
    scheduler.set_view_priority([0.0, 0.0, -1.0], 0.87);
    for _ in 0..12 {
        scheduler.update(&scene, [0.0; 3], 0);
    }
    assert_eq!(scheduler.stats().origin_epoch, anchor_epoch);
    assert_eq!(scheduler.progress(retained), Some(&original));
    assert_eq!(scheduler.progress(behind), Some(&behind_cell));
    assert!(scheduler.progress(visible).is_none());
    assert!(scheduler.take_released_slots().is_empty());

    let mut roomy = make(3);
    roomy.set_view_priority([0.0, 0.0, -1.0], 0.87);
    for _ in 0..12 {
        roomy.update(&scene, [0.0; 3], 0);
    }
    assert!(roomy.progress(behind).is_some());
    let far_face = roomy
        .take_jobs(FaceBudget { near: 0, far: 1 })
        .pop()
        .unwrap();
    assert_eq!(far_face.key, visible);
    roomy.set_view_priority([0.01, 0.0, -1.0], 0.87);
    roomy.update(&scene, [0.0; 3], 0);
    assert!(roomy.is_current(far_face));
}

#[test]
fn shared_capture_pool_serves_visible_probes_first_then_periphery_then_behind() {
    let mut scene = MeshTraceScene::default();
    // Near triangles behind (+Z), to the side (+X) and a far one in view (-Z).
    let visible = [[-0.1, -0.1, -5.0], [0.1, -0.1, -5.0], [0.0, 0.1, -5.0]];
    let behind = [[-0.1, -0.1, 1.5], [0.1, -0.1, 1.5], [0.0, 0.1, 1.5]];
    update_geometry(
        &mut scene,
        &[
            triangle(1, &facing_x(1.5)),
            triangle(2, &behind),
            triangle(3, &visible),
        ],
    );
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    scheduler.set_view_priority([0.0, 0.0, -1.0], 0.87);
    for _ in 0..4 {
        scheduler.update(&scene, [0.0; 3], 0);
    }
    let stats = scheduler.stats();
    assert_eq!(stats.cells, 3);
    assert_eq!((stats.visible_cells, stats.visible_published_cells), (1, 0));
    assert!(stats.visible_missing());

    // One shared pool, regardless of bands: the far visible cell first, then the
    // side cell (periphery), then the near cell behind the camera.
    let jobs = scheduler.take_jobs_shared(13);
    assert_eq!(jobs.len(), 13);
    let order: Vec<_> = jobs.iter().map(|job| job.key).collect();
    assert_eq!(order[0], key(2, CubeFace::NegZ));
    assert_eq!(order[6], key(0, CubeFace::PosX));
    assert_eq!(order[12], key(0, CubeFace::PosZ));
    for job in jobs {
        assert!(scheduler.complete(job));
    }
    let stats = scheduler.stats();
    assert_eq!((stats.visible_cells, stats.visible_published_cells), (1, 1));

    // A relight turns every published probe into a refresh: the never-published
    // remainder of the behind cell still comes first.
    scheduler.update(&scene, [0.0; 3], 1);
    let jobs = scheduler.take_jobs_shared(6);
    assert!(jobs.iter().all(|job| job.key == key(0, CubeFace::PosZ)));
}

#[test]
fn probe_cache_keeps_probes_that_leave_the_grid_and_a_walk_back_adopts_them() {
    let mut scene = MeshTraceScene::default();
    update_geometry(&mut scene, &[triangle(1, &facing_x(-1.5))]);
    let walk = |cache_cells| {
        let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
            cache_cells,
            ..config()
        })
        .unwrap();
        for _ in 0..4 {
            scheduler.update(&scene, [0.0; 3], 0);
            for job in scheduler.take_jobs_shared(64) {
                assert!(scheduler.complete(job));
            }
        }
        let probe = *scheduler.cells().next().expect("captured probe");
        assert!(probe.published && probe.is_ready());
        scheduler.take_released_slots();
        // Walk 12 m away: the wall leaves the 8 m grid.
        scheduler.update(&scene, [12.0, 0.0, 0.0], 0);
        assert!(scheduler.cells().all(|cell| cell.probe != probe.probe));
        let away = scheduler.stats();
        // Walk back: the cached probe is adopted with its slot and faces.
        scheduler.update(&scene, [0.0; 3], 0);
        let back = scheduler
            .cells()
            .find(|cell| cell.probe == probe.probe)
            .copied();
        (
            away,
            back,
            scheduler.take_adopted_slots(),
            scheduler.take_released_slots(),
            probe,
        )
    };
    let (away, back, adopted, released, probe) = walk(8);
    assert_eq!(away.cached_cells, 1);
    let back = back.expect("adopted probe");
    assert_eq!(
        (back.slot, back.ready_faces, back.published),
        (probe.slot, ALL_FACES, true)
    );
    assert!(adopted.contains(&probe.slot));
    assert!(!released.contains(&probe.slot));

    // Without a cache the probe is released and comes back unpublished.
    let (away, back, adopted, released, probe) = walk(0);
    assert_eq!(away.cached_cells, 0);
    // Rediscovery places a fresh probe at the same point; it must be captured again.
    assert!(back.is_none_or(|cell| !cell.published && cell.ready_faces == 0));
    assert!(adopted.is_empty());
    assert!(released.contains(&probe.slot));
}

#[test]
fn a_light_change_clears_the_probe_cache() {
    let mut scene = MeshTraceScene::default();
    update_geometry(&mut scene, &[triangle(1, &facing_x(-1.5))]);
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        cache_cells: 8,
        ..config()
    })
    .unwrap();
    for _ in 0..4 {
        scheduler.update(&scene, [0.0; 3], 0);
        for job in scheduler.take_jobs_shared(64) {
            assert!(scheduler.complete(job));
        }
    }
    scheduler.update(&scene, [12.0, 0.0, 0.0], 0);
    assert_eq!(scheduler.stats().cached_cells, 1);
    scheduler.update(&scene, [12.0, 0.0, 0.0], 1);
    assert_eq!(scheduler.stats().cached_cells, 0);
}

#[test]
fn auto_range_covers_the_scene_bounds_with_hysteresis() {
    let mut scene = MeshTraceScene::default();
    // Geometry from x = -1.5 to x = 30: the fixed 8 m grid would cut it.
    update_geometry(
        &mut scene,
        &[triangle(1, &facing_x(-1.5)), triangle(2, &facing_x(30.0))],
    );
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        auto_range: true,
        ..config()
    })
    .unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    let outer = scheduler.grid().config().z_max;
    assert!(
        outer >= 30.0 && outer < 30.0 * 2f32.powf(0.25) * 1.01,
        "{outer}"
    );
    let shell = scheduler.grid().radial_shell(0).unwrap();
    assert_eq!(shell.inner, 1.0);
    // Both walls get a probe: the far one is inside the last shell.
    for _ in 0..8 {
        scheduler.update(&scene, [0.0; 3], 0);
    }
    assert!(scheduler.cells().any(|cell| cell.probe[0] > 20.0));
    // Walking 2 m toward the far wall keeps the radius (no pumping).
    let epoch = scheduler.stats().origin_epoch;
    scheduler.update(&scene, [2.0, 0.0, 0.0], 0);
    assert_eq!(scheduler.grid().config().z_max, outer);
    assert_eq!(scheduler.stats().origin_epoch, epoch + 1);
    // Walking away grows it at once.
    scheduler.update(&scene, [-6.0, 0.0, 0.0], 0);
    assert!(scheduler.grid().config().z_max >= 36.0);
}

fn center_config() -> SphericalSchedulerConfig {
    SphericalSchedulerConfig {
        center_probe: true,
        ..config()
    }
}

#[test]
fn centre_probe_covers_the_inner_sphere_and_follows_the_camera_beyond_clip_start() {
    let mut scene = MeshTraceScene::default();
    update_geometry(&mut scene, &[triangle(1, &facing_x(1.5))]);
    let mut scheduler = SphericalScheduler::new(center_config()).unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    let center = *scheduler.progress(CellKey::CENTER).unwrap();
    assert_eq!(center.probe, [0.0; 3]);
    assert_eq!(scheduler.stats().near_cells, 2);
    // The centre probe draws first from the near budget.
    let jobs = scheduler.take_jobs(FaceBudget { near: 6, far: 0 });
    assert!(jobs.iter().all(|job| job.key.is_center()));
    for job in jobs {
        assert!(scheduler.complete(job));
    }
    assert!(scheduler.progress(CellKey::CENTER).unwrap().published);
    // Inside clipStart (1 m) the same probe stays the centre without recapture.
    scheduler.update(&scene, [0.5, 0.0, 0.0], 0);
    let kept = *scheduler.progress(CellKey::CENTER).unwrap();
    assert_eq!((kept.slot, kept.probe), (center.slot, center.probe));
    assert!(kept.is_ready());
    // Beyond clipStart it becomes the probe of its own cell and a new centre starts.
    scheduler.update(&scene, [-1.25, 0.0, 0.0], 0);
    let moved = scheduler.progress(key(0, CubeFace::PosX)).unwrap();
    assert_eq!((moved.slot, moved.probe), (center.slot, center.probe));
    assert!(moved.published);
    let fresh = scheduler.progress(CellKey::CENTER).unwrap();
    assert_eq!(fresh.probe, [-1.25, 0.0, 0.0]);
    assert!(!fresh.published);
}

#[test]
fn probes_are_pushed_off_triangles_through_the_cell_centre() {
    let mut scene = MeshTraceScene::default();
    // A wall through the PosX shell-0 centre (radius 1.5).
    let wall = [[1.5, -4.0, -4.0], [1.5, 4.0, -4.0], [1.5, 0.0, 6.0]];
    update_geometry(&mut scene, &[triangle(1, &wall)]);
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    let cell = scheduler.progress(key(0, CubeFace::PosX)).unwrap();
    assert_eq!(cell.center, [1.5, 0.0, 0.0]);
    assert!((cell.probe[0] - 1.5).abs() > 0.05, "probe {:?}", cell.probe);
    assert_eq!(scheduler.grid().key_at(cell.probe), Some(cell.key));
}

#[test]
fn rekey_keeps_the_probe_nearest_each_new_cell_centre() {
    let mut scene = MeshTraceScene::default();
    // Off-centre triangles leave both cell centres clear, so probes sit on them.
    let off = |x: f32| [[x, 0.6, -0.1], [x, 1.0, -0.1], [x, 0.8, 0.2]];
    let (a, b) = (off(1.5), off(3.0));
    update_geometry(&mut scene, &[triangle(1, &a), triangle(2, &b)]);
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        origin_tolerance: 0.5,
        ..config()
    })
    .unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    let near = *scheduler.progress(key(0, CubeFace::PosX)).unwrap();
    let far = *scheduler.progress(key(1, CubeFace::PosX)).unwrap();
    assert_eq!((near.probe, far.probe), ([1.5, 0.0, 0.0], [3.0, 0.0, 0.0]));
    // The anchor snaps to -0.5: both probes now fall in PosX shell 1 (radius 2 to 4,
    // centre 3). The probe at radius 3.5 is nearer that centre than the one at 2.0.
    scheduler.update(&scene, [-0.6, 0.0, 0.0], 0);
    assert_eq!(scheduler.grid().origin(), [-0.5, 0.0, 0.0]);
    let kept = *scheduler.progress(key(1, CubeFace::PosX)).unwrap();
    assert_eq!((kept.slot, kept.probe), (far.slot, far.probe));
    assert!(scheduler.take_released_slots().contains(&near.slot));
}

#[test]
fn snapped_origin_lies_on_the_tolerance_lattice() {
    assert_eq!(snap_origin([0.3, -0.13, 0.0], 0.25), [0.25, -0.25, 0.0]);
    assert_eq!(snap_origin([0.3, -0.13, 0.0], 0.0), [0.3, -0.13, 0.0]);
}

#[test]
fn recentred_origin_snaps_to_the_smallest_cell_world_lattice() {
    let scene = MeshTraceScene::default();
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        origin_tolerance: 0.0,
        ..config()
    })
    .unwrap();
    let step = scheduler.grid().smallest_cell();
    let shell = scheduler.grid().radial_shell(0).unwrap();
    let angular = 2.0 * config().grid.clip_start / f32::from(config().grid.angular_resolution);
    assert_eq!(step, (shell.outer - shell.inner).min(angular));
    for camera in [[0.37, -1.21, 2.05], [5.5, 0.01, -3.33], [-7.9, 2.2, 0.6]] {
        scheduler.update(&scene, camera, 0);
        let origin = scheduler.grid().origin();
        for axis in 0..3 {
            let cells = f64::from(origin[axis]) / f64::from(step);
            assert!(
                (cells - cells.round()).abs() < 1e-4,
                "{origin:?} off the {step} lattice"
            );
            assert!((origin[axis] - camera[axis]).abs() <= step * 0.5 + 1e-5);
        }
    }
}

#[test]
fn view_capture_rank_puts_the_view_cone_first_and_is_a_permutation() {
    for (forward, cos) in [
        ([1.0, 0.0, 0.0], 0.99),
        ([1.0, 1.0, 0.0], 0.75),
        ([-0.1, 1.0, 0.2], 0.0),
    ] {
        let grid = SphericalGrid::new(SphericalGridConfig {
            angular_resolution: 5,
            ..grid_config()
        })
        .unwrap();
        let order = ViewOrder::new(&grid, None, forward, cos).unwrap();
        let mut ranks: Vec<_> = (0..grid.cells_per_shell())
            .map(|index| order.capture_rank(&grid, grid.key(index)))
            .collect();
        let classes: Vec<_> = ranks.iter().map(|rank| rank.0).collect();
        ranks.sort_unstable_by_key(|rank| rank.2);
        assert!(ranks.windows(2).all(|pair| pair[0].2 + 1 == pair[1].2));
        // Angular rank orders the whole view cone first.
        assert!(
            ranks
                .windows(2)
                .all(|pair| pair[0].0 == 0 || pair[1].0 != 0)
        );
        assert!(classes.contains(&0));
    }
}

#[test]
fn rotating_in_place_never_moves_evicts_or_recaptures_probes() {
    let mut scene = MeshTraceScene::default();
    update_geometry(
        &mut scene,
        &[
            triangle(1, &facing_x(1.5)),
            triangle(2, &facing_x(-1.5)),
            triangle(
                3,
                &[[-0.1, -0.1, -5.0], [0.1, -0.1, -5.0], [0.0, 0.1, -5.0]],
            ),
            triangle(4, &[[-0.1, -0.1, 5.0], [0.1, -0.1, 5.0], [0.0, 0.1, 5.0]]),
        ],
    );
    // Capacity below the occupied cells: residency must still not follow the view.
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        max_cells: 3,
        center_probe: true,
        ..config()
    })
    .unwrap();
    let camera = [0.05, 0.0, 0.0];
    for _ in 0..8 {
        scheduler.update(&scene, camera, 0);
        for job in scheduler.take_jobs_shared(64) {
            assert!(scheduler.complete(job));
        }
    }
    let snapshot = |scheduler: &SphericalScheduler| {
        let mut cells: Vec<_> = scheduler.cells().copied().collect();
        cells.sort_by_key(|cell| cell.key);
        cells
    };
    let before = snapshot(&scheduler);
    assert_eq!(before.len(), 3);
    assert!(before.iter().all(|cell| cell.is_ready() && cell.published));
    scheduler.take_released_slots();
    let epoch = scheduler.stats().origin_epoch;
    let mut recaptures = 0;
    for step in 0..=72 {
        let angle = step as f32 * std::f32::consts::TAU / 72.0;
        scheduler.set_view_priority([angle.sin(), 0.0, -angle.cos()], 0.8);
        scheduler.update(&scene, camera, 0);
        recaptures += scheduler.take_jobs_shared(64).len();
    }
    assert_eq!(snapshot(&scheduler), before);
    assert_eq!(recaptures, 0);
    assert!(scheduler.take_released_slots().is_empty());
    assert_eq!(scheduler.stats().origin_epoch, epoch);
}

fn reflected_scene() -> MeshTraceScene {
    let mut scene = MeshTraceScene::default();
    let front = [[3.0, -6.0, -6.0], [3.0, 6.0, -6.0], [3.0, 0.0, 6.0]];
    let back = [[-3.0, -6.0, -6.0], [-3.0, 6.0, -6.0], [-3.0, 0.0, 6.0]];
    update_geometry(&mut scene, &[triangle(1, &front), triangle(2, &back)]);
    scene
}

fn reflected_scheduler() -> SphericalScheduler {
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        grid: SphericalGridConfig {
            angular_resolution: 8,
            ..grid_config()
        },
        max_cells: 8,
        ..config()
    })
    .unwrap();
    scheduler.set_view_priority([1., 0., 0.], 0.8);
    scheduler
}

const REFLECTED_RECEIVER: ReflectedReceiver = ReflectedReceiver {
    position: [-3.0, 0.1, 0.02],
    normal: [1.0, 0.0, 0.0],
    priority: 1,
};

fn advance_reflected(
    scheduler: &mut SphericalScheduler,
    scene: &MeshTraceScene,
    origin: [f32; 3],
    frames: usize,
) {
    for _ in 0..frames {
        scheduler.update(scene, origin, 1);
        assert!(scheduler.stats().cells <= 8);
        assert!(scheduler.stats().last_cell_queries <= 768);
        let jobs = scheduler.take_jobs(FaceBudget { near: 1, far: 4 });
        assert!(jobs.iter().filter(|job| job.key.shell == 0).count() <= 1);
        assert!(jobs.iter().filter(|job| job.key.shell != 0).count() <= 4);
        for job in jobs {
            assert!(scheduler.complete(job));
        }
    }
}

#[test]
fn reflected_priority_reuses_the_pool_and_restores_primary_coverage_when_cleared() {
    let scene = reflected_scene();
    let origin = [0.0; 3];
    let mut scheduler = reflected_scheduler();
    advance_reflected(&mut scheduler, &scene, origin, 20);
    let primary: BTreeMap<_, _> = scheduler.cells().map(|p| (p.key, p.epoch)).collect();
    let receiver_key = scheduler
        .grid()
        .key_at(REFLECTED_RECEIVER.position)
        .unwrap();
    assert!(scheduler.progress(receiver_key).is_none());
    assert_eq!(scheduler.stats().ready_cells, 8);
    let epoch = scheduler.stats().origin_epoch;
    assert!(scheduler.set_reflected_receivers(
        epoch,
        scene.geometry.revision,
        &[REFLECTED_RECEIVER],
    ));
    advance_reflected(&mut scheduler, &scene, origin, 20);
    assert_eq!(scheduler.stats().ready_cells, 8);
    assert_eq!(scheduler.stats().reflected_ready_cells, 2);
    assert!(scheduler.progress(receiver_key).unwrap().is_ready());
    assert_eq!(
        scheduler
            .cells()
            .filter(|p| primary.get(&p.key) == Some(&p.epoch))
            .count(),
        6
    );
    scheduler.clear_reflected_receivers();
    advance_reflected(&mut scheduler, &scene, origin, 20);
    assert_eq!(scheduler.stats().reflected_cells, 0);
    assert_eq!(scheduler.stats().ready_cells, 8);
    assert_eq!(
        scheduler.cells().map(|p| p.key).collect::<Vec<_>>(),
        primary.into_keys().collect::<Vec<_>>()
    );
}

#[test]
fn reflected_priority_keeps_progress_while_walking_and_rejects_stale_feedback() {
    let mut scene = reflected_scene();
    let mut scheduler = reflected_scheduler();
    scheduler.update(&scene, [0.0; 3], 1);
    let mut last_epoch = scheduler.stats().origin_epoch;
    let mut stale_job = None;
    let mut groups_with_progress = BTreeSet::new();
    for frame in 0..80 {
        let origin = [0.0, frame as f32 * 0.02, 0.0];
        scheduler.update(&scene, origin, 1);
        let epoch = scheduler.stats().origin_epoch;
        if epoch != last_epoch {
            assert!(!scheduler.set_reflected_receivers(
                last_epoch,
                scene.geometry.revision,
                &[REFLECTED_RECEIVER]
            ));
            assert!(!scheduler.complete(stale_job.take().unwrap()));
        }
        assert!(scheduler.set_reflected_receivers(
            epoch,
            scene.geometry.revision,
            &[REFLECTED_RECEIVER]
        ));
        advance_reflected(&mut scheduler, &scene, origin, 1);
        if scheduler.stats().reflected_ready_cells > 0 {
            groups_with_progress.insert(epoch);
        }
        if stale_job.is_none() {
            stale_job = scheduler.take_jobs(FaceBudget { near: 0, far: 1 }).pop();
        }
        last_epoch = epoch;
    }
    assert!(groups_with_progress.len() >= 5, "{groups_with_progress:?}");
    let geometry_revision = scene.geometry.revision;
    update_geometry(&mut scene, &[triangle(1, &facing_x(3.0))]);
    scheduler.update(&scene, [0.0, 1.58, 0.0], 1);
    assert!(!scheduler.set_reflected_receivers(
        last_epoch,
        geometry_revision,
        &[REFLECTED_RECEIVER]
    ));
    assert_eq!(scheduler.stats().reflected_cells, 0);
}

#[test]
fn reflected_wrong_side_and_empty_space_do_not_displace_primary() {
    let scene = reflected_scene();
    let mut scheduler = reflected_scheduler();
    let origin = [0.0; 3];
    advance_reflected(&mut scheduler, &scene, origin, 20);
    let before: Vec<_> = scheduler.cells().copied().collect();
    assert!(scheduler.set_reflected_receivers(
        scheduler.stats().origin_epoch,
        scene.geometry.revision,
        &[
            ReflectedReceiver {
                normal: [-1.0, 0.0, 0.0],
                ..REFLECTED_RECEIVER
            },
            ReflectedReceiver {
                position: [0.0, 0.0, -7.0],
                ..REFLECTED_RECEIVER
            },
        ],
    ));
    advance_reflected(&mut scheduler, &scene, origin, 4);
    assert_eq!(scheduler.stats().reflected_cells, 0);
    assert_eq!(scheduler.cells().copied().collect::<Vec<_>>(), before);
}

#[test]
fn reordered_identical_reflected_feedback_does_not_requery_the_bvh() {
    let scene = reflected_scene();
    let mut scheduler = reflected_scheduler();
    let origin = [0.0; 3];
    advance_reflected(&mut scheduler, &scene, origin, 20);
    let wrong_side = ReflectedReceiver {
        normal: [-1.0, 0.0, 0.0],
        ..REFLECTED_RECEIVER
    };
    let nearby_wrong_side = ReflectedReceiver {
        position: [-3.0, 0.12, 0.03],
        ..wrong_side
    };
    let epoch = scheduler.stats().origin_epoch;
    assert!(scheduler.set_reflected_receivers(
        epoch,
        scene.geometry.revision,
        &[wrong_side, nearby_wrong_side]
    ));
    scheduler.update(&scene, origin, 1);
    assert!(scheduler.stats().last_cell_queries > 0);
    assert!(scheduler.set_reflected_receivers(
        epoch,
        scene.geometry.revision,
        &[nearby_wrong_side, wrong_side]
    ));
    scheduler.update(&scene, origin, 1);
    assert_eq!(scheduler.stats().last_cell_queries, 0);
}

/// CPU cost of discovery and probe placement on a real scene. Run with
/// `SPHERICAL_BENCH_TRIANGLES=<N x 3 x 3 float32 file> cargo test --release -- --ignored bench_`.
#[test]
#[ignore]
fn bench_scheduler_update_on_triangle_file() {
    let Ok(path) = std::env::var("SPHERICAL_BENCH_TRIANGLES") else {
        return;
    };
    let bytes = std::fs::read(path).unwrap();
    let positions: Vec<[f32; 3]> = bytes
        .chunks_exact(12)
        .map(|c| {
            std::array::from_fn(|i| f32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap()))
        })
        .collect();
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    let indices: Vec<u32> = (0..positions.len() as u32).collect();
    let mut scene = MeshTraceScene::default();
    let start = std::time::Instant::now();
    scene
        .geometry
        .update(
            &[MeshInput {
                key: 1,
                mesh: 1,
                positions: &positions,
                normals: &normals,
                colors: None,
                uvs: None,
                indices: &indices,
                material: 0,
                world: Mat4::IDENTITY,
            }],
            8 << 30,
            0,
        )
        .unwrap();
    eprintln!(
        "bvh {} triangles {:?}",
        positions.len() / 3,
        start.elapsed()
    );
    let config = SphericalSchedulerConfig {
        grid: SphericalGridConfig {
            angular_resolution: 8,
            radial_growth: 0.3f32.exp2(),
            clip_start: 1.0,
            z_max: 12.0,
            ..SphericalGridConfig::default()
        },
        max_cells: std::env::var("SPHERICAL_BENCH_PROBES").map_or(2048, |v| v.parse().unwrap()),
        cache_cells: 1024,
        auto_range: true,
        ..SphericalSchedulerConfig::default()
    };
    let views = [
        ([-8.807f32, 1.592, -0.858], [-3.004f32, 2.762, 0.237]),
        ([2.703, 1.355, 2.111], [-9.668, 2.493, -2.255]),
    ];
    let mut scheduler = SphericalScheduler::new(config).unwrap();
    for (label, (position, target)) in [("V1", views[0]), ("V2", views[1]), ("V1 again", views[0])]
    {
        let forward = std::array::from_fn(|i| target[i] - position[i]);
        scheduler.set_view_priority(forward, 0.66);
        let mut total = std::time::Duration::ZERO;
        for frame in 0..400 {
            let start = std::time::Instant::now();
            scheduler.update(&scene, position, 0);
            let jobs = scheduler.take_jobs_shared(144);
            let spent = start.elapsed();
            total += spent;
            for job in jobs {
                scheduler.complete(job);
            }
            let stats = scheduler.stats();
            if frame < 8 || !stats.visible_missing() {
                eprintln!(
                    "{label} frame {frame}: {:?} cells {} visible {}/{} queries {} discovery {}",
                    spent,
                    stats.cells,
                    stats.visible_published_cells,
                    stats.visible_cells,
                    stats.last_cell_queries,
                    stats.discovery_pending
                );
            }
            if !stats.visible_missing() {
                eprintln!(
                    "{label}: visible valid after {} updates, CPU {:?}, outer radius {}, cells {}, capacity limited {}",
                    frame + 1,
                    total,
                    scheduler.grid().config().z_max,
                    stats.cells,
                    stats.capacity_limited
                );
                break;
            }
        }
    }
}
