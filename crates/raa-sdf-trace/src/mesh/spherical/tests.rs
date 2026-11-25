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
fn light_and_recenter_reject_old_faces_even_for_the_same_slot_and_key() {
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
    let old_origin_job = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert_eq!(old_origin_job.slot, slot);
    scheduler.update(&scene, [0.3, 0.0, 0.0], 10);
    let new_job = scheduler.take_jobs(FaceBudget { near: 1, far: 0 })[0];
    assert_eq!(
        (new_job.key, new_job.slot),
        (old_origin_job.key, old_origin_job.slot)
    );
    assert_ne!(new_job.center, old_origin_job.center);
    assert_ne!(new_job.epoch, old_origin_job.epoch);
    assert!(scheduler.take_released_slots().contains(&slot));
    assert!(!scheduler.complete(old_origin_job));
    assert!(scheduler.complete(new_job));
}

#[test]
fn sustained_walk_makes_face_progress_in_each_bounded_origin_epoch() {
    let mut scene = MeshTraceScene::default();
    update_geometry(&mut scene, &[triangle(1, &facing_x(1.9))]);
    let mut scheduler = SphericalScheduler::new(config()).unwrap();
    let mut ready_epochs = BTreeSet::new();
    let mut previous = None;
    for step in 0..48 {
        let camera = [step as f32 * 0.02, 0.0, 0.0];
        scheduler.update(&scene, camera, 0);
        let anchor = scheduler.grid().origin();
        assert!((camera[0] - anchor[0]).abs() <= config().origin_tolerance);
        let epoch = scheduler.stats().origin_epoch;
        if let Some((previous_epoch, previous_job)) = previous
            && epoch != previous_epoch
        {
            assert!(ready_epochs.contains(&previous_epoch));
            assert!(!scheduler.complete(previous_job));
            assert_eq!(scheduler.stats().ready_faces, 0);
        }
        for job in scheduler.take_jobs(FaceBudget { near: 1, far: 1 }) {
            assert!(scheduler.complete(job));
            previous = Some((epoch, job));
        }
        if scheduler.stats().ready_cells == 1 {
            ready_epochs.insert(epoch);
        }
    }
    assert!(ready_epochs.len() >= 3);
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
fn rotating_view_admits_visible_geometry_without_resetting_retained_faces() {
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
    let mut scheduler = SphericalScheduler::new(SphericalSchedulerConfig {
        max_cells: 2,
        cell_queries_per_update: 3,
        ..config()
    })
    .unwrap();
    scheduler.update(&scene, [0.0; 3], 0);
    capture_light_metadata(&mut scheduler, |_| vec![]);
    let retained = key(0, CubeFace::PosX);
    let offscreen = key(0, CubeFace::NegX);
    let visible = key(2, CubeFace::NegZ);
    let original = *scheduler.progress(retained).unwrap();
    let anchor_epoch = scheduler.stats().origin_epoch;
    let released = scheduler.progress(offscreen).unwrap().slot;
    assert!(scheduler.progress(visible).is_none());

    scheduler.set_view_priority([0.0, 0.0, -1.0], 0.87);
    scheduler.update(&scene, [0.0; 3], 0);
    assert_eq!(scheduler.stats().last_cell_queries, 3);
    assert_eq!(scheduler.stats().origin_epoch, anchor_epoch);
    assert_eq!(scheduler.progress(retained), Some(&original));
    assert!(scheduler.progress(offscreen).is_none());
    assert!(scheduler.take_released_slots().contains(&released));
    let new_face = scheduler.take_jobs(FaceBudget { near: 0, far: 1 })[0];
    assert_eq!(new_face.key, visible);

    scheduler.set_view_priority([0.01, 0.0, -1.0], 0.87);
    scheduler.update(&scene, [0.0; 3], 0);
    assert_eq!(scheduler.progress(retained), Some(&original));
    assert!(scheduler.is_current(new_face));
}

#[test]
fn view_order_is_bijective_and_shell_ranges_exclude_other_shells() {
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
        let total = grid.cells_per_shell() * u32::from(grid.shell_count());
        for rank in 0..total {
            assert_eq!(order.rank(&grid, order.key(&grid, rank)), rank);
        }
        let ranks: Vec<_> = order.shell_ranges(1..2).into_iter().flatten().collect();
        assert_eq!(ranks.len() as u32, grid.cells_per_shell());
        assert!(
            ranks
                .into_iter()
                .all(|rank| order.key(&grid, rank).shell == 1)
        );
    }
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
