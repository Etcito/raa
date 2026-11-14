//! The distance debug view (S38 `gi.history`, debug view 3) on its CPU twins: the jump flood of
//! the GPU schedule converges to the exact nearest-surface distance, and the band mapping
//! `0.5 + 0.5 cos(2 pi d / 0.4)` of the field sampled `DISTANCE_LIFT_METRES` along the surface
//! normal shows repeating bands parallel to the neighbouring surfaces. Every expected value is
//! derived by hand from a wall-and-floor corner, never from a GPU run.

use raa_gi::clipmap::flood_steps;
use raa_gi::cpu::{
    CpuGrid, CpuVolume, DISTANCE_LIFT_METRES, DISTANCE_PERIOD_METRES, distance_band,
    distance_probe, jump_flood, seed_distance,
};
use raa_gi::shaders::debug_view_module_wgsl;

const R: i32 = 64;
const VOXEL: f32 = 0.2;
/// The wall and the floor sit in cell 4, inside the `(1.5, R - 1.5)` band of `level_at`.
const W: i32 = 4;
/// Centre of the wall and floor cells in metres.
const FACE: f32 = (W as f32 + 0.5) * VOXEL;

fn index_of(cell: [i32; 3]) -> usize {
    (cell[0] + R * (cell[1] + R * cell[2])) as usize
}

/// A 64^3 level of 0.2 m voxels at the origin whose occupied cells are the wall `x = 4` and the
/// floor `y = 4`. Cell (x, y, z) is `0.2 * min(|x - 4|, |y - 4|)` metres from its nearest seed.
fn corner() -> (Vec<Option<[i32; 3]>>, CpuVolume) {
    let mut occupied = vec![false; (R * R * R) as usize];
    for z in 0..R {
        for y in 0..R {
            for x in 0..R {
                occupied[index_of([x, y, z])] = x == W || y == W;
            }
        }
    }
    let grid = CpuGrid {
        resolution: R as u32,
        occupied,
    };
    let seeds = jump_flood(&grid, &flood_steps(R as u32));
    let mut texels = vec![[0.0; 4]; seeds.len()];
    for z in 0..R {
        for y in 0..R {
            for x in 0..R {
                let i = index_of([x, y, z]);
                texels[i][3] = seed_distance([x, y, z], seeds[i], VOXEL);
            }
        }
    }
    let volume = CpuVolume {
        levels: vec![[0.0, 0.0, 0.0, VOXEL]],
        resolution: R as u32,
        texels,
    };
    (seeds, volume)
}

fn band_at(volume: &CpuVolume, probe: [f32; 3]) -> f32 {
    let level = volume.level_at(probe, 0);
    assert_eq!(level, 0, "{probe:?} lies inside level 0");
    distance_band(volume.sample(probe, level)[3])
}

/// Grey levels along the floor (centre of the floor cells) at `i + 0.5` voxels from the wall, sampled
/// `lift` metres straight up, in the middle of the grid along z.
fn floor_line(volume: &CpuVolume, lift: f32) -> Vec<f32> {
    (0..10)
        .map(|i| {
            let surface = [FACE + i as f32 * VOXEL, FACE, 6.5];
            band_at(volume, [surface[0], surface[1] + lift, surface[2]])
        })
        .collect()
}

fn assert_bands(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1.0e-3, "{what}: cell {i}: {g}, want {w}");
    }
}

#[test]
fn band_mapping_has_a_0_4_m_period() {
    assert_eq!(DISTANCE_PERIOD_METRES, 0.4);
    // cos(0) = 1, cos(pi / 2) = 0, cos(pi) = -1, cos(2 pi) = 1.
    for (d, want) in [
        (0.0, 1.0),
        (0.1, 0.5),
        (0.2, 0.0),
        (0.3, 0.5),
        (0.4, 1.0),
        (0.6, 0.0),
    ] {
        assert!((distance_band(d) - want).abs() <= 1.0e-5, "d {d}");
    }
}

#[test]
fn flood_of_the_gpu_schedule_reaches_every_cell_of_the_corner() {
    let (seeds, _) = corner();
    for z in (0..R).step_by(7) {
        for y in 0..R {
            for x in 0..R {
                let cell = [x, y, z];
                let got = seed_distance(cell, seeds[index_of(cell)], VOXEL);
                let want = VOXEL * (x - W).abs().min((y - W).abs()) as f32;
                assert!(
                    (got - want).abs() <= 1.0e-4,
                    "{cell:?}: {got} m, want {want} m"
                );
            }
        }
    }
}

#[test]
fn lifted_probe_shows_repeating_bands_parallel_to_the_wall() {
    let (_, volume) = corner();
    assert_eq!(DISTANCE_LIFT_METRES, 1.4);
    // The probe sits 1.4 m = 7 rows above the floor cells' centre, so the field there is
    // 0.2 min(i, 7) at i cells from the wall:
    // light at even i, dark at odd i (0.2 m light, 0.2 m dark, period 0.4 m) up to 1.4 m from
    // the wall, then the 1.4 m cap (cos 7 pi = -1, dark).
    let lifted: Vec<f32> = (0..10)
        .map(|i| {
            let surface = [FACE + i as f32 * VOXEL, FACE, 6.5];
            band_at(&volume, distance_probe(surface, [0.0, 1.0, 0.0]))
        })
        .collect();
    let want = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    assert_bands(&lifted, &want, "floor, lifted probe");
    assert_bands(
        &floor_line(&volume, DISTANCE_LIFT_METRES),
        &want,
        "floor line",
    );
    let dark = lifted.iter().filter(|&&g| g < 0.5).count();
    assert!(dark >= 3, "repeating dark bands: {dark}");

    // On the wall the probe 1.4 m out reads the floor distance: bands parallel to it.
    let wall: Vec<f32> = (0..10)
        .map(|j| {
            let surface = [FACE, FACE + j as f32 * VOXEL, 6.5];
            band_at(&volume, distance_probe(surface, [1.0, 0.0, 0.0]))
        })
        .collect();
    assert_bands(&wall, &want, "wall, lifted probe");
}

#[test]
fn probe_two_voxels_in_front_draws_at_most_one_dark_line() {
    // The failing mapping: two finest voxels toward a camera looking straight down puts the
    // probe 2 rows above the floor cells, whose field is 0.2 min(i, 2): one dark cell at i = 1, then
    // light everywhere (0.4 m is a whole period). This is the judged image without bands.
    let (_, volume) = corner();
    let near = floor_line(&volume, 2.0 * VOXEL);
    let want = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    assert_bands(&near, &want, "floor, probe 0.4 m in front");
}

#[test]
fn shader_samples_the_lifted_probe_with_the_twin_constants() {
    let wgsl = debug_view_module_wgsl();
    assert!(wgsl.contains(&format!(
        "const RAA_GI_DEBUG_LIFT: f32 = {DISTANCE_LIFT_METRES:?};"
    )));
    assert!(wgsl.contains(&format!(
        "const RAA_GI_DEBUG_PERIOD: f32 = {DISTANCE_PERIOD_METRES:?};"
    )));
    assert!(wgsl.contains("let probe = centre.xyz + normal * RAA_GI_DEBUG_LIFT;"));
    assert!(wgsl.contains("d = raa_gi_sample(probe, level).a;"));
    assert!(wgsl.contains("0.5 + 0.5 * cos(2.0 * RAA_GI_DEBUG_PI * d / RAA_GI_DEBUG_PERIOD)"));
    assert!(!wgsl.contains("2.0 * raa_gi_finest_voxel()"));
}
