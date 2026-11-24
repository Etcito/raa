// Adapted from solstice crates/solstice-render/src/gi/volume.rs at 7e465b5: the clipmap maths of
// the GI volume (constants, level origins, rebuild rounds, relight order, level view, bytes).
//
// MIT License
//
// Copyright (c) 2026 SamG-Coder
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! The voxel clipmap of the GI volume (S37), pure CPU and free of GPU types: the snapped level
//! origins, the choice of the level to rebuild, the rebuild rounds and the relight order, the
//! orthographic view of one level, the volume's byte size and the forward LOD rule.

use std::cmp::Reverse;
use std::collections::VecDeque;

use raa_lit::LodLevel;
use raa_math::Vec3;
use raa_scene::Camera;
use raa_vg::{InstanceFilter, PassView};

use crate::settings::{FINEST_VOXEL_METRES, GiQuality};

/// Clipmap origins snap to multiples of this many voxels, so a level moves in steps
/// (solstice `volume.rs:19`).
pub const SNAP_VOXELS: i32 = 8;

/// Jump-flood passes a pending rebuild advances per frame, of the 18 (solstice `volume.rs:27`).
pub const FLOOD_PASSES_PER_FRAME: usize = 6;

/// Blend weight of the newly relit radiance (solstice `shaders/gi/voxel_light.wgsl:29`, `BLEND`).
pub const LIGHT_BLEND: f32 = 0.22;

/// Distance written for a voxel without a seed, in voxels (solstice
/// `shaders/gi/volume_write.wgsl:22`, `FAR_DISTANCE_VOXELS`).
pub const FAR_DISTANCE_VOXELS: f32 = 64.0;

/// Frames after the last rebuild round before a GI capture counts as settled. No solstice line:
/// ARCHITECTURE S40 `GI_SETTLE_FRAMES`.
pub const SETTLE_FRAMES: u32 = 240;

/// Voxel size in metres of `level`: `0.2 * 2^level` (solstice `volume.rs:17, 234`).
pub fn voxel_size(level: u32) -> f32 {
    FINEST_VOXEL_METRES * (1u32 << level.min(31)) as f32
}

/// `snapped_origin` returns the grid corner of a level centred near `position`, in whole voxels
/// (solstice `volume.rs:738-746`): `floor(position / voxel_size)`, each axis `div_euclid` by the
/// snap step and multiplied back, then minus half of `resolution`.
///
/// Handoff: `raa-gi: clipmap tests and fixture` calls the entry points of this file:
/// `snapped_origin`, `choose_rebuild`, `RebuildRounds`, `RelightCursor::next`, `light_schedule`, `flood_steps`,
/// `flood_pass_count`, `volume_bytes`, `forward_lod`, `level_view` and `voxel_size`. Today they
/// are pure functions of their arguments; the only state crossing is a `RebuildRounds` or
/// `RelightCursor` that the caller owns.
pub fn snapped_origin(position: Vec3, voxel_size: f32, resolution: u32) -> [i32; 3] {
    let half = resolution as i32 / 2;
    (position / voxel_size)
        .to_array()
        .map(|c| c.floor() as i32)
        .map(|c| c.div_euclid(SNAP_VOXELS) * SNAP_VOXELS - half)
}

/// Jump-flood step lengths for a grid of `resolution` voxels; each list spans half the grid.
/// Solstice `volume.rs:22` gives the 96 list; the 64 list is ARCHITECTURE S37.
pub fn flood_steps(resolution: u32) -> [i32; 6] {
    if resolution == 96 {
        [48, 24, 12, 6, 3, 1]
    } else {
        [32, 16, 8, 4, 2, 1]
    }
}

/// Flood passes per rebuild: one per axis per step, an even count ending in buffer A (solstice
/// `volume.rs:714-717`, `flood_pass_count`).
pub fn flood_pass_count() -> usize {
    flood_steps(96).len() * 3
}

/// Relight order over `levels` levels (solstice `volume.rs:24` for five). Five or more levels
/// take the five-level order; three or fewer take the three-level order.
pub fn light_schedule(levels: u32) -> &'static [u32] {
    match levels {
        5.. => &[0, 1, 0, 2, 0, 3, 0, 4],
        4 => &[0, 1, 0, 2, 0, 3],
        _ => &[0, 1, 0, 2],
    }
}

/// Chebyshev distance between two origins, in the voxels of one level.
fn chebyshev(a: [i32; 3], b: [i32; 3]) -> i64 {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| (i64::from(x) - i64::from(y)).abs())
        .max()
        .unwrap_or(0)
}

/// The level to rebuild next and its wanted origin: the level whose origin is furthest behind
/// `camera`, measured in its own voxels. Ties go to the finest level, and a maximum of 0 gives
/// `None`. A `None` origin counts as offset 0 (solstice `volume.rs:721-735`, `choose_rebuild`).
pub fn choose_rebuild(
    origins: &[Option<[i32; 3]>],
    camera: Vec3,
    resolution: u32,
) -> Option<(usize, [i32; 3])> {
    let candidates = origins.iter().enumerate().map(|(level, current)| {
        let wanted = snapped_origin(camera, voxel_size(level as u32), resolution);
        let offset = current.map_or(0, |current| chebyshev(wanted, current));
        (level, wanted, offset)
    });
    let (level, wanted, offset) =
        candidates.max_by_key(|&(level, _, offset)| (offset, Reverse(level)))?;
    (offset > 0).then_some((level, wanted))
}

/// Pending scene refreshes, ordered by when each level was first dirtied. A level
/// takes the current scene when voxelization starts, so edits before that point coalesce.
/// Edits during its flood passes queue it again, behind levels that are still waiting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RebuildRounds {
    levels: usize,
    outdated: VecDeque<usize>,
}

impl RebuildRounds {
    /// Starts with every level current.
    pub fn new(levels: u32) -> Self {
        Self {
            levels: levels as usize,
            outdated: VecDeque::with_capacity(levels as usize),
        }
    }

    /// Queues every level that has not already been queued, coarsest first.
    pub fn mark_stale(&mut self) {
        for level in (0..self.levels).rev() {
            if !self.outdated.contains(&level) {
                self.outdated.push_back(level);
            }
        }
    }

    /// The next level to refresh when the camera needs no recentering.
    pub fn choose_outdated(&self) -> Option<usize> {
        self.outdated.front().copied()
    }

    /// Acknowledges the inputs captured at voxelization, including for a camera rebuild.
    /// Returns whether the level was already queued for a scene change.
    pub fn begin(&mut self, level: usize) -> bool {
        self.outdated
            .iter()
            .position(|&queued| queued == level)
            .and_then(|index| self.outdated.remove(index))
            .is_some()
    }

    /// Whether a scene refresh is queued. A started slice is tracked separately.
    pub fn pending(&self) -> bool {
        !self.outdated.is_empty()
    }

    /// Counts unique levels with an unfinished slice, newer inputs or a different wanted origin.
    pub(crate) fn pending_levels(
        &self,
        origins: &[Option<[i32; 3]>],
        camera: Vec3,
        resolution: u32,
        rebuilding: Option<usize>,
    ) -> u32 {
        origins
            .iter()
            .enumerate()
            .filter(|&(level, origin)| {
                Some(level) == rebuilding
                    || self.outdated.contains(&level)
                    || origin.is_some_and(|origin| {
                        origin != snapped_origin(camera, voxel_size(level as u32), resolution)
                    })
            })
            .count() as u32
    }
}

/// The relight order over the levels (solstice `volume.rs:458-466`): the next scheduled level,
/// skipping the one being rebuilt.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RelightCursor {
    schedule_index: usize,
}

impl RelightCursor {
    /// Picks the level to relight this frame and advances the schedule. A level being rebuilt
    /// is never relit: the next schedule entry is taken instead, and when that one is also being
    /// rebuilt the level is `(entry + 1) % levels`.
    pub fn next(&mut self, levels: u32, being_rebuilt: Option<usize>) -> usize {
        let schedule = light_schedule(levels);
        let mut relight = self.take(schedule);
        if being_rebuilt == Some(relight) {
            relight = self.take(schedule);
            if being_rebuilt == Some(relight) {
                relight = (relight + 1) % (levels.max(1) as usize);
            }
        }
        relight
    }

    /// Reads the schedule entry at the cursor and advances the cursor past it.
    fn take(&mut self, schedule: &[u32]) -> usize {
        self.schedule_index %= schedule.len();
        let entry = schedule.get(self.schedule_index).copied().unwrap_or(0);
        self.schedule_index = (self.schedule_index + 1) % schedule.len();
        entry as usize
    }
}

/// GPU bytes of the volume for resolution `R` and `L` levels: `32 R^3 L + 8 R^3` (ARCHITECTURE
/// S37 Memory; solstice `volume.rs:161-190`, `GI_VOLUME_BYTES`).
pub fn volume_bytes(quality: GiQuality) -> u64 {
    let voxels = u64::from(quality.resolution).pow(3);
    32 * voxels * u64::from(quality.levels) + 8 * voxels
}

/// The orthographic view that selects one level's clusters for driver B, drawing the `Gi`
/// instances with software rasterisation for every cluster (solstice `volume.rs:749-771`,
/// `write_level_view`).
pub fn level_view(origin: [i32; 3], voxel_size: f32, resolution: u32) -> PassView {
    let res = resolution as f32;
    let half = voxel_size * res * 0.5;
    let center = Vec3::from_array(origin.map(|o| (o as f32 + res * 0.5) * voxel_size));
    let view_proj = Camera {
        eye: center + Vec3::Z * half,
        target: center,
        up: Vec3::Y,
        fov_y_rad: 1.0,
        near: 0.0,
        far: 2.0 * half,
        aspect: 1.0,
        ortho_height: Some(2.0 * half),
    }
    .view_proj();
    let mut view = PassView::orthographic(view_proj, center, [res, res], 1.0 / voxel_size, 0.5)
        .drawing(InstanceFilter::Gi);
    view.allow_software = 2;
    view
}

/// The coarsest raa-lit LOD of `chain` whose error, scaled by `world_scale`, is within half a
/// voxel; 0 when none is, or when the chain is empty (ARCHITECTURE S37 driver A; no solstice
/// line).
pub fn forward_lod(chain: &[LodLevel], world_scale: f32, voxel_size: f32) -> usize {
    chain
        .iter()
        .rposition(|level| level.error * world_scale <= 0.5 * voxel_size)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_levels_include_camera_work_after_commit_and_count_overlaps_once() {
        let resolution = 64;
        let camera = Vec3::new(7.0, 0.0, 0.0);
        let mut origins = [Some([-32; 3]); 3];
        let mut rounds = RebuildRounds::new(3);
        rounds.mark_stale();

        let (level, origin) = choose_rebuild(&origins, camera, resolution).unwrap();
        assert_eq!(level, 0);
        assert!(rounds.begin(level));
        assert_eq!(
            rounds.pending_levels(&origins, camera, resolution, Some(level)),
            3
        );
        assert_eq!(rounds.choose_outdated(), Some(2));

        origins[level] = Some(origin);
        assert_eq!(rounds.pending_levels(&origins, camera, resolution, None), 2);
        assert!(rounds.begin(2));
        assert!(rounds.begin(1));
        assert_eq!(rounds.pending_levels(&origins, camera, resolution, None), 2);
        for (level, origin) in origins.iter_mut().enumerate() {
            *origin = Some(snapped_origin(camera, voxel_size(level as u32), resolution));
        }
        assert_eq!(rounds.pending_levels(&origins, camera, resolution, None), 0);
    }
}
