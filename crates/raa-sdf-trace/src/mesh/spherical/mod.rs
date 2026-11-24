//! Sparse CPU index and six-face work scheduler for spherical GI.

mod grid;
mod lighting;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use raa_core::RaaError;

pub use grid::{CellKey, CubeFace, RadialShell, SphericalGrid, SphericalGridConfig};
pub use lighting::{SKY_LIGHT_ID, SUN_LIGHT_ID, SphericalLight};

use super::{MeshTraceScene, geometry::Bounds};
use grid::{dot, sub};
use lighting::{FaceLighting, light_changes, light_hash};

pub const ALL_FACES: u8 = 0b11_1111;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalSchedulerConfig {
    pub grid: SphericalGridConfig,
    pub max_cells: u32,
    /// Shells starting at clipStart that draw from the near face budget.
    pub near_shells: u16,
    /// Maximum candidate cells examined per update, including broad-phase rejects.
    pub cell_queries_per_update: u32,
    /// Maximum camera translation from the current world-space grid anchor, in metres.
    /// The default is 0.25 metres. Camera rotation and translation within this
    /// tolerance preserve face progress. A recenter confirms reprojected world
    /// candidates against the BVH before dispatching a fresh face epoch.
    pub origin_tolerance: f32,
}

impl Default for SphericalSchedulerConfig {
    fn default() -> Self {
        Self {
            grid: SphericalGridConfig::default(),
            max_cells: 128,
            near_shells: 4,
            cell_queries_per_update: 768,
            origin_tolerance: 0.25,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaceBudget {
    pub near: u32,
    pub far: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellProgress {
    pub key: CellKey,
    pub slot: u32,
    pub epoch: u64,
    pub center: [f32; 3],
    /// Point on an overlapping triangle, with its world-space geometric normal.
    pub surface_position: [f32; 3],
    pub normal: [f32; 3],
    pub ready_faces: u8,
    pub in_flight_faces: u8,
    /// Hash of the reported contributing IDs and their capture light values.
    pub light_hash: u64,
}

impl CellProgress {
    pub fn is_ready(self) -> bool {
        self.ready_faces == ALL_FACES
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceJob {
    pub key: CellKey,
    pub slot: u32,
    pub epoch: u64,
    pub face: CubeFace,
    pub center: [f32; 3],
    pub surface_position: [f32; 3],
    pub normal: [f32; 3],
    ticket: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub cells: u32,
    pub near_cells: u32,
    pub far_cells: u32,
    pub ready_cells: u32,
    pub ready_faces: u32,
    pub in_flight_faces: u32,
    pub pending_faces: u32,
    pub origin_epoch: u64,
    pub light_revision: u64,
    pub last_cell_queries: u32,
    /// Resident cells with lighting metadata from all six current faces.
    pub lighting_metadata_ready_cells: u32,
    /// Cumulative cells with ready or in-flight work cleared by lighting changes.
    pub light_invalidated_cells: u64,
    pub light_invalidated_faces: u64,
}

struct Entry {
    progress: CellProgress,
    tickets: [u64; 6],
    lighting: FaceLighting,
}

impl Entry {
    fn reset_faces(&mut self, serial: &mut u64) {
        *serial += 1;
        self.progress.epoch = *serial;
        self.progress.ready_faces = 0;
        self.progress.in_flight_faces = 0;
        self.progress.light_hash = 0;
        self.lighting = FaceLighting::default();
    }
}

struct DirtyScan {
    bounds: Bounds,
    keys: Range<u32>,
}

/// The scheduler owns slot IDs and validity. Consumers must use the grid's anchored
/// origin for lookup and order writes to a recycled GPU slot before publishing its
/// new epoch. A cell can be sampled once all six faces of its current epoch are ready.
pub struct SphericalScheduler {
    config: SphericalSchedulerConfig,
    grid: SphericalGrid,
    cells: BTreeMap<CellKey, Entry>,
    free: BTreeSet<u32>,
    released: BTreeSet<u32>,
    serial: u64,
    revisions: Option<(u64, u64, u64)>,
    local_lights: Vec<SphericalLight>,
    light_invalidated_cells: u64,
    light_invalidated_faces: u64,
    origin_epoch: u64,
    discovery: Range<u32>,
    reprojected: BTreeSet<CellKey>,
    dirty: Option<DirtyScan>,
    pending_dirty: Option<Bounds>,
    last_cell_queries: u32,
}

impl SphericalScheduler {
    pub fn new(config: SphericalSchedulerConfig) -> Result<Self, RaaError> {
        if !(1..=4096).contains(&config.max_cells)
            || config.near_shells > 64
            || !(1..=65536).contains(&config.cell_queries_per_update)
            || !config.origin_tolerance.is_finite()
            || config.origin_tolerance < 0.0
        {
            return Err(invalid_config(
                "invalid capacity, shell split, query budget, or origin tolerance",
            ));
        }
        let grid = SphericalGrid::new(config.grid)?;
        Ok(Self {
            config,
            grid,
            cells: BTreeMap::new(),
            free: (0..config.max_cells).collect(),
            released: BTreeSet::new(),
            serial: 0,
            revisions: None,
            local_lights: Vec::new(),
            light_invalidated_cells: 0,
            light_invalidated_faces: 0,
            origin_epoch: 0,
            discovery: 0..0,
            reprojected: BTreeSet::new(),
            dirty: None,
            pending_dirty: None,
            last_cell_queries: 0,
        })
    }

    pub fn grid(&self) -> &SphericalGrid {
        &self.grid
    }

    pub fn cells(&self) -> impl ExactSizeIterator<Item = &CellProgress> {
        self.cells.values().map(|entry| &entry.progress)
    }

    pub fn progress(&self, key: CellKey) -> Option<&CellProgress> {
        self.cells.get(&key).map(|entry| &entry.progress)
    }

    /// Contributing stable light IDs once all six current face observations arrived.
    pub fn contributing_light_ids(&self, key: CellKey) -> Option<&[u64]> {
        let lighting = &self.cells.get(&key)?.lighting;
        lighting
            .is_known()
            .then_some(lighting.contributors.as_slice())
    }

    /// Slots vacated since the previous drain, including any reused by this update.
    /// Apply clears before recording writes for the new occupants.
    pub fn take_released_slots(&mut self) -> BTreeSet<u32> {
        std::mem::take(&mut self.released)
    }

    /// Update a scene with sun and environment lighting and no local lights.
    pub fn update(
        &mut self,
        scene: &MeshTraceScene,
        camera_origin: [f32; 3],
        light_revision: u64,
    ) -> bool {
        self.update_with_lights(scene, camera_origin, light_revision, &[])
    }

    /// Update before dispatch and lookup. Returns whether the anchor, residency or
    /// face validity changed. The global revision covers sun and environment,
    /// while local lights carry their own stable IDs and shading values.
    pub fn update_with_lights(
        &mut self,
        scene: &MeshTraceScene,
        camera_origin: [f32; 3],
        global_light_revision: u64,
        local_lights: &[SphericalLight],
    ) -> bool {
        self.last_cell_queries = 0;
        if !camera_origin.into_iter().all(f32::is_finite) {
            return false;
        }
        let geometry = &scene.geometry;
        let revisions = (
            geometry.revision,
            scene.material_revision,
            global_light_revision,
        );
        let local_changes = if self.local_lights == local_lights {
            Vec::new()
        } else {
            let mut sorted = local_lights.to_vec();
            sorted.sort_unstable_by_key(|light| light.id());
            let changes = light_changes(&self.local_lights, &sorted);
            self.local_lights = sorted;
            changes
        };
        let displacement = sub(
            camera_origin.map(f64::from),
            self.grid.origin.map(f64::from),
        );
        let recenter = self.revisions.is_none()
            || dot(displacement, displacement) > f64::from(self.config.origin_tolerance).powi(2);
        let mut changed = recenter;
        if recenter {
            self.grid.origin = camera_origin;
            self.origin_epoch += 1;
            let reprojected = self
                .cells()
                .filter_map(|cell| self.grid.key_at(cell.surface_position))
                .collect();
            self.restart();
            self.reprojected = reprojected;
        } else if let Some((old_geometry, old_material, old_light)) = self.revisions {
            if old_geometry != revisions.0 {
                if old_geometry.wrapping_add(1) == revisions.0 {
                    changed |= self.invalidate_geometry(&geometry.dirty_bounds);
                } else {
                    // Only the most recent geometry delta is retained.
                    changed |= !self.cells.is_empty();
                    self.restart();
                }
            }
            let global_changed = old_light != revisions.2;
            let material_changed = old_material != revisions.1;
            if global_changed || material_changed || !local_changes.is_empty() {
                let scene_bounds = geometry.world_bounds();
                for entry in self.cells.values_mut() {
                    let work = entry.progress.ready_faces | entry.progress.in_flight_faces;
                    if work == 0 {
                        continue;
                    }
                    let light_changed =
                        global_changed || entry.lighting.affected_by(&local_changes, scene_bounds);
                    if light_changed {
                        self.light_invalidated_cells += 1;
                        self.light_invalidated_faces += u64::from(work.count_ones());
                    }
                    if light_changed || material_changed {
                        entry.reset_faces(&mut self.serial);
                        changed = true;
                    }
                }
            }
        }
        self.revisions = Some(revisions);
        if geometry.count() == 0 {
            self.discovery.start = self.discovery.end;
            self.reprojected.clear();
            self.dirty = None;
            self.pending_dirty = None;
            return changed;
        }

        // Alternate the dirty scan with forward discovery so continuous edits
        // keep their own progress without preventing initial occupancy discovery.
        let mut prefer_dirty = true;
        let mut reprojection_budget = self.config.cell_queries_per_update.div_ceil(2);
        while self.last_cell_queries < self.config.cell_queries_per_update {
            let reprojected = if reprojection_budget > 0 {
                self.reprojected.pop_first().map(|key| (key, None))
            } else {
                None
            };
            let candidate = if reprojected.is_some() {
                reprojection_budget -= 1;
                reprojected
            } else if prefer_dirty {
                self.next_dirty().or_else(|| self.next_discovery())
            } else {
                self.next_discovery().or_else(|| self.next_dirty())
            };
            let Some((key, bounds)) = candidate else {
                break;
            };
            prefer_dirty = !prefer_dirty;
            self.last_cell_queries += 1;
            if self.cells.contains_key(&key) || !self.before_capacity_cutoff(self.grid.ordinal(key))
            {
                continue;
            }
            let cell = self.grid.cell(key);
            if bounds.is_some_and(|bounds| !bounds.overlaps(cell.min, cell.max)) {
                continue;
            }
            let Some(surface) =
                geometry.find_triangle(cell.min, cell.max, |tri| cell.intersect(tri))
            else {
                continue;
            };
            if self.free.is_empty() {
                let farthest = *self.cells.last_key_value().expect("resident slot").0;
                self.release(farthest);
                // If an edit frees capacity later, rediscover this still-overlapping cell.
                self.discovery.start = self.discovery.start.min(self.grid.ordinal(farthest));
            }
            let slot = self.free.pop_first().expect("free or released slot");
            self.serial += 1;
            self.cells.insert(
                key,
                Entry {
                    progress: CellProgress {
                        key,
                        slot,
                        epoch: self.serial,
                        center: self.grid.center(key).expect("valid cell key"),
                        surface_position: surface.position,
                        normal: surface.normal,
                        ready_faces: 0,
                        in_flight_faces: 0,
                        light_hash: 0,
                    },
                    tickets: [0; 6],
                    lighting: FaceLighting::default(),
                },
            );
            changed = true;
        }
        changed
    }

    /// Emit up to each band's independent face budget, in increasing shell order.
    /// Finish each batch with complete or cancel before asking for another attempt.
    pub fn take_jobs(&mut self, mut budget: FaceBudget) -> Vec<FaceJob> {
        let mut jobs = Vec::new();
        for entry in self.cells.values_mut() {
            let p = &mut entry.progress;
            let budget = if p.key.shell < self.config.near_shells {
                &mut budget.near
            } else {
                &mut budget.far
            };
            for face in CubeFace::ALL {
                let bit = 1 << face as u8;
                if *budget == 0 {
                    break;
                }
                if (p.ready_faces | p.in_flight_faces) & bit != 0 {
                    continue;
                }
                *budget -= 1;
                p.in_flight_faces |= bit;
                self.serial += 1;
                entry.tickets[face as usize] = self.serial;
                jobs.push(FaceJob {
                    key: p.key,
                    slot: p.slot,
                    epoch: p.epoch,
                    face,
                    center: p.center,
                    surface_position: p.surface_position,
                    normal: p.normal,
                    ticket: self.serial,
                });
            }
        }
        jobs
    }

    /// Whether the job is still the active attempt for this face.
    pub fn is_current(&self, job: FaceJob) -> bool {
        self.cells.get(&job.key).is_some_and(|entry| {
            entry.progress.slot == job.slot
                && entry.progress.epoch == job.epoch
                && entry.progress.in_flight_faces & (1 << job.face as u8) != 0
                && entry.tickets[job.face as usize] == job.ticket
        })
    }

    pub fn complete(&mut self, job: FaceJob) -> bool {
        self.finish(job, true)
    }

    /// Requeue an aborted attempt while preserving completed faces.
    pub fn cancel(&mut self, job: FaceJob) -> bool {
        self.finish(job, false)
    }

    /// Attach asynchronous metadata from a submitted face. A retired epoch or
    /// a canceled and retried face cannot overwrite the current dependencies.
    pub fn record_face_lighting(
        &mut self,
        job: FaceJob,
        hit_bounds: Option<([f32; 3], [f32; 3])>,
        contributing_ids: &[u64],
    ) -> bool {
        let Some(entry) = self.cells.get_mut(&job.key) else {
            return false;
        };
        let bit = 1 << job.face as u8;
        if entry.progress.slot != job.slot
            || entry.progress.epoch != job.epoch
            || entry.progress.ready_faces & bit == 0
            || entry.tickets[job.face as usize] != job.ticket
        {
            return false;
        }
        let bounds = hit_bounds.map(|(min, max)| Bounds { min, max });
        if bounds.is_some_and(|b| {
            (0..3).any(|i| !b.min[i].is_finite() || !b.max[i].is_finite() || b.min[i] > b.max[i])
        }) || !entry.lighting.record(bit, bounds, contributing_ids)
        {
            return false;
        }
        let global = self.revisions.map_or(0, |(_, _, light)| light);
        entry.progress.light_hash =
            light_hash(global, &entry.lighting.contributors, &self.local_lights);
        true
    }

    pub fn stats(&self) -> SchedulerStats {
        let mut stats = SchedulerStats {
            cells: self.cells.len() as u32,
            origin_epoch: self.origin_epoch,
            light_revision: self.revisions.map_or(0, |(_, _, light)| light),
            last_cell_queries: self.last_cell_queries,
            light_invalidated_cells: self.light_invalidated_cells,
            light_invalidated_faces: self.light_invalidated_faces,
            ..SchedulerStats::default()
        };
        for entry in self.cells.values() {
            let cell = entry.progress;
            stats.lighting_metadata_ready_cells += u32::from(entry.lighting.is_known());
            stats.near_cells += u32::from(cell.key.shell < self.config.near_shells);
            stats.ready_cells += u32::from(cell.is_ready());
            stats.ready_faces += cell.ready_faces.count_ones();
            stats.in_flight_faces += cell.in_flight_faces.count_ones();
        }
        stats.far_cells = stats.cells - stats.near_cells;
        stats.pending_faces = stats.cells * 6 - stats.ready_faces - stats.in_flight_faces;
        stats
    }

    fn finish(&mut self, job: FaceJob, ready: bool) -> bool {
        if !self.is_current(job) {
            return false;
        }
        let p = &mut self.cells.get_mut(&job.key).expect("current job").progress;
        let bit = 1 << job.face as u8;
        p.in_flight_faces &= !bit;
        if ready {
            p.ready_faces |= bit;
        }
        true
    }

    fn restart(&mut self) {
        for entry in std::mem::take(&mut self.cells).into_values() {
            self.free.insert(entry.progress.slot);
            self.released.insert(entry.progress.slot);
        }
        self.discovery = 0..u32::from(self.grid.shell_count()) * self.grid.cells_per_shell();
        self.reprojected.clear();
        self.dirty = None;
        self.pending_dirty = None;
    }

    fn release(&mut self, key: CellKey) {
        let entry = self.cells.remove(&key).expect("resident cell");
        self.free.insert(entry.progress.slot);
        self.released.insert(entry.progress.slot);
    }

    fn invalidate_geometry(&mut self, bounds: &[Bounds]) -> bool {
        if bounds.is_empty() {
            return false;
        }
        let keys: Vec<_> = self
            .cells
            .keys()
            .copied()
            .filter(|&key| {
                let cell = self.grid.cell(key);
                bounds
                    .iter()
                    .any(|bounds| bounds.overlaps(cell.min, cell.max))
            })
            .collect();
        for &key in &keys {
            self.release(key);
        }
        for &bounds in bounds {
            match &mut self.pending_dirty {
                Some(pending) => pending.include(bounds),
                slot => *slot = Some(bounds),
            }
        }
        !keys.is_empty()
    }

    fn before_capacity_cutoff(&self, index: u32) -> bool {
        !self.free.is_empty()
            || self
                .cells
                .last_key_value()
                .is_some_and(|(&key, _)| index < self.grid.ordinal(key))
    }

    fn next_discovery(&mut self) -> Option<(CellKey, Option<Bounds>)> {
        if !self.before_capacity_cutoff(self.discovery.start) {
            return None;
        }
        self.discovery
            .next()
            .map(|index| (self.grid.key(index), None))
    }

    fn next_dirty(&mut self) -> Option<(CellKey, Option<Bounds>)> {
        loop {
            if let Some(scan) = &self.dirty
                && !scan.keys.is_empty()
                && self.before_capacity_cutoff(scan.keys.start)
            {
                let scan = self.dirty.as_mut().expect("active scan");
                return scan
                    .keys
                    .next()
                    .map(|index| (self.grid.key(index), Some(scan.bounds)));
            }
            if let Some(scan) = self.dirty.take()
                && !scan.keys.is_empty()
            {
                self.discovery.start = self.discovery.start.min(scan.keys.start);
            }
            let bounds = self.pending_dirty.take()?;
            let (min_radius, max_radius) = bounds.radial_range(self.grid.origin());
            let shells = 0..self.grid.shell_count();
            let first = shells.clone().find(|&i| {
                f64::from(self.grid.radial_shell(i).expect("valid shell").outer) >= min_radius
            })?;
            let end = shells.rev().find(|&i| {
                f64::from(self.grid.radial_shell(i).expect("valid shell").inner) <= max_radius
            })? + 1;
            let n = self.grid.cells_per_shell();
            self.dirty = Some(DirtyScan {
                bounds,
                keys: u32::from(first) * n..u32::from(end) * n,
            });
        }
    }
}

fn invalid_config(message: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "spherical_gi.invalid_settings",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests;
