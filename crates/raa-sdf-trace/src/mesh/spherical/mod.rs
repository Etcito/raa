//! Sparse CPU index and six-face work scheduler for spherical GI.

mod grid;
mod lighting;
mod reflected;
mod view;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use raa_core::RaaError;

pub use grid::{CellKey, CubeFace, RadialShell, SphericalGrid, SphericalGridConfig};
pub use lighting::{SKY_LIGHT_ID, SUN_LIGHT_ID, SphericalLight};
pub use reflected::{MAX_REFLECTED_RECEIVERS, ReflectedReceiver};

use super::{MeshTraceScene, geometry::Bounds};
use grid::{Surface, dot, sub};
use lighting::{FaceLighting, light_changes, light_hash};
use view::ViewOrder;

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
    pub reflected_cells: u32,
    pub reflected_ready_cells: u32,
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
    fn take_jobs(
        &mut self,
        budget: &mut FaceBudget,
        near_shells: u16,
        serial: &mut u64,
        jobs: &mut Vec<FaceJob>,
    ) {
        let p = &mut self.progress;
        let budget = if p.key.shell < near_shells {
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
            *serial += 1;
            self.tickets[face as usize] = *serial;
            jobs.push(FaceJob {
                key: p.key,
                slot: p.slot,
                epoch: p.epoch,
                face,
                center: p.center,
                surface_position: p.surface_position,
                normal: p.normal,
                ticket: *serial,
            });
        }
    }

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
    keys: [Range<u32>; 2],
}

impl DirtyScan {
    fn first(&self) -> Option<u32> {
        self.keys
            .iter()
            .find(|range| !range.is_empty())
            .map(|range| range.start)
    }

    fn next(&mut self) -> Option<u32> {
        self.keys.iter_mut().find_map(Iterator::next)
    }
}

/// The scheduler owns slot IDs and validity. Consumers must use the grid's anchored
/// origin for lookup and order writes to a recycled GPU slot before publishing its
/// new epoch. A cell can be sampled once all six faces of its current epoch are ready.
pub struct SphericalScheduler {
    config: SphericalSchedulerConfig,
    grid: SphericalGrid,
    cells: BTreeMap<CellKey, Entry>,
    view_order: Option<ViewOrder>,
    resident_order: BTreeMap<u32, CellKey>,
    reflected_cells: BTreeSet<CellKey>,
    reflected_candidates: Vec<reflected::Candidate>,
    reflected_cursor: usize,
    reflected_face_turn: bool,
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
            view_order: None,
            resident_order: BTreeMap::new(),
            reflected_cells: BTreeSet::new(),
            reflected_candidates: Vec::new(),
            reflected_cursor: 0,
            reflected_face_turn: false,
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

    /// Prioritize columns that can enter the primary view. The cosine is for
    /// the half-angle of a cone enclosing the view frustum, such as the diagonal FOV.
    /// A change to the view order preserves the anchor and existing face epochs.
    pub fn set_view_priority(&mut self, forward: [f32; 3], half_fov_cos: f32) {
        let Some(order) =
            ViewOrder::new(&self.grid, self.view_order.as_ref(), forward, half_fov_cos)
        else {
            return;
        };
        self.resident_order = self
            .cells
            .keys()
            .filter(|key| !self.reflected_cells.contains(key))
            .map(|&key| (order.rank(&self.grid, key), key))
            .collect();
        self.view_order = Some(order);
        self.restart_scan();
    }

    /// Replace reflected receiver demand from a current primary-grid snapshot.
    /// Keep sending supported receivers to preserve their resident face work.
    /// Delayed feedback from a retired origin or geometry revision is ignored.
    pub fn set_reflected_receivers(
        &mut self,
        origin_epoch: u64,
        geometry_revision: u64,
        receivers: &[ReflectedReceiver],
    ) -> bool {
        if self.origin_epoch != origin_epoch
            || self
                .revisions
                .is_none_or(|revision| revision.0 != geometry_revision)
        {
            return false;
        }
        self.reflected_candidates = reflected::candidates(&self.grid, receivers);
        self.reflected_cursor = 0;
        true
    }

    /// Return the reflected allocation allowance to primary discovery.
    pub fn clear_reflected_receivers(&mut self) {
        self.reflected_candidates.clear();
        self.reflected_cursor = 0;
        self.reflected_face_turn = false;
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
                .filter(|cell| !self.reflected_cells.contains(&cell.key))
                .filter_map(|cell| self.grid.key_at(cell.surface_position))
                .collect();
            self.restart();
            self.reprojected = reprojected;
        } else if let Some((old_geometry, old_material, old_light)) = self.revisions {
            if old_geometry != revisions.0 {
                self.clear_reflected_receivers();
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
        changed |= self.update_reflected(scene);
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
            let rank = self.priority_rank(key);
            if self.cells.contains_key(&key) || !self.before_capacity_cutoff(rank) {
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
            self.allocate(key, surface);
            self.resident_order.insert(rank, key);
            changed = true;
        }
        changed
    }

    fn update_reflected(&mut self, scene: &MeshTraceScene) -> bool {
        if self.reflected_cells.is_empty() && self.reflected_candidates.is_empty() {
            return false;
        }
        let candidates: BTreeSet<_> = self
            .reflected_candidates
            .iter()
            .map(|candidate| candidate.key)
            .collect();
        let released: Vec<_> = self
            .reflected_cells
            .iter()
            .copied()
            .filter(|key| !candidates.contains(key))
            .collect();
        let mut changed = !released.is_empty();
        for key in released {
            self.release(key);
            self.discovery.start = 0;
        }
        let geometry = &scene.geometry;
        let quota = self.config.max_cells as usize / 4;
        let limit = self.config.cell_queries_per_update / 4;
        while self.reflected_cells.len() < quota && self.last_cell_queries < limit {
            let Some(&candidate) = self.reflected_candidates.get(self.reflected_cursor) else {
                break;
            };
            self.reflected_cursor += 1;
            self.last_cell_queries += 1;
            let key = candidate.key;
            if self.cells.contains_key(&key) {
                continue;
            }
            let cell = self.grid.cell(key);
            let Some(surface) = geometry.find_triangle(cell.min, cell.max, |tri| {
                cell.intersect(tri)
                    .filter(|surface| candidate.supports(*surface))
            }) else {
                continue;
            };
            self.allocate(key, surface);
            self.reflected_cells.insert(key);
            changed = true;
        }
        changed
    }

    fn allocate(&mut self, key: CellKey, surface: Surface) {
        if self.free.is_empty() {
            let (&rank, &displaced) = self.resident_order.last_key_value().expect("resident slot");
            self.release(displaced);
            // If an edit frees capacity later, rediscover this still-overlapping cell.
            self.discovery.start = self.discovery.start.min(rank);
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
    }

    /// Emit each band's independent face budget in view and radial priority order.
    /// Finish each batch with complete or cancel before asking for another attempt.
    pub fn take_jobs(&mut self, mut budget: FaceBudget) -> Vec<FaceJob> {
        let mut jobs = Vec::new();
        let reserve = |count| match count {
            0 => 0,
            1 => u32::from(self.reflected_face_turn),
            _ => (count / 4).max(1),
        };
        let allowance = FaceBudget {
            near: reserve(budget.near),
            far: reserve(budget.far),
        };
        self.reflected_face_turn = !self.reflected_face_turn && !self.reflected_cells.is_empty();
        let mut reflected = allowance;
        for candidate in &self.reflected_candidates {
            if self.reflected_cells.contains(&candidate.key) {
                self.cells
                    .get_mut(&candidate.key)
                    .expect("reflected cell")
                    .take_jobs(
                        &mut reflected,
                        self.config.near_shells,
                        &mut self.serial,
                        &mut jobs,
                    );
            }
        }
        budget.near -= allowance.near - reflected.near;
        budget.far -= allowance.far - reflected.far;
        for key in self.resident_order.values() {
            self.cells.get_mut(key).expect("resident cell").take_jobs(
                &mut budget,
                self.config.near_shells,
                &mut self.serial,
                &mut jobs,
            );
        }
        for key in &self.reflected_cells {
            self.cells.get_mut(key).expect("reflected cell").take_jobs(
                &mut budget,
                self.config.near_shells,
                &mut self.serial,
                &mut jobs,
            );
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
            reflected_cells: self.reflected_cells.len() as u32,
            reflected_ready_cells: self
                .reflected_cells
                .iter()
                .filter(|key| self.cells[*key].progress.is_ready())
                .count() as u32,
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
        self.resident_order.clear();
        self.reflected_cells.clear();
        self.clear_reflected_receivers();
        self.reprojected.clear();
        self.restart_scan();
    }

    fn restart_scan(&mut self) {
        self.discovery = 0..u32::from(self.grid.shell_count()) * self.grid.cells_per_shell();
        self.dirty = None;
        self.pending_dirty = None;
    }

    fn release(&mut self, key: CellKey) {
        let entry = self.cells.remove(&key).expect("resident cell");
        if !self.reflected_cells.remove(&key) {
            self.resident_order.remove(&self.priority_rank(key));
        }
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

    fn priority_rank(&self, key: CellKey) -> u32 {
        self.view_order.as_ref().map_or_else(
            || self.grid.ordinal(key),
            |order| order.rank(&self.grid, key),
        )
    }

    fn priority_key(&self, rank: u32) -> CellKey {
        self.view_order
            .as_ref()
            .map_or_else(|| self.grid.key(rank), |order| order.key(&self.grid, rank))
    }

    fn before_capacity_cutoff(&self, rank: u32) -> bool {
        !self.free.is_empty()
            || self
                .resident_order
                .last_key_value()
                .is_some_and(|(&worst, _)| rank < worst)
    }

    fn next_discovery(&mut self) -> Option<(CellKey, Option<Bounds>)> {
        if !self.before_capacity_cutoff(self.discovery.start) {
            return None;
        }
        self.discovery
            .next()
            .map(|rank| (self.priority_key(rank), None))
    }

    fn next_dirty(&mut self) -> Option<(CellKey, Option<Bounds>)> {
        loop {
            if let Some(rank) = self.dirty.as_ref().and_then(DirtyScan::first) {
                if self.before_capacity_cutoff(rank) {
                    let scan = self.dirty.as_mut().expect("active scan");
                    let rank = scan.next().expect("pending scan rank");
                    let bounds = scan.bounds;
                    return Some((self.priority_key(rank), Some(bounds)));
                }
                self.discovery.start = self.discovery.start.min(rank);
            }
            self.dirty = None;
            let bounds = self.pending_dirty.take()?;
            let (min_radius, max_radius) = bounds.radial_range(self.grid.origin());
            let shells = 0..self.grid.shell_count();
            let first = shells.clone().find(|&i| {
                f64::from(self.grid.radial_shell(i).expect("valid shell").outer) >= min_radius
            })?;
            let end = shells.rev().find(|&i| {
                f64::from(self.grid.radial_shell(i).expect("valid shell").inner) <= max_radius
            })? + 1;
            let shells = u32::from(first)..u32::from(end);
            let keys = self.view_order.as_ref().map_or_else(
                || {
                    let columns = self.grid.cells_per_shell();
                    [shells.start * columns..shells.end * columns, 0..0]
                },
                |order| order.shell_ranges(shells.clone()),
            );
            self.dirty = Some(DirtyScan { bounds, keys });
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
