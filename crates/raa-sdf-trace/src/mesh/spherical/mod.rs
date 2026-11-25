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

use super::{
    MeshTraceScene,
    geometry::{Bounds, Geometry},
};
use grid::{Surface, dot, point_triangle_distance_sq, sub};
use lighting::{FaceLighting, find_light_by_id, light_changes, light_hash};
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
    /// Maximum camera translation from the current world-space grid anchor, in metres,
    /// and the step of the world lattice a recenter snaps the anchor to. 0 (the
    /// default) uses the grid's smallest cell (`SphericalGrid::smallest_cell`), so
    /// the grid only ever moves by whole small cells. A recenter rekeys the
    /// world-space probes into the moved cells: captured faces survive, only probes
    /// that leave the grid or collide in one cell are dropped.
    pub origin_tolerance: f32,
    /// Keep a free-space probe at the grid anchor for the inner sphere below clipStart.
    pub center_probe: bool,
    /// Extra pool slots for published probes that left the grid, lost a rekey collision
    /// or were evicted (least recently cached first out). A recenter offers them to the
    /// moved cells again, so walking back reuses them without a recapture.
    pub cache_cells: u32,
    /// Derive the outer radius (zMax) from the scene: the farthest corner of the
    /// geometry bounds from the grid anchor, quantised to quarter octaves. It grows as
    /// soon as the scene would be cut and shrinks only below 1/sqrt(2) of the current
    /// radius, so walking does not pump the shells. `grid.z_max` is the initial value.
    pub auto_range: bool,
    /// A probe whose cell centre lies inside closed geometry keeps the centre (no
    /// relocation, no rejection): its capture rays skip every surface met from behind
    /// (inside a solid) and record the first front face or the sky, distances still
    /// measured from the centre, like a camera with its near plane past the enclosing
    /// solid. Off: the older placement at the free offset with the most clearance.
    pub inside_capture: bool,
}

impl SphericalSchedulerConfig {
    /// GPU slots: resident probes plus the probe cache.
    pub fn pool_slots(&self) -> u32 {
        self.max_cells + self.cache_cells
    }
}

impl Default for SphericalSchedulerConfig {
    fn default() -> Self {
        Self {
            grid: SphericalGridConfig::default(),
            max_cells: 1024,
            near_shells: 4,
            cell_queries_per_update: 768,
            origin_tolerance: 0.0,
            center_probe: true,
            cache_cells: 0,
            auto_range: false,
            inside_capture: true,
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
    /// It only witnesses nearby geometry; captures start from `probe`.
    pub surface_position: [f32; 3],
    pub normal: [f32; 3],
    /// Free-space capture origin inside the cell, pushed away from nearby triangles.
    /// It stays fixed in world space while the grid recenters.
    pub probe: [f32; 3],
    pub ready_faces: u8,
    pub in_flight_faces: u8,
    /// Whether a complete capture of this probe was projected at least once. A light
    /// or material refresh keeps the previous projection visible until it is replaced.
    pub published: bool,
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
    pub probe: [f32; 3],
    ticket: u64,
}

/// A resident cell's ready faces at one moment (see `SphericalScheduler::face_tokens`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceToken {
    pub key: CellKey,
    pub slot: u32,
    pub epoch: u64,
    pub ready: u8,
    tickets: [u64; 6],
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
    /// Resident cells in the view cone (and the centre probe), and how many of them
    /// have been published at least once.
    pub visible_cells: u32,
    pub visible_published_cells: u32,
    /// Discovery, reprojection or a dirty rescan still has candidate cells to examine.
    pub discovery_pending: bool,
    /// Published probes held in the cache, outside the grid lookup.
    pub cached_cells: u32,
    /// Cumulative cached probes adopted back into the grid by a recenter.
    pub adopted_cells: u64,
    /// Residency is full while discovery still has farther candidates: the outer
    /// shells are cut by maxProbes and fall back to the ambient there.
    pub capacity_limited: bool,
}

impl SchedulerStats {
    /// Visible probes are missing or not yet discovered: capture should burst.
    pub fn visible_missing(&self) -> bool {
        self.discovery_pending || self.visible_published_cells < self.visible_cells
    }
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
        let p = &self.progress;
        let budget = if p.key.is_center() || p.key.shell < near_shells {
            &mut budget.near
        } else {
            &mut budget.far
        };
        self.take_faces(budget, serial, jobs);
    }

    fn take_faces(&mut self, budget: &mut u32, serial: &mut u64, jobs: &mut Vec<FaceJob>) {
        let p = &mut self.progress;
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
                probe: p.probe,
                ticket: *serial,
            });
        }
    }

    /// Recapture only the faces in `bits`: the epoch stays (the other faces remain valid) and
    /// the published projection stays visible until the cell is complete again.
    fn reset_face_bits(&mut self, bits: u8) -> u8 {
        let bits = bits & self.progress.ready_faces;
        if bits != 0 {
            self.progress.ready_faces &= !bits;
            self.progress.light_hash = 0;
            self.lighting.forget(bits);
        }
        bits
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
    /// Per-face light invalidation with a GPU reach test for added or moved lights
    /// (otherwise whole cells over their faces' hit bounds, the original rule).
    light_reach: bool,
    reach_requests: Vec<u64>,
    light_invalidated_cells: u64,
    light_invalidated_faces: u64,
    origin_epoch: u64,
    discovery: Range<u32>,
    reprojected: BTreeSet<CellKey>,
    dirty: Option<DirtyScan>,
    pending_dirty: Option<Bounds>,
    last_cell_queries: u32,
    /// Probe cache in least-recently-cached order (key = stamp).
    cache: BTreeMap<u64, Entry>,
    cache_stamp: u64,
    adopted: BTreeSet<u32>,
    adopted_cells: u64,
}

impl SphericalScheduler {
    pub fn new(config: SphericalSchedulerConfig) -> Result<Self, RaaError> {
        let min_cells = 1 + u32::from(config.center_probe);
        if !(min_cells..=4096).contains(&config.max_cells)
            || config.cache_cells > 4096
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
            free: (0..config.pool_slots()).collect(),
            released: BTreeSet::new(),
            serial: 0,
            revisions: None,
            local_lights: Vec::new(),
            light_reach: false,
            reach_requests: Vec::new(),
            light_invalidated_cells: 0,
            light_invalidated_faces: 0,
            origin_epoch: 0,
            discovery: 0..0,
            reprojected: BTreeSet::new(),
            dirty: None,
            pending_dirty: None,
            last_cell_queries: 0,
            cache: BTreeMap::new(),
            cache_stamp: 0,
            adopted: BTreeSet::new(),
            adopted_cells: 0,
        })
    }

    /// Prioritize capture of columns that can enter the primary view. The cosine is
    /// for the half-angle of a cone enclosing the view frustum, such as the diagonal
    /// FOV. The grid is anchored on world axes and residency is shell-major in world
    /// order, so a view change only reorders capture: probe positions, slots, valid
    /// faces and residency never depend on the camera orientation.
    pub fn set_view_priority(&mut self, forward: [f32; 3], half_fov_cos: f32) {
        if let Some(order) =
            ViewOrder::new(&self.grid, self.view_order.as_ref(), forward, half_fov_cos)
        {
            self.view_order = Some(order);
        }
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
        let candidates = reflected::candidates(&self.grid, receivers);
        if candidates != self.reflected_candidates {
            self.reflected_candidates = candidates;
            self.reflected_cursor = 0;
        }
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

    /// Slots of cached probes adopted back into the grid since the previous drain;
    /// their projection is intact, the consumer only restarts their fade-in.
    pub fn take_adopted_slots(&mut self) -> BTreeSet<u32> {
        std::mem::take(&mut self.adopted)
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
        // The snapped origin lies within sqrt(3)/2 steps of the camera, inside the
        // tolerance, so a recenter cannot retrigger on the next update.
        let step = if self.config.origin_tolerance > 0.0 {
            self.config.origin_tolerance
        } else {
            self.grid.smallest_cell()
        };
        let tolerance = step;
        let recenter = self.revisions.is_none()
            || dot(displacement, displacement) > f64::from(tolerance).powi(2);
        let origin = if recenter {
            snap_origin(camera_origin, step)
        } else {
            self.grid.origin
        };
        let regrid = self
            .auto_outer_radius(geometry, origin)
            .is_some_and(|z_max| {
                let config = SphericalGridConfig {
                    z_max,
                    ..self.grid.config()
                };
                match SphericalGrid::new(config) {
                    Ok(grid) => {
                        self.grid = grid;
                        true
                    }
                    Err(_) => false,
                }
            });
        let mut changed = recenter || regrid;
        if recenter || regrid {
            // Probes are world space: a new anchor or new shell radii only rekeys them.
            self.grid.origin = origin;
            self.origin_epoch += 1;
            self.rekey();
        }
        if let Some((old_geometry, old_material, old_light)) = self.revisions {
            // Cached captures are not tracked per light: any change drops them.
            if old_geometry != revisions.0
                || old_material != revisions.1
                || old_light != revisions.2
                || !local_changes.is_empty()
            {
                self.clear_cache();
            }
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
            if self.light_reach && !global_changed && !material_changed && !local_changes.is_empty()
            {
                let scene_bounds = geometry.world_bounds();
                for entry in self.cells.values_mut() {
                    let bits = entry.lighting.affected_faces(
                        entry.progress.ready_faces,
                        &local_changes,
                        scene_bounds,
                    );
                    let reset = entry.reset_face_bits(bits);
                    if reset != 0 {
                        self.light_invalidated_cells += 1;
                        self.light_invalidated_faces += u64::from(reset.count_ones());
                        changed = true;
                    }
                }
                for change in &local_changes {
                    if let Some(light) = change.reach_target() {
                        self.reach_requests.push(light.id());
                    }
                }
            } else if global_changed || material_changed || !local_changes.is_empty() {
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
        if self.config.center_probe && !self.cells.contains_key(&CellKey::CENTER) {
            let probe = self.place_probe(geometry, CellKey::CENTER);
            changed |= self.allocate(CellKey::CENTER, None, probe);
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
            let probe = self.place_probe(geometry, key);
            if self.allocate(key, Some(surface), probe) {
                self.resident_order.insert(rank, key);
                changed = true;
            }
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
            let probe = self.place_probe(geometry, key);
            if self.allocate(key, Some(surface), probe) {
                self.reflected_cells.insert(key);
                changed = true;
            }
        }
        changed
    }

    /// Choose a free-space capture origin inside the cell: its centre when no triangle
    /// lies within the clearance radius, otherwise the candidate offset with the most
    /// clearance. The test is exact on the shared BVH, so a probe never sits on a wall.
    fn place_probe(&self, geometry: &Geometry, key: CellKey) -> [f32; 3] {
        let (radial, angular) = self.grid.cell_extent(key);
        let rho = (0.2 * radial.min(angular)).clamp(0.02, 0.5);
        let clearance = |point: [f32; 3]| {
            let p = point.map(f64::from);
            let mut best = rho * rho;
            geometry.find_triangle(p.map(|v| v - rho), p.map(|v| v + rho), |tri| {
                best = best.min(point_triangle_distance_sq(p, tri));
                (best == 0.0).then_some(())
            });
            best.sqrt()
        };
        let center = self.grid.cell_point(key, [0.0; 3]);
        if self.config.inside_capture {
            return center;
        }
        let mut best = (clearance(center), center);
        if best.0 >= rho {
            return center;
        }
        const STEP: f64 = 1.0 / 3.0;
        let mut offsets: Vec<[f64; 3]> = (0..27)
            .filter(|&i| i != 13)
            .map(|i| [i % 3, i / 3 % 3, i / 9].map(|v| f64::from(v as u8) * STEP - STEP))
            .collect();
        offsets.sort_by(|a, b| dot(*a, *a).total_cmp(&dot(*b, *b)));
        for offset in offsets {
            let point = self.grid.cell_point(key, offset);
            let distance = clearance(point);
            if distance > best.0 {
                best = (distance, point);
                if distance >= rho {
                    break;
                }
            }
        }
        best.1
    }

    /// Returns false when every slot belongs to the centre probe.
    fn allocate(&mut self, key: CellKey, surface: Option<Surface>, probe: [f32; 3]) -> bool {
        if self.resident_full() {
            let Some((&rank, &displaced)) = self.resident_order.last_key_value() else {
                return false;
            };
            let entry = self.detach(displaced);
            self.stash(entry, None);
            // If an edit frees capacity later, rediscover this still-overlapping cell.
            self.discovery.start = self.discovery.start.min(rank);
        }
        let slot = self.take_slot();
        self.serial += 1;
        let surface = surface.unwrap_or(Surface {
            position: probe,
            normal: [0.0; 3],
        });
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
                    probe,
                    ready_faces: 0,
                    in_flight_faces: 0,
                    published: false,
                    light_hash: 0,
                },
                tickets: [0; 6],
                lighting: FaceLighting::default(),
            },
        );
        true
    }

    /// Move world-space probes into the cells of the recentred grid. Each probe keeps
    /// its slot, captured faces and publication; in-flight faces are reissued because
    /// their jobs name the retired key. When several probes land in one cell, a
    /// published probe stays first, then the one nearest the new cell centre. The centre probe stays the centre while it
    /// remains inside the inner sphere.
    fn rekey(&mut self) {
        let old = std::mem::take(&mut self.cells);
        self.resident_order.clear();
        self.reflected_cells.clear();
        self.clear_reflected_receivers();
        self.reprojected.clear();
        // A published probe outranks an unpublished one, so a collision never pops a
        // visible probe; among equals the one nearest the new cell centre stays.
        // Cached probes compete like residents; losers return to the cache.
        let mut placed: BTreeMap<CellKey, ((bool, f64), Option<u64>, Entry)> = BTreeMap::new();
        let mut leftovers: Vec<(Option<u64>, Entry)> = Vec::new();
        let mut witnesses = Vec::new();
        let candidates = old
            .into_values()
            .map(|entry| (None, entry))
            .chain(
                std::mem::take(&mut self.cache)
                    .into_iter()
                    .map(|(stamp, entry)| (Some(stamp), entry)),
            )
            .collect::<Vec<_>>();
        for (stamp, mut entry) in candidates {
            let progress = entry.progress;
            if !progress.key.is_center() && stamp.is_none() {
                witnesses.push(progress.surface_position);
            }
            let key = self
                .grid
                .probe_key_at(progress.probe)
                .filter(|key| self.config.center_probe || !key.is_center());
            let Some(key) = key else {
                leftovers.push((stamp, entry));
                continue;
            };
            let center = self.grid.center(key).expect("rekeyed cell");
            let delta = sub(progress.probe.map(f64::from), center.map(f64::from));
            let distance = (!progress.published, dot(delta, delta));
            entry.progress.key = key;
            entry.progress.center = center;
            entry.progress.in_flight_faces = 0;
            entry.tickets = [0; 6];
            let displaced = match placed.get(&key) {
                Some((best, _, _)) if *best <= distance => Some((stamp, entry)),
                _ => placed
                    .insert(key, (distance, stamp, entry))
                    .map(|(_, stamp, entry)| (stamp, entry)),
            };
            leftovers.extend(displaced);
        }
        // Residency stays bounded: the centre first, then priority rank.
        let mut placed: Vec<_> = placed.into_iter().collect();
        placed.sort_by_key(|(key, _)| (!key.is_center(), self.priority_rank(*key)));
        for (index, (key, (_, stamp, entry))) in placed.into_iter().enumerate() {
            if index >= self.config.max_cells as usize {
                leftovers.push((stamp, entry));
                continue;
            }
            if stamp.is_some() {
                self.adopted.insert(entry.progress.slot);
                self.adopted_cells += 1;
            }
            if !key.is_center() {
                self.resident_order.insert(self.priority_rank(key), key);
            }
            self.cells.insert(key, entry);
        }
        // Former residents are the most recent cache entries.
        leftovers.sort_by_key(|(stamp, _)| stamp.is_none());
        for (stamp, entry) in leftovers {
            self.stash(entry, stamp);
        }
        // Old surface witnesses are certain geometry: confirm their cells first.
        self.reprojected = witnesses
            .into_iter()
            .filter_map(|position| self.grid.key_at(position))
            .filter(|key| !self.cells.contains_key(key))
            .collect();
        self.restart_scan();
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
        if let Some(center) = self.cells.get_mut(&CellKey::CENTER) {
            center.take_jobs(
                &mut budget,
                self.config.near_shells,
                &mut self.serial,
                &mut jobs,
            );
        }
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
        for key in self.capture_order() {
            self.cells.get_mut(&key).expect("resident cell").take_jobs(
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

    /// Emit up to `faces` jobs from one shared pool, in capture priority order and
    /// regardless of the near/far band split (used by the GPU-time budget).
    pub fn take_jobs_shared(&mut self, mut faces: u32) -> Vec<FaceJob> {
        let mut jobs = Vec::new();
        let mut order = Vec::with_capacity(self.cells.len());
        if self.cells.contains_key(&CellKey::CENTER) {
            order.push(CellKey::CENTER);
        }
        order.extend(self.capture_order());
        order.extend(self.reflected_cells.iter().copied());
        for key in order {
            if faces == 0 {
                break;
            }
            self.cells.get_mut(&key).expect("resident cell").take_faces(
                &mut faces,
                &mut self.serial,
                &mut jobs,
            );
        }
        jobs
    }

    /// Resident cells (not the centre, not reflected) with faces left to capture,
    /// most pleasant first: never-published probes before refreshes of published
    /// ones, then inside the view, periphery, behind; then nearer shells, then
    /// closeness to the view centre.
    fn capture_order(&self) -> Vec<CellKey> {
        let mut order: Vec<_> = self
            .resident_order
            .values()
            .filter_map(|&key| {
                let p = &self.cells[&key].progress;
                ((p.ready_faces | p.in_flight_faces) != ALL_FACES)
                    .then(|| ((p.published, self.capture_rank(key)), key))
            })
            .collect();
        order.sort_unstable();
        order.into_iter().map(|(_, key)| key).collect()
    }

    fn capture_rank(&self, key: CellKey) -> (u8, u16, u32) {
        if key.is_center() {
            return (0, 0, 0);
        }
        self.view_order.as_ref().map_or_else(
            || (1, key.shell, self.grid.ordinal(key)),
            |order| order.capture_rank(&self.grid, key),
        )
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

    /// Per-face light invalidation with the GPU reach test (see `take_reach_requests`).
    pub fn set_light_reach(&mut self, on: bool) {
        self.light_reach = on;
    }

    /// IDs of lights added, switched on, moved or reshaped since the last call: the caller
    /// runs the reach test over the ready faces and passes the faces it flags to
    /// `invalidate_faces` (or `invalidate_light_bounds` when it cannot test them).
    pub fn take_reach_requests(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.reach_requests)
    }

    /// The ready faces of every resident cell, by slot, to match a reach readback later.
    pub fn face_tokens(&self) -> Vec<FaceToken> {
        self.cells
            .values()
            .filter(|entry| entry.progress.ready_faces != 0)
            .map(|entry| FaceToken {
                key: entry.progress.key,
                slot: entry.progress.slot,
                epoch: entry.progress.epoch,
                ready: entry.progress.ready_faces,
                tickets: entry.tickets,
            })
            .collect()
    }

    /// Recapture the faces a reach test flagged, unless they were recaptured since the test.
    /// Returns the faces reset.
    pub fn invalidate_faces(&mut self, faces: &[(FaceToken, u8)]) -> u32 {
        let mut total = 0;
        for (token, bits) in faces {
            let Some(entry) = self.cells.get_mut(&token.key) else {
                continue;
            };
            if entry.progress.slot != token.slot || entry.progress.epoch != token.epoch {
                continue;
            }
            let mut current = 0;
            for face in 0..6 {
                let bit = 1 << face;
                if bits & token.ready & bit != 0 && entry.tickets[face] == token.tickets[face] {
                    current |= bit;
                }
            }
            let reset = entry.reset_face_bits(current);
            if reset != 0 {
                self.light_invalidated_cells += 1;
                self.light_invalidated_faces += u64::from(reset.count_ones());
                total += reset.count_ones();
            }
        }
        total
    }

    /// Conservative fallback of the reach test for lights it could not take: the faces whose
    /// own hit bounds the light may reach.
    pub(in crate::mesh) fn invalidate_light_bounds(
        &mut self,
        ids: &[u64],
        scene_bounds: Option<Bounds>,
    ) -> u32 {
        let mut total = 0;
        for &id in ids {
            let Some(light) = find_light_by_id(&self.local_lights, id) else {
                continue;
            };
            for entry in self.cells.values_mut() {
                let bits =
                    entry
                        .lighting
                        .faces_in_bounds(entry.progress.ready_faces, light, scene_bounds);
                let reset = entry.reset_face_bits(bits);
                if reset != 0 {
                    self.light_invalidated_cells += 1;
                    self.light_invalidated_faces += u64::from(reset.count_ones());
                    total += reset.count_ones();
                }
            }
        }
        total
    }

    /// 0 inside the view cone (and the centre probe), larger outside: re-shade order.
    pub fn view_rank(&self, key: CellKey) -> u8 {
        self.capture_rank(key).0
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
        stats.cached_cells = self.cache.len() as u32;
        stats.capacity_limited = self.resident_full() && !self.discovery.is_empty();
        stats.adopted_cells = self.adopted_cells;
        // A discovery range blocked by the capacity cutoff has nothing left to admit.
        stats.discovery_pending = (!self.discovery.is_empty()
            && self.before_capacity_cutoff(self.discovery.start))
            || !self.reprojected.is_empty()
            || self.dirty.is_some()
            || self.pending_dirty.is_some();
        for entry in self.cells.values() {
            let cell = entry.progress;
            if self.capture_rank(cell.key).0 == 0 {
                stats.visible_cells += 1;
                stats.visible_published_cells += u32::from(cell.published);
            }
            stats.lighting_metadata_ready_cells += u32::from(entry.lighting.is_known());
            stats.near_cells +=
                u32::from(cell.key.is_center() || cell.key.shell < self.config.near_shells);
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
            p.published |= p.ready_faces == ALL_FACES;
        }
        true
    }

    fn restart(&mut self) {
        self.clear_cache();
        for entry in std::mem::take(&mut self.cells).into_values() {
            self.free_slot(entry.progress.slot);
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
        let entry = self.detach(key);
        self.free_slot(entry.progress.slot);
    }

    fn detach(&mut self, key: CellKey) -> Entry {
        let entry = self.cells.remove(&key).expect("resident cell");
        if !key.is_center() && !self.reflected_cells.remove(&key) {
            self.resident_order.remove(&self.priority_rank(key));
        }
        entry
    }

    fn free_slot(&mut self, slot: u32) {
        self.free.insert(slot);
        self.released.insert(slot);
    }

    fn resident_full(&self) -> bool {
        self.cells.len() >= self.config.max_cells as usize
    }

    /// A free slot, or the slot of the least recently cached probe.
    fn take_slot(&mut self) -> u32 {
        if let Some(slot) = self.free.pop_first() {
            return slot;
        }
        let (_, entry) = self.cache.pop_first().expect("free or cached slot");
        self.released.insert(entry.progress.slot);
        entry.progress.slot
    }

    /// Keep a complete, published probe in the cache (its old stamp when it came from
    /// there), otherwise free its slot. The oldest cached probes leave first.
    fn stash(&mut self, entry: Entry, stamp: Option<u64>) {
        if self.config.cache_cells == 0 || !entry.progress.published || !entry.progress.is_ready() {
            self.free_slot(entry.progress.slot);
            return;
        }
        let stamp = stamp.unwrap_or_else(|| {
            self.cache_stamp += 1;
            self.cache_stamp
        });
        self.cache.insert(stamp, entry);
        while self.cache.len() > self.config.cache_cells as usize {
            let (_, old) = self.cache.pop_first().expect("cached probe");
            self.free_slot(old.progress.slot);
        }
    }

    fn clear_cache(&mut self) {
        for (_, entry) in std::mem::take(&mut self.cache) {
            self.free_slot(entry.progress.slot);
        }
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
                let (min, max) = self.cell_bounds(key);
                bounds.iter().any(|bounds| bounds.overlaps(min, max))
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

    fn cell_bounds(&self, key: CellKey) -> ([f64; 3], [f64; 3]) {
        if key.is_center() {
            let origin = self.grid.origin().map(f64::from);
            let (diameter, _) = self.grid.cell_extent(key);
            return (origin.map(|v| v - diameter), origin.map(|v| v + diameter));
        }
        let cell = self.grid.cell(key);
        (cell.min, cell.max)
    }

    /// New outer radius when the automatic range must change, see `auto_range`.
    fn auto_outer_radius(&self, geometry: &Geometry, origin: [f32; 3]) -> Option<f32> {
        if !self.config.auto_range || geometry.count() == 0 {
            return None;
        }
        let bounds = geometry.world_bounds()?;
        let mut far = 0.0f64;
        for corner in 0..8 {
            let point = [0, 1, 2].map(|axis| {
                let v = if corner & (1 << axis) != 0 {
                    bounds.max[axis]
                } else {
                    bounds.min[axis]
                };
                f64::from(v) - f64::from(origin[axis])
            });
            far = far.max(dot(point, point).sqrt());
        }
        let config = self.grid.config();
        let floor = 2.0 * f64::from(config.clip_start) + 0.5;
        let target = far.max(floor);
        if !target.is_finite() {
            return None;
        }
        let quantised = ((target.log2() * 4.0).ceil() / 4.0).exp2() as f32;
        let current = f64::from(config.z_max);
        (target > current || target * std::f64::consts::SQRT_2 < current).then_some(quantised)
    }

    /// Residency rank: shell-major in fixed world order, independent of the view.
    fn priority_rank(&self, key: CellKey) -> u32 {
        self.grid.ordinal(key)
    }

    fn priority_key(&self, rank: u32) -> CellKey {
        self.grid.key(rank)
    }

    fn before_capacity_cutoff(&self, rank: u32) -> bool {
        !self.resident_full()
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
            let columns = self.grid.cells_per_shell();
            let keys = [shells.start * columns..shells.end * columns, 0..0];
            self.dirty = Some(DirtyScan { bounds, keys });
        }
    }
}

/// Snap a recentred anchor to a cubic lattice so a camera that comes back reuses
/// the same cells. The snapped point is within sqrt(3)/2 steps of the camera, which
/// stays inside the recenter tolerance and so cannot retrigger immediately.
fn snap_origin(camera: [f32; 3], step: f32) -> [f32; 3] {
    if step <= 0.0 {
        return camera;
    }
    camera.map(|v| {
        let snapped = (f64::from(v) / f64::from(step)).round() * f64::from(step);
        snapped as f32
    })
}

fn invalid_config(message: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "spherical_gi.invalid_settings",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests;
