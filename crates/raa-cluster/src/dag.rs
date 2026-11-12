//! Cluster DAG builder: the weld, the per-level partition, group, simplify and re-split loop,
//! and the packing of sibling sets into leaves and packed clusters.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/dag.rs`, revision 7e465b5.

use crate::bounds::Sphere;
use crate::cut::ClusterMesh;
use crate::error::{ClusterError, INVALID_DATA, NOT_FINISHED, TOO_LARGE};
use crate::group::GroupState;
use crate::hierarchy::{Hierarchy, Leaf};
use crate::input::BuildOptions;
use crate::packed::{
    MAX_CLUSTER_TRIANGLES, MAX_CLUSTER_VERTICES, PackedCluster, PackedVertex, ROOT_PARENT_ERROR,
    pack_counts, pack_triangle,
};
use crate::partition::{ClusterLimits, partition_triangles};
use crate::simplify::{Corner, Simplified, Triangle, simplify};
use crate::thin::{NEW_POSITION, find_pieces, set_aside_wide_pieces, thin};
use crate::weld::{WeldedMesh, encode_octahedral};
use raa_math::Vec3;
use std::collections::BTreeMap;

/// A cluster during the build. Its id is its index in the append-only cluster list.
#[derive(Clone, Debug)]
pub struct BuildCluster {
    pub triangles: Vec<Triangle>,
    /// DAG level: 0 for source clusters.
    pub level: u32,
    pub bounds: Sphere,
    /// Sphere and error of the group this cluster was generated from.
    pub lod_bounds: Sphere,
    pub lod_error: f32,
    /// Whether some group has been simplified from this cluster.
    pub has_parent: bool,
}

/// A group of clusters that was successfully simplified; `children` are cluster ids.
#[derive(Clone, Debug)]
pub struct BuiltGroup {
    pub lod_bounds: Sphere,
    pub error: f32,
    pub children: Vec<u32>,
}

/// The reduction of one group of clusters, before its result is re-split.
#[derive(Clone, Debug)]
pub struct GroupResult {
    pub members: Vec<u32>,
    pub lod_bounds: Sphere,
    pub error: f32,
    /// A corner whose position is `NEW_POSITION | i` uses `new_positions[i]`.
    pub triangles: Vec<Triangle>,
    pub new_positions: Vec<Vec3>,
}

/// Bounds of the positions used by `triangles`.
pub fn triangle_bounds(triangles: &[Triangle], positions: &[Vec3]) -> Sphere {
    Sphere::from_points(
        triangles
            .iter()
            .flatten()
            .filter_map(|corner| positions.get(corner.position as usize).copied()),
    )
}

/// Turns one partition cluster of welded-triangle ids into level-0 clusters appended to `out`:
/// it is split until each cluster has at most `max_vertices` distinct corners.
pub fn push_initial_clusters(
    welded: &WeldedMesh,
    triangle_ids: &[u32],
    max_vertices: usize,
    out: &mut Vec<BuildCluster>,
) {
    let triangles: Vec<Triangle> = triangle_ids
        .iter()
        .filter_map(|&t| welded.triangles.get(t as usize).copied())
        .collect();
    let mut split = Vec::new();
    split_until_indexable(triangles, max_vertices, &mut split);
    for triangles in split {
        let bounds = triangle_bounds(&triangles, &welded.positions);
        out.push(BuildCluster {
            triangles,
            level: 0,
            bounds,
            lod_bounds: bounds,
            lod_error: 0.0,
            has_parent: false,
        });
    }
}

/// Partitions `triangles` (at most `PARTITION_BATCH` of them) and halves each cluster until it has
/// at most `limits.max_vertices` distinct corners.
pub fn split_into_clusters(
    triangles: &[Triangle],
    positions: &[Vec3],
    limits: ClusterLimits,
) -> Vec<Vec<Triangle>> {
    let by_position: Vec<[u32; 3]> = triangles.iter().map(|t| t.map(|c| c.position)).collect();
    let mut clusters = Vec::new();
    for members in partition_triangles(&by_position, positions, limits) {
        let cluster: Vec<Triangle> = members
            .iter()
            .filter_map(|&t| triangles.get(t as usize).copied())
            .collect();
        split_until_indexable(cluster, limits.max_vertices, &mut clusters);
    }
    clusters
}

/// A partition counts welded positions, but a cluster stores one vertex per distinct position and
/// attribute pair, so many UV seams or hard edges can exceed the 8-bit index limit. Halving keeps
/// neighbouring triangles together, because partitioning emits them in the order they grew. The
/// explicit work list keeps the recursion's order: a part is emitted before the parts after it.
/// A single triangle is never split.
fn split_until_indexable(
    cluster: Vec<Triangle>,
    max_vertices: usize,
    out: &mut Vec<Vec<Triangle>>,
) {
    let mut pending = vec![cluster];
    while let Some(mut part) = pending.pop() {
        if distinct_corners(&part) <= max_vertices || part.len() < 2 {
            out.push(part);
        } else {
            let second = part.split_off(part.len() / 2);
            pending.push(second);
            pending.push(part);
        }
    }
}

/// The number of distinct (position, attribute) corners of `triangles`.
fn distinct_corners(triangles: &[Triangle]) -> usize {
    let mut corners: Vec<Corner> = triangles.iter().flatten().copied().collect();
    corners.sort_unstable();
    corners.dedup();
    corners.len()
}

/// Reduces one group. Separate pieces are thinned; the rest is simplified to `target_ratio` of its
/// triangles with its shared positions locked, and is retried at 0.75 and 0.9 of its triangles
/// while its error is above the budget. `None` when nothing is left, when more than `min_reduction`
/// of the triangles remain, or when a member id names no cluster. A position id beyond the length
/// of `shared` is not shared.
pub fn reduce_group(
    welded: &WeldedMesh,
    clusters: &[BuildCluster],
    members: Vec<u32>,
    shared: &[bool],
    options: &BuildOptions,
) -> Option<GroupResult> {
    let is_shared = |v: u32| shared.get(v as usize).copied().unwrap_or(false);
    let member_clusters: Vec<&BuildCluster> = members
        .iter()
        .map(|&c| clusters.get(c as usize))
        .collect::<Option<_>>()?;
    let triangles: Vec<Triangle> = member_clusters
        .iter()
        .flat_map(|c| c.triangles.iter().copied())
        .collect();
    let target = |count: usize| ((count as f32 * options.target_ratio) as usize).max(1);

    // Separate pieces, such as blades and leaves, are thinned out. Connected surfaces are
    // simplified by edge collapse.
    let mut found = find_pieces(&triangles, &is_shared);
    set_aside_wide_pieces(&triangles, &mut found, &welded.positions);
    let mut new_positions = Vec::new();
    let mut thinning_error = 0.0f32;
    let mut reduced = Vec::with_capacity(triangles.len());
    if !found.pieces.is_empty() {
        let piece_triangles = found.pieces.iter().map(Vec::len).sum();
        let thinned = thin(
            &triangles,
            &found.pieces,
            &welded.positions,
            &welded.attributes,
            target(piece_triangles),
        );
        reduced.extend(thinned.triangles);
        new_positions = thinned.positions;
        thinning_error = thinned.error;
    }
    let rest: Vec<Triangle> = if found.pieces.is_empty() {
        triangles.clone()
    } else {
        found
            .rest
            .iter()
            .filter_map(|&t| triangles.get(t as usize).copied())
            .collect()
    };
    let mut simplified_error = 0.0f32;
    if !rest.is_empty() {
        let simplify_to = |count: usize| {
            simplify(
                &rest,
                &welded.positions,
                &welded.attributes,
                count,
                &is_shared,
            )
        };
        let mut simplified = simplify_to(target(rest.len()));
        // A group's error is the largest of its parts'. Where thinning carries the reduction, the
        // rest goes only as far as it can without being the coarser part, so it never holds the
        // twigs' detail back.
        if !found.pieces.is_empty() {
            let budget = thinning_error.max(welded.step);
            for ratio in [0.75, 0.9] {
                if simplified.error <= budget {
                    break;
                }
                simplified = simplify_to(((rest.len() as f32 * ratio) as usize).max(1));
            }
            if simplified.error > budget {
                simplified = Simplified {
                    triangles: rest.clone(),
                    error: 0.0,
                };
            }
        }
        reduced.extend(simplified.triangles);
        simplified_error = simplified.error;
    }

    if reduced.is_empty() || reduced.len() as f32 > triangles.len() as f32 * options.min_reduction {
        return None;
    }
    // Spheres and errors grow monotonically up the DAG, which makes the runtime LOD test a
    // consistent, hole-free cut. The small increment keeps parents strictly coarser than their
    // children even when a collapse on a flat region costs nothing.
    let lod_bounds = Sphere::merged_all(member_clusters.iter().map(|c| c.lod_bounds));
    let child_error = member_clusters
        .iter()
        .map(|c| c.lod_error)
        .fold(0.0, f32::max);
    let error = child_error + simplified_error.max(thinning_error) + welded.step * 0.5;
    Some(GroupResult {
        members,
        lod_bounds,
        error,
        triangles: reduced,
        new_positions,
    })
}

/// Most triangles and clusters one unit of a level visits.
const LEVEL_BATCH_TRIANGLES: usize = 4096;
const LEVEL_BATCH_CLUSTERS: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LevelPhase {
    Copy,
    Group,
    Marks,
    Reduce,
    Done,
}

/// Whether a unit that visited `visited` clusters and `triangles` triangles takes one more of
/// `next` triangles. A unit always takes at least one.
fn fits_unit(visited: usize, triangles: usize, next: usize) -> bool {
    visited == 0 || (visited < LEVEL_BATCH_CLUSTERS && triangles + next <= LEVEL_BATCH_TRIANGLES)
}

/// One level of the DAG as a state machine: the position triangles and bounds of the active
/// clusters are copied, grouped with [`GroupState`], the positions used by several groups are
/// marked, and then each group is reduced, re-split and appended, one group per unit.
pub struct LevelState {
    active: Vec<u32>,
    level: u32,
    phase: LevelPhase,
    cursor: usize,
    member: usize,
    cluster_triangles: Vec<Vec<[u32; 3]>>,
    cluster_bounds: Vec<Sphere>,
    grouping: Option<GroupState>,
    groups: Vec<Vec<u32>>,
    owner: Vec<u32>,
    shared: Vec<bool>,
    next_active: Vec<u32>,
    progressed: bool,
    progress: f32,
}

impl LevelState {
    /// A level over the clusters `active` (ids in the cluster list); its new clusters get `level`.
    pub fn new(active: Vec<u32>, level: u32) -> LevelState {
        LevelState {
            grouping: Some(GroupState::new(active.len())),
            active,
            level,
            phase: LevelPhase::Copy,
            cursor: 0,
            member: 0,
            cluster_triangles: Vec::new(),
            cluster_bounds: Vec::new(),
            groups: Vec::new(),
            owner: Vec::new(),
            shared: Vec::new(),
            next_active: Vec::new(),
            progressed: false,
            progress: 0.0,
        }
    }

    /// Runs one unit and returns `true` when every group was handled. A group that cannot be
    /// reduced keeps its clusters active; the others append their `BuiltGroup` and new clusters.
    /// The new positions of a thinned group are appended to `welded.positions` when it is handled.
    pub fn step(
        &mut self,
        welded: &mut WeldedMesh,
        clusters: &mut Vec<BuildCluster>,
        groups: &mut Vec<BuiltGroup>,
        limits: ClusterLimits,
        options: &BuildOptions,
    ) -> bool {
        let fraction = match self.phase {
            LevelPhase::Copy => {
                self.copy(clusters);
                0.1 * self.cursor as f32 / self.active.len().max(1) as f32
            }
            LevelPhase::Group => {
                self.group();
                0.1 + 0.3 * self.grouping.as_ref().map_or(1.0, GroupState::progress)
            }
            LevelPhase::Marks => {
                self.mark(welded.positions.len());
                0.4 + 0.1 * self.cursor as f32 / self.groups.len().max(1) as f32
            }
            LevelPhase::Reduce => {
                self.reduce(welded, clusters, groups, limits, options);
                0.5 + 0.5 * self.cursor as f32 / self.groups.len().max(1) as f32
            }
            LevelPhase::Done => 1.0,
        };
        self.progress = self.progress.max(if self.phase == LevelPhase::Done {
            1.0
        } else {
            fraction.min(0.999)
        });
        self.phase == LevelPhase::Done
    }

    /// The fraction of the level done, in `0..=1`; it never decreases and is 1 once done.
    pub fn progress(&self) -> f32 {
        self.progress
    }

    /// The clusters active at the next level; empty when no group was reduced.
    pub fn finish(self) -> Vec<u32> {
        if self.progressed {
            self.next_active
        } else {
            Vec::new()
        }
    }

    fn copy(&mut self, clusters: &[BuildCluster]) {
        let (mut visited, mut triangles) = (0, 0);
        while let Some(&id) = self.active.get(self.cursor) {
            let cluster = clusters.get(id as usize);
            let count = cluster.map_or(0, |c| c.triangles.len());
            if !fits_unit(visited, triangles, count) {
                return;
            }
            self.cluster_triangles
                .push(cluster.map_or_else(Vec::new, |c| {
                    let positions = c.triangles.iter();
                    positions.map(|t| t.map(|corner| corner.position)).collect()
                }));
            self.cluster_bounds
                .push(cluster.map_or_else(Sphere::default, |c| c.bounds));
            (visited, triangles, self.cursor) = (visited + 1, triangles + count, self.cursor + 1);
        }
        self.phase = LevelPhase::Group;
    }

    fn group(&mut self) {
        let done = self
            .grouping
            .as_mut()
            .is_none_or(|g| g.step(&self.cluster_triangles, &self.cluster_bounds));
        if done {
            self.groups = self
                .grouping
                .take()
                .map_or_else(Vec::new, GroupState::finish);
            (self.phase, self.cursor) = (LevelPhase::Marks, 0);
        }
    }

    /// A position used by two different groups is on a boundary and must not move.
    fn mark(&mut self, position_count: usize) {
        if self.owner.len() < position_count {
            self.owner = vec![u32::MAX; position_count];
            self.shared = vec![false; position_count];
        }
        let (mut visited, mut triangles) = (0, 0);
        while let Some(group) = self.groups.get(self.cursor) {
            let Some(&local) = group.get(self.member) else {
                (self.cursor, self.member) = (self.cursor + 1, 0);
                continue;
            };
            let own = self.cursor as u32;
            let members = self.cluster_triangles.get(local as usize);
            let members = members.map_or(&[][..], Vec::as_slice);
            if !fits_unit(visited, triangles, members.len()) {
                return;
            }
            for &position in members.iter().flatten() {
                let slot = self.owner.get_mut(position as usize);
                let flag = self.shared.get_mut(position as usize);
                if let (Some(slot), Some(flag)) = (slot, flag) {
                    if *slot == u32::MAX {
                        *slot = own;
                    } else if *slot != own {
                        *flag = true;
                    }
                }
            }
            (visited, triangles, self.member) =
                (visited + 1, triangles + members.len(), self.member + 1);
        }
        (self.phase, self.cursor, self.member) = (LevelPhase::Reduce, 0, 0);
    }

    fn reduce(
        &mut self,
        welded: &mut WeldedMesh,
        clusters: &mut Vec<BuildCluster>,
        groups: &mut Vec<BuiltGroup>,
        limits: ClusterLimits,
        options: &BuildOptions,
    ) {
        if let Some(group) = self.groups.get(self.cursor) {
            let members: Vec<u32> = group
                .iter()
                .filter_map(|&m| self.active.get(m as usize).copied())
                .collect();
            match reduce_group(welded, clusters, members.clone(), &self.shared, options) {
                None => self.next_active.extend(members),
                Some(result) => self.append(welded, clusters, groups, limits, result),
            }
            self.cursor += 1;
        }
        if self.cursor >= self.groups.len() {
            self.phase = LevelPhase::Done;
        }
    }

    /// Appends the new positions on the grid, re-splits the reduced triangles and records the
    /// group with its new clusters.
    fn append(
        &mut self,
        welded: &mut WeldedMesh,
        clusters: &mut Vec<BuildCluster>,
        groups: &mut Vec<BuiltGroup>,
        limits: ClusterLimits,
        mut result: GroupResult,
    ) {
        let base = welded.positions.len() as u32;
        for &p in &result.new_positions {
            let snapped = welded.snap(p);
            welded.positions.push(snapped);
        }
        for corner in result.triangles.iter_mut().flatten() {
            if corner.position & NEW_POSITION != 0 {
                corner.position = base + (corner.position & !NEW_POSITION);
            }
        }
        let split = split_into_clusters(&result.triangles, &welded.positions, limits);
        self.progressed = true;
        for &member in &result.members {
            if let Some(cluster) = clusters.get_mut(member as usize) {
                cluster.has_parent = true;
            }
        }
        groups.push(BuiltGroup {
            lod_bounds: result.lod_bounds,
            error: result.error,
            children: result.members,
        });
        for triangles in split {
            let bounds = triangle_bounds(&triangles, &welded.positions);
            self.next_active.push(clusters.len() as u32);
            clusters.push(BuildCluster {
                triangles,
                level: self.level,
                bounds,
                lod_bounds: result.lod_bounds,
                lod_error: result.error,
                has_parent: false,
            });
        }
    }
}

/// Whether a sets unit that has visited `visited` clusters takes `next` more; a unit always takes one set.
fn fits_sets(visited: usize, next: usize) -> bool {
    visited == 0 || visited + next <= LEVEL_BATCH_CLUSTERS
}

/// Most packed clusters: `cluster.too_large` beyond 2^24.
const MAX_PACKED_CLUSTERS: usize = 1 << 24;
/// Parentless clusters share one leaf per run of this many.
const ROOT_RUN: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PackPhase {
    Groups,
    Roots,
    Clusters,
    Done,
}

/// The packing of the sibling sets and clusters: the children of each group form one set, then
/// the parentless clusters form sets of [`ROOT_RUN`]. Each set is one leaf, and the clusters are
/// packed in set order, so a leaf's clusters are consecutive packed clusters.
pub struct PackState {
    source_triangles: u64,
    phase: PackPhase,
    group: usize,
    scan: usize,
    run: Vec<u32>,
    /// Build cluster ids in packed order: a packed index is a position here.
    order: Vec<u32>,
    leaves: Vec<Leaf>,
    cursor: usize,
    max_level: u32,
    vertices: Vec<PackedVertex>,
    triangles: Vec<u32>,
    records: Vec<PackedCluster>,
    progress: f32,
}

impl PackState {
    /// A pack of a mesh whose welded triangle count is `source_triangles`.
    pub fn new(source_triangles: u64) -> PackState {
        PackState {
            source_triangles,
            phase: PackPhase::Groups,
            group: 0,
            scan: 0,
            run: Vec::new(),
            order: Vec::new(),
            leaves: Vec::new(),
            cursor: 0,
            max_level: 0,
            vertices: Vec::new(),
            triangles: Vec::new(),
            records: Vec::new(),
            progress: 0.0,
        }
    }

    /// Runs one unit of sibling sets (at most 1024 clusters) and returns `true` when every set
    /// exists. The groups come first, then the parentless clusters in runs of [`ROOT_RUN`].
    pub fn step_sets(&mut self, clusters: &[BuildCluster], groups: &[BuiltGroup]) -> bool {
        match self.phase {
            PackPhase::Groups => self.groups_into_sets(clusters, groups),
            PackPhase::Roots => self.roots_into_sets(clusters),
            PackPhase::Clusters | PackPhase::Done => {}
        }
        let fraction = match self.phase {
            PackPhase::Groups => 0.25 * ratio(self.group, groups.len()),
            PackPhase::Roots => 0.25 + 0.25 * ratio(self.scan, clusters.len()),
            PackPhase::Clusters | PackPhase::Done => 0.5,
        };
        self.advance(fraction);
        matches!(self.phase, PackPhase::Clusters | PackPhase::Done)
    }

    /// The leaves made so far, one per sibling set in packed order.
    pub fn leaves(&self) -> &[Leaf] {
        &self.leaves
    }

    /// Packs the next units of clusters (at most 1024 clusters and 4096 triangles) in packed order
    /// and returns `true` when every cluster is packed. Refused until every set exists.
    pub fn step_clusters(
        &mut self,
        welded: &WeldedMesh,
        clusters: &[BuildCluster],
    ) -> Result<bool, ClusterError> {
        match self.phase {
            PackPhase::Groups | PackPhase::Roots => {
                return Err(ClusterError::new(
                    NOT_FINISHED,
                    "the sibling sets are not complete",
                ));
            }
            PackPhase::Clusters => {}
            PackPhase::Done => return Ok(true),
        }
        if self.order.len() > MAX_PACKED_CLUSTERS {
            return Err(ClusterError::new(TOO_LARGE, "more than 2^24 clusters"));
        }
        let (mut visited, mut triangles) = (0, 0);
        while let Some(&id) = self.order.get(self.cursor) {
            let cluster = clusters.get(id as usize).ok_or_else(|| {
                ClusterError::new(INVALID_DATA, "a sibling set names a missing cluster")
            })?;
            let count = cluster.triangles.len();
            if !fits_unit(visited, triangles, count) {
                break;
            }
            let record = pack_cluster(welded, cluster, &mut self.vertices, &mut self.triangles)?;
            self.records.push(record);
            self.max_level = self.max_level.max(cluster.level);
            (visited, triangles, self.cursor) = (visited + 1, triangles + count, self.cursor + 1);
        }
        if self.cursor >= self.order.len() {
            (self.phase, self.progress) = (PackPhase::Done, 1.0);
            return Ok(true);
        }
        self.advance(0.5 + 0.5 * ratio(self.cursor, self.order.len()));
        Ok(false)
    }

    /// The fraction packed, in `0..=1`; it never decreases and is 1 once done.
    pub fn progress(&self) -> f32 {
        self.progress
    }

    /// The finished mesh: the packed records and triangles, the hierarchy and the grid header of
    /// `welded`. The bounds are the merge of the leaf bounds.
    pub fn finish(self, welded: &WeldedMesh, hierarchy: Hierarchy) -> ClusterMesh {
        let bounds = Sphere::merged_all(self.leaves.iter().map(|leaf| leaf.bounds));
        ClusterMesh {
            vertices: self.vertices,
            triangles: self.triangles,
            clusters: self.records,
            nodes: hierarchy.nodes,
            root_node: hierarchy.root,
            hierarchy_depth: hierarchy.depth,
            levels: self.max_level.saturating_add(1),
            quant_origin: welded.origin.to_array(),
            quant_step: welded.step,
            uv_origin: welded.uv_origin,
            uv_step: welded.uv_step,
            bounds_center: bounds.center.to_array(),
            bounds_radius: bounds.radius,
            source_triangles: self.source_triangles,
        }
    }

    fn advance(&mut self, fraction: f32) {
        self.progress = self.progress.max(fraction.min(0.999));
    }

    fn groups_into_sets(&mut self, clusters: &[BuildCluster], groups: &[BuiltGroup]) {
        let mut visited = 0;
        while let Some(group) = groups.get(self.group) {
            if !fits_sets(visited, group.children.len()) {
                return;
            }
            self.push_set(clusters, &group.children, group.lod_bounds, group.error);
            (visited, self.group) = (visited + group.children.len(), self.group + 1);
        }
        (self.phase, self.scan) = (PackPhase::Roots, 0);
    }

    fn roots_into_sets(&mut self, clusters: &[BuildCluster]) {
        let mut visited = 0;
        while self.scan < clusters.len() && visited < LEVEL_BATCH_CLUSTERS {
            let id = self.scan;
            (self.scan, visited) = (self.scan + 1, visited + 1);
            if clusters.get(id).is_some_and(|c| !c.has_parent) {
                self.run.push(index_u32(id));
                if self.run.len() == ROOT_RUN {
                    self.flush_run(clusters);
                }
            }
        }
        if self.scan >= clusters.len() {
            self.flush_run(clusters);
            self.phase = PackPhase::Clusters;
        }
    }

    /// Turns the open run of parentless clusters into one set under `ROOT_PARENT_ERROR`.
    fn flush_run(&mut self, clusters: &[BuildCluster]) {
        let run = std::mem::take(&mut self.run);
        let lod_bounds = Sphere::merged_all(
            run.iter()
                .filter_map(|&c| clusters.get(c as usize))
                .map(|c| c.lod_bounds),
        );
        self.push_set(clusters, &run, lod_bounds, ROOT_PARENT_ERROR);
    }

    /// Appends a set of the members that name clusters: its leaf, and its members to the packed
    /// order. A set without members gives no leaf.
    fn push_set(
        &mut self,
        clusters: &[BuildCluster],
        members: &[u32],
        parent_bounds: Sphere,
        parent_error: f32,
    ) {
        let valid: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&m| clusters.get(m as usize).is_some())
            .collect();
        let Some(first) = valid.first().and_then(|&m| clusters.get(m as usize)) else {
            return;
        };
        let bounds = Sphere::merged_all(
            valid
                .iter()
                .filter_map(|&m| clusters.get(m as usize))
                .map(|c| c.bounds),
        );
        self.leaves.push(Leaf {
            level: first.level,
            first_cluster: index_u32(self.order.len()),
            cluster_count: index_u32(valid.len()),
            bounds,
            parent_lod_bounds: parent_bounds,
            parent_error,
        });
        self.order.extend(valid);
    }
}

/// `part` of `whole` as a fraction in `0..=1`; an empty whole counts as one.
fn ratio(part: usize, whole: usize) -> f32 {
    (part as f32 / whole.max(1) as f32).min(1.0)
}

fn index_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Appends the packed vertices and triangles of one cluster and returns its record. Vertices
/// take cluster-local indices in first-use order; a corner shared with another cluster packs to the
/// same vertex data.
fn pack_cluster(
    welded: &WeldedMesh,
    cluster: &BuildCluster,
    vertices: &mut Vec<PackedVertex>,
    triangles: &mut Vec<u32>,
) -> Result<PackedCluster, ClusterError> {
    if cluster.triangles.len() > MAX_CLUSTER_TRIANGLES {
        return Err(ClusterError::new(
            INVALID_DATA,
            "cluster has more triangles than MAX_CLUSTER_TRIANGLES",
        ));
    }
    let too_large = || ClusterError::new(TOO_LARGE, "packed offsets exceed u32");
    let vertex_offset = u32::try_from(vertices.len()).map_err(|_| too_large())?;
    let triangle_offset = u32::try_from(triangles.len()).map_err(|_| too_large())?;
    let mut local: BTreeMap<Corner, u8> = BTreeMap::new();
    for triangle in &cluster.triangles {
        let mut indices = [0u8; 3];
        for (index, &corner) in indices.iter_mut().zip(triangle) {
            *index = match local.get(&corner) {
                Some(&index) => index,
                None => {
                    let index = local_index(local.len())?;
                    vertices.push(pack_vertex(welded, corner)?);
                    local.insert(corner, index);
                    index
                }
            };
        }
        triangles.push(pack_triangle(indices[0], indices[1], indices[2]));
    }
    Ok(PackedCluster {
        center: cluster.bounds.center.to_array(),
        radius: cluster.bounds.radius,
        lod_center: cluster.lod_bounds.center.to_array(),
        lod_radius: cluster.lod_bounds.radius,
        lod_error: cluster.lod_error,
        vertex_offset,
        triangle_offset,
        counts: pack_counts(
            index_u32(local.len()),
            index_u32(cluster.triangles.len()),
            cluster.level,
        ),
    })
}

/// The 8-bit local index of the next vertex, refused beyond `MAX_CLUSTER_VERTICES`.
fn local_index(count: usize) -> Result<u8, ClusterError> {
    match u8::try_from(count) {
        Ok(index) if count < MAX_CLUSTER_VERTICES => Ok(index),
        _ => Err(ClusterError::new(
            INVALID_DATA,
            "cluster exceeds 8-bit vertex indexing",
        )),
    }
}

/// One packed vertex: the grid position, the octahedral normal, and the uv grid index (scene.rs).
fn pack_vertex(welded: &WeldedMesh, corner: Corner) -> Result<PackedVertex, ClusterError> {
    let missing = || ClusterError::new(INVALID_DATA, "a corner names a missing welded element");
    let position = *welded
        .positions
        .get(corner.position as usize)
        .ok_or_else(missing)?;
    let attribute = welded
        .attributes
        .get(corner.attribute as usize)
        .ok_or_else(missing)?;
    let grid = welded.grid(position);
    let normal = encode_octahedral(attribute.normal);
    let uv = |axis: usize| {
        let v = (attribute.uv[axis] - welded.uv_origin[axis]) / welded.uv_step[axis];
        v.round().clamp(0.0, 65535.0) as u16
    };
    Ok(PackedVertex {
        data: [
            grid[0],
            grid[1],
            grid[2],
            normal[0] as u16,
            normal[1] as u16,
            uv(0),
            uv(1),
            0,
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::build_hierarchy;
    use crate::input::OwnedClusterInput;
    use crate::weld::WeldState;
    use raa_geom::MeshData;

    const LIMITS: ClusterLimits = ClusterLimits {
        max_triangles: 128,
        max_vertices: 255,
    };

    /// A flat `cells` x `cells` grid of quads on the XZ plane, two triangles each.
    fn grid(cells: u32) -> OwnedClusterInput {
        rect(cells, cells)
    }

    /// A flat `nx` x `nz` grid of quads on the XZ plane, two triangles each.
    fn rect(nx: u32, nz: u32) -> OwnedClusterInput {
        let side = nx + 1;
        let positions: Vec<[f32; 3]> = (0..=nz)
            .flat_map(|z| (0..side).map(move |x| [x as f32, 0.0, z as f32]))
            .collect();
        let mut indices = Vec::new();
        for z in 0..nz {
            for x in 0..nx {
                let a = z * side + x;
                let (b, c, d) = (a + 1, a + side, a + side + 1);
                indices.extend([a, b, d, a, d, c]);
            }
        }
        owned(positions, indices)
    }

    fn owned(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> OwnedClusterInput {
        let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
        OwnedClusterInput {
            mesh: MeshData {
                positions,
                normals,
                indices,
            },
            uvs: None,
        }
    }

    fn welded(input: &OwnedClusterInput) -> WeldedMesh {
        let mut state = WeldState::new(&input.bounds(), 0.0);
        while !state.step(input) {}
        state.finish()
    }

    /// The initial clusters of every welded triangle, partitioned as the bake does.
    fn initial(welded: &WeldedMesh) -> Vec<BuildCluster> {
        let by_position: Vec<[u32; 3]> = welded
            .triangles
            .iter()
            .map(|t| t.map(|c| c.position))
            .collect();
        let mut out = Vec::new();
        for ids in partition_triangles(&by_position, &welded.positions, LIMITS) {
            push_initial_clusters(welded, &ids, LIMITS.max_vertices, &mut out);
        }
        out
    }

    #[test]
    fn resplit_returns_every_triangle_once_in_small_clusters() {
        let mesh = welded(&grid(39));
        let input = &mesh.triangles[..3000];
        let clusters = split_into_clusters(input, &mesh.positions, LIMITS);
        assert!(clusters.iter().all(|c| c.len() <= 128));
        let mut got: Vec<Triangle> = clusters.concat();
        got.sort();
        let mut want = input.to_vec();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn distinct_attributes_are_halved_until_each_part_is_indexable() {
        let mesh = welded(&grid(10));
        let mut triangles = mesh.triangles.clone();
        for (next, corner) in triangles.iter_mut().flatten().enumerate() {
            corner.attribute = next as u32;
        }
        let clusters = split_into_clusters(&triangles, &mesh.positions, LIMITS);
        assert!(clusters.iter().all(|c| distinct_corners(c) <= 255));
        assert!(clusters.len() >= 3);
        assert_eq!(
            clusters.iter().map(Vec::len).sum::<usize>(),
            triangles.len()
        );
    }

    #[test]
    fn initial_clusters_are_level_zero_with_their_own_bounds() {
        let mesh = welded(&grid(10));
        let clusters = initial(&mesh);
        assert!(!clusters.is_empty());
        for c in &clusters {
            assert_eq!((c.level, c.lod_error, c.has_parent), (0, 0.0, false));
            assert_eq!(c.lod_bounds, c.bounds);
        }
        let total: usize = clusters.iter().map(|c| c.triangles.len()).sum();
        assert_eq!(total, mesh.triangles.len());
    }

    #[test]
    fn flat_grid_group_reduces_with_a_grown_error_and_bounds_its_members() {
        let mesh = welded(&grid(12));
        let clusters = initial(&mesh);
        let members: Vec<u32> = (0..clusters.len() as u32).collect();
        let shared = vec![false; mesh.positions.len()];
        let result = reduce_group(
            &mesh,
            &clusters,
            members.clone(),
            &shared,
            &BuildOptions::default(),
        )
        .expect("a flat grid group reduces");
        assert!(result.triangles.len() as f32 <= mesh.triangles.len() as f32 * 0.85);
        let child = clusters.iter().map(|c| c.lod_error).fold(0.0, f32::max);
        assert!(result.error >= child + 0.5 * mesh.step);
        for &m in &members {
            assert!(result.lod_bounds.contains(clusters[m as usize].lod_bounds));
        }
    }

    #[test]
    fn group_with_every_position_shared_is_not_reduced() {
        let mesh = welded(&grid(12));
        let clusters = initial(&mesh);
        let members: Vec<u32> = (0..clusters.len() as u32).collect();
        let shared = vec![true; mesh.positions.len()];
        let options = BuildOptions::default();
        assert!(reduce_group(&mesh, &clusters, members.clone(), &shared, &options).is_none());
        // Position ids beyond `shared` are not shared, so the same group reduces.
        assert!(reduce_group(&mesh, &clusters, members, &[], &options).is_some());
    }

    #[test]
    fn group_of_separate_quads_returns_new_positions() {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for q in 0..40u32 {
            let x = 2.0 * q as f32;
            let base = 4 * q;
            positions.extend([
                [x, 0.0, 0.0],
                [x + 1.0, 0.0, 0.0],
                [x + 1.0, 0.0, 1.0],
                [x, 0.0, 1.0],
            ]);
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let mesh = welded(&owned(positions, indices));
        let clusters = initial(&mesh);
        let members: Vec<u32> = (0..clusters.len() as u32).collect();
        let shared = vec![false; mesh.positions.len()];
        let result = reduce_group(&mesh, &clusters, members, &shared, &BuildOptions::default())
            .expect("separate quads thin");
        assert!(!result.new_positions.is_empty());
        assert!(
            result
                .triangles
                .iter()
                .flatten()
                .any(|c| c.position & NEW_POSITION != 0)
        );
    }

    /// Everything one level leaves behind.
    struct Level {
        mesh: WeldedMesh,
        clusters: Vec<BuildCluster>,
        groups: Vec<BuiltGroup>,
        next: Vec<u32>,
        progress: Vec<f32>,
    }

    /// Runs a level over every level-0 cluster of a welded `nx` x `nz` grid, one unit at a time.
    fn run_level(nx: u32, nz: u32, options: &BuildOptions) -> Level {
        let mut mesh = welded(&rect(nx, nz));
        let mut clusters = initial(&mesh);
        let mut groups = Vec::new();
        let mut state = LevelState::new((0..clusters.len() as u32).collect(), 1);
        let mut progress = vec![state.progress()];
        while !state.step(&mut mesh, &mut clusters, &mut groups, LIMITS, options) {
            progress.push(state.progress());
        }
        progress.push(state.progress());
        let next = state.finish();
        Level {
            mesh,
            clusters,
            groups,
            next,
            progress,
        }
    }

    #[test]
    fn level_reduces_a_flat_grid_and_keeps_its_border() {
        let level = run_level(80, 40, &BuildOptions::default());
        let initial = level.clusters.iter().filter(|c| c.level == 0).count();
        assert!(!level.groups.is_empty());
        assert!(level.next.len() < initial);
        for group in &level.groups {
            for &child in &group.children {
                let child = &level.clusters[child as usize];
                assert!(child.has_parent && group.error >= child.lod_error);
            }
        }
        for cluster in level.clusters.iter().skip(initial) {
            assert!(cluster.level == 1 && cluster.lod_error > 0.0);
            assert!(
                level.groups.iter().any(|g| {
                    g.lod_bounds == cluster.lod_bounds && g.error == cluster.lod_error
                })
            );
        }
        let used: std::collections::BTreeSet<u32> = level
            .next
            .iter()
            .flat_map(|&c| level.clusters[c as usize].triangles.iter().flatten())
            .map(|corner| corner.position)
            .collect();
        let on_border = |p: Vec3| {
            let edge = |v: f32, far: f32| v.abs() < 0.01 || (v - far).abs() < 0.01;
            edge(p.x, 80.0) || edge(p.z, 40.0)
        };
        let border: Vec<u32> = (0..level.mesh.positions.len() as u32)
            .filter(|&p| on_border(level.mesh.positions[p as usize]))
            .collect();
        assert!(border.len() >= 240);
        assert!(border.iter().all(|p| used.contains(p)));
    }

    #[test]
    fn level_progress_never_decreases_and_reaches_one() {
        let level = run_level(80, 40, &BuildOptions::default());
        assert!(level.progress.windows(2).all(|w| w[0] <= w[1]));
        assert!(level.progress.iter().all(|p| (0.0..=1.0).contains(p)));
        assert_eq!(level.progress.last(), Some(&1.0));
        assert!(level.progress.len() > 4);
    }

    #[test]
    fn level_steps_equal_one_loop() {
        let stepped = run_level(40, 20, &BuildOptions::default());
        let mut mesh = welded(&rect(40, 20));
        let mut clusters = initial(&mesh);
        let mut groups = Vec::new();
        let mut state = LevelState::new((0..clusters.len() as u32).collect(), 1);
        let options = BuildOptions::default();
        while !state.step(&mut mesh, &mut clusters, &mut groups, LIMITS, &options) {}
        assert_eq!(state.finish(), stepped.next);
        assert_eq!(mesh.positions, stepped.mesh.positions);
        assert_eq!(clusters.len(), stepped.clusters.len());
        for (a, b) in clusters.iter().zip(&stepped.clusters) {
            assert_eq!((&a.triangles, a.level), (&b.triangles, b.level));
            assert_eq!((a.bounds, a.lod_bounds), (b.bounds, b.lod_bounds));
            assert_eq!((a.lod_error, a.has_parent), (b.lod_error, b.has_parent));
        }
    }

    #[test]
    fn level_with_every_group_refused_finishes_empty() {
        let options = BuildOptions {
            min_reduction: 0.01,
            ..BuildOptions::default()
        };
        let level = run_level(40, 20, &options);
        assert!(level.next.is_empty() && level.groups.is_empty());
        assert!(level.clusters.iter().all(|c| !c.has_parent && c.level == 0));
    }

    /// A packed level: the mesh from a copy of the leaves, the packed order and the leaves.
    struct Packed {
        mesh: ClusterMesh,
        leaves: Vec<Leaf>,
        order: Vec<u32>,
        units: usize,
        progress: Vec<f32>,
    }

    /// Packs `level` one unit per call, then finishes with the hierarchy of a copy of the leaves.
    fn pack(level: &Level, source: u64) -> Packed {
        let mut state = PackState::new(source);
        let mut progress = vec![state.progress()];
        let mut units = 0;
        while !state.step_sets(&level.clusters, &level.groups) {
            units += 1;
            progress.push(state.progress());
        }
        while !state
            .step_clusters(&level.mesh, &level.clusters)
            .expect("sets exist")
        {
            units += 1;
            progress.push(state.progress());
        }
        progress.push(state.progress());
        let leaves = state.leaves().to_vec();
        let order = state.order.clone();
        let mesh = state.finish(&level.mesh, build_hierarchy(leaves.clone()));
        Packed {
            mesh,
            leaves,
            order,
            units,
            progress,
        }
    }

    #[test]
    fn packed_mesh_validates_and_keeps_the_given_source_count() {
        let level = run_level(80, 40, &BuildOptions::default());
        let packed = pack(&level, 12345);
        assert_eq!(packed.mesh.validate(), Ok(()));
        assert_eq!(packed.mesh.source_triangles, 12345);
        assert_eq!(packed.mesh.clusters.len(), packed.order.len());
        assert!(packed.units >= 2);
        assert!(packed.progress.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(packed.progress.last(), Some(&1.0));
    }

    #[test]
    fn group_children_are_consecutive_under_a_leaf_with_the_group_error() {
        let level = run_level(40, 20, &BuildOptions::default());
        let packed = pack(&level, 0);
        assert!(!level.groups.is_empty());
        let mut first = 0u32;
        for (group, leaf) in level.groups.iter().zip(&packed.leaves) {
            assert_eq!(
                (leaf.first_cluster, leaf.cluster_count),
                (first, group.children.len() as u32)
            );
            assert_eq!(leaf.parent_error, group.error);
            for (i, &child) in group.children.iter().enumerate() {
                assert_eq!(packed.order[first as usize + i], child);
            }
            first += leaf.cluster_count;
            assert!(packed.mesh.nodes.iter().any(|n| {
                n.is_leaf == 1
                    && n.first == leaf.first_cluster
                    && n.count == leaf.cluster_count
                    && n.max_parent_error == group.error
            }));
        }
    }

    #[test]
    fn parentless_clusters_sit_in_runs_under_root_parent_error() {
        let level = run_level(40, 20, &BuildOptions::default());
        let packed = pack(&level, 0);
        let roots: Vec<u32> = (0..level.clusters.len() as u32)
            .filter(|&c| !level.clusters[c as usize].has_parent)
            .collect();
        assert!(!roots.is_empty());
        let root_leaves: Vec<&Leaf> = packed
            .leaves
            .iter()
            .filter(|l| l.parent_error == ROOT_PARENT_ERROR)
            .collect();
        assert_eq!(root_leaves.len(), roots.len().div_ceil(ROOT_RUN));
        assert!(
            root_leaves
                .iter()
                .all(|l| l.cluster_count as usize <= ROOT_RUN)
        );
        let mut packed_roots: Vec<u32> = root_leaves
            .iter()
            .flat_map(|l| l.first_cluster..l.first_cluster + l.cluster_count)
            .map(|p| packed.order[p as usize])
            .collect();
        packed_roots.sort_unstable();
        assert_eq!(packed_roots, roots);
    }

    #[test]
    fn a_vertex_shared_by_two_clusters_packs_to_the_same_eight_words() {
        let level = run_level(40, 20, &BuildOptions::default());
        let packed = pack(&level, 0);
        let mut seen: BTreeMap<Corner, [u16; 8]> = BTreeMap::new();
        let mut shared = 0;
        for (p, &id) in packed.order.iter().enumerate() {
            let record = packed.mesh.clusters[p];
            let mut local: Vec<Corner> = Vec::new();
            for corner in level.clusters[id as usize].triangles.iter().flatten() {
                if !local.contains(corner) {
                    local.push(*corner);
                }
            }
            for (i, corner) in local.iter().enumerate() {
                let data = packed.mesh.vertices[record.vertex_offset as usize + i].data;
                if let Some(old) = seen.insert(*corner, data) {
                    shared += 1;
                    assert_eq!(old, data);
                }
            }
        }
        assert!(shared > 0);
    }

    #[test]
    fn levels_is_two_and_level_one_clusters_carry_level_one_in_counts() {
        let level = run_level(40, 20, &BuildOptions::default());
        let packed = pack(&level, 0);
        assert_eq!(packed.mesh.levels, 2);
        let mut level_one = 0;
        for (p, &id) in packed.order.iter().enumerate() {
            let built = level.clusters[id as usize].level;
            assert_eq!(packed.mesh.clusters[p].level(), built);
            level_one += usize::from(built == 1);
        }
        assert!(level_one > 0);
    }

    #[test]
    fn packing_before_the_sets_is_refused() {
        let level = run_level(8, 4, &BuildOptions::default());
        let mut state = PackState::new(0);
        let code = state
            .step_clusters(&level.mesh, &level.clusters)
            .map_err(|e| e.code());
        assert_eq!(code, Err(NOT_FINISHED));
    }

    #[test]
    fn stepping_one_unit_at_a_time_equals_stepping_to_the_end() {
        let level = run_level(80, 40, &BuildOptions::default());
        let stepped = pack(&level, 7);
        let mut state = PackState::new(7);
        while !state.step_sets(&level.clusters, &level.groups) {}
        while !state
            .step_clusters(&level.mesh, &level.clusters)
            .expect("sets exist")
        {}
        let leaves = state.leaves().to_vec();
        let mesh = state.finish(&level.mesh, build_hierarchy(leaves));
        assert_eq!(mesh, stepped.mesh);
    }
}
