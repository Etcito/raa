//! Cluster DAG builder: the weld, the per-level partition, group, simplify and re-split loop,
//! and the packing of sibling sets into leaves and packed clusters.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/dag.rs`, revision 7e465b5.

use crate::bounds::Sphere;
use crate::group::GroupState;
use crate::input::BuildOptions;
use crate::partition::{ClusterLimits, partition_triangles};
use crate::simplify::{Corner, Simplified, Triangle, simplify};
use crate::thin::{NEW_POSITION, find_pieces, set_aside_wide_pieces, thin};
use crate::weld::WeldedMesh;
use raa_math::Vec3;

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

#[cfg(test)]
mod tests {
    use super::*;
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
}
