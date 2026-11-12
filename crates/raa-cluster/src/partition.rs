//! Partition of the welded triangles into compact clusters: limits, Morton order, adjacency,
//! cluster growth and the chunked partition state that the sliced bake advances.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/partition.rs`, revision 7e465b5.

use std::collections::BTreeMap;

use raa_math::Vec3;

/// Size limits for one cluster.
#[derive(Clone, Copy, Debug)]
pub struct ClusterLimits {
    pub max_triangles: usize,
    pub max_vertices: usize,
}

/// Clusters smaller than this try to merge into a neighbour.
pub const SMALL_CLUSTER: usize = 64;
/// Rings of grid cells searched for a nearby separate piece before giving up.
pub const MAX_SEARCH_RINGS: i32 = 6;

/// Inserts two zero bits between each of the low 10 bits of `x`.
pub fn spread_bits(x: u32) -> u32 {
    let x = x & 0x3ff;
    let x = (x | x << 16) & 0x030000ff;
    let x = (x | x << 8) & 0x0300f00f;
    let x = (x | x << 4) & 0x030c30c3;
    (x | x << 2) & 0x09249249
}

/// Indices of `points` sorted along a Morton curve; equal keys keep index order.
pub fn morton_order(points: &[Vec3]) -> Vec<u32> {
    let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for &p in points {
        min = min.min(p);
        max = max.max(p);
    }
    let extent = max - min;
    let largest = extent.x.max(extent.y).max(extent.z).max(1.0e-12);
    let scale = 1023.0 / largest;
    let mut keyed: Vec<(u32, u32)> = points
        .iter()
        .enumerate()
        .map(|(index, &p)| {
            let q = (p - min) * scale;
            let key = spread_bits(q.x as u32)
                | spread_bits(q.y as u32) << 1
                | spread_bits(q.z as u32) << 2;
            (key, index as u32)
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, index)| index).collect()
}

/// For each triangle, up to three edge-adjacent triangles (`u32::MAX` if none).
pub fn triangle_adjacency(triangles: &[[u32; 3]]) -> Vec<[u32; 3]> {
    let mut edges: Vec<(u64, u32, u8)> = Vec::with_capacity(triangles.len() * 3);
    for (index, triangle) in triangles.iter().enumerate() {
        for side in 0..3 {
            let a = triangle[side];
            let b = triangle[(side + 1) % 3];
            let key = u64::from(a.min(b)) << 32 | u64::from(a.max(b));
            edges.push((key, index as u32, side as u8));
        }
    }
    edges.sort_unstable();

    let mut adjacency = vec![[u32::MAX; 3]; triangles.len()];
    for run in edges.chunk_by(|a, b| a.0 == b.0) {
        // Non-manifold edges link only their first two users; that is enough for
        // growth to continue, and the rest are reached via other edges or seeds.
        if let [(_, first, first_side), (_, second, second_side), ..] = run {
            if let Some(sides) = adjacency.get_mut(*first as usize)
                && let Some(slot) = sides.get_mut(*first_side as usize)
            {
                *slot = *second;
            }
            if let Some(sides) = adjacency.get_mut(*second as usize)
                && let Some(slot) = sides.get_mut(*second_side as usize)
            {
                *slot = *first;
            }
        }
    }
    adjacency
}

/// Triangles re-indexed to a compact vertex range, so scratch arrays stay small
/// even when partitioning a little piece of a huge mesh.
pub struct LocalTriangles {
    pub triangles: Vec<[u32; 3]>,
    pub centroids: Vec<Vec3>,
    pub vertex_count: usize,
}

impl LocalTriangles {
    /// Re-indexes `triangles` in first-use order and computes each triangle centroid.
    pub fn new(triangles: &[[u32; 3]], positions: &[Vec3]) -> LocalTriangles {
        let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
        let mut local = Vec::with_capacity(triangles.len());
        let mut centroids = Vec::with_capacity(triangles.len());
        for triangle in triangles {
            let mut mapped = [0u32; 3];
            let mut sum = Vec3::ZERO;
            for (slot, &vertex) in triangle.iter().enumerate() {
                let next = remap.len() as u32;
                mapped[slot] = *remap.entry(vertex).or_insert(next);
                sum += positions
                    .get(vertex as usize)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
            }
            local.push(mapped);
            centroids.push(sum / 3.0);
        }
        LocalTriangles {
            triangles: local,
            centroids,
            vertex_count: remap.len(),
        }
    }
}

/// Triangle centroids bucketed on a uniform grid, for finding the nearest separate piece.
pub struct PieceGrid {
    origin: Vec3,
    cell: f32,
    cells: BTreeMap<[i32; 3], Vec<u32>>,
}

impl PieceGrid {
    /// Buckets `centroids` on a grid with about eight triangles per occupied cell.
    pub fn new(centroids: &[Vec3]) -> PieceGrid {
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &centroid in centroids {
            min = min.min(centroid);
            max = max.max(centroid);
        }
        let extent = (max - min).max(Vec3::splat(1.0e-6));
        let area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
        let cell = (area / (centroids.len() as f32 / 8.0).max(1.0))
            .sqrt()
            .max(1.0e-6);
        let mut grid = PieceGrid {
            origin: min,
            cell,
            cells: BTreeMap::new(),
        };
        for (index, &centroid) in centroids.iter().enumerate() {
            grid.cells
                .entry(grid.key(centroid))
                .or_default()
                .push(index as u32);
        }
        grid
    }

    fn key(&self, p: Vec3) -> [i32; 3] {
        let q = (p - self.origin) / self.cell;
        [q.x.floor() as i32, q.y.floor() as i32, q.z.floor() as i32]
    }

    /// The unassigned triangle nearest `center` that fits in the cluster; among equal
    /// distances the lowest triangle index wins.
    #[allow(clippy::too_many_arguments)]
    pub fn nearest(
        &self,
        local: &LocalTriangles,
        assigned: &[bool],
        vertex_stamp: &[u32],
        cluster_id: u32,
        vertex_count: usize,
        center: Vec3,
        limits: ClusterLimits,
    ) -> Option<u32> {
        let home = self.key(center);
        let mut best: Option<(f32, u32)> = None;
        for ring in 0..=MAX_SEARCH_RINGS {
            // Anything in this ring or beyond is at least this far away.
            let floor = (ring - 1).max(0) as f32 * self.cell;
            if best.is_some_and(|(distance, _)| distance < floor * floor) {
                break;
            }
            for dz in -ring..=ring {
                for dy in -ring..=ring {
                    for dx in -ring..=ring {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != ring {
                            continue;
                        }
                        let key = [home[0] + dx, home[1] + dy, home[2] + dz];
                        let Some(members) = self.cells.get(&key) else {
                            continue;
                        };
                        for &candidate in members {
                            let index = candidate as usize;
                            if assigned.get(index).copied().unwrap_or(true) {
                                continue;
                            }
                            let (Some(&triangle), Some(&centroid)) =
                                (local.triangles.get(index), local.centroids.get(index))
                            else {
                                continue;
                            };
                            let new_vertices = triangle
                                .iter()
                                .filter(|&&vertex| {
                                    vertex_stamp.get(vertex as usize) != Some(&cluster_id)
                                })
                                .count();
                            if vertex_count + new_vertices > limits.max_vertices {
                                continue;
                            }
                            let distance = (centroid - center).length_squared();
                            let better = best.is_none_or(|(best_distance, best_candidate)| {
                                distance < best_distance
                                    || (distance == best_distance && candidate < best_candidate)
                            });
                            if better {
                                best = Some((distance, candidate));
                            }
                        }
                    }
                }
            }
        }
        best.map(|(_, candidate)| candidate)
    }
}

/// Triangles partitioned per growth batch; a larger mesh is cut along the Morton order
/// of its triangle centroids into chunks of at most this many triangles.
pub const PARTITION_BATCH: usize = 4096;

/// Partitions `triangles` (indices into `positions`) and returns the triangle indices of each
/// cluster in growth order. Every triangle appears in exactly one cluster within `limits`.
pub fn partition_triangles(
    triangles: &[[u32; 3]],
    positions: &[Vec3],
    limits: ClusterLimits,
) -> Vec<Vec<u32>> {
    if triangles.is_empty() {
        return Vec::new();
    }
    let local = LocalTriangles::new(triangles, positions);
    let adjacency = triangle_adjacency(&local.triangles);
    let clusters = grow_clusters(&local, &adjacency, limits);
    merge_small_clusters(clusters, &local, &adjacency, limits)
}

/// Grows one compact cluster per seed, taking frontier triangles by score and handing
/// over to the nearest separate piece when the frontier is empty.
fn grow_clusters(
    local: &LocalTriangles,
    adjacency: &[[u32; 3]],
    limits: ClusterLimits,
) -> Vec<Vec<u32>> {
    let seeds = morton_order(&local.centroids);
    let mut assigned = vec![false; local.triangles.len()];
    // `vertex_stamp[v] == current cluster id` marks a vertex as used by the cluster.
    let mut vertex_stamp = vec![u32::MAX; local.vertex_count];
    let mut clusters = Vec::new();
    let mut frontier: Vec<u32> = Vec::new();

    let pieces = PieceGrid::new(&local.centroids);
    for &seed in &seeds {
        if assigned.get(seed as usize).copied().unwrap_or(true) {
            continue;
        }
        let cluster_id = clusters.len() as u32;
        let mut cluster = Vec::with_capacity(limits.max_triangles);
        let mut vertex_count = 0usize;
        let mut center_sum = Vec3::ZERO;
        frontier.clear();

        let mut next = Some(seed);
        while let Some(triangle) = next {
            let index = triangle as usize;
            let Some(&local_triangle) = local.triangles.get(index) else {
                break;
            };
            assigned[index] = true;
            cluster.push(triangle);
            center_sum += local.centroids.get(index).copied().unwrap_or(Vec3::ZERO);
            for &vertex in &local_triangle {
                if let Some(stamp) = vertex_stamp.get_mut(vertex as usize)
                    && *stamp != cluster_id
                {
                    *stamp = cluster_id;
                    vertex_count += 1;
                }
            }
            for &neighbor in adjacency.get(index).into_iter().flatten() {
                if neighbor != u32::MAX && !assigned.get(neighbor as usize).copied().unwrap_or(true)
                {
                    frontier.push(neighbor);
                }
            }
            if cluster.len() >= limits.max_triangles {
                break;
            }

            // Pick the frontier triangle that best keeps the cluster compact.
            let center = center_sum / cluster.len() as f32;
            let mut best: Option<(f32, usize)> = None;
            let mut slot = 0;
            while slot < frontier.len() {
                let candidate = frontier[slot] as usize;
                if assigned.get(candidate).copied().unwrap_or(true) {
                    frontier.swap_remove(slot);
                    continue;
                }
                let Some(&candidate_triangle) = local.triangles.get(candidate) else {
                    frontier.swap_remove(slot);
                    continue;
                };
                let new_vertices = candidate_triangle
                    .iter()
                    .filter(|&&vertex| vertex_stamp.get(vertex as usize) != Some(&cluster_id))
                    .count();
                if vertex_count + new_vertices <= limits.max_vertices {
                    // Taking triangles with few free neighbours first fills dead ends
                    // that would otherwise be stranded as tiny clusters later.
                    let free_neighbors = adjacency
                        .get(candidate)
                        .map(|sides| {
                            sides
                                .iter()
                                .filter(|&&neighbor| {
                                    neighbor != u32::MAX
                                        && !assigned.get(neighbor as usize).copied().unwrap_or(true)
                                })
                                .count()
                        })
                        .unwrap_or(0);
                    let distance = local
                        .centroids
                        .get(candidate)
                        .map(|&centroid| (centroid - center).length_squared())
                        .unwrap_or(f32::MAX);
                    let score = distance
                        * (1.0 + new_vertices as f32 * 0.5)
                        * (1.0 + free_neighbors as f32 * 0.5);
                    if best.is_none_or(|(best_score, _)| score < best_score) {
                        best = Some((score, slot));
                    }
                }
                slot += 1;
            }
            next = match best {
                Some((_, slot)) => frontier.get(slot).copied(),
                None => pieces.nearest(
                    local,
                    &assigned,
                    &vertex_stamp,
                    cluster_id,
                    vertex_count,
                    center,
                    limits,
                ),
            };
        }
        if !cluster.is_empty() {
            clusters.push(cluster);
        }
    }
    clusters
}

/// Merges clusters below `SMALL_CLUSTER` into their fullest neighbour that still fits.
fn merge_small_clusters(
    mut clusters: Vec<Vec<u32>>,
    local: &LocalTriangles,
    adjacency: &[[u32; 3]],
    limits: ClusterLimits,
) -> Vec<Vec<u32>> {
    let mut owner = vec![0u32; local.triangles.len()];
    for (index, cluster) in clusters.iter().enumerate() {
        for &triangle in cluster {
            if let Some(slot) = owner.get_mut(triangle as usize) {
                *slot = index as u32;
            }
        }
    }

    // Smallest first, so tiny fragments claim the free space before larger ones.
    let mut order: Vec<usize> = (0..clusters.len()).collect();
    order.sort_by_key(|&index| clusters.get(index).map_or(0, Vec::len));

    for index in order {
        let small = clusters.get(index).map_or(0, Vec::len);
        if small == 0 || small >= SMALL_CLUSTER {
            continue;
        }
        let mut neighbors: Vec<u32> = clusters
            .get(index)
            .into_iter()
            .flatten()
            .flat_map(|&triangle| {
                adjacency
                    .get(triangle as usize)
                    .into_iter()
                    .flat_map(|sides| sides.iter().copied())
            })
            .filter(|&neighbor| neighbor != u32::MAX)
            .map(|neighbor| owner.get(neighbor as usize).copied().unwrap_or(u32::MAX))
            .filter(|&cluster| cluster != u32::MAX && cluster as usize != index)
            .collect();
        neighbors.sort_unstable();
        neighbors.dedup();

        // The fullest neighbour that still fits packs the tightest.
        let target = neighbors
            .into_iter()
            .filter(|&candidate| {
                let (Some(other), Some(current)) =
                    (clusters.get(candidate as usize), clusters.get(index))
                else {
                    return false;
                };
                other.len() + current.len() <= limits.max_triangles
                    && unique_vertices(local, other, current) <= limits.max_vertices
            })
            .max_by_key(|&candidate| clusters.get(candidate as usize).map_or(0, Vec::len));
        if let Some(target) = target {
            let Some(moved) = clusters.get_mut(index).map(std::mem::take) else {
                continue;
            };
            for &triangle in &moved {
                if let Some(slot) = owner.get_mut(triangle as usize) {
                    *slot = target;
                }
            }
            if let Some(other) = clusters.get_mut(target as usize) {
                other.extend(moved);
            }
        }
    }

    clusters.retain(|cluster| !cluster.is_empty());
    clusters
}

/// Distinct vertices used by two clusters of the same local mesh.
fn unique_vertices(local: &LocalTriangles, a: &[u32], b: &[u32]) -> usize {
    let mut vertices: Vec<u32> = a
        .iter()
        .chain(b)
        .flat_map(|&triangle| local.triangles.get(triangle as usize).into_iter().flatten())
        .copied()
        .collect();
    vertices.sort_unstable();
    vertices.dedup();
    vertices.len()
}

/// The exposed cell of one triangle centroid on the chunking grid: `(centroid - origin) / step`
/// divided by 64, clamped to ten bits per axis.
fn chunk_key(centroid: Vec3, origin: Vec3, step: f32) -> u32 {
    let scale = if step > 0.0 && step.is_finite() {
        step
    } else {
        1.0
    };
    let q = (centroid - origin) / scale / 64.0;
    let bits = |value: f32| -> u32 { spread_bits(value.clamp(0.0, 1023.0) as u32) };
    bits(q.x) | bits(q.y) << 1 | bits(q.z) << 2
}

/// The chunked partition of a whole mesh, advanced one unit per step: the Morton keys of the
/// next `PARTITION_BATCH` triangles, one sort of the keyed pairs, then one chunk of at most
/// `PARTITION_BATCH` consecutive triangles of that order partitioned on its own.
pub struct PartitionState {
    triangle_count: usize,
    limits: ClusterLimits,
    origin: Vec3,
    step: f32,
    keyed: Vec<(u32, u32)>,
    next_key: usize,
    chunk_start: usize,
    covered: usize,
    finished: Vec<Vec<u32>>,
}

impl PartitionState {
    /// Prepares the partition of a mesh with `triangle_count` triangles.
    pub fn new(
        triangle_count: usize,
        limits: ClusterLimits,
        origin: Vec3,
        step: f32,
    ) -> PartitionState {
        PartitionState {
            triangle_count,
            limits,
            origin,
            step,
            keyed: Vec::new(),
            next_key: 0,
            chunk_start: 0,
            covered: 0,
            finished: Vec::new(),
        }
    }

    /// Runs one unit. Returns true once every triangle is in a cluster.
    pub fn step(&mut self, triangle: &dyn Fn(u32) -> [u32; 3], positions: &[Vec3]) -> bool {
        if self.next_key < self.triangle_count {
            let end = (self.next_key + PARTITION_BATCH).min(self.triangle_count);
            for index in self.next_key..end {
                let corners = triangle(index as u32);
                let mut sum = Vec3::ZERO;
                for vertex in corners {
                    sum += positions
                        .get(vertex as usize)
                        .copied()
                        .unwrap_or(Vec3::ZERO);
                }
                let key = chunk_key(sum / 3.0, self.origin, self.step);
                self.keyed.push((key, index as u32));
            }
            self.next_key = end;
            self.keyed.sort_unstable();
            if self.next_key < self.triangle_count {
                return false;
            }
        }

        let Some(chunk_range) = self.keyed.get(self.chunk_start..) else {
            return true;
        };
        if chunk_range.is_empty() {
            return true;
        }
        let end = self.chunk_start + PARTITION_BATCH.min(chunk_range.len());
        let Some(chunk) = self.keyed.get(self.chunk_start..end) else {
            return true;
        };
        let mesh_indices: Vec<u32> = chunk.iter().map(|&(_, index)| index).collect();
        let chunk_triangles: Vec<[u32; 3]> =
            mesh_indices.iter().map(|&index| triangle(index)).collect();
        for cluster in partition_triangles(&chunk_triangles, positions, self.limits) {
            let mut mapped = Vec::with_capacity(cluster.len());
            for &local in &cluster {
                if let Some(&index) = mesh_indices.get(local as usize) {
                    mapped.push(index);
                }
            }
            if !mapped.is_empty() {
                self.covered += mapped.len();
                self.finished.push(mapped);
            }
        }
        self.chunk_start = end;
        self.chunk_start >= self.keyed.len()
    }

    /// The clusters finished since the last drain, as mesh triangle indices, in production order.
    pub fn drain(&mut self) -> Vec<Vec<u32>> {
        std::mem::take(&mut self.finished)
    }

    /// The covered fraction of the mesh, in 0..=1, never decreasing.
    pub fn progress(&self) -> f32 {
        if self.triangle_count == 0 {
            return 1.0;
        }
        (self.covered as f32 / self.triangle_count as f32).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: ClusterLimits = ClusterLimits {
        max_triangles: 128,
        max_vertices: 255,
    };

    #[test]
    fn spreading_the_widest_10_bit_value_fills_every_third_bit() {
        assert_eq!(spread_bits(0x3ff), 0x0924_9249);
        assert_eq!(spread_bits(0), 0);
        assert_eq!(spread_bits(1), 1);
    }

    #[test]
    fn morton_order_of_the_cube_corners_is_a_permutation() {
        let mut points = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    points.push(Vec3::new(x as f32, y as f32, z as f32));
                }
            }
        }
        let mut order = morton_order(&points);
        assert_eq!(order.len(), points.len());
        order.sort_unstable();
        assert_eq!(order, (0..8).collect::<Vec<u32>>());
    }

    #[test]
    fn morton_order_keeps_index_order_for_equal_points() {
        assert_eq!(morton_order(&[Vec3::splat(0.5); 5]), vec![0, 1, 2, 3, 4]);
        let mut mixed = vec![Vec3::new(1.0, 1.0, 1.0); 3];
        mixed.extend([Vec3::ZERO; 2]);
        assert_eq!(morton_order(&mixed), vec![3, 4, 0, 1, 2]);
    }

    #[test]
    fn shared_edges_are_each_others_neighbour() {
        let adjacency = triangle_adjacency(&[[0, 1, 2], [1, 3, 2]]);
        assert_eq!(adjacency[0][1], 1);
        assert_eq!(adjacency[1][2], 0);
        assert_eq!(adjacency[0][0], u32::MAX);
        assert_eq!(adjacency[0][2], u32::MAX);

        let lone = triangle_adjacency(&[[4, 5, 6]]);
        assert_eq!(lone, vec![[u32::MAX; 3]]);
    }

    #[test]
    fn local_triangles_compact_sparse_ids_and_keep_triangle_means() {
        let mut positions = vec![Vec3::ZERO; 51];
        positions[10] = Vec3::new(3.0, 0.0, 0.0);
        positions[20] = Vec3::new(0.0, 3.0, 0.0);
        positions[30] = Vec3::new(0.0, 0.0, 3.0);
        positions[40] = Vec3::new(1.0, 1.0, 1.0);
        positions[50] = Vec3::new(2.0, 2.0, 2.0);
        let local = LocalTriangles::new(&[[10, 20, 30], [30, 40, 50]], &positions);
        assert_eq!(local.triangles, vec![[0, 1, 2], [2, 3, 4]]);
        assert_eq!(local.vertex_count, 5);
        assert_eq!(local.centroids[0], Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(local.centroids[1], Vec3::new(1.0, 1.0, 2.0));
    }

    #[test]
    fn nearest_piece_skips_assigned_triangles_and_respects_the_vertex_limit() {
        let centroids = [
            Vec3::ZERO,
            Vec3::new(0.001, 0.001, 0.001),
            Vec3::new(0.002, 0.0, 0.0),
        ];
        let local = LocalTriangles {
            triangles: vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            centroids: centroids.to_vec(),
            vertex_count: 9,
        };
        let grid = PieceGrid::new(&centroids);
        let stamps = [u32::MAX; 9];

        let unassigned = [false; 3];
        assert_eq!(
            grid.nearest(&local, &unassigned, &stamps, 0, 0, Vec3::ZERO, LIMITS),
            Some(0)
        );

        let first_taken = [true, false, false];
        assert_eq!(
            grid.nearest(&local, &first_taken, &stamps, 0, 0, Vec3::ZERO, LIMITS),
            Some(1)
        );

        let tight = ClusterLimits {
            max_triangles: 128,
            max_vertices: 2,
        };
        assert_eq!(
            grid.nearest(&local, &unassigned, &stamps, 0, 0, Vec3::ZERO, tight),
            None
        );
    }

    #[test]
    fn nearest_piece_prefers_the_lowest_index_at_equal_distance() {
        let local = LocalTriangles {
            triangles: vec![[0, 1, 2], [3, 4, 5]],
            centroids: vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)],
            vertex_count: 6,
        };
        let mut grid = PieceGrid {
            origin: Vec3::ZERO,
            cell: 1.0,
            cells: BTreeMap::new(),
        };
        grid.cells.insert([-1, 0, 0], vec![1]);
        grid.cells.insert([1, 0, 0], vec![0]);
        let assigned = [false; 2];
        let stamps = [u32::MAX; 6];
        assert_eq!(
            grid.nearest(&local, &assigned, &stamps, 0, 0, Vec3::ZERO, LIMITS),
            Some(0)
        );
    }

    /// A flat quad grid of `size` by `size` cells: two triangles per cell.
    fn grid(size: u32) -> (Vec<[u32; 3]>, Vec<Vec3>) {
        let mut positions = Vec::new();
        for y in 0..=size {
            for x in 0..=size {
                positions.push(Vec3::new(x as f32, 0.0, y as f32));
            }
        }
        let mut triangles = Vec::new();
        let at = |x: u32, y: u32| y * (size + 1) + x;
        for y in 0..size {
            for x in 0..size {
                triangles.push([at(x, y), at(x, y + 1), at(x + 1, y)]);
                triangles.push([at(x + 1, y), at(x, y + 1), at(x + 1, y + 1)]);
            }
        }
        (triangles, positions)
    }

    #[test]
    fn a_grid_partitions_into_full_clusters_within_the_limits() {
        let (triangles, positions) = grid(40);
        assert_eq!(triangles.len(), 3200);
        let clusters = partition_triangles(&triangles, &positions, LIMITS);
        let mut seen = vec![0u32; triangles.len()];
        let mut small = 0;
        for cluster in &clusters {
            assert!(!cluster.is_empty());
            assert!(cluster.len() <= LIMITS.max_triangles);
            let mut vertices: Vec<u32> = cluster
                .iter()
                .flat_map(|&triangle| triangles[triangle as usize])
                .collect();
            vertices.sort_unstable();
            vertices.dedup();
            assert!(vertices.len() <= LIMITS.max_vertices);
            if cluster.len() < SMALL_CLUSTER {
                small += 1;
            }
            for &triangle in cluster {
                seen[triangle as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&count| count == 1));
        assert!(small <= 4, "{small} clusters were below {SMALL_CLUSTER}");
    }

    #[test]
    fn sixteen_triangle_limits_are_respected() {
        let (triangles, positions) = grid(40);
        let limits = ClusterLimits {
            max_triangles: 16,
            max_vertices: 32,
        };
        for cluster in partition_triangles(&triangles, &positions, limits) {
            assert!(cluster.len() <= limits.max_triangles);
            let mut vertices: Vec<u32> = cluster
                .iter()
                .flat_map(|&triangle| triangles[triangle as usize])
                .collect();
            vertices.sort_unstable();
            vertices.dedup();
            assert!(vertices.len() <= limits.max_vertices);
        }
    }

    #[test]
    fn separate_quads_pack_into_clusters_of_more_than_one_piece() {
        let (mut triangles, mut positions) = (Vec::new(), Vec::new());
        for z in 0..20u32 {
            for x in 0..20u32 {
                let base = positions.len() as u32;
                let at = Vec3::new(x as f32 * 0.5, 0.0, z as f32 * 0.5);
                positions.extend([
                    at,
                    at + Vec3::X * 0.1,
                    at + Vec3::Z * 0.1,
                    at + Vec3::new(0.1, 0.0, 0.1),
                ]);
                triangles.push([base, base + 1, base + 2]);
                triangles.push([base + 1, base + 3, base + 2]);
            }
        }
        let clusters = partition_triangles(&triangles, &positions, LIMITS);
        let average = triangles.len() as f32 / clusters.len() as f32;
        assert!(average > 2.0, "average cluster size was only {average}");
        assert!(clusters.iter().any(|cluster| cluster.len() > 2));
    }

    #[test]
    fn the_partition_state_covers_every_triangle_once_and_is_deterministic() {
        let (triangles, positions) = grid(100);
        assert_eq!(triangles.len(), 20000);
        let access = |index: u32| triangles[index as usize];

        let mut state = PartitionState::new(triangles.len(), LIMITS, Vec3::ZERO, 1.0);
        let mut incremental = Vec::new();
        let mut last = 0.0f32;
        loop {
            let done = state.step(&access, &positions);
            let progress = state.progress();
            assert!((0.0..=1.0).contains(&progress));
            assert!(progress >= last, "progress went from {last} to {progress}");
            last = progress;
            incremental.extend(state.drain());
            if done {
                break;
            }
        }
        assert_eq!(last, 1.0);
        assert_eq!(state.progress(), 1.0);
        assert!(state.drain().is_empty());

        let mut state = PartitionState::new(triangles.len(), LIMITS, Vec3::ZERO, 1.0);
        while !state.step(&access, &positions) {}
        let at_end = state.drain();
        assert_eq!(incremental, at_end);

        let mut seen = vec![0u32; triangles.len()];
        for cluster in &at_end {
            for &triangle in cluster {
                seen[triangle as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&count| count == 1));

        let mut second = PartitionState::new(triangles.len(), LIMITS, Vec3::ZERO, 1.0);
        while !second.step(&access, &positions) {}
        assert_eq!(second.drain(), at_end);
    }
}
