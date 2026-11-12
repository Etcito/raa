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
}
