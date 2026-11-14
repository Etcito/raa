//! Edge-collapse simplification of one cluster group with its outer boundary locked: corner
//! types, quadrics, costs, collapse and the `simplify()` entry point of the build.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/simplify.rs`, revision 7e465b5.

use crate::dvec::{DVec2, DVec3};
use raa_math::Vec3;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

/// Shading attributes of a welded vertex.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Attribute {
    pub normal: Vec3,
    pub uv: [f32; 2],
}

/// A triangle corner: which welded position and which attribute it uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Corner {
    pub position: u32,
    pub attribute: u32,
}

pub type Triangle = [Corner; 3];

/// Result of the simplification.
pub struct Simplified {
    pub triangles: Vec<Triangle>,
    /// Largest geometric deviation introduced, in metres.
    pub error: f32,
}

/// Reduces `triangles` towards `target_triangles`. Vertices for which `is_locked` returns true
/// never move, and neither do the ends of any edge not shared by exactly two triangles.
pub fn simplify(
    triangles: &[Triangle],
    positions: &[Vec3],
    attributes: &[Attribute],
    target_triangles: usize,
    is_locked: &dyn Fn(u32) -> bool,
) -> Simplified {
    Simplifier::new(triangles, positions, attributes, is_locked).run(target_triangles)
}

/// Rejects a collapse that would rotate a face by more than about 78 degrees.
pub const MIN_NORMAL_AGREEMENT: f64 = 0.2;
/// Weight of the edge-length term that breaks ties on flat regions, relative to squared length.
pub const EDGE_LENGTH_WEIGHT: f64 = 1.0e-4;
/// How much a normal discontinuity across an edge raises its collapse cost.
pub const NORMAL_DIFFERENCE_WEIGHT: f64 = 0.02;
/// How much a texture displacement counts, relative to moving the surface by the same distance.
pub const TEXTURE_ERROR_WEIGHT: f64 = 0.5;
/// Largest distance outside a triangle, in triangle sizes, to which its texture mapping is trusted.
pub const MAX_EXTRAPOLATION: f64 = 16.0;

/// A symmetric 4x4 error quadric plus the accumulated area weight.
#[derive(Clone, Copy, Default)]
pub struct Quadric {
    aa: f64,
    ab: f64,
    ac: f64,
    ad: f64,
    bb: f64,
    bc: f64,
    bd: f64,
    cc: f64,
    cd: f64,
    dd: f64,
    weight: f64,
}

impl Quadric {
    /// The quadric of the plane `normal . p = d`, weighted by `weight` (usually an area).
    pub fn from_plane(normal: DVec3, d: f64, weight: f64) -> Quadric {
        let (a, b, c) = (normal.x, normal.y, normal.z);
        Quadric {
            aa: a * a * weight,
            ab: a * b * weight,
            ac: a * c * weight,
            ad: a * d * weight,
            bb: b * b * weight,
            bc: b * c * weight,
            bd: b * d * weight,
            cc: c * c * weight,
            cd: c * d * weight,
            dd: d * d * weight,
            weight,
        }
    }

    pub fn add(&mut self, o: &Quadric) {
        self.aa += o.aa;
        self.ab += o.ab;
        self.ac += o.ac;
        self.ad += o.ad;
        self.bb += o.bb;
        self.bc += o.bc;
        self.bd += o.bd;
        self.cc += o.cc;
        self.cd += o.cd;
        self.dd += o.dd;
        self.weight += o.weight;
    }

    /// Weighted mean squared distance from `p` to the accumulated planes.
    pub fn mean_squared_distance(&self, p: DVec3) -> f64 {
        if self.weight <= 0.0 {
            return 0.0;
        }
        let value = self.aa * p.x * p.x
            + 2.0 * self.ab * p.x * p.y
            + 2.0 * self.ac * p.x * p.z
            + 2.0 * self.ad * p.x
            + self.bb * p.y * p.y
            + 2.0 * self.bc * p.y * p.z
            + 2.0 * self.bd * p.y
            + self.cc * p.z * p.z
            + 2.0 * self.cd * p.z
            + self.dd;
        value.max(0.0) / self.weight
    }
}

/// A candidate collapse of `from` into `to`, valid while both stamps still match.
pub struct Candidate {
    pub cost: f64,
    pub from: u32,
    pub to: u32,
    pub from_stamp: u32,
    pub to_stamp: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// Reversed on cost so that `BinaryHeap` pops the cheapest collapse. Vertex ids break
    /// ties so the result does not depend on heap internals.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.from.cmp(&self.from))
            .then_with(|| other.to.cmp(&self.to))
    }
}

/// Distance from `point` to the segment `start`..`end`; a zero-length segment is a point.
pub fn distance_to_segment(point: DVec3, start: DVec3, end: DVec3) -> f64 {
    let along = end - start;
    let length_squared = along.length_squared();
    if length_squared <= 0.0 {
        return point.distance(start);
    }
    let t = ((point - start).dot(along) / length_squared).clamp(0.0, 1.0);
    point.distance(start + along * t)
}

/// Picks the attribute a corner moving onto the collapse target takes: always one of the target's
/// own `wedges`, so the corner never carries a normal or uv into a place the surface does not
/// have them (raa change, 2026-11-14: solstice also offered `current`, the attribute of the vertex
/// being removed. Across a UV seam or a normal change over 45 degrees that won, so a sphere's
/// normal was drawn on the column under it; the sdfm part atlas has seams on a third of its
/// positions). Among the wedges on the corner's side of a hard edge (normal within about 45
/// degrees of `current`) the nearest uv to `wanted` wins; when none is on that side, the wedge whose
/// normal is closest to `current`'s. Ties keep the first candidate in `wedges` order. `current` is
/// kept only when no wedge names a known attribute.
pub fn best_wedge(attributes: &[Attribute], current: u32, wedges: &[u32], wanted: DVec2) -> u32 {
    let Some(current_normal) = attributes.get(current as usize).map(|a| a.normal) else {
        return wedges
            .iter()
            .copied()
            .find(|&wedge| attributes.get(wedge as usize).is_some())
            .unwrap_or(current);
    };
    let known: Vec<(u32, &Attribute)> = wedges
        .iter()
        .filter_map(|&wedge| attributes.get(wedge as usize).map(|a| (wedge, a)))
        .collect();
    let distance = |attribute: &Attribute| DVec2::from_f32(attribute.uv).distance(wanted);
    let same_side = known
        .iter()
        .filter(|(_, attribute)| attribute.normal.dot(current_normal) > 0.7)
        .min_by(|a, b| distance(a.1).total_cmp(&distance(b.1)));
    if let Some(&(wedge, _)) = same_side {
        return wedge;
    }
    known
        .iter()
        .max_by(|a, b| {
            let (da, db) = (
                a.1.normal.dot(current_normal),
                b.1.normal.dot(current_normal),
            );
            // max_by keeps the last of equal elements; compare reversed ids so the first wins.
            da.total_cmp(&db).then_with(|| b.0.cmp(&a.0))
        })
        .map_or(current, |&(wedge, _)| wedge)
}

/// The collapse state of one group: welded local vertices (ids in first-use order, mapped back to
/// input positions through `global`), their quadrics, locks, stamps and the alive triangles.
pub struct Simplifier<'a> {
    attributes: &'a [Attribute],
    /// Global position id per local vertex.
    global: Vec<u32>,
    positions: Vec<DVec3>,
    locked: Vec<bool>,
    quadrics: Vec<Quadric>,
    stamps: Vec<u32>,
    /// Texture displacement built up at each local vertex by earlier collapses, in world units.
    texture_errors: Vec<f64>,
    /// Triangles touching each local vertex.
    vertex_triangles: Vec<Vec<u32>>,
    /// Triangles whose corners' `position` holds local vertex ids.
    triangles: Vec<Triangle>,
    alive: Vec<bool>,
    alive_count: usize,
    heap: BinaryHeap<Candidate>,
    max_error: f64,
}

impl<'a> Simplifier<'a> {
    /// Welds the input to local vertex ids in first-use order and builds the quadrics. A vertex
    /// on an edge not used by exactly two triangles is locked, as is one `is_locked` names. A
    /// position id past `positions` reads as the origin.
    pub fn new(
        input: &[Triangle],
        positions: &[Vec3],
        attributes: &'a [Attribute],
        is_locked: &dyn Fn(u32) -> bool,
    ) -> Simplifier<'a> {
        let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
        let mut global: Vec<u32> = Vec::new();
        let triangles: Vec<Triangle> = input
            .iter()
            .map(|t| {
                t.map(|corner| {
                    let local = *remap.entry(corner.position).or_insert_with(|| {
                        global.push(corner.position);
                        (global.len() - 1) as u32
                    });
                    Corner {
                        position: local,
                        attribute: corner.attribute,
                    }
                })
            })
            .collect();

        let vertex_count = global.len();
        let local_positions: Vec<DVec3> = global
            .iter()
            .map(|&g| {
                positions
                    .get(g as usize)
                    .map_or(DVec3::ZERO, |p| DVec3::from_vec3(*p))
            })
            .collect();
        let mut locked: Vec<bool> = global.iter().map(|&g| is_locked(g)).collect();
        let mut vertex_triangles = vec![Vec::new(); vertex_count];
        let mut quadrics = vec![Quadric::default(); vertex_count];
        let mut edge_uses: BTreeMap<u64, u32> = BTreeMap::new();

        for (index, t) in triangles.iter().enumerate() {
            let [a, b, c] = t.map(|corner| corner.position);
            let (pa, pb, pc) = (
                local_positions[a as usize],
                local_positions[b as usize],
                local_positions[c as usize],
            );
            let cross = (pb - pa).cross(pc - pa);
            let double_area = cross.length();
            if double_area > 0.0 {
                let normal = cross / double_area;
                let quadric = Quadric::from_plane(normal, -normal.dot(pa), double_area * 0.5);
                for v in [a, b, c] {
                    quadrics[v as usize].add(&quadric);
                }
            }
            for (v, next) in [(a, b), (b, c), (c, a)] {
                vertex_triangles[v as usize].push(index as u32);
                let key = (u64::from(v.min(next)) << 32) | u64::from(v.max(next));
                *edge_uses.entry(key).or_insert(0) += 1;
            }
        }

        // Borders and non-manifold edges stay fixed. A border here is either the group's outer
        // boundary or a genuine hole in the source mesh.
        for (key, uses) in edge_uses {
            if uses != 2 {
                locked[(key >> 32) as usize] = true;
                locked[(key & 0xffff_ffff) as usize] = true;
            }
        }

        let alive_count = triangles.len();
        Simplifier {
            attributes,
            global,
            positions: local_positions,
            locked,
            quadrics,
            stamps: vec![0; vertex_count],
            texture_errors: vec![0.0; vertex_count],
            vertex_triangles,
            triangles,
            alive: vec![true; alive_count],
            alive_count,
            heap: BinaryHeap::new(),
            max_error: 0.0,
        }
    }

    /// The vertices sharing a triangle with `v`, ascending; empty for an unknown id.
    pub fn neighbors(&self, v: u32) -> Vec<u32> {
        let Some(triangles) = self.vertex_triangles.get(v as usize) else {
            return Vec::new();
        };
        let mut result: Vec<u32> = triangles
            .iter()
            .filter_map(|&t| self.triangles.get(t as usize))
            .flatten()
            .map(|corner| corner.position)
            .filter(|&p| p != v)
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }

    /// The attribute ids `v` uses across its triangles, ascending; empty for an unknown id.
    pub fn wedges_at(&self, v: u32) -> Vec<u32> {
        let Some(triangles) = self.vertex_triangles.get(v as usize) else {
            return Vec::new();
        };
        let mut wedges: Vec<u32> = triangles
            .iter()
            .filter_map(|&t| self.triangles.get(t as usize))
            .flatten()
            .filter(|corner| corner.position == v)
            .map(|corner| corner.attribute)
            .collect();
        wedges.sort_unstable();
        wedges.dedup();
        wedges
    }

    /// Whether collapsing `from` onto `to` keeps the surface manifold (the link
    /// condition) and flips or degenerates no surviving triangle. A locked `from` never moves.
    pub fn can_collapse(&self, from: u32, to: u32) -> bool {
        let known = self
            .vertex_triangles
            .get(from as usize)
            .zip(self.positions.get(to as usize));
        let Some((from_triangles, &new_position)) = known else {
            return false;
        };
        if from == to || self.locked(from) {
            return false;
        }

        // Link condition: the only vertices adjacent to both ends must be the two opposite the
        // shared edge, otherwise the collapse pinches the surface.
        let from_neighbors = self.neighbors(from);
        let to_neighbors = self.neighbors(to);
        let shared = from_neighbors
            .iter()
            .filter(|n| to_neighbors.binary_search(n).is_ok())
            .count();
        if shared != 2 {
            return false;
        }

        // Reject collapses that flip or degenerate any surviving triangle.
        for &triangle in from_triangles {
            let corners = self.triangles[triangle as usize].map(|c| c.position);
            if corners.contains(&to) {
                continue;
            }
            let old = corners.map(|v| self.positions[v as usize]);
            let new = corners.map(|v| {
                if v == from {
                    new_position
                } else {
                    self.positions[v as usize]
                }
            });
            let old_normal = (old[1] - old[0]).cross(old[2] - old[0]);
            let new_normal = (new[1] - new[0]).cross(new[2] - new[0]);
            let new_length = new_normal.length();
            if new_length <= 0.0 {
                return false;
            }
            if old_normal.dot(new_normal) < MIN_NORMAL_AGREEMENT * old_normal.length() * new_length
            {
                return false;
            }
        }
        true
    }

    /// Whether `v` never moves. An unknown id counts as locked.
    pub fn locked(&self, v: u32) -> bool {
        self.locked.get(v as usize).copied().unwrap_or(true)
    }

    /// Number of triangles not yet collapsed away.
    pub fn alive_triangles(&self) -> usize {
        self.alive_count
    }

    /// Collapses the cheapest admissible edge until the target or no candidate is left, then
    /// reports the surviving triangles with their input position ids.
    pub fn run(mut self, target_triangles: usize) -> Simplified {
        for v in 0..self.positions.len() as u32 {
            self.push_edges_around(v);
        }
        while self.alive_count > target_triangles {
            let Some(candidate) = self.heap.pop() else {
                break;
            };
            if self.stamps[candidate.from as usize] != candidate.from_stamp
                || self.stamps[candidate.to as usize] != candidate.to_stamp
            {
                continue;
            }
            if self.can_collapse(candidate.from, candidate.to) {
                self.collapse(candidate.from, candidate.to);
            }
        }
        self.finish()
    }

    /// Queues every collapse of an unlocked vertex touching `v`, in both directions.
    fn push_edges_around(&mut self, v: u32) {
        for neighbor in self.neighbors(v) {
            for (from, to) in [(v, neighbor), (neighbor, v)] {
                if !self.locked(from) {
                    let candidate = Candidate {
                        cost: self.collapse_cost(from, to),
                        from,
                        to,
                        from_stamp: self.stamps[from as usize],
                        to_stamp: self.stamps[to as usize],
                    };
                    self.heap.push(candidate);
                }
            }
        }
    }

    fn collapse_cost(&self, from: u32, to: u32) -> f64 {
        let mut quadric = self.quadrics[from as usize];
        quadric.add(&self.quadrics[to as usize]);
        let target = self.positions[to as usize];
        let length_squared = self.positions[from as usize].distance_squared(target);
        let texture = self.texture_error(from, to) * TEXTURE_ERROR_WEIGHT;
        quadric.mean_squared_distance(target)
            + texture * texture
            + length_squared
                * (EDGE_LENGTH_WEIGHT + NORMAL_DIFFERENCE_WEIGHT * self.normal_difference(from, to))
    }

    /// How far the texture slides across the surface if `from` collapses into `to`, in world
    /// units, including what earlier collapses already moved at `from`.
    fn texture_error(&self, from: u32, to: u32) -> f64 {
        let wedges = self.wedges_at(to);
        let worst = self.vertex_triangles[from as usize]
            .iter()
            .map(|&t| self.triangles[t as usize])
            .filter(|t| !t.iter().any(|corner| corner.position == to))
            .map(|t| self.moved_corner(&t, from, to, &wedges).1)
            .fold(0.0, f64::max);
        worst.max(self.seam_movement(from, to)) + self.texture_errors[from as usize]
    }

    /// How far the collapse moves a UV seam. Each side's mapping can stay continuous while one
    /// side grows over the other, replacing what the surface showed there.
    fn seam_movement(&self, from: u32, to: u32) -> f64 {
        let seams = self.seam_neighbors(from);
        let start = self.positions[from as usize];
        let end = self.positions[to as usize];
        match seams[..] {
            [] => 0.0,
            // Sliding along a seam only moves it as far as the seam bends here.
            [a, b] if a == to || b == to => {
                let other = self.positions[(if a == to { b } else { a }) as usize];
                distance_to_segment(start, other, end)
            }
            // Leaving the seam, the end of one or a junction of several: it must stay put.
            _ => start.distance(end),
        }
    }

    /// The neighbours of `v` connected by an edge whose two sides use different textures.
    fn seam_neighbors(&self, v: u32) -> Vec<u32> {
        let uv_of = |triangle: &Triangle, vertex: u32| {
            triangle
                .iter()
                .find(|corner| corner.position == vertex)
                .and_then(|corner| self.attributes.get(corner.attribute as usize))
                .map(|attribute| attribute.uv)
        };
        let mut seams = self.neighbors(v);
        seams.retain(|&n| {
            let mut sides = self.vertex_triangles[v as usize]
                .iter()
                .map(|&t| &self.triangles[t as usize])
                .filter(|t| t.iter().any(|corner| corner.position == n));
            let (Some(first), Some(second)) = (sides.next(), sides.next()) else {
                return false;
            };
            uv_of(first, v) != uv_of(second, v) || uv_of(first, n) != uv_of(second, n)
        });
        seams
    }

    /// The attribute the corner of `triangle` at `from` takes on moving to `to`, and how far the
    /// texture slides as a result. A corner the triangle does not touch moves nowhere.
    fn moved_corner(&self, triangle: &Triangle, from: u32, to: u32, wedges: &[u32]) -> (u32, f64) {
        let Some(index) = triangle.iter().position(|corner| corner.position == from) else {
            return (triangle[0].attribute, 0.0);
        };
        let current = triangle[index].attribute;
        let uv_of = |corner: Corner| match self.attributes.get(corner.attribute as usize) {
            Some(attribute) => DVec2::from_f32(attribute.uv),
            None => DVec2::new(0.0, 0.0),
        };
        let (next, last) = (triangle[(index + 1) % 3], triangle[(index + 2) % 3]);
        let origin = self.positions[from as usize];
        let edge_a = self.positions[next.position as usize] - origin;
        let edge_b = self.positions[last.position as usize] - origin;
        let offset = self.positions[to as usize] - origin;
        let uv_origin = uv_of(triangle[index]);
        let uv_a = uv_of(next) - uv_origin;
        let uv_b = uv_of(last) - uv_origin;

        let (aa, ab, bb) = (edge_a.dot(edge_a), edge_a.dot(edge_b), edge_b.dot(edge_b));
        let determinant = aa * bb - ab * ab;
        let uv_area = uv_a.perp_dot(uv_b).abs();
        if determinant <= 1.0e-12 * aa * bb || uv_area <= 0.0 {
            // No usable mapping here, as for a sliver or a mesh without UVs.
            return (best_wedge(self.attributes, current, wedges, uv_origin), 0.0);
        }
        let along_a = (bb * offset.dot(edge_a) - ab * offset.dot(edge_b)) / determinant;
        let along_b = (aa * offset.dot(edge_b) - ab * offset.dot(edge_a)) / determinant;
        let limit = MAX_EXTRAPOLATION;
        let wanted =
            uv_origin + uv_a * along_a.clamp(-limit, limit) + uv_b * along_b.clamp(-limit, limit);

        let chosen = best_wedge(self.attributes, current, wedges, wanted);
        let world_per_uv = (determinant.sqrt() / uv_area).sqrt();
        let moved = self
            .attributes
            .get(chosen as usize)
            .map_or(0.0, |attribute| {
                DVec2::from_f32(attribute.uv).distance(wanted)
            });
        (chosen, moved * world_per_uv)
    }

    /// How much the shading normals disagree between the two ends, in `0..=1`.
    fn normal_difference(&self, from: u32, to: u32) -> f64 {
        let normal_at = |v: u32| {
            self.vertex_triangles[v as usize]
                .iter()
                .flat_map(|&t| self.triangles[t as usize])
                .find(|corner| corner.position == v)
                .and_then(|corner| self.attributes.get(corner.attribute as usize))
                .map_or(Vec3::Y, |attribute| attribute.normal)
        };
        f64::from((1.0 - normal_at(from).dot(normal_at(to))) * 0.5)
    }

    /// Moves `from` onto `to`, merges their quadrics and textures, drops the collapsed triangles
    /// and invalidates the queued work around the changed neighbourhood.
    fn collapse(&mut self, from: u32, to: u32) {
        let target = self.positions[to as usize];
        let mut merged = self.quadrics[to as usize];
        merged.add(&self.quadrics[from as usize]);
        let texture_error = self.texture_error(from, to);
        let weighted = texture_error * TEXTURE_ERROR_WEIGHT;
        self.max_error = self
            .max_error
            .max(merged.mean_squared_distance(target))
            .max(weighted * weighted);
        self.quadrics[to as usize] = merged;
        self.texture_errors[to as usize] = self.texture_errors[to as usize].max(texture_error);

        let target_wedges = self.wedges_at(to);
        let from_triangles = std::mem::take(&mut self.vertex_triangles[from as usize]);
        for triangle in from_triangles {
            let index = triangle as usize;
            if !self.alive[index] {
                continue;
            }
            let corners = self.triangles[index];
            if corners.iter().any(|corner| corner.position == to) {
                self.alive[index] = false;
                self.alive_count -= 1;
                for corner in corners {
                    if corner.position != from {
                        self.vertex_triangles[corner.position as usize].retain(|&t| t != triangle);
                    }
                }
            } else {
                let (attribute, _) = self.moved_corner(&corners, from, to, &target_wedges);
                for corner in &mut self.triangles[index] {
                    if corner.position == from {
                        corner.position = to;
                        corner.attribute = attribute;
                    }
                }
                self.vertex_triangles[to as usize].push(triangle);
            }
        }

        // Invalidate every queued collapse touching the changed neighbourhood.
        self.stamps[from as usize] += 1;
        self.stamps[to as usize] += 1;
        let neighbors = self.neighbors(to);
        for &n in &neighbors {
            self.stamps[n as usize] += 1;
        }
        self.push_edges_around(to);
        for n in neighbors {
            self.push_edges_around(n);
        }
    }

    fn finish(self) -> Simplified {
        let triangles = self
            .triangles
            .iter()
            .zip(&self.alive)
            .filter(|&(_, &alive)| alive)
            .map(|(t, _)| {
                t.map(|corner| Corner {
                    position: self.global[corner.position as usize],
                    attribute: corner.attribute,
                })
            })
            .collect();
        Simplified {
            triangles,
            error: self.max_error.sqrt() as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn quadric_of_a_plane_measures_distance_squared() {
        let quadric = Quadric::from_plane(DVec3::Y, 0.0, 1.0);
        assert_eq!(
            quadric.mean_squared_distance(DVec3::new(0.0, 2.0, 0.0)),
            4.0
        );
        assert_eq!(
            quadric.mean_squared_distance(DVec3::new(1.0, 0.0, -3.0)),
            0.0
        );
        let zero = Quadric::from_plane(DVec3::Y, 0.0, 0.0);
        assert_eq!(zero.mean_squared_distance(DVec3::new(0.0, 2.0, 0.0)), 0.0);
    }

    fn candidate(cost: f64, from: u32, to: u32) -> Candidate {
        Candidate {
            cost,
            from,
            to,
            from_stamp: 0,
            to_stamp: 0,
        }
    }

    #[test]
    fn candidate_order_pops_cheapest_then_lower_ids() {
        let mut heap = BinaryHeap::new();
        for candidate in [
            candidate(1.0, 5, 2),
            candidate(1.0, 3, 9),
            candidate(1.0, 3, 4),
            candidate(0.5, 9, 9),
        ] {
            heap.push(candidate);
        }
        let popped: Vec<(u32, u32)> = std::iter::from_fn(|| heap.pop())
            .map(|c| (c.from, c.to))
            .collect();
        assert_eq!(popped, [(9, 9), (3, 4), (3, 9), (5, 2)]);
    }

    #[test]
    fn segment_distance_at_ends_inside_and_degenerate() {
        let (start, end) = (DVec3::ZERO, DVec3::new(2.0, 0.0, 0.0));
        assert_eq!(distance_to_segment(start, start, end), 0.0);
        assert_eq!(distance_to_segment(end, start, end), 0.0);
        assert_eq!(
            distance_to_segment(DVec3::new(-1.0, 0.0, 0.0), start, end),
            1.0
        );
        assert_eq!(
            distance_to_segment(DVec3::new(1.0, 1.0, 0.0), start, end),
            1.0
        );
        let point = DVec3::new(1.0, 0.0, 0.0);
        assert_eq!(
            distance_to_segment(DVec3::new(1.0, 3.0, 0.0), point, point),
            3.0
        );
    }

    fn attribute(uv: [f32; 2], normal: Vec3) -> Attribute {
        Attribute { normal, uv }
    }

    #[test]
    fn best_wedge_takes_the_nearest_uv_and_skips_hard_edges() {
        let attributes = [
            attribute([0.0, 0.0], Vec3::Y),
            attribute([1.0, 0.0], Vec3::Y),
            attribute([0.2, 0.0], Vec3::Y),
            attribute([0.1, 0.0], Vec3::new(0.0, -1.0, 0.0)),
            attribute([0.2, 0.0], Vec3::Y),
        ];
        assert_eq!(
            best_wedge(&attributes, 0, &[1, 2, 3], DVec2::new(0.25, 0.0)),
            2
        );
        assert_eq!(
            best_wedge(&attributes, 0, &[3], DVec2::new(0.1, 0.0)),
            3,
            "a target with only a far-side wedge still gives its own attribute"
        );
        assert_eq!(
            best_wedge(&attributes, 0, &[2, 4], DVec2::new(0.2, 0.0)),
            2,
            "a tie keeps the first candidate"
        );
        assert_eq!(best_wedge(&attributes, 0, &[99], DVec2::new(0.0, 0.0)), 0);
    }

    /// The removed vertex's attribute is never carried onto the target, even when its uv is the
    /// nearest (a UV seam at the target) or every target wedge is past a hard edge.
    #[test]
    fn best_wedge_never_keeps_the_removed_vertex_attribute() {
        let tilted = Vec3::new(0.0, 0.6, 0.8);
        let attributes = [
            attribute([0.5, 0.5], Vec3::Y),
            attribute([0.9, 0.9], Vec3::Y),
            attribute([0.0, 0.0], Vec3::X),
            attribute([0.0, 0.0], tilted),
        ];
        // Seam: the only same-side wedge at the target is far in uv, `current` is at `wanted`.
        assert_eq!(best_wedge(&attributes, 0, &[1], DVec2::new(0.5, 0.5)), 1);
        // Every wedge past a hard edge: the closest normal wins.
        assert_eq!(best_wedge(&attributes, 0, &[2, 3], DVec2::new(0.5, 0.5)), 3);
    }

    /// A 3x3 grid of 8 triangles. Position ids are numbered in first-use order (the centre is 2),
    /// so local ids equal the input ids.
    fn grid() -> (Vec<Triangle>, Vec<Vec3>, Vec<Attribute>) {
        let corner = |position: u32| Corner {
            position,
            attribute: position,
        };
        let triangles: Vec<Triangle> = [
            [0, 1, 2],
            [0, 2, 3],
            [1, 4, 5],
            [1, 5, 2],
            [3, 2, 6],
            [3, 6, 7],
            [2, 5, 8],
            [2, 8, 6],
        ]
        .iter()
        .map(|t| t.map(corner))
        .collect();
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        ];
        let attributes = vec![
            Attribute {
                normal: Vec3::Z,
                uv: [0.0; 2]
            };
            9
        ];
        (triangles, positions, attributes)
    }

    #[test]
    fn grid_locks_the_border_and_frees_the_centre() {
        let (triangles, positions, attributes) = grid();
        let simplifier = Simplifier::new(&triangles, &positions, &attributes, &|_: u32| false);
        assert!(!simplifier.locked(2));
        for v in [0, 1, 3, 4, 5, 6, 7, 8] {
            assert!(simplifier.locked(v), "border vertex {v} is not locked");
        }
        assert_eq!(simplifier.alive_triangles(), 8);
    }

    #[test]
    fn neighbours_and_wedges_of_the_centre_are_ascending_and_complete() {
        let (triangles, positions, attributes) = grid();
        let simplifier = Simplifier::new(&triangles, &positions, &attributes, &|_: u32| false);
        assert_eq!(simplifier.neighbors(2), [0, 1, 3, 5, 6, 8]);
        assert_eq!(simplifier.wedges_at(2), [2]);
        assert!(simplifier.neighbors(99).is_empty());
    }

    #[test]
    fn centre_collapses_onto_a_neighbour_unless_a_triangle_flips() {
        let (triangles, mut positions, attributes) = grid();
        let simplifier = Simplifier::new(&triangles, &positions, &attributes, &|_: u32| false);
        assert!(simplifier.can_collapse(2, 1));
        // Vertex 1 moved across the edge 0-3: collapsing the centre onto it flips triangle 0-2-3.
        positions[1] = Vec3::new(-0.5, 1.5, 0.0);
        let flipped = Simplifier::new(&triangles, &positions, &attributes, &|_: u32| false);
        assert!(!flipped.can_collapse(2, 1));
    }

    #[test]
    fn a_lock_closure_locks_the_vertex_it_names() {
        let (triangles, positions, attributes) = grid();
        let simplifier = Simplifier::new(&triangles, &positions, &attributes, &|v: u32| v == 2);
        assert!(simplifier.locked(2));
        assert!(
            !simplifier.can_collapse(2, 1),
            "a locked centre does not move"
        );
    }

    #[test]
    fn local_ids_follow_first_use_and_the_closure_sees_input_ids() {
        let corner = |position: u32| Corner {
            position,
            attribute: 0,
        };
        // A fan around input vertex 20: local 0 is 20, then 10, 30, 40 and 50 in first use.
        let triangles = [
            [corner(20), corner(10), corner(30)],
            [corner(20), corner(30), corner(40)],
            [corner(20), corner(40), corner(50)],
            [corner(20), corner(50), corner(10)],
        ];
        let positions = vec![Vec3::ZERO; 51];
        let attributes = [Attribute::default()];
        let simplifier = Simplifier::new(&triangles, &positions, &attributes, &|g: u32| g == 20);
        assert!(simplifier.locked(0), "the closure gets the input id 20");
        assert_eq!(simplifier.neighbors(0), [1, 2, 3, 4]);
        assert!(simplifier.locked(1), "the outer ring is a border");
    }

    /// A flat `cells` x `cells` grid of positions with the identity texture mapping.
    fn flat_grid(cells: u32) -> (Vec<Triangle>, Vec<Vec3>, Vec<Attribute>) {
        let side = cells + 1;
        let at = |x: u32, z: u32| z * side + x;
        let positions: Vec<Vec3> = (0..side)
            .flat_map(|z| (0..side).map(move |x| Vec3::new(x as f32, 0.0, z as f32)))
            .collect();
        let attributes: Vec<Attribute> = positions
            .iter()
            .map(|p| Attribute {
                normal: Vec3::Y,
                uv: [p.x, p.z],
            })
            .collect();
        let mut triangles = Vec::new();
        for z in 0..cells {
            for x in 0..cells {
                let corner = |px: u32, pz: u32| Corner {
                    position: at(px, pz),
                    attribute: at(px, pz),
                };
                let (a, b, c, d) = (
                    corner(x, z),
                    corner(x + 1, z),
                    corner(x, z + 1),
                    corner(x + 1, z + 1),
                );
                triangles.push([a, c, b]);
                triangles.push([b, c, d]);
            }
        }
        (triangles, positions, attributes)
    }

    fn border_vertices(cells: u32) -> Vec<u32> {
        let side = cells + 1;
        let mut border: Vec<u32> = (0..side)
            .flat_map(|x| [x, cells * side + x])
            .chain((0..side).flat_map(|z| [z * side, z * side + cells]))
            .collect();
        border.sort_unstable();
        border.dedup();
        border
    }

    #[test]
    fn a_flat_grid_reaches_the_target_and_keeps_its_border() {
        let (triangles, positions, attributes) = flat_grid(8);
        let result = simplify(&triangles, &positions, &attributes, 64, &|_| false);
        assert!(
            result.triangles.len() <= 64,
            "{} triangles",
            result.triangles.len()
        );
        assert!(result.error < 1.0e-4, "error {}", result.error);
        let present: Vec<u32> = result
            .triangles
            .iter()
            .flatten()
            .map(|corner| corner.position)
            .collect();
        for v in border_vertices(8) {
            assert!(present.contains(&v), "border vertex {v} disappeared");
        }
    }

    #[test]
    fn every_vertex_locked_changes_nothing() {
        let (triangles, positions, attributes) = flat_grid(8);
        let result = simplify(&triangles, &positions, &attributes, 0, &|_| true);
        assert_eq!(result.triangles, triangles);
        assert_eq!(result.error, 0.0);
    }

    #[test]
    fn a_bump_in_the_middle_reports_error_without_moving_locked_vertices() {
        let (triangles, mut positions, attributes) = flat_grid(8);
        positions[(4 * 9 + 4) as usize].y = 0.5;
        let locked = border_vertices(8);
        let result = simplify(&triangles, &positions, &attributes, 0, &|v| {
            locked.contains(&v)
        });
        assert!(result.error > 0.0, "error {}", result.error);
        let present: Vec<u32> = result
            .triangles
            .iter()
            .flatten()
            .map(|corner| corner.position)
            .collect();
        for v in locked {
            assert!(present.contains(&v), "locked vertex {v} moved away");
        }
    }

    #[test]
    fn two_runs_give_identical_triangles() {
        let (triangles, mut positions, attributes) = flat_grid(8);
        positions[(4 * 9 + 4) as usize].y = 0.5;
        let first = simplify(&triangles, &positions, &attributes, 0, &|_| false);
        let second = simplify(&triangles, &positions, &attributes, 0, &|_| false);
        assert_eq!(first.triangles, second.triangles);
        assert_eq!(first.error, second.error);
    }

    /// A flat grid whose right half uses a texture shifted by `jump`, so the middle column is a
    /// UV seam: each side has its own attribute set and the seam vertices carry one of each.
    fn seamed_grid(cells: u32, jump: f32) -> (Vec<Triangle>, Vec<Vec3>, Vec<Attribute>) {
        let side = cells + 1;
        let seam = cells / 2;
        let at = |x: u32, z: u32| z * side + x;
        let count = side * side;
        let mut positions = Vec::new();
        let mut attributes = Vec::new();
        for z in 0..side {
            for x in 0..side {
                positions.push(Vec3::new(x as f32, 0.0, z as f32));
                let shift = if x > seam { jump } else { 0.0 };
                attributes.push(Attribute {
                    normal: Vec3::Y,
                    uv: [x as f32 + shift, z as f32],
                });
            }
        }
        for z in 0..side {
            for x in 0..side {
                attributes.push(Attribute {
                    normal: Vec3::Y,
                    uv: [x as f32 + jump, z as f32],
                });
            }
        }
        let mut triangles = Vec::new();
        for z in 0..cells {
            for x in 0..cells {
                let corner = |px: u32, pz: u32| Corner {
                    position: at(px, pz),
                    attribute: at(px, pz) + if x >= seam { count } else { 0 },
                };
                let (a, b, c, d) = (
                    corner(x, z),
                    corner(x + 1, z),
                    corner(x, z + 1),
                    corner(x + 1, z + 1),
                );
                triangles.push([a, c, b]);
                triangles.push([b, c, d]);
            }
        }
        (triangles, positions, attributes)
    }

    #[test]
    fn a_uv_seam_keeps_its_vertices_on_the_seam_line() {
        const JUMP: f32 = 50.0;
        let (triangles, positions, attributes) = seamed_grid(8, JUMP);
        let result = simplify(&triangles, &positions, &attributes, 64, &|_| false);
        let shift = |corner: Corner| match attributes.get(corner.attribute as usize) {
            Some(attribute) => attribute.uv[0] - positions[corner.position as usize].x,
            None => f32::NAN,
        };
        for triangle in &result.triangles {
            let shifts = triangle.map(shift);
            assert!(
                (shifts[0] - shifts[1]).abs() < 1.0e-3 && (shifts[1] - shifts[2]).abs() < 1.0e-3,
                "a triangle spans the seam: {shifts:?}"
            );
        }
        for corner in result.triangles.iter().flatten() {
            if corner.position % 9 == 4 {
                let x = attributes[corner.attribute as usize].uv[0];
                assert!(
                    (x - 4.0).abs() < 1.0e-3 || (x - (4.0 + JUMP)).abs() < 1.0e-3,
                    "a seam vertex left the seam line: uv.x {x}"
                );
            }
        }
    }
}
