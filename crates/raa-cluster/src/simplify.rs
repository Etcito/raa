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

/// Picks the attribute nearest `wanted` among the wedges at the collapse target and the corner's
/// `current` one. Wedges on the far side of a hard edge do not qualify; ties keep the first
/// candidate in `wedges` order, then `current`.
pub fn best_wedge(attributes: &[Attribute], current: u32, wedges: &[u32], wanted: DVec2) -> u32 {
    let normal = attributes.get(current as usize).map(|a| a.normal);
    let distance = |wedge: u32| match attributes.get(wedge as usize) {
        Some(attribute) => DVec2::from_f32(attribute.uv).distance(wanted),
        None => f64::INFINITY,
    };
    let qualifies = |wedge: u32| match (attributes.get(wedge as usize), normal) {
        (Some(attribute), Some(current_normal)) => attribute.normal.dot(current_normal) > 0.7,
        _ => false,
    };
    wedges
        .iter()
        .copied()
        .filter(|&wedge| qualifies(wedge))
        .chain(std::iter::once(current))
        .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
        .unwrap_or(current)
}

/// The collapse state of one group: welded local vertices (ids in first-use order, mapped back to
/// input positions through `global`), their quadrics, locks, stamps and the alive triangles.
// The costs, the collapse and the run loop of the next share read the heap, stamps, texture
// errors, quadrics, global ids, attributes and the running maximum error; until then they are
// only written here.
#[allow(dead_code)]
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
        assert_eq!(best_wedge(&attributes, 0, &[3], DVec2::new(0.1, 0.0)), 0);
        assert_eq!(
            best_wedge(&attributes, 0, &[2, 4], DVec2::new(0.2, 0.0)),
            2,
            "a tie keeps the first candidate"
        );
        assert_eq!(best_wedge(&attributes, 0, &[99], DVec2::new(0.0, 0.0)), 0);
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
}
