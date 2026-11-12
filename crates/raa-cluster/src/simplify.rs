//! Edge-collapse simplification of one cluster group with its outer boundary locked: corner
//! types, quadrics, costs, collapse and the `simplify()` entry point of the build.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/simplify.rs`, revision 7e465b5.

use crate::dvec::{DVec2, DVec3};
use raa_math::Vec3;
use std::cmp::Ordering;

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
}
