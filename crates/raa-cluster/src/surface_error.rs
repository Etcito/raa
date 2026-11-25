//! Conservative surface deviation of a DAG reduction. A quadric is useful to rank
//! collapses but does not account for a lost component or a hole in a flat surface.

use crate::dvec::DVec3;
use crate::simplify::Triangle;
use crate::thin::NEW_POSITION;
use crate::weld::WeldedMesh;
use raa_math::Vec3;
use std::collections::BTreeSet;

type Points = [DVec3; 3];
type Key = [u32; 3];
const TRIANGLES_PER_UNIT: usize = 12;
const SUBDIVISIONS: usize = 2;

#[derive(Clone, Copy)]
struct Box3 {
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Box3 {
    const EMPTY: Self = Self {
        lo: [f64::INFINITY; 3],
        hi: [f64::NEG_INFINITY; 3],
    };

    fn include(mut self, other: Self) -> Self {
        for i in 0..3 {
            self.lo[i] = self.lo[i].min(other.lo[i]);
            self.hi[i] = self.hi[i].max(other.hi[i]);
        }
        self
    }

    fn distance2(self, point: DVec3) -> f64 {
        let p = [point.x, point.y, point.z];
        (0..3)
            .map(|i| (self.lo[i] - p[i]).max(p[i] - self.hi[i]).max(0.0).powi(2))
            .sum()
    }

    fn widest_axis(self) -> usize {
        let extent = |i: usize| self.hi[i] - self.lo[i];
        (0..3)
            .max_by(|&a, &b| extent(a).total_cmp(&extent(b)))
            .unwrap()
    }
}

#[derive(Clone, Copy)]
struct Face {
    points: Points,
    center: [f64; 3],
    bounds: Box3,
    normal: DVec3,
    normal2: f64,
}

impl Face {
    fn new(points: Points) -> Self {
        let [a, b, c] = points;
        let center = centroid(points);
        let bounds = points.iter().fold(Box3::EMPTY, |bounds, p| {
            let p = [p.x, p.y, p.z];
            bounds.include(Box3 { lo: p, hi: p })
        });
        let normal = (b - a).cross(c - a);
        Self {
            points,
            center: [center.x, center.y, center.z],
            bounds,
            normal,
            normal2: normal.length_squared(),
        }
    }

    fn distance2(self, p: DVec3) -> f64 {
        let [a, b, c] = self.points;
        if self.normal2 > 0.0
            && (b - a).cross(p - a).dot(self.normal) >= 0.0
            && (c - b).cross(p - b).dot(self.normal) >= 0.0
            && (a - c).cross(p - c).dot(self.normal) >= 0.0
        {
            (p - a).dot(self.normal).powi(2) / self.normal2
        } else {
            segment_distance2(p, a, b)
                .min(segment_distance2(p, b, c))
                .min(segment_distance2(p, c, a))
        }
    }
}

fn centroid([a, b, c]: Points) -> DVec3 {
    (a + b + c) / 3.0
}

fn segment_distance2(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let edge = b - a;
    let length2 = edge.length_squared();
    if length2 == 0.0 {
        return a.distance_squared(p);
    }
    let t = ((p - a).dot(edge) / length2).clamp(0.0, 1.0);
    (a + edge * t).distance_squared(p)
}

struct Node {
    bounds: Box3,
    first: usize,
    end: usize,
    children: Option<[usize; 2]>,
}

struct Surface {
    faces: Vec<Face>,
    nodes: Vec<Node>,
}

impl Surface {
    fn new(faces: Vec<Face>) -> Self {
        let mut surface = Self {
            faces,
            nodes: Vec::new(),
        };
        surface.build(0, surface.faces.len());
        surface
    }

    fn build(&mut self, first: usize, end: usize) -> usize {
        let bounds = self.faces[first..end]
            .iter()
            .fold(Box3::EMPTY, |bounds, face| bounds.include(face.bounds));
        let index = self.nodes.len();
        self.nodes.push(Node {
            bounds,
            first,
            end,
            children: None,
        });
        if end - first > 8 {
            let axis = bounds.widest_axis();
            let half = (end - first) / 2;
            self.faces[first..end]
                .select_nth_unstable_by(half, |a, b| a.center[axis].total_cmp(&b.center[axis]));
            let left = self.build(first, first + half);
            let right = self.build(first + half, end);
            self.nodes[index].children = Some([left, right]);
        }
        index
    }

    fn nearest(&self, node: usize, point: DVec3, best: &mut (f64, usize)) {
        let node = &self.nodes[node];
        if let Some([a, b]) = node.children {
            let da = self.nodes[a].bounds.distance2(point);
            let db = self.nodes[b].bounds.distance2(point);
            let children = if da <= db {
                [(da, a), (db, b)]
            } else {
                [(db, b), (da, a)]
            };
            for (distance, index) in children {
                if distance < best.0 {
                    self.nearest(index, point, best);
                }
            }
        } else {
            for i in node.first..node.end {
                let distance = self.faces[i].distance2(point);
                if distance < best.0 {
                    *best = (distance, i);
                }
            }
        }
    }

    /// Distance to a convex triangle is convex. If its distance from each corner is
    /// bounded, the same bound holds on the full source triangle. Choosing the target
    /// by the centroid keeps this useful near hole boundaries. Subdivision tightens
    /// the bound across target edges, with a fixed worst-case number of queries.
    fn bound2(&self, points: Points, known: f64, depth: usize) -> f64 {
        let mut nearest = (f64::INFINITY, 0);
        self.nearest(0, centroid(points), &mut nearest);
        let face = self.faces[nearest.1];
        let upper = points
            .map(|point| face.distance2(point))
            .into_iter()
            .fold(0.0, f64::max);
        let [a, b, c] = points;
        let edge2 = a
            .distance_squared(b)
            .max(b.distance_squared(c))
            .max(c.distance_squared(a));
        // A small acceptable gap avoids chasing a perfect bound around a target edge.
        if depth == 0 || upper <= known.max(nearest.0 + edge2 / 4096.0) {
            return upper;
        }
        let ab = (a + b) * 0.5;
        let bc = (b + c) * 0.5;
        let ca = (c + a) * 0.5;
        let mut best: f64 = 0.0;
        for sub in [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]] {
            best = best.max(self.bound2(sub, known.max(best), depth - 1));
        }
        best
    }
}

fn keyed_faces(
    triangles: &[Triangle],
    welded: &WeldedMesh,
    new_positions: &[Vec3],
) -> Vec<(Key, Face)> {
    triangles
        .iter()
        .map(|triangle| {
            let mut key = triangle.map(|corner| corner.position);
            key.sort_unstable();
            let points = triangle.map(|corner| {
                let id = corner.position;
                let p = if id & NEW_POSITION == 0 {
                    welded.positions[id as usize]
                } else {
                    new_positions[(id & !NEW_POSITION) as usize]
                };
                DVec3::from_vec3(p)
            });
            (key, Face::new(points))
        })
        .collect()
}

/// Measures both directions. Unchanged triangles contribute zero and remain in both
/// search trees. Each step measures a bounded batch so an interactive bake can yield.
pub(crate) struct SurfaceError {
    forward: (Vec<Points>, Surface),
    backward: (Vec<Points>, Surface),
    cursor: usize,
    max2: f64,
}

impl SurfaceError {
    pub(crate) fn new(
        original: &[Triangle],
        reduced: &[Triangle],
        welded: &WeldedMesh,
        new_positions: &[Vec3],
    ) -> Self {
        let snapped: Vec<_> = new_positions.iter().map(|&p| welded.snap(p)).collect();
        let original = keyed_faces(original, welded, &[]);
        let reduced = keyed_faces(reduced, welded, &snapped);
        let changed = |source: &[(Key, Face)], target: &[(Key, Face)]| {
            let target_keys: BTreeSet<_> = target.iter().map(|(key, _)| *key).collect();
            source
                .iter()
                .filter_map(|(key, face)| (!target_keys.contains(key)).then_some(face.points))
                .collect()
        };
        let forward = changed(&original, &reduced);
        let backward = changed(&reduced, &original);
        Self {
            forward: (
                forward,
                Surface::new(reduced.into_iter().map(|(_, face)| face).collect()),
            ),
            backward: (
                backward,
                Surface::new(original.into_iter().map(|(_, face)| face).collect()),
            ),
            cursor: 0,
            max2: 0.0,
        }
    }

    pub(crate) fn step(&mut self) -> bool {
        let split = self.forward.0.len();
        let end = (self.cursor + TRIANGLES_PER_UNIT).min(split + self.backward.0.len());
        while self.cursor < end {
            let (points, tree) = if self.cursor < split {
                (&self.forward.0[self.cursor], &self.forward.1)
            } else {
                (&self.backward.0[self.cursor - split], &self.backward.1)
            };
            self.max2 = self.max2.max(tree.bound2(*points, self.max2, SUBDIVISIONS));
            self.cursor += 1;
        }
        self.cursor == split + self.backward.0.len()
    }

    pub(crate) fn error(&self) -> f32 {
        self.max2.sqrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_missing_coplanar_pieces_and_triangle_interiors() {
        let point = |x, y| DVec3::new(x, y, 0.0);
        let piece = |x| [point(x, 0.0), point(x + 1.0, 0.0), point(x, 1.0)];
        let retained = Surface::new(vec![Face::new(piece(0.0))]);
        assert_eq!(retained.bound2(piece(0.0), 0.0, SUBDIVISIONS), 0.0);
        assert!((retained.bound2(piece(3.0), 0.0, SUBDIVISIONS) - 9.0).abs() < 1e-10);

        let source = [point(0.0, 0.0), point(3.0, 0.0), point(0.0, 3.0)];
        let corners = Surface::new(source.into_iter().map(|p| Face::new([p; 3])).collect());
        // Nearest distances at all three vertices are zero. The interior is uncovered.
        let mut center = (f64::INFINITY, 0);
        corners.nearest(0, centroid(source), &mut center);
        assert!(center.0 > 0.0);
        assert!(corners.bound2(source, 0.0, SUBDIVISIONS) >= center.0);
    }
}
