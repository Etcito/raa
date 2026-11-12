//! Thinning of foliage-like groups made of separate pieces: piece shapes, wide pieces and
//! the enlargement that keeps the covered area while the triangle count falls.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/thin.rs`, revision 7e465b5.

use crate::dvec::{DMat3, DVec3};
use crate::simplify::Triangle;
use raa_math::Vec3;
use std::collections::BTreeMap;

/// Fewer separate pieces than this in a group are left to ordinary simplification, so a lone
/// object is never mistaken for foliage.
pub const MIN_PIECES: usize = 8;
/// A piece holding more than this fraction of its group's triangles is a surface, not foliage.
pub const MAX_PIECE_SHARE: f32 = 0.125;
/// Pieces more than this many times wider than the group's median piece are not thinned.
pub const WIDE: f64 = 3.0;

/// A group's triangles sorted into separate pieces and everything else.
pub struct Pieces {
    /// Triangle indices of each piece, in order of their first triangle.
    pub pieces: Vec<Vec<u32>>,
    /// Triangles that belong to no piece, for ordinary simplification.
    pub rest: Vec<u32>,
}

/// Splits `triangles` into connected pieces. Only pieces that lie wholly inside the group, touching
/// no locked vertex, can be removed, so the rest is everything else.
pub fn find_pieces(triangles: &[Triangle], is_locked: &dyn Fn(u32) -> bool) -> Pieces {
    let mut local: BTreeMap<u32, u32> = BTreeMap::new();
    for triangle in triangles {
        for corner in triangle {
            let next = local.len() as u32;
            local.entry(corner.position).or_insert(next);
        }
    }
    let ids: Vec<[u32; 3]> = triangles
        .iter()
        .map(|triangle| triangle.map(|corner| local[&corner.position]))
        .collect();
    let mut parent: Vec<u32> = (0..local.len() as u32).collect();
    fn root(parent: &mut [u32], mut v: u32) -> u32 {
        while parent[v as usize] != v {
            parent[v as usize] = parent[parent[v as usize] as usize];
            v = parent[v as usize];
        }
        v
    }
    for &[a, b, c] in &ids {
        let ra = root(&mut parent, a);
        for other in [b, c] {
            let ro = root(&mut parent, other);
            parent[ro as usize] = ra;
        }
    }

    // Components are numbered in the order of their first triangle.
    let mut component: BTreeMap<u32, usize> = BTreeMap::new();
    let mut members: Vec<Vec<u32>> = Vec::new();
    let mut touches_lock: Vec<bool> = Vec::new();
    for (index, (triangle, tri_ids)) in triangles.iter().zip(&ids).enumerate() {
        let id = root(&mut parent, tri_ids[0]);
        let slot = *component.entry(id).or_insert_with(|| {
            members.push(Vec::new());
            touches_lock.push(false);
            members.len() - 1
        });
        members[slot].push(index as u32);
        if triangle.iter().any(|corner| is_locked(corner.position)) {
            touches_lock[slot] = true;
        }
    }

    let largest = (triangles.len() as f32 * MAX_PIECE_SHARE).max(1.0) as usize;
    let mut pieces = Vec::new();
    let mut rest = Vec::new();
    for (piece, locked) in members.into_iter().zip(touches_lock) {
        if !locked && piece.len() <= largest {
            pieces.push(piece);
        } else {
            rest.extend(piece);
        }
    }
    if pieces.len() < MIN_PIECES {
        rest.extend(pieces.drain(..).flatten());
        rest.sort_unstable();
    }
    Pieces { pieces, rest }
}

/// A piece's centre, the directions it is longest and widest in, and how long and wide it is.
pub struct Shape {
    pub center: DVec3,
    pub across: DVec3,
    pub length: f64,
    pub width: f64,
}

/// The piece's centre, its length along the direction it is longest in, and its width across it
/// within its own plane.
pub fn shape(triangles: &[Triangle], piece: &[u32], positions: &[Vec3], points: &[DVec3]) -> Shape {
    let center = points.iter().copied().sum::<DVec3>() / points.len() as f64;
    let mut normal = DVec3::ZERO;
    for &t in piece {
        let [a, b, c] = triangles[t as usize]
            .map(|corner| DVec3::from_vec3(positions[corner.position as usize]));
        normal += (b - a).cross(c - a);
    }
    let normal = normal.normalize_or(DVec3::Y);

    let mut covariance = DMat3::ZERO;
    for &p in points {
        let d = p - center;
        covariance.add_outer(d - normal * d.dot(normal));
    }
    // Not `any_orthonormal_vector`: for an axis-aligned flat piece it lands exactly on the
    // narrow axis, which power iteration never leaves.
    let start = normal
        .cross(DVec3::new(0.57, 0.58, 0.58))
        .normalize_or(DVec3::X);
    let axis = power(&covariance, start);
    let across = normal.cross(axis).normalize_or(DVec3::X);
    Shape {
        center,
        across,
        length: extent(points, center, axis),
        width: extent(points, center, across).max(1.0e-9),
    }
}

/// The direction `covariance` stretches most, found by 16 steps of power iteration from `axis`.
fn power(covariance: &DMat3, mut axis: DVec3) -> DVec3 {
    for _ in 0..16 {
        axis = covariance.mul_vec(axis).normalize_or(axis);
    }
    axis
}

/// The span of `points` measured along `direction` from `center`.
fn extent(points: &[DVec3], center: DVec3, direction: DVec3) -> f64 {
    let (lo, hi) = points.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
        let t = (*p - center).dot(direction);
        (lo.min(t), hi.max(t))
    });
    hi - lo
}

/// The distinct position ids of a piece in ascending order, and their points.
pub fn piece_points(
    triangles: &[Triangle],
    piece: &[u32],
    positions: &[Vec3],
) -> (Vec<u32>, Vec<DVec3>) {
    let mut vertices: Vec<u32> = piece
        .iter()
        .flat_map(|&t| triangles[t as usize].map(|c| c.position))
        .collect();
    vertices.sort_unstable();
    vertices.dedup();
    let points = vertices
        .iter()
        .map(|&v| DVec3::from_vec3(positions[v as usize]))
        .collect();
    (vertices, points)
}

/// Moves pieces much wider than is typical in their group, such as a branch among twigs, out of
/// `pieces` and into `rest`. Thinning enlarges the survivors, which moves a wide one so far that
/// it would hold back the whole group's detail; edge collapse shortens such a piece instead.
pub fn set_aside_wide_pieces(triangles: &[Triangle], found: &mut Pieces, positions: &[Vec3]) {
    if found.pieces.is_empty() {
        return;
    }
    let widths: Vec<f64> = found
        .pieces
        .iter()
        .map(|piece| match tube_radius(triangles, piece, positions) {
            Some(radius) => radius * 2.0,
            None => {
                let (_, points) = piece_points(triangles, piece, positions);
                shape(triangles, piece, positions, &points).width
            }
        })
        .collect();
    let mut sorted = widths.clone();
    sorted.sort_by(f64::total_cmp);
    let limit = sorted[sorted.len() / 2] * WIDE;
    let mut kept = Vec::with_capacity(found.pieces.len());
    for (piece, width) in found.pieces.drain(..).zip(widths) {
        if width <= limit {
            kept.push(piece);
        } else {
            found.rest.extend(piece);
        }
    }
    if kept.len() < MIN_PIECES {
        found.rest.extend(kept.drain(..).flatten());
    }
    found.pieces = kept;
    found.rest.sort_unstable();
}

/// The average radius of a piece that is a closed tube, or `None` if it is flat. A tube's faces
/// point every way, so their normals all but cancel; a blade's or a leaf's add up.
pub fn tube_radius(triangles: &[Triangle], piece: &[u32], positions: &[Vec3]) -> Option<f64> {
    let mut sum = DVec3::ZERO;
    let mut area = 0.0;
    let (_, points) = piece_points(triangles, piece, positions);
    for &t in piece {
        let [a, b, c] = triangles[t as usize]
            .map(|corner| DVec3::from_vec3(positions[corner.position as usize]));
        let face = (b - a).cross(c - a);
        sum += face;
        area += face.length() * 0.5;
    }
    if sum.length() * 0.5 > area * 0.3 {
        return None;
    }
    // A tube's area is its circumference times its length.
    let center = points.iter().copied().sum::<DVec3>() / points.len() as f64;
    let mut covariance = DMat3::ZERO;
    for &p in &points {
        covariance.add_outer(p - center);
    }
    let axis = power(&covariance, DVec3::new(0.57, 0.58, 0.58));
    let length = extent(&points, center, axis).max(1.0e-9);
    Some(area / (std::f64::consts::TAU * length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplify::Corner;
    use std::f64::consts::TAU;

    /// Triangles and positions built one vertex at a time; shared vertices join pieces.
    #[derive(Default)]
    struct Mesh {
        triangles: Vec<Triangle>,
        positions: Vec<Vec3>,
    }

    impl Mesh {
        fn vertex(&mut self, x: f32, y: f32, z: f32) -> u32 {
            self.positions.push(Vec3::new(x, y, z));
            (self.positions.len() - 1) as u32
        }

        fn triangle(&mut self, a: u32, b: u32, c: u32) {
            self.triangles.push([a, b, c].map(|position| Corner {
                position,
                attribute: 0,
            }));
        }

        /// A separate quad of size `w` by `h` in the XY plane, starting at `x`.
        fn quad(&mut self, x: f32, w: f32, h: f32) {
            let a = self.vertex(x, 0.0, 0.0);
            let b = self.vertex(x + w, 0.0, 0.0);
            let c = self.vertex(x + w, h, 0.0);
            let d = self.vertex(x, h, 0.0);
            self.triangle(a, b, c);
            self.triangle(a, c, d);
        }

        /// `count` quads sharing edges in a row from `x0`: one connected sheet.
        fn sheet(&mut self, x0: f32, count: u32) {
            let base = self.positions.len() as u32;
            for j in 0..=count {
                self.vertex(x0 + j as f32, 0.0, 0.0);
                self.vertex(x0 + j as f32, 1.0, 0.0);
            }
            for j in 0..count {
                let bottom = base + 2 * j;
                self.triangle(bottom, bottom + 2, bottom + 3);
                self.triangle(bottom, bottom + 3, bottom + 1);
            }
        }

        /// A closed square tube of half width 1 and length 4 along Z: four sides, no caps.
        fn tube(&mut self) {
            let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            let near: Vec<u32> = corners
                .iter()
                .map(|&(x, y)| self.vertex(x, y, 0.0))
                .collect();
            let far: Vec<u32> = corners
                .iter()
                .map(|&(x, y)| self.vertex(x, y, 4.0))
                .collect();
            for side in 0..4 {
                let next = (side + 1) % 4;
                self.triangle(near[side], near[next], far[next]);
                self.triangle(near[side], far[next], far[side]);
            }
        }
    }

    #[test]
    fn twenty_separate_quads_are_twenty_pieces() {
        let mut mesh = Mesh::default();
        for i in 0..20 {
            mesh.quad(10.0 * i as f32, 1.0, 1.0);
        }
        let found = find_pieces(&mesh.triangles, &|_| false);
        assert_eq!(found.pieces.len(), 20);
        assert!(found.rest.is_empty());
    }

    #[test]
    fn five_separate_quads_are_rest_in_order() {
        let mut mesh = Mesh::default();
        for i in 0..5 {
            mesh.quad(10.0 * i as f32, 1.0, 1.0);
        }
        let found = find_pieces(&mesh.triangles, &|_| false);
        assert!(found.pieces.is_empty());
        assert_eq!(found.rest, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn piece_with_a_locked_vertex_is_rest() {
        let mut mesh = Mesh::default();
        for i in 0..20 {
            mesh.quad(10.0 * i as f32, 1.0, 1.0);
        }
        let found = find_pieces(&mesh.triangles, &|v| v == 0);
        assert_eq!(found.pieces.len(), 19);
        assert_eq!(found.rest, vec![0, 1]);
    }

    #[test]
    fn large_sheet_beside_quads_is_rest() {
        let mut mesh = Mesh::default();
        for i in 0..20 {
            mesh.quad(10.0 * i as f32, 1.0, 1.0);
        }
        mesh.sheet(300.0, 20);
        let found = find_pieces(&mesh.triangles, &|_| false);
        assert_eq!(found.pieces.len(), 20);
        assert_eq!(found.rest, (40..80).collect::<Vec<u32>>());
    }

    #[test]
    fn flat_quad_has_no_tube_radius() {
        let mut mesh = Mesh::default();
        mesh.quad(0.0, 1.0, 4.0);
        assert_eq!(tube_radius(&mesh.triangles, &[0, 1], &mesh.positions), None);
    }

    #[test]
    fn closed_square_tube_radius_is_near_its_half_width() {
        let mut mesh = Mesh::default();
        mesh.tube();
        let piece: Vec<u32> = (0..8).collect();
        // Perimeter 8 over a circle's circumference: 8 / TAU, about 1.27 for half width 1.
        let radius = tube_radius(&mesh.triangles, &piece, &mesh.positions).unwrap_or(f64::NAN);
        assert!((radius - 8.0 / TAU).abs() < 1e-9, "radius {radius}");
        assert!((radius - 1.0).abs() < 0.3, "radius {radius}");
    }

    #[test]
    fn shape_of_a_one_by_four_quad() {
        let mut mesh = Mesh::default();
        mesh.quad(0.0, 1.0, 4.0);
        let piece = [0u32, 1];
        let (_, points) = piece_points(&mesh.triangles, &piece, &mesh.positions);
        let s = shape(&mesh.triangles, &piece, &mesh.positions, &points);
        assert!((s.length - 4.0).abs() < 1e-6, "length {}", s.length);
        assert!((s.width - 1.0).abs() < 1e-6, "width {}", s.width);
    }

    #[test]
    fn quad_ten_times_wider_is_set_aside() {
        let mut mesh = Mesh::default();
        for i in 0..20 {
            mesh.quad(10.0 * i as f32, 1.0, 1.0);
        }
        mesh.quad(300.0, 10.0, 10.0);
        let mut found = find_pieces(&mesh.triangles, &|_| false);
        assert_eq!(found.pieces.len(), 21);
        set_aside_wide_pieces(&mesh.triangles, &mut found, &mesh.positions);
        assert_eq!(found.pieces.len(), 20);
        assert_eq!(found.rest, vec![40, 41]);
    }
}
