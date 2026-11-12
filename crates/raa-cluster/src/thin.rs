//! Thinning of foliage-like groups made of separate pieces: piece shapes, wide pieces and
//! the enlargement that keeps the covered area while the triangle count falls.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/thin.rs`, revision 7e465b5.

use crate::dvec::{DMat3, DVec3};
use crate::simplify::{Attribute, Triangle};
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
    let ids: Vec<Option<[u32; 3]>> = triangles
        .iter()
        .map(|triangle| {
            let mut ids = [0u32; 3];
            for (slot, corner) in ids.iter_mut().zip(triangle) {
                *slot = *local.get(&corner.position)?;
            }
            Some(ids)
        })
        .collect();
    let mut parent: Vec<u32> = (0..local.len() as u32).collect();
    fn root(parent: &mut [u32], mut v: u32) -> u32 {
        while parent[v as usize] != v {
            parent[v as usize] = parent[parent[v as usize] as usize];
            v = parent[v as usize];
        }
        v
    }
    for &[a, b, c] in ids.iter().flatten() {
        let ra = root(&mut parent, a);
        for other in [b, c] {
            let ro = root(&mut parent, other);
            parent[ro as usize] = ra;
        }
    }

    // Components are numbered in the order of their first triangle. A triangle whose corner
    // cannot be resolved joins the rest.
    let mut component: BTreeMap<u32, usize> = BTreeMap::new();
    let mut members: Vec<Vec<u32>> = Vec::new();
    let mut touches_lock: Vec<bool> = Vec::new();
    let mut rest: Vec<u32> = Vec::new();
    for (index, (triangle, tri_ids)) in triangles.iter().zip(&ids).enumerate() {
        let Some(tri_ids) = tri_ids else {
            rest.push(index as u32);
            continue;
        };
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

/// The piece triangle `t`'s corners in f64, or `None` when the triangle id or one of its corner
/// positions is out of range.
fn triangle_corners(triangles: &[Triangle], positions: &[Vec3], t: u32) -> Option<[DVec3; 3]> {
    let triangle = triangles.get(t as usize)?;
    let mut corners = [DVec3::ZERO; 3];
    for (slot, corner) in corners.iter_mut().zip(triangle) {
        *slot = DVec3::from_vec3(*positions.get(corner.position as usize)?);
    }
    Some(corners)
}

/// The piece's centre, its length along the direction it is longest in, and its width across it
/// within its own plane. With no resolvable point it returns a zero-length shape at the origin.
pub fn shape(triangles: &[Triangle], piece: &[u32], positions: &[Vec3], points: &[DVec3]) -> Shape {
    if points.is_empty() {
        return Shape {
            center: DVec3::ZERO,
            across: DVec3::X,
            length: 0.0,
            width: 1.0e-9,
        };
    }
    let center = points.iter().copied().sum::<DVec3>() / points.len() as f64;
    let mut normal = DVec3::ZERO;
    for &t in piece {
        if let Some([a, b, c]) = triangle_corners(triangles, positions, t) {
            normal += (b - a).cross(c - a);
        }
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
        .filter_map(|&t| triangles.get(t as usize))
        .flat_map(|triangle| triangle.map(|c| c.position))
        .filter(|&v| positions.get(v as usize).is_some())
        .collect();
    vertices.sort_unstable();
    vertices.dedup();
    let points = vertices
        .iter()
        .filter_map(|&v| positions.get(v as usize).copied().map(DVec3::from_vec3))
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
        let [a, b, c] = triangle_corners(triangles, positions, t)?;
        let face = (b - a).cross(c - a);
        sum += face;
        area += face.length() * 0.5;
    }
    if sum.length() * 0.5 > area * 0.3 {
        return None;
    }
    if points.is_empty() {
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

/// Result of [`thin`].
pub struct Thinned {
    /// The surviving pieces. A corner whose position is `NEW_POSITION | i` uses `positions[i]`.
    pub triangles: Vec<Triangle>,
    pub positions: Vec<Vec3>,
    /// Furthest any surviving vertex moved, in metres.
    pub error: f32,
}

/// Marks a position index as one of [`Thinned::positions`] rather than an existing vertex.
pub const NEW_POSITION: u32 = 1 << 31;

/// The summed area of the piece's resolvable triangles.
fn piece_area(triangles: &[Triangle], piece: &[u32], positions: &[Vec3]) -> f64 {
    piece
        .iter()
        .filter_map(|&t| triangle_corners(triangles, positions, t))
        .map(|[a, b, c]| (b - a).cross(c - a).length() * 0.5)
        .sum()
}

/// Keeps pieces until at least `target_triangles` remain, and enlarges them to keep the total area.
///
/// Which pieces survive depends only on where they are, so the choice is the same however the
/// pieces happen to be listed. Triangles or vertices that do not resolve are skipped.
pub fn thin(
    triangles: &[Triangle],
    pieces: &[Vec<u32>],
    positions: &[Vec3],
    attributes: &[Attribute],
    target_triangles: usize,
) -> Thinned {
    let mut order: Vec<(u64, usize)> = pieces
        .iter()
        .enumerate()
        .map(|(index, piece)| {
            (
                placement_hash(piece_centroid(triangles, piece, positions)),
                index,
            )
        })
        .collect();
    order.sort_unstable();

    let mut kept = Vec::new();
    let mut kept_triangles = 0;
    for &(_, index) in &order {
        if kept_triangles >= target_triangles {
            break;
        }
        if let Some(piece) = pieces.get(index) {
            kept.push(piece);
            kept_triangles += piece.len();
        }
    }

    let total_area: f64 = pieces
        .iter()
        .map(|p| piece_area(triangles, p, positions))
        .sum();
    let kept_area: f64 = kept
        .iter()
        .map(|p| piece_area(triangles, p, positions))
        .sum();
    let growth = if kept_area > 0.0 {
        total_area / kept_area
    } else {
        1.0
    };

    let mut out = Thinned {
        triangles: Vec::with_capacity(kept_triangles),
        positions: Vec::new(),
        error: 0.0,
    };
    for piece in kept {
        let (vertices, points) = piece_points(triangles, piece, positions);
        let tube = tube_radius(triangles, piece, positions);
        let enlarge = enlargement(triangles, piece, positions, &points, growth);
        // Each vertex's outward direction, from the normals of the corners that use it.
        let mut outward: BTreeMap<u32, DVec3> = BTreeMap::new();
        if tube.is_some() {
            for triangle in piece.iter().filter_map(|&t| triangles.get(t as usize)) {
                for corner in triangle {
                    if let Some(a) = attributes.get(corner.attribute as usize) {
                        *outward.entry(corner.position).or_default() += DVec3::from_vec3(a.normal);
                    }
                }
            }
        }
        let mut moved: BTreeMap<u32, u32> = BTreeMap::new();
        for (&v, &p) in vertices.iter().zip(&points) {
            let to = match (tube, outward.get(&v)) {
                // A tube's area grows with its radius, so every point moves out by the same
                // fraction of the average radius.
                (Some(radius), Some(n)) => p + n.normalize_or_zero() * (radius * (growth - 1.0)),
                (Some(_), None) => p,
                (None, _) => enlarge(p),
            };
            out.error = out.error.max(to.distance(p) as f32);
            moved.insert(v, NEW_POSITION | out.positions.len() as u32);
            out.positions.push(to.to_vec3());
        }
        for triangle in piece.iter().filter_map(|&t| triangles.get(t as usize)) {
            let mut corners = *triangle;
            let mut resolved = true;
            for corner in &mut corners {
                match moved.get(&corner.position) {
                    Some(&position) => corner.position = position,
                    None => resolved = false,
                }
            }
            if resolved {
                out.triangles.push(corners);
            }
        }
    }
    out
}

/// How to move a piece's vertices so its area grows by `growth`.
fn enlargement(
    triangles: &[Triangle],
    piece: &[u32],
    positions: &[Vec3],
    points: &[DVec3],
    growth: f64,
) -> impl Fn(DVec3) -> DVec3 {
    let Shape {
        center,
        across,
        length,
        width,
    } = shape(triangles, piece, positions, points);
    let widen = growth.min((length / width).max(1.0));
    let scale = (growth / widen).sqrt();

    move |p: DVec3| {
        let d = p - center;
        center + (d + across * (d.dot(across) * (widen - 1.0))) * scale
    }
}

/// The f32 mean of the positions of all corners of the piece's resolvable triangles.
fn piece_centroid(triangles: &[Triangle], piece: &[u32], positions: &[Vec3]) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut count = 0usize;
    for triangle in piece.iter().filter_map(|&t| triangles.get(t as usize)) {
        for corner in triangle {
            if let Some(p) = positions.get(corner.position as usize) {
                sum += *p;
                count += 1;
            }
        }
    }
    sum / count.max(1) as f32
}

/// A well-mixed number from a position, so survivors are spread evenly rather than in rows.
fn placement_hash(p: Vec3) -> u64 {
    let mut x = (p.x.to_bits() as u64)
        ^ (p.y.to_bits() as u64).rotate_left(21)
        ^ (p.z.to_bits() as u64).rotate_left(42);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
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

    #[test]
    fn out_of_range_ids_do_not_panic() {
        let mut mesh = Mesh::default();
        mesh.quad(0.0, 1.0, 1.0);
        let bad = Corner {
            position: 999,
            attribute: 0,
        };
        mesh.triangles.push([bad; 3]);
        // Triangle 5 does not exist; triangle 2 references a missing position.
        let piece = [0u32, 1, 2, 5];
        let (ids, points) = piece_points(&mesh.triangles, &piece, &mesh.positions);
        assert_eq!(ids.len(), 4);
        assert_eq!(points.len(), 4);
        assert_eq!(tube_radius(&mesh.triangles, &piece, &mesh.positions), None);
        let s = shape(&mesh.triangles, &piece, &mesh.positions, &points);
        assert!((s.length - 2.0_f64.sqrt()).abs() < 1e-6 || s.length >= 1.0);
        let none = shape(&mesh.triangles, &[7], &mesh.positions, &[]);
        assert_eq!(none.length, 0.0);
        let mut found = Pieces {
            pieces: vec![piece.to_vec()],
            rest: Vec::new(),
        };
        set_aside_wide_pieces(&mesh.triangles, &mut found, &mesh.positions);
        assert!(found.pieces.is_empty());
        let all = find_pieces(&mesh.triangles, &|_| false);
        assert!(all.pieces.is_empty());
        assert_eq!(all.rest, vec![0, 1, 2]);
    }

    fn area_of(triangles: &[Triangle], positions: &[Vec3]) -> f64 {
        let all: Vec<u32> = (0..triangles.len() as u32).collect();
        piece_area(triangles, &all, positions)
    }

    fn quads(n: u32) -> (Mesh, Vec<Vec<u32>>) {
        let mut mesh = Mesh::default();
        for i in 0..n {
            mesh.quad(10.0 * i as f32, 1.0, 1.0);
        }
        let pieces = (0..n).map(|i| vec![2 * i, 2 * i + 1]).collect();
        (mesh, pieces)
    }

    #[test]
    fn thinning_forty_quads_keeps_about_half_and_the_area() {
        let (mut mesh, pieces) = quads(40);
        // Pack the quads into a small grid so f32 rounding stays far below the tolerance.
        for (i, p) in mesh.positions.iter_mut().enumerate() {
            let quad = (i / 4) as f32;
            p.x += 2.0 * (quad % 8.0) - 10.0 * quad;
            p.z = 2.0 * (quad / 8.0).floor();
        }
        let out = thin(&mesh.triangles, &pieces, &mesh.positions, &[], 40);
        assert!(out.triangles.len() >= 40 && out.triangles.len() <= 42);
        let before = area_of(&mesh.triangles, &mesh.positions);
        let plain: Vec<Triangle> = out
            .triangles
            .iter()
            .map(|t| {
                t.map(|c| Corner {
                    position: c.position & !NEW_POSITION,
                    attribute: 0,
                })
            })
            .collect();
        let after = area_of(&plain, &out.positions);
        assert!(
            (after - before).abs() / before < 1e-6,
            "{after} vs {before}"
        );
        for t in &out.triangles {
            for c in t {
                assert!(c.position & NEW_POSITION != 0);
                assert!(((c.position & !NEW_POSITION) as usize) < out.positions.len());
            }
        }
        assert!(out.error > 0.0);
    }

    #[test]
    fn error_is_the_largest_displacement() {
        let (mesh, pieces) = quads(40);
        let out = thin(&mesh.triangles, &pieces, &mesh.positions, &[], 40);
        let mut largest = 0.0f32;
        let mut seen = 0;
        for t in &out.triangles {
            for c in t {
                let p = out.positions[(c.position & !NEW_POSITION) as usize];
                let nearest = mesh
                    .positions
                    .iter()
                    .map(|q| (p - *q).length())
                    .fold(f32::MAX, f32::min);
                largest = largest.max(nearest);
                seen += 1;
            }
        }
        assert!(seen > 0);
        assert!(
            out.error > 0.0 && out.error >= largest - 1e-4,
            "{} {largest}",
            out.error
        );
    }

    #[test]
    fn a_blade_grows_across_its_narrow_side_first() {
        // Two equal blades, one survives: growth 2, so only the width doubles.
        let mut mesh = Mesh::default();
        mesh.quad(0.0, 1.0, 4.0);
        mesh.quad(10.0, 1.0, 4.0);
        let pieces = vec![vec![0, 1], vec![2, 3]];
        let out = thin(&mesh.triangles, &pieces, &mesh.positions, &[], 1);
        assert_eq!(out.triangles.len(), 2);
        let ys: Vec<f32> = out.positions.iter().map(|p| p.y).collect();
        let length = ys.iter().cloned().fold(f32::MIN, f32::max)
            - ys.iter().cloned().fold(f32::MAX, f32::min);
        assert!((length - 4.0).abs() < 1e-6, "length {length}");
        let xs: Vec<f32> = out.positions.iter().map(|p| p.x).collect();
        let width = xs.iter().cloned().fold(f32::MIN, f32::max)
            - xs.iter().cloned().fold(f32::MAX, f32::min);
        assert!((width - 2.0).abs() < 1e-5, "width {width}");
    }

    #[test]
    fn survivors_do_not_depend_on_the_order_of_the_list() {
        let (mesh, pieces) = quads(40);
        let mut reversed = pieces.clone();
        reversed.reverse();
        let key = |out: &Thinned| {
            let mut v: Vec<u32> = out
                .triangles
                .iter()
                .map(|t| {
                    (out.positions[(t[0].position & !NEW_POSITION) as usize].x / 10.0).round()
                        as u32
                })
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let a = thin(&mesh.triangles, &pieces, &mesh.positions, &[], 40);
        let b = thin(&mesh.triangles, &reversed, &mesh.positions, &[], 40);
        assert_eq!(key(&a).len(), key(&b).len());
        // Growth moves vertices, but quads stay near their slot: compare slot sets.
        assert_eq!(key(&a), key(&b));
    }

    #[test]
    fn a_target_at_the_total_keeps_everything_unmoved() {
        let (mesh, pieces) = quads(10);
        let out = thin(&mesh.triangles, &pieces, &mesh.positions, &[], 20);
        assert_eq!(out.triangles.len(), 20);
        assert_eq!(out.error, 0.0);
        let out = thin(&mesh.triangles, &pieces, &mesh.positions, &[], 500);
        assert_eq!(out.triangles.len(), 20);
        assert_eq!(out.error, 0.0);
    }

    #[test]
    fn thin_skips_unresolvable_ids() {
        let (mut mesh, mut pieces) = quads(3);
        let bad = Corner {
            position: 999,
            attribute: 77,
        };
        mesh.triangles.push([bad; 3]);
        pieces[0].extend([6, 50]);
        let out = thin(&mesh.triangles, &pieces, &mesh.positions, &[], 100);
        assert_eq!(out.triangles.len(), 6);
    }
}
