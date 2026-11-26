//! Whole-tree convex hull construction for native foliage (M7, contract m7-hull).
//!
//! [`build_foliage_hull`] turns every retained source-tree position into one closed,
//! outward-wound convex volume in source-root metres. The hull is a solid index mesh, not a
//! billboard: the native bake keeps the source silhouette and its holes through coverage.
//! Construction only reads [`FoliageSource`]; the retained triangles, normals, UVs,
//! materials and tags are left untouched.

use std::collections::{HashMap, HashSet};

use raa_core::RaaError;
use raa_geom::MeshData;

use crate::foliage_resources::FoliageSource;

type Point = [f64; 3];

#[derive(Clone, Copy)]
struct Face {
    vertices: [usize; 3],
    normal: Point,
    offset: f64,
}

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

fn finite3(v: [f32; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Every source position once, in ascending source-vertex order. A position repeated by
/// adjacent triangles keeps its first (lowest) index, which is the tie-break order the
/// hull uses.
fn collect_points(source: &FoliageSource) -> Result<Vec<Point>, RaaError> {
    if !finite3(source.min) || !finite3(source.max) {
        return Err(invalid("foliage source bounds must be finite"));
    }
    let corners = source
        .triangles
        .len()
        .checked_mul(3)
        .ok_or_else(|| invalid("foliage source has too many triangles for a hull"))?;
    let mut seen = HashSet::new();
    seen.try_reserve(corners).map_err(|_| {
        invalid(format!(
            "foliage hull could not reserve {corners} position key(s)"
        ))
    })?;
    let mut points = Vec::new();
    points.try_reserve_exact(corners).map_err(|_| {
        let bytes = corners.saturating_mul(std::mem::size_of::<Point>());
        invalid(format!("foliage hull could not allocate {bytes} bytes"))
    })?;
    for triangle in &source.triangles {
        for position in triangle.positions {
            if !finite3(position) {
                return Err(invalid("foliage source positions must be finite"));
            }
            let key = position.map(|v| {
                if v == 0.0 {
                    0.0f32.to_bits()
                } else {
                    v.to_bits()
                }
            });
            if seen.insert(key) {
                points.push(position.map(f64::from));
            }
        }
    }
    Ok(points)
}

/// A distance tolerance that covers the source extent and the `f32` output precision.
fn tolerance(points: &[Point]) -> f64 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut magnitude = 0.0_f64;
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
            magnitude = magnitude.max(point[axis].abs());
        }
    }
    let extent = (0..3).map(|axis| max[axis] - min[axis]).fold(0.0, f64::max);
    (extent * 1e-7).max(magnitude * 1e-7).max(1e-9)
}

/// The first four non-coplanar positions, chosen with strict comparisons so equal
/// distances keep the lower source index.
fn initial_tetrahedron(points: &[Point], eps: f64) -> Option<[usize; 4]> {
    if points.len() < 4 {
        return None;
    }
    let p0 = points[0];
    let mut i1 = 1;
    let mut best = -1.0;
    for (i, point) in points.iter().enumerate().skip(1) {
        let d = dot(sub(*point, p0), sub(*point, p0));
        if d > best {
            best = d;
            i1 = i;
        }
    }
    if best <= eps * eps {
        return None;
    }
    let e1 = sub(points[i1], p0);
    let e1_len2 = dot(e1, e1);
    let mut i2 = 0;
    let mut best = -1.0;
    for (i, point) in points.iter().enumerate() {
        if i == i1 {
            continue;
        }
        let c = cross(e1, sub(*point, p0));
        let d = dot(c, c) / e1_len2;
        if d > best {
            best = d;
            i2 = i;
        }
    }
    if best <= eps * eps {
        return None;
    }
    let plane = cross(e1, sub(points[i2], p0));
    let area2 = dot(plane, plane).sqrt();
    let mut i3 = 0;
    let mut best = -1.0;
    for (i, point) in points.iter().enumerate() {
        if i == 0 || i == i1 || i == i2 {
            continue;
        }
        let d = dot(plane, sub(*point, p0)).abs();
        if d > best {
            best = d;
            i3 = i;
        }
    }
    if best <= eps * area2 {
        return None;
    }
    Some([0, i1, i2, i3])
}

/// One face oriented so `interior` lies on its negative side (outward normal and winding).
fn face_of(points: &[Point], vertices: [usize; 3], interior: Point) -> Face {
    let mut vertices = vertices;
    let [a, b, c] = vertices;
    let mut normal = cross(sub(points[b], points[a]), sub(points[c], points[a]));
    let length = dot(normal, normal).sqrt();
    normal = if length > 0.0 {
        [normal[0] / length, normal[1] / length, normal[2] / length]
    } else {
        [0.0; 3]
    };
    if dot(normal, sub(interior, points[a])) > 0.0 {
        vertices.swap(1, 2);
        normal = [-normal[0], -normal[1], -normal[2]];
    }
    Face {
        vertices,
        normal,
        offset: dot(normal, points[vertices[0]]),
    }
}

/// Adds one position to the hull: drops visible faces, then fans the horizon to `index`.
fn insert_point(faces: &mut Vec<Face>, points: &[Point], interior: Point, index: usize, eps: f64) {
    let point = points[index];
    let visible: Vec<bool> = faces
        .iter()
        .map(|face| dot(face.normal, point) - face.offset > eps)
        .collect();
    if !visible.contains(&true) {
        return;
    }
    let mut edge_face = HashMap::with_capacity(faces.len() * 3);
    for (face_index, face) in faces.iter().enumerate() {
        for k in 0..3 {
            edge_face.insert((face.vertices[k], face.vertices[(k + 1) % 3]), face_index);
        }
    }
    let mut horizon = Vec::new();
    for (face_index, face) in faces.iter().enumerate() {
        if !visible[face_index] {
            continue;
        }
        for k in 0..3 {
            let a = face.vertices[k];
            let b = face.vertices[(k + 1) % 3];
            let neighbour_visible = edge_face
                .get(&(b, a))
                .map(|&neighbour| visible[neighbour])
                .unwrap_or(false);
            if !neighbour_visible {
                horizon.push((a, b));
            }
        }
    }
    let mut kept = Vec::with_capacity(faces.len() + horizon.len());
    for (face_index, face) in std::mem::take(faces).into_iter().enumerate() {
        if !visible[face_index] {
            kept.push(face);
        }
    }
    for (a, b) in horizon {
        kept.push(face_of(points, [b, a, index], interior));
    }
    *faces = kept;
}

/// Rotates every face to start at its lowest vertex index and sorts the faces, without
/// changing their winding.
fn canonical_order(faces: &mut [Face]) {
    for face in faces.iter_mut() {
        let vertices = face.vertices;
        let first = (0..3).min_by_key(|&k| vertices[k]).unwrap_or(0);
        face.vertices = [
            vertices[first],
            vertices[(first + 1) % 3],
            vertices[(first + 2) % 3],
        ];
    }
    faces.sort_by_key(|face| face.vertices);
}

/// Builds the closed convex hull of every retained source-tree position.
///
/// The result is indexed triangle-list [`MeshData`] in source-root metres with one
/// outward-wound face normal per triangle and no vertex colour. An empty, non-finite,
/// collinear or coplanar source (one without three-dimensional extent), or one whose
/// buffers cannot be allocated, is refused with `foliage.invalid`; this function never
/// panics. Repeated calls on one source return numerically equal meshes.
///
/// Handoff: 'Publish completed foliage assets in the renderer' (card #9135) calls
/// `build_foliage_hull` and uploads the returned mesh as the native foliage hull; today
/// this function constructs and returns that hull and leaves the source untouched. The
/// earlier handoff for the same downstream card is
/// `crates/raa-lit/src/foliage_resources.rs:301` `FoliageResources::new`.
pub fn build_foliage_hull(source: &FoliageSource) -> Result<MeshData, RaaError> {
    let points = collect_points(source)?;
    if points.is_empty() {
        return Err(invalid("foliage source is empty: a hull needs positions"));
    }
    let eps = tolerance(&points);
    let tetrahedron = initial_tetrahedron(&points, eps)
        .ok_or_else(|| invalid("foliage source must enclose a non-zero volume"))?;
    let interior = {
        let mut sum = [0.0; 3];
        for index in tetrahedron {
            let point = points[index];
            for axis in 0..3 {
                sum[axis] += point[axis];
            }
        }
        [sum[0] / 4.0, sum[1] / 4.0, sum[2] / 4.0]
    };
    let mut faces = Vec::with_capacity(4);
    for omitted in 0..4 {
        let vertices: Vec<usize> = tetrahedron
            .iter()
            .enumerate()
            .filter(|(slot, _)| *slot != omitted)
            .map(|(_, &index)| index)
            .collect();
        faces.push(face_of(
            &points,
            [vertices[0], vertices[1], vertices[2]],
            interior,
        ));
    }
    for index in 0..points.len() {
        if tetrahedron.contains(&index) {
            continue;
        }
        insert_point(&mut faces, &points, interior, index, eps);
    }
    for face in &mut faces {
        if dot(face.normal, sub(interior, points[face.vertices[0]])) > 0.0 {
            face.vertices.swap(1, 2);
            face.normal = [-face.normal[0], -face.normal[1], -face.normal[2]];
        }
    }
    canonical_order(&mut faces);

    let mut mesh = MeshData::new();
    let corners = faces.len().saturating_mul(3);
    let refused = |_| {
        invalid(format!(
            "foliage hull could not allocate {corners} corner(s)"
        ))
    };
    mesh.positions.try_reserve_exact(corners).map_err(refused)?;
    mesh.normals.try_reserve_exact(corners).map_err(refused)?;
    mesh.indices.try_reserve_exact(corners).map_err(refused)?;
    for face in faces {
        let normal = face.normal.map(|v| v as f32);
        let base = mesh.positions.len() as u32;
        for vertex in face.vertices {
            mesh.positions.push(points[vertex].map(|v| v as f32));
            mesh.normals.push(normal);
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    Ok(mesh)
}
