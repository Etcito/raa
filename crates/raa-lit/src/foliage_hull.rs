//! Whole-tree convex hull construction for native foliage (M7, contract m7-hull).
//!
//! [`build_foliage_hull`] turns every retained source-tree position into one closed,
//! outward-wound convex volume in source-root metres. The hull is a solid index mesh, not a
//! billboard: the native bake keeps the source silhouette and its holes through coverage.
//! Construction only reads [`FoliageSource`]; the retained triangles, normals, UVs,
//! materials and tags are left untouched.

use std::collections::HashMap;

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

/// The source corners read in place as hull input: corner `i` is position `i % 3` of
/// source triangle `i / 3`, widened to `f64` on read.
///
/// The hull used to copy every distinct position into a `Vec<[f64; 3]>` reserved for all
/// `3 x triangles` corners (24 B each) beside a `HashSet<[u32; 3]>` of position keys
/// reserved for the same count (hashbrown rounds `3 x triangles x 8 / 7` buckets up to a
/// power of two, 12 B key plus 1 control byte each). For the 2 609 340-triangle 41-part
/// sample tree that is 187 872 480 B of points plus 218 103 812 B of keys (16 777 216
/// buckets), about 406 MB on top of the 281 808 720 B source snapshot. Reading the corners
/// in place keeps the hull's working set to its faces and the [`FirstCorners`] set.
struct Corners<'a> {
    source: &'a FoliageSource,
    len: usize,
}

impl Corners<'_> {
    fn raw(&self, index: usize) -> [f32; 3] {
        self.source.triangles[index / 3].positions[index % 3]
    }

    fn get(&self, index: usize) -> Point {
        self.raw(index).map(f64::from)
    }
}

/// Validates the source and returns its corners. Every position must be finite.
fn corners(source: &FoliageSource) -> Result<Corners<'_>, RaaError> {
    if !finite3(source.min) || !finite3(source.max) {
        return Err(invalid("foliage source bounds must be finite"));
    }
    let len = source
        .triangles
        .len()
        .checked_mul(3)
        .ok_or_else(|| invalid("foliage source has too many triangles for a hull"))?;
    for triangle in &source.triangles {
        if !triangle.positions.iter().all(|&position| finite3(position)) {
            return Err(invalid("foliage source positions must be finite"));
        }
    }
    Ok(Corners { source, len })
}

/// Bytes of one [`FirstCorners`] slot: a `u32` corner index.
pub const HULL_SLOT_BYTES: usize = 4;

/// Host bytes of the hull's distinct-position set for `distinct` distinct source positions:
/// the power-of-two slot count of [`FirstCorners`] (at most half full) times
/// [`HULL_SLOT_BYTES`]. While the set grows the previous table (half this size) is live
/// beside it until the rehash ends. `None` when the size overflows.
pub fn hull_set_bytes(distinct: usize) -> Option<usize> {
    distinct
        .max(8)
        .checked_mul(2)?
        .checked_next_power_of_two()?
        .checked_mul(HULL_SLOT_BYTES)
}

/// The set of first-occurrence corners: an open-addressing table of `u32` corner indices
/// whose key is the corner's exact position (`-0.0` and `0.0` equal), read from the source
/// on demand. It replaces the former `HashSet<[u32; 3]>` of position keys and the
/// `Vec<[f64; 3]>` of distinct points: a slot is 4 B instead of 12 B plus a control byte
/// and 24 B of copied point, and the table grows with the distinct positions (fallibly,
/// one doubling at a time) instead of being reserved for every corner up front.
///
/// The hull still inserts exactly the first corner of each distinct position, in source
/// order, so its faces and their order are those of the former distinct-point list (the
/// corner indices of first occurrences rise with the former point indices).
struct FirstCorners {
    slots: Vec<u32>,
    len: usize,
}

impl FirstCorners {
    const EMPTY: u32 = u32::MAX;

    fn key(position: [f32; 3]) -> [u32; 3] {
        position.map(|v| {
            if v == 0.0 {
                0.0f32.to_bits()
            } else {
                v.to_bits()
            }
        })
    }

    fn hash(key: [u32; 3]) -> usize {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for word in key {
            hash = (hash ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        (hash ^ (hash >> 29)) as usize
    }

    fn table(slots: usize) -> Result<Vec<u32>, RaaError> {
        let mut table = Vec::new();
        table.try_reserve_exact(slots).map_err(|_| {
            invalid(format!(
                "foliage hull could not allocate {} bytes",
                slots.saturating_mul(HULL_SLOT_BYTES)
            ))
        })?;
        table.resize(slots, Self::EMPTY);
        Ok(table)
    }

    fn new() -> Result<Self, RaaError> {
        Ok(FirstCorners {
            slots: Self::table(16)?,
            len: 0,
        })
    }

    /// The slot holding a corner at position `key`, or the empty slot where it belongs.
    fn find(slots: &[u32], points: &Corners<'_>, key: [u32; 3]) -> usize {
        let mask = slots.len() - 1;
        let mut slot = Self::hash(key) & mask;
        loop {
            let held = slots[slot];
            if held == Self::EMPTY || Self::key(points.raw(held as usize)) == key {
                return slot;
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Records corner `index` and reports whether it is the first corner at its position.
    fn first(&mut self, points: &Corners<'_>, index: usize) -> Result<bool, RaaError> {
        let key = Self::key(points.raw(index));
        let slot = Self::find(&self.slots, points, key);
        if self.slots[slot] != Self::EMPTY {
            return Ok(false);
        }
        if (self.len + 1) * 2 > self.slots.len() {
            // The table of `len + 1` positions: twice the current one.
            let doubled = hull_set_bytes(self.len + 1)
                .ok_or_else(|| invalid("foliage hull position set overflows"))?
                / HULL_SLOT_BYTES;
            let mut grown = Self::table(doubled)?;
            for &held in self.slots.iter().filter(|&&held| held != Self::EMPTY) {
                let at = Self::find(&grown, points, Self::key(points.raw(held as usize)));
                grown[at] = held;
            }
            self.slots = grown;
        }
        let slot = Self::find(&self.slots, points, key);
        self.slots[slot] = u32::try_from(index)
            .ok()
            .filter(|&index| index != Self::EMPTY)
            .ok_or_else(|| invalid("foliage source has too many corners for a hull"))?;
        self.len += 1;
        Ok(true)
    }
}

/// A distance tolerance that covers the source extent and the `f32` output precision.
fn tolerance(points: &Corners<'_>) -> f64 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut magnitude = 0.0_f64;
    for index in 0..points.len {
        let point = points.get(index);
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
fn initial_tetrahedron(points: &Corners<'_>, eps: f64) -> Option<[usize; 4]> {
    if points.len < 4 {
        return None;
    }
    let p0 = points.get(0);
    let mut i1 = 1;
    let mut best = -1.0;
    for i in 1..points.len {
        let point = points.get(i);
        let d = dot(sub(point, p0), sub(point, p0));
        if d > best {
            best = d;
            i1 = i;
        }
    }
    if best <= eps * eps {
        return None;
    }
    let e1 = sub(points.get(i1), p0);
    let e1_len2 = dot(e1, e1);
    let mut i2 = 0;
    let mut best = -1.0;
    for i in 0..points.len {
        if i == i1 {
            continue;
        }
        let c = cross(e1, sub(points.get(i), p0));
        let d = dot(c, c) / e1_len2;
        if d > best {
            best = d;
            i2 = i;
        }
    }
    if best <= eps * eps {
        return None;
    }
    let plane = cross(e1, sub(points.get(i2), p0));
    let area2 = dot(plane, plane).sqrt();
    let mut i3 = 0;
    let mut best = -1.0;
    for i in 0..points.len {
        if i == 0 || i == i1 || i == i2 {
            continue;
        }
        let d = dot(plane, sub(points.get(i), p0)).abs();
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
fn face_of(points: &Corners<'_>, vertices: [usize; 3], interior: Point) -> Face {
    let mut vertices = vertices;
    let [a, b, c] = vertices;
    let origin = points.get(a);
    let mut normal = cross(sub(points.get(b), origin), sub(points.get(c), origin));
    let length = dot(normal, normal).sqrt();
    normal = if length > 0.0 {
        [normal[0] / length, normal[1] / length, normal[2] / length]
    } else {
        [0.0; 3]
    };
    if dot(normal, sub(interior, origin)) > 0.0 {
        vertices.swap(1, 2);
        normal = [-normal[0], -normal[1], -normal[2]];
    }
    Face {
        vertices,
        normal,
        offset: dot(normal, points.get(vertices[0])),
    }
}

fn reserve<T>(list: &mut Vec<T>, count: usize, what: &str) -> Result<(), RaaError> {
    list.try_reserve_exact(count).map_err(|_| {
        invalid(format!(
            "foliage hull could not allocate {} bytes of {what}",
            count.saturating_mul(std::mem::size_of::<T>())
        ))
    })
}

/// The faces removed for a point: the faces it sees, normalized so their union is one disc
/// whose boundary (the horizon) is a single simple cycle. `seen` holds the faces farther
/// than the tolerance in front of the point and `seed` the farthest of them.
///
/// For a point outside a convex hull the seen faces already form such a disc and are
/// returned unchanged. Near-coplanar faces judged with the tolerance can instead give
/// several patches, a patch with a hole, or two patches touching at one vertex; the former
/// insertion then fanned a horizon that was not a simple cycle, left edges shared by more
/// than two faces and multiplied the faces without bound (a 6 000-triangle curved-leaf
/// source reached 4 139 279 faces after 3 250 points). Here only the patch of `seed` is
/// kept, every hole of it is filled (all non-removed faces but the largest connected
/// group), the faces around a vertex the horizon passes twice are removed, and the face
/// across a horizon edge `(a, b)` whose fan face would face the interior (`inverted`; the
/// former insertion then wound that face against its neighbours) is removed, until the
/// horizon is simple and every fan face faces outward. A vertex absorbed that way lies
/// within the tolerance band of its neighbouring faces. `None` when no face would remain.
fn removed_faces(
    faces: &[Face],
    neighbours: &[[usize; 3]],
    seen: &[bool],
    seed: usize,
    inverted: impl Fn(usize, usize) -> bool,
) -> Result<Option<Vec<bool>>, RaaError> {
    let count = faces.len();
    let mut removed = Vec::new();
    reserve(&mut removed, count, "face flags")?;
    removed.resize(count, false);
    let mut stack = Vec::new();
    reserve(&mut stack, count, "face stack")?;
    removed[seed] = true;
    stack.push(seed);
    while let Some(face) = stack.pop() {
        for &next in &neighbours[face] {
            if next < count && seen[next] && !removed[next] {
                removed[next] = true;
                stack.push(next);
            }
        }
    }
    let mut group = Vec::new();
    reserve(&mut group, count, "face groups")?;
    let mut uses = Vec::new();
    loop {
        // Fill the holes: keep only the largest connected group of remaining faces.
        group.clear();
        group.resize(count, usize::MAX);
        let mut sizes: Vec<usize> = Vec::new();
        for start in 0..count {
            if removed[start] || group[start] != usize::MAX {
                continue;
            }
            let label = sizes.len();
            let mut size = 0usize;
            group[start] = label;
            stack.push(start);
            while let Some(face) = stack.pop() {
                size += 1;
                for &next in &neighbours[face] {
                    if next < count && !removed[next] && group[next] == usize::MAX {
                        group[next] = label;
                        stack.push(next);
                    }
                }
            }
            sizes.push(size);
        }
        let Some(largest) =
            (0..sizes.len()).max_by_key(|&label| (sizes[label], usize::MAX - label))
        else {
            return Ok(None);
        };
        for face in 0..count {
            if !removed[face] && group[face] != largest {
                removed[face] = true;
            }
        }
        // A vertex starting two horizon edges pinches the disc: remove the faces around it.
        uses.clear();
        for face in 0..count {
            if !removed[face] {
                continue;
            }
            for k in 0..3 {
                let next = neighbours[face][k];
                if next >= count || !removed[next] {
                    uses.push(faces[face].vertices[k]);
                }
            }
        }
        uses.sort_unstable();
        let pinched: Vec<usize> = uses
            .windows(2)
            .filter(|pair| pair[0] == pair[1])
            .map(|pair| pair[0])
            .collect();
        if !pinched.is_empty() {
            for face in 0..count {
                if faces[face]
                    .vertices
                    .iter()
                    .any(|vertex| pinched.binary_search(vertex).is_ok())
                {
                    removed[face] = true;
                }
            }
            continue;
        }
        // A fan face that would face the interior: absorb the face across its edge.
        let mut grew = false;
        for face in 0..count {
            if !removed[face] {
                continue;
            }
            for k in 0..3 {
                let next = neighbours[face][k];
                let (a, b) = (faces[face].vertices[k], faces[face].vertices[(k + 1) % 3]);
                if next < count && !removed[next] && inverted(a, b) {
                    removed[next] = true;
                    grew = true;
                }
            }
        }
        if !grew {
            return Ok(Some(removed));
        }
    }
}

/// Adds one position to the hull: drops the faces it sees ([`removed_faces`]), then fans
/// the horizon to `index`. A point no face sees (inside or on the hull) returns before any
/// allocation; a refused allocation is `foliage.invalid`.
fn insert_point(
    faces: &mut Vec<Face>,
    points: &Corners<'_>,
    interior: Point,
    index: usize,
    eps: f64,
) -> Result<(), RaaError> {
    let point = points.get(index);
    let distance = |face: &Face| dot(face.normal, point) - face.offset;
    let mut seed = None;
    let mut farthest = eps;
    for (face_index, face) in faces.iter().enumerate() {
        let d = distance(face);
        if d > farthest {
            farthest = d;
            seed = Some(face_index);
        }
    }
    let Some(seed) = seed else {
        return Ok(());
    };
    let mut seen = Vec::new();
    reserve(&mut seen, faces.len(), "face flags")?;
    seen.extend(faces.iter().map(|face| distance(face) > eps));
    let mut edge_face = HashMap::new();
    edge_face
        .try_reserve(faces.len().saturating_mul(3))
        .map_err(|_| invalid("foliage hull could not allocate its edge map"))?;
    for (face_index, face) in faces.iter().enumerate() {
        for k in 0..3 {
            edge_face.insert((face.vertices[k], face.vertices[(k + 1) % 3]), face_index);
        }
    }
    let mut neighbours = Vec::new();
    reserve(&mut neighbours, faces.len(), "face neighbours")?;
    neighbours.extend(faces.iter().map(|face| {
        [0, 1, 2].map(|k| {
            let (a, b) = (face.vertices[k], face.vertices[(k + 1) % 3]);
            edge_face.get(&(b, a)).copied().unwrap_or(usize::MAX)
        })
    }));
    drop(edge_face);
    // The fan face of horizon edge `(a, b)` is `[b, index, a]` (`face_of` turns
    // `[b, a, index]` to it); it is inverted when that winding faces the interior.
    let inverted = |a: usize, b: usize| {
        let origin = points.get(b);
        let normal = cross(sub(point, origin), sub(points.get(a), origin));
        dot(normal, sub(interior, origin)) >= 0.0
    };
    let Some(visible) = removed_faces(faces, &neighbours, &seen, seed, inverted)? else {
        return Ok(());
    };
    let mut horizon = Vec::new();
    for (face_index, face) in faces.iter().enumerate() {
        if !visible[face_index] {
            continue;
        }
        for k in 0..3 {
            let next = neighbours[face_index][k];
            if next >= faces.len() || !visible[next] {
                horizon.push((face.vertices[k], face.vertices[(k + 1) % 3]));
            }
        }
    }
    let mut kept = Vec::new();
    reserve(&mut kept, faces.len() + horizon.len(), "faces")?;
    for (face_index, face) in std::mem::take(faces).into_iter().enumerate() {
        if !visible[face_index] {
            kept.push(face);
        }
    }
    for (a, b) in horizon {
        kept.push(face_of(points, [b, a, index], interior));
    }
    *faces = kept;
    Ok(())
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
/// The corners are read in place from the source ([`Corners`]); besides the faces, the
/// only working buffer is the [`FirstCorners`] set of 4 B slots ([`hull_set_bytes`]),
/// sized by the distinct positions rather than reserved for every corner.
///
/// Handoff: 'Publish completed foliage assets in the renderer' (card #9135) calls
/// `build_foliage_hull` and uploads the returned mesh as the native foliage hull; today
/// this function constructs and returns that hull and leaves the source untouched. The
/// earlier handoff for the same downstream card is
/// `crates/raa-lit/src/foliage_resources.rs:301` `FoliageResources::new`.
pub fn build_foliage_hull(source: &FoliageSource) -> Result<MeshData, RaaError> {
    let points = corners(source)?;
    if points.len == 0 {
        return Err(invalid("foliage source is empty: a hull needs positions"));
    }
    let eps = tolerance(&points);
    let tetrahedron = initial_tetrahedron(&points, eps)
        .ok_or_else(|| invalid("foliage source must enclose a non-zero volume"))?;
    let interior = {
        let mut sum = [0.0; 3];
        for index in tetrahedron {
            let point = points.get(index);
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
    let mut seen = FirstCorners::new()?;
    for index in 0..points.len {
        if !seen.first(&points, index)? || tetrahedron.contains(&index) {
            continue;
        }
        insert_point(&mut faces, &points, interior, index, eps)?;
    }
    for face in &mut faces {
        if dot(face.normal, sub(interior, points.get(face.vertices[0]))) > 0.0 {
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
            mesh.positions.push(points.get(vertex).map(|v| v as f32));
            mesh.normals.push(normal);
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foliage_resources::SourceTriangle;
    use std::collections::HashSet;

    fn next(state: &mut u32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state >> 8) as f32 / 16_777_216.0
    }

    /// Curved square leaves of shared grid corners scattered through a crown: straight
    /// rows of collinear corners and near-coplanar patches, the input that made the former
    /// insertion multiply its faces without bound.
    fn curved_leaves(leaves: usize, grid: usize, seed: u32) -> FoliageSource {
        let mut state = seed | 1;
        let mut triangles = Vec::new();
        for _ in 0..leaves {
            let centre = [
                next(&mut state) * 8.0 - 4.0,
                next(&mut state) * 6.0 + 3.0,
                next(&mut state) * 8.0 - 4.0,
            ];
            let mut corners = Vec::new();
            for i in 0..=grid {
                for j in 0..=grid {
                    corners.push([
                        centre[0] + i as f32 * 0.03,
                        centre[1] + j as f32 * 0.03 + (i * j) as f32 * 0.001,
                        centre[2]
                            + 0.02 * ((i * i + j * j) as f32).sqrt()
                            + next(&mut state) * 0.001,
                    ]);
                }
            }
            for i in 0..grid {
                for j in 0..grid {
                    let a = i * (grid + 1) + j;
                    let (b, c, d) = (a + 1, a + grid + 1, a + grid + 2);
                    for positions in [
                        [corners[a], corners[b], corners[d]],
                        [corners[a], corners[d], corners[c]],
                    ] {
                        triangles.push(SourceTriangle {
                            positions,
                            normals: [[0.0, 0.0, 1.0]; 3],
                            uv: [[0.0, 0.0]; 3],
                            material: 0,
                            source_tag: 1,
                            foliage: true,
                        });
                    }
                }
            }
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for triangle in &triangles {
            for position in triangle.positions {
                for axis in 0..3 {
                    min[axis] = min[axis].min(position[axis]);
                    max[axis] = max[axis].max(position[axis]);
                }
            }
        }
        FoliageSource {
            triangles,
            min,
            max,
            root_world: raa_math::Mat4::IDENTITY,
            generation: 1,
        }
    }

    /// Every directed edge once and its reverse present: a closed, consistently wound
    /// surface whose face count is bounded by its vertices.
    #[test]
    fn degenerate_leaves_keep_a_closed_bounded_hull() {
        for seed in [10, 12, 15, 25] {
            let source = curved_leaves(50 + seed as usize * 7, 3 + seed as usize % 4, seed);
            let mesh = build_foliage_hull(&source).expect("curved leaf hull");
            let key = |index: u32| mesh.positions[index as usize].map(f32::to_bits);
            let mut edges = HashSet::new();
            for face in mesh.indices.as_chunks::<3>().0 {
                for k in 0..3 {
                    let edge = (key(face[k]), key(face[(k + 1) % 3]));
                    assert!(edges.insert(edge), "seed {seed}: a directed edge repeats");
                }
            }
            for &(a, b) in &edges {
                assert!(edges.contains(&(b, a)), "seed {seed}: an open edge");
            }
            let vertices: HashSet<_> = (0..mesh.positions.len() as u32).map(key).collect();
            assert_eq!(
                mesh.indices.len() / 3,
                2 * vertices.len() - 4,
                "seed {seed}"
            );
        }
    }
}
