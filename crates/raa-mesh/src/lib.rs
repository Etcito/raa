//! Attributed meshes for the lit renderer: S13 validation and JS array intake (S10, S13).

use std::fmt;

use raa_core::MaterialId;
use raa_geom::MeshData;
use raa_math::Aabb;

/// Every S13 refusal raa-mesh can produce. `code` is stable (`mesh.<snake_case>`);
/// `message` names the array or index at fault.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshError {
    pub code: &'static str,
    pub message: String,
}

impl MeshError {
    fn new(code: &'static str, message: String) -> MeshError {
        MeshError { code, message }
    }

    /// Stable machine-readable code of the form `mesh.<snake_case>`.
    pub fn code(&self) -> &'static str {
        self.code
    }
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for MeshError {}

/// A contiguous index range drawn with one material.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SubMesh {
    /// Offset into the index array, a multiple of 3.
    pub first_index: u32,
    /// Number of indices, a multiple of 3 and > 0.
    pub index_count: u32,
    /// Material of the range; `None` uses the instance's material.
    pub material: Option<MaterialId>,
}

/// Vertex colours as handed over by JS (S13): linear rgba floats or sRGB rgba8.
#[derive(Copy, Clone, Debug)]
pub enum Colors<'a> {
    Linear(&'a [f32]),
    Srgb8(&'a [u8]),
}

/// An S2 mesh plus the optional S10 attributes and its sub-meshes.
///
/// After [`AttributedMesh::validate`] succeeds every attribute array has one entry per
/// vertex, every value is finite, indices are in range, normals and tangent xyz are
/// non-zero, tangent w is +1 or -1 and sub-meshes are valid, disjoint index ranges.
#[derive(Clone, Debug, Default)]
pub struct AttributedMesh {
    pub base: MeshData,
    pub uvs: Option<Vec<[f32; 2]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Linear rgba.
    pub colors: Option<Vec<[f32; 4]>>,
    pub submeshes: Vec<SubMesh>,
}

impl AttributedMesh {
    /// Wraps `m` with one sub-mesh over all its indices and no attributes.
    pub fn from_mesh(m: MeshData) -> AttributedMesh {
        let index_count = u32::try_from(m.indices.len()).unwrap_or(u32::MAX);
        AttributedMesh {
            base: m,
            uvs: None,
            tangents: None,
            colors: None,
            submeshes: vec![SubMesh {
                first_index: 0,
                index_count,
                material: None,
            }],
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.base.vertex_count()
    }

    pub fn triangle_count(&self) -> usize {
        self.base.triangle_count()
    }

    pub fn bounds(&self) -> Aabb {
        self.base.bounds()
    }

    /// Checks the S13 rules in a fixed order (empty, lengths, finiteness, index range,
    /// normals, tangents, sub-mesh ranges, sub-mesh overlap); the first broken rule is
    /// returned.
    pub fn validate(&self) -> Result<(), MeshError> {
        let vertex_count = self.base.positions.len();
        let indices = &self.base.indices;
        if vertex_count == 0 || indices.is_empty() {
            return Err(MeshError::new(
                "mesh.empty",
                format!(
                    "mesh has {vertex_count} vertices and {} indices; both must be non-zero",
                    indices.len()
                ),
            ));
        }
        check_count("normals", self.base.normals.len(), vertex_count)?;
        if let Some(uvs) = &self.uvs {
            check_count("uvs", uvs.len(), vertex_count)?;
        }
        if let Some(tangents) = &self.tangents {
            check_count("tangents", tangents.len(), vertex_count)?;
        }
        if let Some(colors) = &self.colors {
            check_count("colors", colors.len(), vertex_count)?;
        }
        if !indices.len().is_multiple_of(3) {
            return Err(MeshError::new(
                "mesh.length_mismatch",
                format!(
                    "indices length {} is not a multiple of 3 (triangle list)",
                    indices.len()
                ),
            ));
        }

        check_finite("positions", &self.base.positions)?;
        check_finite("normals", &self.base.normals)?;
        if let Some(uvs) = &self.uvs {
            check_finite("uvs", uvs)?;
        }
        if let Some(tangents) = &self.tangents {
            check_finite("tangents", tangents)?;
        }
        if let Some(colors) = &self.colors {
            check_finite("colors", colors)?;
        }

        if let Some((offset, index)) = indices
            .iter()
            .enumerate()
            .find(|(_, i)| **i as usize >= vertex_count)
        {
            return Err(MeshError::new(
                "mesh.index_range",
                format!("indices[{offset}] = {index} is out of range for {vertex_count} vertices"),
            ));
        }

        if let Some(i) = self.base.normals.iter().position(|n| length(n) == 0.0) {
            return Err(MeshError::new(
                "mesh.invalid_normal",
                format!("normals[{i}] has zero length"),
            ));
        }

        if let Some(tangents) = &self.tangents {
            for (i, t) in tangents.iter().enumerate() {
                if length(&[t[0], t[1], t[2]]) == 0.0 {
                    return Err(MeshError::new(
                        "mesh.invalid_tangent",
                        format!("tangents[{i}] has a zero-length xyz"),
                    ));
                }
                if t[3] != 1.0 && t[3] != -1.0 {
                    return Err(MeshError::new(
                        "mesh.invalid_tangent",
                        format!("tangents[{i}].w = {} is not +1 or -1", t[3]),
                    ));
                }
            }
        }

        check_submeshes(&self.submeshes, indices.len())
    }
}

/// Builds and validates a mesh from the flat JS arrays of an S13 `MeshDesc`.
///
/// `positions`/`normals` hold 3 floats per vertex, `uvs` 2, `tangents` 4 and `colors`
/// 4 (linear floats or sRGB rgba8, converted with the exact sRGB EOTF; alpha / 255 stays
/// linear). Empty `submeshes` means one sub-mesh over all indices. Normals and tangent
/// xyz are stored normalised.
pub fn from_js_arrays(
    positions: &[f32],
    normals: &[f32],
    uvs: Option<&[f32]>,
    tangents: Option<&[f32]>,
    colors: Option<Colors<'_>>,
    indices: &[u32],
    submeshes: &[SubMesh],
) -> Result<AttributedMesh, MeshError> {
    if positions.is_empty() || indices.is_empty() {
        return Err(MeshError::new(
            "mesh.empty",
            format!(
                "positions has {} floats and indices has {} entries; both must be non-zero",
                positions.len(),
                indices.len()
            ),
        ));
    }
    if !positions.len().is_multiple_of(3) {
        return Err(MeshError::new(
            "mesh.length_mismatch",
            format!(
                "positions length {} is not a multiple of 3",
                positions.len()
            ),
        ));
    }
    let vertex_count = positions.len() / 3;
    check_flat_len("normals", normals.len(), 3, vertex_count)?;
    if let Some(uvs) = uvs {
        check_flat_len("uvs", uvs.len(), 2, vertex_count)?;
    }
    if let Some(tangents) = tangents {
        check_flat_len("tangents", tangents.len(), 4, vertex_count)?;
    }
    match colors {
        Some(Colors::Linear(c)) => check_flat_len("colors", c.len(), 4, vertex_count)?,
        Some(Colors::Srgb8(c)) => check_flat_len("colors", c.len(), 4, vertex_count)?,
        None => {}
    }

    let index_count = u32::try_from(indices.len()).map_err(|_| {
        MeshError::new(
            "mesh.length_mismatch",
            format!("indices length {} exceeds u32 range", indices.len()),
        )
    })?;
    let submeshes = if submeshes.is_empty() {
        vec![SubMesh {
            first_index: 0,
            index_count,
            material: None,
        }]
    } else {
        submeshes.to_vec()
    };

    let mut mesh = AttributedMesh {
        base: MeshData {
            positions: to_arrays(positions),
            normals: to_arrays(normals),
            indices: indices.to_vec(),
        },
        uvs: uvs.map(to_arrays),
        tangents: tangents.map(to_arrays),
        colors: colors.map(|c| match c {
            Colors::Linear(c) => to_arrays(c),
            Colors::Srgb8(c) => c
                .chunks_exact(4)
                .map(|px| {
                    let mut out = [0.0; 4];
                    for (o, v) in out.iter_mut().zip(px) {
                        *o = srgb8_to_linear(*v);
                    }
                    if let (Some(a), Some(v)) = (out.get_mut(3), px.get(3)) {
                        *a = f32::from(*v) / 255.0;
                    }
                    out
                })
                .collect(),
        }),
        submeshes,
    };
    mesh.validate()?;

    for n in &mut mesh.base.normals {
        *n = normalize(*n);
    }
    if let Some(tangents) = &mut mesh.tangents {
        for t in tangents.iter_mut() {
            let [x, y, z] = normalize([t[0], t[1], t[2]]);
            *t = [x, y, z, t[3]];
        }
    }
    Ok(mesh)
}

/// Exact sRGB EOTF (IEC 61966-2-1) of an 8-bit channel, evaluated in f64.
fn srgb8_to_linear(v: u8) -> f32 {
    let c = f64::from(v) / 255.0;
    let linear = if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    };
    linear as f32
}

fn to_arrays<const N: usize>(flat: &[f32]) -> Vec<[f32; N]> {
    flat.chunks_exact(N)
        .map(|chunk| {
            let mut out = [0.0; N];
            out.copy_from_slice(chunk);
            out
        })
        .collect()
}

/// Euclidean length in f64 so tiny (subnormal) vectors are not mistaken for zero.
fn length(v: &[f32; 3]) -> f64 {
    v.iter()
        .map(|c| f64::from(*c) * f64::from(*c))
        .sum::<f64>()
        .sqrt()
}

/// Normalises a non-zero vector (callers checked the length).
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(&v);
    v.map(|c| (f64::from(c) / len) as f32)
}

fn check_flat_len(
    name: &str,
    len: usize,
    per_vertex: usize,
    vertex_count: usize,
) -> Result<(), MeshError> {
    let expected = per_vertex * vertex_count;
    if len == expected {
        return Ok(());
    }
    Err(MeshError::new(
        "mesh.length_mismatch",
        format!(
            "{name} length {len} does not match {per_vertex} x {vertex_count} vertices = {expected}"
        ),
    ))
}

fn check_count(name: &str, len: usize, vertex_count: usize) -> Result<(), MeshError> {
    if len == vertex_count {
        return Ok(());
    }
    Err(MeshError::new(
        "mesh.length_mismatch",
        format!("{name} has {len} entries for {vertex_count} vertices"),
    ))
}

fn check_finite<const N: usize>(name: &str, values: &[[f32; N]]) -> Result<(), MeshError> {
    for (i, v) in values.iter().enumerate() {
        if let Some(c) = v.iter().position(|x| !x.is_finite()) {
            return Err(MeshError::new(
                "mesh.not_finite",
                format!("{name}[{i}][{c}] (flat index {}) is not finite", i * N + c),
            ));
        }
    }
    Ok(())
}

fn check_submeshes(submeshes: &[SubMesh], index_count: usize) -> Result<(), MeshError> {
    if submeshes.is_empty() {
        return Err(MeshError::new(
            "mesh.submesh_range",
            "submeshes is empty; at least one sub-mesh is required".to_string(),
        ));
    }
    for (i, s) in submeshes.iter().enumerate() {
        let end = u64::from(s.first_index) + u64::from(s.index_count);
        let reason = if !s.first_index.is_multiple_of(3) {
            Some("first is not a multiple of 3")
        } else if s.index_count == 0 {
            Some("count is 0")
        } else if !s.index_count.is_multiple_of(3) {
            Some("count is not a multiple of 3")
        } else if end > index_count as u64 {
            Some("range ends past the index array")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(MeshError::new(
                "mesh.submesh_range",
                format!(
                    "subMeshes[{i}] (first {}, count {}) is invalid for {index_count} indices: {reason}",
                    s.first_index, s.index_count
                ),
            ));
        }
    }
    let mut ranges: Vec<(u64, u64, usize)> = submeshes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let first = u64::from(s.first_index);
            (first, first + u64::from(s.index_count), i)
        })
        .collect();
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if let [(a_first, a_end, a), (b_first, b_end, b)] = pair
            && b_first < a_end
        {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            return Err(MeshError::new(
                "mesh.submesh_overlap",
                format!(
                    "subMeshes[{lo}] and subMeshes[{hi}] overlap (indices {a_first}..{a_end} and {b_first}..{b_end})"
                ),
            ));
        }
    }
    Ok(())
}
