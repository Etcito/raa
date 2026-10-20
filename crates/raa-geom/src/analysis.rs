//! Mesh analysis: welded topology, area, volume and normal checks (S2, WELD_EPS, NORMAL_RULE).

use std::collections::{BTreeMap, BTreeSet};

use crate::MeshData;

/// Relative weld tolerance: eps = WELD_EPS_FACTOR x max(1, bbox diagonal).
const WELD_EPS_FACTOR: f64 = 1e-6;
/// NORMAL_RULE: allowed abs(len - 1) of a vertex normal.
const NORMAL_LENGTH_TOL: f32 = 1e-4;
/// NORMAL_RULE: allowed angle between a vertex normal and its incident face normals' mean.
const NORMAL_DEVIATION_MAX_DEG: f32 = 45.0;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Result of [`analyze`].
#[derive(Clone, Debug, PartialEq)]
pub struct MeshReport {
    pub vertices: u32,
    /// Positions after welding with WELD_EPS.
    pub unique_positions: u32,
    pub triangles: u32,
    /// `boundary_edges`, `nonmanifold_edges` and `inconsistent_edges` are all 0.
    pub closed: bool,
    /// Welded undirected edges used by exactly one triangle.
    pub boundary_edges: u32,
    /// Welded undirected edges used by more than two triangles.
    pub nonmanifold_edges: u32,
    /// Welded undirected edges used by two triangles in the same direction.
    pub inconsistent_edges: u32,
    /// Triangles with welded-duplicate corners or with area < eps^2.
    pub degenerate_triangles: u32,
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
    pub area: f64,
    /// Signed volume (divergence theorem); > 0 means outward winding.
    pub volume: f64,
    pub max_normal_deviation_deg: f32,
    pub bad_normal_lengths: u32,
    /// Vertices whose normal has a negative dot with an incident face normal.
    pub inward_normals: u32,
}

fn to_f64(p: [f32; 3]) -> [f64; 3] {
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Welds positions within `eps` through a quantised grid; ids are given in first-seen order.
fn weld(positions: &[[f32; 3]], eps: f64) -> (Vec<u32>, u32) {
    let mut grid: BTreeMap<[i64; 3], Vec<u32>> = BTreeMap::new();
    let mut representatives: Vec<[f64; 3]> = Vec::new();
    let mut ids = Vec::with_capacity(positions.len());
    for p in positions {
        let p = to_f64(*p);
        let cell = [
            (p[0] / eps).floor() as i64,
            (p[1] / eps).floor() as i64,
            (p[2] / eps).floor() as i64,
        ];
        let mut found = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    let Some(candidates) = grid.get(&key) else {
                        continue;
                    };
                    for &id in candidates {
                        let close = representatives
                            .get(id as usize)
                            .is_some_and(|r| length(sub(*r, p)) <= eps);
                        if close && found.is_none_or(|f| id < f) {
                            found = Some(id);
                        }
                    }
                }
            }
        }
        let id = match found {
            Some(id) => id,
            None => {
                let id = representatives.len() as u32;
                representatives.push(p);
                grid.entry(cell).or_default().push(id);
                id
            }
        };
        ids.push(id);
    }
    (ids, representatives.len() as u32)
}

/// Analyses `mesh` (S2 semantics). Triangles with an out-of-range index are counted as
/// degenerate and otherwise ignored, so the analysis never panics on invalid data.
pub fn analyze(mesh: &MeshData) -> MeshReport {
    let positions = &mesh.positions;
    let mut bbox_min = [f64::INFINITY; 3];
    let mut bbox_max = [f64::NEG_INFINITY; 3];
    for p in positions {
        let p = to_f64(*p);
        for axis in 0..3 {
            bbox_min[axis] = bbox_min[axis].min(p[axis]);
            bbox_max[axis] = bbox_max[axis].max(p[axis]);
        }
    }
    if positions.is_empty() {
        bbox_min = [0.0; 3];
        bbox_max = [0.0; 3];
    }
    let diagonal = length(sub(bbox_max, bbox_min));
    let eps = WELD_EPS_FACTOR * diagonal.max(1.0);
    let (welded, unique_positions) = weld(positions, eps);

    // Undirected welded edge (low, high) -> (uses, uses in the low -> high direction).
    let mut edges: BTreeMap<(u32, u32), (u32, u32)> = BTreeMap::new();
    let mut face_sum = vec![[0.0f64; 3]; positions.len()];
    let mut inward = vec![false; positions.len()];
    let mut degenerate_triangles = 0u32;
    let mut area = 0.0f64;
    let mut volume = 0.0f64;

    for tri in mesh.indices.chunks_exact(3) {
        let idx = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let (Some(p0), Some(p1), Some(p2)) = (
            positions.get(idx[0]),
            positions.get(idx[1]),
            positions.get(idx[2]),
        ) else {
            degenerate_triangles += 1;
            continue;
        };
        let (p0, p1, p2) = (to_f64(*p0), to_f64(*p1), to_f64(*p2));
        // Unnormalised face normal; its length is twice the triangle area.
        let n = cross(sub(p1, p0), sub(p2, p0));
        let tri_area = 0.5 * length(n);
        area += tri_area;
        volume += dot(p0, cross(p1, p2)) / 6.0;

        // Indices were range-checked through `positions.get` above.
        let w = [welded[idx[0]], welded[idx[1]], welded[idx[2]]];
        let duplicate = w[0] == w[1] || w[1] == w[2] || w[0] == w[2];
        if duplicate || tri_area < eps * eps {
            degenerate_triangles += 1;
        }
        if !duplicate {
            for k in 0..3 {
                let (a, b) = (w[k], w[(k + 1) % 3]);
                let entry = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                entry.0 += 1;
                if a < b {
                    entry.1 += 1;
                }
            }
        }

        for &v in &idx {
            if let Some(sum) = face_sum.get_mut(v) {
                for axis in 0..3 {
                    sum[axis] += n[axis];
                }
            }
            if let (Some(vn), Some(flag)) = (mesh.normals.get(v), inward.get_mut(v))
                && dot(to_f64(*vn), n) < 0.0
            {
                *flag = true;
            }
        }
    }

    let (mut boundary_edges, mut nonmanifold_edges, mut inconsistent_edges) = (0u32, 0u32, 0u32);
    for &(uses, forward) in edges.values() {
        match uses {
            1 => boundary_edges += 1,
            2 if forward != 1 => inconsistent_edges += 1,
            2 => {}
            _ => nonmanifold_edges += 1,
        }
    }

    let mut bad_normal_lengths = 0u32;
    let mut max_dev_rad = 0.0f64;
    for (i, normal) in mesh.normals.iter().enumerate() {
        let vn = to_f64(*normal);
        let len = length(vn);
        if (len - 1.0).abs() > f64::from(NORMAL_LENGTH_TOL) {
            bad_normal_lengths += 1;
        }
        let Some(mean) = face_sum.get(i) else {
            continue;
        };
        let mean_len = length(*mean);
        if len > 0.0 && mean_len > 0.0 {
            let cos = (dot(vn, *mean) / (len * mean_len)).clamp(-1.0, 1.0);
            max_dev_rad = max_dev_rad.max(cos.acos());
        }
    }
    let inward_normals = inward.iter().filter(|f| **f).count() as u32;

    MeshReport {
        vertices: positions.len() as u32,
        unique_positions,
        triangles: mesh.triangle_count() as u32,
        closed: boundary_edges == 0 && nonmanifold_edges == 0 && inconsistent_edges == 0,
        boundary_edges,
        nonmanifold_edges,
        inconsistent_edges,
        degenerate_triangles,
        bbox_min: bbox_min.map(|v| v as f32),
        bbox_max: bbox_max.map(|v| v as f32),
        area,
        volume,
        max_normal_deviation_deg: max_dev_rad.to_degrees() as f32,
        bad_normal_lengths,
        inward_normals,
    }
}

/// Checks closedness, volume > 0, no degenerate triangle and NORMAL_RULE.
/// The error lists every failed rule with its numbers.
pub fn passes(report: &MeshReport) -> Result<(), String> {
    let mut failures = Vec::new();
    if !report.closed {
        failures.push(format!(
            "not closed (boundary edges {}, nonmanifold edges {}, inconsistent edges {})",
            report.boundary_edges, report.nonmanifold_edges, report.inconsistent_edges
        ));
    }
    if report.volume.is_nan() || report.volume <= 0.0 {
        failures.push(format!("volume {} <= 0", report.volume));
    }
    if report.degenerate_triangles > 0 {
        failures.push(format!(
            "{} degenerate triangles",
            report.degenerate_triangles
        ));
    }
    if report.max_normal_deviation_deg.is_nan()
        || report.max_normal_deviation_deg > NORMAL_DEVIATION_MAX_DEG
    {
        failures.push(format!(
            "max normal deviation {} deg > {} deg",
            report.max_normal_deviation_deg, NORMAL_DEVIATION_MAX_DEG
        ));
    }
    if report.bad_normal_lengths > 0 {
        failures.push(format!(
            "{} normals with abs(length - 1) > {}",
            report.bad_normal_lengths, NORMAL_LENGTH_TOL
        ));
    }
    if report.inward_normals > 0 {
        failures.push(format!("{} inward normals", report.inward_normals));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

/// FNV-1a 64 over the little-endian bytes of positions, then normals, then indices.
pub fn mesh_hash(mesh: &MeshData) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    let mut feed = |bytes: [u8; 4]| {
        for b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    for v in mesh.positions.iter().chain(mesh.normals.iter()).flatten() {
        feed(v.to_le_bytes());
    }
    for i in &mesh.indices {
        feed(i.to_le_bytes());
    }
    hash
}

/// Number of distinct positions after rounding each coordinate to 1e-6 (S8).
pub fn unique_positions_1e6(mesh: &MeshData) -> u32 {
    let mut seen = BTreeSet::new();
    for p in &mesh.positions {
        seen.insert(to_f64(*p).map(|v| (v * 1e6).round() as i64));
    }
    seen.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mesh_hash_is_fnv_offset_basis() {
        assert_eq!(mesh_hash(&MeshData::new()), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn mesh_hash_changes_with_any_buffer() {
        let base = MeshData {
            positions: vec![[0.0, 1.0, 2.0]],
            normals: vec![[0.0, 0.0, 1.0]],
            indices: vec![0, 0, 0],
        };
        let mut other = base.clone();
        other.indices[2] = 1;
        assert_eq!(mesh_hash(&base), mesh_hash(&base.clone()));
        assert_ne!(mesh_hash(&base), mesh_hash(&other));
    }

    #[test]
    fn unique_positions_1e6_merges_positions_closer_than_rounding() {
        let mesh = MeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1e-8, -1e-8, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 2e-6],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 5],
            indices: Vec::new(),
        };
        assert_eq!(unique_positions_1e6(&mesh), 3);
        assert_eq!(unique_positions_1e6(&MeshData::new()), 0);
    }

    #[test]
    fn debug_cube_passes_and_flipped_triangle_is_inconsistent() {
        use crate::{GeometryProvider, ParamValue, Params, Quality, builtin::DebugCube};
        let mut mesh = MeshData::new();
        let params = Params::new().with("half", ParamValue::Scalar(0.5));
        DebugCube
            .generate(&params, Quality::LOD0, &mut mesh)
            .unwrap();
        let report = analyze(&mesh);
        assert_eq!(passes(&report), Ok(()));
        assert_eq!(report.unique_positions, 8);
        assert!((report.volume - 1.0).abs() < 1e-9);
        assert!((report.area - 6.0).abs() < 1e-9);
        mesh.indices.swap(0, 1);
        let report = analyze(&mesh);
        assert_eq!(report.inconsistent_edges, 3);
        assert!(passes(&report).unwrap_err().contains("not closed"));
    }
}
