//! Tangent generation: orthonormality, handedness, determinism and the default basis.

use raa_geom::MeshData;
use raa_mesh::{AttributedMesh, default_tangent, generate_tangents};

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn xyz(t: [f32; 4]) -> [f32; 3] {
    [t[0], t[1], t[2]]
}

/// A latitude/longitude grid on the unit sphere between -80 and +80 degrees (no pole
/// collapse); u runs east with longitude, v north with latitude, normals outward.
struct Grid {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

fn sphere_grid(rings: u32, segments: u32) -> Grid {
    let lat_max = 80f32.to_radians();
    let mut grid = Grid {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::new(),
    };
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let lat = -lat_max + 2.0 * lat_max * v;
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let lon = u * std::f32::consts::TAU;
            // Longitude grows from +Z towards +X, so u increases eastward (outward normal).
            let p = [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()];
            grid.positions.push(p);
            grid.normals.push(p);
            grid.uvs.push([u, v]);
        }
    }
    let row = segments + 1;
    for r in 0..rings {
        for s in 0..segments {
            let a = r * row + s;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            grid.indices.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
    grid
}

#[test]
fn sphere_grid_tangents_are_orthonormal() {
    let g = sphere_grid(12, 24);
    let mut out = Vec::new();
    generate_tangents(&g.positions, &g.normals, &g.uvs, &g.indices, &mut out);
    assert_eq!(out.len(), g.positions.len());
    for (i, (t, n)) in out.iter().zip(&g.normals).enumerate() {
        let txyz = xyz(*t);
        let len = dot(txyz, txyz).sqrt();
        assert!((len - 1.0).abs() < 1e-4, "vertex {i}: |t| = {len}");
        let d = dot(txyz, *n);
        assert!(d.abs() < 1e-4, "vertex {i}: t.n = {d}");
        assert!(t[3] == 1.0 || t[3] == -1.0, "vertex {i}: w = {}", t[3]);
    }
}

#[test]
fn sphere_grid_tangents_follow_u_with_positive_handedness() {
    let g = sphere_grid(6, 12);
    let mut out = Vec::new();
    generate_tangents(&g.positions, &g.normals, &g.uvs, &g.indices, &mut out);
    for (i, (t, p)) in out.iter().zip(&g.positions).enumerate() {
        // dP/du on the sphere points east: (cos lon, 0, -sin lon) scaled by cos lat.
        let east = [p[2], 0.0, -p[0]];
        let east_len = dot(east, east).sqrt();
        let d = dot(xyz(*t), east) / east_len;
        assert!(d > 0.95, "vertex {i}: tangent not along +u ({d})");
        assert_eq!(t[3], 1.0, "vertex {i}");
    }
}

/// A unit quad in the XY plane facing +Z with u = 1 - x (mirrored) and v = y.
#[test]
fn mirrored_u_quad_has_negative_handedness() {
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let normals = [[0.0, 0.0, 1.0]; 4];
    let indices = [0, 1, 2, 0, 2, 3];
    let mut out = Vec::new();

    let plain = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    generate_tangents(&positions, &normals, &plain, &indices, &mut out);
    for t in &out {
        assert_eq!(*t, [1.0, 0.0, 0.0, 1.0]);
    }

    let mirrored = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    generate_tangents(&positions, &normals, &mirrored, &indices, &mut out);
    assert_eq!(out.len(), 4);
    for t in &out {
        assert_eq!(*t, [-1.0, 0.0, 0.0, -1.0]);
    }
}

#[test]
fn second_run_is_byte_identical() {
    let g = sphere_grid(9, 17);
    let mut first = Vec::new();
    let mut second = vec![[9.0; 4]; 3];
    generate_tangents(&g.positions, &g.normals, &g.uvs, &g.indices, &mut first);
    generate_tangents(&g.positions, &g.normals, &g.uvs, &g.indices, &mut second);
    let bits = |v: &[[f32; 4]]| -> Vec<u32> { v.iter().flatten().map(|c| c.to_bits()).collect() };
    assert_eq!(bits(&first), bits(&second));
}

#[test]
fn degenerate_uvs_fall_back_to_default_tangent() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]];
    let n = [0.0, -0.6, 0.8];
    let normals = [n; 3];
    let uvs = [[0.5, 0.5]; 3];
    let mut out = Vec::new();
    generate_tangents(&positions, &normals, &uvs, &[0, 1, 2], &mut out);
    assert_eq!(out, vec![default_tangent(n); 3]);
}

#[test]
fn out_of_range_triangles_are_skipped_without_panic() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let normals = [[0.0, 1.0, 0.0]; 2];
    let uvs = [[0.0, 0.0], [1.0, 0.0]];
    let mut out = Vec::new();
    generate_tangents(&positions, &normals, &uvs, &[0, 1, 7], &mut out);
    assert_eq!(out, vec![default_tangent([0.0, 1.0, 0.0]); 2]);
}

fn assert_default_basis(n: [f32; 3]) {
    let len = dot(n, n).sqrt();
    let n = n.map(|c| c / len);
    let t = default_tangent(n);
    let txyz = xyz(t);
    assert_eq!(t[3], 1.0, "n = {n:?}");
    assert!(
        (dot(txyz, txyz).sqrt() - 1.0).abs() < 1e-4,
        "|t| for n = {n:?}"
    );
    assert!(dot(txyz, n).abs() < 1e-4, "t.n for n = {n:?}");
    let b = cross(n, txyz);
    assert!(
        (dot(b, b).sqrt() - 1.0).abs() < 1e-4,
        "|n x t| for n = {n:?}"
    );
}

#[test]
fn default_tangent_is_orthonormal_for_axes() {
    for n in [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ] {
        assert_default_basis(n);
    }
    assert_eq!(default_tangent([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(default_tangent([0.0, 0.0, -1.0]), [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn default_tangent_is_orthonormal_for_random_normals() {
    // xorshift32: deterministic pseudo-random normals.
    let mut state = 0x9e37_79b9u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut tested = 0;
    while tested < 1000 {
        let n = [next(), next(), next()];
        if dot(n, n) < 1e-6 {
            continue;
        }
        assert_default_basis(n);
        tested += 1;
    }
}

#[test]
fn ensure_tangents_generates_only_when_uvs_present_and_tangents_absent() {
    let base = MeshData {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        indices: vec![0, 1, 2],
    };

    let mut no_uvs = AttributedMesh::from_mesh(base.clone());
    no_uvs.ensure_tangents();
    assert!(no_uvs.tangents.is_none());

    let mut with_uvs = AttributedMesh::from_mesh(base.clone());
    with_uvs.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
    with_uvs.ensure_tangents();
    assert_eq!(with_uvs.tangents, Some(vec![[1.0, 0.0, 0.0, 1.0]; 3]));
    assert!(with_uvs.validate().is_ok());

    let given = vec![[0.0, 1.0, 0.0, -1.0]; 3];
    let mut with_tangents = AttributedMesh::from_mesh(base);
    with_tangents.uvs = Some(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
    with_tangents.tangents = Some(given.clone());
    with_tangents.ensure_tangents();
    assert_eq!(with_tangents.tangents, Some(given));
}
