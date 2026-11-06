use raa_core::MeshId;
use raa_geom::MeshData;
use raa_lit::{
    LodLevel, LodParams, LodRegistry, MAX_LOD_LEVELS, camera_level, lod_error, shadow_level,
};

fn cube(size: f32) -> MeshData {
    MeshData {
        positions: (0..8)
            .map(|i| std::array::from_fn(|axis| if i & (1 << axis) == 0 { -size } else { size }))
            .collect(),
        indices: vec![
            0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 4, 6, 0, 6,
            2, 1, 3, 7, 1, 7, 5,
        ],
        normals: vec![],
    }
}

fn sphere(columns: u32, rows: u32) -> MeshData {
    let mut mesh = MeshData::new();
    for y in 0..=rows {
        let theta = std::f32::consts::PI * y as f32 / rows as f32;
        for x in 0..=columns {
            let phi = std::f32::consts::TAU * x as f32 / columns as f32;
            mesh.positions.push([
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ]);
        }
    }
    for y in 0..rows {
        for x in 0..columns {
            let a = y * (columns + 1) + x;
            let b = a + columns + 1;
            if y > 0 {
                mesh.indices.extend([a, b, a + 1]);
            }
            if y + 1 < rows {
                mesh.indices.extend([a + 1, b, b + 1]);
            }
        }
    }
    mesh
}

#[test]
fn identical_mesh() {
    assert_eq!(lod_error(&cube(0.5), &cube(0.5)).unwrap(), 0.0);
}

#[test]
fn sphere_error_is_symmetric() {
    let full = sphere(64, 32);
    let level = sphere(8, 4);
    let error = lod_error(&full, &level).unwrap();
    assert!(error > 0.01 && error < 0.2, "{error}");
    assert_eq!(error, lod_error(&level, &full).unwrap());
}

#[test]
fn enlarged_cube() {
    let err = lod_error(&cube(0.5), &cube(0.55)).unwrap();
    // enlarged corner is 0.05 out on all three axes: sqrt(3)*0.05
    assert!((err - 0.08660254).abs() < 1e-4);
}

#[test]
fn invalid_geometry() {
    let good = cube(0.5);
    let mut bad = good.clone();
    bad.positions[0][0] = f32::NAN;
    let mut index = good.clone();
    index.indices[0] = 100;
    let mut partial = good.clone();
    partial.indices.pop();
    for mesh in [MeshData::new(), cube(0.0), bad, index, partial] {
        assert_eq!(
            lod_error(&good, &mesh).unwrap_err().code(),
            "render.invalid_lod"
        );
        assert!(lod_error(&mesh, &good).is_err());
    }
}

fn chain() -> Vec<LodLevel> {
    [0.0, 0.01, 0.04, 0.16]
        .into_iter()
        .enumerate()
        .map(|(i, error)| LodLevel {
            mesh: MeshId(i as u32),
            error,
        })
        .collect()
}

#[test]
fn camera_projection_bias_and_band() {
    let c = chain();
    let p = LodParams::default();
    let select = |d, prev, p: &LodParams| camera_level(&c, 1.0, d, 1000.0, prev, p);
    for (d, expected) in [(5.0, 0), (50.0, 2), (1000.0, 3)] {
        assert_eq!(select(d, None, &p), expected);
    }
    assert_eq!(select(10.0, Some(0), &p), 0);
    assert_eq!(select(12.4, Some(0), &p), 0);
    assert_eq!(select(12.5, Some(0), &p), 1);
    assert_eq!(select(9.0, Some(1), &p), 1);
    assert_eq!(select(8.0, Some(1), &p), 0);
    assert_eq!(select(25.0, None, &p), 1);
    assert_eq!(select(25.0, None, &LodParams { bias: 1.0, ..p }), 2);
    assert_eq!(select(1000.0, Some(999), &p), 3);
    for d in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(select(d, None, &p), 0);
    }
    assert_eq!(camera_level(&[], 1.0, 1.0, 1.0, None, &p), 0);
}

#[test]
fn shadow_projection_and_band() {
    let c = chain();
    let p = LodParams::default();
    assert_eq!(shadow_level(&c, 1.0, 0.01, None, &p), 1);
    assert_eq!(shadow_level(&c, 1.0, 0.01, Some(0), &p), 0);
    assert_eq!(shadow_level(&c, 1.0, 0.0125, Some(0), &p), 1);
    assert_eq!(shadow_level(&c, 1.0, 0.009, Some(1), &p), 1);
    assert_eq!(shadow_level(&c, 1.0, 0.008, Some(1), &p), 0);
    assert_eq!(shadow_level(&c, 1.0, 100.0, None, &p), 3);
    for texel in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(shadow_level(&c, 1.0, texel, None, &p), 0);
    }
}

#[test]
fn registry_validation_is_atomic() {
    let mut registry = LodRegistry::new();
    let c = chain();
    registry.set(c[0].mesh, &c[1..]).unwrap();
    assert_eq!(registry.chain(c[0].mesh), Some(c.as_slice()));
    assert!(!registry.is_level(c[0].mesh));
    assert!(registry.is_level(c[1].mesh));
    let level = |id, error| LodLevel {
        mesh: MeshId(id),
        error,
    };
    let too_many: Vec<_> = (1..=MAX_LOD_LEVELS)
        .map(|i| level(i as u32, i as f32))
        .collect();
    for bad in [
        vec![],
        too_many,
        vec![level(0, 1.0)],
        vec![level(1, 1.0), level(1, 2.0)],
        vec![level(4, 0.0)],
        vec![level(4, -1.0)],
        vec![level(4, f32::NAN)],
        vec![level(4, f32::INFINITY)],
        vec![level(4, 2.0), level(5, 1.0)],
        vec![level(4, 1.0), level(5, 1.0)],
    ] {
        assert_eq!(
            registry.set(MeshId(0), &bad).unwrap_err().code(),
            "render.invalid_lod"
        );
        assert_eq!(registry.chain(MeshId(0)), Some(c.as_slice()));
    }
    assert!(registry.set(MeshId(9), &[level(1, 1.0)]).is_err());
    assert!(registry.set(MeshId(9), &[level(0, 1.0)]).is_err());
    assert!(registry.set(MeshId(1), &[level(9, 1.0)]).is_err());
    registry
        .set(MeshId(0), &[level(1, 0.1), level(4, 0.2)])
        .unwrap();
    assert!(!registry.is_level(MeshId(2)));
    registry.clear(MeshId(99));
    registry.clear(MeshId(0));
    assert!(registry.chain(MeshId(0)).is_none());
    assert!(!registry.is_level(MeshId(1)));
}

#[test]
fn flat_face_and_distant_grid_queries() {
    let face = MeshData {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        indices: vec![0, 1, 2],
        normals: vec![],
    };
    for offset in [0.05, 1000.0] {
        let mut moved = face.clone();
        for p in &mut moved.positions {
            p[2] += offset;
        }
        assert!((lod_error(&face, &moved).unwrap() - offset).abs() < 1e-4);
    }
}
