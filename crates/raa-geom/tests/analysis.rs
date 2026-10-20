use raa_geom::analysis::{analyze, mesh_hash, passes};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};

fn cube(half: f32) -> MeshData {
    let mut mesh = MeshData::new();
    DebugCube
        .generate(
            &Params::new().with("half", ParamValue::Scalar(half)),
            Quality::LOD0,
            &mut mesh,
        )
        .unwrap();
    mesh
}

#[test]
fn debug_cube_is_closed_unit_cube_and_passes() {
    let report = analyze(&cube(0.5));
    assert!(report.closed, "{report:?}");
    assert_eq!(report.vertices, 24);
    assert_eq!(report.unique_positions, 8);
    assert_eq!(report.triangles, 12);
    assert_eq!(report.boundary_edges, 0);
    assert_eq!(report.nonmanifold_edges, 0);
    assert_eq!(report.inconsistent_edges, 0);
    assert_eq!(report.degenerate_triangles, 0);
    assert!(
        (report.volume - 1.0).abs() < 1e-9,
        "volume {}",
        report.volume
    );
    assert!((report.area - 6.0).abs() < 1e-9, "area {}", report.area);
    assert_eq!(report.inward_normals, 0);
    assert_eq!(report.bad_normal_lengths, 0);
    assert_eq!(passes(&report), Ok(()));
}

#[test]
fn missing_triangle_is_reported_as_boundary() {
    let mut mesh = cube(0.5);
    mesh.indices.truncate(mesh.indices.len() - 3);
    let report = analyze(&mesh);
    assert!(report.boundary_edges > 0, "{report:?}");
    assert!(!report.closed);
    let err = passes(&report).unwrap_err();
    assert!(err.contains("closed"), "{err}");
}

#[test]
fn flipped_triangle_is_reported_as_inconsistent() {
    let mut mesh = cube(0.5);
    mesh.indices.swap(0, 1);
    let report = analyze(&mesh);
    assert!(report.inconsistent_edges > 0, "{report:?}");
    assert!(!report.closed);
    assert!(passes(&report).is_err());
}

#[test]
fn inverted_normals_are_all_counted_inward() {
    let mut mesh = cube(0.5);
    for n in &mut mesh.normals {
        *n = n.map(|c| -c);
    }
    let report = analyze(&mesh);
    assert_eq!(report.inward_normals, 24);
    let err = passes(&report).unwrap_err();
    assert!(err.contains("24 inward"), "{err}");
}

#[test]
fn scaled_mesh_welds_with_relative_eps() {
    let mut mesh = cube(500.0);
    // Jitter every vertex by up to 1e-4: far above an absolute 1e-6, below 1e-6 x diagonal.
    for (i, p) in mesh.positions.iter_mut().enumerate() {
        p[0] += (i % 5) as f32 * 2.5e-5;
    }
    let report = analyze(&mesh);
    assert_eq!(report.unique_positions, 8, "{report:?}");
    assert!(report.closed, "{report:?}");
    assert!(
        (report.volume - 1e9).abs() < 1e9 * 1e-4,
        "{}",
        report.volume
    );
}

#[test]
fn mesh_hash_is_deterministic_and_position_sensitive() {
    let a = cube(0.5);
    let b = cube(0.5);
    assert_eq!(mesh_hash(&a), mesh_hash(&b));
    let mut c = cube(0.5);
    c.positions[3][1] += 1e-3;
    assert_ne!(mesh_hash(&a), mesh_hash(&c));
}
