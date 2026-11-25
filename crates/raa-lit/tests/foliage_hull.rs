//! Whole-tree convex hull (M7, contract m7-hull): a tetrahedron's four outward faces and
//! volume 1/6, a cube's containment and volume, empty and refused inputs, the untouched
//! source and numerically stable repeated results.

use raa_core::RaaError;
use raa_geom::MeshData;
use raa_lit::foliage_resources::{FoliageSource, SourceTriangle};
use raa_math::Mat4;

mod foliage_resources {
    pub use raa_lit::foliage_resources::*;
}

#[path = "../src/foliage_hull.rs"]
mod foliage_hull;

use foliage_hull::build_foliage_hull;

fn triangle(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> SourceTriangle {
    SourceTriangle {
        positions: [a, b, c],
        normals: [[0.0, 0.0, 1.0]; 3],
        uv: [[0.0, 0.0]; 3],
        material: 0,
        source_tag: 1,
        foliage: true,
    }
}

fn source(triangles: Vec<SourceTriangle>) -> FoliageSource {
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
    if triangles.is_empty() {
        min = [0.0; 3];
        max = [0.0; 3];
    }
    FoliageSource {
        triangles,
        min,
        max,
        root_world: Mat4::IDENTITY,
        generation: 3,
    }
}

fn tetrahedron() -> FoliageSource {
    let o = [0.0, 0.0, 0.0];
    let x = [1.0, 0.0, 0.0];
    let y = [0.0, 1.0, 0.0];
    let z = [0.0, 0.0, 1.0];
    source(vec![
        triangle(o, y, x),
        triangle(o, x, z),
        triangle(o, z, y),
        triangle(x, y, z),
    ])
}

fn cube() -> FoliageSource {
    let corners = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let quads = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    let mut triangles = Vec::new();
    for quad in quads {
        triangles.push(triangle(
            corners[quad[0]],
            corners[quad[1]],
            corners[quad[2]],
        ));
        triangles.push(triangle(
            corners[quad[0]],
            corners[quad[2]],
            corners[quad[3]],
        ));
    }
    source(triangles)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn corner(mesh: &MeshData, index: u32) -> [f64; 3] {
    mesh.positions[index as usize].map(f64::from)
}

fn volume(mesh: &MeshData) -> f64 {
    let mut total = 0.0;
    for indices in mesh.indices.as_chunks::<3>().0 {
        let a = corner(mesh, indices[0]);
        let b = corner(mesh, indices[1]);
        let c = corner(mesh, indices[2]);
        total += dot(a, cross(b, c)) / 6.0;
    }
    total
}

fn contains(mesh: &MeshData, point: [f32; 3]) -> bool {
    let point = point.map(f64::from);
    for indices in mesh.indices.as_chunks::<3>().0 {
        let a = corner(mesh, indices[0]);
        let b = corner(mesh, indices[1]);
        let c = corner(mesh, indices[2]);
        if dot(cross(sub(b, a), sub(c, a)), sub(point, a)) > 1e-6 {
            return false;
        }
    }
    true
}

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

fn assert_invalid(source: &FoliageSource) {
    let err = build_foliage_hull(source).expect_err("source must be refused");
    assert_eq!(err.code(), invalid_code(), "{}", err.message());
}

#[test]
fn tetrahedron_has_four_outward_faces_and_volume_one_sixth() {
    let mesh = build_foliage_hull(&tetrahedron()).expect("tetrahedron hull");
    mesh.validate().expect("valid mesh");
    assert_eq!(mesh.triangle_count(), 4);
    assert_eq!(mesh.vertex_count(), 12);
    let volume = volume(&mesh);
    assert!((volume - 1.0 / 6.0).abs() < 1e-6, "volume {volume}");
    assert!(volume > 0.0, "outward winding gives a positive volume");

    let interior = [0.25, 0.25, 0.25];
    for indices in mesh.indices.as_chunks::<3>().0 {
        let a = corner(&mesh, indices[0]);
        let b = corner(&mesh, indices[1]);
        let c = corner(&mesh, indices[2]);
        let center = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        let normal = mesh.normals[indices[0] as usize].map(f64::from);
        let length = dot(normal, normal).sqrt();
        assert!((length - 1.0).abs() < 1e-6, "unit normal {length}");
        assert!(
            dot(normal, sub(center, interior)) > 0.0,
            "normal points out"
        );
    }

    for vertex in [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ] {
        assert!(
            contains(&mesh, vertex),
            "source vertex {vertex:?} is inside"
        );
    }
}

#[test]
fn cube_is_contained_with_volume_one_and_repeats_exactly() {
    let source = cube();
    let mesh = build_foliage_hull(&source).expect("cube hull");
    mesh.validate().expect("valid mesh");
    assert_eq!(mesh.triangle_count(), 12);
    assert!(
        (volume(&mesh) - 1.0).abs() < 1e-5,
        "volume {}",
        volume(&mesh)
    );

    for x in [-0.5_f32, 0.0, 0.5] {
        for y in [-0.5_f32, 0.0, 0.5] {
            for z in [-0.5_f32, 0.0, 0.5] {
                assert!(
                    contains(&mesh, [x, y, z]),
                    "cube point {x},{y},{z} is inside"
                );
            }
        }
    }

    let again = build_foliage_hull(&source).expect("cube hull again");
    assert_eq!(mesh.positions, again.positions);
    assert_eq!(mesh.normals, again.normals);
    assert_eq!(mesh.indices, again.indices);
}

#[test]
fn empty_source_returns_an_empty_mesh() {
    let mesh = build_foliage_hull(&source(Vec::new())).expect("empty hull");
    assert!(mesh.positions.is_empty());
    assert!(mesh.normals.is_empty());
    assert!(mesh.indices.is_empty());
    mesh.validate().expect("valid empty mesh");
}

#[test]
fn coplanar_source_is_refused() {
    let square = source(vec![
        triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        triangle([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
    ]);
    assert_invalid(&square);
}

#[test]
fn nonfinite_source_is_refused() {
    let broken = source(vec![triangle(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, f32::NAN, 0.0],
    )]);
    assert_invalid(&broken);
}

#[test]
fn hull_construction_does_not_modify_the_source() {
    let source = tetrahedron();
    let before = source.clone();
    let _ = build_foliage_hull(&source).expect("tetrahedron hull");
    assert_eq!(source, before);
}
