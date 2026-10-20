//! intersect_convex and subtract_convex on code-generated meshes (DebugCube and raa-prims at LOD0).

use raa_csg::{CsgError, convex_planes, intersect_convex, subtract_convex};
use raa_geom::analysis::{analyze, mesh_hash};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_math::{Mat4, Vec3};
use raa_prims::PrimParams;

fn moved(src: &MeshData, t: [f32; 3]) -> MeshData {
    let mut m = MeshData::new();
    m.append_transformed(src, &Mat4::from_translation(Vec3::from_array(t)));
    m
}

fn cube(half: f32, t: [f32; 3]) -> MeshData {
    let mut m = MeshData::new();
    let params = Params::new().with("half", ParamValue::Scalar(half));
    DebugCube
        .generate(&params, Quality::LOD0, &mut m)
        .expect("cube");
    moved(&m, t)
}

fn prim(p: PrimParams, t: [f32; 3]) -> MeshData {
    let mut m = MeshData::new();
    raa_prims::generate(&p, Quality::LOD0, &mut m).expect("prim");
    moved(&m, t)
}

fn intersect(a: &MeshData, b: &MeshData) -> Result<MeshData, CsgError> {
    let mut out = MeshData::new();
    intersect_convex(a, b, &mut out).map(|()| out)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn overlapping_cubes_give_closed_box_of_expected_volume() {
    // At z = 0.1 the tool top face would lie on the host top face (coplanar, refused below),
    // so the tool sits at z = 0.05: x [-0.1, 0.5], y [-0.2, 0.5], z [-0.35, 0.45], 0.6 x 0.7 x 0.8.
    let out = intersect(&cube(0.5, [0.0; 3]), &cube(0.4, [0.3, 0.2, 0.05])).expect("ok");
    let r = analyze(&out);
    assert!(r.closed, "{r:?}");
    assert!(rel(r.volume, 0.336) <= 1e-4, "volume {}", r.volume);
}

#[test]
fn cube_touching_host_top_face_from_inside_gives_coplanar() {
    let e = intersect(&cube(0.5, [0.0; 3]), &cube(0.4, [0.3, 0.2, 0.1])).unwrap_err();
    assert_eq!(e.code(), "csg.coplanar");
}

#[test]
fn small_cube_inside_big_cube_returns_the_small_one() {
    let small = cube(0.2, [0.1, -0.05, 0.03]);
    let out = intersect(&cube(0.5, [0.0; 3]), &small).expect("ok");
    assert_eq!(out, small);
    let out = intersect(&small, &cube(0.5, [0.0; 3])).expect("ok");
    assert_eq!(out, small);
}

#[test]
fn disjoint_cubes_give_empty_result() {
    let e = intersect(&cube(0.5, [0.0; 3]), &cube(0.5, [3.0, 0.0, 0.0])).unwrap_err();
    assert_eq!(e.code(), "csg.empty_result");
}

#[test]
fn shared_face_plane_with_overlap_gives_coplanar() {
    // Tool top face at y = 0.5 lies on the host top face.
    let e = intersect(&cube(0.5, [0.0; 3]), &cube(0.3, [0.1, 0.2, 0.0])).unwrap_err();
    assert_eq!(e.code(), "csg.coplanar");
    assert!(e.to_string().contains("d=0.5"), "{e}");
}

fn box_and_sphere() -> (MeshData, MeshData) {
    let host = prim(
        PrimParams::Box {
            half_extents: [0.7, 0.6, 0.5],
        },
        [0.0; 3],
    );
    let tool = prim(PrimParams::Sphere { radius: 0.8 }, [0.013, -0.011, 0.017]);
    (host, tool)
}

#[test]
fn box_intersect_sphere_is_closed_and_smaller_than_both() {
    let (host, tool) = box_and_sphere();
    let out = intersect(&host, &tool).expect("ok");
    let r = analyze(&out);
    assert!(r.closed && r.volume > 0.0, "{r:?}");
    assert!(r.volume < analyze(&host).volume);
    assert!(r.volume < analyze(&tool).volume);
}

#[test]
fn two_runs_give_identical_mesh_hash() {
    let (host, tool) = box_and_sphere();
    let a = intersect(&host, &tool).expect("ok");
    let b = intersect(&host, &tool).expect("ok");
    assert_eq!(mesh_hash(&a), mesh_hash(&b));
}

fn subtract(host: &MeshData, tools: &[&MeshData]) -> Result<MeshData, CsgError> {
    let planes = convex_planes(host, 1e-6).expect("host planes");
    let mut out = MeshData::new();
    subtract_convex(host, &planes, tools, &mut out).map(|()| out)
}

#[test]
fn cube_minus_corner_cube_is_closed_with_volume_minus_overlap() {
    // Tool x [0.1, 0.7], y [0.05, 0.65], z [0, 0.6]; overlap 0.4 x 0.45 x 0.5 = 0.09.
    let out = subtract(&cube(0.5, [0.0; 3]), &[&cube(0.3, [0.4, 0.35, 0.3])]).expect("ok");
    let r = analyze(&out);
    assert!(r.closed, "{r:?}");
    assert!(rel(r.volume, 1.0 - 0.09) <= 1e-4, "volume {}", r.volume);
}

#[test]
fn two_disjoint_tools_remove_both_overlaps() {
    // Overlaps: 0.4 x 0.45 x 0.5 = 0.09 and 0.25 x 0.3 x 0.35 = 0.02625.
    let a = cube(0.3, [0.4, 0.35, 0.3]);
    let b = cube(0.2, [-0.45, -0.4, -0.35]);
    let out = subtract(&cube(0.5, [0.0; 3]), &[&a, &b]).expect("ok");
    let r = analyze(&out);
    assert!(r.closed, "{r:?}");
    assert!(
        rel(r.volume, 1.0 - 0.09 - 0.02625) <= 1e-4,
        "volume {}",
        r.volume
    );
}

#[test]
fn overlapping_tools_give_overlapping_tools() {
    let a = cube(0.3, [0.4, 0.35, 0.3]);
    let b = cube(0.3, [0.2, 0.15, 0.1]);
    let e = subtract(&cube(0.5, [0.0; 3]), &[&a, &b]).unwrap_err();
    assert_eq!(e.code(), "csg.overlapping_tools");
    assert!(
        e.to_string().contains("tool#0") && e.to_string().contains("tool#1"),
        "{e}"
    );
}

#[test]
fn far_tool_leaves_host_byte_identical() {
    let host = cube(0.5, [0.0; 3]);
    let out = subtract(&host, &[&cube(0.3, [5.0, 0.0, 0.0])]).expect("ok");
    assert_eq!(out, host);
    assert_eq!(mesh_hash(&out), mesh_hash(&host));
}

#[test]
fn host_inside_tool_gives_empty_result() {
    let e = subtract(&cube(0.2, [0.05, 0.0, 0.0]), &[&cube(0.5, [0.0; 3])]).unwrap_err();
    assert_eq!(e.code(), "csg.empty_result");
}

fn box_minus_sphere_volume_identity(t: [f32; 3], radius: f32) {
    let host = prim(
        PrimParams::Box {
            half_extents: [0.7, 0.6, 0.5],
        },
        [0.0; 3],
    );
    let tool = prim(PrimParams::Sphere { radius }, t);
    let out = subtract(&host, &[&tool]).expect("ok");
    let r = analyze(&out);
    assert!(r.closed && r.volume > 0.0, "{r:?}");
    let vh = analyze(&host).volume;
    let vi = analyze(&intersect(&host, &tool).expect("ok")).volume;
    assert!(vi > 0.0);
    let err = (r.volume - (vh - vi)).abs() / vh;
    assert!(err <= 1e-4, "{} vs {} (rel {err:e})", r.volume, vh - vi);
}

#[test]
fn box_minus_sphere_dent_is_closed_and_satisfies_volume_identity() {
    // Set A dent (case box-sub-sphere-dent): sphere r 0.4 centred just above the top face.
    box_minus_sphere_volume_identity([0.013, 0.613, 0.007], 0.4);
}

#[test]
fn box_minus_sphere_across_edges_is_closed_and_satisfies_volume_identity() {
    // The sphere crosses two and three host faces; its pole facets graze the x = 0.7 face.
    for t in [[0.9, 0.5, 0.0], [0.9, 0.5, 0.3], [0.5, 0.5, 0.5]] {
        box_minus_sphere_volume_identity(t, 0.8);
    }
}
