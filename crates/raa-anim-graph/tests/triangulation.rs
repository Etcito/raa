use raa_anim::{Clip, Rig};
use raa_anim_graph::{Graph, Triangulation, triangulate};

fn tri(points: &[[f32; 2]]) -> Triangulation {
    triangulate(points, "/p").unwrap()
}

#[test]
fn orders_a_single_triangle_counter_clockwise() {
    let t = tri(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]);
    assert_eq!(t.triangles, vec![[0, 2, 1]]);
    assert_eq!(t.hull, vec![[0, 2], [2, 1], [1, 0]]);
}

#[test]
fn picks_the_empty_circle_diagonal() {
    // Flat rhombus: the short diagonal 1-3 is Delaunay, 0-2 is not.
    let t = tri(&[[0.0, 0.0], [2.0, -1.0], [4.0, 0.0], [2.0, 1.0]]);
    assert_eq!(t.triangles, vec![[0, 1, 3], [1, 2, 3]]);
    assert_eq!(t.hull, vec![[0, 1], [1, 2], [2, 3], [3, 0]]);
}

#[test]
fn breaks_cocircular_ties_by_source_index() {
    let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let t = tri(&square);
    assert_eq!(t.triangles, vec![[0, 1, 2], [0, 2, 3]]);
    assert_eq!(t.hull, vec![[0, 1], [1, 2], [2, 3], [3, 0]]);
    assert_eq!(tri(&square), t);
    // The same square listed differently splits along the other diagonal, (1,0)-(0,1).
    let t = tri(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]);
    assert_eq!(t.triangles, vec![[0, 1, 2], [0, 3, 1]]);
    assert_eq!(t.hull, vec![[0, 3], [3, 1], [1, 2], [2, 0]]);
}

#[test]
fn keeps_interior_and_collinear_hull_points_as_vertices() {
    let points = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    let t = tri(&points);
    assert_eq!(
        t.triangles,
        vec![[0, 5, 3], [0, 4, 5], [1, 2, 5], [1, 5, 4], [2, 3, 5]]
    );
    assert_eq!(t.hull, vec![[0, 4], [4, 1], [1, 2], [2, 3], [3, 0]]);
}

#[test]
fn triangulates_a_capped_cocircular_grid() {
    let grid: Vec<[f32; 2]> = (0..64).map(|i| [(i % 8) as f32, (i / 8) as f32]).collect();
    let t = tri(&grid);
    assert_eq!(
        (t.triangles.len(), t.hull.len(), t.hull[0], t.hull[27]),
        (98, 28, [0, 1], [8, 0])
    );
    assert_eq!(&t.triangles[..2], &[[0, 1, 8], [1, 2, 9]]);
}

#[test]
fn refuses_degenerate_point_sets() {
    let cases: [&[[f32; 2]]; 5] = [
        &[[0.0, 0.0], [1.0, 1.0]],
        &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        &[[0.0, 0.0], [1.0, 0.0], [0.0, f32::NAN]],
        &[[0.0, 0.0], [3e19, 0.0], [0.0, 3e19], [3e19, 3e19]],
    ];
    for points in cases {
        let e = triangulate(points, "/blendSpaces/2").unwrap_err();
        assert_eq!(
            (e.code, e.path.as_str()),
            ("graph.blend_space", "/blendSpaces/2"),
            "{points:?}"
        );
    }
}

const DOC: &str = r#"{"format":"raa.animgraph","version":1,"name":"g","rig":"rig-basic","params":[{"name":"x","type":"float","default":0},{"name":"y","type":"float","default":0}],"blendSpaces":[{"name":"line","params":["x"],"points":[{"clip":"idle","at":[0]},{"clip":"run","at":[1]}]},{"name":"plane","params":["x","y"],"points":[{"clip":"idle","at":[0,0]},{"clip":"run","at":[1,0]},{"clip":"idle","at":[1,1]},{"clip":"run","at":[0,1]}]}],"layers":[{"name":"base","machine":"m","mask":null,"weight":1,"additive":false}],"machines":[{"name":"m","entry":"s","states":[{"name":"s","motion":"plane","loop":true,"speedParam":null}],"transitions":[]}],"rootMotion":{"bone":"root","mode":"none"}}"#;

fn assets() -> (Rig, Vec<Clip>) {
    let rig = Rig::load(include_str!("../../raa-anim/tests/fixtures/rig-basic.json")).unwrap();
    let clip = |name: &str| {
        format!(
            r#"{{"format":"raa.clip","version":2,"name":"{name}","rig":"rig-basic","lengthTicks":10,"loop":true,"rootMotion":"none","tracks":[],"events":[]}}"#
        )
    };
    let clips = ["idle", "run"]
        .map(|n| Clip::load(&clip(n), &rig).unwrap())
        .to_vec();
    (rig, clips)
}

#[test]
fn load_retains_geometry_for_2d_spaces_only() {
    let (rig, clips) = assets();
    let compiled = Graph::load_compiled(DOC, &rig, &clips).unwrap();
    let plane = compiled.triangulations[1].as_ref().unwrap();
    assert_eq!(compiled.triangulations[0], None);
    assert_eq!(plane.triangles, vec![[0, 1, 2], [0, 2, 3]]);
    assert_eq!(plane.hull, vec![[0, 1], [1, 2], [2, 3], [3, 0]]);
    assert_eq!(Graph::load(DOC, &rig, &clips).unwrap(), compiled.graph);
}

#[test]
fn load_refuses_degenerate_2d_spaces() {
    let (rig, clips) = assets();
    let collinear = DOC
        .replace(r#""at":[1,1]"#, r#""at":[2,0]"#)
        .replace(r#""at":[0,1]"#, r#""at":[3,0]"#);
    let duplicate = DOC.replace(r#""at":[0,1]"#, r#""at":[1,0]"#);
    for (doc, path) in [
        (collinear, "/blendSpaces/1"),
        (duplicate, "/blendSpaces/1/points/3"),
    ] {
        let e = Graph::load_compiled(&doc, &rig, &clips).unwrap_err();
        assert_eq!((e.code, e.path.as_str()), ("graph.blend_space", path));
    }
}
