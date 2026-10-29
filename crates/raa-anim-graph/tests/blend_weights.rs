use raa_anim_graph::{
    BlendPoint, BlendSpace, BlendWeight, Num, evaluate, evaluate_1d, evaluate_2d, triangulate,
};

fn point(clip: &str, at: &[f32]) -> BlendPoint {
    BlendPoint {
        clip: clip.into(),
        at: at.iter().map(|&value| Num::new(value.into())).collect(),
    }
}

fn blend_space(params: &[&str], points: &[(&str, &[f32])]) -> BlendSpace {
    BlendSpace {
        name: "blend".into(),
        params: params.iter().map(|name| (*name).into()).collect(),
        points: points.iter().map(|(clip, at)| point(clip, at)).collect(),
        sync: false,
    }
}

fn line() -> BlendSpace {
    blend_space(
        &["speed"],
        &[("idle", &[0.0]), ("waddle", &[1.5]), ("run", &[4.0])],
    )
}

fn square() -> BlendSpace {
    blend_space(
        &["x", "y"],
        &[
            ("idle", &[0.0, 0.0]),
            ("run", &[1.0, 0.0]),
            ("idle2", &[1.0, 1.0]),
            ("run2", &[0.0, 1.0]),
        ],
    )
}

fn expect(got: &[BlendWeight], want: &[(usize, f32)]) {
    assert_eq!(got.len(), want.len(), "{got:?}");
    for (got, &(point, weight)) in got.iter().zip(want) {
        assert_eq!(got.point, point, "{got:?}");
        assert!((got.weight - weight).abs() <= 1e-6, "{got:?} != {want:?}");
    }
    let sum: f32 = got.iter().map(|w| w.weight).sum();
    assert!((sum - 1.0).abs() <= 1e-6, "{got:?} sums to {sum}");
    assert!(got.iter().all(|w| w.weight.is_finite() && w.weight >= 0.0), "{got:?}");
}

fn weights(space: &BlendSpace, at: &[f32]) -> Vec<BlendWeight> {
    let geometry = (space.params.len() == 2).then(|| {
        let points: Vec<[f32; 2]> = space.points.iter().map(|p| [p.at[0].rt, p.at[1].rt]).collect();
        triangulate(&points, "/blendSpaces/0").unwrap()
    });
    evaluate(space, geometry.as_ref(), at).unwrap()
}

#[test]
fn interpolates_one_dimensional_neighbours_and_endpoints() {
    let space = line();
    expect(&weights(&space, &[-1.0]), &[(0, 1.0)]);
    expect(&weights(&space, &[0.0]), &[(0, 1.0)]);
    expect(&weights(&space, &[0.75]), &[(0, 0.5), (1, 0.5)]);
    expect(&weights(&space, &[1.5]), &[(1, 1.0)]);
    expect(&weights(&space, &[2.75]), &[(1, 0.5), (2, 0.5)]);
    expect(&weights(&space, &[4.0]), &[(2, 1.0)]);
    expect(&weights(&space, &[9.0]), &[(2, 1.0)]);
    let unsorted = blend_space(
        &["speed"],
        &[("run", &[4.0]), ("idle", &[0.0]), ("waddle", &[1.5])],
    );
    expect(&weights(&unsorted, &[-2.0]), &[(1, 1.0)]);
    expect(&weights(&unsorted, &[0.75]), &[(1, 0.5), (2, 0.5)]);
    expect(&weights(&unsorted, &[3.0]), &[(0, 0.6), (2, 0.4)]);
    assert_eq!(weights(&unsorted, &[3.0]), weights(&unsorted, &[3.0]));
    let one = blend_space(&["speed"], &[("idle", &[2.0])]);
    expect(&weights(&one, &[-5.0]), &[(0, 1.0)]);
    expect(&weights(&one, &[5.0]), &[(0, 1.0)]);
}

#[test]
fn interpolates_two_dimensional_triangle_interiors_in_source_order() {
    let space = square();
    expect(&weights(&space, &[0.25, 0.1]), &[(0, 0.75), (1, 0.15), (2, 0.1)]);
    expect(&weights(&space, &[0.1, 0.5]), &[(0, 0.5), (2, 0.1), (3, 0.4)]);
    let single = blend_space(
        &["x", "y"],
        &[("a", &[0.0, 0.0]), ("c", &[0.0, 1.0]), ("b", &[1.0, 0.0])],
    );
    expect(
        &weights(&single, &[0.25, 0.25]),
        &[(0, 0.5), (1, 0.25), (2, 0.25)],
    );
}

#[test]
fn handles_vertices_edges_and_hull_boundaries() {
    let space = square();
    expect(&weights(&space, &[0.0, 0.0]), &[(0, 1.0)]);
    expect(&weights(&space, &[1.0, 1.0]), &[(2, 1.0)]);
    expect(&weights(&space, &[0.5, 0.5]), &[(0, 0.5), (2, 0.5)]);
    expect(&weights(&space, &[0.5, 0.0]), &[(0, 0.5), (1, 0.5)]);
    expect(&weights(&space, &[0.0, 0.5]), &[(0, 0.5), (3, 0.5)]);
    expect(&weights(&space, &[1.0, 0.5]), &[(1, 0.5), (2, 0.5)]);
}

#[test]
fn projects_outside_points_onto_the_nearest_hull_edge() {
    let space = square();
    expect(&weights(&space, &[2.0, 0.5]), &[(1, 0.5), (2, 0.5)]);
    expect(&weights(&space, &[0.5, 2.0]), &[(2, 0.5), (3, 0.5)]);
    expect(&weights(&space, &[0.5, -0.25]), &[(0, 0.5), (1, 0.5)]);
    expect(&weights(&space, &[-1.0, -1.0]), &[(0, 1.0)]);
    expect(&weights(&space, &[2.0, 2.0]), &[(2, 1.0)]);
    assert_eq!(
        weights(&space, &[2.0, 2.0]),
        vec![BlendWeight { point: 2, weight: 1.0 }]
    );
}

#[test]
fn refuses_bad_runtime_coordinates() {
    let line = line();
    let square = square();
    let points: Vec<[f32; 2]> = square.points.iter().map(|p| [p.at[0].rt, p.at[1].rt]).collect();
    let geometry = triangulate(&points, "/b").unwrap();
    assert!(evaluate(&line, None, &[1.0, 2.0]).is_err());
    assert!(evaluate_1d(&line, f32::NAN).is_err());
    assert!(evaluate(&square, None, &[0.5, 0.5]).is_err());
    assert!(evaluate_2d(&square, &geometry, [f32::INFINITY, 0.0]).is_err());
}
