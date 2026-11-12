//! The sliced cluster bake (S29) through its public API: the result does not depend on the
//! budget, the progress contract holds on every step, and the refusals and counts are exact.

use raa_cluster::bake::{BakeProgress, BakeStage, BakeStatus, ClusterBake, WorkBudget, bake};
use raa_cluster::cut::ClusterMesh;
use raa_cluster::error::{EMPTY, INVALID_OPTIONS, NOT_FINISHED};
use raa_cluster::input::{BuildOptions, ClusterInput, OwnedClusterInput};
use raa_geom::MeshData;

/// Bakes `input` with `units` per step, checking the progress contract after every step.
fn stepped(
    input: ClusterInput<'_>,
    units: u32,
) -> Result<ClusterMesh, raa_cluster::error::ClusterError> {
    let mut bake = ClusterBake::new(input, &BuildOptions::default()).expect("valid input");
    let mut last: Option<BakeProgress> = None;
    loop {
        let status = bake.step(WorkBudget { units });
        let now = bake.progress();
        if let Some(before) = last {
            assert!(now.fraction >= before.fraction, "fraction decreased");
        }
        assert!(
            now.done_units <= now.total_units,
            "more units done than estimated"
        );
        match status {
            BakeStatus::Working => assert!(now.fraction < 1.0, "fraction reached 1 while working"),
            BakeStatus::Done => {
                assert_eq!(now.stage, BakeStage::Done);
                assert_eq!(now.fraction, 1.0);
                break;
            }
        }
        last = Some(now);
    }
    bake.finish()
}

fn sphere() -> OwnedClusterInput {
    OwnedClusterInput::displaced_sphere(3)
}

fn foliage() -> OwnedClusterInput {
    OwnedClusterInput::foliage_patch(600)
}

#[test]
fn budgets_give_equal_meshes_and_section_bytes() {
    for input in [sphere(), foliage()] {
        let one = stepped(input.as_input(), 1).expect("bakes");
        let seven = stepped(input.as_input(), 7).expect("bakes");
        let all = stepped(input.as_input(), u32::MAX).expect("bakes");
        assert_eq!(one, seven);
        assert_eq!(one, all);
        assert_eq!(one.to_section_bytes(), all.to_section_bytes());
        assert_eq!(seven.to_section_bytes(), all.to_section_bytes());
    }
}

#[test]
fn finish_before_done_is_refused() {
    let input = sphere();
    let bake = ClusterBake::new(input.as_input(), &BuildOptions::default()).expect("valid");
    assert_eq!(bake.finish().map_err(|e| e.code()), Err(NOT_FINISHED));
    let mut bake = ClusterBake::new(input.as_input(), &BuildOptions::default()).expect("valid");
    assert_eq!(bake.step(WorkBudget { units: 1 }), BakeStatus::Working);
    assert_eq!(bake.finish().map_err(|e| e.code()), Err(NOT_FINISHED));
}

#[test]
fn a_mesh_without_index_is_refused_by_the_constructor() {
    let mesh = MeshData {
        positions: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
    };
    let input = ClusterInput {
        mesh: &mesh,
        uvs: None,
    };
    let result = ClusterBake::new(input, &BuildOptions::default()).map(|_| ());
    assert_eq!(result.map_err(|e| e.code()), Err(EMPTY));
}

#[test]
fn collapsed_triangles_are_refused_by_the_finish() {
    // Three corners within 1e-10 of each other and a far fourth point: the triangle collapses.
    let mesh = MeshData {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0e-10, 0.0, 0.0],
            [0.0, 1.0e-10, 0.0],
            [1.0, 1.0, 1.0],
        ],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        indices: vec![0, 1, 2],
    };
    let input = ClusterInput {
        mesh: &mesh,
        uvs: None,
    };
    assert_eq!(
        stepped(input, 1).map_err(|e| e.code()),
        Err(EMPTY),
        "stepped bake"
    );
    assert_eq!(
        bake(input, &BuildOptions::default()).map_err(|e| e.code()),
        Err(EMPTY),
        "one-call bake"
    );
}

#[test]
fn single_triangle_is_one_cluster_one_leaf_and_the_root() {
    let input = OwnedClusterInput::single_triangle();
    let mesh = stepped(input.as_input(), 1).expect("bakes");
    assert_eq!(mesh.clusters.len(), 1);
    assert_eq!(mesh.nodes.len(), 2, "one leaf node and the root");
    assert_eq!(mesh.levels, 1);
    assert_eq!(mesh.source_triangles, 1);
    mesh.validate().expect("valid");
}

#[test]
fn invalid_options_are_refused_by_the_constructor() {
    let input = sphere();
    let options = BuildOptions {
        max_cluster_triangles: 200,
        ..BuildOptions::default()
    };
    let result = ClusterBake::new(input.as_input(), &options).map(|_| ());
    assert_eq!(result.map_err(|e| e.code()), Err(INVALID_OPTIONS));
}

#[test]
fn one_call_bake_equals_the_stepped_bake_and_validates() {
    let input = sphere();
    let whole = bake(input.as_input(), &BuildOptions::default()).expect("bakes");
    assert_eq!(whole, stepped(input.as_input(), 1).expect("bakes"));
    whole.validate().expect("valid");
}

#[test]
fn source_triangles_match_the_level_zero_clusters() {
    let mesh = stepped(sphere().as_input(), u32::MAX).expect("bakes");
    assert_eq!(mesh.source_triangles, 1280);
    assert!(
        mesh.levels > 1,
        "the sphere must go through simplified levels"
    );
    let level_zero: u64 = mesh
        .clusters
        .iter()
        .filter(|cluster| cluster.level() == 0)
        .map(|cluster| u64::from(cluster.triangle_count()))
        .sum();
    assert_eq!(level_zero, mesh.source_triangles);
}

#[test]
fn a_collapsed_second_triangle_leaves_one_source_triangle() {
    // The second triangle has two corners 1e-10 apart beside a far point, so it collapses.
    let mesh_data = MeshData {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0e-10, 1.0, 0.0],
        ],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
    };
    let input = ClusterInput {
        mesh: &mesh_data,
        uvs: None,
    };
    let mesh = bake(input, &BuildOptions::default()).expect("bakes");
    assert_eq!(mesh.source_triangles, 1);
}
