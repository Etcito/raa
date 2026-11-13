//! The S29 guarantees on generated inputs through the public bake and cut API: the displaced
//! sphere keeps the cluster limits and its source count, errors and bounds grow toward the root,
//! two bakes agree, the foliage patch keeps its covered area across levels, the single triangle
//! sits on the grid within one step, the empty and out-of-range meshes are refused, and the
//! build options bound every cluster.

use std::collections::BTreeMap;

use raa_cluster::bake::bake;
use raa_cluster::cut::{ClusterMesh, cut_triangles, select_cut};
use raa_cluster::error::{ClusterError, EMPTY, INVALID_MESH};
use raa_cluster::input::{BuildOptions, ClusterInput, OwnedClusterInput};
use raa_cluster::packed::{
    MAX_CLUSTER_TRIANGLES, MAX_CLUSTER_VERTICES, PackedCluster, PackedNode, ROOT_PARENT_ERROR,
};
use raa_geom::MeshData;

/// The summed triangle area, in f64, of the decoded uniform cut at `error_m`.
fn cut_area(mesh: &ClusterMesh, error_m: f32) -> f64 {
    let cut = cut_triangles(mesh, &select_cut(mesh, error_m));
    let mut area = 0.0;
    for triangle in cut.indices.chunks_exact(3) {
        let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
        let (Some(pa), Some(pb), Some(pc)) = (
            cut.positions.get(a as usize),
            cut.positions.get(b as usize),
            cut.positions.get(c as usize),
        ) else {
            continue;
        };
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]].map(f64::from);
        let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]].map(f64::from);
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        area += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    }
    area
}

/// For each DAG level above 0 present in the mesh, the smallest `lod_error` among its clusters;
/// the levels ascend.
fn level_errors(mesh: &ClusterMesh) -> Vec<f32> {
    let mut levels: BTreeMap<u32, f32> = BTreeMap::new();
    for cluster in &mesh.clusters {
        let level = cluster.level();
        if level == 0 {
            continue;
        }
        let entry = levels.entry(level).or_insert(f32::INFINITY);
        *entry = entry.min(cluster.lod_error);
    }
    levels.into_values().collect()
}

/// Whether `(outer_center, outer_radius)` contains `(inner_center, inner_radius)`: the f32
/// rounding allowance of `Sphere::contains`, in f64, so the packed records are judged as built.
fn sphere_contains(outer: ([f32; 3], f32), inner: ([f32; 3], f32)) -> bool {
    let offset = [
        inner.0[0] - outer.0[0],
        inner.0[1] - outer.0[1],
        inner.0[2] - outer.0[2],
    ]
    .map(f64::from);
    let distance = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
    distance + f64::from(inner.1) <= f64::from(outer.1) * (1.0 + 1.0e-3) + 1.0e-5
}

/// The clusters of a leaf node, by its `first` and `count`.
fn leaf_clusters<'a>(mesh: &'a ClusterMesh, node: &PackedNode) -> &'a [PackedCluster] {
    let first = node.first as usize;
    let end = first + node.count as usize;
    mesh.clusters
        .get(first..end)
        .unwrap_or_else(|| panic!("a leaf names clusters {first}..{end} outside the mesh"))
}

#[test]
fn displaced_sphere_passes_validation_with_the_s29_limits() {
    let input = OwnedClusterInput::displaced_sphere(4);
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the sphere bakes");
    mesh.validate()
        .unwrap_or_else(|error| panic!("the bake fails validation: {error}"));
    for (i, cluster) in mesh.clusters.iter().enumerate() {
        let triangles = cluster.triangle_count();
        let vertices = cluster.vertex_count();
        assert!(
            (1..=MAX_CLUSTER_TRIANGLES as u32).contains(&triangles),
            "cluster {i} has {triangles} triangles"
        );
        assert!(
            (1..=MAX_CLUSTER_VERTICES as u32).contains(&vertices),
            "cluster {i} has {vertices} vertices"
        );
    }
    assert_eq!(mesh.source_triangles, 5120, "source triangles");
    let finest = cut_triangles(&mesh, &select_cut(&mesh, 0.0));
    assert_eq!(
        finest.indices.len() / 3,
        5120,
        "the finest cut must hold every source triangle"
    );
    assert!(mesh.levels >= 2, "levels {}", mesh.levels);
    let top = mesh.levels - 1;
    let parentless: Vec<&PackedNode> = mesh
        .nodes
        .iter()
        .filter(|node| node.is_leaf == 1 && node.max_parent_error == ROOT_PARENT_ERROR)
        .collect();
    assert!(
        !parentless.is_empty(),
        "no leaf carries ROOT_PARENT_ERROR: /{}",
        mesh.nodes.len()
    );
    assert!(
        parentless.iter().any(|node| leaf_clusters(&mesh, node)
            .iter()
            .any(|cluster| cluster.level() == top)),
        "no parentless cluster at level {top}"
    );
}

#[test]
fn errors_and_bounds_are_monotonic_toward_the_root() {
    let input = OwnedClusterInput::displaced_sphere(4);
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the sphere bakes");
    for (i, node) in mesh.nodes.iter().enumerate() {
        if node.is_leaf == 1 {
            let first = node.first as usize;
            for (offset, cluster) in leaf_clusters(&mesh, node).iter().enumerate() {
                let cluster_index = first + offset;
                assert!(
                    cluster.lod_error <= node.max_parent_error,
                    "leaf {i} cluster {cluster_index}: lod_error {} above max_parent_error {}",
                    cluster.lod_error,
                    node.max_parent_error
                );
                assert!(
                    sphere_contains(
                        (node.lod_center, node.lod_radius),
                        (cluster.lod_center, cluster.lod_radius)
                    ),
                    "leaf {i}: lod sphere does not contain cluster {cluster_index}"
                );
            }
        } else {
            let first = node.first as usize;
            let end = first + node.count as usize;
            let children = mesh
                .nodes
                .get(first..end)
                .unwrap_or_else(|| panic!("inner node {i} names children {first}..{end}"));
            for (offset, child) in children.iter().enumerate() {
                let child_index = first + offset;
                assert!(
                    node.max_parent_error >= child.max_parent_error,
                    "node {i}: max_parent_error {} below node {child_index}'s {}",
                    node.max_parent_error,
                    child.max_parent_error
                );
                assert!(
                    sphere_contains((node.center, node.radius), (child.center, child.radius)),
                    "node {i}: sphere does not contain node {child_index}"
                );
                assert!(
                    sphere_contains(
                        (node.lod_center, node.lod_radius),
                        (child.lod_center, child.lod_radius)
                    ),
                    "node {i}: lod sphere does not contain node {child_index}"
                );
            }
        }
    }
}

#[test]
fn two_bakes_of_the_same_input_are_equal() {
    let input = OwnedClusterInput::displaced_sphere(3);
    let options = BuildOptions::default();
    let first = bake(input.as_input(), &options).expect("the first bake");
    let second = bake(input.as_input(), &options).expect("the second bake");
    assert_eq!(first, second, "two bakes differ");
    assert_eq!(
        first.to_section_bytes(),
        second.to_section_bytes(),
        "two bakes give different section bytes"
    );
}

#[test]
fn foliage_patch_keeps_its_covered_area_across_levels() {
    let input = OwnedClusterInput::foliage_patch(2000);
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the foliage bakes");
    mesh.validate()
        .unwrap_or_else(|error| panic!("the foliage bake fails validation: {error}"));
    let finest = cut_triangles(&mesh, &select_cut(&mesh, 0.0));
    let finest_triangles = finest.indices.len() / 3;
    let coarsest = select_cut(&mesh, 1.0e29);
    assert!(!coarsest.is_empty(), "the coarsest cut is empty");
    let coarse_triangles = cut_triangles(&mesh, &coarsest).indices.len() / 3;
    assert!(
        coarse_triangles < finest_triangles,
        "coarsest cut {coarse_triangles} is not below the finest {finest_triangles}"
    );
    let finest_area = cut_area(&mesh, 0.0);
    assert!(finest_area > 0.0, "the finest cut has area {finest_area}");
    let levels = level_errors(&mesh);
    assert!(!levels.is_empty(), "no DAG level above 0");
    for error in levels {
        let area = cut_area(&mesh, error);
        let deviation = (area - finest_area).abs() / finest_area;
        assert!(
            deviation <= 0.10,
            "cut at {error} has area {area} against the finest {finest_area}: {deviation} deviation"
        );
    }
}

#[test]
fn single_triangle_is_one_cluster_on_the_grid() {
    let input = OwnedClusterInput::single_triangle();
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the triangle bakes");
    assert_eq!(mesh.clusters.len(), 1, "clusters {}", mesh.clusters.len());
    let cut = cut_triangles(&mesh, &[0]);
    assert_eq!(cut.positions.len(), 3, "positions {}", cut.positions.len());
    for (i, position) in input.mesh.positions.iter().enumerate() {
        let nearest = cut
            .positions
            .iter()
            .map(|decoded| {
                [0, 1, 2]
                    .map(|axis| (decoded[axis] - position[axis]).abs())
                    .into_iter()
                    .fold(0.0f32, f32::max)
            })
            .fold(f32::INFINITY, f32::min);
        assert!(
            nearest <= mesh.quant_step,
            "input vertex {i} {position:?} is {nearest} from the grid, one step is {}",
            mesh.quant_step
        );
    }
}

#[test]
fn empty_and_out_of_range_meshes_are_refused() {
    let options = BuildOptions::default();
    let empty = MeshData::new();
    let input = ClusterInput {
        mesh: &empty,
        uvs: None,
    };
    let code = bake(input, &options).map(|_| ()).map_err(|e| e.code());
    assert_eq!(code, Err(EMPTY), "an empty mesh must be refused");

    let broken = MeshData {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        indices: vec![0, 1, 9],
    };
    let input = ClusterInput {
        mesh: &broken,
        uvs: None,
    };
    let error: ClusterError =
        bake(input, &options).expect_err("an out-of-range index must be refused");
    assert_eq!(error.code(), INVALID_MESH, "{}", error.message);
}

#[test]
fn options_bound_every_cluster() {
    let input = OwnedClusterInput::displaced_sphere(3);
    let options = BuildOptions {
        max_cluster_triangles: 32,
        max_cluster_vertices: 64,
        ..BuildOptions::default()
    };
    let mesh = bake(input.as_input(), &options).expect("the sphere bakes");
    mesh.validate()
        .unwrap_or_else(|error| panic!("the limited bake fails validation: {error}"));
    for (i, cluster) in mesh.clusters.iter().enumerate() {
        assert!(
            cluster.triangle_count() <= 32,
            "cluster {i} has {} triangles over 32",
            cluster.triangle_count()
        );
        assert!(
            cluster.vertex_count() <= 64,
            "cluster {i} has {} vertices over 64",
            cluster.vertex_count()
        );
    }
}
