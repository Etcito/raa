//! The uniform cut (S29) through its public API on the displaced sphere at subdivision 4: the
//! finest cut is the welded source mesh, every cut is closed, the triangle count falls as the
//! threshold grows, every cluster is selected at its own error, and the decoded cut is well formed.

use std::collections::BTreeMap;

use raa_cluster::bake::bake;
use raa_cluster::cut::{ClusterMesh, cut_triangles, select_cut};
use raa_cluster::input::{BuildOptions, OwnedClusterInput};
use raa_cluster::packed::unpack_triangle;
use raa_cluster::weld::WeldState;

/// A corner on the 16-bit grid of the bake.
type Grid = [u16; 3];

fn sphere() -> (OwnedClusterInput, ClusterMesh) {
    let input = OwnedClusterInput::displaced_sphere(4);
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the sphere bakes");
    (input, mesh)
}

/// The thresholds of the closure and count tests: the finest cut, two grid steps, four errors and
/// a threshold above every error.
fn thresholds(mesh: &ClusterMesh) -> [f32; 6] {
    [0.0, 2.0 * mesh.quant_step, 0.005, 0.02, 0.1, 1e29]
}

/// The triangles of `clusters` as grid corners, read from the packed vertices in cluster order.
fn packed_triangles(mesh: &ClusterMesh, clusters: &[u32]) -> Vec<[Grid; 3]> {
    let mut out = Vec::new();
    for &c in clusters {
        let cluster = &mesh.clusters[c as usize];
        let first = cluster.vertex_offset as usize;
        let vertices = &mesh.vertices[first..first + cluster.vertex_count() as usize];
        let first = cluster.triangle_offset as usize;
        for &packed in &mesh.triangles[first..first + cluster.triangle_count() as usize] {
            out.push(unpack_triangle(packed).map(|k| {
                let d = vertices[k as usize].data;
                [d[0], d[1], d[2]]
            }));
        }
    }
    out
}

/// The corners of `t` rotated so the smallest comes first; the cyclic order is kept.
fn normalised(t: [Grid; 3]) -> [Grid; 3] {
    [[t[0], t[1], t[2]], [t[1], t[2], t[0]], [t[2], t[0], t[1]]]
        .into_iter()
        .min()
        .unwrap_or(t)
}

/// The welded triangles of `input` on the grid of an unpadded weld, as grid corners.
fn source_triangles(input: &OwnedClusterInput) -> Vec<[Grid; 3]> {
    let mut weld = WeldState::new(&input.bounds(), 0.0);
    while !weld.step(input) {}
    let welded = weld.finish();
    welded
        .triangles
        .iter()
        .map(|t| t.map(|corner| welded.grid(welded.positions[corner.position as usize])))
        .collect()
}

/// Every directed grid edge is used once and its reverse once: each undirected edge is shared by
/// exactly two triangles, in opposite directions, and no triangle is degenerate.
fn is_closed(triangles: &[[Grid; 3]]) -> bool {
    let mut directed: BTreeMap<(Grid, Grid), u32> = BTreeMap::new();
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if a == b {
                return false;
            }
            *directed.entry((a, b)).or_default() += 1;
        }
    }
    directed
        .iter()
        .all(|(&(a, b), &n)| n == 1 && directed.get(&(b, a)) == Some(&1))
}

#[test]
fn finest_cut_is_the_welded_source_mesh() {
    let (input, mesh) = sphere();
    let level0: Vec<u32> = (0..mesh.clusters.len() as u32)
        .filter(|&c| mesh.clusters[c as usize].level() == 0)
        .collect();
    let finest = select_cut(&mesh, 0.0);
    assert_eq!(finest, level0);
    let mut cut: Vec<_> = packed_triangles(&mesh, &finest)
        .into_iter()
        .map(normalised)
        .collect();
    let mut source: Vec<_> = source_triangles(&input)
        .into_iter()
        .map(normalised)
        .collect();
    cut.sort_unstable();
    source.sort_unstable();
    assert_eq!(cut, source);
}

#[test]
fn every_uniform_cut_is_closed() {
    let (_, mesh) = sphere();
    for error in thresholds(&mesh) {
        let cut = select_cut(&mesh, error);
        assert!(!cut.is_empty(), "threshold {error} selects nothing");
        assert!(
            is_closed(&packed_triangles(&mesh, &cut)),
            "threshold {error} is open"
        );
    }
}

#[test]
fn triangle_count_falls_as_the_threshold_grows() {
    let (_, mesh) = sphere();
    let counts: Vec<usize> = thresholds(&mesh)
        .into_iter()
        .map(|error| {
            cut_triangles(&mesh, &select_cut(&mesh, error))
                .indices
                .len()
                / 3
        })
        .collect();
    assert!(counts.windows(2).all(|w| w[1] <= w[0]), "{counts:?}");
    assert!(counts.last() < counts.first(), "{counts:?}");
}

#[test]
fn every_cluster_is_selected_at_its_own_error() {
    let (_, mesh) = sphere();
    for (c, cluster) in mesh.clusters.iter().enumerate() {
        let cut = select_cut(&mesh, cluster.lod_error);
        assert!(cut.contains(&(c as u32)), "cluster {c} is not in its cut");
    }
}

#[test]
fn decoded_cut_is_well_formed() {
    let (_, mesh) = sphere();
    for error in thresholds(&mesh) {
        let clusters = select_cut(&mesh, error);
        let cut = cut_triangles(&mesh, &clusters);
        assert_eq!(cut.indices.len() % 3, 0);
        assert_eq!(cut.cluster_of_triangle.len(), cut.indices.len() / 3);
        assert_eq!(cut.normals.len(), cut.positions.len());
        assert_eq!(cut.uvs.len(), cut.positions.len());
        assert!(
            cut.indices
                .iter()
                .all(|&i| (i as usize) < cut.positions.len())
        );
        assert!(cut.cluster_of_triangle.iter().all(|c| clusters.contains(c)));
        for n in &cut.normals {
            let length = n.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((length - 1.0).abs() <= 1e-3, "normal length {length}");
        }
    }
    assert!(cut_triangles(&mesh, &[u32::MAX]).indices.is_empty());
}
