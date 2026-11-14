//! The S29 guarantees on generated inputs through the public bake and cut API: the displaced
//! sphere keeps the cluster limits and its source count, errors and bounds grow toward the root,
//! two bakes agree, the foliage patch keeps its covered area across levels, the single triangle
//! sits on the grid within one step, the empty and out-of-range meshes are refused, the build
//! options bound every cluster, the open strip keeps its boundary at every cut and the seam mesh
//! stays closed across its duplicated edges.

use std::collections::{BTreeMap, BTreeSet};

use raa_cluster::bake::bake;
use raa_cluster::cut::{ClusterMesh, cut_triangles, select_cut};
use raa_cluster::error::{ClusterError, EMPTY, INVALID_MESH};
use raa_cluster::input::{BuildOptions, ClusterInput, OwnedClusterInput};
use raa_cluster::packed::{
    MAX_CLUSTER_TRIANGLES, MAX_CLUSTER_VERTICES, PackedCluster, PackedNode, ROOT_PARENT_ERROR,
    unpack_triangle,
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

/// The numbers of the flat array under `key` in a fixture text, up to its closing bracket.
fn fixture_array(text: &str, key: &str) -> Vec<f32> {
    let start = text
        .find(&format!("\"{key}\""))
        .unwrap_or_else(|| panic!("the fixture has no {key} key"));
    let rest = &text[start..];
    let open = rest
        .find('[')
        .unwrap_or_else(|| panic!("{key} has no opening bracket"));
    let close = rest[open..]
        .find(']')
        .unwrap_or_else(|| panic!("{key} has no closing bracket"));
    rest[open + 1..open + close]
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|token| !token.is_empty())
        .map(|token| {
            token
                .parse::<f32>()
                .unwrap_or_else(|_| panic!("{key} has a bad number {token:?}"))
        })
        .collect()
}

/// Reads a fixture of flat `positions`, optional `uvs` and `indices` arrays and computes the
/// vertex normals as the normalised area-weighted sums of the incident face normals.
fn read_fixture(text: &str) -> OwnedClusterInput {
    let positions = fixture_array(text, "positions");
    let uvs = text.contains("\"uvs\"").then(|| fixture_array(text, "uvs"));
    let indices = fixture_array(text, "indices");
    let place = |i: usize| [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
    let mut normals = vec![[0.0f32; 3]; positions.len() / 3];
    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let (pa, pb, pc) = (place(a), place(b), place(c));
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let face = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for vertex in [a, b, c] {
            for (component, value) in normals[vertex].iter_mut().zip(face) {
                *component += value;
            }
        }
    }
    for normal in &mut normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        *normal = if length > 0.0 {
            normal.map(|component| component / length)
        } else {
            [0.0, 1.0, 0.0]
        };
    }
    OwnedClusterInput {
        mesh: MeshData {
            positions: positions
                .chunks_exact(3)
                .map(|p| [p[0], p[1], p[2]])
                .collect(),
            normals,
            indices: indices.iter().map(|index| *index as u32).collect(),
        },
        uvs: uvs.map(|flat| flat.chunks_exact(2).map(|uv| [uv[0], uv[1]]).collect()),
    }
}

/// For each undirected grid edge of the uniform cut at `error_m`, keyed with its smaller end
/// first, the number of triangles using it in each direction: the keyed one, then its reverse.
fn cut_edge_uses(mesh: &ClusterMesh, error_m: f32) -> BTreeMap<([u16; 3], [u16; 3]), [u32; 2]> {
    let mut uses: BTreeMap<([u16; 3], [u16; 3]), [u32; 2]> = BTreeMap::new();
    for index in select_cut(mesh, error_m) {
        let Some(cluster) = mesh.clusters.get(index as usize) else {
            continue;
        };
        let first_vertex = cluster.vertex_offset as usize;
        let vertex_end = first_vertex.saturating_add(cluster.vertex_count() as usize);
        let Some(vertices) = mesh.vertices.get(first_vertex..vertex_end) else {
            continue;
        };
        let first_triangle = cluster.triangle_offset as usize;
        let triangle_end = first_triangle.saturating_add(cluster.triangle_count() as usize);
        let Some(triangles) = mesh.triangles.get(first_triangle..triangle_end) else {
            continue;
        };
        for &packed in triangles {
            let corners = unpack_triangle(packed).map(|corner| {
                vertices
                    .get(corner as usize)
                    .map(|vertex| [vertex.data[0], vertex.data[1], vertex.data[2]])
            });
            let [Some(a), Some(b), Some(c)] = corners else {
                continue;
            };
            for (from, to) in [(a, b), (b, c), (c, a)] {
                let (key, direction) = if from <= to {
                    ((from, to), 0)
                } else {
                    ((to, from), 1)
                };
                uses.entry(key).or_insert([0, 0])[direction] += 1;
            }
        }
    }
    uses
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

#[test]
fn open_strip_keeps_its_boundary_at_every_threshold() {
    let input = read_fixture(include_str!("fixtures/open-strip.json"));
    assert_eq!(input.mesh.positions.len(), 153, "vertices");
    assert_eq!(input.mesh.indices.len(), 256 * 3, "indices");
    assert!(input.uvs.is_none(), "the open strip has no uvs");
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the strip bakes");
    mesh.validate()
        .unwrap_or_else(|error| panic!("the strip bake fails validation: {error}"));
    let boundary = |uses: &BTreeMap<([u16; 3], [u16; 3]), [u32; 2]>| {
        uses.iter()
            .filter(|(_, counts)| counts[0] + counts[1] == 1)
            .map(|(edge, _)| *edge)
            .collect::<BTreeSet<_>>()
    };
    let finest = boundary(&cut_edge_uses(&mesh, 0.0));
    assert!(!finest.is_empty(), "the finest cut has no boundary");
    for error in [0.0, 0.01, 0.1, 1.0e29] {
        let uses = cut_edge_uses(&mesh, error);
        for (edge, counts) in &uses {
            assert!(
                counts[0] + counts[1] <= 2,
                "at {error} edge {edge:?} is used by {counts:?}"
            );
        }
        assert_eq!(boundary(&uses), finest, "the boundary changed at {error}");
    }
}

#[test]
fn seam_mesh_cuts_are_closed_on_the_grid() {
    let input = read_fixture(include_str!("fixtures/seam-mesh.json"));
    assert_eq!(input.mesh.positions.len(), 294, "vertices");
    assert_eq!(input.mesh.indices.len(), 432 * 3, "indices");
    assert_eq!(input.uvs.as_ref().map(Vec::len), Some(294), "uvs");
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the cube bakes");
    mesh.validate()
        .unwrap_or_else(|error| panic!("the cube bake fails validation: {error}"));
    assert_eq!(
        cut_triangles(&mesh, &select_cut(&mesh, 0.0)).indices.len() / 3,
        432,
        "the finest cut must hold every source triangle"
    );
    for error in [0.0, 0.01, 0.1, 1.0e29] {
        let uses = cut_edge_uses(&mesh, error);
        assert!(!uses.is_empty(), "the cut at {error} is empty");
        for (edge, counts) in &uses {
            assert_eq!(
                counts,
                &[1, 1],
                "the cut at {error} leaves edge {edge:?} with uses {counts:?}"
            );
        }
    }
}

/// The geometric unit normals of a decoded cut's triangles; degenerate triangles are skipped.
fn cut_normals(mesh: &ClusterMesh, error_m: f32) -> Vec<[f64; 3]> {
    let cut = cut_triangles(mesh, &select_cut(mesh, error_m));
    let mut normals = Vec::new();
    for triangle in cut.indices.chunks_exact(3) {
        let corner = |i: usize| triangle.get(i).and_then(|&v| cut.positions.get(v as usize));
        let (Some(a), Some(b), Some(c)) = (corner(0), corner(1), corner(2)) else {
            continue;
        };
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]].map(f64::from);
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]].map(f64::from);
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if length > 1.0e-12 {
            normals.push(n.map(|x| x / length));
        }
    }
    normals
}

/// How far a normal leans from the horizontal plane, in degrees: 0 for an upright blade's face.
fn tilt_degrees(normal: [f64; 3]) -> f64 {
    normal[1].abs().clamp(0.0, 1.0).asin().to_degrees()
}

/// The cells of a `cells` x `cells` grid over the square `[-half, half]` in x and z that the
/// triangles of the cut touch when seen from above (an upright blade is a line there).
fn covered_cells(
    mesh: &ClusterMesh,
    error_m: f32,
    half: f64,
    cells: usize,
) -> BTreeSet<(usize, usize)> {
    const STEPS: usize = 24;
    let cut = cut_triangles(mesh, &select_cut(mesh, error_m));
    let cell = |value: f64| -> Option<usize> {
        let at = ((value + half) / (2.0 * half) * cells as f64).floor();
        (at >= 0.0 && at < cells as f64).then_some(at as usize)
    };
    let mut covered = BTreeSet::new();
    for triangle in cut.indices.chunks_exact(3) {
        let corner = |i: usize| triangle.get(i).and_then(|&v| cut.positions.get(v as usize));
        let (Some(a), Some(b), Some(c)) = (corner(0), corner(1), corner(2)) else {
            continue;
        };
        for i in 0..=STEPS {
            for j in 0..=(STEPS - i) {
                let (s, t) = (i as f64 / STEPS as f64, j as f64 / STEPS as f64);
                let at = |axis: usize| {
                    f64::from(a[axis]) * (1.0 - s - t)
                        + f64::from(b[axis]) * s
                        + f64::from(c[axis]) * t
                };
                if let (Some(x), Some(z)) = (cell(at(0)), cell(at(2))) {
                    covered.insert((x, z));
                }
            }
        }
    }
    covered
}

#[test]
fn thinned_foliage_stays_upright_and_covers_the_patch_footprint() {
    // 4096 blades on a 0.05 * 64 = 3.2 m square centred at the origin.
    let input = OwnedClusterInput::foliage_patch(4096);
    let half = 1.6;
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the foliage bakes");
    mesh.validate()
        .unwrap_or_else(|error| panic!("the foliage bake fails validation: {error}"));
    let levels = level_errors(&mesh);
    assert!(
        levels.len() >= 4,
        "only {} DAG levels above 0",
        levels.len()
    );
    // The thinned levels the page shows: a third and a half of the way up the DAG, each on a
    // grid as fine as that level can still cover.
    let third = levels
        .get(levels.len().div_ceil(3) - 1)
        .copied()
        .expect("a third level");
    let middle = levels
        .get(levels.len().div_ceil(2) - 1)
        .copied()
        .expect("a middle level");

    let finest_normals = cut_normals(&mesh, 0.0);
    let finest_tilt = finest_normals
        .iter()
        .copied()
        .map(tilt_degrees)
        .fold(0.0, f64::max);
    for error in [third, middle] {
        let normals = cut_normals(&mesh, error);
        assert!(
            normals.len() < finest_normals.len(),
            "the level at error {error} has {} triangles, the finest {}",
            normals.len(),
            finest_normals.len()
        );
        // The quantisation grid moves a normal by well under a degree; a re-oriented blade by far
        // more.
        let tilt = normals
            .iter()
            .copied()
            .map(tilt_degrees)
            .fold(0.0, f64::max);
        assert!(
            tilt <= finest_tilt + 2.0,
            "blades at error {error} lean up to {tilt} degrees, the finest ones {finest_tilt}"
        );
    }

    for (error, cells) in [(third, 16), (middle, 12)] {
        let finest_cells = covered_cells(&mesh, 0.0, half, cells);
        assert_eq!(
            finest_cells.len(),
            cells * cells,
            "the finest level leaves cells of a {cells} x {cells} grid uncovered"
        );
        let thinned_cells = covered_cells(&mesh, error, half, cells);
        let empty: Vec<_> = finest_cells.difference(&thinned_cells).collect();
        assert!(
            empty.is_empty(),
            "the level at error {error} leaves {} of {} cells of a {cells} x {cells} grid empty: {empty:?}",
            empty.len(),
            finest_cells.len()
        );
    }
}

/// The foliage patch of the virtual-geometry page (`examples/virtual-geometry/lib/content.js`,
/// `foliagePatch()`), rebuilt here with the same numbers: 64 x 64 upright blades 0.03 m wide and
/// 0.25 m high, one per cell of a 4 m square centred at the origin, offset and yawed by a
/// 32-bit LCG seeded with 0x2f6e2b1 (x, z, yaw per cell in row-major order).
fn page_foliage_patch() -> OwnedClusterInput {
    const GRID: u32 = 64;
    const EXTENT: f64 = 4.0;
    let mut state = 0x02f6_e2b1_u32;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(state) / 4_294_967_296.0
    };
    let cell = EXTENT / f64::from(GRID);
    let (hw, height) = (0.015, 0.25);
    let mut mesh = MeshData::new();
    let mut uvs = Vec::new();
    for j in 0..GRID {
        for i in 0..GRID {
            let bx = -EXTENT / 2.0 + (f64::from(i) + next()) * cell;
            let bz = -EXTENT / 2.0 + (f64::from(j) + next()) * cell;
            let (sin, cos) = (next() * std::f64::consts::TAU).sin_cos();
            // right = (cos, 0, -sin) and the normal is right x up = (sin, 0, cos).
            let normal = [sin, 0.0, cos];
            let base = mesh.positions.len() as u32;
            for (lx, ly, u, v) in [
                (-hw, 0.0, 0.0, 0.0),
                (hw, 0.0, 1.0, 0.0),
                (hw, height, 1.0, 1.0),
                (-hw, height, 0.0, 1.0),
            ] {
                mesh.positions
                    .push([(bx + lx * cos) as f32, ly as f32, (bz - lx * sin) as f32]);
                mesh.normals.push(normal.map(|x| x as f32));
                uvs.push([u, v]);
            }
            mesh.indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    OwnedClusterInput {
        mesh,
        uvs: Some(uvs),
    }
}

/// The smallest `lod_error` of the clusters at DAG level `level`, as the scene picks its cut.
fn smallest_error_at_level(mesh: &ClusterMesh, level: u32) -> f32 {
    mesh.clusters
        .iter()
        .filter(|cluster| cluster.level() == level)
        .map(|cluster| cluster.lod_error)
        .fold(f32::INFINITY, f32::min)
}

#[test]
fn page_foliage_thinned_cut_stays_upright_and_covers_the_patch_footprint() {
    let input = page_foliage_patch();
    let half = 2.0;
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the foliage bakes");
    mesh.validate()
        .unwrap_or_else(|error| panic!("the foliage bake fails validation: {error}"));
    // The scene's right patch: the level a third of the way up the DAG (scenes/bake.js).
    let level = mesh.levels.div_ceil(3);
    let error = smallest_error_at_level(&mesh, level);
    assert!(error.is_finite(), "no cluster at DAG level {level}");

    let finest_normals = cut_normals(&mesh, 0.0);
    let thinned_normals = cut_normals(&mesh, error);
    assert!(
        thinned_normals.len() < finest_normals.len(),
        "the thinned cut has {} triangles, the finest {}",
        thinned_normals.len(),
        finest_normals.len()
    );
    let finest_tilt = finest_normals
        .iter()
        .copied()
        .map(tilt_degrees)
        .fold(0.0, f64::max);
    let tilt = thinned_normals
        .iter()
        .copied()
        .map(tilt_degrees)
        .fold(0.0, f64::max);
    assert!(
        tilt <= finest_tilt + 2.0,
        "blades at error {error} lean up to {tilt} degrees, the finest ones {finest_tilt}"
    );

    // The scene's own fact `foliage-area-kept`: the summed area stays within 15 percent.
    let (finest_area, thinned_area) = (cut_area(&mesh, 0.0), cut_area(&mesh, error));
    println!("page foliage area: {thinned_area} thinned, {finest_area} finest m2");
    assert!(
        (thinned_area - finest_area).abs() <= 0.15 * finest_area,
        "area {thinned_area} thinned against {finest_area} finest"
    );

    let cells = 16;
    let finest_cells = covered_cells(&mesh, 0.0, half, cells);
    let thinned_cells = covered_cells(&mesh, error, half, cells);
    let empty: Vec<_> = finest_cells.difference(&thinned_cells).collect();
    println!(
        "page foliage: level {level}, error {error}, {} of {} triangles, {} of {} cells empty",
        thinned_normals.len(),
        finest_normals.len(),
        empty.len(),
        finest_cells.len()
    );
    assert!(
        empty.is_empty(),
        "the page's thinned cut at error {error} leaves {} of {} cells of a {cells} x {cells} grid empty: {empty:?}",
        empty.len(),
        finest_cells.len()
    );
}

/// The sdfm monument (plinth, column, ball) with its own SDF normals and its part-atlas uvs, whose
/// seams split a third of the positions. Every corner of every simplified cut must carry a normal
/// the source has at that same grid position: a corner that kept the removed vertex's attribute
/// drew the ball's normals down the column ("dripping" columns, sdfm card 7923, 2026-11-14).
#[test]
fn monument_cuts_keep_source_normals_at_every_corner() {
    let input = monument_input();
    assert_eq!(input.mesh.indices.len(), 1294 * 3, "indices");
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the monument bakes");
    mesh.validate()
        .unwrap_or_else(|error| panic!("the monument bake fails validation: {error}"));
    let cell = |p: [f32; 3]| -> [i64; 3] {
        [0, 1, 2].map(|a| ((p[a] - mesh.quant_origin[a]) / mesh.quant_step).round() as i64)
    };
    let mut source: BTreeMap<[i64; 3], Vec<[f32; 3]>> = BTreeMap::new();
    for (p, n) in input.mesh.positions.iter().zip(&input.mesh.normals) {
        source.entry(cell(*p)).or_default().push(*n);
    }
    let finest = cut_triangles(&mesh, &select_cut(&mesh, 0.0)).indices.len() / 3;
    assert_eq!(
        finest, 1294,
        "the finest cut must hold every source triangle"
    );
    let mut simplified_cuts = 0;
    for error in [0.0, 0.003, 0.01, 0.03, 0.1, 1.0e29] {
        let cut = cut_triangles(&mesh, &select_cut(&mesh, error));
        if cut.indices.len() / 3 < finest {
            simplified_cuts += 1;
        }
        let mut stray = Vec::new();
        for (v, (p, n)) in cut.positions.iter().zip(&cut.normals).enumerate() {
            let best = source.get(&cell(*p)).map_or(-2.0, |list| {
                list.iter()
                    .map(|s| s[0] * n[0] + s[1] * n[1] + s[2] * n[2])
                    .fold(-2.0f32, f32::max)
            });
            if best < 0.999 {
                stray.push((v, best));
            }
        }
        assert!(
            stray.is_empty(),
            "the cut at {error} ({} triangles) has {} corners whose normal the source lacks at \
             their position (vertex, best dot): {:?}",
            cut.indices.len() / 3,
            stray.len(),
            &stray[..stray.len().min(8)]
        );
    }
    assert!(
        simplified_cuts >= 2,
        "the monument must simplify at least twice"
    );
}

/// The sdfm monument fixture as a cluster input: its positions, SDF normals and atlas uvs.
fn monument_input() -> OwnedClusterInput {
    let text = include_str!("fixtures/monument-atlas.json");
    let positions = fixture_array(text, "positions");
    let normals = fixture_array(text, "normals");
    let uvs = fixture_array(text, "uvs");
    let indices = fixture_array(text, "indices");
    let triple = |flat: &[f32]| -> Vec<[f32; 3]> {
        flat.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect()
    };
    OwnedClusterInput {
        mesh: MeshData {
            positions: triple(&positions),
            normals: triple(&normals),
            indices: indices.iter().map(|index| *index as u32).collect(),
        },
        uvs: Some(uvs.chunks_exact(2).map(|uv| [uv[0], uv[1]]).collect()),
    }
}

type D3 = [f64; 3];

fn d3(v: [f32; 3]) -> D3 {
    v.map(f64::from)
}

fn d3_sub(a: D3, b: D3) -> D3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn d3_dot(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `a * wa + b * wb + c * wc`.
fn d3_mix(corners: [D3; 3], weights: [f64; 3]) -> D3 {
    [0, 1, 2].map(|axis| (0..3).map(|i| corners[i][axis] * weights[i]).sum())
}

fn d3_unit(a: D3) -> D3 {
    let length = d3_dot(a, a).sqrt();
    if length > 0.0 {
        a.map(|x| x / length)
    } else {
        a
    }
}

fn d3_face(p: [D3; 3]) -> D3 {
    let (u, v) = (d3_sub(p[1], p[0]), d3_sub(p[2], p[0]));
    d3_unit([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ])
}

/// The barycentric weights of the point of triangle `t` closest to `p` (Ericson, 5.1.5).
fn closest_weights(p: D3, t: [D3; 3]) -> [f64; 3] {
    let (ab, ac, ap) = (d3_sub(t[1], t[0]), d3_sub(t[2], t[0]), d3_sub(p, t[0]));
    let (d1, d2) = (d3_dot(ab, ap), d3_dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = d3_sub(p, t[1]);
    let (d3v, d4) = (d3_dot(ab, bp), d3_dot(ac, bp));
    if d3v >= 0.0 && d4 <= d3v {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3v * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3v <= 0.0 {
        let v = d1 / (d1 - d3v);
        return [1.0 - v, v, 0.0];
    }
    let cp = d3_sub(p, t[2]);
    let (d5, d6) = (d3_dot(ab, cp), d3_dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3v * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3v >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3v) / ((d4 - d3v) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let denominator = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denominator, vc * denominator);
    [1.0 - v - w, v, w]
}

/// Every simplified cluster of the monument shades within what its `lod_error` claims. The shading
/// error of a cluster triangle at a point is the chord between its interpolated normal and the
/// source's interpolated normal at the nearest source point (on a face within 60 degrees of the
/// triangle's), times the triangle's longest edge: metres, the unit of `NORMAL_ERROR_WEIGHT`.
/// Bound, at 15 points of each triangle of each cluster above level 0: the 99th percentile is at
/// most `lod_error` and every point at most 1.5 x `lod_error` (a normal change inherited from the
/// level below counts again on the coarser level's longer triangles). A point whose interpolated
/// normal is shorter than 0.5 before normalising has no shading direction and is skipped: the
/// plinth's bottom face interpolates opposite corner normals in the source itself. Before the
/// simplifier weighed shading, position-only quadrics let the level-1 clusters (the 646-triangle
/// cut the Game view drew, sdfm 2026-11-14) reach 180 x their `lod_error` (normals 141 degrees off,
/// 22 degrees at the 95th percentile) for 12.8 mm off the surface: the dripping columns.
#[test]
fn monument_clusters_shade_within_their_lod_error() {
    let input = monument_input();
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the monument bakes");
    let source_positions: Vec<D3> = input.mesh.positions.iter().map(|p| d3(*p)).collect();
    let source_normals: Vec<D3> = input.mesh.normals.iter().map(|n| d3_unit(d3(*n))).collect();
    // Each source triangle with its face normal and bounding sphere (centroid, radius).
    let source: Vec<([usize; 3], D3, D3, f64)> = input
        .mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            let corners = [t[0], t[1], t[2]].map(|i| i as usize);
            let p = corners.map(|i| source_positions[i]);
            let centre = d3_mix(p, [1.0 / 3.0; 3]);
            let radius = p
                .iter()
                .map(|q| d3_dot(d3_sub(*q, centre), d3_sub(*q, centre)).sqrt())
                .fold(0.0, f64::max);
            (corners, d3_face(p), centre, radius)
        })
        .collect();
    let source_normal_near = |point: D3, face: D3| -> Option<D3> {
        let mut best: Option<(f64, D3)> = None;
        for (corners, normal, centre, radius) in &source {
            let reach = d3_dot(d3_sub(point, *centre), d3_sub(point, *centre)).sqrt() - radius;
            if d3_dot(*normal, face) < 0.5 || best.is_some_and(|(d, _)| reach > d) {
                continue;
            }
            let p = corners.map(|i| source_positions[i]);
            let weights = closest_weights(point, p);
            let offset = d3_sub(point, d3_mix(p, weights));
            let distance = d3_dot(offset, offset).sqrt();
            if best.is_none_or(|(d, _)| distance < d) {
                best = Some((
                    distance,
                    d3_mix(corners.map(|i| source_normals[i]), weights),
                ));
            }
        }
        best.map(|(_, normal)| normal)
    };

    let mut checked = 0;
    for (index, cluster) in mesh.clusters.iter().enumerate() {
        if cluster.level() == 0 {
            continue;
        }
        let cut = cut_triangles(&mesh, &[index as u32]);
        let mut errors = Vec::new();
        for triangle in cut.indices.chunks_exact(3) {
            let corners = [triangle[0], triangle[1], triangle[2]].map(|i| i as usize);
            let p = corners.map(|i| d3(cut.positions[i]));
            let n = corners.map(|i| d3(cut.normals[i]));
            let face = d3_face(p);
            let longest = [(0, 1), (1, 2), (2, 0)]
                .map(|(a, b)| d3_dot(d3_sub(p[a], p[b]), d3_sub(p[a], p[b])).sqrt())
                .into_iter()
                .fold(0.0, f64::max);
            for i in 0..=4 {
                for j in 0..=4 - i {
                    let weights = [1.0 - (i + j) as f64 / 4.0, i as f64 / 4.0, j as f64 / 4.0];
                    let drawn = d3_mix(n, weights);
                    let Some(wanted) = source_normal_near(d3_mix(p, weights), face) else {
                        continue;
                    };
                    if d3_dot(drawn, drawn) < 0.25 || d3_dot(wanted, wanted) < 0.25 {
                        continue;
                    }
                    let chord = d3_sub(d3_unit(drawn), d3_unit(wanted));
                    errors.push(d3_dot(chord, chord).sqrt() * longest);
                }
            }
        }
        errors.sort_by(f64::total_cmp);
        let Some(&worst) = errors.last() else {
            continue;
        };
        let p99 = errors[(errors.len() - 1) * 99 / 100];
        let claimed = f64::from(cluster.lod_error);
        println!(
            "monument cluster {index}: level {}, lod_error {claimed:.4} m, shading p99 {p99:.4} m, max {worst:.4} m",
            cluster.level()
        );
        assert!(
            p99 <= claimed && worst <= 1.5 * claimed,
            "monument cluster {index} (level {}) shades beyond its lod_error {claimed:.4} m: \
             99th percentile {p99:.4} m, worst {worst:.4} m",
            cluster.level()
        );
        checked += 1;
    }
    assert!(
        checked >= 4,
        "only {checked} simplified clusters were checked"
    );
}
