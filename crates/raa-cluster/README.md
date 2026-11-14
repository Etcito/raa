# raa-cluster

The CPU cluster-DAG builder of RAA's virtualized geometry (S29). A saved mesh is welded onto one
16-bit grid, split into clusters of at most 128 triangles and 255 vertices, simplified level by
level into a hierarchy, and packed into 16 B vertex, 48 B cluster and 48 B node records. The
crate has no GPU code, depends only on `raa-math` and `raa-geom`, and is deterministic: the same
input gives the same bytes for every budget sequence.

## Pipeline

1. **Weld.** Positions snap to the quantisation grid and coincident vertices merge; triangles that
   collapse under quantisation drop. A mesh of separate thin pieces is welded again with a grid margin.
2. **Per level** (until one active cluster remains or a round makes no progress):
   - *partition* compacts the triangles into clusters (`SMALL_CLUSTER = 64`, `MAX_SEARCH_RINGS = 6`);
   - *group* collects adjacent clusters (`TARGET_GROUP_SIZE = 16`, `MAX_GROUP_SIZE = 24`, `MIN_GROUP_SIZE = 6`);
   - *simplify* reduces each group to `target_ratio` of its triangles with its outer boundary locked,
     and the result is split back into the parents of the group's clusters. Foliage groups made of
     separate pieces are *thinned* instead, keeping their covered area.
3. **Hierarchy.** Sibling sets become leaves of a tree with fan-out 8. A node's sphere bounds its
   children, and `max_parent_error` is the largest parent error below it. Roots carry `ROOT_PARENT_ERROR`.
4. **Output.** `ClusterMesh` holds the packed arrays and the header fields; `to_section_bytes()`
   gives the cluster and node byte arrays of the saved-mesh section (S30).

The build is a sliced step API: `ClusterBake::step(WorkBudget { units })` runs bounded units (one
weld or partition batch of at most 4096 triangles, one group simplification of at most 24 clusters,
one hierarchy or packing batch of at most 1024 clusters), and `bake()` runs it to the end.

## Module map

| module | contents |
|---|---|
| `error` | `ClusterError { code, message, path }` and the S29 code constants |
| `packed` | `PackedVertex`, `PackedCluster`, `PackedNode`, triangle packing, constants |
| `bounds` | `Sphere` fitting, merging and containment |
| `input` | `ClusterInput`, `OwnedClusterInput`, `BuildOptions` and validation |
| `dvec` | private f64 vector helpers (`#[doc(hidden)]`) |
| `weld` | grid weld, octahedral normals, the weld state |
| `partition` | limits, Morton order, adjacency, cluster growth, the partition state |
| `group` | cluster adjacency, group seeding, reach and small-group merge |
| `simplify` | corner types, quadrics, candidates, the simplifier and `simplify()` |
| `thin` | piece shapes, wide pieces, `thin()` and placement |
| `hierarchy` | leaves, the batched node tree and the hierarchy state |
| `dag` | cluster build, re-split, per-group reduction, levels and the pack state |
| `bake` | `ClusterBake`, `WorkBudget`, `BakeStatus`, `BakeStage`, `BakeProgress`, `bake()` |
| `cut` | `ClusterMesh`, `SectionHeader`, section bytes, `validate()`, `select_cut`, `cut_triangles` |
| `store` | `ClusterStore`, the CPU registry of sections per mesh key |

## Usage

One call bakes a mesh to the end:

```rust
let input = raa_cluster::ClusterInput { mesh: &mesh, uvs: None };
let section = raa_cluster::bake(input, &raa_cluster::BuildOptions::default())?;
section.validate()?;
let (cluster_bytes, node_bytes) = section.to_section_bytes();
```

A sliced bake runs the same units in bounded steps. The result does not depend on the budgets:

```rust
let mut bake = raa_cluster::ClusterBake::new(input, &options)?;
while bake.step(raa_cluster::WorkBudget { units: 1 }) == raa_cluster::BakeStatus::Working {
    let fraction = bake.progress().fraction; // never decreases, below 1 until done
}
let section = bake.finish()?;
```

`select_cut(&section, error_m)` gives the uniform cut and `cut_triangles(&section, &clusters)`
decodes it; `ClusterMesh::from_section_parts` reads a saved section back with full validation.

## Store

`ClusterStore` keys sections by the caller's mesh key. `attach` stores ready sections,
`start_bake` queues one bake per input, and `step_all(&mut clock)` runs units while the clock
allows, lowest key first. `state(key)` reports `None`, `Baking` with the mean progress,
`Ready` (`Attached` or `Runtime`) or `Refused` with a code and message. `generation()` rises
whenever a ready set appears, is replaced or is removed.

## Unit rule

No loop of one unit visits more than 4096 triangles or 1024 clusters. A whole-array sort of keys
and the single piece search over the welded mesh are each one unit. One weld or partition batch
is at most 4096 triangles, one group simplification or thinning at most 24 clusters, and one
hierarchy or packing batch at most 1024 clusters.

## Guarantees pinned by the tests

The integration tests (`tests/packed.rs`, `tests/sliced.rs`, `tests/cuts.rs`, `tests/bake.rs`) and
the unit tests of `store` check, on generated inputs (displaced sphere, foliage patch, single
triangle) and on the JSON fixtures (`tests/fixtures/open-strip.json`, `seam-mesh.json`,
`monument-atlas.json`):

- the finest cut is the welded source mesh, and every uniform cut of a closed source is closed,
  with each undirected grid edge used by exactly two triangles in opposite directions;
- `lod_error` and the bounds grow monotonically toward the root, and every section validates;
- `lod_error` covers the shading change as well as the geometric one: the simplifier weighs and
  reports a turned interpolated normal as its chord times the triangle's longest edge
  (`NORMAL_ERROR_WEIGHT`), and every simplified monument cluster shades within its `lod_error`;
- the output bytes are the same for unit budgets and for one unbounded step;
- a mesh without triangle, a collapsed mesh and invalid options are refused with their codes.

## What is not ported

Paging (`solstice-builder/src/paging.rs`, `solstice-asset/src/{file,paged}.rs`), texture
coordinate tiers (`solstice-builder/src/texture.rs`) and the solstice streaming module are not
part of this crate. Vertex colours and tangents are not carried by the 16 B vertex. rayon becomes
sequential code, glam becomes `raa-math` plus private f64 helpers, bytemuck becomes `to_le_bytes`
writers, and `FxHashMap` becomes `BTreeMap` or sorted vectors so that no result depends on a hash order.

## Attribution

Adapted from solstice (MIT) at revision 7e465b599c13da165948ebaa44df824c36c3ae07, Copyright (c)
2026 SamG-Coder. Each adapted module names its solstice source file in its module doc. The
dvec and store modules are new code.
