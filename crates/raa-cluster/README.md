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
