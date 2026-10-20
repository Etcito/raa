# raa-csg

Simple booleans of a primitive against a convex host: exact convex face cutting, otherwise a
typed refusal. A boolean never fails silently: the output is a closed, consistently wound mesh
with positive volume, or a `CsgError` with a stable code and a message that names the nodes and numbers.

Depends on `raa-math`, `raa-geom` and `raa-loader` (dev: `raa-prims`). No third-party crates.

## Public items

- `Plane { n: [f64; 3], d: f64 }`: the points `p` with `n.p = d`. A point is outside when
  `n.p - d > eps`.
- `CsgError` and its `code()`:

  | variant | code |
  |---|---|
  | `HostNotConvex { kind }` | `csg.host_not_convex` |
  | `ToolNotConvex { kind }` | `csg.tool_not_convex` |
  | `ToolHasChildren { node }` | `csg.tool_has_children` |
  | `MultipleIntersect` | `csg.multiple_intersect` |
  | `OverlappingTools { a, b }` | `csg.overlapping_tools` |
  | `Coplanar { detail }` | `csg.coplanar` |
  | `EmptyResult` | `csg.empty_result` |
  | `NotWatertight { boundary_edges, nonmanifold_edges, inconsistent_edges }` | `csg.not_watertight` |
  | `Degenerate(String)` | `csg.degenerate` |

- `convex_planes(mesh, eps) -> Result<Vec<Plane>, CsgError>`: face planes in f64, zero-area
  triangles skipped. Planes merge when normals agree within 1e-9 and offsets within `eps`, in
  first-occurrence order. The mesh is convex when every vertex has `n.p - d <= eps` for every plane.
  Otherwise the function returns an error, and the caller maps it to HostNotConvex or ToolNotConvex.
- `intersect_convex(host, tool, out)`: host faces clipped to the inside of every tool plane,
  plus tool faces clipped to the inside of every host plane. Disjoint inputs give EmptyResult.
- `subtract_convex(host, host_planes, tools, out)`: removes every tool from the convex host.
  Tools must have pairwise disjoint bboxes (otherwise OverlappingTools). Tool faces are kept
  inside the host with reversed winding. A host entirely inside a tool gives EmptyResult.
- `CsgResolver` (implements `raa_loader::BooleanResolver`). It works in a fixed order:
  1. no tools: the host is copied unchanged;
  2. any tool with children: `ToolHasChildren`;
  3. more than one intersect tool: `MultipleIntersect` (both nodes named);
  4. host kind not convex, or `convex_planes(host)` fails: `HostNotConvex`;
  5. each tool in document order: kind not convex or planes fail: `ToolNotConvex`;
  6. the single intersect tool, if present, is applied first (the result stays convex);
  7. all subtract tools at once, through `subtract_convex` against the post-intersect planes.

  Errors become `LoadError { code: "boolean." + CsgError::code(), path: "", message }`. The
  loader fills an empty path with the first tool's JSON pointer. For that reason every message
  names the offending host and tool node ids.
- `CsgCase { id, expected, asset }` and `cases()`: the fixed table of CSG cases (prims v0 text,
  set A parameters). `expected` is `"ok"` or the expected `boolean.csg.*` code.

## Eps rules

- `eps = 1e-6 x max(1, d)`, where `d` is the f64 diagonal of the union of the host and all tool
  bboxes (CSG_EPS). The resolver computes it once per host and uses it for every step.
- A vertex within `eps` of a plane is ON, and both sides keep it as a boundary.
- Two planes are coincident when they are parallel with either sign (normals within 1e-9) and
  their offsets are within `eps`. Coincident tool and host planes whose faces overlap give `Coplanar`.
- A split point on an edge is computed from the edge endpoints in lexicographic order, so
  neighbouring faces agree on it.
- Output is welded on an `eps` grid (first occurrence wins). T-junctions are repaired in up to
  3 passes, and the result is checked with `raa_geom::analysis::analyze`. A result that is not
  closed, not consistent or without positive volume gives `NotWatertight` with the counts.
- The volume identity `V(host - tool) = V(host) - V(host intersect tool)` must hold within a
  relative difference of 1e-4.
- Every iteration order is fixed (no hash maps), so the same input gives byte-identical output.

## Limitations

- roundBox hosts and tools are refused with `boolean.csg.host_not_convex` /
  `boolean.csg.tool_not_convex`: the raa-prims LOD0 roundBox mesh (SDF Modeler port) folds its
  face-grid quads along a fixed diagonal, so the mesh is not convex within CSG_EPS even though
  the analytic shape is. `tests/cases.rs` asserts the host refusal.
- Torus hosts and tools are refused (the kind is not convex).

## How to test

```sh
cargo test -p raa-csg
cargo run --release -p raa-csg --example csg-cases -- [--only <regex>] [--json <path>]
```

The example loads every case through `PrimsLoader` + `CsgResolver` at LOD0. It prints one line
per case and exits 1 when a case differs from its expectation, when an ok case is not watertight
or misses the volume identity, or when a refusal has an empty message. It runs up to `NMC_CPUS`
threads. The milestone proof is `/root/nmc/engines/bin/node proofs/m5-booleans.mjs`.
