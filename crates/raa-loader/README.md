# raa-loader

Asset loading trait and the primitive-asset loader (prims v0, format `raa.prims`).
A loader turns bytes into a `LoadedAsset`: closed meshes in node-local primitive space
with their asset-space transforms, materials, anchors and colliders. The prims loader
lives with its format; glTF can be another `AssetLoader` later.

Depends on raa-math, raa-geom and raa-material (dev: raa-prims).

## Public items

- `json`: `JsonValue`, `parse` (RFC 8259, duplicate keys refused, depth <= 32),
  `write_pretty` (canonical text), `format_number` (JS Number-to-string rules).
- `LoadError { code, path, message }`: stable S5 code, JSON pointer (or `@<offset> line:col`
  for JSON syntax), human message. Display is `<code> at <path>: <message>`.
- `AssetLoader` (`format`, `sniff`, `load`) and `LoaderRegistry` (`new`, `register`,
  `load`, `formats`).
- `LoadContext { geometry, booleans, quality, strict_kinds }`.
- `LoadedAsset`, `LoadedMesh` (meshes in depth-first pre-order; reserved material `_default`
  first in `materials`).
- Booleans: `BoolOp`, `BooleanHost`, `BooleanTool` (mesh already in host local space),
  `BooleanResolver`, `NoBooleans` (copies tool-less hosts, refuses any tool).
- Colliders: `ColliderShape` (ball, cuboid, roundCuboid, capsuleY, cylinderY, coneY,
  convexHull) and `Collider { node_id, shape, transform }`.
- `prims`: `FORMAT`, `VERSION`, `DEFAULT_TRIANGLE_BUDGET` (1500), `MAX_NODES` (64),
  `MAX_LEVEL` (8); the document model (`PrimsAsset`, `PrimNode`, `TransformDoc`, `NodeOp`,
  `PrimMaterial`, `ColliderDoc`); `parse`, `to_canonical_json`, `estimate_triangles`,
  `build`, `PrimsLoader`; `prims::samples` (`sample_assets`, `invalid_cases`).

## S5 codes and their semantics

| code | when |
|---|---|
| `json.syntax` | malformed JSON or invalid UTF-8; path `@<offset> line:col` |
| `json.duplicate_key` | a key repeated in one object |
| `json.depth` | nesting deeper than 32 |
| `format.unknown` | `format` is not `raa.prims` |
| `format.version` | `version` is not 0 |
| `schema.missing_field` | a required key is absent |
| `schema.unknown_field` | a key the schema does not know |
| `schema.type` | a value of the wrong JSON type or arity |
| `value.not_finite` | a number that is not finite (for example `1e999` overflow) |
| `value.range` | a finite value outside its rule: primitive params (S3 rules, provider errors), quaternion norm off by more than 1e-3, scale component <= 0, `triangleBudget` not a positive integer, `op`, material values |
| `id.invalid` | a node id not matching `[A-Za-z0-9_-]+` |
| `id.duplicate` | a node id used twice |
| `material.unknown` | a node names a material that is not declared |
| `kind.unknown` | a kind outside the 8 S3 kinds (strict), or not registered in the geometry registry |
| `kind.params` | params missing, extra or of the wrong shape for the kind |
| `cap.nodes` | more than 64 nodes |
| `cap.depth` | a node at level 9 or deeper (roots are level 0) |
| `cap.budget` | the estimate before booleans, or the actual total after booleans, is above the budget; message `<n> triangles > budget <b>`, path `/triangleBudget` (empty when the key is absent) |
| `collision.torus_needs_override` | a torus union node and no `collision` override |
| `collision.shape` | an override entry with an unknown shape, a bad, missing or extra param (pointer at the param), or an unknown `node` |
| `boolean.unsupported` | a subtract/intersect child with no resolver installed (`NoBooleans` refuses any tool, with or without children); path is the first tool's pointer |
| `boolean.<csg code>` | a CSG resolver refusal, prefixing its CsgError code (for example `boolean.csg.tool_has_children`) |
| `loader.unknown_format` | no registered loader sniffs the bytes |
| `loader.duplicate_format` | a loader for that format is already registered |

Parse-time errors follow document order and the first one wins. `cap.budget`,
`collision.*` derivation and `boolean.*` are build-time. A collision override that names a
node is placed by its own transform after that node's world transform.

A subtract/intersect tool may have children: the parser validates them like any node
(ids, kinds, params, caps, materials) and does not refuse the tool. The build passes it to
the `BooleanResolver` with `BooleanTool.has_children = true`, and the resolver decides
(`NoBooleans` refuses any tool; a CSG resolver refuses with `boolean.csg.tool_has_children`).
A tool's children add no mesh, no collider and no triangle estimate.

## How to test

```sh
cargo test -p raa-loader
cargo run --release -p raa-loader --example prims-check -- [--only <regex>] [--json <path>]
```

`tests/roundtrip.rs` checks canonical round trips, `tests/refusals.rs` runs every invalid
case as parse then build (it prints `refused k/n`; currently `refused 33/33`), and
`tests/build.rs` builds the samples (closed meshes, mesh counts, budgets, colliders,
anchors, materials, determinism, the loader registry, non-strict kinds and boolean
delegation). The `prims-check` example runs the same round-trip, refusal and build cases
one line per case and exits 1 on any failure.

## Report: triangles per sample at LOD0

From `cargo test -p raa-loader --test build -- --nocapture`
(`raa_prims::register_all`, `Quality::LOD0`, `NoBooleans`):

| sample | triangles | budget | explicit `triangleBudget` |
|---|---|---|---|
| lamp | 3840 | 3840 | 3840 (cylinders 160 x2, sphere 1520, cone 80, torus 1920) |
| critter | 9680 | 9680 | 9680 (ellipsoid 1520, 2 spheres 1520, 4 capsules 1280) |
| crate | 1236 | 1500 | none (default) |
