# raa-geom

Geometry provider trait, shared mesh data (S2) and mesh analysis. Depends on `raa-math` only.

## Public items

- `MeshData` — indexed triangle list (`positions`, `normals`, `indices: Vec<u32>`): `new`, `clear` (keeps capacity), `vertex_count`, `triangle_count`, `validate`, `append_transformed`, `bounds`, `write_vertex_bytes` (S1 layout, 24 B stride), `write_index_bytes`.
- `Params` / `ParamValue` — named parameters in insertion order (`Scalar(f32)` or `Vec3([f32; 3])`), with typed getters `scalar` and `vec3`.
- `Quality` — tessellation segments: `LOW` = 8, `LOD0` = 24, `HIGH` = 64; `Quality::new` clamps to [3, 256].
- `GeometryProvider` — trait for a geometry kind: `kind`, `param_names`, `is_convex`, `triangle_count`, `generate` (clears the output, deterministic, validates parameters, closed outward mesh).
- `GeometryRegistry` — providers by kind: `register`, `get`, `kinds` (registration order), `generate`.
- `GeomError` — the crate's single error type, with stable codes:
  `geom.unknown_kind`, `geom.missing_param`, `geom.invalid_param`, `geom.duplicate_kind`, `geom.invalid_mesh`.
- `analysis` — `MeshReport`, `analyze` (welded topology, area, volume, normal checks), `passes` (lists every failed rule with its numbers), `mesh_hash` (FNV-1a 64 over LE bytes of positions, normals, indices), `unique_positions_1e6` (S8 count).
- `builtin::DebugCube` — kind `debug_cube`, param `half > 0`; 24 vertices, 12 triangles, hard edges, convex.

## Rules

- WELD_EPS: positions closer than 1e-6 x max(1, bbox diagonal) are welded (deterministic quantised grid) before topology checks.
- NORMAL_RULE: per vertex, abs(len - 1) <= 1e-4, angle to the area-weighted mean of incident face normals <= 45 degrees, and no negative dot with an incident face normal.

## Testing

```sh
cargo test -p raa-geom
```
