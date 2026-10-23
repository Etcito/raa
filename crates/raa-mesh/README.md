# raa-mesh

Attributed meshes for the lit renderer. An `AttributedMesh` is an S2 `raa_geom::MeshData`
(positions, normals, u32 triangle-list indices) plus the optional S10 attributes (uvs,
tangents, linear vertex colours) and its sub-meshes. The crate validates meshes handed over
by JS (`addMesh`, S13 refusal codes), generates tangents and writes the GPU vertex streams
(S1 stream 0, S10 stream 1) and the index buffer.

## Public items

- `AttributedMesh { base, uvs, tangents, colors, submeshes }`
  - `from_mesh(MeshData)`: one sub-mesh over all indices, no attributes.
  - `validate()`: S13 rules, first broken rule as a `MeshError`.
  - `ensure_tangents()`: generates tangents when uvs are present and tangents absent.
  - `write_stream0_bytes(&mut Vec<u8>)`: S1 vertex, 24 B per vertex (position @0, normal @12).
  - `write_stream1_bytes(&mut Vec<u8>)`: S10 stream 1, 40 B per vertex (uv @0, tangent @8,
    colour @24). Absent attributes take the S10 defaults: uv (0, 0),
    `default_tangent(normal)`, colour (1, 1, 1, 1). It does not generate tangents: call
    `ensure_tangents()` first for meshes with uvs.
  - `write_index_bytes(&mut Vec<u8>)`: 4 B per index.
  - `bounds()`, `vertex_count()`, `triangle_count()`.

  The writers are little-endian (`to_le_bytes`, no unsafe) and append to `out` without
  clearing it.
- `SubMesh { first_index, index_count, material }`.
- `Colors::{Linear, Srgb8}` and `from_js_arrays(positions, normals, uvs, tangents, colors,
  indices, submeshes)`: builds and validates a mesh from flat JS arrays; sRGB8 colours go
  through the exact sRGB EOTF (alpha / 255, linear); normals and tangent xyz are stored
  normalised.
- `MeshError { code, message }` (`Display`, `Error`, `code()`), codes `mesh.empty`,
  `mesh.length_mismatch`, `mesh.not_finite`, `mesh.index_range`, `mesh.invalid_normal`,
  `mesh.invalid_tangent`, `mesh.submesh_range`, `mesh.submesh_overlap`.
- `generate_tangents(positions, normals, uvs, indices, out)`: per-triangle dP/du and dP/dv
  accumulated per vertex with corner-angle weights, Gram-Schmidt against the normal,
  w = sign(dot(cross(n, t), bitangent)). Triangles with degenerate uvs contribute nothing;
  a vertex without a usable tangent gets `default_tangent(n)`. `out` is cleared and gets
  one tangent per position. The output is deterministic (fixed order, f64 accumulation).
- `default_tangent(n)`: the branchless orthonormal basis of Duff et al. 2017 ("Building an
  Orthonormal Basis, Revisited"); xyz is b1, so `cross(n, b1)` is b2 and w = 1.

## How to test

```sh
cargo test -p raa-mesh
cargo clippy -p raa-mesh --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown -p raa-mesh
```

`tests/validate.rs` covers every S13 code and valid round-trips, `tests/tangents.rs`
orthonormality, handedness on a mirrored quad, determinism and the default basis, and
`tests/streams.rs` the byte offsets, strides, defaults and index byte count.
