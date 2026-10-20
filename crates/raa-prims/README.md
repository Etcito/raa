# raa-prims

SDF primitives turned into polygons straight from their parameters: a port of SDF
Modeler's `leafGeometry` maths (`sdfm:src/core/hybrid/leaf-mesh.ts@50b45d5`, see
`tools/sdf-ref/README.md`), with no distance-field sampling. Every mesh is closed,
outward-wound, deterministic and follows S2. The crate depends on `raa-math` and
`raa-geom` only.

## Public items

- `PrimKind`: `Sphere, Box, RoundBox, Capsule, Cylinder, Cone, Torus, Ellipsoid`;
  `ALL` (in this order), `as_str` / `parse` (S3 names), `is_convex` (false only for
  torus), `param_names` (S3 keys in S3 order).
- `PrimParams`: one variant per kind; `kind`, `from_params` (missing gives
  `geom.missing_param`, a wrong type or an extra name gives `geom.invalid_param`,
  then it validates), `to_params`, `validate` (S3 rules plus finiteness, each failure
  a `geom.invalid_param` naming the kind, the param and the rule), `bounds`
  (analytic), `reference_set(kind, set)` (S3 sets A and B; any char other than `'B'`
  gives A).
- `generate(p, q, out)`: validates `p` first (an invalid `p` leaves `out` untouched),
  clears `out`, clamps `q.segments` to [3, 256] (the field is public), then dispatches
  on the `PrimParams` variant to the per-kind tessellator.
- `triangle_count(p, q)`: the closed per-kind formula below, equal to the generated
  count, without building a mesh.
- `PrimProvider(pub PrimKind)`: implements `raa_geom::GeometryProvider`. `kind` and
  `param_names` come from `PrimKind`; `generate` and `triangle_count` go through
  `PrimParams::from_params`, so the params always have the provider's kind.
- `register_all(reg)`: registers the 8 kinds in `PrimKind::ALL` order. Calling it on a
  registry that already holds one of the kinds gives `geom.duplicate_kind`.
- `reference` (added with the SDF reference fixture): reads the S8 fixture and
  mapping v1, and compares a mesh under SDF_REF_TOL.

Internal layout: one file per kind (`sphere.rs`, `capsule.rs`, `cylinder.rs`,
`cone.rs`, `torus.rs` and `ellipsoid.rs` with `triangle_count(<fields>, segments)` and
`generate(<fields>, segments, out)`; `boxes.rs` with `box_*` and `round_box_*`).
`segments` is already clamped, and `out` is already cleared. Until a kind is ported,
its `generate` returns `geom.invalid_mesh` ("<kind>: tessellation not ported yet")
and its `triangle_count` returns 0.

## Quality to tessellation counts

Each kind has the fixed counts that `leafGeometry` hard-codes (mapping v1
`paramMap.<kind>.segments`, `tools/sdf-ref/README.md` §3.2 and §4). RAA derives every
count from `s = clamp(Quality.segments, 3, 256)` with

```
count(s) = max(min, multiple * round(ref * s / (24 * multiple)))     round half away from zero (f64::round)
```

so LOD0 (24) reproduces the reference defaults exactly. Every `multiple` is 1.

| kind | count (meaning) | ref | min | LOW 8 | LOD0 24 | HIGH 64 | source (sdfm@50b45d5) |
|---|---|---|---|---|---|---|---|
| sphere | `nu` azimuth segments | 40 | 3 | 13 | 40 | 107 | leaf-mesh.ts:103 |
| sphere | `nv` latitude bands (`nv + 1` rings incl. both poles) | 20 | 2 | 7 | 20 | 53 | leaf-mesh.ts:103 |
| box | none (same mesh at every quality) | — | — | — | — | — | — |
| roundBox | `cells` per face edge (each face a `(cells+1)²` grid) | 10 | 1 | 3 | 10 | 27 | leaf-mesh.ts:149 |
| capsule | `nu` azimuth segments | 32 | 3 | 11 | 32 | 85 | leaf-mesh.ts:177 |
| capsule | `nCap` rings per hemisphere (pole to equator) | 10 | 1 | 3 | 10 | 27 | leaf-mesh.ts:177 |
| cylinder | `nu` azimuth segments | 40 | 3 | 13 | 40 | 107 | leaf-mesh.ts:197 |
| cone | `nu` azimuth segments | 40 | 3 | 13 | 40 | 107 | leaf-mesh.ts:197 |
| torus | `nu` segments around the Y axis | 48 | 3 | 16 | 48 | 128 | leaf-mesh.ts:222 |
| torus | `nv` segments around the tube | 20 | 3 | 7 | 20 | 53 | leaf-mesh.ts:222 |
| ellipsoid | `nu` azimuth segments | 40 | 3 | 13 | 40 | 107 | leaf-mesh.ts:103 |
| ellipsoid | `nv` latitude bands | 20 | 2 | 7 | 20 | 53 | leaf-mesh.ts:103 |

Parameter mapping to the reference (mapping v1, axisMap identity, det 1): sphere
`radius` → `ellipsoid(r, r, r)`; ellipsoid `radii` → `ellipsoid(rx, ry, rz)`; box and
roundBox `halfExtents × 2` → `size`, roundBox `radius` unchanged (S3 guarantees the
reference's own clamp is inactive); cylinder `halfHeight × 2` → `height`, `radius` →
`frustum(height, r, r)`; cone `halfHeight × 2` → `height`, `radius` → `radiusBottom`,
`radiusTop = 0`; torus `majorRadius`, `minorRadius` unchanged. Capsule `radius` maps
unchanged; its `height` is the open finding of `tools/sdf-ref/README.md` §5 (the
reference's total height is `2 × halfHeight + 2 × radius`, an affine relation that
mapping v1 cannot express yet).

## Triangle count per kind

Closed formulas, copied from `tools/sdf-ref/README.md` §3.3, with the counts above.
The reference drops zero-area triangles (`done()`, leaf-mesh.ts:50-59), so pole and
apex fans are not counted.

| kind | triangles | raw vertices | LOW | LOD0 | HIGH |
|---|---|---|---|---|---|
| sphere | `2·nu·(nv − 1)` | `(nu+1)(nv−1)+2` | 156 | 1520 | 11128 |
| box | `12` | `24` | 12 | 12 | 12 |
| roundBox | `12·cells²` | `6(cells+1)²` | 108 | 1200 | 8748 |
| capsule | `2·nu·(2·nCap+1) − 2·nu` = `4·nu·nCap` | `(nu+1)(2·nCap+2)` | 132 | 1280 | 9180 |
| cylinder | `4·nu` (side `2·nu`, caps `nu` each) | `4·nu + 6` | 52 | 160 | 428 |
| cone | `2·nu` (side `nu` after the apex drop, base cap `nu`) | `3·nu + 4` | 26 | 80 | 214 |
| torus | `2·nu·nv` | `(nu+1)(nv+1)` | 224 | 1920 | 13568 |
| ellipsoid | `2·nu·(nv − 1)` | `(nu+1)(nv−1)+2` | 156 | 1520 | 11128 |

Port note (capsule): when `halfHeight = 0` the two equator rings coincide, the
reference drops the `2·nu` triangles of the middle band, and the count becomes
`4·nu·nCap − 2·nu`. The capsule port's `triangle_count` must follow whatever its
`generate` emits.

## Hard edges and seams (S2)

- Hard edges duplicate vertices, one copy per face with that face's normal: the 6
  box faces (4 vertices each), the cylinder's two cap discs and the cone's base disc
  (rim vertices duplicated against the side, cap normal `[0, ±1, 0]`, plus a centre
  vertex per cap).
- Smooth surfaces keep exact normals. Like the reference, the azimuth seam
  (`i = 0` and `i = nu`), the pole rows of capsule, the cone's
  apex row and both torus seam rows are exact f32 duplicates with equal normals. The
  roundBox face grids meet on the rounded borders with coincident positions and equal
  normals. Closedness is judged on the position-welded topology (WELD_EPS), and
  `uniquePositions` (rounding to 1e-6) must match the reference exactly, so a port
  reproduces the reference's vertex positions and triangle list.
- Sphere and ellipsoid poles are single shared vertices. The reference keeps
  `nu+1` copies, which NORMAL_RULE's per-vertex measure cannot accept for
  ellipsoid B at LOW. Sharing each pole measures its entire fan with the analytic
  pole normal. Positions and the triangle list still match the reference; only
  pole indices are remapped, and the azimuth seam stays duplicated (Q6).

## How to test

```
cargo test -p raa-prims                       # unit tests + tests/closed.rs, determinism.rs, sdf_reference.rs
cargo clippy -p raa-prims --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown -p raa-prims
cargo run --release -p raa-prims --example prims-report -- \
  [--only <pattern on kind>] [--json <path>] \
  [--compare crates/raa-prims/tests/fixtures/sdf-leaf-reference.json]
```

`prims-report` prints one line per kind × set (A, B) × quality (LOW, LOD0, HIGH):
triangles, vertices, uniquePositions, closed, volume, normal deviation and hash
(generated twice). With `--compare` it also prints the deltas against the fixture next
to SDF_REF_TOL. It exits 1 when any case fails `analysis::passes`, determinism or the
reference comparison. The milestone proof is
`/root/nmc/engines/bin/node proofs/m3-prims.mjs`.
