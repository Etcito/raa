# tools/sdf-ref — SDF Modeler `leafGeometry` reference and mapping v1

This directory is the written source of truth for `gen-leaf-fixtures.mjs` (the
one-off fixture generator) and for the `raa-prims` ports: what SDF Modeler's
`leafGeometry` does per kind, every tessellation count it uses, and the binding
`mapping v1` object that connects RAA's S3 parameters to it. The reference is read
**read-only** through the cross-repository alias `sdfm` (branch node 3312); nothing
inside it is ever written.

> Status: the mapping is complete. It is filled per the Q3 amendment (per-component
> vec3 keys, count rules) and the Q5 amendment (one optional additive term, used only
> by `capsule.height = 2 × halfHeight + 2 × radius`). §5 has no open finding left.
> `gen-leaf-fixtures.mjs` (card 3296) validates it and wrote the 48-case S8 fixture
> `crates/raa-prims/tests/fixtures/sdf-leaf-reference.json` (§6), capsule included.

## 1. Access path and commit

- Alias `sdfm` (read-only). Card 3295 saw it at `<sdfm checkout>` (git objects
  only); card 3296 sees a materialised checkout at
  `<sdfm checkout>` (all tracked files, clean, no
  `node_modules`). The path differs per card, which is why the tool takes `--ref`.
- Read: `src/core/hybrid/leaf-mesh.ts` (781 lines, 38841 bytes) and the imports it
  pulls in: `../doc/types.js`, `../math/vec.js`, `../sdf/profile.js`, `../registry.js`,
  `../sweep/{constants,pack,open,stations}.js`,
  `../nurbs/{piece-params,piece-tess}.js` (the last two only for the non-basic
  kinds), `../errors.js`; the per-kind parameter specs are in
  `src/core/sdf/primitives/basic.ts` (478 lines).
- Commit: **`50b45d5862be55ba85492d6bb0e457ed3b244dd6`** — the 40-hex HEAD of
  `sdfm` (`refs/heads/main`, 2026-10-15), i.e. what the alias resolves to.
  `sha256(leaf-mesh.ts)` = `048318c72eb806201535cbfa83acd5d96056de1da0f866e8a56f94068cd63083`;
  that is the exact text the `literal` path would copy and the hash the fixture
  records. Throughout this file `@50b45d5` shortens this commit.
- Card 3295 found the alias worktree empty (only `.git`), so its reads went
  through the git object store (`git -C <sdfm checkout> show 50b45d58...:<path>`)
  and its measurements through `git archive HEAD` extracted into
  `<scratch>/sdfm` (outside both repositories). Card 3296 read the materialised
  checkout directly and re-measured every §3 number with the tool. Nothing inside
  `sdfm` was ever written.

### 1.1 Loading `leaf-mesh.ts` (node built-in type stripping)

`node` is v24.21.0 (type stripping on by default). A direct
`import` of the `.ts` file fails for two independent reasons, both observed:

1. The source writes `.js` specifiers for `.ts` siblings: `ERR_MODULE_NOT_FOUND:
   Cannot find module '<...>/src/core/sdf/profile.js' imported from
   '<...>/src/core/hybrid/leaf-mesh.ts'`; node's type stripping does not rewrite
   `.js` → `.ts` on resolution.
2. `src/core/svgplus/shape-region.ts` (pulled in by leaf-mesh.ts via
   `../nurbs/piece-params.js`, for the `nurbsPiece` kind) imports the npm package
   `polygon-clipping`, and the alias has no `node_modules`.

`gen-leaf-fixtures.mjs` therefore loads the module with node built-ins only
(`module.registerHooks`, synchronous hooks, no npm dependency, nothing written):

- `resolve`: node's own resolution first; only when it fails with
  `ERR_MODULE_NOT_FOUND` for an importer under `<ref>/src/`, a relative `.js`
  specifier is retried as its existing `.ts` sibling, and a bare npm specifier that
  is not installed in the checkout (here only `polygon-clipping`) resolves to a
  stub whose default export **throws on any use** (property access, call or
  `new`). A stub can therefore never alter a mesh: if a reference path used the
  package, the tool would fail. The 8 basic kinds never reach it (measured: every
  case is produced, and the tool prints the stubbed packages).
- `load`: serves the in-memory copies of §1.2 (`?sdf-ref=call`,
  `?sdf-ref=literal&...` URLs of `leaf-mesh.ts`) as `module-typescript`; their
  relative imports resolve to the same files, so copies and the unmodified module
  share one instance of every dependency.

The esbuild fallback of the original module text cannot run here (the checkout has
no `node_modules`, so no `esbuild`) and is not needed: the hooks resolve every
import with node built-ins. With an installed `node_modules`, node's own
resolution finds `polygon-clipping` and no stub is used. The sandbox details are in
`docs/dogfood/m3.md`.

### 1.2 How the `call` path reaches the counting functions

The five generators that own the 8 kinds' counts (`ellipsoid`, `roundBox`,
`capsule`, `frustum`, `torus`) are module-private, and `leafGeometry` itself takes
no quality or count argument. Path `call` therefore evaluates an **in-memory copy**
of `leaf-mesh.ts` whose only change is one appended line:

```ts
export { ellipsoid, roundBox, capsule, frustum, torus }
```

The copy is evaluated with type stripping (§1.1); no literal is substituted,
nothing is written inside `sdfm`, and at LOD0 the explicit-count call must equal
unmodified `leafGeometry` output byte for byte (positions, normals and indices,
asserted per kind and set). The tool decides the path per kind from the `source`
lines, never from a hard-coded list:

- `call` when every count of the kind is a parameter `<count> = <ref>` of the
  `function` declared on its `source` line, that function is the one
  `leafGeometry` calls for the kind (`<f>:695-703`), and the parameters after the
  shape arguments are exactly the listed counts (a hard-coded count missing from
  the mapping is refused). The generator gets the same shape arguments as in
  `leafGeometry`'s switch, followed by the explicit counts.
- `literal` when every count is instead exactly one literal `<ref>` on its
  `source` line: one in-memory copy per kind and quality, with only those literals
  replaced, then `leafGeometry(kind, sdfInput)` of that copy.
- A kind without counts (`box`, `segments: []`) calls unmodified `leafGeometry` at
  every quality and is recorded as `call` (a call with an empty count list).

Mixed paths inside one kind, or a `source` line that fits neither, are refused.
All 8 kinds resolve to `call` at `@50b45d5`. The `literal` path was exercised once
by forcing it for the 7 counted kinds (scratch run, §6): all 48 cases were identical
to the `call` run.

## 2. Quoted excerpts (sdfm@50b45d5)

`<f>` is `src/core/hybrid/leaf-mesh.ts`.
`<b>` is `src/core/sdf/primitives/basic.ts`.
All line numbers are at commit `50b45d5862be55ba85492d6bb0e457ed3b244dd6`.

`<f>:31-32@50b45d5` — the kinds `leafGeometry` can generate (the 8 S3 kinds are the
first 8 of 12):

```ts
/** Kinds `leafGeometry` can generate (others, e.g. plane or plugin primitives, keep their SDF / baked display). */
export const LEAF_KINDS = ['sphere', 'box', 'roundBox', 'capsule', 'cylinder', 'cone', 'torus', 'ellipsoid', 'extrude', 'lathe', 'sweep', 'nurbsPiece'] as const
```

`<f>:20-29@50b45d5` — the vertex layout: SoA `Float32Array`, triangle list, no UV,
same vertex count for positions and normals; `materials` only for sweeps (never set
by the 8 kinds):

```ts
export interface LeafGeom {
  positions: Float32Array
  normals: Float32Array
  indices: Uint32Array
  /**
   * Per-triangle material (sweep sections with per-span materialIds): `ids[tri[t]]` is the material id of triangle t, null =
   * the node's own material. A vertex belongs to the triangles of one material only (vertices are split at material changes).
   */
  materials?: { ids: (string | null)[]; tri: Uint16Array }
}
```

`<f>:690-703@50b45d5` — the entry point: no quality/segment input anywhere, the 8
basic kinds call the private generators without counts, and an advanced param
(present at all, even `false`) makes it return `null`:

```ts
export function leafGeometry(kind: string, params: Record<string, ParamValue> = {}, doc?: SdfDocument, nodeId?: string): LeafGeom | null {
  // v23 advanced shape options (taper, per-edge/corner bevels, cap bevels) have no generated mesh: keep the SDF display
  if (registry.primitive(kind)?.params.some((s) => s.advanced && params[s.name] !== undefined)) return null
  // a drafted extrude (sloped walls) has no generated mesh either
  if (kind === 'extrude' && typeof params.draft === 'number' && params.draft !== 0) return null
  switch (kind) {
    case 'sphere': { const r = num(kind, params, 'radius'); return ellipsoid(r, r, r) }
    case 'ellipsoid': { const r = vec3(kind, params, 'radii'); return ellipsoid(r[0], r[1], r[2]) }
    case 'box': { const s = vec3(kind, params, 'size'); return box(s[0], s[1], s[2]) }
    case 'roundBox': { const s = vec3(kind, params, 'size'); return roundBox(s[0], s[1], s[2], num(kind, params, 'radius')) }
    case 'capsule': return capsule(num(kind, params, 'height'), num(kind, params, 'radius'))
    case 'cylinder': { const r = num(kind, params, 'radius'); return frustum(num(kind, params, 'height'), r, r) }
    case 'cone': return frustum(num(kind, params, 'height'), num(kind, params, 'radiusBottom'), num(kind, params, 'radiusTop'))
    case 'torus': return torus(num(kind, params, 'majorRadius'), num(kind, params, 'minorRadius'))
```

`<f>:48-59@50b45d5` — quad winding ("counter-clockwise seen from outside") and the
zero-area triangle drop that removes pole/apex fans at the end of every generator:

```ts
  /** Quad a b c d (counter-clockwise seen from outside). */
  quad(a: number, b: number, c: number, d: number): void { this.idx.push(a, b, c, a, c, d) }
  done(): LeafGeom {
    // drop zero-area triangles (pole and axis rows of revolved surfaces)
    const p = this.pos, keep: number[] = []
    for (let t = 0; t < this.idx.length; t += 3) {
      const a = this.idx[t] * 3, b = this.idx[t + 1] * 3, c = this.idx[t + 2] * 3
      const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]], e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]]
      const cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0]
      if (cx * cx + cy * cy + cz * cz > 1e-20) keep.push(this.idx[t], this.idx[t + 1], this.idx[t + 2])
    }
    return { positions: Float32Array.from(this.pos), normals: Float32Array.from(this.nrm), indices: Uint32Array.from(keep) }
  }
```

`<f>:136-146@50b45d5` — `fixWinding`, applied by every generator, flips any triangle
whose geometric normal points against its summed vertex normals, so winding is
outward regardless of the grid orientation:

```ts
/** Flips triangles whose geometric normal points against the vertex normals (keeps generators simple and robust). */
function fixWinding(g: LeafGeom): LeafGeom {
  const p = g.positions, n = g.normals, ix = g.indices
  for (let t = 0; t < ix.length; t += 3) {
    const a = ix[t], b = ix[t + 1], c = ix[t + 2]
    const e1: Vec3 = [p[3 * b] - p[3 * a], p[3 * b + 1] - p[3 * a + 1], p[3 * b + 2] - p[3 * a + 2]]
    const e2: Vec3 = [p[3 * c] - p[3 * a], p[3 * c + 1] - p[3 * a + 1], p[3 * c + 2] - p[3 * a + 2]]
    const fn: Vec3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
    const vn: Vec3 = [n[3 * a] + n[3 * b] + n[3 * c], n[3 * a + 1] + n[3 * b + 1] + n[3 * c + 1], n[3 * a + 2] + n[3 * b + 2] + n[3 * c + 2]]
    if (fn[0] * vn[0] + fn[1] * vn[1] + fn[2] * vn[2] < 0) { ix[t + 1] = c; ix[t + 2] = b }
  }
  return g
}
```

`<f>:103,149,177,197,222@50b45d5` — the five generator signatures that carry every
count (defaults repeated in §3.2):

```ts
function ellipsoid(rx: number, ry: number, rz: number, nu = 40, nv = 20): LeafGeom {
function roundBox(sx: number, sy: number, sz: number, radius: number, cells = 10): LeafGeom {
function capsule(height: number, radius: number, nu = 32, nCap = 10): LeafGeom {
function frustum(height: number, rb: number, rt: number, nu = 40): LeafGeom {
function torus(major: number, minor: number, nu = 48, nv = 20): LeafGeom {
```

`<b>:52-57@50b45d5` — every length is in metres (`unit: 'm'`, `MIN_LENGTH = 0.001`,
`MAX_LENGTH = 1000`), vec3 params are three metres:

```ts
export function lengthSpec(name: string, def: number, doc: string, min = MIN_LENGTH): ParamSpec {
  return { name, type: 'number', default: def, min, max: MAX_LENGTH, step: 0.01, unit: 'm', doc }
}
function vec3Spec(name: string, def: Vec3, doc: string): ParamSpec {
  return { name, type: 'vec3', default: def, min: MIN_LENGTH, max: MAX_LENGTH, step: 0.01, unit: 'm', doc }
}
```

## 3. Per-kind reference facts (the 8 S3 kinds)

### 3.1 Axis convention, units, winding, seams (all kinds)

- SDF Modeler is right-handed, Y up, metres, like RAA: the polar axis of the
  ellipsoid/sphere is Y (`<f>:109`), capsule/cylinder/cone run along Y
  (`<b>:244,297,341`), the torus ring lies in XZ (`<b>:400`), the ground plane is
  `y <= 0` (`<b>:464`). Every length is metres (`<b>:52-57`). Measured on both
  reference sets: the signed volume is positive for all 8 kinds in the raw
  reference frame, so `axisMap` is the identity and `det = 1` (no index reversal).
- Winding: quads are pushed CCW seen from outside (`<f>:48-49`), `fixWinding`
  (`<f>:136-146`) orients every triangle outward and `done()` (`<f>:50-59`) drops
  zero-area triangles. Verified for all kinds on sets A and B: closed on the
  position-welded topology (every undirected edge used by exactly two triangles in
  opposite directions) and signed volume > 0.
- Seams: the generators never weld. Azimuth seams (`i = 0` vs `i = nu`), poles and
  apexes (one coincident copy per azimuth), flat box faces and cap rims keep
  separate vertices with their own normals (hard edges). Coincident positions are
  exact f32 duplicates, so a consumer welding at WELD_EPS sees the closed
  topology; the port may share or duplicate them as long as positions, triangle
  list and normals match the reference.
- Quality/segments: `leafGeometry` has no quality argument and none of the 8 kinds
  reads a count from `params`; every count is a default argument of the generator
  (`<f>:103,149,177,197,222`). RAA derives each count with
  `count(s) = max(min, multiple * round(ref * s / (24 * multiple)))` (round half
  away from zero) from the `segments` list in §4, where `s` is the clamped
  `QUALITY.segments` value (LOW = 8, LOD0 = 24, HIGH = 64, clamp [3, 256]). At
  LOD0 every count equals the reference default exactly.

### 3.2 Every tessellation count

`source` is `file:line@50b45d5`; `8/24/64` are the derived counts at LOW/LOD0/HIGH.
`path` is how `gen-leaf-fixtures.mjs` produces the reference at other qualities
(§1.2). `min` is the smallest value the reference maths yields a well-formed mesh
for (reason listed below); `multiple` is the divisibility constraint of the count
(none of these counts has one).

| kind | count | source | default | min | multiple/parity | path | 8 / 24 / 64 |
|---|---|---|---|---|---|---|---|
| sphere | `nu` | `leaf-mesh.ts:103` | 40 | 3 | 1 / none | call | 13 / 40 / 107 |
| sphere | `nv` | `leaf-mesh.ts:103` | 20 | 2 | 1 / none | call | 7 / 20 / 53 |
| box | — | — | — | — | — | — | no count (segments `[]`) |
| roundBox | `cells` | `leaf-mesh.ts:149` | 10 | 1 | 1 / none | call | 3 / 10 / 27 |
| capsule | `nu` | `leaf-mesh.ts:177` | 32 | 3 | 1 / none | call | 11 / 32 / 85 |
| capsule | `nCap` | `leaf-mesh.ts:177` | 10 | 1 | 1 / none | call | 3 / 10 / 27 |
| cylinder | `nu` | `leaf-mesh.ts:197` | 40 | 3 | 1 / none | call | 13 / 40 / 107 |
| cone | `nu` | `leaf-mesh.ts:197` | 40 | 3 | 1 / none | call | 13 / 40 / 107 |
| torus | `nu` | `leaf-mesh.ts:222` | 48 | 3 | 1 / none | call | 16 / 48 / 128 |
| torus | `nv` | `leaf-mesh.ts:222` | 20 | 3 | 1 / none | call | 7 / 20 / 53 |
| ellipsoid | `nu` | `leaf-mesh.ts:103` | 40 | 3 | 1 / none | call | 13 / 40 / 107 |
| ellipsoid | `nv` | `leaf-mesh.ts:103` | 20 | 2 | 1 / none | call | 7 / 20 / 53 |

All counts are parameters (with defaults) of the private generators, so all use
path `call`; box has no count at all, its mesh is byte-identical at every quality
(`segments: []`), which is correct for a flat-faced box (R3's smoothness criterion
does not apply to it). Measured minima (all verified on both reference sets, and
the values 8/24/64 produce closed, positive-volume meshes for every kind):

- `nu` (azimuth/ring) min 3: with 2 the ring is flat — ellipsoid/capsule/torus are
  not closed and the frustum's volume is ~4e-17 with zero Z extent. 3 is the
  smallest genuine ring.
- ellipsoid/sphere `nv` (latitude bands) min 2: `nv = 1` drops every triangle (the
  two rows are the poles); `nv = 2` is the smallest valid closed surface.
- torus `nv` (tube ring) min 3: `nv = 2` puts every cross-section ring at y = 0
  (not closed, volume ~0); `nv = 3` is the smallest valid tube.
- capsule `nCap` min 1: `nCap = 0` computes `0 / 0` in `(Math.PI / 2) * (j / nCap)`
  and yields NaN positions; `nCap = 1` is valid.
- roundBox `cells` min 1: `cells = 0` divides by zero (`-1 + (2 * i) / cells`) and
  yields NaN; `cells = 1` is valid.

Odd values are exercised by the derived counts themselves (13, 7, 3, 11, 27) and
also produce closed, positive-volume meshes; no count requires an even value or any
other multiple.

### 3.3 Per kind

Counts, raw vertex count, welded vertex count (WELD_EPS =
1e-6 × max(1, bbox diagonal)) and triangle count are the measured default values at
the reference counts on sets A and B; all 8 kinds pass closure, outward winding and
positive volume in both sets.

- **sphere** — S3 `radius` (m) → SDF `radius`, scale 1. Calls `ellipsoid(r, r, r)`
  (`<f>:696`). Counts `nu`, `nv`. Raw `(nu+1)(nv+1)` = 861, welded
  `2 + nu(nv−1)` = 762, tris `2·nu·(nv−1)` = 1520. Exact normals `unit(d/r)`
  (`<f>:110`); azimuth seam and both pole rows are duplicated (pole fans removed by
  `done()`).
- **box** — S3 `halfExtents` (m) → SDF `size` (full extents), scale 2 per
  component. No counts. 24 raw / 8 welded / 12 tris: six faces, each with its own
  4 vertices and a CCW quad; the 8 corners weld, hard edges stay split (no seam
  welding inside the generator).
- **roundBox** — `halfExtents` × 2 → `size`, `radius` scale 1 (S3 guarantees
  `radius <= min(halfExtents)`, so the reference's own clamp `<f>:151` is inactive).
  Count `cells` (per face, each face a `(cells+1)²` grid). Raw `6(cells+1)²` = 726,
  welded 602, tris `12·cells²` = 1200. The rounded borders coincide between faces
  and weld; the flat middle of each face keeps distinct vertices.
- **capsule** — `radius` scale 1; SDF `height` is the total height including the
  caps (`<b>:240`), so it is `2 × halfHeight + 2 × radius` through the Q5 additive
  term (§4; set A 1.8, set B 2.9). The reference's `half = max(0, height / 2 − r)`
  (`<f>:179`) is then S3 `halfHeight`, and the Y extent is
  `PrimParams::bounds`' `halfHeight + radius`. Counts `nu`, `nCap`. `22` rows at the defaults (2 poles + 2×`nCap` circles),
  raw `(nu+1)(2·nCap+2)` = 726, welded 642, tris `2·nu·(2·nCap+1) − 2·nu` = 1280.
  Pole rows hold `nu+1` coincident copies whose `2·nu` zero-area triangles are
  dropped; the azimuth seam is duplicated.
- **cylinder** — `halfHeight` (S3 half-height) × 2 → SDF `height` (total), `radius`
  scale 1. Calls `frustum(height, r, r)` (`<f>:701`). Count `nu`. Raw `4·nu+6` =
  166, welded 82, tris `4·nu` = 160 (80 side + 80 cap). Side rim and cap ring
  coincide positionally but are separate vertices (cap normals `[0, ±1, 0]`, hard
  edge); the azimuth seam is duplicated.
- **cone** — `halfHeight` × 2 → `height`; `radius` → `radiusBottom`; `radiusTop`
  has no S3 source and is the constant 0 (S3 cone is pointed, apex at
  `+halfHeight`). Count `nu`. Raw `3·nu+4` = 124, welded 42, tris `2·nu` = 80.
  The apex ring (`r <= 0`) is skipped for the top cap (`<f>:210`) and its `nu`
  zero-area triangles are dropped; the bottom cap and both rings are duplicated.
- **torus** — `majorRadius`, `minorRadius` scale 1 (m); ring in XZ, axis Y.
  Counts `nu`, `nv`. Raw `(nu+1)(nv+1)` = 1029, welded `nu·nv` = 960, tris
  `2·nu·nv` = 1920. Both seam rows (`i = 0/nu`, `j = 0/nv`) are duplicated, all
  triangles non-degenerate.
- **ellipsoid** — `radii` (m, semi-axes) → SDF `radii` component-wise, scale 1.
  Calls `ellipsoid(rx, ry, rz)` (`<f>:697`), normals `unit(d[0]/rx, d[1]/ry,
  d[2]/rz)`. Counts `nu`, `nv`; same layout and counts as sphere (861 / 762 /
  1520). Seam and pole duplication as sphere.

## 4. mapping v1

Binding object (Q3 and Q5 amendments). Conventions used by every consumer:

- `sdf_value = scale × S3 value`. A vec3 SDF parameter is written as three keys
  `<sdfParam>.x|.y|.z` (suffix in the SDF frame); each has `component` = the RAA
  component after `axisMap`, and each RAA component is used exactly once. A scalar
  SDF parameter keeps its bare name with `component: null`. A vector S3 `from` with
  `component: null` is invalid. The main `scale` is finite and > 0.
- Additive term (Q5): a `params` entry may carry **one** optional
  `"add": {"from", "component", "scale"}`. Then
  `sdf_value = scale × S3[from(.component)] + add.scale × S3[add.from(.component)]`,
  evaluated in f64 in exactly that order (`scale*a + add.scale*b`). `add.from` must
  be an S3 key of the same kind, and `add.component` follows the same rules as the
  main term on the same entry: `null` for a scalar key (and then the SDF parameter is
  a scalar), and for a vector key the RAA-frame component that `axisMap` gives for
  the entry's SDF axis suffix (`null` on a vector key is invalid). `add.scale` is
  finite and non-zero. `add` has exactly these three keys, cannot contain another
  `add`, and an entry has at most one. Coverage: the main `from` (per component for
  vectors) counts toward the each-S3-key-once rule below; the key named in `add` is
  an extra use and does not. Only `capsule.height` uses `add`.
- `constants` carries the SDF shape parameters that have no S3 source (here only
  cone `radiusTop`). Advanced/optional params (`edgeStyle`, `taper`, `shift`,
  `edges`, `corners`, `topBevel*`, `bottomBevel*`, ...) must stay absent: their
  mere presence makes `leafGeometry` return `null` (`<f>:692`), so the registry
  default is the "absent" case, not a constant.
- `segments` is the list of count rules `{count, ref, min, multiple, source}`;
  `box` has `[]`. RAA and the fixture compute every count with
  `count(s) = max(min, multiple * round(ref * s / (24 * multiple)))`, round half
  away from zero, `s` = clamped segments (LOW 8 / LOD0 24 / HIGH 64, clamp [3,
  256]); at LOD0 every count equals `ref`.
- Validation (binding for `gen-leaf-fixtures.mjs` and `raa-prims::reference`):
  version 1, `reference.commit` 40 lowercase hex, `axisMap` a signed permutation
  whose `det` matches, all 8 kinds present, every non-advanced shape parameter of a
  kind covered exactly once by `params`+`constants`, each S3 key consumed by
  exactly one SDF parameter's main term (per component for vectors; `add` terms
  are extra uses), every `add` well formed as above. Any violation is refused
  (`prims.mapping_invalid` / `sdf-ref: <message>`), never defaulted.

```json
{
  "version": 1,
  "reference": {
    "repo": "sdfm",
    "path": "src/core/hybrid/leaf-mesh.ts",
    "commit": "50b45d5862be55ba85492d6bb0e457ed3b244dd6"
  },
  "axisMap": { "x": "+x", "y": "+y", "z": "+z", "det": 1 },
  "paramMap": {
    "sphere": {
      "sdfKind": "sphere",
      "params": { "radius": { "from": "radius", "component": null, "scale": 1 } },
      "constants": {},
      "segments": [
        { "count": "nu", "ref": 40, "min": 3, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:103" },
        { "count": "nv", "ref": 20, "min": 2, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:103" }
      ]
    },
    "box": {
      "sdfKind": "box",
      "params": {
        "size.x": { "from": "halfExtents", "component": "x", "scale": 2 },
        "size.y": { "from": "halfExtents", "component": "y", "scale": 2 },
        "size.z": { "from": "halfExtents", "component": "z", "scale": 2 }
      },
      "constants": {},
      "segments": []
    },
    "roundBox": {
      "sdfKind": "roundBox",
      "params": {
        "size.x": { "from": "halfExtents", "component": "x", "scale": 2 },
        "size.y": { "from": "halfExtents", "component": "y", "scale": 2 },
        "size.z": { "from": "halfExtents", "component": "z", "scale": 2 },
        "radius": { "from": "radius", "component": null, "scale": 1 }
      },
      "constants": {},
      "segments": [
        { "count": "cells", "ref": 10, "min": 1, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:149" }
      ]
    },
    "capsule": {
      "sdfKind": "capsule",
      "params": {
        "height": { "from": "halfHeight", "component": null, "scale": 2, "add": { "from": "radius", "component": null, "scale": 2 } },
        "radius": { "from": "radius", "component": null, "scale": 1 }
      },
      "constants": {},
      "segments": [
        { "count": "nu", "ref": 32, "min": 3, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:177" },
        { "count": "nCap", "ref": 10, "min": 1, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:177" }
      ]
    },
    "cylinder": {
      "sdfKind": "cylinder",
      "params": {
        "height": { "from": "halfHeight", "component": null, "scale": 2 },
        "radius": { "from": "radius", "component": null, "scale": 1 }
      },
      "constants": {},
      "segments": [
        { "count": "nu", "ref": 40, "min": 3, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:197" }
      ]
    },
    "cone": {
      "sdfKind": "cone",
      "params": {
        "height": { "from": "halfHeight", "component": null, "scale": 2 },
        "radiusBottom": { "from": "radius", "component": null, "scale": 1 }
      },
      "constants": { "radiusTop": 0 },
      "segments": [
        { "count": "nu", "ref": 40, "min": 3, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:197" }
      ]
    },
    "torus": {
      "sdfKind": "torus",
      "params": {
        "majorRadius": { "from": "majorRadius", "component": null, "scale": 1 },
        "minorRadius": { "from": "minorRadius", "component": null, "scale": 1 }
      },
      "constants": {},
      "segments": [
        { "count": "nu", "ref": 48, "min": 3, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:222" },
        { "count": "nv", "ref": 20, "min": 3, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:222" }
      ]
    },
    "ellipsoid": {
      "sdfKind": "ellipsoid",
      "params": {
        "radii.x": { "from": "radii", "component": "x", "scale": 1 },
        "radii.y": { "from": "radii", "component": "y", "scale": 1 },
        "radii.z": { "from": "radii", "component": "z", "scale": 1 }
      },
      "constants": {},
      "segments": [
        { "count": "nu", "ref": 40, "min": 3, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:103" },
        { "count": "nv", "ref": 20, "min": 2, "multiple": 1, "source": "src/core/hybrid/leaf-mesh.ts:103" }
      ]
    }
  }
}
```

Each S3 key of a kind is consumed exactly once by a main term (`halfExtents` and
`radii` once per component of their single vector SDF param; capsule's `radius` by
`radius`, its extra use in `height.add` not counted), all scales are final, `box`'s `segments` is `[]`, `segments` sources
are `file:line` relative to the `sdfm` root, and the `sdfInput` of a fixture case is
`params` + `constants` passed to `leafGeometry` unchanged. The fixture records the
path per kind (all `call`) plus the sdfm commit and the sha256 of the original
`leaf-mesh.ts`.

## 5. Unmappable

Nothing is unmappable: every kind, parameter and count of the 8 kinds fits mapping
v1 as amended (Q3, Q5). The one finding recorded against
`sdfm@50b45d5862be55ba85492d6bb0e457ed3b244dd6` is kept below for the record.

1. **Resolved by amendment Q5 (2026-10-18): `capsule.height` is an affine
   combination, not `scale × S3 value`.** The lead adopted the proposed additive term
   (§4), with the binding entry
   `"height": {"from":"halfHeight","component":null,"scale":2,"add":{"from":"radius","component":null,"scale":2}}`;
   the capsule stays in the comparison (48 cases). The original finding:
   - S3 (`ARCHITECTURE.md`) gives a capsule `radius` and `halfHeight`, with the core
     segment from `-halfHeight` to `+halfHeight`; accordingly
     `raa_prims::PrimParams::bounds` returns `[radius, halfHeight + radius, radius]`
     (`crates/raa-prims/src/params.rs`). The RAA mesh's Y extent is therefore
     `2 × (halfHeight + radius)`.
   - SDF Modeler's `capsule.height` is the **total** height including the caps
     (`<b>:240`: "Total height along Y, caps included"; `<f>:179`:
     `half = Math.max(0, height / 2 - r)`). Measured:
     `leafGeometry('capsule', { height: 1, radius: 0.4 })` has Y bbox
     `[-0.5, 0.5]`.
   - Matching the reference to an S3 capsule therefore needs
     `height = 2 × halfHeight + 2 × radius` (set A: 1.8, set B: 2.9). Before Q5,
     mapping v1 defined `sdf_value = scale × S3 value` with a single `scale` per
     entry and no offset or second term, so no `{from, component, scale}` entry
     could carry it.
   - Consequence before Q5: `gen-leaf-fixtures.mjs` refused the mapping (no
     `height` entry, so the coverage rule failed), nothing fell back to the registry
     default (`num()` would silently use `height = 1`, `<f>:78-83`) and no case was
     written.
   - Proposed minimal amendment, adopted as Q5: one optional additive term per
     param entry. Re-reading the S3 capsule as "half of the total height" was
     **not** proposed, because it contradicts S3's
     `(segment from -halfHeight to +halfHeight)` and the analytic bounds already
     implemented in `raa-prims`.
   - Verified after Q5 (§6): at LOW, LOD0 and HIGH the capsule's Y bbox is
     `±(halfHeight + radius)` to f32 precision on both sets (A ±0.9, B ±1.45),
     the `PrimParams::bounds` Y extent.

## 6. Regeneration command

Exact command, run from the `raa` repository root (this string is also the
fixture's `command` and `regenerate`):

```
node tools/sdf-ref/gen-leaf-fixtures.mjs --ref <sdfm checkout root> --out crates/raa-prims/tests/fixtures/sdf-leaf-reference.json
```

`--ref` is the root of the read-only `sdfm` checkout (card 3296:
`<sdfm checkout>`); its HEAD must be
`mapping.reference.commit` and `src/` must have no local change, otherwise the
tool refuses. Determinism check: run it twice (the second time with
`--out <scratch>/sdf-leaf-reference.json`) and `cmp` the two files; they must be
byte-identical.

What the tool does, in order (it refuses with `sdf-ref: <message>` and exit code 1
at the first failing step, and never writes anything inside the checkout):

1. Reads the §4 fenced json block of this README; that parsed object is the
   fixture's `mapping` (same keys, same order).
2. Loads `leaf-mesh.ts` and `basic.ts` (§1.1), and validates the mapping against
   `LEAF_KINDS` and the `basicPrimitives` parameter specs with the §4 rules:
   version 1, 40-hex commit, signed-permutation `axisMap` with matching `det`, the
   8 S3 kinds, every `sdfKind` in `LEAF_KINDS`, per-component vec3 keys whose
   `component` is the RAA axis that `axisMap` gives (each once), `component: null`
   exactly for scalars, `scale` finite and > 0, at most one `add` per entry with
   exactly `from`/`component`/`scale`, no nested `add`, the same component rules and
   `add.scale` finite and non-zero, every S3 key (per component) used once by a main
   term (`add` uses are extra), every non-advanced SDF parameter covered once by
   `params` or `constants`,
   no advanced parameter, and `segments` rules whose count at LOD0 equals `ref`.
   Every violation is printed, then it exits.
3. Checks the checkout's HEAD and cleanliness, and decides each kind's path (§1.2).
4. For each kind (S3 order) × set A/B × QUALITY 8/24/64: `s` = clamped segments,
   `counts` from the §4 rule, `sdfInput` = `params` + `constants`, the reference
   mesh from the kind's path; at LOD0 it asserts byte equality with unmodified
   `leafGeometry(sdfKind, sdfInput)`. The mesh goes through `axisMap` (triangles
   reversed when `det = -1`), then the metrics are computed in f64 in RAA's frame,
   mirroring `raa_geom::analysis::analyze`: `uniquePositions` welded at WELD_EPS
   (which must equal the 1e-6 rounding count of S8), `triangles`, `bboxMin`,
   `bboxMax`, `area`, signed `volume` (must be > 0) and welded-edge `closed`.
   Counted kinds must give three distinct triangle counts per set; there must be
   48 distinct cases.
5. Writes the whole fixture at once (temporary file in the target directory, then
   rename): `JSON.stringify(fixture, null, 2)` plus a final LF.

Fixture keys, in order: `generatedBy`, `command`, `regenerate` (both the exact
command above), `sdfModelerRevision` (the sdfm commit), `leafMeshSha256` (sha256
of the original `leaf-mesh.ts` bytes), `axisMap` and `paramMap` (the S8 forms,
derived from `mapping`: `[{axis, sign}] x3` and
`{kind: [{ref, refIndex, raa, index, scale, value}]}`; an entry whose mapping term has
`add` also carries `add: {raa, index, scale}`, the same additive term in S8 names, so
`refParams[ref][refIndex] = scale × params[raa][index] + add.scale × params[add.raa][add.index]`;
only capsule `height` has it), `mapping` (§4 verbatim),
`kinds` (`{kind: {sdfKind, path, commit, sha256}}`, the per-kind path record) and
`cases`. Each case: `kind`, `set`, `quality`, `params` (S3 names), `refParams`
(S8 form of the leafGeometry params), `sdfInput` (`{kind, params}`, the exact
`leafGeometry` arguments), `counts` (`{name: value}`, `{}` for box),
`uniquePositions`, `triangles`, `bboxMin`, `bboxMax`, `area`, `volume`, `closed`.

Current result (2026-10-18, card 3296, `--ref <sdfm checkout>`):
exit 0 in about 1 s, 48 cases written, fixture sha256
`bb4296bcb535147f10111e357c21764e2742f6794c55941d475bae1708d84b6a`; a second run
gave byte-identical bytes. Every kind used path `call`, every LOD0 mesh equalled
unmodified `leafGeometry` byte for byte, all 48 reference meshes are closed with
positive volume, and the LOD0 counts and sizes match §3.3. Triangles per set
(identical for A and B) at LOW / LOD0 / HIGH:

| kind | counts at 8 / 24 / 64 | triangles 8 / 24 / 64 | uniquePositions 8 / 24 / 64 |
|---|---|---|---|
| sphere | nu,nv 13,7 / 40,20 / 107,53 | 156 / 1520 / 11128 | 80 / 762 / 5566 |
| box | — | 12 / 12 / 12 | 8 / 8 / 8 |
| roundBox | cells 3 / 10 / 27 | 108 / 1200 / 8748 | 56 / 602 / 4376 |
| capsule | nu,nCap 11,3 / 32,10 / 85,27 | 132 / 1280 / 9180 | 68 / 642 / 4592 |
| cylinder | nu 13 / 40 / 107 | 52 / 160 / 428 | 28 / 82 / 216 |
| cone | nu 13 / 40 / 107 | 26 / 80 / 214 | 15 / 42 / 109 |
| torus | nu,nv 16,7 / 48,20 / 128,53 | 224 / 1920 / 13568 | 112 / 960 / 6784 |
| ellipsoid | nu,nv 13,7 / 40,20 / 107,53 | 156 / 1520 / 11128 | 80 / 762 / 5566 |

Also checked from scratch copies outside both repositories (never committed): a
forced `literal` path for the 7 counted kinds gave identical cases; a `det = -1`
mapping (`z: "-z"`) kept every case closed with the same positive volume and a
mirrored Z bbox; and each `add` violation (zero or non-finite `add.scale`, nested
`add`, an unknown key, a missing key, an unknown `add.from`, a component on a scalar
key, `null` on a vector key, `add` not an object) was refused with its own
`sdf-ref: invalid mapping v1 ...` line, exit 1 and no file written.
