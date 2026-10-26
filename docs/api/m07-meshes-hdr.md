# RAA API m07: meshes and HDR base

Additions of chain milestone 1 to `@etcito/raa`: attributed meshes from typed arrays, the configurable pass pipeline, HDR rendering with AgX / ACES tone mapping and MSAA. Types: `packages/raa/types/m07-meshes-hdr.d.ts` (appended to `index.d.ts` when the package is staged, so `EngineOptions` and `Engine` merge with their v0 declarations). Implementation: `crates/raa-js/src/api/m07_meshes.rs`, `crates/raa-js/src/api/m07_pipeline.rs` and the engine options in `crates/raa-js/src/convert.rs`.

Everything in `docs/API.md` (v0) stays valid. Conventions are those of v0: right-handed, Y up, the camera looks down -Z, metres, front faces counter-clockwise. Every failure throws a `RaaError` whose `code` is listed per method; after `dispose` every method throws `api.disposed`.

## Rendering model

The engine renders scene-referred linear HDR into `rgba16float` targets (with MSAA in the forward pass) and runs one display transform at the end, the `tonemap` pass: exposure, the tone mapping curve, the sRGB transfer function when the canvas format is not an `*-srgb` format, and dither. Colours given to the engine are linear RGB unless documented as sRGB.

## Engine options

### `LitEngineOptions`

`{ msaa?: 1 | 4; toneMapper?: 'agx' | 'aces' | 'linear'; exposureEv?: number }`

Options of the lit pipeline, read by `createEngine` together with the v0 options.

| field | default | meaning |
|---|---|---|
| `msaa` | `4` | MSAA sample count of the forward pass; `1` disables multisampling. |
| `toneMapper` | `'agx'` | Initial tone mapping curve (see `setToneMapping`). |
| `exposureEv` | `0` | Initial exposure in EV, finite: the HDR colour is multiplied by 2^exposureEv. |

Any other value (an `msaa` other than 1 or 4, an unknown mapper name, a non-finite or non-number `exposureEv`) makes `createEngine` reject with `api.invalid_argument`.

### `EngineOptions`

`export interface EngineOptions extends LitEngineOptions {}`

The v0 `EngineOptions` (`maxInstances`, `maxMaterials`, `powerPreference`) merged with `LitEngineOptions`, so `raa.createEngine(canvas, { msaa: 1, toneMapper: 'aces' })` type-checks. Omitted fields take their defaults.

## Meshes (S13)

### `MeshDesc`

```ts
interface MeshDesc {
  positions: Float32Array;              // 3n, metres
  normals: Float32Array;                // 3n
  uvs?: Float32Array;                   // 2n
  tangents?: Float32Array;              // 4n, w = +1 | -1
  colors?: Float32Array | Uint8Array;   // 4n: linear floats | sRGB rgba8
  indices: Uint32Array;                 // 3m, triangle list
  subMeshes?: SubMeshDesc[] | null;
}
```

An indexed triangle list in the mesh's own space (`n` vertices, `m` triangles). The mesh need not be closed. Rules, checked in this order, the first broken one is thrown:

- `positions`, `normals` and `indices` are required. A missing one, or an array of the wrong type (for example a plain `Array` or a `Float64Array`), throws `api.invalid_argument` naming the field.
- Positions and indices both non-empty, else `mesh.empty`.
- Each array has exactly its per-vertex length (`positions` 3n, `normals` 3n, `uvs` 2n, `tangents` 4n, `colors` 4n), else `mesh.length_mismatch` naming the array; `indices.length` is a multiple of 3.
- Every float is finite, else `mesh.not_finite` naming the array and the index.
- Every index is < n, else `mesh.index_range` with the first bad index.
- Normals have a non-zero length (`mesh.invalid_normal`); they are normalised by the engine.
- Given tangents have a non-zero xyz and w equal to +1 or -1 (`mesh.invalid_tangent`); xyz is normalised.
- Sub-meshes as in `SubMeshDesc` (`mesh.submesh_range`, `mesh.submesh_overlap`, `api.unknown_handle`).

Attributes and defaults:

- `uvs`: default (0, 0) per vertex.
- `tangents`: when absent and `uvs` are given, tangents are **generated** from positions, normals and uvs (per-triangle dP/du accumulated per corner weighted by the corner angle, orthogonalised against the normal, w = the sign of the bitangent's handedness; vertices without a usable uv parametrisation get the default below). When absent without `uvs`, every vertex gets the default tangent: an orthonormal vector perpendicular to the normal (branchless basis of Duff et al. 2017), w = +1.
- `colors`: per-vertex colour that multiplies the material's base colour (alpha multiplies opacity). **A `Float32Array` is linear** rgba as given. **A `Uint8Array` is sRGB rgba8**: rgb goes through the exact sRGB-to-linear transfer function, alpha is `a / 255` (linear). Default (1, 1, 1, 1).

### `SubMeshDesc`

`{ first: number; count: number; material?: number }`

One contiguous index range drawn with one material. `first` is an index offset (a multiple of 3, >= 0), `count` a number of indices (a multiple of 3, > 0), and `first + count <= indices.length`; a non-integer, negative or out-of-range value throws `mesh.submesh_range`. Two sub-meshes must not share indices (`mesh.submesh_overlap`); they need not cover every index (uncovered triangles are not drawn). `material` is a material handle from `addMaterial` or `loadPrimsAsset`; an unknown handle throws `api.unknown_handle`. Absent `subMeshes` (or null) means one sub-mesh over all indices with no material.

### `MeshInfo`

`{ vertices: number; triangles: number; subMeshes: { first: number; count: number; material: number | null }[]; bounds: { min: Vec3; max: Vec3 } }`

What `meshInfo` returns: the vertex count, the triangle count, the sub-meshes as stored (one entry when the desc had none; `material` is the handle, or `null` for none) and the axis-aligned bounds of the positions in mesh space (metres).

### `addMesh`

`engine.addMesh(desc: MeshDesc): number`

Validates `desc` (rules of `MeshDesc`), uploads it and returns a new mesh handle. The handle is accepted everywhere a v0 mesh handle is: `addInstance` (every sub-mesh drawn with the given material), `addMeshInstance`, `updateMesh`, `meshInfo` and `removeMesh`. Primitive and prims-asset meshes get the attribute defaults (uv (0, 0), default tangent, white colour). Codes: `mesh.empty`, `mesh.length_mismatch`, `mesh.not_finite`, `mesh.index_range`, `mesh.invalid_normal`, `mesh.invalid_tangent`, `mesh.submesh_range`, `mesh.submesh_overlap`, `api.unknown_handle`, `api.invalid_argument`.

### `updateMesh`

`engine.updateMesh(id: number, desc: MeshDesc): void`

Replaces the geometry, attributes and sub-meshes of mesh `id` (any mesh handle, including primitive and asset meshes). Every instance of that mesh is rebuilt at the next frame from the new sub-meshes, keeping its node, visibility and instance handle. A refused `desc` leaves the mesh unchanged. Codes: `api.unknown_handle` (mesh) and those of `addMesh`.

### `meshInfo`

`engine.meshInfo(id: number): MeshInfo`

The counts, sub-meshes and bounds of mesh `id`. Code: `api.unknown_handle`.

### `addMeshInstance`

`engine.addMeshInstance(node: number, mesh: number): number`

Draws `mesh` at `node`, each sub-mesh with its own `material` (the engine's default material, a grey standard material, when the sub-mesh has none), and returns an instance handle. The handle behaves as a v0 instance: `removeInstance` and `setVisible` act on all of its sub-meshes. Codes: `api.unknown_handle`, `scene.stale_node`.

## Pipeline (S12)

The engine runs one ordered list of passes each frame. Passes run by ascending order (unless `setPassOrder` set another one); contributors draw inside the forward pass after the opaque draws, at their own position. Milestone 1 installs these rows of the S12 table; later features add their own rows (sky, shadows, AO, bloom...):

| order | name | role | required | what it does |
|---|---|---|---|---|
| 30 | `prepass` | pass | yes | Depth (Depth32Float, single-sample), view-space normal + roughness and material (F0, metallic) targets. |
| 60 | `forward` | pass | yes | Lit shading into the HDR target (`rgba16float`) with MSAA (`msaa` samples), resolved into the HDR chain. |
| 200 | `tonemap` | pass | yes | Display transform: exposure, mapper, look, sRGB encoding when the canvas is not sRGB, dither; writes the canvas. Last HDR pass; display overlays follow it. |

Required passes cannot be disabled; `prepass` runs before `forward`, which runs before `tonemap`, and `tonemap` is last among HDR passes. Display overlays such as `editor` follow it.

**Background rule.** With no sky, the background colour is display-referred linear RGB and reaches the canvas unchanged: `tonemap` writes it as is (no exposure, no curve; only the sRGB encoding of a non-sRGB canvas) wherever the depth is 1 (nothing drawn). An empty scene therefore shows exactly the background, which defaults to CLEAR_COLOR (0.05, 0.06, 0.08), and `probeNonEmpty` keeps its v0 meaning.

### `PassInfo`

`{ name: string; role: 'pass' | 'contributor'; enabled: boolean; order: number; required: boolean }`

One entry of `passes`. `role` is `'pass'` for a pipeline pass and `'contributor'` for a draw inside `forward`. `order` is the pass's sort key (its position inside `forward` for a contributor). `required` is true for `prepass`, `forward` and `tonemap`.

### `passes`

`engine.passes(): PassInfo[]`

Every pass and contributor in execution order (contributors listed where they run). Code: `pipeline.busy` (called from inside a frame).

### `setPassEnabled`

`engine.setPassEnabled(name: string, enabled: boolean): void`

Enables or disables a pass or contributor (all are enabled by default). A disabled feature resets its lighting resources to neutral values, so the frame stays valid without it. Codes: `pipeline.unknown_pass`, `pipeline.required_pass` (disabling `prepass`, `forward` or `tonemap`; enabling them is a no-op).

### `setPassOrder`

`engine.setPassOrder(names: string[]): void`

Replaces the execution order of the passes. `names` lists every pass-role name exactly once (contributors are not listed; they keep their position inside `forward`). Codes, in the order they are checked: `pipeline.unknown_pass` (unknown name or a contributor), `pipeline.duplicate_pass` (a name listed twice), `pipeline.order_incomplete` (a pass missing), `pipeline.required_order` (`prepass`, `forward`, `tonemap` out of order or an HDR pass after `tonemap`, or a display overlay before an HDR pass). A refused order leaves the current one unchanged.

## Tone mapping

### `ToneMappingDesc`

`{ mapper?: 'agx' | 'aces' | 'linear'; exposureEv?: number; look?: 'none' | 'punchy' | 'golden'; dither?: boolean }`

| field | default | meaning |
|---|---|---|
| `mapper` | `'agx'` | `'agx'`: AgX, the default; a smooth highlight roll-off without hue skews. `'aces'`: the ACES filmic curve, higher contrast and saturation. `'linear'`: no curve, values above 1 clip (for debugging and comparisons). |
| `exposureEv` | `0` | Exposure in EV (stops); the HDR colour is multiplied by 2^exposureEv before the curve. Finite. |
| `look` | `'none'` | AgX look: `'punchy'` (more contrast and saturation) or `'golden'` (warm tint). |
| `dither` | `true` | Adds a small noise before quantisation to hide banding. |

The defaults (`mapper` agx, `look` none, `exposureEv` 0, `dither` on) are TONEMAP_DEFAULT; `createEngine` options `toneMapper` and `exposureEv` set the initial values.

### `setToneMapping`

`engine.setToneMapping(settings: ToneMappingDesc): void`

Merges the given fields over the current tone mapping; omitted fields keep their current value (`setToneMapping({ mapper: 'aces' })` keeps the exposure). Codes: `api.invalid_argument` (unknown `mapper` or `look`, non-finite `exposureEv`, a field of the wrong type), `post.invalid` (settings refused by the tonemap pass). A refusal changes nothing.

## Background

### `setBackground`

`engine.setBackground(color: Vec3 | Float32Array): void`

Sets the background colour: 3 components, display-referred linear RGB, each finite and >= 0 (values above 1 clip at the display). It is shown unchanged where nothing is drawn (background rule above). Default `[0.05, 0.06, 0.08]` (CLEAR_COLOR). Code: `api.invalid_argument` (not 3 components, non-finite or negative).

## Capabilities

### `Capabilities`

`{ coreWebGpu: true; features: string[]; msaa: number; hdrFormat: 'rgba16float'; maxStorageBufferBindingSize: number }`

| field | meaning |
|---|---|
| `coreWebGpu` | Always `true`: the engine requires a core WebGPU adapter and has no degraded path. |
| `features` | The enabled device features as WebGPU names. The optional `float32-filterable` and `rg11b10ufloat-renderable` are requested when the adapter has them and never required. |
| `msaa` | The MSAA sample count in use (1 or 4, from `EngineOptions.msaa`). |
| `hdrFormat` | Format of the HDR targets, `'rgba16float'`. |
| `maxStorageBufferBindingSize` | The device limit in bytes (the adapter's limits are requested). |

### `capabilities`

`engine.capabilities(): Capabilities`

The device capabilities the engine runs with. Code: `api.disposed` only.

## Example

```js
const engine = await raa.createEngine(canvas, { msaa: 4, toneMapper: 'agx' });
const red = engine.addMaterial({ color: [0.9, 0.2, 0.2] });
const mesh = engine.addMesh({
  positions: new Float32Array([-1, 0, 0, 1, 0, 0, 0, 1, 0]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  uvs: new Float32Array([0, 0, 1, 0, 0.5, 1]),
  colors: new Uint8Array([255, 255, 255, 255, 255, 128, 0, 255, 0, 128, 255, 255]),
  indices: new Uint32Array([0, 1, 2]),
  subMeshes: [{ first: 0, count: 3, material: red }],
});
engine.addMeshInstance(engine.addNode(null), mesh);
engine.setToneMapping({ mapper: 'aces', exposureEv: 0.5 });
engine.setBackground([0.05, 0.06, 0.08]);
console.log(engine.passes().map((p) => p.name), engine.capabilities().features);
```
