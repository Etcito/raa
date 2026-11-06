# @etcito/raa-sdfm

Adapter that loads `sdfm.game-asset` v1 files and turns them into RAA meshes and materials, so an asset can be shown side by side with SDF Modeler.

Meshing and material compilation are done by MAMA (the SDF document engine, package `mama`). RAA renders the result through `@etcito/raa`. The consuming app provides both; the package declares no dependencies and has no build step.

## Skinned assets

A part of a group with a `rig` carries `skin: { joints, weights, rig }`. The loader
passes those arrays through unchanged as the mesh's `MeshDesc.joints` and
`MeshDesc.weights`; joint indices are in the rig's `bones` order. The loaded asset
exposes the group's S16 `rig` and S17 `clips` as `asset.rig` and `asset.clips`;
unrigged assets load exactly as before. An invalid rig is refused as `sdfm.rig`,
with the S16 code in the message.

## Procedural scenes

`loadProceduralScene(engine, loader, scene, options)` loads one `sdfm.procedural-scene` v1 file
(section 66). It has two variants:

- **sdf-refs**: asset sources are read through `options.resolveAsset(file)` (once per distinct
  file) and loaded with `loader.load`. Composite sources (csg, realized) are built by MAMA's
  `sceneSourceDocument` into an in-memory `sdfm.game-asset` and meshed by the loader; a bent
  instance (`deform`) gets its own wrapper document with a `bend` deformer and is meshed once per
  distinct bend per source. Colliders, anchors and bounds follow the source.
- **mesh-refs**: each mesh file is read through `options.resolveMesh(path)`, decoded from the S24
  saved-mesh format with `decodeMeshFile` / `meshFilePayload`, uploaded with `Engine.addMesh`, and
  each scene instance becomes `Engine.addMeshInstance` under its own node. A mesh-refs instance
  never carries a deform.

`mode: 'trace'` loads an sdf-refs scene without meshing: MAMA's `buildSceneTraceInput` builds the
traces, `packTrace` (evaluator op 4, given `options.procgen`) packs them in Rust and the engine
installs the program with `Engine.setTraceScene`. Live handles share that one slot: `setVisible`
and `dispose` repack the visible inputs (`placement` offsets a scene's instances), and the last
handle clears it.

Either variant spawns one engine node per scene instance and its parts under it: instances are
never flattened, merged or baked into the root. A scene instance's `attrs` and `lod` are forwarded
to every part it spawned through `Engine.setInstanceParams` and, when the engine has it,
`Engine.setInstanceLodBias`; `stats().attributed` counts the scene instances whose parameters were
set. `stats()` reports the instance count, the number of distinct loaded assets' parts and the
pending uploads; `setVisible` covers every spawned part instance; `dispose` removes the root node
and every mesh the handle added.

Refusals: `sdfm.scene_format` for a parse failure, an unknown `mode`, a mesh-refs scene in trace
mode or a deform on a mesh-refs instance; `sdfm.scene_resolve` when a resolve callback, the S24
decode, `sceneSourceDocument` or trace packing fails; `sdfm.worker` when the default MAMA import
fails.

MAMA is reached only through `src/mama-bridge.js`. The calls below are the ones that file makes, reviewed against MAMA 0.1.0 at commit 864ebc3 (sibling checkout `../MAMA`).

## MAMA calls

| specifier | export | signature used | returns | MAMA file @ 864ebc3 |
|---|---|---|---|---|
| `mama` | `serializeDocument` | `serializeDocument(document)` | Document text whose UTF-8 bytes define the asset sha256 | MAMA index export |
| `mama` | `readSdfm` | `readSdfm(document)` (the asset's `document`, unchanged) | `SdfDocument` | `src/sdfm.ts:26` |
| `mama/core/game/mesh-assets.js` | `meshAssets` | `meshAssets(doc, { pool, groups })`; `pool: ComputePool` (node: the caller's pool; browser: from `setupBrowserHost`); `groups: NodeId[]` | `Promise<AssetMeshResult[]>`; each result `{ group, parts: [{ partId, name, frame: Float64Array, subMeshes: [{ material: string \| null, positions: Float32Array, normals: Float32Array, uvs?: Float32Array, indices: Uint32Array, edges?: Uint32Array }] }], materials, error: { code, message, fix } \| null }` | `src/core/game/mesh-assets.ts:133` |
| `mama` | `buildMaterial` | not called by the bridge: `meshAssets` calls `buildMaterial(subset, id)` for the materials of each group; the bridge reads the values in `AssetMeshResult.materials` | `BuiltMaterial { id, name, def, graph: CompiledGraph \| null, error }`; the bridge reads `id`, `error`, `graph.wgslBody`, `graph.helpers`, `graph.constants` | `src/sdfm.ts:77` |
| `mama/core/material-graph/helpers.js` | `HELPERS` | `HELPERS[id].code`, `HELPERS[id].deps` (helper id -> WGSL text) | `Record<HelperId, { deps?: HelperId[], code: string }>` | `src/core/material-graph/helpers.ts:73` |
| `mama/core/material-graph/helpers.js` | `BASE_WGSL` | read as text (MAMA emits it in front of every material module, `src/core/material-graph/slots.ts:402`) | `string`: the always-emitted constants and scalar helpers (`mat_clamp01`, `mat_smoothstep`, ...) | `src/core/material-graph/helpers.ts:37` |
| `mama/browser` | `setupBrowserHost` | `setupBrowserHost({ workerUrl, manifoldWasmUrl, size?, workerMode? })`; `workerMode: 'blob'` starts classic `blob:` workers (opaque-origin pages) | `ComputePool` (a meshing worker pool) | `src/browser/index.ts:111` |
| `mama/procgen/scene-file.js` | `parseProceduralScene` | `parseProceduralScene(text)` (the `sdfm.procedural-scene` v1 text) | MAMA's strict `ProceduralScene` value | `src/procgen/scene-file.ts` |
| `mama/procgen/scene-build.js` | `sceneSourceDocument` | `sceneSourceDocument(scene, sourceIndex, lookupAsset)`; `lookupAsset(file)` returns the parsed asset object or null | `SdfDocument` of one composite (csg, realized) or of the asset source a bend wraps | `src/procgen/scene-build.ts` |
| `mama/procgen/trace.js` | `buildSceneTraceInput`, `packTrace` | `buildSceneTraceInput(scene, lookupAsset)` resolves every sdf-refs source; `packTrace(procgen, input)` is evaluator op 4, given `options.procgen` | section 67 `TraceInput`, then the little-endian TraceProgram v1 bytes `Engine.setTraceScene` takes | `src/procgen/trace.ts` @ 2086284 |
| `mama/core/material-graph/cpu-eval.js` | `cpuGraphOrNull` | compare page only (`examples/sdfm-compare/lib/mama-view.js`), not the bridge: `cpuGraphOrNull(def, doc).eval(p, p, n, masks, {})` per vertex, `p` and `n` in world space | `CpuGraph \| null`; the page reads `MaterialValues.baseColor` (linear) | `src/core/material-graph/cpu-eval.ts:482` |

Notes on the table:

- `buildMaterial` calls `compileMaterialGraph` (`src/core/material-graph/compile.ts:97`) internally. The bridge calls neither directly. A `BuiltMaterial` with `graph: null` throws `sdfm.material_unmappable` with MAMA's error.
- Helpers: `BASE_WGSL` comes first as helper `base` (the graph bodies call its `mat_clamp01` without listing it in `graph.helpers`); then each id of `graph.helpers` and its `deps` (dependencies first) is resolved through `HELPERS`; an id without WGSL source throws `sdfm.material_unmappable` naming the material and the helper.
- Imports: `importMama(base?)` in `src/mama-bridge.js` imports `mama`, `mama/core/game/mesh-assets.js`, `mama/core/material-graph/helpers.js` and `mama/browser` (bare specifiers, or the matching files under MAMA's `dist/` URL `base`).
- Procedural scenes: `importProcedural(base?)` in `src/mama-bridge.js` imports `mama/procgen/scene-file.js` (for `parseProceduralScene`), `mama/procgen/scene-build.js` (for `sceneSourceDocument`), `mama/procgen/trace.js` (for `buildSceneTraceInput` and `packTrace`) and `mama` (for `serializeDocument`); with `base` (MAMA's `dist/` URL) it imports `procgen/scene-file.js`, `procgen/scene-build.js`, `procgen/trace.js` and `index.js` under it. It is the default `options.mama()` of `loadProceduralScene`; an import failure is `sdfm.worker` naming the specifier.
- Output of `meshDocument(mama, document, { pool, groups? })`: `{ groups: [{ group, parts: [{ partId, name, frame: Float64Array, meshes: [{ material, positions: Float32Array, normals: Float32Array, uvs: Float32Array | null, indices: Uint32Array, skin?: { joints: Uint16Array, weights: Float32Array, rig } }] }], rig?, clips? }], materials: MamaMaterial[] }`. Every part is rigid and the hierarchy is flat (one node per part under the spawn root, local transform = `frame`); no rigid flag or parent is read from the document. Each MAMA sub-mesh is one mesh; sub-meshes are not merged. MAMA's S18 skin arrays are copied with their joint order and rig association; `rig` and `clips` are the group's authored S16/S17 values.
- Refusals: a group whose `error` is set (`error.code` `MESHER_GROUP_UNSUPPORTED`, message naming the group) becomes `SdfmError('sdfm.mesher_refused', '<group>: <error.message>' plus `error.fix` when present, path '/document')`. A group error with code `sdfm.rig` becomes `SdfmError('sdfm.rig', ...)` with the S16 code in the message. The first refused group in result order wins. Any other exception from MAMA becomes `sdfm.worker`.
- Browser host: MAMA has no viewport host. Nothing in MAMA mounts a canvas, loads a document into a viewport, sets a camera or renders a frame. `setupBrowserHost` returns a meshing worker pool only, so this README has no rows for mount, load, camera or render. The MAMA pane of the side-by-side page is therefore a local reference viewport, `examples/sdfm-compare/lib/mama-view.js`, labelled `MAMA 864ebc3 mesher (reference viewport)`. It meshes the document with `readSdfm`, `setupBrowserHost` (`workerMode: 'blob'`) and `meshAssets` through `meshDocument`, colours each vertex with the `cpuGraphOrNull` base colour, and draws the result with plain WebGPU (Lambert shading, the same camera and size as the RAA pane). It is not SDF Modeler's viewport.

## MaterialFull fields

MAMA's material graph returns `MaterialFull`, declared in `src/core/material-graph/wgsl-parts.ts:242` @ 864ebc3:

`struct MaterialFull { base_color: vec3f, metallic: f32, roughness: f32, emissive: vec3f, height: f32 }`

The field names, in declaration order, are `base_color`, `metallic`, `roughness`, `emissive`, `height`. The bridge pins them in `MamaMaterial.fields` with this commit.

`MamaMaterial` is `{ name, key, body, helpers: [{ id, wgsl }], constants: number[], fields }`, where `key` is the sha256 hex of `canonicalJson({ body, helpers, constants, fields })`.

`mapMaterial` maps `MaterialFull` onto `MaterialOutput` (S11) as follows. Every other `MaterialOutput` field comes from `raa_default_output(in)`.

| MaterialFull | MaterialOutput | mapping |
|---|---|---|
| `base_color` | `base_color` | copied |
| `metallic` | `metallic` | copied |
| `roughness` | `roughness` | copied |
| `emissive` | `emissive` | copied |
| `height` | none | dropped; named in the `dropped` list |

## Per-pixel field shading

A saved game mesh part carries `shading` (MAMA `core/paramesh/field-program.ts` `PartShading`, mesh format 3): the part's SDF as a field program (numbers), each field operand's colour, material (or folded `constant` values) and node frame, the part frame and the look. `fieldPartShading(shading, materials, { field: FIELD_PROGRAM_WGSL })` returns ONE interpreter material function and the part's data block:

- `name` and `wgsl`: the function. Its text is fixed. Every part whose operands are constant materials or colours gets the same text, so a scene compiles one pipeline for all of them. A part whose operands run material graphs (a pattern, the player colour) adds those graphs, and parts with the same graphs share that text too. Register it once per `name` with `Engine.addFunction`.
- `words`: the part's data block (layout `FIELD_DATA_*` in `src/field-shading.js`). Add it with `Engine.addMaterialData(words)` and pass the returned base as the surface material's `data`. Free it with `removeMaterialData` when the part goes away.

The caller hands in the evaluator text, MAMA's `FIELD_PROGRAM_WGSL` from `core/paramesh/field-program-wgsl.js`, so this package imports nothing from MAMA. `debug` (the normal as an emissive colour) and `single` (one material per object) are data flags, so they reuse the same pipeline. The vertex streams come from `fieldSurfaceAttributes(positions, normals)`.

## Baked field shading

MAMA's game render option `shading: 'baked'` (`core/game/render-options.ts`) bakes a part's normal and material once, at load time, on the GPU, into 2D textures laid out by the part's UV atlas (saved mesh format 4; MAMA `core/game/part-atlas.ts`). Nothing is downloaded and nothing is a 3D volume (`src/field-bake.js`):

- `createFieldBaker(device).bake({ shading, materials, field, positions, normals, uvs, tangents, indices, size })` rasterises the part's triangles in UV space with the same interpreter as the per-pixel shading (`fieldShadingCore`), dilates every chart into its gutter and reads back three RGBA8 images: the tangent-space normal with metallic, the sRGB colour with roughness, and the emissive (null when black). Parts with the same material graphs share one bake pipeline. `device` is the `GPUDevice` RAA renders with.
- `tangents = partTangents(positions, normals, uvs, indices)`: pass the same array to the bake and to the drawn mesh (`MeshDesc.tangents`).
- `size`: MAMA `bakeTextureSize(atlasScale(positions, uvs, indices), options)`, then `fitBakeSizes(parts)` for every part of the engine so the bakes fit RAA's 4096 x 4096 texture atlas (the densest parts halve first).
- Upload the images with `Engine.addTexture`, register `BAKED_MATERIAL_WGSL` once as `BAKED_FUNCTION`, and give each part a surface material with `data: addMaterialData(bakedMaterialWords(normal, color, emissive))`. A part whose material takes the player colour keeps its per-pixel shading.
- Or let `createBakeScheduler(engine, { device, field, textureSize })` do all of it (`src/bake-scheduler.js`), as Royal Fiesta and SDF Modeler's Game view do: `prepare(part)` at upload gives the part's surface material and tangents, `queue(parts)` when a scene first shows them starts one run that fits everything queued into `BAKE_RUN_SHARE` (3/4) of the atlas left and bakes `BAKE_BATCH` parts at a time; `onChange` says which parts became done (show the baked draw) or failed (keep the vertex-colour copy, reported once by `onFailed`); `release(part)` gives its textures and data block back.

## Instance parameters

Procgen attributes of one placed instance (`instanceParams(attrs)`, `src/instance-params.js`) become the 16 floats that `engine.setInstanceParams` stores in the instance's `InstanceExtra.params`. The per-pixel field shading (`fieldPartShading`) multiplies the base colour by the tint in slot 0 when its flag is set, so one material and one pipeline serve every variant:

| floats | slot | content |
|---|---|---|
| 0..2 | `INSTANCE_PARAM_SLOTS.tint` (0) | linear rgb from `tint.r`, `tint.g`, `tint.b` (sRGB 0..1 converted; a missing channel is 1) |
| 3 | tint flag | 1 when any `tint.*` key exists, else 0 (and the rgb is 0) |
| 4 | `INSTANCE_PARAM_SLOTS.wind` (4) | `wind` (0 if absent) |
| 5 | `INSTANCE_PARAM_SLOTS.scale` (5) | `scale` (0 if absent) |
| 6..15 | `INSTANCE_PARAM_SLOTS.free` (6) | the other attributes, in ascending key byte order, at most 10; the rest are dropped |

`instanceParams` returns `null` when `attrs` is absent or has no keys, so an instance without attributes sets nothing.

## LOD levels

`createSdfmLoader({ engine, mama, lods: 2 })` makes two coarser Paramesh meshes of each
unskinned part by default (`LOD_LEVELS_DEFAULT = 2`). `lods` must be an integer from 0 to 3;
0 disables levels and any other value throws `TypeError` naming `options.lods`.

After uploading the full meshes, the loader meshes `lodDocument(asset.document, k)` for
each level through the same MAMA path and mesher settings. `lodDocument` clones the document
and halves `settings.paramesh.subdivisions` and `fieldResolution` per level, flooring to
integers and clamping at 1 and 16 respectively. Missing quality values use MAMA's defaults
of 16 and 96. Other document content stays intact.

Parts match by `group/partId`; their levels reuse the full mesh's material handles.
`attachLods(engine, full, fullDesc, levels, levelDescs)` asks Rust's `meshLodError` to measure
each level against the full mesh, registers the `{ mesh, error }` pairs with `setMeshLods`
once, and returns the measured errors. The engine selects a level per instance for the
camera and shadows, keeping the same shading at every level.

Each loaded part exposes `lods: number[]`, the attached handles in increasing level order.
Skinned parts and engines without `setMeshLods` keep `[]`. A failed level, missing matching
part, or upload/attachment failure leaves that part with `[]` and still resolves the load;
unattached level meshes are removed with `removeMesh`. Progress includes stage `'lod'`
when the loader builds levels.

## Browser build

MAMA's build is `dist/` in the MAMA checkout. Build it with `npm run build` in `../MAMA`. That runs the `@real3d/quadmesh` build and then `tsc -p tsconfig.json`. That output imports npm packages by bare specifier, so `scripts/build-sdfm-compare.sh --mama ../MAMA` stages a browser ESM bundle of the entry points above (`index.js`, `core/game/mesh-assets.js`, `core/material-graph/helpers.js`, `browser/index.js`, `browser/compute-worker.js`, plus `core/material-graph/cpu-eval.js` for the compare page's reference viewport) as `mama/` with MAMA's own esbuild, at the same paths as under `dist/`, plus `manifold.wasm`. The `base` of `importMama` in the page is that `mama/` URL.

## How to test

`node --test packages/raa-sdfm/test/`

## API

MAMA is reached only through `src/mama-bridge.js`: no other module of this package imports it. Outside this package, only the compare page's reference viewport (`examples/sdfm-compare/lib/mama-view.js`) uses MAMA: it calls this bridge, plus `readSdfm` and `cpuGraphOrNull` directly (see the table above). Every failure of the adapter is an `SdfmError` with one of the S15 codes below; refusals of the engine propagate unchanged as `RaaError`.

Every exported name of `index.d.ts`:

| name | description |
|---|---|
| `createSdfmLoader` | Creates a loader bound to one engine; `mama` is resolved once and the hash cache is per loader. |
| `SdfmLoaderOptions` | `{ engine, mama, workerUrl?, workers?, lods? }`; `mama` resolves to MAMA's `dist/` URL (workers) or to MAMA's exports (in-process). |
| `SdfmLoader` | `validate`, `load` (validate, hash check, cache, mesh, upload), `spawn` (root plus one node and instance per part), `dispose`. |
| `SdfmAsset` | A loaded asset: the v1 fields `id`, `hash`, `name`, `category`, `world`, `anchors`, `bounds`, `collision` passed through, plus `parts` and, when rigged, `rig` and `clips`. |
| `SdfmPart` | One rigid part: `name`, `groupId`, `rigid` (always true), `parent` (always null), `t`, `r`, `mesh`, `lods`, `materials`, and `skin?` for a part of a rigged group. |
| `SdfmRig`, `SdfmRigBone`, `SdfmRigBinding`, `SdfmClip`, `SdfmPartSkin` | The S16 rig and S17 clips MAMA exports (`bones` order = joint order) and the part skin arrays uploaded as `MeshDesc.joints`/`weights`. |
| `SdfmSpawned` | `{ root, parts }`: the spawn root node and, per part name, its `node` and `instance` handles. |
| `SdfmProgress` | One `onProgress` record: `stage` (`validate`, `mesh`, `upload`, `lod`), `done`, `total`. |
| `SdfmValidationResult` | The `validate` result: `{ ok, errors }`; validation is synchronous and needs no MAMA. |
| `SdfmValidationError` | One validation entry: an S15 `code`, a JSON pointer `path` and a human `message`. |
| `SdfmMamaModule` | The MAMA exports used for in-process meshing: `readSdfm`, `serializeDocument`, `meshAssets`, `HELPERS`, `BASE_WGSL`, `setupBrowserHost?`, `pool?`. |
| `SdfmCompiledMaterial` | A MAMA compiled material as the bridge normalises it: `name`, `key` (sha256 hex), `body`, `helpers`, `constants`, `fields`. |
| `SdfmMappedMaterial` | The `mapMaterial` result: `functionName`, `wgsl` (an S11 `material_main`), `desc` (always `blend: 'opaque'`), `dropped`. |
| `mapMaterial` | Wraps a `SdfmCompiledMaterial` into an S11 user material function and a surface-material description. |
| `SdfmError` | The adapter's refusal: `code` (an S15 code) and an optional JSON pointer `path`; message prefixed `<code>: `. |
| `SdfmErrorCode` | Union of the thirteen S15 codes listed in the table below, plus the S18 `sdfm.rig`. |
| `loadProceduralScene` | Loads an `sdfm.procedural-scene` v1 file (sdf-refs or mesh-refs) into one engine node per instance; `mode: 'trace'` packs an sdf-refs scene into the engine trace slot; returns a `ProceduralSceneHandle`. |
| `ProceduralSceneHandle` | `root`, `instanceCount`, `sourceCount`, `setVisible`, `stats` (`{ instances, meshes, pending, attributed? }`) and `dispose`. |
| `ProceduralSceneOptions` | `{ resolveAsset?, resolveMesh?, parent?, mode?: 'mesh' \| 'trace', procgen?, placement?, mama? }`; `mode` defaults to `'mesh'` and `mama` to the bridge's `importProcedural`. |
| `SdfmProceduralMama` | The MAMA functions `loadProceduralScene` calls: `parseProceduralScene`, `sceneSourceDocument` and `serializeDocument`, plus `buildSceneTraceInput` and `packTrace` in trace mode. |
| `decodeMeshFile` | Decodes an S24 saved `.mesh` file into its header and typed-array views. |
| `meshFilePayload` | Validates a decoded S24 payload and wraps its groups, parts and materials. |
| `INSTANCE_PARAM_SLOTS` | The first slot of each group of an instance's 16 parameter floats: `{ tint: 0, wind: 4, scale: 5, free: 6 }`. |
| `instanceParams` | Converts procgen attributes into the 16 floats of `setInstanceParams`, or `null` without attributes. |
| `LOD_LEVELS_DEFAULT` | Default number of coarser Paramesh meshes (2). |
| `lodDocument` | Copies a document with Paramesh subdivisions and field resolution halved per level. |
| `attachLods` | Measures coarser meshes in Rust, registers the chain with the engine, and returns its errors. |
| `Vec3` | `[x, y, z]`, metres for positions; same shape as `@etcito/raa`'s `Vec3`. |
| `Quat` | `[x, y, z, w]`, a rotation quaternion; same shape as `@etcito/raa`'s `Quat`. |
| `Engine` | Re-exported from `@etcito/raa`: the engine the loader uploads meshes and materials to. |
| `SurfaceBlend` | Re-exported from `@etcito/raa`: the surface blend mode; mapped materials are always `'opaque'`. |

### S15 codes

| code | when |
|---|---|
| `sdfm.format` | `format` is not `sdfm.game-asset` (path `/format`). |
| `sdfm.version` | `version` is not 1 (path `/version`). |
| `sdfm.schema` | Root is not an object, or a key is missing, unknown, out of canonical order, has the wrong type, or an anchor/bounds entry is malformed (JSON pointer). |
| `sdfm.not_finite` | A number anywhere in the asset is not finite (JSON pointer). |
| `sdfm.id` | `id` does not match `^[a-z][a-z0-9]*(\.[a-z0-9-]+)+$` (path `/id`). |
| `sdfm.cap_document` | The canonical document text exceeds 2 MiB (path `/document`). |
| `sdfm.hash_mismatch` | The recomputed sha256 of the UTF-8 bytes of MAMA `serializeDocument(document)` differs from `hash` (path `/hash`). |
| `sdfm.mesher_refused` | MAMA refused a group with a value error (`MESHER_GROUP_UNSUPPORTED`); the first refused group in result order wins, the message names the group and its fix (path `/document`). |
| `sdfm.rig` | MAMA refused a group's S16/S18 rig; the message names the group and the S16 code (path `/document`). |
| `sdfm.material_unmappable` | MAMA could not build a material or resolve a helper, or the mapped WGSL is not an S11 function (unknown field, forbidden token, `raa_` identifier, more than 65536 bytes); the message names the material and the token. |
| `sdfm.worker` | MAMA could not be imported or threw, a worker crashed or broke the protocol, or `options.mama()` failed. |
| `sdfm.disposed` | `load`/`spawn` after `dispose` (or a worker pool request after termination). |
| `sdfm.scene_format` | A procedural scene does not parse, `mode` is unknown, a mesh-refs instance carries a deform, or trace mode is asked for a mesh-refs scene. |
| `sdfm.scene_resolve` | A procedural scene asset or mesh could not be resolved or decoded, `sceneSourceDocument` failed, or trace input building or packing failed; the message names the file or the reason. |

### Usage

```js
import { createSdfmLoader } from '@etcito/raa-sdfm';

const loader = createSdfmLoader({
  engine,
  // Browser: MAMA's dist/ URL, so meshing runs in worker.js workers.
  mama: () => new URL('../../MAMA/dist/', import.meta.url),
  // Node alternative: MAMA's exports, so meshing runs in-process through the bridge.
  // mama: async () => ({ ...(await import('mama')), pool: myMamaPool }),
});

const asset = await loader.load(assetText, {
  onProgress: ({ stage, done, total }) => console.log(stage, done, total),
});
const spawned = loader.spawn(asset); // { root, parts: { [name]: { node, instance } } }
loader.dispose();
```

### Worker protocol

`workerUrl` defaults to `new URL('../worker.js', import.meta.url)`; the pool creates module workers with `new Worker(workerUrl, { type: 'module' })` and dispatches requests round-robin, keyed by `id`.

| direction | message | meaning |
|---|---|---|
| loader -> worker | `{ type: 'init', mamaUrl }` | load MAMA once; `mamaUrl` is MAMA's `dist/` base URL, or absent for bare specifiers. |
| loader -> worker | `{ type: 'mesh', id, document, mesher }` | mesh the asset document; `id` correlates the reply. |
| worker -> loader | `{ type: 'meshed', id, groups, materials }` | the `meshDocument` bridge output; its typed arrays are transferred. |
| worker -> loader | `{ type: 'error', id, code, message, path? }` | an `SdfmError`; any other worker exception becomes `sdfm.worker`. |

## Packaging

`scripts/build-package.sh` packs this JavaScript adapter alongside RAA at the same version. The published manifest declares no peer dependencies: the consuming app provides RAA and MAMA, which are distributed outside the npm registry. This preserves the packaging policy introduced in 0.1.1.

Asset hashes use MAMA’s exported `serializeDocument(document)` exactly, without
extra whitespace or a newline from the adapter. `importMama` includes this export;
callers supplying exports directly must include it. Material graph keys still use
`canonicalJson`. Optional MAMA submesh polygon `edges` pass through as `MeshDesc.edges`,
with vertex offsets after material grouping. Missing lists use triangle edges; an
explicit empty list stays empty.
