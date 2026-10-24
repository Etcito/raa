# @etcito/raa-sdfm

Adapter that loads `sdfm.game-asset` v1 files and turns them into RAA meshes and materials, so an asset can be shown side by side with SDF Modeler.

Meshing and material compilation are done by MAMA (the SDF document engine, package `mama`). RAA renders the result through `@etcito/raa`. Both are peer dependencies: `@etcito/raa` `^0.1.0` and `mama` `^0.1.0`. The package has no dependencies and no build step.

MAMA is reached only through `src/mama-bridge.js`. The calls below are the ones that file makes, reviewed against MAMA 0.1.0 at commit 864ebc3 (sibling checkout `../MAMA`).

## MAMA calls

| specifier | export | signature used | returns | MAMA file @ 864ebc3 |
|---|---|---|---|---|
| `mama` | `readSdfm` | `readSdfm(document)` (the asset's `document`, unchanged) | `SdfDocument` | `src/sdfm.ts:26` |
| `mama/core/game/mesh-assets.js` | `meshAssets` | `meshAssets(doc, { pool, groups })`; `pool: ComputePool` (node: the caller's pool; browser: from `setupBrowserHost`); `groups: NodeId[]` | `Promise<AssetMeshResult[]>`; each result `{ group, parts: [{ partId, name, frame: Float64Array, subMeshes: [{ material: string \| null, positions: Float32Array, normals: Float32Array, uvs?: Float32Array, indices: Uint32Array }] }], materials, error: { code, message, fix } \| null }` | `src/core/game/mesh-assets.ts:133` |
| `mama` | `buildMaterial` | not called by the bridge: `meshAssets` calls `buildMaterial(subset, id)` for the materials of each group; the bridge reads the values in `AssetMeshResult.materials` | `BuiltMaterial { id, name, def, graph: CompiledGraph \| null, error }`; the bridge reads `id`, `error`, `graph.wgslBody`, `graph.helpers`, `graph.constants` | `src/sdfm.ts:77` |
| `mama/core/material-graph/helpers.js` | `HELPERS` | `HELPERS[id].code`, `HELPERS[id].deps` (helper id -> WGSL text) | `Record<HelperId, { deps?: HelperId[], code: string }>` | `src/core/material-graph/helpers.ts:73` |
| `mama/browser` | `setupBrowserHost` | `setupBrowserHost({ workerUrl, manifoldWasmUrl, size? })` | `ComputePool` (a meshing worker pool) | `src/browser/index.ts:111` |

Notes on the table:

- `buildMaterial` calls `compileMaterialGraph` (`src/core/material-graph/compile.ts:97`) internally. The bridge calls neither directly. A `BuiltMaterial` with `graph: null` throws `sdfm.material_unmappable` with MAMA's error.
- Helpers: each id of `graph.helpers` and its `deps` (dependencies first) is resolved through `HELPERS`; an id without WGSL source throws `sdfm.material_unmappable` naming the material and the helper.
- Imports: `importMama(base?)` in `src/mama-bridge.js` imports `mama`, `mama/core/game/mesh-assets.js`, `mama/core/material-graph/helpers.js` and `mama/browser` (bare specifiers, or the matching files under MAMA's `dist/` URL `base`).
- Output of `meshDocument(mama, document, { pool, groups? })`: `{ groups: [{ group, parts: [{ partId, name, frame: Float64Array, meshes: [{ material, positions: Float32Array, normals: Float32Array, uvs: Float32Array | null, indices: Uint32Array }] }] }], materials: MamaMaterial[] }`. Every part is rigid and the hierarchy is flat (one node per part under the spawn root, local transform = `frame`); no rigid flag or parent is read from the document. Each MAMA sub-mesh is one mesh; sub-meshes are not merged.
- Refusals: a group whose `error` is set (`error.code` `MESHER_GROUP_UNSUPPORTED`, message naming the group) becomes `SdfmError('sdfm.mesher_refused', '<group>: <error.message>' plus `error.fix` when present, path '/document')`. The first refused group in result order wins. Any other exception from MAMA becomes `sdfm.worker`.
- Browser host: MAMA has no viewport host. Nothing in MAMA mounts a canvas, loads a document into a viewport, sets a camera or renders a frame. `setupBrowserHost` returns a meshing worker pool only, so this README has no rows for mount, load, camera or render. The compare viewport (SDF Modeler side of the side-by-side page) is held for the product owner.

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

## Browser build

MAMA's browser build is `dist/browser/` in the MAMA checkout (`compute-worker.js` and `index.js`). Build it with `npm run build` in `../MAMA`. That runs the `@real3d/quadmesh` build and then `tsc -p tsconfig.json`.

## How to test

`node --test packages/raa-sdfm/test/`

## API

Filled by a later task.
