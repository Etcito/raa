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

MAMA's build is `dist/` in the MAMA checkout. Build it with `npm run build` in `../MAMA`. That runs the `@real3d/quadmesh` build and then `tsc -p tsconfig.json`. That output imports npm packages by bare specifier, so `scripts/build-sdfm-compare.sh --mama ../MAMA` stages a browser ESM bundle of the entry points above (`index.js`, `core/game/mesh-assets.js`, `core/material-graph/helpers.js`, `browser/index.js`, `browser/compute-worker.js`) as `mama/` with MAMA's own esbuild, at the same paths as under `dist/`, plus `manifold.wasm`. The `base` of `importMama` in the page is that `mama/` URL.

## How to test

`node --test packages/raa-sdfm/test/`

## API

MAMA is reached only through `src/mama-bridge.js`: no other module of this package imports it, and the compare page makes no MAMA call of its own. Every failure of the adapter is an `SdfmError` with one of the S15 codes below; refusals of the engine propagate unchanged as `RaaError`.

Every exported name of `index.d.ts`:

| name | description |
|---|---|
| `createSdfmLoader` | Creates a loader bound to one engine; `mama` is resolved once and the hash cache is per loader. |
| `SdfmLoaderOptions` | `{ engine, mama, workerUrl?, workers? }`; `mama` resolves to MAMA's `dist/` URL (workers) or to MAMA's exports (in-process). |
| `SdfmLoader` | `validate`, `load` (validate, hash check, cache, mesh, upload), `spawn` (root plus one node and instance per part), `dispose`. |
| `SdfmAsset` | A loaded asset: the v1 fields `id`, `hash`, `name`, `category`, `world`, `anchors`, `bounds`, `collision` passed through, plus `parts`. |
| `SdfmPart` | One rigid part: `name`, `groupId`, `rigid` (always true), `parent` (always null), `t`, `r`, `mesh`, `materials`. |
| `SdfmSpawned` | `{ root, parts }`: the spawn root node and, per part name, its `node` and `instance` handles. |
| `SdfmProgress` | One `onProgress` record: `stage` (`validate`, `mesh`, `upload`), `done`, `total`. |
| `SdfmValidationResult` | The `validate` result: `{ ok, errors }`; validation is synchronous and needs no MAMA. |
| `SdfmValidationError` | One validation entry: an S15 `code`, a JSON pointer `path` and a human `message`. |
| `SdfmMamaModule` | The MAMA exports used for in-process meshing: `readSdfm`, `meshAssets`, `HELPERS`, `setupBrowserHost?`, `pool?`. |
| `SdfmCompiledMaterial` | A MAMA compiled material as the bridge normalises it: `name`, `key` (sha256 hex), `body`, `helpers`, `constants`, `fields`. |
| `SdfmMappedMaterial` | The `mapMaterial` result: `functionName`, `wgsl` (an S11 `material_main`), `desc` (always `blend: 'opaque'`), `dropped`. |
| `mapMaterial` | Wraps a `SdfmCompiledMaterial` into an S11 user material function and a surface-material description. |
| `SdfmError` | The adapter's refusal: `code` (an S15 code) and an optional JSON pointer `path`; message prefixed `<code>: `. |
| `SdfmErrorCode` | Union of the eleven S15 codes listed in the table below. |
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
| `sdfm.hash_mismatch` | The recomputed sha256 of the canonical document differs from `hash` (path `/hash`). |
| `sdfm.mesher_refused` | MAMA refused a group with a value error (`MESHER_GROUP_UNSUPPORTED`); the first refused group in result order wins, the message names the group and its fix (path `/document`). |
| `sdfm.material_unmappable` | MAMA could not build a material or resolve a helper, or the mapped WGSL is not an S11 function (unknown field, forbidden token, `raa_` identifier, more than 65536 bytes); the message names the material and the token. |
| `sdfm.worker` | MAMA could not be imported or threw, a worker crashed or broke the protocol, or `options.mama()` failed. |
| `sdfm.disposed` | `load`/`spawn` after `dispose` (or a worker pool request after termination). |

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
