# RAA API m29: cluster bake

The declarations are in `packages/raa/types/m29-cluster-bake.d.ts`; the package build
appends them to `packages/raa/index.d.ts`. The module bakes a single-material mesh into a
cluster DAG (a `ClusterSection`, S29 and S30) on the CPU. The module functions need neither
an Engine nor WebGPU and run in Node. The Engine methods keep the sections of a mesh on the
CPU side, either attached by the caller or baked at runtime over frames.

```ts
interface RaaModule {
  bakeClusters(input: ClusterBakeInput): ClusterSection
  createClusterBake(input: ClusterBakeInput): ClusterBake
  validateClusters(section: ClusterSection): void
  clusterCutMesh(section: ClusterSection, errorMetres: number): ClusterCutMesh
}
interface Engine {
  attachMeshClusters(mesh: number, sections: ClusterSection[]): void
  bakeMeshClusters(mesh: number, options?: ClusterBakeOptions): void
  clearMeshClusters(mesh: number): void
  meshClusters(mesh: number): MeshClusterState
  setClusterBakeBudget(ms: number): void
  clusterBakeBudget(): number
}
```

## Module functions

The `RaaModule` object comes from `loadRaa(options?)`, which accepts
`wasm?: BufferSource | WebAssembly.Module`. When `wasm` is present it is used instead of
`wasmUrl`. Every call, overlapping or not, shares the first call's single wasm
instantiation; a failed one is forgotten so a later call retries.

- `bakeClusters(input)` bakes one mesh to the end and returns its `ClusterSection`.
- `createClusterBake(input)` returns a sliced `ClusterBake` (see below). The input is
  copied and validated at creation.
- `validateClusters(section)` runs the complete structural validation of a section and
  returns nothing when it passes.
- `clusterCutMesh(section, errorMetres)` returns a `ClusterCutMesh` (plain typed arrays) of the
  uniform cut of the section at
  `errorMetres` metres, a finite number >= 0. `0` gives the finest cut, which is the welded
  source mesh. The result is plain typed arrays.

All refusals are a `RaaError` (`code` set, see [Codes](#codes)). A refused call changes
nothing.

## Cluster section

`ClusterSection` is a plain object with exactly these sixteen keys, in this order:

| key | type | meaning |
|---|---|---|
| `version` | `1` | section version (`CLUSTER_SECTION_VERSION`) |
| `algorithm` | `1` | builder algorithm (`CLUSTER_ALGORITHM`) |
| `vertices` | `Uint16Array` | 8 u16 per vertex (16 B): grid x, y, z, octahedral normal x, y, uv u, v, zero |
| `triangles` | `Uint32Array` | one packed triangle per u32: three cluster-local vertex indices, 8 bits each |
| `clusters` | `Uint8Array` | 48 B per cluster record, little-endian; a multiple of 48 |
| `nodes` | `Uint8Array` | 48 B per node record, little-endian; a multiple of 48 |
| `rootNode` | `number` | index of the root node, below `nodes.length / 48` |
| `hierarchyDepth` | `number` | integer in 1..64 |
| `levels` | `number` | DAG levels, integer in 1..256 |
| `quantOrigin` | `[number, number, number]` | metres; origin of the one 16-bit position grid of the mesh |
| `quantStep` | `number` | metres; finite and above 0 |
| `uvOrigin` | `[number, number]` | uv origin |
| `uvStep` | `[number, number]` | uv step per component; finite and not negative |
| `boundsCenter` | `[number, number, number]` | metres |
| `boundsRadius` | `number` | metres; finite and not negative |
| `sourceTriangles` | `number` | triangles of the source mesh; a safe non-negative integer |

A vertex position is `quantOrigin + grid * quantStep` (metres); a uv is
`uvOrigin + packed * uvStep`. Typed arrays are copied in both directions: a section returned
by a bake or a cut is independent of the module's memory.

Validation runs in two steps. The wire rules check the exact key set, the versions, the
typed-array types and lengths, the scalar ranges and the per-cluster and per-node counts.
Then the complete check runs: every packed index below its cluster's vertex count, every
range inside its array, every cluster reachable from `rootNode` exactly once, and the error
order (`lod_error` and the bounds grow toward the root). A refusal is `vg.invalid` naming the
field, for example `vertices` or `clusters`; inside `attachMeshClusters` the field is
prefixed by `sections[i].`.

## Bake input and options

`ClusterBakeInput` holds the flat arrays of one single-material mesh, in metres:

- `positions: Float32Array`, 3 values per vertex; its length is a multiple of 3.
- `normals: Float32Array`, 3 values per vertex; the same length as `positions`.
- `uvs?: Float32Array`, 2 values per vertex. Absent means zero uvs for every vertex.
- `indices: Uint32Array`, 3 indices per triangle, each below the vertex count.
- `options?: ClusterBakeOptions`.

A mesh without a triangle is `vg.mesh_empty`. A mesh whose triangles all collapse under the
position quantisation is `vg.mesh_empty` too. A mesh beyond the S29 limits is
`vg.mesh_too_large`.

`ClusterBakeOptions` fields are optional. An omitted field (`undefined`) takes its default.

| option | domain | default |
|---|---|---|
| `maxClusterTriangles` | integer in 8..128 | 128 |
| `maxClusterVertices` | integer in 16..255 | 255 |
| `minReduction` | number in (0, 1) | 0.85 |
| `targetRatio` | number in (0, 1) | 0.5 |

`null`, a value of the wrong type, a non-finite or out-of-domain value and an unknown key
reject with `vg.invalid` naming `options.<key>` (`options` itself when the object is `null`
or not an object). An unknown key is named `options.<key>`. Nothing is baked on a refusal.

## Sliced bake: the `ClusterBake` object

`createClusterBake(input)` returns an object with four methods. A bake runs in bounded units
of work (S29), so a large mesh builds over several calls on the caller's clock.

- `step(budgetMs?)` runs units until `budgetMs` milliseconds of `performance.now()` have
  passed or the bake is done. At least one unit runs per call. The default is 4 ms. A value
  that is not finite or not above 0 is `vg.invalid` naming `budgetMs`. The result does not
  depend on the budgets: any sequence of steps gives the same section.
- `progress()` returns the `ClusterBakeProgress` after the last step:
  `{ stage, level, fraction, done }`. `stage` runs through `weld`, `partition`, `level`,
  `hierarchy`, `pack` and `done`. `level` counts from 1 and is 0 before the first level.
  `fraction` lies in [0, 1], never decreases, and is 1 only when `done` is true.
- `result()` returns the finished `ClusterSection`. Before the bake is done it throws
  `vg.invalid` naming `result` ("the bake is not done").
- `free()` releases the bake. A later call on the same object throws `vg.invalid` naming
  `bake`; a second `free()` does nothing.

A bake error (`vg.mesh_empty`, `vg.mesh_too_large`, or a `vg.invalid` from the builder) is
thrown by `step` and again by `result`.

## Engine: mesh clusters

The six methods of the `Engine` hold the clusters of each mesh handle. They are the CPU side
of the virtualized-geometry module; drawing the clusters is a separate part of m29. Each
method throws `api.disposed` on a disposed Engine and `api.unknown_handle` for an unknown
mesh handle. Refusals leave the mesh's state unchanged.

- `attachMeshClusters(mesh, sections)` takes one section per sub-mesh, in sub-mesh order.
  Every section is validated completely; only when all pass are the mesh's clusters
  replaced, atomically. The state becomes `ready` with source `attached`. A `sections` value
  that is not an array, or whose length differs from the sub-mesh count, is `vg.invalid`
  naming `sections`. A mesh with a colour stream is `vg.mesh_vertex_colors`.
- `bakeMeshClusters(mesh, options?)` starts a runtime bake of every sub-mesh from the
  geometry the Engine retained for the mesh. The bake is sliced over frames: each Engine
  render invocation spends at most `clusterBakeBudget()` milliseconds on it. The state reads
  `baking` with `progress` in [0, 1), then `ready` with source `runtime`. A new bake replaces
  the previous data of the mesh. The options follow the rules above. A refusal at start
  throws: `vg.invalid` (options, or `mesh` with "no retained geometry"),
  `vg.mesh_vertex_colors`, `vg.mesh_empty` or `vg.mesh_too_large`. A refusal found while the
  bake runs is not thrown: the state becomes `refused` with its code and message.
- `clearMeshClusters(mesh)` cancels a bake and drops the data: the state becomes `none`.
- `meshClusters(mesh)` returns a fresh `MeshClusterState`:

| field | type | meaning |
|---|---|---|
| `state` | `'none' \| 'baking' \| 'ready' \| 'refused'` | the state of the mesh |
| `progress` | `number` | runtime bake fraction in [0, 1); 1 when ready, 0 otherwise |
| `source` | `'attached' \| 'runtime' \| null` | origin of a ready set; null when not ready |
| `code` | `string \| null` | refusal code (`vg.invalid`, `vg.mesh_empty`, `vg.mesh_too_large`, ...) |
| `message` | `string \| null` | refusal message |
| `sections` | `number` | sections of a ready mesh (one per sub-mesh); else 0 |
| `clusters` | `number` | clusters of a ready mesh, summed over its sections; else 0 |
| `sourceTriangles` | `number` | source triangles of a ready mesh, summed over its sections; else 0 |
| `levels` | `number` | largest DAG level of a ready mesh; else 0 |

`code` and `message` are null unless the state is `refused`.

- `setClusterBakeBudget(ms)` sets the runtime bake budget. `ms` must be finite and in
  [0.5, 100] milliseconds, else `vg.invalid` naming `ms`. The default is 4.
- `clusterBakeBudget()` returns the current budget in milliseconds.

The budget is spent per render invocation across all baking meshes in mesh-handle order
(ascending handle, the order of creation). The steps of the earliest handle run first; a
bake that does not finish within the budget continues on the next invocation.

## Lifecycle

- `updateMesh(id, desc)` drops the clusters of that mesh and cancels its bake. The state
  becomes `none`; attach or bake again after the update.
- `removeMesh(id)` frees the clusters of the mesh, cancels its bake and drops its retained
  geometry.
- `dispose()` frees every mesh's clusters and cancels every bake.
- A disposed Engine answers `api.disposed` to every Engine method above.
- A `ClusterSection` returned by a module function, or passed to an attach, is the caller's
  own plain object; the module keeps no reference to it between calls.

## Codes

| code | thrown by | meaning |
|---|---|---|
| `vg.invalid` | every function and method | a malformed or out-of-domain value; the message is `<field>: <reason>` |
| `vg.mesh_vertex_colors` | `attachMeshClusters`, `bakeMeshClusters` | the mesh has a colour stream, which the 16 B cluster vertex cannot carry |
| `vg.mesh_empty` | `bakeClusters`, `createClusterBake`, `bakeMeshClusters` | no triangle, or none survives quantisation |
| `vg.mesh_too_large` | `bakeClusters`, `createClusterBake`, `bakeMeshClusters` | beyond the S29 limits (2^24 clusters, 2^32 - 1 packed vertices or triangles) |
| `api.unknown_handle` | Engine mesh methods | the mesh handle is unknown or removed |
| `api.invalid_argument` | Engine mesh methods | the mesh handle is not a valid numeric handle |
| `api.disposed` | Engine mesh methods | the Engine was disposed |

The `vg.invalid` field names used by this module are `positions`, `normals`, `uvs`,
`indices`, `options`, `options.<key>`, `budgetMs`, `result`, `bake`, `errorMetres`,
`sections`, `sections[i].<field>`, `ms` and `mesh`. A refusal changes nothing: the arrays,
the options, the budget, the bake and the mesh's state stay as they were. The one state a
refusal can set is `refused` on `meshClusters`, from a runtime bake that fails after it
started.

## Example: Node, without an Engine

```js
import { readFile } from 'node:fs/promises';
import { loadRaa } from '@etcito/raa';

const raa = await loadRaa({
  wasm: await readFile('node_modules/@etcito/raa/pkg/raa_js_bg.wasm'),
});

// An 8 x 8 grid of quads in the XZ plane, metres, normals +Y.
const n = 8;
const positions = new Float32Array((n + 1) * (n + 1) * 3);
const normals = new Float32Array(positions.length);
const indices = new Uint32Array(n * n * 6);
for (let j = 0; j <= n; j++) {
  for (let i = 0; i <= n; i++) {
    const v = (j * (n + 1) + i) * 3;
    positions.set([i / n, 0, j / n], v);
    normals.set([0, 1, 0], v);
  }
}
let k = 0;
for (let j = 0; j < n; j++) {
  for (let i = 0; i < n; i++) {
    const a = j * (n + 1) + i;
    const c = a + n + 1;
    indices.set([a, c, a + 1, a + 1, c, c + 1], k);
    k += 6;
  }
}

// One call, or a sliced bake that yields between steps.
const section = raa.bakeClusters({ positions, normals, indices, options: { targetRatio: 0.5 } });
raa.validateClusters(section);
const finest = raa.clusterCutMesh(section, 0);
console.log(section.levels, finest.clusters, finest.triangles);

const bake = raa.createClusterBake({ positions, normals, indices });
let progress;
do {
  progress = bake.step(2); // milliseconds per call
} while (!progress.done);
const sliced = bake.result();
bake.free();
```

## Example: Engine, runtime bake over frames

```js
const engine = await raa.createEngine(canvas);
const mesh = engine.addMesh({ positions, normals, indices });

engine.setClusterBakeBudget(2); // milliseconds per render invocation, in [0.5, 100]
engine.bakeMeshClusters(mesh, { maxClusterTriangles: 128 });

function frame(nowMs) {
  engine.render(nowMs); // spends at most the budget on the baking meshes
  const state = engine.meshClusters(mesh);
  if (state.state === 'refused') console.error(state.code, state.message);
  if (state.state === 'ready') {
    console.log(state.source, state.sections, state.clusters, state.levels);
  } else {
    requestAnimationFrame(frame);
  }
}
requestAnimationFrame(frame);
```

To attach sections baked elsewhere instead, pass one `ClusterSection` per sub-mesh, in
sub-mesh order, to `engine.attachMeshClusters(mesh, sections)`.
