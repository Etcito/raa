# @etcito/raa

A small, extensible rendering engine in Rust, compiled to WebAssembly over WebGPU.
It has 8 primitive kinds, prims v0 assets with CSG booleans, instancing, GPU
skinning and a PBR-lite forward renderer. Version 0.5.0 (public API v0).

## Install

The package is built offline into a tarball (`scripts/build-package.sh` in the RAA
repository writes `dist/etcito-raa-0.5.0.tgz`). Install it from that file:

```sh
npm install ./dist/etcito-raa-0.5.0.tgz
```

The package is a plain ES module with no dependencies. It needs a browser with WebGPU
and a secure origin (https, or http on localhost or 127.0.0.1). The `.wasm` is loaded
from `pkg/raa_js_bg.wasm` next to `index.js`. Pass `wasmUrl` to load it from elsewhere.

## Usage

```js
import { loadRaa } from '@etcito/raa';

const raa = await loadRaa(); // or loadRaa({ wasmUrl: '/assets/raa_js_bg.wasm' })
const canvas = document.querySelector('canvas');
const engine = await raa.createEngine(canvas, { maxInstances: 4096 });

const sphere = engine.primitiveMesh('sphere', { radius: 0.5 });      // segments default 24
const box = engine.primitiveMesh('box', { halfExtents: [0.4, 0.4, 0.4] }, 8);
const red = engine.addMaterial({ color: [0.9, 0.2, 0.2], roughness: 0.4 });
const grey = engine.addMaterial({ color: [0.6, 0.6, 0.6], model: 'flat' });

const root = engine.addNode(null);
const a = engine.addNode(root, [-1, 0, 0]);
const b = engine.addNode(root, [1, 0, 0], [0, 0.383, 0, 0.924]);    // quaternion [x, y, z, w]
engine.addInstance(a, sphere, red);
engine.addInstance(b, box, grey);

engine.setCamera({ eye: [0, 2, 6], target: [0, 0, 0], fovYDeg: 60 });
engine.setLight({ direction: [-0.4, -1, -0.3], color: [1, 1, 1], intensity: 3 });
engine.setAmbient({ color: [0.6, 0.7, 0.9], intensity: 0.3 });

function frame(nowMs) {
  engine.setTransform(b, [1, 0, 0], [0, Math.sin(nowMs / 2000), 0, Math.cos(nowMs / 2000)], [1, 1, 1]);
  engine.render(nowMs);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

A prims v0 asset is loaded once and spawned any number of times:

```js
const info = engine.loadPrimsAsset(jsonText);   // { id, name, triangles, budget, meshes, materials, anchors }
const spawned = engine.spawnAsset(info.id, root); // { root, nodes, instances }
```

To move many nodes per frame, use `setTransforms(nodes: Uint32Array, trs: Float32Array)`
with 10 floats per node (t3, r4 `[x, y, z, w]`, s3). Reuse the typed arrays from frame
to frame.

## Skinned meshes

`addMesh` and `updateMesh` accept `joints: Uint16Array` and `weights: Float32Array`
(four slots per vertex; supply both or neither; each vertex's weights sum to 1
within 1e-4). Pose an instance with `setSkinPose(instance, matrices)`, or many at
once with `setSkinPoses(instances, matrices)`: column-major mesh-space
`model(pose) * inverse_bind` matrices, 16 floats per joint in S16 bone order.
Instances sharing a mesh keep independent poses; an instance without a pose draws
at rest.

The `skin` compute pass (order 15, feature bit 20) is enabled by default; toggle it
with `setPassEnabled('skin', false)`. When disabled, the frame draws at rest.
Prepass, forward, shadows, transparent, editor wireframe and selection outline all
follow the pose. Declarations are in `types/m18-skinning.d.ts`; details are in
`docs/api/m18-skinning.md`.

Units: metres, radians (except `fovYDeg`, in degrees), milliseconds. Coordinates are
right-handed with Y up. Material `color` is sRGB. Light, ambient and emissive colours
are linear.

## Error model

Every failure throws (or, for promises, rejects with) a `RaaError`: a JS `Error` whose
`name` is `'RaaError'`, whose `code` is a stable `<area>.<snake_case>` string and whose
`message` is `"<code>: <human message>"`. Refusals are always errors. The engine never
returns NaN, an empty result or a silent skip.

```js
try {
  engine.primitiveMesh('sphere', { radius: -1 });
} catch (e) {
  if (e.name === 'RaaError') console.warn(e.code); // 'geom.invalid_param'
}
```

Common codes: `api.invalid_argument`, `api.unknown_param`, `api.unknown_handle`,
`api.length_mismatch`, `api.disposed`, `geom.*`, `material.*`, `scene.stale_node`,
`scene.stale_instance`, `render.too_many_materials`, `core.*`. Asset loader errors keep
their code (for example `kind.unknown`, `cap.budget`, `boolean.csg.*`). Handles are
numbers that are never reused. After `removeNode`, the handles of the removed
descendants throw `scene.stale_node` or `scene.stale_instance`.

## API

The full reference is in [docs/API.md](../../docs/API.md). The types are in `index.d.ts`.

## License

MIT
