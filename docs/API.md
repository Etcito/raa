# RAA public API v0 (frozen)

This document describes the public API of `@etcito/raa` 0.1.0 (the JS package) and, in the Rust section, of the RAA crates. It is written from `packages/raa/index.d.ts` and `crates/raa-js/src/engine.rs`.

Conventions: right-handed, Y up, the camera looks down -Z, lengths in metres, angles in radians unless the name says `Deg`, times in milliseconds. Matrices are column-major. Quaternions are `[x, y, z, w]`. Colours are linear RGB unless documented as sRGB (asset and material `color` are sRGB in [0, 1] and converted to linear on entry).

## Stability rule

v0 is frozen: until v1, changes are additive only. Nothing is renamed or removed; a new method, a new optional field or a new error code may be added. A behaviour documented here does not change.

## JS API

```js
import { loadRaa } from '@etcito/raa';
const raa = await loadRaa();
const engine = await raa.createEngine(canvas);
```

### `loadRaa`

`loadRaa(options?: { wasmUrl?: string | URL }): Promise<RaaModule>`

Initialises the wasm module (idempotent) and resolves to a fresh frozen `RaaModule`. `wasmUrl` is where the `.wasm` is fetched from; the default is `pkg/raa_js_bg.wasm` next to `index.js`. Each call returns an independent object; the package keeps no module-level mutable state.

### `RaaModule`

`{ readonly version: '0.1.0'; readonly createEngine(canvas, options?) => Promise<Engine> }`

`createEngine(canvas: HTMLCanvasElement, options?: EngineOptions): Promise<Engine>` creates an engine on the canvas at its current pixel width and height. Rejects with `api.invalid_argument`, `core.no_adapter`, `core.request_device` or `core.surface`.

### `EngineOptions`

All fields optional.

- `maxInstances`: capacity of nodes and instances, default 16384.
- `maxMaterials`: maximum number of materials (asset materials included), default 256.
- `powerPreference`: `'high-performance'` (default) or `'low-power'`; anything else throws `api.invalid_argument`.

### `Vec3`

`[number, number, number]`: positions in metres, directions unitless.

### `Quat`

`[number, number, number, number]` as `[x, y, z, w]`; identity `[0, 0, 0, 1]`.

### `PrimKind`

One of `'sphere' | 'box' | 'roundBox' | 'capsule' | 'cylinder' | 'cone' | 'torus' | 'ellipsoid'`. All are centred at the origin; axial kinds use the Y axis. Parameters (metres):

| kind | params | rules |
|---|---|---|
| sphere | `radius` | > 0 |
| box | `halfExtents: Vec3` | each > 0 |
| roundBox | `halfExtents: Vec3`, `radius` | 0 < radius <= min(halfExtents) |
| capsule | `radius`, `halfHeight` | radius > 0, halfHeight >= 0 |
| cylinder | `radius`, `halfHeight` | both > 0 |
| cone | `radius`, `halfHeight` | both > 0; base at y = -halfHeight, apex at +halfHeight |
| torus | `majorRadius`, `minorRadius` | 0 < minorRadius < majorRadius; ring in the XZ plane |
| ellipsoid | `radii: Vec3` | each > 0 |

### `MaterialDesc`

- `color: Vec3`: required, sRGB in [0, 1].
- `roughness?: number`: [0, 1], default 0.6.
- `metallic?: number`: [0, 1], default 0.
- `emissive?: Vec3`: linear RGB, each >= 0, default `[0, 0, 0]`.
- `model?: 'flat' | 'pbr'`: `'flat'` is unlit (base + emissive), `'pbr'` is lit PBR-lite; default `'pbr'`.

### `CameraDesc`

- `eye: Vec3` (metres), `target: Vec3` (metres, different from `eye`): required.
- `up?: Vec3`: default `[0, 1, 0]`.
- `fovYDeg?: number`: vertical field of view in degrees, in (0, 180), default 60.
- `near?: number`: metres, > 0, default 0.1.
- `far?: number`: metres, > near, default 200.

The aspect ratio follows the canvas.

### `LightDesc`

The directional light; all fields required. `direction: Vec3` is the direction the light travels (normalised by the engine, non-zero), `color: Vec3` linear RGB, `intensity: number` >= 0.

### `AmbientDesc`

All fields required: `color: Vec3` (linear RGB), `intensity: number` >= 0.

### `AdapterInfo`

`{ vendor, architecture, device, description: string; isFallbackAdapter: boolean }` as reported by WebGPU. The strings may be empty. `isFallbackAdapter` is true for a software adapter handed out by the browser.

### `FrameStats`

Returned by `stats`.

- `adapter: AdapterInfo`.
- `hardware: boolean`: true when the adapter is not a fallback adapter and no adapter field matches swiftshader, llvmpipe, lavapipe, software or basic render (case-insensitive).
- `firstFrameMs: number | null`: milliseconds, navigation-relative, when frame 0's GPU work completed; null before.
- `frames: number`: frames rendered so far.
- `cpuMsMedian`, `cpuMsP95`: milliseconds of CPU per frame (update + record + submit), median and 95th percentile.
- `instances`, `drawCalls`, `triangles`: counts for the last frame.
- `allocsPerFrameTotal: number | null`: heap allocations in the last frame when an allocation counter is linked, else null.

### `AssetInfo`

Returned by `loadPrimsAsset`.

- `id: number`: asset handle for `spawnAsset`.
- `name: string`: the asset's name.
- `triangles: number`: total triangles at the load quality, after booleans.
- `budget: number`: the asset's triangle budget (default 1500).
- `meshes: number[]`: mesh handles, one per asset mesh, baked into asset space.
- `materials: number[]`: material handles in the asset's material order.
- `anchors: Record<string, Float32Array>`: anchor name to an asset-space matrix of 16 floats, column-major (translation in elements 12 to 14).

### `SpawnedAsset`

Returned by `spawnAsset`; all handles are new. `root: number` is the new root node, `nodes: number[]` has one identity child node per asset mesh (mesh order), `instances: number[]` has one instance per child node (mesh order).

### `RaaError`

`interface RaaError extends Error { code: string; name: 'RaaError' }`. Every failure throws (or rejects with) a `RaaError`; `message` is `"<code>: <human message>"`. Codes are stable, `<area>.<snake_case>`:

- `api.*` (raised by the JS layer): `api.invalid_argument` (a value of the wrong type or out of range), `api.unknown_param` (an extra primitive parameter key), `api.unknown_handle` (a handle that was never issued, or was removed), `api.length_mismatch` (`setTransforms` array sizes), `api.disposed` (any call after `dispose`).
- Pass-through codes, kept as the Rust crates define them: `geom.unknown_kind`, `geom.missing_param`, `geom.invalid_param`, `geom.invalid_mesh`; `material.invalid`, `material.unknown_id`, `material.unknown`; `scene.stale_node`, `scene.stale_instance`, `scene.parent_cycle`; `render.too_many_materials`; `core.no_adapter`, `core.request_device`, `core.surface`, `core.resource`, `core.gpu`, `core.unsupported`.
- Asset loader codes, kept as is: `json.*`, `format.*`, `schema.*`, `value.*`, `id.*`, `kind.unknown`, `kind.params`, `cap.nodes`, `cap.depth`, `cap.budget`, `collision.*`, `boolean.unsupported`, `boolean.csg.*`, `loader.*`.

### `Engine`

One canvas, one scene, a forward renderer with the 8 primitive kinds and CSG booleans registered.

Handles (meshes, materials, nodes, instances, assets) are plain numbers issued from a monotonically increasing counter, starting at 1, and never reused. A handle that was never issued or whose object was removed through its own remove method throws `api.unknown_handle`. Descendant nodes and instances removed as a side effect of `removeNode` are stale: using them throws `scene.stale_node` or `scene.stale_instance`. After `dispose` every method throws `api.disposed`.

#### `addMaterial`

`addMaterial(desc: MaterialDesc): number`. Adds a material, returns its handle. Throws `material.invalid`, `api.invalid_argument`, `render.too_many_materials`.

#### `setMaterial`

`setMaterial(id: number, desc: MaterialDesc): void`. Replaces the material behind `id`; takes effect on the next frame. Throws `api.unknown_handle`, `material.invalid`, `api.invalid_argument`.

#### `primitiveMesh`

`primitiveMesh(kind: PrimKind, params: Record<string, number | Vec3>, segments?: number): number`. Generates a primitive (see `PrimKind` for parameters, metres), uploads it and returns the mesh handle. `segments` is the tessellation quality, default 24 (LOD0), rounded and clamped to [3, 256]. Throws `geom.unknown_kind`, `geom.missing_param`, `geom.invalid_param`, `api.unknown_param`, `api.invalid_argument`.

#### `removeMesh`

`removeMesh(id: number): void`. Frees a mesh; the handle is never reused. Throws `api.unknown_handle`.

#### `loadPrimsAsset`

`loadPrimsAsset(json: string, segments?: number): AssetInfo`. Loads a prims v0 asset (format `raa.prims`, version 0) from its JSON text at `segments` (default 24, clamped to [3, 256]), evaluating booleans. Registers its materials and meshes. Throws the loader's code (for example `kind.unknown`, `cap.budget`, `boolean.csg.*`) or `render.too_many_materials`.

#### `spawnAsset`

`spawnAsset(asset: number, parent?: number | null): SpawnedAsset`. Creates a new root node (under `parent` when given), one identity child node per asset mesh, and one instance per child using the asset's mesh and material. Throws `api.unknown_handle`, `scene.stale_node`.

#### `addNode`

`addNode(parent: number | null, t?: Vec3 | Float32Array, r?: Quat | Float32Array, s?: Vec3 | Float32Array): number`. Adds a node and returns its handle. `parent` null makes a root. Defaults: `t` `[0, 0, 0]` (metres), `r` `[0, 0, 0, 1]`, `s` `[1, 1, 1]`. Throws `api.unknown_handle`, `scene.stale_node`, `api.invalid_argument`.

#### `setTransform`

`setTransform(node: number, t, r, s): void`. Sets the local transform: translation `t` (metres), rotation `r` `[x, y, z, w]`, scale `s`; each a `Vec3`/`Quat` or a `Float32Array`. Throws `api.unknown_handle`, `scene.stale_node`, `api.invalid_argument`.

#### `setTransforms`

`setTransforms(nodes: Uint32Array, trs: Float32Array): void`. Sets many local transforms without allocating in steady state. `trs` holds 10 floats per node, in the order of `nodes`: `t` (3, metres), `r` (4, `[x, y, z, w]`), `s` (3). So `trs.length` must equal `10 * nodes.length`, otherwise it throws `api.length_mismatch` before changing anything. An entry error (`api.unknown_handle`, `scene.stale_node`, `api.invalid_argument`) aborts at that entry; nodes before it keep their new transform.

#### `removeNode`

`removeNode(node: number): void`. Removes the node, its subtree and all their instances. The handles of the removed descendants and instances become stale (`scene.stale_node`, `scene.stale_instance`); the node's own handle then throws `api.unknown_handle` or `scene.stale_node`. Throws `api.unknown_handle`, `scene.stale_node`.

#### `addInstance`

`addInstance(node: number, mesh: number, material: number): number`. Draws `mesh` with `material` at `node`; returns the instance handle. Throws `api.unknown_handle`, `scene.stale_node`.

#### `removeInstance`

`removeInstance(id: number): void`. Removes an instance; the handle is never reused. Throws `api.unknown_handle`, `scene.stale_instance`.

#### `setVisible`

`setVisible(id: number, visible: boolean): void`. Shows or hides an instance (visible by default). Throws `api.unknown_handle`, `scene.stale_instance`.

#### `setCamera`

`setCamera(desc: CameraDesc): void`. Sets the camera. Throws `api.invalid_argument` for fovYDeg outside (0, 180), near <= 0, far <= near or eye equal to target.

#### `orbit`

`orbit(dYawRad: number, dPitchRad: number, zoomFactor: number): void`. Rotates the orbit camera by the deltas (radians; pitch clamped to +-89 degrees) and multiplies the distance by `zoomFactor` (> 0; 1 = no zoom). Throws `api.invalid_argument`.

#### `setOrbit`

`setOrbit(target: Vec3 | Float32Array, distance: number, yawRad: number, pitchRad: number): void`. Replaces the orbit camera. Default before any call: target `[0, 0, 0]`, distance 6. `distance` is in metres and > 0; pitch is clamped to +-89 degrees. Throws `api.invalid_argument`.

#### `setLight`

`setLight(desc: LightDesc): void`. Sets the directional light. Throws `api.invalid_argument` for a zero direction or a negative intensity.

#### `setAmbient`

`setAmbient(desc: AmbientDesc): void`. Sets the ambient term. Throws `api.invalid_argument` for a negative intensity.

#### `resize`

`resize(width: number, height: number): void`. Resizes the canvas backing store in pixels (0 clamps to 1); the camera aspect follows. Throws `api.disposed`.

#### `render`

`render(nowMs: number): void`. Renders one frame; `nowMs` is the current time in milliseconds (for example the `requestAnimationFrame` time). Throws `core.surface`, `core.gpu`, `api.disposed`.

#### `stats`

`stats(): FrameStats`. Frame statistics, the adapter and the hardware flag. Throws `api.disposed`.

#### `probeNonEmpty`

`probeNonEmpty(nowMs: number): Promise<boolean>`. Renders one frame at `nowMs` and resolves to true when any pixel differs from the clear colour. Rejects with `core.gpu` or `api.disposed`.

#### `dispose`

`dispose(): void`. Releases the GPU objects and the scene; every later call throws `api.disposed`.

## Size table

Source: `proofs/results/m6-sizes.json` (gzip level 9, node zlib). Limit `WASM_GZIP_MAX` = 300000 bytes gzip per shipped `.wasm`.

| name | raw (bytes) | gzip (bytes) | within 300000 |
|---|---|---|---|
| m1-boot | 167847 | 68097 | yes |
| m2-scene | 243851 | 97217 | yes |
| m3-prims | 265428 | 105867 | yes |
| m4-asset | 394161 | 160891 | yes |
| m5-booleans | 458519 | 182582 | yes |
| example-basic | 465568 | 185085 | yes |
| raa (the package) | 465568 | 185085 | yes |

## Measured numbers

Source: `proofs/results/example-basic.json`, captured through the GPU client at 1280x720, 60 warm-up and 300 measured frames.

- Adapter: `ANGLE (Microsoft Corporation, D3D12 (NVIDIA GeForce RTX 3090), OpenGL ES 3.1 Mesa 26.0.3-1ubuntu1)`, not a fallback adapter. This is a hardware GPU.
- `firstFrameMs`: 235.2 ms against the limit of 500 ms.
- `cpuMsMedian`: 0.40 ms; `cpuMsP95`: 0.60 ms per frame.
- Scene: 220 instances, 13 draw calls, 207068 triangles. The page is non-empty and the resize cycle (1280x720 to 960x540 and back) is ok.

## Rust API

Every crate compiles for the host and for `wasm32-unknown-unknown`. Items below are the public re-exports of each crate's `src/lib.rs`. Errors carry a stable `code()` of the form `<area>.<snake_case>`.

### Crates

**raa-core.** Device, surface, handles, the minimal frame graph and the shared error type. Items: `RaaError`, `AdapterSummary`, `GpuContext`, `GpuOptions`, `Handle`, `Pool`, `MeshId`, `MaterialId`, `CLEAR_COLOR`, `non_empty`, `ClearPass`, the `graph` module (`FrameGraph`, `RenderPass`, `PassBuilder`, `PassId`, `PrepareContext`, `RecordContext`, `ResourceId`, `BACKBUFFER`, `TargetDesc`, `TargetSize`), `CanvasSurface` and `SurfaceFrame` (wasm32 only), `OffscreenTarget` (native only), and the native `testing` module (`software_gpu`, `skip`). M6 hook: `pub use wgpu;` (`raa_core::wgpu`) re-exports the exact wgpu version raa-core uses, so an extension crate implements `RenderPass` without its own wgpu dependency.

**raa-math.** Dependency-free f32 maths. Items: `Vec3`, `Vec4`, `Quat`, `Mat4` (column-major, `M * v`), `Aabb`, `approx_eq`.

**raa-geom.** Mesh data, geometry-provider trait and registry, mesh analysis. Items: `MeshData`, `ParamValue`, `Params`, `Quality` (`LOW` 8, `LOD0` 24, `HIGH` 64, clamped to [3, 256]), `GeometryProvider`, `GeometryRegistry`, `GeomError`, the `analysis` module (`analyze`, `passes`, `mesh_hash`, `unique_positions_1e6`, `MeshReport`) and the `builtin` module (`DebugCube`).

**raa-material.** Simple flat and PBR-lite materials. Items: `ShadingModel`, `Material`, `MaterialLibrary`, `MaterialError`, `MaterialId` (re-exported from raa-core), `srgb_to_linear`, `MATERIAL_GPU_SIZE`, `WGSL_SHADING`.

**raa-scene.** Transforms, hierarchy, instances, camera and lights. Items: `Scene`, `Transform`, `NodeId`, `InstanceId`, `Camera`, `OrbitController`, `DirectionalLight`, `Ambient`, `RenderItem`, `Batch`, `RenderList`, `SceneError` (`scene.stale_node`, `scene.stale_instance`, `scene.parent_cycle`).

**raa-render.** The forward renderer as a frame-graph pass. Items: `Renderer`, `RendererOptions`, `FrameCounts`, `DrawRange`, `FramePrep`, `ForwardPass`, `FORWARD_WGSL`, `FRAME_UNIFORM_SIZE`, `INSTANCE_STRIDE`.

**raa-prims.** The eight SDF primitives as polygons. Items: `PrimKind`, `PrimParams`, `PrimProvider`, `register_all`, `generate`, `triangle_count`, and the `reference` module (SDF reference fixture reader and comparison).

**raa-loader.** The asset-loader trait and the prims v0 loader. Items: `AssetLoader`, `LoaderRegistry`, `LoadContext`, `LoadedAsset`, `LoadedMesh`, `LoadError`, `BooleanResolver`, `BooleanHost`, `BooleanTool`, `BoolOp`, `NoBooleans`, `Collider`, `ColliderShape`, the `json` module and the `prims` module (`PrimsLoader`, `parse`, `build`, `estimate_triangles`, `to_canonical_json`, `PrimsAsset`, `PrimNode`, `PrimMaterial`, `TransformDoc`, `ColliderDoc`, `NodeOp`, `samples`, and the consts `FORMAT`, `VERSION`, `DEFAULT_TRIANGLE_BUDGET`, `MAX_NODES`, `MAX_LEVEL`).

**raa-csg.** Booleans of a primitive against a convex host, otherwise typed refusals. Items: `CsgResolver`, `CsgError` (`csg.*` codes, mapped by the loader to `boolean.csg.*`), `Plane`, `convex_planes`, `intersect_convex`, `subtract_convex`, `CsgCase`, `cases`.

**raa-web.** Web runtime used by the demos and by raa-js. Items: `FrameInput`, `FrameCounters`, `WebApp`, `RuntimeOptions`, `FrameStats`, `stats_to_js`, `error_to_js`, `WebRuntime` (wasm32 only), and the `stats` module. On native only the stats module is built.

**raa-js.** The wasm-bindgen layer behind `@etcito/raa`. Items: `Engine` (the JS class documented above), and the `convert` module (`api_error`, `geom_error`, `load_error`, `scene_error`, `material_error`, `decode_trs`, `check_trs_len`, `quality`, `HandleMap`, `MaterialDesc`, `CameraDesc`, `ModelName`). It is not meant to be used from Rust.

### Extension points

`tests/raa-ext-test` is a crate that depends only on `raa-*` crates and that nothing depends on. It adds a geometry provider and a render pass without editing any RAA crate. This is the proof check `extension` of `proofs/m6-api.mjs` (`cargo test -p raa-ext-test` plus a `cargo metadata` check). The examples below come from it.

#### GeometryProvider

Implement `GeometryProvider` (`kind`, `param_names`, `is_convex`, `triangle_count`, `generate`) and register it in a `GeometryRegistry`. `generate` clears `out`, is deterministic, validates its parameters (`geom.invalid_param`) and emits a closed mesh with outward winding. `StarProvider` (kind `star`, params `points`, `outerRadius`, `innerRadius`, `halfHeight`) registers next to the eight primitives:

```rust
let mut geometry = GeometryRegistry::new();
raa_prims::register_all(&mut geometry)?;          // the 8 kinds
geometry.register(Box::new(StarProvider))?;       // kinds() is now 9 long, last = "star"
```

A prims v0 asset using kind `star` loads when `strict_kinds` is false; with `true` it is refused with `kind.unknown`:

```rust
let mut loaders = LoaderRegistry::new();
loaders.register(Box::new(PrimsLoader))?;
let ctx = LoadContext {
    geometry: &geometry,
    booleans: &NoBooleans,
    quality: Quality::LOD0,
    strict_kinds: false,
};
let asset = loaders.load(star_text.as_bytes(), &ctx)?; // 1 mesh, 40 triangles (8 * points)
```

The collider of a non-analytic kind is derived as a convex hull.

#### RenderPass

Implement `RenderPass` (`name`, `declare`, optional `prepare`, `record`, optional `resize`). `declare` states reads and writes through `PassBuilder`; `record` writes into the command encoder. `TintPass` (name `ext-tint`) reads and writes `BACKBUFFER` with `LoadOp::Load` and draws a fullscreen triangle with alpha blending of a fixed tint. Its wgpu types come from `raa_core::wgpu`:

```rust
use raa_core::wgpu;
impl RenderPass for TintPass {
    fn name(&self) -> &str { "ext-tint" }
    fn declare(&self, b: &mut PassBuilder) { b.read(BACKBUFFER); b.write(BACKBUFFER); }
    fn record(&mut self, ctx: &mut RecordContext<'_>) -> Result<(), RaaError> {
        // wgpu render pass on ctx.view(BACKBUFFER)?, LoadOp::Load
        Ok(())
    }
}
```

#### FrameGraph: add_pass_after and add_pass_before

`add_pass` appends. `add_pass_after(name, pass)` and `add_pass_before(name, pass)` insert relative to a named pass; an unknown name gives the code `framegraph.unknown_pass`. Compile checks the order (`framegraph.duplicate_pass`, `framegraph.unknown_resource`, `framegraph.read_before_write`, `framegraph.no_output`).

```rust
let mut graph = FrameGraph::new(wgpu::TextureFormat::Rgba8UnormSrgb);
graph.add_pass(Box::new(ClearPass::new("forward", BACKBUFFER, CLEAR_COLOR)));
graph.add_pass_after("forward", Box::new(TintPass::new()))?;
graph.compile()?;
assert_eq!(graph.pass_names(), ["forward", "ext-tint"]);
let e = graph.add_pass_after("missing", Box::new(TintPass::new())).unwrap_err();
assert_eq!(e.code(), "framegraph.unknown_pass");
// add_pass_before("forward", ...) places the new pass ahead of it, with the same error code.
```

With the real renderer, `Renderer::install(&mut graph)` appends the pass named `forward`, and `add_pass_after("forward", ...)` puts the extension after it.

#### AssetLoader and LoaderRegistry

Implement `AssetLoader` (`format`, `sniff`, `load`). `LoaderRegistry::load` asks the loaders in registration order and uses the first whose `sniff(bytes)` is true; if none matches it returns `loader.unknown_format`. Registering the same `format` twice gives `loader.duplicate_format`. `PrimsLoader` sniffs the format string in the first 512 bytes. Registration and loading are shown in the GeometryProvider example above (`LoaderRegistry::new`, `register(Box::new(PrimsLoader))`, `load`).

#### BooleanResolver

`LoadContext::booleans` is a `&dyn BooleanResolver`. `NoBooleans` refuses any subtract or intersect node with `boolean.unsupported`. `CsgResolver` (raa-csg) applies convex booleans and refuses everything else with `boolean.csg.*` codes (for example `boolean.csg.host_not_convex`). A custom resolver implements `resolve(&self, host, tools, out)`:

```rust
let with_csg = LoadContext { geometry: &geometry, booleans: &CsgResolver, quality: Quality::LOD0, strict_kinds: true };
let without = LoadContext { booleans: &NoBooleans, ..with_csg }; // subtract nodes now fail with boolean.unsupported
```
