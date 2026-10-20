# RAA architecture

RAA is an extensible rendering engine library: small Rust crates compiled to WebAssembly over WebGPU, a JS package `@etcito/raa`, demo pages and an example page. This document is the single contract for its modules. Conventions apply to every module. The numbered shared sections (S1 to S9) define data shapes that modules implement and never redefine.

## Conventions

### Repository, toolchain, layout
- One repository: `raa`. All paths are relative to it. The repository contains `Cargo.toml`, `Cargo.lock` (committed), `LICENSE` (MIT), `README.md` and `.gitignore`. The `.gitignore` is kept as is and blocks target/, node_modules/, dist/, pkg/, *.log, *.glb, *.gltf, *.png, *.jpg, *.avif, *.ktx2 and *.bin.
- Rust edition 2024. `[workspace.package]` sets rust-version to 1.96.0, the stable rustc installed at M1. The root `Cargo.toml` has `members = ["crates/*", "demos/*", "tests/*"]`. These are globs, so adding a crate never edits the root manifest. All third-party versions live in `[workspace.dependencies]`. Member crates use `{ workspace = true }` and may only add features. The rustfmt and clippy toolchain components are required.
- Closed dependency list:
  - `wgpu` 30.0.1, with `default-features = false` and features `["wgsl"]`. wasm32 crates add `webgpu` only. Native targets add `vulkan` and `gl`.
  - `wasm-bindgen = "=0.2.129"`, which must equal the installed CLI.
  - `wasm-bindgen-futures = "=0.4.79"`, `js-sys = "=0.3.106"`, `web-sys = "=0.3.106"`.
  - `pollster` 1.0.1, for native tests and the bench only.

  RAA has no serde, no glam or nalgebra, no bytemuck and no log framework. Adding a dependency is a lead-authorised minimal edit of the root manifest, never a silent one. A task that adds or changes a dependency runs `cargo fetch` for all targets, so the offline proof sandbox has every crate in CARGO_HOME.
- Library crates use `crates/<name>/{Cargo.toml, README.md, src/lib.rs, tests/<topic>.rs}`. The README states the purpose, the public items and how to test.
  - Demo pages: `demos/<mN-name>/`, a cdylib crate plus `web/` and `gpu-input.json`.
  - Test-only crates: `tests/<name>/`.
  - Proofs: `proofs/`. Scripts: `scripts/`. Shared web helpers: `web/`. One-off tools: `tools/`.
  - JS package: `packages/raa/`. Example page: `examples/basic/`.
- Every workspace crate compiles for the host and for `wasm32-unknown-unknown`. Native-only code sits behind `#[cfg(not(target_arch = "wasm32"))]` and web-only code behind `#[cfg(target_arch = "wasm32")]`.
- Profiles: `[profile.wasm-release]` inherits `release` with `opt-level = "z"`, `lto = true`, `codegen-units = 1`, `panic = "abort"`, `strip = true` and `debug = false`. The native `release` profile keeps its defaults.
- Tools:
  - node is `/root/nmc/engines/bin/node`. Scripts use `${NODE:-/root/nmc/engines/bin/node}`. npm is in the same directory.
  - The `wasm-bindgen` CLI 0.2.129 is on PATH.
  - wasm-opt is not used by default. If a binaryen binary is fetched, it lives outside the repository, and the size table reports sizes with and without it.

### Coordinates, units, colour
- Right-handed, Y up, the camera looks down -Z, units are metres. Front faces are counter-clockwise, back-face culling is on and normals point outward.
- Matrices are column-major f32 with column vectors (`M * v`). Clip depth is [0, 1] (WebGPU). The depth format is `Depth32Float`, compare `Less`, cleared to 1.0. Quaternions are `[x, y, z, w]`.
- Colours are linear RGB everywhere in Rust and on the GPU. Asset files store `color` as sRGB in [0, 1], converted at load. When the canvas format is not an `*Srgb` format, the shader encodes linear to sRGB itself (a flag in the frame uniform).

### Errors and refusals
- Each crate has one error type implementing `Display` and `std::error::Error`. It has a stable `code(&self) -> &'static str` of the form `<area>.<snake_case>` and a human message. Refusals are values (a typed error with code and message), never NaN, an empty output, a panic or a skip.
- No `unwrap`, `expect` or indexing panics on paths that public API with user data can reach. They are allowed in tests and on invariants that carry a comment.

### No global state, no hidden allocation, unsafe
- No `static mut`, no `thread_local!` state and no lazy globals in library crates. The only exception is `crates/raa-alloc-count` (atomic counters plus `CountingAlloc`). It is linked as `#[global_allocator]` only by `crates/raa-bench` and by demo crates under the cargo feature `alloc-count`.
- The frame path is `Scene::update_world`, `Scene::build_render_list`, `FramePrep::prepare_cpu`, and RAA's own code inside `FrameGraph::execute` and `WebRuntime::frame`. It performs no heap allocation after warm-up: buffers are preallocated and reused (`clear()` keeps capacity) and sorting is in place. Allocations made inside wgpu or web-sys calls are outside RAA's code. The web demo reports them as `allocsPerFrameTotal`, which is information, not a threshold.
- `unsafe` is allowed only with a `// SAFETY: <reason> (test: <test_fn_name>)` comment on the line directly above, and the named test must exist. `scripts/check.sh` enforces this. The only unsafe code is `unsafe impl GlobalAlloc for CountingAlloc`.

### Testing
- Unit tests go in `#[cfg(test)] mod tests` inside the file. Integration tests go in `crates/<c>/tests/<topic>.rs`. Test names are snake_case and describe the behaviour.
- Native GPU tests call `raa_core::testing::software_gpu()`. When it returns `None`, the test prints `SKIP <test name>: no software adapter (<reason>)` to stderr and returns. A test never fails because there is no adapter, and never skips without that message. CPU tests never need a GPU.
- Code generates the test geometry and test assets. JSON text fixtures are allowed. Binary fixtures never are.

### Web build and demo pages
- `scripts/build-web.sh <crate-dir> [--features <list>]` does the following:
  1. Runs `cargo build --profile wasm-release --target wasm32-unknown-unknown -p <crate>`.
  2. Runs `wasm-bindgen --target web --out-dir dist/<dir-name>/pkg <wasm>`.
  3. Copies `<crate-dir>/web/*` to `dist/<dir-name>/` and `web/common/*` to `dist/<dir-name>/common/`.
  4. Writes `dist/<dir-name>/sizes.json` (S7) and prints the raw and gzip sizes.
- Every `demos/*` crate follows the demo crate pattern:
  - It is a cdylib exporting `#[wasm_bindgen] pub async fn start(canvas: web_sys::HtmlCanvasElement, options: JsValue) -> Result<Demo, JsValue>`.
  - It also exports `#[wasm_bindgen] impl Demo { pub fn frame(&mut self, now_ms: f64) -> Result<(), JsValue>; pub fn resize(&mut self, width: u32, height: u32); pub fn stats(&self) -> JsValue; pub async fn probe_non_empty(&mut self, now_ms: f64) -> Result<bool, JsValue> }`.
  - The demo is a `raa_web::WebApp` driven by a `WebRuntime`.
  - `web/index.html` and `web/main.js` import `./pkg/<crate_name>.js`, drive `requestAnimationFrame` and register `window.nmcGpu.ops` through `common/nmc-gpu.js`.
  - `web/nmc.gpu.json` follows nomorechat's GPU-client protocol (`docs/gpu-client.md` of nomorechat).
  - The optional feature `alloc-count` installs `CountingAlloc` and sets `RuntimeOptions.alloc_probe`.
- Local serving uses `node web/serve.mjs <dir> [--port N]`:
  - It binds to 127.0.0.1 only. Port 0 means an ephemeral port.
  - It sends correct MIME types, including `application/wasm`, and `Cache-Control: no-store`.
  - It prints `serving <dir> on http://127.0.0.1:<port>/`.

### Browser captures (GPU client only)
- The project has no GPU. Browser rendering goes only through nomorechat's GPU client, run as `node scripts/gpu-capture.mjs <name> [--input <gpu-input.json>]`. The script does the following:
  1. Runs `node "$NMC_GPU_CLIENT" publish dist/<name>/`.
  2. Runs `node "$NMC_GPU_CLIENT" render <input> --page-ref REF -o target/gpu-out/<name>/`.
  3. Writes the committed `proofs/results/<name>.json` (S6).
  4. Copies the image to `proofs/captures/<name>.png`.

  The `.gitignore` excludes the PNGs. They are review artefacts, and the JSON is the committed record.
- Linux Chrome is never launched inside the sandbox. `NMC_GPU_TOKEN` is never printed.
- If the client is unavailable, the task stops visibly and writes a dogfood entry. It never substitutes CPU output.
- Hardware labelling: `hardware = !isFallbackAdapter && none of /swiftshader|llvmpipe|lavapipe|software|basic render/i matches vendor, architecture, device or description`. Every report states in words whether the adapter is hardware or software.
- Timing:
  - `firstFrameMs` is the navigation-relative `performance.now()` at the moment `queue.onSubmittedWorkDone()` of frame 0 resolves.
  - `cpuMs` of a frame is the `performance.now()` delta around `Demo.frame` (update, record and submit).
  - Reports give the median and p95 over `frames` measured frames after `warmup` frames.

### Values (decided once, cited everywhere)
| name | value |
|---|---|
| WASM_GZIP_MAX | 300000 bytes per shipped .wasm (gzip level 9 via node zlib), raw size always reported next to it |
| FIRST_FRAME_MAX_MS | 500 |
| CPU_MS_MEDIAN_MAX | 4.0 ms median CPU per frame, 1000 instances, hardware WebGPU only |
| ALLOC_RUN | 1000 instances, 120 warm-up frames, 600 measured frames, allowed allocations = 0 |
| BROWSER_RUN | warmup 60, frames 300 |
| CAPTURE | canvas 1280x720, devicePixelRatio 1; resize cycle 1280x720 to 960x540 (30 frames) and back |
| CLEAR_COLOR | linear (0.05, 0.06, 0.08, 1.0) |
| QUALITY | `Quality::LOW` = 8, `Quality::LOD0` = 24, `Quality::HIGH` = 64 segments; any value clamped to [3, 256] |
| TRIANGLE_BUDGET_DEFAULT | 1500 triangles per asset at LOD0 |
| ASSET_CAPS | nodes <= 64; nesting level <= 8 (roots are level 0); quaternion norm within 1e-3 of 1; scale components > 0 |
| WELD_EPS | 1e-6 x max(1, bbox diagonal) (mesh analysis) |
| NORMAL_RULE | per vertex: abs(len - 1) <= 1e-4; angle to the area-weighted mean of incident face normals (same index topology) <= 45 degrees; inward count (normal dot face normal < 0) = 0 |
| SDF_REF_TOL | uniquePositions and triangles equal exactly; each bbox coordinate abs diff <= 1e-5 x max(1, diagonal); area and volume relative diff <= 1e-4 |
| CSG_EPS | 1e-6 x max(1, diagonal of host bbox union tool bbox); volume identity relative diff <= 1e-4 |

### Proofs
- There is one proof script per milestone: `proofs/mN-<name>.mjs`, built on `proofs/lib/proof.mjs`.
  - Run it with `/root/nmc/engines/bin/node proofs/mN-<name>.mjs [--quick|--full] [--only <regex>]`.
  - The default is `--quick`, the cheap subset that still guards the milestone.
  - A proof has at most 5 checks, covering core behaviour only.
- Each check prints one line: `CHECK <label> | measured: <value> | threshold: <threshold> | PASS|FAIL`.
  - The script exits non-zero at the first failing selected check.
  - `--only <regex>` selects checks by label and is forwarded to case-level runners (prims kinds, CSG cases).
- Independent cases run in parallel, up to `NMC_CPUS` processes (1 when unset), never more.
- Wall budget: a milestone proof runs in at most 3 minutes on a warm cargo cache, and every artefact is built once per run. `timeout_s` covers the cold first compile. A threshold is never lowered for it.
- Timing thresholds exist only for FIRST_FRAME_MAX_MS and CPU_MS_MEDIAN_MAX. They read the committed BrowserResult and are evaluated first and alone.
- Look and visual quality are review items. Proofs only check `nonEmpty` and that the capture record exists. The owner reviews `proofs/captures/<name>.png` against the Quality bar.
- A milestone closeout runs the build, the boot, its own proof and the project smoke test. Earlier proofs are runnable on demand.
- Dogfood: friction caused by nomorechat (GPU client, sandbox, tooling) goes in `docs/dogfood/mN.md`: what broke, what was done by hand and what nomorechat must do instead. The lead transfers it to `bench/kart/HANDOVER.md`, which is in another repository and never edited by tasks.

### Commits and scope
- One commit per task, never pushed. No binaries or asset files (`scripts/check-no-assets.sh`).
- A task edits only files of modules owned by its milestone. Fixing another module needs the lead's explicit authorisation and stays minimal.
- The public API (crate items, JS package, docs/API.md) never names a consumer game: no Royal Fiesta, no Monster Racer and no game concepts. `scripts/check.sh` enforces this on `crates/`, `demos/`, `tests/`, `packages/`, `examples/` and `docs/API.md`.

### Quality bar
Criteria controlled by commands:
- **Q1 Clean build:** `scripts/check.sh` exits 0. It covers fmt, clippy `-D warnings` on host and wasm32, all tests, every demo wasm build and the asset, SAFETY and neutrality guards.
- **Q2 Size:** every shipped .wasm has a gzip size <= WASM_GZIP_MAX. Raw and gzip sizes are reported in each milestone proof.
- **Q3 Startup and speed:** every BrowserResult has `firstFrameMs` <= FIRST_FRAME_MAX_MS. The 1000-instance scene has `cpuMsMedian` <= CPU_MS_MEDIAN_MAX with `hardware: true`. A software run is reported as such and does not pass the speed criterion.
- **Q4 Frame loop:** `raa-bench frame-allocs` reports 0 allocations over ALLOC_RUN.
- **Q5 Primitive meshes:** this covers the 8 kinds at LOW, LOD0 and HIGH on both reference parameter sets. Each mesh:
  - is closed with consistent winding and positive volume;
  - satisfies NORMAL_RULE;
  - hashes byte-identically across two runs;
  - matches the SDF Modeler reference within SDF_REF_TOL (48 cases, capsule included).
- **Q6 Assets:** canonical prims v0 text round-trips byte-identical. Every invalid case is refused with its expected code and a non-empty message. The triangle budget is enforced.
- **Q7 Booleans:** every supported CSG case is watertight and satisfies the volume identity within CSG_EPS. Every unsupported case returns its expected typed refusal code and message.
- **Q8 Extensibility and API:** the extension test crate adds a geometry provider and a render pass with no other crate depending on it and no edit to other crates (cargo metadata check). `docs/API.md` covers every export of `packages/raa/index.d.ts`. The neutrality guard passes.
- **Q9 Honest GPU labelling:** every BrowserResult names the adapter and states `hardware` true or false.

Review criteria. The owner answers yes or no on each saved capture:
- **R1 (M1):** a single solid triangle is clearly visible and centred on the clear colour, with no garbage, stripes or tearing.
- **R2 (M2):** the 1000 instances are visibly lit by one directional light:
  - lit and unlit sides are distinguishable on each object, and ambient light keeps dark sides readable;
  - occlusion is correct;
  - flat and PBR-lite materials are distinguishable, with PBR-lite showing a specular highlight;
  - nothing is stretched after the resize cycle.
- **R3 (M3):** on the contact sheet, each of the 8 kinds is recognisable at every quality. Shading is smooth, with no cracks, holes, dark inverted faces or seam lines. Higher quality is visibly smoother.
- **R4 (M4):** the code-generated assets look like their declared composition, with parts in place and declared colours.
- **R5 (M5):** boolean results show flat, clean cut faces with dents and holes where expected, no slivers and no see-through gaps.
- **R6 (M6):** the example page shows a loaded primitive asset and instanced primitives, rendered through the `@etcito/raa` package only.

## S1. GPU layouts
All layouts are little-endian, written with `to_le_bytes`, with no unsafe code.

- **Vertex** (stride 24 B):
  - `position: f32x3` at offset 0 (location 0) and `normal: f32x3` at offset 12 (location 1).
  - Indices are `u32`, triangle list.
- **Instance** (stride 128 B, `raa_render::INSTANCE_STRIDE`; vertex buffer with step mode Instance):
  - `model: 4 x f32x4` (columns, locations 2-5) at offset 0.
  - `normal_matrix: 3 x f32x4` at offset 64 (locations 6-8). These are the columns of the inverse-transpose of the upper 3x3, with w = 0.
  - `material_index: u32` at offset 112 (location 9), followed by 12 B of padding.
- **MaterialGpu** (48 B): element of a read-only storage buffer `array<MaterialGpu>` at group 0, binding 1.
  - `base_color: vec4<f32>`: linear rgb, a = 1.
  - `emissive: vec4<f32>`: linear rgb, w = 0.
  - `params: vec4<f32>` = (roughness clamped to [0.04, 1], metallic in [0, 1], model where 0 = Flat and 1 = PbrLite, 0).
- **FrameUniform** (144 B, `raa_render::FRAME_UNIFORM_SIZE`): uniform at group 0, binding 0.
  - `view_proj: mat4x4<f32>` at offset 0.
  - `camera_pos: vec4<f32>` at offset 64.
  - `light_dir: vec4<f32>` at offset 80: the normalised travel direction of the light, w = 0.
  - `light_color: vec4<f32>` at offset 96: rgb x intensity.
  - `ambient: vec4<f32>` at offset 112: rgb x intensity.
  - `flags: vec4<u32>` at offset 128: x = 1 when the shader must encode linear to sRGB.
- **Shading:**
  - Flat is unlit: `base_color + emissive`.
  - PbrLite = `ambient * base * (1 - 0.5 * metallic) + (Lambert diffuse * (1 - metallic) + GGX specular with Schlick Fresnel, F0 = mix(0.04, base, metallic), Smith-Schlick G) * light_color * max(dot(n, -light_dir), 0) + emissive`.
- Integer and flat varyings are declared `@interpolate(flat, either)`. This is valid in both core and compatibility modes, so pipelines can be created on compatibility-level adapters.

## S2. Mesh semantics (`raa_geom::MeshData`)
- A mesh is an indexed triangle list in the mesh's own space.
- `positions.len() == normals.len()`, `indices.len() % 3 == 0`, all indices are in range and all values are finite.
- Hard edges (caps, box faces) duplicate vertices. Smooth surfaces share them.
- Closedness is judged on the position-welded topology (WELD_EPS): every undirected edge is used by exactly 2 triangles, in opposite directions.
- Signed volume (divergence theorem) > 0 means outward winding.

## S3. Primitive kinds and parameters
This is RAA's reading of prims v0. The names are the JSON `params` keys. All primitives are centred at the origin, and axial kinds use the Y axis.

| kind | params | rules |
|---|---|---|
| sphere | `radius: f` | radius > 0 |
| box | `halfExtents: [f,f,f]` | each > 0 |
| roundBox | `halfExtents: [f,f,f]`, `radius: f` | radius > 0 and radius <= min(halfExtents) |
| capsule | `radius: f`, `halfHeight: f` | radius > 0, halfHeight >= 0 (segment from -halfHeight to +halfHeight; bounds [radius, halfHeight + radius, radius]) |
| cylinder | `radius: f`, `halfHeight: f` | both > 0 |
| cone | `radius: f`, `halfHeight: f` | both > 0; base of radius `radius` at y = -halfHeight, apex at +halfHeight |
| torus | `majorRadius: f`, `minorRadius: f` | 0 < minorRadius < majorRadius; ring in the XZ plane |
| ellipsoid | `radii: [f,f,f]` | each > 0 |

- Every kind except torus is convex.
- Tessellation ports SDF Modeler's `leafGeometry` scheme. The count rules are in S8.
- `crates/raa-prims/README.md` documents the per-kind mapping from `Quality.segments` to rings and segments, and the triangle-count formula.

Reference parameter sets, used by the fixture, the contact sheet and the CSG cases:
- **Set A:** sphere r 0.8; box [0.7,0.6,0.5]; roundBox [0.7,0.6,0.5] r 0.15; capsule r 0.4 hh 0.5; cylinder r 0.6 hh 0.7; cone r 0.7 hh 0.8; torus R 0.6 r 0.22; ellipsoid [0.9,0.6,0.5].
- **Set B:** sphere r 1.7; box [1.2,0.3,2.0]; roundBox [1.0,1.0,0.4] r 0.35; capsule r 0.25 hh 1.2; cylinder r 1.5 hh 0.2; cone r 0.3 hh 1.5; torus R 1.5 r 0.1; ellipsoid [0.3,1.4,0.8].

## S4. Primitive asset format v0
The format is owned by raa-assets-tools. RAA keeps its own parser in raa-loader. Shared conformance fixtures reconcile any difference, and the fix lands in raa-loader.

**Top-level object.** Keys in canonical order:
- `format`: `raa.prims`.
- `version`: 0.
- `id`: string.
- `name`: string.
- `triangleBudget`: optional positive integer, default TRIANGLE_BUDGET_DEFAULT.
- `nodes`: array, required.
- `materials`: object, required, may be empty.
- `anchors`: object, required, may be empty.
- `collision`: optional. Absent means derived.

**Node.** Keys in order:
- `id`: unique in the asset, `[A-Za-z0-9_-]+`.
- `kind`.
- `params`: keys in the S3 order for the kind, all required, no extras.
- `transform`: optional. When present it is `{t:[3], r:[4], s:[3]}` with all three keys required.
- `material`: optional. Absent means inherit from the parent. The root default is linear grey 0.5, roughness 0.6, metallic 0.
- `op`: optional. `union` is the default and is omitted in canonical form. The other values are `subtract` and `intersect`.
- `children`: optional, omitted when empty.

A subtract or intersect node may have children. They are parsed and validated like any node, but they add no mesh, no collider and no triangle estimate. Refusing such a tool is the BooleanResolver's decision.

**Material.** Keys in order: `color: [r,g,b]` (sRGB [0,1]), `roughness` ([0,1]), `metallic` ([0,1]), `emissive: [r,g,b]` (linear, >= 0). Always PbrLite.

**Anchor.** A name mapped to `{t, r, s}`.

**`collision` override:**
- Shape: `{"colliders": [{"node": id (optional), "shape": "ball"|"cuboid"|"roundCuboid"|"capsuleY"|"cylinderY"|"coneY", "params": {...}, "transform": {t,r,s}}]}`.
- Params per shape: ball takes `radius`. cuboid takes `halfExtents`. roundCuboid takes `halfExtents` and `borderRadius`. capsuleY, cylinderY and coneY take `halfHeight` and `radius`.

**Derived colliders.** One collider per union node, in asset space:
- sphere becomes ball, box becomes cuboid, roundBox becomes roundCuboid, capsule becomes capsuleY, cylinder becomes cylinderY and cone becomes coneY.
- When the world scale is non-uniform, or the kind is ellipsoid, the collider is a convexHull of the node's LOW mesh points.
- A torus without an override is refused with `collision.torus_needs_override`.
- Subtract and intersect children add no collider; the host's collider is used.

**Semantics:**
- Root nodes and `union` children each become one separate closed mesh, positioned by the composed transforms.
- `subtract` and `intersect` children modify their parent (the host). The tool is expressed in the host's local space.
- Boolean evaluation is delegated to a `BooleanResolver`.

**Canonical text:**
- The equivalent of `JSON.stringify(value, null, 2)` plus a final LF.
- Object keys follow the orders above. `materials` and `anchors` keys are sorted bytewise.
- Numbers use JavaScript Number-to-string rules: shortest round-trip, integers without `.0`, exponent form (`1e-7`, `1e+21`) outside [1e-6, 1e21), and `-0` written as `0`.
- Canonical input round-trips byte-identical. Non-canonical input is canonicalised idempotently.

**Caps.** ASSET_CAPS plus `triangleBudget`, which bounds the total generated triangles after booleans at the load quality. Unknown keys, non-finite numbers, duplicate ids, unknown materials and unknown kinds are refused.

## S5. LoadError codes (raa-loader)
`json.syntax`, `json.duplicate_key`, `json.depth`, `format.unknown`, `format.version`, `schema.missing_field`, `schema.unknown_field`, `schema.type`, `value.not_finite`, `value.range`, `id.invalid`, `id.duplicate`, `material.unknown`, `kind.unknown`, `kind.params`, `cap.nodes`, `cap.depth`, `cap.budget`, `collision.torus_needs_override`, `collision.shape`, `boolean.unsupported`, `boolean.<csg code>` (the CsgError code with a prefix), `loader.unknown_format`, `loader.duplicate_format`.

`path` is a JSON pointer, for example `/nodes/2/children/0/params/radius`. A resolver error with an empty path gets the first tool's JSON pointer.

## S6. BrowserResult (`proofs/results/<name>.json`)
`scripts/gpu-capture.mjs` writes this file from the page op result:

`{"name", "milestone", "capturedAt" (ISO), "adapter": {"vendor", "architecture", "device", "description", "isFallbackAdapter"}, "hardware": bool, "canvas": [w,h], "nonEmpty": bool, "firstFrameMs": number, "warmup", "frames", "cpuMsMedian", "cpuMsP95", "instances", "drawCalls", "triangles", "allocsPerFrameTotal": number|null, "resize": {"to": [w,h], "back": [w,h], "ok": bool}|null, "wasm": {"raw", "gzip", "sha256"}, "capture": "proofs/captures/<name>.png", "captureSha256"}`

Page op contract:
- `window.nmcGpu.ops.run(args)` takes `args = {width, height, warmup, frames, resize: bool, ...demo args}`.
- It resolves to the fields above except `name`, `milestone`, `capturedAt`, `wasm` and `capture*`, which the script adds.
- The client takes the screenshot after `run` resolves, at CAPTURE size.

## S7. sizes.json (`dist/<name>/sizes.json`, `dist/raa-sizes.json`)
`{"wasm": "pkg/<crate>_bg.wasm", "raw": bytes, "gzip": bytes, "sha256": hex}`.

## S8. SDF reference fixture and mapping v1
`tools/sdf-ref/gen-leaf-fixtures.mjs` produces the fixture `crates/raa-prims/tests/fixtures/sdf-leaf-reference.json`. `raa_prims::reference` reads it.

**Top-level keys.** `generatedBy`, `command`, `regenerate` (the exact command), `sdfModelerRevision` (the sdfm commit), `leafMeshSha256` (sha256 of the original `leaf-mesh.ts`), `axisMap`, `paramMap`, `mapping` (mapping v1, embedded verbatim from the README), `kinds` (per kind, the generation path used: `call` or `literal`) and `cases`. Required keys: `command`, `sdfModelerRevision`, `axisMap`, `paramMap`, `mapping` and `cases`.

**Legacy-shaped maps:**
- `axisMap` is `[{"axis": "x"|"y"|"z", "sign": 1|-1}] x3`. Entry i is the reference axis that becomes RAA axis i.
- `paramMap` is `{"<kind>": [{"ref", "refIndex": int|null, "raa": string|null, "index": int|null, "scale": number, "value": number|null, "add"?: {...}}]}`.
- In both maps, `refParams[ref][refIndex] = raa == null ? value : scale * params[raa][index] (+ add.scale * params[add.raa][add.index])`.

**Cases.** 48 distinct cases: 8 kinds x sets A/B x quality 8/24/64. Each case has the keys `kind`, `set`, `quality`, `params` (RAA names), `refParams` (`{name: number | number[]}`), `sdfInput` (the exact `leafGeometry` arguments, opaque), `counts`, `uniquePositions`, `triangles`, `bboxMin`, `bboxMax`, `area`, `volume` and `closed`.
- All metrics are in RAA's frame (after axisMap).
- uniquePositions counts positions deduplicated after rounding to 1e-6.
- Area and volume are computed in f64, and the volume must be > 0.

**mapping v1.** Its shape:

```json
{"version":1,
 "reference":{"repo":"sdfm","path":"src/core/hybrid/leaf-mesh.ts","commit":"<hex40>"},
 "axisMap":{"x":S,"y":S,"z":S,"det":1|-1},
 "paramMap":{"<S3 kind>":{
   "sdfKind":"<LEAF_KINDS entry>",
   "params":{"<sdf param>":{"from":"<S3 key>","component":"x"|"y"|"z"|null,"scale":n,
                            "add":{"from":"<S3 key>","component":...,"scale":n}}},
   "constants":{"<sdf param>":<value>},
   "segments":[{"count","ref","min","multiple","source"}]}}}
```

- **axisMap:** S is one of +x -x +y -y +z -z, and axisMap is a signed permutation (RAA axis from a signed SDF axis). It is applied to positions and normals, and index order is reversed when det = -1.
- **Params:**
  - A vec3 SDF parameter is written as three keys `<p>.x|.y|.z` (SDF-frame axis). Each takes a vector S3 key and its RAA-frame component, and each RAA component is used exactly once. Scalars keep their bare name with `component: null`.
  - Each S3 key is used exactly once. `constants` covers every SDF parameter without an S3 source.
- **Additive term:** a params entry may carry one optional `add`. It then evaluates as `sdf_value = scale * S3[from] + add.scale * S3[add.from]`, in f64 and in that order.
  - `add.from` is an S3 key of the kind and follows the same component rules.
  - `add.scale` is finite and non-zero, and `add` cannot be nested.
  - The `add` use is not counted by the each-key-once rule.
  - Capsule uses `"height": {"from":"halfHeight","component":null,"scale":2,"add":{"from":"radius","component":null,"scale":2}}`.
- **Segments:** `count(s) = max(min, multiple * round(ref * s / (24 * multiple)))`, rounding half away from zero, where s is the clamped QUALITY segments. LOD0 (24) therefore reproduces the reference defaults. Box has `segments: []` and is quality-independent.
- **Refusals:** `raa_prims::reference` refuses any malformed entry with `prims.mapping_invalid`:
  - version != 1, a commit that is not 40 lowercase hex characters, an axisMap that is not a signed permutation, or a det mismatch;
  - a missing or duplicate kind;
  - a null component on a vector `from`, a nested or invalid `add`, or an unknown key;
  - segments rules that differ from raa-prims' own COUNT_TABLE.

  A case whose `counts` differ from RAA's computed counts is refused with `prims.reference_counts_mismatch`, and a shape error with `prims.fixture_invalid`.

## S9. CSG cases table (`proofs/results/m5-csg-cases.json`)
`[{"id", "op", "host": {kind, params, transform}, "tools": [{op, kind, params, transform, hasChildren}], "status": "ok"|"refused", "triangles", "watertight", "volume", "volumeIdentityRelErr"|null, "code"|null, "message"|null, "expected": "ok"|<code>}]`

## Module ownership

| Module | Milestone | Files | Consumes |
|---|---|---|---|
| workspace | M1 | `Cargo.toml`, `Cargo.lock`, `.gitignore` | none |
| tooling | M1 | `scripts/check.sh`, `scripts/build-web.sh`, `scripts/check-no-assets.sh`, `scripts/hooks/pre-commit`, `scripts/gpu-capture.mjs`, `web/serve.mjs`, `web/common/nmc-gpu.js`, `web/common/run-loop.js`, `proofs/lib/proof.mjs` | workspace |
| raa-core | M1 | `crates/raa-core/**` | workspace |
| raa-web | M1 | `crates/raa-web/**` | raa-core |
| demo-m1-boot | M1 | `demos/m1-boot/**` | raa-core, raa-web, tooling |
| proof-m1 | M1 | `proofs/m1-boot.mjs`, `proofs/results/m1-boot.json`, `docs/dogfood/m1.md` | tooling, demo-m1-boot |
| raa-math | M2 | `crates/raa-math/**` | none |
| raa-geom | M2 | `crates/raa-geom/**` | raa-math |
| raa-material | M2 | `crates/raa-material/**` | raa-core |
| raa-scene | M2 | `crates/raa-scene/**` | raa-math, raa-core |
| raa-render | M2 | `crates/raa-render/**` | raa-core, raa-math, raa-geom, raa-material, raa-scene |
| raa-alloc-count | M2 | `crates/raa-alloc-count/**` | none |
| raa-bench | M2 | `crates/raa-bench/**` | raa-alloc-count, raa-scene, raa-render, raa-math, raa-core |
| demo-m2-scene | M2 | `demos/m2-scene/**` | raa-web, raa-render, raa-scene, raa-geom, raa-material, raa-alloc-count, tooling |
| proof-m2 | M2 | `proofs/m2-scene.mjs`, `proofs/results/m2-scene.json`, `docs/dogfood/m2.md` | tooling, raa-bench, demo-m2-scene |
| raa-prims | M3 | `crates/raa-prims/**` | raa-math, raa-geom |
| sdf-ref-tool | M3 | `tools/sdf-ref/gen-leaf-fixtures.mjs`, `tools/sdf-ref/README.md` | none |
| demo-m3-prims | M3 | `demos/m3-prims/**` | raa-web, raa-render, raa-scene, raa-prims, raa-material, tooling |
| proof-m3 | M3 | `proofs/m3-prims.mjs`, `proofs/results/m3-prims.json`, `proofs/results/m3-prims-table.json`, `docs/dogfood/m3.md` | tooling, raa-prims, demo-m3-prims |
| raa-loader | M4 | `crates/raa-loader/**` | raa-math, raa-geom, raa-material, raa-prims (dev) |
| demo-m4-asset | M4 | `demos/m4-asset/**` | raa-web, raa-render, raa-scene, raa-prims, raa-loader, tooling |
| proof-m4 | M4 | `proofs/m4-loader.mjs`, `proofs/results/m4-asset.json`, `docs/dogfood/m4.md` | tooling, raa-loader, demo-m4-asset |
| raa-csg | M5 | `crates/raa-csg/**` | raa-math, raa-geom, raa-loader, raa-prims (dev) |
| demo-m5-booleans | M5 | `demos/m5-booleans/**` | raa-web, raa-render, raa-scene, raa-prims, raa-loader, raa-csg, tooling |
| proof-m5 | M5 | `proofs/m5-booleans.mjs`, `proofs/results/m5-booleans.json`, `proofs/results/m5-csg-cases.json`, `docs/dogfood/m5.md` | tooling, raa-csg, demo-m5-booleans |
| raa-js | M6 | `crates/raa-js/**` | raa-core, raa-web, raa-math, raa-geom, raa-material, raa-scene, raa-render, raa-prims, raa-loader, raa-csg |
| js-package | M6 | `packages/raa/{package.json, index.js, index.d.ts, README.md}`, `scripts/build-package.sh` | raa-js |
| example-basic | M6 | `examples/basic/**` | js-package, tooling |
| raa-ext-test | M6 | `tests/raa-ext-test/**` | raa-core, raa-math, raa-geom, raa-render, raa-scene, raa-material, raa-loader, raa-prims |
| project-docs | M6 | `README.md`, `docs/API.md`, `docs/ARCHITECTURE-notes.md` | js-package, raa-js |
| proof-m6 | M6 | `proofs/m6-api.mjs`, `proofs/results/example-basic.json`, `proofs/results/m6-sizes.json`, `docs/dogfood/m6.md` | tooling, raa-ext-test, js-package, example-basic, project-docs |

## Module: workspace
Files: `Cargo.toml`, `Cargo.lock`, `.gitignore`. Owner: M1.

- `[workspace] resolver = "2"`, `members = ["crates/*", "demos/*", "tests/*"]`.
- `[workspace.package]`: version 0.1.0, edition 2024, license MIT, rust-version 1.96.0.
- `[workspace.dependencies]` holds the closed dependency list (Conventions). It also lists every internal library crate by path (raa-core, raa-web, raa-math, raa-geom, raa-material, raa-scene, raa-render, raa-alloc-count, raa-prims, raa-loader, raa-csg), so crates reference them with `{ workspace = true }`.
- `[profile.wasm-release]` follows Conventions.
- `Cargo.lock` is committed. `.gitignore` is unchanged.

## Module: tooling
Files: `scripts/check.sh`, `scripts/build-web.sh`, `scripts/check-no-assets.sh`, `scripts/hooks/pre-commit`, `scripts/gpu-capture.mjs`, `web/serve.mjs`, `web/common/nmc-gpu.js`, `web/common/run-loop.js`, `proofs/lib/proof.mjs`. Owner: M1.

**`scripts/check.sh [--quick]`** is bash with `set -euo pipefail`. It prints `== <step>` per step and exits non-zero on the first failure. Steps:
1. `cargo fmt --all --check`.
2. `cargo clippy --workspace --all-targets -- -D warnings`.
3. `cargo clippy --workspace --lib --target wasm32-unknown-unknown -- -D warnings`.
4. `cargo test --workspace` (skipped by `--quick`).
5. `scripts/build-web.sh` for every directory in `demos/*`. The directories are discovered, never listed.
6. `scripts/check-no-assets.sh`.
7. SAFETY guard: every Rust line under crates/, demos/ and tests/ containing the token `unsafe` has `// SAFETY: ... (test: NAME)` directly above, and `fn NAME` exists in the workspace.
8. Neutrality guard: no case-insensitive match of `royal|fiesta|monster|racer` in crates/, demos/, tests/, packages/, examples/ or docs/API.md.

Before step 1, the script runs `cargo metadata --no-deps --format-version 1`. With zero packages, it prints `check.sh: workspace has no member crates yet; skipping cargo fmt/check/clippy/test`, runs the non-cargo checks and exits 0. Otherwise every cargo step is fatal. A missing rustfmt or clippy component is a failure that names the component.

**`scripts/build-web.sh <crate-dir> [--features <list>]`** follows Conventions. Output goes to `dist/<basename>/`, including `sizes.json` (S7).

**`scripts/check-no-assets.sh`** fails when `git ls-files` contains a blocked extension or a file larger than 1 MB, and prints each offender. With `--staged` it checks staged files. Blocked extensions: glb, gltf, png, jpg, jpeg, avif, ktx2, bin, wasm, hdr, exr. `scripts/hooks/pre-commit` runs it with `--staged`. Install the hook with `git config core.hooksPath scripts/hooks`.

**`scripts/gpu-capture.mjs <name> [--input <path>] [--milestone mN]`** follows Conventions (Browser captures).
- The default input is `demos/<name>/gpu-input.json`. The script reads `dist/<name>/sizes.json`.
- It writes `proofs/results/<name>.json` (S6) and `proofs/captures/<name>.png`.
- When `NMC_GPU_CLIENT` is unset or the client fails, it exits non-zero with a message and never fakes a result.

**`web/serve.mjs <dir> [--port N]`** is a static server per Conventions. It accepts `GET` only and allows no traversal outside `<dir>`.

**`web/common/nmc-gpu.js`** exports `registerNmcGpu({ run })`. It sets `window.nmcGpu = { ops: { run } }` and `window.nmcGpu.ready = true`.

**`web/common/run-loop.js`** exports `async function runMeasured(demo, canvas, args)`. It does the following:
1. Sets the canvas to `args.width` x `args.height` at dpr 1.
2. Runs `args.warmup + args.frames` frames through `requestAnimationFrame`, calling `demo.frame(now)` and measuring cpuMs.
3. Performs the CAPTURE resize cycle when `args.resize` is set.
4. Calls `demo.probe_non_empty(now)`.
5. Returns the merged S6 page fields, with `demo.stats()` supplying adapter, hardware, firstFrameMs, instances, drawCalls, triangles and allocsPerFrameTotal.

**`proofs/lib/proof.mjs`** exports:
- `parseArgs(argv) -> { quick, full, only: RegExp|null }`.
- `async check(label, fn)`: `fn` returns `{ measured, threshold, ok }`. It prints the CHECK line, skips when `only` does not match and exits 1 on failure.
- `run(cmd, args, opts?) -> { code, stdout, stderr }`: spawnSync with no shell and the inherited environment.
- `async pool(items, worker)`: concurrency `NMC_CPUS`, default 1.
- `readJson(path)`.
- `gzipSize(path)`: zlib level 9.
- `browserResultCheck(name, { requireHardware, maxCpuMs, instances })`: validates S6, meaning nonEmpty, adapter present, hardware stated and firstFrameMs <= FIRST_FRAME_MAX_MS, plus the optional speed rule.
- `VALUES`: the Values table numbers.

## Module: raa-core
Files: `crates/raa-core/{Cargo.toml, README.md}`, `src/{lib, error, gpu, surface, handle, graph, clear, offscreen, testing}.rs`, `tests/{handles, graph, gpu_readback}.rs`. Owner: M1.

raa-core holds the device, surface, handles, a minimal frame graph and the error type. It depends on wgpu, plus web-sys and wasm-bindgen-futures on wasm32 and pollster on native.

**`pub use wgpu;`** re-exports the exact wgpu version raa-core uses. With it, extension crates implement RenderPass while depending only on raa-* crates.

**`pub enum RaaError`**
- Variants: `NoAdapter(String)`, `RequestDevice(String)`, `Surface(String)`, `FrameGraph { code: &'static str, message: String }`, `Resource(String)`, `Gpu(String)`, `Unsupported(String)`.
- `code()` returns `core.no_adapter`, `core.request_device`, `core.surface`, the FrameGraph code, `core.resource`, `core.gpu` or `core.unsupported`.
- Implements Display and Error.

**Adapter and GPU context**
- `pub struct AdapterSummary { vendor, architecture, device, description, backend: String, is_fallback: bool }` with `is_hardware()`, which applies the hardware rule.
- `pub struct GpuOptions { power_preference: wgpu::PowerPreference, allow_software: bool }`. Default: HighPerformance with allow_software true.
- `pub struct GpuContext` provides `device()`, `queue()`, `adapter()` and `summary()`.
  - wasm32: `async for_canvas(canvas, &GpuOptions) -> Result<(GpuContext, CanvasSurface), RaaError>`. WebGPU only; a missing `navigator.gpu` gives `core.no_adapter`.
  - native: `async headless(&GpuOptions) -> Result<GpuContext, RaaError>`. When no hardware adapter exists and allow_software is set, it retries with `force_fallback_adapter = true`.

**Surface (wasm32)**
- `CanvasSurface` provides `format()`, `is_srgb()`, `size()` and `resize(gpu, w, h)` (zero sizes clamp to 1).
- `acquire(gpu) -> Result<SurfaceFrame, RaaError>` reconfigures and retries once on Outdated or Lost; otherwise it fails with `core.surface`. Surface usage is RENDER_ATTACHMENT | COPY_SRC.
- `SurfaceFrame { pub view }` provides `texture()` and `present(self)`.

**Handles and ids**
- `Handle<T>` (Copy, Eq, Hash, Debug) has `index()` and `generation()`.
- `Pool<T>` provides `new`, `with_capacity`, `insert`, `get`, `get_mut`, `remove` (bumps the generation, so stale handles return None), `len` and `iter` (index order). Free slots are reused LIFO, deterministically.
- `MeshId(pub u32)` and `MaterialId(pub u32)` are engine-wide ids (Copy, Eq, Ord, Hash, Debug).
- `pub const CLEAR_COLOR: wgpu::Color`.

**Frame graph types** (module `graph`, re-exported)
- `ResourceId`. `BACKBUFFER` (id 0) is imported each frame.
- `TargetSize { Surface, Fixed(u32, u32) }` and `TargetDesc { label, format, size, usage }`.
- `PassId`. `PassBuilder` provides `read(r)` and `write(r)`.
- `trait RenderPass { name; declare(&self, &mut PassBuilder); prepare(&mut self, &PrepareContext) (default Ok); record(&mut self, &mut RecordContext); resize(&mut self, &GpuContext, (u32, u32)) (default no-op) }`.
- `PrepareContext { gpu, size, frame }`.
- `RecordContext { gpu, encoder, size, frame }` provides `view(r)`, `format(r)` and `is_first_write(r)`. A pass clears its target when `is_first_write` is true and loads it otherwise.

**`FrameGraph` API**
- `new(backbuffer_format)` and `create_target(desc) -> ResourceId`.
- `add_pass(pass)` appends a pass. `add_pass_after(name, pass)` and `add_pass_before(name, pass)` give `framegraph.unknown_pass` for an unknown name.
- `compile()`, `pass_names()` and `execute(gpu, backbuffer_view, size) -> Result<wgpu::CommandBuffer, RaaError>`.

**Compile rules**, checked in order:
1. Pass names are unique (`framegraph.duplicate_pass`).
2. Every read and write targets a known resource (`framegraph.unknown_resource`).
3. Every read of a non-BACKBUFFER target has a write in an earlier pass (`framegraph.read_before_write`).
4. At least one pass writes BACKBUFFER (`framegraph.no_output`).

Compiling caches the order and the first-write flags. Adding a pass invalidates compilation, and executing before compiling gives `framegraph.not_compiled`.

**Execution.** Execute recreates Surface-sized targets only when the size changes, and calls `resize` on the passes then. It calls prepare on all passes, then record in order, into one encoder. RAA code makes no allocation after the first frame at a given size.

**Other items**
- `ClearPass::new(name, target, color)` writes its target with LoadOp::Clear.
- Native `OffscreenTarget` provides `new(gpu, w, h, format)`, `view()` and `read_rgba8(gpu)`, which handles the 256-byte row padding.
- `non_empty(pixels_rgba8, clear_rgba8) -> bool` is true when any channel of any pixel differs by more than 2.
- Native `testing`: `software_gpu() -> Option<GpuContext>` and `skip(test, reason)`, which prints the SKIP line.

Tests: handle generations and reuse, every graph compile error code, add_pass_after/before ordering, and a ClearPass into an OffscreenTarget reading back CLEAR_COLOR (with SKIP when there is no adapter).

## Module: raa-web
Files: `crates/raa-web/{Cargo.toml, README.md}`, `src/{lib, runtime, stats, js}.rs`. Owner: M1.

raa-web is the web runtime used by every demo and by raa-js.

**Types**
- `FrameInput { frame: u64, now_ms: f64, dt_ms: f64, size: (u32, u32) }`.
- `trait WebApp`:
  - `init(&mut self, gpu, graph: &mut FrameGraph, size, surface_format, surface_is_srgb) -> Result<(), RaaError>`;
  - `update(&mut self, gpu, &FrameInput) -> Result<FrameCounters, RaaError>`;
  - `resize(&mut self, gpu, size)` (default no-op).
- `FrameCounters { instances: u32, draw_calls: u32, triangles: u64 }`.
- `RuntimeOptions { width, height, gpu: GpuOptions, stats_window: usize (default 300), alloc_probe: Option<fn() -> u64> }`.
- `FrameStats { frames, first_frame_ms: Option<f64>, cpu_ms_median, cpu_ms_p95, counters, allocs_last_frame: Option<u64> }`.

**`WebRuntime`** (wasm32)
- `async new(canvas, app: Box<dyn WebApp>, opts)` creates the GpuContext, CanvasSurface and FrameGraph, calls `app.init` and compiles the graph.
- `frame(now_ms)` runs `app.update`, acquire, `graph.execute`, submit and present. It records CPU ms in a preallocated ring. On frame 0 it registers `on_submitted_work_done`, which stores `performance.now()` into an `Rc<Cell<Option<f64>>>`.
- `resize(w, h)`.
- `stats()` computes median and p95 through a preallocated scratch buffer.
- Accessors: `adapter()`, `gpu()`, `app_mut()`.
- `async probe_non_empty(now_ms)` renders one frame, copies the backbuffer to a mapped buffer and applies `raa_core::non_empty` against CLEAR_COLOR.

**JS conversion**
- `stats_to_js(&FrameStats, &AdapterSummary) -> JsValue` builds a plain object with adapter{vendor, architecture, device, description, isFallbackAdapter}, hardware, firstFrameMs, frames, cpuMsMedian, cpuMsP95, instances, drawCalls, triangles and allocsPerFrameTotal. It uses js_sys::Reflect, no serde.
- `error_to_js(&RaaError) -> JsValue` creates a JS `Error` named `RaaError` with a `code` property.

On native, the crate compiles to the stats module only: FrameStats and percentile helpers, unit-tested.

## Module: demo-m1-boot
Files: `demos/m1-boot/{Cargo.toml, src/lib.rs, src/triangle.wgsl, web/index.html, web/main.js, web/nmc.gpu.json, gpu-input.json}`. Owner: M1.

Follows the demo crate pattern. The WebApp has a ClearPass named `clear` (CLEAR_COLOR), followed by a `triangle` pass that draws one solid orange triangle centred on the canvas. The shader generates the clip-space vertices from `vertex_index`; there are no buffers. The `run` op follows S6 (instances 1, drawCalls 1, triangles 1).

## Module: proof-m1
Files: `proofs/m1-boot.mjs`, `proofs/results/m1-boot.json`, `docs/dogfood/m1.md`. Owner: M1.

Checks:
- `check-sh`: `scripts/check.sh` exits 0.
- `wasm-size`: dist/m1-boot gzip vs WASM_GZIP_MAX, raw size shown.
- `serve`: `web/serve.mjs` on port 0 serves index.html, main.js, pkg/*.js, the wasm as application/wasm and nmc.gpu.json, all with 200.
- `browser-run`: m1-boot.json is nonEmpty, names the adapter, states hardware and has firstFrameMs <= FIRST_FRAME_MAX_MS.

The task produces the browser result with `scripts/gpu-capture.mjs` and commits it.

## Module: raa-math
Files: `crates/raa-math/{Cargo.toml, README.md}`, `src/{lib, vec, quat, mat4, aabb}.rs`, `tests/math.rs`. Owner: M2.

No dependencies. Types are f32 with `#[derive(Copy, Clone, Debug, PartialEq, Default)]`.

- **`Vec3 { x, y, z }`:**
  - `new`, `splat`, consts `ZERO`, `ONE`, `X`, `Y`, `Z`;
  - `dot`, `cross`, `length`, `length_squared`, `normalize_or_zero`, `min`, `max`, `abs`, `lerp`;
  - `to_array`, `from_array`, `is_finite`;
  - ops Add, Sub, Neg, Mul<f32>, Div<f32>, Mul<Vec3> (component-wise), AddAssign, SubAssign.
- **`Vec4 { x, y, z, w }`:** `new`, `from_vec3(v, w)`, `xyz`, `dot`, `to_array`, ops Add, Sub, Mul<f32>.
- **`Quat { x, y, z, w }`:**
  - `IDENTITY`, `from_xyzw`, `from_array`, `to_array`, `from_axis_angle`, `from_rotation_y`;
  - `mul` (Hamilton, plus `impl Mul`), `rotate`, `normalize`, `length`, `conjugate`, `slerp`.
- **`Mat4 { cols: [[f32; 4]; 4] }`:**
  - constructors: `IDENTITY`, `from_cols`, `from_translation`, `from_scale`, `from_quat`, `from_trs` (= T * R * S);
  - products: `mul` (plus `impl Mul`), `mul_vec4`;
  - transforms: `transform_point` (w = 1, divides when w != 1), `transform_vector` (w = 0);
  - `transpose`, `determinant`, `inverse() -> Option<Mat4>` (None when abs(det) < 1e-12);
  - `normal_matrix() -> [[f32; 4]; 3]` (inverse-transpose of the upper 3x3, identity when singular);
  - cameras: `perspective_rh_zo(fov_y, aspect, near, far)`, `look_at_rh(eye, target, up)`;
  - output: `to_cols_array`, `write_le_bytes(out)` (64 B).
- **`Aabb { min, max }`:** `EMPTY` (min +inf, max -inf), `from_points`, `include`, `union`, `intersects`, `center`, `extent`, `diagonal`, `transform(&Mat4)` (8 corners).
- **`approx_eq(a, b, eps)`.**

Tests: TRS composition, inverse round trip, quaternion rotate vs matrix, perspective near maps to 0 and far to 1, look_at gives -Z forward.

## Module: raa-geom
Files: `crates/raa-geom/{Cargo.toml, README.md}`, `src/{lib, mesh, params, provider, analysis, builtin}.rs`, `tests/{analysis, registry}.rs`. Owner: M2.

raa-geom holds the geometry provider trait, mesh data and mesh analysis. It depends on raa-math.

**`MeshData { positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>, indices: Vec<u32> }`** (S2)
- `new`, `clear` (keeps capacity), `vertex_count`, `triangle_count`.
- `validate()` fails with `geom.invalid_mesh`.
- `append_transformed(&other, &m)` maps positions by m and normals by the normal matrix (renormalised), and flips winding when det < 0.
- `bounds() -> Aabb`, `write_vertex_bytes(&mut Vec<u8>)` (S1), `write_index_bytes`.

**Parameters**
- `ParamValue { Scalar(f32), Vec3([f32; 3]) }`.
- `Params` keeps insertion order. It provides `new`, `with`, `set`, `get`, `scalar(kind, name)` and `vec3(kind, name)`. A missing parameter gives `geom.missing_param` and a wrong type `geom.invalid_param`. `names()` lists the keys.
- `Quality { segments: u32 }` has `LOW`, `LOD0` and `HIGH`. `new(segments)` clamps to [3, 256].

**Providers**
- `trait GeometryProvider { kind; param_names; is_convex; triangle_count(&Params, Quality) -> Result<u32, GeomError>; generate(&Params, Quality, &mut MeshData) -> Result<(), GeomError> }`.
  - `generate` clears `out` first, is deterministic (byte-identical output) and validates parameters (`geom.invalid_param` naming the param and the rule).
  - It emits a closed mesh with outward winding.
  - `triangle_count` equals the generated count without generating.
- `GeometryRegistry` provides `new`, `register` (`geom.duplicate_kind`), `get`, `kinds` (registration order) and `generate` (`geom.unknown_kind`).

**`GeomError`**
- Variants: `UnknownKind`, `MissingParam { kind, name }`, `InvalidParam { kind, name, reason }`, `DuplicateKind`, `InvalidMesh`.
- Codes: `geom.unknown_kind`, `geom.missing_param`, `geom.invalid_param`, `geom.duplicate_kind`, `geom.invalid_mesh`.

**`analysis`**
- `MeshReport { vertices, unique_positions, triangles, closed, boundary_edges, nonmanifold_edges, inconsistent_edges, degenerate_triangles, bbox_min, bbox_max, area: f64, volume: f64, max_normal_deviation_deg, bad_normal_lengths, inward_normals }`.
- `analyze(&MeshData)`:
  - welds deterministically with WELD_EPS through a quantised grid;
  - sets closed when the boundary, nonmanifold and inconsistent counts are all 0;
  - computes area and volume in f64;
  - applies NORMAL_RULE.
- `passes(&MeshReport) -> Result<(), String>` requires closed, volume > 0, no degenerate triangles and NORMAL_RULE. The message lists every failed rule with its numbers.
- `mesh_hash(&MeshData) -> u64`: FNV-1a 64 over the little-endian bytes of positions, normals and indices.
- `unique_positions_1e6(&MeshData) -> u32`: the S8 definition.

**`builtin`**: `DebugCube`, kind `debug_cube`, param `half > 0`. It has 24 vertices and 12 triangles with hard edges, and is convex.

Tests:
- analysis catches a missing triangle, a flipped triangle and inverted normals;
- DebugCube passes;
- registry duplicate and unknown kind codes.

## Module: raa-material
Files: `crates/raa-material/{Cargo.toml, README.md}`, `src/{lib.rs, shading.wgsl}`, `tests/material.rs`. Owner: M2.

Depends on raa-core and re-exports `MaterialId`.

- **`ShadingModel { Flat, PbrLite }`.**
- **`Material { base_color: [f32; 3] (linear), roughness, metallic, emissive: [f32; 3] (linear), model }`:**
  - constructors: `flat(c)`, `pbr(c, roughness, metallic)`, `from_srgb(c_srgb, roughness, metallic, emissive_linear)` (PbrLite);
  - `validate()` requires finite values, colour channels >= 0, and roughness and metallic in [0, 1]. Failures give `material.invalid` naming the field;
  - `write_gpu(out)` writes 48 B (S1).
  - Default: linear grey 0.5, roughness 0.6, metallic 0, PbrLite.
- **`srgb_to_linear(c)`:** the exact piecewise sRGB EOTF.
- **`MaterialLibrary`:**
  - `new`, `add` (dense ids from 0), `set` (`material.unknown_id`), `get`, `len`;
  - `is_dirty`, and `write_gpu_all(&mut Vec<u8>)`, which clears the dirty flag.
- **`MaterialError { code, message }`.**
- **`WGSL_SHADING`:** WGSL `struct MaterialGpu`, `struct FrameUniform` (S1) and `fn raa_shade(m, n, v, f) -> vec3<f32>`, which implements S1 shading.

## Module: raa-scene
Files: `crates/raa-scene/{Cargo.toml, README.md}`, `src/{lib, transform, graph, camera, light, render_list}.rs`, `tests/{hierarchy, camera, render_list}.rs`. Owner: M2.

Depends on raa-math and raa-core.

**Transforms and ids**
- `Transform { translation, rotation, scale }` has `IDENTITY`, `from_trs` and `to_mat4`.
- `NodeId` and `InstanceId` are generational (Copy, Eq, Hash, Debug) and provide `to_bits() -> u64`.

**Camera**
- `Camera { eye, target, up, fov_y_rad, near, far, aspect }`. Default: eye (0, 2, 6), target 0, up Y, fov 60°, near 0.1, far 200, aspect 16/9.
- Methods: `view`, `projection` (perspective_rh_zo), `view_proj` and `set_viewport(w, h)`.
- `OrbitController { target, distance, yaw_rad, pitch_rad, min_distance, max_distance }`:
  - `new(target, distance)`;
  - `rotate` clamps pitch to ±89°;
  - `zoom` multiplies the distance by a factor, clamped;
  - `pan` moves in the camera plane, scaled by distance;
  - `apply(&mut Camera)` sets `eye = target + distance * (cos p sin y, sin p, cos p cos y)`.

**Lights**
- `DirectionalLight { direction, color, intensity }`. Default: normalize(-0.4, -1, -0.3), white, 3.0.
- `Ambient { color, intensity }`. Default: (0.6, 0.65, 0.75), 0.3.

**Render list**
- `RenderItem { mesh, material, world }` and `Batch { mesh, first, count }`.
- `RenderList { items, batches }` provides `with_capacity` and `clear`.

**`Scene { pub camera, pub light, pub ambient, .. }`**
- Construction: `new`, `with_capacity(nodes, instances)`.
- Nodes: `add_node(parent, local)`, `remove_node` (removes the subtree and its instances), `set_local` (marks the subtree dirty), `local`, `world` (as of the last update).
- Instances: `add_instance(node, mesh, material)`, `remove_instance`, `set_instance_visible`, `set_instance_material`.
- Counts: `node_count`, `instance_count`.
- `update_world` processes parents before children, using an order list kept on insert and compacted on remove. It recomputes only dirty subtrees.
- `build_render_list(&mut RenderList)` sorts visible instances in place by (mesh, material, slot), and builds batches from contiguous runs per mesh. It is deterministic and does not allocate when capacity suffices.

**`SceneError`**: `StaleNode`, `StaleInstance` and `ParentCycle`, with codes `scene.stale_node`, `scene.stale_instance` and `scene.parent_cycle`.

## Module: raa-render
Files: `crates/raa-render/{Cargo.toml, README.md}`, `src/{lib, renderer, prep, forward}.rs`, `src/forward.wgsl`, `tests/{prep, gpu_forward}.rs`. Owner: M2.

raa-render is a forward renderer that runs as a frame-graph pass. It depends on raa-core, raa-math, raa-geom, raa-material and raa-scene.

**Options and counts**
- `RendererOptions { max_instances (16384), max_materials (256) }`.
- `FrameCounts { instances, draw_calls, triangles: u64 }`.
- `DrawRange { mesh, first_instance, instance_count }`.
- Consts `INSTANCE_STRIDE` (128) and `FRAME_UNIFORM_SIZE` (144).

**`FramePrep`** is CPU-only and GPU-free.
- `new(max_instances)` preallocates the instance bytes.
- `prepare_cpu(&Scene, &RenderList, surface_is_srgb) -> Result<FrameCounts, RaaError>`:
  - writes S1 instance bytes and the FrameUniform, and fills the draws;
  - leaves triangles at 0;
  - fails with `render.too_many_instances` (RaaError::Resource) beyond max_instances, with no truncation;
  - makes no allocation.
- Accessors: `instance_bytes()`, `frame_uniform_bytes()`, `draws()`.

**`Renderer`**
- `new(gpu, surface_format, surface_is_srgb, opts)`.
- `install(&self, &mut FrameGraph) -> ResourceId` creates the Surface-sized `depth` target (Depth32Float) and appends the `forward` pass. That pass writes BACKBUFFER and depth and clears each on first write: colour to CLEAR_COLOR, depth to 1.0.
- Meshes:
  - `upload_mesh(gpu, &MeshData) -> Result<MeshId, RaaError>` validates the mesh (`render.invalid_mesh`);
  - `remove_mesh` and `mesh_triangles` manage uploaded meshes.
- Materials: `materials()` and `materials_mut()`. A dirty library is uploaded at the next prepare.
- `prepare(gpu, &mut Scene, &mut RenderList)` does the following:
  1. Runs `update_world`, then `build_render_list`.
  2. Runs `prepare_cpu`.
  3. Computes triangles from the mesh table; an unknown mesh gives `render.unknown_mesh`.
  4. Writes the instance, uniform and dirty material buffers with `queue.write_buffer`.

**Pass and shader**
- `ForwardPass` implements RenderPass and shares renderer state through `Rc<RefCell<ForwardState>>`; there is no global state. It issues one `draw_indexed` per DrawRange, with back-face culling and depth Less.
- `FORWARD_WGSL` is forward.wgsl appended to `WGSL_SHADING`. The flat varying `material_index` uses `@interpolate(flat, either)`, so the pipeline is creatable on compatibility-level adapters.

Tests: prep byte layout (S1 offsets), the too_many_instances refusal, deterministic bytes, and a DebugCube rendered into an OffscreenTarget that is non_empty (SKIP without an adapter).

## Module: raa-alloc-count
Files: `crates/raa-alloc-count/{Cargo.toml, README.md, src/lib.rs}`. Owner: M2.

- `pub struct CountingAlloc;` with `unsafe impl GlobalAlloc`. It forwards to `std::alloc::System` and counts alloc, alloc_zeroed and realloc calls and bytes in `static AtomicU64` counters. This is the single documented global-state exception. Its SAFETY comment names the test `counting_alloc_forwards_to_system`, which lives in src.
- Functions: `allocations()`, `bytes()`, `reset()`.
- Use it only in binaries or feature-gated demos: `#[global_allocator] static A: raa_alloc_count::CountingAlloc = raa_alloc_count::CountingAlloc;`.

## Module: raa-bench
Files: `crates/raa-bench/{Cargo.toml, README.md, src/main.rs}`. Owner: M2.

raa-bench builds the binary `frame-allocs`, with CountingAlloc as the global allocator.

Usage: `cargo run --release -p raa-bench --bin frame-allocs -- [--instances 1000] [--warmup 120] [--frames 600]`.

**Setup.** A Scene of instances/100 parent nodes x 100 children, using fake MeshId 0..3 and MaterialId 0..7 and no GPU, plus a RenderList and a FramePrep.

**Per frame:**
1. `set_local` on every parent (rotation by frame).
2. `update_world`.
3. `build_render_list`.
4. `prepare_cpu`.

The counters are reset after warm-up. The binary prints one JSON line `{"instances","warmup","frames","allocations","bytes","prepCpuMsMedian"}` and exits 0. The proof judges the value.

## Module: demo-m2-scene
Files: `demos/m2-scene/{Cargo.toml, src/lib.rs, web/index.html, web/main.js, web/nmc.gpu.json, gpu-input.json}`. Owner: M2.

Follows the demo crate pattern, with feature `alloc-count`.

**Scene:**
- A ground slab: a scaled DebugCube with a flat material.
- Exactly 1000 DebugCube instances in a 10x10x10 jittered lattice, under 10 rotating parent nodes.
- 8 materials: 4 Flat and 4 PbrLite with varied roughness and metallic.
- The orbit camera turns slowly.

**Run op.** `run` accepts `resize: true` and reports 1000 instances (plus the ground). gpu-input.json requests the CAPTURE size, BROWSER_RUN and the resize cycle.

**Builds.** The capture uses `--features alloc-count`, so allocsPerFrameTotal is reported. The size check uses the default build.

## Module: proof-m2
Files: `proofs/m2-scene.mjs`, `proofs/results/m2-scene.json`, `docs/dogfood/m2.md`. Owner: M2.

Checks, in order:
1. `cpu-ms-1000` runs first and alone: hardware true, instances 1000, cpuMsMedian <= CPU_MS_MEDIAN_MAX.
2. `browser-run`: nonEmpty, resize.ok, firstFrameMs <= FIRST_FRAME_MAX_MS, adapter named.
3. `frame-allocs`: raa-bench reports allocations == 0 over ALLOC_RUN.
4. `wasm-size`: the default build of dist/m2-scene.
5. `scene-tests`: `cargo test -p raa-math -p raa-geom -p raa-material -p raa-scene -p raa-render`.

## Module: raa-prims
Files: `crates/raa-prims/{Cargo.toml, README.md}`, `src/{lib, kind, params, sphere, boxes, capsule, cylinder, cone, torus, ellipsoid, provider, reference}.rs`, `examples/prims-report.rs`, `tests/{closed, determinism, sdf_reference}.rs`, `tests/fixtures/sdf-leaf-reference.json`. Owner: M3.

raa-prims turns SDF primitives into polygons straight from their parameters. It ports SDF Modeler's `leafGeometry` maths and samples no distance field. It depends on raa-math and raa-geom.

**Kinds and parameters**
- `PrimKind { Sphere, Box, RoundBox, Capsule, Cylinder, Cone, Torus, Ellipsoid }` provides `ALL` (this order), `as_str` (S3 names), `parse`, `is_convex` and `param_names` (S3 order).
- `PrimParams` has one variant per kind with snake_case fields: `half_extents`, `radius`, `half_height`, `major_radius`, `minor_radius` and `radii`. Methods:
  - `kind`, `from_params(kind, &Params)`, `to_params`;
  - `validate` (S3 rules, finite values);
  - `bounds` (analytic);
  - `reference_set(kind, set)`: S3 set A or B. Any char other than `'B'` gives A.

**Generation**
- `generate(&PrimParams, Quality, &mut MeshData)` validates and clears `out`. It emits a closed, outward mesh that satisfies NORMAL_RULE and is deterministic, with hard-edge duplication per S2.
- `triangle_count(&PrimParams, Quality) -> u32` is a closed formula equal to the generated count.
- Every port derives its tessellation counts from `reference::COUNT_TABLE` with the S8 count rule. The table:
  - sphere and ellipsoid: nu 40, nv 20;
  - box: no counts;
  - roundBox: cells 10;
  - capsule: nu 32, nCap 10;
  - cylinder and cone: nu 40;
  - torus: nu 48, nv 20.
- The README documents seam welding, pole handling and the segment mapping per kind. uniquePositions and triangles match the reference exactly, with no per-case code.

**Provider.** `PrimProvider(pub PrimKind)` implements GeometryProvider. `register_all(&mut GeometryRegistry)` registers the 8 kinds in `ALL` order.

**`pub mod reference`** is the fixture reader and the SDF_REF_TOL comparison, shared by prims-report and the tests. It has no per-kind code and fixed tolerances.
- Tolerances and the count rule:
  - `BBOX_TOL = 1e-5`, multiplied by max(1, reference diagonal), and `REL_TOL = 1e-4`;
  - `CASE_COUNT = 48`;
  - `CountSpec { name, reference, min, multiple }`, `COUNT_TABLE`, `count_rules(kind)` and `count(reference, min, multiple, segments)`;
  - `sdf_value(scale, a, add)`.
- `RefError`:
  - variants `Fixture`, `MappingInvalid` and `CountsMismatch`, with codes `prims.fixture_invalid`, `prims.mapping_invalid` and `prims.reference_counts_mismatch`;
  - `path()` returns a JSON pointer and `reason()` the reason;
  - `From<RefError> for GeomError` maps it to `InvalidParam { kind: "fixture", .. }`.
- Mapping v1 types:
  - `ReferenceMapping { commit, axis_map: [(u8, i8); 3], det, kinds: Vec<(PrimKind, String)>, entries: Vec<KindMapping> }` with `kind()`;
  - `KindMapping { kind, sdf_kind, params: Vec<ParamTerm>, constants, segments: Vec<CountRule> }` with `counts(segments)`;
  - `ParamTerm` (with `evaluate`), `AddTerm`, `CountRule` and `ConstValue`.
- `parse_mapping(text)` validates a standalone mapping v1 block.
- Fixture types:
  - `ParamMapEntry { ref_name, ref_index, raa_name, raa_index, scale, value, add: Option<ParamMapAdd> }` with `evaluate`;
  - `RefCase { kind, set, quality, params, ref_params, sdf_input, counts, unique_positions, triangles, bbox_min, bbox_max, area, volume, closed }`;
  - `RefFixture { command, sdf_modeler_revision, axis_map, param_map, cases }` with `mapping()` and `case(kind, set, quality)`.
- `parse_fixture(text) -> Result<RefFixture, RefError>` uses its own JSON reader, no serde. It validates mapping v1, the counts and the S8 shape.
- `RefDelta { unique_positions_eq, triangles_eq, bbox_max_abs_err, area_rel_err, volume_rel_err, ok, message }`.
- `compare(&RefFixture, &RefCase, &MeshData) -> RefDelta` does the following:
  - runs `analyze` and `unique_positions_1e6`;
  - checks that the case params equal `reference_set(kind, set)`;
  - checks that paramMap reproduces refParams within 1e-6 relative;
  - applies SDF_REF_TOL.

  Fixture metrics are already in RAA's frame, so compare never re-maps them. A reference case with `closed: false` is a mismatch with the message `reference mesh not closed: report to the lead`.
- `map_to_raa(&ReferenceMapping, sdf, out)` applies axisMap to a reference mesh.

**Example `prims-report`.** Usage: `cargo run --release -p raa-prims --example prims-report -- [--only <pattern on kind>] [--json <path>] [--compare <fixture>]`.
- `--only` supports `^`, `( )`, `|` and `$` over literal kind names. Other metacharacters exit 2.
- For each selected kind x set (A, B) x quality (LOW, LOD0, HIGH), it prints a line with kind, set, quality, triangles, vertices, uniquePositions, closed, volume, normal deviation and hash. The mesh is generated twice, and both hashes must match.
- With `--compare` it prints the deltas next to SDF_REF_TOL and the fixture's `mapping.reference.commit`. On failure it also prints the case's `sdfInput`.
- `--json` writes `[{kind, set, quality, triangles, vertices, uniquePositions, closed, passes, hash, deterministic, counts, ref: {uniquePositionsEq, trianglesEq, bboxMaxAbsErr, areaRelErr, volumeRelErr, ok}|null}]`.
- It exits 1 on any failure of `passes`, determinism or the reference comparison.

**Tests** (rustfmt-clean):
- closed.rs: 8 kinds x 3 qualities x 2 sets pass `analysis::passes`, and invalid params give `geom.invalid_param`.
- determinism.rs: stable hashes, and the triangle_count formula holds.
- sdf_reference.rs: all 48 fixture cases are within SDF_REF_TOL, capsule included.

## Module: sdf-ref-tool
Files: `tools/sdf-ref/gen-leaf-fixtures.mjs`, `tools/sdf-ref/README.md`. Owner: M3.

A one-off node script, never imported by the engine.

**Command:** `/root/nmc/engines/bin/node tools/sdf-ref/gen-leaf-fixtures.mjs --ref <sdfm checkout root> --out crates/raa-prims/tests/fixtures/sdf-leaf-reference.json`

**Reference access.**
- The reference is read only, through the cross-repository alias `sdfm`: `src/core/hybrid/leaf-mesh.ts` and its imports.
- The recorded commit is sdfm HEAD.
- The script loads the file with node's built-in type stripping, with no npm dependency.
- It refuses an `--out` path inside the reference checkout.

**Generation.**
- The script embeds the README's mapping v1 block verbatim and validates it the same way raa-prims does (S8).
- It produces every quality from sdfm's own `leafGeometry` maths, through one of two paths:
  - `call`: invokes the counting functions with explicit counts.
  - `literal`: builds an in-memory copy of leaf-mesh.ts with only the `source` literals substituted. It never writes to sdfm.
- It asserts that the LOD0 output equals unmodified `leafGeometry` exactly.
- It writes the 48 S8 cases with `counts`, `sdfInput` and metrics after axisMap, plus the per-kind path, the commit and `leafMeshSha256`.

**Output and refusals.** Output is deterministic: stable key order, and numbers through `JSON.stringify` of f64. Any invalid mapping, missing path, absent kind, load failure or non-positive volume exits non-zero with `sdf-ref: <message>`. The script never writes a partial fixture.

**README.md**, in this order:
1. Access path and commit.
2. Quoted excerpts of `leafGeometry`, `LEAF_KINDS` and the vertex layout, each with file:line@commit.
3. Per kind: axis convention, parameter names and units, every count with its file:line, segments handling, winding, and seam welding or duplication.
4. The mapping v1 fenced JSON block and its rules, including the additive term and the capsule entry.
5. An "Unmappable" section, which records the capsule finding as resolved by the additive term.
6. The regeneration command.

A count or param that does not fit is a stop-and-ask finding.

## Module: demo-m3-prims
Files: `demos/m3-prims/{Cargo.toml, src/lib.rs, src/web.rs, web/index.html, web/main.js, web/nmc.gpu.json, gpu-input.json}`. Owner: M3.

Follows the demo crate pattern. The page is a contact sheet:
- 8 columns in `PrimKind::ALL` order, left to right, by 3 rows: LOW at the top, LOD0 in the middle, HIGH at the bottom.
- Reference set A, 2.5 m spacing, one PbrLite colour per kind.
- A fixed camera frames the whole grid at CAPTURE size.
- Each primitive rotates slowly about Y (time-based). The capture is taken after BROWSER_RUN.

The page reports the triangles of all 24 meshes.

## Module: proof-m3
Files: `proofs/m3-prims.mjs`, `proofs/results/m3-prims.json`, `proofs/results/m3-prims-table.json`, `docs/dogfood/m3.md`. Owner: M3.

Checks:
- `prims-closed-normals`: `cargo test -p raa-prims --test closed`.
- `prims-deterministic` and `sdf-reference`: prims-report with `--compare`, run per kind in parallel through `pool` and honouring `--only`.
  - Measured = cases passed / total, and the threshold is all 48 cases, capsule included.
  - The comparison line cites `mapping.reference.commit`.
  - Writes `target/proofs/m3-prims-table.json`.
- `wasm-size`: dist/m3-prims.
- `browser-run`: m3-prims.json nonEmpty, firstFrameMs.

The committed `m3-prims-table.json` lists, per kind and quality, the derived counts next to the triangle count. The box row is identical at every quality.

## Module: raa-loader
Files: `crates/raa-loader/{Cargo.toml, README.md}`, `src/{lib, json, error, registry, boolean, collider}.rs`, `src/prims/{mod, model, parse, write, build, samples}.rs`, `examples/prims-check.rs`, `tests/{json, roundtrip, refusals, build}.rs`. Owner: M4.

raa-loader holds the asset loading trait and the prims v0 loader. glTF can become another AssetLoader later. It depends on raa-math, raa-geom and raa-material (dev: raa-prims).

**`json`**
- `JsonValue { Null, Bool, Number(f64), String, Array, Object(Vec<(String, JsonValue)>) }`.
- `parse(text)` follows RFC 8259 with UTF-8 input. It refuses duplicate keys and limits nesting depth to 32. The error path is `@<offset> line:col`.
- `write_pretty(&JsonValue, &mut String)` follows the S4 canonical text rules, with keys in stored order.
- `format_number(x, &mut String)` follows JS Number-to-string rules.

**Errors and registry**
- `LoadError { code, path, message }` uses the S5 codes. Display is `<code> at <path>: <message>`.
- `trait AssetLoader { format; sniff(&[u8]) -> bool; load(&[u8], &LoadContext) -> Result<LoadedAsset, LoadError> }`.
- `LoaderRegistry`:
  - `new`;
  - `register` (`loader.duplicate_format`);
  - `load`, which uses the first loader whose sniff is true, else `loader.unknown_format`;
  - `formats`.
- `LoadContext<'a> { geometry: &GeometryRegistry, booleans: &dyn BooleanResolver, quality: Quality, strict_kinds: bool }`.
  - With strict_kinds true, only the 8 S3 kinds are accepted (`kind.unknown`).
  - With false, any registered kind is accepted and its provider validates the params.

**Loaded asset**
- `LoadedAsset { format, id, name, meshes: Vec<LoadedMesh>, materials: Vec<(String, Material)>, anchors: Vec<(String, Mat4)>, colliders: Vec<Collider>, triangles, budget }`.
- `LoadedMesh { node_id, kind, mesh, transform, material }`:
  - `mesh` is in node-local primitive space, with booleans applied;
  - `transform` is in asset space;
  - `material` is resolved, including inheritance.
- Meshes come in depth-first pre-order. The default material has the reserved name `_default`.

**Booleans**
- `BoolOp { Subtract, Intersect }`.
- `BooleanHost { node_id, kind, convex, mesh }`.
- `BooleanTool { node_id, kind, op, convex, has_children, mesh }`, where `mesh` is in host local space.
- `trait BooleanResolver { resolve(&BooleanHost, &[BooleanTool], &mut MeshData) -> Result<(), LoadError> }`.
- `NoBooleans` refuses any tool with `boolean.unsupported`, naming the first tool and op ("no boolean resolver installed").

**Colliders**
- `ColliderShape { Ball, Cuboid, RoundCuboid, CapsuleY, CylinderY, ConeY, ConvexHull { points } }`.
- `Collider { node_id: Option<String>, shape, transform: Mat4 }`, derived or overridden per S4. A uniform scale is folded into the shape dimensions.

**`prims` model**
- Consts: `FORMAT = "raa.prims"`, `VERSION = 0`, `DEFAULT_TRIANGLE_BUDGET = 1500`, `MAX_NODES = 64`, `MAX_LEVEL = 8`.
- Document types: `PrimsAsset { id, name, triangle_budget, nodes, materials, anchors, collision }`, `PrimNode { id, kind, params, transform, material, op, children }`, `TransformDoc { t, r, s }`, `NodeOp { Union, Subtract, Intersect }`, `PrimMaterial { color, roughness, metallic, emissive }` and `ColliderDoc { node, shape, params, transform }`.

**`prims` functions**
- `parse(text)` checks syntax, schema, unknown keys, finiteness, ASSET_CAPS, ids, material references, kind names and param shapes. The first error in document order wins. Children of tools are parsed and validated, not refused.
- `to_canonical_json(&PrimsAsset)` writes S4 canonical text.
- `estimate_triangles(&PrimsAsset, &LoadContext)` counts before booleans, through the providers' `triangle_count`.
- `build(&PrimsAsset, &LoadContext)`:
  - generates each node at `ctx.quality`;
  - sends hosts with boolean children through `ctx.booleans`, with tool meshes transformed into host space and `has_children = !tool.children.is_empty()`;
  - fails with `cap.budget`, giving both numbers, when the total exceeds the budget;
  - derives colliders;
  - is deterministic.
- `PrimsLoader` implements AssetLoader. Sniff: the first 512 bytes contain the format string. Load: parse, then build.

**`prims::samples`**
- `sample_assets()` returns canonical texts built in code:
  - `lamp`: cylinder, sphere, cone, torus.
  - `critter`: ellipsoid body, sphere eyes, capsule limbs, anchors.
  - `crate`: roundBox, boxes.
- `invalid_cases()` returns (label, text, expected code), with at least one case per S5 code reachable from text. Required cases include 65 nodes, level 9, NaN written as an exponent overflow, a budget of 10, an unknown kind, an unknown key, a duplicate id, an unknown material, a torus without override, a bad quaternion, scale 0 and a subtract with NoBooleans.

**Example `prims-check`.** Usage: `cargo run --release -p raa-loader --example prims-check -- [--only <regex>] [--json <path>]`. It runs three groups of cases:
- roundtrip: canonical text in gives identical text out, and non-canonical text is canonicalised idempotently.
- refusal: the expected code with a non-empty message.
- build: closed meshes per `passes`, mesh count = union nodes, triangles <= budget, colliders derived.

It prints one line per case and exits 1 on any failure.

## Module: demo-m4-asset
Files: `demos/m4-asset/{Cargo.toml, src/lib.rs, web/index.html, web/main.js, web/nmc.gpu.json, gpu-input.json}`. Owner: M4.

Follows the demo crate pattern. The demo:
1. Loads the three `sample_assets()` texts through LoaderRegistry and PrimsLoader (LOD0, strict kinds, NoBooleans).
2. Uploads their meshes and materials.
3. Spawns each asset 3 times side by side on a ground slab, with the orbit camera.

It reports triangles and draw calls.

## Module: proof-m4
Files: `proofs/m4-loader.mjs`, `proofs/results/m4-asset.json`, `docs/dogfood/m4.md`. Owner: M4.

Checks:
- `roundtrip`: prims-check `--only roundtrip`.
- `refusals`: prims-check `--only refusal`. Measured = cases with the expected code / total.
- `build`: prims-check `--only build`, plus `cargo test -p raa-loader`.
- `wasm-size`: dist/m4-asset.
- `browser-run`: m4-asset.json nonEmpty, firstFrameMs.

## Module: raa-csg
Files: `crates/raa-csg/{Cargo.toml, README.md}`, `src/{lib, plane, clip, weld, resolver, cases}.rs`, `examples/csg-cases.rs`, `tests/{convex, cases, loader}.rs`. Owner: M5.

raa-csg evaluates simple booleans of a primitive against a convex host with exact face cutting, and returns a typed refusal otherwise. It depends on raa-math, raa-geom and raa-loader (dev: raa-prims).

**`Plane { n: [f64; 3], d: f64 }`.** A point p is outside when `n.p - d > eps`.

**`CsgError`**

| Variant | Code |
|---|---|
| `HostNotConvex { kind }` | `csg.host_not_convex` |
| `ToolNotConvex { kind }` | `csg.tool_not_convex` |
| `ToolHasChildren { node }` | `csg.tool_has_children` |
| `MultipleIntersect` | `csg.multiple_intersect` |
| `OverlappingTools { a, b }` | `csg.overlapping_tools` |
| `Coplanar { detail }` | `csg.coplanar` |
| `EmptyResult` | `csg.empty_result` |
| `NotWatertight { boundary_edges, nonmanifold_edges, inconsistent_edges }` | `csg.not_watertight` |
| `Degenerate(String)` | `csg.degenerate` |

The Display message names the nodes and numbers.

**`convex_planes(mesh, eps)`** returns face planes, merged when normals agree within 1e-9 and offsets within eps, in first-occurrence order. A non-convex mesh is an error that the caller maps to HostNotConvex or ToolNotConvex.

**`intersect_convex(host, tool, out)`**
- Host faces are clipped to the inside of every tool plane, and tool faces to the inside of every host plane, in their original winding.
- Disjoint inputs give EmptyResult. A host inside the tool returns the host, and a tool inside the host returns the tool.

**`subtract_convex(host, host_planes, tools, out)`**
- Tools must have pairwise disjoint bboxes, else OverlappingTools.
- Each host polygon is split by each tool's planes in order. Outside parts are emitted, and the final inside part is discarded.
- Tool triangles are clipped to the inside of the host, then emitted with reversed winding and negated normals.
- A tool that does not overlap the host leaves it untouched. A host entirely inside a tool gives EmptyResult.

**Common rules**
- eps = CSG_EPS. A vertex within eps of a plane is ON, and both sides keep it as a boundary.
- A tool plane that coincides with a host plane (parallel either sign within 1e-9, offset within eps) and whose faces overlap gives Coplanar.
- Split points on an edge are computed from the endpoints in canonical (lexicographic) order, so neighbours agree. Host normals are interpolated along split edges and renormalised.
- Convex polygons are fan-triangulated. Then come a deterministic weld (CSG_EPS grid, first occurrence wins), T-junction repair (up to 3 passes) and `analysis::analyze`.
- Any output that is not closed, consistent and of positive volume gives NotWatertight with the counts. There is never a silent output. All iteration orders are fixed.

**`CsgResolver`** implements BooleanResolver.
- It refuses ToolHasChildren, MultipleIntersect, HostNotConvex (host kind not convex, or plane extraction fails) and ToolNotConvex.
- It applies the single intersect first; the result stays convex. Then it applies all subtract tools together against the post-intersect planes.
- Errors map to `LoadError { code: "boolean.<csg code>", path: <tool's JSON pointer>, message }`.
- The raa-prims LOD0 roundBox mesh is not convex within CSG_EPS, although the analytic shape is. A roundBox host or tool is therefore refused (`boolean.csg.host_not_convex` / `tool_not_convex`). The README states this limitation. CSG_EPS is never relaxed.

**Case table.** `CsgCase { id, expected, asset }`, where `asset` is canonical prims v0 text. `cases()` returns the fixed table of 20 cases built from S3 set A, 11 ok then 9 refusals.
- **Ok:**
  - `box-sub-sphere-dent`, `box-sub-cylinder-through`, `sphere-sub-box-corner`, `cylinder-sub-capsule-slot`;
  - `box-sub-ellipsoid-scoop`: box [0.7,0.6,0.5] minus ellipsoid [0.9,0.6,0.5] at t [0.713,0.011,0.007], identity rotation, s [0.5,0.5,0.5];
  - `box-int-sphere`, `sphere-int-cylinder`, `cone-int-box`, `box-sub-two-spheres`, `box-int-sphere-sub-cylinder`;
  - `box-sub-sphere-outside` (unchanged host).
- **Refusals:**
  - `torus-host-sub-sphere` (host_not_convex);
  - `box-sub-torus` and `box-int-torus` (tool_not_convex);
  - `box-sub-box-coplanar` (coplanar);
  - `box-int-sphere-disjoint` and `sphere-sub-box-containing` (empty_result);
  - `box-sub-overlapping-spheres` (overlapping_tools);
  - `box-sub-tool-with-children` (tool_has_children);
  - `box-two-intersects` (multiple_intersect).

  No case gets specific code. tests/cases.rs also asserts the host refusal for a roundBox-host asset outside `cases()`.

**Example `csg-cases`.** Usage: `cargo run --release -p raa-csg --example csg-cases -- [--only <regex>] [--json <path>]`.
- It loads every case through PrimsLoader and CsgResolver at LOD0. Budgets are raised in the case text where needed, never in code.
- For ok cases it checks the volume identity V(host - tool) = V(host) - V(host ∩ tool), computed with the same algorithm functions.
- It prints one S9 line per case and runs cases in threads, up to NMC_CPUS.
- It exits 1 when a case differs from its expectation, an ok case is not watertight or exceeds the identity tolerance, or a refusal has an empty message.

## Module: demo-m5-booleans
Files: `demos/m5-booleans/{Cargo.toml, src/lib.rs, web/index.html, web/main.js, web/nmc.gpu.json, gpu-input.json}`. Owner: M5.

Follows the demo crate pattern. The demo shows every ok case of `raa_csg::cases()`, loaded through PrimsLoader and CsgResolver. The grid follows table order, 4 per row, and each case rotates slowly so the cut faces are visible. Refused cases are not drawn. The demo reports triangles.

**Streaming (first frame).** `init` does only the cheap setup: registries, the ok-case list (`expected == "ok"`), the Renderer, install, the camera, and then load, upload and spawn of case 0 only.
- The layout is fixed and independent of CSG results: count = number of ok cases, pitch = constant `PITCH`. A host test asserts spacing(all baked diagonals) <= PITCH <= 1.5 x spacing, and the in-view/in-slot test uses PITCH.
- `update` loads, uploads and spawns at most one further case per frame, in table order, into its fixed slot. All 11 cases are present from frame 10, inside the warm-up. After that, `update` only calls `set_local` and `prepare`, with no allocation.
- Scene and RenderList start at capacity (count, count) and may grow only during the loading frames.
- `load_cases()` loops over the same per-case step, for the host tests.
- A load error in `update` is returned unchanged.

## Module: proof-m5
Files: `proofs/m5-booleans.mjs`, `proofs/results/m5-booleans.json`, `proofs/results/m5-csg-cases.json`, `docs/dogfood/m5.md`. Owner: M5.

Checks:
- `csg-supported`: the 11 ok cases are watertight with volume identity <= 1e-4. Measured = passing / total.
- `csg-refusals`: the 9 refusal cases give the expected code with a non-empty message.
- `loader-integration`: `cargo test -p raa-csg --test loader`. A subtract asset builds with CsgResolver and is refused with NoBooleans.
- `wasm-size`: dist/m5-booleans.
- `browser-run`: m5-booleans.json nonEmpty, firstFrameMs <= FIRST_FRAME_MAX_MS.

The committed `m5-csg-cases.json` is the S9 table produced by the same command.

`m5-booleans.json` is captured with the streaming demo. Its instances, drawCalls and triangles are those of the full 11-case grid, because streaming ends inside warm-up. The result is never edited or re-run selectively. `docs/dogfood/m5.md` records that cases stream in after frame 0.

## Module: raa-js
Files: `crates/raa-js/{Cargo.toml, README.md}`, `src/{lib, engine, convert}.rs`, `tests/convert.rs`. Owner: M6.

raa-js is the frozen wasm-bindgen API v0 (cdylib + rlib). It depends on raa-core, raa-web, raa-math, raa-geom, raa-material, raa-scene, raa-render, raa-prims, raa-loader and raa-csg.

**Engine.** `#[wasm_bindgen(js_name = Engine)] pub struct Engine` is a WebApp inside a WebRuntime. It installs the forward renderer, registers the CsgResolver and the 8 prims, and uses strict kinds.
- Handles exposed to JS are monotonically increasing u32 values mapped to internal generational ids, never reused.
- Every error is thrown through `raa_web::error_to_js`.

**Methods** (JS names):
- Lifecycle and frame:
  - `static create(canvas, options) -> Promise<Engine>`. Options: maxInstances (16384), maxMaterials (256), powerPreference.
  - `resize(width, height)`, `render(nowMs)`, `stats() -> FrameStats`, `probeNonEmpty(nowMs) -> Promise<boolean>`, `dispose()`.
- Materials: `addMaterial(desc) -> u32` and `setMaterial(id, desc)`. Desc fields: color (sRGB), roughness (0.6), metallic (0), emissive (linear, 0), model `'flat'|'pbr'` (default `'pbr'`).
- Meshes:
  - `primitiveMesh(kind, params, segments?) -> u32` (S3 names, LOD0 default) and `removeMesh(id)`.
  - `loadPrimsAsset(json, segments?) -> AssetInfo`.
  - `spawnAsset(asset, parent?) -> SpawnedAsset` creates one node per LoadedMesh under a new root.
- Nodes:
  - `addNode(parent|null, t?, r?, s?) -> u32`, `setTransform(node, t, r, s)`, `removeNode(node)`.
  - `setTransforms(nodes: Uint32Array, trs: Float32Array)` takes 10 floats per node. A length mismatch throws `api.length_mismatch`.
- Instances: `addInstance(node, mesh, material) -> u32`, `removeInstance(id)`, `setVisible(id, visible)`.
- Camera and lights:
  - `setCamera(desc)`: eye, target, up, fovYDeg (60), near (0.1), far (200).
  - `orbit(dYawRad, dPitchRad, zoomFactor)` and `setOrbit(target, distance, yawRad, pitchRad)`.
  - `setLight(desc)` and `setAmbient(desc)`.

No method or doc names a consumer game.

## Module: js-package
Files: `packages/raa/{package.json, index.js, index.d.ts, README.md}`, `scripts/build-package.sh`. Owner: M6.

**Package.** `@etcito/raa` 0.1.0 with `"type": "module"` and `exports: { ".": { types: ./index.d.ts, default: ./index.js } }`. `files` lists index.js, index.d.ts, README.md and pkg/ (the wasm-bindgen output of raa-js, gitignored).

**`scripts/build-package.sh`** runs entirely offline:
1. Builds raa-js with the wasm-release profile.
2. Runs wasm-bindgen `--target web` into `packages/raa/pkg/`.
3. Writes `dist/raa-sizes.json` (S7).
4. Builds examples/basic into `dist/example-basic/`, with the package copied to `dist/example-basic/raa/` and web/common copied too.
5. Runs `npm pack --pack-destination dist`, which produces `dist/etcito-raa-0.1.0.tgz`.

**`index.js`** exports `async function loadRaa(options = {}) -> Promise<RaaModule>`. It initialises the wasm from `options.wasmUrl` or the packaged pkg URL. There is no module-level mutable state, and each call is independent. `RaaModule = { version: '0.1.0', createEngine(canvas, options) -> Promise<Engine> }`.

**`index.d.ts`** is hand-written. The generated pkg/*.d.ts is kept but not exported. It declares:
- `Vec3`, `Quat`, `PrimKind`, `EngineOptions`, `MaterialDesc`, `CameraDesc`, `LightDesc`, `AmbientDesc`, `AdapterInfo`, `FrameStats`;
- `AssetInfo { id, name, triangles, budget, meshes: number[], materials: number[], anchors: Record<string, Float32Array> }`, where each anchor is 16 floats, column-major;
- `SpawnedAsset { root, nodes, instances }`;
- `RaaError extends Error { code: string }`;
- `Engine`, with the raa-js methods and exact types;
- `RaaModule`, and `loadRaa(options?: { wasmUrl?: string | URL }): Promise<RaaModule>`.

## Module: example-basic
Files: `examples/basic/{index.html, main.js, nmc.gpu.json, gpu-input.json}`. Owner: M6.

The example page uses only `./raa/index.js`, the packed package. It:
- loads a prims v0 asset text built in code inside main.js and spawns it 4 times;
- adds 200 instanced primitives, using `primitiveMesh` for each kind;
- uses the orbit camera and a light.

It registers `window.nmcGpu.ops.run` with the semantics of `web/common/run-loop.js` (S6).

## Module: raa-ext-test
Files: `tests/raa-ext-test/{Cargo.toml, README.md}`, `src/{lib, star, tint_pass}.rs`, `tests/extension.rs`. Owner: M6.

This crate shows that extensibility is tested, not just claimed. It depends only on RAA library crates (raa-core, raa-math, raa-geom, raa-render, raa-scene, raa-material, raa-loader, raa-prims), with wgpu reached through `raa_core::wgpu`. No crate depends on it. It adds new behaviour without editing any of them.

- `StarProvider` implements GeometryProvider with kind `star`. Params: `points` (integer >= 3), `outerRadius`, `innerRadius` and `halfHeight`. It builds a closed extruded star prism with hard edges, and is not convex.
- `TintPass` implements RenderPass with the name `ext-tint`. It reads and writes BACKBUFFER with LoadOp::Load and draws a fullscreen triangle that alpha-blends a fixed tint.

Tests (extension.rs):
- The star provider registers next to `raa_prims::register_all` and passes `analysis::passes` at 3 qualities.
- PrimsLoader loads a `star` asset when strict_kinds is false and refuses it with `kind.unknown` when true.
- `Renderer::install` plus `add_pass_after("forward", TintPass)` compiles in the order [forward, ext-tint].
- A GPU readback shows the tint over CLEAR_COLOR (SKIP without an adapter).

## Module: project-docs
Files: `README.md`, `docs/API.md`, `docs/ARCHITECTURE-notes.md`. Owner: M6.

**`docs/API.md`** is public API v0, frozen. It contains:
- A JS section covering every export of `packages/raa/index.d.ts`, with signature, semantics, units, defaults and error codes.
- A Rust section listing, per crate, the public items of this contract and the extension points (GeometryProvider, RenderPass, `FrameGraph::add_pass_after`/`add_pass_before`, AssetLoader, BooleanResolver), each with a minimal example.
- The stability rule: v0 allows additive changes only until v1.
- The size table from `dist/*sizes.json` and the measured numbers from `proofs/results`.

**`README.md`** covers:
- what RAA is and the crates map;
- the build and check commands;
- the demo and example regeneration commands;
- the hook install command.

The stack sentence stays in README only, outside the neutrality scope.

**`docs/ARCHITECTURE-notes.md`** describes the data flow for contributors: asset, loader, meshes, renderer, frame graph.

## Module: proof-m6
Files: `proofs/m6-api.mjs`, `proofs/results/example-basic.json`, `proofs/results/m6-sizes.json`, `docs/dogfood/m6.md`. Owner: M6.

Checks:
- `extension`: `cargo test -p raa-ext-test`, plus `cargo metadata --format-version 1 --no-deps`. raa-ext-test must depend only on raa-* library crates, and no workspace crate may depend on it.
- `api-doc`: every name exported by `packages/raa/index.d.ts` appears in `docs/API.md`. Measured = covered / total.
- `npm-pack`: `scripts/build-package.sh` exits 0, and `dist/etcito-raa-0.1.0.tgz` lists package.json, index.js, index.d.ts, README.md, pkg/*_bg.wasm and pkg/*.js.
- `wasm-size-table`: every `dist/*/sizes.json` and `dist/raa-sizes.json` has gzip <= WASM_GZIP_MAX. The table is printed.
- `browser-run`: example-basic.json nonEmpty, firstFrameMs.

The committed `proofs/results/m6-sizes.json` is the size table written by the task.

## Milestone proofs
- M1 Boot: workspace, checks and a WebGPU triangle from WASM in a real browser: `/root/nmc/engines/bin/node proofs/m1-boot.mjs`
- M2 Scene basics: 1000 lit, depth-tested instances with zero allocations in the frame loop: `/root/nmc/engines/bin/node proofs/m2-scene.mjs`
- M3 Primitives to polygons: 8 SDF kinds as closed meshes matching SDF Modeler: `/root/nmc/engines/bin/node proofs/m3-prims.mjs`
- M4 Primitive asset loader: prims v0 parsed, validated, round-tripped and built into meshes: `/root/nmc/engines/bin/node proofs/m4-loader.mjs`
- M5 Simple booleans: exact convex cutting or a typed refusal: `/root/nmc/engines/bin/node proofs/m5-booleans.mjs`
- M6 Extensibility proven and public API v0 frozen: @etcito/raa package, docs and example page: `/root/nmc/engines/bin/node proofs/m6-api.mjs`
