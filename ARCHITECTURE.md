# RAA architecture

RAA is an extensible rendering engine library: small Rust crates compiled to WebAssembly over WebGPU, a JS package `@etcito/raa`, demo pages and an example page. This document is the single contract for its modules. Conventions apply to every module. The numbered shared sections (S1 to S9) define data shapes that modules implement and never redefine.

## Conventions

### Repository, toolchain, layout
- One repository: `raa`. Paths are relative to it. It contains `Cargo.toml`, `Cargo.lock` (committed), `LICENSE` (MIT), `README.md` and `.gitignore`. The `.gitignore` is kept as is and blocks target/, node_modules/, dist/, pkg/, *.log, *.glb, *.gltf, *.png, *.jpg, *.avif, *.ktx2 and *.bin.
- Rust edition 2024, rust-version 1.96.0, `members = ["crates/*", "demos/*", "tests/*"]`. These are globs, so adding a crate never edits the root manifest. Third-party versions live in `[workspace.dependencies]`. Members use `{ workspace = true }` and may only add features. rustfmt and clippy are required.
- Closed dependency list:
  - `wgpu` 30.0.1, `default-features = false`, features `["wgsl"]`. wasm32 adds `webgpu`, native adds `vulkan` and `gl`.
  - `naga`, at the version Cargo.lock already has through wgpu, with `features = ["wgsl-in"]` and default features off. Used for WGSL validation only.
  - `wasm-bindgen = "=0.2.129"` (equal to the installed CLI), `wasm-bindgen-futures = "=0.4.79"`, `js-sys = "=0.3.106"`, `web-sys = "=0.3.106"`.
  - `pollster` 1.0.1, for native tests and the bench only.

  RAA has no serde, no glam or nalgebra, no bytemuck and no log framework. Adding a dependency needs the lead's authorisation and a minimal edit of the root manifest. The task that adds it runs `cargo fetch` for all targets, so the offline sandbox has every crate.
- Library crates: `crates/<name>/{Cargo.toml, README.md, src/lib.rs, tests/<topic>.rs}`. The README states the purpose, the public items and how to test.
- Other locations:
  - demos in `demos/<mN-name>/` (a cdylib, `web/` and `gpu-input.json`);
  - test-only crates in `tests/<name>/`;
  - `proofs/`, `scripts/`, `web/` (shared web helpers) and `tools/`;
  - JS packages `packages/raa/` (`@etcito/raa`) and `packages/raa-sdfm/` (`@etcito/raa-sdfm`, plain ES modules);
  - pages `examples/basic/`, `examples/lookdev/` and `examples/sdfm-compare/`.
- Every crate compiles for the host and for `wasm32-unknown-unknown`. Native-only code sits behind `#[cfg(not(target_arch = "wasm32"))]` and web-only code behind `#[cfg(target_arch = "wasm32")]`.
- `[profile.wasm-release]` inherits `release` with `opt-level = "z"`, `lto = true`, `codegen-units = 1`, `panic = "abort"`, `strip = true` and `debug = false`.
- Tools: `node` (scripts use `${NODE:-node}`, npm next to it), the `wasm-bindgen` CLI 0.2.129 on PATH, and `python3` for the judge client. wasm-opt is not used by default.

### Coordinates, units, colour
- Right-handed, Y up, the camera looks down -Z, units are metres. Front faces are counter-clockwise, back faces are culled and normals point outward.
- Matrices are column-major f32 with column vectors (`M * v`). Clip depth is [0, 1]. Depth is `Depth32Float`, compare `Less`, cleared to 1.0. Quaternions are `[x, y, z, w]`.
- Colours are linear RGB in Rust and on the GPU. Asset files and JS colour inputs are sRGB in [0, 1] unless the API says linear, and are converted at the boundary.
- The v0 renderer shades straight to the canvas (S1). The lit renderer shades linear HDR into HDR_FORMAT, and only `tonemap` converts to display (S12). When the canvas is not `*Srgb`, the shader encodes sRGB itself.

### Errors and refusals
- Each crate has one error type implementing `Display` and `std::error::Error`, with a stable `code() -> &'static str` of the form `<area>.<snake_case>` and a human message.
  - P2 crates return `raa_core::RaaError`, using `RaaError::FrameGraph { code, message }` for their area codes.
  - Refusals are typed values, never NaN, an empty output, a panic or a skip.
  - A refused setter leaves the previous state unchanged.
- No `unwrap`, `expect` or indexing panic on paths that public API with user data can reach. They are allowed in tests and on commented invariants.

### No global state, no hidden allocation, unsafe
- No `static mut`, no `thread_local!` state and no lazy globals in library crates. The only exception is `crates/raa-alloc-count`, linked as `#[global_allocator]` only by raa-bench and by demos with the feature `alloc-count`. Shared state between a feature and its handle goes through `Rc<RefCell<..>>` or `Rc<Cell<..>>`.
- The v0 frame path makes no heap allocation after warm-up (Q4): `Scene::update_world`, `Scene::build_render_list`, `FramePrep::prepare_cpu`, and RAA's code inside `FrameGraph::execute` and `WebRuntime::frame`. Allocations inside wgpu or web-sys are reported as `allocsPerFrameTotal`, as information. P2 crates preallocate and reuse their buffers.
- `unsafe` needs a `// SAFETY: <reason> (test: <test_fn_name>)` comment on the line directly above, and the named test must exist (`scripts/check.sh`). The only unsafe code is `unsafe impl GlobalAlloc for CountingAlloc`.

### Testing
- Unit tests go in `#[cfg(test)] mod tests`, integration tests in `crates/<c>/tests/<topic>.rs`. Test names are snake_case and describe the behaviour.
- Native GPU tests call `raa_core::testing::software_gpu()`. Without an adapter they print `SKIP <test name>: no software adapter (<reason>)` to stderr and return. They never fail for lack of an adapter, and never skip silently.
- Shader code is tested by naga validation of every composed variant (`variants.rs`). GPU algorithms have CPU twins tested on the CPU.
- Code generates test geometry and assets. JSON text fixtures are allowed, binary fixtures never are.

### Web build and demo pages
- `scripts/build-web.sh <crate-dir> [--features <list>]` runs these steps:
  1. `cargo build --profile wasm-release --target wasm32-unknown-unknown -p <crate>`.
  2. `wasm-bindgen --target web --out-dir dist/<dir>/pkg`.
  3. Copies `<crate-dir>/web/*` and `web/common/*` (into `common/`).
  4. Writes `dist/<dir>/sizes.json` (S7).
- Demo crate pattern:
  - `#[wasm_bindgen] pub async fn start(canvas, options) -> Result<Demo, JsValue>`;
  - `impl Demo { frame(now_ms), resize(w, h), stats(), async probe_non_empty(now_ms) }`;
  - a `raa_web::WebApp` driven by a `WebRuntime`;
  - `web/{index.html, main.js, nmc.gpu.json}`, which register `window.nmcGpu.ops` through `common/nmc-gpu.js`;
  - optional feature `alloc-count`.
- `node web/serve.mjs <dir> [--port N]` binds 127.0.0.1 only (port 0 = ephemeral). It sends correct MIME types (including `application/wasm`) and `Cache-Control: no-store`, and prints `serving <dir> on http://127.0.0.1:<port>/`.

### Browser captures (GPU client only)
- The project has no GPU. Browser rendering goes only through nomorechat's GPU client (`NMC_GPU_CLIENT`):
  - `scripts/gpu-capture.mjs` for demos and example-basic (S6);
  - `scripts/lookdev-capture.mjs` for the lookdev and sdfm-compare pages (S14).
- Both publish `dist/<page>/`, render, write the committed JSON record and copy the PNG to `proofs/captures/`. The PNGs are gitignored review artefacts.
- Linux Chrome is never launched in the sandbox, and `NMC_GPU_TOKEN` is never printed. When the client is unavailable, the task stops visibly and writes a dogfood entry. It never substitutes CPU output.
- Hardware rule: `hardware = !isFallbackAdapter && none of /swiftshader|llvmpipe|lavapipe|software|basic render/i matches vendor, architecture, device or description`. Every report states in words whether the adapter is hardware or software.
- Timing:
  - `firstFrameMs` is the navigation-relative `performance.now()` when `onSubmittedWorkDone` of frame 0 resolves;
  - `cpuMs` is the delta around one frame call;
  - reports give the median and p95 over `frames` measured frames after `warmup` frames.

### Judge (lookdev look)
- Command: `python3 "$NMC_JUDGE_CLIENT" <capture.png> --description <variant description> --reference <ref-prefix><ref>.png ... --min-score JUDGE_MIN_SCORE`. Timeout 900 s.
- The score is the JSON `score`, else the last `JUDGE PASS|FAIL: N/10` line, else the last `score: N`. The stored output is redacted (paths and env values) and kept within 4000 characters.
- Pass = exit 0. `--judge-record-only` records a low score without failing the capture run. The milestone proof's `judge` check still requires exit 0.
- Lookdev captures render at a fixed time with dt 0, so jiggle springs never move in a capture. A scene that shows jiggle in a capture writes the jiggle record directly through `setInstanceExtras` (floats 8..11). Springs (`setJiggle`) are for interactive loops with a real dt.

### Values (decided once, cited everywhere)
| name | value |
|---|---|
| WASM_GZIP_MAX | 300000 bytes per shipped .wasm (gzip level 9 via node zlib), raw size reported next to it |
| FIRST_FRAME_MAX_MS | 500 |
| CPU_MS_MEDIAN_MAX | 4.0 ms median CPU per frame, 1000 instances, hardware WebGPU only |
| ALLOC_RUN | 1000 instances, 120 warm-up frames, 600 measured frames, allowed allocations = 0 |
| BROWSER_RUN | warmup 60, frames 300 |
| CAPTURE | canvas 1280x720, devicePixelRatio 1; resize cycle 1280x720 to 960x540 (30 frames) and back |
| CLEAR_COLOR | linear (0.05, 0.06, 0.08, 1.0) |
| QUALITY | `Quality::LOW` = 8, `LOD0` = 24, `HIGH` = 64 segments; clamped to [3, 256] |
| TRIANGLE_BUDGET_DEFAULT | 1500 triangles per asset at LOD0 |
| ASSET_CAPS | nodes <= 64; nesting level <= 8 (roots are level 0); quaternion norm within 1e-3 of 1; scale components > 0 |
| WELD_EPS | 1e-6 x max(1, bbox diagonal) |
| NORMAL_RULE | per vertex: abs(len - 1) <= 1e-4; angle to the area-weighted mean of incident face normals <= 45 degrees; inward count = 0 |
| SDF_REF_TOL | uniquePositions and triangles equal; bbox coordinate diff <= 1e-5 x max(1, diagonal); area and volume relative diff <= 1e-4 |
| CSG_EPS | 1e-6 x max(1, diagonal of host bbox union tool bbox); volume identity relative diff <= 1e-4 |
| HDR_FORMAT | `Rgba16Float` (`raa_pipeline::HDR_FORMAT`) |
| MSAA | forward samples 4 by default, 1 allowed (`LitOptions.msaa`, JS `msaa`) |
| TONEMAP_DEFAULT | mapper agx, look none, exposure 0 EV, dither on |
| WEBGPU_OPTIONAL | `FLOAT32_FILTERABLE`, `RG11B10UFLOAT_RENDERABLE`: requested when the core adapter has them, never required |
| SKY_ENV | sky cube 256 px per face, HDR_FORMAT, 6 mips |
| LOOKDEV_CAPTURE | 1280x720, 30 warm-up frames + 1 frame, all at the variant's fixed `timeMs` (default 1000) |
| JUDGE_MIN_SCORE | 6 (out of 10) |

### Proofs
- One proof script per milestone, `proofs/mN-<name>.mjs`, built on `proofs/lib/proof.mjs` (P2 proofs also use `proofs/lib/p2.mjs`). Run it as `node proofs/mN-<name>.mjs [--quick|--full] [--only <regex>]`. The default is `--quick`. A proof has at most 5 checks.
- Each check prints `CHECK <label> | measured: <value> | threshold: <threshold> | PASS|FAIL`. The script exits non-zero at the first failing selected check. `--only` also reaches case-level runners.
- Independent cases run in parallel, up to `NMC_CPUS` (1 when unset), never more.
- Wall budget: at most 3 minutes on a warm cargo cache, and each artefact is built once per run. A threshold is never lowered.
- Timing thresholds exist only for FIRST_FRAME_MAX_MS and CPU_MS_MEDIAN_MAX.
- P2 milestone proofs check `unit`, `wasm-check`, `api-doc`, `captures` and, when variants are judged, `judge`.
- Look is never measured by a proof, except through the judge on `judge: true` variants. The owner reviews the captures.
- Dogfood: friction caused by nomorechat goes in `docs/dogfood/mN.md`: what broke, what was done by hand and what nomorechat must do instead.

### Commits and scope
- One commit per task, never pushed. No binaries or asset files (`scripts/check-no-assets.sh`).
- A task edits only files of modules owned by its milestone. Fixing another module needs the lead's explicit authorisation and stays minimal.
- Neutrality: the public API, docs and code comments in guarded paths never name a consumer game or game concept. `scripts/check.sh` rejects `royal|fiesta|monster|racer` (case-insensitive) in crates/, demos/, tests/, packages/, examples/ and docs/API.md, comments included.
- The JS API grows additively: each P2 milestone adds `packages/raa/types/mNN-*.d.ts` and `docs/api/mNN-*.md`. `index.d.ts` and `docs/API.md` stay API v0.

### Quality bar
Criteria controlled by commands:
- **Q1 Clean build:** `scripts/check.sh` exits 0.
- **Q2 Size:** every shipped .wasm has gzip <= WASM_GZIP_MAX. Raw and gzip sizes are reported.
- **Q3 Startup and speed:** every BrowserResult has `firstFrameMs` <= FIRST_FRAME_MAX_MS. The 1000-instance scene has `cpuMsMedian` <= CPU_MS_MEDIAN_MAX with `hardware: true`.
- **Q4 Frame loop:** `raa-bench frame-allocs` reports 0 allocations over ALLOC_RUN.
- **Q5 Primitive meshes:** the 8 kinds at LOW, LOD0 and HIGH on both sets are closed with positive volume, satisfy NORMAL_RULE, hash identically across runs and match the SDF Modeler reference within SDF_REF_TOL (48 cases).
- **Q6 Assets:** canonical prims v0 round-trips byte-identical. Invalid cases are refused with their code. The budget is enforced.
- **Q7 Booleans:** supported CSG cases are watertight within CSG_EPS. Unsupported ones return their typed refusal.
- **Q8 Extensibility and API:** raa-ext-test extends RAA with no edit to other crates. Every exported JS name is documented (`docs/API.md` for index.d.ts, `docs/api/<tag>-*.md` for `types/<tag>-*.d.ts`, the raa-sdfm README for its index.d.ts). The neutrality guard passes.
- **Q9 Honest GPU labelling:** every record names the adapter and states `hardware`.
- **Q10 Lookdev records:** every declared variant of a milestone's scenes has an S14 record with `nonEmpty` and `hardware` true, and no GPU errors.
- **Q11 Judged look:** every `judge: true` variant has `judge.exit == 0` against its references.
- **Q15 Feature toggles:** every toggleable feature has variants `<feature>-on` and `<feature>-off`, and the frame stays valid with the feature disabled (S12).

Review criteria. The owner answers yes or no on each saved capture:
- **R1-R6 (base M1-M6):**
  - a centred solid triangle;
  - 1000 lit instances with correct occlusion, flat versus PBR-lite, nothing stretched after resize;
  - 8 recognisable kinds per quality without cracks;
  - assets matching their composition;
  - clean boolean cuts;
  - the example page rendered through `@etcito/raa` only.
- **P2 lookdev:**
  - each on/off pair shows what its feature adds, with no artefacts (banding, acne, haloes, flicker between pairs);
  - **R12:** sdfm-compare shows RAA's rendering beside SDF Modeler's for the same asset and camera, with matching shape and materials.

## S1. GPU layouts
All layouts are little-endian, written with `to_le_bytes`, with no unsafe code. These are the v0 forward renderer layouts (raa-render). The lit renderer extends them in S10 and keeps the Vertex and Instance layouts.

- **Vertex** (stride 24 B): `position: f32x3` at offset 0 (location 0), `normal: f32x3` at offset 12 (location 1). Indices are `u32`, triangle list.
- **Instance** (stride 128 B, `raa_render::INSTANCE_STRIDE`, step mode Instance):
  - `model: 4 x f32x4` at offset 0 (locations 2-5);
  - `normal_matrix: 3 x f32x4` at offset 64 (locations 6-8): the columns of the inverse-transpose of the upper 3x3, w = 0;
  - `material_index: u32` at offset 112 (location 9), then 12 B of padding.
- **MaterialGpu** (48 B), read-only storage `array<MaterialGpu>` at group 0, binding 1:
  - `base_color: vec4` (linear rgb, a = 1);
  - `emissive: vec4` (linear, w = 0);
  - `params: vec4` = (roughness clamped to [0.04, 1], metallic in [0, 1], model 0 Flat / 1 PbrLite, 0).
- **FrameUniform** (144 B, `FRAME_UNIFORM_SIZE`), uniform at group 0, binding 0:
  - `view_proj` at offset 0 and `camera_pos` at 64;
  - `light_dir` at 80 (normalised travel direction, w = 0);
  - `light_color` at 96 (rgb x intensity) and `ambient` at 112 (rgb x intensity);
  - `flags: vec4<u32>` at 128 (x = 1: the shader encodes linear to sRGB).
- **Shading:**
  - Flat = `base_color + emissive`.
  - PbrLite = `ambient * base * (1 - 0.5 metallic) + (Lambert * (1 - metallic) + GGX specular, Schlick Fresnel F0 = mix(0.04, base, metallic), Smith-Schlick G) * light_color * max(dot(n, -light_dir), 0) + emissive`.
- Integer and flat varyings use `@interpolate(flat, either)`.

## S2. Mesh semantics (`raa_geom::MeshData`)
- A mesh is an indexed triangle list in the mesh's own space.
- `positions.len() == normals.len()`, `indices.len() % 3 == 0`, all indices are in range and all values are finite.
- Hard edges duplicate vertices. Smooth surfaces share them.
- Closedness is judged on the position-welded topology (WELD_EPS): every undirected edge is used by exactly 2 triangles, in opposite directions. Signed volume > 0 means outward winding.
- Attributed meshes from JS (S13) need not be closed.

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

## S10. P2 GPU layouts (lit pipeline)
Little-endian, written with `to_le_bytes`. Owned by raa-mesh (streams) and raa-lit (buffers, bind groups). Every P2 shader uses exactly these.

**Vertex and instance streams**
- Stream 0 = the S1 Vertex (24 B, locations 0-1).
- Stream 1 (40 B):
  - `uv: f32x2` at 0 (location 10);
  - `tangent: f32x4` at 8 (location 11; xyz unit, w = ±1);
  - `color: f32x4` at 24 (location 12; linear rgba).
- Defaults: uv (0, 0); tangent = `default_tangent(normal)`; colour (1, 1, 1, 1). Primitive and prims-asset meshes get the defaults.
- The instance stream is the S1 Instance (128 B, locations 2-9) in draw order. `@builtin(instance_index)` indexes the extras array.

**Group 0 (frame)**: 0 `FrameGpu` uniform (512 B), 1 `array<SurfaceMaterialGpu>`, 2 `array<InstanceExtraGpu>` (read-only storage).

**FrameGpu** (512 B)
- Matrices: `view_proj` @0, `view` @64, `proj` @128, `inv_proj` @192, `inv_view` @256.
- `camera_pos` @320 (w = time s).
- `sun_dir` @336: travel direction, w = sun angular radius (default 0.0047).
- `sun_color` @352 and `ambient` @368: rgb x intensity.
- `viewport` @384: (w, h, 1/w, 1/h).
- `clip` @400: (near, far, frame index, dt s).
- `flags: vec4<u32>` @416: x = 1 to encode sRGB in tonemap, y = msaa, z = enabled-feature bits (1 << S12 row index), w = 0.
- `exposure` @432: (ev, 2^ev, mapper 0 agx / 1 aces / 2 linear, look 0 none / 1 punchy / 2 golden).
- `reserved` @448..512, zero.

**SurfaceMaterialGpu** (256 B)
- `base` @0: linear rgb, a = opacity.
- `emissive` @16: linear rgb x strength.
- `surface` @32: (roughness clamped to [0.03, 1], metallic, clearcoat, clearcoat roughness).
- `sheen` @48: (rgb, roughness).
- `transmission` @64: (transmission, ior, thickness m, 0).
- `attenuation` @80: (rgb, distance m).
- `header: vec4<u32>` @96: (model 0 flat / 1 standard / 2 function, function id, flags, 0).
- `p: array<vec4, 8>` @112.
- `fur` @240: (length m, density per cm², droop, shells).
- Flags: 1 TRANSPARENT, 2 TRANSMISSIVE, 4 FUR, 8 DOUBLE_SIDED, 16 NO_SHADOW, 32 VERTEX_FN, 64 UNLIT.
- A v0 Material maps to model 0 or 1.

**InstanceExtraGpu** (128 B)
- `tint` @0: default 1.
- `squash` @16: (axis, amount), default (0, 1, 0, 0).
- `jiggle` @32: (local offset m, height m), default 0.
- `params: array<vec4, 4>` @48.
- `reserved` @112.
- Keyed by the scene instance tag, so every sub-mesh draw of one JS instance shares it.

**Group 1 (lighting)**, used by the forward, transparent, fur and particle passes. It is built by `LightingResources`, with 1x1 dummies until a feature sets a resource:
- 0 `shadow_map: texture_depth_2d_array` (dummy 1);
- 1 `shadow_cmp` (LessEqual, linear);
- 2 `ShadowGpu` (320 B): `cascade_view_proj[4]` @0, `splits` @256, `params` @272 (penumbra scale, normal offset texels, texel size, enabled), `params2` @288 (light size, blend band, 0, 0), reserved @304;
- 3 `env_specular: texture_cube` (dummy = ambient);
- 4 `EnvSh` (160 B): 9 x vec4 SH9 @0, `info` @144 (specular, diffuse, max mip, enabled);
- 5 `brdf_lut` (Rg16Float scale/bias, dummy (1, 0));
- 6 `linear_clamp`;
- 7 `ao` (dummy 1);
- 8 `contact` (dummy 1);
- 9 `opaque_color` (HDR_FORMAT with mips, dummy black);
- 10 `opaque_depth` (R32Float, textureLoad, dummy 1);
- 11 `EnvGpu` (256 B):
  - `sun_disc` @0 (radius, intensity);
  - `zenith` @16, `horizon` @32, `ground` @48;
  - `sky_params` @64 (intensity, horizon softness, background mode 0 constant / 1 sky);
  - `background` @80 (display-referred);
  - `clouds` @96 (coverage, density, altitude, scale), `clouds2` @112 (wind x, wind z, silver lining), `cloud_color` @128;
  - `fog` @144 (density, falloff, base height, start), `fog_color` @160 (rgb, inscatter);
  - `aerial` @176 (strength, distance scale);
  - reserved @192.

**Group 2** is private to the pass that binds it. **Group 3** is reserved for extensions.

## S11. Shader and material function contract (raa-shader)
A composed module = `PRELUDE` (S10 structs, groups 0-1) + `LIB` + one material function + the winning provider of each slot + the pass entry points. Variant key = (material fn id, pass kind, msaa, flags, slots hash).
```wgsl
struct MaterialInput { position: vec3<f32>, local_position: vec3<f32>, normal: vec3<f32>, tangent: vec4<f32>, uv: vec2<f32>, color: vec4<f32>, view_dir: vec3<f32>, time: f32, front_facing: bool, frag_coord: vec4<f32>, instance_params: array<vec4<f32>, 4>, record: SurfaceMaterialGpu }
struct MaterialOutput { base_color: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32>, normal: vec3<f32>, ao: f32, opacity: f32, clearcoat: f32, clearcoat_roughness: f32, sheen_color: vec3<f32>, sheen_roughness: f32, transmission: f32, ior: f32, thickness: f32, attenuation_color: vec3<f32>, attenuation_distance: f32 }
struct VertexMaterialInput { local_position: vec3<f32>, normal: vec3<f32>, uv: vec2<f32>, color: vec4<f32>, time: f32, instance_params: array<vec4<f32>, 4>, record: SurfaceMaterialGpu }
struct VertexMaterialOutput { local_position: vec3<f32>, normal: vec3<f32> }
fn material_main(in: MaterialInput) -> MaterialOutput            // required
fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput // optional (sets VERTEX_FN)
```
**Material functions**
- Positions and normals are in world space. `in.normal` is the interpolated geometric normal, flipped when back-facing and DOUBLE_SIDED. `MaterialOutput.normal` is the unit shading normal.
- `raa_default_output(in)` fills every field from `in.record`; the `standard` built-in is this alone. The prelude applies the tint after `material_main`.
- User rules (`material.fn_invalid` with the reason; naga messages with line:col):
  - defines `material_main` with that signature;
  - no `@group`, `@binding`, `@vertex`, `@fragment`, `@compute`, `var<storage`, `var<uniform`, `var<workgroup`, `enable` or `override`;
  - no identifier starting `raa_`;
  - <= 65536 bytes;
  - the composed Forward variant validates.
  A duplicate name gives `material.fn_duplicate`.
- LIB helpers: `raa_hash3`, `raa_noise3`, `raa_fbm3(p, octaves)`, `raa_voronoi3(p) -> vec2`, `raa_triplanar_weights`, `raa_perturb_normal`, `raa_luma`, `raa_srgb_to_linear`.

**Slots.** The default providers (priority 0) live in raa-lit. The highest priority wins; equal priority on one slot gives `shader.slot_conflict`.
- `Deform`: `raa_slot_deform(p, n, x: InstanceExtraGpu, t) -> VertexMaterialOutput` (local space, after material_vertex; default identity).
- `Shadow`: `raa_slot_shadow(world_pos, n, frag) -> f32` (sun visibility including contact; default 1).
- `Ao`: `raa_slot_ao(frag) -> f32` (default 1).
- `Ambient`: `raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3` (default = the S1 ambient term).
- `Direct`: `raa_slot_direct(s, c, l, radiance) -> vec3` (default = S1 GGX + Lambert).
- `Transmission`: `raa_slot_transmission(s, c) -> vec3` (default 0; reached only by TRANSMISSIVE materials).
- `Fog`: `raa_slot_fog(color, world_pos, c) -> vec3` (transparent, fur and blended particles; default identity).
- `ShadeCtx { position, view_dir, frag, n_dot_v, ao, shadow, f0 }`.

**Final colour**
- Opaque: `emissive + raa_slot_direct(s, c, -sun_dir, sun_color * shadow) + raa_slot_ambient(s, c) * min(s.ao, ao)`.
- Transparent passes add `raa_slot_transmission` and apply `raa_slot_fog`.
- Flat or UNLIT returns `base + emissive`.

**Pass kinds**
- `Prepass`: vs + fs writing the S12 normal and material targets.
- `Forward`: vs + fs to HDR, MSAA.
- `Shadow`: vs only, with material_vertex and deform.
- `Transparent`: vs + fs appending to the OIT list.
- `FurShell`: vs extruding shells + fs with alpha-to-coverage.

## S12. Pipeline passes and targets (raa-pipeline)
- One raa-core RenderPass, `raa.pipeline` (writes BACKBUFFER), runs the enabled feature passes by `order`. The order is overridable from JS. Required passes cannot be disabled, and `prepass` < `forward` < `tonemap` with `tonemap` last (`pipeline.required_order`).

| order | name | role | crate |
|---|---|---|---|
| 10 | sky.env | pass | raa-sky |
| 20 | shadow.csm | pass | raa-shadow |
| 30 | prepass | pass, required | raa-lit |
| 40 | ao.gtao | pass | raa-gtao |
| 45 | shadow.contact | pass | raa-contact |
| 50 | particles.sim | pass | raa-particles |
| 60 | forward | pass, required | raa-lit |
| 500 | fur.shells | contributor of forward | raa-fur |
| 600 | particles.opaque | contributor of forward | raa-particles |
| 900 | sky.background | contributor of forward | raa-sky |
| 65 | ssr | pass | raa-ssr |
| 68 | fog | pass | raa-fog |
| 69 | sunshafts | pass | raa-shafts |
| 70 | opaque.copy | pass | raa-transmission |
| 80 | transparent.oit | pass | raa-oit |
| 85 | particles.blend | pass | raa-particles |
| 110 | dof | pass | raa-dof |
| 120 | bloom | pass | raa-bloom |
| 130 | grading | pass | raa-grading |
| 200 | tonemap | pass, required | raa-tonemap |

- Contributor orders are positions inside the forward MSAA pass, after the opaque draws. The row index of this table is the feature-bit index of `FrameGpu.flags.z`.
- Targets (`raa_pipeline::Targets`, surface-sized, recreated on resize):
  - `depth` (Depth32Float, single-sample, written by prepass, RENDER_ATTACHMENT | TEXTURE_BINDING);
  - `normal` (Rgba16Float: view-space normal, roughness);
  - `material` (Rgba8Unorm: F0 sRGB-encoded, metallic);
  - `hdr` chain A/B (HDR_FORMAT, RENDER_ATTACHMENT | TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC | COPY_DST).
- HDR chain use:
  - `forward` resolves into `hdr.dst()` and swaps;
  - each later HDR pass reads `src`, writes `dst` and swaps;
  - `tonemap` reads `src`.
- Disabling a pass calls `set_enabled(false, lighting)`, which resets its group-1 resources to dummies, so the frame stays valid.
- Background rule: with EnvGpu background mode 0 (no sky contributor enabled), tonemap writes `background` unchanged where `depth == 1`. An empty scene therefore equals CLEAR_COLOR, and `probeNonEmpty` keeps its v0 meaning.

## S13. Mesh API data (JS `addMesh`, raa-mesh)
`MeshDesc { positions: Float32Array (3n), normals: Float32Array (3n), uvs?: Float32Array (2n), tangents?: Float32Array (4n), colors?: Float32Array (4n linear) | Uint8Array (4n sRGB rgba8), indices: Uint32Array (3m), subMeshes?: [{ first (index offset, multiple of 3), count (multiple of 3, > 0), material?: number }] }`.

- Absent subMeshes means one sub-mesh over all indices. Sub-meshes must not overlap and must cover only valid ranges.
- Tangents absent with uvs present are generated. Absent without uvs, they use the S10 defaults.
- Refusal codes:
  - `mesh.empty`;
  - `mesh.length_mismatch` (names the array);
  - `mesh.not_finite` (array and index);
  - `mesh.index_range` (first bad index);
  - `mesh.invalid_normal` (zero length);
  - `mesh.invalid_tangent` (w not ±1, or zero xyz);
  - `mesh.submesh_range`, `mesh.submesh_overlap`;
  - `api.unknown_handle` (sub-mesh material).

## S14. Lookdev scenes and records
**Scene module** `examples/lookdev/scenes/mNN-<name>.js` (also `examples/sdfm-compare/scenes/`). It has no top-level side effects, so node can import it for metadata.
- `export default { id, milestone: 'mNN', title, engine?, variants, async build(engine, lib, variant), frame?(engine, timeMs, variant) }`.
- Each variant: `{ id (^[a-z0-9-]+$), description, references: ['ref-1', ...], judge: boolean, timeMs? (1000), engine?, passes?: { disable?, order? }, settings? }`.
- `lib` holds the page helpers. A toggleable feature has variants `<feature>-on` and `<feature>-off` (Q15).

**Page op** `window.nmcGpu.ops.run({ scene, variant, width, height })`:
- creates a fresh engine with the variant's engine options merged over the scene's, builds, applies the passes, and renders LOOKDEV_CAPTURE;
- resolves `{ adapter, hardware, canvas, nonEmpty, firstFrameMs, warmup, frames, cpuMsMedian, cpuMsP95, instances, drawCalls, triangles, allocsPerFrameTotal: null, resize: null, scene, variant, passes: PassInfo[], gpuErrors }`;
- non-empty gpuErrors reject the op.

**Record** `proofs/results/lookdev/<scene>--<variant>.json`:
- `{ scene, variant, milestone, capturedAt, adapter, hardware, canvas, nonEmpty, passes: [enabled names in order], instances, triangles, firstFrameMs, wasm: {raw, gzip, sha256}, capture: 'proofs/captures/lookdev/<scene>--<variant>.png', captureSha256, judge }`;
- `judge` is `{ description, references, minScore: 6, exit, score|null, output (redacted, <= 4000 chars), at }` or null;
- Pass = `judge.exit == 0`.

`dist/lookdev/scenes.json` and `dist/sdfm-compare/scenes.json`: `[{ id, milestone, file }]` sorted by id.

## S15. sdfm adapter shapes (`@etcito/raa-sdfm`)
The adapter reads the shared `sdfm.game-asset` v1 file.
- Fields: format, version 1, game, id, name, category, world, mesher, document, collision, anchors, bounds, hash.
- Rules: canonical key order, document <= 2 MB, finite numbers, id `^[a-z][a-z0-9]*(\\.[a-z0-9-]+)+$`.

RAA uses only:
- `document`, meshed by MAMA in workers;
- `mesher`, passed to MAMA unchanged;
- `anchors` and `bounds`;
- `hash`, the cache key: the recomputed sha256 of the canonical document must match.

`collision` is passed through untouched and never used for rendering. MAMA's typed mesher refusals surface as `sdfm.mesher_refused`, with MAMA's message naming the group.

Codes: `sdfm.format`, `sdfm.version`, `sdfm.schema` (JSON pointer), `sdfm.not_finite`, `sdfm.id`, `sdfm.cap_document`, `sdfm.hash_mismatch`, `sdfm.mesher_refused`, `sdfm.material_unmappable`, `sdfm.worker`, `sdfm.disposed`.

## Module ownership

| Module | Files | Consumes |
|---|---|---|
| workspace | `Cargo.toml`<br>`Cargo.lock`<br>`.gitignore` | none |
| tooling | `scripts/check.sh`<br>`scripts/build-web.sh`<br>`scripts/check-no-assets.sh`<br>`scripts/hooks/pre-commit`<br>`scripts/gpu-capture.mjs`<br>`web/serve.mjs`<br>`web/common/nmc-gpu.js`<br>`web/common/run-loop.js`<br>`proofs/lib/proof.mjs` | workspace |
| proof-m1 | `proofs/m1-boot.mjs`<br>`proofs/results/m1-boot.json`<br>`docs/dogfood/m1.md` | tooling, demo-m1-boot |
| proof-m2 | `proofs/m2-scene.mjs`<br>`proofs/results/m2-scene.json`<br>`docs/dogfood/m2.md` | tooling, raa-bench, demo-m2-scene |
| sdf-ref-tool | `tools/sdf-ref/gen-leaf-fixtures.mjs`<br>`tools/sdf-ref/README.md` | none |
| proof-m3 | `proofs/m3-prims.mjs`<br>`proofs/results/m3-prims.json`<br>`proofs/results/m3-prims-table.json`<br>`docs/dogfood/m3.md` | tooling, raa-prims, demo-m3-prims |
| proof-m4 | `proofs/m4-loader.mjs`<br>`proofs/results/m4-asset.json`<br>`docs/dogfood/m4.md` | tooling, raa-loader, demo-m4-asset |
| proof-m5 | `proofs/m5-booleans.mjs`<br>`proofs/results/m5-booleans.json`<br>`proofs/results/m5-csg-cases.json`<br>`docs/dogfood/m5.md` | tooling, raa-csg, demo-m5-booleans |
| raa-js | `crates/raa-js/Cargo.toml`<br>`crates/raa-js/README.md`<br>`crates/raa-js/build.rs`<br>`crates/raa-js/src/lib.rs`<br>`crates/raa-js/src/engine.rs`<br>`crates/raa-js/src/convert.rs`<br>`crates/raa-js/src/state.rs`<br>`crates/raa-js/src/feature_map.rs`<br>`crates/raa-js/src/api/m07_meshes.rs`<br>`crates/raa-js/src/api/m07_pipeline.rs`<br>`crates/raa-js/tests/convert.rs`<br>`crates/raa-js/tests/api_registry.rs` | raa-core, raa-web, raa-math, raa-geom, raa-material, raa-scene, raa-prims, raa-loader, raa-csg, raa-mesh, raa-shader, raa-pipeline, raa-lit, raa-tonemap |
| js-package | `scripts/build-package.sh`<br>`packages/raa/package.json`<br>`packages/raa/index.js`<br>`packages/raa/index.d.ts`<br>`packages/raa/README.md`<br>`packages/raa/LICENSE` | raa-js, api-m7 |
| project-docs | `README.md`<br>`docs/API.md`<br>`docs/ARCHITECTURE-notes.md` | js-package, raa-js |
| proof-m6 | `proofs/m6-api.mjs`<br>`proofs/results/example-basic.json`<br>`proofs/results/m6-sizes.json`<br>`docs/dogfood/m6.md` | tooling, raa-ext-test, js-package, example-basic, project-docs |
| api-m7 | `packages/raa/types/m07-meshes-hdr.d.ts`<br>`docs/api/m07-meshes-hdr.md` | raa-js |
| lookdev-page | `examples/lookdev/index.html`<br>`examples/lookdev/main.js`<br>`examples/lookdev/nmc.gpu.json`<br>`examples/lookdev/lib/geometry.js`<br>`examples/lookdev/lib/objects.js`<br>`examples/lookdev/lib/runner.js` | js-package, api-m7, tooling |
| lookdev-tools | `scripts/build-lookdev.sh`<br>`scripts/lookdev-capture.mjs`<br>`proofs/lib/p2.mjs`<br>`proofs/lib/p2-smoke.mjs` | tooling, js-package, lookdev-page |
| proof-m7 | `proofs/m7-meshes-hdr.mjs`<br>`examples/lookdev/scenes/m07-base.js`<br>`docs/dogfood/m7.md` | lookdev-tools, lookdev-page, raa-js |
| api-m8 | `crates/raa-js/src/api/m08_sky_ibl.rs`<br>`packages/raa/types/m08-sky-ibl.d.ts`<br>`docs/api/m08-sky-ibl.md` | raa-js, raa-sky, raa-ibl |
| proof-m8 | `proofs/m8-sky-ibl.mjs`<br>`examples/lookdev/scenes/m08-sky-ibl.js`<br>`docs/dogfood/m8.md` | lookdev-tools, api-m8 |
| api-m9 | `crates/raa-js/src/api/m09_shadows_ao.rs`<br>`packages/raa/types/m09-shadows-ao.d.ts`<br>`docs/api/m09-shadows-ao.md` | raa-js, raa-shadow, raa-gtao, raa-contact |
| proof-m9 | `proofs/m9-shadows-ao.mjs`<br>`examples/lookdev/scenes/m09-shadows-ao.js`<br>`docs/dogfood/m9.md` | lookdev-tools, api-m9 |
| api-m10 | `crates/raa-js/src/api/m10_materials.rs`<br>`packages/raa/types/m10-materials.d.ts`<br>`docs/api/m10-materials.md` | raa-js, raa-matfn, raa-brdf, raa-fur, raa-bloom |
| proof-m10 | `proofs/m10-materials.mjs`<br>`examples/lookdev/scenes/m10-materials.js`<br>`docs/dogfood/m10.md` | lookdev-tools, api-m10 |
| api-m11 | `crates/raa-js/src/api/m11_transmission.rs`<br>`packages/raa/types/m11-transmission.d.ts`<br>`docs/api/m11-transmission.md` | raa-js, raa-transmission, raa-oit |
| proof-m11 | `proofs/m11-transmission.mjs`<br>`examples/lookdev/scenes/m11-transmission.js`<br>`docs/dogfood/m11.md` | lookdev-tools, api-m11 |
| sdfm-compare | `scripts/build-sdfm-compare.sh` | raa-sdfm-pkg, lookdev-tools |
| proof-m12 | `proofs/m12-sdfm.mjs`<br>`examples/sdfm-compare/scenes/m12-fixtures.js`<br>`docs/dogfood/m12.md` | lookdev-tools, sdfm-compare |
| api-m13 | `crates/raa-js/src/api/m13_liquids_particles.rs`<br>`packages/raa/types/m13-liquids-particles.d.ts`<br>`docs/api/m13-liquids-particles.md` | raa-js, raa-liquid, raa-particles |
| proof-m13 | `proofs/m13-liquids-particles.mjs`<br>`examples/lookdev/scenes/m13-liquids-particles.js`<br>`docs/dogfood/m13.md` | lookdev-tools, api-m13 |
| api-m14 | `crates/raa-js/src/api/m14_camera_effects.rs`<br>`packages/raa/types/m14-camera-effects.d.ts`<br>`docs/api/m14-camera-effects.md` | raa-js, raa-ssr, raa-dof, raa-fog, raa-shafts, raa-grading, api-m8, api-m10 |
| proof-m14 | `proofs/m14-camera-effects.mjs`<br>`examples/lookdev/scenes/m14-camera-effects.js`<br>`docs/dogfood/m14.md` | lookdev-tools, api-m14 |
| api-m15 | `crates/raa-js/src/api/m15_animation.rs`<br>`packages/raa/types/m15-animation.d.ts`<br>`docs/api/m15-animation.md` | raa-js, raa-deform |
| proof-m15 | `proofs/m15-animation.mjs`<br>`examples/lookdev/scenes/m15-waddle.js`<br>`docs/dogfood/m15.md` | lookdev-tools, api-m15 |

## Module: workspace
Files: `Cargo.toml`, `Cargo.lock`, `.gitignore`.

- `[workspace] resolver = "2"`, `members = ["crates/*", "demos/*", "tests/*"]`.
- `[workspace.package]`: version 0.1.0, edition 2024, license MIT, rust-version 1.96.0.
- `[workspace.dependencies]`:
  - the closed dependency list (Conventions), including `naga` (`wgsl-in`, default features off);
  - the pre-P2 internal crates by path (raa-core, raa-web, raa-math, raa-geom, raa-material, raa-scene, raa-render, raa-alloc-count, raa-prims, raa-loader, raa-csg).
- P2 crates reference each other through path dependencies.
- `[profile.wasm-release]` follows Conventions. `Cargo.lock` is committed and is updated by any task that adds a member. `.gitignore` is unchanged.

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
Files: `crates/raa-core/**`.

raa-core holds the device, surface, handles, a minimal frame graph and the error type. It depends on wgpu, plus web-sys and wasm-bindgen-futures on wasm32 and pollster on native.

- **`pub use wgpu;`** re-exports the exact wgpu version, so extension crates depend only on raa-* crates.

**`pub enum RaaError`**
- Variants: `NoAdapter(String)`, `RequestDevice(String)`, `Surface(String)`, `FrameGraph { code: &'static str, message: String }`, `Resource(String)`, `Gpu(String)`, `Unsupported(String)`.
- `code()` returns `core.no_adapter`, `core.request_device`, `core.surface`, the FrameGraph code, `core.resource`, `core.gpu` or `core.unsupported`.

**Adapter and GPU context**
- `AdapterSummary { vendor, architecture, device, description, backend, is_fallback }` with `is_hardware()`.
- `GpuOptions { power_preference, allow_software }`, default HighPerformance with allow_software true.
- `#[derive(Clone, Debug, Default)] pub struct DeviceRequest { pub optional_features: wgpu::Features, pub adapter_limits: bool }`.
- `GpuContext`: `device()`, `queue()`, `adapter()`, `summary()`, `features()` (the device's enabled features), `limits()`.
  - wasm32: `async for_canvas_with(canvas, &GpuOptions, &DeviceRequest) -> Result<(GpuContext, CanvasSurface), RaaError>`.
    - It requests a core adapter, never featureLevel compatibility.
    - The device gets `req.optional_features & adapter.features()`, plus `adapter.limits()` as required limits when `adapter_limits` is set.
    - `for_canvas` uses the default request.
    - A missing `navigator.gpu` gives `core.no_adapter`.
  - native: `async headless_with(&GpuOptions, &DeviceRequest)` follows the same rules. `headless` uses the default request. Without a hardware adapter and with allow_software set, it retries with `force_fallback_adapter = true`.

**Surface (wasm32)**
- `CanvasSurface`: `format()`, `is_srgb()`, `size()`, `resize(gpu, w, h)` (zero sizes clamp to 1).
- `acquire(gpu) -> Result<SurfaceFrame, RaaError>` reconfigures and retries once on Outdated or Lost. Usage is RENDER_ATTACHMENT | COPY_SRC.
- `SurfaceFrame { pub view }`: `texture()`, `present(self)`.

**Handles and ids**
- `Handle<T>` has `index()` and `generation()`.
- `Pool<T>`: `new`, `with_capacity`, `insert`, `get`, `get_mut`, `remove` (bumps the generation), `len`, `iter`. Free slots are reused LIFO.
- `MeshId(pub u32)`, `MaterialId(pub u32)`, `pub const CLEAR_COLOR`.

**Frame graph** (module `graph`, re-exported)
- Types:
  - `ResourceId` and `BACKBUFFER` (id 0, imported each frame);
  - `TargetSize { Surface, Fixed(w, h) }`, `TargetDesc { label, format, size, usage }`;
  - `PassId` and `PassBuilder { read, write }`.
- `trait RenderPass { name; declare; prepare (default Ok); record; resize (default no-op) }`.
- Contexts: `PrepareContext { gpu, size, frame }` and `RecordContext { gpu, encoder, size, frame }` with `view(r)`, `format(r)` and `is_first_write(r)`. A pass clears on first write and loads otherwise.
- `FrameGraph`: `new(backbuffer_format)`, `create_target`, `add_pass`, `add_pass_after`/`add_pass_before` (`framegraph.unknown_pass`), `compile()`, `pass_names()`, `execute(gpu, backbuffer_view, size) -> Result<wgpu::CommandBuffer, RaaError>`.
- Compile rules, in order:
  1. unique names (`framegraph.duplicate_pass`);
  2. known resources (`framegraph.unknown_resource`);
  3. every read of a non-BACKBUFFER target has an earlier write (`framegraph.read_before_write`);
  4. some pass writes BACKBUFFER (`framegraph.no_output`).
  Adding a pass invalidates compilation. Executing before compiling gives `framegraph.not_compiled`.
- Execution recreates Surface-sized targets and calls `resize` only when the size changes, then runs prepare on all passes and record in order into one encoder.

**Other items**
- `ClearPass::new(name, target, color)`.
- Native `OffscreenTarget { new, view, read_rgba8 }`.
- `non_empty(pixels, clear) -> bool`: true when any channel differs by more than 2.
- Native `testing::{software_gpu, skip}`.

Tests: handle generations, every graph compile code, after/before ordering, a ClearPass readback, `device_request_intersects_adapter_features` (SKIP without an adapter).

## Module: raa-web
Files: `crates/raa-web/**`.

The web runtime used by every demo and by raa-js.

**Types**
- `FrameInput { frame: u64, now_ms: f64, dt_ms: f64, size: (u32, u32) }`.
- `trait WebApp`:
  - `init(&mut self, gpu, graph, size, surface_format, surface_is_srgb)`;
  - `update(&mut self, gpu, &FrameInput) -> Result<FrameCounters, RaaError>`;
  - `resize` (default no-op).
- `FrameCounters { instances, draw_calls, triangles: u64 }`.
- `RuntimeOptions { width, height, gpu, stats_window (300), alloc_probe }`. Default 1280x720.
- `FrameStats { frames, first_frame_ms, cpu_ms_median, cpu_ms_p95, counters, allocs_last_frame }`.

**`WebRuntime`** (wasm32)
- `async new_with(canvas, app, opts, req: &raa_core::DeviceRequest)` creates the context through `GpuContext::for_canvas_with`, the surface and the FrameGraph, calls `app.init` and compiles. `new` uses the default request.
- `frame(now_ms)`: update, acquire, execute, submit, present. CPU ms goes into a preallocated ring. Frame 0 registers `on_submitted_work_done` for firstFrameMs.
- `resize(w, h)`, `stats()`, `adapter()`, `gpu()`, `app_mut()`.
- `async probe_non_empty(now_ms)`: renders one frame, reads the backbuffer back and applies `non_empty` against CLEAR_COLOR.

**JS conversion**
- `stats_to_js(&FrameStats, &AdapterSummary)`: the S6 adapter and stats fields, built with js_sys::Reflect.
- `error_to_js(&RaaError)`: a JS `Error` named `RaaError` with a `code` property.

On native the crate compiles to the stats module only.

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
Files: `crates/raa-scene/**`.

Depends on raa-math and raa-core.

**Transforms and ids**
- `Transform { translation, rotation, scale }` with `IDENTITY`, `from_trs`, `to_mat4`.
- `NodeId` and `InstanceId` are generational, with `to_bits() -> u64`.

**Camera**
- `Camera { eye, target, up, fov_y_rad, near, far, aspect }`. Default: eye (0, 2, 6), target 0, fov 60°, near 0.1, far 200, aspect 16/9.
- Methods: `view`, `projection` (perspective_rh_zo), `view_proj`, `set_viewport`.
- `OrbitController { target, distance, yaw_rad, pitch_rad, min_distance, max_distance }`:
  - `new`;
  - `rotate` (pitch clamped to ±89°);
  - `zoom` (factor, clamped);
  - `pan`;
  - `apply(&mut Camera)`: eye = target + distance x (cos p sin y, sin p, cos p cos y).

**Lights**
- `DirectionalLight { direction, color, intensity }`. Default: normalize(-0.4, -1, -0.3), white, 3.0.
- `Ambient { color, intensity }`. Default: (0.6, 0.65, 0.75), 0.3.

**Render list**
- `RenderItem { mesh, material, world }`, `Batch { mesh, first, count }`.
- `RenderList { items, batches }` with `with_capacity` and `clear`.

**`Scene { pub camera, pub light, pub ambient, .. }`**
- Construction: `new`, `with_capacity`.
- Nodes: `add_node(parent, local)`, `remove_node` (removes the subtree and its instances), `set_local`, `local`, `world`.
- Instances: `add_instance(node, mesh, material)`, `remove_instance`, `set_instance_visible`, `set_instance_material`.
- Tags: every instance carries a `tag: u32` (default 0). `set_instance_tag(id, tag)` and `instance_tag(id)` refuse stale ids with `scene.stale_instance`.
- Counts: `node_count`, `instance_count`.
- `update_world` processes parents before children and recomputes only dirty subtrees.
- `build_render_list(&mut RenderList)` sorts visible instances in place by (mesh, material, slot) and batches contiguous runs per mesh. It is deterministic and does not allocate when capacity suffices.
- `build_render_list_tagged(&mut RenderList, tags: &mut Vec<u32>)` gives identical items and order. `tags[i]` is the tag of `items[i]`.

**`SceneError`**: `scene.stale_node`, `scene.stale_instance`, `scene.parent_cycle`.

Tests: hierarchy, camera, render_list, tags (sorted order, stale refusal, removal).

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

**Command:** `node tools/sdf-ref/gen-leaf-fixtures.mjs --ref <sdfm checkout root> --out crates/raa-prims/tests/fixtures/sdf-leaf-reference.json`

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
Files: `crates/raa-js/{Cargo.toml, README.md, build.rs}`, `src/{lib, engine, convert, state, feature_map}.rs`, `src/api/*.rs`, `tests/{convert, api_registry}.rs`.

The wasm-bindgen engine (cdylib + rlib). `#[wasm_bindgen(js_name = Engine)] pub struct Engine` is a WebApp inside a WebRuntime.
- The runtime is created with `WebRuntime::new_with` and `DeviceRequest { optional_features: WEBGPU_OPTIONAL, adapter_limits: true }`.
- `EngineApp::init` creates `raa_pipeline::Pipeline`, `raa_lit::LitRenderer` and `raa_tonemap::TonemapPass`, runs `install_all`, then installs `raa.pipeline` in the runtime graph.
- The CsgResolver and the 8 prims are registered, with strict kinds.
- JS handles are monotonically increasing u32 values, never reused. Errors are thrown through `raa_web::error_to_js`.

**API v0** (unchanged names, arguments, defaults, results and codes):
- Lifecycle:
  - `static create(canvas, options)`. Options: maxInstances 16384, maxMaterials 256, powerPreference, and `msaa?: 1|4`, `toneMapper?: 'agx'|'aces'|'linear'`, `exposureEv?` (finite). Other values throw `api.invalid_argument`.
  - `resize`, `render(nowMs)`, `stats()`, `probeNonEmpty(nowMs)`, `dispose()`.
- Materials: `addMaterial(desc)`, `setMaterial(id, desc)`. Both map through `SurfaceMaterial::from_v0`.
- Meshes:
  - `primitiveMesh(kind, params, segments?)` and `loadPrimsAsset(json, segments?)`, uploaded as `AttributedMesh` with S10 default attributes;
  - `removeMesh(id)` (any mesh handle);
  - `spawnAsset(asset, parent?)`.
- Nodes: `addNode`, `setTransform`, `removeNode`, `setTransforms(nodes, trs)` (10 floats per node, `api.length_mismatch`).
- Instances:
  - `addInstance(node, mesh, material)` draws every sub-mesh with `material`;
  - each JS instance is an instance group: one scene instance per sub-mesh, tag = the JS handle;
  - `removeInstance` and `setVisible` act on the whole group.
- Camera and light: `setCamera`, `orbit`, `setOrbit`, `setLight`, `setAmbient`.
- `stats` and `probeNonEmpty` keep their meaning (S12 background rule).

**State**
- `src/state.rs`: `pub(crate) struct EngineState` holds the scene, list, tags, orbit, geometry, loaders, booleans, scratch, `lit: LitRenderer`, `pipeline: Rc<RefCell<Pipeline>>`, `tonemap: TonemapControl`, `features: FeatureMap` and the handle maps (meshes, materials, nodes, instance groups, assets). Helpers: `node(h)`, `material(h)`, `mesh(h)`, `instance_group(h)`.
- `src/feature_map.rs`: `FeatureMap`, type-keyed, with `insert<T>`, `get<T>`, `get_mut<T>` and `require_mut<T>(what)` (`api.feature_missing`).

**API files**
- `build.rs` lists `src/api/*.rs` sorted by file name and writes `$OUT_DIR/api_registry.rs`, with:
  - `#[path] mod <stem>;` per file;
  - `install_all(ctx: &mut InstallCtx)` and `frame_all(st, gpu, input)`, calling each file's `install` and `frame` in that order.
  - It emits `cargo:rerun-if-changed=src/api`.
- Each api file defines `pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError>`, `pub(crate) fn frame(st: &mut EngineState, gpu: &GpuContext, input: &FrameInput) -> Result<(), RaaError>` and `#[wasm_bindgen(js_class = Engine)] impl Engine` blocks using `Engine::with_state(&self, f)`.
- `InstallCtx { gpu, surface_format, pipeline, lit, features }`.
- `install_all` runs at init after `prepass`, `forward` and `tonemap` exist. `frame_all` runs in `update` before `lit.prepare`.
- Files: `m07_meshes`, `m07_pipeline` (api-m7), `m08_sky_ibl` (api-m8), `m09_shadows_ao`, `m10_materials`, `m11_transmission`, `m13_liquids_particles`, `m14_camera_effects`, `m15_animation`.

**`m07_meshes`**
- `addMesh(desc: MeshDesc) -> u32` (S13).
- `updateMesh(id, desc)`: instance groups of that mesh are rebuilt at the next frame, keeping node, visibility and tag.
- `meshInfo(id) -> {vertices, triangles, subMeshes: [{first, count, material|null}], bounds: {min, max}}`.
- `addMeshInstance(node, mesh) -> u32`: each sub-mesh with its own material, or the default material.

**`m07_pipeline`**
- `passes() -> PassInfo[]` (`{name, role: 'pass'|'contributor', enabled, order, required}` in execution order).
- `setPassEnabled(name, enabled)` (`pipeline.unknown_pass`, `pipeline.required_pass`).
- `setPassOrder(names)`: every pass-role name exactly once (`pipeline.order_incomplete`, `pipeline.unknown_pass`, `pipeline.required_order`).
- `setToneMapping({mapper?, exposureEv?, look?: 'none'|'punchy'|'golden', dither?})`.
- `setBackground(color)`: display-referred linear, finite, >= 0.
- `capabilities() -> {coreWebGpu: true, features: string[], msaa, hdrFormat: 'rgba16float', maxStorageBufferBindingSize}`.

Tests: `convert.rs` (v0 conversions), `api_registry.rs` (build.rs output lists every api file, install/frame order).

## Module: js-package
Files: `packages/raa/{package.json, index.js, index.d.ts, README.md, LICENSE}`, `scripts/build-package.sh`.

**Package.** `@etcito/raa` 0.1.0 (a version bump is an owner decision), `"type": "module"`, `exports: { ".": { types: ./index.d.ts, default: ./index.js } }`. `files` lists index.js, index.d.ts, README.md, LICENSE and pkg/ (gitignored wasm-bindgen output of raa-js).

**`index.js`** exports `async function loadRaa(options = {}) -> Promise<RaaModule>`. It initialises the wasm from `options.wasmUrl` or the packaged pkg URL. It has no module-level mutable state. `RaaModule = { version: '0.1.0', createEngine(canvas, options) }`.

**`index.d.ts`** is hand-written API v0:
- `Vec3`, `Quat`, `PrimKind`, `EngineOptions`, `MaterialDesc`, `CameraDesc`, `LightDesc`, `AmbientDesc`, `AdapterInfo`, `FrameStats`;
- `AssetInfo` and `SpawnedAsset`;
- `RaaError`, `Engine`, `RaaModule`, `loadRaa`.

P2 additions live in `packages/raa/types/mNN-*.d.ts`, which repeat `export interface Engine` / `EngineOptions` declarations for TypeScript to merge.

**`scripts/build-package.sh [--tag vX.Y.Z]`** runs offline. Steps:
1. Builds raa-js with wasm-release and runs wasm-bindgen `--target web` into `packages/raa/pkg/`.
2. Copies LICENSE and writes `dist/raa-sizes.json` (S7).
3. Stages the package in `target/package-stage/raa/`. The shipped `index.d.ts` is `packages/raa/index.d.ts` followed by every `packages/raa/types/*.d.ts` in file-name order. The stage is also copied to `dist/package/raa/` for the P2 pages.
4. Builds `dist/example-basic/`.
5. Runs `npm pack` from the stage into `release/` with `SHA256SUMS`, and checks the tarball listing. The same tgz is written to `dist/etcito-raa-<version>.tgz`.

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
Files: `README.md`, `docs/API.md`, `docs/ARCHITECTURE-notes.md`.

**`docs/API.md`** is public API v0. Its frozen sections are:
- the JS section for every export of `packages/raa/index.d.ts`;
- the Rust section with the extension points GeometryProvider, RenderPass, `add_pass_after`/`add_pass_before`, AssetLoader and BooleanResolver;
- the additive-only stability rule;
- the size table.

Its last section, `## Additions after v0`, points to `docs/api/` (one file per milestone, additions only) and to the Rust extension points `FeaturePass`, `ForwardContributor` and slot providers.

**`README.md`** covers:
- what RAA is and the crates map, P2 crates included;
- build, check and hook commands;
- demo and example regeneration;
- lookdev commands (`scripts/build-lookdev.sh`, `node scripts/lookdev-capture.mjs`).

The stack sentence stays in README only.

**`docs/ARCHITECTURE-notes.md`** describes the data flows: asset, loader, meshes, renderer, frame graph (v0); mesh, LitRenderer, pipeline passes, tonemap (lit).

## Module: proof-m6
Files: `proofs/m6-api.mjs`, `proofs/results/example-basic.json`, `proofs/results/m6-sizes.json`, `docs/dogfood/m6.md`. Owner: M6.

Checks:
- `extension`: `cargo test -p raa-ext-test`, plus `cargo metadata --format-version 1 --no-deps`. raa-ext-test must depend only on raa-* library crates, and no workspace crate may depend on it.
- `api-doc`: every name exported by `packages/raa/index.d.ts` appears in `docs/API.md`. Measured = covered / total.
- `npm-pack`: `scripts/build-package.sh` exits 0, and `dist/etcito-raa-0.1.0.tgz` lists package.json, index.js, index.d.ts, README.md, pkg/*_bg.wasm and pkg/*.js.
- `wasm-size-table`: every `dist/*/sizes.json` and `dist/raa-sizes.json` has gzip <= WASM_GZIP_MAX. The table is printed.
- `browser-run`: example-basic.json nonEmpty, firstFrameMs.

The committed `proofs/results/m6-sizes.json` is the size table written by the task.

## Module: raa-mesh
Files: `crates/raa-mesh/**`.

Attributed meshes for the lit renderer (S10 streams, S13 rules). Depends on raa-math, raa-geom, raa-core.

**Types**
- `AttributedMesh { pub base: MeshData, pub uvs: Option<Vec<[f32; 2]>>, pub tangents: Option<Vec<[f32; 4]>>, pub colors: Option<Vec<[f32; 4]>>, pub submeshes: Vec<SubMesh> }`.
- `SubMesh { first_index, index_count, material: Option<MaterialId> }`.

**`AttributedMesh` methods**
- `from_mesh(m)`: one sub-mesh, no attributes.
- `validate()`: S13 codes; normals stored normalised.
- `ensure_tangents()`: generates tangents when uvs are present and tangents absent.
- `write_stream0_bytes` (S1 24 B), `write_stream1_bytes` (S10 40 B, defaults), `write_index_bytes`.
- `bounds`, `vertex_count`, `triangle_count`.

**Free functions**
- `from_js_arrays(positions, normals, uvs?, tangents?, colors?: Colors, indices, submeshes) -> Result<AttributedMesh, MeshError>`. `enum Colors { Linear(&[f32]), Srgb8(&[u8]) }`. sRGB8 uses the exact EOTF, and alpha is /255 linear.
- `generate_tangents(positions, normals, uvs, indices, out)`:
  - angle-weighted per-vertex dP/duv;
  - Gram-Schmidt against the normal;
  - w = sign(dot(cross(n, t), bitangent));
  - degenerate uv triangles fall back to `default_tangent`.
- `default_tangent(n) -> [f32; 4]`: Duff et al. 2017, w = 1.

`MeshError { code, message }`.

Tests: validate.rs (every S13 code), tangents.rs, streams.rs.

## Module: raa-shader
Files: `crates/raa-shader/**`.

WGSL composition and validation (S11). Depends on raa-core and naga.

**Types**
- `PassKind { Prepass, Forward, Shadow, Transparent, FurShell }`.
- `SlotId { Deform, Shadow, Ao, Ambient, Direct, Transmission, Fog }` with `ALL` and `fn_name()`.
- `SlotProvider { slot, priority: i32, provider: &'static str, wgsl: Cow<'static, str> }`. Private helpers are prefixed `raa_<provider>_`.
- `MaterialFnId(pub u32)`; `MaterialFnId::STANDARD` = 0.
- `VariantKey { material_fn, pass, msaa, flags }`.

**`MaterialFnRegistry`**
- `new()` registers `raa.standard` as id 0.
- `register_builtin(name, wgsl)`.
- `register_user(name, wgsl, &Composer)`: S11 user rules, then the Forward variant is validated.
- `get(id) -> Option<&MaterialFnEntry { name, wgsl, has_vertex, builtin }>`, `id_of`, `names`, `len`.

**`Composer`**
- `new()` (no providers).
- `add_slot(p)` (`shader.slot_conflict`), `winner(slot)`, `slots_hash()` (FNV-1a over the winners).
- `compose(&reg, &key)` emits, in order:
  1. PRELUDE and LIB;
  2. the material fn, plus a passthrough `material_vertex` when the fn has none;
  3. the slot winners in `SlotId::ALL` order;
  4. the pass entry points.
  Codes: `shader.slot_missing`, `shader.unknown_fn`.
- `validate(wgsl)`: naga front-end + validator with the core WebGPU capabilities. Messages carry line:col.

**WGSL consts**
- `PRELUDE_WGSL`, `LIB_WGSL`, and `PASS_{PREPASS,FORWARD,SHADOW,TRANSPARENT,FURSHELL}_WGSL` (entry points `vs_main`, `fs_main`).
- The Transparent fs appends to the OIT list in group 2: binding 0 `heads: array<atomic<u32>>`, 1 `nodes: array<OitNode>`, 2 `counter: atomic<u32>`, 3 `OitParams` (width, height, capacity, 0).
- The FurShell pass reads its shell index from the `RaaFurShellGpu` group-2 uniform.

**`ShaderError { code, message }`**: `shader.parse`, `shader.validate`, `shader.slot_conflict`, `shader.slot_missing`, `shader.unknown_fn`, `material.fn_invalid`, `material.fn_duplicate`.

Tests: compose.rs, user_fn.rs, variants.rs.

## Module: raa-pipeline
Files: `crates/raa-pipeline/**`.

Configurable pass pipeline (S12) and group-1 lighting resources (S10). Depends on raa-core.

**Pass interface**
- `pub const HDR_FORMAT`.
- `pub trait FeaturePass`:
  - `name`, `order`;
  - `required` (default false);
  - `resize(gpu, &Targets)`;
  - `set_enabled(enabled, &mut LightingResources)`;
  - `prepare(&mut PrepareCtx)` (default Ok);
  - `record(&mut RecordCtx)`.
- `PrepareCtx { gpu, size, frame, time_s, dt_s, targets, lighting: &mut LightingResources }`.
- `RecordCtx { gpu, encoder, size, frame, time_s, targets, lighting, backbuffer, backbuffer_format, surface_is_srgb }`.

**Targets**
- `Targets`: `size`, `depth`, `normal`, `material` (views and `*_texture()`), `hdr() -> &HdrChain`.
- `HdrChain`: `src`, `dst`, `src_texture`, `dst_texture`, `swap()` (Cell index), `reset()` at frame start.

**`LightingResources`**
- `new(gpu)` creates dummies.
- `layout()` and `bind_group()` (rebuilt when dirty).
- Setters and clears:
  - `set_shadow(view, &[u8; 320])` / `clear_shadow`;
  - `set_env(cube, sh: [[f32; 4]; 9], info)` / `clear_env`;
  - `set_brdf_lut`;
  - `set_ao` / `clear_ao`;
  - `set_contact` / `clear_contact`;
  - `set_opaque_copy(color, depth)` / `clear_opaque_copy`.
- `env()` / `env_mut() -> &mut EnvGpu` (S10 fields, uploaded when changed).

**`Pipeline`**
- `new(gpu, surface_format, surface_is_srgb)`.
- `add(Box<dyn FeaturePass>)` (`pipeline.duplicate_pass`).
- `add_contributor_entry(name, order, enabled: Rc<Cell<bool>>)`.
- `passes() -> Vec<PassInfo>`, `set_enabled(name, on)`, `set_order(names)`, `set_time(now_ms)`.
- `lighting`, `lighting_mut`, `targets`.
- `install(this: &Rc<RefCell<Pipeline>>, graph)` adds RenderPass `raa.pipeline` writing BACKBUFFER. Prepare uploads the lighting resources and runs each enabled pass's prepare. Record runs each enabled pass's record in order.
- `PassInfo { name, role: PassRole { Pass, Contributor }, enabled, order, required }`.
- Codes: `pipeline.duplicate_pass`, `unknown_pass`, `required_pass`, `order_incomplete`, `required_order`.

Tests: config.rs, resources.rs (GPU part SKIP without an adapter).

## Module: raa-lit
Files: `crates/raa-lit/**`.

The HDR lit renderer: meshes, surface materials, instance extras, draw lists, prepass and MSAA forward (S10-S12). Depends on raa-core, raa-math, raa-geom, raa-material, raa-scene, raa-mesh, raa-shader, raa-pipeline.

**Options and data types**
- `LitOptions { max_instances 16384, max_materials 256, msaa 4 }`. msaa other than 1 or 4 gives `render.invalid_msaa`.
- `SurfaceModel { Flat, Standard, Function(MaterialFnId) }`.
- `SurfaceMaterial { model, base_color, opacity, emissive, roughness, metallic, clearcoat, clearcoat_roughness, sheen_color, sheen_roughness, transmission, ior (1.5), thickness, attenuation_color, attenuation_distance (INFINITY written as 1e30), flags, params: [[f32; 4]; 8], fur: [f32; 4] }`:
  - `from_v0(&Material)`;
  - `validate()` (`material.invalid` naming the field);
  - `write_gpu` (256 B);
  - `class()`.
- `InstanceExtra { tint, squash, jiggle, params: [[f32; 4]; 4] }` (Default per S10), `write_gpu` (128 B).
- `DrawClass { Opaque, Transparent, Fur }`: TRANSPARENT or TRANSMISSIVE gives Transparent; FUR gives Fur, also drawn as an opaque base.
- `LitMeshId(pub u32)` and `LitMeshInfo { vertices, triangles, submeshes, bounds }`.
- `LitDraw { mesh, material, material_fn, class, first_instance, instance_count, view_depth }`.
- `pub trait ForwardContributor { name; order; prepare(&LitHandle, &mut PrepareCtx) (default Ok); record(&self, &mut wgpu::RenderPass, &LitHandle) }`.

**`LitRenderer`**
- `new(gpu, &Rc<RefCell<Pipeline>>, opts)`: default slot providers (priority 0), FeaturePasses `prepare` and `forward`, default material id 0.
- Meshes:
  - `upload(gpu, &AttributedMesh) -> LitMeshId` (`render.invalid_mesh` with the MeshError code);
  - `update`, `remove`;
  - `submesh_ids(id)`: one MeshId per sub-mesh, dense, lowest free reused;
  - `info(id)`.
- Materials and functions:
  - `materials()` / `materials_mut() -> &mut SurfaceLibrary` (`add`, `set` (`material.unknown_id`), `get`, `len`, `is_dirty`, `write_gpu_all`);
  - `fns()` / `fns_mut()`, `composer()`, `add_slot(p)` (clears the pipeline cache).
- Classes and extras:
  - `register_class_handler(class, pass_name)`, `class_handled(class)` (Opaque always true);
  - `set_instance_extra(tag, x)`, `instance_extra(tag)`, `clear_instance_extra(tag)`.
- `add_contributor(Box<dyn ForwardContributor>)`, which also registers its pipeline entry. `set_background(rgb)`.
- `prepare(gpu, &mut Scene, &mut RenderList, &mut Vec<u32>) -> FrameCounts`:
  - updates the world and builds the tagged render list;
  - writes FrameGpu from the camera, light, ambient and tonemap state;
  - writes instance bytes and extras in draw order. Opaque and Fur draws are batched by (mesh, material). Transparent draws are one per instance, back-to-front by the view depth of the world translation, ties by tag;
  - codes `render.too_many_instances`, `render.unknown_mesh`, `render.too_many_materials`, `material.blend_unavailable`.
- `handle() -> LitHandle`.

**`LitHandle`** (Rc to the shared state)
- `draws(class)`.
- `bind_mesh(pass, mesh) -> index count` (streams 0/1 and indices) and `submesh_range(mesh)`.
- `instance_buffer`, `frame_layout`, `frame_bind_group`, `msaa`, `material_record(id)`.
- `pipeline_for(gpu, VariantKey, &PipelineSpec { color_targets, depth, cull, alpha_to_coverage, extra_layout })`, cached by key, spec and slots hash.

**Passes**
- `prepare` (order 30, required): Opaque + Fur draws, single sample, writes `depth`, `normal` and `material`.
- `forward` (order 60, required):
  - MSAA HDR colour + MSAA Depth32Float (Less), colour cleared to the background;
  - Opaque + Fur draws, culling back faces unless DOUBLE_SIDED;
  - enabled contributors by order;
  - resolves into `hdr.dst()` and swaps.

Tests: material_gpu.rs, extras.rs, prep.rs, variants.rs, gpu_lit.rs (SKIP without an adapter).

## Module: raa-tonemap
Files: `crates/raa-tonemap/**`.

Display transform (S12 `tonemap`). Depends on raa-core, raa-pipeline, raa-shader.

**Types**
- `ToneMapper { Agx, Aces, Linear }`, `Look { None, Punchy, Golden }`.
- `ToneMapping { mapper, exposure_ev, look, dither }`. Default TONEMAP_DEFAULT. `validate` gives `post.invalid` for a non-finite ev.
- `TonemapControl` (Rc<Cell<ToneMapping>>): `get`, `set`.

**`TonemapPass`** (FeaturePass `tonemap`, order 200, required)
- `new(gpu, &Pipeline) -> (TonemapPass, TonemapControl)`.
- Draws a fullscreen triangle into the backbuffer. Colour = hdr.src x 2^ev, then the mapper:
  - AgX: Sobotka base (inset, log2 over [-12.47393, 4.026069] EV, 6th-order sigmoid, outset), looks as slope/power/saturation presets;
  - ACES: Hill's fitted RRT+ODT;
  - Linear: clamp.
- Then the sRGB OETF when the canvas is not *Srgb, and an 8x8 Bayer dither of ±0.5/255 when enabled.
- S12 background rule.
- `TONEMAP_WGSL`.

Tests: curves.rs (monotonic, 0 to 0, mid grey 0.18 in [0.4, 0.55], white below 1.0001), variants.rs.

## Module: api-m7
Files: `crates/raa-js/src/api/m07_meshes.rs`, `crates/raa-js/src/api/m07_pipeline.rs`, `packages/raa/types/m07-meshes-hdr.d.ts`, `docs/api/m07-meshes-hdr.md`.

The Rust files are described in raa-js. The hand-written types declare:
- `MeshDesc`, `SubMeshDesc`, `MeshInfo`, `PassInfo`, `ToneMappingDesc`, `Capabilities`;
- `LitEngineOptions` (msaa, toneMapper, exposureEv), merged into `EngineOptions`;
- `Engine { addMesh, updateMesh, meshInfo, addMeshInstance, passes, setPassEnabled, setPassOrder, setToneMapping, setBackground, capabilities }`.

The doc gives each name its signature, semantics, units, defaults and codes (S12, S13).

## Module: lookdev-page
Files: `examples/lookdev/{index.html, main.js, nmc.gpu.json}`, `examples/lookdev/lib/{geometry, objects, runner}.js`.

The lookdev page is built only on `./raa/index.js` (the staged package), with no compatibility-adapter wrapper.

**`lib/geometry.js`** (pure, node-importable)
- Builders, each returning a MeshDesc: `lathe(profile, segments, opts?)`, `sweepTube(path, radius, sides, opts?)`, `roundedBox(half, radius, segments)`, `uvSphere(radius, segU, segV)`, `grid(w, d, nx, nz)`.
- `merge(parts)` keeps sub-meshes. `withTangents(desc)`.
- Every builder emits positions, normals, uvs, tangents and indices.

**`lib/objects.js`** (pure)
- `inflatableHammer()`: two sub-meshes with vertex-colour stripes.
- `iceBlock()`, `plushSphere()`, `neonTube()`.
- `slimePool()`: basin plus a surface sub-mesh.
- `floorTiles(n)`: per-tile vertex colours.

**`lib/runner.js`**
- `runScene(raa, canvas, scene, variant, args)`:
  - creates a fresh engine with merged EngineOptions and runs `scene.build`;
  - applies the variant's passes (disable, order);
  - renders LOOKDEV_CAPTURE at the fixed `timeMs`, calling `scene.frame`;
  - returns the S14 op result.
- Exports `WARMUP`, `FRAMES`, `DEFAULT_TIME_MS`, `variantTimeMs` and `renderAt`.

**`main.js`** loads `scenes.json` and registers `window.nmcGpu.ops.run` (S14). It offers a scene picker and collects uncaptured GPU errors. `nmc.gpu.json` has `{page: index.html, ops: [render]}`.

## Module: lookdev-tools
Files: `scripts/build-lookdev.sh`, `scripts/lookdev-capture.mjs`, `proofs/lib/p2.mjs`, `proofs/lib/p2-smoke.mjs`.

**`scripts/build-lookdev.sh`**
- Runs `scripts/build-package.sh`.
- Builds `dist/lookdev/` from `examples/lookdev/**`, `dist/package/raa/` (as `raa/`) and `web/common/*` (as `common/`).
- Writes `scenes.json` (S14) and `sizes.json` (S7).

**`node scripts/lookdev-capture.mjs [--page lookdev|sdfm-compare] [--milestone mNN] [--only <regex>] [--judge [--judge-record-only]] [--ref-prefix <p>]`**
- Requires NMC_GPU_CLIENT and publishes `dist/<page>/` once.
- Per (scene, variant):
  - writes `target/lookdev/<pair>.input.json` (`{args: {scene, variant, width: 1280, height: 720}}`);
  - renders into `target/gpu-out/lookdev/<pair>/`, up to NMC_CPUS in parallel;
  - writes the S14 record and `proofs/captures/lookdev/<pair>.png`.
- With `--judge`, runs the judge (Conventions) on `judge: true` variants.
- Redacts env values and never writes a record the client did not produce.
- Exits non-zero after finishing the others when any pair failed, listing the failures.

**`proofs/lib/p2.mjs`**
- `cargoTest(crates)`.
- `wasmCheck()`: `cargo check --target wasm32-unknown-unknown -p raa-js`.
- `apiDocCheck(tag)`: every exported name and Engine member of `types/<tag>-*.d.ts` appears in backticks in `docs/api/<tag>-*.md`, plus the neutrality regex on both.
- `lookdevCheck(tag, {page?})`: every declared variant has a nonEmpty, hardware record, and every `-on` has its `-off`.
- `judgeCheck(tag)`: every judged variant has exit 0.

**`node proofs/lib/p2-smoke.mjs`** validates every lookdev record against S14 and runs `apiDocCheck` for every tag present.

## Module: proof-m7
Files: `proofs/m7-meshes-hdr.mjs`, `examples/lookdev/scenes/m07-base.js`, `proofs/results/lookdev/m07-*.json`, `docs/dogfood/m7.md`.

Scene `m07-base`: the six lookdev objects on tiles with standard materials, a bright sun and a background.
- Variants: `agx`, `aces`, `exposure-up` (+1.5 EV), `msaa-on`, `msaa-off`, `vertex-colors`, `empty`. None is judged.
- Checks:
  - `unit`: raa-mesh, raa-shader, raa-pipeline, raa-lit, raa-tonemap, raa-scene, raa-js;
  - `wasm-check`;
  - `api-doc` (m07 plus v0 index.d.ts coverage);
  - `captures` (m07);
  - `v0-mapping` (`cargo test -p raa-js --test convert`).

## Module: raa-ibl
Files: `crates/raa-ibl/**`.

Image-based lighting from the sky cube (S10 bindings 3-5). Depends on raa-core, raa-pipeline, raa-lit, raa-shader.

**`IblFeature::install(gpu, pipeline, lit) -> IblHandle`**
- Computes the BRDF LUT once: a compute pass writes pack2x16float into a storage buffer, which is copied into the Rg16Float texture (core WebGPU has no Rg16Float storage textures). It is a split-sum scale/bias with Fdez-Aguera multiscatter, set through `set_brdf_lut`.
- Adds the `Ambient` provider `ibl` (priority 10).

**`IblHandle`**
- `filter(gpu, encoder, cube, lighting)`. The cube is SKY_ENV with TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC. Steps:
  1. source mips 1..5 by box downsample;
  2. mip 0 copied into the IBL output cube;
  3. output mips 1..5 by GGX importance sampling with a PDF-based source mip;
  4. SH9 projection by compute reduction with solid-angle weights;
  5. a copy into a staging ring of 3 buffers for asynchronous readback.
- `apply_pending(lighting) -> bool`, called every frame by `sky.env`:
  - it calls `set_env(cube, sh, [specular, diffuse, 5, 1])` when a new SH arrived or the intensities changed;
  - the env stays cleared until the first SH arrives;
  - it returns true when a readback was skipped, so the caller filters again.
- `set_intensity(diffuse, specular)`: finite, >= 0.

**WGSL and ambient provider**
- `IBL_LIB_WGSL`: `raa_ibl_irradiance`, `raa_ibl_specular`, `raa_ibl_dfg`, `raa_specular_occlusion` (Lagarde). The SH is cosine-convolved and divided by π, so a constant environment c gives irradiance c.
- With `EnvSh.info.w < 0.5` the provider returns the S1 ambient term. Otherwise:
  - diffuse = SH irradiance x base x (1 - metallic) x multiscatter compensation, with Jimenez multi-bounce AO;
  - specular = prefiltered(R, roughness x max mip) x (F0 A + B) x energy compensation x specular occlusion;
  - each term is scaled by its intensity.

Tests: sh.rs, variants.rs, gpu_ibl.rs.

## Module: raa-sky
Files: `crates/raa-sky/**`.

Procedural sky (S12 `sky.env`, `sky.background`). Depends on raa-core, raa-pipeline, raa-lit, raa-shader, raa-ibl.

**Descriptions**
- `CloudDesc { coverage, density, altitude_m (1500), scale_m (2500), wind_mps: [f32; 2], color, silver_lining }`.
- `SkyDesc { zenith, horizon, ground, intensity, horizon_softness, sun_disc_angular_radius, sun_disc_intensity, clouds, env_update: EnvUpdate { OnChange, EveryFrame } }`.
- `SkyDesc::preset("day"|"snow"|"jungle"|"night")`. `validate` gives `sky.invalid` naming the field.

**`SKY_WGSL`** defines `raa_sky_radiance(dir, env, sun_dir, sun_color, time)`:
- a gradient with the horizon blend pow(1 - max(dir.y, 0), softness);
- HG glow (g 0.76) and a limb-darkened sun disc;
- a cloud layer by ray-plane intersection: 5-octave domain-warped fbm over the coverage threshold, a 4-step light march with Beer-powder, an HG silver lining and a horizon fade.

**`SkyFeature::install(gpu, pipeline, lit, ibl) -> SkyHandle`**
- FeaturePass `sky.env` renders the SKY_ENV cube and calls `ibl.filter`, on change or every frame per env_update. It calls `ibl.apply_pending` each frame. `set_enabled` writes the EnvGpu background mode and clears the env.
- ForwardContributor `sky.background`: fullscreen at depth 1 with LessEqual.

`SkyHandle`: `set`, `get`, `mark_dirty`.

Tests: desc.rs, variants.rs.

## Module: api-m8
Files: `crates/raa-js/src/api/m08_sky_ibl.rs`, `packages/raa/types/m08-sky-ibl.d.ts`, `docs/api/m08-sky-ibl.md`.

`install` creates IblFeature, then SkyFeature, and stores the handles in the FeatureMap. `frame` marks the sky dirty when the scene light changed.
- `setSky(desc)`: `{ preset?, zenith?, horizon?, ground?, intensity?, horizonSoftness?, sunDiscSize? (rad), sunDiscIntensity?, clouds?: { coverage?, density?, altitude?, scale?, wind?: [x, z], color?, silverLining? }, envUpdate?: 'onChange'|'everyFrame' }`. The preset applies first, then the overrides (`sky.invalid`).
- `sky()` returns the resolved values.
- `setEnvironmentLighting({ diffuse?, specular? })` (values >= 0).

## Module: proof-m8
Files: `proofs/m8-sky-ibl.mjs`, `examples/lookdev/scenes/m08-sky-ibl.js`, `proofs/results/lookdev/m08-*.json`, `docs/dogfood/m8.md`.

Scene `m08-sky-ibl`: the lookdev objects under the `day` sky with clouds.
- Variants: `sky-on`/`sky-off`, `ibl-on`/`ibl-off`, `preset-snow`, `preset-night`, `glossy`. `sky-on` and `ibl-on` are judged against ref-1.
- Checks: `unit` (raa-sky, raa-ibl), `wasm-check`, `api-doc` (m08), `captures`, `judge`.

## Module: raa-shadow
Files: `crates/raa-shadow/**`.

Cascaded soft sun shadows (S12 `shadow.csm`). Depends on raa-core, raa-math, raa-pipeline, raa-lit, raa-shader.

**Description and CPU functions**
- `ShadowDesc { softness (1; (0, 16], multiplies the sun angular radius), distance_m (60), resolution (2048: 1024|2048|4096) }`. `validate` gives `shadow.invalid`.
- `cascade_splits(near, far, lambda, n) -> [f32; 4]`: the practical scheme. n is clamped to 1..=4, and unused entries = far.
- `cascade_matrix(view, fov_y, aspect, split, sun_dir, resolution) -> Mat4`:
  - the bounding sphere of the slice, radius rounded up to 1/16 m;
  - centre snapped to texels;
  - ortho with z in [0, 1];
  - IDENTITY for a singular view.

**`ShadowFeature::install(gpu, pipeline, lit) -> ShadowHandle`**
- FeaturePass `shadow.csm` (order 20):
  - renders Opaque + Fur draws without NO_SHADOW into a 4-layer Depth32Float array (PassKind::Shadow, group-2 caster view_proj with dynamic offset, slope bias 2.0 + constant 2);
  - then `set_shadow` with params (softness, 1.5 normal-offset texels, 1/resolution, 1) and params2 (0, 0.1 blend band, 0, 0);
  - no view or disabled gives `clear_shadow`.
- `Shadow` slot `csm` (priority 10):
  - cascade by view depth with a blend band and a normal-offset receiver;
  - PCSS: 16-sample blocker search; penumbra = (d_r - d_b) x tan(sun radius x softness), min 1.5 texels; 32-sample Poisson PCF rotated by interleaved gradient noise;
  - multiplied by the contact texture;
  - the contact term alone when shadows are cleared.

`ShadowHandle`: `set`, `get`, `set_view(view, fov_y, aspect, near, far, sun_dir)` (called every frame by api-m9).

Tests: cascades.rs, variants.rs.

## Module: raa-gtao
Files: `crates/raa-gtao/**`.

Ground-truth ambient occlusion (S12 `ao.gtao`). Depends on raa-core, raa-math, raa-pipeline, raa-lit, raa-shader.

- `AoDesc { radius_m (0.8; (0, 10]), intensity (1; [0, 1]), power (1.5; (0, 8]) }`. `validate` gives `ao.invalid`.
- `GtaoFeature::install(gpu, pipeline, lit) -> GtaoHandle`. FeaturePass `ao.gtao` (order 40) runs these compute steps at full resolution:
  1. Jimenez 2016 from `depth` and `normal`: 4 slice directions rotated by 4x4 noise, 8 horizon steps per side, cosine-weighted visibility with a thickness heuristic; out = 1 - intensity x (1 - vis^power), into R32Float.
  2. Two separable 7-tap bilateral passes, weight exp(-|dz| / (0.02 z)). The final texture is Rgba16Float with AO in .r.
  3. `set_ao`; no camera or disabled gives `clear_ao`.
- `Ao` slot `gtao` (priority 10): textureLoad of binding 7 .r.
- `GtaoHandle`: `set`, `get`, `set_camera(proj, inv_proj)` (every frame, api-m9).

Tests: desc.rs, variants.rs.

## Module: raa-contact
Files: `crates/raa-contact/**`.

Screen-space contact shadows (S12 `shadow.contact`). Depends on raa-core, raa-math, raa-pipeline, raa-lit, raa-shader.

- `ContactDesc { distance_m (0.35; (0, 5]), thickness_m (0.05; (0, 1]), intensity (1; [0, 1]) }`. `validate` gives `shadow.invalid`.
- `ContactFeature::install(gpu, pipeline, lit) -> ContactHandle`. FeaturePass `shadow.contact` (order 45, compute):
  - 24 jittered steps toward the sun; a pixel is occluded when the depth buffer is in front by less than the thickness;
  - fades at the end and at the edges;
  - writes Rgba16Float with .r = 1 for lit;
  - `set_contact`, or `clear_contact` with no camera or when disabled.
- `ContactHandle`: `set`, `get`, `set_camera(view, proj, inv_proj, sun_dir)` (every frame, api-m9).

Tests: variants.rs.

## Module: api-m9
Files: `crates/raa-js/src/api/m09_shadows_ao.rs`, `packages/raa/types/m09-shadows-ao.d.ts`, `docs/api/m09-shadows-ao.md`.

- `setShadows({ softness?, distance?, resolution?: 1024|2048|4096 })`.
- `setAo({ radius?, intensity?, power? })`.
- `setContactShadows({ distance?, thickness?, intensity? })`.

Each call is validated with its code. Enabling and disabling go through `setPassEnabled('shadow.csm'|'ao.gtao'|'shadow.contact', on)`. `frame` feeds the camera and sun to the three handles.

## Module: proof-m9
Files: `proofs/m9-shadows-ao.mjs`, `examples/lookdev/scenes/m09-shadows-ao.js`, `proofs/results/lookdev/m09-*.json`, `docs/dogfood/m9.md`.

Scene `m09-shadows-ao`: an obstacle-like arrangement with a tall pillar under the day sky.
- Variants: `all-on`, `shadows-on`/`-off`, `ao-on`/`-off`, `contact-on`/`-off`, `softness-wide`, `long-shadow`. `all-on` is judged against ref-1.
- Checks: `unit` (raa-shadow, raa-gtao, raa-contact), `wasm-check`, `api-doc` (m09), `captures`, `judge`.

## Module: raa-matfn
Files: `crates/raa-matfn/**`.

Built-in material functions and the JS surface description mapping. Depends on raa-core, raa-shader, raa-lit.

**Built-in tables**
- `ParamKind { Float, Vec2, Vec3, Vec4, Color }` with `components()`.
- `ParamSpec { name, kind, default (JS units; Color sRGB), slot: (p index, first component) }`.
- `BuiltinFn { name: raa.<fn>, wgsl, params, flags, class }`.
- `BUILTINS`: `raa.inflatable`, `raa.candy`, `raa.plush`, `raa.neon`, `raa.mud`, `raa.snow`, `raa.tiles`.
  - `raa.plush` computes the Kajiya-Kay primary and shifted secondary highlights tinted by the sheen colour.
  - Helpers are prefixed `<fn>_`, never `raa_`.
- `register_builtins(&mut MaterialFnRegistry)`.
- `ExtraBuiltins` (tables registered by other crates): `push(table)` (earlier tables win), `find(name)`, `tables()`.

**Surface description**
- `SurfaceDesc { function (default raa.standard), params: Vec<(String, Vec<f32>)>, blend: Blend { Opaque, Transparent, Transmissive }, double_sided, cast_shadow (true), overrides: StandardOverrides }`.
- `StandardOverrides` holds an optional value for every standard field, with colours in sRGB and emissive linear, plus `fur_length` (>= 0), `fur_density` (> 0, default FUR_DENSITY_DEFAULT 2000), `fur_droop` ([0, 1]) and `fur_shells` (an integer in [1, FUR_SHELLS_MAX 64], default 32). Any fur field sets FUR.

**`#[doc(hidden)] map_desc_with(desc, reg, extras, class_handled)`** maps in this order:
1. function lookup (`material.unknown_fn`);
2. the row from BUILTINS, then extras. Its params are refused with `material.unknown_param` or `material.param_type`, and non-finite values with `material.invalid`. User functions take `p0`..`p7` with 1..4 numbers;
3. overrides;
4. blend and flags (DOUBLE_SIDED, NO_SHADOW). Opacity < 1 without blend transparent, or transmission > 0 without blend transmissive, gives `material.invalid`;
5. `validate`;
6. for Transmissive, opacity becomes opacity x (1 - transmission) + transmission (the OIT alpha);
7. `material.blend_unavailable` when the class has no handler.

`map_desc` uses an empty ExtraBuiltins. `to_surface_material(desc, lit)` and `to_surface_material_with(desc, lit, extras)` use the lit renderer's functions and class handlers.

Tests: desc.rs (including extra-table mapping and transmissive alpha), variants.rs.

## Module: raa-brdf
Files: `crates/raa-brdf/**`.

Layered BRDF slot providers. Depends on raa-shader, raa-lit, raa-ibl.

- `providers() -> [SlotProvider; 2]`:
  - `Direct` provider `layered` (priority 20):
    - base: Lambert x (1 - metallic) + height-correlated Smith GGX with multiscatter compensation;
    - sheen: Charlie NDF + Neubelt visibility, with the base scaled by (1 - max(sheen) x E_sheen);
    - clearcoat: GGX, F0 0.04, Kelemen visibility on the geometric normal, attenuating the lower layers by (1 - Fc).
  - `Ambient` provider `layered` (priority 20): the raa-ibl base terms, a clearcoat prefiltered reflection and sheen indirect from irradiance, with the same attenuation.
- `install(lit)`. `BRDF_WGSL`. CPU `eval_direct(..)`.

Tests: energy.rs (white furnace <= 1 + 1e-3; clearcoat 0 equals base), variants.rs.

## Module: raa-fur
Files: `crates/raa-fur/**`.

Fur shells (S12 `fur.shells`). Depends on raa-core, raa-shader, raa-lit, raa-pipeline (raa-matfn, raa-brdf, raa-ibl for tests).

**`FurContributor::install(gpu, lit)`** registers the Fur class handler and the ForwardContributor `fur.shells` (order 500).
- Per Fur draw it draws N = clamp(round(fur.w), 1, 64) shells with the FurShell variant, one draw per shell. The shell index comes from the `RaaFurShellGpu` group-2 uniform: a 64-slot buffer with a dynamic offset, written once.
- The FurShell shader provides:
  - the offset along the normal, length x (i+1)/N, with droop -Y x droop x length x h²;
  - a strand mask from 3D cellular noise x density, with tapering;
  - alpha-to-coverage under MSAA (discard below 0.5 at 1 sample);
  - root-to-tip darkening, and the shadow and ambient slots.
- Pipeline state: HDR target without blending, depth Less with write, cull none.

Tests: variants.rs (raa.plush FurShell with default, ibl and layered providers; uniform layout).

## Module: raa-bloom
Files: `crates/raa-bloom/**`.

Physically based bloom (S12 `bloom`). Depends on raa-core, raa-pipeline, raa-shader.

- `BloomDesc { strength (0.04; [0, 1]), radius (0.85; [0, 1]) }`. `validate` gives `bloom.invalid`.
- `BloomFeature::install(gpu, pipeline) -> BloomHandle`. FeaturePass `bloom` (order 120):
  1. a downsample chain from hdr.src with a 13-tap filter (Karis average on the first level), up to 8 mips, stopping at 8 px;
  2. a 3x3 tent upsample scaled by radius, accumulated additively;
  3. the composite `mix(hdr, bloom, strength)` into hdr.dst, then a swap.
  There is no threshold.
- `BloomHandle`: `set`, `get`.

Tests: desc.rs, variants.rs, mips.rs.

## Module: api-m10
Files: `crates/raa-js/src/api/m10_materials.rs`, `packages/raa/types/m10-materials.d.ts`, `docs/api/m10-materials.md`.

`install` registers the built-ins, the layered BRDF, the fur contributor and bloom.
- `addMaterialFunction(name, wgsl)`: a user function (S11 rules).
- `materialFunctions(): MaterialFunctionInfo[]` returns `{ name, builtin, params: { name, kind: 'scalar'|'vec3'|'color', default }[] }`. Params come from BUILTINS, then the ExtraBuiltins tables.
- `addSurfaceMaterial(desc: SurfaceMaterialDesc): number` and `setSurfaceMaterial(id, desc)`:
  - desc = `{ function?, params?, blend?: 'opaque'|'transparent'|'transmissive', doubleSided?, castShadow?, color?, roughness?, metallic?, emissive?, opacity?, clearcoat?, clearcoatRoughness?, sheenColor?, sheenRoughness?, transmission?, ior?, thickness?, attenuationColor?, attenuationDistance?, furLength?, furDensity?, furDroop?, furShells? }`;
  - mapped by `raa_matfn::to_surface_material_with` with the FeatureMap's ExtraBuiltins (empty when absent).
- `setBloom({ strength?, radius? })`.

## Module: proof-m10
Files: `proofs/m10-materials.mjs`, `examples/lookdev/scenes/m10-materials.js`, `proofs/results/lookdev/m10-*.json`, `docs/dogfood/m10.md`.

Scene `m10-materials` variants:
- `inflatables` (judged vs ref-1);
- `plush` (judged vs ref-1 and ref-2);
- `neon` (night, judged vs ref-4);
- `bloom-on`/`-off`, `fur-on`/`-off`, `clearcoat-on`/`-off`;
- `user-fn` (a WGSL function added from JS);
- `mud-snow-tiles`.

Comments in the scene file follow the neutrality guard.

Checks: `unit` (raa-matfn, raa-brdf, raa-fur, raa-bloom), `wasm-check`, `api-doc` (m10), `captures`, `judge`.

## Module: raa-transmission
Files: `crates/raa-transmission/**`.

Transmission and refraction (S12 `opaque.copy`). Depends on raa-core, raa-pipeline, raa-lit, raa-shader, raa-ibl, raa-matfn.

**`TransmissionFeature::install(gpu, pipeline, lit)`** registers TRANSMISSIVE_BUILTINS, the `Transmission` slot `screen` (priority 10) and the FeaturePass `opaque.copy` (order 70).

**`opaque.copy`** is active while a Transparent draw exists. It calls set_opaque_copy on activation and clear_opaque_copy on deactivation or disable, never every frame. It copies hdr.src into opaque_color (full mip chain by 2x2 downsample) and depth into opaque_depth (R32Float).

**Slot `screen`**
- direction = refract(-v, n, 1/ior); exit = position + dir x thickness;
- lod = log2(max(1, roughness² x 0.5 x viewport h x thickness / view depth)); the unrefracted position is used when the sample's opaque depth is in front of the surface;
- Beer-Lambert T = exp(-σ path), σ = -ln(attenuation_color) / attenuation_distance, path = thickness / max(dot(-dir, n), 0.2);
- result = transmitted x T x (1 - F) x transmission. Reflection comes from the Direct and Ambient slots.

**`TRANSMISSIVE_BUILTINS`** (flags 0, class Opaque; blend transmissive makes them Transparent):
- `raa.ice` (tint, absorption, cloudiness, crackScale);
- `raa.slime` (color, absorption, glossiness, bubbleScale);
- `raa.water` (deepColor, shallowColor, clarity);
- `raa.glass` (tint, roughness).

The built-ins scale their diffuse base by (1 - transmission). api-m11 pushes the table into ExtraBuiltins.

Tests: variants.rs, absorption.rs.

## Module: raa-oit
Files: `crates/raa-oit/**`.

Sorted transparency with per-pixel linked lists (S12 `transparent.oit`). Depends on raa-core, raa-pipeline, raa-lit, raa-shader.

**Data**
- OitNode (16 B): pack2x16float(r, g), pack2x16float(b, alpha) with straight linear colour, then depth f32 and next u32 (0xffffffff ends the list). The resolve premultiplies.
- `pool_nodes(w, h, factor, limit_bytes) = max(1, min(w h factor, limit / 16))`.
- `OitSettings { max_fragments (4..=32, 16), pool_factor (2..=32, 8) }`, `OitStats { pool_nodes, overflow_frames }`.

**`OitFeature::install(gpu, pipeline, lit) -> OitHandle`** registers the Transparent class handler and the FeaturePass `transparent.oit` (order 80):
1. a compute pass clears the heads and the counter;
2. the Transparent draws run with PassKind::Transparent, msaa 1, no colour target, depth Less against `depth` without writes, culling back faces unless DOUBLE_SIDED, group 2 = heads/nodes/counter/params;
3. the counter is read back asynchronously (ring of 3); an overflow frame doubles the factor, bounded by the storage-binding limit;
4. a fullscreen resolve keeps the max_fragments nearest nodes (insertion sort), composites them back to front over hdr.src into hdr.dst, and swaps.

Without Transparent draws the chain is untouched.

`OitHandle`: `set` (`oit.invalid`, unchanged on refusal), `get`, `stats`.

Tests: resolve.rs, variants.rs, gpu_oit.rs (order independence; SKIP without an adapter).

## Module: api-m11
Files: `crates/raa-js/src/api/m11_transmission.rs`, `packages/raa/types/m11-transmission.d.ts`, `docs/api/m11-transmission.md`.

`install` creates TransmissionFeature and OitFeature. It adds no material method: `addSurfaceMaterial` accepts `blend: 'transparent'|'transmissive'` and the transmissive built-ins.
- `setTransparency({ maxFragments?: 4..32, poolFactor?: 2..32 })` (`oit.invalid`).
- `transparencyStats(): { poolNodes, overflowFrames }`, for debugging only.

The types and docs document these and the transmission fields of SurfaceMaterialDesc.

## Module: proof-m11
Files: `proofs/m11-transmission.mjs`, `examples/lookdev/scenes/m11-transmission.js`, `proofs/results/lookdev/m11-*.json`, `docs/dogfood/m11.md`.

Scene `m11-transmission` variants:
- `ice` (judged vs ref-2);
- `slime` and `water` (judged vs ref-1 and ref-3);
- `overlap`;
- `transmission-on`/`-off` (blend opaque);
- `oit-on`/`-off`.

Checks: `unit` (raa-transmission, raa-oit), `wasm-check`, `api-doc` (m11), `captures`, `judge`.

## Module: raa-liquid
Files: `crates/raa-liquid/**`.

Animated liquid surfaces as material functions with vertex stages. Depends on raa-shader, raa-lit, raa-matfn, raa-transmission.

**`LIQUID_BUILTINS`**, registered by `install(lit)`:
- `raa.liquid.water` and `raa.liquid.slime` (Transmissive class);
- `raa.liquid.mud` (Opaque, wet clearcoat).

Each function has:
- a `material_vertex`:
  - 4 Gerstner waves (amplitude, wavelength, direction angle, steepness and speed packed in p) with analytic normals;
  - ripple rings from up to 4 impacts in the instance params (x, z, start s, strength), a radial damped sine.
- a `material_main`:
  - detail normals from 2 scrolling fbm layers;
  - foam = shore term (opaque_depth - depth < foamWidth) + crest term + noise breakup, white, rough and opaque;
  - water and slime refract through the Transmission slot;
  - mud adds voronoi bubble pops.

`wave_height(params, x, z, t)` is the CPU twin.

Tests: waves.rs, variants.rs.

## Module: raa-particles
Files: `crates/raa-particles/**`.

GPU particles (S12 `particles.sim` 50, contributor `particles.opaque` 600, `particles.blend` 85). Depends on raa-core, raa-math, raa-pipeline, raa-lit, raa-shader.

**Types**
- `ParticleKind { Splash, Snow, Confetti, Sparkles }`.
- `EmitterDesc { kind, position, direction, spread_rad, rate_per_s, lifetime_s: [f32; 2], speed, size, gravity, drag, colors: Vec<[f32; 3]> (1..8 linear), emissive, volume_half, ground_y: Option<f32>, max_particles (<= 65536), seed }`.
- `EmitterDesc::for_kind(kind)`: gravity 9.81, except snow 1.0 and sparkles 0.
- `validate` gives `particles.invalid` naming the field.

**Spawn CPU twin**
- `pcg_hash` (PCG-RXS-M-XS).
- `SpawnSample { position, velocity, life, size, rotation, color_index }`.
- `spawn_sample(desc, serial)`: deterministic per (seed, serial) and mirrored by the sim WGSL.
- `PARTICLE_STRIDE` = 48: pos + age, vel + life, size, rotation, color index, kind.

**`ParticleFeature::install(gpu, pipeline, lit) -> ParticleHandle`** (atomic refusals `pipeline.busy`, `pipeline.duplicate_pass`)
- `ParticleHandle`: `add(d) -> id` (dense ids, lowest free reused), `set(id, d)`, `remove(id)`, `burst(id, count)`.
- `count()`: the sum over emitters of min(max_particles, spawned), with no readback.
- Codes: `particles.invalid`, `particles.capacity` (more than 64 emitters or more than 262144 particles), `particles.unknown_emitter`.

**Passes**
- `particles.sim` (compute):
  - atomic spawn from the accumulated rate and bursts;
  - gravity, drag and ground bounce 0.3;
  - per kind: confetti flutter, snow curl-noise drift, ballistic splash, sparkle drift.
  - The clock advances in fixed 1/60 s steps toward the pipeline time: at most 240 steps per frame, and the first frame starts at max(0, t - 4 s). A fixed-time frame (dt 0) therefore shows the state at t.
- `particles.opaque` (confetti, snow): lit quads with alpha-to-coverage and the installed Shadow, Ao, Ambient and Direct slots.
- `particles.blend`: splash with env Fresnel (premultiplied) and additive twinkling sparkles. It copies hdr.src to dst, fades softly against `depth` and swaps. Without such emitters it records nothing.

Draw shaders = PRELUDE + LIB + the slot winners snapshotted at install + the pass body. Hidden builders: `sim_module_wgsl`, `opaque_module_wgsl(composer)`, `blend_module_wgsl(composer)`.

Tests: desc.rs, spawn.rs, variants.rs.

## Module: api-m13
Files: `crates/raa-js/src/api/m13_liquids_particles.rs`, `packages/raa/types/m13-liquids-particles.d.ts`, `docs/api/m13-liquids-particles.md`.

- `addEmitter(desc): number`, `setEmitter(id, desc)`, `removeEmitter(id)`, `burst(id, count)`.
  - `EmitterDesc = { kind: 'splash'|'snow'|'confetti'|'sparkles', position, direction?, spread?, rate?, lifetime?: [min, max], speed?, size?, gravity?, drag?, colors?: Vec3[] (sRGB), emissive?, volume?, groundY?, maxParticles?, seed? }`.
- `setLiquidImpacts(instance, impacts: { x, z, time, strength }[])`: at most 4, written to instance params 0..3 (`api.invalid_argument`).

The liquid built-ins appear in `materialFunctions()`.

## Module: proof-m13
Files: `proofs/m13-liquids-particles.mjs`, `examples/lookdev/scenes/m13-liquids-particles.js`, `proofs/results/lookdev/m13-*.json`, `docs/dogfood/m13.md`.

Scene `m13-liquids-particles` variants:
- `water` (judged vs ref-1 and ref-3);
- `slime` (vs ref-1) and `mud` (vs ref-3);
- `splash`;
- `snow` (vs ref-2), `confetti` (vs ref-1), `sparkles` (vs ref-4);
- `waves-t0`/`waves-t1` (timeMs 1000 and 1600).

Checks: `unit` (raa-liquid, raa-particles), `wasm-check`, `api-doc` (m13), `captures`, `judge`.

## Module: raa-sdfm-pkg
Files: `packages/raa-sdfm/**`.

`@etcito/raa-sdfm` 0.1.0: plain ES modules, a hand-written `index.d.ts`, no build step.
- `package.json`: `type: module`, `exports` index.js / index.d.ts, `peerDependencies` on `@etcito/raa` ^0.1.0 and MAMA.
- MAMA (0.1.0 @ 864ebc3) is reached only through `src/mama-bridge.js`.
- The README's `## MAMA calls` table (specifier | export | signature | returns | MAMA file @ commit) lists every MAMA call of the package and of the compare page. No call appears in code without a row.

**API**
- `createSdfmLoader({ engine, mama, workerUrl?, workers? (min(4, hardwareConcurrency || 2)) }) -> SdfmLoader`.
- `SdfmLoader`:
  - `validate(asset)`: `{ ok, errors: { code, path, message }[] }` (S15, without MAMA);
  - `load(asset, { onProgress? })`: validate, hash check, cache by hash, worker meshing with transferable arrays, then per mesh `engine.addMesh`, `addMaterialFunction` (once per key) and `addSurfaceMaterial`;
  - `spawn(asset, parent?)`;
  - `dispose()` (later calls reject with `sdfm.disposed`).
- Shapes:
  - `SdfmAsset { id, hash, name, category, world, parts, anchors, bounds, collision }`;
  - `SdfmPart { name, groupId, rigid, parent, t, r, mesh, materials }`;
  - `SdfmSpawned { root, parts: Record<name, { node, instance }> }`.
- `SdfmError extends Error { code, path? }` (S15 codes).

**Meshing.** The bridge calls `meshAssets(doc, { pool, groups })` from `mama/core/game/mesh-assets.js`. The pool comes from `setupBrowserHost` (`mama/browser`) in the browser, or from the caller in node.
- A group result with `error` gives `sdfm.mesher_refused` (path `/document`, message naming the group with MAMA's message and fix). Any other MAMA exception gives `sdfm.worker`.
- Every part is rigid and the hierarchy is flat: one node per part under the spawn root, local transform = the MAMA frame. Each MAMA subMesh becomes one S13 mesh with its own material.
- Bridge output: `{ groups: [{ group, parts: [{ partId, name, frame, meshes: [{ material, positions, normals, uvs|null, indices }] }] }], materials: MamaMaterial[] }`.

**Materials**
- `MamaMaterial { name, key, body, helpers: [{ id, wgsl }], constants, fields }`:
  - key = sha256 of canonicalJson({ body, helpers, constants, fields });
  - body = wgslBody verbatim;
  - constants = the binding-7 u32 words;
  - an unresolvable helper gives `sdfm.material_unmappable`.
- `mapMaterial(m) -> { functionName: 'sdfm.' + key.slice(0, 16), wgsl, desc: { function, blend: 'opaque' }, dropped }`. The wgsl is an S11 `material_main`:
  - MaterialFull is declared under a non-raa_ name and the helpers are inlined;
  - the binding-7 reads become `const SDFM_K = array<u32, N>(...)`;
  - fields are mapped onto `raa_default_output(in)`, and the unmatched ones are listed in `dropped`;
  - a forbidden token, a raa_ identifier or more than 65536 bytes gives `sdfm.material_unmappable`.
- Every material is opaque and passes no params. The README records the MaterialFull to MaterialOutput table.

**`worker.js`** protocol: `{type: 'init', mamaUrl}`; `{type: 'mesh', id, document, mesher}` answered by `{type: 'meshed', id, parts}` or `{type: 'error', id, code, message}`.

Tests (`node --test packages/raa-sdfm/test/`): validate, map-material, loader (`test/fake-mama.js` reproduces the real meshAssets shape). Fixtures `test/fixtures/*.asset.json` use `game: 'raa-test'`.

## Module: sdfm-compare
Files: `examples/sdfm-compare/**`, `scripts/build-sdfm-compare.sh`.

A side-by-side page for the same fixture and camera: MAMA's browser host viewport on the left, RAA through `@etcito/raa-sdfm` on the right.

`scripts/build-sdfm-compare.sh --mama <MAMA browser build dir>` builds `dist/sdfm-compare/`:
- the staged raa package and the raa-sdfm sources;
- the MAMA build, copied and never committed;
- `scenes.json` from `examples/sdfm-compare/scenes/*.js` and `sizes.json`.

The page op and records follow S14 (`--page sdfm-compare`), with LOOKDEV_CAPTURE timing.

## Module: proof-m12
Files: `proofs/m12-sdfm.mjs`, `examples/sdfm-compare/scenes/m12-fixtures.js`, `proofs/results/lookdev/m12-*.json`, `docs/dogfood/m12.md`.

Scene `m12-fixtures` has one `compare-<fixture>` variant per fixture (at least 3: a multi-material rounded crate, blended shapes, a figure with parts). No variant is judged; the owner reviews them (R12).

Checks:
- `unit`: `node --test packages/raa-sdfm/test/`;
- `isolation`: nothing under crates/ names MAMA, and packages/raa-sdfm passes neutrality;
- `api-doc`: every export of `packages/raa-sdfm/index.d.ts` is in its README;
- `captures` (m12, page sdfm-compare).

## Module: raa-ssr
Files: `crates/raa-ssr/**`.

Screen-space reflections (S12 `ssr`). Depends on raa-core, raa-pipeline, raa-lit, raa-shader, raa-ibl, raa-math.

- `SsrDesc { intensity (1), max_roughness (0.6), thickness_m (0.15) }`. `validate` gives `post.invalid`.
- `pub mod hiz`: the CPU twin of the Hi-Z reduction (mips to 1x1; 2x2 min including odd edges).
- `SsrFeature::install(gpu, pipeline) -> SsrHandle`:
  1. builds a Hi-Z min pyramid of `depth` and a mip chain of hdr.src;
  2. for pixels with roughness <= max_roughness, marches the reflected ray through Hi-Z (64 steps + 8 binary);
  3. samples by roughness and cone footprint, fading at the edges, by length and on back-facing hits;
  4. composites `hdr + (ssr - env_specular) x F(F0) x confidence x intensity` into hdr.dst, and swaps.
- `SsrHandle`: `set`, `get`, `set_camera(view, proj)`. A singular matrix keeps the previous camera.

Tests: variants.rs, hiz.rs.

## Module: raa-dof
Files: `crates/raa-dof/**`.

Depth of field (S12 `dof`). Depends on raa-core, raa-pipeline, raa-shader, raa-math.

**Description and CoC**
- `DofDesc { focus_distance_m (10), f_stop (2.8), focal_length_mm (50), autofocus: Option<[f32; 2]> (uv), max_coc_px (24 at 720p) }`. `validate` gives `post.invalid`.
- `coc_px(z, d, viewport_h)`: thin lens with a 36 mm sensor mapped onto viewport_h. It is signed (near negative) and clamped to max_coc_px x h / 720.
- `coc_px_at_focus(z, desc, viewport_h, s_prime)` equals `coc_px` with focus s' within 1e-3 relative.

**`DofFeature::install(gpu, pipeline) -> DofHandle`**
1. Autofocus reads depth at the uv (compute, smoothed; the first frame snaps).
2. Full-res CoC.
3. 16x16 tile min/max CoC with dilation.
4. A half-res near/far prefilter.
5. A 48-tap spiral gather per layer; the near alpha comes from coverage.
6. A bilateral upsample composite into hdr.dst, then a swap.

**DofParams** (128 B): after `lens` (focus s m, K = A f / (s - f) px, max CoC px, near) comes `lens2` = (f in metres, 0, 0, 0).
- In cs_coc, s' = max(autofocus s', f + 0.001), K' = lens.y (s - f) / (s' - f), and CoC = clamp(K' (z - s') / z, ±lens.z).
- With autofocus off, s' = s.

`DofHandle`: `set`, `get`, `set_camera(view, proj)`.

Tests: coc.rs (including 600 mm at 1 m), the layout test (size 128, lens2 offset), variants.rs.

## Module: raa-fog
Files: `crates/raa-fog/**`.

Height fog and aerial perspective (S12 `fog`, EnvGpu fog fields). Depends on raa-core, raa-pipeline, raa-lit, raa-shader.

- `FogDesc { density, height_falloff, base_height_m, start_m, color, inscatter, aerial_strength ([0, 2], 0 = off), aerial_distance_m }`. Presets: `clear`, `haze`, `snow`, `jungle`, `night`. `validate` gives `post.invalid` naming the field.
- `FogFeature::install(gpu, pipeline, lit) -> FogHandle`. FeaturePass `fog` (order 68):
  - integrates exponential height fog analytically along the view ray from `depth`;
  - fog colour = env cube at max mip x color + HG (g 0.6) sun inscatter;
  - aerial amount = clamp((1 - exp(-dist / aerial_distance)) x strength, 0, 1) toward the horizon radiance;
  - sky pixels get the height part near the horizon only;
  - writes the EnvGpu fog fields;
  - adds the `Fog` slot `height` (priority 10).
- `FogHandle`: `set`, `get`.

Tests: integral.rs (versus quadrature within 1e-3), desc.rs (aerial boundaries -0.1, 0, 2, 2.01), variants.rs.

## Module: raa-shafts
Files: `crates/raa-shafts/**`.

Volumetric sun shafts (S12 `sunshafts`). Depends on raa-core, raa-pipeline, raa-shader, raa-shadow, raa-math.

- `ShaftsDesc { intensity (1), density (0.02), max_distance_m (60) }`. `validate` gives `post.invalid`.
- `ShaftsFeature::install(gpu, pipeline) -> ShaftsHandle`:
  - a half-res, 32-step, jittered march sampling the CSM map (bindings 0-2) and the fog density (binding 11), with HG g 0.7;
  - a depth-aware bilateral upsample added into hdr.dst, then a swap;
  - a no-op when `shadow.csm` is disabled (ShadowGpu.params.w = 0).
- `ShaftsHandle`: `set`, `get`, `set_camera(view, proj)`, `set_sun(dir, color)`.

Tests: variants.rs.

## Module: raa-grading
Files: `crates/raa-grading/**`.

HDR colour grading before tonemap (S12 `grading`). Depends on raa-core, raa-pipeline, raa-shader.

- `GradingDesc { temperature_k (6500), tint, exposure_ev, contrast (1, around 0.18 in log2), saturation (1), vibrance, lift, gamma, gain, shadows_tint, highlights_tint, split_balance, vignette }`. Presets: `neutral`, `candy`, `ice`, `jungle`, `neon`. `validate` gives `post.invalid`.
- `white_balance_matrix(temperature_k, tint)`: CAT02 in LMS against D65.
- `GradingFeature::install(gpu, pipeline) -> GradingHandle`: a fullscreen pass from hdr.src to hdr.dst, applying in order:
  1. white balance;
  2. exposure;
  3. log contrast;
  4. lift/gamma/gain (ASC CDL in log);
  5. split toning;
  6. luma-preserving saturation and vibrance;
  7. vignette.
  Then a swap.
- `GradingHandle`: `set`, `get`.

Tests: wb.rs (6500 K identity within 1e-3), variants.rs.

## Module: api-m14
Files: `crates/raa-js/src/api/m14_camera_effects.rs`, `packages/raa/types/m14-camera-effects.d.ts`, `docs/api/m14-camera-effects.md`.

- `setSsr({ intensity?, maxRoughness?, thickness? })`.
- `setDof({ focusDistance?, fStop?, focalLength?, autofocus?: [u, v] | null, maxCoc? })`.
- `setFog({ preset?, density?, heightFalloff?, baseHeight?, start?, color?, inscatter?, aerialStrength? ([0, 2]), aerialDistance? })`.
- `setSunShafts({ intensity?, density?, maxDistance? })`.
- `setGrading({ preset?, temperature?, tint?, exposure?, contrast?, saturation?, vibrance?, lift?, gamma?, gain?, shadowsTint?, highlightsTint?, splitBalance?, vignette? })`.
- `applyEnvironment({ sky?, sun?: { dir, color, intensity }, fog?, post? })`:
  - `post` is a preset name (`candy`, `ice`, `jungle`, `neon`, `neutral`: grading + bloom + tone mapping) or `{ grading?, bloom?, dof?, ssr?, sunShafts?, toneMapping? }`;
  - applied in that order;
  - unknown names throw `post.unknown_preset`.

`frame` feeds the camera and sun to the ssr, dof and shafts handles.

## Module: proof-m14
Files: `proofs/m14-camera-effects.mjs`, `examples/lookdev/scenes/m14-camera-effects.js`, `proofs/results/lookdev/m14-*.json`, `docs/dogfood/m14.md`.

Scene `m14-camera-effects`: an obstacle-course-like vista with every P2 feature.
- Variants:
  - `world-candy`, `world-ice`, `world-jungle`, `world-neon`, judged vs ref-1 to ref-4 respectively, all through applyEnvironment presets;
  - on/off pairs for `ssr`, `dof`, `fog`, `sunshafts`, `grading`.
- Checks: `unit` (raa-ssr, raa-dof, raa-fog, raa-shafts, raa-grading), `wasm-check`, `api-doc` (m14), `captures`, `judge`.

## Module: raa-deform
Files: `crates/raa-deform/**`.

Per-instance squash-and-stretch and jiggle (S10 InstanceExtra, S11 Deform slot). Depends on raa-math, raa-shader, raa-lit.

**`provider()`**: `Deform` provider `squash-jiggle` (priority 10, helpers `raa_squash_jiggle_`).
- Squash with axis a and amount s (s clamped to >= -0.9): p' = (p.a) a (1 + s) + (p - (p.a) a) / sqrt(1 + s). Normals use the inverse-transpose, renormalised.
- Jiggle: p'' = p' + offset x smoothstep(0, 1, clamp(dot(p', up) / h, 0, 1))². The normal is tilted by the weight gradient.

**CPU twins**: `squash_point`, `squash_normal`, `jiggle_offset_weight`.

**Springs**
- `JiggleConfig { stiffness (120), damping (0.35), amount (1), height_m }`. `validate` gives `deform.invalid`.
- `JiggleSprings`: `set(tag, Option<JiggleConfig>)` and `step(dt, tag, world) -> [f32; 3]`:
  - a spring on the lag of the world translation, semi-implicit Euler;
  - expressed in node-local axes and clamped to 0.5 h;
  - deterministic.

`install(lit)`.

Tests: squash.rs (volume within 1e-4), jiggle.rs, variants.rs.

## Module: api-m15
Files: `crates/raa-js/src/api/m15_animation.rs`, `packages/raa/types/m15-animation.d.ts`, `docs/api/m15-animation.md`.

`frame` steps the jiggle springs of configured instances from their node world matrices and writes their extras.
- `setInstanceTint(instance, [r, g, b, a])`: sRGB rgb, linear alpha multiplier.
- `setInstanceParams(instance, params)`: 16 numbers.
- `setSquash(instance, { axis? ([0, 1, 0]), amount })`.
- `setJiggle(instance, { stiffness?, damping?, amount?, height } | null)`.
- `setInstanceExtras(instances: Uint32Array, data: Float32Array)`: 32 floats per instance:
  - tint 4;
  - squash 4;
  - jiggle 4 (local offset m, height m; written as given, and overwritten next frame for instances with a spring);
  - params 16;
  - reserved 4.
  A length mismatch gives `api.length_mismatch`. Everything is checked before any write.

Rigid hierarchies use `addNode` and `setTransforms`.

## Module: proof-m15
Files: `proofs/m15-animation.mjs`, `examples/lookdev/scenes/m15-waddle.js`, `proofs/results/lookdev/m15-*.json`, `docs/dogfood/m15.md`.

Scene `m15-waddle`: a procedural plush penguin (body, belly, head, beak, eyes, flippers, feet as rigid parts) in three tinted copies.
- `frame(engine, t)` drives a 1.2 s waddle: roll, foot lift and yaw, flipper flap, head bob, squash on foot plant and belly jiggle. Jiggle is written through `setInstanceExtras` for captures.
- Variants: `f00`..`f11` (timeMs 0..1100; `f00` and `f06` judged vs ref-1 and ref-2), `deform-on`/`deform-off`, `tints`.
- Checks: `unit` (raa-deform), `wasm-check`, `api-doc` (m15), `captures`, `judge`.

## Milestone proofs

Each milestone's proof, run from the repository root at its final commit.

- M1 Meshes and HDR base: addMesh, configurable pass pipeline, HDR + AgX/ACES + MSAA 4x, lookdev page: `node proofs/m7-meshes-hdr.mjs` — timeout_s 3600
- M2 Light I: procedural sky with clouds and image-based lighting derived from it: `node proofs/m8-sky-ibl.mjs` — timeout_s 3600
- M3 Light II: soft cascaded sun shadows, GTAO and contact shadows: `node proofs/m9-shadows-ao.mjs` — timeout_s 3600
- M4 Materials I: WGSL material functions, layered BRDF (clearcoat, sheen), fur shells, emissive with bloom: `node proofs/m10-materials.mjs` — timeout_s 3600
- M5 Materials II: transmission and refraction with absorption, sorted translucency with linked-list OIT: `node proofs/m11-transmission.mjs` — timeout_s 3600
- M6 Effects I: animated liquid surfaces and GPU particles: `node proofs/m13-liquids-particles.mjs` — timeout_s 3600
- M7 SDF loader @etcito/raa-sdfm: sdfm.game-asset to RAA meshes and materials, side by side with SDF Modeler: `node proofs/m12-sdfm.mjs` — timeout_s 3600
- M8 Effects II: SSR on glossy floors, depth of field, height fog and aerial perspective, sun shafts, colour grading, environ: `node proofs/m14-camera-effects.mjs` — timeout_s 3600
- M9 Animation hooks: rigid part hierarchies from JS, squash-and-stretch and jiggle in the vertex stage, per-instance tint an: `node proofs/m15-animation.mjs` — timeout_s 3600
