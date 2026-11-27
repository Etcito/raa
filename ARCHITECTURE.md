# RAA architecture

This contract governs the linked `raa`, `mama` and `sdfm` repositories. RAA is an extensible Rust/WebAssembly/WebGPU renderer with the JS package `@etcito/raa`; MAMA owns procedural scene data; SDF Modeler owns document authoring. The consolidated contract lives only in `raa:ARCHITECTURE.md`. Conventions apply to every module. The numbered shared sections define data shapes that modules implement and never redefine.

## Conventions

### Repositories, toolchain, layout
- Writable in P4: `raa` and `sdfm`, merged raa then sdfm (earlier projects also wrote `mama`, merged first). `raa:`, `mama:`, `sdfm:` qualify paths; unqualified paths are RAA's. MAMA (`1c4ad64c5195a01b42576716fb7391c22489b298`) owns procedural scene data and the saved mesh format; in P4 it is a read-only sibling built with `npm run build` before sdfm reads it. Solstice (`7e465b599c13da165948ebaa44df824c36c3ae07`) is read-only; adapted files keep the MIT notice and `Copyright (c) 2026 SamG-Coder`. royal-fiesta and haa are read-only, never tested in. Kong resolves linked packages; host `dist/` and abandoned branches are never inputs. MAMA never imports RAA, nothing under `crates/` names MAMA; `packages/raa-sdfm` is the passage.
- One owner module per file; an explicit file or narrower directory wins over a glob. A task writes one repository, only its milestone's module files, in one unpushed commit; repairing an earlier owner needs the lead. A `Cargo.lock` refresh, one api path-dependency line in `crates/raa-js/Cargo.toml` appended `.nmc/commands.json` entries and the bounded seam edits of S51 are not ownership changes.
- Workspace: edition 2024, rust-version 1.96.0, version 0.5.0, resolver 2, members `crates/*`, `demos/*`, `tests/*`, `Cargo.lock` committed.
- Closed dependencies: `wgpu` 30.0.1 (no default features, `wgsl`; `webgpu` on wasm32, `vulkan`, `gl` native), `naga` 30.0.1 (`wgsl-in`, validation), `wasm-bindgen` =0.2.129, `wasm-bindgen-futures` =0.4.79, `js-sys`/`web-sys` =0.3.106, `pollster` 1.0.1. No serde, glam, nalgebra, bytemuck, rayon, rustc-hash, log framework, network service or threads.
- Every crate builds for host and `wasm32-unknown-unknown`; `[profile.wasm-release]`: opt-level z, LTO, one codegen unit, panic abort, strip. raa-pipeline, raa-shader, raa-lit, raa-mesh, raa-sky, raa-oit, raa-fog, raa-cluster, raa-vg, raa-gi and raa-taa are sibling `path` dependencies.
- Layout: `crates/<name>/{Cargo.toml, README.md, src/lib.rs, tests/<topic>.rs}`, `demos/<mN-name>/`, `tests/<name>/`, `proofs/`, `scripts/`, `web/`, `tools/`, `packages/raa/` (`@etcito/raa`), `packages/raa-sdfm/`, `examples/{basic,lookdev,sdfm-compare,sky-weather,virtual-geometry,global-illumination,post}/`.
- Tools: `node` (`${NODE:-node}`), `wasm-bindgen` CLI, `python3` (judge), repository-local TypeScript; no fetching `npx`.

### Coordinates, units, colour
- Right-handed, +Y up, camera forward -Z, metres; CCW front faces, outward normals; column-major f32 matrices, column vectors; quaternions `[x, y, z, w]`. Clip depth [0, 1], `Depth32Float`, `Less`, clear 1; the sky draws at depth 1 with `LessEqual`. RAA is standard-Z, solstice reversed-Z: adapted depth code uses the S32 depth helper.
- Rust and GPU colours are linear; asset and JS colours sRGB unless stated (sky colours and `setLight` linear, S17 lights sRGB). MAMA `ProcLight.color` is linear and gets the adapter's OETF before S17; document lights never do.
- Kilometres only behind S23's conversion. Latitude degrees north; `north` radians about +Y (north = -Z, east = +X at zero); wind = compass direction FROM, degrees. Cluster error is object-space metres; the LOD threshold is pixels. GI voxel sizes are metres.
- Exposure values are EV (base-2 stops), a linear multiplier is `2^ev`; luminance is `dot(rgb, (0.2126, 0.7152, 0.0722))` of linear HDR colour.

### Errors, state, allocation
- One error type per crate, stable `code()` `<area>.<snake_case>`; lit crates return `raa_core::RaaError`. Refusals are typed, never NaN, panic or skip; a refused setter changes nothing. Sky, virtualized-geometry and GI edits reject atomically with `sky.invalid`, `vg.invalid`, `gi.invalid` and `<key>: <reason>`, post settings with `post.invalid` naming the JS key; a malformed argument is `api.invalid_argument`; a disposed Engine answers `api.disposed`. S17 light operations are atomic; `setTransforms` keeps entries applied before a failure.
- Shader setters validate with naga synchronously; WebGPU errors arrive asynchronously. No `unwrap`, `expect` or indexing panic on public-data paths.
- No `static mut`, `thread_local!` or lazy globals in libraries except `raa-alloc-count`'s `#[global_allocator]`; features share `Rc<RefCell<..>>` with their handle. The v0 frame path allocates nothing after warm-up; lit crates reuse scratch.
- One Engine invocation advances playback once for all views; GPU histories are view-local and commit only for submitted work.
- `unsafe` needs `// SAFETY: <reason> (test: <test_fn_name>)` above it; the only unsafe code is `unsafe impl GlobalAlloc for CountingAlloc`.
- Post features (S46-S48) default off; their compute runs in WGSL or Rust, never TypeScript; they add no group-0/1 binding, variant or feature bit; ported maths has a CPU twin tested against hand-derived values.

### Testing and checks
- Unit tests in `#[cfg(test)] mod tests`, integration in `crates/<c>/tests/<topic>.rs`, text fixtures only. Native GPU tests use `raa_core::testing::software_gpu()` and print `SKIP <test name>: no software adapter (<reason>)`. Every composed variant is naga-validated.
- Sky, weather, virtualized geometry, GI and post passes use no reference image, pixel metric, output hash or CPU/GPU parity: CPU tests cover data, parsers, state, layouts and variants; CPU twins of adapted kernels run on the CPU only against hand-derived or analytic JSON fixtures; GPU results are judged renders. A card's tests cover its subject, at most three files.
- Branch checks: RAA `cargo check`, affected `node --test`, `npm run typecheck:types` when shipped declarations or `packages/raa/index.js` change, `bash scripts/check.sh` at milestone verification; MAMA `npm run typecheck`, `npm run build`, `node scripts/test.mjs --only <selector>`; SDFM `npm run typecheck`, `npm run build`, `npm run guard`, `node scripts/proof/unit.mjs --only <anchored selector> --jobs 1` after `node node_modules/typescript/bin/tsc -p tsconfig.node.json`, and the managed Chrome boot check. Readers of dist, package or WASM have a build predecessor; host work respects `NMC_CPUS` (default 1).
- rustfmt on changed files; clippy per changed crate `--all-targets --no-deps -- -D warnings`. Asserted shader, API or binding text changes with its pinning tests (S27, S35, the P4 tests named in S36-S45).

### Builds, browser runs, judge, commands
- `scripts/build-web.sh <crate-dir> [--features <list>]`: `wasm-release`, `wasm-bindgen --target web` into `dist/<dir>/pkg`, copies `<crate-dir>/web/*` and `web/common/*`, writes `sizes.json` (S7). `node web/serve.mjs <dir> [--port N]` serves 127.0.0.1 with `Cache-Control: no-store`.
- Browser rendering is a page published to managed Windows Chrome on a physical GPU through nomorechat's GPU client; nothing substitutes, a missing client is a visible failure (S22). Credentials are never printed.
- `hardware = !isFallbackAdapter && none of /swiftshader|llvmpipe|lavapipe|software|basic render/i matches vendor, architecture, device or description`; records name the adapter. `firstFrameMs` = navigation-relative time when `onSubmittedWorkDone` of frame 0 resolves; `cpuMs` = delta around one frame call.
- Judge: `python3 $NMC_JUDGE_CLIENT <capture.png> --description <text> [--reference <png> ...] [--aspect <a>] --min-score 6`, pass = exit 0, output redacted within 4000 characters. References come only from lookdev (S14) and rows with `judgeReference` (S40). Lookdev renders at a fixed time with dt 0; jiggle scenes write `setInstanceExtras` floats 8..11.
- Captures and binaries are never committed (`scripts/check-no-assets.sh`); `proofs/captures/` and `out/` (sky-weather, virtual-geometry, global-illumination, post and their `-editor` stores) are ignored; preset content and test textures are generated at page load. VG and GI captures use exactly one viewport.
- `.nmc/commands.json`: appended single-program entries `{name, command, timeout_s, cwd?, after?}` validated by `nmc-commands-check`; a run names its ordered selection. Builds, GPU jobs (<= 300 s) and judge chains run programmatically; model cards never run GPU commands, programmatic cards never write code; one GPU card at a time.

### Values
| name | value |
|---|---|
| WASM_GZIP_MAX | 300000 bytes gzip (level 9, node zlib) per shipped .wasm |
| FIRST_FRAME_MAX_MS | 500 |
| CPU_MS_MEDIAN_MAX | 4.0 ms median CPU per frame, 1000 instances, hardware only |
| ALLOC_RUN | 1000 instances, 120 warm-up, 600 measured frames, 0 allocations |
| BROWSER_RUN | warmup 60, frames 300 |
| CAPTURE | 1280x720, dpr 1; resize to 960x540 (30 frames) and back |
| CLEAR_COLOR | linear (0.05, 0.06, 0.08, 1.0) |
| QUALITY | `Quality::LOW` 8, `LOD0` 24, `HIGH` 64 segments, clamped [3, 256] |
| TRIANGLE_BUDGET_DEFAULT | 1500 triangles per asset at LOD0 |
| ASSET_CAPS | nodes <= 64; nesting <= 8 (roots 0); quaternion norm within 1e-3 of 1; scale > 0 |
| WELD_EPS | 1e-6 x max(1, bbox diagonal) |
| NORMAL_RULE | abs(len - 1) <= 1e-4; <= 45 degrees from the area-weighted face-normal mean; 0 inward |
| SDF_REF_TOL | uniquePositions and triangles equal; bbox diff <= 1e-5 x max(1, diagonal); area, volume rel. diff <= 1e-4 |
| CSG_EPS | 1e-6 x max(1, diagonal of host and tool bbox union); volume identity rel. diff <= 1e-4 |
| HDR_FORMAT | `Rgba16Float` |
| MSAA | forward 4 by default, 1 allowed |
| TONEMAP_DEFAULT | agx, look none, 0 EV, dither on |
| WEBGPU_OPTIONAL | `FLOAT32_FILTERABLE`, `RG11B10UFLOAT_RENDERABLE`, requested when available |
| SKY_ENV | cube 256 px per face, HDR_FORMAT, 6 mips |
| LOOKDEV_CAPTURE | 1280x720, 30 warm-up + 1 frame at `timeMs` (default 1000) |
| JUDGE_MIN_SCORE | 6 of 10 |
| PAIR_RUN | on/off pairs: 30 warm-up + 300 measured rAF intervals, 1280x720, dpr 1, no threshold (S26) |
| VG_PAIR_RUN | PAIR_RUN plus drawn clusters and triangles (S33) |
| VG_CLUSTER | <= 128 triangles, 255 vertices; 16 B vertex, 48 B cluster and node (S29) |
| VG_ERROR_PIXELS | 1 px, [0.25, 64] |
| VG_BAKE_BUDGET | 4 ms per render invocation, [0.5, 100] |
| VG_MSAA | virtualized geometry needs msaa 1 |
| VG_VIEW_SLOTS | 11 slots of 512 B: main 0-1, cascades 2-5, GI 6-10 |
| VG_HIGH_POLY | displaced icosphere, subdivision 6, 81920 triangles |
| POST_SETTLE_MAX_FRAMES | 600; a post row draws until `settled` holds on 2 consecutive reads (S49) |
| POST_UPSCALE_SCALE | 0.67 `setRenderScale` (S49, S50) |
| ATLAS_MIP_LEVELS | 5 (S48) |
| ATLAS_ALIGN | 16 texels (S48) |
| GI_FINEST_VOXEL | 0.2 m; level i voxel = 0.2 x 2^i m |
| GI_QUALITY_DEFAULT | resolution 96, levels 5, probeRays 8 (64 or 96; 3-5; 6 or 8) |
| GI_VOLUME_BYTES | 32 R^3 L + 8 R^3 (148,635,648 at 96/5; 27,262,976 at 64/3) |
| GI_SETTLE_FRAMES | 240 frames after the last rebuild round |
| GI_PROBE_SPACING | 16 px |
| GI_PAIR_RUN | PAIR_RUN plus GI quality and `allocatedBytes` (S40) |

### Proofs and scope
- Proofs: `node proofs/mN-<name>.mjs [--quick|--full] [--only <regex>]` on `proofs/lib/proof.mjs` (`p2.mjs` for lookdev), <= 5 checks printing `CHECK <label> | measured: <value> | threshold: <threshold> | PASS|FAIL`; lookdev checks `unit`, `wasm-check`, `api-doc`, `captures`, `judge`. Sky, weather, VG, GI and post add no proof script or threshold. Dogfood notes: `docs/dogfood/mN.md`.
- Neutrality: `scripts/check.sh` rejects `royal|fiesta|monster|racer` (case-insensitive, comments included) in crates/, demos/, tests/, packages/, examples/ and docs/API.md. raa is public: no local path, machine name or private project name.
- The JS API grows through `packages/raa/types/mNN-*.d.ts` and `docs/api/mNN-*.md`; `index.d.ts` and `docs/API.md` stay v0 apart from index lines and the `addTexture` text. VG is m29, GI m30, TAA m31 (`docs/api/m31-taa.md`, no `.d.ts`). Automatic exposure is m33, sharpening m34.
- Not provided: a dedicated upscaler (render scale + TAA + sharpening is the path), BC7/BC5 compression, a GPU profiler, precipitation particles, a second distance fog, virtualized skinned, deformed, transparent or fur instances, cluster streaming or paging, glossy GI reflections. Local-light shadows exist only through GI (S39) and the area-light tiers (S44).

### Quality bar
- **Q1** `scripts/check.sh` exits 0. **Q2** shipped .wasm gzip <= WASM_GZIP_MAX. **Q3** `firstFrameMs` <= FIRST_FRAME_MAX_MS, 1000-instance `cpuMsMedian` <= CPU_MS_MEDIAN_MAX on hardware. **Q4** `raa-bench frame-allocs` reports 0 over ALLOC_RUN. **Q5** the 8 primitive kinds at LOW, LOD0, HIGH are closed, satisfy NORMAL_RULE, hash identically and match the reference within SDF_REF_TOL. **Q6** prims v0 round-trips byte-identical; invalid cases refuse. **Q7** supported CSG cases are watertight within CSG_EPS, others refuse. **Q8** raa-ext-test extends RAA without editing other crates; every JS export is documented. **Q9** records name the adapter and state `hardware`. **Q10** every lookdev variant has an S14 record, `nonEmpty` and `hardware` true, no GPU errors. **Q11** judged variants exit 0. **Q15** every toggleable feature has `-on`/`-off` variants and stays valid when disabled.
- **R1-R6, R12** (owner review): centred triangle; 1000 lit instances; crack-free primitives; assets matching composition; clean boolean cuts; example page through `@etcito/raa`; sdfm-compare beside SDF Modeler. **QB1-QB9** (local lights): earlier paths unchanged; lights per S11; ownership kept through move, rotate, detach, delete, reattach; bind-group and storage limits hold; overflow reported.
- **P2-Q1..Q8** (sky): day, twilight and night skies; background, reflections, IBL and fog share the sky state; presets differ and clouds and their shadows move; transitions leave no stale history; readable Game controls kept across reload; invalid edits keep the last valid state; layouts and limits hold; every image current and judged >= 6.
- **P3-Q1..Q10** (VG): (command) watertight monotonic cuts independent of step budget; a dense field virtualized with every feature; no cracks or popping with distance; occlusion lowers drawn clusters; cascade shadows both ways; animated and transparent content stays forward with coded refusals; forward while baking; (command + judge) passes off allocate nothing, `-on`/`-off` presets, distinct debug views; SDF Modeler export and setting; (command) images current, judged, pairs with VG_PAIR_RUN.
- **P4-Q1..Q10** (GI): (command) twins pass fixtures, entry points validate, layout twins match, check green; debug views show voxels, radiance, contours and levels; colour bleed, interior bounce and emissive light; point and spot light stop at a wall; open ground keeps brightness, overhang gains bounce, glossy floor reflects through the volume; both qualities lit, animated and transparent content receives GI, a moved instance relights; (msaa 1) VG field voxelized in one grid with forward; (command + judge) disabled GI records nothing, 0 bytes, dummies bound, `-off` = the earlier frame; SDF Modeler setting labelled, persistent, renderer kept, refusals prefixed; (command) images current, judged, pairs with GI_PAIR_RUN.
- **P5** (post): **P5-Q1** (command) S46-S48 CPU twins pass, WGSL validates, refusals change nothing. **P5-Q2** auto exposure makes interior and sunlit exterior readable once settled. **P5-Q3** sharpening is crisper at scale 1 and 0.67 without halos. **P5-Q4** atlas mips remove distant moire and sparkle, near field as sharp. **P5-Q5** defaults off keep earlier looks; `-on`/`-off` rows exist. **P5-Q6** the SDF Modeler Game-view post controls are labelled, off, kept across reload and work. **P5-Q7** (command) images current, judged >= 6, pairs carry PAIR_RUN.

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
Little-endian `to_le_bytes`, byte offsets, ranges exclude the upper end, matrix columns 16 B apart. Owners: raa-mesh (streams), raa-lit (frame, instance, light writers), raa-pipeline (lighting resources), raa-shader (declarations).

**Streams**: 0 = S1 Vertex (24 B, locations 0-1); 1 (40 B) = `uv` @0 (loc 10), `tangent: f32x4` @8 (loc 11, w ±1), `color` @24 (loc 12, linear); defaults (0, 0), `default_tangent(normal)`, white. Stream 0 and indices also have STORAGE usage. The lit path replaces the S1 Instance stream with a `u32` scene-instance slot at location 2; `raa_instance(id)` reads binding 7. `RaaInstanceGpu` (128 B): model columns @0-48, inverse-transpose normal columns @64-96, `material_index` @112, `tag` @116, `palette_base` @120. Posed variants read a 64 B rest-skin record from vertex slot 3 (position @0, normal @12, tangent @24 loc 11, joints u16x4 @40 loc 13, weights @48 loc 14). Each scene instance owns one slot for bindings 2, 7, 8; the tag selects the group's extras.

**Group 0** (0-7 VERTEX|FRAGMENT, 8 VERTEX; read-only storage): 0 `FrameGpu` (768 B); 1 `array<SurfaceMaterialGpu>` (256 B); 2 `array<InstanceExtraGpu>` (128 B); 3 material atlas (`Rgba8Unorm` 4096², 5 mips once a texture exists, 1 x 1 before; S48), 4 its sampler (linear incl. mipmap, clamp), 5 atlas rects (16 B (x, y, w, h) / 4096; negative w = level 0 only; id 0 and removed ids zero); 6 `array<u32>` material data (`raa_material_word`, `raa_material_f32`, `raa_material_data_base`; words 0-3 zero); 7 `array<RaaInstanceGpu>`; 8 `array<RaaPreviousInstanceGpu>`.

**FrameGpu** (768): `view_proj` @0, `view` @64, `proj` @128, `inv_proj` @192, `inv_view` @256, `camera_pos` @320 (w time s), `sun_dir` @336 (travel, w angular radius 0.0047), `sun_color` @352, `ambient` @368, `viewport` @384 (w, h, 1/w, 1/h), `clip` @400 (near, far, frame index, dt), `flags: vec4<u32>` @416 (sRGB encode, msaa, feature words z and w), `exposure` @432 (ev, 2^ev, mapper agx 0 / aces 1 / linear 2, look none 0 / punchy 1 / golden 2; static, never written by S46), `previous_view_proj` @512, `unjittered_view_proj` @576, `camera_jitter` @640 (current, previous NDC), `light_info: vec4<u32>` @656 (lights, tile columns, rows, stride 128), `temporal: vec4<u32>` @672 (parity, accumulation, camera history valid, 0); other bytes 0. `view`/`proj` are unjittered; the Halton 2/3 jitter (samples 1-8 minus 0.5, (2dx/w, -2dy/h)) applies to `view_proj` and `inv_proj` only while `taa` is enabled, with `temporal.y` 1 (S43). Creation, resize, camera discontinuity and jitter-consumer changes invalidate camera history; history commits per submitted view.

**Feature ids** (`raa_feature_enabled(id)`: 0-31 flags.z, 32-63 flags.w): 24 light tiling; 27, 28, 36 raa-vg (S32); 29 GI active, 30 GI local-light visibility, 37 GI reflections (S37); 39 area form-factor mode (S44). Reserved: 25, 26, 31 (TAA sets no bit), 32-35, 38.

**RaaPreviousInstanceGpu** (64 B): affine rows @0, 16, 32, `previous_valid` @48. New, reused or re-posed slots and removals clear validity; only submitted slots commit; a settled frame uploads nothing.

**SurfaceMaterialGpu** (256 B): `base` @0 (linear rgb, opacity), `emissive` @16 (rgb; w = diffuse transmission colour as the exact float r + 256 g + 65536 b of 8-bit channels), `surface` @32 (roughness [0.03, 1], metallic, clearcoat, clearcoat roughness), `sheen` @48, `transmission` @64 (amount, ior, thickness m, diffuse transmission weight), `attenuation` @80 (rgb, distance m), `header: vec4<u32>` @96 (model flat 0 / standard 1 / function 2, function id, flags, data base word), `p[8]` @112, `fur` @240 (length m, density per cm², droop, shells). Flags 1 TRANSPARENT, 2 TRANSMISSIVE, 4 FUR, 8 DOUBLE_SIDED, 16 NO_SHADOW, 32 VERTEX_FN, 64 UNLIT, 128 CEL, 256 TRACE_MIRROR, 512 DIFFUSE_TRANSMISSION (written by `SurfaceMaterial::gpu_flags` exactly when the weight is > 0).

**Diffuse transmission** (opaque thin translucency, `SurfaceMaterial::diffuse_transmission` / JS `diffuseTransmission`): the material stays in the Opaque class (prepass, depth, shadow caster, no OIT). The camera forward key of its draws carries flag 512 (raa-lit `FORWARD_VARIANT_FLAGS`, never the prepass), and only that variant compiles the back term of `opaque_lighting.wgsl`: the front diffuse keeps base x (1 - t), and a Lambert term with albedo base x colour x t is gathered around -N from the sun (shadow slot re-evaluated with the receiver pushed along -N), the tile's local lights (falloff, cone, n.l, LocalVisibility slot; area lights by their diffuse form factor), and `raa_ddgi_sample(p, -N, -view)` over the SH / flat ambient. t is scaled by 1 - metallic. Virtual-geometry shading, the transparent and fur passes and the GI/probe ray shaders ignore it.

**InstanceExtraGpu** (128 B): `tint` @0 (1), `squash` @16 ((0, 1, 0, 0)), `jiggle` @32 (offset m, height m), `params[4]` @48, `reserved` @112 (x `tint_set`); keyed by tag.

**Group 1 (lighting)**, `LightingResources` with dummies; 0-11 VERTEX|FRAGMENT, 12-20 FRAGMENT: 0 `shadow_map` depth array (cascades 0-3, hero layers 4-7); 1 `shadow_cmp`; 2 `ShadowGpu` (320 B: `cascade_view_proj[4]` @0, `splits` @256, `params` @272 (penumbra, normal offset, texel size, enabled), `params2` @288 (light size, blend band)); 3 `env_specular` cube; 4 `EnvSh` (176 B, S45); 5 `raa_brdf_lut` Rgba16Float 128x192 atlas: split-sum at x, y 0..127, three.js LTC_MAT_1 at x 0..63 and LTC_MAT_2 at x 64..127 over y 128..191 (`create_brdf_atlas`, dummy fill (1, 0)); 6 `linear_clamp`; 7 `ao`; 8 `contact`; 9 `opaque_color`; 10 `opaque_depth` R32Float; 11 `EnvGpu` (256 B: `sun_disc` @0, `zenith` @16, `horizon` @32, `ground` @48, `sky_params` @64, `background` @80, `clouds` @96, `clouds2` @112, `cloud_color` @128, `fog` @144, `fog_color` @160, `aerial` @176, `cel` @192); 12 `raa_local_lights: array<LightGpu, 256>` uniform; 13 `raa_light_tiles` R32Uint array (ceil(w/16) x ceil(h/16) x 128 layers, written by `light.tiles`); 14 `raa_gi_screen` (dummy (0, 0, 0, 1)); 15, 16 (S28); 17 `raa_gi_volume` texture_3d (dummy (0, 0, 0, 12.8)); 18 `RaaGiVolumeParams` (96 B, S38); 19 `RaaAreaStochasticGpu` (80 B: `lights: array<vec4<u32>, 2>` @0 = packed light index of slot s at byte 4s or 0xFFFFFFFF, `weights` @32 = w_stoch, `params: vec4<u32>` @64 = limit 0..8, cells 4|9); 20 `raa_area_shadow_mask` Rgba16Float array, 2 layers, viewport-sized (slot s = layer s/4, channel s%4; dummy ones).
- Layouts: full `layout()` 0..=20; `opaque_layout()` without 9, 10 (vg.gbuffer, vg.shade, ssr, shafts); `base_layout()` 0..=13, 15, 16 (sky.env, sky.background). Setters `set_shadow`, `set_env`, `set_env_gi`, `set_brdf_lut`, `set_ao`, `set_contact`, `set_opaque_copy`, `set_cloud_transmittance`, `set_gi`, `set_area_shadow` (bad view `core.resource`) and clears restoring dummies; values set in prepare bind the same frame.
- Fragment budgets: full 12 textures, 2 samplers, 7 uniforms; opaque 10/2/7; lit Forward, Transparent, FurShell 13/3/8-9; every other consumer <= 16 textures. GI compute group 1 = 3, 6, 12, 15, 16, 17, 18.

**LightGpu** (64 B): `position_range` @0, `color_inv_range_sq` @16, `direction_cos_outer` @32, `cos_inner` @48, `kind` @52 (point 0, spot 1, directional 2, area 3), `width` @56, `height` @60; for kind 3 lanes @28 (w_direct), @44 (hero layer, -1 none), @48 (w_cone) carry S44 weights.

**Group 2** is pass-private; **group 3** is the posed skin palette (64 B mat4 from `palette_base`). Group 0 has five fragment storage buffers, OIT adds three (limit eight); vertex storage six (seven posed); no fifth group. Pins: raa-lit `src/{frame,previous_instances,instance_store,handle,local_lights}.rs`, raa-shader `src/{gpu_constants,lib}.rs`, raa-pipeline `src/lighting.rs`, `tests/resources.rs`, `tests/variants.rs` of raa-lit, raa-oit, raa-fur, raa-sky.

## S11. Shader and material function contract (raa-shader)
A composed module = `PRELUDE` + `LIB` + one material function + the winning provider of each slot + `LOCAL_LIGHTING_WGSL` + the pass entry points. Variant key = (material fn id, pass kind, msaa, flags, slots hash).
```wgsl
struct MaterialInput { position: vec3<f32>, local_position: vec3<f32>, normal: vec3<f32>, tangent: vec4<f32>, uv: vec2<f32>, color: vec4<f32>, view_dir: vec3<f32>, time: f32, front_facing: bool, frag_coord: vec4<f32>, instance_params: array<vec4<f32>, 4>, record: SurfaceMaterialGpu }
struct MaterialOutput { base_color: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32>, normal: vec3<f32>, ao: f32, opacity: f32, clearcoat: f32, clearcoat_roughness: f32, sheen_color: vec3<f32>, sheen_roughness: f32, transmission: f32, ior: f32, thickness: f32, attenuation_color: vec3<f32>, attenuation_distance: f32 }
struct VertexMaterialInput { local_position: vec3<f32>, normal: vec3<f32>, uv: vec2<f32>, color: vec4<f32>, time: f32, instance_params: array<vec4<f32>, 4>, record: SurfaceMaterialGpu }
struct VertexMaterialOutput { local_position: vec3<f32>, normal: vec3<f32> }
fn material_main(in: MaterialInput) -> MaterialOutput            // required
fn material_vertex(in: VertexMaterialInput) -> VertexMaterialOutput // optional (sets VERTEX_FN)
```
- World-space positions and normals; `in.normal` is flipped when back-facing and DOUBLE_SIDED; `raa_default_output(in)` is the `standard` built-in; the tint applies after `material_main`.
- User rules (`material.fn_invalid`, naga line:col): defines `material_main`; no `@group`, `@binding`, `@vertex`, `@fragment`, `@compute`, `var<storage`, `var<uniform`, `var<workgroup`, `enable`, `override` or `raa_` identifier; <= 65536 bytes; the Forward variant validates. Duplicates give `material.fn_duplicate`. LIB: `raa_hash3`, `raa_noise3`, `raa_fbm3`, `raa_voronoi3`, `raa_triplanar_weights`, `raa_perturb_normal`, `raa_luma`, `raa_srgb_to_linear`.

**Slots** (defaults priority 0 in raa-lit; highest wins; ties `shader.slot_conflict`): `Deform` `raa_slot_deform(p, n, x, t)`; `Shadow` `raa_slot_shadow(world_pos, n, frag)` (sun incl. contact and cloud); `Ao` `raa_slot_ao(frag)`; `Ambient` `raa_slot_ambient(s, c)` (applies `c.ao` once: default ambient x base x (1 - 0.5 metallic) x c.ao; IBL/BRDF multi-bounce AO on diffuse, specular occlusion); `Direct` `raa_slot_direct(s, c, l, radiance)`; `LocalVisibility` `raa_slot_local_visibility(index, position, normal, dir, distance)` (default 1); `Indirect` `raa_slot_indirect(s, c)` (default 0); `Transmission`; `Fog`. `ShadeCtx { material_flags, position, view_dir, frag, n_dot_v, ao, shadow, f0 }`.

**Final colour**: lit passes set `c.ao = min(s.ao, raa_slot_ao(frag))`; opaque = `emissive + raa_slot_direct(s, c, -sun_dir, raa_directional_radiance(sun_color) * shadow) + raa_slot_ambient(s, c) + raa_local_lighting(s, c) + raa_slot_indirect(s, c)`. AO darkens only ambient, once; direct and local light never (`tests/ao_once.rs`). Transparent adds transmission and fog; fur multiplies by `mix(0.35, 1.0, shell_height)`; flat or UNLIT = base + emissive.

**Local lighting** (`raa_local_lighting`, tile lights; the sun is separate): point zero at/beyond range or facing away, `window = clamp(1 - (d/range)^4, 0, 1)`, falloff `window^2 / max(d^2, 0.04)` (solstice `deferred.wgsl`); spot x `smoothstep(cosOuter, cosInner, ...)`, absent inner = 0.8 x outer; directional unattenuated. These call Direct with clamped LocalVisibility. Area (kind 3) = U x W x V: basis `t = normalize(cross(up0, d))` (up0 Y unless |d.Y| >= 0.99, then X), `b = cross(d, t)`, one-sided along +d; the Karis representative point gives the visibility direction, distance and window `raa_area_window`; U = `raa_area_ltc` (LTC atlas) or `raa_area_diffuse` with feature 39, radiance = intensity / (w x h); never Direct; V per S44. Loop providers avoid implicit derivatives. Fixtures: `crates/raa-lit/tests/fixtures/local-light-math.txt`.

**Pass kinds**: Prepass, Forward, Shadow, Transparent, FurShell; VgGbuffer, VgShade (S32).

## S12. Pipeline passes and targets (raa-pipeline)
- The RenderPass `raa.pipeline` runs enabled feature passes by `order`, overridable from JS; `Pipeline::entry_enabled(name)` reads an entry. Required passes cannot be disabled; `prepass` < `forward` < `tonemap`, tonemap last among HDR passes; `output_overlay()` passes follow it. Sky passes: S28.

| order | name | role | crate |
|---|---|---|---|
| 10 | sky.env | pass | raa-sky |
| 20 | shadow.csm | pass | raa-shadow |
| 21 | vg.cull | pass | raa-vg |
| 22 | vg.raster | pass | raa-vg |
| 23 | vg.resolve | pass | raa-vg |
| 25 | gi.volume | pass | raa-gi |
| 30 | prepass | pass, required | raa-lit |
| 32 | vg.gbuffer | pass | raa-vg |
| 35 | light.tiles | pass | raa-lit |
| 40 | ao.gtao | pass | raa-gtao |
| 42 | gi.screen | pass (compute) | raa-gi |
| 43 | gi.area-shadow | pass (compute) | raa-gi |
| 44 | gi.ssgi | pass (compute) | raa-gi |
| 45 | shadow.contact | pass | raa-contact |
| 50 | particles.sim | pass | raa-particles |
| 60 | forward | pass, required | raa-lit |
| 450 | vg.shade | contributor of forward | raa-vg |
| 500 | fur.shells | contributor of forward | raa-fur |
| 600 | particles.opaque | contributor of forward | raa-particles |
| 900 | sky.background | contributor of forward | raa-sky |
| 64 | gi.history | pass | raa-gi |
| 65 | ssr | pass | raa-ssr |
| 68 | fog | pass | raa-fog |
| 69 | sunshafts | pass | raa-shafts |
| 70 | opaque.copy | pass | raa-transmission |
| 80 | transparent.oit | pass | raa-oit |
| 85 | particles.blend | pass | raa-particles |
| 100 | taa | pass (compute), installed disabled | raa-taa |
| 110 | dof | pass | raa-dof |
| 120 | bloom | pass | raa-bloom |
| 125 | exposure.auto | pass, off by default | raa-exposure |
| 130 | grading | pass | raa-grading |
| 190 | cel.resolve | pass | raa-cel |
| 195 | sharpen | pass, off by default | raa-sharpen |
| 200 | tonemap | pass, required | raa-tonemap |

- `pipeline.required_order` refuses inverted prepass < light.tiles < {ao.gtao, forward} and the S28, S32 and S38 pairs; a missing pass drops only its own pairs.
- Targets: `depth` Depth32Float; `normal` Rgba16Float (view normal, roughness); `material` Rgba8Unorm (F0, metallic); `motion` Rgba16Float (location 2 of prepass and vg.gbuffer, cleared 0, lanes S43); `hdr` A/B chain (HDR_FORMAT, render, texture, storage, copy). Forward resolves into `hdr.dst()` and swaps; later passes read src, write dst, swap.
- Disabling calls `set_enabled(false, lighting)` (dummies back). With background mode 0 tonemap writes `background` where depth == 1, so `probeNonEmpty` keeps its v0 meaning.

**Tiling**: 16x16 px tiles of 128 words in binding 13, tile = floor(pixel/16); standard-Z reduction over prepass depth; spheres test sides and depth planes, area lights range + hypot(w, h)/2, directionals every tile (solstice `light_tiles*.wgsl`). 256 lights max (`lights.invalid`); each tile keeps the strongest 127 per **M2-TILE-SELECTION** (`crates/raa-lit/README.md`, `tests/fixtures/tile-selection.txt`), ties by handle; `droppedLightReferences` sums exclusions. Readback (`light_stats.rs`): three-sample async pool, `lightStats()` publishes the latest fully completed invocation, never a fence.

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
- Reads `sdfm.game-asset` v1 (format, version, game, id, name, category, world, mesher, document, collision, anchors, bounds, hash; canonical key order, document <= 2 MB, finite, id `^[a-z][a-z0-9]*(\.[a-z0-9-]+)+$`). RAA uses `document` (MAMA-meshed in workers), `mesher` (unchanged), `anchors`, `bounds`, `hash` (sha256 of MAMA `serializeDocument(document)` must match); `collision` passes through.
- Codes: `sdfm.format`, `sdfm.version`, `sdfm.schema`, `sdfm.not_finite`, `sdfm.id`, `sdfm.cap_document`, `sdfm.hash_mismatch`, `sdfm.mesher_refused`, `sdfm.material_unmappable`, `sdfm.worker`, `sdfm.disposed`, `sdfm.scene_format`, `sdfm.scene_resolve`.
- `createSdfmLoader(options)` (`mama` accessor resolving once to a URL for workers or a module) and `loadProceduralScene(engine, loader, scene, { resolveAsset?, resolveMesh?, parent?, mode?: 'mesh'|'trace', procgen?, placement?, mama? })` returning `{ root, instanceCount, sourceCount, setVisible, stats(), runtime(procgen), dispose() }`; runtime `{ step(tick), set(values), dispose() }`. MAMA imports live only in `src/mama-bridge.js`.
- `toMamaLight` maps S16 to S17 (position `t`, direction `R(r) * S(s) * [0,0,-1]`, metres unscaled, absent inner kept absent, linear to sRGB). `createMamaLightOwner` keeps one world-space id-to-handle registry per scene for static and runtime adapters; trace placement adds once.
- Lights are validated before allocation, staged unlit, committed after the scene applies; failures restore and release; engine refusals propagate. Complete snapshots reset removed optional fields; missing `delta.output` keeps the set, `[]` clears it. `setVisible` hides through intensity 0; disposal removes only owned lights.
- `render-options.js` `applyRenderOptions` keys include `renderScale`, `autoExposure` and `sharpening` (S47).

## S16. Procedural light interchange
Owners: mama-light-record and mama-light-transport; consumers raa-sdfm-pkg, sdfm-light-authoring.
- `mama:src/procgen/output-types.ts`: `ProcLightKind = 'point' | 'spot' | 'directional' | 'area'`; `ProcLight extends ProcGameplayEntry { id, node, key, t, r, s }` adds `kind`, `color` (linear), `intensity`, `range`, `shadows`, `angleDeg?`, `innerAngleDeg?`, `size?: [w, h]`. Rust keeps Json: `game_markers.rs` emits, `passes/gameplay.rs` flattens and converts colour to linear, `output.rs` holds `OutputSections.lights`; order kept, 1e-6 and 1e-9 grids, empty family emits nothing.
- `game_light` defaults: point, `#fff1d6`, intensity 2, range 6 m (> 0), 35 degrees, size `[1,1,0]`, shadows false. Angles are full; explicit `innerAngleDeg` in [1, 179] and <= `angleDeg`, refused without clamping; absence = 0.8 x outer, never materialized.
- `ProceduralScene` (`sdfm.procedural-scene` v1) has optional `lights?: ProcLight[]` (`null` invalid). `parseProceduralScene`/`serializeProceduralScene` validate and keep order, identity, placement, colour and inner-angle omission (key u32, intensity >= 0, range > 0, spot angle [1, 179], area two sizes; unknown keys refuse; `PSCENE_NOT_FINITE`). `planProceduralScene` copies lights into both variants. A successful change of the lights section makes the next `Runtime.step` `delta.output` complete; a failed evaluation is not an empty snapshot.

## S17. Local-light API and lifecycle
Owner: api-m28 (`packages/raa/types/m28-lights.d.ts`). `setLight` stays the sun API.
- `setLightCutoff(threshold)` (default 2/255 of display white, 0 disables): influence cutoff of local lights (`crates/raa-lit/src/light_cutoff.rs`). `T` = scene-linear radiance at which a neutral grey reaches `threshold` after the fixed exposure, mapper and look (`raa_tonemap::curves::visible_radiance`; auto-exposure ignored). With `L = max(r,g,b) * intensity` (times `max(1, w*h)` for area lights) a point/spot/area light gets `range = min(range, sqrt(2L/T))` (cut error at most `T`, window reaches 0 with zero slope, no pop); `25 L <= T/2` (or `L < T` for directional) switches it off. The registry applies it to every GPU record and world descriptor (tiles, forward, DDGI, GI epochs, hero/stochastic area shadow assignment skip off lights); `light_info.x` counts records up to the last live light, so all-off lights cost no tile pass and no probe-ray loop. The threshold is shared: each light uses `T/k`, `k` = 1 + authored-on lights whose first-pass spheres overlap its own (directional overlaps all, `k` capped at 32; beyond the cap the split is no longer strictly conservative), computed on the CPU once per registry change, so the *total* cut error stays below the threshold. Forward tier: `setLightCutoff(threshold, forwardThreshold = 16/255)`; a point or spot light without `shadows` is forward-shaded (and tiled) only within `r_f = sqrt(L/T2)` (width lane of its record, 0 = everywhere, faded over [0.75 r_f, r_f]); farther out it is probe-only: DDGI injects its shadowed analytic direct light (one shadow ray per probe and light, up to 16 lights per probe, weight complementary to the forward fade) into the probe irradiance texels. Per-tile forward budget: `setLightCutoff(threshold, forwardThreshold, forwardBudget = 8)` (0 = unlimited): in each tile the N strongest point/spot lights (tile score) stay forward; a weaker light fades with `smoothstep(t, 1.5 t, score)`, `t` = score of the (N+1)-th (continuous in the score, stored as 255*(1-w) in bits 16..23 of the tile word; N rides in `light_info.w >> 16`) and its probe share is the complement, evaluated at each probe with the same importance. The probe-only lights of a probe (at most 16) are compacted in light-index order, so the injection is deterministic. Authored values (`get`) are never changed; `lightStats().activeLights` stays the authored count.
- Local-light cube shadows (`setLocalShadows`, `crates/raa-sdf-trace/src/mesh/local_shadow.rs`, pass `shadow.local` order 36, kept enabled under every graph plan by the raa-graph executor extension `local_shadow`): a point/spot light with `shadows: true` gets a stable slot (`LocalLights::shadow_slots`, at most `MAX_LOCAL_SHADOWS` = 32, published with the light records through `LightingResources::set_local_shadow_slots`; the record's height lane carries `slot + 1`). Each slot owns 6 layers (+X, -X, +Y, -Y, +Z, -Z) of an R32Float array (group 1 binding 22, 0 = not traced = unoccluded) holding the distance from the light to the first occluder, traced through the shared mesh BVH (present only while a BVH-consuming GI such as DDGI is active; otherwise nothing is traced) with rays starting at the light's `radius` (its emitter: the light's own housing casts nothing, so a lantern still lights its diffuse-transmission glass) and ending at its effective range. Static caching: a face is re-traced only when its light's position/range/radius changes, the BVH changes inside its sphere (dirty bounds; a rebuild re-traces all) or the array is reallocated; at most `facesPerFrame` (default 2, 128x128 faces: about 1.4 ms of tracing per frame on an RTX 3090 in Intel Sponza, 3.7 M triangles, only while faces are pending) per frame, nearest the camera first, so loads and edits never produce a long frame. The forward (`raa_local_shadow`, local_lighting.wgsl) pushes the receiver 1.5 texels along its lit normal (-N for the diffuse-transmission back term) and takes 2x2 bilinear PCF of textureLoad compares; DDGI keeps its own one-ray light visibility, and shadow-casting lights are never probe-only tiered.
- `addLight(desc): LightHandle`, `updateLight(handle, changes)`, `removeLight`, `attachLight(handle, instanceId, localOffset)`, `detachLight`, `lightStats(): { activeLights, attachedLights, tileColumns, tileRows, tileCapacity: 127, droppedLightReferences | null, statsFrameIndex | null, pending }`. `LocalLightDesc { kind?, position?, direction?, color?, intensity?, range?, angleDeg?, innerAngleDeg?, size?, radius?, shadows? }`, `LocalLightChanges = Partial<LocalLightDesc>`.
- Defaults: point, [0,0,0], direction [0,0,-1], sRGB white, intensity 2, range 6, 35 degrees, no inner, size [1,1], radius 0, shadows false. Colour [0,1], intensity and radius >= 0, range and size > 0, direction non-zero (normalized). Radius and shadows are metadata: no local shadows.
- Updates merge and validate before mutation; omitted keeps, explicit `undefined` resets; colour converts only when supplied. Handles are never reused; the 257th light refuses with `lights.invalid`.
- Attachment uses the instance-group handle: position = `model * [offset, 1]`, direction by the linear part; an attached position edit refuses with `light.attached_position`; detach and parent removal keep the last world placement.
- Codes: `light.invalid_desc`, `light.invalid_attachment`, `light.attached_position`, `api.invalid_argument`, `api.unknown_handle`, `lights.invalid`, `api.disposed`. Stats follow S12.

## S18. Document-light compatibility and inspector
Owner: sdfm-light-authoring.
- `sdfm:src/core/scene-lights.ts` keeps kind `pointLight` with `params.kind` point (default), spot, directional or area. `documentGameLights(doc)` lists every light in traversal order with world placement, sRGB colour, radius and visibility. The legacy point-light packers stay limited to eight visible point lights.
- Document defaults: intensity 8, range 4 m, radius 0.08 m, spot 35 degrees, area [1,1]; the add command uses `#ffad55` and one undo step. Inner-angle absence and hidden kind values survive save, reload and kind switches.
- `PointLightSection.vue` edits kind, colour, intensity, range, radius, contact shadows, angles (with reset) and area size through `setDocumentLightParams(doc, id, changes): Op[]`, validating the whole state; a refusal shows in the inspector and changes nothing. It states that Game has no local shadows. Document colours reach S17 as sRGB; light-only edits do not invalidate geometry. `src/node/schema/{docs,json-schema}.ts` generate schema and catalog.

## S19. GameDisplay ownership, frames and lifecycle
Owner: sdfm-game-light-sync.
- `sdfm:src/app/paramesh/game-light-sync.ts` exports `GameLightAttachment { instance, instanceWorld, ownerDocumentWorld, ownerWorld }`, `ResolveGameLightAttachment(light) -> GameLightAttachment | null`, `PlannedGameLight { id, visible, desc: LocalLightDesc, attachment: { instance, offset } | null }`, `GameLightEngine` (the five S17 light methods of Engine), `DocumentLightHandle { id, handle, instance | null, visible }`, `GameLightSync { sync(lights, resolve), beforeInstanceRemoval(instance), snapshot(), dispose() }`, `planGameLights(lights, resolve)` and `createGameLightSync(engine)`.
- `GameDisplay` resolves the nearest mapped ancestor through `Entry.members`, `Draw.pick.world` and the public `Draw.instance` group handles, excluding retiring draws, in the decomposed TRS frame sent to the engine. With C = ownerWorld x inverse(ownerDocumentWorld): world placement = C x document placement; attachment offset = inverse(instanceWorld) x corrected position; directions use the matching normalized linear transforms. Range and size stay metres. Before a mapping exists the composed world placement is used.
- The reconciler validates the whole plan before any engine call, keeps one stable handle per surviving document id, suppresses unchanged calls, resets omitted kind-specific fields, hides through intensity 0 and stages new handles at 0. A failure rolls back and keeps the published observations and the originating error; a failed rollback keeps an AggregateError and stops normal synchronization. Disposal removes only owned document handles, never adapter lights.
- `game-display.ts` exports `GameDisplayLightState = { lights: DocumentLightHandle[]; engine: LightStats | null; ready: boolean; error: string | null }` through `documentLightState()`: reconciliation readiness, not GPU submission or geometry readiness. Light-only changes do not invalidate geometry. Types come from the `@etcito/raa` root, never deep imports.
- The sky observation `skyWeatherState()` (S25) is separate and leaves this hand-off unchanged.

## S20. M4 browser facts and the single capture inventory
Owner: sdfm-light-evidence (`sdfm:src/core/scene-lights-evidence.ts`). `M4_BROWSER_FACTS = { format: 'sdfm.scene-lights.browser-facts', version: 1 }`: `LightSnapshot { document, selection, projection, game }`, `job`, `provenance`, `captures` and the observations of jobs point, spot, kinds, ownership, many, invalid; validated for canonical text, projection, unique ids, provenance and exact coverage (facts, never pixels). `SCENE_LIGHT_CAPTURES` (16 identities) is the only selection source; `assertLightInspectorReadable` runs right before capture.

## S21. Node-side M4_EVIDENCE_RECORDS lifecycle
Owner: sdfm-light-evidence (`topics/scene-lights.ts`), records under ignored `out/solstice-m4/`, `M4_EVIDENCE_RECORDS = { schema: 1, milestone: 'm4' }`. Runtime receipts (mama, then raa), `select`, `clean` (isolated UUID attempt, `current.json` pointer, never rerun), `fresh` (rejects stale, partial or failed jobs and images) and `judge` (minimum 6, all verdicts kept) follow the S26 pattern; rendering uses `node scripts/gpu-evidence.mjs scene-lights`.

## S22. Execution, verified recovery and host completion
- Executable authority is the current `.nmc/commands.json` of each repository and the configured linked targets.
- A visual milestone closes after its branch checks, its current managed render, fresh and all-image judge chain (S21, S26) and the host's archive of every passed image and fact, with the delivery receipt of the informational email, which is not an approval step.
- Technical failures keep their originating error and enter the configured Doctor/Operator recovery; they are never waived. A delivery-only failure retries delivery without rendering again.

## S23. Solstice adaptation and atmosphere/celestial data
Source paths refer to the solstice revision of Conventions; raa-sky implements this section.

**Atmosphere** (`shaders/common/atmosphere.wgsl`, `shaders/sky/luts.wgsl`, `src/sky.rs`)
- `atmospheric_radius_km(height_m) = 6360 + max(height_m, 1) * 0.001` is the only metres-to-kilometres conversion.
- Ground radius 6360 km, top 6460 km. Rayleigh (5.802e-3, 13.558e-3, 33.1e-3) per km, scale height 8 km. Mie scattering 14e-3, absorption 4.40e-3, scale height 1.6 km, g 0.8. Ozone (0.650e-3, 1.881e-3, 0.085e-3), triangular profile centred at 25 km, half-width 15 km. Rayleigh and Cornette-Shanks phases.
- Rgba16Float LUTs: transmittance 256x64 (40 steps), multiple scattering 32x32 (8x8 directions, 20 steps), per-view sky-view 256x144 (32 quadratically spaced steps). Ground albedo is authored (source 0.3). Static LUTs rebuild only on atmosphere parameter changes; the sky-view follows each view's height and the accepted light.

**Time and bodies** (`src/time_of_day.rs`)
- Lunar month 29.53059 days, new-moon anchor day 6, obliquity 23.44 degrees, dominant-light step 0.05 degrees, moonlight 1/2500 of daylight, night glow 1.2e-4, moon radiance 0.03 by day and 0.0024 at night, starlight 0.002.
- Solar declination = -obliquity * cos(2pi * (day + 10) / 365); hour angle = 2pi * (hours - 12) / 24. Moon age = fract((day + hours / 24 - 6) / 29.53059); lit fraction = 0.5 - 0.5 * cos(2pi * age); the moon's hour angle subtracts 2pi * age and its declination is the solar one times cos(2pi * age).
- Sun twilight fade over altitudes [-10, -5] degrees, night fade [-14, -6], moon rise [-4, 6], hand-off to the moon below -10 degrees. Body directions point toward the body; a directional light stores the negated travel direction. The undimmed daylight baseline is stored apart from the dominant light.

**Sky bodies** (`shaders/sky/bodies.wgsl`, `milky_way.wgsl`)
- Atmosphere-filtered sun and moon discs with limb darkening, procedural moon surface, phase and earthshine, temperature-coloured star layers occluded by the moon. Moon angular radius 0.0075; `sunDiscSize` and `sunDiscIntensity` drive the sun disc.
- The Milky Way (square-root encoding over range 4) is baked once into an Rgba8Unorm 2048x1024 map. Moon, stars and Milky Way are part of `raa_sky_radiance`; there is no celestial fullscreen pass.
- Not adopted: the combined solstice environment bind group, GI resources, `volume.wgsl`, the solstice aerial perspective.

## S24. Public sky, time and weather API (m08)
Declared in `packages/raa/types/m08-sky-ibl.d.ts`, documented in `docs/api/m08-sky-ibl.md`. Untyped members are numbers.

```ts
type SkyModel = 'physical' | 'legacy'; type SkyPreset = 'day' | 'snow' | 'jungle' | 'night'; type SkyEnvUpdate = 'onChange' | 'everyFrame'
type WeatherPreset = 'clear' | 'fair' | 'broken' | 'overcast' | 'storm' | 'cirrus'
interface CloudsDesc { coverage?; density?; altitude?; scale?; wind?: [number, number]; color?: Vec3; silverLining? }
interface SkyDesc { preset?; model?; zenith?: Vec3; horizon?: Vec3; ground?: Vec3; intensity?; horizonSoftness?; sunDiscSize?; sunDiscIntensity?; clouds?: CloudsDesc; envUpdate?; atmosphere?: { groundAlbedo?: Vec3; mieMultiplier? } }
interface TimeOfDaySettings { day; hours; latitude; north; enabled: boolean; playing: boolean; speed }
interface TimeOfDayState extends TimeOfDaySettings { currentDay; currentHours; sunDirection: Vec3; moonDirection: Vec3; moonLit; starPole: Vec3; starEquinox: Vec3; dominant: 'manual' | 'sun' | 'moon' }
interface WeatherState { coverage; type; base; thickness; density; cirrus; wind: { speed; from }; evolution; rain }
interface WeatherSettings { enabled: boolean; shadows: boolean; changing: boolean; seed; transitionSeconds; target: WeatherState }
interface ResolvedWeather extends WeatherSettings { current: WeatherState; cyclePreset: WeatherPreset | null; text: string }
```
- Engine: `setSky(SkyDesc)`, `sky(): ResolvedSkyDesc` (every field resolved, `profile` instead of `preset`), `setEnvironmentLighting({ diffuse?, specular? })`, `setTimeOfDay(Partial<TimeOfDaySettings>)`, `timeOfDay()`, `setWeather(WeatherDesc)` (`preset?`, `text?`, the settings flags and every WeatherState field, all optional), `weather()`, `skyWeatherDiagnostics()` (the S25 diagnostics in camelCase, null when absent). Package root: `parseWeatherText(text): WeatherState`, `formatWeatherText(weather): string`.
- Validation: finite numbers that stay finite as f32. Time, weather, model and atmosphere fields reject wrong types, null, unknown keys and out-of-domain values with `sky.invalid` and the field path. Undefined is omission; omitted fields merge over the authored settings; a preset applies before the overrides; nothing changes before validation completes. Getters return fresh objects. Earlier `setSky` fields keep their null compatibility and ranges.

**Time.** `day` integer 1..365, `hours` [0, 24), `latitude` [-90, 90], `north` radians, `speed` >= 0 simulated seconds per second; defaults 172, 9, 38, 0, disabled, paused, 1. `enabled` gives the dominant light to sun and moon; `playing` advances the clock only while enabled, once per Engine invocation by the real delta. While enabled, `setLight` updates the stored daylight baseline and the astronomical direction wins; disabling restores the manual light. Editing `day` or `hours` resets the runtime cursor (`currentDay`, `currentHours`); pause and resume do not.

**Weather.** Defaults: enabled, shadows, not changing, seed 1 (u32), transitionSeconds 0, target fair. Domains: coverage, type, cirrus, rain in [0, 1]; base >= 0 m; thickness >= 100 m; density, wind.speed (m/s), evolution, transitionSeconds >= 0; wind.from finite degrees.

| preset | coverage | type | base | thickness | density | cirrus | wind speed / from | evolution | rain |
|---|---|---|---|---|---|---|---|---|---|
| fair | 0.3 | 0.9 | 1200 | 1600 | 1 | 0.15 | 7 / 250 | 1 | 0 |
| clear | 0 | | | | | 0.05 | 5 / 250 | | |
| broken | 0.62 | 0.95 | 1000 | 2600 | 1.15 | 0.3 | 10 / 240 | 1.3 | |
| overcast | 0.97 | 0.12 | 600 | 900 | 1.3 | 0 | 8 / 220 | 0.6 | |
| storm | 0.85 | 1 | 800 | 7000 | 2 | 0.6 | 16 / 230 | 2 | 1 |
| cirrus | 0.04 | 0.9 | | | | 0.75 | 20 / 260 | | |

An empty cell is the fair value. `preset` with `text` rejects; either is the base under the structured overrides.
- Changing weather: a seeded cycle dwells 120-300 s, then moves over 90 s along clear -> {fair, cirrus}, cirrus -> {fair, broken}, fair -> {broken, clear}, broken -> {overcast, fair}, overcast -> {storm, broken}, storm -> {broken, overcast}. An explicit preset names the cycle start (text starts at fair). It restarts on an explicit preset or text, a changed seed, or changing turning on. A manual target edit stops it unless the same patch sets `changing: true`.
- `current` eases to the target by `1 - exp(-dt / transitionSeconds)` with shortest-turn wind. Cloud animation uses the invocation delta capped at 0.25 s; wind drift wraps at 252000 m, evolution at 7000 m (4 m/s at multiplier 1); wind is rotated by `north`.
- Weather text: starts from fair, strips `#` comments, takes the last repeated value, accepts `wind` with omitted components, rejects unknown keys and malformed numbers with a one-based line message, and clamps to the domains where the structured API rejects. Output is the authored target, one setting per line (coverage, type, base, thickness, density, cirrus, wind, evolution, rain), final LF.

**Model and presets.** `physical` is the atmosphere with volumetric clouds; `legacy` the gradient and planar layer, which approximates the authored weather. Physical rendering reads artistic colours and intensity relative to the profile's original values. `groundAlbedo` in [0, 1], `mieMultiplier` > 0.
- An explicit physical `preset` sets sky, weather, day 172 and hour before the overrides and keeps place, enabled, playing and speed: day = fair, 9 h, albedo 0.3, mie 1; snow = overcast, 12 h, albedo 0.9, mie 1.5; jungle = broken, 15 h, albedo (0.18, 0.26, 0.14), mie 1.2; night = clear, 0 h, albedo 0.3, mie 1. A patch without `preset` resets nothing; a model-only switch keeps the weather.
- In physical mode cloud coverage, density, altitude and wind write the weather target; scale, color and silverLining stay artistic.

## S25. Sky backend, clouds, fog and editor integration
**Feature** (raa-sky)
- `SkyFeature::install(gpu, pipeline: &Rc<RefCell<Pipeline>>, lit: &mut LitRenderer, ibl: IblHandle) -> Result<SkyHandle, RaaError>` installs the S28 passes; one feature evaluates both models behind `raa_sky_radiance`.
- `SkyHandle`: `set(SkyDesc)`, `set_transaction(SkyTransaction)`, `get`, `mark_dirty`, `set_time_of_day(TimeOfDaySettings)`, `time_of_day`, `set_manual_light(DirectionalLight)`, `advance_frame(invocation: u64, dt_seconds: f64) -> Result<DirectionalLight, RaaError>`, `set_weather(WeatherSettings)`, `set_weather_update(WeatherUpdate)`, `weather`, `set_view(view, camera, history_valid)`, `observe_submission(invocation, submitted_views: u8, complete)`, `remove_view`, `diagnostics`. Setters validate the whole candidate before mutation; a repeated invocation id returns the same snapshot; a steady frame does not allocate.
- `SkyTransaction { desc, preset: Option<SkyPreset>, cloud_fields: LegacyCloudFields }` (COVERAGE 1, DENSITY 2, ALTITUDE 4, WIND 8) and `WeatherUpdate { settings, cycle_start: Option<WeatherPreset>, restart_cycle }` carry explicit intent; it is never inferred by comparing descriptors.

**Engine wiring** (raa-js `engine.rs`, `state.rs`)
- `EngineState.sky_invocation_id` starts at 0; `begin_scene_invocation` increments it once per `render` or `probeNonEmpty` (checked, `sky.invalid` on exhaustion), independently of frame and submission counters.
- `EngineApp::update` calls `advance_sky_frame(input)` before `api::frame_all`: `advance_frame(id, dt_ms / 1000)` runs once and its light becomes `scene.light`. A validated `setLight` reaches `set_manual_light` first. Each view's camera reaches `set_view` before the lit view preparation; `finish_scene_invocation` forwards the acknowledged view mask to `observe_submission`.

**Readiness**
- New or reallocated atmosphere resources stay pending until an acknowledged submission; the Milky Way is ready only after a complete submitted invocation that carried its bake.
- `SkyDiagnostics { invocation, submitted_invocation, weather_generation, atmosphere_generation, environment: IblPublication, views: [Option<SkyViewDiagnostics>; 4], readiness }`; a view reports size, invocations, sky-view generation, history validity, reset reasons (bits Resize 1, CameraDiscontinuity 2, Weather 4, Model 8, Enable 16, ViewRemoved 32) and both cloud validity flags. Readiness is pending, ready or failed at stage install, atmosphere, milkyWay, cloudLighting, cloudScreen, environment or submission; a failure keeps its message. Generation 0 means no accepted work.
- raa-ibl: `IblPublication { requested_generation, published_generation: Option<u64>, pending }`, `IblHandle::publication()`, `publication_error()` (`ibl.readback`), `sh()`, and `invalidate(lighting)`, which clears the environment and every earlier request so a late callback cannot republish. `sky.env` invalidates on disable, model or source change and rejected source.
- `envUpdate`: `onChange` republishes the IBL on accepted sky, time, weather or light changes, not on cloud drift; `everyFrame` republishes continuously. Views share time and weather and own their LUTs and histories.

**Cloud seam** (raa-cloud-seam)
- `CloudFrame { invocation, view, size, camera, previous_view_proj, history_valid, weather, sky, dominant, animation_seconds, wind_offset, wind_velocity, evolution, weather_enabled, cloud_shadows, physical }`; wind axes x = east, z = south, rotated by `Place.north`.
- `CloudViews { panorama, screen, transmittance, screen_valid, panorama_valid }`: Rgba16Float, RGB cloud radiance and A transmittance, composed as `clear_sky * A + RGB`; screen alpha -1 marks an un-marched texel, which falls back to the panorama or the neutral (0, 0, 0, 1). `CloudInvalidation`: Resize, CameraDiscontinuity, Weather, Model, Enable, ViewRemoved.
- Private trait `CloudRenderer`: `prepare`, `record_lighting`, `record_screen`, `views(view)`, `invalidate(view: Option<u32>, reason)`, `submitted(view, invocation)`. `build.rs` always registers `neutral` and registers `m2` (`clouds::create(gpu)`) when `src/clouds/mod.rs` exists. No backend owns the clock, the weather state or the lights.

**Volumetric clouds** (raa-volumetric-weather; solstice `src/clouds/mod.rs`, `shaders/clouds/*`)
- Rgba8Unorm noise: shape 128^3, detail 32^3, weather map 512^2; 256-bin coverage CDF; panorama 192x108; half-resolution ping-pong screen targets; Rgba16Float ground shadow 256^2 over 12000 m. Samples: screen 56 primary and 5 light, panorama 32 and 4, shadow 16.
- Coverage-rank remapping, stratus to cumulus height profiles, Perlin-Worley erosion, curved-layer intersection, two HG lobes, four multiple-scattering octaves, Beer-powder, rain darkening, cirrus.
- The screen march blends 0.12 of the new sample into cloud-only history reprojected with the previous view-projection; every `CloudInvalidation` resets it. Ground shadow: minimum transmittance 0.04, rain multiplier `1 - 0.3 * rain`; disabled clouds or shadows publish 1.

**Fog.** raa-fog is the only scene aerial perspective; its height integral and Fog slot sample the published environment and call the S28 directional boundary once.

**SDFM.** `RenderSettings` (key `sdfm.render-settings.v1`) adds `gameSky: Pick<Required<SkyDesc>, 'model' | 'preset'>`, `gameTime: TimeOfDaySettings`, `gameWeather: WeatherSettings`, validated strictly and atomically; the authored candidate is saved, never runtime playback. `GameDisplay.skyWeatherState(): { ready, error, sky, time, weather, submittedFrames }` reports the actual Engine and its submitted frames.

## S26. Sky-weather page, inventory and evidence
**Page** (`examples/sky-weather/`, built to `dist/sky-weather/` by `bash scripts/build-sky-weather.sh`)
- The build runs `scripts/build-bricks.sh`, stages the page and scene modules, writes the sorted `scenes.json` `[{ id, milestone, file }]` and, last, the ready `build.json`. Stages `foundation`, `atmosphere`, `weather` are cumulative; a scene's `milestone` is its stage.
- A scene keeps the S14 default export and adds `inventory`, `factNames` and `observe(engine, variant)`. A variant adds `selection { camera, subjects [{ name, min, max }] }` (metres) and optional `measurement { pair, state, preset, features }` and `controls`.
- Inventory entry `{ id, criterionName, scenario, mode, filename, expectedVisibleBehavior, requiredFacts, pair }` is the only source of selection, filenames and judge descriptions.
- `requiredFacts` selectors: `snapshot:sky|time|weather|diagnostics`, `runtime:passes|viewSizes|ready`, `assertion:<name>` (declared in `factNames`, exactly one, `passed === true`). `validateInventory(scenes, stage)` also checks that the subjects overlap the camera frustum; `validateRequiredFacts(row, facts, factNames)` throws on missing, null, duplicate or false evidence.
- `window.nmcGpu.ops.run({ selectionId, attemptId, buildId, scenario, mode, width, height, remainingMs })` returns `{ provenance, browser, adapter, hardware, images [{ id, filename, mime, data, observation }], facts, measurements, errors { console, page, gpu } }`. `remainingMs` (at most 300000) becomes one local deadline for setup, waits, measurements and captures.
- `facts` = the scene's `{ sky, time, weather, diagnostics, assertions [{ name, passed, detail }], edits [{ operation, accepted, error, before, after }] }` plus the runtime's `passes`, `viewSizes` and `readiness { status, stage, error }`. Unavailable values are null; nothing is inferred.
- `FeatureMeasurement { pair, state, preset, features, width, height, warmup, frames, frameTimesMs, fps, p50Ms, p95Ms, p99Ms }` (PAIR_RUN): fps = 1000 / mean, nearest-rank percentiles `ceil(p * N) - 1`, a missing frame expires after min(10 s, remaining deadline). No pass threshold.

**Records** (`node web/common/sky-weather-job.mjs <check|select|clean|render|fresh|judge> --page dist/sky-weather --out out/sky-weather/<stage> --stage <stage>`)
- `build.json`: `{ buildId, preparationId, inputs, builtAt, page, files, status, error }`; `inputs` holds repository, revision, dependencies, source texts, toolchain, commands and criteria text.
- Stage root: `selection.json` `{ selectionId, stage, buildId, preparationId, inputs, inventory }` and `current.json` `{ selectionId, attemptId, directory }`, a pointer and never a pass. `clean` allocates `attempts/<uuid>/` with frozen selection and build copies and deletes only image files of attempts of another build. An attempt that left `allocated` is never reused.
- `run.json`: `{ selectionId, attemptId, buildId, preparationId, pageRef, jobs, images, kept, status, errors }`. `render` publishes at most once and runs one managed operation per (scenario, mode) group; a group of an earlier attempt of the same build (`buildKey` = receipt inputs without preparationId) is kept when its job passes every freshness check and its images are intact.
- `fresh.json`: current inputs, identity linkage, exact image coverage, Windows Chrome physical-GPU provenance, readiness, requiredFacts, measurements and empty console, page and GPU errors, over kept and new images.
- `judge.json`: one verdict `{ id, filename, description, minScore: 6, exit, score, output, judgedAt }` per selected image, from the inventory behaviour without a reference; a kept byte-identical image with the same description reuses its passed verdict.

**Commands**
- `raa:.nmc/commands.json`: `p2-sky-build` (1800 s), then per stage `p2-sky-<stage>-check`, `-select`, `-clean`, `-render` (900 s), `-fresh`, `-judge` (600 s), chained in that order with judge after render and fresh. The weather judge uses `--budget-s 1020` (timeout 1200 s); `p2-sky-weather-judge-local` runs the job, contract and weather `node --test` files.
- `sdfm:.nmc/commands.json`: `p2-editor-mama-build`, `-raa-build`, `-node-build`, `-build`, `-check`, `-select`, `-clean`, `-render`, `-fresh`, `-judge`; records under `out/sky-weather-editor/`.

**Coverage.** Atmosphere: clear day, twilight, night, a changed moment, the four presets, physical versus legacy, fog, IBL and flat-background pairs, invalid input, multi-view facts, and resize, camera-cut, model, weather and re-enable transitions. Weather: the six presets, drift and evolution, ground shadow and weather pairs, panorama and IBL publication, history resets. Editor: the Render settings panel beside the Game view, changed time and weather across reload, `.weather` import/export, pause/play, an invalid input with unchanged accepted state.

## S27. Pinning tests of the sky and weather contracts
- Sky: `crates/raa-sky/tests/{desc,variants,time_of_day,weather,lifecycle,backend_registry}.rs` and the inline tests of `desc.rs`, `state.rs`, `feature.rs`, `background.rs`, `env_pass.rs`; `variants.rs` pins `raa_sky_radiance`, the legacy and physical compositions and the cube orientation. Cloud backend: `crates/raa-sky/src/clouds/tests/`.
- Shared layouts: `crates/raa-pipeline/src/lighting.rs` (group 1 bindings 0..13, 15, 16 and dummies), `tests/resources.rs`; `crates/raa-shader/src/lib.rs`, `src/gpu_constants.rs`, `tests/{compose,variants,user_fn}.rs`.
- Composed budgets and directional consumers: `tests/variants.rs` of raa-lit, raa-oit, raa-fur, raa-shadow, raa-fog, raa-particles and raa-shafts.
- Engine and package: `crates/raa-js/tests/{api_registry,frame_lifecycle,views,sky_weather}.rs`, `packages/raa/test/sky-weather.test.mjs`.
- SDFM: `src/core/sky-weather-settings.test.ts`, `src/app/paramesh/game-sky-weather.test.ts`, `src/app/gpu-page/sky-weather-controls.test.ts`, `src/node/evidence/topics/sky-weather.test.ts`.

## S28. Sky and weather GPU layouts, passes and directional boundary
Completes S10-S12. FrameGpu, EnvGpu, instance records and feature ids are unchanged; ids 25 and 26 stay reserved.

**Group 1** (FRAGMENT-only additions; binding 14 unallocated; no new storage buffer)
- 15 `raa_cloud_transmittance: texture_2d_array<f32>`, Rgba16Float: layer 0 the 256x256 ground map, layer 1 texel (0, 0) its mean; disabled = 1x1 two-layer ones; sampled with binding 6.
- 16 `raa_weather_lighting: WeatherLightingGpu` (64 B, UNIFORM | COPY_DST): `map` @0 (origin x, origin z, extent m, cloud enabled), `light` @16 (direction toward the dominant light, 0), `atmosphericTransmission` @32 (rgb, 1), `flags: vec4<u32>` @48 (physical enabled). Defaults (0, 0, 1, 0), (0, 1, 0, 0), ones, zeros. The CPU writes map, light and flags; the atmosphere compute result (16 B) is copied to offset 32 after dispatch and is never overwritten by a stale upload.
- `LightingResources`: `set_cloud_transmittance(view, map, light) -> Result<(), RaaError>`, `clear_cloud_transmittance()`, `set_physical_atmosphere(enabled)`, `atmospheric_transmission_buffer()`.

**Directional boundary** (raa-shader)
- `WEATHER_LIGHTING_WGSL` declares the record, both bindings and `raa_directional_radiance(raw: vec3<f32>) -> vec3<f32>` (`raw`, times the transmission when physical is enabled); `PRELUDE_WGSL` includes it once. The Forward, Transparent and FurShell passes call Direct with `raa_directional_radiance(sun_color) * shadow`; lit particles, the fog slot's inscatter and the sun shafts use the same boundary. Local lights do not; atmosphere and cloud evaluation receive raw daylight.
- `CLOUD_SHADOW_WGSL` defines `raa_cloud_shadow(world_pos) -> f32`: 1 when `map.w < 0.5`, otherwise the map sampled where the line to the dominant light meets the ground, fading to the mean across the edge band 0.75..1. The default Shadow slot returns it; the `csm` provider returns `vis * contact * raa_cloud_shadow`, or `contact * raa_cloud_shadow` when CSM is cleared.

**Passes** (raa-sky): `sky.atmosphere` 5 (Milky Way bake, LUTs, directional transmission), `sky.cloud-lighting` 8 (panorama, ground transmittance), `sky.env` 10, `sky.cloud-screen` 55 (screen march), contributor `sky.background` 900. Required order: atmosphere < cloud-lighting < sky.env and prepass < cloud-screen < forward. Disabling `sky.cloud-lighting` invalidates the backend and publishes neutral transmittance; disabling `sky.env` clears the IBL and cancels pending publication; the flat background still writes `EnvGpu.background`.

**Sky group 2** (env and background): 0 `PhysicalSkyGpu` (256 B), 1 transmittance LUT, 2 sky-view, 3 cloud panorama, 4 screen clouds, 5 Milky Way, 6 clamp sampler, 7 repeat-U sampler; unused textures are dummies. Compute modules use group 2 with empty groups 0 and 1.

**PhysicalSkyGpu** (16 B lanes): `viewer` @0 (world xyz m, radius km); `sun` @16 (direction, moon lit); `moon` @32 (direction, radiance); `pole` @48 (xyz, star intensity); `equinox` @64 (xyz, night glow); `dominant` @80 (direction, raw intensity); `daylight` @96 (rgb, baseline intensity); `discs` @112 (sun radius, sun multiplier, moon radius, pixel angle); `ground_albedo` @128 (rgb, mie multiplier); `rayleigh` @144 (coefficients, scale height); `mie` @160 (scattering, absorption, scale height, g); `ozone` @176 (coefficients, rain); `planet` @192 (ground radius, top radius, animation seconds, 0); `flags: vec4<u32>` @208 (physical, cloud screen valid, cloud panorama valid, 0); zeros @224..256.

## S29. Cluster DAG data (raa-cluster)
CPU, deterministic for any step budget. Weld onto a 16-bit grid, then per level partition, group (target 16, max 24) and simplify to `target_ratio` with locked group boundaries (foliage is thinned); then a fan-out-8 node tree.
- `BuildOptions`: triangles 128 (8..128), vertices 255 (16..255), `min_reduction` 0.85, `target_ratio` 0.5.
- `PackedVertex` 16 B (grid position, octahedral normal, uv, 0). `PackedCluster` 48 B: `center` @0, `radius` @12, `lod_center` @16, `lod_radius` @28, `lod_error` @32 (m), `vertex_offset` @36, `triangle_offset` @40, `counts` @44 (vertices 0-7, triangles 8-15, level 16-23). `PackedNode` 48 B: same bounds, `max_parent_error` @32, `first` @36, `count` @40, `is_leaf` @44 (section-relative). Triangle = three 8-bit local indices.
- One step unit visits <= 4096 triangles or 1024 clusters. Codes `cluster.invalid_mesh`, `invalid_options`, `empty`, `too_large`, `not_finished`, `invalid_data`. `ClusterStore` states None, Baking, Ready (Attached or Runtime), Refused.

## S30. Saved-mesh cluster section
Optional key `data.groups[g].parts[p].meshes[m].clusters` of a MAMA saved mesh; encoding and `MESH_COMPATIBILITY` unchanged. Shape (`ClusterSection`, MAMA `MeshClusters`): `{ version: 1, algorithm: 1, vertices: Uint16Array, triangles: Uint32Array, clusters: Uint8Array, nodes: Uint8Array, rootNode, hierarchyDepth (1..64), levels (1..256), quantOrigin, quantStep (> 0), uvOrigin, uvStep (>= 0), boundsCenter, boundsRadius (>= 0), sourceTriangles }`. Wire rules (`mama:docs/mesh-clusters.md`): exact keys, array lengths multiples of 8 or 48, integer and finite fields, cluster counts and ranges, node ranges. Full validation is `validateClusters`. Refusals: MAMA `mesh.clusters_invalid`, raa-sdfm `sdfm.scene_resolve`, RAA `vg.invalid`, each naming the field.

## S31. Virtualized-geometry API (m29)
`packages/raa/types/m29-{cluster-bake,virtual-geometry}.d.ts`, `docs/api/m29-*.md`.
- Module (no Engine): `bakeClusters(input)`, `createClusterBake(input)` (`step(budgetMs = 4)`, `progress`, `result`, `free`), `validateClusters(section)`, `clusterCutMesh(section, errorMetres)`.
- Engine meshes: `attachMeshClusters(mesh, sections)` (atomic, one per sub-mesh), `bakeMeshClusters(mesh, options?)`, `clearMeshClusters`, `meshClusters(mesh)` (`state` none/baking/ready/refused, `progress`, `source`, `code`, `message`, counts), `setClusterBakeBudget(ms)`, `clusterBakeBudget()`. Codes `vg.mesh_vertex_colors`, `vg.mesh_empty`, `vg.mesh_too_large`, `vg.invalid`.
- `setVirtualGeometry(patch)` / `virtualGeometry()`: `enabled` false (needs msaa 1), `errorPixels` 1, `softwareRaster`, `occlusion`, `shadows`, `runtimeBake`, `instanceDefault` true, `maxVisibleClusters` 262144 ([1024, 4194304]), `debugView` none/clusters/lod/raster/occlusion. Bad values: `vg.invalid` `<key>: <reason>`.
- `setInstanceVirtualGeometry(instance, boolean | null)`, `instanceVirtualGeometry(instance)` (`requested`, `path`, `pending`, `code`, `message`); codes `vg.skinned_unsupported`, `vg.deformed_unsupported`, `vg.transparent_material`.
- `virtualGeometryStats()`: `active`, `inactiveReason` (disabled, passDisabled, multiView, noInstance), instance and mesh counts, `refused`, drawn, hardware, software, shadow and dropped clusters, `drawnTriangles`, `statsFrameIndex`, `pending` (published after complete submission).

## S32. Virtualized-geometry GPU layouts and passes (raa-vg)
Adapted from solstice `src/vg/*` and `shaders/vg/*`, assembled by Rust builders. Groups 0 and 1 are unchanged; feature ids 27, 28, 36 are set while active.
- Depth helper `raa_shader::DEPTH_WGSL`: `raa_depth_nearer/nearest/farthest/is_far`, `raa_depth_key(d)` = `bitcast<u32>(1 - d)` (0 empty), `raa_depth_from_key`; every adapted comparison uses it; cascade key = `key >> 8`.
- `PassView` 432 B in 512 B slots (main 0-1, cascades 2-5, GI 6-10 reserved): view-projections, planes, LOD scale and threshold (px), phase, occlusion, `allow_software`, eye data, receiver planes. Software raster below 40 px (16 texels in cascades).
- Scene pool: storage `instances` (`VgInstanceGpu` 192 B: model, previous rows, quantization, `ranges` = section bases, `ids` = slot, material, flags, hierarchy word, bounds), `nodes`, `clusters`, `vertices`, `triangles`. Cull entries 8 B (instance, cluster word: index 0-23, software 30, second phase 31); args 56 B (dispatch @0, software @12, draw @24, merge @40). Visibility `vis_depth` and `vis_id` per pixel; stamped cascade depth; R32Float occlusion pyramid.
- Resolve outputs (`VgSurface`): `ids` Rgba32Uint, `normal` Rgba16Float, `uv_motion` Rgba32Float, `albedo` Rgba8Unorm. Bindings live in pass-private group 2 (cull also group 3).
- Required order: vg.cull and shadow.csm < vg.raster < vg.resolve; vg.resolve and prepass < vg.gbuffer < light.tiles, ao.gtao, forward. `vg.cull` runs phase First for the main view and cascades; `vg.raster` rasterises hardware and software lists, builds the pyramid, runs phase Second and draws cascades into the CSM layers; `vg.resolve` fills the outputs; `vg.gbuffer` (`PassKind::VgGbuffer`) writes S12 targets and depth per material function; `vg.shade` (`PassKind::VgShade`) writes the S11 opaque colour and depth. Material functions run unchanged on the rebuilt `MaterialInput`.
- raa-lit's `VirtualSet` removes virtualized instances from its draw lists. Inactive or disabled: nothing records and nothing is allocated.

## S33. Virtual-geometry page, inventory and evidence
`bash scripts/build-virtual-geometry.sh` builds `dist/virtual-geometry/` (bricks, page, `lib/`, `scenes/`, `shared/`, `scenes.json`, `build.json`). Stages `bake` (4 rows) and `passes` (21 rows over seven presets, `msaa: 1`, `-on`/`-off` pairs with VG_PAIR_RUN). Records follow S26 through `node web/common/vg-job.mjs <action> --page dist/virtual-geometry --out out/virtual-geometry/<stage> --stage <bake|passes>`; commands `p3-vg-*`. The judge uses no reference.

## S34. SDF Modeler virtualized-geometry integration
- Export bakes every non-skinned mesh with `bakeClusters` and stores it with MAMA `setMeshClusters`.
- `RenderSettings.gameVirtualGeometry` (default false) is applied to the Game engine by `applyGameVirtualGeometry`; refusals are prefixed `Game virtual geometry refused: `.
- GameDisplay uses msaa 1 while it is on, attaches saved sections to static draws, keeps other draws forward and reports `virtualGeometryState()`.
- Evidence: scenario `virtual-geometry-app`, milestone `p3-editor`, `out/virtual-geometry-editor`, commands `p3-editor-*`.

## S35. Pinning tests of the virtualized-geometry contracts
raa-cluster, raa-vg, raa-shader `vg_variants.rs`, raa-lit `virtual_set.rs`, raa-shadow `cascade_access.rs`, raa-pipeline `config.rs`, raa-js `{cluster_bake,virtual_geometry}.rs`, `examples/virtual-geometry/*.test.mjs`, `web/common/{page-job,vg-job}.test.mjs`, MAMA and SDFM mesh-clusters and virtual-geometry tests.

## S36. Global-illumination API (m30)
`types/m30-global-illumination.d.ts`, `docs/api/m30-global-illumination.md`, raa-js `m30_global_illumination.rs`.
- `setGlobalIllumination(patch)` / `globalIllumination()`: `enabled` false, `intensity` 1 [0, 16], `environmentDiffuse` 0 [0, 1], `localLightShadows`, `reflections`, `screenTraces` true, `debugView` none/voxels/radiance/distance/levels, `screenSpace` false, `screenSpaceQuality` low/medium/high (medium). `setGiQuality` / `giQuality()`: resolution 64|96, levels 3-5, probeRays 6|8; a change reallocates and rebuilds. Patches merge in key order; bad keys or values throw `gi.invalid` `<key>: <reason>`, nothing changes.
- `setInstanceGlobalIllumination(instance, boolean | null)`, `instanceGlobalIllumination(instance)` `{ requested, voxelized, driver, code, message }` (`api.unknown_handle`).
- `giStats()` `{ active, inactiveReason (disabled, passDisabled, multiView, noInstance), quality, finestVoxelMetres, voxelizedForwardInstances, voxelizedVirtualInstances, pendingRebuild, framesSinceRebuild, allocatedBytes, statsFrameIndex, pending }`, published on complete submission. Rebuild diagnostics report the cause, level and remaining flood passes of the submitted rebuild, plus the number of pending levels and the number relit.
- `setMeshVertexColors(mesh, colors, components?)`, `setMeshClusterVertexColors(mesh, ordinal, colors, components?)`: GI albedo = base x tint x vertex colour.
- `frame` feeds camera, sun and light count, invalidates history on cuts and calls `prepare_frame`; `finish_scene_invocation` calls `observe_submission`.

## S37. GI clipmap volume, membership and voxel drivers (raa-gi)
Adapted from solstice `src/gi/*` and `shaders/gi/*`; CPU twins in `clipmap.rs`, `cpu.rs` and part `cpu` modules.
- L levels of R^3 voxels around the camera, origins snapped by 8 voxels; `choose_rebuild` picks the level furthest behind (ties finest); a rebuild clears, voxelizes through every driver, seeds, jump-floods (18 passes, 6 per frame), commits and writes the Rgba16Float atlas R x R x (R x L) (rgb radiance, a distance). One level relit per frame (`light_schedule`). Memory = GI_VOLUME_BYTES, only while active.
- Membership: every item receives GI; voxelized unless rank 0 `gi.opted_out`, 1 `gi.skinned_instance`, 2 `gi.deformed_instance`, 3 `gi.non_opaque_material`, 4 `gi.no_shadow_material`. A signature change queues every level for refresh. Each level consumes the current scene at voxelization, and an edit during its flood passes queues that level again.
- Driver A: coarsest raa-lit LOD within half a voxel, one dispatch per item over `LitHandle::mesh_storage`. Driver B (part `virtual_driver`, with VG): claims virtual instances, sets raa-vg `INSTANCE_GI`, culls each level's view (slot 6 + level) in `VgHandle::with_gi_frame` and voxelizes clusters; one shared grid.
- `GiFeature::install` registers `gi.volume` (25), `gi.history` (64), driver A and the parts of `src/parts/*.rs`; inactive frames release everything and set no feature ids (29, 30, 37 while active).

## S38. GI GPU records and passes
Twins in `crates/raa-gi/src/layouts.rs`. `GiFrameGpu` 432 B: matrices view_proj, inv_view_proj, inv_proj, inv_view, previous_view_proj @0-320, camera_position @320, sun_direction @336, sun_color @352, viewport @368, counts u32x4 @384 (lights, frame, debug view, levels), settings @400 (intensity, environment diffuse, near, far), flags @416 (history, screen traces, local shadows, reflections). 32 B `GiVoxelizeGpu`, `GiFloodGpu`, `GiLightGpu`, `GiWriteGpu`; `GiForwardWorkGpu` 48 B in 256 B slots. `RaaGiVolumeParams` 96 B: `levels[5]` (origin, voxel size) @0, resolution @80, intensity @84, level_count @88, flags @92 (1 valid, 2 reflections, 4 local shadows).
- `GI_VOLUME_WGSL` (raa-shader): `raa_gi_level_at`, `raa_gi_sample`, `raa_gi_trace`, `raa_gi_visibility`, `raa_gi_cone_visibility`, `raa_gi_edge_fade`, `raa_gi_finest_voxel`; no binding, `textureSampleLevel` only.
- Compute modules (voxelize, seed, flood, write, voxel light, debug view) use the GI compute group 1 and a private group 2 (<= 8 storage buffers). `gi.volume` calls `set_gi`; `gi.history` copies lit HDR and draws debug views. Pairs: vg.resolve < gi.volume < gi.screen; prepass, vg.gbuffer < gi.screen < forward < gi.history < ssr.

## S39. GI screen probes, slots and reflection fallback
- `gi.screen` (42): depth pyramid (`GiShared::pyramid_view`), one probe per 16 px tile, P x P rays through the previous frame then the volume, SH9 probes, integrated into Rgba16Float (rgb irradiance, a occlusion; `probe_output_view`); history needs a valid, on-screen, unrejected motion neighbourhood.
- `gi.ssgi` (44, `screenSpace`): visibility-bitmask indirect diffuse (32 sectors, 2 m, reconstructed thickness at most 0.5 m), temporally accumulated. Screen coverage weights the probe replacement, with output `(E_ss + V x E_probe, probe.a)`. Off allocates nothing. Binding 14 is the last part screen output.
- Slots `gi`: Indirect = (1 - metallic) x base / pi x g.rgb x intensity x s.ao; Ao = min(GTAO, g.a); LocalVisibility = volume (cone for area) visibility. The part also sets IBL diffuse scale and GI specular (S45).
- SSR with flag 2 falls back to a volume trace on a screen miss.

## S40. Global-illumination page, inventory and evidence
`bash scripts/build-global-illumination.sh` (bricks, then `build-page.mjs global-illumination --stages volume,lighting,virtual --shared examples/virtual-geometry/lib/content.js --shared examples/trace-first-pixel/scene.js`). Runner topic `gi` with GI facts, readiness active + settled 240 frames (disabled: inactive, 0 bytes), extras `quality`, `allocatedBytes`. Rows add `heldConstant`, `producer`, optional `judgeReference` (`/refs/*.png`) and `judgeAspect`. Scenes: volume, lighting, area-shadows, area-stochastic, area-restir, temporal, ssgi, gi-quality, virtual (msaa 1); Cornell rows (millimetres / 100) judged against `/refs/cornell-box-reference.png`. Records: `node web/common/gi-job.mjs <action> --page dist/global-illumination --out out/global-illumination/<stage> --stage <stage>`; commands `p4-gi-*`.

## S41. SDF Modeler global-illumination integration
`RenderSettings.gameGlobalIllumination` (false) and `gameGiQuality` (high; low 64/3/6, medium 96/4/8, high 96/5/8), labelled `Global illumination (Game view)` and `Global illumination quality`. `applyGameGlobalIllumination` applies on change only; refusals `Game global illumination refused: ` under status `global-illumination`; the engine and msaa are kept. `GameDisplay.globalIlluminationState()` is ready when active, settled and voxelized. Saved meshes may carry linear vertex `colors`. Game lighting controls (area tiers, TAA, screen-space GI) apply through `applyGameLighting` (`Game lighting refused: `). Evidence: scenario `global-illumination-app`, milestone `p4-editor`, commands `p4-editor-*`.

## S43. Temporal anti-aliasing and the motion target
- `taa` (100, installed disabled; `setPassEnabled('taa', on)`): Catmull-Rom history, YCoCg clip, standard-Z, own `HistoryPair` per viewport; uses history only with `temporal.y` and `temporal.z` and no invalidation (resize, disable, camera cut). Enabling it turns on the FrameGpu jitter.
- Motion: xy = UV displacement current minus previous; z = 1 rejects (no previous model or vertex function: camera-only xy; no camera history or bad clip: xy 0); vg.gbuffer z 0; sky 0. Deformed instances re-evaluate deform at `t - dt` (`VARIANT_FLAG_MOTION_DEFORM`).
- `HistoryPair` and `REPROJECT_WGSL` also serve area-shadow, ReSTIR and SSGI histories. Evidence: GI `temporal`, lookdev `m31-taa`.

## S44. Area-light shading and soft-shadow tiers
- `setAreaLightShading({ mode: 'ltc' | 'formFactor' })` (feature 39). `setAreaLightShadows` / `areaLightShadows()`: `tier` auto|hero|cone|fade, `heroCount` 0..4 (0), `heroDistance` 25, `fadeDistance` 80, `band` 5, `stochasticCount` 0..8 (4), `stochasticDistance` 40, `stochasticCells` 4|9, `stochasticMode` stratified|restir; plus `activeHeroes`, `activeStochastic`.
- Cone: GI cone visibility; fade: direct term fades on GI frames; hero: 140 degree maps in shadow layers 4-7 with the shared PCSS kernel (opt-in).
- Mid tier (GI active, `localLightShadows`): `gi.area-shadow` (43) traces one stratified ray per pixel and slot (pyramid, volume, SDF trace scene), reprojects S and U, filters bilaterally and writes V = S / U into binding 20. ReSTIR tier: same pass with ReSTIR DI reservoirs.
- Blend: `V_far = mix(V_cone, V_mask, w_stoch)`, `V = mix(mix(V_far, V_hero, map weight), V_far, w_cone)`.

## S45. GI-occluded environment specular
`EnvSh` is 176 B: `sh[9]` @0, `info` @144 (specular, diffuse x scale, max mip, enabled), `gi` @160 (x = w_gi). `set_env_gi` stores it; raa-gi sets 1 while active through `IblHandle::set_gi_specular`. Ambient IBL specular and the SSR-subtracted specular are mix(L_sky, E_gi / pi, w) x (f0 dfg.x + dfg.y).

## S46. Automatic exposure (raa-exposure, pass `exposure.auto`, JS m33)
Adapted from solstice `crates/solstice-render/shaders/post/exposure.wgsl:21-83` (adapted files carry `Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/exposure.wgsl, MIT licence, Copyright (c) 2026 SamG-Coder`).

**Constants** (`lib.rs`): `AUTO_EXPOSURE_PASS_NAME` "exposure.auto", `AUTO_EXPOSURE_ORDER` 125, `HISTOGRAM_BINS` 128, `MIN_LOG_LUMINANCE` -8, `LOG_LUMINANCE_RANGE` 18, `SAMPLE_STRIDE` 4, `HISTOGRAM_WORKGROUP` 16, `LUMINANCE_FLOOR` 1e-5, `MIDDLE_GREY` 0.18, `SETTLED_EV` 0.01, `PARAMS_SIZE` 48, `RESULT_SIZE` 16, `READBACK_SLOTS` 3.

**Settings** `AutoExposureDesc` (`Default`, `validate()` in row order, first failure `post.invalid` "`<jsKey>` must be <rule>, got <value>"):

| JS key | default | rule |
|---|---|---|
| `enabled` | false | |
| `compensationEv` | 0.263 (log2 1.2) | finite, -8..=8 |
| `lowPercent` | 0.2 | finite, 0..=1, < highPercent |
| `highPercent` | 0.7 | finite, 0..=1 |
| `speedUp` / `speedDown` | 2.4 / 1.1 1/s | finite, > 0 |
| `minEv` / `maxEv` | -6 / 6 | finite, minEv <= maxEv |

**Maths** (CPU twin `src/maths.rs`; the WGSL computes the same):
```
bin_index(l) = None if l < 1e-5, else Some(u32(clamp((log2(l) + 8) / 18, 0, 1) * 127))
band_average(hist, low, high): weights each bin by its overlap with [total*low, total*high],
  log_l = -8 + (i+0.5)/128*18; Some(exp2(sum(log_l*used) / max(sum(used), 1))), None if empty
adapt(prev, meas, dt, up, down): prev None -> meas or 1; meas None -> prev;
  else mix(meas, prev, exp(-dt * (meas > prev ? up : down)))
exposure_from_average(avg, d) = clamp(2^compensation_ev * 0.18 / max(avg, 1e-6), 2^min_ev, 2^max_ev)
settled(a, b) = abs(b - a) < 0.01
```
The pass snaps on its first prepared frame, a real HDR size change, `invalidate()` and any frame gap (disabled or skipped). Zero counted pixels keep the previous average (1.0 without one).

**GPU.** `ExposureParams` (48 B): `dt_s`, `first`, `compensation`, `low_percent`, `high_percent`, `speed_up`, `speed_down`, `min_exposure`, `max_exposure`, `frame` (@0-36). `ExposureState` (16 B, also the readback record): `average_luminance`, `exposure`, `valid`, `frame`. Histogram `array<atomic<u32>, 128>`. `exposure_compute.wgsl`: `build_histogram` (16 x 16, every 4th pixel of `hdr().src()`) and `resolve` (1 thread, clears the histogram); `exposure_apply.wgsl`: `vs_main`, `fs_apply` (`rgb * exposure` into `hdr().dst()`, then swap). The state is copied into an idle slot of three MAP_READ buffers, mapped in the next `prepare`, never waited on; results older than the latest snap are ignored. No group-1 binding, no change to `FrameGpu` or tonemap; `setToneMapping`'s `exposureEv` multiplies after.

**State** `AutoExposureState { exposure_ev, average_luminance, settled, pending }`: disabled -> `{None, None, true, false}`; no result since the snap -> `{None, None, false, true}`; else the latest result, `settled` over the two latest with distinct `frame`.

**JS (m33)**: `setAutoExposure(settings)` merges, validates, sets the pipeline flag and the handle (nothing changes on refusal); `autoExposure()` (`enabled` from the pipeline flag); `autoExposureState(viewport = 0)` -> `{ exposureEv, averageLuminance, settled, pending }`. The average is per view: `ViewPost.exposure` in `m07_pipeline.rs` receives the main desc in `prepare_view`; the flag is global.

## S47. Sharpening (raa-sharpen, pass `sharpen`, JS m34) and raa-sdfm render options
Kernel adapted from solstice `crates/solstice-render/shaders/post/tonemap.wgsl:95-103` (MIT, attribution as S46); taps are range-compressed because the chain is HDR.

`SHARPEN_PASS_NAME` "sharpen", `SHARPEN_ORDER` 195, `SHARPEN_WEIGHT` 0.2, `COMPRESS_LIMIT` 0.999, `PARAMS_SIZE` 16. `SharpenDesc { enabled, sharpness }` default `{ false, 0.9 }`; sharpness outside [0, 1] -> `post.invalid`.
```
compress(c) = max(c, 0) / (1 + max channel); expand(d) = d / (1 - min(max channel, 0.999))
sharpen_texel: compress centre and N, S, W, E; low/high = min/max of the five
  amount = sqrt(clamp(min(low, 1 - high) / max(high, 1e-4), 0, 1)); w = -amount * sharpness * 0.2
  out = clamp((c + (n+s+w+e) * w) / (1 + 4w), low, high) -> (expand(out), centre alpha)
```
CPU twin `src/kernel.rs`; `wgsl/sharpen.wgsl` (`vs_main`, `fs_sharpen`; params, HDR texture; clamped `textureLoad` taps). Standard post pass at render resolution, no per-view state, no halo by construction.

**JS (m34)**: `setSharpening({ enabled?, sharpness? })` and `sharpening()`, refusals as m33.

**raa-sdfm** `applyRenderOptions`: `autoExposure?` and `sharpening?` take a boolean (`{ enabled }`) or a settings object; other values or a missing engine method are skipped with a reason; engine refusals throw.

## S48. Material atlas mip chain (raa-lit textures, prelude `raa_texture_grad`, m22 options)
**Helpers** (`crates/raa-lit/src/textures.rs`, re-exported): `ATLAS_MIP_LEVELS` 5, `ATLAS_ALIGN` 16, `texture_levels(w, h) = min(5, 1 + floor(log2(min(w, h))))`, `level_size` (halve rounding up), `TextureOptions { mips: true, srgb: false }`, `texture_lod(size, duv_dx, duv_dy, levels) = clamp(0.5 * log2(max(rho2, 1e-12)), 0, levels - 1)` (rho2 = max squared texel-space derivative), `level_inset(lod, atlas) = 0.5 * exp2(ceil(lod)) / atlas`.

`src/mipgen.rs` `downsample_rgba8(src, w, h, srgb)`: 2 x 2 box, edge repeated on odd sides, linear `(a+b+c+d+2)/4` or sRGB-correct rgb; each level from the previous. Textures reserve 16-aligned extents (`texture.atlas_full` on reserved size); with `mips` levels 1.. upload at `(x >> k, y >> k)`; the rect record negates w for `mips: false`. Sampler mipmap filter Linear.

**Prelude**: `RAA_ATLAS_MIP_LEVELS = 5.0`; `raa_texture(id, uv)` samples level 0; `raa_texture_grad(id, uv, duv_dx, duv_dy)` computes the lod, insets the uv and calls `textureSampleLevel` (safe in non-uniform flow). Callers take `dpdx`/`dpdy` of the unwrapped uv at the top of `material_main`.

**JS (m22)**: `addTexture({ width, height, data, mips?, srgb? })`; `LitRenderer::add_texture_with(.., options)`. Users: raa-sdfm `field-bake.js`, `bake-scheduler.js` (colour `srgb: true`), sdfm `import-textures.ts`.

## S49. Post preset page `examples/post/` and its job
Built like `examples/global-illumination/`: `main.js` (`discoverScenes`, `attachPostPage`), `lib/inventory.js` (`validateInventory(namespaces, stage)`, cumulative exposure < sharpen < mips; `validateRequiredFacts`), `lib/runtime.js` (`createPostPageRunner`), `lib/content.js` (generated geometry and `stripes`/`checker` textures), `scenes/{exposure,sharpen,mips}.js`. `scripts/build-post.sh` runs `build-bricks.sh` then `build-page.mjs post --stages exposure,sharpen,mips --shared examples/virtual-geometry/lib/content.js`. `web/common/post-job.mjs` configures `createPageJob` (topic `post`, `keepSameBuildOnly`, store `$NMC_CACHE_DIR/post/<stage>`, out `out/post/<stage>`); `web/common/view-check.mjs` (`boxInView`, `assertSubjectsInView`) checks every variant's framing.

Rows (1280 x 720, below scale 1 drawn into a 1280 x 720 canvas; measured rows carry PAIR_RUN; judge text in each scene's `inventory`): `exposure-room` exposure-interior-on/-off, exposure-exterior-on/-off (`-on` draws until settled, POST_SETTLE_MAX_FRAMES); `sharpen-detail` sharpen-on/-off, upscale-067-sharpen-on/-off (TAA on, scale 0.67 for upscale); `mips-ground` mips-on/-off (`-off` adds textures with `mips: false`). Commands `p5-post-build` and `p5-<stage>-{check,select,clean,render,fresh,judge}`.

## S50. SDF Modeler Game view post controls (sdfm)
**Settings** `src/core/game-post-settings.ts` (spread into `render-settings.ts`, persisted): `gameAutoExposure`, `gameExposureCompensation` 0.263, `gameExposureMinEv` -6, `gameExposureMaxEv` 6, `gameExposureSpeedUp` 2.4, `gameExposureSpeedDown` 1.1, `gameSharpening`, `gameSharpness` 0.9, `gameTemporalUpscaling` (booleans false). Controls in groups 'Auto exposure' and 'Sharpening' (incl. 'Temporal upscaling (0.67 scale, TAA, sharpening)'), Game view only. Helpers `autoExposureOf`, `sharpeningOf`, `temporalUpscalingPatch`, `effectiveRenderScale(option, s, cel)` = cel ? 1 : upscaling ? 0.67 : option; `src/app/temporal-upscaling.ts` turns on TAA and sharpening once when upscaling turns on.

**Apply** `src/app/paramesh/game-post.ts`: `applyGamePost(engine, s)` (skip reasons), `nextExposureWatch` (draw until 2 settled reads, at most `EXPOSURE_FRAME_WATCH` 600 frames); GameDisplay arms it in `setRenderSettings` and runs `watchExposure()` per frame. `import-textures.ts` samples with `raa_texture_grad` and adds base colour and emissive with `srgb: true`.

**Evidence** `src/core/post-evidence.ts`, `src/node/evidence/topics/post.ts`, scenario `post-app.ts`; game `post-editor`, assets `post.room-opening`, `post.textured-tiles`, out `out/post-editor`; entries `post-exposure-off`, `post-exposure-on`, `post-scale-1`, `post-upscale-on`, `post-texture-far`; commands `p5-editor-*`.

## S51. Post passes beside the render graph
Seam edits of the post work: `m07_pipeline.rs` (`ViewPost.exposure`, `view_auto_exposure`), raa-lit `add_texture_with`, `index.d.ts` typings, sdfm `render-settings.ts`, `game-display.ts`. When the render graph lands, `exposure.auto` and `sharpen` become nodes `pass.exposure.auto` and `pass.sharpen` (HDR `color` in and out); until then they keep the state their setters gave them.

## Module ownership

| Module | Files | Consumes |
|---|---|---|
| workspace | `Cargo.toml`<br>`Cargo.lock`<br>`.gitignore` | none |
| tooling | `scripts/check.sh`<br>`scripts/build-web.sh`<br>`scripts/check-no-assets.sh`<br>`scripts/hooks/pre-commit`<br>`scripts/gpu-capture.mjs`<br>`web/serve.mjs`<br>`web/common/nmc-gpu.js`<br>`web/common/run-loop.js`<br>`proofs/lib/proof.mjs` | workspace |
| proof-m1 | `proofs/m1-boot.mjs`<br>`proofs/results/m1-boot.json`<br>`docs/dogfood/m1.md` | tooling |
| proof-m2 | `proofs/m2-scene.mjs`<br>`proofs/results/m2-scene.json`<br>`docs/dogfood/m2.md` | tooling |
| sdf-ref-tool | `tools/sdf-ref/gen-leaf-fixtures.mjs`<br>`tools/sdf-ref/README.md` | raa-prims |
| proof-m3 | `proofs/m3-prims.mjs`<br>`proofs/results/m3-prims.json`<br>`proofs/results/m3-prims-table.json`<br>`docs/dogfood/m3.md` | tooling |
| proof-m4 | `proofs/m4-loader.mjs`<br>`proofs/results/m4-asset.json`<br>`docs/dogfood/m4.md` | tooling |
| proof-m5 | `proofs/m5-booleans.mjs`<br>`proofs/results/m5-booleans.json`<br>`proofs/results/m5-csg-cases.json`<br>`docs/dogfood/m5.md` | tooling |
| raa-js | `crates/raa-js/Cargo.toml`<br>`crates/raa-js/README.md`<br>`crates/raa-js/build.rs`<br>`crates/raa-js/src/lib.rs`<br>`crates/raa-js/src/engine.rs`<br>`crates/raa-js/src/convert.rs`<br>`crates/raa-js/src/state.rs`<br>`crates/raa-js/src/feature_map.rs`<br>`crates/raa-js/tests/convert.rs`<br>`crates/raa-js/tests/api_registry.rs` | raa-gi-core |
| js-package | `scripts/build-package.sh`<br>`packages/raa/package.json`<br>`packages/raa/index.js`<br>`packages/raa/index.d.ts`<br>`packages/raa/README.md`<br>`packages/raa/LICENSE`<br>`scripts/build-bricks.sh`<br>`packages/raa/sky-weather.js`<br>`packages/raa/test/sky-weather.test.mjs`<br>`packages/raa/test/load-once.test.mjs` | raa-js, api-m8, api-m29-bake |
| project-docs | `README.md`<br>`docs/API.md`<br>`docs/ARCHITECTURE-notes.md` | js-package, raa-js |
| proof-m6 | `proofs/m6-api.mjs`<br>`proofs/results/example-basic.json`<br>`proofs/results/m6-sizes.json`<br>`docs/dogfood/m6.md` | tooling, js-package, project-docs |
| api-m7 | `packages/raa/types/m07-meshes-hdr.d.ts`<br>`docs/api/m07-meshes-hdr.md`<br>`crates/raa-js/src/api/m07_meshes.rs`<br>`crates/raa-js/src/api/m07_pipeline.rs` | raa-js |
| lookdev-page | `examples/lookdev/index.html`<br>`examples/lookdev/main.js`<br>`examples/lookdev/nmc.gpu.json`<br>`examples/lookdev/lib/geometry.js`<br>`examples/lookdev/lib/objects.js`<br>`examples/lookdev/lib/runner.js` | js-package, api-m7, tooling |
| lookdev-tools | `scripts/build-lookdev.sh`<br>`scripts/lookdev-capture.mjs`<br>`proofs/lib/p2.mjs`<br>`proofs/lib/p2-smoke.mjs` | tooling, js-package, lookdev-page |
| proof-m7 | `proofs/m7-meshes-hdr.mjs`<br>`examples/lookdev/scenes/m07-base.js`<br>`docs/dogfood/m7.md` | lookdev-tools, lookdev-page, raa-js |
| api-m8 | `crates/raa-js/src/api/m08_sky_ibl.rs`<br>`packages/raa/types/m08-sky-ibl.d.ts`<br>`docs/api/m08-sky-ibl.md`<br>`crates/raa-js/tests/sky_weather.rs` | raa-js |
| proof-m8 | `proofs/m8-sky-ibl.mjs`<br>`examples/lookdev/scenes/m08-sky-ibl.js`<br>`docs/dogfood/m8.md` | lookdev-tools, api-m8 |
| api-m9 | `crates/raa-js/src/api/m09_shadows_ao.rs`<br>`packages/raa/types/m09-shadows-ao.d.ts`<br>`docs/api/m09-shadows-ao.md` | raa-js |
| proof-m9 | `proofs/m9-shadows-ao.mjs`<br>`examples/lookdev/scenes/m09-shadows-ao.js`<br>`docs/dogfood/m9.md` | lookdev-tools, api-m9 |
| api-m10 | `crates/raa-js/src/api/m10_materials.rs`<br>`packages/raa/types/m10-materials.d.ts`<br>`docs/api/m10-materials.md` | raa-js |
| proof-m10 | `proofs/m10-materials.mjs`<br>`examples/lookdev/scenes/m10-materials.js`<br>`docs/dogfood/m10.md` | lookdev-tools, api-m10 |
| api-m11 | `crates/raa-js/src/api/m11_transmission.rs`<br>`packages/raa/types/m11-transmission.d.ts`<br>`docs/api/m11-transmission.md` | raa-js |
| proof-m11 | `proofs/m11-transmission.mjs`<br>`examples/lookdev/scenes/m11-transmission.js`<br>`docs/dogfood/m11.md` | lookdev-tools, api-m11 |
| sdfm-compare | `scripts/build-sdfm-compare.sh` | lookdev-tools |
| proof-m12 | `proofs/m12-sdfm.mjs`<br>`examples/sdfm-compare/scenes/m12-fixtures.js`<br>`docs/dogfood/m12.md` | lookdev-tools, sdfm-compare |
| api-m13 | `crates/raa-js/src/api/m13_liquids_particles.rs`<br>`packages/raa/types/m13-liquids-particles.d.ts`<br>`docs/api/m13-liquids-particles.md` | raa-js |
| proof-m13 | `proofs/m13-liquids-particles.mjs`<br>`examples/lookdev/scenes/m13-liquids-particles.js`<br>`docs/dogfood/m13.md` | lookdev-tools, api-m13 |
| api-m14 | `crates/raa-js/src/api/m14_camera_effects.rs`<br>`packages/raa/types/m14-camera-effects.d.ts`<br>`docs/api/m14-camera-effects.md` | raa-js, api-m8, api-m10 |
| proof-m14 | `proofs/m14-camera-effects.mjs`<br>`examples/lookdev/scenes/m14-camera-effects.js`<br>`docs/dogfood/m14.md` | lookdev-tools, api-m14 |
| api-m15 | `crates/raa-js/src/api/m15_animation.rs`<br>`packages/raa/types/m15-animation.d.ts`<br>`docs/api/m15-animation.md` | raa-js |
| api-m28 | `crates/raa-js/src/api/m28_lights.rs`<br>`packages/raa/types/m28-lights.d.ts`<br>`docs/api/m28-lights.md` | raa-js |
| mama-light-record | `mama:crates/procgen/src/nodes/game_markers.rs`<br>`mama:crates/procgen/src/passes/gameplay.rs`<br>`mama:crates/procgen/src/output.rs`<br>`mama:crates/procgen/SPEC.md`<br>`mama:src/procgen/output-types.ts` | none |
| mama-light-transport | `mama:src/procgen/scene-file.ts`<br>`mama:src/procgen/scene-build.ts`<br>`mama:src/procgen/{scene-file,scene-build,runtime}.test.ts`<br>`mama:crates/procgen/src/ext/runtime/delta.rs`<br>`mama:crates/procgen/tests/runtime.rs`<br>`mama:src/procgen/scene-file.test.ts`<br>`mama:src/procgen/scene-build.test.ts`<br>`mama:src/procgen/runtime.test.ts` | mama-light-record |
| sdfm-light-authoring | `sdfm:src/core/scene-lights.ts`<br>`sdfm:src/plugins/ui/30-point-lights.ts`<br>`sdfm:src/components/editor/PointLightSection.vue`<br>`sdfm:src/node/schema/{docs,json-schema}.ts`<br>`sdfm:docs/schema/{sdfm-document.schema,tools}.json`<br>`sdfm:src/core/scene-lights.test.ts`<br>`sdfm:src/core/scene-lights-authoring.test.ts`<br>`sdfm:src/node/schema/docs.ts`<br>`sdfm:src/node/schema/json-schema.ts`<br>`sdfm:src/app/gpu-page/scene-lights-controls.test.ts` | mama-light-record |
| sdfm-game-light-sync | `sdfm:src/app/paramesh/game-light-sync.ts`<br>`sdfm:src/app/paramesh/game-display.ts`<br>`sdfm:src/app/paramesh/game-light-sync.test.ts`<br>`sdfm:src/app/paramesh/game-display-lights.test.ts` | sdfm-light-authoring, api-m28, sdfm-sky-weather-settings, sdfm-vg-settings, api-m29-bake, api-m29-vg, mama-mesh-clusters, sdfm-gi-settings, api-m30 |
| sdfm-light-evidence | `sdfm:src/core/scene-lights-evidence.ts`<br>`sdfm:src/node/evidence/topics/scene-lights.ts`<br>`sdfm:src/app/gpu-page/scenarios/{scene-lights-app,ui-driver}.ts`<br>`sdfm:src/app/gpu-page/ui-driver.test.ts`<br>`sdfm:src/node/evidence/run.ts`<br>`sdfm:.nmc/commands.json`<br>`sdfm:src/core/scene-lights-evidence.test.ts`<br>`sdfm:src/node/evidence/topics/scene-lights.test.ts`<br>`sdfm:src/app/gpu-page/scenarios/scene-lights-app.ts`<br>`sdfm:src/app/gpu-page/scenarios/ui-driver.ts`<br>`sdfm:src/app/gpu-page/scene-lights-visibility.ts`<br>`sdfm:src/app/gpu-page/scene-lights-requests.ts` | sdfm-light-authoring, sdfm-game-light-sync |
| solstice-architecture-finalization | `ARCHITECTURE.md` | api-m8, sdfm-sky-weather-settings, sdfm-sky-weather-evidence |
| proof-m15 | `proofs/m15-animation.mjs`<br>`examples/lookdev/scenes/m15-waddle.js`<br>`docs/dogfood/m15.md` | lookdev-tools, api-m15 |
| raa-cel | `crates/raa-js/src/api/m27_cel.rs`<br>`packages/raa/types/m27-cel.d.ts`<br>`docs/api/m27-cel.md` | raa-js |
| p2-managed-transport | `web/common/lights-job.mjs`<br>`web/common/lights-job.test.mjs`<br>`web/common/gpu-job.mjs`<br>`web/common/gpu-job.test.mjs` | tooling |
| sky-weather-page | `examples/sky-weather/index.html`<br>`examples/sky-weather/main.js`<br>`examples/sky-weather/nmc.gpu.json`<br>`examples/sky-weather/scenes/foundation.js`<br>`examples/sky-weather/contract.test.mjs`<br>`scripts/build-sky-weather.sh`<br>`.nmc/commands.json` | js-package, api-m7, lookdev-page, p2-managed-transport |
| sky-atmosphere-scenes | `examples/sky-weather/scenes/atmosphere.js`<br>`examples/sky-weather/atmosphere.test.mjs` | api-m8, sky-weather-page |
| sky-weather-scenes | `examples/sky-weather/scenes/weather.js`<br>`examples/sky-weather/weather.test.mjs` | sky-weather-page, sky-atmosphere-scenes, api-m8 |
| sdfm-sky-weather-settings | `sdfm:src/core/render-settings.ts`<br>`sdfm:src/core/sky-weather-settings.ts`<br>`sdfm:src/core/sky-weather-settings.test.ts`<br>`sdfm:src/app/render-settings.ts`<br>`sdfm:src/components/shell/RenderSettingsPanel.vue`<br>`sdfm:src/components/shell/SkyWeatherSettings.vue`<br>`sdfm:src/app/paramesh/game-sky-weather.test.ts`<br>`sdfm:src/node/ops/render-settings.ts`<br>`sdfm:docs/schema/sdfm-document.schema.json`<br>`sdfm:docs/schema/tools.json`<br>`sdfm:src/app/procedural/raytrace-display.ts`<br>`sdfm:src/plugins/ui/36-display.ts`<br>`sdfm:src/app/paramesh/game-render-settings.ts`<br>`sdfm:docs/llm/catalog.md`<br>`sdfm:tsconfig.node.json` | api-m8, js-package, api-m29-vg, sdfm-vg-settings, sdfm-gi-settings, api-m30 |
| sdfm-sky-weather-evidence | `sdfm:src/core/sky-weather-evidence.ts`<br>`sdfm:src/core/sky-weather-evidence.test.ts`<br>`sdfm:src/app/gpu-page/scenarios/sky-weather-app.ts`<br>`sdfm:src/app/gpu-page/sky-weather-visibility.ts`<br>`sdfm:src/app/gpu-page/sky-weather-controls.test.ts`<br>`sdfm:src/node/evidence/topics/sky-weather.ts`<br>`sdfm:src/node/evidence/topics/sky-weather.test.ts` | sdfm-sky-weather-settings, sdfm-game-light-sync, sdfm-light-evidence, sky-weather-scenes, sdfm-editor-evidence |
| raa-cloud-seam | `crates/raa-sky/src/cloud_backend.rs`<br>`crates/raa-sky/tests/backend_registry.rs` | raa-sky, raa-pipeline, raa-scene, raa-math |
| sky-weather-job-resume | `web/common/sky-weather-job.mjs`<br>`web/common/sky-weather-job.test.mjs` | sky-weather-page, p2-managed-transport |
| raa-cluster | `crates/raa-cluster/Cargo.toml`<br>`crates/raa-cluster/README.md`<br>`crates/raa-cluster/src/lib.rs`<br>`crates/raa-cluster/src/error.rs`<br>`crates/raa-cluster/src/packed.rs`<br>`crates/raa-cluster/src/bounds.rs`<br>`crates/raa-cluster/src/dvec.rs`<br>`crates/raa-cluster/src/input.rs`<br>`crates/raa-cluster/src/weld.rs`<br>`crates/raa-cluster/src/partition.rs`<br>`crates/raa-cluster/src/group.rs`<br>`crates/raa-cluster/src/simplify.rs`<br>`crates/raa-cluster/src/thin.rs`<br>`crates/raa-cluster/src/hierarchy.rs`<br>`crates/raa-cluster/src/dag.rs`<br>`crates/raa-cluster/src/bake.rs`<br>`crates/raa-cluster/src/cut.rs`<br>`crates/raa-cluster/src/store.rs`<br>`crates/raa-cluster/tests/bake.rs`<br>`crates/raa-cluster/tests/cuts.rs`<br>`crates/raa-cluster/tests/sliced.rs`<br>`crates/raa-cluster/tests/packed.rs`<br>`crates/raa-cluster/tests/fixtures/open-strip.json`<br>`crates/raa-cluster/tests/fixtures/seam-mesh.json` | — |
| api-m29-bake | `crates/raa-js/src/api/m29_cluster_bake.rs`<br>`crates/raa-js/tests/cluster_bake.rs`<br>`packages/raa/types/m29-cluster-bake.d.ts`<br>`docs/api/m29-cluster-bake.md`<br>`packages/raa/test/cluster-bake.test.mjs` | raa-js, raa-cluster |
| mama-mesh-clusters | `mama:src/core/game/mesh-clusters.ts`<br>`mama:src/core/game/mesh-clusters.test.ts`<br>`mama:src/core/game/mesh-file.ts`<br>`mama:src/core/game/mesh-file.test.ts`<br>`mama:docs/mesh-clusters.md` | — |
| raa-sdfm-clusters | `packages/raa-sdfm/src/mesh-clusters.js`<br>`packages/raa-sdfm/test/mesh-clusters.test.js`<br>`packages/raa-sdfm/src/mesh-file.js`<br>`packages/raa-sdfm/src/procedural.js`<br>`packages/raa-sdfm/index.js`<br>`packages/raa-sdfm/index.d.ts`<br>`packages/raa-sdfm/README.md` | api-m29-bake |
| vg-page | `examples/virtual-geometry/index.html`<br>`examples/virtual-geometry/main.js`<br>`examples/virtual-geometry/nmc.gpu.json`<br>`examples/virtual-geometry/lib/inventory.js`<br>`examples/virtual-geometry/lib/runtime.js`<br>`examples/virtual-geometry/lib/content.js`<br>`examples/virtual-geometry/lib/cluster-preview.js`<br>`examples/virtual-geometry/lib/facts.js`<br>`examples/virtual-geometry/scenes/bake.js`<br>`examples/virtual-geometry/contract.test.mjs`<br>`examples/virtual-geometry/bake.test.mjs`<br>`scripts/build-virtual-geometry.sh` | js-package, api-m29-bake, lookdev-page, sky-weather-page, p2-managed-transport, vg-job, page-kit |
| vg-job | `web/common/vg-job.mjs`<br>`web/common/vg-job.test.mjs` | page-job, sky-weather-page |
| raa-shader-vg | `crates/raa-shader/src/wgsl/depth.wgsl`<br>`crates/raa-shader/src/wgsl/pass_vg_gbuffer.wgsl`<br>`crates/raa-shader/src/wgsl/pass_vg_shade.wgsl`<br>`crates/raa-shader/src/vg.rs`<br>`crates/raa-shader/tests/vg_variants.rs`<br>`crates/raa-shader/src/ids.rs`<br>`crates/raa-shader/src/composer.rs`<br>`crates/raa-shader/src/lib.rs` | raa-shader-gi |
| raa-pipeline-vg-order | `crates/raa-pipeline/src/pipeline.rs`<br>`crates/raa-pipeline/tests/config.rs`<br>`crates/raa-pipeline/README.md` | — |
| raa-lit-vg-seam | `crates/raa-lit/src/virtual_set.rs`<br>`crates/raa-lit/tests/virtual_set.rs`<br>`crates/raa-lit/src/prep.rs`<br>`crates/raa-lit/src/renderer.rs`<br>`crates/raa-lit/src/handle.rs`<br>`crates/raa-lit/src/lib.rs` | raa-lit-gi-seam |
| raa-shadow-vg-seam | `crates/raa-shadow/src/cascade_access.rs`<br>`crates/raa-shadow/src/feature.rs`<br>`crates/raa-shadow/src/caster.rs`<br>`crates/raa-shadow/src/lib.rs`<br>`crates/raa-shadow/tests/cascade_access.rs` | — |
| raa-vg-scene | `crates/raa-vg/Cargo.toml`<br>`crates/raa-vg/README.md`<br>`crates/raa-vg/src/lib.rs`<br>`crates/raa-vg/src/error.rs`<br>`crates/raa-vg/src/settings.rs`<br>`crates/raa-vg/src/scene.rs`<br>`crates/raa-vg/src/eligibility.rs`<br>`crates/raa-vg/tests/scene.rs`<br>`crates/raa-vg/tests/settings.rs` | raa-cluster, raa-lit-vg-seam |
| raa-vg-cull | `crates/raa-vg/src/culling.rs`<br>`crates/raa-vg/src/hzb.rs`<br>`crates/raa-vg/src/wgsl/pass_view.wgsl`<br>`crates/raa-vg/src/wgsl/cull.wgsl`<br>`crates/raa-vg/src/wgsl/cull_control.wgsl`<br>`crates/raa-vg/src/wgsl/hzb_seed.wgsl`<br>`crates/raa-vg/src/wgsl/hzb_down.wgsl`<br>`crates/raa-vg/tests/pass_view.rs`<br>`crates/raa-vg/tests/cull_variants.rs` | raa-vg-scene, raa-shader-vg |
| raa-vg-raster | `crates/raa-vg/src/raster.rs`<br>`crates/raa-vg/src/shadow.rs`<br>`crates/raa-vg/src/wgsl/raster_common.wgsl`<br>`crates/raa-vg/src/wgsl/raster_hw.wgsl`<br>`crates/raa-vg/src/wgsl/raster_hw_fragment.wgsl`<br>`crates/raa-vg/src/wgsl/raster_sw.wgsl`<br>`crates/raa-vg/src/wgsl/raster_sw_body.wgsl`<br>`crates/raa-vg/src/wgsl/raster_sw_depth.wgsl`<br>`crates/raa-vg/src/wgsl/vis_write.wgsl`<br>`crates/raa-vg/src/wgsl/depth_stamp.wgsl`<br>`crates/raa-vg/src/wgsl/shadow_merge.wgsl`<br>`crates/raa-vg/tests/raster_variants.rs` | raa-vg-scene, raa-vg-cull, raa-shader-vg, raa-shadow-vg-seam |
| raa-vg-resolve | `crates/raa-vg/src/resolve.rs`<br>`crates/raa-vg/src/gbuffer.rs`<br>`crates/raa-vg/src/shade.rs`<br>`crates/raa-vg/src/feature.rs`<br>`crates/raa-vg/src/frame_passes.rs`<br>`crates/raa-vg/src/stats.rs`<br>`crates/raa-vg/src/wgsl/resolve.wgsl`<br>`crates/raa-vg/tests/resolve_variants.rs`<br>`crates/raa-vg/tests/feature.rs` | raa-vg-scene, raa-vg-cull, raa-vg-raster, raa-shader-vg, raa-pipeline-vg-order, raa-lit-vg-seam, raa-shadow-vg-seam |
| api-m29-vg | `crates/raa-js/src/api/m29_virtual_geometry.rs`<br>`crates/raa-js/tests/virtual_geometry.rs`<br>`packages/raa/types/m29-virtual-geometry.d.ts`<br>`docs/api/m29-virtual-geometry.md` | raa-js, api-m29-bake, raa-vg-resolve, raa-vg-scene, api-m9 |
| vg-pass-scenes | `examples/virtual-geometry/scenes/passes.js`<br>`examples/virtual-geometry/passes.test.mjs` | vg-page, vg-job, api-m29-vg, api-m29-bake, api-m28, api-m8, api-m9 |
| sdfm-vg-export | `sdfm:src/node/games/mesh-clusters.ts`<br>`sdfm:src/node/games/mesh-clusters.test.ts`<br>`sdfm:src/node/games/asset-mesh.ts`<br>`sdfm:src/node/procedural/export.ts` | mama-mesh-clusters, api-m29-bake, js-package |
| sdfm-vg-settings | `sdfm:src/core/virtual-geometry-settings.ts`<br>`sdfm:src/core/virtual-geometry-settings.test.ts`<br>`sdfm:src/components/shell/VirtualGeometrySetting.vue`<br>`sdfm:src/app/paramesh/game-virtual-geometry.ts`<br>`sdfm:src/app/paramesh/game-virtual-geometry.test.ts` | api-m29-vg, api-m29-bake, mama-mesh-clusters |
| sdfm-vg-evidence | `sdfm:src/core/virtual-geometry-evidence.ts`<br>`sdfm:src/core/virtual-geometry-evidence.test.ts`<br>`sdfm:src/app/gpu-page/scenarios/virtual-geometry-app.ts`<br>`sdfm:src/app/gpu-page/virtual-geometry-visibility.ts`<br>`sdfm:src/app/gpu-page/virtual-geometry-controls.test.ts`<br>`sdfm:src/node/evidence/topics/virtual-geometry.ts`<br>`sdfm:src/node/evidence/topics/virtual-geometry.test.ts` | sdfm-vg-export, sdfm-vg-settings, sdfm-game-light-sync, sdfm-sky-weather-settings, sdfm-sky-weather-evidence, sdfm-light-evidence, sdfm-editor-evidence, mama-mesh-clusters, api-m29-bake, api-m29-vg |
| page-job | `web/common/page-job.mjs`<br>`web/common/page-job.test.mjs`<br>`docs/page-job.md` | p2-managed-transport, sky-weather-page |
| sdfm-editor-evidence | `sdfm:src/node/evidence/editor-evidence.ts`<br>`sdfm:src/node/evidence/editor-evidence.test.ts` | sdfm-light-evidence |
| page-kit | `web/common/page-runner.js`<br>`web/common/page-runner.test.mjs`<br>`scripts/build-page.mjs`<br>`scripts/build-page.test.mjs` | p2-managed-transport, page-job, sky-weather-page |
| raa-shader-gi | `crates/raa-shader/src/wgsl/prelude.wgsl`<br>`crates/raa-shader/src/wgsl/gi_volume.wgsl`<br>`crates/raa-shader/tests/gi_volume.rs`<br>`crates/raa-shader/tests/variants.rs`<br>`crates/raa-shader/tests/compose.rs` | raa-shader-vg |
| raa-pipeline-gi | `crates/raa-pipeline/src/lighting.rs`<br>`crates/raa-pipeline/tests/resources.rs` | raa-shader-gi |
| raa-sky-gi-budget | `crates/raa-sky/src/env_pass.rs`<br>`crates/raa-sky/src/background.rs`<br>`crates/raa-sky/src/feature.rs` | raa-pipeline-gi |
| raa-group1-pins | `crates/raa-oit/tests/variants.rs`<br>`crates/raa-fur/tests/variants.rs`<br>`crates/raa-sky/tests/variants.rs` | raa-shader-gi, raa-pipeline-gi |
| raa-lit-gi-seam | `crates/raa-lit/src/meshes.rs`<br>`crates/raa-lit/tests/gi_access.rs` | raa-lit-vg-seam |
| raa-gi-core | `crates/raa-gi/Cargo.toml`<br>`crates/raa-gi/README.md`<br>`crates/raa-gi/build.rs`<br>`crates/raa-gi/src/lib.rs`<br>`crates/raa-gi/src/error.rs`<br>`crates/raa-gi/src/settings.rs`<br>`crates/raa-gi/src/clipmap.rs`<br>`crates/raa-gi/src/cpu.rs`<br>`crates/raa-gi/src/membership.rs`<br>`crates/raa-gi/src/stats.rs`<br>`crates/raa-gi/src/parts.rs`<br>`crates/raa-gi/src/feature.rs`<br>`crates/raa-gi/tests/settings.rs`<br>`crates/raa-gi/tests/clipmap.rs`<br>`crates/raa-gi/tests/cpu_volume.rs`<br>`crates/raa-gi/tests/membership.rs`<br>`crates/raa-gi/tests/fixtures/voxelize.json`<br>`crates/raa-gi/tests/fixtures/flood.json`<br>`crates/raa-gi/tests/fixtures/codecs.json`<br>`crates/raa-gi/tests/fixtures/clipmap.json`<br>`crates/raa-gi/tests/fixtures/trace.json` | raa-pipeline-gi, raa-shader-gi, raa-lit-gi-seam, raa-lit-vg-seam, raa-pipeline-vg-order, raa-vg-scene, raa-vg-cull, raa-gi-volume, raa-taa |
| raa-gi-volume | `crates/raa-gi/src/layouts.rs`<br>`crates/raa-gi/src/shaders.rs`<br>`crates/raa-gi/src/volume.rs`<br>`crates/raa-gi/src/forward_driver.rs`<br>`crates/raa-gi/src/debug.rs`<br>`crates/raa-gi/src/wgsl/voxelize_kernel.wgsl`<br>`crates/raa-gi/src/wgsl/voxelize_forward.wgsl`<br>`crates/raa-gi/src/wgsl/radiance_codec.wgsl`<br>`crates/raa-gi/src/wgsl/seed.wgsl`<br>`crates/raa-gi/src/wgsl/jump_flood.wgsl`<br>`crates/raa-gi/src/wgsl/volume_write.wgsl`<br>`crates/raa-gi/src/wgsl/voxel_light.wgsl`<br>`crates/raa-gi/src/wgsl/debug_view.wgsl`<br>`crates/raa-gi/tests/layouts.rs`<br>`crates/raa-gi/tests/variants.rs` | raa-gi-core, raa-shader-gi, raa-pipeline-gi, raa-lit-vg-seam, raa-lit-gi-seam |
| api-m30 | `crates/raa-js/src/api/m30_global_illumination.rs`<br>`crates/raa-js/tests/global_illumination.rs`<br>`packages/raa/types/m30-global-illumination.d.ts`<br>`docs/api/m30-global-illumination.md` | raa-js, raa-gi-core, api-m29-vg, api-m8 |
| gi-page | `examples/global-illumination/index.html`<br>`examples/global-illumination/main.js`<br>`examples/global-illumination/nmc.gpu.json`<br>`examples/global-illumination/lib/content.js`<br>`examples/global-illumination/lib/facts.js`<br>`examples/global-illumination/lib/inventory.js`<br>`examples/global-illumination/lib/runtime.js`<br>`examples/global-illumination/scenes/volume.js`<br>`examples/global-illumination/contract.test.mjs`<br>`examples/global-illumination/volume.test.mjs`<br>`scripts/build-global-illumination.sh`<br>`web/common/gi-job.mjs`<br>`web/common/gi-job.test.mjs` | page-kit, page-job, js-package, api-m30, vg-page, sky-weather-page, p2-managed-transport |
| raa-gi-screen | `crates/raa-gi/src/parts/screen.rs`<br>`crates/raa-gi/src/parts/screen/cpu.rs`<br>`crates/raa-gi/src/parts/screen/passes.rs`<br>`crates/raa-gi/src/parts/slots.rs`<br>`crates/raa-gi/src/wgsl/screen/depth_pyramid.wgsl`<br>`crates/raa-gi/src/wgsl/screen/probe_common.wgsl`<br>`crates/raa-gi/src/wgsl/screen/screen_trace.wgsl`<br>`crates/raa-gi/src/wgsl/screen/probe_trace.wgsl`<br>`crates/raa-gi/src/wgsl/screen/probe_project.wgsl`<br>`crates/raa-gi/src/wgsl/screen/probe_integrate.wgsl`<br>`crates/raa-gi/src/wgsl/slots/slot_indirect.wgsl`<br>`crates/raa-gi/src/wgsl/slots/slot_ao.wgsl`<br>`crates/raa-gi/src/wgsl/slots/slot_local_visibility.wgsl`<br>`crates/raa-gi/tests/screen_twins.rs`<br>`crates/raa-gi/tests/screen_variants.rs`<br>`crates/raa-gi/tests/fixtures/sh9.json`<br>`crates/raa-gi/tests/fixtures/probes.json`<br>`crates/raa-gi/tests/fixtures/temporal.json` | raa-gi-core, raa-gi-volume, raa-shader-gi, raa-pipeline-gi, raa-ibl-gi-seam |
| raa-ibl-gi-seam | `crates/raa-ibl/src/feature.rs`<br>`crates/raa-ibl/tests/diffuse_scale.rs` | raa-pipeline-gi |
| raa-ssr-gi-fallback | `crates/raa-ssr/src/wgsl/ssr_trace.wgsl`<br>`crates/raa-ssr/src/shaders.rs`<br>`crates/raa-ssr/tests/variants.rs` | raa-shader-gi, raa-pipeline-gi |
| gi-lighting-scenes | `examples/global-illumination/scenes/lighting.js`<br>`examples/global-illumination/lighting.test.mjs` | gi-page, page-kit, api-m30, api-m28, api-m8, api-m10, api-m11, api-m15, api-m14 |
| raa-gi-virtual | `crates/raa-gi/src/parts/virtual_driver.rs`<br>`crates/raa-gi/src/wgsl/voxelize_virtual.wgsl`<br>`crates/raa-gi/tests/virtual_variants.rs` | raa-gi-core, raa-gi-volume, raa-vg-resolve, raa-vg-scene, raa-vg-cull |
| gi-virtual-scenes | `examples/global-illumination/scenes/virtual.js`<br>`examples/global-illumination/virtual.test.mjs` | gi-page, page-kit, api-m30, api-m29-vg, api-m29-bake, vg-page |
| sdfm-gi-settings | `sdfm:src/core/global-illumination-settings.ts`<br>`sdfm:src/core/global-illumination-settings.test.ts`<br>`sdfm:src/components/shell/GlobalIlluminationSetting.vue`<br>`sdfm:src/app/paramesh/game-global-illumination.ts`<br>`sdfm:src/app/paramesh/game-global-illumination.test.ts` | api-m30, js-package |
| sdfm-gi-evidence | `sdfm:src/core/global-illumination-evidence.ts`<br>`sdfm:src/core/global-illumination-evidence.test.ts`<br>`sdfm:src/app/gpu-page/scenarios/global-illumination-app.ts`<br>`sdfm:src/app/gpu-page/global-illumination-visibility.ts`<br>`sdfm:src/app/gpu-page/global-illumination-controls.test.ts`<br>`sdfm:src/app/gpu-page/editor-capture.ts`<br>`sdfm:src/node/evidence/topics/global-illumination.ts`<br>`sdfm:src/node/evidence/topics/global-illumination.test.ts` | sdfm-gi-settings, sdfm-sky-weather-settings, sdfm-game-light-sync, sdfm-editor-evidence, sdfm-vg-evidence, sdfm-vg-export, sdfm-light-evidence, sdfm-sky-weather-evidence, mama-mesh-clusters, api-m30, api-m29-vg, api-m29-bake |
| raa-ssr-gi-budget | `crates/raa-ssr/src/resources.rs` | raa-pipeline-gi |
| raa-ao-once | `crates/raa-shader/src/wgsl/pass_forward.wgsl`<br>`crates/raa-shader/src/wgsl/pass_transparent.wgsl`<br>`crates/raa-shader/src/wgsl/pass_furshell.wgsl`<br>`crates/raa-particles/src/wgsl/particles_opaque.wgsl`<br>`crates/raa-matfn/src/wgsl/plush.wgsl`<br>`crates/raa-lit/src/wgsl/slots_default.wgsl`<br>`crates/raa-ibl/src/wgsl/ambient.wgsl`<br>`crates/raa-brdf/src/wgsl/ambient.wgsl`<br>`crates/raa-gtao/src/desc.rs`<br>`crates/raa-gtao/README.md`<br>`crates/raa-gtao/tests/desc.rs`<br>`crates/raa-shader/tests/ao_once.rs` | raa-shader-gi, raa-shader-vg, raa-gi-screen, api-m9, raa-pipeline-gi |
| raa-area-ltc | `crates/raa-shader/src/wgsl/local_lighting.wgsl`<br>`crates/raa-shader/tests/area_form_factor.rs`<br>`crates/raa-pipeline/src/brdf_atlas.rs`<br>`crates/raa-pipeline/src/ltc_tables.txt`<br>`crates/raa-pipeline/src/lib.rs`<br>`crates/raa-ibl/src/lut.rs`<br>`crates/raa-ibl/src/wgsl/brdf_lut.wgsl`<br>`crates/raa-ibl/src/wgsl/ibl_lib.wgsl`<br>`crates/raa-ibl/README.md`<br>`crates/raa-lit/src/light_math.rs`<br>`crates/raa-lit/tests/fixtures/local-light-math.txt`<br>`crates/raa-js/tests/local_lights.rs`<br>`scripts/gi-cornell-ratio.mjs`<br>`scripts/gi-cornell-ratio.test.mjs`<br>`crates/raa-tonemap/src/tonemap.wgsl`<br>`crates/raa-tonemap/src/curves.rs`<br>`crates/raa-tonemap/tests/curves.rs`<br>`crates/raa-tonemap/tests/variants.rs` | raa-shader-gi, raa-pipeline-gi, raa-gi-volume, raa-gi-screen, raa-ssr-gi-fallback, raa-ibl-gi-seam, api-m28, gi-lighting-scenes, gi-page |
| raa-area-shadow | `crates/raa-lit/src/area_shadow.rs`<br>`crates/raa-lit/tests/area_shadow.rs`<br>`crates/raa-lit/tests/area_shadow_frame.rs`<br>`crates/raa-lit/src/local_lights.rs`<br>`crates/raa-lit/src/select.rs`<br>`crates/raa-lit/tests/select.rs`<br>`crates/raa-shadow/tests/hero_layers.rs`<br>`crates/raa-shadow/Cargo.toml` | raa-lit-vg-seam, raa-area-ltc, raa-shadow-vg-seam |
| raa-area-shadow-shading | `crates/raa-shadow/src/wgsl/csm.wgsl`<br>`crates/raa-shadow/tests/pcss_shared.rs`<br>`crates/raa-shader/tests/shadow_pcss.rs`<br>`crates/raa-shader/tests/area_shadow_tiers.rs`<br>`crates/raa-gi/tests/fixtures/cone.json` | raa-shader-gi, raa-area-ltc, raa-gi-screen, raa-gi-core, raa-area-shadow |
| gi-area-shadow-scenes | `examples/global-illumination/scenes/area-shadows.js`<br>`examples/global-illumination/area-shadows.test.mjs` | gi-page, gi-lighting-scenes, api-m28 |
| raa-taa | `crates/raa-taa/Cargo.toml`<br>`crates/raa-taa/README.md`<br>`crates/raa-taa/src/lib.rs`<br>`crates/raa-taa/src/history.rs`<br>`crates/raa-taa/src/shaders.rs`<br>`crates/raa-taa/src/resources.rs`<br>`crates/raa-taa/src/feature.rs`<br>`crates/raa-taa/src/wgsl/reproject.wgsl`<br>`crates/raa-taa/src/wgsl/taa.wgsl`<br>`crates/raa-taa/tests/history.rs`<br>`crates/raa-taa/tests/reproject.rs`<br>`crates/raa-taa/tests/variants.rs`<br>`crates/raa-taa/tests/frame.rs` | raa-pipeline-vg-order, raa-lit-vg-seam |
| api-m31-taa | `crates/raa-js/src/api/m31_taa.rs`<br>`crates/raa-js/tests/taa.rs`<br>`docs/api/m31-taa.md` | raa-taa, api-m7, raa-lit-vg-seam, raa-js |
| gi-taa-scenes | `examples/global-illumination/scenes/temporal.js`<br>`examples/global-illumination/temporal.test.mjs` | gi-page, gi-lighting-scenes, page-kit, api-m31-taa |
| lookdev-taa-scene | `examples/lookdev/scenes/m31-taa.js`<br>`examples/lookdev/test/m31-taa.test.mjs` | lookdev-page, lookdev-tools, api-m31-taa |
| raa-motion-v2 | `crates/raa-pipeline/src/targets.rs`<br>`crates/raa-pipeline/tests/targets.rs`<br>`crates/raa-lit/src/passes.rs`<br>`crates/raa-shader/src/wgsl/pass_prepass.wgsl`<br>`crates/raa-lit/tests/motion.rs`<br>`crates/raa-lit/tests/fixtures/motion.txt` | raa-lit-vg-seam, raa-shader-vg, raa-pipeline-vg-order |
| raa-area-stochastic | `crates/raa-lit/src/area_stochastic.rs`<br>`crates/raa-lit/tests/area_stochastic.rs`<br>`crates/raa-lit/tests/area_stochastic_frame.rs` | raa-area-shadow, raa-pipeline-gi, raa-lit-vg-seam |
| raa-gi-area-stochastic | `crates/raa-gi/src/parts/area_shadow.rs`<br>`crates/raa-gi/src/parts/area_shadow/cpu.rs`<br>`crates/raa-gi/src/parts/area_shadow/passes.rs`<br>`crates/raa-gi/src/parts/area_shadow/resources.rs`<br>`crates/raa-gi/src/wgsl/area_shadow/trace.wgsl`<br>`crates/raa-gi/src/wgsl/area_shadow/temporal.wgsl`<br>`crates/raa-gi/src/wgsl/area_shadow/spatial.wgsl`<br>`crates/raa-gi/tests/area_shadow_twins.rs`<br>`crates/raa-gi/tests/area_shadow_variants.rs`<br>`crates/raa-gi/tests/area_shadow_frame.rs`<br>`crates/raa-gi/tests/fixtures/area_stochastic.json` | raa-gi-core, raa-gi-screen, raa-taa, raa-area-stochastic, raa-pipeline-gi, raa-shader-gi, raa-lit-vg-seam |
| gi-area-stochastic-scenes | `examples/global-illumination/scenes/area-stochastic.js`<br>`examples/global-illumination/area-stochastic.test.mjs` | gi-page, gi-lighting-scenes, gi-area-shadow-scenes, api-m28 |
| raa-gi-area-restir | `crates/raa-gi/src/parts/area_shadow/restir_cpu.rs`<br>`crates/raa-gi/src/parts/area_shadow/restir.rs`<br>`crates/raa-gi/src/parts/area_shadow/restir_resources.rs`<br>`crates/raa-gi/src/wgsl/area_shadow/restir_common.wgsl`<br>`crates/raa-gi/src/wgsl/area_shadow/restir_initial.wgsl`<br>`crates/raa-gi/src/wgsl/area_shadow/restir_temporal.wgsl`<br>`crates/raa-gi/src/wgsl/area_shadow/restir_spatial.wgsl`<br>`crates/raa-gi/tests/fixtures/area_restir.json` | raa-gi-area-stochastic, raa-area-stochastic, raa-taa, raa-pipeline-gi, raa-shader-gi, raa-area-ltc, raa-gi-screen, raa-gi-core |
| gi-area-restir-scenes | `examples/global-illumination/scenes/area-restir.js`<br>`examples/global-illumination/area-restir.test.mjs` | gi-area-stochastic-scenes, gi-lighting-scenes, gi-page, api-m28, page-job |
| raa-gi-ssgi | `crates/raa-gi/src/parts/ssgi.rs`<br>`crates/raa-gi/src/parts/ssgi/cpu.rs`<br>`crates/raa-gi/src/parts/ssgi/passes.rs`<br>`crates/raa-gi/src/parts/ssgi/resources.rs`<br>`crates/raa-gi/src/wgsl/ssgi/trace.wgsl`<br>`crates/raa-gi/src/wgsl/ssgi/resolve.wgsl`<br>`crates/raa-gi/tests/ssgi_twins.rs`<br>`crates/raa-gi/tests/ssgi_variants.rs`<br>`crates/raa-gi/tests/ssgi_frame.rs`<br>`crates/raa-gi/tests/fixtures/ssgi.json` | raa-gi-core, raa-gi-screen, raa-taa, raa-shader-gi, raa-motion-v2, raa-lit-vg-seam |
| gi-ssgi-scenes | `examples/global-illumination/scenes/ssgi.js`<br>`examples/global-illumination/ssgi.test.mjs` | gi-page, gi-lighting-scenes, gi-area-stochastic-scenes |
| gi-quality-scenes | `examples/global-illumination/scenes/gi-quality.js`<br>`examples/global-illumination/gi-quality.test.mjs` | gi-lighting-scenes, gi-area-stochastic-scenes, gi-page, api-m28, api-m30 |
| sdfm-game-lighting-settings | `sdfm:src/core/game-lighting-settings.ts`<br>`sdfm:src/core/game-lighting-settings.test.ts`<br>`sdfm:src/app/paramesh/game-lighting.ts`<br>`sdfm:src/app/paramesh/game-lighting.test.ts` | api-m28, api-m30, api-m31-taa, sdfm-sky-weather-settings |
| raa-exposure | `crates/raa-exposure/Cargo.toml`<br>`crates/raa-exposure/README.md`<br>`crates/raa-exposure/src/lib.rs`<br>`crates/raa-exposure/src/desc.rs`<br>`crates/raa-exposure/src/maths.rs`<br>`crates/raa-exposure/src/feature.rs`<br>`crates/raa-exposure/src/readback.rs`<br>`crates/raa-exposure/src/shaders.rs`<br>`crates/raa-exposure/src/wgsl/exposure_compute.wgsl`<br>`crates/raa-exposure/src/wgsl/exposure_apply.wgsl`<br>`crates/raa-exposure/tests/maths.rs`<br>`crates/raa-exposure/tests/desc.rs`<br>`crates/raa-exposure/tests/variants.rs` | — |
| api-m33 | `crates/raa-js/src/api/m33_auto_exposure.rs`<br>`crates/raa-js/tests/auto_exposure.rs`<br>`packages/raa/types/m33-auto-exposure.d.ts`<br>`docs/api/m33-auto-exposure.md` | raa-js, raa-exposure, api-m7 |
| post-page | `examples/post/index.html`<br>`examples/post/main.js`<br>`examples/post/nmc.gpu.json`<br>`examples/post/lib/inventory.js`<br>`examples/post/lib/runtime.js`<br>`examples/post/lib/content.js`<br>`examples/post/scenes/exposure.js`<br>`examples/post/contract.test.mjs`<br>`examples/post/exposure.test.mjs`<br>`scripts/build-post.sh`<br>`web/common/post-job.mjs`<br>`web/common/post-job.test.mjs`<br>`web/common/view-check.mjs`<br>`web/common/view-check.test.mjs` | page-job, p2-managed-transport, sky-weather-page, vg-page, js-package, api-m33, tooling |
| raa-sharpen | `crates/raa-sharpen/Cargo.toml`<br>`crates/raa-sharpen/README.md`<br>`crates/raa-sharpen/src/lib.rs`<br>`crates/raa-sharpen/src/desc.rs`<br>`crates/raa-sharpen/src/kernel.rs`<br>`crates/raa-sharpen/src/feature.rs`<br>`crates/raa-sharpen/src/shaders.rs`<br>`crates/raa-sharpen/src/wgsl/sharpen.wgsl`<br>`crates/raa-sharpen/tests/kernel.rs`<br>`crates/raa-sharpen/tests/variants.rs` | — |
| api-m34 | `crates/raa-js/src/api/m34_sharpening.rs`<br>`crates/raa-js/tests/sharpening.rs`<br>`packages/raa/types/m34-sharpening.d.ts`<br>`docs/api/m34-sharpening.md` | raa-js, raa-sharpen |
| raa-sdfm-post-options | `packages/raa-sdfm/src/render-options.js`<br>`packages/raa-sdfm/test/render-options.test.js` | api-m33, api-m34, raa-sdfm-clusters |
| post-sharpen-scenes | `examples/post/scenes/sharpen.js`<br>`examples/post/sharpen.test.mjs` | post-page, api-m34 |
| raa-atlas-mips | `crates/raa-lit/src/textures.rs`<br>`crates/raa-lit/src/mipgen.rs`<br>`crates/raa-lit/tests/atlas_mips.rs`<br>`crates/raa-lit/tests/gpu_tint_textures.rs`<br>`crates/raa-shader/src/wgsl/prelude.wgsl`<br>`crates/raa-shader/tests/texture_grad.rs`<br>`crates/raa-js/src/api/m22_textures.rs`<br>`crates/raa-js/tests/textures.rs`<br>`docs/api/m22-textures.md`<br>`packages/raa-sdfm/src/field-bake.js`<br>`packages/raa-sdfm/src/bake-scheduler.js`<br>`packages/raa-sdfm/test/field-bake.test.js`<br>`packages/raa-sdfm/test/bake-scheduler.test.js` | raa-lit-vg-seam, raa-shader-vg, raa-js, js-package |
| post-mips-scenes | `examples/post/scenes/mips.js`<br>`examples/post/mips.test.mjs` | post-page, raa-atlas-mips |
| sdfm-game-post-settings | `sdfm:src/core/game-post-settings.ts`<br>`sdfm:src/core/game-post-settings.test.ts` | sdfm-sky-weather-settings |
| sdfm-game-post | `sdfm:src/app/paramesh/game-post.ts`<br>`sdfm:src/app/paramesh/game-post.test.ts`<br>`sdfm:src/app/temporal-upscaling.ts` | sdfm-game-post-settings, raa-sdfm-post-options, api-m33, api-m34, sdfm-game-light-sync |
| sdfm-import-texture-mips | `sdfm:src/app/paramesh/import-textures.ts`<br>`sdfm:src/core/display/import-textures.test.ts` | raa-atlas-mips |
| sdfm-post-evidence | `sdfm:src/core/post-evidence.ts`<br>`sdfm:src/core/post-evidence.test.ts`<br>`sdfm:src/app/gpu-page/scenarios/post-app.ts`<br>`sdfm:src/node/evidence/topics/post.ts`<br>`sdfm:src/node/evidence/topics/post.test.ts` | sdfm-game-post, sdfm-game-post-settings, sdfm-import-texture-mips, sdfm-editor-evidence, sdfm-light-evidence, sdfm-game-light-sync |

## Module: workspace
Files: `Cargo.toml`, `Cargo.lock`, `.gitignore`.

- `[workspace]` resolver 2, member globs; `[workspace.package]` version 0.5.0, edition 2024, license MIT, rust-version 1.96.0.
- `[workspace.dependencies]`: the closed list (Conventions) and the v0 crates by path (raa-core, raa-web, raa-math, raa-geom, raa-material, raa-scene, raa-render, raa-alloc-count, raa-prims, raa-loader, raa-csg). Lit crates use path dependencies.
- `.gitignore` blocks target/, node_modules/, dist/, release/, out/, .nmc/, pkg/, *.log and binary assets (*.glb, *.gltf, *.png, *.jpg, *.avif, *.ktx2, *.bin).

## Module: tooling
Files: `scripts/check.sh`, `scripts/build-web.sh`, `scripts/check-no-assets.sh`, `scripts/hooks/pre-commit`, `scripts/gpu-capture.mjs`, `web/serve.mjs`, `web/common/nmc-gpu.js`, `web/common/run-loop.js`, `proofs/lib/proof.mjs`.
- `npm run typecheck:types` (`scripts/typecheck-types.mjs`, `tools/types-check/`) stages the package declarations as `build-package.sh` does and runs strict `tsc` with a consumer fixture of every v0 `loadRaa` option. Existing v0 options are declared in `index.d.ts`; new API in `types/mNN-*.d.ts`.
- `scripts/check.sh [--quick]`: `== <step>`, stops on failure: fmt, clippy (host, wasm32 lib), tests (not quick), `build-web.sh` per discovered demo, `check-no-assets.sh`, the SAFETY guard and the neutrality guard; with no member crates it skips cargo steps.
- `check-no-assets.sh [--staged]` refuses blocked extensions (glb, gltf, png, jpg, jpeg, avif, ktx2, bin, wasm, hdr, exr) and files > 1 MB; the pre-commit hook runs it (`git config core.hooksPath scripts/hooks`).
- `gpu-capture.mjs <name>` writes `proofs/results/<name>.json` (S6) and the capture, never faking a result. `serve.mjs` is GET-only without traversal. `nmc-gpu.js` exports `registerNmcGpu({ run })`; `run-loop.js` `runMeasured(demo, canvas, args)` returns the S6 fields.
- `proof.mjs`: `parseArgs`, `check`, `run`, `pool` (NMC_CPUS), `readJson`, `gzipSize`, `browserResultCheck`, `VALUES`.

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
Files: `crates/raa-js/{Cargo.toml, README.md, build.rs}`, `src/{lib, engine, convert, state, feature_map}.rs`, `tests/{convert, api_registry}.rs`; `src/api/*.rs` belong to their api modules.

The wasm-bindgen `Engine` (WebApp in a WebRuntime with `DeviceRequest { optional_features: WEBGPU_OPTIONAL, adapter_limits: true }`). Init creates `Pipeline`, `LitRenderer`, `TonemapPass`, runs `install_all`, installs `raa.pipeline`; CsgResolver and the 8 prims use strict kinds. Handles are never-reused u32; errors via `raa_web::error_to_js`. `finish_scene_invocation` forwards submissions to `VgHandle` and `GiHandle`.

**API v0** (unchanged): `create(canvas, options)` (maxInstances 16384, maxMaterials 256, powerPreference, msaa 1|4, toneMapper, exposureEv), `resize`, `render`, `stats`, `probeNonEmpty`, `dispose`; `addMaterial`, `setMaterial`; `primitiveMesh`, `loadPrimsAsset`, `removeMesh`, `spawnAsset`; `addNode`, `setTransform`, `removeNode`, `setTransforms` (10 floats per node); `addInstance` (instance group, tag = handle), `removeInstance`, `setVisible`; `setCamera`, `orbit`, `setOrbit`, `setLight`, `setAmbient`.

`EngineState` holds scene, lit, pipeline, tonemap, `FeatureMap` (type-keyed, `api.feature_missing`) and handle maps. `build.rs` registers `src/api/*.rs` by file name (`install_all`, `frame_all` before `lit.prepare`).

`m07_meshes`: `addMesh` (S13), `updateMesh`, `meshInfo`, `addMeshInstance`. `m07_pipeline`: `passes`, `setPassEnabled`, `setPassOrder` (`pipeline.*` codes), `setToneMapping`, `setBackground`, `capabilities`.

## Module: js-package
Files: `packages/raa/{package.json, index.js, sky-weather.js, index.d.ts, README.md, LICENSE}`, `packages/raa/test/{sky-weather,load-once}.test.mjs`, `scripts/build-package.sh`, `scripts/build-bricks.sh`.
- `@etcito/raa` 0.5.0, `type: module`. `loadRaa(options)` loads the wasm from `options.wasm`, else `options.wasmUrl`, else the packaged URL, with one shared instantiation. `RaaModule = { version, createEngine, bakeClusters, createClusterBake, validateClusters, clusterCutMesh }`. Re-exports `parseWeatherText` and `formatWeatherText`.
- `index.d.ts` is API v0; additions in `types/mNN-*.d.ts` merge into `Engine` and `RaaModule`.
- `build-package.sh` builds `packages/raa/pkg/`, `dist/raa-sizes.json`, `dist/package/raa/`, `dist/example-basic/` and `npm pack` into `release/`. `build-bricks.sh` stages `dist/bricks/{raa,raa-sdfm,raa-anim}`.

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
  - `output_overlay` (default false): display overlays follow all HDR passes and tonemap;
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
Files: `crates/raa-sky/**`, except `src/clouds/**` (raa-volumetric-weather) and `src/cloud_backend.rs` with `tests/backend_registry.rs` (raa-cloud-seam).

Physical and legacy sky, time of day, weather data and the sky passes (S23-S25, S28). Depends on raa-core, raa-math, raa-scene, raa-pipeline, raa-lit, raa-shader, raa-ibl.

- `desc.rs`: `CloudDesc { coverage, density, altitude_m (1500), scale_m (2500), wind_mps, color, silver_lining }`, `AtmosphereDesc { ground_albedo, mie_multiplier }`, `EnvUpdate`, `SkyModel`, `SkyPreset`, `SkyDesc { model, profile, zenith, horizon, ground, intensity, horizon_softness, sun_disc_angular_radius, sun_disc_intensity, clouds, env_update, atmosphere }`, `SkyDesc::preset(name)`, `validate` (`sky.invalid` naming the field), `SkyTransaction`, `LegacyCloudFields`.
- `time_of_day.rs`: `Place { latitude, north }`; `TimeOfDay { day: u16, hours: f64 }` with `validate`, `advance(hours)` (signed, wraps the year) and `sky(&place) -> CelestialSky { sun, moon, moon_lit, pole, equinox }`; `CelestialSky::celestial(right_ascension, declination)`; `TimeOfDaySettings { time, place, enabled, playing, speed }`; `TimeOfDayState { settings, current, sky, dominant }` with `DominantLight` Manual, Sun or Moon.
- `weather.rs` (presets in `weather/*.weather`): `Weather { coverage, cloud_type, base, thickness, density, cirrus, wind_speed, wind_from, evolution, rain }` with `names`, `by_name`, `parse`, `to_text`, `validate`, `lerp`, `wind_velocity`, `observed(&Observation)`; `Observation { cover_low, cover_mid, cover_high, wind_kmh, wind_from, code, precipitation }` is a pure conversion from a weather report (nonfinite values become 0, an unknown WMO code uses the cover-derived branch); `WeatherPreset`, `WeatherSettings`, `ResolvedWeatherState { settings, current, cycle_preset }`, `WeatherUpdate`, `WeatherCycle::{new(start, seed), current, advance(seconds)}` with `DWELL = (120, 300)` and `TRANSITION = 90`.
- `atmosphere.rs`, `shaders.rs`, `wgsl/`: `atmospheric_radius_km`, `PhysicalSkyGpu`, `AtmosphereResources` (LUTs, per-view sky-view, Milky Way bake, submission acknowledgement) and the WGSL module builders for the LUTs, the Milky Way bake, `raa_celestial_radiance`, and the physical and legacy env and background modules.
- `SKY_WGSL`, the legacy `raa_sky_radiance(dir, env, sun_dir, sun_color, time)`: gradient with horizon blend pow(1 - max(dir.y, 0), softness), HG glow (g 0.76), limb-darkened sun disc, and a ray-plane cloud layer (5-octave domain-warped fbm, 4-step Beer-powder light march, HG silver lining, horizon fade).
- `feature.rs`, `env_pass.rs`, `background.rs`, `state.rs`, `observations.rs`: `SkyFeature`, `SkyHandle`, the passes and the diagnostics records.

Tests: desc.rs, variants.rs, time_of_day.rs, weather.rs, lifecycle.rs.

## Module: api-m8
Files: `crates/raa-js/src/api/m08_sky_ibl.rs`, `crates/raa-js/tests/sky_weather.rs`, `packages/raa/types/m08-sky-ibl.d.ts`, `docs/api/m08-sky-ibl.md`.

Implements S24. `install` creates IblFeature, then SkyFeature, and stores the handles in the FeatureMap. `frame` observes the snapshot advanced by `EngineApp::update` and marks the sky dirty on light changes; it never advances the clock.
- `setSky` resolves the full descriptor and calls `set_transaction` once with the explicit preset and the mask of supplied cloud fields.
- `setWeather` parses the patch into a complete `WeatherUpdate` without mutation (an unknown preset rejects) and calls `set_weather_update` once; `weather()` generates `text` from the authored target at call time.
- `skyWeatherDiagnostics()` serializes `SkyDiagnostics` into fresh objects.

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
Files: `crates/raa-gtao/**` except raa-ao-once's files.

GTAO (S12 `ao.gtao`, order 40): `AoDesc { radius_m 0.8, intensity 0.6, power 1.0 }` (`ao.invalid`); Jimenez 2016 horizon search, two bilateral passes, `set_ao`/`clear_ao`; `Ao` slot `gtao` (10). It darkens only ambient, once (S11). `GtaoHandle`: `set`, `get`, `set_camera`.

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

`setShadows({ softness?, distance?, resolution? })`, `setAo({ radius?, intensity?, power? })` (0.8, 0.6, 1.0), `setContactShadows({ distance?, thickness?, intensity? })`; toggles through `setPassEnabled`.

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
Files: `packages/raa-sdfm/**` except raa-sdfm-clusters' files.

`@etcito/raa-sdfm` 0.1.0, plain ES modules, hand-written `index.d.ts`. MAMA is reached only through `src/mama-bridge.js`; the README's `## MAMA calls` table lists every MAMA call.
- `createSdfmLoader({ engine, mama, workerUrl?, workers?, lods? }) -> { validate, load, spawn, dispose }` (S15 codes, `SdfmError { code, path? }`).
- `loadProceduralScene` dispatches through `src/procedural.js` (mesh) or `src/procedural-trace.js`; `src/procedural-runtime.js` applies deltas; `createMamaLightOwner` owns lights (S16, S17).
- Meshing via MAMA `meshAssets`; materials via `mapMaterial` to an S11 `material_main` (`sdfm.material_unmappable` on failure). Tests: `node --test` on `packages/raa-sdfm/test/*.test.js`.

## Module: api-m28
Files: `crates/raa-js/src/api/m28_lights.rs`, `packages/raa/types/m28-lights.d.ts`, `docs/api/m28-lights.md`. Owner: landed M2. Consumes: raa-lit.

The existing filename-discovered API registration installs `light.tiles` at order 35 against the renderer-owned registry. The additive declarations and Engine methods implement S17 without overloading the sun. Existing parser/lifecycle tests are `crates/raa-js/tests/local_lights.rs` and `tests/api_registry.rs` in that crate; source pins distinguish property absence from explicit reset. The shipped root exports already include LocalLightDesc, LocalLightChanges and LightStats through package assembly; consumers import those owning declarations.

## Module: sdfm-light-authoring
Implements S18 in sdfm: the registered `pointLight` kind, the document-light projection, the add command, `PointLightSection.vue` and the generated schema and catalog. Tests: `src/core/scene-lights.test.ts`, `src/core/scene-lights-authoring.test.ts`, `src/app/gpu-page/scene-lights-controls.test.ts`.

## Module: sdfm-game-light-sync
Implements S19 (`game-light-sync.ts`, `game-display.ts`); GameDisplay also reports sky (S25), GI (S41) and game-lighting state and refusals and passes vertex colours to m30.

## Module: sdfm-light-evidence
Implements S20 and S21 in sdfm through the compiled-topic discovery of `src/node/evidence/run.ts` and the `scene-lights-app` viewport scenario, with the UI driver and the inspector readability check. `sdfm:.nmc/commands.json` is written by sdfm-sky-weather-evidence.

## Module: solstice-architecture-finalization
Files: `raa:ARCHITECTURE.md`, the single consolidated contract of the linked repositories.

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

## Module: raa-cel

`crates/raa-cel` owns the `cel.resolve` HDR pass at order 190 (after grading,
before the required final HDR pass, tonemap). It reads HDR, depth and normals,
quantizes colours in Oklab and composites protected geometric ink. CelPost is
112 bytes; a separate 128-byte camera uniform supplies inverse projection/view.
The priority-30 Direct slot uses material flag 128 and preserves layered BRDF
lighting for unflagged materials. ShadeCtx carries the material flags. EnvGpu's
formerly reserved lane at byte 192 holds toon enabled, softness and specular;
the remaining reserved bytes and total 256-byte ABI are unchanged.

The JS module m27 installs cel disabled. `setCel` enables it and selects linear,
zero-exposure tone mapping without display dither; `getCel` returns its controls.
See `docs/api/m27-cel.md` for the exact API and phase-one limitations.

## Module: p2-managed-transport
Files: `web/common/{gpu-job,lights-job}.mjs` and their tests.

`gpu-job.mjs` is the managed transport of the lights and sky-weather runners: `publishGpuPage(directory, { timeoutMs }) -> Promise<{ pageRef }>` and `runGpuPageOp(pageRef, op, input, { timeoutMs }) -> Promise<{ jobId, clientId, result }>`, which forwards the input unchanged and returns the complete `reply.result`. Work timeout is at most 300000 ms; queue availability waits `NMC_GPU_WAIT_S` (default 60 s) separately. Missing configuration, failed publication or job, timeout and malformed responses reject with attribution. `lights-job.mjs` keeps its exports on top of it.

## Module: sky-weather-page
Files: `examples/sky-weather/{index.html, main.js, nmc.gpu.json, lib/**, scenes/foundation.js, contract.test.mjs, page-attach.test.mjs}`, `scripts/build-sky-weather.sh`, `.nmc/commands.json`.

The S26 page and the RAA commands.
- `lib/inventory.js`: `validateInventory`, `validateRequiredFacts`. `lib/readiness.js`: `createReadiness(stage) -> { snapshot, pending, ready, fail }`.
- `lib/runtime.js`: `createOperationBudget(remainingMs, clock)` and `createSkyPageRunner({ canvas, status, load, scenes, lib }) -> { run, engine, dispose }`; concurrent runs reject. `main.js` validates the scenes of `scenes.json` and attaches `run` to `window.nmcGpu.ops`.
- `lib/capture.js`: `captureCurrent(canvas, status)` (base64 PNG with the displayed status or error). `lib/measure.js`: `collectFeatureMeasurement(render, settings, clock)`, `validateFeatureMeasurement`.
- `lib/evidence.mjs`: `containedPath`, `readJsonRecord`, `writeJsonRecord` (atomic), `captureInputs(root, preparationId)`, `assertCurrentInputs(root, receipt)`.
- `lib/scene.js`: `buildSkySample(engine, lib, camera)` (ground, matte reference, reflective sphere, distant forms) and `sampleSelection(camera)`. `scenes/foundation.js` captures `foundation-sky-ground-and-reflective-sphere.png`.

## Module: sky-atmosphere-scenes
Files: `examples/sky-weather/scenes/atmosphere.js`, `examples/sky-weather/atmosphere.test.mjs`.

The atmosphere stage of S26, with time, place and playback controls. `observe` reads the public getters and diagnostics and waits for the publication generation each capture needs. Multi-view and transition rows require the assertion `cloud-flags-current` (both cloud validity flags true on every view). Pair metadata drives the measurement collector.

## Module: raa-volumetric-weather
Files: `crates/raa-sky/src/clouds/**`.

The `m2` cloud backend of S25 (`clouds::create`): noise, coverage, volume, shadow, panorama and screen modules (`shaders.rs`), `CloudGpu` (192 B, `resources.rs`) and the march group 2 (0 `CloudGpu`, 1 `PhysicalSkyGpu`, 2-4 noise and weather textures, 5-6 LUTs, 7-8 samplers, 9 coverage CDF, 10 output; screen adds 11 history, 12 depth). Validity flags follow acknowledged submissions; history is view-local. Tests: `src/clouds/tests/`.

## Module: sky-weather-scenes
Files: `examples/sky-weather/scenes/weather.js`, `examples/sky-weather/weather.test.mjs`.

The weather stage of S26, discovered by the page like every scene; it reuses the atmosphere scene's camera and helpers.

## Module: sdfm-sky-weather-settings
Files (sdfm): render settings core, app and node routes, panels, `game-render-settings.ts`, raytrace display, schema and catalog docs, `tsconfig.node.json`. `resolveSkyWeatherSettings`, `authoredSkyWeather`, the Render settings panel (sky, time, weather, VG, GI, game lighting) and `applyGameRenderSettings` (sky, then VG, GI and game lighting; throws failures); `applySharedRenderSettings` never throws.

## Module: sdfm-sky-weather-evidence
Files (sdfm): `src/core/sky-weather-evidence.ts`, `src/app/gpu-page/scenarios/sky-weather-app.ts`, `src/app/gpu-page/sky-weather-visibility.ts`, `src/node/evidence/topics/sky-weather.ts`, their tests, `.nmc/commands.json`.

- Inventory: modes `authoring`, `reload`, `invalid`; entries extend the S26 shape with `heldConstant` and `producer`, scenario `sky-weather-app`. Images: `editor-time-weather-controls-beside-game.png`, `editor-changed-time-weather-reloaded.png`, `editor-invalid-weather-last-valid-state.png`.
- `SkyWeatherEditorFacts { format: 'sdfm.sky-weather.editor-facts', version: 1, mode, provenance, captures, observations (authored, persisted, `GameSkyWeatherState`, engine serial), controls (field bounds and value), playback, weatherText, invalid }`, checked by `validateSkyWeatherEditorFacts`.
- `assertSkyWeatherControlsReadable(host, fields, errorRequired)` checks bounds, clipping and occlusion of the panel, the Game canvas, each named field and the validation error immediately before capture. `SkyWeatherApp` drives the real panel (`openPanel`, `setField`, `typeField`, `observe`, `capture`, `reload`).
- `SkyWeatherEvidence`: `runtimeBuild('mama' | 'raa')`, `select`, `clean`, `render`, `fresh`, `judge` over schema-1 records of milestone `p2-editor`. Jobs use `runGpuJob` with a 300 s limit and keep the direct result. The render is resumable: only entries without a valid image for the current build render again.

## Module: raa-cloud-seam
Files: `crates/raa-sky/src/cloud_backend.rs`, `crates/raa-sky/tests/backend_registry.rs`.

The S25 seam: `CloudFrame`, `CloudViews`, `CloudInvalidation`, the `CloudRenderer` trait, `NeutralCloudRenderer` (1x1 neutral views, validity flags false), the backend registry and the passes `sky.cloud-lighting` and `sky.cloud-screen`. Public helpers: `cloud_backend_names(m2_source_present)`, `registered_cloud_backend_names()`, `M2_CLOUD_SOURCE_PRESENT`, `CLOUD_SCREEN_UNKNOWN_ALPHA`. `views` returns neutral views for an absent view; `prepare` and recording reject an invalid index with `sky.invalid`.

## Module: sky-weather-job-resume
Files: `web/common/sky-weather-job.mjs`, `web/common/sky-weather-job.test.mjs`.

The node runner of the S26 records: `check`, `select`, `clean`, `render`, `fresh`, `judge`, `parseArgs`, `main`, with `STAGES`, `ACTIONS`, `RECORD_FIELDS`, `readBuildReceipt`, `loadScenes`, `verifyPage`, `storeRoot`, `attemptRecords` and `buildKey`. `render` gives each page operation a 270000 ms allowance inside the 300 s job cap. Judge calls run up to `JUDGE_CONCURRENCY` = 4 at a time and start only within the budget (`--budget-s`, default 480 s).

## Module: raa-cluster
S29 in `crates/raa-cluster`.

## Module: api-m29-bake
The cluster half of S31 (`m29_cluster_bake.rs`).

## Module: mama-mesh-clusters
MAMA's S30 validator (`src/core/game/mesh-clusters.ts`).

## Module: raa-sdfm-clusters
The S30 reader of `@etcito/raa-sdfm`.

## Module: page-job
`createPageJob` (S26 actions, verdict reuse, `judgeReference`/`judgeAspect`, `<envPrefix>_SCENARIO`), configured by vg-job and gi-job.

## Module: vg-job
The S33 page-job configuration.

## Module: vg-page
The S33 page on page-kit (`build-page.mjs virtual-geometry --stages bake,passes`).

## Module: raa-shader-vg
S32 shader parts; also `GI_VOLUME_WGSL` and `VARIANT_FLAG_MOTION_DEFORM` exports.

## Module: raa-pipeline-vg-order
The S32 order pairs.

## Module: raa-lit-vg-seam
`VirtualSet` and `LitHandle` accessors (S32, S37), the TAA jitter consumer and `frame_buffer()`.

## Module: raa-shadow-vg-seam
`ShadowHandle::cascade_targets`.

## Module: raa-vg-scene
raa-vg settings, eligibility, scene pool and `INSTANCE_GI`.

## Module: raa-vg-cull
raa-vg culling and the occlusion pyramid.

## Module: raa-vg-raster
raa-vg rasterisers and cascade casters.

## Module: raa-vg-resolve
raa-vg resolve, gbuffer, shade, stats, `VgFeature`, `with_gi_frame`, `set_gi_members`.

## Module: api-m29-vg
The settings half of S31 (`m29_virtual_geometry.rs`).

## Module: vg-pass-scenes
The S33 `passes` scene.

## Module: sdfm-vg-export
The S34 export bake.

## Module: sdfm-vg-settings
The S34 setting helpers.

## Module: sdfm-vg-evidence
The S34 editor evidence.

## Module: sdfm-editor-evidence
`EditorEvidenceStore` and `exportEditorAssets`.

## Module: page-kit
`createPageRunner` and `scripts/build-page.mjs`.

## Module: raa-shader-gi
S10/S38 prelude and `gi_volume.wgsl`.

## Module: raa-pipeline-gi
S10 group-1 layouts and setters.

## Module: raa-sky-gi-budget
Sky base layout (S10).

## Module: raa-group1-pins
S10 variant pins.

## Module: raa-lit-gi-seam
STORAGE mesh buffers (S10).

## Module: raa-gi-core
S36, S37.

## Module: raa-gi-volume
S37, S38.

## Module: api-m30
S36.

## Module: gi-page
S40.

## Module: raa-gi-screen
S39.

## Module: raa-ibl-gi-seam
S39, S45.

## Module: raa-ssr-gi-fallback
S39, S45.

## Module: raa-ssr-gi-budget
SSR split group 0 (S10).

## Module: gi-lighting-scenes
S40 lighting.

## Module: raa-gi-virtual
S37 driver B.

## Module: gi-virtual-scenes
S40 virtual.

## Module: sdfm-gi-settings
S41.

## Module: sdfm-gi-evidence
S41.

## Module: raa-ao-once
S11 AO rule.

## Module: raa-area-ltc
S11, S44.

## Module: raa-area-shadow
S44.

## Module: raa-area-shadow-shading
S44.

## Module: gi-area-shadow-scenes
S40.

## Module: raa-taa
S43.

## Module: api-m31-taa
S43.

## Module: gi-taa-scenes
S40.

## Module: lookdev-taa-scene
S43.

## Module: raa-motion-v2
S43.

## Module: raa-area-stochastic
S44.

## Module: raa-gi-area-stochastic
S44.

## Module: gi-area-stochastic-scenes
S40.

## Module: raa-gi-area-restir
S44.

## Module: gi-area-restir-scenes
S40.

## Module: raa-gi-ssgi
S39.

## Module: gi-ssgi-scenes
S40.

## Module: gi-quality-scenes
S40.

## Module: sdfm-game-lighting-settings
S41.

## Module: raa-exposure
S46 in `crates/raa-exposure`.

## Module: api-m33
The m33 API of S46 (`m33_auto_exposure.rs`, `merge_auto_exposure`).

## Module: post-page
The S49 page, `scenes/exposure.js`, `post-job.mjs`, `view-check.mjs`.

## Module: raa-sharpen
S47 in `crates/raa-sharpen`.

## Module: api-m34
The m34 API of S47 (`m34_sharpening.rs`, `merge_sharpening`).

## Module: raa-sdfm-post-options
The S47 `applyRenderOptions` keys.

## Module: post-sharpen-scenes
The S49 `sharpen-detail` scene.

## Module: raa-atlas-mips
S48 textures, mipgen, prelude helpers and m22 options.

## Module: post-mips-scenes
The S49 `mips-ground` scene.

## Module: sdfm-game-post-settings
The S50 settings.

## Module: sdfm-game-post
S50 apply, exposure watch and temporal upscaling.

## Module: sdfm-import-texture-mips
The S50 imported-texture material.

## Module: sdfm-post-evidence
The S50 editor evidence.
