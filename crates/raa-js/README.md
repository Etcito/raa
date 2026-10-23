# raa-js

The wasm-bindgen engine of RAA (cdylib + rlib). API v0 stays valid: every v0
method keeps its JS name, arguments, defaults, results and error codes. The
`Engine` (wasm32 only) wraps a `WebRuntime` running the HDR lit pipeline.

## Lit pipeline

`EngineApp::init` builds a `raa_pipeline::Pipeline` with the `raa_lit::LitRenderer`
passes (`prepass`, `forward`: MSAA HDR) and the `raa_tonemap::TonemapPass`
(`tonemap`), runs the api files' `install`, then installs `raa.pipeline` in the
frame graph. The device is requested on a core adapter with the WEBGPU_OPTIONAL
features the adapter has and the adapter's limits.

- v0 mapping: `addMaterial`/`setMaterial` become `SurfaceMaterial::from_v0`;
  `primitiveMesh`/`loadPrimsAsset` meshes are uploaded as `AttributedMesh` with the
  S10 default attributes; an instance handle is an instance group (one scene
  instance per sub-mesh, tag = the JS handle). `addInstance(node, mesh, material)`
  draws every sub-mesh with `material`; `removeInstance` and `setVisible` act on the
  whole group; `stats` and `probeNonEmpty` keep their meaning.
- `EngineOptions` additions: `msaa` (1 or 4, default 4), `toneMapper` ('agx',
  'aces', 'linear', default 'agx'), `exposureEv` (finite, default 0); other values
  throw `api.invalid_argument`.
- Each `update` runs the api files' `frame`, rebuilds the instance groups of meshes
  replaced since the last frame (keeping node, visibility and tag), then
  `lit.prepare`.

## API files

`build.rs` lists `src/api/*.rs` sorted by file name and writes
`$OUT_DIR/api_registry.rs` (a `#[path]` module per file, `install_all` and
`frame_all` calling each file's `install` and `frame` in that order) and
`$OUT_DIR/api_files.txt`. Adding a file needs no registration edit. Each file
defines `install(ctx: &mut InstallCtx)` (run at init, after the M1 passes exist),
`frame(st, gpu, input)` (run in `update` before `lit.prepare`) and its JS methods in
`#[wasm_bindgen(js_class = Engine)] impl Engine` blocks using `Engine::with_state`.

- `m07_meshes.rs`:
  - `addMesh(desc: MeshDesc) -> handle` (S13: `positions`, `normals`, `indices`,
    optional `uvs`, `tangents`, `colors` as Float32Array linear or Uint8Array sRGB
    rgba8, `subMeshes: [{first, count, material?}]`), built with
    `raa_mesh::from_js_arrays`.
  - `updateMesh(id, desc)` replaces the geometry and sub-meshes of any mesh handle;
    its instance groups are rebuilt at the next frame.
  - `meshInfo(id) -> {vertices, triangles, subMeshes: [{first, count, material:
    handle | null}], bounds: {min, max}}`.
  - `addMeshInstance(node, mesh) -> handle`: each sub-mesh with its own material
    (the default material when it has none).
  - Refusals: the S13 `mesh.*` codes (kept), `api.unknown_handle` (sub-mesh
    material, mesh, node), `api.invalid_argument` (missing array, wrong array type).
- `m07_pipeline.rs`: `passes`, `setPassEnabled`, `setPassOrder`, `setToneMapping`,
  `setBackground`, `capabilities`.

## Public items

- `convert` (native-testable): `api_error`, mappers `geom_error`, `load_error`,
  `scene_error`, `material_error` (each keeps its crate's stable code),
  `HandleMap`, `MaterialDesc`, `CameraDesc`, `EngineOptions`, `engine_options`,
  `tone_mapper`, `decode_trs`, `check_trs_len`, `quality`.
- `Engine` v0 JS methods: `create`, `addMaterial`, `setMaterial`, `primitiveMesh`,
  `removeMesh` (every mesh handle), `loadPrimsAsset`, `spawnAsset`, `addNode`,
  `setTransform`, `setTransforms`, `removeNode`, `addInstance`, `removeInstance`,
  `setVisible`, `setCamera`, `orbit`, `setOrbit`, `setLight`, `setAmbient`,
  `resize`, `render`, `stats`, `probeNonEmpty`, `dispose`; m07 methods above.
- Error codes: `api.length_mismatch`, `api.unknown_handle`, `api.invalid_argument`,
  `api.unknown_param`, `api.disposed`, `api.feature_missing`, `pipeline.busy`,
  plus the codes of the underlying crates (`mesh.*`, `pipeline.*`, `render.*`, ...).

## Error carrier

`RaaError` has no generic variant, so non-core codes travel in
`RaaError::FrameGraph { code, message }`, whose `code()` returns the given code
verbatim; `raa_web::error_to_js` throws it as a JS `RaaError` with `code`.

## Test

`cargo test -p raa-js` (convert, feature map, instance groups, `api_registry`:
the build.rs output lists every api file and calls install/frame in order);
`cargo check --target wasm32-unknown-unknown -p raa-js` and wasm32 clippy for the
engine. The package build is `scripts/build-package.sh`.
