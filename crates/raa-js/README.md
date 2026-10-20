# raa-js

Frozen wasm-bindgen API v0 of RAA (cdylib + rlib). The `Engine` (wasm32 only,
added in later tasks) wraps a `WebRuntime` with the forward renderer, the
primitive providers and the CSG resolver.

## Public items

- `convert` (native-testable): `api_error`, mappers `geom_error`, `load_error`,
  `scene_error`, `material_error` (each keeps its crate's stable code),
  `HandleMap`, `MaterialDesc`, `CameraDesc`, `decode_trs`, `check_trs_len`, `quality`.
- `Engine` JS methods: `create`, `addMaterial`, `setMaterial`, `primitiveMesh`,
  `removeMesh`, `loadPrimsAsset`, `spawnAsset`, `addNode`, `setTransform`,
  `setTransforms`, `removeNode`, `addInstance`, `removeInstance`, `setVisible`,
  `setCamera`, `orbit`, `setOrbit`, `setLight`, `setAmbient`, `resize`, `render`,
  `stats`, `probeNonEmpty`, `dispose`.
- Error codes: `api.length_mismatch`, `api.unknown_handle`, `api.invalid_argument`,
  `api.unknown_param`, `api.disposed`, plus the codes of the underlying crates.

## Error carrier

`RaaError` has no generic variant, so non-core codes travel in
`RaaError::FrameGraph { code, message }`, whose `code()` returns the given code verbatim.

## Test

`cargo test -p raa-js`. The package build is `scripts/build-package.sh` (M6).
