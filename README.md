# RAA: Render Anything Anywhere

RAA is a WebGPU rendering engine written in Rust and compiled to WebAssembly. It loads primitive assets and turns their primitive descriptions into polygons during loading, so asset files can describe geometry without shipping baked meshes.

Status: M6 — public API v0 frozen. Part of the Etcito open-source game stack (GameSaloon, gamesaloon-rollback, Royal Fiesta).

## Crates

- `raa-alloc-count`: counting allocator for test, benchmark, and opted-in demo binaries.
- `raa-bench`: frame preparation and allocation benchmarks.
- `raa-core`: GPU context, resources, and frame graph.
- `raa-csg`: mesh Boolean operations used by asset loading.
- `raa-geom`: mesh data, geometry providers, registry, and analysis.
- `raa-js`: frozen wasm-bindgen `Engine` API.
- `raa-loader`: prims v0 parser, loader registry, asset construction, and Boolean delegation.
- `raa-material`: material descriptions and shared shading code.
- `raa-math`: engine vector, quaternion, and matrix types.
- `raa-prims`: the eight built-in primitive geometry providers.
- `raa-render`: forward renderer and CPU frame preparation.
- `raa-scene`: scene graph, camera, lights, instances, and render list.
- `raa-web`: browser runtime, frame statistics, and JavaScript error conversion.
- `raa-ext-test` (`tests/raa-ext-test`): integration coverage for geometry and render-pass extensions.

## JavaScript package and example

The [`@etcito/raa` package](packages/raa/README.md) exposes the frozen v0 JavaScript API. The basic example in [`examples/basic`](examples/basic/) loads a primitive asset and renders primitive instances using the package.

## Build and check

```sh
scripts/check.sh [--quick]
cargo test --workspace
```

Regenerate a demo (for example, `m3-prims`) and capture it with the GPU client:

```sh
scripts/build-web.sh demos/<name>
node scripts/gpu-capture.mjs <name>
```

Regenerate the package and basic example, then capture the example for M6:

```sh
bash scripts/build-package.sh
node scripts/gpu-capture.mjs example-basic --input examples/basic/gpu-input.json --milestone m6
```

Run a milestone proof with the bundled Node runtime:

```sh
node proofs/mN-*.mjs
```

## Releases

Each release attaches a built npm tarball, with the wasm included, to its GitHub Release. Consumers install it by URL and need no Rust toolchain:

```sh
npm install https://github.com/Etcito/raa/releases/download/v0.1.0/etcito-raa-0.1.0.tgz
```

The asset URL pattern is `https://github.com/Etcito/raa/releases/download/v<version>/etcito-raa-<version>.tgz`. The same release carries `SHA256SUMS`; check the tarball with `sha256sum -c SHA256SUMS`.

To cut a release (maintainers):

1. Bump the version in `packages/raa/package.json`, `VERSION` in `packages/raa/index.js` and `[workspace.package] version` in `Cargo.toml`, then commit.
2. Optionally dry-run the build locally: `bash scripts/build-package.sh --tag v<version>`. It needs `wasm-bindgen` at the version pinned in `Cargo.lock`, and it refuses a tag that differs from the package version. The output is `release/etcito-raa-<version>.tgz` and `release/SHA256SUMS`.
3. Tag and push: `git tag v<version> && git push origin v<version>`. The workflow in `.github/workflows/release.yml` runs the checks, builds the package and creates the GitHub Release with both files.

## Git hook

Install the repository pre-commit asset check:

```sh
git config core.hooksPath scripts/hooks
```

## Documentation and licence

- [Public API v0](docs/API.md)
- [Architecture notes for contributors](docs/ARCHITECTURE-notes.md)
- MIT licence: [LICENSE](LICENSE)
