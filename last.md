# RAA cel handoff

Built from `07e269cd7b104eef2eb427d84e347e3fc0ad7f86` for `CellShading_V01`.
All source edits and generated artifacts are inside `/root/nmc/worktrees/direct/cel/raa`.
Reference checkouts were read only. No main branch was merged or pushed.

## Added and changed

New rendering crate `crates/raa-cel/`:

- `Cargo.toml`, `src/lib.rs`: crate dependencies and public exports.
- `src/desc.rs`: `CelSettings`, seven `Dither` modes, validation with `cel.invalid`, 112-byte `CelPost` packing.
- `src/feature.rs`: `CelFeature::install`, shared `CelHandle::get/set/set_camera`, HDR pass `cel.resolve`, order 190, ping-pong swap and resize bindings.
- `src/shaders.rs`, `src/wgsl/cel.wgsl`: fullscreen linear-HDR resolve and protected ink composition.
- `src/wgsl/color.wgsl`: Oklab lightness/hue/chroma quantization and chroma-bisection gamut mapping.
- `src/wgsl/dither.wgsl`: Bayer 2/4/8/16, IGN, white hash and a deterministic 16×16 void-and-cluster blue mask; optional world anchoring.
- `src/wgsl/outline.wgsl`: depth/normal ink, eight taps, two rings, fractional widths, tangent-plane depth rejection and 3×3 coverage feather.
- `src/provider.rs`, `src/wgsl/direct.wgsl`: priority-30 Direct provider, six warm/cool ramp rows, banded diffuse and shaped highlights. Only flag-128 materials opt in; other materials retain the layered BRDF.
- `src/palette.rs`, `src/color.rs`: pure palette/band and Oklab functions/tests.
- `tests/variants.rs`: naga resolve ABI validation, every built-in material and every PassKind at MSAA 1/4, plus actual opaque/blended particle modules.

Shader plumbing adds `ShadeCtx.material_flags` in `raa-shader` and passes it through forward, transparent and fur shading. `raa-pipeline::EnvGpu.cel` uses the formerly reserved lane at byte 192: toon enabled, softness, specular, unused. EnvGpu remains 256 bytes; its packing and fog preservation tests cover the new lane. Resolve camera inverses use a separate 128-byte uniform.

New `crates/raa-js/src/api/m27_cel.rs` installs the feature disabled and supplies the camera. The existing pipeline API supports the `cel` alias and per-view cel handles. The materials API accepts `cel` and restores `unlit` from the earlier commit. The environment API applies `post.cel`. Feature bit 23 identifies `cel.resolve`; existing bits stay unchanged.

New `packages/raa/types/m27-cel.d.ts` exports `CelDither`, `CelSettings` and `CelState`; `packages/raa/index.d.ts` re-exports them and declares the Engine methods. The material/environment declarations are extended. Package staging retains the exported type module in `types/`, including the npm archive and example package.

New `docs/api/m27-cel.md` documents the controls and limitations. `ARCHITECTURE.md` and the camera-effects documentation describe the integration. `packages/raa-sdfm/src/map-material.js`, its declarations and README accept the material opt-in and explain host-owned style; `loader.js` is unchanged.

Separate prerequisite commits repair baseline formatting/clippy findings, document an existing animation ABI borrow precondition, and remove pre-existing neutrality violations. These are required for the requested workspace gate and are separate from the feature commits.

## Exact JS API

```ts
engine.setCel(settings: CelSettings): void;
engine.getCel(): CelState; // Required<CelSettings>, fresh object
engine.setPassEnabled('cel', enabled: boolean): void;
// 'cel.resolve' is also accepted and is the name shown by passes().
```

All `CelSettings` fields are optional and merge over current settings:

| Name | Type and range | Default |
| --- | --- | --- |
| `bands` | integer, 2–32 | 10 |
| `hues` | integer, 4–32 | 16 |
| `chroma` | integer, 2–12 | 6 |
| `dither` | `'bayer2' \| 'bayer4' \| 'bayer8' \| 'bayer16' \| 'ign' \| 'white' \| 'blue'` | `'bayer4'` |
| `ditherAmount` | finite number, 0–1 | 0.65 |
| `outlineWidth` | finite number, 0.5–6 rendered pixels | 1 |
| `outlineSharpness` | finite number, 0–1 | 0.85 |
| `interiorEdges` | boolean | true |
| `edgeRetention` | finite number, 0–1 | 0.85 |
| `worldSpace` | boolean | true |
| `worldScale` | finite number, 1–512 cells/world unit | 48 |
| `softness` | finite number, 0–1 | 0.12 |
| `specular` | finite number, 0–2 | 0.7 |
| `toon` | boolean | true |
| `palette` | boolean | true |
| `outlines` | boolean | true |
| `ditherOutlines` | boolean | false |

`setCel` validates before changing state, enables the pass and sets tone mapping to `{mapper:'linear', exposureEv:0, look:'none', dither:false}`. The final display pass still encodes sRGB. Invalid values/types, nonfinite numbers, fractional integer fields and unknown threshold modes throw `cel.invalid` without changing controls, enable state or tone mapping.

`getCel` returns all 17 controls, not the enable state; inspect `passes()` for that. `setPassEnabled('cel', false)` disables resolve and toon lighting, retains controls and leaves tone mapping unchanged. Re-enable with `setCel({})` to select the neutral tone mapper again, or with `setPassEnabled` to keep the current tone mapper.

```js
engine.setCel({ bands: 10, hues: 16, chroma: 6, dither: 'bayer4' });
const material = engine.addSurfaceMaterial({ color: [0.6, 0.4, 0.2], cel: true });
engine.applyEnvironment({ post: { cel: { worldSpace: true, worldScale: 48 } } });
```

- `addSurfaceMaterial` and `setSurfaceMaterial` accept `cel?: boolean` (default false), which sets bit 128. The latter replaces the description; include the flag again when replacing a cel material.
- `unlit?: boolean` (default false) sets bit 64 and uses base plus emissive directly. It takes precedence over cel lighting; the final image resolve still applies.
- `toon:false` restores layered Direct lighting even for flagged materials. It does not disable resolve or outlines.
- `palette:false` selects scalar lightness bands instead of the Oklab hue/chroma palette; it does not disable banding.
- `ditherOutlines:false` protects ink after quantization; true includes ink in quantization.
- `applyEnvironment({post:{cel: settings}})` calls `setCel`. Within the post object, cel runs after grading/bloom/DOF/SSR/sun shafts and before `toneMapping`. An explicit `toneMapping` key can override neutrality.
- `mapMaterial(material, {cel:true})` returns `desc.cel:true`; the material's optional `cel:true` is also accepted. An explicit mapper option overrides the material value. The engine maps the boolean to the flag. The loader does not read a document style; the host calls `setCel` directly.

## Deliberately missing / practical limits

- Native-resolution phase only: internal detail factor is 1. Linear averaging, nearest depth hit and average/max ink coverage are retained, but no reduction/present target or pixel-size/sampling API is exposed. `edgeRetention` is inert at factor 1.
- No 480p pixel grid or nearest-neighbour enlargement yet; pixel-stable panning stays with the host.
- No point lights, material-id outlines, field curvature, CPU error diffusion/DBS or temporal dithering.
- Transparent surfaces and particles share the opaque prepass depth/normal footprint for resolve outlines; no separate transparent geometry target is added.
- No GPU-vs-CPU parity test, benchmark card or visual acceptance capture was added. New cel tests are device-free.

## Verification

- `cargo test -p raa-cel`: PASS, 5 unit tests and 3 shader/variant tests.
- `cargo test --workspace`: PASS, including the final rerun after neutrality cleanup.
- `bash scripts/check.sh --quick`: PASS: formatting, host and wasm32 clippy with `-D warnings`, demo wasm builds, asset guard, safety guard and neutrality guard. This script explicitly skips tests under `--quick`, so the workspace test command was run separately.
- `bash scripts/build-bricks.sh --out dist/bricks`: PASS after the green quick gate. `npm_config_cache` was set to `$PWD/target/npm-cache` because the default cache is read-only. `GIT_DIR` was set as below for correct build provenance.
- `node --test packages/raa-sdfm/test/map-material.test.js`: PASS, 22 tests. Additional direct mapper checks passed for true, material-carried true, and an explicit false override.
- TypeScript strict/noEmit checks of staged declarations and a complete cel usage sample: PASS. The final packaged `dist/bricks/raa/index.d.ts` also passes.
- Generated wasm declarations expose `setCel(settings: any): void` and `getCel(): any`; the hand-written types narrow those to `CelSettings` and `CelState`. Material/environment declarations and `types/m27-cel.d.ts` are present in the package.
- `git diff --check`: PASS.

Initial gate attempts identified and then resolved baseline formatting, lint, safety-comment and neutrality failures. The first package attempt could not write npm's default cache; rerunning with the in-worktree cache succeeded.

Artifacts: `dist/bricks/raa`, `dist/bricks/raa-sdfm`, `dist/bricks/raa-anim`, and `dist/bricks/bricks.json`. RAA wasm is 2,552,845 bytes raw / 858,430 gzip. Animation wasm is 406,564 raw / 156,267 gzip, below its existing 300,000-byte gzip limit.

## Git handoff

The worktree's `.git` file points into the shared read-only repository. Normal `git add` was refused while trying to create its index lock; no shared metadata was changed. To create the authorized commits entirely inside this worktree, a separate Git directory was initialized at `target/cel-git`, fetched the exact base from origin, and uses this directory as its worktree.

Use the following prefix when inspecting these commits locally:

```sh
export GIT_DIR="$PWD/target/cel-git"
git status
git log --oneline -5
```

The original shared worktree HEAD intentionally remains at `07e269c`; its ordinary status therefore still shows the source changes. The isolated Git directory contains the commits on `CellShading_V01`. Do not reset those source files using the original metadata. The branch is committed locally for owner review, with no main push or merge.

**Publication is blocked; the branch has not been pushed.** The exact requested command
`GIT_DIR="$PWD/target/cel-git" git push origin CellShading_V01` failed: the credential
prompt reported a WSL socket failure and could not read the GitHub username. The connected
GitHub publication fallback was also refused before writing: "MCP tool call requires approval,
but approval policy is never". No approval can be requested in this session.

A bundle of these commits is available at `dist/cel-handoff.bundle` (requires base `07e269c`).
From an authenticated session in this worktree, publish with:

```sh
GIT_DIR="$PWD/target/cel-git" git push origin CellShading_V01
```

The bundle preserves the branch independently of the ignored `target` directory. The build
manifest records implementation commit `79dec38`; the final handoff commit changes only this report.

Implementation commits:

- `16ec3a2` — baseline formatting and lint repairs.
- `c6335f2` — cel rendering crate and shader plumbing.
- `0131fd0` — JS API, material adapter, declarations and documentation.
- `79dec38` — baseline neutrality cleanup.

Every commit, including this handoff, ends with `Co-Authored-By: OpenAI Codex <noreply@openai.com>`.
