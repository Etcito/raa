# raa-liquid

Animated liquid surfaces of the RAA lit renderer. The crate owns the material functions
`raa.liquid.water` and `raa.liquid.slime` (Transmissive) and `raa.liquid.mud` (Opaque,
wet clearcoat), with Gerstner vertex stages, ripple rings from instance impacts, scrolling
detail normals, shore and crest foam, and refraction through the `Transmission` slot.
Depends on raa-core, raa-shader, raa-lit and raa-matfn; raa-transmission and raa-brdf are
dev-dependencies for the variants test of the built-in functions.

The liquid WGSL is the shared Gerstner vertex stage (`material_vertex`), the shared
fragment helpers (detail normals, foam) and each function's `material_main`.

## Public items

- `LIQUID_BUILTINS: &[raa_matfn::BuiltinFn]`: the three rows, in this order:

  | name | flags | class |
  |---|---|---|
  | `raa.liquid.water` | `FLAG_VERTEX_FN \| FLAG_TRANSMISSIVE` | Transparent |
  | `raa.liquid.slime` | `FLAG_VERTEX_FN \| FLAG_TRANSMISSIVE` | Transparent |
  | `raa.liquid.mud` | `FLAG_VERTEX_FN` | Opaque (wet clearcoat, voronoi bubble pops) |

  Mud is drawn with the opaque draws, before the frame's opaque copy, so it has crest and
  noise foam but no shore term.
- `install(lit: &mut LitRenderer) -> Result<(), RaaError>`: registers every row with
  `lit.fns_mut().register_builtin`. Atomic: every row is first registered on a clone of the
  registry, so a refusal registers nothing. Refusal: the registry's `ShaderError` as
  `RaaError::FrameGraph` with the same code and message (`material.fn_duplicate` when a
  liquid name is already registered, e.g. a second install). The transmissive liquids also
  need the `screen` Transmission provider and a Transparent handler (raa-transmission,
  raa-oit), which `install` does not add.
- `WAVE_COUNT` (4): number of waves packed in the material record's `p` array.
- `wave_height(params, x, z, t) -> f32`: sum of the rest-position heights
  `A sin(k (D.x x + D.y z - speed t))` of the four waves.

`params` is `[[f32; 4]; 8]`: `p[i]` for `i` in `0..4` is wave `i` as `(amplitude A in m,
wavelength L in m, direction angle theta in rad in XZ from +X towards +Z, steepness Q in
[0, 1])`; `p[4]` holds the four speeds in m/s; `p[5..8]` are function-specific and are not
read by the twin. A wave with `L <= 0`, `A == 0` or any non-finite field contributes 0, so
the result is never NaN.

The `wave_height` rustdoc also fixes the horizontal Gerstner displacement
`dxz = D Qi A cos(phase)` with `Qi = clamp(Q / (k A WAVE_COUNT), 0, 1)`, the displaced
position and the analytic normal from the GPU Gems 1 chapter 1 partial derivatives, so the
WGSL vertex stages and the CPU twin stay identical.

## Parameter packing

Every row packs its parameters in the material record's `p: array<vec4<f32>, 8>`
(`(index, component)`; colours take 3 components, floats 1, `wave*`/`speeds` 4):

| slot | water | slime | mud |
|---|---|---|---|
| (0..3, 0) | `wave0`..`wave3` (A, L, theta, Q) | same layout, 0.4x amplitude | same layout, 0.25x amplitude |
| (4, 0) | `speeds` (deep-water dispersion) | `speeds` x0.25 | `speeds` x0.08 |
| (5, 0) | `deepColor` [0.05, 0.45, 0.55] | `color` [0.42, 1.0, 0.12] | `color` [0.32, 0.22, 0.14] |
| (5, 3) | `clarity` 0.7 | `absorption` 5.0 | `wetness` 0.8 |
| (6, 0) | `shallowColor` [0.55, 0.92, 0.88] | `glossiness` 0.9 | `bubbleScale` 0.18 m |
| (6, 1) | (in `shallowColor`) | `bubbleScale` 0.05 | `bubbleRate` 0.6 pops/s per cell |
| (6, 3) | `foamWidth` 0.12 m | `foamWidth` 0.06 m | - |
| (7, 0) | `foamAmount` 0.6 | `foamAmount` 0.25 | `foamAmount` 0.15 |
| (7, 1) | `detailStrength` 0.15 | `detailStrength` 0.08 | `detailStrength` 0.25 |
| (7, 2) | `detailScale` 3.0 | `detailScale` 2.0 | `detailScale` 4.0 |

Ripple rings read up to 4 impacts from the instance params (`x, z, start time s,
strength`).

## Testing

```
cargo test -p raa-liquid
```

`tests/waves.rs` (CPU only) checks on a grid that the height is finite with degenerate waves
(`L = 0`, NaN amplitude and other non-finite fields), that one wave is periodic in time with
period `L / speed` and along its direction with period `L`, that four waves with commensurate
periods are periodic, and that the height is bounded by the sum of the amplitudes.

`tests/variants.rs` (Q11, CPU only) composes every liquid built-in for every `PassKind`
(Prepass, Forward, Shadow, Transparent, FurShell) at MSAA 1 and 4 with the row flags (plus
`FLAG_TRANSPARENT | FLAG_TRANSMISSIVE` for Transparent) over raa-lit's default slot
providers, `raa_brdf::providers()` and the `screen` Transmission provider, validates each
with `Composer::validate` and asserts the count (30 variants). The unit tests in
`src/builtins.rs` check the rows, and those in `src/feature.rs` check without a GPU that a
second registration refuses `material.fn_duplicate` and leaves the registry count unchanged
(`LitRenderer::new` needs a GPU and a pipeline, so the tests drive the registry part of
`install` directly).

```
cargo clippy -p raa-liquid --all-targets -- -D warnings
```

The crate must also stay clean for the web target:

```
cargo clippy -p raa-liquid --target wasm32-unknown-unknown -- -D warnings
```
