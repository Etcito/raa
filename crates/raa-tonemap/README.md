# raa-tonemap

Display transform of the lit renderer (S12 `tonemap`): scene-referred linear HDR in, display
image out. It is the only place where exposure, tone mapping, the sRGB OETF and dither are
applied. Depends on raa-core, raa-pipeline and raa-shader (the tests validate the shader with
`raa_shader::Composer::validate`).

## Public items

- `ToneMapper { Agx, Aces, Linear }`, `Look { None, Punchy, Golden }` (AgX only),
  `ToneMapping { mapper, exposure_ev, look, dither }` (`Default` = TONEMAP_DEFAULT: agx, look
  none, 0 EV, dither on). `ToneMapping::validate` refuses a non-finite `exposure_ev` with
  `post.invalid`.
- `TonemapControl`: shared handle (`get`, `set`); `set` validates and keeps the old settings on
  refusal. The pass reads it each frame in `prepare`.
- `TonemapPass::new(gpu, &Pipeline) -> (TonemapPass, TonemapControl)`: a `FeaturePass` named
  `tonemap`, order 200, required. Add it with `Pipeline::add`.
- `TONEMAP_WGSL`: the shader (`vs_main` fullscreen triangle, `fs_main`).
- `curves`: CPU ports of the same curves (`agx`, `aces`, `linear`, `tonemap`, `srgb_oetf`).

## What the pass draws

A fullscreen triangle into the backbuffer, reading `hdr.src()`, the depth target and the
`EnvGpu` uniform (copied from `LightingResources::env()` in `prepare` into the pass's own bind
group 0, so it does not depend on group 1):

1. Background rule (S12): where depth == 1 and `EnvGpu.sky_params.z` (background mode) is 0,
   the output is `EnvGpu.background` unchanged: no exposure, curve, OETF or dither. An empty
   scene therefore equals CLEAR_COLOR.
2. Otherwise `colour = hdr * 2^ev`, then AgX (Sobotka base with the inset/outset matrices,
   log2 encoding over [-12.47393, 4.026069] EV, 6th-order sigmoid, punchy/golden looks), ACES
   (Hill's fitted RRT+ODT) or Linear (clamp).
3. The sRGB OETF unless the surface format is *Srgb (the hardware encodes then).
4. Dither of +-0.5/255 with an 8x8 Bayer matrix when enabled.

Render pipelines are cached per backbuffer format; the bind group is rebuilt only when the HDR
source or depth texture changes (resize, different chain parity).

## Tests

`cargo test -p raa-tonemap`

- `tests/curves.rs`: CPU curve properties (monotonic, 0 -> 0, mid grey, white saturation,
  agx != aces).
- `tests/variants.rs`: `TONEMAP_WGSL` validates (CPU, no GPU needed). GPU tests (SKIP without a
  software adapter) run Pipeline + TonemapPass on an offscreen Rgba8Unorm target: an empty
  frame equals the background colour, the shader agrees with the CPU curves for every mapper
  and look, dither stays within one level, and a sky background is tone mapped.
