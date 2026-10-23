# raa-sky

Procedural sky of the lit renderer (S12 `sky.env`, `sky.background`): a zenith/horizon/ground
gradient with a sun disc and a procedural cloud layer, described in scene-referred linear HDR.
The sky renders an env cube that raa-ibl filters into image-based lighting, and is drawn per
pixel behind the opaque geometry. Depends on raa-core, raa-pipeline, raa-lit, raa-shader and
raa-ibl.

## Public items

- `SkyFeature::install(gpu, pipeline, lit, ibl) -> Result<SkyHandle, RaaError>`: creates the
  shared sky state (preset `day`), adds the FeaturePass `sky.env` to the pipeline and the
  forward contributor `sky.background` through `LitRenderer::add_contributor`. Refusals:
  `pipeline.busy` (pipeline borrowed), `pipeline.duplicate_pass` (already installed).
- `SkyHandle` (`Clone`, clones share one state):
  - `set(d)`: validates (`sky.invalid`, nothing changed on a refusal), stores the description
    and marks the env cube and the `EnvGpu` sky fields dirty; `pipeline.busy` during a frame.
  - `get()`: the stored description.
  - `mark_dirty()`: the sun changed, re-render and re-filter the env cube.
- `SkyDesc { zenith, horizon, ground, intensity, horizon_softness, sun_disc_angular_radius,
  sun_disc_intensity, clouds, env_update }`: the sky description (linear rgb colours).
- `CloudDesc { coverage, density, altitude_m, scale_m, wind_mps, color, silver_lining }`
  (`Default`: altitude 1500 m, scale 2500 m).
- `EnvUpdate { OnChange, EveryFrame }` (`Default` = `OnChange`): when the env cube is
  re-rendered.
- `SkyDesc::preset(name)`: `day`, `snow`, `jungle`, `night`; `None` for any other name.
- `SkyDesc::validate()`: refuses with code `sky.invalid` and a message naming the field
  (e.g. `clouds.coverage`). Rules: every number finite; colours >= 0; intensity >= 0;
  horizon_softness > 0; sun disc radius in [0, 0.2] rad; sun disc intensity >= 0; coverage in
  [0, 1]; density >= 0; altitude > 0; scale > 0; silver lining in [0, 1].
- `SKY_WGSL`: defines `raa_sky_radiance(dir, env, sun_dir, sun_color, time)` (gradient,
  Henyey-Greenstein sun glow, limb-darkened sun disc, ray-plane cloud layer with domain-warped
  fbm, 4-step Beer-powder light march, silver lining, horizon fade), shared by both passes.

## Passes and contributors

| name | role | order | what |
|---|---|---|---|
| `sky.env` | FeaturePass, not required | 10 | writes the `EnvGpu` sky fields when changed, drives `IblHandle::apply_pending`, renders the SKY_ENV cube (256 px, Rgba16Float, 6 mips) and calls `IblHandle::filter`, when the sky changed or every frame (`EnvUpdate::EveryFrame`) |
| `sky.background` | forward contributor | 900 | fullscreen triangle at depth 1 (LessEqual, no depth write), per-pixel `raa_sky_radiance` |

Disabling `sky.env` clears the env (flat ambient) and sets the constant background mode;
enabling it again rebuilds the env.

## Background-mode rule

`EnvGpu.sky_params.z` is the background mode (0 constant, 1 sky). `sky.env` sets it to 0 in
each prepare; `sky.background`, which only prepares while enabled, sets it back to 1 later in
the same frame. So with the contributor disabled (or the sky not installed) tonemap writes the
display-referred background colour unchanged where depth is 1 (S12).

## Testing

```sh
cargo test -p raa-sky
```

`tests/desc.rs` checks every preset is valid, unknown presets, each refusal and that presets
differ; `tests/variants.rs` composes and validates the sky, env and background shaders (naga,
CPU). Unit tests: `src/desc.rs` (`EnvGpu` field placement), `src/state.rs` (default state),
`src/feature.rs` (`SkyHandle` set/get/refusal/mark_dirty over a device-free state),
`src/env_pass.rs` and `src/background.rs` (S12 names and orders). The GPU test in
`src/env_pass.rs` records frames through Pipeline + LitRenderer + IblFeature and checks the
SH9 arrives without a validation error; it prints `SKIP` without a software adapter.
