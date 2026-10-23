# raa-sky

Procedural sky of the lit renderer (S12 `sky.env`, `sky.background`): a zenith/horizon/ground
gradient with a sun disc and a procedural cloud layer, described in scene-referred linear HDR.
Depends on raa-core, raa-pipeline, raa-lit and raa-shader (raa-ibl joins with the env pass).

## Public items

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
- Crate-internal: `SkyDesc::write_env` fills the S10 `EnvGpu` sky and cloud fields (not the
  background mode, background colour, fog or aerial fields).

## Testing

```sh
cargo test -p raa-sky
```

`tests/desc.rs` checks every preset is valid, unknown presets, each refusal and that presets
differ; the unit test in `src/desc.rs` checks the `EnvGpu` field placement. No GPU needed.
