# raa-fog

Height fog and aerial perspective of the lit renderer (S12 `fog`, order 68). The description
packs into the S10 `EnvGpu` fog fields: `fog` = (density, height falloff, base height, start
distance), `fog_color` = (linear rgb colour, sun inscatter strength) and `aerial` = (strength,
distance scale, 0, 0). Depends on raa-core, raa-pipeline, raa-lit and raa-shader.

## Public items

- `FogDesc { density, height_falloff, base_height_m, start_m, color, inscatter,
  aerial_strength, aerial_distance_m }`: exponential height fog `density *
  exp(-height_falloff * (y - base_height_m))` from `start_m`, plus aerial perspective. `Default`
  is `clear`: density 0, so fog and aerial perspective are off.
- `FogDesc::preset(name)`: `clear`, `haze`, `snow`, `jungle`, `night`; `None` for any other
  name. Starting values (the look tuning refines them):

  | preset | density | falloff | base m | colour | inscatter | aerial |
  |---|---|---|---|---|---|---|
  | `clear` | 0 | 0.1 | 0 | white | 0 | off |
  | `haze` | 0.012 | 0.12 | 0 | warm white | 0.6 | 0.5 over 400 m |
  | `snow` | 0.02 | 0.08 | 0 | cold blue-white | 0.4 | 0.6 over 250 m |
  | `jungle` | 0.03 | 0.3 | 2 | green-grey humid | 0.35 | 0.4 over 300 m |
  | `night` | 0.015 | 0.1 | 0 | deep blue-violet | 0.2 | 0.3 over 500 m |

- `FogDesc::validate()`: refuses with code `post.invalid` and a message naming the field.
  Rules: every number finite; `density` in [0, 1]; `height_falloff` in (0, 10]; `start_m` >= 0;
  every `color` component >= 0; `inscatter` in [0, 4]; `aerial_strength` in [0, 2]
  (0 = off; above 1 the aerial fade saturates closer); `aerial_distance_m` > 0.
- `FogDesc::write_env(&mut raa_pipeline::EnvGpu)`: writes exactly the three S10 fog fields
  above (padding 0) and leaves every other `EnvGpu` field untouched.

Planned with the pass task (not exported yet): `FogFeature::install(gpu, pipeline, lit) ->
Result<FogHandle, RaaError>` (the `fog` FeaturePass, order 68, and the `height` Fog slot
provider, priority 10) and `FogHandle` (`set`, `get`).

## Testing

```sh
cargo test -p raa-fog
```

`tests/desc.rs` checks that every preset validates, that unknown names return `None`, that
every refusal carries `post.invalid` and names its field, and that `write_env` puts the values
at the S10 fields while leaving the other `EnvGpu` fields untouched.
