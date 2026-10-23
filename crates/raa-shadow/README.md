# raa-shadow

Cascaded soft sun shadows (CSM + PCSS), the S12 `shadow.csm` pass (order 20).

## Public items

- `ShadowDesc { softness, distance_m, resolution }` with `Default` (1, 60, 2048) and
  `validate()`, which refuses with `shadow.invalid` naming the field.

Cascade splits and matrices, `ShadowFeature` and `ShadowHandle` are added by later tasks.

## Test

`cargo test -p raa-shadow`
