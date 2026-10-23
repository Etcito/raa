# raa-shadow

Cascaded soft sun shadows (CSM + PCSS), the S12 `shadow.csm` pass (order 20).

## Public items

- `ShadowDesc { softness, distance_m, resolution }` with `Default` (1, 60, 2048) and
  `validate()`, which refuses with `shadow.invalid` naming the field.
- `cascade_splits(near, far, lambda, n) -> [f32; 4]`: view-space far distance per cascade by
  the practical scheme (n clamped to 1..=4, unused entries and the last used entry = far).
- `cascade_matrix(view, fov_y_rad, aspect, split, sun_dir, resolution) -> Mat4`: cascade
  view-projection from the bounding sphere of the frustum slice (radius rounded up to 1/16 m),
  a light view along `sun_dir`, the centre snapped to whole texels and an orthographic
  projection with clip z [0, 1]; a singular view gives `Mat4::IDENTITY`.

`ShadowFeature` and `ShadowHandle` are added by a later task.

## Test

`cargo test -p raa-shadow`
