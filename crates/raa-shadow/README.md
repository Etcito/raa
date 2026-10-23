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
- `ShadowFeature::install(gpu, pipeline, lit) -> Result<ShadowHandle, RaaError>`: adds the
  `csm` `Shadow` slot provider (priority 10) to `lit` and the FeaturePass `shadow.csm`
  (order 20, not required) to `pipeline`. Refusals, with nothing installed: `pipeline.busy`
  when the pipeline is borrowed, `shader.slot_conflict` when installed twice, and
  `pipeline.duplicate_pass`.
- The `shadow.csm` pass: creates the 4-layer Depth32Float caster array lazily at the
  description's resolution (recreated when it changes), computes the practical splits and
  their view-projections from the view set by `set_view`, renders the Opaque and Fur casters
  into the layers (slope bias 2.0 + constant bias 2) and sets the array and its 320 B
  `ShadowGpu` uniform on the lighting (`set_shadow`; `params` = softness, 1.5 normal-offset
  texels, 1/resolution, enabled 1; `params2` = 0, 10% blend band, 0, 0; the shader takes the
  sun angular radius from `FrameGpu.sun_dir.w`). Without a view, or while disabled, it calls
  `clear_shadow`.
- `ShadowHandle` (clone shares the state): `set(desc)` validates (`shadow.invalid`, unchanged
  on refusal), `get()`, and `set_view(view, fov_y_rad, aspect, near, far, sun_dir)`, called
  every frame with the scene camera and the sun's world travel direction.

## Test

`cargo test -p raa-shadow`
