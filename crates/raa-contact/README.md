# raa-contact

Screen-space contact shadows of the lit renderer (S12 `shadow.contact`, order 45, CONTACT
values: 24 steps, max distance 0.35 m, thickness 0.05 m). A compute pass reconstructs each
pixel's view position from the prepass depth, marches toward the sun with a per-pixel
interleaved-gradient-noise jitter, and marks the pixel occluded when the depth buffer lies in
front of the ray by less than the thickness. The occlusion fades over the last 25% of the
distance and within 5% of the screen edges. The result is written to an Rgba16Float texture
(value in .r, 1 = lit) that the lighting reads at S10 group 1 binding 8. Depends on raa-core,
raa-math, raa-pipeline, raa-lit and raa-shader.

## Public items

- `ContactDesc { distance_m, thickness_m, intensity }` (`Default`: 0.35, 0.05, 1).
- `ContactDesc::validate()`: refuses with code `shadow.invalid` and a message naming the field
  when `distance_m` is not finite or not in (0, 5], `thickness_m` not finite or not in (0, 1],
  or `intensity` not finite or not in [0, 1].

Crate-private: `src/shaders.rs` (`CONTACT_WGSL`, `PARAMS_SIZE` = 176) and
`src/wgsl/contact.wgsl`, the standalone compute module (entry `contact`, 8x8, group 0:
0 depth, 1 `ContactParams { proj, inv_proj, viewport, sun_view, params }`, 2 storage output).
The feature (`ContactFeature`, `ContactHandle`) is added by a later task.

## Testing

```sh
cargo test -p raa-contact
```

Unit tests in `src/desc.rs` cover the defaults and each refusal. `tests/variants.rs` (CPU,
naga) validates the compute module, checks it binds only group 0, has exactly one compute entry
`contact`, does not reference `raa_frame` and declares the `ContactParams` fields in the
contract order with the size `PARAMS_SIZE`.
