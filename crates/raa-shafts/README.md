# raa-shafts

Volumetric sun shafts for the lit pipeline (S12 `sunshafts`, order 69, after `fog` at 68).

Public items:
- `ShaftsDesc { intensity (1), density (0.02), max_distance_m (60) }` with `Default` and `validate` (`post.invalid`, message names the field): all finite; intensity in [0, 8]; density in [0, 1]; max_distance_m in (0, 1000].

Later in this milestone: `ShaftsFeature` (`install(gpu, pipeline) -> ShaftsHandle`, half-resolution 32-step ray march of the CSM shadow map and height fog, HG phase g 0.7, bilateral upsample added into `hdr.dst`; no-op when `shadow.csm` is disabled) and `ShaftsHandle` (`set`, `get`, `set_camera`, `set_sun`).

Test: `cargo test -p raa-shafts`.
