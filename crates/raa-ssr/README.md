# raa-ssr

Screen-space reflections for the lit pipeline (S12 `ssr`, order 65). This crate currently holds
the module contract's first public items; the feature pass (`SsrFeature`) and its handle
(`SsrHandle` with `set_camera`) come with the pass.

Depends on raa-core, raa-pipeline, raa-lit, raa-shader, raa-ibl and raa-math.

## Public items

- `SsrDesc { intensity, max_roughness, thickness_m }` (defaults 1, 0.6, 0.15) with `validate`:
  refuses with `post.invalid`, naming the field. Rules: every value finite; intensity in
  [0, 2]; max_roughness in [0, 1]; thickness_m in (0, 10].
- `hiz` (CPU twin of the GPU Hi-Z reduction, so the rule is testable without a GPU):
  - `mip_count(width, height) -> u32`: levels until 1x1, `floor(log2(max(w, h))) + 1`.
  - `next_size(width, height) -> (u32, u32)`: `max(1, w / 2) x max(1, h / 2)`.
  - `reduce_min(src, width, height, dst: &mut Vec<f32>) -> (u32, u32)`: each destination texel is
    the minimum of its 2x2 footprint. When a source dimension is odd, the last destination
    column or row also takes the extra source column or row, so no depth sample is lost. `dst`
    is cleared and reused by the caller.

The shader rule the WGSL reduction mirrors is documented in the `hiz` module doc.

## Tests

`cargo test -p raa-ssr`: `tests/hiz.rs` covers the mip count (1280x720 gives 11, 1x1 gives 1),
the odd-edge reduction (5x3 corner is a 3x3 footprint), the whole pyramid's last level equal to
the global minimum of random data, and every `SsrDesc` refusal with its field name.
