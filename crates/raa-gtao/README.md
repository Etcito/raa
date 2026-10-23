# raa-gtao

Ground-truth ambient occlusion (GTAO) for the lit renderer (S12 `ao.gtao`).

## Purpose

`raa-gtao` owns the `ao.gtao` FeaturePass (order 40, full resolution) and the `Ao` slot
provider `gtao`. The pass follows Jimenez 2016:

- 4 slice directions rotated by a 4x4 noise, 8 horizon steps per side over the projected
  radius, and a cosine-weighted visibility integral with a thickness heuristic, reading the
  prepass `depth` and `normal` targets.
- It writes `1 - intensity * (1 - vis^power)` into an R32Float texture, then runs 2 separable
  7-tap depth-aware bilateral passes (horizontal into R32Float, vertical into the final
  texture) with the weight `exp(-|dz| / (0.02 z))`.
- The final texture is Rgba16Float with the AO in `.r`, because R16Float is not a core WebGPU
  storage format. The `Ao` provider samples it with `textureLoad` of binding 7 `.r`; the
  multi-bounce term is applied in the ambient slot.
- Fixed values: radius 0.8 m, 4 slices x 8 steps per side, 2 denoise passes.

## Public items

- `AoDesc` (S12): `radius_m: f32` (0.8; `(0, 10]`), `intensity: f32` (1; `[0, 1]`),
  `power: f32` (1.5; `(0, 8]`); `Default`, and `validate` which refuses invalid values with
  `RaaError::FrameGraph { code: "ao.invalid" }` and a message naming the field.
- Shaders, `GtaoFeature` and `GtaoHandle` follow in the next task of this crate.

## Test

`cargo test -p raa-gtao`:

- `tests/desc.rs`: the default is valid, every field and bound is refused with `ao.invalid`
  and the field named in the message, and NaN/infinities are refused (Q12).
