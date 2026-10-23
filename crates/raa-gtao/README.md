# raa-gtao

Ground-truth ambient occlusion (GTAO) for the lit renderer (S12 `ao.gtao`).

## Purpose

`raa-gtao` owns the `ao.gtao` FeaturePass (order 40, full resolution) and the `Ao` slot
provider `gtao`. The pass follows Jimenez 2016:

- 4 slice directions rotated by a 4x4 noise, 8 horizon steps per side over the projected
  radius (at least one pixel apart, read at pixel centres), each sample faded toward the
  tangent-plane horizon as its distance reaches the radius (thickness heuristic), and the
  analytic cosine-weighted visibility between the two horizons, normalised by the unoccluded
  integral of the slice so an open flat surface is exactly 1; it reads the prepass `depth`
  and `normal` targets.
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
- `GtaoFeature::install(gpu, pipeline, lit) -> Result<GtaoHandle, RaaError>`: adds the `Ao`
  slot provider `gtao` (priority 10) to `lit`, builds the `gtao`, `denoise_h` and `denoise_v`
  compute pipelines (layouts derived from the modules) and their 160 B uniform, and adds the
  FeaturePass `ao.gtao` (order 40). Refusals, with nothing installed: `pipeline.busy` when the
  pipeline is borrowed, `pipeline.duplicate_pass` when `ao.gtao` is already installed, and the
  `add_slot` refusal with its code (`shader.slot_conflict` when another `Ao` provider holds
  priority 10).
- `GtaoHandle` (clones share the state): `set(desc)` validates (`ao.invalid`, unchanged on
  refusal), `get()`, and `set_camera(proj, inv_proj)`, called every frame with the scene
  camera's projection and its inverse.

## Pass and textures

`ao.gtao` (order 40, after `prepass`):

- resize: recreates three full-resolution textures, all STORAGE_BINDING | TEXTURE_BINDING:
  `raw` (R32Float, GTAO output), `blur` (R32Float, horizontal denoise) and `out` (Rgba16Float,
  vertical denoise, AO in `.r`), and the bind group of each dispatch over the `depth` and
  `normal` targets.
- prepare: without a camera or textures the lighting is reset to the dummy (`clear_ao`).
  Otherwise it writes `GtaoParams` (proj @0, inv_proj @64, viewport (w, h, 1/w, 1/h) @128,
  params (radius_m, intensity, power, frame index) @144) and gives `out` to the lighting
  (`set_ao`, S10 group 1 binding 7) when it does not hold it yet.
- record: one compute pass with three dispatches of ceil(w/8) x ceil(h/8) workgroups:
  `gtao` (depth, normal -> raw), `denoise_h` (raw -> blur), `denoise_v` (blur -> out).
- Disabling the pass calls `clear_ao`; enabling it gives `out` back at the next prepare.

Crate-private: `src/shaders.rs` (`GTAO_WGSL`, `DENOISE_WGSL`, `PARAMS_SIZE` = 160,
`ao_provider`) and `src/wgsl/` (`gtao.wgsl`, `denoise.wgsl`, `slot_ao.wgsl`).

## Test

`cargo test -p raa-gtao`:

- `tests/desc.rs`: the default is valid, every field and bound is refused with `ao.invalid`
  and the field named in the message, and NaN/infinities are refused (Q12).
- `tests/variants.rs` (CPU, naga): the compute modules validate and bind only group 0, the
  `GtaoParams` field order, `gtao` wins `Ao`, and the standard function composes and validates
  for every pass kind, MSAA 1/4 and the flag sets (Q11).
- `src/feature.rs`: the `GtaoParams` packing (viewport @128, params @144) and `set` keeping
  the description on refusal.
