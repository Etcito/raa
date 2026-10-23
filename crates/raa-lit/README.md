# raa-lit

The HDR lit renderer of RAA (ARCHITECTURE S10, S11, S12).

## Public items (so far)

- `SurfaceModel` (`Flat`, `Standard`, `Function(MaterialFnId)`).
- `SurfaceMaterial`: the lit material record. `from_v0` maps a v0 `raa_material::Material`
  (Flat -> flat, PbrLite -> standard), `validate` refuses `material.invalid` naming the field,
  `write_gpu` appends the 256-byte S10 `SurfaceMaterialGpu`, `class` gives its `DrawClass`.
- `DrawClass` (`Opaque`, `Transparent`, `Fur`) and the flag consts `FLAG_TRANSPARENT`,
  `FLAG_TRANSMISSIVE`, `FLAG_FUR`, `FLAG_DOUBLE_SIDED`, `FLAG_NO_SHADOW`, `FLAG_VERTEX_FN`,
  `FLAG_UNLIT`; `SURFACE_MATERIAL_GPU_SIZE` (256).
- `SurfaceLibrary`: dense material ids with `add`, `set` (`material.unknown_id`), `get`, `len`,
  `is_empty`, `is_dirty`, `write_gpu_all`.

Crate-private: `default_slot_providers()` (`src/slots.rs`) returns one S11 slot provider per
`SlotId` (priority 0, provider `raa.default`), cut from the `// @slot <name>` sections of
`src/wgsl/slots_default.wgsl`: identity deform, shadow 1, AO 1, the S1 PbrLite ambient term,
S1 GGX + Lambert direct light, no transmission, identity fog.

## Testing

`cargo test -p raa-lit` (CPU only: `tests/material_gpu.rs` checks every GPU offset, the v0
mapping, each validation refusal and the class mapping; `tests/variants.rs` composes and
validates the default slots x the standard function x every pass kind x MSAA 1/4).
