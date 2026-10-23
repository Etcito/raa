# raa-fur

Fur shells of the lit renderer (S12 `fur.shells`, FUR values): every draw of a FUR material
is drawn by the renderer as its opaque base, then again by this crate as N stacked shells with
alpha-to-coverage. Depends on raa-core, raa-shader, raa-lit and raa-pipeline.

## Public items

- `FurContributor::install(gpu, lit: &mut LitRenderer) -> Result<(), RaaError>`: creates the
  shell uniform buffer, its group-2 layout and bind group, adds the forward contributor
  `fur.shells` (order 500) and registers it as the `DrawClass::Fur` handler (so FUR materials
  are no longer refused with `material.blend_unavailable`). Refusals (nothing registered):
  `pipeline.duplicate_pass` (installed twice), `pipeline.busy`.
- `FurContributor` (`ForwardContributor`):
  - `prepare`: for every draw of `lit.draws(DrawClass::Fur)`, the pipeline from
    `LitHandle::pipeline_for` with `VariantKey { material fn of the draw, PassKind::FurShell,
    lit.msaa(), record flags & (DOUBLE_SIDED | VERTEX_FN | UNLIT) }` and the shell count.
    A draw whose material record is missing is refused with `material.unknown_id`.
  - `record`: per Fur draw, `bind_mesh`, then for shell i in 0..N group 2 at offset
    i x stride and `draw_indexed` over the sub-mesh with the draw's instance range.
- `shell_count(fur_w) -> u32`: N = clamp(round(fur.w), 1, 64); `FUR_SHELLS_DEFAULT` = 32 (the
  FUR value), `FUR_SHELLS_MAX` = 64.
- `shell_slot_stride(min_uniform_offset_alignment) -> u64` and
  `write_shell_slots(stride, out)`: the bytes of the shell uniform buffer (64 slots, slot i
  holds `index = i`, `SHELL_UNIFORM_SIZE` = 16 B per `RaaFurShellGpu`).
- `FUR_NAME` (`fur.shells`), `FUR_ORDER` (500).

## Pipeline state

One HDR_FORMAT colour target without blending, Depth32Float compare Less with depth write,
no culling, alpha-to-coverage when the forward pass has more than 1 sample, group 2 =
`@group(2) @binding(0) var<uniform> raa_fur_shell: RaaFurShellGpu` of
`raa_shader::PASS_FURSHELL_WGSL` (uniform, dynamic offset, 16 B binding, vertex and fragment).
The buffer, layout and bind group are created once at install; `prepare` keeps its per-draw
list's capacity, so a warm frame allocates nothing.

What the shells look like is not computed here: the FurShell pass shader of raa-shader
supplies the shell offset along the normal (length x (i + 1) / N), the gravity droop, the
strand mask, the 1-sample discard, root-to-tip darkening and the shadow and ambient slots;
the Kajiya-Kay highlights come from the `raa.plush` material function.

## Testing

`cargo test -p raa-fur` (CPU only): shell-count clamp, slot stride, uniform slot bytes and the
pipeline spec.
