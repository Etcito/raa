# raa-lit

The HDR lit renderer of RAA (ARCHITECTURE S10, S11, S12): surface materials, GPU meshes,
the per-frame draw lists and the two required feature passes of the lit pipeline,
`prepass` and `forward`, which draw opaque and fur geometry into the `raa-pipeline` targets.

## Public items

- `LitRenderer`: owns everything below. `new(gpu, &pipeline, LitOptions)` registers the
  `prepass` (order 30) and `forward` (order 60) passes in the pipeline (refusals
  `pipeline.duplicate_pass` with nothing added, `pipeline.busy`). It also offers:
  - meshes: `upload`, `update`, `remove`, `submesh_ids`, `info`;
  - materials and functions: `materials(_mut)`, `fns(_mut)`, `register_user_fn`,
    `composer`, `add_slot`;
  - draw classes: `register_class_handler`, `class_handled`;
  - instance extras: `set_instance_extra`, `instance_extra`, `clear_instance_extra`;
  - `add_contributor`, `set_background`, `handle`;
  - `prepare(gpu, scene, list, tags) -> FrameCounts`.
- `LitOptions` (`max_instances`, `max_materials`, `msaa` 1 or 4) and `FrameCounts`
  (instances, draw calls, triangles).
- `LitHandle`: what passes and contributors share. It offers `draws(class)`,
  `bind_mesh` (vertex slots 0..2 and the index buffer of a sub-mesh), `submesh_range`,
  `instance_buffer`, `frame_layout`, `frame_bind_group`, `msaa`, `material_record`, and
  `pipeline_for(gpu, VariantKey, &PipelineSpec)`, which caches one render pipeline per
  shader variant.
- `PipelineSpec`: colour targets, depth state, cull mode, alpha-to-coverage and the
  optional group-2 layout of a pipeline built by `pipeline_for`.
- `ForwardContributor`: an extra draw recorded inside the forward render pass (`name`,
  `order`, `prepare`, `record`), toggled through the pipeline like a pass.
- `SurfaceModel` (`Flat`, `Standard`, `Function(MaterialFnId)`).
- `SurfaceMaterial`: the lit material record.
  - `from_v0` maps a v0 `raa_material::Material` (Flat -> flat, PbrLite -> standard).
  - `validate` refuses `material.invalid`, naming the field.
  - `write_gpu` appends the 256-byte S10 `SurfaceMaterialGpu`.
  - `class` gives its `DrawClass`.
- `DrawClass` (`Opaque`, `Transparent`, `Fur`), the flag consts (`FLAG_TRANSPARENT`,
  `FLAG_TRANSMISSIVE`, `FLAG_FUR`, `FLAG_DOUBLE_SIDED`, `FLAG_NO_SHADOW`,
  `FLAG_VERTEX_FN`, `FLAG_UNLIT`) and `SURFACE_MATERIAL_GPU_SIZE` (256).
- `SurfaceLibrary`: dense material ids with `add`, `set` (`material.unknown_id`), `get`,
  `len`, `is_empty`, `is_dirty`, `write_gpu_all`. Material 0 of a renderer is the grey
  standard default.
- `InstanceExtra` (`INSTANCE_EXTRA_GPU_SIZE` 128): per-instance extra data keyed by tag.
- `write_frame_gpu`, `FrameParams`, `FRAME_GPU_SIZE` (512) and
  `SUN_ANGULAR_RADIUS_DEFAULT`: the S10 `FrameGpu` writer.
- `DrawPrep`, `PrepInput`, `PrepCapacity`, `LitDraw`, `MaterialClass` and
  `INSTANCE_GPU_SIZE` (128): the CPU draw-list builder.
  - Opaque and fur draws are batched by (mesh, material).
  - Transparent draws are one per instance, sorted back to front.
- `LitMeshId`, `LitMeshInfo`, and the re-exports `MeshId`, `MaterialId`,
  `AttributedMesh`, `SubMesh`, `MaterialFnId`.

Crate-private: `default_slot_providers()` (`src/slots.rs`) returns one S11 slot provider
per `SlotId` (priority 0, provider `raa.default`), cut from the `// @slot <name>` sections
of `src/wgsl/slots_default.wgsl`:
- identity deform;
- shadow 1 and AO 1;
- the S1 PbrLite ambient term;
- S1 GGX + Lambert direct light;
- no transmission;
- identity fog.

## Data flow

1. `LitRenderer::prepare` updates the scene and builds its render list, then the draw
   lists, instance and extras bytes, and `FrameGpu`. It writes them to the GPU queue.
2. The pipeline creates or resizes its targets, then prepares each pass. `forward` (re)creates
   its own colour and depth attachments on resize:
   - MSAA HDR_FORMAT colour and MSAA Depth32Float with `msaa` samples;
   - at `msaa` 1, a single-sample depth and no colour texture.
   - It then prepares its enabled contributors.
3. `prepass` (`src/passes.rs`, `PassKind::Prepass`) clears and writes the targets `depth`
   (Depth32Float, `Less`, cleared to 1.0), `normal` and `material`. It draws the Opaque
   and Fur draws single sample. `raa-tonemap` reads this depth for the S12 background rule.
4. `forward` (`PassKind::Forward`) clears colour to the `EnvGpu` background, alpha 1.
   - It draws the same classes: back faces culled unless `FLAG_DOUBLE_SIDED`, one pipeline
     per material function variant, bind groups 0 (frame) and 1 (lighting).
   - It records the enabled contributors by order.
   - The draws resolve into `hdr.dst()`; at `msaa` 1 they render into it directly.
   - The HDR chain swaps, so the next pass (for example tonemap) reads the result from
     `hdr.src()`.

Every draw is `draw_indexed(0..index_count, 0, first_instance..first_instance + count)`
over the sub-mesh range bound by `LitHandle::bind_mesh`.

The pass is named `prepass`, as ARCHITECTURE S12 and `raa_pipeline::REQUIRED_ORDER`
(`prepass`, `forward`, `tonemap`) require.

## Testing

`cargo test -p raa-lit` runs:
- CPU tests:
  - `tests/material_gpu.rs`: every GPU offset, the v0 mapping, each validation refusal and
    the class mapping.
  - `tests/prep.rs`: batching, the transparent sort and its ties, extras, refusals.
  - `tests/extras.rs`: the `InstanceExtra` layout.
  - unit tests of the pass specs, names and orders.
- `tests/variants.rs`: composes and naga-validates the default slots x the standard
  function x every pass kind x MSAA 1/4.
- `tests/gpu_lit.rs` (software adapter, prints `SKIP` without one): uploads a raa_geom
  `DebugCube` and renders it through `LitRenderer` and `Pipeline` (no tonemap) at MSAA 4
  and 1.
  - It reads back `hdr.src()`: the cube shows, the corners keep the background, and no
    validation error occurs.
  - It also checks the pass registration and its duplicate refusal, and that enabled
    contributors are recorded by order.
