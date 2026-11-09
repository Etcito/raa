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
- `IdPrep`, `SelectInput`, `SelectCapacity`, `LitDraw`, `MaterialClass` and
  `INSTANCE_GPU_SIZE` (128): the CPU id-list builder.
  - Opaque and fur draws are batched by (level mesh, material).
  - Transparent draws are one per instance, sorted back to front.
- `InstanceStore`, `StoreKind`, `write_instance_record` and `InstanceExtra`: the persistent
  128-byte instance and extras records by stable slot, with per-page uploads.
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

## Editor display

The optional `editor` output overlay (order 210) records no GPU commands while off.
`WireframeMode` and `LitHandle::{set_wireframe, clear_wireframe, set_outline,
set_mesh_edges}` control it. Surface contributors use `draw_surface_instances`
to respect edges-only overrides. See [editor features](../../docs/editor-features.md).

## M2-TILE-SELECTION

The CPU oracle `tile_selection::select_tile` is crate-visible in library tests; run
`cargo test -p raa-lit --lib tile_selection::tests`. Text fixtures specify the estimator
and exact rank before the separate GPU coverage, selection and order-35 recording shares.
The shared `raa_shader` constants are 16 pixels, 127 lights and 128 words, matching
solstice@7e465b5 `crates/solstice-render/shaders/lighting/light_tiles_common.wgsl:5,10,12`.
The nondeterministic source overflow algorithm is not used.

For each clipped 16 by 16 tile, ignore out-of-viewport samples, nonfinite depth and
depth outside `[0,1)`. Standard-Z nearest is the minimum remaining depth; farthest
is the maximum. An empty tile emits count zero, including for directionals. Reconstruct
the world point at the centre of the clipped pixel rectangle at nearest depth:
NDC is `(2*x/width-1, 1-2*y/height, depth)`; apply inverse jittered projection,
divide by homogeneous w, then inverse view. Bounds use the four inward normalized
tile-frustum side planes and near/far view-axis planes reconstructed at these depths.
A point or spot qualifies when its range sphere intersects all six half-spaces
(tangency included). An area uses radius `range + hypot(width,height)/2` about its
centre, covering every representative point. Directionals qualify on every covered tile.
Cone, emission-side and centre-point falloff never exclude conservative candidates.

At the reconstructed point, score is luminance of linear colour times intensity,
`0.2126*R + 0.7152*G + 0.0722*B`, times S11 `windowed_falloff(distance, range)`.
Spots additionally multiply by `spot_factor(dot(travelDirection,directionFromLight),
cosOuter,cosInner)`. Areas measure distance from S11 `representative_point` and multiply
by `area_light_emits` (0 or 1). Directionals use luminance alone. No surface-normal,
visibility or shadow term enters ranking. Clamp scores to `[0,f32::MAX]`, saturating
positive overflow and mapping undefined arithmetic to zero; every score is finite.

Retain qualified zero-score candidates too. A fixed 256-wide bitonic sorting network
orders eligible entries first, descending numeric score then ascending stable handle;
signed zeros tie. Padding has separate eligibility, never a sentinel score/handle.
GPU indices already follow handle order in `LocalLights::write_gpu`, including after
removal. Emit exactly `min(qualifying_count,127)` entries, zeroing unused CPU slots.
`droppedLightReferences` is `qualifying_count-min(qualifying_count,127)`, summed over
tiles and submitted views of the invocation, counting references rather than objects.
The CPU fixture passes no mutable registry state and never compares with GPU output.
