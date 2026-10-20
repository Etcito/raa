# raa-render

Forward renderer as a frame-graph pass. It turns a `raa_scene::Scene` into instanced,
depth-tested, lit draws through `raa_core::FrameGraph`. Depends on raa-core, raa-math,
raa-geom, raa-material and raa-scene (plus wgpu).

## Public items

- `RendererOptions { max_instances, max_materials }`: capacities preallocated once
  (defaults 16384 and 256).
- `FrameCounts { instances, draw_calls, triangles }`: what one frame submitted.
- `DrawRange { mesh, first_instance, instance_count }`: one instanced draw.
- `FramePrep`: the CPU half of a frame, with no GPU objects (raa-bench measures it on its own).
  `new(max_instances)` preallocates `max_instances x 128 B`; `prepare_cpu(scene, list,
  surface_is_srgb)` writes the S1 instance bytes, the 144 B FrameUniform and the draws
  without allocating; `instance_bytes()`, `frame_uniform_bytes()`, `draws()`.
  `INSTANCE_STRIDE` (128) and `FRAME_UNIFORM_SIZE` (144) are exported.
- `Renderer`:
  - `new(gpu, surface_format, surface_is_srgb, opts)` builds the pipeline, the uniform,
    instance and material buffers and the bind group. The material library is seeded with
    one default `Material`, so `MaterialId(0)` is always valid.
  - `install(&self, graph)` creates the `depth` target (Depth32Float, `TargetSize::Surface`)
    and appends the `forward` pass. It returns the depth id.
  - `upload_mesh(gpu, mesh)` returns a dense `MeshId` and reuses removed slots.
  - `remove_mesh(id)`.
  - `mesh_triangles(id)`.
  - `materials()` and `materials_mut()`: a dirty library is uploaded at the next `prepare`.
  - `prepare(gpu, scene, list)` runs `update_world`, then `build_render_list`, then
    `prepare_cpu`. It counts triangles, writes the instances, the uniform and the dirty
    materials, and returns `FrameCounts`.
- `ForwardPass`: `impl RenderPass`, created by `install`. It shares the renderer state
  through `Rc<RefCell<..>>` because WebGPU objects are single-threaded, so there is no global
  state. It writes BACKBUFFER and `depth`. On first write it clears colour to `CLEAR_COLOR`
  and depth to 1.0. It issues one `draw_indexed` per `DrawRange`, with back-face culling and
  depth compare `Less`.
- `FORWARD_WGSL`: the complete shader source.

## FORWARD_WGSL composition

`FORWARD_WGSL = concat!(raa-material/src/shading.wgsl, src/forward.wgsl)`. The first part is
`raa_material::WGSL_SHADING`. It provides `struct MaterialGpu`, `struct FrameUniform`,
`raa_shade` and `raa_encode_srgb`. `forward.wgsl` adds the bindings:

- `@group(0) @binding(0)`: the uniform `FrameUniform`.
- `@group(0) @binding(1)`: the read-only storage `array<MaterialGpu>`.

It also adds `vs_main` and `fs_main`. The fragment shader encodes linear to sRGB itself when
`frame.flags.x == 1`, which is the case when `surface_is_srgb` is false.

## S1 layouts used

| buffer | layout |
|---|---|
| vertex (slot 0, stride 24 B) | `position f32x3` @0 (loc 0), `normal f32x3` @12 (loc 1); u32 triangle-list indices |
| instance (slot 1, step Instance, stride 128 B) | `model` 4 x f32x4 @0 (loc 2-5), `normal_matrix` 3 x f32x4 @64 (loc 6-8), `material_index u32` @112 (loc 9), 12 B padding |
| MaterialGpu (storage, 48 B each) | `base_color vec4` @0, `emissive vec4` @16, `params vec4` @32 = (roughness, metallic, model, 0) |
| FrameUniform (uniform, 144 B) | `view_proj mat4` @0, `camera_pos` @64, `light_dir` @80, `light_color` @96, `ambient` @112, `flags vec4<u32>` @128 |

All bytes are little-endian and written with `to_le_bytes`.

## Error codes

Every refusal is a `RaaError::Resource`, whose `code()` is `core.resource`. Its message starts
with the render code:

| code | when |
|---|---|
| `render.invalid_mesh` | `upload_mesh` of a mesh failing `MeshData::validate` (the reason follows) |
| `render.unknown_mesh` | `prepare` draws a `MeshId` that is not uploaded (or was removed) |
| `render.too_many_instances` | the render list has more items than `max_instances` (nothing is truncated) |
| `render.too_many_materials` | the library holds more materials than `max_materials` |

When `prepare` refuses, it writes nothing to the GPU.

## How to test

```sh
cargo test -p raa-render                                  # CPU tests + GPU tests
cargo test -p raa-render --test prep                      # FramePrep byte layout, refusal, determinism
cargo test -p raa-render --test gpu_forward -- --nocapture
```

`tests/gpu_forward.rs` renders one DebugCube (half 0.5, rotated 30 degrees about Y) into a
128x128 Rgba8Unorm `OffscreenTarget` and reads the pixels back. It asserts:

- `non_empty` is true;
- the centre pixel is not the clear colour;
- the corner pixel is the clear colour, within 2;
- the counts are 1 instance, 1 draw call and 12 triangles.

It also checks that an unknown mesh gives `render.unknown_mesh`. Each test prints the adapter
and whether it is `hardware` or `software` to stderr.

The GPU tests call `raa_core::testing::software_gpu()`. When no adapter exists, they print
`SKIP <test name>: no software adapter (<reason>)` to stderr and return. A missing adapter
never makes them fail. The GPU objects are leaked with `std::mem::forget` at the end of each
test, because Mesa's D3D12 GL adapter on WSL crashes at teardown.
