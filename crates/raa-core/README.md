# raa-core

Device, queue and surface, resource handles, a minimal frame graph and the error type.
Every other RAA crate builds on it. It compiles for the host and for
`wasm32-unknown-unknown`: canvas items are wasm32-only, and headless/offscreen/testing
items are native-only.

## Public items

- `RaaError`: the crate's one error type (`Display` + `std::error::Error`), with
  `code()` (stable `<area>.<snake_case>` code: `core.no_adapter`, `core.request_device`,
  `core.surface`, the `framegraph.*` code, `core.resource`, `core.gpu`, `core.unsupported`)
  and `message()`.
- `AdapterSummary` (vendor, architecture, device, description, backend, is_fallback) with
  `is_hardware()`; `GpuOptions` (default: HighPerformance, allow_software true).
- `GpuContext`: `device()`, `queue()`, `adapter()`, `summary()`; native
  `GpuContext::headless(&opts)`; wasm32 `GpuContext::for_canvas(canvas, &opts)`.
- wasm32 only: `CanvasSurface` (`format`, `is_srgb`, `size`, `resize`, `acquire`) and
  `SurfaceFrame` (`view`, `texture()`, `present()`).
- `Handle<T>` and `Pool<T>`: generational handles; stale handles return `None`.
- `MeshId(pub u32)`, `MaterialId(pub u32)`: engine-wide ids (Copy, Eq, Ord, Hash, Debug).
- `CLEAR_COLOR`: the frame clear colour, linear (0.05, 0.06, 0.08, 1.0).
- Frame graph (module `graph`, re-exported): `FrameGraph`, `RenderPass`, `PassBuilder`,
  `PrepareContext`, `RecordContext`, `ResourceId`, `BACKBUFFER`, `TargetDesc`,
  `TargetSize`, `PassId`. Test-visible helpers: `FrameGraph::is_compiled()` and
  `FrameGraph::first_write_of(pass, resource)`.
- `ClearPass::new(name, target, color)`: clears one target.
- `non_empty(pixels_rgba8, clear_rgba8)`: true when any pixel differs from the clear
  colour by more than 2 in any channel (all targets).
- Native only: `OffscreenTarget::new(gpu, w, h, format)` (usage RENDER_ATTACHMENT |
  COPY_SRC) with `view()`, `texture()`, `size()`, `format()` and `read_rgba8(gpu)`. It strips
  the 256-byte row padding and supports Rgba8Unorm/Rgba8UnormSrgb (`core.unsupported` for any
  other format).
- Native only: `testing::software_gpu() -> Option<GpuContext>` (headless with
  allow_software; prints the reason to stderr on `None`), `testing::skip(test, reason)`
  and `testing::block_on(future)` (a minimal std-only executor).

## Testing

```sh
cargo test -p raa-core
```

CPU unit tests and `tests/graph.rs`/`tests/handles.rs` never need a GPU.
`tests/gpu_readback.rs` renders a `ClearPass` into a 64x64 `OffscreenTarget`, reads it back
and checks every pixel against `CLEAR_COLOR` quantised to unorm8 (within 1). When no
adapter is available it does not fail; it prints

```
SKIP clear_pass_into_offscreen_target_reads_back_clear_color: no software adapter (<reason>)
```

to stderr and returns. Use `--nocapture` to see that line (and the adapter used):

```sh
cargo test -p raa-core --test gpu_readback -- --nocapture
```

Workspace gate (fmt, clippy on host and wasm32, tests, demo builds, asset/SAFETY/neutrality
guards), run before every commit:

```sh
scripts/check.sh
```

Install the pre-commit hook (refuses asset/binary files) once per clone:

```sh
git config core.hooksPath scripts/hooks
```
