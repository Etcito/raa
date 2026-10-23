# raa-web

The web runtime library behind every RAA demo page: it owns the canvas, the frame graph and
the application loop, records CPU frame times, and converts results to plain JavaScript
values. The frozen wasm-bindgen demo API and the `@etcito/raa` npm package are separate
modules (`raa-js` / js-package), not this crate.

## Public items

- `FrameInput { frame, now_ms, dt_ms, size }` — the clock and canvas size of one update.
- `FrameCounters { instances, draw_calls, triangles }` — per-frame work reported by an app.
- `WebApp` — the application trait: `init` adds passes to the frame graph (called once,
  before compilation), `update` does the per-frame CPU work, `resize` reacts to canvas size
  changes. It lives on every target so applications compile and unit-test on the host.
- `FrameStats { frames, first_frame_ms, cpu_ms_median, cpu_ms_p95, counters,
  allocs_last_frame }` — the stats snapshot (module `stats`).
- `stats::CpuRing` — the fixed-capacity ring and the median/p95 helper behind the stats,
  unit-tested on the host.
- `stats_to_js(&FrameStats, &AdapterSummary) -> JsValue` (wasm32) — a plain object with the
  keys `adapter {vendor, architecture, device, description, isFallbackAdapter}`, `hardware`,
  `firstFrameMs`, `frames`, `cpuMsMedian`, `cpuMsP95`, `instances`, `drawCalls`, `triangles`
  and `allocsPerFrameTotal`. Unknown values are `null`.
- `error_to_js(&RaaError) -> JsValue` (wasm32) — a JS `Error` named `RaaError`, with the
  `Display` text as the message and the stable code in a `code` property.

wasm32 only:

- `RuntimeOptions { width, height, gpu, stats_window, alloc_probe }` with `Default`
  1280x720, `stats_window` 300 and `alloc_probe` `None`.
- `WebRuntime::new(canvas, app, opts)` — sizes the canvas, creates the GPU context and the
  surface, calls `WebApp::init` and compiles the frame graph.
- `WebRuntime::new_with(canvas, app, opts, &DeviceRequest)` — `new` with the context created
  through `GpuContext::for_canvas_with` (core adapter, `optional_features` intersected with
  the adapter's, adapter limits when `adapter_limits`); `new` calls it with
  `DeviceRequest::default()`.
- `WebRuntime::frame(now_ms)` — update, acquire, execute, submit, present; records the
  `performance.now()` delta in the ring and registers the frame-0
  `on_submitted_work_done` callback that captures `first_frame_ms`.
- `WebRuntime::resize(width, height)`, `adapter()`, `gpu()`, `app_mut()`.
- `WebRuntime::stats()` — median/p95 over the ring using a preallocated scratch buffer, plus
  the last counters and allocation count.
- `WebRuntime::probe_non_empty(now_ms)` — renders one frame, copies the backbuffer (or an
  offscreen target when the surface lacks `COPY_SRC`) into a mapped buffer and runs
  `raa_core::non_empty` against `CLEAR_COLOR` quantised to the canvas encoding.

## Native vs wasm32

- Native (host): the shared types (`FrameInput`, `FrameCounters`, `WebApp`, `FrameStats`,
  the `stats` module and the colour-quantisation tests). No GPU is touched, so
  `cargo test -p raa-web` runs anywhere.
- wasm32: additionally the runtime module (`WebRuntime`, `RuntimeOptions`, the frame loop and
  `probe_non_empty`) and the JS conversion module (`js.rs`). Both are gated with
  `#[cfg(target_arch = "wasm32")]`, so the native build pulls in neither web-sys nor the
  browser frame loop.

## How to test

- `cargo test -p raa-web` — native unit tests (ring statistics, colour quantisation).
- `cargo check -p raa-web --target wasm32-unknown-unknown` — the wasm build of the runtime
  and the JS conversions.
- `cargo clippy -p raa-web --lib --target wasm32-unknown-unknown -- -D warnings` and
  `cargo clippy -p raa-web --all-targets -- -D warnings` — lint gates.
- Real browser behaviour (frame loop, resize, `probe_non_empty`, the exact JS keys) is
  verified by the demo capture: `scripts/build-web.sh` plus
  `scripts/gpu-capture.mjs <demo>`, whose result JSON is the committed record.
