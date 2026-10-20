# raa-bench

Frame loop benchmarks. Native binary only: it needs no GPU.

## frame-allocs

Counts heap allocations on the CPU frame path. It links
`raa_alloc_count::CountingAlloc` as `#[global_allocator]` and builds a scene of
`ceil(instances / 100)` parent nodes, each with up to 100 child nodes on a grid, one
instance per child (fake `MeshId(i % 4)`, `MaterialId(i % 8)`), a `RenderList` and a
`FramePrep`. Each frame it rotates every parent (`set_local`), then runs
`Scene::update_world`, `Scene::build_render_list` and `FramePrep::prepare_cpu`. The
counters are reset after the warm-up frames; the measured frames are timed with
`std::time::Instant`.

```sh
cargo run --release -p raa-bench --bin frame-allocs -- --instances 1000 --warmup 120 --frames 600
```

Output, exactly one JSON line on stdout:

```json
{"instances":1000,"warmup":120,"frames":600,"allocations":0,"bytes":0,"prepCpuMsMedian":0.05}
```

- `allocations`, `bytes`: alloc, alloc_zeroed and realloc calls (and their sizes) over
  the measured frames.
- `prepCpuMsMedian`: median CPU milliseconds of one frame (steps above).

The binary exits 0 whatever the count is; the milestone proof judges it. Quality bar Q4
requires `allocations` = 0 over ALLOC_RUN (1000 instances, 120 warm-up, 600 measured
frames). An invalid argument prints the usage to stderr and exits 2; a frame error exits 1.

Last measured (2026-10-17): `{"instances":1000,"warmup":120,"frames":600,"allocations":0,"bytes":0,"prepCpuMsMedian":0.0834}`

## Test

```sh
cargo run --release -p raa-bench --bin frame-allocs
cargo clippy -p raa-bench -- -D warnings
```
