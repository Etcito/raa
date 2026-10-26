# raa-pipeline

Configurable pass pipeline of the lit renderer (S12) and its group-1 lighting resources
(S10). Feature crates (raa-lit, raa-tonemap, later sky, shadows, AO, bloom...) implement
`FeaturePass` and add themselves to a `Pipeline`; the pipeline runs the enabled passes in
order inside one raa-core frame graph pass, `raa.pipeline`, which writes BACKBUFFER. Passes
can be enabled, disabled and reordered from JS; invalid configurations are refused with a
typed code. Depends on raa-core only.

## Public items

- `trait FeaturePass`: `name`, `order`, `required` (default false), `resize(gpu, targets)`,
  `set_enabled(enabled, lighting)` (disabling resets the pass's group-1 resources to the
  dummies), `prepare(&mut PrepareCtx)` (default nothing), `record(&mut RecordCtx)`.
- `PrepareCtx { gpu, size, frame, time_s, dt_s, targets, lighting: &mut }` and
  `RecordCtx { gpu, encoder, size, frame, time_s, targets, lighting, backbuffer,
  backbuffer_format, surface_is_srgb }`.
- `PassInfo { name, role, enabled, order, required }` and `PassRole::{Pass, Contributor}`.
- `Pipeline`:
  - `new(gpu, surface_format, surface_is_srgb)`: empty, with dummy lighting resources.
  - `add(Box<dyn FeaturePass>)`: inserted, enabled, before the first pass of greater order
    in the current execution order (after equal orders).
  - `add_contributor_entry(name, order, enabled: Rc<Cell<bool>>)`: a draw inside the
    forward pass (fur shells, opaque particles, sky background); the pipeline lists it and
    toggles the shared flag, its owner reads the flag.
  - `passes()`: every entry in execution order; contributors (by order) come right after
    their host `forward`, or at the end while no `forward` pass is added.
  - `set_enabled(name, enabled)`: sets the flag and, for a pass, calls its
    `set_enabled(enabled, lighting)` (every time, also when the state does not change).
  - `set_order(names)`: replaces the execution order of the passes.
  - `set_time(now_ms)`: `time_s = now_ms / 1000`; `dt_s` = seconds since the previous call
    (0 on the first call and when time goes backwards); non-finite values are ignored.
  - `lighting()`, `lighting_mut()`, `targets()` (`None` before the first prepare),
    `surface_format()`, `surface_is_srgb()`.
  - `install(&Rc<RefCell<Pipeline>>, &mut FrameGraph)`: adds the RenderPass `raa.pipeline`.
    Its prepare creates the targets (first frame) or recreates them on a size change, resets
    the HDR chain (forward writes A), uploads the lighting resources, then for each enabled
    pass in order calls `resize` when the targets are new to it and `prepare`; it uploads the
    lighting again afterwards so resources a pass set in its prepare are bound in the same
    frame. Its record calls each enabled pass's `record` in order with the backbuffer view
    and format of the graph. The frame path does not allocate: it iterates the stored
    entries; targets are only created on a size change and the bind group is only rebuilt
    after a lighting resource changed.
- `PipelineConfig`: the device-free configuration core `Pipeline` delegates to (`new`,
  `add`, `add_contributor_entry`, `passes`, `set_enabled` returning the pass to call,
  `set_order`). It never calls the passes' GPU methods, so the configuration rules are
  tested without a GPU.
- `Targets`, `HdrChain`, `HDR_FORMAT`: surface-sized depth (Depth32Float), normal
  (Rgba16Float), material (Rgba8Unorm) and the HDR A/B chain (Rgba16Float) with
  `src`/`dst`/`swap`/`reset`.
- `LightingResources` (group-1 layout, 1x1 dummies, setters and `clear_*`, bind group
  rebuilt when dirty) and `EnvGpu` (the 256 B S10 env uniform).

## Refusals

Every configuration refusal is `RaaError::FrameGraph { code, message }` with a non-empty
message; a refused call changes nothing.

| code | when |
|---|---|
| `pipeline.duplicate_pass` | `add` or `add_contributor_entry` with a name already used by a pass or a contributor; `set_order` listing a name twice |
| `pipeline.unknown_pass` | `set_enabled` or `set_order` with a name that is not in the pipeline; `set_order` with a contributor name (contributors keep their order inside `forward`) |
| `pipeline.required_pass` | `set_enabled(name, false)` on a required pass (prepass, forward, tonemap) |
| `pipeline.order_incomplete` | `set_order` not listing every pass (the message names the missing ones) |
| `pipeline.required_order` | `set_order`, or `add` by its order, giving `forward` before `prepass`, `tonemap` before either, or any pass after `tonemap` |

`set_order` checks per name (unknown, then duplicate), then completeness, then the required
order. The frame graph pass reports a pipeline still borrowed during the frame as
`core.resource`; errors returned by a feature pass's `prepare` or `record` stop the frame
and are returned by `FrameGraph::execute` unchanged.

## How to test

```sh
cargo test -p raa-pipeline
cargo clippy -p raa-pipeline --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown -p raa-pipeline
```

`tests/config.rs` uses mock passes. Its CPU tests drive `PipelineConfig`: `passes()` order
with contributors after `forward`, enable/disable of passes and contributors, order overrides
and passes added after one, and every refusal code with its message. Its GPU test (prints
`SKIP pipeline_on_the_software_gpu: ...` without a software adapter) checks that
`Pipeline::set_enabled` hands the lighting resources to the pass, and that the installed
`raa.pipeline` pass resizes, prepares and records the enabled passes in order with the frame
time, the backbuffer view and format, resets the HDR chain every frame, recreates the targets
on a new size and follows an order override. `tests/resources.rs` covers the lighting
resources, `tests/targets.rs` the targets.

Display-space passes can return true from `FeaturePass::output_overlay()`. They
follow every HDR pass and tonemap; the ordering validator refuses an HDR pass
after an overlay. Existing passes retain the default false behaviour.
