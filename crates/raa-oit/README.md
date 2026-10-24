# raa-oit

Per-pixel linked-list order-independent transparency. It holds the data contract, the CPU reference of the resolve and the GPU feature pass `transparent.oit` (S12, order 80).

## Public items

- `node::OitNode` (16 B): `color_rg`, `color_ba` (pack2x16float of straight linear r,g and b,alpha), `depth: f32`, `next: u32` (`END` = `0xffffffff`). Helpers `OitNode::pack`, `unpack`, `to_le_bytes`, `from_le_bytes`; `pack2x16float`, `unpack2x16float`, `f32_to_f16`, `f16_to_f32`; `NODE_SIZE`; `pool_nodes(width, height, factor, limit_bytes)`.
- `resolve_cpu`: `Fragment`, `collect_nearest`, `composite_back_to_front`, `resolve_pixel`, `DEFAULT_MAX_FRAGMENTS` (16). Keeps the nearest `max` fragments by insertion sort on depth, composites far to near with `dst = c*a + dst*(1-a)` (the resolve premultiplies the straight colour). The WGSL resolve uses the same algorithm.
- `OitSettings { max_fragments (4..=32, default 16), pool_factor (2..=32, default 8) }`, `OitStats { pool_nodes, overflow_frames }`.
- `OitFeature::install(gpu, pipeline, lit) -> Result<OitHandle, RaaError>`: refuses `pipeline.busy` and `pipeline.duplicate_pass`; registers the `DrawClass::Transparent` handler in the lit renderer and adds the FeaturePass `transparent.oit`. Each frame with Transparent draws it clears the heads (compute), draws the Transparent draws with `PassKind::Transparent` (msaa 1, no colour target, Less against the prepass `depth` read-only, cull Back unless DOUBLE_SIDED, group 2 = heads/nodes/counter/OitParams), copies the counter into a readback ring of 3 (an overflow frame doubles the pool factor, bounded by the adapter limit), resolves hdr.src into hdr.dst and swaps. Without Transparent draws the HDR chain is untouched.
- `OitHandle` (`Clone`): `set(OitSettings)` (`oit.invalid` naming the field; unchanged on refusal), `get()`, `stats()`.
- Shaders: `CLEAR_WGSL` (`CLEAR_ENTRY`, `CLEAR_WORKGROUP_SIZE`, `CLEAR_PARAMS_SIZE`), `RESOLVE_WGSL` (`RESOLVE_VS_ENTRY`, `RESOLVE_FS_ENTRY`, `RESOLVE_PARAMS_SIZE`).

## Test

`cargo test -p raa-oit`
