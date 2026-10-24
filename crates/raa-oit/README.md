# raa-oit

Per-pixel linked-list order-independent transparency. This first part holds the data contract and the CPU reference; the GPU feature (`transparent.oit`) builds on it.

## Public items

- `node::OitNode` (16 B): `color_rg`, `color_ba` (pack2x16float of straight linear r,g and b,alpha), `depth: f32`, `next: u32` (`END` = `0xffffffff`). Helpers `OitNode::pack`, `unpack`, `to_le_bytes`, `from_le_bytes`; `pack2x16float`, `unpack2x16float`, `f32_to_f16`, `f16_to_f32`; `NODE_SIZE`; `pool_nodes(width, height, factor, limit_bytes)`.
- `resolve_cpu`: `Fragment`, `collect_nearest`, `composite_back_to_front`, `resolve_pixel`, `DEFAULT_MAX_FRAGMENTS` (16). Keeps the nearest `max` fragments by insertion sort on depth, composites far to near with `dst = c*a + dst*(1-a)` (the resolve premultiplies the straight colour). The WGSL resolve uses the same algorithm.

## Test

`cargo test -p raa-oit`
