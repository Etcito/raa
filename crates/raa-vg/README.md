# raa-vg

Virtualized geometry of RAA (module m29, S31 and S32). A mesh with clusters from `raa-cluster` is
drawn through a GPU-driven hierarchy: each frame the cut of every virtualized instance is culled
against the view, the surviving clusters are rasterised into visibility buffers, resolved into a
G-buffer and shaded as a forward contributor. The same clusters cast into the existing cascade
shadow maps.

## What the crate does

- **Settings and refusals** (S31, `settings.rs`, `error.rs`): the `VgSettings` record with the S31
  defaults, the `VgSettingsPatch` merged over it and validated as a whole, and the `vg.*` refusal
  codes. A refusal is `vg.invalid` and names the camelCase field.
- **Path eligibility** (`eligibility.rs`): skinned, deformed, transparent and fur content stays on the
  forward path. The first refusal of the rank table (skinned, deformed or vertex function, material
  class) decides the code, and `InstancePath` reports the path of each instance.
- **Scene pool** (S32, `scene.rs`, planned): the read-only instance, node, cluster, vertex and
  triangle buffers, grown in pages, with the path plan for each instance group.
- **Passes** (S32, planned): cull (`vg.cull`, compute, hierarchy traversal with occlusion from the
  previous frame's depth), raster (`vg.raster`, hardware and software rasterisers into the
  visibility buffers, the occlusion pyramid and the cascade casters), resolve (`vg.resolve`, four
  outputs: ids, normal, uv motion and albedo), the G-buffer merge (`vg.gbuffer`, which writes the
  material output and `frag_depth`) and the forward contributor (`vg.shade`, which writes the opaque
  final colour before the forward passes that follow).

Virtualized instances leave the lit renderer's draw lists but keep their persistent instance records
(`raa-lit` `VirtualSet`). Skinned, deformed, transparent and fur instances stay on forward beside them.

## Disabled state

With every vg pass disabled, or the path inactive (`enabled` false, a multi-view frame, or no
instance), no vg pass is recorded, no vg buffer or texture is allocated and feature ids 27, 28 and 36
are clear: the pool allocates nothing until a path is active.

## Adapted solstice sources

Later cards of this crate adapt the following files of solstice at `7e465b5`. Each adapted file keeps
the MIT notice below and names its source in its module docs.

Rust, from `crates/solstice-render/src/vg/`:

- `culling.rs`, `hzb.rs`, `raster.rs`, `resolve.rs`

WGSL, from `crates/solstice-render/shaders/vg/`:

- `cull.wgsl`, `cull_control.wgsl`, `hzb_seed.wgsl`, `hzb_down.wgsl`, `pass_view.wgsl`
- `raster_common.wgsl`, `raster_hw.wgsl`, `raster_hw_fragment.wgsl`, `raster_sw.wgsl`,
  `raster_sw_body.wgsl`, `raster_sw_depth.wgsl`, `vis_write.wgsl`
- `resolve.wgsl`, `shadow_merge.wgsl`, `depth_stamp.wgsl`

The solstice include preprocessor is not ported: each module is assembled by a Rust builder that
concatenates WGSL constants. `resolve.wgsl` drops its wetness include, its waterline bindings and its
material texture tiers. Every comparison of the adapted shaders goes through the `raa-shader` depth
helper (standard-Z), never a hand-flipped reversed-Z comparison.

MIT notice (from solstice `LICENSE`):

```text
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Checks

```sh
cargo test -p raa-vg --test settings
cargo check -p raa-vg --target wasm32-unknown-unknown
cargo clippy -p raa-vg --all-targets --no-deps -- -D warnings
```
