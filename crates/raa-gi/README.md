# raa-gi

Global illumination of RAA (module m30, S36-S39): a voxel clipmap volume in which forward and
virtualized instances are voxelized, lit with bounce light and traced by screen probes. The volume
is rebuilt in slices as the camera moves, relights one clipmap level per frame and is disabled
without cost: no gi pass records and no resource is allocated while GI is inactive.

## What the crate does

- **Settings and refusals** (S36, `settings.rs`, `error.rs`): the `GiSettings` record with the S36
  defaults, the `GiSettingsPatch` merged over it in key order, the `GiQuality` volume size (96/5/8
  by default) and the `gi.*` refusal codes. A refusal is `gi.invalid` and names the camelCase field.
- **Clipmap volume** (S37, `clipmap.rs`, `cpu.rs`, planned): the snapped level origins, the rebuild
  rounds and flood schedule, the relocation choice, the volume bytes and the CPU twins of the
  voxelize, flood, codec and trace maths.
- **Membership** (S37, `membership.rs`, planned): the rank table that decides which instances are
  voxelized and by which driver; every instance receives GI, opted-out groups included.
- **Feature, parts and drivers** (S37, S38, `feature.rs`, `parts.rs`, `stats.rs`, planned): the
  `gi.volume` and `gi.history` passes, the forward driver A, the milestone-2 and milestone-3 part
  seam and the published `GiStats`.

## Disabled state

With GI disabled, `gi.volume` disabled or more than one submitted view, no gi pass records, every
gi buffer and texture is released, the group-1 GI dummies are bound and `allocatedBytes` is 0.

## Adapted solstice sources

Later cards of this crate adapt the following files of solstice at `7e465b5`. Each adapted file
keeps the MIT notice below and names its source in its module docs.

Rust, from `crates/solstice-render/src/gi/`:

- `volume.rs`, `screen.rs`, `depth_pyramid.rs`

WGSL, from `crates/solstice-render/shaders/gi/`:

- `voxelize.wgsl`, `jump_flood.wgsl`, `volume_write.wgsl`, `radiance_codec.wgsl`, `voxel_light.wgsl`
- `probe_common.wgsl`, `probe_trace.wgsl`, `screen_trace.wgsl`, `screen_history.wgsl`,
  `depth_pyramid.wgsl`, `probe_project.wgsl`, `probe_integrate.wgsl`

The volume half of `crates/solstice-render/shaders/common/environment.wgsl` (lines 140-219) and the
Indirect slot of `shaders/lighting/deferred.wgsl` are adapted too. The solstice include
preprocessor is not ported: each module is assembled by a Rust builder that concatenates WGSL
constants. Every comparison of the adapted shaders goes through the `raa-shader` depth helper
(standard-Z), never a hand-flipped reversed-Z comparison.

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
cargo check
cargo build -p raa-gi --target wasm32-unknown-unknown
cargo clippy -p raa-gi --all-targets --no-deps
```
