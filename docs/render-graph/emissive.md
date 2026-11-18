# Emissive sources of the GI pass nodes

The three GI pass nodes of the render graph each carry an `emissive` parameter. It selects
whether emissive surfaces feed that stage of global illumination.

| node | parameter | default | group | rebuild | binding |
|---|---|---|---|---|---|
| `pass.gi.volume` | `emissive` | `true` | Emissive | pipeline | node (`ParamBinding::Node`) |
| `pass.gi.screen` | `emissive` | `true` | Emissive | pipeline | node |
| `pass.gi.ssgi` | `emissive` | `true` | Emissive | pipeline | node |

The parameter is a Bool with label `Emissive`. `pass.gi.area-shadow` and `pass.gi.history` have no
such parameter. The node family is `crates/raa-graph/src/nodes/gi_emissive.rs`; the executor
extension that writes the value to the engine is `crates/raa-graph/src/exec_ext/gi_emissive.rs`,
which calls `GiHandle::set_emissive_sources` (`crates/raa-gi/src/feature.rs`).

## What the emission is

The emission is the material record's `emissive` of `addSurfaceMaterial`
(`docs/api/m10-materials.md`): a linear RGB value, the colour times the strength, unbounded above.
A surface with a zero `emissive` emits nothing, so the parameter has no effect on it.

## What each node does

- **`pass.gi.volume` (`emissive`)**: the emitted radiance is injected into the voxel radiance that
  every volume ray reads. When the parameter is off, the voxel light keeps only the lit albedo term
  and the emitted term is dropped. Changing the emission of a surface, or toggling this source,
  restarts every voxel's average at the next frame.
- **`pass.gi.screen` (`emissive`)**: the emission of surfaces hit by the screen-probe and ReSTIR GI
  screen rays. When the parameter is off, the emission at the hit point is subtracted from the
  radiance those rays read, never below zero.
- **`pass.gi.ssgi` (`emissive`)**: the emission of surfaces hit by the SSGI samples. When the
  parameter is off, the emission at the hit point is subtracted from the radiance the samples read,
  never below zero.

The screen and SSGI nodes read the emission of a hit from the GI voxel buffer, so both need the
voxel buffer to be bound (see the budget change below).

## Default and equivalence

The parameter defaults to on. With every source on, a graph renders as the engine did before this
parameter existed: the default Full GI and Debug starters carry `"emissive": true` on their three
nodes, and the no-graph export writes `true` for each.

## Rebuild class

Each node's `emissive` has rebuild class `pipeline`. Changing it rebuilds the pipeline of the
stage: the voxel light pipeline for `pass.gi.volume`, the probe-trace and ReSTIR GI main pipelines
for `pass.gi.screen`, and the SSGI trace pipeline for `pass.gi.ssgi`.

## Voxel-buffer budget change

The emission lookup binds the GI voxel buffer as a read-only storage buffer, so the Compute stage
budgets of two nodes change:

| node | Compute `storage_buffers` before | after |
|---|---|---|
| `pass.gi.screen` | 2 | 3 (the voxel buffer at group 2 binding 6) |
| `pass.gi.ssgi` | 0 | 1 (the voxel buffer at group 2 binding 6) |

The other budget fields are unchanged. Both values stay within the device-free limits
`GraphLimits::CORE`, which allow 8 storage buffers per stage. `pass.gi.volume` has no budget
change.
