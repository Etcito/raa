# RAA API m30: global illumination

The declarations are in `packages/raa/types/m30-global-illumination.d.ts`. Global illumination
(GI) keeps a voxel clipmap of the scene around the camera and reads the traced light back as
indirect light. Milestone 1 builds the clipmap (driver A, the jump flood, voxel lighting), the
seven Engine methods and the volume debug views. Screen probes, the lighting slots and driver B
land in milestones 2 and 3 and are configured through these same settings (see
[Milestones](#milestones)).

```ts
type GiDebugView = 'none' | 'voxels' | 'radiance' | 'distance' | 'levels'
interface GlobalIlluminationSettings { enabled: boolean; intensity: number; environmentDiffuse: number; localLightShadows: boolean; reflections: boolean; screenTraces: boolean; debugView: GiDebugView }
type GlobalIlluminationPatch = Partial<GlobalIlluminationSettings>
interface GiQuality { resolution: 64 | 96; levels: 3 | 4 | 5; probeRays: 6 | 8 }
type GiQualityPatch = Partial<GiQuality>
type GiInactiveReason = 'disabled' | 'passDisabled' | 'multiView' | 'noInstance'
interface InstanceGlobalIllumination { requested: boolean | null; voxelized: boolean; driver: 'forward' | 'virtual' | null; code: string | null; message: string | null }
interface GiStats { active: boolean; inactiveReason: GiInactiveReason | null; quality: GiQuality; finestVoxelMetres: number; voxelizedForwardInstances: number; voxelizedVirtualInstances: number; pendingRebuild: boolean; framesSinceRebuild: number; allocatedBytes: number; statsFrameIndex: number | null; pending: boolean }
interface Engine {
  setGlobalIllumination(patch: GlobalIlluminationPatch): void
  globalIllumination(): GlobalIlluminationSettings
  setGiQuality(patch: GiQualityPatch): void
  giQuality(): GiQuality
  setInstanceGlobalIllumination(instance: number, value: boolean | null): void
  instanceGlobalIllumination(instance: number): InstanceGlobalIllumination
  giStats(): GiStats
}
```

## Engine methods

| method | returns | refusals |
|---|---|---|
| `setGlobalIllumination(patch)` | nothing | `gi.invalid`, `api.disposed` |
| `globalIllumination()` | a fresh `GlobalIlluminationSettings` | `api.disposed` |
| `setGiQuality(patch)` | nothing | `gi.invalid`, `api.disposed` |
| `giQuality()` | a fresh `GiQuality` | `api.disposed` |
| `setInstanceGlobalIllumination(instance, value)` | nothing | `api.unknown_handle`, `gi.invalid`, `api.disposed` |
| `instanceGlobalIllumination(instance)` | a fresh `InstanceGlobalIllumination` | `api.unknown_handle`, `api.disposed` |
| `giStats()` | a fresh `GiStats` | `api.disposed` |

## Units

- `finestVoxelMetres` is in metres. It is 0.2 for every quality; level `i` has voxel size
  `0.2 x 2^i` m.
- `allocatedBytes` is in bytes: every GPU byte raa-gi owns.
- `framesSinceRebuild` is in frames.
- `intensity` and `environmentDiffuse` are dimensionless multipliers.
- `statsFrameIndex` is the id of an Engine render invocation. It is not a GPU fence.

## Settings

Defaults and domains, in the key order of `globalIllumination()`:

| key | type | default | domain | meaning |
|---|---|---|---|---|
| `enabled` | boolean | `false` | | GI on; off releases every GI resource |
| `intensity` | number | `1` | finite, [0, 16] | multiplier of the Indirect term |
| `environmentDiffuse` | number | `0` | finite, [0, 1] | scale over raa-ibl's authored diffuse while GI is active |
| `localLightShadows` | boolean | `true` | | local lights are shadowed by the volume (feature 30) |
| `reflections` | boolean | `true` | | SSR misses fall back to a volume trace (feature 37) |
| `screenTraces` | boolean | `true` | | screen probes first trace the previous lit frame |
| `debugView` | `GiDebugView` | `'none'` | one of five | the volume debug view, see [Debug views](#debug-views) |

## Quality

Defaults and domains, in the key order of `giQuality()`:

| key | type | default | domain | meaning |
|---|---|---|---|---|
| `resolution` | number | `96` | 64 or 96 | voxels per axis in each level (R) |
| `levels` | number | `5` | 3, 4 or 5 | clipmap levels (L) |
| `probeRays` | number | `8` | 6 or 8 | rays per side of a screen probe (milestone 2) |

A level spans R voxels per axis. At 96, level 0 spans 19.2 m and level 4 spans 307.2 m. At 64
with 3 levels the spans are 12.8 m, 25.6 m and 51.2 m.

## Patches and merge

- `undefined` is omission. An omitted key keeps its current value: `setGiQuality({ levels: 3 })`
  keeps `resolution` and `probeRays`.
- A patch is read in entry order (the order of its own keys). The first offending key is the one
  reported. The refusal is atomic: no key of the patch is applied, and the state and the next
  frame do not change.
- Every refusal is `gi.invalid` with the message `<key>: <reason>`:

| case | message |
|---|---|
| unknown key, whatever its value | `<key>: unknown setting` |
| `null` | `<key>: must not be null` |
| wrong type | `<key>: must be a boolean`, `must be a number` or `must be a string` |
| non-finite or outside the domain | `intensity: must be finite in [0, 16]`, `environmentDiffuse: must be finite in [0, 1]` |
| a number that is not a listed integer | `resolution: must be 64 or 96`, `levels: must be 3, 4 or 5`, `probeRays: must be 6 or 8` |
| unknown view | `debugView: must be none, voxels, radiance, distance or levels, got <name>` |
| non-object argument | `settings: must be an object`, or `quality: must be an object` for `setGiQuality` |

For example, `setGlobalIllumination({ intensity: 99, debugView: 'bad' })` throws
`gi.invalid` with `intensity: must be finite in [0, 16]`, and neither key is applied.

- `globalIllumination()` and `giQuality()` return fresh objects. Changing a returned object
  changes nothing in GI.

## Quality changes

- A quality that differs from the current one reallocates the volume and the screen resources at
  the next active frame. Every level is then rebuilt in that frame: the first-build path.
- An unchanged quality does nothing.
- A quality set while GI is inactive is kept; the allocation happens at the next active frame.

## Instances

`setInstanceGlobalIllumination(instance, value)` and `instanceGlobalIllumination(instance)` take a
live instance-group handle. An unknown handle throws `api.unknown_handle`.

- `value` must be `true`, `false` or `null`. Anything else, `undefined` included, throws
  `gi.invalid` with `value: expected a boolean or null`.

| `value` | `requested` | membership |
|---|---|---|
| `null` | `null` | the default: a voxel member unless a refusal below applies |
| `true` | `true` | the default, recorded as an explicit request |
| `false` | `false` | opted out (rank 0): not voxelized, still receives GI |

- `instanceGlobalIllumination(instance)` returns `requested`, `voxelized`, `driver`, `code` and
  `message`. `voxelized` and `driver` describe the latest prepared frame. `driver` is `forward`
  when driver A voxelizes the group, `virtual` when driver B does (milestone 3), and `null` when
  it is not voxelized. Before any frame, `voxelized` is `false` and `driver`, `code` and `message`
  are `null`.
- `code` and `message` give the first refusal of the rank table below, or `null` when there is
  none.

## Membership codes

A group's `code` is the first matching rank:

| rank | code | condition |
|---|---|---|
| 0 | `gi.opted_out` | the request is `false` |
| 1 | `gi.skinned_instance` | the instance is skinned |
| 2 | `gi.deformed_instance` | the instance is deformed, or its material has a vertex function |
| 3 | `gi.non_opaque_material` | a material class other than Opaque: transparent, transmissive or fur |
| 4 | `gi.no_shadow_material` | the material has the NO_SHADOW flag |

Ranks 1 to 3 are the rank table of raa-vg. Voxel members are the visible scene instances with no
refusal. Every instance receives GI, member or not: an opted-out group is kept out of the voxels
but still receives GI, and so do refused skinned, deformed, transparent and fur instances.

## Stats

`giStats()` returns these keys in this order:

| key | unit and type | meaning |
|---|---|---|
| `active` | boolean | GI is built and shading for the frame; true exactly when `inactiveReason` is null |
| `inactiveReason` | `GiInactiveReason` or null | the first matching rank below; null when active |
| `quality` | `GiQuality` | the current quality |
| `finestVoxelMetres` | metres | 0.2 |
| `voxelizedForwardInstances` | count | scene instances voxelized by driver A in the latest prepared frame |
| `voxelizedVirtualInstances` | count | scene instances voxelized by driver B in the latest prepared frame |
| `pendingRebuild` | boolean | a rebuild slice is pending, or a round is outdated or waiting |
| `framesSinceRebuild` | frames | frames since the last rebuild round completed; saturates at 1000000; 0 while pending |
| `allocatedBytes` | bytes | every GPU byte raa-gi owns; 0 when inactive |
| `statsFrameIndex` | invocation id or null | the newest invocation whose submission completed; null before the first |
| `pending` | boolean | a prepared frame whose submission is not yet observed |

Inactive reasons, first match wins:

| rank | `inactiveReason` | condition |
|---|---|---|
| 0 | `disabled` | `enabled` is false |
| 1 | `passDisabled` | the `gi.volume` pass is disabled |
| 2 | `multiView` | the invocation submitted a number of views other than 1 |
| 3 | `noInstance` | no instance is a voxel member |

The counters come from CPU state published on complete submission. No call waits for the GPU and
there is no readback. Before the first published frame, `statsFrameIndex` is null.

## Lifecycle

- GI is active only with exactly one submitted view, and it follows that view's camera. Any other
  view count is `multiView`.
- While inactive, no gi pass records, every GI buffer and texture is released, and the group-1 GI
  dummies are bound.
- Disabling GI (`disabled`) or the `gi.volume` pass (`passDisabled`) releases everything.
  Re-enabling starts the first-build path: every level is rebuilt in one frame.
- A rebuild round starts when a voxel member is added or removed, or when a member's signature
  changes (slot, mesh, material id, world matrix, base colour, emissive or material flags). The
  levels then rebuild over several frames, six of the 18 flood passes per frame, while their old
  contents keep serving. A moved instance lights its new place after the round completes.
- A camera move rebuilds the level whose snapped origin changed, in cells of 8 voxels.
- Resize: in milestone 1 the volume does not depend on the surface size, so a resize changes
  nothing. From milestone 2, a resize recreates only the screen resources and invalidates the
  screen histories.
- Camera cut: a camera discontinuity of view 0 (`scene_invocation.is_discontinuous(0)`) makes the
  frame call `invalidate_history()`. From milestone 2 that clears the screen histories. The volume
  is not reset: its levels follow the camera by rebuild rounds, and old atlas contents serve until
  each level commits.
- Disposal: `dispose()` drops GI with the Engine state. Every later call on the disposed Engine
  throws `api.disposed`.

## Volume sizes

The clipmap volume of quality R and L levels takes `32 x R^3 x L + 8 x R^3` bytes: voxel words,
seeds, radiance, atlas and the two flood buffers.

| resolution | levels | volume bytes |
|---|---|---|
| 96 | 5 | 148,635,648 |
| 96 | 4 | 120,324,096 |
| 96 | 3 | 92,012,544 |
| 64 | 5 | 44,040,192 |
| 64 | 4 | 35,651,584 |
| 64 | 3 | 27,262,976 |

These are on top of the raa-sdf-trace volume, the 4096 x 4096 atlas and the per-viewport targets.
While GI is active, `allocatedBytes` is at least the volume size of the current quality.

## Debug views

With `debugView` other than `none`, the GI history pass replaces the shaded frame with the view
below while GI is active. With `none` the frame is unchanged.

| `debugView` | shows |
|---|---|
| `none` | the shaded frame |
| `voxels` | the first occupied voxel along the camera ray, as a flat-shaded 0.2 m cube: albedo x (0.35 + 0.65 x N.L) |
| `radiance` | the atlas radiance at the first hit |
| `distance` | iso-bands `0.5 + 0.5 cos(2 pi d / 0.4 m)` of the distance field, sampled 2 finest voxels in front of the depth surface |
| `levels` | the finest level containing the depth surface: 0 red, 1 orange, 2 yellow, 3 green, 4 blue |

## Milestones

| part | milestone | what it configures or adds |
|---|---|---|
| clipmap, driver A, flood, voxel lighting, Engine methods, volume debug views | 1 | this document |
| screen probes, lighting slots (Indirect, Ao and LocalVisibility `gi`), `environmentDiffuse` scale, SSR fallback | 2 | `intensity`, `environmentDiffuse`, `localLightShadows`, `reflections`, `screenTraces`, `probeRays` |
| driver B, virtualized members (`driver: 'virtual'`) | 3 | the members raa-vg draws virtually |

Until milestone 2 the lit frame does not read the volume. In milestone 1 the settings above are
stored and read back, and they change the output only when their part lands.

## Known limits

- The finest voxel is 0.2 m, so walls thinner than about two finest voxels leak light.
- GI follows the single view's camera.
- Double counting with raa-ibl's SH irradiance is mitigated by `environmentDiffuse` (default 0: diffuse from traced irradiance alone, as in Solstice).
- Material functions are not evaluated for voxel albedo; driver A uses base colour times tint and driver B base colour only. Vertex colours are not voxelized.
- Budgets stay nearly exhausted: fragment storage 8 of 8 in OIT variants, compute storage 8 of 8 in driver B, sampled textures 16 of 16 in vg.shade, and feature words z and w.
- Glossy GI reflections are not provided; SSR misses fall back to a volume trace.
