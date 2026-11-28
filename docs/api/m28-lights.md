# RAA API m28: local lights and attachments

The declarations are in `packages/raa/types/m28-lights.d.ts`; the package build
appends this fragment to `packages/raa/index.d.ts`. These methods add four local
light kinds while preserving the existing `setLight(LightDesc)` sun API.

```ts
interface Engine {
  addLight(desc: LocalLightDesc): LightHandle
  updateLight(handle: LightHandle, changes: LocalLightChanges): void
  removeLight(handle: LightHandle): void
  attachLight(handle: LightHandle, instanceId: number,
              localOffset: [number, number, number]): void
  detachLight(handle: LightHandle): void
  lightStats(): LightStats
  setAreaLightShading(settings: AreaLightShadingDesc): void
  areaLightShading(): AreaLightShadingDesc
  setAreaLightShadows(settings: AreaLightShadowsDesc): void
  areaLightShadows(): AreaLightShadows
}
```

## Descriptions and defaults

`LocalLightDesc` fields are optional. Defaults are `kind: 'point'`,
`position: [0, 0, 0]`, `direction: [0, 0, -1]`, `color: [1, 1, 1]`,
`intensity: 2`, `range: 6`, `angleDeg: 35`, `innerAngleDeg: undefined`,
`size: [1, 1]`, `radius: 0`, `shadows: false`, `gi: true`,
`shadowMode: 'traced'` and `shadowMinScreenSize: 0`. Position, range, size and
radius use metres. JS `color` is sRGB in [0, 1] and is converted once to linear
RGB; intensity is nonnegative. Direction must be finite and nonzero and is
normalized when accepted.

`angleDeg` and `innerAngleDeg` are full cone angles in degrees, each in [1, 179].
An explicit inner angle cannot exceed the outer angle. When omitted, the effective
inner angle is 80% of the outer angle. `size` contains positive full width and
height values. All numeric values must be finite; colours must be in [0, 1].
`shadows: true` makes a point or spot light cast shadows (see "Local-light cube
shadows" below); `radius` is its emitter radius, where its shadow rays start. Both
are ignored by area lights (own tiers) and directional lights.

An area light emits radiance intensity / (width x height) per unit colour over its
rectangle, so intensity is radiance x area.

## Local-light cube shadows

A point or spot light with `shadows: true` holds one of 32 shadow slots (later
shadow-casting lights stay unshadowed): 6 cube faces of R32Float distances from the
light to the first occluder, traced through the shared mesh BVH (the one DDGI traces;
without a BVH-consuming GI node nothing is traced and the lights stay unshadowed).
Rays start `radius` metres from the light, so geometry inside it (its bulb, a
lantern's glass) casts nothing and stays lit, including its diffuse-transmission back
term. Faces are cached: one is traced again only when its light moves (position,
range, radius), the mesh geometry changes inside its range, or the array is
reallocated. At most `facesPerFrame` faces are traced per frame (pass `shadow.local`,
order 36), nearest lights first; an untraced face reads as unoccluded. The forward
reads the faces with textureLoad and 2x2 PCF (group 1 binding 22; the light record
carries `slot + 1` in its height lane). Shadow-casting lights are never put in the
probe-only tier (light cutoff); the per-tile forward budget still applies.

A traced light that moved (position, range or radius) is re-traced in full in the frame
it moves, before and outside the `facesPerFrame` budget (at most 24 faces), so a torch
carried every frame keeps a current shadow (about 0.8 ms for one point light's 6 faces at
128 texels on an RTX 3090). The BVH holds static geometry only: dynamic nodes
(`setRuntimeNodeDynamic`) cast no traced shadow; use `shadowMode: 'map'` for those.

## Per-light GI and shadow options (dynamic lights)

- `gi: false` (direct only): the light is zeroed in the copy of the light records that the
  GI passes bind (`gi_local_lights_buffer`: DDGI, spherical GI, NRC, mesh-trace pass), so a
  moving light causes no probe relight (DDGI counts no light edit) and no GI flicker. It is
  never put in the probe-only tier. The voxel GI (`raa-gi`) still reads all lights.
- `shadowMode: 'map'` (point/spot with `shadows: true`): the light's own depth map, rendered
  every frame by `shadow.csm` from the caster lists (dynamic nodes included) into extra
  layers of the sun shadow array after the area-light heroes: one layer for a spot (half
  angle tangent 1.1 tan(outer/2) + 0.02, at most tan 70 degrees), six cube faces for a
  point light (tangent 1.05). Near 0.05 m, far = range; 4 comparison taps. At most 12 map
  layers (two point lights, or twelve spots); the array is reallocated only when map lights
  are added or removed. The light record's height lane holds `-layer`.
- `shadowMinScreenSize` (impact threshold X in [0, 1], default 0): a map light whose impact
  is below it uses its traced cube instead. Impact is the area-light hero metric: the
  fraction of the screen area covered by the range sphere, `min(1, pi r_px^2 / (w h))` with
  `r_px = f r / sqrt(d^2 - r^2)`, `f = h / (2 tan(fovY / 2))`, and 1 when the eye is inside
  the range. The highest-impact map lights get the layers first.

Invalid values throw `light.invalid_desc` ("'shadowMode' must be 'traced' or 'map'",
"shadowMinScreenSize must be in [0, 1]").

```ts
interface LocalShadowsDesc { enabled?: boolean; resolution?: number; facesPerFrame?: number }
```

`setLocalShadows(settings)` (defaults `enabled: true`, `resolution: 128` texels per
face side, a multiple of 8 in [64, 1024], `facesPerFrame: 2` in [1, 24]); a refusal
throws `api.invalid_argument` (non-object) or `shadow.local_invalid` and changes
nothing. `invalidateLocalShadows()` re-traces every face. `localShadowStats()`
returns the settings, `lights`, `capacity`, `readyFaces`, `pendingFaces`,
`tracedFaces`, `tracedTotal`, `triangles` and `bytes`, or null.

## Area-light shading

`setAreaLightShading(settings)` selects how area lights are shaded. The setting is
engine-wide (every view) and persists until changed; a new Engine starts at `'ltc'`.

```ts
type AreaLightShadingMode = 'ltc' | 'formFactor'
interface AreaLightShadingDesc { mode: AreaLightShadingMode }
```

- `'ltc'` integrates diffuse and GGX specular over the rectangle with the LTC tables.
- `'formFactor'` is the cheap far or low option: the diffuse polygon form factor
  only, with no table and no specular.

`areaLightShading()` returns the current `{ mode }`. A non-object argument, a missing
`mode`, or any other value throws `api.invalid_argument` with the message
`setAreaLightShading: 'mode' must be 'ltc' or 'formFactor'`; a refused call changes
nothing.

## Area-light soft shadows

`setAreaLightShadows(settings)` selects how area lights are shadowed. The setting is
engine-wide (every view), persists until changed, and applies to every view.

```ts
type AreaLightShadowTier = 'auto' | 'hero' | 'cone' | 'fade'
type AreaLightStochasticMode = 'stratified' | 'restir'
interface AreaLightShadowsDesc {
  tier?: AreaLightShadowTier     // default 'auto'
  heroCount?: number             // integer 0..4, default 0 (no hero maps)
  heroDistance?: number          // metres, > 0, default 25
  fadeDistance?: number          // metres, > heroDistance, default 80
  band?: number                  // metres, > 0 and <= heroDistance, default 5
  stochasticCount?: number       // integer 0..8, default 4 (0 disables the mid tier)
  stochasticDistance?: number    // metres, > 0, default 40
  stochasticCells?: 4 | 9        // strata of the light rectangle, default 4
  stochasticMode?: AreaLightStochasticMode  // default 'stratified'
}
interface AreaLightShadows {
  tier: AreaLightShadowTier; heroCount: number; heroDistance: number
  fadeDistance: number; band: number; activeHeroes: number
  stochasticCount: number; stochasticDistance: number; stochasticCells: number
  stochasticMode: AreaLightStochasticMode
  activeStochastic: number
}
```

Absent keys keep their current value. The merged settings are validated as a whole.
The defaults are `{ tier: 'auto', heroCount: 0, heroDistance: 25, fadeDistance: 80,
band: 5, stochasticCount: 4, stochasticDistance: 40, stochasticCells: 4,
stochasticMode: 'stratified' }`; all distances are metres.

- `'hero'`: perspective shadow maps of the `heroCount` area lights with the largest
  screen coverage (hysteresis 1.25), filtered with PCSS whose light size is the
  rectangle size. It works with GI off.
- `'cone'`: soft visibility through the GI voxel distance field. Without GI the
  visibility is 1.
- `'fade'`: no direct term while GI is active, so the light reaches the scene
  through the GI volume only.
- `'auto'`: hero maps for the selected lights nearer than `heroDistance`, blended to
  the cone over `band` metres; the direct term is faded out over `band` metres before
  `fadeDistance` while GI is active. While GI is active with `localLightShadows` on,
  `'auto'` also uses the mid tier: one stratified shadow ray per pixel and light toward
  a 2x2 or 3x3 cell of the rectangle, traced against the screen depth pyramid, the GI
  distance field and SDF trace content, combined with the analytic light by the ratio
  estimator and denoised spatially and temporally, for up to `stochasticCount` lights
  nearer than `stochasticDistance`, blended to the cone tier over `band` metres.

`stochasticCount` is the number of area lights per view that get the mid tier; 0
disables the tier. Beyond `stochasticDistance`, after the `band`, lights use the cone
tier. `stochasticCells` is the strata of the light rectangle, 4 (2x2) or 9 (3x3).
`stochasticMode` selects the stochastic tier: `'stratified'` (default) is the M6 mid
tier, and `'restir'` is the ultra tier, ReSTIR DI with compute-traced visibility in the
same `gi.area-shadow` pass. The mid tier requires GI active with `localLightShadows` on;
without it the tier falls back to the cone.

`areaLightShadows()` reports `stochasticMode` between `stochasticCells` and
`activeStochastic`.

`activeHeroes` is the number of area lights that had a hero shadow map in the last
prepared frame. `activeStochastic` is the number of area lights that held a mid-tier
slot in the last prepared frame. Both are 0 before the first frame.

Refusals throw `light.invalid_area_shadows` and change nothing. A non-object argument
(`null`, an array or a primitive) gives `settings: must be an object`. A present `tier`
that is not one of the four strings gives `tier: must be 'auto', 'hero', 'cone' or
'fade'`. A present `heroCount`, `heroDistance`, `fadeDistance`, `band`,
`stochasticCount`, `stochasticDistance` or `stochasticCells` that is not a number
(`undefined` and `null` included) gives `<key>: must be a number`. A `heroCount`
number that is not an integer in [0, 4] gives `heroCount: must be an integer in [0, 4]
(got <value>)`. A `stochasticCount` number that is not an integer in [0, 8] gives
`stochasticCount: must be an integer in [0, 8] (got <value>)`. A `stochasticCells`
number other than 4 or 9 gives `stochasticCells: must be 4 or 9 (got <value>)`, and a
`stochasticDistance` that is not finite or not > 0 gives
`stochasticDistance: must be finite and > 0 (got <value>)`. A `stochasticMode` string
other than `'stratified'` or `'restir'`, or any non-string, gives
`stochasticMode: must be 'stratified' or 'restir'`. A merged state that breaks
the distance rules gives `<key>: <reason>`, for example
`fadeDistance: must be finite and > heroDistance (got 10)`.

## Lifecycle

`addLight(desc)` returns a positive safe-integer `LightHandle`. The engine holds
at most 256 local lights; adding a 257th refuses with `lights.invalid` and a
capacity message. Removed handles are never reused, and exhausted safe-integer
handles are refused. The existing sun is independent and continues to use
`setLight`.

`updateLight(handle, changes)` merges fields onto the current description and
validates the complete result before applying any change. Omitted fields retain
their current values, including `radius` and `shadows`. A property explicitly set
to `undefined` resets that field to its default; for example,
`updateLight(id, { innerAngleDeg: undefined })` restores the implicit 80% angle.
An outer-angle change is refused if a retained explicit inner angle would exceed
it. Changing kind keeps other authored fields. Updating an attached light's
`position` is refused; use `attachLight` to change its offset. A rejected update
leaves the light unchanged.

`attachLight(handle, instanceId, localOffset)` uses the public instance-group
handle returned by `addInstance` or `addMeshInstance`. The light position is the
instance world transform applied to `localOffset`; its authored direction is
transformed by the instance's linear transform and normalized. Object scale does
not change light range or size. Reattachment replaces the parent and offset
atomically. `detachLight` preserves the latest world position and direction.
Removing a parent detaches its lights at their last valid world transform;
removing a light removes its attachment. Invalid attachment changes do not
partially apply.

## Statistics and errors

`lightStats()` is synchronous and never waits for the GPU. It returns
`{ activeLights, attachedLights, tileColumns, tileRows, tileCapacity,
droppedLightReferences, statsFrameIndex, pending }`. `tileCapacity` is 127;
dimensions describe viewport 0. Overflow counts excluded light/tile references,
summed over the submitted views of one invocation. The result is published only
after all readbacks for that invocation complete. Before a completed result,
`droppedLightReferences` and `statsFrameIndex` are `null`; `pending` indicates an
outstanding readback. Counts reflect current registry state independently of
readback completion.

Methods throw `RaaError`. `light.invalid_desc` covers malformed descriptions,
including non-finite or out-of-range values. `api.invalid_argument` covers an
invalid numeric light handle; `api.unknown_handle` covers a valid but unknown or
removed light handle. `light.attached_position` rejects a position update on an
attached light. `light.invalid_attachment` covers invalid offsets, parents or
transformed directions. Capacity refusal uses `lights.invalid`. Calls after
engine disposal use `api.disposed`.
