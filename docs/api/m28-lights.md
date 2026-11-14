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
}
```

## Descriptions and defaults

`LocalLightDesc` fields are optional. Defaults are `kind: 'point'`,
`position: [0, 0, 0]`, `direction: [0, 0, -1]`, `color: [1, 1, 1]`,
`intensity: 2`, `range: 6`, `angleDeg: 35`, `innerAngleDeg: undefined`,
`size: [1, 1]`, `radius: 0` and `shadows: false`. Position, range, size and
radius use metres. JS `color` is sRGB in [0, 1] and is converted once to linear
RGB; intensity is nonnegative. Direction must be finite and nonzero and is
normalized when accepted.

`angleDeg` and `innerAngleDeg` are full cone angles in degrees, each in [1, 179].
An explicit inner angle cannot exceed the outer angle. When omitted, the effective
inner angle is 80% of the outer angle. `size` contains positive full width and
height values. All numeric values must be finite; colours must be in [0, 1].
`radius` and `shadows` are retained authoring metadata only: radius does not
control shadows, and local lights do not cast shadows.

An area light emits radiance intensity / (width x height) per unit colour over its
rectangle, so intensity is radiance x area.

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
