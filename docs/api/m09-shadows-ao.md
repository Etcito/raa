# RAA API m09: shadows and ambient occlusion

Additions of chain milestone 3 to `@etcito/raa`: soft cascaded sun shadows, horizon-based ambient
occlusion (GTAO) and screen-space contact shadows. Types:
`packages/raa/types/m09-shadows-ao.d.ts` (appended to `index.d.ts` when the package is staged, so
the `Engine` interface merges with its v0 declaration).

Everything in `docs/API.md` (v0), `docs/api/m07-meshes-hdr.md` and `docs/api/m08-sky-ibl.md` stays
valid. Conventions are those of v0: right-handed, Y up, the camera looks down -Z, metres, radians,
front faces counter-clockwise. Colours are scene-referred **linear HDR** RGB triples unless a
field says otherwise. Every failure throws a `RaaError` whose `code` is listed per method; after
`dispose` every method throws `api.disposed`.

## Rendering model

Three rows of the pipeline take part (`S12`), each enabled by default and each feeding the forward
pass through group 1 (`S10`):

| order | name | role | what it does |
|---|---|---|---|
| 20 | `shadow.csm` | pass | Renders a stable 4-cascade sun shadow map before `prepass`. |
| 40 | `ao.gtao` | pass | Computes full-resolution ambient occlusion after `prepass`, before `forward`. |
| 45 | `shadow.contact` | pass | Computes 24-step screen-space contact shadows after `prepass`, before `forward`. |

The forward pass reads the shadow map and comparison sampler, the AO texture and the contact
texture. `setShadows`, `setAo` and `setContactShadows` change the parameters of their pass;
enabling and disabling a feature stays `setPassEnabled('shadow.csm' | 'ao.gtao' |
'shadow.contact', on)` (codes `pipeline.unknown_pass`, `pipeline.required_pass`).

### Disabling with `setPassEnabled`

- `setPassEnabled('shadow.csm', false)`: the shadow resources reset to dummies (no sun shadow),
  so only ambient and image-based lighting remain.
- `setPassEnabled('ao.gtao', false)`: the AO texture resets to 1, so creases receive the full
  ambient light.
- `setPassEnabled('shadow.contact', false)`: the contact texture resets to 1, so small objects
  lose their grounding shadow.

## Shadow settings

### `ShadowSettings`

```ts
interface ShadowSettings {
  softness?: number;                    // > 0 and <= 16, multiplier of the sun angular radius
  distance?: number;                    // metres, > 0
  resolution?: 1024 | 2048 | 4096;      // texels, edge length of one cascade
}
```

The argument of `setShadows`. Every field is optional and merged over the current values. The
algorithm is a **4-cascade stable cascaded shadow map (CSM) with percentage-closer soft shadows
(PCSS)**: the view frustum is split with the practical split scheme (lambda 0.75), each cascade
renders into one slice of a `Depth32Float` array, and the sampled cascade is snapped to whole
texels in light space so the map does not swim as the camera moves. PCSS widens the penumbra with
occluder distance (16 blocker + 32 PCF samples, slope bias 2.0, constant bias 2, normal offset
1.5 texels) and a 10% blend band hides the cascade boundary.

| field | unit | default | valid range | meaning |
|---|---|---|---|---|
| `softness` | - | `1` | > 0 and <= 16 | Multiplier of the sun's angular radius used as the PCSS penumbra; 1 matches the real sun (0.0047 rad), larger values widen the soft shadow. |
| `distance` | metres | `60` | > 0 | Distance from the camera to the far cascade split. |
| `resolution` | texels | `2048` | `1024`, `2048` or `4096` | Edge length of one cascade's shadow map. |

A wrong type, a non-finite number or a value outside its range throws `shadow.invalid` naming the
field.

### `setShadows`

`engine.setShadows(settings: ShadowSettings): void`

Validates `settings` and merges it over the current values (omitted fields keep their current
value). A refusal changes nothing. Codes: `shadow.invalid` (a field is out of range, non-finite or
of the wrong type; the message names the field), `api.invalid_argument` (the argument is not an
object), `api.feature_missing` (no shadow feature is installed).

## Ambient occlusion

### `AoSettings`

```ts
interface AoSettings {
  radius?: number;     // metres, > 0 and <= 10
  intensity?: number;  // [0, 1]
  power?: number;      // > 0 and <= 8
}
```

The argument of `setAo`. Every field is optional and merged over the current values. The
algorithm is **full-resolution ground-truth ambient occlusion (GTAO)**: 4 slices x 8 steps per
side over the depth buffer, followed by **2 separable 7-tap depth-aware denoise passes** that
smooth the raw signal without bleeding across depth edges.

| field | unit | default | valid range | meaning |
|---|---|---|---|---|
| `radius` | metres | `0.8` | > 0 and <= 10 | World-space sampling radius. |
| `intensity` | - | `1` | `[0, 1]` | Strength of the AO darkening: 0 leaves the ambient light unchanged, 1 applies it fully. |
| `power` | - | `1.5` | > 0 and <= 8 | Exponent shaping the AO falloff. |

A wrong type, a non-finite number or a value outside its range throws `ao.invalid` naming the
field.

### `setAo`

`engine.setAo(settings: AoSettings): void`

Validates `settings` and merges it over the current values. A refusal changes nothing. Codes:
`ao.invalid` (a field is out of range, non-finite or of the wrong type; the message names the
field), `api.invalid_argument` (the argument is not an object), `api.feature_missing` (no ambient
occlusion feature is installed).

## Contact shadows

### `ContactShadowSettings`

```ts
interface ContactShadowSettings {
  distance?: number;   // metres, > 0 and <= 5
  thickness?: number;  // metres, > 0 and <= 1
  intensity?: number;  // [0, 1]
}
```

The argument of `setContactShadows`. Every field is optional and merged over the current values.
The algorithm is **24-step screen-space contact shadows**: a ray is marched in screen space from
each fragment towards the sun against the depth buffer, and a hit within `thickness` darkens the
fragment; this grounds small objects and sharpens the shadow at contact where the cascaded shadow
map is too coarse.

| field | unit | default | valid range | meaning |
|---|---|---|---|---|
| `distance` | metres | `0.35` | > 0 and <= 5 | Maximum screen-space ray-march distance. |
| `thickness` | metres | `0.05` | > 0 and <= 1 | Assumed occluder thickness. |
| `intensity` | - | `1` | `[0, 1]` | Strength of the contact darkening. |

A wrong type, a non-finite number or a value outside its range throws `shadow.invalid` naming the
field.

### `setContactShadows`

`engine.setContactShadows(settings: ContactShadowSettings): void`

Validates `settings` and merges it over the current values. A refusal changes nothing. Codes:
`shadow.invalid` (a field is out of range, non-finite or of the wrong type; the message names the
field), `api.invalid_argument` (the argument is not an object), `api.feature_missing` (no contact
shadow feature is installed).

## Refusals

| code | thrown by | when |
|---|---|---|
| `shadow.invalid` | `setShadows`, `setContactShadows` | A `ShadowSettings` or `ContactShadowSettings` field is out of range, non-finite or of the wrong type; the message names the field. |
| `ao.invalid` | `setAo` | An `AoSettings` field is out of range, non-finite or of the wrong type; the message names the field. |
| `api.invalid_argument` | `setShadows`, `setAo`, `setContactShadows` | The argument is not an object. |
| `api.feature_missing` | `setShadows`, `setAo`, `setContactShadows` | The shadow, ambient occlusion or contact shadow feature is not installed in this engine. |
| `api.disposed` | all | Any method after `dispose`. |

## Example

```js
const engine = await raa.createEngine(canvas, { msaa: 4, toneMapper: 'agx' });
engine.setShadows({ softness: 2, distance: 80, resolution: 4096 });
engine.setAo({ radius: 1.0, intensity: 0.9, power: 1.5 });
engine.setContactShadows({ distance: 0.5, thickness: 0.08, intensity: 1 });
console.log(engine.passes().map((p) => p.name));
engine.setPassEnabled('ao.gtao', false); // keep the sun shadows, drop the ambient occlusion
```
