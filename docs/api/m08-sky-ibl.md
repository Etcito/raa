# RAA API m08: sky and environment lighting

Additions of chain milestone 2 to `@etcito/raa`: a procedural sky with clouds, and the
image-based lighting derived from it. Types: `packages/raa/types/m08-sky-ibl.d.ts` (appended to
`index.d.ts` when the package is staged, so `Engine` merges with its v0 declaration).

Everything in `docs/API.md` (v0) and `docs/api/m07-meshes-hdr.md` stays valid. Conventions are
those of v0: right-handed, Y up, the camera looks down -Z, metres, radians, front faces
counter-clockwise. Colours are scene-referred **linear HDR** RGB triples (values above 1 are
allowed) unless a field says otherwise. Every failure throws a `RaaError` whose `code` is listed
per method; after `dispose` every method throws `api.disposed`.

## Rendering model

The sky is both a background image and the light source of the scene. From the current sky the
engine builds an **environment cube** (256 px per face, `HDR_FORMAT`, 6 mips, `SKY_ENV`) and from
it an **irradiance** signal (spherical harmonics, order 2) and **prefiltered specular**
reflections with a BRDF LUT (`IBL`). Opaque objects are lit by the sun and by that environment;
glossy objects reflect it. The sky itself is drawn as the background behind everything else.

Two rows of the pipeline take part (`S12`), both enabled by default:

| order | name | role | what it does |
|---|---|---|---|
| 10 | `sky.env` | pass | Rebuilds the environment cube, the irradiance SH and the prefiltered specular from the current sky. Runs before `prepass`. |
| 900 | `sky.background` | contributor of `forward` | Draws the sky (gradient, sun disc, clouds) where nothing is drawn, inside the forward pass. |

`sky.env` runs once per sky change by default (`envUpdate` below) and again when a light changes
the sun. It is the source of the image-based lighting; `sky.background` is only the visible image.

### Disabling with `setPassEnabled`

- `setPassEnabled('sky.env', false)`: the environment resources reset to neutral dummies, so the
  scene falls back to flat ambient lighting; the sky is still drawn by `sky.background`.
- `setPassEnabled('sky.background', false)`: the sky image is not drawn, so the background shows
  the flat background colour (`setBackground`, default CLEAR_COLOR); image-based lighting still
  applies while `sky.env` is enabled.
- Both disabled: flat ambient lighting and the flat background colour, as before milestone 2.

Codes: `pipeline.unknown_pass`, `pipeline.required_pass`.

## Sky

### `SkyPreset`

`'day' | 'snow' | 'jungle' | 'night'`

A built-in base for every `SkyDesc` field. `'day'` is the default and its values are the defaults
listed below. The other presets change the mood (a high-key overcast for `'snow'`, a warm hazy
green cast for `'jungle'`, a dark blue sky with a dim ground for `'night'`); any field given
alongside a preset overrides it. An unknown name throws `sky.invalid` naming `preset`.

### `CloudsDesc`

```ts
interface CloudsDesc {
  coverage?: number;      // [0, 1]
  density?: number;       // >= 0
  altitude?: number;      // metres, > 0
  scale?: number;         // metres, > 0
  wind?: [number, number]; // [x, z] metres per second
  color?: Vec3;           // linear RGB
  silverLining?: number;  // [0, 1]
}
```

The procedural cloud layer of `SkyDesc`. Every field is optional and merged over the preset's
clouds (default: the `day` values below). Refusals throw `sky.invalid` naming the field.

| field | unit | default (`day`) | meaning |
|---|---|---|---|
| `coverage` | fraction | `0.5` | Fraction of the sky covered by clouds, in [0, 1]. |
| `density` | - | `0.5` | Optical density of the layer, finite and >= 0; 0 is a thin veil. |
| `altitude` | metres | `1200` | Cloud base altitude, finite and > 0. |
| `scale` | metres | `1000` | Horizontal scale of the cloud pattern, finite and > 0. |
| `wind` | m/s | `[1, 0.5]` | Wind velocity `[x, z]`, both finite; drifts the pattern with time. |
| `color` | linear RGB | `[1, 1, 1]` | Cloud colour. |
| `silverLining` | - | `0.5` | Bright rim around dense cloud edges, in [0, 1]. |

### `SkyDesc`

```ts
interface SkyDesc {
  preset?: SkyPreset;
  zenith?: Vec3;            // linear RGB
  horizon?: Vec3;           // linear RGB
  ground?: Vec3;            // linear RGB
  intensity?: number;       // >= 0
  horizonSoftness?: number; // [0, 1]
  sunDiscSize?: number;     // angular radius, radians
  sunDiscIntensity?: number; // >= 0
  clouds?: CloudsDesc;
  envUpdate?: 'onChange' | 'everyFrame';
}
```

The argument of `setSky` and the result of `sky`. Every field is optional: the `preset` is applied
first, then the remaining fields are merged over it.

| field | unit | default (`day`) | meaning |
|---|---|---|---|
| `preset` | - | `'day'` | Built-in base values. |
| `zenith` | linear RGB | `[0.2, 0.42, 1.0]` | Sky colour straight up. |
| `horizon` | linear RGB | `[0.7, 0.8, 0.95]` | Sky colour at the horizon. |
| `ground` | linear RGB | `[0.22, 0.2, 0.18]` | Colour below the horizon. |
| `intensity` | - | `1` | Overall sky radiance multiplier, finite and >= 0. |
| `horizonSoftness` | - | `0.15` | Blend width of the zenith-to-horizon gradient, in [0, 1]: 0 a hard band, 1 smooth. |
| `sunDiscSize` | radians | `0.0047` | Angular radius of the sun disc, finite and > 0 (0.0047 rad is the real sun). |
| `sunDiscIntensity` | - | `20` | Radiance multiplier of the sun disc, finite and >= 0. |
| `clouds` | - | see `CloudsDesc` | Cloud layer; merged field by field over the preset's clouds. |
| `envUpdate` | - | `'onChange'` | `'onChange'`: rebuild the environment only when the sky or the light changes. `'everyFrame'`: rebuild every frame (animated clouds, higher cost). |

Colours are linear HDR RGB triples: each component finite and >= 0. A wrong type, a non-finite
number or a value outside its range throws `sky.invalid` naming the field.

### `setSky`

`engine.setSky(desc: SkyDesc): void`

Validates `desc` and applies it: the `preset` selects the base values, every other given field
overrides them, and omitted fields keep the preset's value. A refusal changes nothing. Codes:
`sky.invalid` (a field is out of range, non-finite or of the wrong type; the message names the
field), `api.invalid_argument` (the argument is not an object), `api.feature_missing` (no sky
feature is installed).

### `sky`

`engine.sky(): SkyDesc`

The sky in effect, with every field resolved to a concrete value (the preset merged with the
overrides given to `setSky`), including a fully resolved `clouds` object. Code: `api.feature_missing`
(no sky feature is installed).

## Environment lighting

### `EnvironmentLighting`

```ts
interface EnvironmentLighting {
  diffuse?: number;   // >= 0
  specular?: number;  // >= 0
}
```

The two intensities of the image-based lighting derived from the sky. Every field is optional and
merged over the current values (default 1 for both); `diffuse` scales the irradiance from the sky,
`specular` the prefiltered reflections. Both are finite and >= 0; 0 removes that part of the sky
light without changing the visible sky.

### `setEnvironmentLighting`

`engine.setEnvironmentLighting(settings: EnvironmentLighting): void`

Merges `settings` over the current environment lighting (`setEnvironmentLighting({ specular: 0 })`
keeps the diffuse term). A refusal changes nothing. Codes: `api.invalid_argument` (a value is not
finite or < 0, or the argument is not an object), `api.feature_missing` (no environment lighting
feature is installed).

## Refusals

| code | thrown by | when |
|---|---|---|
| `sky.invalid` | `setSky` | A `SkyDesc` or `CloudsDesc` field is out of range, non-finite or of the wrong type; the message names the field. |
| `api.invalid_argument` | `setSky`, `setEnvironmentLighting` | The argument is not an object, or an `EnvironmentLighting` value is non-finite or < 0. |
| `api.feature_missing` | `setSky`, `sky`, `setEnvironmentLighting` | The sky or environment lighting feature is not installed in this engine. |
| `api.disposed` | all | Any method after `dispose`. |

## Example

```js
const engine = await raa.createEngine(canvas, { msaa: 4, toneMapper: 'agx' });
engine.setSky({
  preset: 'day',
  sunDiscIntensity: 24,
  clouds: { coverage: 0.6, wind: [2, 0.5] },
  envUpdate: 'everyFrame',
});
engine.setEnvironmentLighting({ diffuse: 1, specular: 1 });
console.log(engine.sky().zenith, engine.passes().map((p) => p.name));
engine.setPassEnabled('sky.background', false); // keep the sky light, drop the sky image
```
