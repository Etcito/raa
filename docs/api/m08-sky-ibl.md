# RAA API m08: sky and environment lighting

Additions of chain milestone 2 to `@etcito/raa`: a procedural sky with clouds, the celestial
time of day, the typed sky/weather lifecycle diagnostics and the image-based lighting derived
from the sky. Types: `packages/raa/types/m08-sky-ibl.d.ts` (appended to `index.d.ts` when the
package is staged, so `Engine` merges with its v0 declaration).

Everything in `docs/API.md` (v0) and `docs/api/m07-meshes-hdr.md` stays valid. Conventions are
those of v0: right-handed, Y up, the camera looks down -Z, metres, radians, front faces
counter-clockwise. Colours are scene-referred **linear HDR** RGB triples (values above 1 are
allowed) unless a field says otherwise. Every failure throws a `RaaError` whose `code` is listed
per method; after `dispose` every method throws `api.disposed`.

## Rendering model

The sky is both a background image and the light source of the scene. From the current sky the
engine builds an **environment cube** (256 px per face, `HDR_FORMAT`, 6 mips, `SKY_ENV`) and from
it an **irradiance** signal (spherical harmonics, order 2) and **prefiltered specular**
reflections with a BRDF LUT (`IBL`).  (default 1, finite in [0.25, 4]) scales the
screen-cloud march step count (56 at 1, clamped to 8..224); it is not part of the  text.
Opaque objects are lit by the sun and by that environment;
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

### `SkyModel`

`'physical' | 'legacy'`

`'physical'` feeds the common physical atmosphere and celestial evaluator; `'legacy'` keeps the
original gradient/layer evaluator. The two share one sky feature, one public description and the
same `sky.env`/`sky.background` rows. Physical interpretation treats the artistic colours and
`intensity` relative to the selected `profile`'s original values, so preset defaults never
double-tint the atmosphere and an explicit override stays effective; the legacy model consumes
the authored values verbatim. `profile` is the resolved preset identity the physical model uses
as that reference. A `model`-only switch keeps every authored value and the authored weather.

### `SkyPreset`

`'day' | 'snow' | 'jungle' | 'night'`

A built-in base for every `SkyDesc` field. `'day'` is the default and its values are the defaults
listed below. The other presets change the mood (a high-key overcast for `'snow'`, a warm hazy
green cast for `'jungle'`, a dark blue sky with a dim ground for `'night'`); any field given
alongside a preset overrides it. An unknown name throws `sky.invalid` naming `preset`.

In the physical model an explicit preset also selects its S24 moment and weather defaults
(`day` → fair/09:00, `snow` → overcast/12:00, `jungle` → broken/15:00, `night` → clear/00:00,
day 172) while retaining the accepted place, automatic/playing flags and speed. That selection
is an explicit event even when the resolved values already match the accepted state. In the
legacy model a preset keeps its original gradient/layer behaviour.

### `CloudsDesc`

```ts
interface CloudsDesc {
  coverage?: number;       // [0, 1]
  density?: number;        // >= 0
  altitude?: number;       // metres, > 0
  scale?: number;          // metres, > 0
  wind?: [number, number]; // [x, z] metres per second
  color?: Vec3;            // linear RGB
  silverLining?: number;   // [0, 1]
}
```

The procedural cloud layer of `SkyDesc`. Every field is optional and merged over the preset's
clouds (default: the `day` values below). Refusals throw `sky.invalid` naming the complete field
path (for example `clouds.wind[1]`).

| field | unit | default (`day`) | meaning |
|---|---|---|---|
| `coverage` | fraction | `0.45` | Fraction of the sky covered by clouds, in [0, 1]. |
| `density` | - | `1` | Optical density of the layer, finite and >= 0; 0 is a thin veil. |
| `altitude` | metres | `1500` | Cloud base altitude, finite and > 0. |
| `scale` | metres | `2500` | Horizontal scale of the cloud pattern, finite and > 0. |
| `wind` | m/s | `[4, 1.5]` | Wind velocity `[x, z]`, both finite; drifts the pattern with time. |
| `color` | linear RGB | `[0.95, 0.95, 0.97]` | Cloud colour. |
| `silverLining` | - | `0.8` | Bright rim around dense cloud edges, in [0, 1]. |

`coverage`, `density`, `altitude` and `wind` are the shared legacy fields: in the physical model
they update the corresponding authored weather target only when the patch supplies them (see
`setSky`). `scale`, `color` and `silverLining` remain artistic volume controls.

`null` (and `undefined`) is an omission for these old fields.

### `AtmosphereDesc`

```ts
interface AtmosphereDesc {
  groundAlbedo?: Vec3;    // each component in [0, 1]
  mieMultiplier?: number; // > 0
}
```

Physical atmosphere controls, merged field by field over the preset. `groundAlbedo` defaults to
`[0.3, 0.3, 0.3]` and `mieMultiplier` to `1`. These are S24 fields: `null`, unknown keys, wrong
types, out-of-domain values and values that overflow f32 throw `sky.invalid` with the complete
path (`atmosphere.groundAlbedo[1]`, `atmosphere.mieMultiplier`). They are stored and applied in
both models; the legacy evaluator ignores them.

### `SkyDesc`

```ts
interface SkyDesc {
  preset?: SkyPreset;
  model?: SkyModel;
  zenith?: Vec3;            // linear RGB
  horizon?: Vec3;           // linear RGB
  ground?: Vec3;            // linear RGB
  intensity?: number;       // >= 0
  horizonSoftness?: number; // > 0
  sunDiscSize?: number;     // angular radius, radians, [0, 0.2]
  sunDiscIntensity?: number; // >= 0
  clouds?: CloudsDesc;
  envUpdate?: 'onChange' | 'everyFrame';
  atmosphere?: AtmosphereDesc;
}
```

The argument of `setSky`. Every field is optional: an explicit `preset` supplies the base
descriptor, then the remaining fields are merged over it. Without a preset the current accepted
sky is the base, so an empty patch changes nothing.

| field | unit | default (`day`) | meaning |
|---|---|---|---|
| `preset` | - | `'day'` | Built-in base values and, when explicitly supplied, a preset event. |
| `model` | - | `'physical'` | `'physical'` common atmosphere or the retained `'legacy'` gradient. |
| `zenith` | linear RGB | `[0.15, 0.35, 0.85]` | Sky colour straight up. |
| `horizon` | linear RGB | `[0.85, 0.8, 0.7]` | Sky colour at the horizon. |
| `ground` | linear RGB | `[0.35, 0.3, 0.25]` | Colour below the horizon. |
| `intensity` | - | `1.6` | Overall sky radiance multiplier, finite and >= 0. |
| `horizonSoftness` | - | `4` | Exponent of the zenith-to-horizon blend, finite and > 0. |
| `sunDiscSize` | radians | `0.0047` | Angular radius of the sun disc, in [0, 0.2] (0.0047 rad is the real sun). |
| `sunDiscIntensity` | - | `40` | Radiance multiplier of the sun disc, finite and >= 0. |
| `clouds` | - | see `CloudsDesc` | Cloud layer; merged field by field over the preset's clouds. |
| `envUpdate` | - | `'onChange'` | `'onChange'`: rebuild the environment only when the sky or the light changes. `'everyFrame'`: rebuild every frame (animated clouds, higher cost). |
| `atmosphere` | - | see `AtmosphereDesc` | Physical ground albedo and Mie multiplier. |

Colours are linear HDR RGB triples: each component finite and >= 0. Every numeric value must
remain finite when stored as f32, so values that overflow f32 throw `sky.invalid`. A wrong type,
an out-of-range value or an unknown key throws `sky.invalid` naming the complete field path.
Omitted, `undefined` and (for the old fields) `null` all mean omission.

### `ResolvedSkyDesc`

```ts
interface ResolvedSkyDesc {
  model: SkyModel;
  profile: SkyPreset;
  zenith: Vec3;
  horizon: Vec3;
  ground: Vec3;
  intensity: number;
  horizonSoftness: number;
  sunDiscSize: number;
  sunDiscIntensity: number;
  clouds: Required<CloudsDesc>;
  envUpdate: 'onChange' | 'everyFrame';
  atmosphere: Required<AtmosphereDesc>;
}
```

The complete accepted sky returned by `sky()`. `profile` is the resolved preset identity (the
physical reference profile); the explicit `preset` intent is not part of a resolved result.

### `setSky`

`engine.setSky(desc: SkyDesc): void`

Validates `desc` and applies it as **one atomic transaction**. The explicit `preset` selects the
base values first, every other given field overrides them, and omitted fields keep the base
value. The transaction keeps two pieces of intent independent of resolved equality:

- **explicit preset presence**: `setSky({ preset: 'day' })` is a preset event (it resets the
  preset moment, for example after a time edit), while `setSky({})` is not and retains the
  authored moment. This holds even when both calls resolve to the identical `SkyDesc`.
- **shared-cloud presence**: `clouds.coverage`, `clouds.density`, `clouds.altitude` and
  `clouds.wind` are forwarded only when the patch actually supplies them. An omitted field never
  resets or overrides a later authored weather edit; a supplied one applies after the preset, so
  later setters keep precedence.

A model-only switch (`setSky({ model: 'legacy' })`) keeps the authored weather and every other
value. The complete merged descriptor is validated before any sky, time, weather, light, dirty
flag or resource changes; a refusal leaves the previous state untouched. Codes: `sky.invalid`
(a field is out of range, non-finite, overflows f32, of the wrong type or unknown; the message
names the complete field path), `api.invalid_argument` (the argument is not an object),
`api.feature_missing` (no sky feature is installed), `pipeline.busy` (called while a frame
executes).

### `sky`

`engine.sky(): ResolvedSkyDesc`

The sky in effect, with every field resolved to a concrete value (the preset merged with the
overrides given to `setSky`), including a fully resolved `clouds` object and the physical
`atmosphere` controls. The result is a fresh object: mutating it never changes the engine. Code:
`api.feature_missing` (no sky feature is installed).

## Celestial time

### `TimeOfDaySettings`

```ts
interface TimeOfDaySettings {
  day: number;      // integer [1, 365]
  hours: number;    // [0, 24)
  latitude: number; // degrees north [-90, 90]
  north: number;    // radians about +Y
  enabled: boolean; // automatic sun/moon light
  playing: boolean; // advance the clock with the frame delta
  speed: number;    // simulated seconds per real second, >= 0
}
```

| field | unit | default | meaning |
|---|---|---|---|
| `day` | day of year | `172` | One-based day in [1, 365]. |
| `hours` | local solar hours | `9` | Finite and in [0, 24); noon is 12. |
| `latitude` | degrees north | `38` | Finite and in [-90, 90]. |
| `north` | radians | `0` | Scene orientation about +Y; zero puts north at -Z and east at +X. |
| `enabled` | - | `false` | Automatic dominant light: the sun/moon direction wins, the manual descriptor is the stored daylight baseline. |
| `playing` | - | `false` | Advance the runtime clock; it only matters while `enabled` is true. |
| `speed` | s/s | `1` | Simulated seconds per real second, finite and >= 0. |

`TimeOfDayDesc` is `Partial<TimeOfDaySettings>`. The authored moment (`day`/`hours`) and the
runtime cursor (`currentDay`/`currentHours`) are separate: changing an authored day or hour resets
the cursor to that moment, while pause/resume and speed/place/orientation edits keep it. Save the
authored settings, never the current cursor or frame counters.

### `TimeOfDayState`

```ts
interface TimeOfDayState extends TimeOfDaySettings {
  currentDay: number;
  currentHours: number;
  sunDirection: Vec3;  // toward the sun
  moonDirection: Vec3; // toward the moon
  moonLit: number;
  starPole: Vec3;
  starEquinox: Vec3;
  dominant: 'manual' | 'sun' | 'moon';
}
```

The authored settings plus the current playback moment and its celestial snapshot.
`sunDirection`/`moonDirection`/`starPole`/`starEquinox` point toward the bodies (RAA directional
lights store the opposite travel direction). `dominant` names the source of the accepted light.

### `setTimeOfDay`

`engine.setTimeOfDay(desc: TimeOfDayDesc): void`

Merges the supplied fields over the accepted authored settings; `undefined` is an omission and
the complete merged result is validated before anything changes. A changed authored `day` or
`hours` resets the runtime cursor to that moment; the manual light descriptor and the undimmed
daylight baseline remain owned by the sky. A physical `setSky` preset applies its moment and
weather defaults first and retains the accepted place, `enabled`, `playing` and speed; a later
`setTimeOfDay` wins. Manual `setLight` stays effective while automatic control is disabled; while
it is enabled, the astronomical direction wins and the stored daylight baseline supplies colour
and intensity. Disabling restores the stored manual descriptor.

The clock advances once per Engine invocation: the engine applies
`FrameInput.dt_ms / 1000.0` under its monotonic sky invocation identity before the per-view sky
work. The first and nonpositive deltas advance neither clock; a repeated invocation identity
returns the accepted snapshot without advancing again; a nonfinite delta refuses before any
mutation. The per-view frame hook only observes the accepted light and marks the sky dirty; it
never advances time a second time and uses no submitted-frame counter.

Codes: `sky.invalid` (a field is non-finite, out of range, of the wrong type, unknown, or a
place value overflows f32; `day` must be an integer and `hours` in [0, 24); the message names
the complete field path), `api.invalid_argument` (the argument is not an object),
`api.feature_missing` (no sky feature is installed), `pipeline.busy` (called while a frame
executes).

### `timeOfDay`

`engine.timeOfDay(): TimeOfDayState`

The authored settings, current playback moment and celestial snapshot. The result is a fresh
object; mutating it never changes the engine, and reading it never advances the clock. Code:
`api.feature_missing` (no sky feature is installed).

## Weather

### `WeatherState`, `WeatherSettings`, `WeatherDesc`, `ResolvedWeather`

`WeatherState` is `{coverage, type, base, thickness, density, cirrus, wind:{speed, from},
evolution, rain}`: fractions (`coverage`, `type`, `cirrus`, `rain`) in [0, 1], `base` >= 0 and
`thickness` >= 100 metres, `density`/`evolution` >= 0, `wind.speed` >= 0 m/s and `wind.from` a
finite compass direction in degrees FROM north (rotated by `TimeOfDaySettings.north` at the
scene boundary). `base + thickness` must stay finite. `WeatherSettings` is
`{enabled, shadows, changing, seed, transitionSeconds, target}` with defaults `true`, `true`,
`false`, `1`, `0` and the fair target (coverage 0.3, type 0.9, base 1200, thickness 1600,
density 1, cirrus 0.15, wind 7 from 250, evolution 1, rain 0). `seed` is an integer u32 and
`transitionSeconds` a finite value >= 0 (0 is immediate; otherwise the state eases with
`1 - exp(-dt / transitionSeconds)`). `cloudSamples` (integer-valued number in [8, 256], default 56)
is the screen-cloud march step count per ray and `cloudScale` (finite in [0.25, 1], default 0.5) the
screen-cloud buffer scale: the buffer is `ceil(size * cloudScale)` texels, so 0.5 is half resolution
and 1 is full resolution. Changing either restarts the screen-cloud accumulation (and a changed scale
recreates the buffer, resetting its history); neither is part of the `.weather` text. The screen
clouds average successive frames: while the camera and the sky stand still the average converges
(blend 1/(n + 6) after n unchanged frames), and when nothing is animated (no wind, no cloud boil, no
weather transition or cycle, no playing time of day) after 64 unchanged frames the march stops
entirely (a cheap refresh pass every 8th frame keeps newly uncovered sky filled). Texels that enter
the view while the camera turns are marched with twice the steps.
`WeatherDesc` is the flat patch: `preset`
(`clear|fair|broken|overcast|storm|cirrus`), `text`, the settings fields and the state fields
with a partial `wind`. These records mirror Rust `raa_sky::WeatherSettings` (`type` is
`cloud_type`, `wind` is `wind_speed`/`wind_from`) and `raa_sky::ResolvedWeatherState`.

**Authored versus runtime.** `target` is what the caller authored; `current` is the runtime
state that eases towards the authored target, or towards the evolving cycle target while
`changing` is true. Persist the authored settings (`WeatherSettings`), never `current`,
`cyclePreset` or `text`. `text` is the `.weather` export of the authored `target`, generated
from `settings.target.to_text()` only when `weather()` is called; nothing stores or formats
text per frame.

### `setWeather`

`engine.setWeather(desc: WeatherDesc): void`

The patch is parsed completely into one owned candidate, then committed once through
`SkyHandle::set_weather_update`. `undefined` omits a field; omitted fields (including
`wind.speed`/`wind.from` separately) keep the authored values. `preset` and `text` together
reject. Either supplies the base before the structured overrides; `text` is parsed fully
before the candidate exists.

Cycle-start identity is intent, not equality: an explicit `preset` keeps that preset as the
cycle start even with overrides (`{preset:'storm', coverage:0.5}` starts from storm), and
`text` starts from fair because text has no preset identity. An explicit preset or text is a
reset/restart event even when every resolved value equals the current one; omitting `preset`
never resets. A changed `seed` or `changing` false→true also restarts the cycle; an unchanged
`changing: true` alone keeps the running cycle. A manual target edit stops changing unless the
same patch sets `changing: true`. With `changing` false the start is retained for a later
enable and `cyclePreset` is `null`. Codes: `sky.invalid` (wrong type, `null`, unknown key,
out-of-range/non-finite/f32-overflowing field, unknown preset, `preset` with `text`,
malformed text with a one-based line message; the message names the field path),
`api.invalid_argument` (not an object), `api.feature_missing`, `pipeline.busy`. A refusal
changes neither settings, cycle, light nor any dirty flag.

### `weather`

`engine.weather(): ResolvedWeather`

The authored settings flattened with `current`, `cyclePreset` (`null` without an active cycle)
and `text`, as a fresh object; reading it never advances weather. Code: `api.feature_missing`.

### Environment publication and the legacy model

Accepted weather changes mark the sky environment dirty. With `envUpdate: 'onChange'` the
environment/IBL is an authored snapshot republished on accepted changes only, so continuous
cloud drift and cycle evolution do not refresh reflections; select `'everyFrame'` when
reflections must follow drift. The background always evaluates the current state. The
`legacy` model keeps the full authored weather but renders it with its planar gradient/layer
approximation: volume-only properties (`type`, `thickness`, `cirrus`, evolution structure)
have no volumetric equivalent there, and no parity with the physical model is claimed.

### Pure text helpers

The package root exports `parseWeatherText(text): WeatherState` and
`formatWeatherText(weather): string` (declared here, implemented in
`packages/raa/sky-weather.js`). They need no Engine or GPU. Parsing follows the `.weather`
file rules (fair defaults, `#` comments, last value wins, file-format clamping, one-based line
errors); formatting writes coverage, type, base, thickness, density, cirrus, wind, evolution
and rain, one per line, with a final LF.

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

## Diagnostics

### `SkyWeatherDiagnostics`

```ts
interface SkyWeatherDiagnostics {
  invocation: number | null;
  submittedInvocation: number | null;
  weatherGeneration: number;
  atmosphereGeneration: number;
  environment: { requestedGeneration: number; publishedGeneration: number | null; pending: boolean };
  views: SkyViewDiagnostics[];
  readiness: SkyReadiness;
}
interface SkyViewDiagnostics {
  view: number; width: number; height: number;
  invocation: number | null; submittedInvocation: number | null;
  skyViewGeneration: number; historyValid: boolean;
  resetReasons: ('Resize' | 'CameraDiscontinuity' | 'Weather' | 'Model' | 'Enable' | 'ViewRemoved')[];
  cloudScreenValid: boolean; cloudPanoramaValid: boolean;
}
interface SkyReadiness { status: 'pending' | 'ready' | 'failed'; stage: string; error: string | null }
```

`engine.skyWeatherDiagnostics(): SkyWeatherDiagnostics`

The typed lifecycle observation of the sky pipeline: the invocation whose recording was
accepted, the last invocation with an actual submitted view, the weather/atmosphere counter
generations, the IBL environment publication, the prepared views and the readiness. It is
observational: reading it never advances a clock, changes a resource or becomes a saved setting,
and it allocates fresh objects only on request.

- `views` holds only prepared views, in ascending slot order; no prepared view yields `[]`.
  Removed views become absent after their `ViewRemoved` invalidation.
- `resetReasons` lists the set reset bits in the order `Resize`, `CameraDiscontinuity`,
  `Weather`, `Model`, `Enable`, `ViewRemoved`. A newly prepared view starts with no reasons
  until an actual invalidation.
- `readiness.stage` is one of `install`, `atmosphere`, `milkyWay`, `cloudLighting`,
  `cloudScreen`, `environment`, `submission` (see `SkyReadiness` for the exact union).
  Defaults are `pending`/`install`/`null`. A failure retains its originating `error` message
  until a new successful preparation or submission; recording or allocation alone never
  becomes `ready`.
- Counters are nonnegative safe JS integers. A value that could not be represented exactly
  throws `sky.invalid` instead of rounding; nothing is redefined or reused.

Codes: `sky.invalid` (a counter exceeds the safe JS integer range), `api.feature_missing`
(no sky feature is installed).

## Refusals

| code | thrown by | when |
|---|---|---|
| `sky.invalid` | `setSky`, `setTimeOfDay`, `setWeather`, `skyWeatherDiagnostics` | A `SkyDesc`, `CloudsDesc`, `AtmosphereDesc`, `TimeOfDayDesc` or `WeatherDesc` field is out of range, non-finite, overflows f32, of the wrong type or unknown (the message names the complete field path), or a diagnostics counter exceeds the safe JS integer range. |
| `api.invalid_argument` | `setSky`, `setTimeOfDay`, `setWeather`, `setEnvironmentLighting` | The argument is not an object, or an `EnvironmentLighting` value is non-finite or < 0. |
| `api.feature_missing` | `setSky`, `sky`, `setTimeOfDay`, `timeOfDay`, `setWeather`, `weather`, `skyWeatherDiagnostics`, `setEnvironmentLighting` | The sky or environment lighting feature is not installed in this engine. |
| `pipeline.busy` | `setSky`, `setTimeOfDay`, `setWeather` | Called while a frame executes (the sky state is borrowed). |
| `api.disposed` | all | Any method after `dispose`. |

## Example

```js
const engine = await raa.createEngine(canvas, { msaa: 4, toneMapper: 'agx' });
engine.setSky({
  preset: 'day',
  model: 'physical',
  sunDiscIntensity: 24,
  clouds: { coverage: 0.6, wind: [2, 0.5] },
  atmosphere: { groundAlbedo: [0.25, 0.25, 0.25], mieMultiplier: 1.2 },
  envUpdate: 'everyFrame',
});
engine.setEnvironmentLighting({ diffuse: 1, specular: 1 });
engine.setTimeOfDay({ day: 172, hours: 18.5, latitude: 38, enabled: true, playing: true, speed: 30 });
console.log(engine.sky().profile, engine.sky().zenith, engine.passes().map((p) => p.name));
console.log(engine.timeOfDay().currentHours, engine.timeOfDay().dominant);
engine.setWeather({ preset: 'broken', changing: true, seed: 7, transitionSeconds: 30 });
console.log(engine.weather().cyclePreset, engine.weather().text);
console.log(engine.skyWeatherDiagnostics().readiness); // { status, stage, error }
engine.setPassEnabled('sky.background', false); // keep the sky light, drop the sky image
```
