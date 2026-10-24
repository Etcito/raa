# RAA API m14: camera effects

Additions of chain milestone 8 to `@etcito/raa`: screen-space reflections on glossy floors, depth
of field, height fog with aerial perspective, sun shafts, colour grading and named environment
presets. Types: `packages/raa/types/m14-camera-effects.d.ts` (appended to `index.d.ts` when the
package is staged; `Engine` merges with the v0 declaration). Everything in v0 and
`docs/api/m07`..`m13` stays valid. Failures throw a `RaaError` with a `code`.

Every setter merges its fields over the current values, so omitted fields keep what was set
before. A refusal changes nothing. Lengths are metres, colours are linear RGB unless noted.

## Exports

- `SsrSettings`: `{ intensity?, maxRoughness?, thickness? }`.
- `DofSettings`: `{ focusDistance?, fStop?, focalLength?, autofocus?, maxCoc? }`.
- `FogPreset`: `'clear' | 'haze' | 'snow' | 'jungle' | 'night'`.
- `FogSettings`: `{ preset?, density?, heightFalloff?, baseHeight?, start?, color?, inscatter?, aerialStrength?, aerialDistance? }`.
- `SunShaftsSettings`: `{ intensity?, density?, maxDistance? }`.
- `GradingPreset`: `'neutral' | 'candy' | 'ice' | 'jungle' | 'neon'`.
- `GradingSettings`: `{ preset?, temperature?, tint?, exposure?, contrast?, saturation?, vibrance?, lift?, gamma?, gain?, shadowsTint?, highlightsTint?, splitBalance?, vignette? }`.
- `PostPreset`: `'neutral' | 'candy' | 'ice' | 'jungle' | 'neon'`.
- `PostSettings`: `{ grading?, bloom?, dof?, ssr?, sunShafts?, toneMapping? }`.
- `EnvironmentDesc`: `{ sky?, sun?, fog?, post? }`.
- `setSsr(settings)`, `setDof(settings)`, `setFog(settings)`, `setSunShafts(settings)`,
  `setGrading(settings)`, `applyEnvironment(env)`: methods of `Engine`.

`Vec3` (from `index.d.ts`), `SkyDesc` (m08), `BloomSettings` (m10) and `ToneMappingDesc` (m07)
are reused, not redefined.

Each effect is a pass of the pipeline, switched with `setPassEnabled(name, on)`. The pass names
are `ssr`, `dof`, `fog`, `sunshafts` and `grading`. Passes run in the order ssr, fog, sunshafts,
dof, grading, after the forward pass and before tone mapping. A disabled pass leaves the frame
as it would be without it.

## Screen-space reflections

`setSsr` reflects the lit scene in glossy surfaces (floors, wet paint). Pass `ssr`.

| field | default | range |
|---|---|---|
| `intensity` | 1 | finite, [0, 2] |
| `maxRoughness` | 0.35 | [0, 1]; rougher surfaces get no reflection |
| `thickness` | 0.1 | metres (view space), > 0 |

## Depth of field

`setDof` blurs what is out of focus with a physically sized circle of confusion. Pass `dof`.

| field | default | range |
|---|---|---|
| `focusDistance` | 8 | metres, > 0; ignored while `autofocus` is set |
| `fStop` | 2.8 | [1, 32] |
| `focalLength` | 0.05 | metres, (0, 0.3] (0.05 is 50 mm) |
| `autofocus` | `null` | `[u, v]` in [0, 1] (top left origin), or `null` for manual focus |
| `maxCoc` | 12 | pixels, [0, 64] |

With `autofocus: [u, v]` the focal plane follows the depth under that screen point. Focus eases
towards the measured depth a fixed fraction per frame, not per second, so warm-up frames rendered
at a fixed time (dt 0) converge the same way as live frames.

## Height fog and aerial perspective

`setFog` fills the air with fog that thins with height, and adds aerial perspective (distant
objects shift towards the fog colour). Pass `fog`. `preset` is applied first; the other fields
then override it.

| field | default | range |
|---|---|---|
| `preset` | `clear` | `FogPreset` |
| `density` | 0 (clear) | >= 0, per metre at the base height |
| `heightFalloff` | 0.2 (clear) | >= 0, per metre |
| `baseHeight` | 0 (clear) | metres, finite |
| `start` | 0 (clear) | metres, >= 0 |
| `color` | [0.75, 0.82, 0.9] (clear) | linear RGB |
| `inscatter` | 0.5 (clear) | [0, 1] |
| `aerialStrength` | 0 (clear) | [0, 2]; 0 turns aerial perspective off |
| `aerialDistance` | 1000 (clear) | metres, > 0 |

Fog presets. Each lists the fields it changes; the others keep the `clear` values.

| preset | changes |
|---|---|
| `clear` | none: the defaults above |
| `haze` | `density` 0.015, `start` 10, `aerialStrength` 1, `aerialDistance` 800 |
| `snow` | `density` 0.03, `heightFalloff` 0.35, `start` 5, `color` [0.88, 0.92, 0.97], `inscatter` 0.4, `aerialStrength` 1, `aerialDistance` 500 |
| `jungle` | `density` 0.04, `heightFalloff` 0.5, `start` 2, `color` [0.6, 0.72, 0.55], `inscatter` 0.6, `aerialStrength` 1.5, `aerialDistance` 400 |
| `night` | `density` 0.02, `heightFalloff` 0.25, `color` [0.18, 0.22, 0.36], `inscatter` 0.3, `aerialStrength` 0.5, `aerialDistance` 900 |

## Sun shafts

`setSunShafts` draws light rays where the sun is occluded by geometry in the fog. Pass `sunshafts`.

| field | default | range |
|---|---|---|
| `intensity` | 1 | finite, [0, 4] |
| `density` | 0.5 | [0, 1]; more samples along each shaft |
| `maxDistance` | 150 | metres, > 0; how far the shafts march towards the sun |

## Colour grading

`setGrading` applies white balance, tone adjustments, lift, gamma, gain, split tones and a
vignette. Pass `grading`. `preset` is applied first; the other fields then override it.

| field | default | range |
|---|---|---|
| `preset` | `neutral` | `GradingPreset` |
| `temperature` | 0 | [-1, 1]; positive is warmer |
| `tint` | 0 | [-1, 1]; positive shifts towards magenta |
| `exposure` | 0 | EV, [-5, 5] |
| `contrast` | 1 | [0, 2], around mid grey |
| `saturation` | 1 | [0, 2] |
| `vibrance` | 0 | [-1, 1]; boosts the less saturated colours |
| `lift` | [0, 0, 0] | per channel, each [-1, 1] |
| `gamma` | [1, 1, 1] | per channel, each > 0 |
| `gain` | [1, 1, 1] | per channel, each >= 0 |
| `shadowsTint` | [1, 1, 1] | linear RGB, each >= 0 |
| `highlightsTint` | [1, 1, 1] | linear RGB, each >= 0 |
| `splitBalance` | 0.5 | [0, 1]; luminance where the shadow and highlight tints meet |
| `vignette` | 0 | [0, 1]; darkening at the corners |

Grading presets. Each lists the fields it changes from the defaults above.

| preset | changes |
|---|---|
| `neutral` | none |
| `candy` | `temperature` 0.1, `contrast` 1.05, `saturation` 1.2, `vibrance` 0.3, `highlightsTint` [1.04, 0.98, 1.02], `vignette` 0.1 |
| `ice` | `temperature` -0.25, `tint` 0.05, `contrast` 1.1, `saturation` 0.85, `shadowsTint` [0.85, 0.95, 1.1], `vignette` 0.15 |
| `jungle` | `temperature` 0.05, `tint` -0.1, `saturation` 1.15, `vibrance` 0.4, `gamma` [1, 1.03, 0.97], `vignette` 0.2 |
| `neon` | `contrast` 1.2, `saturation` 1.35, `vibrance` 0.2, `shadowsTint` [1.05, 0.9, 1.15], `highlightsTint` [1.1, 0.95, 1.05], `vignette` 0.25 |

## Post settings and environment presets

`PostSettings` groups the post effects: `grading` (as `setGrading`), `bloom` (as `setBloom`,
`BloomSettings` = `{ strength, radius }`), `dof` (as `setDof`), `ssr` (as `setSsr`), `sunShafts`
(as `setSunShafts`) and `toneMapping` (as `setToneMapping`, `ToneMappingDesc`). Every field merges
over the current values with the same range and refusal rules as its own setter.

`PostPreset` names a post look. A preset sets the grading preset, the bloom strength and radius,
and the tone mapping; fields given next to it then override those values. Tone mapping defaults
to AgX (`mapper` `'agx'`) in every preset; ACES is selected with `mapper: 'aces'`.

| preset | grading | bloom `strength`, `radius` | tone mapping |
|---|---|---|---|
| `neutral` | `neutral` | 0.04, 0.85 | `agx`, look `none` |
| `candy` | `candy` | 0.08, 0.85 | `agx`, look `punchy` |
| `ice` | `ice` | 0.05, 0.7 | `agx`, look `none` |
| `jungle` | `jungle` | 0.06, 0.8 | `agx`, look `golden` |
| `neon` | `neon` | 0.12, 0.9 | `agx`, look `punchy` |

## Applying an environment

`applyEnvironment(env)` sets a whole look in one call. The sections are applied in this order:

1. `sky`: a string names a sky preset (`'day'`, `'snow'`, `'jungle'` or `'night'`, as
   `setSky`); an object is a `SkyDesc` applied with `setSky`.
2. `sun`: `{ dir, color, intensity }` is applied with `setLight` as
   `{ direction: dir, color, intensity }`.
3. `fog`: a string is a `FogPreset`; an object is `FogSettings` applied with `setFog`.
4. `post`: a string is a `PostPreset` (see above); an object is `PostSettings`. A preset is
   applied first and the object's fields override it.

Each section that is omitted keeps the current state. The whole description is validated before
anything changes, so a refusal changes nothing.

```js
engine.applyEnvironment({
  sky: 'snow',
  sun: { dir: [-0.3, -1, -0.2], color: [1, 0.96, 0.9], intensity: 3 },
  fog: 'snow',
  post: { grading: { vignette: 0.3 } },
});
```

## Refusals

Every method throws a `RaaError` with one of these codes. Each names the offending field in its
message where a field is out of range.

| code | raised by |
|---|---|
| `api.invalid_argument` | any method, when the argument (or a section of `applyEnvironment`) is not an object of the documented shape |
| `ssr.invalid` | `setSsr`: a field is out of range or non-finite |
| `dof.invalid` | `setDof`: a field is out of range or non-finite |
| `fog.unknown_preset` | `setFog`, `applyEnvironment`: the preset name is not a `FogPreset` |
| `fog.invalid` | `setFog`, `applyEnvironment`: a field is out of range or non-finite |
| `sunshafts.invalid` | `setSunShafts`: a field is out of range or non-finite |
| `grading.unknown_preset` | `setGrading`: the preset name is not a `GradingPreset` |
| `grading.invalid` | `setGrading`: a field is out of range or non-finite |
| `post.unknown_preset` | `applyEnvironment`: the `post` name is not a `PostPreset` |
| `post.invalid` | `applyEnvironment`: a field of the `post` settings is out of range or non-finite |
| `sky.invalid` | `applyEnvironment`: the sky preset or a sky field is refused by `setSky` |

After `dispose` every method throws `api.disposed`, as in v0.
