# RAA API m11: transmission and sorted translucency

Additions of chain milestone 5 to `@etcito/raa`: refraction with absorption, translucent
surfaces and order-independent transparency (per-pixel linked list). Types:
`packages/raa/types/m11-transmission.d.ts` (appended to `index.d.ts` when the package is staged;
`Engine` merges with the v0 declaration). Everything in v0 and `docs/api/m07`..`m10` stays valid.
Failures throw a `RaaError` with a `code`.

## Exports

- `TransparencySettings`: `{ maxFragments?, poolFactor? }`.
- `TransparencyStats`: `{ poolNodes, overflowFrames }`.
- `setTransparency(settings)`, `transparencyStats()`: methods of `Engine`.

No new material method: `addSurfaceMaterial` now accepts the blends and built-ins below.

## Transmission fields of `SurfaceMaterialDesc`

Shared with m10; they take effect once this feature is installed (always, in `@etcito/raa`).

| field | default | meaning |
|---|---|---|
| `blend` | `opaque` | `'transparent'` or `'transmissive'` draw the surface in the sorted pass |
| `transmission` | 0 | [0, 1]; share of light that passes through; > 0 requires `blend: 'transmissive'` |
| `ior` | 1.5 | [1, 3]; index of refraction |
| `thickness` | 0 | metres, >= 0; path length of the refracted ray inside the volume |
| `attenuationColor` | [1, 1, 1] | sRGB; colour the light takes after `attenuationDistance` |
| `attenuationDistance` | Infinity | metres, > 0; Beer-Lambert absorption distance |

Absorption is `T = exp(-sigma * path)` with `sigma = -ln(attenuationColor) / attenuationDistance`,
so thicker volumes tint more strongly. The refracted background is read from a copy of the
opaque colour, blurred by roughness and thickness; where the refracted sample would lie in front
of the surface, the unrefracted position is used.

Transmissive alpha rule: transmission is carried by the refracted colour, not by alpha. A
`transmissive` surface composites with alpha = `opacity` (default 1), so it covers what is behind
it with the refracted, tinted copy of it; lowering `opacity` below 1 lets the plain background show
through as well. A `transparent` surface composites with alpha = `opacity` and no refraction.
Overlapping translucent surfaces give the same pixels in any submission order. Opaque objects are
never sorted; translucent fragments behind opaque depth are dropped.

## Built-in material functions

Four functions, listed by `materialFunctions()` with `builtin: true`. They use
`blend: 'transmissive'` (or `'transparent'`); with `blend: 'opaque'` they draw as opaque
surfaces. Colour params are sRGB.

| name | params (default) |
|---|---|
| `raa.ice` | `tint` color (0.7, 0.9, 0.98), `absorption` scalar (1.5), `cloudiness` scalar (0.3), `crackScale` scalar (0.25) |
| `raa.slime` | `color` color (0.45, 0.95, 0.15), `absorption` scalar (3.0), `glossiness` scalar (0.85), `bubbleScale` scalar (0.06) |
| `raa.water` | `deepColor` color (0.05, 0.45, 0.55), `shallowColor` color (0.55, 0.92, 0.88), `clarity` scalar (0.7) |
| `raa.glass` | `tint` color (0.95, 0.98, 0.97), `roughness` scalar (0.03) |

`raa.ice` has inner cracks and bubbles and frosted roughness variation; `raa.slime` has bubbles
and a glossy skin; `raa.water` blends `deepColor` into `shallowColor` with depth and `clarity`.

## Passes

- `opaque.copy` (order 70): copies the HDR colour (with mips) and the depth of everything opaque
  so transmissive surfaces can refract it. It is active only while a transparent or transmissive
  draw exists.
- `transparent.oit` (order 80): collects translucent fragments into a per-pixel linked list,
  sorts the nearest `maxFragments` per pixel and composites them back to front.

`setPassEnabled('opaque.copy', false)` makes transmissive surfaces refract black.
`setPassEnabled('transparent.oit', false)` leaves translucent surfaces undrawn. Both are
toggled like any pass (`passes()` lists them); neither is required.

## Transparency settings

`setTransparency(settings)` merges the given fields over the current values; a refusal changes
nothing.

| field | default | range |
|---|---|---|
| `maxFragments` | 16 | integer 4..32 |
| `poolFactor` | 8 | integer 2..32 |

`transparencyStats()` returns `{ poolNodes, overflowFrames }`. After an overflow frame the pool
grows for the next frame (bounded by the adapter limit). It is debugging information and never a
gate: do not branch on it for correctness.

## Refusal codes

| code | when |
|---|---|
| `oit.invalid` | `setTransparency` field is not an integer or out of range; the message names the field |
| `material.blend_unavailable` | `transparent` or `transmissive` while no transparency feature is installed |
| `api.invalid_argument` | `setTransparency` argument is not an object, or a field is not a number |
