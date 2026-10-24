# RAA API m13: liquids and particles

Additions of chain milestone 6 to `@etcito/raa`: animated liquid surfaces (Gerstner waves,
ripple impacts, foam and refraction) and GPU particle emitters. Types:
`packages/raa/types/m13-liquids-particles.d.ts` (appended to `index.d.ts` when the package is
staged; `Engine` merges with the v0 declaration). Everything in v0 and `docs/api/m07`..`m11`
stays valid. Failures throw a `RaaError` with a `code`.

## Exports

- `EmitterKind`: `'splash' | 'snow' | 'confetti' | 'sparkles'`.
- `EmitterDesc`: the emitter description, below.
- `LiquidImpact`: `{ x, z, time, strength }`.
- `addEmitter(desc)`, `setEmitter(id, desc)`, `removeEmitter(id)`, `burst(id, count)`,
  `setLiquidImpacts(instance, impacts)`: methods of `Engine`.

No new material method: `addSurfaceMaterial` now accepts the three liquid built-ins below and
`materialFunctions()` lists them with `builtin: true`.

## Emitters

`addEmitter(desc)` validates the whole description and returns its id; ids are dense and the
lowest free id is reused, so the ids of the current emitters are not stable across
`removeEmitter`. `setEmitter(id, desc)` replaces the whole description (omitted fields take the
kind's defaults again, not the previous values); its live particles are carried over while
`maxParticles` is unchanged, and changing `maxParticles` reallocates the emitter's slot range.
`burst` adds `count` particles at the next simulation step on top of the continuous `rate`.
`removeEmitter(id)` also removes its particles.

At most 64 emitters and at most 262144 particles in total (the sum of `maxParticles`) are
allowed.

### Fields

| field | meaning |
|---|---|
| `kind` | `EmitterKind`; required |
| `position` | emitter centre in world space; required |
| `direction` | mean spawn direction, non-zero; the simulation normalises it |
| `spread` | cone half-angle around `direction`, radians in [0, pi] |
| `rate` | continuous spawn rate in particles per second, >= 0 |
| `lifetime` | `[min, max]` seconds, `0 < min <= max` |
| `speed` | initial speed in m/s: one number or `[min, max]`, `0 <= min <= max` |
| `size` | particle size in metres: one number or `[min, max]`, `0 < min <= max` |
| `gravity` | downward acceleration in m/s²; negative values float upwards |
| `drag` | velocity damping per second, >= 0 |
| `colors` | 1 to 8 sRGB colours, each component in [0, 1]; converted to linear on use |
| `emissive` | emissive strength added to the palette colour, >= 0 |
| `volume` | half extents of the spawn volume around `position`, metres, each >= 0 |
| `groundY` | ground plane height in metres; omitted disables the bounce (restitution 0.3) |
| `maxParticles` | hard cap on live particles of this emitter, integer 1..65536 |
| `seed` | seed of the deterministic spawn sequence, a non-negative integer |

### Per-kind defaults

| field | `splash` | `snow` | `confetti` | `sparkles` |
|---|---|---|---|---|
| `direction` | [0, 1, 0] | [0, -1, 0] | [0, 1, 0] | [0, 1, 0] |
| `spread` | 0.5 | 0.2 | 0.6 | pi |
| `rate` | 0 | 400 | 120 | 60 |
| `lifetime` | [0.6, 1.2] | [6, 10] | [3, 5] | [0.6, 1.6] |
| `speed` | [1.5, 3.5] | [0.2, 0.4] | [3, 6] | [0.1, 0.4] |
| `size` | [0.015, 0.04] | [0.01, 0.025] | [0.02, 0.035] | [0.02, 0.06] |
| `gravity` | 9.81 | 1.0 | 9.81 | 0 |
| `drag` | 0.2 | 2.0 | 1.5 | 0.5 |
| `colors` | (0.8, 0.9, 1.0) | (0.95, 0.97, 1.0) | 6 saturated colours | (1, 0.85, 0.3), (1, 1, 1) |
| `emissive` | 0 | 0 | 0 | 8 |
| `volume` | [0, 0, 0] | [4, 2, 4] | [0, 0, 0] | [0.5, 0.5, 0.5] |
| `groundY` | none | none | none | none |
| `maxParticles` | 2048 | 8192 | 4096 | 1024 |
| `seed` | 1 | 1 | 1 | 1 |

`position` defaults to the origin but is required in JS. The confetti palette is red, orange,
yellow, green, blue and violet: (1, 0, 0), (1, 0.5, 0), (1, 0.9, 0), (0, 0.8, 0.2), (0, 0.4, 1),
(0.8, 0, 1). Splash droplets are ballistic; snow drifts on curl noise; confetti flutters with
spin and lift; sparkles drift and twinkle.

## Simulation clock

The simulation advances in fixed 1/60 s steps towards the pipeline time; `dt` is not used. The
first frame starts at `max(0, t - 4 s)`, so a scene created at time `t` already shows four
seconds of steady-state emission. At most 240 steps run per frame; a frame further behind jumps
its clock forward and simulates only the last 240 steps. Fixed-time frames (`dt == 0`) therefore
show the state at `t`, and two runs with the same `seed` and clock produce the same particles.

## Liquid material functions

Three built-ins, listed by `materialFunctions()` with `builtin: true`. They define a
`material_vertex` stage: the surface is a flat XZ grid at local `y == 0`, and the stage replaces
the local Y with the wave height plus the ripple rings and distorts XZ by the Gerstner
displacement. Place the surface with the instance transform. `water` and `slime` are
transmissive (built-in flag `FLAG_TRANSMISSIVE`): use `blend: 'transmissive'` with
`transmission` > 0 and `thickness` in metres so they refract and absorb the opaque copy.
`mud` is opaque: a thin wet clear coat over a rough body with slow popping bubbles. Being
opaque it cannot read the opaque depth copy, so its shore foam reads the mud depth above the
ground from the mesh's vertex colour alpha (metres, clamped to [0, 1]; the default alpha 1
means deep, no shore foam): foam covers the band shallower than `foamWidth`.

| name | class | look |
|---|---|---|
| `raa.liquid.water` | transmissive | clear water, deep-to-shallow absorption, foam at shores and crests |
| `raa.liquid.slime` | transmissive | thick glossy slime, voronoi bubbles, subtle shore foam |
| `raa.liquid.mud` | opaque | wet mud, dome-and-pop bubble field, darkens with `wetness` |

### `raa.liquid.water` params

| param | kind | default |
|---|---|---|
| `wave0`..`wave3` | vec4 | (0.05, 2.2, 0.3, 0.6), (0.03, 1.2, 1.4, 0.5), (0.016, 0.7, -0.7, 0.4), (0.008, 0.42, 2.4, 0.35) |
| `speeds` | vec4 | (1.8533, 1.3688, 1.0454, 0.8098) |
| `deepColor` | color (sRGB) | (0.0, 0.42, 0.62) |
| `shallowColor` | color (sRGB) | (0.3, 0.9, 0.82) |
| `clarity` | scalar | 0.6 |
| `foamWidth` | scalar (m) | 0.3 |
| `foamAmount` | scalar | 0.9 |
| `detailStrength` | scalar | 0.03 |
| `detailScale` | scalar | 1.4 |

### `raa.liquid.slime` params

| param | kind | default |
|---|---|---|
| `wave0`..`wave3` | vec4 | (0.03, 2.2, 0.3, 0.6), (0.018, 1.2, 1.4, 0.5), (0.0096, 0.7, -0.7, 0.4), (0.0048, 0.42, 2.4, 0.35) |
| `speeds` | vec4 | (0.46334, 0.34220, 0.26136, 0.20245) |
| `color` | color (sRGB) | (0.42, 1.0, 0.12) |
| `absorption` | scalar (1/m) | 5.0 |
| `glossiness` | scalar | 0.9 |
| `bubbleScale` | scalar (m) | 0.2 |
| `foamWidth` | scalar (m) | 0.14 |
| `foamAmount` | scalar | 0.45 |
| `detailStrength` | scalar | 0.02 |
| `detailScale` | scalar | 1.4 |

### `raa.liquid.mud` params

| param | kind | default |
|---|---|---|
| `wave0`..`wave3` | vec4 | (0.02, 2.2, 0.3, 0.6), (0.012, 1.2, 1.4, 0.5), (0.0064, 0.7, -0.7, 0.4), (0.0032, 0.42, 2.4, 0.35) |
| `speeds` | vec4 | (0.14827, 0.10950, 0.08363, 0.06478) |
| `color` | color (sRGB) | (0.48, 0.32, 0.18) |
| `wetness` | scalar | 0.7 |
| `bubbleScale` | scalar (m) | 0.55 |
| `bubbleRate` | scalar (1/s) | 0.18 |
| `foamWidth` | scalar (m) | 0.12 |
| `foamAmount` | scalar | 0.2 |
| `detailStrength` | scalar | 0.02 |
| `detailScale` | scalar | 1.8 |

## Wave parameter packing

Every liquid reads the same layout from the material record's `p` array (S10
`SurfaceMaterialGpu.p`):

- `p[i]` for `i` in 0..3 is wave `i`: `(amplitude A in m, wavelength L in m, direction angle
  theta in rad measured in XZ from +X towards +Z, steepness Q in [0, 1])`.
- `p[4]` is `(speed0, speed1, speed2, speed3)` in m/s.
- `p[5]`..`p[7]` are function-specific (the tables above).

With `k = 2 pi / L` and `D = (cos theta, sin theta)`, the phase is
`k (D.x x + D.y z - speed t)`. The rest height is `A sin(phase)`; the vertex stage adds the
horizontal Gerstner displacement and derives the analytic normal from the slope. A wave with
`L <= 0`, `A == 0` or a non-finite field contributes nothing, so a degenerate parameter can
never produce NaN. The water speeds are the deep-water dispersion `sqrt(9.81 L / 2 pi)`; slime
scales the amplitudes by 0.6 and the speeds by 0.25; mud scales them by 0.4 and 0.08 (a slow
viscous swell).

## Ripple impacts

`setLiquidImpacts(instance, impacts)` writes up to 4 impacts into the instance params rows 0..3
of the JS instance handle; every call replaces all four rows and zeroes the unused ones. The
vertex stage adds, per impact,

`strength exp(-1.6 a) sin(14 q) (1 - smoothstep(0, 0.12, q)) / (1 + 6 r)` with
`a = t - time`, `q = r - 0.9 a` and `r` the distance to `(x, z)`, plus the analytic radial slope
to the normal.

- `x` and `z` are in the liquid mesh's **local** space (the same XZ the waves displace), not
  world space.
- `time` is the start time in seconds on the engine clock, i.e. `render(timeMs) / 1000` — the
  same clock as the shader's `in.time`. A ring is visible only once the clock reaches `time`;
  its amplitude decays with age.
- `strength` must be finite and > 0 to be visible.
- Over 4 impacts, a missing field or a non-finite value is refused; other materials of the same
  instance read the same rows only if they choose to.

## Passes

- `particles.sim` (order 50, pass, not required): the compute simulation. Disabling it freezes
  the particles and their clock.
- `particles.opaque` (order 600, contributor of the forward pass): confetti and snow as lit,
  camera-facing or oriented quads with alpha-to-coverage, shaded with the installed Shadow, Ao,
  Ambient and Direct slots.
- `particles.blend` (order 85): splash with an environment Fresnel term (premultiplied) and
  sparkles as additive emissive points with twinkle; it copies the HDR colour, applies a soft
  depth fade and records nothing when no splash or sparkles emitter exists.

Both draw passes are listed by `passes()` and toggled with `setPassEnabled`; neither is required.

## Refusal codes

| code | when |
|---|---|
| `api.invalid_argument` | malformed JS: a desc is not an object, `kind` is not one of the four names, a required field is missing, a typed array has the wrong length, `maxParticles`/`seed` is not a non-negative integer, or an impact is missing a field / has a non-finite value / there are more than 4 impacts |
| `particles.invalid` | `EmitterDesc` range rule (finite numbers, direction non-zero, spread, rate, lifetime, speed, size, drag, palette, emissive, volume, `maxParticles`); the message names the field |
| `particles.capacity` | more than 64 emitters, more than 262144 particles in total, or no contiguous slot range |
| `particles.unknown_emitter` | `setEmitter`, `removeEmitter` or `burst` with an id that does not exist |
| `api.feature_missing` | the particle feature is not installed in this engine (never in the packaged `@etcito/raa`) |
| `api.unknown_handle` | `setLiquidImpacts` with an instance handle that does not exist |

## Example

```js
const water = engine.addSurfaceMaterial({
  function: 'raa.liquid.water',
  blend: 'transmissive',
  transmission: 1,
  ior: 1.33,
  thickness: 0.4,
  params: { clarity: 0.8, deepColor: [0.02, 0.3, 0.45], foamWidth: 0.15 },
});
// `gridMesh` is a flat XZ grid at local y = 0; the instance transform lifts it.
const pool = engine.addInstance(poolNode, gridMesh, water);

engine.setLiquidImpacts(pool, [
  { x: 0.2, z: -0.1, time: 2.5, strength: 0.6 },
  { x: -0.4, z: 0.3, time: 3.1, strength: 0.35 },
]);

const snow = engine.addEmitter({
  kind: 'snow',
  position: [0, 6, 0],
  rate: 600,
  volume: [4, 0.5, 4],
});
engine.burst(snow, 300);
engine.setEmitter(snow, { kind: 'snow', position: [0, 6, 0], rate: 300, seed: 7 });
engine.removeEmitter(snow);
```
