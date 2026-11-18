# RAA API m34: contrast-adaptive sharpening

Additive to [m07 meshes and pipeline](m07-meshes-hdr.md), [m31 TAA](m31-taa.md) and
[m33 automatic exposure](m33-auto-exposure.md).
Declarations live in `packages/raa/types/m34-sharpening.d.ts`; `scripts/build-package.sh`
appends them to the package's `index.d.ts`, merging `Engine`. The pass is `sharpen`
(crate `raa-sharpen`); the JS methods are in `crates/raa-js/src/api/m34_sharpening.rs`.

```ts
interface SharpeningSettings {
  enabled?: boolean; sharpness?: number
}
interface Engine {
  setSharpening(settings: SharpeningSettings): void
  sharpening(): Required<SharpeningSettings>
}
```

## Settings

`setSharpening(settings)` merges the keys it is given over the current settings, validates the
merged result and applies it only when that passes. Omitted keys keep their values. The
defaults are:

| JS key | type | default | rule |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | none; sets the pass flag, as `setPassEnabled('sharpen', enabled)` does |
| `sharpness` | number | `0.9` | finite, 0..1; `0` leaves the image unchanged |

A rule failure is refused with `post.invalid`, and its message names `sharpness`. A refusal
changes nothing. `sharpening()` returns both keys, with `enabled` read from the pass flag.

## Methods and refusals

- `setSharpening(settings: SharpeningSettings): void` merges, validates and then sets the
  pass flag. `enabled` sets the pass the same way `setPassEnabled('sharpen', enabled)` does.
- `sharpening(): Required<SharpeningSettings>` returns the complete settings.

| Code | When |
| --- | --- |
| `api.invalid_argument` | `setSharpening`: the argument is not an object, or a key has the wrong JS type (the message names the key). |
| `post.invalid` | `setSharpening`: `sharpness` is not a finite number in 0..1 (the message names `sharpness`). |
| `pipeline.busy` | `setSharpening` while a frame is executing. |

```js
engine.setSharpening({ enabled: true, sharpness: 0.9 });
engine.sharpening(); // { enabled: true, sharpness: 0.9 }
engine.setSharpening({ sharpness: 2 }); // throws post.invalid; the settings stay as they were
```

## Kernel

The pass is a 5-tap contrast-adaptive kernel: the centre `C` and its neighbours `N`, `S`, `W`
and `E` (one texel up, down, left and right). The taps are `textureLoad`s clamped to the
texture, and each one is range-compressed first (see the next section). Then:

1. `low` and `high` are the per-channel smallest and largest of the five compressed taps.
2. `amount = sqrt(clamp(min(low, 1 - high) / max(high, 1e-4), 0, 1))`. It is largest where the
   five taps are close together and small across a strong edge, so the sharpening is strong on
   flat detail and gentle at high contrast.
3. `weight = -amount * sharpness * 0.2`.
4. `out = clamp((C + (N + S + W + E) * weight) / (1 + 4 * weight), low, high)`.

The weights sum to one, so flat areas do not change. The centre tap gains weight and the
neighbours take a negative one, which sharpens edges. The final clamp keeps the result inside
the range of the five taps, so there is no halo: no dark or bright outline forms beside an
edge. With `sharpness` 0 the weight is 0 and the centre tap passes through unchanged, apart
from the range compression and expansion below.

The kernel is adapted from solstice (see [Attribution](#attribution)). Solstice sharpens
display values; RAA's chain is HDR, so the taps are compressed first.

## HDR range compression

Each tap is compressed before the kernel and the result is expanded after it:

- compress: negative values are clamped to zero, then the colour is divided by one plus its
  largest channel: `c / (1 + max(c))`. Every compressed value lies in [0, 1).
- expand: `d / (1 - min(max(d), 0.999))`.

For a colour whose largest channel is at most 999, the expansion inverts the compression
exactly. Brighter colours are expanded with the divisor fixed at 0.001, so they are bounded by
the neighbour clamp and do not blow up. Very bright pixels therefore stay within the range of
their neighbours, as the clamp in the kernel requires.

## The pass

`sharpen` is an HDR-chain pass at order 195, after `cel.resolve` (190) and before `tonemap`
(200). It reads the chain's source, writes its destination and swaps the chain. It is not an
overlay. It is installed disabled: with no call to `setSharpening` or `setPassEnabled`, existing
pages render as before. Enable it with either of:

- `setSharpening({ enabled: true })`;
- `setPassEnabled('sharpen', true)`.

Both set the same flag, so `sharpening().enabled` and `passes()` agree.

## Composition

- **`setToneMapping`.** The pass runs on the HDR colour before the tone mapper. The tone
  mapping `exposureEv` keeps its meaning and applies after the sharpening.
- **Automatic exposure.** `exposure.auto` (order 125) runs before this pass, so the sharpening
  works on the exposed colour. The two compose and neither changes the other's settings.

## Render scale

The pass runs at the render resolution. Below `setRenderScale(1)` the canvas is smaller than
the display, and the browser upsamples the sharpened canvas. The sharpening is applied before
that upsampling.

For temporal upscaling, use TAA with sharpening: `setPassEnabled('taa', true)` (see
[m31 TAA](m31-taa.md)), `setSharpening({ enabled: true })` and a render scale below 1, such as
0.67. TAA accumulates samples over frames and softens edges; sharpening restores the crispness
that the temporal filter softens.

## Cel lines

With cel shading on (`setCel`, declared in `packages/raa/types/m27-cel.d.ts`), the cel lines
are part of the colour this pass reads, so they are sharpened too. This is one reason the
default is off: leave the pass off for cel scenes unless the sharper lines are wanted.

## Attribution

The kernel is adapted from solstice@7e465b5 `crates/solstice-render/shaders/post/tonemap.wgsl`
(MIT, Copyright (c) 2026 SamG-Coder).

## Example

```js
import { loadRaa } from '@etcito/raa';

const raa = await loadRaa();
const canvas = document.querySelector('canvas');
const engine = await raa.createEngine(canvas);
engine.setPassEnabled('taa', true);
engine.setRenderScale(0.67);
engine.setSharpening({ enabled: true, sharpness: 0.9 });
```
