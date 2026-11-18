# RAA API m33: automatic exposure (S46)

Additive to [m07 meshes and pipeline](m07-meshes-hdr.md) and [m21 viewports](m21-views.md).
Declarations live in `packages/raa/types/m33-auto-exposure.d.ts`; `scripts/build-package.sh`
appends them to the package's `index.d.ts`, merging `Engine`. The pass is `exposure.auto`
(crate `raa-exposure`); the JS methods are in `crates/raa-js/src/api/m33_auto_exposure.rs`.

```ts
interface AutoExposureSettings {
  enabled?: boolean; compensationEv?: number; lowPercent?: number; highPercent?: number;
  speedUp?: number; speedDown?: number; minEv?: number; maxEv?: number
}
interface AutoExposureState {
  exposureEv: number | null; averageLuminance: number | null; settled: boolean; pending: boolean
}
interface Engine {
  setAutoExposure(settings: AutoExposureSettings): void
  autoExposure(): Required<AutoExposureSettings>
  autoExposureState(viewport?: number): AutoExposureState
}
```

## Settings

`setAutoExposure(settings)` merges the keys it is given over the current settings, validates the
merged result and applies it only when that passes. Omitted keys keep their values. The
defaults are:

| JS key | type | default | rule |
| --- | --- | --- | --- |
| `enabled` | boolean | `false` | none; sets the pass flag |
| `compensationEv` | number, EV | `0.263` (log2 1.2) | finite, -8..8 |
| `lowPercent` | number | `0.2` | finite, 0..1, below `highPercent` |
| `highPercent` | number | `0.7` | finite, 0..1 |
| `speedUp` | number, 1/s | `2.4` | finite, > 0 (adaptation towards a brighter scene) |
| `speedDown` | number, 1/s | `1.1` | finite, > 0 (adaptation towards a darker scene) |
| `minEv` | number, EV | `-6` | finite, at most `maxEv` |
| `maxEv` | number, EV | `6` | finite |

The rules are checked in row order. The first failure is refused with `post.invalid`, and its
message names the JS key. A refusal changes nothing. `autoExposure()` returns all eight keys,
with `enabled` read from the pass flag.

## Maths

Luminance is `dot(rgb, (0.2126, 0.7152, 0.0722))` of the linear HDR colour. Each frame the pass
does the following.

1. **Histogram.** Pixels whose luminance is below 1e-5 are not counted. Any other luminance `L`
   goes to bin `trunc(clamp((log2 L + 8) / 18, 0, 1) * 127)` of 128 bins spanning log2
   luminance from -8 to 10. Only every 4th pixel in x and y is read, one pixel in 16.
2. **Band average.** Let `total` be the number of counted pixels. The bins are accumulated, and
   the counts that fall between `lowPercent * total` and `highPercent * total` form the band.
   The band's mean log2 luminance is the count-weighted mean of the bin centres
   `-8 + (i + 0.5) / 128 * 18`, and the band average is 2 to that power. With no counted
   pixel the previous average is kept, or 1.0 when there is none.
3. **Adaptation.** The average moves exponentially towards the band average:
   `average = previous + (measured - previous) * (1 - exp(-dt * rate))`, where `dt` is the frame
   time in seconds and `rate` is `speedUp` when the scene gets brighter and `speedDown` when
   it gets darker.
4. **Exposure.** `exposure = clamp(2^compensationEv * 0.18 / max(average, 1e-6), 2^minEv, 2^maxEv)`,
   a linear multiplier. `exposureEv` is its log2.
5. **Settled.** `settled` holds when the two latest results with distinct frame numbers have
   `|exposureEv(latest) - exposureEv(previous)| < 0.01`. It is false while only one result exists.

**Snap.** The average is taken from the measurement with no smoothing on the first frame of the
pass, when the HDR target size changes, after the pass is invalidated, and on a frame gap: the
pass was disabled or skipped, including `setAutoExposure({ enabled: false })` followed by any
later re-enable. A snap starts a new history. Results older than the latest snap are ignored,
so the state reads `pending` until the first new result arrives.

## The pass

`exposure.auto` is an HDR-chain pass at order 125, between `bloom` (120) and `grading` (130).
It multiplies the HDR colour by the exposure multiplier, keeps the alpha, and swaps the chain.
It is installed disabled: with no call to `setAutoExposure` or `setPassEnabled`, existing pages
render as before. Enable it with either of:

- `setAutoExposure({ enabled: true })`;
- `setPassEnabled('exposure.auto', true)`.

Both set the same flag, so `autoExposure().enabled` and `passes()` agree.

## Composition with setToneMapping

`setToneMapping({ exposureEv })` keeps its meaning. Its `exposureEv` multiplies the HDR colour
in the tone mapper, after this pass, so the two compose in EV: the shown exposure is the
automatic `exposureEv` plus the tone mapping `exposureEv`. With the default 0 EV the automatic
result is shown as it is. The automatic pass never writes the tone mapping setting.

## Viewports

With m21 viewports, each viewport adapts on its own. Each has its own pass instance and its
own average and history, so a snap in one view never changes another view's exposure.
The settings and the enabled flag are global: `setAutoExposure` and `setPassEnabled` apply
to every view. `autoExposureState(viewport)` reads that view. Without
viewports only 0 (the main pass) is valid.

```js
engine.setViewports([
  { x: 0, y: 0, width: 0.5, height: 1 },
  { x: 0.5, y: 0, width: 0.5, height: 1 },
]);
const left = engine.autoExposureState(0);
const right = engine.autoExposureState(1);
```

## Drawing on demand

The histogram is read back through an asynchronous readback, so results arrive a few frames
after the frame they measure. `autoExposureState()` therefore describes an earlier frame than
the one on screen. A host that draws only on demand keeps drawing until `settled` is true, and
reads `autoExposureState()` after each frame it draws. A host that draws continuously needs
no change.

## Example

```js
import { loadRaa } from '@etcito/raa';

const raa = await loadRaa();
const canvas = document.querySelector('canvas');
const engine = await raa.createEngine(canvas);
engine.setToneMapping({ exposureEv: 0 });
engine.setAutoExposure({ enabled: true, compensationEv: 0.263, minEv: -6, maxEv: 6 });

function frame(nowMs) {
  engine.render(nowMs);
  // On-demand drawing: keep drawing until the exposure has settled. Start the loop again
  // after a change to the scene or the camera.
  if (!engine.autoExposureState().settled) requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

## Errors

| Code | When |
| --- | --- |
| `api.invalid_argument` | `setAutoExposure`: the argument is not an object, or a key has the wrong JS type (the message names the key). `autoExposureState`: the viewport is not an integer index of an existing viewport. |
| `post.invalid` | `setAutoExposure`: a field breaks its rule in the table above; the message names the JS key. |
| `pipeline.busy` | `setAutoExposure` while a frame is executing. |
