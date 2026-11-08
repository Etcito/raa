# Cel palette and toon lighting

`engine.setCel(settings)` merges settings, enables `cel.resolve` (HDR order 190), and selects neutral tone mapping: `mapper: 'linear'`, `exposureEv: 0`, `look: 'none'`, `dither: false`. The final tonemap still encodes sRGB. The feature is installed **disabled** so existing scenes keep their appearance.

`engine.getCel()` returns a fresh object with every setting below. `engine.setPassEnabled('cel', false)` disables resolve and toon lighting; `'cel.resolve'` is also accepted. Disabling keeps settings and leaves tone mapping as last set. Re-enabling with `setPassEnabled` keeps the current tone mapper; calling `setCel` selects the neutral mapper again.

| Setting | Type / range | Default |
| --- | --- | --- |
| `bands` | integer 2–32 | 10 |
| `hues` | integer 4–32 | 16 |
| `chroma` | integer 2–12 | 6 |
| `dither` | `'bayer2' \| 'bayer4' \| 'bayer8' \| 'bayer16' \| 'ign' \| 'white' \| 'blue'` | `'bayer4'` |
| `ditherAmount` | number 0–1 | 0.65 |
| `outlineWidth` | number 0.5–6 rendered pixels | 1 |
| `outlineSharpness` | number 0–1 | 0.85 |
| `interiorEdges` | boolean | true |
| `edgeRetention` | number 0–1 | 0.85 |
| `worldSpace` | boolean | true |
| `worldScale` | number 1–512 cells per world unit | 48 |
| `softness` | number 0–1 | 0.12 |
| `specular` | number 0–2 | 0.7 |
| `toon` | boolean | true |
| `palette` | boolean | true |
| `outlines` | boolean | true |
| `ditherOutlines` | boolean | false |

All fields are optional; omitted fields retain their current value. Numbers must be finite. Invalid types, fractions in integer fields, unknown dither modes, nulls and out-of-range values throw `cel.invalid`; the settings, enable state and tone mapper remain unchanged.

```js
engine.setCel({ bands: 10, hues: 16, chroma: 6, worldSpace: true });
const material = engine.addSurfaceMaterial({ color: [0.6, 0.4, 0.2], cel: true });
engine.applyEnvironment({ post: { cel: { dither: 'blue', softness: 0.12 } } });
const settings = engine.getCel();
```

`applyEnvironment({post: {cel: settings}})` invokes the same setter. Within a post object, `cel` is applied after grading/bloom/DOF/SSR/sun shafts and before `toneMapping`; an explicit `toneMapping` key can override the neutral transform.

The surface option `cel?: boolean` defaults to false on both `addSurfaceMaterial` and `setSurfaceMaterial` (replacement semantics). It sets material flag 128. Only those materials use the priority-30 Direct provider when cel and `toon` are enabled. Unflagged materials retain the priority-20 layered BRDF. The restored `unlit?: boolean` option sets flag 64 and uses base colour plus emission directly; it takes precedence over cel lighting. Resolve still applies to the resulting image.

The resolve reads linear HDR colour, single-sample depth and view normals. It quantizes hit pixels in Oklab: lightness bands, snapped hue sectors, sqrt-spaced chroma, chroma bisection into gamut and threshold-biased rounding. `palette: false` selects scalar lightness bands instead of hue/chroma reduction. Blue thresholds use a deterministic 16×16 void-and-cluster rank tile. World anchoring projects `floor(world * worldScale)` into integer threshold coordinates.

Outlines use eight depth/normal taps at two rings, fractional widths, tangent-plane rejection on smooth surfaces and a 3×3 coverage feather. With `ditherOutlines: false`, ink is composited in linear space after quantization. Transparency and particles use the opaque prepass depth/normal footprint; they have no separate outline target. Background without geometry is preserved.

This phase runs at native target resolution: the internal detail factor is 1. The retained linear sample averaging and average/max coverage formula are ready for later reduction; `edgeRetention` has no visible effect at factor 1. There is no pixel-grid or sampling-size API yet. Point lights, material-id outlines, field curvature, CPU error diffusion/DBS, temporal dithering and pixel-stable panning are outside this module.

Rust hosts install `raa_cel::install(&mut lit)` for the Direct provider and `CelFeature::install(gpu, pipeline)` for resolve, update the handle's camera each frame and select neutral linear tonemapping themselves. The resolve `CelPost` uniform is 112 bytes; inverse projection/view matrices occupy a separate camera uniform. Toon controls use the formerly reserved environment lane at byte 192 without changing its 256-byte ABI.
