# RAA API m21: viewports, viewport cameras, resize and overlays (S25, S26)

Additive to [m07 meshes and pipeline](m07-meshes-hdr.md). Declarations live in
`packages/raa/types/m21-views.d.ts`; `scripts/build-package.sh` appends them to the
package's `index.d.ts`, merging `Engine`. The Rust side is
`crates/raa-js/src/api/m21_views.rs`. It holds the state, and m07_pipeline reads it to
render one pass chain per viewport and to draw the overlays.

```ts
interface ViewRect { x: number; y: number; width: number; height: number }
interface ViewCamera { eye: [number, number, number]; target: [number, number, number];
                       fovYDeg: number; near: number; far: number }
interface Engine {
  setViewports(rects: ViewRect[]): void
  setViewportCamera(index: number, cam: ViewCamera): void
  resize(cssWidth: number, cssHeight: number, dpr: number): void
  setOverlayLines(layer: number, positions: Float32Array, colors: Float32Array,
                  opts: { depthTest: boolean; widthPx: number }): void
  clearOverlay(layer: number): void
}
```

## Viewports

`setViewports(rects)` takes 1..4 rects as fractions of the canvas, with the origin at the
top left. Each of `x`, `y`, `width` and `height` must be:

- a finite number in 0..1;
- for sizes, non-zero;
- such that `x + width <= 1` and `y + height <= 1`.

Rects must not overlap. Shared edges are allowed, and both checks allow 1e-9 of rounding
slack. The engine starts with one full-canvas rect, `[{x: 0, y: 0, width: 1, height: 1}]`,
which is today's single view at today's cost. The 2x2 split screen is four quarters.
A refused call changes nothing.

## Viewport cameras

`setViewportCamera(index, cam)` needs `index` to be an integer below the current viewport
count. The camera is perspective with +Y up, and is checked with the `setCamera` rules:

- `fovYDeg` is in (0, 180);
- `near > 0` and `far > near`;
- `eye` differs from `target`;
- the view direction is not vertical.

Viewport 0 has no camera of its own: it is the scene camera. `setCamera` and
`setViewportCamera(0, cam)` write the same camera, so code that only calls `setCamera`
keeps working. A viewport i > 0 without a camera renders with viewport 0's camera.
Cameras of indices that leave the range on `setViewports` are kept, and apply again
when that viewport comes back.

## Resize

`resize(cssWidth, cssHeight, dpr)` sets the canvas backing store to
`round(css × min(dpr, 2))` per axis (DPR_CAP = 2), clamped to at least 1. Like v0, the
camera aspect follows, and an unchanged size does not reconfigure the canvas.

The v0 call `resize(width, height)` stays valid. When `dpr` is omitted it is 1, so the
two numbers are backing pixels, exactly as before. The game's
`engine.resize(target.width, target.height)` therefore keeps its meaning.

The JS method moved from `engine.rs` to `m21_views.rs`. `engine.rs` keeps the
backing-store update as the crate-private `resize_backing`.

## Overlays

`setOverlayLines(layer, positions, colors, opts)` replaces the segments of one layer
(0..7):

- `positions` holds xyz pairs (segments) in world space;
- `colors` holds linear rgba per vertex;
- `opts.depthTest` tests the segments against the scene depth;
- `opts.widthPx` is the line width in pixels (finite, > 0).

An empty pair of arrays is valid and empties the layer. `clearOverlay(layer)` empties a
layer. Overlays draw after tonemap in every viewport, and are never captured into
lighting. The data is stored per layer now; drawing lands with the m07_pipeline viewport
pass.

## Errors

| Code | When |
| --- | --- |
| `views.count` | 0 or more than 4 rects |
| `views.rect` | a rect value missing, non-finite or outside 0..1, a zero size, `x + width > 1` or `y + height > 1`, or two rects overlapping |
| `views.index` | a viewport index that is not an integer below the viewport count |
| `overlay.layer` | a layer that is not an integer in 0..7 |
| `overlay.length` | `positions.length` not a multiple of 6, or `colors.length` not 4 per vertex |
| `api.invalid_argument` | a malformed argument, a bad camera, a non-finite size, a `dpr` that is not finite and > 0, a bad `widthPx`, non-finite overlay values |
| `api.feature_missing` | the views state is not installed |
