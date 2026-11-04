// S25/S26 additions; see docs/api/m21-views.md. build-package.sh appends this file
// to index.d.ts. These interfaces merge with the v0 Engine declaration, whose
// `resize(width, height)` stays valid as the dpr-less form of `resize` below.

/** A viewport as fractions of the canvas (0..1), origin at the top left. */
export interface ViewRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A perspective viewport camera; up is +Y. */
export interface ViewCamera {
  eye: [number, number, number];
  target: [number, number, number];
  fovYDeg: number;
  near: number;
  far: number;
}

export interface Engine {
  /**
   * Sets 1..4 non-overlapping viewports (edges may touch). Each value is finite and in
   * 0..1, sizes are non-zero, x + width <= 1 and y + height <= 1.
   * `[{x: 0, y: 0, width: 1, height: 1}]` is the default single view.
   * Nothing changes on a refusal.
   * @throws RaaError `views.count`, `views.rect`, `api.invalid_argument`.
   */
  setViewports(rects: ViewRect[]): void;

  /**
   * Sets the camera of viewport `index` (0 .. viewport count - 1). Viewport 0 is the scene
   * camera: `setCamera` and `setViewportCamera(0, cam)` write the same camera. A viewport
   * without its own camera renders with viewport 0's.
   * @throws RaaError `views.index`, `api.invalid_argument` (bad camera).
   */
  setViewportCamera(index: number, cam: ViewCamera): void;

  /**
   * Resizes the canvas backing store to round(css * min(dpr, 2)) (0 clamps to 1).
   * Called with 2 arguments, dpr is 1 and the sizes are backing pixels (the v0 call).
   * @throws RaaError `api.invalid_argument` (non-finite size, dpr not finite and > 0).
   */
  resize(cssWidth: number, cssHeight: number, dpr: number): void;

  /**
   * Draws the 3D image at `scale` (0.25..2) times the backing size {@link Engine.resize} gives:
   * below 1 the canvas holds fewer pixels and the browser upsamples it to its CSS size
   * (bilinear, or nearest under CSS `image-rendering: pixelated`), above 1 it holds more and
   * the browser downsamples it (supersampling; at most 8192 px per side). Viewports are canvas
   * fractions, so they follow. Applied at once when `resize` was called, else at the first
   * `resize`. Default 1.
   * @throws RaaError `api.invalid_argument` (outside 0.25..2 or non-finite).
   */
  setRenderScale(scale: number): void;

  /** The render scale {@link Engine.setRenderScale} set (1 by default). */
  renderScale(): number;

  /**
   * Replaces the segments of overlay `layer` (0..7). `positions` holds xyz pairs in world
   * space (length a multiple of 6), `colors` linear rgba per vertex (4 per vertex).
   * Overlays draw after tonemap and are never captured into lighting.
   * @throws RaaError `overlay.layer`, `overlay.length`, `api.invalid_argument`.
   */
  setOverlayLines(
    layer: number,
    positions: Float32Array,
    colors: Float32Array,
    opts: { depthTest: boolean; widthPx: number },
  ): void;

  /**
   * Empties overlay `layer` (0..7).
   * @throws RaaError `overlay.layer`.
   */
  clearOverlay(layer: number): void;
}
