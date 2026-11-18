// Sharpening additions (pass `sharpen`, order 195, off by default). See
// docs/api/m34-sharpening.md. scripts/build-package.sh appends this file to index.d.ts, and
// scripts/typecheck-types.mjs stages it the same way; the `Engine` interface below merges with
// the v0 declaration of the same name.

/**
 * Contrast-adaptive sharpening for {@link Engine.setSharpening}. Every field is optional and
 * merged over the current settings.
 */
export interface SharpeningSettings {
  /**
   * Enables the pass `sharpen`, the same as `setPassEnabled('sharpen', on)`. A boolean. Default
   * false: with cel shading on, the cel lines are sharpened too.
   */
  enabled?: boolean;
  /**
   * Sharpening amount; finite, in [0, 1]. 0 leaves the image unchanged. Default 0.9.
   */
  sharpness?: number;
}

/** The sharpening methods (m34), merged into the v0 Engine. */
export interface Engine {
  /**
   * Merges `settings` over the current sharpening settings, validates the merged result, then
   * sets the pass flag: `enabled` sets `sharpen` the same as `setPassEnabled('sharpen', enabled)`
   * does. A refusal changes nothing.
   * @throws RaaError `api.invalid_argument` (settings is not an object, or a key has the wrong
   *   type; the message names the key), `post.invalid` (sharpness is not a finite number in
   *   [0, 1]; the message names `sharpness`), `pipeline.busy` (a frame is executing).
   */
  setSharpening(settings: SharpeningSettings): void;

  /**
   * The complete sharpening settings. `enabled` is read from the pass flag, so
   * `setPassEnabled('sharpen', on)` is shown here too.
   */
  sharpening(): Required<SharpeningSettings>;
}
