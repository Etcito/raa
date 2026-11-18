// S46 additions: automatic exposure (pass `exposure.auto`, order 125, off by default). See
// docs/api/m33-auto-exposure.md. scripts/build-package.sh appends this file to index.d.ts, and
// scripts/typecheck-types.mjs stages it the same way; the `Engine` interface below merges with
// the v0 declaration of the same name.

/**
 * Automatic exposure for {@link Engine.setAutoExposure}. Every field is optional and merged over
 * the current settings. Exposure is in EV (base-2 stops): a linear multiplier is 2^ev.
 */
export interface AutoExposureSettings {
  /**
   * Enables the pass `exposure.auto`, the same as `setPassEnabled('exposure.auto', on)`. A
   * boolean; no range. Default false.
   */
  enabled?: boolean;
  /**
   * Exposure compensation in EV; finite, in [-8, 8]. The exposure aims at 2^compensationEv times
   * 0.18 over the measured average luminance. Default 0.263 (about 1.2 times).
   */
  compensationEv?: number;
  /**
   * Lower edge of the histogram band that sets the average, a fraction; finite, in [0, 1] and
   * below highPercent. Default 0.2.
   */
  lowPercent?: number;
  /**
   * Upper edge of the histogram band that sets the average, a fraction; finite, in [0, 1].
   * Default 0.7.
   */
  highPercent?: number;
  /**
   * Adaptation rate in 1/s while the scene gets brighter; finite and > 0. Default 2.4.
   */
  speedUp?: number;
  /**
   * Adaptation rate in 1/s while the scene gets darker; finite and > 0. Default 1.1.
   */
  speedDown?: number;
  /**
   * Lowest exposure in EV; finite and <= maxEv. Default -6.
   */
  minEv?: number;
  /**
   * Highest exposure in EV; finite. Default 6.
   */
  maxEv?: number;
}

/**
 * The state {@link Engine.autoExposureState} returns for one viewport. `exposureEv` and
 * `averageLuminance` are null until the pass has a result since its last snap. `settled` holds
 * when the two latest results differ by less than 0.01 EV. `pending` holds while an enabled pass
 * waits for that first result. A disabled pass reports settled true and pending false.
 */
export interface AutoExposureState {
  /** log2 of the exposure multiplier of the latest result, or null. */
  exposureEv: number | null;
  /** The adapted average linear luminance the latest result used, or null. */
  averageLuminance: number | null;
  /** True when the exposure has stopped moving (see above). */
  settled: boolean;
  /** True while an enabled pass has no result since its last snap. */
  pending: boolean;
}

/** The automatic exposure methods (m33), merged into the v0 Engine. */
export interface Engine {
  /**
   * Merges `settings` over the current automatic exposure settings and validates the merged
   * result; nothing is applied unless it passes. `enabled` sets the pass flag, as
   * `setPassEnabled('exposure.auto', enabled)` does. A refusal changes nothing.
   * @throws RaaError `api.invalid_argument` (settings is not an object, or a key has the wrong
   *   type; the message names the key), `post.invalid` (a field breaks its rule; the message
   *   names the JS key), `pipeline.busy` (a frame is executing).
   */
  setAutoExposure(settings: AutoExposureSettings): void;

  /**
   * The complete automatic exposure settings. `enabled` is read from the pass flag, so
   * `setPassEnabled('exposure.auto', on)` is shown here too.
   */
  autoExposure(): Required<AutoExposureSettings>;

  /**
   * The adapted exposure of one viewport: `viewport` is an index below the viewport count and
   * defaults to 0. Without viewports only 0 (the main pass) is valid. Each viewport adapts on its
   * own; the settings are shared.
   * @throws RaaError `api.invalid_argument` (the viewport is not an integer index of a viewport
   *   that exists).
   */
  autoExposureState(viewport?: number): AutoExposureState;
}
