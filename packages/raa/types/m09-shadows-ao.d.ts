// Hand-written types of the chain milestone 3 additions (soft cascaded sun shadows, horizon-based
// ambient occlusion and screen-space contact shadows). Documented in docs/api/m09-shadows-ao.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): the `Engine` interface below merges with the v0 declaration of the
// same name.

/**
 * Soft cascaded sun shadows for {@link Engine.setShadows}. Every field is optional and merged
 * over the current values. Lengths are metres, angles radians.
 */
export interface ShadowSettings {
  /**
   * Multiplier of the sun's angular radius used as the PCSS penumbra; finite, > 0 and <= 16.
   * 1 matches the real sun (0.0047 rad); larger values widen the soft shadow. Default 1.
   */
  softness?: number;
  /**
   * Distance from the camera to the far cascade split, in metres; finite and > 0. Default 60.
   */
  distance?: number;
  /**
   * Edge length of one cascade's shadow map in texels; 1024, 2048 or 4096. Default 2048.
   */
  resolution?: 1024 | 2048 | 4096;
}

/**
 * Horizon-based ambient occlusion (GTAO) for {@link Engine.setAo}. Every field is optional and
 * merged over the current values.
 */
export interface AoSettings {
  /**
   * World-space sampling radius in metres; finite, > 0 and <= 10. Default 0.8.
   */
  radius?: number;
  /**
   * Strength of the AO darkening, in [0, 1]: 0 leaves the ambient light unchanged, 1 applies it
   * fully. Default 1.
   */
  intensity?: number;
  /**
   * Exponent shaping the AO falloff; finite, > 0 and <= 8. Default 1.5.
   */
  power?: number;
}

/**
 * Screen-space contact shadows for {@link Engine.setContactShadows}. Every field is optional and
 * merged over the current values. Lengths are metres.
 */
export interface ContactShadowSettings {
  /**
   * Maximum screen-space ray-march distance in metres; finite, > 0 and <= 5. Default 0.35.
   */
  distance?: number;
  /**
   * Assumed occluder thickness in metres; finite, > 0 and <= 1. Default 0.05.
   */
  thickness?: number;
  /**
   * Strength of the contact darkening, in [0, 1]. Default 1.
   */
  intensity?: number;
}

/** The chain milestone 3 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Validates and applies soft cascaded sun shadows. Merges `settings` over the current values:
   * omitted fields keep their current value. A refusal changes nothing. Enabling or disabling
   * the feature stays `setPassEnabled('shadow.csm', on)`.
   * @throws RaaError `shadow.invalid` (a field is out of range, non-finite or of the wrong type;
   *   the message names the field), `api.invalid_argument` (the argument is not an object),
   *   `api.feature_missing` (no shadow feature is installed).
   */
  setShadows(settings: ShadowSettings): void;

  /**
   * Validates and applies horizon-based ambient occlusion (GTAO). Merges `settings` over the
   * current values. A refusal changes nothing. Enabling or disabling the feature stays
   * `setPassEnabled('ao.gtao', on)`.
   * @throws RaaError `ao.invalid` (a field is out of range, non-finite or of the wrong type;
   *   the message names the field), `api.invalid_argument` (the argument is not an object),
   *   `api.feature_missing` (no ambient occlusion feature is installed).
   */
  setAo(settings: AoSettings): void;

  /**
   * Validates and applies screen-space contact shadows. Merges `settings` over the current
   * values. A refusal changes nothing. Enabling or disabling the feature stays
   * `setPassEnabled('shadow.contact', on)`.
   * @throws RaaError `shadow.invalid` (a field is out of range, non-finite or of the wrong type;
   *   the message names the field), `api.invalid_argument` (the argument is not an object),
   *   `api.feature_missing` (no contact shadow feature is installed).
   */
  setContactShadows(settings: ContactShadowSettings): void;
}
