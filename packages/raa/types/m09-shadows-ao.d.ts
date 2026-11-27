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
   * Keep the sun shadowed in the whole scene. The first three cascades keep the practical
   * splits of `distance`; the last one reaches the farthest corner of the scene bounds (capped
   * by the camera far plane) when that is farther, and a cascade is fitted to the scene's
   * light-space bounds when they are smaller than its view slice. `false` restores the plain
   * range `min(far, distance)`, past which the sun is unshadowed. Default true.
   */
  cover?: boolean;
  /**
   * Edge length of one cascade's shadow map in texels; 1024, 2048 or 4096. Default 2048.
   */
  resolution?: 1024 | 2048 | 4096;
  /**
   * Number of cascades rendered, 1..4 (practical splits of `distance` for that count; the
   * unused layers are not rendered and the shader clamps to the last used cascade). Default 4.
   */
  cascades?: 1 | 2 | 3 | 4;
  /**
   * Minimum PCF filter radius in metres, 0..0.5: the same world size in every cascade (at
   * least 1.5 texels of each), widened by the sun penumbra. Default 0.015.
   */
  filter?: number;
  /**
   * Practical split weight of the cascades, 0..1 (0 uniform, 1 logarithmic); higher values
   * give the near cascades more texels per metre. Default 0.75.
   */
  split?: number;
}

/** What the last `shadow.csm` frame computed, from {@link Engine.shadowStats}. */
export interface ShadowStats {
  /** The `cover` setting. */
  cover: boolean;
  /** View-space far distance of each of the 4 cascades, metres. */
  splits: number[];
  /** How each cascade box was fitted: `slice` (sphere of its view slice) or `scene` (scene bounds). */
  fits: ('slice' | 'scene')[];
  /** World size of one shadow-map texel of each cascade, metres. */
  texels: number[];
  /** Farthest scene-bounds corner from the camera (rounded up to 2^(k/4)), metres, or null. */
  coverDistance: number | null;
  /** The scene bounds used, `[minX, minY, minZ, maxX, maxY, maxZ]`, or null. */
  scene: number[] | null;
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
   * fully. Default 0.6.
   */
  intensity?: number;
  /**
   * Exponent shaping the AO falloff; finite, > 0 and <= 8. Default 1.0.
   */
  power?: number;
  /**
   * Sampling effort: slice directions per pixel, a whole number in 1..8. Default 4. Fewer is
   * cheaper and noisier.
   */
  slices?: number;
  /**
   * Sampling effort: horizon samples per side of a slice, a whole number in 1..16. Default 8.
   */
  steps?: number;
  /**
   * Apply artistic GTAO to diffuse GI and model-assisted diffuse bounce. Default false.
   * The independent path reference always bypasses screen-space AO.
   */
  indirectDiffuse?: boolean;
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
   * The cascades of the last frame the `shadow.csm` pass prepared (diagnostics).
   * @throws RaaError `api.feature_missing` (no shadow feature is installed).
   */
  shadowStats(): ShadowStats;

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
