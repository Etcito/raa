// Hand-written types of the chain milestone 2 additions (procedural sky with clouds and the
// image-based lighting derived from it). Documented in docs/api/m08-sky-ibl.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): `Vec3` comes from there, and the `Engine` interface below
// merges with the v0 declaration of the same name.

/** The built-in sky presets of {@link Engine.setSky}. */
export type SkyPreset = 'day' | 'snow' | 'jungle' | 'night';

/**
 * The procedural cloud layer of {@link SkyDesc}. Every field is optional and merged over the
 * preset (default: the values of the `day` preset).
 */
export interface CloudsDesc {
  /** Fraction of the sky covered by clouds, in [0, 1]. Default 0.5. */
  coverage?: number;
  /** Optical density of the cloud layer, finite and >= 0. Default 0.5. */
  density?: number;
  /** Cloud base altitude in metres, finite and > 0. Default 1200. */
  altitude?: number;
  /** Horizontal scale of the cloud pattern in metres, finite and > 0. Default 1000. */
  scale?: number;
  /** Wind velocity `[x, z]` in metres per second, both finite. Default [1, 0.5]. */
  wind?: [number, number];
  /** Cloud colour, linear RGB. Default [1, 1, 1]. */
  color?: Vec3;
  /** Bright rim around dense cloud edges, in [0, 1]. Default 0.5. */
  silverLining?: number;
}

/**
 * A procedural sky description for {@link Engine.setSky}. Every field is optional: the `preset`
 * is applied first and the remaining fields are merged over it. Colours are scene-referred
 * linear HDR RGB triples (values above 1 are allowed); lengths are metres and angles radians.
 */
export interface SkyDesc {
  /** Built-in preset applied first. Default `'day'`. */
  preset?: SkyPreset;
  /** Sky colour straight up, linear RGB. Default [0.2, 0.42, 1.0]. */
  zenith?: Vec3;
  /** Sky colour at the horizon, linear RGB. Default [0.7, 0.8, 0.95]. */
  horizon?: Vec3;
  /** Colour below the horizon, linear RGB. Default [0.22, 0.2, 0.18]. */
  ground?: Vec3;
  /** Overall sky radiance multiplier, finite and >= 0. Default 1. */
  intensity?: number;
  /** Softness of the zenith-to-horizon blend, in [0, 1] (0 a hard band, 1 smooth). Default 0.15. */
  horizonSoftness?: number;
  /** Angular radius of the sun disc in radians, finite and > 0. Default 0.0047. */
  sunDiscSize?: number;
  /** Radiance multiplier of the sun disc, finite and >= 0. Default 20. */
  sunDiscIntensity?: number;
  /** Cloud layer; merged field by field over the preset's clouds. */
  clouds?: CloudsDesc;
  /** When the environment cube and its lighting are rebuilt. Default `'onChange'`. */
  envUpdate?: 'onChange' | 'everyFrame';
}

/**
 * The image-based lighting derived from the sky for
 * {@link Engine.setEnvironmentLighting}. Every field is optional; omitted fields keep their
 * current value (default 1).
 */
export interface EnvironmentLighting {
  /** Diffuse (irradiance) contribution of the sky, finite and >= 0. Default 1. */
  diffuse?: number;
  /** Specular (reflection) contribution of the sky, finite and >= 0. Default 1. */
  specular?: number;
}

/** The chain milestone 2 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Validates and applies a sky. `desc.preset` selects the base values first, then every other
   * given field overrides them; omitted fields keep the preset's value. A refusal changes
   * nothing.
   * @throws RaaError `sky.invalid` (a field is out of range, non-finite or of the wrong type;
   *   the message names the field), `api.invalid_argument` (the argument is not an object),
   *   `api.feature_missing` (no sky feature is installed).
   */
  setSky(desc: SkyDesc): void;

  /**
   * The sky in effect, with every field resolved to a concrete value (the preset merged with
   * the overrides given to {@link Engine.setSky}).
   * @throws RaaError `api.feature_missing` (no sky feature is installed).
   */
  sky(): SkyDesc;

  /**
   * Sets how strongly the sky lights the scene through image-based lighting: `diffuse` scales
   * the irradiance, `specular` the prefiltered reflections. Merges over the current values.
   * @throws RaaError `api.invalid_argument` (a value is not finite or < 0, or the argument is
   *   not an object), `api.feature_missing` (no environment lighting feature is installed).
   */
  setEnvironmentLighting(settings: EnvironmentLighting): void;
}
