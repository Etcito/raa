// Hand-written types of the chain milestone 2 additions (procedural sky with clouds and the
// image-based lighting derived from it). Documented in docs/api/m08-sky-ibl.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): `Vec3` comes from there, and the `Engine` interface below
// merges with the v0 declaration of the same name.

/** The S24 sky evaluation model. `'physical'` is the default. */
export type SkyModel = 'physical' | 'legacy';

/** When the environment cube and its image-based lighting are rebuilt. */
export type SkyEnvUpdate = 'onChange' | 'everyFrame';

/** The built-in sky presets of {@link Engine.setSky}. */
export type SkyPreset = 'day' | 'snow' | 'jungle' | 'night';

/**
 * The procedural cloud layer of {@link SkyDesc}. Every field is optional and merged over the
 * preset (default: the values of the `day` preset).
 */
export interface CloudsDesc {
  /** Fraction of the sky covered by clouds, in [0, 1]. Default 0.45. */
  coverage?: number;
  /** Optical density of the cloud layer, finite and >= 0. Default 1. */
  density?: number;
  /** Cloud base altitude in metres, finite and > 0. Default 1500. */
  altitude?: number;
  /** Horizontal scale of the cloud pattern in metres, finite and > 0. Default 2500. */
  scale?: number;
  /** Wind velocity `[x, z]` in metres per second, both finite. Default [4, 1.5]. */
  wind?: [number, number];
  /** Cloud colour, linear RGB. Default [0.95, 0.95, 0.97]. */
  color?: Vec3;
  /** Bright rim around dense cloud edges, in [0, 1]. Default 0.8. */
  silverLining?: number;
}

/**
 * The physical atmosphere controls of {@link SkyDesc}. Both fields are optional and merged
 * over the preset. Refusals use `sky.invalid` and name `atmosphere.<field>`.
 */
export interface AtmosphereDesc {
  /** Ground albedo, linear RGB with every component in [0, 1]. Default [0.3, 0.3, 0.3]. */
  groundAlbedo?: Vec3;
  /** Multiplier of the Mie scattering coefficient, finite and > 0. Default 1. */
  mieMultiplier?: number;
}

/**
 * A procedural sky description for {@link Engine.setSky}. Every field is optional: the `preset`
 * is applied first and the remaining fields are merged over it; without a preset the current
 * sky is the base. Colours are scene-referred linear HDR RGB triples (values above 1 are
 * allowed); lengths are metres and angles radians. Every numeric field must stay finite when
 * stored as f32. Old fields treat omitted, `undefined` and `null` alike; the S24 fields
 * (`model`, `atmosphere` and its members) reject `null`, unknown keys and values that do not
 * fit f32.
 */
export interface SkyDesc {
  /** Built-in preset applied first. Its selection is an explicit event (see `setSky`). */
  preset?: SkyPreset;
  /** Evaluation model. Default `'physical'` (or the current model without a preset). */
  model?: SkyModel;
  /** Sky colour straight up, linear RGB. Default [0.15, 0.35, 0.85]. */
  zenith?: Vec3;
  /** Sky colour at the horizon, linear RGB. Default [0.85, 0.8, 0.7]. */
  horizon?: Vec3;
  /** Colour below the horizon, linear RGB. Default [0.35, 0.3, 0.25]. */
  ground?: Vec3;
  /** Overall sky radiance multiplier, finite and >= 0. Default 1.6. */
  intensity?: number;
  /** Zenith-to-horizon blend exponent, finite and > 0. Default 4. */
  horizonSoftness?: number;
  /** Angular radius of the sun disc in radians, in [0, 0.2]. Default 0.0047. */
  sunDiscSize?: number;
  /** Radiance multiplier of the sun disc, finite and >= 0. Default 40. */
  sunDiscIntensity?: number;
  /** Cloud layer; merged field by field over the preset's clouds. */
  clouds?: CloudsDesc;
  /** When the environment cube and its lighting are rebuilt. Default `'onChange'`. */
  envUpdate?: SkyEnvUpdate;
  /** Physical atmosphere controls; merged field by field over the preset. */
  atmosphere?: AtmosphereDesc;
}

/**
 * The complete resolved sky returned by {@link Engine.sky}: the base (preset or previous
 * sky) merged with the accepted overrides, with the explicit `preset` intent replaced by the
 * resolved `profile` identity.
 */
export interface ResolvedSkyDesc {
  model: SkyModel;
  profile: SkyPreset;
  zenith: Vec3;
  horizon: Vec3;
  ground: Vec3;
  intensity: number;
  horizonSoftness: number;
  sunDiscSize: number;
  sunDiscIntensity: number;
  clouds: Required<CloudsDesc>;
  envUpdate: SkyEnvUpdate;
  atmosphere: Required<AtmosphereDesc>;
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
   * Validates and applies a sky as one atomic transaction. `desc.preset` selects the base
   * values first (and is an explicit preset event: it resets the preset moment even when
   * every resolved value is unchanged), then every other given field overrides it; an empty
   * patch keeps the accepted sky, moment and weather. Shared `clouds.coverage`, `density`,
   * `altitude` and `wind` values are forwarded only when the patch supplies them, so an
   * omitted field never overrides a later authored weather edit. A refusal changes nothing.
   * @throws RaaError `sky.invalid` (a field is out of range, non-finite, overflows f32, of
   *   the wrong type or unknown; the message names the complete field path),
   *   `api.invalid_argument` (the argument is not an object), `api.feature_missing` (no sky
   *   feature is installed), `pipeline.busy` (called while a frame executes).
   */
  setSky(desc: SkyDesc): void;

  /**
   * The resolved sky in effect: model, profile identity, every artistic value, the complete
   * cloud layer, the environment update policy and the physical atmosphere controls. The
   * result is a fresh object; mutating it never changes the engine. The explicit `preset`
   * intent is not part of the resolved result.
   * @throws RaaError `api.feature_missing` (no sky feature is installed).
   */
  sky(): ResolvedSkyDesc;

  /**
   * Sets how strongly the sky lights the scene through image-based lighting: `diffuse` scales
   * the irradiance, `specular` the prefiltered reflections. Merges over the current values.
   * @throws RaaError `api.invalid_argument` (a value is not finite or < 0, or the argument is
   *   not an object), `api.feature_missing` (no environment lighting feature is installed).
   */
  setEnvironmentLighting(settings: EnvironmentLighting): void;
}
