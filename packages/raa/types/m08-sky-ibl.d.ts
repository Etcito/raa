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

/**
 * The authored celestial time of {@link Engine.setTimeOfDay}. `day` is one-based, `hours`
 * uses local solar time, `latitude` is degrees north and `north` is the scene orientation in
 * radians about +Y. `enabled` controls automatic lighting, `playing` advances the clock (both
 * only while `enabled`), and `speed` is simulated seconds per real second.
 */
export interface TimeOfDaySettings {
  /** Day of the year, an integer in [1, 365]. Default 172. */
  day: number;
  /** Local solar hours, finite and in [0, 24). Default 9. */
  hours: number;
  /** Degrees north, finite and in [-90, 90]. Default 38. */
  latitude: number;
  /** Scene orientation in radians about +Y; finite. Default 0. */
  north: number;
  /** Automatic dominant light (sun/moon) instead of the manual light. Default false. */
  enabled: boolean;
  /** Advance the clock with the engine frame delta. Default false. */
  playing: boolean;
  /** Simulated seconds per real second, finite and >= 0. Default 1. */
  speed: number;
}

/**
 * A partial authored time-of-day patch for {@link Engine.setTimeOfDay}: only supplied
 * fields change. `undefined` is an omission; `null`, unknown keys, wrong types and
 * out-of-domain values throw `sky.invalid` with the complete field path.
 */
export type TimeOfDayDesc = Partial<TimeOfDaySettings>;

/**
 * The authored time settings plus the current playback moment and its celestial snapshot,
 * returned by {@link Engine.timeOfDay}. Day/hours are the authored moment; currentDay and
 * currentHours are the runtime cursor. Directions point toward the bodies.
 */
export interface TimeOfDayState extends TimeOfDaySettings {
  currentDay: number;
  currentHours: number;
  sunDirection: Vec3;
  moonDirection: Vec3;
  moonLit: number;
  starPole: Vec3;
  starEquinox: Vec3;
  dominant: 'manual' | 'sun' | 'moon';
}

/** The ordered cloud-history reset reasons of a prepared view. */
export type SkyResetReason =
  | 'Resize'
  | 'CameraDiscontinuity'
  | 'Weather'
  | 'Model'
  | 'Enable'
  | 'ViewRemoved';

/** The lifecycle status of the physical-sky pipeline. */
export type SkyReadinessStatus = 'pending' | 'ready' | 'failed';

/** The pipeline stage a readiness observation refers to. */
export type SkyReadinessStage =
  | 'install'
  | 'atmosphere'
  | 'milkyWay'
  | 'cloudLighting'
  | 'cloudScreen'
  | 'environment'
  | 'submission';

/** The IBL environment publication copied from its handle; generations are counters. */
export interface SkyEnvironmentPublication {
  requestedGeneration: number;
  publishedGeneration: number | null;
  pending: boolean;
}

/** One prepared view's lifecycle observation, in ascending view-slot order. */
export interface SkyViewDiagnostics {
  view: number;
  width: number;
  height: number;
  invocation: number | null;
  submittedInvocation: number | null;
  skyViewGeneration: number;
  historyValid: boolean;
  resetReasons: SkyResetReason[];
  cloudScreenValid: boolean;
  cloudPanoramaValid: boolean;
  /** Screen-cloud accumulation: committed count n, the animated flag the freeze decision used,
   * whether the last frame skipped the march, and whether history was usable. */
  cloud: {
    count: number;
    animated: boolean;
    frozen: boolean;
    historyUsed: boolean;
    /** Frames that ran the in-frame convergence burst of the screen march (up to 16 marches
     * in one frame after a sky change, so the picture shown is already converged). */
    burstFrames: number;
    /** The world-space cloud dome (shared by every view): `active` = cloudDome is on, `count`
     * its committed accumulation count, `frozen` = the last frame did not march it,
     * `burstFrames` = frames that ran its convergence burst. */
    dome: { active: boolean; count: number; frozen: boolean; burstFrames: number };
  };
}

/** Whether the sky pipeline is ready, pending or failed, with its stage and error. */
export interface SkyReadiness {
  status: SkyReadinessStatus;
  stage: SkyReadinessStage;
  /** The originating failure message while `status` is `'failed'`; otherwise null. */
  error: string | null;
}

/**
 * The exact typed sky/weather lifecycle observation returned by
 * {@link Engine.skyWeatherDiagnostics}. Every counter is a nonnegative safe JS integer; a
 * value beyond that range throws `sky.invalid` instead of rounding. All fields are
 * observations of renderer state: reading them never advances a clock, changes a resource
 * or becomes a saved setting.
 */
export interface SkyWeatherDiagnostics {
  invocation: number | null;
  submittedInvocation: number | null;
  weatherGeneration: number;
  atmosphereGeneration: number;
  environment: SkyEnvironmentPublication;
  /** Prepared views in ascending slot order; no prepared view yields `[]`. */
  views: SkyViewDiagnostics[];
  readiness: SkyReadiness;
}

/** The built-in weather presets, in rank order clear=0 … cirrus=5. */
export type WeatherPreset = 'clear' | 'fair' | 'broken' | 'overcast' | 'storm' | 'cirrus';

/**
 * One complete weather state. Fractions lie in [0, 1]; `base` >= 0 and `thickness` >= 100
 * are metres; `density`, `evolution` and `wind.speed` (m/s) are >= 0; `wind.from` is a
 * finite compass direction in degrees FROM north. Fair defaults: coverage 0.3, type 0.9,
 * base 1200, thickness 1600, density 1, cirrus 0.15, wind 7 from 250, evolution 1, rain 0.
 */
export interface WeatherState {
  coverage: number;
  type: number;
  base: number;
  thickness: number;
  density: number;
  cirrus: number;
  wind: { speed: number; from: number };
  evolution: number;
  rain: number;
}

/**
 * The authored weather settings. Defaults: enabled true, shadows true, changing false,
 * seed 1, transitionSeconds 0 (immediate), the fair target.
 */
export interface WeatherSettings {
  enabled: boolean;
  shadows: boolean;
  /** Whether the seeded weather cycle drives the evolving target. */
  changing: boolean;
  /** Cycle seed, an integer u32. */
  seed: number;
  /** Exponential easing time towards a new target, finite and >= 0; 0 is immediate. */
  transitionSeconds: number;
  /** Screen-cloud march steps per ray, an integer-valued number in [8, 256]. Default 56. */
  cloudSamples: number;
  /** Screen-cloud buffer scale in [0.25, 1]: the buffer is ceil(size * cloudScale). Default 0.5. */
  cloudScale: number;
  /** Per-pixel march-offset noise of the screen clouds. false = constant offset: no crawling while the camera moves (fixed step banding instead; raise cloudSamples). Default true. */
  cloudDither: boolean;
  /** Noise grain, an integer in [1, 16]: one noise value per cloudNoiseSize x cloudNoiseSize buffer texels; larger = coarser. Default 1. */
  cloudNoiseSize: number;
  /** World-space cloud dome: marches the clouds into a persistent 2D texture indexed by world view direction (hemi-octahedral), so a moving or rotating camera never re-marches them; no screen-space cloud march runs while on. Default false. */
  cloudDome: boolean;
  /** Edge of the dome texture in texels, an integer in [128, 2048]. Default 1024. */
  cloudDomeSize: number;
  /** The authored target weather (not the runtime state). */
  target: WeatherState;
}

/**
 * A weather patch for {@link Engine.setWeather}. `undefined` is omission; omitted fields,
 * including `wind.speed`/`wind.from`, keep the authored values. `preset` and `text` are
 * mutually exclusive and supply the base before the structured overrides.
 */
export interface WeatherDesc {
  preset?: WeatherPreset;
  /** A complete `.weather` file, parsed fully before anything is applied. */
  text?: string;
  enabled?: boolean;
  shadows?: boolean;
  changing?: boolean;
  seed?: number;
  transitionSeconds?: number;
  /** Screen-cloud march steps per ray, an integer-valued number in [8, 256]. Default 56. */
  cloudSamples?: number;
  /** Screen-cloud buffer scale in [0.25, 1]; 0.5 is half resolution. Default 0.5. */
  cloudScale?: number;
  /** See the `cloudDither` field of the weather read back by `weather()`. Default true. */
  cloudDither?: boolean;
  /** Integer in [1, 16]. Default 1. */
  cloudNoiseSize?: number;
  /** See the `cloudDome` field of the weather read back by `weather()`. Default false. */
  cloudDome?: boolean;
  /** Integer in [128, 2048]. Default 1024. */
  cloudDomeSize?: number;
  coverage?: number;
  type?: number;
  base?: number;
  thickness?: number;
  density?: number;
  cirrus?: number;
  wind?: { speed?: number; from?: number };
  evolution?: number;
  rain?: number;
}

/**
 * The resolved weather: the authored settings, the runtime `current` state, the running
 * cycle's preset (`null` when no cycle is active) and the `.weather` text of the authored
 * target, generated when the getter is called.
 */
export interface ResolvedWeather extends WeatherSettings {
  current: WeatherState;
  cyclePreset: WeatherPreset | null;
  text: string;
}

/**
 * Parses `.weather` text into a complete state (pure; no Engine or GPU). Starts from fair
 * defaults, strips `#` comments, keeps the last repeated value and clamps as the file
 * format does.
 * @throws Error with a one-based line message for an unknown key or malformed number.
 */
export declare function parseWeatherText(text: string): WeatherState;

/**
 * Formats a weather state as `.weather` text: coverage, type, base, thickness, density,
 * cirrus, wind speed/from, evolution, rain, one setting per line with a final LF (pure).
 */
export declare function formatWeatherText(weather: WeatherState): string;

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

  /**
   * Merges a validated time-of-day patch over the authored settings and applies it once.
   * A changed authored `day` or `hours` resets the runtime cursor to that moment;
   * pause/resume and speed/place/orientation edits keep it. Automatic lighting and the
   * stored manual daylight baseline stay owned by the sky; changing either authored moment
   * does not by itself enable automatic control.
   * @throws RaaError `sky.invalid` (a field is non-finite, out of range, of the wrong type,
   *   unknown or overflows f32; `day` must be an integer and `hours` in [0, 24)),
   *   `api.invalid_argument` (the argument is not an object), `api.feature_missing`,
   *   `pipeline.busy` (called while a frame executes).
   */
  setTimeOfDay(desc: TimeOfDayDesc): void;

  /**
   * The authored time settings, the runtime playback moment and its celestial snapshot as a
   * fresh object. It is never persisted and never advances the clock.
   * @throws RaaError `api.feature_missing` (no sky feature is installed).
   */
  timeOfDay(): TimeOfDayState;

  /**
   * The exact typed sky/weather lifecycle observation: invocation identities, weather and
   * atmosphere generations, the environment publication, the prepared views in slot order
   * with their ordered reset reasons, and readiness. Allocates fresh objects only on
   * request and is observational, never persisted.
   * @throws RaaError `api.feature_missing` (no sky feature is installed), `sky.invalid` (a
   *   counter would not be a nonnegative safe JS integer).
   */
  skyWeatherDiagnostics(): SkyWeatherDiagnostics;

  /**
   * Parses a weather patch fully, then applies it atomically once. A `preset` (even with
   * overrides) is an explicit cycle start with that preset's identity; `text` starts from
   * fair. Either explicit base, a changed `seed` or `changing` false→true restarts the
   * cycle; an unchanged `changing: true` alone keeps it. A manual target edit stops
   * changing unless the same patch sets `changing: true`. A refusal changes nothing.
   * @throws RaaError `sky.invalid` (wrong type, null, unknown key, out-of-range or
   *   non-finite field, unknown preset, `preset` with `text`, malformed text; the message
   *   names the field path), `api.invalid_argument` (not an object), `api.feature_missing`,
   *   `pipeline.busy`.
   */
  setWeather(desc: WeatherDesc): void;

  /**
   * The authored settings, runtime current state, active cycle preset and authored target
   * text as a fresh object; it never advances weather and is the value to persist (without
   * `current`/`cyclePreset`).
   * @throws RaaError `api.feature_missing` (no sky feature is installed).
   */
  weather(): ResolvedWeather;
}
