// Hand-written types of the chain milestone 8 additions (screen-space reflections, depth of field,
// height fog with aerial perspective, sun shafts, colour grading and named environment presets).
// Documented in docs/api/m14-camera-effects.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): `Vec3` comes from there, `SkyDesc` from the m08 file, `ToneMappingDesc`
// from m07 and `BloomSettings` from m10, and the `Engine` interface below merges with the v0
// declaration of the same name.

/** Screen-space reflections for {@link Engine.setSsr}. Every field is optional and merged over the current values. */
export interface SsrSettings {
  /** Reflection strength, finite, [0, 2]. Default 1. */
  intensity?: number;
  /** Roughness above which a surface gets no reflection, [0, 1]. Default 0.35. */
  maxRoughness?: number;
  /** Thickness of a surface in metres (view space) for ray hits, > 0. Default 0.1. */
  thickness?: number;
}

/** Depth of field for {@link Engine.setDof}. Every field is optional and merged over the current values. */
export interface DofSettings {
  /** Focus distance in metres, > 0. Default 8. Ignored while `autofocus` is set. */
  focusDistance?: number;
  /** Aperture f-number, [1, 32]. Default 2.8. */
  fStop?: number;
  /** Focal length in metres, (0, 0.3]. Default 0.05 (50 mm). */
  focalLength?: number;
  /** Screen point [u, v] in [0, 1] (top left origin) to focus on, or `null` for manual focus. Default `null`. */
  autofocus?: [number, number] | null;
  /** Largest circle of confusion in pixels, [0, 64]. Default 12. */
  maxCoc?: number;
}

/** A height fog preset of {@link FogSettings}. */
export type FogPreset = 'clear' | 'haze' | 'snow' | 'jungle' | 'night';

/** Height fog and aerial perspective for {@link Engine.setFog}. `preset` applies first, then the other fields. */
export interface FogSettings {
  /** Preset applied first. Default `'clear'`. */
  preset?: FogPreset;
  /** Fog density per metre at the base height, >= 0. */
  density?: number;
  /** Decay of the density per metre of height, >= 0. */
  heightFalloff?: number;
  /** Height in metres where the density equals `density`, finite. */
  baseHeight?: number;
  /** Distance in metres before the fog starts, >= 0. */
  start?: number;
  /** Fog colour, linear RGB. */
  color?: Vec3;
  /** Strength of the in-scatter lighting of the fog, [0, 1]. */
  inscatter?: number;
  /** Aerial perspective strength, [0, 2]; 0 turns it off. */
  aerialStrength?: number;
  /** Distance in metres over which aerial perspective reaches full strength, > 0. */
  aerialDistance?: number;
}

/** Sun shafts for {@link Engine.setSunShafts}. Every field is optional and merged over the current values. */
export interface SunShaftsSettings {
  /** Shaft strength, finite, [0, 4]. Default 1. */
  intensity?: number;
  /** Number of samples along the shaft, as a density in [0, 1]. Default 0.5. */
  density?: number;
  /** Distance in metres the shafts march towards the sun, > 0. Default 150. */
  maxDistance?: number;
}

/** A colour grading preset of {@link GradingSettings}. */
export type GradingPreset = 'neutral' | 'candy' | 'ice' | 'jungle' | 'neon';

/** Colour grading for {@link Engine.setGrading}. `preset` applies first, then the other fields. */
export interface GradingSettings {
  /** Preset applied first. Default `'neutral'`. */
  preset?: GradingPreset;
  /** White balance warmth, [-1, 1]; positive is warmer. Default 0. */
  temperature?: number;
  /** White balance green-magenta shift, [-1, 1]. Default 0. */
  tint?: number;
  /** Exposure in EV, [-5, 5]. Default 0. */
  exposure?: number;
  /** Contrast around mid grey, [0, 2]. Default 1. */
  contrast?: number;
  /** Saturation, [0, 2]. Default 1. */
  saturation?: number;
  /** Saturation of the less saturated colours only, [-1, 1]. Default 0. */
  vibrance?: number;
  /** Per-channel shadow offset, linear RGB, each [-1, 1]. Default [0, 0, 0]. */
  lift?: Vec3;
  /** Per-channel midtone exponent, linear RGB, each > 0. Default [1, 1, 1]. */
  gamma?: Vec3;
  /** Per-channel highlight multiplier, linear RGB, each >= 0. Default [1, 1, 1]. */
  gain?: Vec3;
  /** Tint applied to the shadows, linear RGB, each >= 0. Default [1, 1, 1]. */
  shadowsTint?: Vec3;
  /** Tint applied to the highlights, linear RGB, each >= 0. Default [1, 1, 1]. */
  highlightsTint?: Vec3;
  /** Luminance at which the shadow and highlight tints meet, [0, 1]. Default 0.5. */
  splitBalance?: number;
  /** Vignette darkening at the screen corners, [0, 1]. Default 0. */
  vignette?: number;
}

/** A named post preset of {@link EnvironmentDesc.post}: grading, bloom and tone mapping together. */
export type PostPreset = 'neutral' | 'candy' | 'ice' | 'jungle' | 'neon';

/** Post settings of {@link EnvironmentDesc.post}. Each field is applied with the same rules as its own setter. */
export interface PostSettings {
  /** Grading settings, merged as {@link Engine.setGrading} does. */
  grading?: GradingSettings;
  /** Bloom settings, merged as {@link Engine.setBloom} does. */
  bloom?: BloomSettings;
  /** Depth of field settings, merged as {@link Engine.setDof} does. */
  dof?: DofSettings;
  /** Reflection settings, merged as {@link Engine.setSsr} does. */
  ssr?: SsrSettings;
  /** Sun shaft settings, merged as {@link Engine.setSunShafts} does. */
  sunShafts?: SunShaftsSettings;
  /** Tone mapping settings, merged as {@link Engine.setToneMapping} does. */
  toneMapping?: ToneMappingDesc;
}

/**
 * A whole environment for {@link Engine.applyEnvironment}. Every field is optional; the sections
 * are applied in the order sky, sun, fog, post.
 */
export interface EnvironmentDesc {
  /** A sky preset name (`'day' | 'snow' | 'jungle' | 'night'`), or a sky description applied to {@link Engine.setSky}. */
  sky?: string | SkyDesc;
  /** The sun, applied with {@link Engine.setLight}. `dir` is the direction the light travels. */
  sun?: { dir: Vec3; color: Vec3; intensity: number };
  /** A fog preset name, or fog settings applied to {@link Engine.setFog}. */
  fog?: string | FogSettings;
  /** A post preset name (sets grading, bloom and tone mapping), or post settings. */
  post?: string | PostSettings;
}

/** The chain milestone 8 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Validates and applies screen-space reflection settings, merged over the current values. A
   * refusal changes nothing. Enabling and disabling is `setPassEnabled('ssr', on)`.
   * @throws RaaError `ssr.invalid` (field out of range or non-finite; names the field),
   *   `api.invalid_argument` (argument is not an object).
   */
  setSsr(settings: SsrSettings): void;

  /**
   * Validates and applies depth of field settings, merged over the current values. A refusal
   * changes nothing. Enabling and disabling is `setPassEnabled('dof', on)`.
   * @throws RaaError `dof.invalid` (field out of range or non-finite; names the field),
   *   `api.invalid_argument` (argument is not an object).
   */
  setDof(settings: DofSettings): void;

  /**
   * Validates and applies fog settings. `preset` is applied first, then the other fields override
   * it. A refusal changes nothing. Enabling and disabling is `setPassEnabled('fog', on)`.
   * @throws RaaError `fog.unknown_preset` (name is not a {@link FogPreset}), `fog.invalid` (field
   *   out of range or non-finite; names the field), `api.invalid_argument` (argument is not an object).
   */
  setFog(settings: FogSettings): void;

  /**
   * Validates and applies sun shaft settings, merged over the current values. A refusal changes
   * nothing. Enabling and disabling is `setPassEnabled('sunshafts', on)`.
   * @throws RaaError `sunshafts.invalid` (field out of range or non-finite; names the field),
   *   `api.invalid_argument` (argument is not an object).
   */
  setSunShafts(settings: SunShaftsSettings): void;

  /**
   * Validates and applies colour grading. `preset` is applied first, then the other fields
   * override it. A refusal changes nothing. Enabling and disabling is `setPassEnabled('grading', on)`.
   * @throws RaaError `grading.unknown_preset` (name is not a {@link GradingPreset}),
   *   `grading.invalid` (field out of range or non-finite; names the field),
   *   `api.invalid_argument` (argument is not an object).
   */
  setGrading(settings: GradingSettings): void;

  /**
   * Applies a named environment: the sky, the sun, the fog and the post stack, in that order. A
   * string names a preset; an object overrides it. A refusal changes nothing.
   * @throws RaaError `sky.invalid` (sky preset or field, from {@link Engine.setSky}),
   *   `fog.unknown_preset`, `fog.invalid`, `post.unknown_preset` (name is not a {@link PostPreset}),
   *   `post.invalid` (a post field is out of range or non-finite; names the field),
   *   `api.invalid_argument` (argument or a section is not of the documented shape).
   */
  applyEnvironment(env: EnvironmentDesc): void;
}
