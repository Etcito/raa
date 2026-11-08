/** Deterministic GPU threshold source. Blue uses a repeating 16 × 16 rank mask. */
export type CelDither = 'bayer2' | 'bayer4' | 'bayer8' | 'bayer16' | 'ign' | 'white' | 'blue';

/** Controls merged by Engine.setCel. Invalid fields throw cel.invalid without changing state. */
export interface CelSettings {
  /** Integer [2, 32]. Default 10. */
  bands?: number;
  /** Integer [4, 32]. Default 16. */
  hues?: number;
  /** Integer [2, 12]. Default 6. */
  chroma?: number;
  /** Default 'bayer4'. */
  dither?: CelDither;
  /** [0, 1]. Default 0.65. */
  ditherAmount?: number;
  /** [0.5, 6] rendered pixels. Default 1. */
  outlineWidth?: number;
  /** [0, 1]. Default 0.85. */
  outlineSharpness?: number;
  /** Depth and normal breaks as well as silhouettes. Default true. */
  interiorEdges?: boolean;
  /** [0, 1]. Default 0.85. Average/max ink blend; inert at the current detail factor of 1. */
  edgeRetention?: number;
  /** Anchor thresholds to reconstructed world positions. Default true. */
  worldSpace?: boolean;
  /** [1, 512] cells per world unit. Default 48. */
  worldScale?: number;
  /** [0, 1]. Default 0.12. Toon transition width. */
  softness?: number;
  /** [0, 2]. Default 0.7. Toon highlight strength. */
  specular?: number;
  /** Toon Direct lighting on materials with cel:true. Default true. */
  toon?: boolean;
  /** Oklab hue/chroma palette; false retains scalar lightness bands. Default true. */
  palette?: boolean;
  /** Default true. */
  outlines?: boolean;
  /** Quantize ink with the surface; false protects it after the palette. Default false. */
  ditherOutlines?: boolean;
}
/** Complete settings returned by getCel; enabled state is available through passes(). */
export type CelState = Required<CelSettings>;
