// Hand-written types of the chain milestone 4 additions (WGSL material functions, layered
// surface materials with clear coat and sheen, fur shells, emissive bloom). Documented in
// docs/api/m10-materials.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): the `Engine` interface below merges with the v0 declaration of the
// same name.

/** Kind of a material function parameter: 1, 3 or 3 (colour, sRGB [0, 1]) numbers. */
export type ParamKind = 'scalar' | 'vec3' | 'color';

/** One parameter of a material function. */
export interface MaterialFunctionParam {
  /** Parameter name, as used in `SurfaceMaterialDesc.params`. */
  name: string;
  /** `scalar` takes a number, `vec3` and `color` take three numbers. */
  kind: ParamKind;
  /** Default value: 1 number for `scalar`, 3 for `vec3` and `color` (sRGB [0, 1]). */
  default: number[];
}

/** Description of a registered material function, as listed by {@link Engine.materialFunctions}. */
export interface MaterialFunctionInfo {
  name: string;
  /** True for the seven functions shipped with the engine, false for `addMaterialFunction`. */
  builtin: boolean;
  /** Parameters in positional order p0..p7 of the shader record. */
  params: MaterialFunctionParam[];
}

/** How a surface is composited. */
export type SurfaceBlend = 'opaque' | 'transparent' | 'transmissive';

/**
 * Description of a surface material. Every field is optional. Colours are sRGB in [0, 1]
 * except `emissive`, which is linear and may exceed 1 (HDR, feeds bloom). Lengths are metres.
 */
export interface SurfaceMaterialDesc {
  /** Material function name. Default `raa.standard`. */
  function?: string;
  /** Function parameters by name: a number for `scalar`, three numbers for `vec3`/`color`. Missing = default. */
  params?: Record<string, number | number[]>;
  /** Default `opaque`. `transparent` and `transmissive` need the transparency feature, else `material.blend_unavailable`. */
  blend?: SurfaceBlend;
  /** Disables back-face culling. Default false. */
  doubleSided?: boolean;
  /** Casts sun shadows. Default true. */
  castShadow?: boolean;
  /** Base colour, sRGB [0, 1]. Default [0.8, 0.8, 0.8]. */
  color?: [number, number, number];
  /** Perceptual roughness [0, 1]. Default 0.5. */
  roughness?: number;
  /** Metallic [0, 1]. Default 0. */
  metallic?: number;
  /** Emissive radiance, linear RGB, each >= 0 (not clamped to 1). Default [0, 0, 0]. */
  emissive?: [number, number, number];
  /** Opacity [0, 1]; < 1 requires a non-opaque blend. Default 1. */
  opacity?: number;
  /** Clear coat layer strength [0, 1]. Default 0. */
  clearcoat?: number;
  /** Clear coat roughness [0, 1]. Default 0.1. */
  clearcoatRoughness?: number;
  /** Sheen colour, sRGB [0, 1]. Default [0, 0, 0] (no sheen). */
  sheenColor?: [number, number, number];
  /** Sheen roughness [0, 1]. Default 0.5. */
  sheenRoughness?: number;
  /** Transmission [0, 1]; > 0 requires `blend: 'transmissive'`. Default 0. */
  transmission?: number;
  /** Index of refraction, [1, 3]. Default 1.5. */
  ior?: number;
  /** Volume thickness in metres, >= 0. Default 0. */
  thickness?: number;
  /** Absorption tint, sRGB [0, 1]. Default [1, 1, 1]. */
  attenuationColor?: [number, number, number];
  /** Absorption distance in metres, > 0. Default Infinity (no absorption). */
  attenuationDistance?: number;
  /** Fur shell extrusion length in metres, >= 0; > 0 enables fur. Default 0. */
  furLength?: number;
  /** Strands per cm2, > 0. Default 2000. */
  furDensity?: number;
  /** Gravity droop of the strands [0, 1]. Default 0. */
  furDroop?: number;
  /** Number of shells, integer 1..64. Default 32. */
  furShells?: number;
}

/** Bloom settings for {@link Engine.setBloom}; fields merge over the current values. */
export interface BloomSettings {
  /** Bloom mix strength, finite, [0, 1]. Default 0.04. */
  strength?: number;
  /** Blur radius, finite, [0, 1]. Default 0.85. */
  radius?: number;
}

/** The chain milestone 4 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Registers a user WGSL material function under `name`. The source must define
   * `material_main` (rules in docs/api/m10-materials.md). It then composes for every pass kind.
   * @throws RaaError `material.fn_invalid` (rule broken or naga message with line:col),
   *   `material.fn_duplicate` (name already registered, built-in or user),
   *   `api.invalid_argument` (name or wgsl is not a non-empty string).
   */
  addMaterialFunction(name: string, wgsl: string): void;

  /** Lists built-in and user material functions, built-ins first, in registration order. */
  materialFunctions(): MaterialFunctionInfo[];

  /**
   * Creates a surface material and returns its handle, usable with `addInstance` and sub-mesh
   * materials.
   * @throws RaaError `material.unknown_fn`, `material.unknown_param`, `material.param_type`
   *   (params entry has the wrong arity or is non-finite), `material.invalid` (a field is out of
   *   range, non-finite or inconsistent; the message names it), `material.blend_unavailable`,
   *   `api.invalid_argument` (desc is not an object).
   */
  addSurfaceMaterial(desc: SurfaceMaterialDesc): number;

  /**
   * Replaces the material `id` with `desc` (omitted fields take their defaults). A refusal
   * changes nothing.
   * @throws RaaError the codes of {@link Engine.addSurfaceMaterial}, plus `api.unknown_handle`.
   */
  setSurfaceMaterial(id: number, desc: SurfaceMaterialDesc): void;

  /**
   * Validates and applies bloom settings, merged over the current values. A refusal changes
   * nothing. Enabling and disabling is `setPassEnabled('bloom', on)`.
   * @throws RaaError `bloom.invalid` (field out of range or non-finite; names the field),
   *   `api.invalid_argument` (argument is not an object).
   */
  setBloom(settings: BloomSettings): void;
}
