/** The four supported local-light kinds. */
export type LocalLightKind = 'point' | 'spot' | 'directional' | 'area';

/** Stable positive safe-integer handle returned by {@link Engine.addLight}. */
export type LightHandle = number;

/** Description accepted by {@link Engine.addLight}; omitted values use S17 defaults. */
export interface LocalLightDesc {
  kind?: LocalLightKind;
  /** World position in metres, or local position while attached. Default [0, 0, 0]. */
  position?: [number, number, number];
  /** Nonzero travel direction; normalized on acceptance. Default [0, 0, -1]. */
  direction?: [number, number, number];
  /** sRGB colour in [0, 1], converted once to linear RGB. Default [1, 1, 1]. */
  color?: [number, number, number];
  /** Nonnegative intensity. Default 2. */
  intensity?: number;
  /** Positive range in metres. Default 6. */
  range?: number;
  /** Full spot-cone angle in degrees, [1, 179]. Default 35. */
  angleDeg?: number;
  /** Optional full inner spot-cone angle in degrees, [1, 179] and <= angleDeg. */
  innerAngleDeg?: number;
  /** Positive full width and height in metres. Default [1, 1]. */
  size?: [number, number];
  /** Nonnegative authoring metadata; does not control shadows. Default 0. */
  radius?: number;
  /** Authoring metadata; local-light shadows are not enabled. Default false. */
  shadows?: boolean;
}

/** Partial changes merged atomically onto an existing light. */
export type LocalLightChanges = Partial<LocalLightDesc>;

/** Current light counts and the latest completed tile-overflow readback. */
export interface LightStats {
  activeLights: number;
  attachedLights: number;
  tileColumns: number;
  tileRows: number;
  tileCapacity: 127;
  /** Excluded light/tile references, or null before a completed result exists. */
  droppedLightReferences: number | null;
  /** Invocation frame described by the completed result, or null before one exists. */
  statsFrameIndex: number | null;
  /** True while overflow readback is pending. */
  pending: boolean;
}

/** Area-light shading mode: LTC, or the diffuse-only polygon form factor. */
export type AreaLightShadingMode = 'ltc' | 'formFactor';

/** Settings accepted by {@link Engine.setAreaLightShading}. */
export interface AreaLightShadingDesc {
  mode: AreaLightShadingMode;
}

/**
 * Area-light soft-shadow tier. `'auto'` (default): hero maps for the selected lights nearer
 * than `heroDistance`, blended to the cone over `band` metres, and the direct term faded out
 * over `band` metres before `fadeDistance` while GI is active. `'hero'`: perspective shadow
 * maps of the `heroCount` area lights with the largest screen coverage, filtered with PCSS
 * whose light size is the rectangle size; works with GI off. `'cone'`: soft visibility through
 * the GI voxel distance field (visibility 1 without GI). `'fade'`: no direct term while GI is
 * active, so the light reaches the scene through the GI volume only.
 */
export type AreaLightShadowTier = 'auto' | 'hero' | 'cone' | 'fade';

/**
 * Settings accepted by {@link Engine.setAreaLightShadows}. Absent keys keep the current value;
 * the merged settings are validated as a whole. A refusal throws `light.invalid_area_shadows`
 * with the message `<key>: <reason>` and changes nothing.
 */
export interface AreaLightShadowsDesc {
  /** Visibility tier. Default `'auto'`. */
  tier?: AreaLightShadowTier;
  /** Integer in [0, 4]: hero shadow maps, 0 disables them. Default 0. */
  heroCount?: number;
  /** Metres, > 0: where the hero weight starts to fade. Default 25. */
  heroDistance?: number;
  /** Metres, > heroDistance: where the direct term has fully faded. Default 80. */
  fadeDistance?: number;
  /** Metres, > 0 and <= heroDistance: width of the blend and fade bands. Default 5. */
  band?: number;
}

/**
 * Current area-light soft-shadow tiers, returned by {@link Engine.areaLightShadows}. The
 * setting is engine-wide: it persists until changed and applies to every view.
 */
export interface AreaLightShadows {
  tier: AreaLightShadowTier;
  heroCount: number;
  heroDistance: number;
  fadeDistance: number;
  band: number;
  /**
   * Area lights that had a hero shadow map in the last prepared frame; 0 before the first
   * frame.
   */
  activeHeroes: number;
}

export interface Engine {
  /** Adds a local light; capacity is 256. The sun remains configured through setLight. */
  addLight(desc: LocalLightDesc): LightHandle;
  /** Merges and validates changes before mutation; explicit undefined resets a field. */
  updateLight(handle: LightHandle, changes: LocalLightChanges): void;
  /** Removes a light; its handle is never reused. */
  removeLight(handle: LightHandle): void;
  /** Attaches to an instance-group handle returned by addInstance/addMeshInstance. */
  attachLight(
    handle: LightHandle,
    instanceId: number,
    localOffset: [number, number, number],
  ): void;
  /** Detaches while preserving the latest world position and direction. */
  detachLight(handle: LightHandle): void;
  /** Returns synchronous counts and the latest fully completed asynchronous stats. */
  lightStats(): LightStats;
  /**
   * Selects the engine-wide area-light shading mode: `'ltc'` integrates diffuse and GGX
   * specular over the rectangle with the LTC tables, `'formFactor'` uses the diffuse polygon
   * form factor only. The setting persists until changed and applies to every view; a new
   * Engine starts at `'ltc'`. A non-object argument, a missing `mode` or any other value
   * throws `api.invalid_argument` and changes nothing.
   */
  setAreaLightShading(settings: AreaLightShadingDesc): void;
  /** Returns the current area-light shading mode as `{ mode }`. */
  areaLightShading(): AreaLightShadingDesc;
  /**
   * Sets the engine-wide area-light soft-shadow tiers (defaults `tier: 'auto'`, `heroCount: 0`,
   * `heroDistance: 25`, `fadeDistance: 80`, `band: 5`). A non-object argument throws
   * `light.invalid_area_shadows` with `settings: must be an object`; a present key that is not
   * a number throws `<key>: must be a number`; a `heroCount` that is not an integer in [0, 4]
   * throws `heroCount: must be an integer in [0, 4] (got <value>)`. Every refusal changes nothing.
   */
  setAreaLightShadows(settings: AreaLightShadowsDesc): void;
  /** Returns the current area-light soft-shadow tiers and the active hero count. */
  areaLightShadows(): AreaLightShadows;
}
