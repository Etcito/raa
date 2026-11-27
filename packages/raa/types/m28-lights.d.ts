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
  /**
   * Nonnegative emitter radius in metres. For a shadow-casting point or spot light the shadow
   * rays start this far from the light, so geometry inside it (the light's own bulb, lantern
   * glass) casts nothing and stays lit. Default 0.
   */
  radius?: number;
  /**
   * Point and spot lights: cast shadows from a cached cube shadow map traced through the mesh
   * BVH of a BVH-consuming GI (DDGI graphs; none otherwise, the light stays unshadowed); see
   * {@link Engine.setLocalShadows}. At most 32 shadow-casting lights; later ones stay
   * unshadowed. Area lights use their own tiers; directional lights ignore it. Default false.
   */
  shadows?: boolean;
}

/** Settings accepted by {@link Engine.setLocalShadows}; omitted keys keep their value. */
export interface LocalShadowsDesc {
  /** Default true. */
  enabled?: boolean;
  /** Texels per cube-face side, a multiple of 8 in [64, 1024]. Default 256. */
  resolution?: number;
  /** Cube faces traced per frame at most, [1, 24]. Default 3. */
  facesPerFrame?: number;
}

/** {@link Engine.localShadowStats}: settings and the last frame's cache state. */
export interface LocalShadowStats {
  enabled: boolean;
  resolution: number;
  facesPerFrame: number;
  /** Shadow-casting lights holding a slot. */
  lights: number;
  /** Slots the face array holds (a multiple of 4). */
  capacity: number;
  /** Faces of the current lights traced and still valid, and still to trace. */
  readyFaces: number;
  pendingFaces: number;
  /** Faces traced in the last frame and since installation. */
  tracedFaces: number;
  tracedTotal: number;
  /** Triangles of the traced BVH (0: no trace geometry, nothing is traced). */
  triangles: number;
  /** Bytes of the R32Float face array. */
  bytes: number;
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
 * over `band` metres before `fadeDistance` while GI is active. While GI is active, `'auto'`
 * also uses the mid tier: one stratified shadow ray per pixel and light toward a 2x2 or 3x3
 * cell of the rectangle, traced against the screen depth pyramid, the GI distance field and
 * SDF trace content, combined with the analytic light by the ratio estimator and denoised
 * spatially and temporally, for up to `stochasticCount` lights nearer than
 * `stochasticDistance`, blended to the cone tier over `band` metres. `'hero'`: perspective
 * shadow maps of the `heroCount` area lights with the largest screen coverage, filtered with
 * PCSS whose light size is the rectangle size; works with GI off. `'cone'`: soft visibility
 * through the GI voxel distance field (visibility 1 without GI). `'fade'`: no direct term
 * while GI is active, so the light reaches the scene through the GI volume only.
 */
export type AreaLightShadowTier = 'auto' | 'hero' | 'cone' | 'fade';

/**
 * Stochastic shadow mode. `'stratified'` (default) is the M6 mid tier: one stratified shadow
 * ray per pixel and light toward a cell of the rectangle. `'restir'` is the M7 ultra tier:
 * ReSTIR DI with compute-traced visibility in the same `gi.area-shadow` pass.
 */
export type AreaLightStochasticMode = 'stratified' | 'restir';

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
  /**
   * Integer in [0, 8]: area lights that get the mid tier while GI is active; 0 disables the
   * tier. Default 4.
   */
  stochasticCount?: number;
  /**
   * Metres, > 0: beyond it, after the `band`, lights use the cone tier. Default 40.
   */
  stochasticDistance?: number;
  /** Strata of the light rectangle: 4 (2x2) or 9 (3x3). Default 4. */
  stochasticCells?: 4 | 9;
  /**
   * Stochastic shadow mode: `'stratified'` (default) is the M6 mid tier; `'restir'` is the
   * ultra tier, ReSTIR DI with compute-traced visibility.
   */
  stochasticMode?: AreaLightStochasticMode;
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
  stochasticCount: number;
  stochasticDistance: number;
  stochasticCells: number;
  stochasticMode: AreaLightStochasticMode;
  /**
   * Area lights that held a mid-tier slot in the last prepared frame; 0 before the first
   * frame.
   */
  activeStochastic: number;
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
   * Sets the engine-wide area-light soft-shadow tiers and the M6/M7 stochastic mid and ultra
   * tiers (defaults `tier: 'auto'`, `heroCount: 0`, `heroDistance: 25`, `fadeDistance: 80`,
   * `band: 5`, `stochasticCount: 4`, `stochasticDistance: 40`, `stochasticCells: 4`,
   * `stochasticMode: 'stratified'`). A non-object
   * argument throws `light.invalid_area_shadows` with `settings: must be an object`; a present
   * key that is not a number throws `<key>: must be a number`; a `heroCount` that is not an
   * integer in [0, 4] throws `heroCount: must be an integer in [0, 4] (got <value>)`; a
   * `stochasticCount` that is not an integer in [0, 8] throws `stochasticCount: must be an
   * integer in [0, 8] (got <value>)`; a `stochasticCells` other than 4 or 9 throws
   * `stochasticCells: must be 4 or 9 (got <value>)`; a `stochasticDistance` that is not finite
   * or not > 0 throws `stochasticDistance: must be finite and > 0 (got <value>)`; a
   * `stochasticMode` other than `'stratified'` or `'restir'`, or a non-string, throws
   * `stochasticMode: must be 'stratified' or 'restir'`. Every refusal changes nothing.
   */
  setAreaLightShadows(settings: AreaLightShadowsDesc): void;
  /**
   * Returns the current area-light soft-shadow tiers, the active hero and mid-tier slot counts
   * and the stochastic mode.
   */
  areaLightShadows(): AreaLightShadows;
  /**
   * Configures the cached cube shadows of the point and spot lights with `shadows: true`. Each
   * such light holds 6 R32Float faces (distance from the light to the first occluder) traced once
   * through the shared mesh BVH, starting at the light's `radius`; a face is traced again only
   * when its light moves (position, range, radius) or the geometry changes inside its range. At
   * most `facesPerFrame` faces are traced per frame, nearest lights first; an untraced face reads
   * as unoccluded. The forward reads them with 2x2 PCF (direct and diffuse-transmission terms).
   * A non-object argument throws `api.invalid_argument`; an out-of-range value throws
   * `shadow.local_invalid`; nothing changes on refusal.
   */
  setLocalShadows(settings: LocalShadowsDesc): void;
  /** Re-traces every cached local shadow face. */
  invalidateLocalShadows(): void;
  /** Settings and cache state of the local shadows, or null when not installed. */
  localShadowStats(): LocalShadowStats | null;
}
