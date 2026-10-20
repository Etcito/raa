// Hand-written types of @etcito/raa 0.1.0 (public API v0). The wasm-bindgen output in
// pkg/*.d.ts is an implementation detail and is not re-exported.
//
// Conventions: right-handed, Y up, the camera looks down -Z, lengths in metres, angles in
// radians unless the name says `Deg`, times in milliseconds. Matrices are column-major.
// Colours are linear RGB unless documented as sRGB. Every failure throws a RaaError.

/** A 3-component vector `[x, y, z]` (metres for positions, unitless for directions). */
export type Vec3 = [number, number, number];

/** A rotation quaternion `[x, y, z, w]`; the identity is `[0, 0, 0, 1]`. */
export type Quat = [number, number, number, number];

/**
 * The 8 primitive kinds. All are centred at the origin; axial kinds use the Y axis.
 * Parameters (metres) per kind:
 * - `sphere`: `{ radius }`, radius > 0
 * - `box`: `{ halfExtents: Vec3 }`, each > 0
 * - `roundBox`: `{ halfExtents: Vec3, radius }`, 0 <= radius <= min(halfExtents)
 * - `capsule`: `{ radius, halfHeight }`, radius > 0, halfHeight >= 0
 * - `cylinder`: `{ radius, halfHeight }`, both > 0
 * - `cone`: `{ radius, halfHeight, radiusTop? }`, radius and radiusTop >= 0 and not both 0,
 *   halfHeight > 0; base radius at y = -halfHeight, top radius (default 0, a point) at +halfHeight
 *
 * A raa.prims asset may also give the published v0 names (box `size`, capsule and cylinder
 * `height`, cone `radiusBottom` and `radiusTop`); the loader maps them to these.
 * - `torus`: `{ majorRadius, minorRadius }`, 0 < minorRadius < majorRadius; ring in XZ
 * - `ellipsoid`: `{ radii: Vec3 }`, each > 0
 */
export type PrimKind =
  | 'sphere'
  | 'box'
  | 'roundBox'
  | 'capsule'
  | 'cylinder'
  | 'cone'
  | 'torus'
  | 'ellipsoid';

/** Options of {@link RaaModule.createEngine}. Every field is optional. */
export interface EngineOptions {
  /** Capacity of nodes and instances. Default 16384. */
  maxInstances?: number;
  /** Maximum number of materials (asset materials included). Default 256. */
  maxMaterials?: number;
  /** GPU adapter preference. Default 'high-performance'. Another value throws `api.invalid_argument`. */
  powerPreference?: 'high-performance' | 'low-power';
}

/** A material description for {@link Engine.addMaterial} and {@link Engine.setMaterial}. */
export interface MaterialDesc {
  /** Base colour, sRGB in [0, 1] (converted to linear). Required. */
  color: Vec3;
  /** Roughness in [0, 1]. Default 0.6. */
  roughness?: number;
  /** Metallic in [0, 1]. Default 0. */
  metallic?: number;
  /** Emissive colour, linear RGB, each >= 0. Default [0, 0, 0]. */
  emissive?: Vec3;
  /** 'flat' = unlit base + emissive; 'pbr' = lit PBR-lite. Default 'pbr'. */
  model?: 'flat' | 'pbr';
}

/** A camera description for {@link Engine.setCamera}. The aspect follows the canvas. */
export interface CameraDesc {
  /** Camera position (metres). Required. */
  eye: Vec3;
  /** Point looked at (metres); must differ from `eye`. Required. */
  target: Vec3;
  /** Up direction. Default [0, 1, 0]. */
  up?: Vec3;
  /** Vertical field of view in degrees, in (0, 180). Default 60. */
  fovYDeg?: number;
  /** Near plane distance (metres), > 0. Default 0.1. */
  near?: number;
  /** Far plane distance (metres), > near. Default 200. */
  far?: number;
}

/** The directional light for {@link Engine.setLight}. All fields are required. */
export interface LightDesc {
  /** Direction the light travels (normalised by the engine; must be non-zero). */
  direction: Vec3;
  /** Linear RGB colour. */
  color: Vec3;
  /** Intensity multiplier, >= 0. */
  intensity: number;
}

/** The ambient term for {@link Engine.setAmbient}. All fields are required. */
export interface AmbientDesc {
  /** Linear RGB colour. */
  color: Vec3;
  /** Intensity multiplier, >= 0. */
  intensity: number;
}

/** The GPU adapter the engine runs on, as reported by WebGPU. */
export interface AdapterInfo {
  /** Adapter vendor (may be empty). */
  vendor: string;
  /** Adapter architecture (may be empty). */
  architecture: string;
  /** Adapter device (may be empty). */
  device: string;
  /** Adapter description (may be empty). */
  description: string;
  /** True when the browser handed out a fallback (software) adapter. */
  isFallbackAdapter: boolean;
}

/** Frame statistics returned by {@link Engine.stats}. */
export interface FrameStats {
  /** The adapter in use. */
  adapter: AdapterInfo;
  /**
   * True for a hardware GPU: not a fallback adapter and no adapter field matches
   * swiftshader, llvmpipe, lavapipe, software or basic render.
   */
  hardware: boolean;
  /** Milliseconds (navigation-relative) at which frame 0's GPU work completed; null before that. */
  firstFrameMs: number | null;
  /** Number of frames rendered so far. */
  frames: number;
  /** Median CPU time per frame in milliseconds (update + record + submit). */
  cpuMsMedian: number;
  /** 95th percentile CPU time per frame in milliseconds. */
  cpuMsP95: number;
  /** Instances drawn in the last frame. */
  instances: number;
  /** Draw calls in the last frame. */
  drawCalls: number;
  /** Triangles drawn in the last frame. */
  triangles: number;
  /** Heap allocations in the last frame when an allocation counter is linked, else null. */
  allocsPerFrameTotal: number | null;
}

/** The result of {@link Engine.loadPrimsAsset}. */
export interface AssetInfo {
  /** Asset handle, for {@link Engine.spawnAsset}. */
  id: number;
  /** The asset's `name`. */
  name: string;
  /** Total triangles generated at the load quality (after booleans). */
  triangles: number;
  /** The asset's triangle budget (default 1500). */
  budget: number;
  /** Mesh handles, one per asset mesh, baked into asset space. */
  meshes: number[];
  /** Material handles, in the asset's material order. */
  materials: number[];
  /** Anchor name to its asset-space matrix: 16 floats, column-major. */
  anchors: Record<string, Float32Array>;
}

/** The result of {@link Engine.spawnAsset}. All handles are new. */
export interface SpawnedAsset {
  /** The new root node handle. */
  root: number;
  /** One identity child node handle per asset mesh, in mesh order. */
  nodes: number[];
  /** One instance handle per child node, in mesh order. */
  instances: number[];
}

/**
 * Every error thrown by the engine. `message` is `"<code>: <human message>"`.
 * Codes are stable, of the form `<area>.<snake_case>`:
 * - `api.invalid_argument`, `api.unknown_param`, `api.unknown_handle`,
 *   `api.length_mismatch`, `api.disposed`
 * - `geom.unknown_kind`, `geom.missing_param`, `geom.invalid_param`, `geom.invalid_mesh`
 * - `material.invalid`, `material.unknown_id`, `material.unknown`
 * - `scene.stale_node`, `scene.stale_instance`, `scene.parent_cycle`
 * - `render.too_many_materials`
 * - `core.no_adapter`, `core.request_device`, `core.surface`, `core.resource`,
 *   `core.gpu`, `core.unsupported`
 * - prims asset loader codes, kept as is: `json.*`, `format.*`, `schema.*`, `value.*`,
 *   `id.*`, `material.unknown`, `kind.unknown`, `kind.params`, `cap.nodes`, `cap.depth`,
 *   `cap.budget`, `collision.*`, `boolean.unsupported`, `boolean.csg.*`, `loader.*`
 */
export interface RaaError extends Error {
  /** The stable error code. */
  code: string;
  /** Always 'RaaError'. */
  name: 'RaaError';
}

/**
 * One engine: one canvas, one scene, a forward renderer. Handles (meshes, materials,
 * nodes, instances, assets) are plain numbers, starting at 1, increasing, never reused.
 * An unknown or removed handle throws `api.unknown_handle`. Handles of descendants and
 * instances removed through {@link Engine.removeNode} throw `scene.stale_node` or
 * `scene.stale_instance`. After {@link Engine.dispose} every method throws `api.disposed`.
 */
export interface Engine {
  /**
   * Adds a material and returns its handle.
   * @throws RaaError `material.invalid`, `api.invalid_argument`, `render.too_many_materials`.
   */
  addMaterial(desc: MaterialDesc): number;

  /**
   * Replaces the material behind `id`.
   * @throws RaaError `api.unknown_handle`, `material.invalid`, `api.invalid_argument`.
   */
  setMaterial(id: number, desc: MaterialDesc): void;

  /**
   * Generates a primitive, uploads it and returns its mesh handle.
   * @param kind One of the 8 {@link PrimKind} names.
   * @param params Exactly the kind's parameters in metres (see {@link PrimKind}); an extra
   *   key throws `api.unknown_param`, a missing one `geom.missing_param`.
   * @param segments Tessellation quality, default 24 (LOD0); rounded and clamped to [3, 256].
   * @throws RaaError `geom.unknown_kind`, `geom.missing_param`, `geom.invalid_param`,
   *   `api.unknown_param`, `api.invalid_argument`.
   */
  primitiveMesh(kind: PrimKind, params: Record<string, number | Vec3>, segments?: number): number;

  /**
   * Frees a mesh; its handle is never reused.
   * @throws RaaError `api.unknown_handle`.
   */
  removeMesh(id: number): void;

  /**
   * Loads a prims v0 asset (JSON text) at `segments` with the 8 primitive kinds (strict)
   * and CSG booleans. Registers its materials and its meshes (baked into asset space).
   * @param json The asset text (format `raa.prims`, version 0).
   * @param segments Tessellation quality, default 24 (LOD0); rounded and clamped to [3, 256].
   * @throws RaaError with the loader's code (for example `kind.unknown`, `cap.budget`,
   *   `boolean.csg.*`), or `render.too_many_materials`.
   */
  loadPrimsAsset(json: string, segments?: number): AssetInfo;

  /**
   * Spawns a loaded asset: a new root node under `parent` (a root when absent), one
   * identity child node per asset mesh, each with one instance of that mesh and material.
   * @throws RaaError `api.unknown_handle`, `scene.stale_node`.
   */
  spawnAsset(asset: number, parent?: number | null): SpawnedAsset;

  /**
   * Adds a node and returns its handle.
   * @param parent Parent node handle, or null for a root.
   * @param t Translation in metres, default [0, 0, 0].
   * @param r Rotation quaternion [x, y, z, w], default [0, 0, 0, 1].
   * @param s Scale, default [1, 1, 1].
   * @throws RaaError `api.unknown_handle`, `scene.stale_node`, `api.invalid_argument`.
   */
  addNode(
    parent: number | null,
    t?: Vec3 | Float32Array,
    r?: Quat | Float32Array,
    s?: Vec3 | Float32Array,
  ): number;

  /**
   * Sets the local transform of `node`: translation `t` (metres), rotation `r`
   * [x, y, z, w], scale `s`.
   * @throws RaaError `api.unknown_handle`, `scene.stale_node`, `api.invalid_argument`.
   */
  setTransform(
    node: number,
    t: Vec3 | Float32Array,
    r: Quat | Float32Array,
    s: Vec3 | Float32Array,
  ): void;

  /**
   * Sets many local transforms at once, without allocating in steady state.
   * @param nodes Node handles.
   * @param trs 10 floats per node: t (3, metres), r (4, [x, y, z, w]), s (3).
   * @throws RaaError `api.length_mismatch` when `trs.length !== 10 * nodes.length`;
   *   `api.unknown_handle`, `scene.stale_node`, `api.invalid_argument` for an entry
   *   (nodes before the failing entry keep their new transform).
   */
  setTransforms(nodes: Uint32Array, trs: Float32Array): void;

  /**
   * Removes `node`, its subtree and their instances. Handles of removed descendants and
   * instances then throw `scene.stale_node` or `scene.stale_instance`.
   * @throws RaaError `api.unknown_handle`, `scene.stale_node`.
   */
  removeNode(node: number): void;

  /**
   * Draws `mesh` with `material` at `node` and returns the instance handle.
   * @throws RaaError `api.unknown_handle`, `scene.stale_node`.
   */
  addInstance(node: number, mesh: number, material: number): number;

  /**
   * Removes an instance; its handle is never reused.
   * @throws RaaError `api.unknown_handle`, `scene.stale_instance`.
   */
  removeInstance(id: number): void;

  /**
   * Shows or hides an instance (visible by default).
   * @throws RaaError `api.unknown_handle`, `scene.stale_instance`.
   */
  setVisible(id: number, visible: boolean): void;

  /**
   * Sets the camera.
   * @throws RaaError `api.invalid_argument` (fovYDeg outside (0, 180), near <= 0,
   *   far <= near, eye equal to target).
   */
  setCamera(desc: CameraDesc): void;

  /**
   * Rotates the orbit camera and zooms it.
   * @param dYawRad Yaw delta in radians.
   * @param dPitchRad Pitch delta in radians (pitch clamped to +-89 degrees).
   * @param zoomFactor Multiplies the orbit distance, > 0 (1 = no zoom).
   * @throws RaaError `api.invalid_argument`.
   */
  orbit(dYawRad: number, dPitchRad: number, zoomFactor: number): void;

  /**
   * Replaces the orbit camera (default target [0, 0, 0], distance 6).
   * @param target Orbit centre in metres.
   * @param distance Distance to the target in metres, > 0.
   * @param yawRad Yaw in radians.
   * @param pitchRad Pitch in radians (clamped to +-89 degrees).
   * @throws RaaError `api.invalid_argument`.
   */
  setOrbit(target: Vec3 | Float32Array, distance: number, yawRad: number, pitchRad: number): void;

  /**
   * Sets the directional light.
   * @throws RaaError `api.invalid_argument` (zero direction, negative intensity).
   */
  setLight(desc: LightDesc): void;

  /**
   * Sets the ambient term.
   * @throws RaaError `api.invalid_argument` (negative intensity).
   */
  setAmbient(desc: AmbientDesc): void;

  /**
   * Resizes the canvas backing store in pixels (0 clamps to 1); the camera aspect follows.
   * @throws RaaError `api.disposed`.
   */
  resize(width: number, height: number): void;

  /**
   * Renders one frame.
   * @param nowMs Current time in milliseconds (for example the requestAnimationFrame time).
   * @throws RaaError `core.surface`, `core.gpu`, `api.disposed`.
   */
  render(nowMs: number): void;

  /**
   * Frame statistics, the adapter and the hardware flag.
   * @throws RaaError `api.disposed`.
   */
  stats(): FrameStats;

  /**
   * Renders one frame at `nowMs` (milliseconds) and resolves to true when any pixel
   * differs from the clear colour.
   * @throws RaaError `core.gpu`, `api.disposed` (as a rejection).
   */
  probeNonEmpty(nowMs: number): Promise<boolean>;

  /** Releases the GPU objects and the scene; every later call throws `api.disposed`. */
  dispose(): void;
}

/** What {@link loadRaa} resolves to. Each call of loadRaa returns an independent object. */
export interface RaaModule {
  /** The package version, '0.1.0'. */
  readonly version: '0.1.0';
  /**
   * Creates an engine on `canvas` at its current width and height (pixels).
   * @param options Engine options; omitted fields take their defaults.
   * @throws RaaError (as a rejection) `api.invalid_argument`, `core.no_adapter`,
   *   `core.request_device`, `core.surface`.
   */
  readonly createEngine: (canvas: HTMLCanvasElement, options?: EngineOptions) => Promise<Engine>;
}

/**
 * Initialises the wasm module (idempotent) and returns a fresh, frozen {@link RaaModule}.
 * @param options `wasmUrl`: where to fetch the .wasm; default `pkg/raa_js_bg.wasm` next
 *   to index.js.
 */
export function loadRaa(options?: { wasmUrl?: string | URL }): Promise<RaaModule>;
