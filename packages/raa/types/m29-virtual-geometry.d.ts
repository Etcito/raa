/**
 * The virtualized-geometry debug views (S31). `none` draws the frame as shaded; the others
 * replace the shading with one diagnostic: cluster colours, level of detail, hardware or
 * software raster, or occlusion.
 */
export type VirtualGeometryDebugView = 'none' | 'clusters' | 'lod' | 'raster' | 'occlusion';

/**
 * A patch of the engine-wide virtualized-geometry settings, accepted by
 * {@link Engine.setVirtualGeometry}. Every key is optional; an omitted key (`undefined`) keeps
 * its value. `null`, a wrong type, an unknown key or an out-of-domain value throws `vg.invalid`
 * with the message `<key>: <reason>`, and nothing changes.
 */
export interface VirtualGeometrySettings {
  /**
   * Turns the five virtualized-geometry passes on or off through the pipeline. Default false.
   * `true` needs an Engine created with `msaa: 1`, otherwise `vg.invalid` with the message
   * `msaa: virtualized geometry needs msaa 1`.
   */
  enabled?: boolean;
  /** Pixels of screen-space error: the level-of-detail threshold. Finite, in [0.25, 64]. Default 1. */
  errorPixels?: number;
  /** Small clusters are also rasterised in software (compute). Default true. */
  softwareRaster?: boolean;
  /** Occlusion culling against the previous frame's depth pyramid. Default true. */
  occlusion?: boolean;
  /** Virtualized instances cast into the four cascade shadow layers. Default true. */
  shadows?: boolean;
  /** A requested instance whose mesh has no cluster data starts a runtime bake. Default true. */
  runtimeBake?: boolean;
  /** The request of an instance without an explicit value. Default true. */
  instanceDefault?: boolean;
  /**
   * Capacity of the cull queue and of the visible list of each view, in clusters. An integer in
   * [1024, 4194304]. Default 262144. Clusters beyond it are dropped and counted in
   * `droppedClusters`.
   */
  maxVisibleClusters?: number;
  /** The debug view of the frame. Default 'none'. */
  debugView?: VirtualGeometryDebugView;
}

/**
 * The effective virtualized-geometry settings, returned by {@link Engine.virtualGeometry}. Every
 * key is present; the record is a fresh object.
 */
export interface ResolvedVirtualGeometry {
  enabled: boolean;
  /** Pixels of screen-space error. */
  errorPixels: number;
  softwareRaster: boolean;
  occlusion: boolean;
  shadows: boolean;
  runtimeBake: boolean;
  instanceDefault: boolean;
  /** Clusters: capacity of the cull queue and visible list of each view. */
  maxVisibleClusters: number;
  debugView: VirtualGeometryDebugView;
}

/**
 * The virtualized-geometry state of one instance group, returned by
 * {@link Engine.instanceVirtualGeometry}.
 */
export interface InstanceVirtualGeometry {
  /** The explicit request, or `null` when the instance follows `instanceDefault`. */
  requested: boolean | null;
  /** The path of the last submitted frame. */
  path: 'virtual' | 'forward';
  /**
   * True when the instance is requested and eligible but a mesh has no ready cluster data yet
   * (state `none` with a runtime bake started when `runtimeBake` is true, or `baking`). It draws
   * on forward meanwhile and switches to `virtual` once every mesh is ready. False on `virtual`,
   * for a non-requested instance and for a refused one.
   */
  pending: boolean;
  /**
   * An instance code (`vg.skinned_unsupported`, `vg.deformed_unsupported`,
   * `vg.transparent_material`) or a mesh code (`vg.mesh_vertex_colors`, `vg.mesh_empty`,
   * `vg.mesh_too_large`, `vg.invalid`). `null` when pending, virtual or not requested.
   */
  code: string | null;
  /** The refusal message, or `null`. */
  message: string | null;
}

/**
 * Counters of virtualized geometry, returned by {@link Engine.virtualGeometryStats}. The record
 * is fresh. The counters come from the last published readback, never from a GPU fence.
 */
export interface VirtualGeometryStats {
  active: boolean;
  /**
   * The first matching reason: `disabled` (settings), `passDisabled` (one of the five vg passes),
   * `multiView` (submitted views != 1), `noInstance` (no qualifying instance). `null` exactly
   * when `active`.
   */
  inactiveReason: 'disabled' | 'passDisabled' | 'multiView' | 'noInstance' | null;
  /** Instance groups on the virtual path. */
  virtualInstances: number;
  /** Instance groups on the forward path. */
  forwardInstances: number;
  /**
   * Requested instance groups refused with each instance code. A code without a refusal is
   * absent. Mesh codes are not listed here.
   */
  refused: { [code: string]: number };
  /** Distinct meshes of the listed instances in state `baking`. */
  bakingMeshes: number;
  /** Distinct meshes of the listed instances in state `ready`. */
  readyMeshes: number;
  /** Clusters drawn in the main view, both occlusion phases; `null` before a readback. */
  drawnClusters: number | null;
  /** Triangles drawn in the main view, both occlusion phases; `null` before a readback. */
  drawnTriangles: number | null;
  /** The part of `drawnClusters` rasterised in hardware. */
  hardwareClusters: number | null;
  /** The part of `drawnClusters` rasterised in software. */
  softwareClusters: number | null;
  /** Clusters drawn into the four cascades, summed; 0 when no cascade is drawn. */
  shadowClusters: number | null;
  /** Clusters dropped at `maxVisibleClusters`, main view and cascades. */
  droppedClusters: number | null;
  /** The Engine render invocation the counters come from. */
  statsFrameIndex: number | null;
  /** True while a readback is outstanding. */
  pending: boolean;
}

/**
 * The virtualized-geometry part of the Engine (S31). Every method throws `api.disposed` on a
 * disposed Engine. The instance methods throw `api.invalid_argument` or `api.unknown_handle`
 * like the other instance methods.
 */
export interface Engine {
  /** Merges a settings patch atomically; see {@link VirtualGeometrySettings}. */
  setVirtualGeometry(settings: VirtualGeometrySettings): void;
  /** The effective settings, a fresh {@link ResolvedVirtualGeometry}. */
  virtualGeometry(): ResolvedVirtualGeometry;
  /**
   * Sets the request of one instance group: `true`, `false`, or `null` for the default request.
   * An explicit `true` on an ineligible instance throws its instance code.
   */
  setInstanceVirtualGeometry(instance: number, enabled: boolean | null): void;
  /** The state of one instance group, a fresh {@link InstanceVirtualGeometry}. */
  instanceVirtualGeometry(instance: number): InstanceVirtualGeometry;
  /** The latest published counters, a fresh {@link VirtualGeometryStats}. */
  virtualGeometryStats(): VirtualGeometryStats;
}
