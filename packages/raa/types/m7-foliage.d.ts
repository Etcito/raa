// M7 tree impostors and relightable foliage: the asynchronous preparation, removal and
// whole-tree LOD group methods of the Engine. scripts/build-package.sh appends this file to
// index.d.ts, and scripts/typecheck-types.mjs stages it the same way; the `Engine` interface
// below merges with the v0 declaration of the same name. Declarations only: the native
// methods are bound by the foliage API card, and this file adds no runtime code.

/** The presentation modes of a foliage asset, from source-faithful to differentiable. */
export type FoliageMode = 'spom' | 'pom' | 'offset' | 'shell' | 'stochastic' | 'differentiable';

/**
 * Input of {@link Engine.prepareFoliage}. Every field is required. `instances` is the nonempty
 * unique set of live whole-tree instance handles; `foliageInstances` is the subset identifying
 * the source foliage. `material` is a caller-owned, generation-exclusive compiled LOD1 material
 * lease that the asset borrows; the caller keeps it until preparation settles and removal
 * completes.
 */
export interface FoliageDesc {
  /** Nonempty unique live whole-tree instance handles. */
  instances: number[];
  /** Subset of `instances` identifying the source foliage. */
  foliageInstances: number[];
  /** Live source-root node handle. */
  root: number;
  /** Caller generation number (u32). */
  generation: number;
  /** The distillation or presentation mode. */
  mode: FoliageMode;
  /** Voxel resolution per axis. */
  resolution: 16 | 32;
  /** Requested triangle count, a positive integer. */
  triangles: number;
  /** Bake seed (u32). */
  seed: number;
  /** Number of views, 64..128. */
  views: number;
  /** Optimisation iterations, a positive integer. */
  iterations: number;
  /** Finite nonnegative weight; not all three weights may be zero. */
  silhouetteWeight: number;
  /** Finite nonnegative weight; not all three weights may be zero. */
  normalWeight: number;
  /** Finite nonnegative weight; not all three weights may be zero. */
  edgeWeight: number;
  /** Translucency, 0..1. */
  translucency: number;
  /** Alpha cutoff, 0..1. */
  cutoff: number;
  /** Compiled LOD1 surface-material handle (u32). */
  material: number;
}

/**
 * A completed foliage preparation. `triangles` is the actual output count; `material` echoes
 * the {@link FoliageDesc.material} handle the asset borrows.
 */
export interface FoliageAssetInfo {
  /** Native foliage asset id. */
  id: number;
  /** The prepared native generation. */
  generation: number;
  /** The prepared mode. */
  mode: FoliageMode;
  /** Actual output triangle count. */
  triangles: number;
  /** Voxel resolution per axis actually used. */
  volumeResolution: 16 | 32;
  /** The owned proxy mesh handle. */
  mesh: number;
  /** The borrowed material handle, echoed unchanged. */
  material: number;
}

/**
 * Input of {@link Engine.setFoliageGroup}: a whole-tree LOD group. `full` is the nonempty
 * list of LOD0 instances; `switchPixels` is the positive screen-size switch threshold.
 */
export interface FoliageGroupDesc {
  /** A prepared foliage asset id. */
  asset: number;
  /** The complete nonempty set of LOD0 instance handles. */
  full: number[];
  /** The scene-node handle shared by `full`. */
  root: number;
  /** Positive switch threshold in pixels. */
  switchPixels: number;
  /** `'auto'` selects the level per view; otherwise the level is fixed. */
  level: 'auto' | 0 | 1;
}

/** The last prepared status of a foliage group. */
export interface FoliageState {
  /** Preparation phase. */
  phase: 'pending' | 'ready' | 'failed';
  /** The prepared native generation. */
  generation: number;
  /** The associated asset id, or null when none. */
  asset: number | null;
  /** The prepared mode. */
  mode: FoliageMode;
  /** The selected level: 0 for full geometry, 1 for the foliage asset. */
  level: 0 | 1;
  /** The failure message, or null when the group is not failed. */
  error: string | null;
}

export interface Engine {
  /**
   * Handoff: 9140 'Coordinate foliage presentations in the game engine' takes from packages/raa/types/m7-foliage.d.ts: prepareFoliage, removeFoliage, setFoliageGroup, foliageGroupState, FoliageMode, FoliageDesc, FoliageAssetInfo, FoliageGroupDesc, FoliageState
   * Prepares a foliage asset from `desc` asynchronously. Invalid descriptors refuse with
   * `foliage.invalid` before any mutation; execution failures reject with
   * `foliage.bake_failed`. The material lease is borrowed, not released.
   * @throws RaaError `foliage.invalid`, `foliage.bake_failed`, `api.disposed`.
   */
  prepareFoliage(desc: FoliageDesc): Promise<FoliageAssetInfo>;

  /**
   * Removes foliage asset `id`: detaches its dependent groups and the material association,
   * then releases its owned mesh and volume. The borrowed material and data are never
   * deleted. Unknown ids refuse before any mutation.
   * @throws RaaError `foliage.invalid`, `api.disposed`.
   */
  removeFoliage(id: number): void;

  /**
   * Registers or updates the foliage group `id` with `desc`, or removes it with `null`.
   * Removal is idempotent. Invalid or unknown-asset descriptors refuse before any change.
   * @throws RaaError `foliage.invalid`, `api.disposed`.
   */
  setFoliageGroup(id: string, desc: FoliageGroupDesc | null): void;

  /**
   * The last prepared status of group `id`, or null when no such group is registered.
   * @throws RaaError `api.disposed`.
   */
  foliageGroupState(id: string): FoliageState | null;
}
