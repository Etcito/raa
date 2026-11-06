// M26 instance detail chains, ray queries and frame statistics.
// build-package.sh appends this fragment to index.d.ts, merging Engine.

/** Geometry fields read by {@link Engine.meshLodError}. */
export interface MeshLodGeometry {
  /** 3n finite vertex coordinates as xyz triples. */
  positions: Float32Array;
  /** Triangle-list indices. */
  indices: Uint32Array;
}

/** One coarser mesh and its geometric error in object units. */
export interface MeshLodLevel {
  /** Handle returned by {@link Engine.addMesh}. */
  mesh: number;
  /** Positive, finite two-sided error, increasing from finest to coarsest. */
  error: number;
}

/** Partial camera and shadow LOD settings; omitted fields keep their current values. */
export interface LodOptions {
  /** Camera error limit in pixels; positive and finite. */
  thresholdPx?: number;
  /** Selection hysteresis from 0 to 0.9. */
  hysteresis?: number;
  /** Finite camera bias; positive values select coarser levels sooner. */
  bias?: number;
  /** Finite shadow bias; positive values select coarser levels sooner. */
  shadowBias?: number;
}

/** An instance intersected by {@link Engine.queryInstancesRay}. */
export interface InstanceRayHit {
  /** Instance handle. */
  instance: number;
  /** Ray parameter in `origin + t * dir` (the direction is not normalized). */
  t: number;
}

/** Statistics from the most recently prepared frame. */
export interface InstanceStats {
  /** Visible scene instances represented by persistent records. */
  instances: number;
  /** Instances selected for the camera. */
  camera: number;
  /** Caster counts for the four shadow cascades. */
  cascades: number[];
  /** Camera instance counts at levels 0 through 7; posed instances count at level 0. */
  lods: number[];
  /** Bytes uploaded for changed persistent instance pages in the frame. */
  uploadedBytes: number;
}

export interface Engine {
  /**
   * Measures the two-sided vertex-to-triangle distance between meshes, in object units.
   * Only positions and triangle indices are read. Posed instances always draw their
   * full-resolution mesh (level 0) for the camera and every shadow cascade.
   * @throws RaaError `api.invalid_argument` (malformed geometry), `lod.invalid` (a mesh has
   *   no triangles or contains non-finite positions), `api.disposed`.
   */
  meshLodError(full: MeshLodGeometry, level: MeshLodGeometry): number;

  /**
   * Registers one or more coarser levels, finest first. Each level must have the same
   * sub-mesh layout; sub-mesh k pairs with sub-mesh k of `mesh`. A refused registration
   * leaves no partial chain on the mesh.
   * @throws RaaError `lod.unknown_mesh`, `lod.too_many_levels`, `lod.invalid`
   *   (invalid chain or layout), `api.invalid_argument` (malformed levels), `api.disposed`.
   */
  setMeshLods(mesh: number, levels: MeshLodLevel[]): void;

  /**
   * Clears every sub-mesh chain of `mesh`; returns true if any chain existed.
   * @throws RaaError `lod.unknown_mesh`, `api.disposed`.
   */
  clearMeshLods(mesh: number): boolean;

  /**
   * Updates camera and shadow LOD settings. Shadows use a fixed one-texel threshold and
   * `shadowBias` (default 0); omitted fields retain their prior values.
   * @throws RaaError `api.invalid_argument` (non-finite/out-of-range values or malformed
   *   options), `api.disposed`.
   */
  setLodOptions(options: LodOptions): void;

  /**
   * Sets an instance's whole-number LOD bias from -3 to 3; positive values choose coarser
   * levels sooner.
   * @throws RaaError `api.invalid_argument` (bias is not a whole number in -3..3),
   *   `api.unknown_handle` (instance), `api.disposed`.
   */
  setInstanceLodBias(instance: number, bias: number): void;

  /**
   * Returns the nearest hit for each intersected instance, sorted by `t` then handle.
   * `origin` and `dir` are xyz arrays or Float32Arrays; `max` defaults to 64.
   * @throws RaaError `api.invalid_argument` (vectors or max malformed), `api.disposed`.
   */
  queryInstancesRay(
    origin: Vec3 | Float32Array,
    dir: Vec3 | Float32Array,
    max?: number,
  ): InstanceRayHit[];

  /**
   * Returns instance counts and changed-page upload bytes for the last prepared frame.
   * @throws RaaError `api.disposed`.
   */
  instanceStats(): InstanceStats;
}
