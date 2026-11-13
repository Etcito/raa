/**
 * The saved-mesh cluster section (S30): the baked cluster DAG of one single-material mesh.
 * Typed arrays are copied in both directions; the tuples are plain arrays of numbers.
 */
export interface ClusterSection {
  /** Section version, 1. */
  version: 1;
  /** Builder algorithm, 1. */
  algorithm: 1;
  /** Eight u16 per vertex (16 B packed vertex): grid position, octahedral normal, uv, zero. */
  vertices: Uint16Array;
  /** One packed triangle per u32: three cluster-local vertex indices, 8 bits each. */
  triangles: Uint32Array;
  /** 48 B per cluster record, little-endian. */
  clusters: Uint8Array;
  /** 48 B per node record, little-endian; `rootNode` indexes these. */
  nodes: Uint8Array;
  /** Index of the root node, below `nodes.length / 48`. */
  rootNode: number;
  /** Hierarchy depth, an integer in 1..64. */
  hierarchyDepth: number;
  /** DAG levels, an integer in 1..256. */
  levels: number;
  /** Quantisation origin in metres. */
  quantOrigin: [number, number, number];
  /** Quantisation step in metres, finite and above 0. */
  quantStep: number;
  /** UV origin. */
  uvOrigin: [number, number];
  /** UV step per component, finite and not negative. */
  uvStep: [number, number];
  /** Bounding sphere centre in metres. */
  boundsCenter: [number, number, number];
  /** Bounding sphere radius in metres, finite and not negative. */
  boundsRadius: number;
  /** Triangles of the source mesh, a safe non-negative integer. */
  sourceTriangles: number;
}

/**
 * Options of a bake; an omitted field takes its default. A bad value is `vg.invalid` naming
 * `options.<field>`.
 */
export interface ClusterBakeOptions {
  /** Integer in 8..128, the triangles of one cluster. Default 128. */
  maxClusterTriangles?: number;
  /** Integer in 16..255, the vertices of one cluster. Default 255. */
  maxClusterVertices?: number;
  /** In (0, 1): a group that cannot shrink below this ratio is retried. Default 0.85. */
  minReduction?: number;
  /** In (0, 1): the triangle ratio each simplified group aims at. Default 0.5. */
  targetRatio?: number;
}

/**
 * One single-material mesh for a bake, in metres. `positions` and `normals` hold 3 values per
 * vertex, `uvs` 2 per vertex (absent means zero uvs), `indices` 3 per triangle.
 * Refusals: `vg.invalid` naming the array, `vg.mesh_empty` with no triangle,
 * `vg.mesh_too_large` beyond the S29 limits.
 */
export interface ClusterBakeInput {
  positions: Float32Array;
  normals: Float32Array;
  uvs?: Float32Array;
  indices: Uint32Array;
  options?: ClusterBakeOptions;
}

/**
 * Progress of a sliced bake; `fraction` never decreases, lies in [0, 1] and is 1 only when done.
 */
export interface ClusterBakeProgress {
  stage: 'weld' | 'partition' | 'level' | 'hierarchy' | 'pack' | 'done';
  /** The level being built, counted from 1; 0 before the first level. */
  level: number;
  /** Fraction of the whole bake, in [0, 1]. */
  fraction: number;
  /** True once the bake is finished. */
  done: boolean;
}

/**
 * A sliced bake: `step` runs bounded units on the CPU, so a large mesh builds over several calls.
 * A bake error is thrown by `step` and again by `result`.
 */
export interface ClusterBake {
  /**
   * Runs units until `budgetMs` milliseconds of `performance.now()` have passed or the bake is
   * done; at least one unit runs. Default 4 ms; finite and above 0, else `vg.invalid` naming
   * `budgetMs`.
   */
  step(budgetMs?: number): ClusterBakeProgress;
  /** Stage, level and fraction after the last step. */
  progress(): ClusterBakeProgress;
  /** The finished section. `vg.invalid` naming `result` ('the bake is not done') before done. */
  result(): ClusterSection;
  /** Releases the bake; a later call on this object is refused. */
  free(): void;
}

/** The uniform cut of a section as plain typed arrays, positions and normals in metres. */
export interface ClusterCutMesh {
  /** 3 values per vertex, metres. */
  positions: Float32Array;
  /** 3 values per vertex, unit normals. */
  normals: Float32Array;
  /** 2 values per vertex. */
  uvs: Float32Array;
  /** 3 indices per triangle. */
  indices: Uint32Array;
  /** The section cluster index of each triangle. */
  clusterOfTriangle: Uint32Array;
  /** Clusters in the cut. */
  clusters: number;
  /** Triangles in the cut. */
  triangles: number;
}

/**
 * The cluster data of one mesh (S31), returned by {@link Engine.meshClusters}. `progress` is a
 * fraction in [0, 1] (1 only when ready). The section, cluster, source-triangle and level
 * counts are zero unless the state is `ready`.
 */
export interface MeshClusterState {
  /** `none`, `baking` (a runtime bake is sliced over frames), `ready` or `refused`. */
  state: 'none' | 'baking' | 'ready' | 'refused';
  /** Fraction of the runtime bake, in [0, 1); 1 when ready, 0 otherwise. */
  progress: number;
  /** Where a ready set came from: `attached` or `runtime`; null when not ready. */
  source: 'attached' | 'runtime' | null;
  /** A refusal code (`vg.invalid`, `vg.mesh_empty`, `vg.mesh_too_large`, ...), else null. */
  code: string | null;
  /** The refusal message, else null. */
  message: string | null;
  /** Sections of a ready mesh (one per sub-mesh), else 0. */
  sections: number;
  /** Clusters of a ready mesh, summed over its sections, else 0. */
  clusters: number;
  /** Source triangles of a ready mesh, summed over its sections, else 0. */
  sourceTriangles: number;
  /** Largest DAG level of a ready mesh, else 0. */
  levels: number;
}

/**
 * The CPU-side cluster API of the Engine (S31). Every method throws `api.disposed` on a
 * disposed Engine and `api.unknown_handle` for an unknown mesh handle; a refusal leaves the
 * mesh's cluster state unchanged. `updateMesh` drops the clusters of the updated mesh and
 * cancels its bake, and `removeMesh` and `dispose` free them.
 */
export interface Engine {
  /**
   * Attaches one section per sub-mesh of `mesh`, in sub-mesh order: all are validated
   * completely before the mesh's clusters are replaced atomically (state `ready`, source
   * `attached`). A mesh with a colour stream throws `vg.mesh_vertex_colors`; a wrong number of
   * sections or an invalid section throws `vg.invalid`.
   */
  attachMeshClusters(mesh: number, sections: ClusterSection[]): void;
  /**
   * Starts a runtime bake of every sub-mesh of `mesh`, sliced over frames: each Engine render
   * invocation spends at most `clusterBakeBudget()` milliseconds. The state is `baking` with
   * `progress` in [0, 1), then `ready` with source `runtime`. An omitted `options` uses the S31
   * defaults; a mesh without retained geometry, with a colour stream or an empty mesh throws
   * (`vg.invalid`, `vg.mesh_vertex_colors`, `vg.mesh_empty` or `vg.mesh_too_large`).
   */
  bakeMeshClusters(mesh: number, options?: ClusterBakeOptions): void;
  /** Cancels a bake and drops the clusters of `mesh`: state `none`. */
  clearMeshClusters(mesh: number): void;
  /** Returns a fresh {@link MeshClusterState} of `mesh`. */
  meshClusters(mesh: number): MeshClusterState;
  /**
   * Sets the CPU budget of the runtime bake per Engine render invocation, in milliseconds:
   * finite and inside [0.5, 100]; anything else throws `vg.invalid` naming `ms`.
   */
  setClusterBakeBudget(ms: number): void;
  /** The current runtime bake budget in milliseconds per render invocation (default 4). */
  clusterBakeBudget(): number;
}

/** The cluster-bake functions of the module (S31); they need no engine and run in Node. */
export interface RaaModule {
  /**
   * Bakes one mesh to the end and returns its section (S29 limits, S31 options).
   * @throws RaaError `vg.invalid`, `vg.mesh_empty` or `vg.mesh_too_large`; nothing is changed.
   */
  bakeClusters(input: ClusterBakeInput): ClusterSection;
  /**
   * Creates a sliced bake; see {@link ClusterBake}.
   * @throws RaaError as {@link RaaModule.bakeClusters}.
   */
  createClusterBake(input: ClusterBakeInput): ClusterBake;
  /**
   * Validates a section completely: the S30 wire rules, every index range and the reachability
   * and error order of the DAG.
   * @throws RaaError `vg.invalid` naming the failing field.
   */
  validateClusters(section: ClusterSection): void;
  /**
   * The uniform cut at `errorMetres`, a finite number >= 0 in metres; 0 gives the finest cut.
   * @throws RaaError `vg.invalid` naming `errorMetres` or the section field.
   */
  clusterCutMesh(section: ClusterSection, errorMetres: number): ClusterCutMesh;
}
