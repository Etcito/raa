// Hand-written types of the chain milestone 1 additions (meshes from typed arrays, the
// configurable pass pipeline, HDR tone mapping, MSAA). Documented in docs/api/m07-meshes-hdr.md.
//
// This file is appended to packages/raa/index.d.ts when the package is staged
// (scripts/build-package.sh): `Vec3` comes from there, and the `EngineOptions` and `Engine`
// interfaces below merge with the v0 declarations of the same name.

/**
 * One contiguous index range of a mesh drawn with one material (S13).
 * Sub-meshes of one mesh must not overlap and must lie inside the index array.
 */
export interface SubMeshDesc {
  /** Offset into `indices`, a multiple of 3, >= 0. */
  first: number;
  /** Number of indices, a multiple of 3, > 0. */
  count: number;
  /**
   * Material handle drawn by {@link Engine.addMeshInstance}; absent = the default material.
   * An unknown handle throws `api.unknown_handle`.
   */
  material?: number;
}

/**
 * An attributed triangle mesh for {@link Engine.addMesh} and {@link Engine.updateMesh} (S13).
 * `n` is the vertex count, `m` the triangle count. Positions in metres in the mesh's own
 * space; front faces CCW.
 */
export interface MeshDesc {
  /** 3n floats: x, y, z per vertex (metres). Required. */
  positions: Float32Array;
  /** 3n floats: a non-zero normal per vertex (normalised by the engine). Required. */
  normals: Float32Array;
  /** 2n floats: u, v per vertex. Default (0, 0). */
  uvs?: Float32Array;
  /**
   * 4n floats: xyz non-zero (normalised by the engine), w = +1 or -1 (handedness).
   * Absent with `uvs` present: generated from the uvs; absent without `uvs`: an
   * orthonormal default derived from the normal.
   */
  tangents?: Float32Array;
  /**
   * 4n vertex colours, multiplying the base colour: a Float32Array is linear rgba, a
   * Uint8Array is sRGB rgba8 (rgb converted to linear, alpha / 255). Default (1, 1, 1, 1).
   */
  colors?: Float32Array | Uint8Array;
  /** 3m vertex indices, triangle list. Required. */
  indices: Uint32Array;
  /** Sub-meshes; absent or null = one sub-mesh over all indices with no material. */
  subMeshes?: SubMeshDesc[] | null;
}

/** The result of {@link Engine.meshInfo}. */
export interface MeshInfo {
  /** Vertex count. */
  vertices: number;
  /** Triangle count (indices / 3). */
  triangles: number;
  /** The sub-meshes; `material` is a material handle, or null for the default material. */
  subMeshes: { first: number; count: number; material: number | null }[];
  /** Axis-aligned bounds of the positions in mesh space (metres). */
  bounds: { min: Vec3; max: Vec3 };
}

/** One entry of {@link Engine.passes}: a pipeline pass or a contributor of `forward`. */
export interface PassInfo {
  /** Pass name, for example 'prepass', 'forward' or 'tonemap'. */
  name: string;
  /** 'pass' = a pipeline pass; 'contributor' = drawn inside the forward pass. */
  role: 'pass' | 'contributor';
  /** Whether the pass runs. */
  enabled: boolean;
  /** Sort key (passes) or position inside forward (contributors). */
  order: number;
  /** True for prepass, forward and tonemap: they cannot be disabled. */
  required: boolean;
}

/**
 * Tone mapping settings for {@link Engine.setToneMapping}. Every field is optional and
 * merged over the current settings (default: agx, exposureEv 0, look none, dither on).
 */
export interface ToneMappingDesc {
  /** Tone mapping curve. Another value throws `api.invalid_argument`. */
  mapper?: 'agx' | 'aces' | 'linear';
  /** Exposure in EV (the HDR colour is multiplied by 2^exposureEv); finite. */
  exposureEv?: number;
  /** AgX look. Another value throws `api.invalid_argument`. */
  look?: 'none' | 'punchy' | 'golden';
  /** Dither the display output. */
  dither?: boolean;
}

/** The result of {@link Engine.capabilities}. */
export interface Capabilities {
  /** Always true: the engine runs on a core WebGPU adapter. */
  coreWebGpu: true;
  /** Enabled device features as WebGPU names, for example 'float32-filterable'. */
  features: string[];
  /** MSAA sample count in use (1 or 4). */
  msaa: number;
  /** Format of the HDR targets. */
  hdrFormat: 'rgba16float';
  /** The device's maxStorageBufferBindingSize limit in bytes. */
  maxStorageBufferBindingSize: number;
}

/** The lit pipeline's engine options; merged into {@link EngineOptions}. */
export interface LitEngineOptions {
  /** MSAA sample count of the forward pass. Default 4. Another value throws `api.invalid_argument`. */
  msaa?: 1 | 4;
  /** Initial tone mapping curve. Default 'agx'. Another value throws `api.invalid_argument`. */
  toneMapper?: 'agx' | 'aces' | 'linear';
  /** Initial exposure in EV, finite. Default 0. A non-finite value throws `api.invalid_argument`. */
  exposureEv?: number;
}

/** {@link LitEngineOptions} merged into the v0 options of {@link RaaModule.createEngine}. */
export interface EngineOptions extends LitEngineOptions {}

/** The chain milestone 1 methods, merged into the v0 Engine. */
export interface Engine {
  /**
   * Validates and uploads an attributed mesh and returns its mesh handle (usable by
   * addInstance, addMeshInstance, updateMesh, meshInfo and removeMesh).
   * @throws RaaError `mesh.empty`, `mesh.length_mismatch`, `mesh.not_finite`,
   *   `mesh.index_range`, `mesh.invalid_normal`, `mesh.invalid_tangent`,
   *   `mesh.submesh_range`, `mesh.submesh_overlap`, `api.unknown_handle` (sub-mesh
   *   material), `api.invalid_argument` (missing array or wrong array type).
   */
  addMesh(desc: MeshDesc): number;

  /**
   * Replaces the geometry and sub-meshes of mesh `id` (any mesh handle). Its instances are
   * rebuilt at the next frame, keeping node, visibility and handle. A refused desc changes
   * nothing.
   * @throws RaaError `api.unknown_handle` and the codes of {@link Engine.addMesh}.
   */
  updateMesh(id: number, desc: MeshDesc): void;

  /**
   * Vertex and triangle counts, sub-meshes and bounds of mesh `id`.
   * @throws RaaError `api.unknown_handle`.
   */
  meshInfo(id: number): MeshInfo;

  /**
   * Draws `mesh` at `node`, each sub-mesh with its own material (the default material when
   * it has none), and returns the instance handle.
   * @throws RaaError `api.unknown_handle`, `scene.stale_node`.
   */
  addMeshInstance(node: number, mesh: number): number;

  /** Every pass and forward contributor, in execution order. */
  passes(): PassInfo[];

  /**
   * Enables or disables a pass or contributor.
   * @throws RaaError `pipeline.unknown_pass`, `pipeline.required_pass`.
   */
  setPassEnabled(name: string, enabled: boolean): void;

  /**
   * Sets the execution order: every pass-role name exactly once, prepass before forward
   * before tonemap; display overlays follow tonemap. A refused order changes nothing.
   * @throws RaaError `pipeline.unknown_pass`, `pipeline.duplicate_pass`,
   *   `pipeline.order_incomplete`, `pipeline.required_order`.
   */
  setPassOrder(names: string[]): void;

  /**
   * Merges `settings` over the current tone mapping. A refusal changes nothing.
   * @throws RaaError `api.invalid_argument` (unknown mapper or look, non-finite
   *   exposureEv, wrong field type), `post.invalid`.
   */
  setToneMapping(settings: ToneMappingDesc): void;

  /**
   * Sets the background colour: display-referred linear RGB, each finite and >= 0, shown
   * unchanged where nothing is drawn. Default [0.05, 0.06, 0.08].
   * @throws RaaError `api.invalid_argument`.
   */
  setBackground(color: Vec3 | Float32Array): void;

  /** The device capabilities the engine runs with. */
  capabilities(): Capabilities;
}
