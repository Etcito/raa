// Hand-written types of @etcito/raa-sdfm 0.1.0 (ARCHITECTURE S15). Plain ES modules, no build step.
// The adapter turns `sdfm.game-asset` v1 files into RAA meshes and surface materials, meshed and
// compiled by MAMA. Engine refusals (RaaError) propagate unchanged; the adapter's own refusals are
// SdfmError with one of the S15 codes (plus S18 `sdfm.rig`). MAMA is reached only through
// src/mama-bridge.js.

import type { Engine, MeshDesc, SurfaceBlend } from '@etcito/raa';

/**
 * Re-exported from `@etcito/raa`: the engine the loader uploads meshes and materials to, and the
 * surface blend mode of a mapped material (`'opaque'` in this package).
 * Refusals: none (types only); the engine's own refusals are RaaError.
 */
export type { Engine, SurfaceBlend };

/**
 * A 3-component vector `[x, y, z]` (metres for positions, unitless for directions).
 * Refusals: none (type only).
 */
export type Vec3 = [number, number, number];

/**
 * A rotation quaternion `[x, y, z, w]`; the identity is `[0, 0, 0, 1]`.
 * Refusals: none (type only).
 */
export type Quat = [number, number, number, number];

/**
 * The S15 refusal codes plus the S18 `sdfm.rig` and the two procedural-scene codes (S66):
 * `sdfm.format`, `sdfm.version`, `sdfm.schema`, `sdfm.not_finite`, `sdfm.id`,
 * `sdfm.cap_document`, `sdfm.hash_mismatch`, `sdfm.mesher_refused`,
 * `sdfm.material_unmappable`, `sdfm.worker`, `sdfm.disposed`, `sdfm.rig`,
 * `sdfm.scene_format`, `sdfm.scene_resolve`.
 * Refusals: every code listed here is a possible `SdfmError.code`; the README table `S15 codes`
 * says when each one is raised.
 */
export type SdfmErrorCode =
  | 'sdfm.format'
  | 'sdfm.version'
  | 'sdfm.schema'
  | 'sdfm.not_finite'
  | 'sdfm.id'
  | 'sdfm.cap_document'
  | 'sdfm.hash_mismatch'
  | 'sdfm.mesher_refused'
  | 'sdfm.material_unmappable'
  | 'sdfm.worker'
  | 'sdfm.disposed'
  | 'sdfm.rig'
  | 'sdfm.scene_format'
  | 'sdfm.scene_resolve';

/**
 * A refusal of the sdfm adapter. The message is prefixed with `<code>: `, like RaaError, and
 * every failure of this package is one of these codes (engine refusals stay RaaError).
 * Refusals: any {@link SdfmErrorCode}; `path` is a JSON pointer into the asset where one exists.
 */
export class SdfmError extends Error {
  /** One of the S15 codes, or `sdfm.rig` for an invalid S16 rig. */
  readonly code: SdfmErrorCode;
  /** JSON pointer of the offending asset value, when the refusal has one (e.g. `/nodes/2/radius`). */
  readonly path?: string;
  /**
   * @param code one of the S15 codes
   * @param message human message (stored as `<code>: <message>`)
   * @param path JSON pointer into the asset, when there is one
   */
  constructor(code: SdfmErrorCode, message: string, path?: string);
}

/**
 * One entry of {@link SdfmLoader.validate}: an S15 code, a JSON pointer (`''` for the root) and a
 * human message. Validation reports every problem, it never throws.
 * Refusals: the codes are {@link SdfmErrorCode} values.
 */
export interface SdfmValidationError {
  code: SdfmErrorCode;
  path: string;
  message: string;
}

/**
 * Result of {@link SdfmLoader.validate}: `ok` is `errors.length === 0`. Validation is synchronous
 * and needs no MAMA.
 * Refusals: none (the errors array carries the codes).
 */
export interface SdfmValidationResult {
  ok: boolean;
  errors: SdfmValidationError[];
}

/**
 * One progress report of {@link SdfmLoader.load}, passed to `opts.onProgress`.
 * `stage` runs validate, then mesh, then upload, then lod when supported; `done` is between 0 and `total`.
 * Refusals: none; progress is informational.
 */
export interface SdfmProgress {
  stage: 'validate' | 'mesh' | 'upload' | 'lod';
  done: number;
  total: number;
}

/** One S16 rig bone; `rest` is local to `parent` (null at the root). `bones` order is joint order. */
export interface SdfmRigBone { name: string; parent: string | null; rest: { t: Vec3; r: Quat; s: Vec3 } }

/** One S16 IK chain (`twoBone` has exactly 3 bones; `ccd` and `fabrik` have 2 to 8). */
export interface SdfmRigChain {
  name: string; solver: 'twoBone' | 'ccd' | 'fabrik'; bones: string[];
  pole?: Vec3 | null; iterations?: number;
}

/** One S16 pose-mode limit: XYZ Euler degrees relative to rest. */
export interface SdfmRigLimit { bone: string; min: Vec3; max: Vec3 }

/** One S16 named attachment point on a bone (props and costumes bind to it). */
export interface SdfmRigAttachment { bone: string; t: Vec3; r: Quat }

/** One S18 binding: a shape or a Paramesh group maps to weighted bones. */
export interface SdfmRigBinding { source: { group: string } | { shape: string }; bones: { bone: string; factor: number }[] }

/**
 * The validated S16 rig plus S18 authoring fields MAMA exports for a rigged group. An invalid rig
 * is refused by {@link SdfmLoader.load} as `sdfm.rig`, with the S16 code in the message.
 */
export interface SdfmRig {
  format: 'raa.rig'; version: 1; name: string; bones: SdfmRigBone[]; roles?: Record<string, string>;
  ikChains?: SdfmRigChain[]; limits?: SdfmRigLimit[]; attach?: Record<string, SdfmRigAttachment>;
  bindings?: SdfmRigBinding[]; smooth?: number;
}

/** One S17 clip key; `hl`/`hr` are `[dtTicks, dv...]` and only exist on BEZIER keys. */
export interface SdfmClipKey {
  tick: number; v: number[]; interp: 'CONSTANT' | 'LINEAR' | 'BEZIER'; hl?: number[]; hr?: number[];
}

/** One S17 clip track: one bone and channel, keys in strictly increasing tick order. */
export interface SdfmClipTrack { bone: string; channel: 'translation' | 'rotation' | 'scale'; keys: SdfmClipKey[] }

/** One S17 clip event. */
export interface SdfmClipEvent { tick: number; name: string }

/** One S17 `raa.clip` v2 clip exported with a rigged group, unchanged. */
export interface SdfmClip {
  format: 'raa.clip'; version: 2; name: string; rig: string; lengthTicks: number; loop: boolean;
  rootMotion: 'none' | 'xz' | 'xyz'; tracks: SdfmClipTrack[]; events?: SdfmClipEvent[];
}

/**
 * S18 skin of one part: 4n joint indices into `rig.bones` (authored order) and 4n weights, four
 * per mesh vertex, plus the group rig. The same arrays are uploaded as `MeshDesc.joints`/`weights`.
 */
export interface SdfmPartSkin { joints: Uint16Array; weights: Float32Array; rig: SdfmRig }

/**
 * One uploaded part of an {@link SdfmAsset}: a rigid MAMA part with its mesh and sub-mesh
 * material handles. MAMA has no rigid flag or parent link, so `rigid` is always true and
 * `parent` is always null: every part is flat under the spawn root and carries its own frame.
 * A part of a rigged group also carries its S18 `skin`.
 * Refusals: none of its own; {@link SdfmLoader.load} raises the codes of {@link SdfmErrorCode}
 * before a part exists.
 */
export interface SdfmPart {
  /** Part name; `group/partId` when two parts of the asset share a name. */
  name: string;
  /** MAMA group id the part belongs to. */
  groupId: string;
  /** Always true (every MAMA part is rigid). */
  rigid: boolean;
  /** Always null (the MAMA hierarchy is flat). */
  parent: string | null;
  /** Local translation in metres. */
  t: Vec3;
  /** Local rotation quaternion `[x, y, z, w]`. */
  r: Quat;
  /** Engine mesh handle returned by `Engine.addMesh`. */
  mesh: number;
  /** Attached coarser mesh handles, finest first; empty for skinned parts or unavailable LODs. */
  lods: number[];
  /** Surface material handles of the sub-meshes, in sub-mesh order. */
  materials: number[];
  /**
   * S18 skin of a part of a rigged group; the same joints and weights are uploaded as the mesh's
   * `joints`/`weights` attributes. Absent for an unrigged part.
   */
  skin?: SdfmPartSkin;
}

/**
 * A loaded asset: the `sdfm.game-asset` v1 fields passed through untouched (`id`, `hash`, `name`,
 * `category`, `world`, `anchors`, `bounds`, `collision`) plus the uploaded `parts` and, for a
 * rigged asset, the S18 `rig` and `clips`.
 * Refusals: {@link SdfmLoader.load} raises the validation/hash/mesh/material/rig codes of
 * {@link SdfmErrorCode}; engine refusals are RaaError.
 */
export interface SdfmAsset {
  /** Asset id (`^[a-z][a-z0-9]*(\.[a-z0-9-]+)+$`). */
  id: string;
  /** sha256 of UTF-8 MAMA serializeDocument(document); the load cache key. */
  hash: string;
  name: string;
  category: string;
  world: string;
  parts: SdfmPart[];
  anchors: Record<string, { t: Vec3; r: Quat }>;
  bounds: { min: Vec3; max: Vec3 };
  /** Passed through untouched; never used for rendering. */
  collision: unknown[];
  /**
   * The validated S16 rig of the first group that carries one; every skinned part's `skin.rig` is
   * this object. Absent for an unrigged asset. Joint indices are its `bones` order.
   */
  rig?: SdfmRig;
  /** The S17 clips of the loaded groups, in group result order; absent when no group has clips. */
  clips?: SdfmClip[];
}

/**
 * Nodes and instances created by {@link SdfmLoader.spawn}: the root node plus one node and one
 * mesh instance per part, keyed by part name.
 * Refusals: the engine throws RaaError (`api.unknown_handle`, `scene.stale_node`,
 * `api.invalid_argument`) unchanged; `sdfm.disposed` after {@link SdfmLoader.dispose}.
 */
export interface SdfmSpawned {
  /** Root node handle (parented to `parent ?? null`). */
  root: number;
  parts: Record<string, { node: number; instance: number }>;
}

/**
 * The MAMA exports the adapter uses when `mama` resolves to an object instead of a URL (node,
 * in-process meshing through src/mama-bridge.js). Only these members are read.
 * Refusals: `SdfmLoader.load` raises `sdfm.worker` when a member throws or is missing,
 * `sdfm.mesher_refused` for a group value refusal and `sdfm.material_unmappable` for a material
 * MAMA cannot build.
 */
export interface SdfmMamaModule {
  /** MAMA serializer used verbatim for the asset document hash. */
  serializeDocument(document: unknown): string;
  /** MAMA's `readSdfm`: builds MAMA's document from the asset's `document`. */
  readSdfm(document: unknown): unknown;
  /** MAMA's `meshAssets(doc, { pool, groups })`; each result is `{ group, parts, materials, error, rig?, clips? }`. */
  meshAssets(document: unknown, options: { pool: unknown; groups?: string[] }): Promise<unknown[]>;
  /** MAMA's helper table: helper id -> WGSL source and dependencies. */
  HELPERS: Record<string, { code: string; deps?: string[] }>;
  /** MAMA's always-emitted material WGSL (constants and scalar helpers such as `mat_clamp01`). */
  BASE_WGSL: string;
  /** MAMA's browser meshing pool factory (`mama/browser`); unused in node. */
  setupBrowserHost?(options: { workerUrl: string | URL; manifoldWasmUrl: string; size?: number; workerMode?: 'module' | 'blob' }): unknown;
  /** A MAMA `ComputePool` used for in-process meshing. */
  pool?: unknown;
}

/**
 * Options of {@link createSdfmLoader}.
 * Refusals: a missing `engine` or `mama`, or invalid `lods`, throws `TypeError`; failures during `load` are
 * `SdfmError` (`sdfm.worker`, `sdfm.disposed`) or RaaError from the engine.
 */
export interface SdfmLoaderOptions {
  /** The RAA engine that receives the meshes, material functions and surface materials. */
  engine: Engine;
  /**
   * Resolves once per loader. A string or URL is the base URL of MAMA's `dist/` directory and
   * meshes in `worker.js` workers; an object is MAMA's exports (see {@link SdfmMamaModule}) and
   * meshes in-process through src/mama-bridge.js.
   */
  mama: () => Promise<string | URL | SdfmMamaModule>;
  /** URL of `worker.js`; default `new URL('../worker.js', import.meta.url)`. */
  workerUrl?: string | URL;
  /** Worker count of the pool; default `min(4, navigator.hardwareConcurrency || 2)`. */
  workers?: number;
  /** Coarser Paramesh levels, an integer 0..3; default LOD_LEVELS_DEFAULT (2). Zero disables LODs. */
  lods?: number;
}

/**
 * Loads `sdfm.game-asset` v1 files as RAA meshes, surface materials and nodes. One loader owns
 * its material-function set, its hash cache and, only in worker mode, its worker pool.
 * Refusals: all S15 codes; engine refusals propagate unchanged as RaaError.
 */
export interface SdfmLoader {
  /**
   * Validates an asset against S15 (synchronous, no MAMA) and returns every error.
   * Refusals: none; the errors carry the codes and JSON pointers.
   */
  validate(asset: unknown): SdfmValidationResult;
  /**
   * Validates, checks the document hash, then meshes (workers or in-process through the bridge),
   * uploads one mesh and its materials per part and caches the result by hash. A second load of
   * the same hash resolves from the cache and calls MAMA no more. The returned asset exposes the
   * meshed group's S16 `rig` and S17 `clips` beside the S15 fields (absent when unrigged).
   * Coarser meshes reuse the full mesh's material handles. LOD failures leave that part's
   * `lods` empty and do not fail the load; unattached level meshes are removed.
   * Refusals: the first validation error as `SdfmError` (with its path), otherwise
   * `sdfm.hash_mismatch`, `sdfm.mesher_refused`, `sdfm.rig` (an invalid S16/S18 rig, with the S16
   * code in the message), `sdfm.material_unmappable`, `sdfm.worker`, `sdfm.disposed`; engine
   * refusals are RaaError.
   */
  load(
    asset: string | object,
    opts?: { onProgress?: (progress: SdfmProgress) => void },
  ): Promise<SdfmAsset>;
  /**
   * Adds the spawn root under `parent` (null = scene root), one node per part under the root with
   * the part's `t`/`r`, and one mesh instance per part.
   * Refusals: `sdfm.disposed` after dispose; engine refusals are RaaError.
   */
  spawn(asset: SdfmAsset, parent?: number | null): SdfmSpawned;
  /**
   * Terminates the worker pool and clears the hash cache and material-function set; later
   * `load`/`spawn` calls raise `sdfm.disposed`.
   * Refusals: none (idempotent).
   */
  dispose(): void;
}

/**
 * A compiled material as src/mama-bridge.js normalises MAMA's `CompiledGraph` (`MamaMaterial`):
 * `key` is the sha256 hex of the canonical `{ body, helpers, constants, fields }`, `body` ends in
 * `return MaterialFull(...)`, `constants` are the binding-7 u32 words and `fields` are MAMA's
 * MaterialFull field names in declaration order.
 * Refusals: {@link mapMaterial} raises `sdfm.material_unmappable` for a malformed record.
 */
export interface SdfmCompiledMaterial {
  /** MAMA material id. */
  name: string;
  /** 64 lowercase hex digits: the material graph hash. */
  key: string;
  /** MAMA's `wgslBody` verbatim. */
  body: string;
  /** Helpers with their WGSL source, dependencies first. */
  helpers: { id: string; wgsl: string }[];
  /** The binding-7 `u32` words, inlined as `const SDFM_K`. */
  constants: number[];
  /** MaterialFull field names in declaration order. */
  fields: string[];
}

/**
 * Result of {@link mapMaterial}: an S11 user material function and its surface description.
 * Refusals: none of its own; {@link mapMaterial} raises `sdfm.material_unmappable`.
 */
export interface SdfmMappedMaterial {
  /** `'sdfm.'` plus the first 16 hex digits of the graph key; register it once per key. */
  functionName: string;
  /** WGSL defining `fn material_main(in: MaterialInput) -> MaterialOutput` (S11). */
  wgsl: string;
  /** Description for `Engine.addSurfaceMaterial`; `blend` is always `'opaque'` (MAMA has none). */
  desc: { function: string; blend: SurfaceBlend };
  /** MaterialFull fields with no MaterialOutput counterpart, in declaration order. */
  dropped: string[];
}

/**
 * Creates a loader for one engine. `mama` is resolved once; the worker pool is created lazily on
 * the first load whose `mama` resolves to a URL, and the hash cache is per loader.
 * Refusals: `TypeError` for a missing engine or `mama`; later calls raise the codes of
 * {@link SdfmLoader}.
 */
export function createSdfmLoader(options: SdfmLoaderOptions): SdfmLoader;

/**
 * The MAMA procedural scene functions `loadProceduralScene` calls, exactly what
 * src/mama-bridge.js `importProcedural()` resolves (bare `mama/procgen/scene-file.js`,
 * `mama/procgen/scene-build.js`, `mama/procgen/trace.js` and `mama`, or the matching files under a `base` URL).
 * The scene is typed `string | object` because this package declares no MAMA types: pass the parsed
 * `sdfm.procedural-scene` v1 file (section 66) in the shape MAMA's `ProceduralScene` defines.
 * Refusals: none of its own; an import failure is `sdfm.worker`.
 */
export interface SdfmProceduralMama {
  /** MAMA `parseProceduralScene(text)`: strict parse of an `sdfm.procedural-scene` v1 file. */
  parseProceduralScene(text: string): object;
  /**
   * MAMA `sceneSourceDocument(scene, sourceIndex, lookupAsset)`: the SDF document of one composite
   * source (csg, realized) or of the asset source a bent instance wraps.
   */
  sceneSourceDocument(scene: string | object, sourceIndex: number,
    lookupAsset: (file: string) => object | null): object;
  /** MAMA `serializeDocument(document)`: canonical document text. */
  serializeDocument(document: object): string;
  /** MAMA `buildSceneTraceInput(scene, lookupAsset)`; trace mode only. */
  buildSceneTraceInput?(scene: string | object, lookupAsset: (file: string) => object | null):
    { sources: object[]; instances: { source: number }[] };
  /** MAMA `packTrace(procgen, input)`: evaluator op 4 packs the TraceProgram in Rust; trace mode only. */
  packTrace?(procgen: SdfmProcgen, input: { sources: object[]; instances: { source: number }[] }): Uint8Array;
}

export interface SdfmProcgen {
  call(op: number, input: Uint8Array): Uint8Array;
}

/**
 * Options of {@link loadProceduralScene}. `resolveAsset` is used by an sdf-refs scene and
 * `resolveMesh` by a mesh-refs scene; the other one is ignored.
 * Refusals: none of its own; the resolve callbacks' failures become `sdfm.scene_resolve`.
 */
export interface ProceduralSceneOptions {
  /** Return the asset text (JSON string) or its parsed game-asset object, once per distinct `file`. */
  resolveAsset?: (file: string) => Promise<string | object>;
  /** Return the saved S24 `.mesh` bytes, once per mesh source. */
  resolveMesh?: (path: string) => Promise<Uint8Array | ArrayBuffer>;
  /** Parent node of the scene root (null: a top-level root). */
  parent?: number | null;
  /** `'mesh'` (default) meshes through the loader; `'trace'` packs an sdf-refs scene for the trace scene. */
  mode?: 'mesh' | 'trace';
  /** The loaded MAMA procedural evaluator; trace mode needs it to pack. */
  procgen?: SdfmProcgen;
  /** Trace mode only: a translation in metres added to every scene instance. */
  placement?: Vec3;
  /** MAMA accessor; the default is mama-bridge `importProcedural()`. */
  mama?: () => Promise<SdfmProceduralMama>;
}

/**
 * The handle of one loaded `sdfm.procedural-scene` (section 66). Every scene instance is one engine
 * node under `root`; its parts are spawned under it, never flattened. In trace mode `setVisible`
 * packs or removes the handle's instances from the shared trace scene.
 * Refusals: `sdfm.disposed` from `setVisible` and `stats` after {@link ProceduralSceneHandle.dispose}.
 */
export interface ProceduralSceneHandle {
  /** Engine node every instance node is parented to; trace mode parents nothing to it. */
  root: number;
  /** Number of scene instances (one engine node each). */
  instanceCount: number;
  /** Number of scene sources (assets, composites, or meshes). */
  sourceCount: number;
  /** Show or hide every spawned part instance; trace mode packs or removes the handle's traced objects. */
  setVisible(visible: boolean): void;
  /** `instances` is the scene instance count, `meshes` the distinct loaded assets' parts, `pending` the uploads left. */
  stats(): { instances: number; meshes: number; pending: number };
  /** Remove the root node and every mesh this handle added; idempotent. */
  dispose(): void;
}

/**
 * Loads one `sdfm.procedural-scene` v1 file (`scene` text or parsed object). An sdf-refs scene
 * resolves its asset sources and loads them, plus an in-memory game asset per composite (csg,
 * realized) and per distinct bend. A mesh-refs scene decodes each S24 `.mesh` (section 24) and
 * uploads it with `Engine.addMesh`, then one instance per scene instance. Instances are never
 * flattened.
 * Trace mode resolves an sdf-refs scene through MAMA's `buildSceneTraceInput` without meshing,
 * merges it with every live handle and installs the op 4 packed program with
 * `Engine.setTraceScene`; `setVisible` and `dispose` repack it, and the last handle calls
 * `Engine.clearTraceScene`.
 * Refusals: `sdfm.scene_format` for a parse failure, an unknown `mode`, a deform on a mesh-refs
 * instance or a mesh-refs scene in trace mode; `sdfm.scene_resolve` when `resolveAsset`,
 * `resolveMesh`, the mesh decode, `sceneSourceDocument`, trace input building or trace packing
 * fails; `sdfm.worker` when the default MAMA import fails; `sdfm.disposed` from the handle after
 * disposal; engine refusals (RaaError) propagate unchanged.
 */
export function loadProceduralScene(engine: Engine, loader: SdfmLoader,
  scene: string | object, options?: ProceduralSceneOptions): Promise<ProceduralSceneHandle>;

/**
 * Decodes an S24 saved mesh file (`assets/<world>/<id>.mesh`): format byte 1, big-endian u32 header
 * length and buffer count, a UTF-8 JSON header with `{$buffer}` / `{$array}` markers, then each
 * buffer's big-endian u32 length and bytes. Typed arrays are views on the decoded buffers.
 * Refusals: `sdfm.scene_resolve` for a bad format byte, a truncated buffer, trailing bytes, an
 * unknown array type or an invalid JSON header.
 */
export function decodeMeshFile(bytes: Uint8Array): object;

/**
 * Validates the `data` payload of a decoded S24 file with the saved-asset rules and wraps it with
 * its materials, naming the group and part of a bad record.
 * Refusals: `sdfm.scene_resolve` naming the group and part (a non-16-number frame, positions,
 * normals, uvs or indices of the wrong type or length, or an out-of-range index).
 */
export function meshFilePayload(value: object): {
  groups: { group: string; parts: { partId: string; name: string; frame: Float64Array;
    meshes: { material: string | null; positions: Float32Array; normals: Float32Array;
      uvs: Float32Array | null; indices: Uint32Array }[] }[] }[];
  materials: object[];
};

/**
 * Wraps a MAMA compiled material into an S11 `material_main` function: MAMA's MaterialFull and
 * MatMasks are declared under non-raa_ names, the helpers are inlined, the binding-7 reads become
 * a `const SDFM_K`, and every MaterialFull field with a MaterialOutput counterpart is copied onto
 * `raa_default_output(in)`; the rest are named in `dropped`. Pure text work, no MAMA import.
 * Refusals: `sdfm.material_unmappable` naming the material and the offending token (malformed
 * record, unknown field, unresolvable constant read, a remaining S11-forbidden token, an
 * identifier starting `raa_`, or more than 65536 bytes).
 */
export function mapMaterial(material: SdfmCompiledMaterial): SdfmMappedMaterial;

/** S11 user functions are at most this many bytes. */
export const MATERIAL_FUNCTION_MAX_BYTES: 65536;
/**
 * WGSL of `sdfm_posed_normal(in, g)`: a rest-space normal carried onto the posed surface through the rest frame
 * (rest normal in uv + colour w, tangent cross(REST_TANGENT_AXIS, rest normal)); declares `SDFM_TANGENT_AXIS`.
 */
export const FIELD_FRAME_WGSL: string;
/** WGSL of `sdff_normal(in, s: PmfSample)`: the field normal on the posed surface, the mesh normal past 60 degrees. */
export const FIELD_NORMAL_WGSL: string;
/** How one field operand is shaded (MAMA field-program.ts FieldOperandShading). */
export interface SdfmFieldOperandShading {
  color: [number, number, number];
  material: string | null;
  /** Rows of the program -> node space 3x4 affine: 12 floats. */
  toNode: number[];
  /** The material's output when its graph reads no point, normal, mask or player colour: no graph code runs for it. */
  constant?: { color: [number, number, number]; metallic: number; roughness: number; emissive: [number, number, number] };
}
/**
 * The per-pixel shading of a field part (MAMA field-program.ts PartShading; a saved game mesh part's `shading`, mesh
 * format 3): the field program (read by MAMA FIELD_PROGRAM_WGSL, part frame), its operands' shading, the part -> program
 * frame (absent = identity), the look, and whether the source used lattices or deformers (approximate normals).
 */
export interface SdfmPartShading {
  program: number[];
  operands: SdfmFieldOperandShading[];
  frame?: number[];
  single: boolean;
  look: { roughness: number; metallic: number };
  deformed?: true;
}
/** Field program layout version the interpreter reads. */
export const FIELD_PROGRAM_VERSION: 1;
/** Data block layout of a part: version, words before the program, words per operand, flags of word 1. */
export const FIELD_DATA_VERSION: 1;
export const FIELD_DATA_HEADER: 20;
export const FIELD_OPERAND_WORDS: 24;
export const FIELD_FLAG_SINGLE: 1;
export const FIELD_FLAG_DEBUG: 2;
/** Checks a part shading record; refusals: `sdfm.schema` naming the first bad field. */
export function checkPartShading(shading: unknown): SdfmPartShading;
/**
 * The per-pixel shading of a field part, the one material path of SDF Modeler's Game view and of the game: ONE
 * interpreter material function (fixed text; every part whose operands are constant or colours shares it, a part with
 * material graphs adds them, named by the set of graph bodies) and the part's data block. Register the function once
 * per `name` (`Engine.addFunction`), add `words` with `Engine.addMaterialData` and pass the base as the surface
 * material's `data`. The field normal comes from the rest point ({@link fieldSurfaceAttributes}), the operands' materials
 * (constant values, graphs at the program-space and node-space points, else colours) are mixed by the field's weights. A
 * graph material missing from `materials` wears its operand colour and is listed in `missing`. `single` leaves
 * roughness/metallic to the surface record (`look`); `debug` shows the normal (a data flag, same function).
 * Refusals: `sdfm.schema` (record, evaluator), `sdfm.material_unmappable` (a material that cannot be inlined, over 65536 bytes).
 */
export function fieldPartShading(shading: SdfmPartShading, materials: ReadonlyMap<string, SdfmCompiledMaterial> | readonly SdfmCompiledMaterial[],
  options: { field: string; debug?: boolean }): { name: string; wgsl: string; words: Uint32Array; tintMode: 'graph' | 'multiply'; missing: string[] };
/** The vertex streams of a field-shaded mesh: colours = rest position + rest normal z, uvs = rest normal xy, tangents = restTangent. */
export function fieldSurfaceAttributes(positions: Float32Array, normals: Float32Array):
  { colors: Float32Array; uvs: Float32Array; tangents: Float32Array };
/**
 * The interpreter's text without an entry point (structs, graphs, the field evaluator, the operand shading
 * `sdff_material(base, s, q, n)`) and the part's data block; it reads the block through `raa_material_word` /
 * `raa_material_f32`, which the RAA prelude (fieldPartShading) or the bake shim (field-bake) define.
 */
export function fieldShadingCore(shading: SdfmPartShading, materials: ReadonlyMap<string, SdfmCompiledMaterial> | readonly SdfmCompiledMaterial[],
  options: { field: string; debug?: boolean }): { core: string; words: Uint32Array; tintMode: 'graph' | 'multiply'; missing: string[] };

/** The one material function every baked part draws with (game render option shading 'baked'). */
export const BAKED_FUNCTION: 'sdfb.baked';
/** Its WGSL (S11 user function): samples the part's normal (tangent space + metallic) and colour (sRGB + roughness) textures. */
export const BAKED_MATERIAL_WGSL: string;
export const BAKED_DATA_VERSION: 1;
export const BAKED_DATA_WORDS: 4;
/** The emissive texture id of a baked part without emissive. */
export const BAKED_NO_TEXTURE: 4294967295;
export const BAKE_DILATE_PASSES: number;
export const BAKE_DILATE_WGSL: string;
/** A baked part's material data words: version, normal, colour and emissive texture ids (Engine.addTexture ids). */
export function bakedMaterialWords(normal: number, color: number, emissive?: number): Uint32Array;
/** Per-vertex tangents (xyz + handedness) from the atlas uvs; the bake and the drawn mesh must use the same array. */
export function partTangents(positions: Float32Array, normals: Float32Array, uvs: Float32Array, indices: Uint32Array): Float32Array;
/** UV units per metre of a part's atlas (sqrt(uv area / surface area)); 0 without area. The bake side is MAMA bakeTextureSize(scale, options). */
export function atlasScale(positions: Float32Array, uvs: Float32Array, indices: Uint32Array): number;
/** Texels the bakes of one engine may take in RAA's 4096 x 4096 texture atlas. */
export const BAKE_ATLAS_BUDGET: number;
/** The smallest side fitBakeSizes goes down to. */
export const BAKE_MIN_SIZE: number;
/** Bake sides that fit `budget` texels: the densest part (size x atlas scale) is halved first, down to BAKE_MIN_SIZE. */
export function fitBakeSizes(parts: readonly { size: number; textures: number; scale?: number }[], budget?: number): number[];
/** The bake shader of an interpreter core ({@link fieldShadingCore}). */
export function fieldBakeWgsl(core: string): string;
/** One part's bake input: the rest-space mesh with its atlas uvs and tangents, its shading and materials, MAMA's evaluator. */
export interface SdfmBakeInput {
  shading: SdfmPartShading;
  materials: ReadonlyMap<string, SdfmCompiledMaterial> | readonly SdfmCompiledMaterial[];
  field: string;
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  tangents: Float32Array;
  indices: Uint32Array;
  /** Texture side in texels (MAMA render-options.ts bakeTextureSize). */
  size: number;
}
/** A part's baked textures, RGBA8 rows top first (Engine.addTexture data); emissive null when every texel is black. */
export interface SdfmBakeResult { size: number; normal: Uint8Array; color: Uint8Array; emissive: Uint8Array | null; missing: string[]; pipeline: string }
/** A baker on the GPUDevice RAA renders with: pipelines cached by shader text (one for all constant-material parts). */
export function createFieldBaker(device: unknown): { bake(input: SdfmBakeInput): Promise<SdfmBakeResult>; readonly pipelines: number };

/** What one bake run may take of the texture atlas left (0.75); the rest stays for what a later scene brings. */
export const BAKE_RUN_SHARE: number;
/** Parts baked at once by a bake run. */
export const BAKE_BATCH: number;
/** Textures a part's bake uploads: 2 (normal, colour), 3 when an operand may glow (a graph or an emissive constant). */
export function bakeTextureCount(shading: SdfmPartShading): number;
/**
 * A part the 'baked' shading draws (SdfmBakeScheduler.prepare): draw its mesh (positions, normals, uvs, `tangents`,
 * indices) with `material`, hidden while `state` is not 'done' (the renderer's vertex-colour copy covers it; 'failed'
 * keeps that copy, `error` says why).
 */
export interface SdfmBakedPart {
  state: 'idle' | 'queued' | 'done' | 'failed' | 'released';
  label: string;
  shading: SdfmPartShading;
  materials: SdfmBakeInput['materials'];
  positions: Float32Array; normals: Float32Array; uvs: Float32Array; indices: Uint32Array;
  /** partTangents of the part: the drawn mesh carries the same array. */
  tangents: Float32Array;
  /** Its material data block (bakedMaterialWords; texture ids written when baked) and its surface material handle. */
  data: number; material: number;
  /** Engine.addTexture ids once baked, the bake side and the texels they take of the atlas. */
  textures: number[]; size: number; texels: number;
  error: string | null;
}
export interface SdfmBakeStats {
  parts: number; failed: number; ms: number; texels: number; bytes: number; textures: number; pipelines: number;
  /** What the done, unreleased parts hold now. */
  resident: { textures: number; texels: number };
  /** Texels of RAA's texture atlas the bakes take now (the next run fits in what remains). */
  used: number;
}
export interface SdfmBakeScheduler {
  /** Registers the shared baked function once, adds the part's data block and surface material, computes its tangents. */
  prepare(part: { shading: SdfmPartShading; materials: SdfmBakeInput['materials']; positions: Float32Array; normals: Float32Array;
    uvs: Float32Array; indices: Uint32Array; look?: { roughness: number; metallic: number }; label?: string }): SdfmBakedPart;
  /** Queues the idle parts a scene shows; one run (next macrotask) fits and bakes everything queued. */
  queue(parts: readonly SdfmBakedPart[]): void;
  /** Frees a part (textures, texels, data block); idempotent. */
  release(part: SdfmBakedPart): void;
  /** Settles once nothing waits. */
  done(): Promise<void>;
  stats(): SdfmBakeStats;
  dispose(): void;
}
/**
 * The bake scheduling shared by sdfm game renderers (Royal Fiesta's port, SDF Modeler's Game view): parts prepared at
 * upload, queued when a scene first shows them, each run fitted into BAKE_RUN_SHARE of the atlas left, baked
 * BAKE_BATCH at a time on `device()`, texture ids written into the parts' data blocks.
 */
export function createBakeScheduler(engine: Engine, options: {
  device: () => unknown;
  field: string;
  /** MAMA render-options.ts bakeTextureSize(scale, options). */
  textureSize: (scale: number) => number;
  onChange?: (parts: SdfmBakedPart[]) => void;
  onFailed?: (part: SdfmBakedPart, why: string) => void;
  onRun?: (stats: SdfmBakeStats) => void;
  budget?: number;
  runShare?: number;
  batch?: number;
  /** A baker to use instead of createFieldBaker(device()) (tests). */
  baker?: { bake(input: SdfmBakeInput): Promise<SdfmBakeResult>; readonly pipelines: number };
}): SdfmBakeScheduler;

/** The fixed axis of the rest tangent: (1, 2, 3) / sqrt(14). */
export const REST_TANGENT_AXIS: readonly [number, number, number];
/** The rest tangent cross(REST_TANGENT_AXIS, n) of a rest normal, not normalised (linear in n). */
export function restTangent(n: readonly number[]): [number, number, number];

/**
 * The player-facing render options an sdfm game applies to RAA (the values of MAMA's GameRenderOptions contract,
 * src/core/game/render-options.ts). Every field is optional; keys this adapter does not apply to the engine (shading,
 * bake sizes, frame pacing, lodOffset) are reported in `skipped` with the reason.
 */
export interface SdfmRenderOptions {
  /** 0 = AO off, 1..4 = the GTAO effort of {@link AO_QUALITY_LEVELS} (3 = RAA's default, 4 slices x 8 steps). */
  aoQuality?: number;
  /** Engine.setRenderScale: 0.25..2 times the backing size; the browser up/downsamples the canvas. */
  renderScale?: number;
  /** false turns the bloom pass off; true leaves it as the scene look sets it. */
  postBloom?: boolean;
  /** false turns the colour grading pass off; true leaves it as the scene look sets it. */
  postGrading?: boolean;
  /** Engine.setPostStrength: 0..1, the grade mixed with the ungraded colour and the bloom scaled. */
  postStrength?: number;
  [key: string]: unknown;
}

/** GTAO slices x steps per AO quality level; level 0 is null (the `ao.gtao` pass off). */
export const AO_QUALITY_LEVELS: readonly ({ readonly slices: number; readonly steps: number } | null)[];

/**
 * Applies `options` to `engine`. `look` names the post passes the current scene look uses (an option only turns a
 * pass off, never on against the look); call it again after every `applyEnvironment`. Engine refusals (RaaError)
 * propagate. Returns the keys applied and, for every other key given, why not (an @etcito/raa without the call, or
 * an option with no engine side).
 */
export function applyRenderOptions(engine: Engine, options: SdfmRenderOptions,
  look?: { bloom?: boolean; grading?: boolean }): { applied: string[]; skipped: { key: string; reason: string }[] };

/** The slots of the 16 floats of an instance's parameters (engine.setInstanceParams): tint 0..3, wind 4, scale 5, free 6..15. */
export const INSTANCE_PARAM_SLOTS: { readonly tint: 0; readonly wind: 4; readonly scale: 5; readonly free: 6 };
/**
 * The 16 floats of one instance's procgen attributes: linear tint rgb (tint.r/g/b, sRGB 0..1; a missing channel is 1)
 * and a flag in float 3 when any tint.* key exists, wind, scale, then the other attributes in ascending key byte order.
 * Null when `attrs` is absent or has no keys.
 */
export function instanceParams(attrs?: Record<string, number> | null): Float32Array | null;

/** Default number of coarser Paramesh meshes made by createSdfmLoader. */
export const LOD_LEVELS_DEFAULT: 2;

/**
 * Clone a canonical document and halve settings.paramesh subdivisions and fieldResolution
 * per level, flooring to integers and clamping at 1 and 16. Missing values use MAMA's
 * defaults (16 and 96). Does not mutate the input. A non-integer or negative level throws TypeError.
 */
export function lodDocument<T extends object>(document: T, level: number): T;

/**
 * Compute Rust meshLodError(fullDesc, levelDesc) for each uploaded level and call
 * setMeshLods(full, [{ mesh, error }]) once. Returns the errors in level order, or []
 * for no levels or an engine without the LOD API. Engine refusals propagate unchanged.
 */
export function attachLods(engine: Engine, full: number, fullDesc: MeshDesc,
  levels: number[], levelDescs: MeshDesc[]): number[];
