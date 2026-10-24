// Hand-written types of @etcito/raa-sdfm 0.1.0 (ARCHITECTURE S15). Plain ES modules, no build step.
// The adapter turns `sdfm.game-asset` v1 files into RAA meshes and surface materials, meshed and
// compiled by MAMA. Engine refusals (RaaError) propagate unchanged; the adapter's own refusals are
// SdfmError with one of the eleven S15 codes. MAMA is reached only through src/mama-bridge.js.

import type { Engine, SurfaceBlend } from '@etcito/raa';

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
 * The eleven S15 refusal codes:
 * `sdfm.format`, `sdfm.version`, `sdfm.schema`, `sdfm.not_finite`, `sdfm.id`,
 * `sdfm.cap_document`, `sdfm.hash_mismatch`, `sdfm.mesher_refused`,
 * `sdfm.material_unmappable`, `sdfm.worker`, `sdfm.disposed`.
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
  | 'sdfm.disposed';

/**
 * A refusal of the sdfm adapter. The message is prefixed with `<code>: `, like RaaError, and
 * every failure of this package is one of these codes (engine refusals stay RaaError).
 * Refusals: any {@link SdfmErrorCode}; `path` is a JSON pointer into the asset where one exists.
 */
export class SdfmError extends Error {
  /** One of the eleven S15 codes. */
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
 * `stage` runs validate, then mesh, then upload; `done` is between 0 and `total`.
 * Refusals: none; progress is informational.
 */
export interface SdfmProgress {
  stage: 'validate' | 'mesh' | 'upload';
  done: number;
  total: number;
}

/**
 * One uploaded part of an {@link SdfmAsset}: a rigid MAMA part with its mesh and sub-mesh
 * material handles. MAMA has no rigid flag or parent link, so `rigid` is always true and
 * `parent` is always null: every part is flat under the spawn root and carries its own frame.
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
  /** Surface material handles of the sub-meshes, in sub-mesh order. */
  materials: number[];
}

/**
 * A loaded asset: the `sdfm.game-asset` v1 fields passed through untouched (`id`, `hash`, `name`,
 * `category`, `world`, `anchors`, `bounds`, `collision`) plus the uploaded `parts`.
 * Refusals: {@link SdfmLoader.load} raises the validation/hash/mesh/material codes of
 * {@link SdfmErrorCode}; engine refusals are RaaError.
 */
export interface SdfmAsset {
  /** Asset id (`^[a-z][a-z0-9]*(\.[a-z0-9-]+)+$`). */
  id: string;
  /** sha256 of the canonical document; the load cache key. */
  hash: string;
  name: string;
  category: string;
  world: string;
  parts: SdfmPart[];
  anchors: Record<string, { t: Vec3; r: Quat }>;
  bounds: { min: Vec3; max: Vec3 };
  /** Passed through untouched; never used for rendering. */
  collision: unknown[];
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
  /** MAMA's `readSdfm`: builds MAMA's document from the asset's `document`. */
  readSdfm(document: unknown): unknown;
  /** MAMA's `meshAssets(doc, { pool, groups })`; each result is `{ group, parts, materials, error }`. */
  meshAssets(document: unknown, options: { pool: unknown; groups?: string[] }): Promise<unknown[]>;
  /** MAMA's helper table: helper id -> WGSL source and dependencies. */
  HELPERS: Record<string, { code: string; deps?: string[] }>;
  /** MAMA's browser meshing pool factory (`mama/browser`); unused in node. */
  setupBrowserHost?(options: { workerUrl: string | URL; manifoldWasmUrl: string; size?: number }): unknown;
  /** A MAMA `ComputePool` used for in-process meshing. */
  pool?: unknown;
}

/**
 * Options of {@link createSdfmLoader}.
 * Refusals: a missing `engine` or `mama` throws `TypeError`; failures during `load` are
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
   * the same hash resolves from the cache and calls MAMA no more.
   * Refusals: the first validation error as `SdfmError` (with its path), otherwise
   * `sdfm.hash_mismatch`, `sdfm.mesher_refused`, `sdfm.material_unmappable`, `sdfm.worker`,
   * `sdfm.disposed`; engine refusals are RaaError.
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
 * Wraps a MAMA compiled material into an S11 `material_main` function: MAMA's MaterialFull and
 * MatMasks are declared under non-raa_ names, the helpers are inlined, the binding-7 reads become
 * a `const SDFM_K`, and every MaterialFull field with a MaterialOutput counterpart is copied onto
 * `raa_default_output(in)`; the rest are named in `dropped`. Pure text work, no MAMA import.
 * Refusals: `sdfm.material_unmappable` naming the material and the offending token (malformed
 * record, unknown field, unresolvable constant read, a remaining S11-forbidden token, an
 * identifier starting `raa_`, or more than 65536 bytes).
 */
export function mapMaterial(material: SdfmCompiledMaterial): SdfmMappedMaterial;
