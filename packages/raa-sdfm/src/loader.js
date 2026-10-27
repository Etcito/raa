// createSdfmLoader: sdfm.game-asset v1 -> RAA meshes, surface materials and nodes (raa-sdfm-pkg).
// Validation and hash check (validate.js), meshing by MAMA (worker pool, or in-process through the
// bridge), upload through the engine (addMesh, addMaterialFunction, addSurfaceMaterial), spawn.
import { SdfmError } from './errors.js';
import { validateAsset, checkHash } from './validate.js';
import { mapMaterial } from './map-material.js';
import { importMama, meshDocument } from './mama-bridge.js';
import { createPool } from './pool.js';

const DISPOSED = 'the sdfm loader was disposed';

/**
 * @param {{ engine: object, mama: () => Promise<string | URL | object>, workerUrl?: string | URL,
 *   workers?: number }} options `mama` resolves to the URL of MAMA's `dist/` (meshing in the worker
 *   pool) or to MAMA's exports as `importMama` returns them, optionally with `pool` (a MAMA
 *   ComputePool), meshed in-process (node).
 * @returns {{ validate(asset: unknown): { ok: boolean, errors: { code: string, path: string, message: string }[] },
 *   load(asset: string | object, opts?: { onProgress?: (p: { stage: string, done: number, total: number }) => void }): Promise<object>,
 *   spawn(asset: object, parent?: number | null): { root: number, parts: Record<string, { node: number, instance: number }> },
 *   dispose(): void }}
 */
export function createSdfmLoader(options) {
  const { engine, mama, workerUrl = new URL('../worker.js', import.meta.url), workers } = options ?? {};
  if (!engine || typeof engine !== 'object') throw new TypeError('createSdfmLoader: options.engine is required');
  if (typeof mama !== 'function') throw new TypeError('createSdfmLoader: options.mama must be a function');

  /** @type {Map<string, Promise<object>>} hash -> SdfmAsset */
  const cache = new Map();
  /** Material function names registered by this loader (one per graph hash). */
  const functions = new Set();
  let source = null;
  let pool = null;
  let hashMama = null;
  let disposed = false;

  const disposedError = () => new SdfmError('sdfm.disposed', DISPOSED);
  const checkLive = () => {
    if (disposed) throw disposedError();
  };

  /** MAMA as the caller's `mama()` gives it, resolved once per loader. */
  function resolveMama() {
    if (!source) {
      source = Promise.resolve()
        .then(() => mama())
        .catch((e) => {
          source = null;
          if (e instanceof SdfmError) throw e;
          throw new SdfmError('sdfm.worker', `options.mama() failed: ${messageOf(e)}`);
        });
    }
    return source;
  }

  async function mesh(asset) {
    const m = await resolveMama();
    checkLive();
    if (typeof m === 'string' || m instanceof URL) {
      pool ??= createPool({ size: workers, workerUrl, mamaUrl: m });
      return pool.mesh(asset.document, asset.mesher);
    }
    if (m && typeof m === 'object') return meshDocument(m, asset.document, { pool: m.pool });
    throw new SdfmError('sdfm.worker', 'options.mama() must resolve to a MAMA URL or to MAMA exports');
  }

  async function build(asset, progress) {
    progress('mesh', 0, 1);
    const meshed = await mesh(asset);
    checkLive();
    progress('mesh', 1, 1);

    const partsIn = meshed.groups.flatMap((g) => g.parts.map((p) => ({ group: g.group, part: p })));
    const total = meshed.materials.length + partsIn.length;
    let done = 0;
    progress('upload', done, total);

    /** MAMA material id -> surface material handle */
    const handles = new Map();
    for (const m of meshed.materials) {
      const mapped = mapMaterial(m);
      registerFunction(mapped.functionName, mapped.wgsl);
      handles.set(m.name, engine.addSurfaceMaterial({ ...mapped.desc }));
      progress('upload', (done += 1), total);
    }

    const names = new Set();
    const parts = [];
    for (const { group, part } of partsIn) {
      const { desc, materials } = partMesh(part, handles);
      const handle = engine.addMesh(desc);
      const name = names.has(part.name) ? `${group}/${part.partId}` : part.name;
      names.add(name);
      const { t, r } = frameToTr(part.frame);
      parts.push({ name, groupId: group, rigid: true, parent: null, t, r, mesh: handle, materials });
      progress('upload', (done += 1), total);
    }

    return {
      id: asset.id,
      hash: asset.hash,
      name: asset.name,
      category: asset.category,
      world: asset.world,
      parts,
      anchors: asset.anchors,
      bounds: asset.bounds,
      collision: asset.collision,
    };
  }

  function registerFunction(name, wgsl) {
    if (functions.has(name)) return;
    try {
      engine.addMaterialFunction(name, wgsl);
    } catch (e) {
      if (e?.code !== 'material.fn_duplicate') throw e;
    }
    functions.add(name);
  }

  return {
    validate(asset) {
      return validateAsset(asset);
    },

    async load(input, opts = {}) {
      checkLive();
      const onProgress = opts?.onProgress;
      const progress = (stage, done, total) => {
        if (typeof onProgress === 'function') onProgress({ stage, done, total });
      };
      progress('validate', 0, 1);
      const asset = parse(input);
      const { errors } = validateAsset(asset);
      if (errors.length > 0) {
        const [first] = errors;
        throw new SdfmError(first.code, first.message, first.path);
      }
      const m = await resolveMama();
      hashMama ??= typeof m === 'string' || m instanceof URL
        ? importMama(m, { browser: false }) : Promise.resolve(m);
      const exports = await hashMama;
      if (typeof exports?.serializeDocument !== 'function') {
        throw new SdfmError('sdfm.worker', 'MAMA serializeDocument export is required for asset hashes');
      }
      await checkHash(asset, exports.serializeDocument);
      checkLive();
      progress('validate', 1, 1);

      const hit = cache.get(asset.hash);
      if (hit) return hit;
      const pending = build(asset, progress);
      cache.set(asset.hash, pending);
      pending.catch(() => {
        if (cache.get(asset.hash) === pending) cache.delete(asset.hash);
      });
      return pending;
    },

    spawn(asset, parent = null) {
      checkLive();
      const root = engine.addNode(parent ?? null);
      const byName = new Map(asset.parts.map((p) => [p.name, p]));
      const out = {};
      for (const part of parentsFirst(asset.parts, byName)) {
        const parentNode = part.parent !== null && byName.has(part.parent) ? out[part.parent].node : root;
        const node = engine.addNode(parentNode, part.t, part.r);
        const instance = engine.addMeshInstance(node, part.mesh);
        out[part.name] = { node, instance };
      }
      return { root, parts: out };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      pool?.terminate();
      pool = null;
      cache.clear();
      functions.clear();
    },
  };
}

function parse(input) {
  if (typeof input !== 'string') return input;
  try {
    return JSON.parse(input);
  } catch (e) {
    throw new SdfmError('sdfm.schema', `asset text is not JSON: ${messageOf(e)}`, '');
  }
}

/**
 * One S13 MeshDesc for a part: its MAMA meshes concatenated, one sub-mesh per material (meshes of
 * one material are contiguous, materials in order of first use). uvs are (0, 0) for a mesh
 * without them, and absent when no mesh has any.
 *
 * @returns {{ desc: object, materials: number[] }} materials: the sub-mesh material handles, in sub-mesh order
 */
function partMesh(part, handles) {
  const byMaterial = new Map();
  for (const m of part.meshes) {
    const key = m.material ?? null;
    if (!byMaterial.has(key)) byMaterial.set(key, []);
    byMaterial.get(key).push(m);
  }
  const ordered = [...byMaterial.values()].flat();
  const vertices = ordered.reduce((n, m) => n + m.positions.length / 3, 0);
  const indexCount = ordered.reduce((n, m) => n + m.indices.length, 0);
  const withUvs = ordered.some((m) => m.uvs);
  const positions = new Float32Array(vertices * 3);
  const normals = new Float32Array(vertices * 3);
  const uvs = withUvs ? new Float32Array(vertices * 2) : null;
  const indices = new Uint32Array(indexCount);

  const edges = ordered.some((m) => m.edges != null) ? [] : null;
  const subMeshes = [];
  const materials = [];
  let base = 0;
  let first = 0;
  for (const [key, meshes] of byMaterial) {
    const start = first;
    for (const m of meshes) {
      positions.set(m.positions, base * 3);
      normals.set(m.normals, base * 3);
      if (uvs && m.uvs) uvs.set(m.uvs, base * 2);
      for (let i = 0; i < m.indices.length; i += 1) indices[first + i] = m.indices[i] + base;
      if (edges) {
        // Preserve polygon topology where supplied; use triangle edges for older meshes.
        if (m.edges != null) {
          for (const index of m.edges) edges.push(index + base);
        } else {
          const seen = new Set();
          for (let i = 0; i < m.indices.length; i += 3) {
            for (const [a, b] of [[m.indices[i], m.indices[i + 1]], [m.indices[i + 1], m.indices[i + 2]], [m.indices[i + 2], m.indices[i]]]) {
              const key = `${Math.min(a, b)},${Math.max(a, b)}`;
              if (!seen.has(key)) { seen.add(key); edges.push(a + base, b + base); }
            }
          }
        }
      }
      base += m.positions.length / 3;
      first += m.indices.length;
    }
    const sub = { first: start, count: first - start };
    if (key !== null) {
      const handle = handles.get(key);
      if (handle === undefined) {
        throw new SdfmError('sdfm.material_unmappable', `part '${part.name}' uses material '${key}' that MAMA did not build`);
      }
      sub.material = handle;
      materials.push(handle);
    }
    subMeshes.push(sub);
  }
  const desc = { positions, normals, indices, subMeshes };
  if (uvs) desc.uvs = uvs;
  if (edges) desc.edges = Uint32Array.from(edges);
  return { desc, materials };
}

/**
 * Translation and rotation of a column-major 4x4 frame (MAMA part frame). The rotation is read
 * from the normalised basis columns, so a scale in the frame is not carried.
 *
 * @param {ArrayLike<number>} f 16 numbers
 * @returns {{ t: [number, number, number], r: [number, number, number, number] }}
 */
function frameToTr(f) {
  const t = [f[12], f[13], f[14]];
  const col = (c) => {
    const x = f[c * 4];
    const y = f[c * 4 + 1];
    const z = f[c * 4 + 2];
    const len = Math.hypot(x, y, z) || 1;
    return [x / len, y / len, z / len];
  };
  const [c0, c1, c2] = [col(0), col(1), col(2)];
  // m(row, col) = c<col>[row]
  const m00 = c0[0], m10 = c0[1], m20 = c0[2];
  const m01 = c1[0], m11 = c1[1], m21 = c1[2];
  const m02 = c2[0], m12 = c2[1], m22 = c2[2];
  const trace = m00 + m11 + m22;
  let q;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s];
  }
  const n = Math.hypot(...q) || 1;
  return { t, r: q.map((v) => v / n) };
}

/** Parts ordered so that every part comes after its parent (unknown parents count as the root). */
function parentsFirst(parts, byName) {
  const out = [];
  const state = new Map();
  const visit = (part) => {
    const s = state.get(part.name);
    if (s === 'done') return;
    if (s === 'visiting') {
      throw new SdfmError('sdfm.schema', `part '${part.name}' is its own ancestor`);
    }
    state.set(part.name, 'visiting');
    const parent = part.parent !== null ? byName.get(part.parent) : undefined;
    if (parent) visit(parent);
    state.set(part.name, 'done');
    out.push(part);
  };
  for (const part of parts) visit(part);
  return out;
}

function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}
