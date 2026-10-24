// The only module of @etcito/raa-sdfm that reaches MAMA (MAMA 0.1.0 @ 864ebc3). Every call made
// here has a row in the README section `## MAMA calls`. MAMA's values are normalised into plain
// typed arrays and MamaMaterial records; every refusal leaves this module as an SdfmError.
import { canonicalJson, sha256Hex } from './canonical.js';
import { SdfmError } from './errors.js';

/**
 * MaterialFull field names in declaration order, pinned to MAMA 864ebc3
 * (`src/core/material-graph/wgsl-parts.ts`: `struct MaterialFull`).
 */
export const MATERIAL_FULL_FIELDS = Object.freeze(['base_color', 'metallic', 'roughness', 'emissive', 'height']);

/** MAMA import specifiers -> paths of MAMA's `dist/` (its package.json `exports`). */
const SPECIFIERS = Object.freeze({
  root: ['mama', 'index.js'],
  meshAssets: ['mama/core/game/mesh-assets.js', 'core/game/mesh-assets.js'],
  helpers: ['mama/core/material-graph/helpers.js', 'core/material-graph/helpers.js'],
  browser: ['mama/browser', 'browser/index.js'],
});

/**
 * Imports the MAMA exports the bridge uses and returns them as one object (the shape the fake
 * MAMA of the tests also has). Without `base` the bare specifiers are imported (node, or a page
 * with an import map); with `base` (the URL of MAMA's `dist/` directory) the files under it are.
 *
 * @param {string | URL} [base]
 * @param {{ browser?: boolean }} [options] browser false skips `mama/browser` (node)
 * @returns {Promise<{ readSdfm: Function, meshAssets: Function, HELPERS: object, BASE_WGSL: string,
 *   setupBrowserHost?: Function }>}
 * @throws SdfmError `sdfm.worker` when a module cannot be imported
 */
export async function importMama(base, options = {}) {
  const load = async ([bare, file]) => {
    const specifier = base === undefined ? bare : new URL(file, base).href;
    try {
      return await import(specifier);
    } catch (e) {
      throw new SdfmError('sdfm.worker', `cannot import MAMA module ${specifier}: ${messageOf(e)}`);
    }
  };
  const browser = options.browser ?? true;
  const [root, mesh, helpers, host] = await Promise.all([
    load(SPECIFIERS.root),
    load(SPECIFIERS.meshAssets),
    load(SPECIFIERS.helpers),
    browser ? load(SPECIFIERS.browser) : null,
  ]);
  return {
    readSdfm: root.readSdfm,
    meshAssets: mesh.meshAssets,
    HELPERS: helpers.HELPERS,
    BASE_WGSL: helpers.BASE_WGSL,
    ...(host ? { setupBrowserHost: host.setupBrowserHost } : {}),
  };
}

/**
 * The browser meshing pool: MAMA's `setupBrowserHost`. In node the caller builds its own pool.
 *
 * @param {{ setupBrowserHost?: Function }} mama the object of {@link importMama}
 * @param {{ workerUrl: string | URL, manifoldWasmUrl: string, size?: number, workerMode?: 'module' | 'blob' }} options
 *   workerMode 'blob' (classic blob: workers that import the worker) for a page with an opaque origin
 * @returns {object} MAMA's ComputePool
 * @throws SdfmError `sdfm.worker` when MAMA has no browser host or refuses to set it up
 */
export function createBrowserPool(mama, options) {
  if (typeof mama.setupBrowserHost !== 'function') {
    throw new SdfmError('sdfm.worker', 'MAMA browser host (setupBrowserHost of mama/browser) is not loaded');
  }
  const { workerUrl, manifoldWasmUrl, size, workerMode } = options;
  try {
    return mama.setupBrowserHost({
      workerUrl,
      manifoldWasmUrl,
      ...(size !== undefined ? { size } : {}),
      ...(workerMode !== undefined ? { workerMode } : {}),
    });
  } catch (e) {
    throw new SdfmError('sdfm.worker', `MAMA setupBrowserHost failed: ${messageOf(e)}`);
  }
}

/**
 * Meshes an sdfm document with MAMA's `meshAssets(doc, { pool, groups })` and normalises the
 * result. Every part is rigid and the hierarchy is flat: no rigid flag or parent is read from
 * the document. Sub-meshes are not merged: each MAMA sub-mesh becomes one mesh of its part.
 *
 * @param {{ readSdfm: Function, meshAssets: Function, HELPERS: object }} mama the object of {@link importMama}
 * @param {unknown} document the asset's `document`, unchanged
 * @param {{ pool: object, groups?: string[] }} options pool: MAMA ComputePool; groups: MAMA's default when absent
 * @returns {Promise<{
 *   groups: { group: string, parts: { partId: string, name: string, frame: Float64Array,
 *     meshes: { material: string | null, positions: Float32Array, normals: Float32Array,
 *       uvs: Float32Array | null, indices: Uint32Array }[] }[] }[],
 *   materials: MamaMaterial[] }>}
 * @throws SdfmError `sdfm.mesher_refused` (first refused group in result order, path /document),
 *   `sdfm.material_unmappable` (a material MAMA cannot build or a helper without source),
 *   `sdfm.worker` (any other MAMA exception)
 */
export async function meshDocument(mama, document, options) {
  const { pool, groups } = options;
  let results;
  try {
    const doc = mama.readSdfm(document);
    results = await mama.meshAssets(doc, { pool, groups });
  } catch (e) {
    throw new SdfmError('sdfm.worker', `MAMA meshing failed: ${messageOf(e)}`);
  }
  if (!Array.isArray(results)) {
    throw new SdfmError('sdfm.worker', 'MAMA meshAssets did not return an array of group results');
  }

  const refused = results.find((r) => r.error);
  if (refused) {
    const { message, fix } = refused.error;
    const tail = fix ? ` (fix: ${fix})` : '';
    throw new SdfmError('sdfm.mesher_refused', `${refused.group}: ${message}${tail}`, '/document');
  }

  const out = { groups: [], materials: [] };
  const seen = new Set();
  for (const result of results) {
    out.groups.push({
      group: result.group,
      parts: result.parts.map((part) => ({
        partId: part.partId,
        name: part.name,
        frame: Float64Array.from(part.frame),
        meshes: part.subMeshes.map((sub) => ({
          material: sub.material ?? null,
          positions: Float32Array.from(sub.positions),
          normals: Float32Array.from(sub.normals),
          uvs: sub.uvs ? Float32Array.from(sub.uvs) : null,
          indices: Uint32Array.from(sub.indices),
        })),
      })),
    });
    for (const built of result.materials) {
      if (seen.has(built.id)) continue;
      seen.add(built.id);
      out.materials.push(await toMamaMaterial(mama, built));
    }
  }
  return out;
}

/**
 * @typedef {{ name: string, key: string, body: string, helpers: { id: string, wgsl: string }[],
 *   constants: number[], fields: string[] }} MamaMaterial
 */

/**
 * Turns one MAMA BuiltMaterial (`buildMaterial`, which runs `compileMaterialGraph`) into a
 * MamaMaterial. MAMA's `BASE_WGSL` (constants and scalar helpers such as `mat_clamp01`, which MAMA
 * emits in front of every material module, slots.ts) comes first as helper `base`; the graph's
 * helpers follow, resolved through MAMA's `HELPERS` with their dependencies first.
 *
 * @param {{ HELPERS: object, BASE_WGSL: string }} mama
 * @param {{ id: string, graph: { wgslBody: string, helpers: string[], constants: Uint32Array } | null,
 *   error: { code: string, message: string, fix: string } | null }} built
 * @returns {Promise<MamaMaterial>}
 * @throws SdfmError `sdfm.material_unmappable` (build error, no BASE_WGSL, or a helper without WGSL source)
 */
export async function toMamaMaterial(mama, built) {
  if (!built.graph) {
    const reason = built.error ? `${built.error.code}: ${built.error.message}` : 'no compiled graph';
    throw new SdfmError('sdfm.material_unmappable', `material '${built.id}' was not built by MAMA (${reason})`);
  }
  if (typeof mama.BASE_WGSL !== 'string') {
    throw new SdfmError('sdfm.material_unmappable', `material '${built.id}': MAMA BASE_WGSL has no WGSL source`);
  }
  const body = built.graph.wgslBody;
  const helpers = [{ id: 'base', wgsl: mama.BASE_WGSL }, ...resolveHelpers(mama.HELPERS, built.graph.helpers, built.id)];
  const constants = Array.from(built.graph.constants);
  const fields = [...MATERIAL_FULL_FIELDS];
  const key = await sha256Hex(canonicalJson({ body, helpers, constants, fields }));
  return { name: built.id, key, body, helpers, constants, fields };
}

function resolveHelpers(table, ids, material) {
  const out = [];
  const done = new Set();
  const visit = (id, chain) => {
    if (done.has(id)) return;
    const helper = table && Object.prototype.hasOwnProperty.call(table, id) ? table[id] : undefined;
    if (!helper || typeof helper.code !== 'string') {
      throw new SdfmError('sdfm.material_unmappable', `material '${material}': helper '${id}' has no WGSL source in MAMA`);
    }
    if (chain.includes(id)) {
      throw new SdfmError('sdfm.material_unmappable', `material '${material}': helper '${id}' depends on itself`);
    }
    for (const dep of helper.deps ?? []) visit(dep, [...chain, id]);
    done.add(id);
    out.push({ id, wgsl: helper.code });
  };
  for (const id of ids) visit(id, []);
  return out;
}

function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}
