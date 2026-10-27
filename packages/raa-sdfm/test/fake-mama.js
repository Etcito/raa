// Fake MAMA for node tests: the exports src/mama-bridge.js uses, with MAMA 0.1.0 @ 864ebc3
// signatures and result shapes (src/core/game/mesh-assets.ts, src/sdfm.ts,
// src/core/material-graph/compile.ts and helpers.ts, src/browser/index.ts). Geometry and
// compiled graphs are not computed: the fake document carries them under `fake` keys.
//
// Fake document: { nodes: { <id>: { kind, name, asset?, fake?: { parts?, refuse?, throw? } } },
//   materials: { <id>: { name, fake: { wgslBody, helpers, constants } | { error } } } }
// fake.parts: [{ partId, name, frame: number[16], subMeshes: [{ material, vertices, uvs? }] }]

/** Calls seen by the fake, for assertions. */
export const calls = [];

/** MaterialFull as MAMA declares it (src/core/material-graph/wgsl-parts.ts @ 864ebc3). */
export const MATERIAL_FULL_WGSL =
  'struct MaterialFull { base_color: vec3f, metallic: f32, roughness: f32, emissive: vec3f, height: f32 }';

/** Always-emitted constants and scalar helpers (src/core/material-graph/helpers.ts BASE_WGSL, shortened). */
export const BASE_WGSL =
  '\nconst MAT_PI: f32 = 3.141592653589793;\n\nfn mat_clamp01(x: f32) -> f32 {\n  if (x < 0.0) { return 0.0; }\n  if (x > 1.0) { return 1.0; }\n  return x;\n}\n';

/** Helper id -> { deps?, code } (src/core/material-graph/helpers.ts). */
export const HELPERS = {
  axes: { code: '\nfn mat_axis(a: u32, p: vec3f) -> f32 {\n  return p[min(a, 2u)];\n}\n' },
  tri: { code: '\nfn mat_tri(x: f32) -> f32 {\n  return abs(fract(x) * 2.0 - 1.0);\n}\n' },
  pat_stripes: {
    deps: ['axes', 'tri'],
    code: '\nfn mat_pat_stripes(a: u32, p: vec3f, s: f32) -> f32 {\n  return mat_tri(mat_axis(a, p) * s);\n}\n',
  },
};

export class FakeSdfmError extends Error {
  constructor(code, message, fix = '') {
    super(message);
    this.code = code;
    this.fix = fix;
  }
}

/** readSdfm(json: string | unknown): SdfDocument (src/sdfm.ts). */
export function readSdfm(json) {
  calls.push({ fn: 'readSdfm' });
  const doc = typeof json === 'string' ? JSON.parse(json) : structuredClone(json);
  if (!doc || typeof doc.nodes !== 'object') throw new FakeSdfmError('DOC_INVALID', 'document has no nodes');
  doc.materials ??= {};
  return doc;
}

/** buildMaterial(doc, id): BuiltMaterial (src/sdfm.ts), through a fake compileMaterialGraph. */
export function buildMaterial(doc, id) {
  const def = doc.materials[id];
  if (!def) throw new FakeSdfmError('MATERIAL_NOT_FOUND', `no material with id '${id}'`);
  if (def.fake.error) {
    return { id, name: def.name, def, graph: null, error: { code: 'MATERIAL_GRAPH_CYCLE', message: def.fake.error, fix: '' } };
  }
  return { id, name: def.name, def, graph: compileMaterialGraph(def), error: null };
}

/** CompiledGraph (src/core/material-graph/compile.ts) with the fields the bridge reads plus the rest at rest values. */
function compileMaterialGraph(def) {
  return {
    wgslBody: def.fake.wgslBody,
    helpers: [...def.fake.helpers],
    constants: Uint32Array.from(def.fake.constants),
    needMask: 0,
    aoWord: null,
    maskWords: { curvature: null, cavity: null, aoScale: null },
    order: ['out'],
    bump: null,
    displace: false,
    authored: { normal: false, ao: false, opacity: false },
    bitmapBlocks: [],
    svgplusBlocks: [],
    imageBlocks: [],
    coordsBlocks: [],
    usesCoords: false,
  };
}

/** The value refusal of mesh-assets.ts `unsupported()`. */
function unsupported(name, output, inner) {
  return {
    code: 'MESHER_GROUP_UNSUPPORTED',
    message: `group '${name}' cannot be meshed with ${output} (${output}): ${inner.message}`,
    fix: inner.fix ? inner.fix : 'Pick another mesher in the Mesher section.',
  };
}

function fakeSubMesh(spec) {
  const n = spec.vertices;
  const positions = new Float32Array(n * 3).map((_, i) => (i % 3) * 0.25 + Math.floor(i / 3));
  const normals = new Float32Array(n * 3).map((_, i) => (i % 3 === 1 ? 1 : 0));
  const indices = new Uint32Array(n).map((_, i) => i);
  const uvs = spec.uvs ? new Float32Array(n * 2).map((_, i) => i / (n * 2)) : undefined;
  return { material: spec.material, positions, normals, ...(uvs ? { uvs } : {}), indices, ...(spec.edges != null ? { edges: Uint32Array.from(spec.edges) } : {}) };
}

/** meshAssets(doc, { groups?, output?, pool?, onProgress?, onNotice? }): Promise<AssetMeshResult[]>. */
export async function meshAssets(doc, o = {}) {
  calls.push({ fn: 'meshAssets', options: o });
  const ids = Object.keys(doc.nodes);
  const targets = o.groups ?? ids.filter((id) => doc.nodes[id].kind === 'group' && doc.nodes[id].asset !== undefined);
  const docOutput = 'paramesh';
  const out = [];
  for (const group of targets) {
    const node = Object.prototype.hasOwnProperty.call(doc.nodes, group) ? doc.nodes[group] : undefined;
    if (!node || node.kind !== 'group') {
      const what = node ? `is a ${node.kind}, not a group` : 'is not a node of the scene';
      out.push({
        group, asset: null, output: docOutput, parts: [], materials: [],
        error: {
          code: 'ASSET_FIELD_NOT_GROUP', message: `'${group}' ${what}, so it cannot be meshed as an asset`,
          fix: 'Pick a group in the scene tree and mesh it again',
        },
      });
      continue;
    }
    const name = node.name !== '' ? node.name : group;
    const asset = node.asset ?? null;
    const fake = node.fake ?? {};
    if (fake.throw) throw new TypeError(fake.throw);
    if (fake.refuse) {
      out.push({ group, asset, output: docOutput, parts: [], materials: [], error: unsupported(name, docOutput, fake.refuse) });
      continue;
    }
    const parts = (fake.parts ?? []).map((p) => ({
      partId: p.partId, name: p.name, frame: Float64Array.from(p.frame), subMeshes: p.subMeshes.map(fakeSubMesh),
    }));
    const used = new Set();
    for (const part of parts) for (const s of part.subMeshes) if (s.material !== null) used.add(s.material);
    const materials = [...used].sort().map((id) => buildMaterial(doc, id));
    out.push({ group, asset, output: docOutput, parts, materials, error: null });
  }
  return out;
}

/** setupBrowserHost({ workerUrl, manifoldWasmUrl, size?, workerMode? }): ComputePool (src/browser/index.ts). */
export function setupBrowserHost(opts) {
  calls.push({ fn: 'setupBrowserHost', options: opts });
  return fakePool(opts.size ?? 2);
}

/** A ComputePool (src/core/compute/pool.ts) that runs nothing. */
export function fakePool(size = 1) {
  return {
    size,
    run: async () => { throw new Error('fake pool runs no job'); },
    map: async () => { throw new Error('fake pool runs no job'); },
    dispose() {},
  };
}

// Deliberately differs from adapter pretty JSON: tests must use the supplied serializer.
export const serializeDocument = (document) => JSON.stringify(document);
