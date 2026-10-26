// MAMA material (MamaMaterial, produced by mama-bridge.js from a MAMA CompiledGraph) -> an S11 user material function.
// Pure text work: no MAMA import, no engine call. The engine validates the result again when it is registered.
import { SdfmError } from './errors.js';

/** S11 user functions are at most this many bytes. */
const MAX_BYTES = 65536;

/**
 * WGSL type of each MaterialFull field MAMA declares (src/core/material-graph/wgsl-parts.ts @ 864ebc3:
 * `struct MaterialFull { base_color: vec3f, metallic: f32, roughness: f32, emissive: vec3f, height: f32 }`).
 * @type {Readonly<Record<string, string>>}
 */
const MAMA_FIELD_TYPES = Object.freeze({
  base_color: 'vec3<f32>',
  metallic: 'f32',
  roughness: 'f32',
  emissive: 'vec3<f32>',
  height: 'f32',
});

/**
 * MaterialOutput fields of S11 and their WGSL types. A MaterialFull field with the same name and type is copied; any
 * other field is dropped (README, "MaterialFull fields").
 * @type {Readonly<Record<string, string>>}
 */
const OUTPUT_TYPES = Object.freeze({
  base_color: 'vec3<f32>', metallic: 'f32', roughness: 'f32', emissive: 'vec3<f32>', normal: 'vec3<f32>', ao: 'f32',
  opacity: 'f32', clearcoat: 'f32', clearcoat_roughness: 'f32', sheen_color: 'vec3<f32>', sheen_roughness: 'f32',
  transmission: 'f32', ior: 'f32', thickness: 'f32', attenuation_color: 'vec3<f32>', attenuation_distance: 'f32',
});

/** Names the wrapper gives MAMA's structs (S11 forbids only the raa_ prefix). */
const FULL_STRUCT = 'SdfmMaterialFull';
const MASKS_STRUCT = 'SdfmMatMasks';
const GRAPH_FN = 'sdfm_graph';

/**
 * MAMA's MatMasks (wgsl-parts.ts @ 864ebc3) and the fallback values MAMA itself uses where no SDF tap exists
 * (slots.ts mat_dh_K: curvature 0.5, cavity 1, ao 1, height 0, proximity 1).
 */
const MASKS_DECL = `struct ${MASKS_STRUCT} { curvature: f32, cavity: f32, ao: f32, height: f32, proximity: f32 }`;
const MASKS_FALLBACK = `${MASKS_STRUCT}(0.5, 1.0, 1.0, 0.0, 1.0)`;

/** A binding-7 read of the graph's constants block: `extra_words[b + Ku]`. */
const CONSTANT_READ = /extra_words\s*\[\s*b\s*\+\s*(\d+)u?\s*\]/g;

/**
 * Maps a MAMA material onto an S11 material function.
 *
 * The WGSL declares MAMA's MaterialFull and MatMasks under non-raa_ names, inlines the helpers, inlines the constants
 * as `const SDFM_K = array<u32, N>(...)` (every `extra_words[b + K]` read becomes `SDFM_K[K]`; MAMA already wraps
 * float reads in `bitcast<f32>`), wraps the body in `fn sdfm_graph(b, p, pn, n, mk)` and defines
 * `fn material_main(in: MaterialInput) -> MaterialOutput`, which starts from `raa_default_output(in)` and copies
 * every MaterialFull field that has a MaterialOutput counterpart. Mesh attributes are not set, so MAMA's graph sees
 * the SDF surface values (no mesh UV, white vertex colour), as in SDF Modeler's viewport.
 *
 * @param {{ name: string, key: string, body: string, helpers: { id: string, wgsl: string }[],
 *   constants: number[], fields: string[] }} m MamaMaterial
 * @returns {{ functionName: string, wgsl: string, desc: { function: string, blend: 'opaque' }, dropped: string[] }}
 * @throws {SdfmError} `sdfm.material_unmappable` naming the material and the offending token
 */
export function mapMaterial(m) {
  const name = checkShape(m);
  const fail = (what) => {
    throw new SdfmError('sdfm.material_unmappable', `material '${name}': ${what}`);
  };

  for (const f of m.fields) {
    if (!(f in MAMA_FIELD_TYPES)) fail(`unknown MaterialFull field '${f}'`);
  }
  if (new Set(m.fields).size !== m.fields.length) fail('duplicate MaterialFull field');

  const n = m.constants.length;
  const rewrite = (text) => text
    .replace(CONSTANT_READ, (read, k) => {
      if (Number(k) >= n) fail(`'${read}' reads past the ${n} constant words`);
      return `SDFM_K[${k}u]`;
    })
    .replace(/\bMaterialFull\b/g, FULL_STRUCT)
    .replace(/\bMatMasks\b/g, MASKS_STRUCT);

  const body = rewrite(m.body);
  const helpers = m.helpers.map((h) => `// MAMA helper ${h.id}\n${rewrite(h.wgsl).trim()}\n`);

  // MAMA-sourced text: no binding read may remain and no raa_ identifier may appear.
  for (const text of [body, ...helpers]) {
    const toks = tokens(text);
    const left = toks.find((t) => t === 'extra_words');
    if (left) fail(`'extra_words' (a binding-7 read that is not a constant of this graph)`);
    const raa = toks.find((t) => /^raa_/.test(t));
    if (raa) fail(`'${raa}' (identifiers starting with raa_ are reserved)`);
  }

  const fields = m.fields.map((f) => `${f}: ${MAMA_FIELD_TYPES[f]}`).join(', ');
  const copied = m.fields.filter((f) => OUTPUT_TYPES[f] === MAMA_FIELD_TYPES[f]);
  const dropped = m.fields.filter((f) => !copied.includes(f));
  const words = m.constants.map((w) => `${w}u`).join(', ');

  const wgsl = [
    `// sdfm graph ${m.key}`,
    `struct ${FULL_STRUCT} { ${fields} }`,
    MASKS_DECL,
    ...(n > 0 ? [`const SDFM_K = array<u32, ${n}>(${words});`] : []),
    '',
    ...helpers,
    `fn ${GRAPH_FN}(b: u32, p: vec3<f32>, pn: vec3<f32>, n: vec3<f32>, mk: ${MASKS_STRUCT}) -> ${FULL_STRUCT} {`,
    body.replace(/\s+$/, ''),
    '}',
    '',
    'fn material_main(in: MaterialInput) -> MaterialOutput {',
    `  let m = ${GRAPH_FN}(0u, in.position, in.local_position, in.normal, ${MASKS_FALLBACK});`,
    '  var out = raa_default_output(in);',
    ...copied.map((f) => `  out.${f} = m.${f};`),
    '  return out;',
    '}',
    '',
  ].join('\n');

  const forbidden = forbiddenToken(wgsl);
  if (forbidden) fail(`'${forbidden}' is forbidden in a material function (S11)`);
  const bytes = new TextEncoder().encode(wgsl).length;
  if (bytes > MAX_BYTES) fail(`'${bytes} bytes' exceeds the ${MAX_BYTES}-byte limit`);

  const functionName = `sdfm.${m.key.slice(0, 16)}`;
  return { functionName, wgsl, desc: { function: functionName, blend: 'opaque' }, dropped };
}

/** Checks the MamaMaterial shape; returns its name. */
function checkShape(m) {
  const bad = (what) => {
    const label = m && typeof m.name === 'string' ? m.name : '(unnamed)';
    throw new SdfmError('sdfm.material_unmappable', `material '${label}': ${what}`);
  };
  if (m === null || typeof m !== 'object') bad('not a MamaMaterial object');
  if (typeof m.name !== 'string' || m.name === '') bad("'name' is not a non-empty string");
  if (typeof m.key !== 'string' || !/^[0-9a-f]{64}$/.test(m.key)) bad("'key' is not 64 lowercase hex digits");
  if (typeof m.body !== 'string' || !/\breturn\s+MaterialFull\s*\(/.test(m.body)) {
    bad("'body' does not return MaterialFull(...)");
  }
  if (!Array.isArray(m.helpers) || m.helpers.some((h) => !h || typeof h.id !== 'string' || typeof h.wgsl !== 'string')) {
    bad("'helpers' is not an array of { id, wgsl }");
  }
  if (!Array.isArray(m.constants) || m.constants.some((w) => !Number.isInteger(w) || w < 0 || w > 0xffffffff)) {
    bad("'constants' is not an array of u32 words");
  }
  if (!Array.isArray(m.fields) || m.fields.length === 0 || m.fields.some((f) => typeof f !== 'string')) {
    bad("'fields' is not a non-empty array of names");
  }
  return m.name;
}

/** WGSL tokens of `src` with comments removed (line comments and nested block comments). */
function tokens(src) {
  let out = '';
  for (let i = 0; i < src.length;) {
    if (src.startsWith('//', i)) {
      while (i < src.length && src[i] !== '\n') i++;
    } else if (src.startsWith('/*', i)) {
      let depth = 0;
      do {
        if (src.startsWith('/*', i)) { depth++; i += 2; } else if (src.startsWith('*/', i)) { depth--; i += 2; } else i++;
      } while (depth > 0 && i < src.length);
      out += ' ';
    } else {
      out += src[i++];
    }
  }
  return out.match(/[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]/g) ?? [];
}

/** The first S11-forbidden token of `src` (token scan, comments ignored, as raa-shader does), or null. */
function forbiddenToken(src) {
  const t = tokens(src);
  for (let i = 0; i < t.length; i++) {
    if (t[i] === '@' && ['group', 'binding', 'vertex', 'fragment', 'compute'].includes(t[i + 1])) return `@${t[i + 1]}`;
    if (t[i] === 'var' && t[i + 1] === '<' && ['storage', 'uniform', 'workgroup'].includes(t[i + 2])) {
      return `var<${t[i + 2]}`;
    }
    if (t[i] === 'enable' || t[i] === 'override') return t[i];
  }
  return null;
}
