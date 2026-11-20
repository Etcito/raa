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
 * `text` with every `extra_words[i]` read (MAMA's binding 7: the graph's constants at `b`, the image pages they point to)
 * replaced by `fn(i)`, a read of RAA material data words. Nested and balanced indices are kept.
 */
export function wordReads(text, fn) {
  let out = '', at = 0;
  for (const m of text.matchAll(/\bextra_words\s*\[/g)) {
    if (m.index < at) continue;
    let depth = 1, i = m.index + m[0].length;
    for (; i < text.length && depth > 0; i++) depth += text[i] === '[' ? 1 : text[i] === ']' ? -1 : 0;
    if (depth > 0) break;
    out += `${text.slice(at, m.index)}${fn}(${wordReads(text.slice(m.index + m[0].length, i - 1), fn)})`;
    at = i;
  }
  return out + text.slice(at);
}

/** 16 hex digits of two independent 32-bit passes (FNV-1a, a rotating odd multiply) over the UTF-16 units of `text` (a name, not a security hash). */
export function textHash(text) {
  let a = 0x811c9dc5, b = 0x9e3779b9;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    a = Math.imul(a ^ c, 0x01000193) >>> 0;
    b = Math.imul(((b ^ c) << 13) | ((b ^ c) >>> 19), 0x5bd1e995) >>> 0;
  }
  return a.toString(16).padStart(8, '0') + b.toString(16).padStart(8, '0');
}

/**
 * The `pages` of a MamaMaterial (indices of its constant words that point into the host's image page words, MAMA
 * slots.ts graphPagesOfDoc), [] when absent. Throws through `fail` when malformed.
 */
export function materialPages(m, fail) {
  if (m.pages === undefined) return [];
  if (!Array.isArray(m.pages) || m.pages.some((i) => !Number.isInteger(i) || i < 0 || i >= m.constants.length)) {
    fail(`'pages' is not an array of indices of its ${m.constants.length} constant words`);
  }
  return m.pages;
}

/** The material data base of the host's image page words (`options.pagesBase`), checked when a material has pages. */
export function pagesBaseOf(options, fail) {
  const base = options?.pagesBase;
  if (!Number.isInteger(base) || base < 0 || base > 0xffffffff) {
    fail("samples image pages: pass options.pagesBase, the material data base (Engine.addMaterialData) of the document's page words (MAMA graphPagesOfDoc)");
  }
  return base;
}

/** True when the MAMA text declares mat_footprint (the M4 prelude): the host then sets it before the graph runs. */
export function declaresFootprint(texts) {
  return texts.some((t) => /\bvar\s*<\s*private\s*>\s*mat_footprint\b/.test(t));
}

/**
 * `sdfm_footprint(in)`: metres per screen pixel at the shaded point, what SDF Modeler's viewport writes into mat_footprint
 * before material_eval (ray distance x pixel angle; the pixel size in an orthographic view). SVG PLUS textures pick
 * their mip level from it.
 */
export const FOOTPRINT_WGSL = /* wgsl */ `
fn sdfm_footprint(in: MaterialInput) -> f32 {
  let pixel = 2.0 * raa_frame.viewport.w / max(abs(raa_frame.proj[1][1]), 1e-6);
  if (raa_frame.proj[3][3] == 1.0) { return pixel; }
  return length(raa_frame.camera_pos.xyz - in.position) * pixel;
}
`;

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
 *   constants: number[], fields: string[], cel?: boolean, pages?: number[] }} m MamaMaterial
 * @param {{cel?: boolean, pagesBase?: number}} [options] Host override; cel:true sets the surface option (material flag 128).
 * `tintMode` is 'graph' when the MAMA WGSL reads `mat_instance_tint` (only the S23 `input.player_color` node emits
 * it): RAA then skips the post-material tint multiply. Otherwise it is 'multiply'.
 * A material with `pages` (it samples image pages, MAMA graphPagesOfDoc) needs `options.pagesBase` and returns `words`:
 * see mapPagedMaterial; the host adds them as material data and sets `desc.data` to their base.
 *
 * @returns {{ functionName: string, wgsl: string, desc: { function: string, blend: 'opaque', cel?: boolean }, dropped: string[],
 *   tintMode: 'graph' | 'multiply', words?: Uint32Array }}
 * @throws {SdfmError} `sdfm.material_unmappable` naming the material and the offending token
 */
export function mapMaterial(m, options = {}) {
  const name = checkShape(m);
  const fail = (what) => {
    throw new SdfmError('sdfm.material_unmappable', `material '${name}': ${what}`);
  };

  for (const f of m.fields) {
    if (!(f in MAMA_FIELD_TYPES)) fail(`unknown MaterialFull field '${f}'`);
  }
  if (new Set(m.fields).size !== m.fields.length) fail('duplicate MaterialFull field');
  const pages = materialPages(m, fail);
  if (pages.length > 0) return mapPagedMaterial(m, pages, options, fail);

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
  let tintMode = 'multiply';
  for (const text of [body, ...helpers]) {
    const toks = tokens(text);
    if (toks.includes('mat_instance_tint')) tintMode = 'graph';
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
  return { functionName, wgsl, desc: { function: functionName, blend: 'opaque', ...((options.cel ?? m.cel) === true ? { cel: true } : {}) }, dropped, tintMode };
}

/**
 * mapMaterial of a graph that samples image pages (texture.svgplus, texture.bitmap, surface image pages): its helpers
 * read the pages through binding-7 indices it takes from its constants, so constants and pages are both RAA material
 * data words here. `words` (the constants, each `pages` word plus options.pagesBase) is the host's to place
 * (Engine.addMaterialData) and to name as the record's `data`; every `extra_words[i]` read becomes a material data read
 * at `i`, the graph's `b` is the record's data base. The text does not depend on the words, so the function is shared
 * by every material with the same graph structure.
 */
function mapPagedMaterial(m, pages, options, fail) {
  const pagesBase = pagesBaseOf(options, fail);
  const rewrite = (text) => wordReads(text, 'sdfm_word')
    .replace(/\bMaterialFull\b/g, FULL_STRUCT)
    .replace(/\bMatMasks\b/g, MASKS_STRUCT);
  const body = rewrite(m.body);
  const helpers = m.helpers.map((h) => `// MAMA helper ${h.id}\n${rewrite(h.wgsl).trim()}\n`);
  let tintMode = 'multiply';
  for (const text of [body, ...helpers]) {
    const toks = tokens(text);
    if (toks.includes('mat_instance_tint')) tintMode = 'graph';
    if (toks.includes('extra_words')) fail(`'extra_words' is not read as extra_words[...]`);
    const raa = toks.find((t) => /^raa_/.test(t));
    if (raa) fail(`'${raa}' (identifiers starting with raa_ are reserved)`);
  }
  const footprint = declaresFootprint(m.helpers.map((h) => h.wgsl));
  const fields = m.fields.map((f) => `${f}: ${MAMA_FIELD_TYPES[f]}`).join(', ');
  const copied = m.fields.filter((f) => OUTPUT_TYPES[f] === MAMA_FIELD_TYPES[f]);
  const dropped = m.fields.filter((f) => !copied.includes(f));
  const wgsl = [
    '// sdfm graph with image pages (constants and pages in material data words)',
    `struct ${FULL_STRUCT} { ${fields} }`,
    MASKS_DECL,
    'fn sdfm_word(i: u32) -> u32 { return raa_material_word(i); }',
    ...(footprint ? [FOOTPRINT_WGSL] : []),
    '',
    ...helpers,
    `fn ${GRAPH_FN}(b: u32, p: vec3<f32>, pn: vec3<f32>, n: vec3<f32>, mk: ${MASKS_STRUCT}) -> ${FULL_STRUCT} {`,
    body.replace(/\s+$/, ''),
    '}',
    '',
    'fn material_main(in: MaterialInput) -> MaterialOutput {',
    ...(footprint ? ['  mat_footprint = sdfm_footprint(in);'] : []),
    `  let m = ${GRAPH_FN}(raa_material_data_base(in.record), in.position, in.local_position, in.normal, ${MASKS_FALLBACK});`,
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
  const words = Uint32Array.from(m.constants, (w) => w >>> 0);
  for (const i of pages) words[i] = (words[i] + pagesBase) >>> 0;
  const functionName = `sdfm.p${textHash(wgsl)}`;
  const cel = (options.cel ?? m.cel) === true;
  return { functionName, wgsl, desc: { function: functionName, blend: 'opaque', ...(cel ? { cel: true } : {}) }, dropped, tintMode, words };
}

/** Checks the MamaMaterial shape; returns its name. */
export function checkShape(m) {
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
export function tokens(src) {
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
export function forbiddenToken(src) {
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
