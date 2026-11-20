// Image pages (SVG PLUS, Bitmap, surface images) of a MAMA graph in RAA material data: mapMaterial and fieldPartShading
// with `pages` / `pagesBase` (MAMA slots.ts graphPagesOfDoc).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SdfmError } from '../src/errors.js';
import { fieldPartShading } from '../src/field-shading.js';
import { mapMaterial } from '../src/map-material.js';

const FIELD = `struct PmfSample { d: f32, g: vec3f, la: u32, lb: u32, t: f32 }
fn pmf_eval(base: u32, p: vec3f) -> PmfSample { return PmfSample(length(p) - pmf_f(base + 3u), p, 0u, 1u, 0.5); }`;
const ROWS = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

/** A graph shaped like texture.svgplus: its block at b (word 0 = page base, a pointer into the pages), its helper reads the pages. */
const paged = (over = {}) => ({
  name: 'slate', key: 'a'.repeat(64), fields: ['base_color', 'metallic', 'roughness', 'emissive', 'height'],
  constants: [7, 2, 0x3f000000], pages: [0],
  helpers: [
    { id: 'mat_m4_prelude', wgsl: 'var<private> mat_footprint: f32 = 0.0;\nvar<private> mat_label: u32 = 0u;' },
    { id: 'svg', wgsl: 'fn mat_svg(b: u32) -> f32 { return f32(extra_words[extra_words[b] + extra_words[b + 1u]]) * mat_footprint; }' },
  ],
  body: '  let v = mat_svg(b + 0u) * bitcast<f32>(extra_words[b + 2u]);\n  return MaterialFull(vec3f(v), 0.0, 0.5, vec3f(0.0), 0.0);\n',
  ...over,
});

test('mapMaterial: a paged graph reads constants and pages as material data, its words carry the relocated pointers', () => {
  const r = mapMaterial(paged(), { pagesBase: 1000 });
  assert.deepEqual(Array.from(r.words), [1007, 2, 0x3f000000]);
  assert.ok(!/\bextra_words\b/.test(r.wgsl));
  assert.ok(r.wgsl.includes('sdfm_word(sdfm_word(b) + sdfm_word(b + 1u))'), 'nested page reads become data reads');
  assert.ok(r.wgsl.includes('sdfm_graph(raa_material_data_base(in.record), '), 'b is the record data base');
  assert.ok(r.wgsl.includes('mat_footprint = sdfm_footprint(in);'), 'the footprint is set before the graph');
  // The text is independent of the words: two materials of the same structure share one function.
  assert.equal(mapMaterial(paged({ name: 'eroded', constants: [9, 3, 0] }), { pagesBase: 5 }).functionName, r.functionName);
  assert.throws(() => mapMaterial(paged()), (e) => e instanceof SdfmError && e.code === 'sdfm.material_unmappable' && e.message.includes('pagesBase'));
});

test('fieldPartShading: page pointers of each graph move to options.pagesBase in the part block', () => {
  const shading = { program: [1, 0, 8, 0.25, 0, 0, 0, 0], single: false, look: { roughness: 0.4, metallic: 0 },
    operands: [{ color: [1, 0, 0], material: 'slate', toNode: ROWS }] };
  const r = fieldPartShading(shading, [paged()], { field: FIELD, pagesBase: 500 });
  const table = r.words[3], constants = r.words[table + 5];
  assert.deepEqual(Array.from(r.words.subarray(constants, constants + 3)), [507, 2, 0x3f000000]);
  assert.ok(r.wgsl.includes('sdff_w(sdff_w(b) + sdff_w(b + 1u))'));
  assert.ok(r.wgsl.includes('mat_footprint = sdfm_footprint(in);') && r.wgsl.includes('fn sdfm_footprint('));
  assert.throws(() => fieldPartShading(shading, [paged()], { field: FIELD }), (e) => e.code === 'sdfm.material_unmappable' && e.message.includes("'slate'"));
});
