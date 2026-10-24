import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mapMaterial } from '../src/map-material.js';
import { canonicalJson, sha256Hex } from '../src/canonical.js';
import { SdfmError } from '../src/errors.js';

// Fixtures: MamaMaterial objects whose body, helpers and constants come from a MAMA 864ebc3 compile.
// compiled-basic: a constant-only output node. compiled-unmappable: noise -> mix_color, three helpers plus MAMA's base
// and noise WGSL, 22 binding-7 words; tests derive the refused variants from it.
const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), 'utf8'));

/** A copy of `m` with one helper's WGSL (or the body) replaced, and its key left as is. */
function withHelper(m, wgsl) {
  return { ...m, helpers: [...m.helpers, { id: 'injected', wgsl }] };
}

function expectUnmappable(m, token) {
  assert.throws(() => mapMaterial(m), (e) => {
    assert.ok(e instanceof SdfmError);
    assert.equal(e.code, 'sdfm.material_unmappable');
    assert.ok(e.message.startsWith('sdfm.material_unmappable: '));
    assert.ok(e.message.includes(`'${m.name}'`), `message names the material: ${e.message}`);
    assert.ok(e.message.includes(token), `message names ${token}: ${e.message}`);
    return true;
  });
}

test('fixture keys are the sha256 of canonicalJson({ body, helpers, constants, fields })', async () => {
  for (const name of ['compiled-basic', 'compiled-unmappable']) {
    const m = fixture(name);
    const { body, helpers, constants, fields } = m;
    assert.equal(m.key, await sha256Hex(canonicalJson({ body, helpers, constants, fields })));
  }
});

test('basic material maps onto material_main with an opaque desc', () => {
  const m = fixture('compiled-basic');
  const r = mapMaterial(m);
  assert.equal(r.functionName, 'sdfm.' + m.key.slice(0, 16));
  assert.deepEqual(r.desc, { function: r.functionName, blend: 'opaque' });
  assert.ok(r.wgsl.includes('fn material_main(in: MaterialInput) -> MaterialOutput {'));
  assert.ok(r.wgsl.includes('var out = raa_default_output(in);'));
  assert.ok(!/\bMaterialFull\b/.test(r.wgsl), 'MaterialFull is renamed');
  assert.match(r.wgsl, /struct SdfmMaterialFull \{ base_color: vec3<f32>, metallic: f32, roughness: f32, emissive: vec3<f32>, height: f32 \}/);
  assert.ok(!('params' in r), 'S11 user functions take no params');
});

test('functionName is stable and depends only on the key', () => {
  const m = fixture('compiled-unmappable');
  const a = mapMaterial(m);
  const b = mapMaterial(fixture('compiled-unmappable'));
  assert.equal(a.functionName, b.functionName);
  assert.equal(a.wgsl, b.wgsl);
  assert.equal(a.functionName, 'sdfm.914aaa9f632122e9');
  assert.equal(mapMaterial({ ...m, name: 'renamed' }).functionName, a.functionName);
  assert.notEqual(mapMaterial(fixture('compiled-basic')).functionName, a.functionName);
});

test('binding-7 reads become indices of an inlined SDFM_K', () => {
  const m = fixture('compiled-unmappable');
  const r = mapMaterial(m);
  assert.ok(m.body.includes('bitcast<f32>(extra_words[b + 3u])'), 'the fixture reads binding 7');
  assert.ok(r.wgsl.includes(`const SDFM_K = array<u32, ${m.constants.length}>(${m.constants.map((w) => `${w}u`).join(', ')});`));
  assert.ok(!r.wgsl.includes('extra_words'));
  assert.ok(r.wgsl.includes('var sc = bitcast<f32>(SDFM_K[3u]);'), 'float reads keep bitcast<f32>');
  assert.ok(r.wgsl.includes('let tri = SDFM_K[6u] == 1u;'), 'u32 reads stay u32');
  assert.ok(r.wgsl.includes('mat_pat_noise(qj, P, SDFM_K[5u], SDFM_K[4u])'));
  // the same rewrite applies inside helpers
  const h = mapMaterial(withHelper(m, 'fn sdfm_probe(b: u32) -> f32 { return bitcast<f32>(extra_words[b + 21u]); }'));
  assert.ok(h.wgsl.includes('return bitcast<f32>(SDFM_K[21u]);'));
});

test('helpers are inlined in order', () => {
  const m = fixture('compiled-unmappable');
  const r = mapMaterial(m);
  let at = -1;
  for (const h of m.helpers) {
    const i = r.wgsl.indexOf(`// MAMA helper ${h.id}\n`);
    assert.ok(i > at, `helper ${h.id} inlined after the previous one`);
    at = i;
  }
  for (const fn of ['fn mat_blend(', 'fn mat_tri_weights(', 'fn mat_pat_noise(', 'fn fbm3(', 'fn pcg_hash(']) {
    assert.ok(r.wgsl.includes(fn), fn);
  }
  assert.ok(at < r.wgsl.indexOf('fn sdfm_graph('));
});

test('fields with a MaterialOutput counterpart are copied, the others dropped', () => {
  const r = mapMaterial(fixture('compiled-basic'));
  assert.deepEqual(r.dropped, ['height']);
  for (const f of ['base_color', 'metallic', 'roughness', 'emissive']) assert.ok(r.wgsl.includes(`  out.${f} = m.${f};`), f);
  assert.ok(!r.wgsl.includes('out.height'));
});

test('a constant read past the constants block is refused', () => {
  const m = fixture('compiled-basic');
  expectUnmappable({ ...m, constants: m.constants.slice(0, 5) }, 'extra_words[b + 5u]');
});

test('a binding-7 read that is not a graph constant is refused', () => {
  const m = fixture('compiled-basic');
  expectUnmappable(withHelper(m, 'fn sdfm_probe(i: u32) -> u32 { return extra_words[i]; }'), 'extra_words');
});

for (const [token, wgsl] of [
  ['@group', '@group(3) @binding(0) var<private> sdfm_x: f32;'],
  ['@binding', 'var<private> @binding(0) sdfm_x: f32;'],
  ['@vertex', '@vertex fn sdfm_vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }'],
  ['@fragment', '@fragment fn sdfm_fs() {}'],
  ['@compute', '@compute @workgroup_size(1) fn sdfm_cs() {}'],
  ['var<storage', 'var<storage, read> sdfm_x: array<f32>;'],
  ['var<uniform', 'var<uniform> sdfm_x: vec4<f32>;'],
  ['var<workgroup', 'var<workgroup> sdfm_x: f32;'],
  ['enable', 'enable f16;'],
  ['override', 'override sdfm_x: f32 = 1.0;'],
]) {
  test(`forbidden token ${token} is refused`, () => {
    expectUnmappable(withHelper(fixture('compiled-basic'), wgsl), `'${token}'`);
  });
}

test('forbidden words inside comments are not tokens', () => {
  const r = mapMaterial(withHelper(fixture('compiled-basic'), '// enable override @group var<storage\n/* @vertex /* nested */ */\n'));
  assert.ok(r.wgsl.includes('// enable override'));
});

test('a raa_ identifier in MAMA text is refused', () => {
  const m = fixture('compiled-basic');
  expectUnmappable(withHelper(m, 'fn raa_hash3(p: vec3<f32>) -> f32 { return 0.0; }'), 'raa_hash3');
  expectUnmappable({ ...m, body: m.body.replace('mat_clamp01(', 'raa_luma(vec3f(') }, 'raa_luma');
});

test('more than 65536 bytes is refused', () => {
  const big = `const SDFM_PAD: f32 = 0.0;\n${'// pad\n'.repeat(10000)}`;
  expectUnmappable(withHelper(fixture('compiled-basic'), big), '65536');
});

test('malformed MamaMaterial inputs are refused', () => {
  const m = fixture('compiled-basic');
  expectUnmappable({ ...m, key: 'abc' }, "'key'");
  expectUnmappable({ ...m, body: '  return vec3f(0.0);\n' }, "'body'");
  expectUnmappable({ ...m, constants: [1.5] }, "'constants'");
  expectUnmappable({ ...m, fields: [...m.fields, 'sparkle'] }, "'sparkle'");
  assert.throws(() => mapMaterial(null), { code: 'sdfm.material_unmappable' });
});
