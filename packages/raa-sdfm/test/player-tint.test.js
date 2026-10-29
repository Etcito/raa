import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mapMaterial } from '../src/map-material.js';
import * as fakeMama from './fake-mama.js';
import { createFakeEngine } from './fake-engine.js';
import { createSdfmLoader } from '../index.js';
import { documentHash } from '../src/canonical.js';

// S23: a material whose WGSL reads mat_instance_tint (the input.player_color emit) maps to tint mode 'graph'.
const fixture = (name) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8'));
const TINT_BODY = '  let c = select(vec3f(0.5, 0.5, 0.5), mat_instance_tint.rgb, mat_instance_tint_set > 0.5);\n'
  + '  return MaterialFull(c, 0.0, 0.5, vec3f(0.0), 0.0);\n';

test('a material using mat_instance_tint maps to tint mode graph', () => {
  const m = { ...fixture('compiled-basic.json'), body: TINT_BODY };
  const r = mapMaterial(m);
  assert.equal(r.tintMode, 'graph');
  assert.ok(r.wgsl.includes('mat_instance_tint_set > 0.5'));
});

test('a plain material maps to tint mode multiply', () => {
  assert.equal(mapMaterial(fixture('compiled-basic.json')).tintMode, 'multiply');
});

test('the loader calls setMaterialTintMode only for the graph material', async () => {
  const engine = createFakeEngine();
  const tintCalls = [];
  engine.setMaterialTintMode = (material, mode) => tintCalls.push([material, mode]);
  const asset = fixture('rigid.asset.json');
  asset.id = 'raatest.tinted';
  asset.document.materials.paint.fake.wgslBody = TINT_BODY;
  asset.hash = await documentHash(asset.document, fakeMama.serializeDocument);
  const loader = createSdfmLoader({ engine, mama: async () => fakeMama });
  await loader.load(asset);
  const surfaces = engine.calls.addSurfaceMaterial.length;
  assert.ok(surfaces >= 2, `two materials registered, got ${surfaces}`);
  assert.equal(tintCalls.length, 1);
  assert.equal(tintCalls[0][1], 'graph');
  assert.ok(engine.materials.has(tintCalls[0][0]));
});
