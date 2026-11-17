// Sharpen-detail scene (S49, stage sharpen): the inventory through validateInventory, the variant settings,
// framing, observe and build on stubs. Form follows examples/post/exposure.test.mjs.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { validateInventory } from './lib/inventory.js';
import * as exposure from './scenes/exposure.js';
import * as sharpen from './scenes/sharpen.js';
import { assertSubjectsInView } from '../../web/common/view-check.mjs';

const variantOf = (id) => sharpen.default.variants.find((v) => v.id === id);

// Consumes validateInventory (examples/post/lib/inventory.js:120), given the exposure and sharpen namespaces and stage 'sharpen'.
test('validateInventory returns the exposure rows first, then the four sharpen-detail rows, with their ids and filenames', () => {
  const rows = validateInventory([exposure, sharpen], 'sharpen');
  assert.equal(rows.length, exposure.inventory.length + sharpen.inventory.length);
  assert.deepEqual(rows.slice(0, exposure.inventory.length), exposure.inventory);
  assert.deepEqual(rows.slice(exposure.inventory.length), sharpen.inventory);
  const sharp = rows.filter((row) => row.scenario === 'sharpen-detail');
  assert.deepEqual(sharp.map((row) => row.id), ['sharpen-on', 'sharpen-off', 'upscale-067-sharpen-on', 'upscale-067-sharpen-off']);
  assert.deepEqual(sharp.map((row) => row.filename), [
    'sharpen-on-fine-detail-crisp.png',
    'sharpen-off-fine-detail-soft.png',
    'upscale-067-sharpen-on-fine-detail-crisp.png',
    'upscale-067-sharpen-off-fine-detail-blurred.png',
  ]);
});

test('each row names the pair of its variant; each pair has one on and one off variant with one preset', () => {
  for (const row of sharpen.inventory) {
    assert.equal(row.pair, variantOf(row.mode).measurement.pair);
  }
  for (const [pair, preset] of [['sharpen', 'sharpen-detail-scale-1'], ['upscale-067-sharpen', 'sharpen-detail-scale-067']]) {
    const members = sharpen.default.variants.filter((v) => v.measurement.pair === pair);
    assert.deepEqual(members.map((v) => v.measurement.state).sort(), ['off', 'on']);
    assert.deepEqual(members.map((v) => v.measurement.preset), [preset, preset]);
  }
});

test('every variant enables TAA; render scale 0.67 only in the upscale rows; no settle; a pair differs only in sharpening.enabled', () => {
  for (const v of sharpen.default.variants) {
    assert.equal(v.settings.passes.taa, true, v.id);
    if (v.id.startsWith('upscale-067-')) assert.equal(v.settings.renderScale, 0.67, v.id);
    else assert.equal(Object.hasOwn(v.settings, 'renderScale'), false, v.id);
    assert.equal(Object.hasOwn(v, 'settle'), false, v.id);
    assert.equal(v.measurement.features.sharpening, v.settings.sharpening.enabled, v.id);
  }
  for (const on of sharpen.default.variants.filter((v) => v.id.endsWith('-on'))) {
    const off = variantOf(on.id.replace(/-on$/, '-off'));
    assert.equal(on.settings.sharpening.enabled, true, on.id);
    assert.equal(off.settings.sharpening.enabled, false, off.id);
    assert.deepEqual({ ...on.settings, sharpening: null }, { ...off.settings, sharpening: null }, on.id);
    assert.deepEqual(on.selection, off.selection, on.id);
  }
});

// Consumes assertSubjectsInView (web/common/view-check.mjs:80) on each variant's selection: the one detail camera.
test('assertSubjectsInView passes for every variant', () => {
  for (const v of sharpen.default.variants) {
    assertSubjectsInView(v.selection, `${sharpen.default.id}/${v.id}`);
  }
});

const engineStub = {
  renderScale: () => 0.67,
  passes: () => [{ name: 'taa', enabled: true, order: 100 }],
  sharpening: () => ({ enabled: true, sharpness: 0.9 }),
};

test('observe on upscale-067-sharpen-on reads 0.67, TAA on and sharpening on: three passing assertions in order', async () => {
  const result = await sharpen.observe(engineStub, variantOf('upscale-067-sharpen-on'), { instances: [] });
  assert.deepEqual(result.values, { 'render-scale': 0.67, 'taa-enabled': true, 'sharpen-enabled': true });
  assert.deepEqual(result.assertions.map((a) => a.name), ['render-scale', 'taa-enabled', 'sharpen-enabled']);
  for (const a of result.assertions) {
    assert.equal(a.passed, true, a.name);
    assert.ok(typeof a.detail === 'string' && a.detail.length > 0, a.name);
  }
});

test('observe on sharpen-off fails render-scale and sharpen-enabled against the stub and passes taa-enabled', async () => {
  const result = await sharpen.observe(engineStub, variantOf('sharpen-off'), { instances: [] });
  assert.deepEqual(result.assertions.map((a) => [a.name, a.passed]), [['render-scale', false], ['taa-enabled', true], ['sharpen-enabled', false]]);
  for (const a of result.assertions) assert.ok(typeof a.detail === 'string' && a.detail.length > 0, a.name);
});

test('build sets the sky, light, ambient, bloom and camera, and never sets sharpening, the render scale or a pass', async () => {
  const calls = [];
  let handle = 0;
  const engine = new Proxy({}, {
    get: (_, method) => (...args) => {
      calls.push([method, ...args]);
      return handle++;
    },
  });
  const variant = variantOf('upscale-067-sharpen-off');
  const { instances } = await sharpen.default.build(engine, { raa: {}, content: {} }, variant);
  const named = (method) => calls.filter(([name]) => name === method);
  assert.deepEqual(named('setCamera'), [['setCamera', variant.selection.camera]]);
  assert.equal(named('setSharpening').length, 0);
  assert.equal(named('setRenderScale').length, 0);
  assert.equal(named('setPassEnabled').length, 0);
  assert.equal(instances.length, 13);
  assert.equal(new Set(instances.map((i) => i.name)).size, instances.length);
});
