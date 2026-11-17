// Mip-chain ground scene (S49, stage mips): the inventory through validateInventory, the pair and its variants,
// the material WGSL text, framing, build and observe on stubs. Form follows examples/post/sharpen.test.mjs.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { validateInventory } from './lib/inventory.js';
import * as exposure from './scenes/exposure.js';
import * as sharpen from './scenes/sharpen.js';
import * as mips from './scenes/mips.js';
import { MIPS_MATERIAL_FUNCTION, MIPS_MATERIAL_WGSL, MIPS_PAIR_PRESET, MIPS_SCENE_ID } from './scenes/mips.js';
import { assertSubjectsInView } from '../../web/common/view-check.mjs';

const variantOf = (id) => mips.default.variants.find((v) => v.id === id);

// Consumes validateInventory (examples/post/lib/inventory.js:120), given the exposure, sharpen and mips namespaces and stage 'mips'.
test('validateInventory returns the exposure rows, then the sharpen rows, then the two mips rows, with their ids and filenames', () => {
  const rows = validateInventory([exposure, sharpen, mips], 'mips');
  const head = exposure.inventory.length + sharpen.inventory.length;
  assert.equal(rows.length, head + mips.inventory.length);
  assert.deepEqual(rows.slice(0, exposure.inventory.length), exposure.inventory);
  assert.deepEqual(rows.slice(exposure.inventory.length, head), sharpen.inventory);
  assert.deepEqual(rows.slice(head), mips.inventory);
  const scene = rows.filter((row) => row.scenario === MIPS_SCENE_ID);
  assert.deepEqual(scene.map((row) => row.id), ['mips-on', 'mips-off']);
  assert.deepEqual(scene.map((row) => row.filename), ['mips-on-distant-ground-smooth.png', 'mips-off-distant-ground-moire.png']);
});

test('the mips pair has one on and one off variant with the preset label; the scene id is not the preset label', () => {
  assert.equal(mips.default.id, MIPS_SCENE_ID);
  assert.equal(mips.default.milestone, 'mips');
  assert.notEqual(mips.default.id, MIPS_PAIR_PRESET);
  const members = mips.default.variants.filter((v) => v.measurement.pair === 'mips');
  assert.deepEqual(members.map((v) => v.measurement.state).sort(), ['off', 'on']);
  assert.deepEqual(members.map((v) => v.measurement.preset), [MIPS_PAIR_PRESET, MIPS_PAIR_PRESET]);
  for (const row of mips.inventory) assert.equal(row.pair, 'mips', row.id);
});

test('the two variants differ only in the state and the mips feature; both share selection and settings {} and have no settle', () => {
  const on = variantOf('mips-on');
  const off = variantOf('mips-off');
  assert.deepEqual(on.measurement, { pair: 'mips', state: 'on', preset: MIPS_PAIR_PRESET, features: { mips: true } });
  assert.deepEqual(off.measurement, { pair: 'mips', state: 'off', preset: MIPS_PAIR_PRESET, features: { mips: false } });
  assert.deepEqual(on.selection, off.selection);
  assert.deepEqual(on.settings, {});
  assert.deepEqual(off.settings, {});
  assert.equal(Object.hasOwn(on, 'settle'), false);
  assert.equal(Object.hasOwn(off, 'settle'), false);
});

test('the material WGSL takes dpdx of the unwrapped uv before raa_texture_grad and never differentiates fract or calls raa_texture', () => {
  const wgsl = MIPS_MATERIAL_WGSL;
  const firstDpdx = wgsl.indexOf('dpdx(');
  assert.ok(firstDpdx >= 0);
  assert.ok(firstDpdx < wgsl.indexOf('raa_texture_grad('));
  assert.ok(wgsl.includes('dpdy(in.uv)'));
  assert.equal(wgsl.includes('dpdx(fract'), false);
  assert.equal(/\braa_texture\(/.test(wgsl), false);
});

// Consumes assertSubjectsInView (web/common/view-check.mjs:80) on each mips variant's selection: the grazing camera.
test('assertSubjectsInView passes for mips-on and for mips-off', () => {
  for (const id of ['mips-on', 'mips-off']) {
    assertSubjectsInView(variantOf(id).selection, `${MIPS_SCENE_ID}/${id}`);
  }
});

test('build adds two textures with the variant mips flag, registers the material function once, gives every mesh uvs, sets the variant camera and never sets sharpening, exposure, render scale or a pass', async () => {
  for (const id of ['mips-on', 'mips-off']) {
    const calls = [];
    let handle = 0;
    const engine = new Proxy({}, {
      get: (_, method) => (...args) => {
        calls.push([method, ...args]);
        return handle++;
      },
    });
    const variant = variantOf(id);
    const flag = variant.measurement.features.mips;
    const { instances, textures } = await mips.default.build(engine, { raa: {}, content: {} }, variant);
    const named = (method) => calls.filter(([name]) => name === method);
    assert.deepEqual(named('addTexture').map(([, desc]) => desc.mips), [flag, flag], id);
    assert.deepEqual(named('addMaterialFunction'), [['addMaterialFunction', MIPS_MATERIAL_FUNCTION, MIPS_MATERIAL_WGSL]], id);
    assert.equal(named('addMesh').length, 4, id);
    for (const [, desc] of named('addMesh')) assert.equal(desc.uvs.length, 2 * desc.positions.length / 3, id);
    for (const method of ['setSharpening', 'setAutoExposure', 'setRenderScale', 'setPassEnabled']) {
      assert.equal(named(method).length, 0, `${id}: ${method}`);
    }
    assert.deepEqual(named('setCamera'), [['setCamera', variant.selection.camera]], id);
    assert.deepEqual(instances.map((i) => i.name), ['ground', 'far-cube-0', 'far-cube-1', 'far-cube-2'], id);
    assert.equal(new Set(instances.map((i) => i.name)).size, instances.length, id);
    assert.equal(textures.length, 2, id);
    assert.deepEqual(textures.map((t) => t.mips), [flag, flag], id);
  }
});

test('observe on two mips-false textures passes both assertions for mips-off and fails mips-enabled for mips-on', async () => {
  const state = { textures: [{ id: 0, mips: false }, { id: 1, mips: false }] };
  const off = await mips.observe({}, variantOf('mips-off'), state);
  assert.deepEqual(off.values, { 'mips-enabled': false, 'texture-count': 2 });
  assert.deepEqual(off.assertions.map((a) => [a.name, a.passed]), [['mips-enabled', true], ['texture-count', true]]);
  const on = await mips.observe({}, variantOf('mips-on'), state);
  assert.deepEqual(on.assertions.map((a) => [a.name, a.passed]), [['mips-enabled', false], ['texture-count', true]]);
  for (const a of [...off.assertions, ...on.assertions]) assert.ok(typeof a.detail === 'string' && a.detail.length > 0, a.name);
});

test('observe on two mips-true textures passes both assertions for mips-on', async () => {
  const state = { textures: [{ id: 0, mips: true }, { id: 1, mips: true }] };
  const result = await mips.observe({}, variantOf('mips-on'), state);
  assert.deepEqual(result.values, { 'mips-enabled': true, 'texture-count': 2 });
  assert.deepEqual(result.assertions.map((a) => [a.name, a.passed]), [['mips-enabled', true], ['texture-count', true]]);
});
