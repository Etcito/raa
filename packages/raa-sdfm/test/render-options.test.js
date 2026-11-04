import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AO_QUALITY_LEVELS, applyRenderOptions } from '../index.js';

function engine(methods = ['setPassEnabled', 'setAo', 'setRenderScale', 'setPostStrength']) {
  const calls = [];
  const e = { calls };
  for (const name of methods) e[name] = (...args) => calls.push([name, ...args]);
  return e;
}

test('aoQuality 0 turns GTAO off, 1..4 set its slices x steps (3 = RAA default 4 x 8)', () => {
  assert.deepEqual(AO_QUALITY_LEVELS[3], { slices: 4, steps: 8 });
  const off = engine();
  assert.deepEqual(applyRenderOptions(off, { aoQuality: 0 }), { applied: ['aoQuality'], skipped: [] });
  assert.deepEqual(off.calls, [['setPassEnabled', 'ao.gtao', false]]);
  const low = engine();
  applyRenderOptions(low, { aoQuality: 1 });
  assert.deepEqual(low.calls, [['setPassEnabled', 'ao.gtao', true], ['setAo', { slices: 2, steps: 4 }]]);
  const bad = engine();
  const r = applyRenderOptions(bad, { aoQuality: 9 });
  assert.equal(r.skipped[0].key, 'aoQuality');
  assert.deepEqual(bad.calls, []);
});

test('post toggles only turn a pass off, never on against the scene look', () => {
  const e = engine();
  applyRenderOptions(e, { postBloom: true, postGrading: false }, { bloom: false, grading: true });
  assert.deepEqual(e.calls, [['setPassEnabled', 'bloom', false], ['setPassEnabled', 'grading', false]]);
  const on = engine();
  applyRenderOptions(on, { postBloom: true, postGrading: true });
  assert.deepEqual(on.calls, [['setPassEnabled', 'bloom', true], ['setPassEnabled', 'grading', true]]);
});

test('render scale and post strength go to their engine calls', () => {
  const e = engine();
  const r = applyRenderOptions(e, { renderScale: 0.75, postStrength: 0.5 });
  assert.deepEqual(r.applied, ['renderScale', 'postStrength']);
  assert.deepEqual(e.calls, [['setRenderScale', 0.75], ['setPostStrength', 0.5]]);
});

test('options without an engine side, and calls an older engine lacks, are reported, not faked', () => {
  const old = engine(['setPassEnabled', 'setAo']);
  const r = applyRenderOptions(old, { renderScale: 0.5, postStrength: 1, shading: 'field', lodOffset: 1, frameRate: 'cap60', mystery: 1 });
  assert.deepEqual(r.applied, []);
  assert.deepEqual(r.skipped.map((s) => s.key), ['renderScale', 'postStrength', 'shading', 'lodOffset', 'frameRate', 'mystery']);
  assert.match(r.skipped.find((s) => s.key === 'lodOffset').reason, /LOD/);
  assert.deepEqual(old.calls, []);
});
