import test from 'node:test';
import assert from 'node:assert/strict';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import { createReadiness } from './lib/readiness.js';

const camera = { eye: [0, 1, 5], target: [0, 1, 0], fovYDeg: 60, near: .1, far: 100 };
const selection = { camera, subjects: [{ name: 'ground', min: [-100, -1, -100], max: [100, 0, 100] }, { name: 'sphere', min: [-1, 0, -1], max: [1, 2, 1] }] };
function scene(stage = 'foundation') {
  const id = `${stage}-scene`;
  return { default: { id, milestone: stage, title: 'Scene', build() {}, variants: [{ id: 'default', selection }] },
    factNames: ['sky-visible'], observe: async () => ({}), inventory: [{ id: `${stage}-image`, criterionName: 'Sky', scenario: id, mode: 'default', filename: `${stage}-sky-and-sphere.png`, expectedVisibleBehavior: 'Sky above the sphere', requiredFacts: ['snapshot:sky', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:sky-visible'], pair: null }] };
}
function facts() {
  return { sky: { preset: 'day', zenith: [0, .4, 1], intensity: 1 }, time: null, weather: null, diagnostics: null,
    assertions: [{ name: 'sky-visible', passed: true, detail: 'Observed' }],
    edits: [{ operation: 'invalid edit', accepted: false, error: { code: 'sky.invalid', message: 'invalid value' }, before: { sky: { preset: 'day' }, time: null, weather: null }, after: { sky: { preset: 'day' }, time: null, weather: null } }],
    passes: [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }], viewSizes: [{ view: 0, width: 1280, height: 720 }], readiness: { status: 'ready', stage: 'foundation', error: null } };
}
test('inventory resolves cumulative stages and requires the requested stage', () => {
  const all = [scene(), scene('atmosphere'), scene('weather')];
  assert.deepEqual(validateInventory(all, 'foundation').map(x => x.id), ['foundation-image']);
  assert.equal(validateInventory(all, 'atmosphere').length, 2);
  assert.equal(validateInventory(all, 'weather').length, 3);
  assert.throws(() => validateInventory(all.slice(0, 1), 'atmosphere'), /unavailable/);
  assert.throws(() => validateInventory(all, 'unknown'), /stage/);
});
test('inventory rejects invalid identity, registry, selectors, modes and framing', () => {
  for (const mutate of [
    s => s.inventory[0].filename = '../bad.png', s => s.inventory[0].mode = 'missing',
    s => s.factNames.push('sky-visible'), s => s.inventory[0].requiredFacts.push('assertion:unknown'),
    s => s.inventory[0].requiredFacts.push('snapshot:sky'), s => s.default.variants[0].selection = { camera, subjects: [{ name: 'behind', min: [0, 0, 10], max: [1, 1, 11] }] },
  ]) { const s = structuredCloneScene(); mutate(s); assert.throws(() => validateInventory([s], 'foundation')); }
  const a = scene(), b = scene('atmosphere'); b.inventory[0].filename = a.inventory[0].filename;
  assert.throws(() => validateInventory([a, b], 'atmosphere'));
});
function structuredCloneScene() { const s = scene(); s.default.variants[0].selection = structuredClone(selection); return s; }
test('measurement pairs require both states and matching metadata', () => {
  const s = scene();
  s.default.variants[0].measurement = { pair: 'fog', state: 'on', preset: 'day', features: { fog: true } };
  s.inventory[0].pair = 'fog';
  assert.throws(() => validateInventory([s], 'foundation'), /pair/);
  s.default.variants.push({ id: 'off', selection, measurement: { pair: 'fog', state: 'off', preset: 'day', features: { fog: false } } });
  assert.equal(validateInventory([s], 'foundation').length, 1);
  s.default.variants[1].measurement.preset = 'night';
  assert.throws(() => validateInventory([s], 'foundation'), /pair/);
});
test('required facts validate the complete foundation observation and strict selectors', () => {
  const row = scene().inventory[0], names = ['sky-visible'];
  validateRequiredFacts(row, facts(), names);
  for (const mutate of [
    f => f.sky = null, f => f.sky.intensity = NaN, f => f.passes = [], f => f.viewSizes = [],
    f => f.readiness.status = 'pending', f => f.readiness.error = 'failure',
    f => f.assertions[0].passed = false, f => f.assertions.push({ ...f.assertions[0] }),
    f => delete f.edits[0].after.weather, f => f.diagnostics = { frames: Infinity },
  ]) { const f = facts(); mutate(f); assert.throws(() => validateRequiredFacts(row, f, names), /foundation-image/); }
  for (const key of ['snapshot:time', 'snapshot:weather', 'snapshot:diagnostics', 'sky.intensity', 'assertion:missing']) {
    assert.throws(() => validateRequiredFacts({ ...row, requiredFacts: [key] }, facts(), names), new RegExp(key.replace('.', '\\.')));
  }
  assert.throws(() => validateRequiredFacts(row, facts(), ['sky-visible', 'sky-visible']), /duplicate/);
});
test('pending and rejected preparations stay visibly unready until a new preparation', async () => {
  const readiness = createReadiness('loading');
  let resolve;
  const preparation = new Promise(r => { resolve = r; });
  const pending = preparation.then(() => readiness.ready(), error => readiness.fail('loading', error));
  assert.deepEqual(readiness.snapshot(), { status: 'pending', stage: 'loading', error: null });
  resolve(); await pending;
  assert.equal(readiness.snapshot().status, 'ready');
  readiness.pending('loading assets');
  let reject;
  const rejected = new Promise((_, r) => { reject = r; }).then(() => readiness.ready(), error => readiness.fail('loading assets', error));
  assert.equal(readiness.snapshot().status, 'pending');
  reject(new Error('originating preparation error')); await rejected;
  assert.deepEqual(readiness.snapshot(), { status: 'failed', stage: 'loading assets', error: 'originating preparation error' });
  assert.throws(() => readiness.ready(), /new pending/);
  assert.equal(readiness.snapshot().status, 'failed');
  readiness.pending('retry'); readiness.ready();
  assert.deepEqual(readiness.snapshot(), { status: 'ready', stage: 'retry', error: null });
});
test('later snapshot selectors require complete finite observations', () => {
  const f = facts();
  f.time = { day: 172, hours: 9, latitude: 38, north: 0, enabled: true, playing: false, speed: 1, currentDay: 172, currentHours: 9, sunDirection: [0, 1, 0], moonDirection: [0, -1, 0], moonLit: .5, starPole: [0, 1, 0], starEquinox: [1, 0, 0], dominant: 'sun' };
  const w = { coverage: .3, type: .9, base: 1200, thickness: 1600, density: 1, cirrus: .15, wind: { speed: 7, from: 250 }, evolution: 1, rain: 0 };
  f.weather = { enabled: true, shadows: false, changing: false, seed: 1, transitionSeconds: 0, target: w, current: structuredClone(w), cyclePreset: null, text: '' };
  f.diagnostics = { submittedFrames: 1, views: [{ view: 0, ready: true }] };
  const row = { id: 'later-image', requiredFacts: ['snapshot:time', 'snapshot:weather', 'snapshot:diagnostics'] };
  validateRequiredFacts(row, f, []);
  f.weather.current.wind.speed = Infinity;
  assert.throws(() => validateRequiredFacts(row, f, []), /later-image.*snapshot:weather/);
  f.weather.current.wind.speed = 7; f.diagnostics.submittedFrames = Number.MAX_SAFE_INTEGER + 1;
  assert.throws(() => validateRequiredFacts(row, f, []), /snapshot:diagnostics/);
});
