// Contract checks of the post page (S49, module post-page): the inventory and fact validators on fixtures, and the
// runtime (lib/runtime.js) on stub engines. The real exposure scene namespace is checked by examples/post/exposure.test.mjs.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SETTING_ORDER, validateInventory, validateRequiredFacts } from './lib/inventory.js';
import { SETTLE_FRAME_MS, advancingRender, applyVariantSettings, captureDisplayed, createPostPageRunner, waitExposureSettled } from './lib/runtime.js';

// A camera at the origin looking down -Z, and a 1 m box five metres in front of it (inside the frustum).
const camera = { eye: [0, 0, 0], target: [0, 0, -1], fovYDeg: 60, near: 0.1, far: 50 };
const selection = { camera, subjects: [{ name: 'box', min: [-0.5, -0.5, -5.5], max: [0.5, 0.5, -4.5] }] };
const settings = { toneMapping: { exposureEv: 0 }, passes: { tonemap: true } };
const on = { pair: 'demo', state: 'on', preset: 'p', features: { exposure: true } };
const off = { pair: 'demo', state: 'off', preset: 'p', features: { exposure: false } };
const onVariant = (overrides = {}) => ({ id: 'demo-on', selection, settings, measurement: on, ...overrides });
const offVariant = () => ({ id: 'demo-off', selection, settings, measurement: off });
const names = ['exposure-settled', 'exposure-ev'];

const onRow = {
  id: 'fixture-room-demo-on',
  criterionName: 'exposure',
  scenario: 'fixture-room',
  mode: 'demo-on',
  filename: 'fixture-room-demo-on.png',
  expectedVisibleBehavior: 'The room is readable.',
  requiredFacts: ['runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'fact:exposure-settled', 'fact:exposure-ev', 'assertion:exposure-settled'],
  pair: 'demo',
  heldConstant: ['room geometry'],
  producer: 'fixture',
};
const offRow = { ...onRow, id: 'fixture-room-demo-off', mode: 'demo-off', filename: 'fixture-room-demo-off.png', expectedVisibleBehavior: 'The room is dark.' };

// A scene namespace (default, inventory, factNames, observe) with the fixture room unless overridden.
function namespace({ id = 'fixture-room', milestone = 'exposure', variants = [onVariant(), offVariant()], inventory = [onRow, offRow], factNames = names } = {}) {
  return {
    default: { id, milestone, title: 'Fixture room', variants, build: async () => ({}) },
    inventory,
    factNames,
    observe: async () => ({ values: {}, assertions: [] }),
  };
}

test('validateInventory returns the rows of the stage in namespace then inventory order', () => {
  assert.deepEqual(validateInventory([namespace()], 'exposure'), [onRow, offRow]);
});

test('an unknown stage is refused', () => {
  assert.throws(() => validateInventory([namespace()], 'blur'), { message: 'invalid stage or scenes: blur' });
});

test('a stage with no scene of that milestone is unavailable', () => {
  assert.throws(() => validateInventory([namespace()], 'sharpen'), { message: 'stage sharpen is unavailable' });
});

test('duplicate scene, variant, row and filename are refused, and a bad filename too', () => {
  assert.throws(() => validateInventory([namespace(), namespace()], 'exposure'), { message: 'invalid or duplicate scene fixture-room' });
  assert.throws(() => validateInventory([namespace({ variants: [onVariant(), onVariant()] })], 'exposure'), { message: 'fixture-room: invalid or duplicate variant' });
  assert.throws(() => validateInventory([namespace({ inventory: [onRow, { ...offRow, id: onRow.id }] })], 'exposure'), { message: 'fixture-room: invalid inventory row fixture-room-demo-on' });
  assert.throws(() => validateInventory([namespace({ inventory: [onRow, { ...offRow, filename: onRow.filename }] })], 'exposure'), { message: 'fixture-room: invalid inventory row fixture-room-demo-off' });
  assert.throws(() => validateInventory([namespace({ inventory: [{ ...onRow, filename: 'Bad.png' }, offRow] })], 'exposure'), { message: 'fixture-room: invalid inventory row fixture-room-demo-on' });
});

test('a measurement state that does not match the variant suffix is refused', () => {
  const variants = [onVariant({ measurement: off }), offVariant()];
  assert.throws(() => validateInventory([namespace({ variants })], 'exposure'), { message: 'fixture-room/demo-on: invalid measurement or variant name' });
});

test('a measurement pair with only an on state is refused', () => {
  assert.throws(() => validateInventory([namespace({ variants: [onVariant()], inventory: [onRow] })], 'exposure'), { message: 'measurement pair demo requires on and off states with the same preset' });
});

test('a subject behind the camera is refused', () => {
  const behind = { camera, subjects: [{ name: 'box', min: [-0.5, -0.5, 4.5], max: [0.5, 0.5, 5.5] }] };
  const variants = [onVariant({ selection: behind }), offVariant()];
  assert.throws(() => validateInventory([namespace({ variants })], 'exposure'), { message: 'fixture-room/demo-on: subject box outside camera frustum' });
});

test('unknown settings, a non-boolean settle and an unknown selector are refused', () => {
  assert.throws(() => validateInventory([namespace({ variants: [onVariant({ settings: { blur: 1 } }), offVariant()] })], 'exposure'), { message: 'fixture-room/demo-on: unknown setting blur' });
  assert.throws(() => validateInventory([namespace({ variants: [onVariant({ settle: 'yes' }), offVariant()] })], 'exposure'), { message: 'fixture-room/demo-on: settle must be a boolean' });
  assert.throws(() => validateInventory([namespace({ inventory: [{ ...onRow, requiredFacts: ['snapshot:x'] }, offRow] })], 'exposure'), { message: 'fixture-room-demo-on: unknown selector snapshot:x' });
});

test('a scene of a later milestone is skipped at an earlier stage and included at its own', () => {
  const sharp = namespace({
    id: 'fixture-sharp',
    milestone: 'sharpen',
    variants: [{ id: 'plain', selection, settings }],
    inventory: [{ ...onRow, id: 'fixture-sharp-plain', scenario: 'fixture-sharp', mode: 'plain', filename: 'fixture-sharp-plain.png', pair: null, requiredFacts: [] }],
    factNames: [],
  });
  assert.deepEqual(validateInventory([namespace(), sharp], 'exposure').map(row => row.id), ['fixture-room-demo-on', 'fixture-room-demo-off']);
  assert.deepEqual(validateInventory([namespace(), sharp], 'sharpen').map(row => row.id), ['fixture-room-demo-on', 'fixture-room-demo-off', 'fixture-sharp-plain']);
});

test('SETTING_ORDER is the rank table of the settings', () => {
  assert.deepEqual(SETTING_ORDER, ['toneMapping', 'passes', 'renderScale', 'autoExposure', 'sharpening']);
});

// Facts of a passing exposure run; each refusal below changes one field of it.
const facts = {
  values: { 'exposure-settled': true, 'exposure-ev': null },
  assertions: [{ name: 'exposure-settled', passed: true, detail: '' }],
  passes: [{ name: 'tonemap', role: 'pass', enabled: true, order: 200, required: true }],
  viewSizes: [{ view: 0, width: 1280, height: 720 }],
  readiness: { status: 'ready', stage: 'exposure', error: null },
};

test('validateRequiredFacts accepts the facts of a passing row', () => {
  assert.equal(validateRequiredFacts(onRow, facts, names), undefined);
});

test('validateRequiredFacts refuses a missing value', () => {
  const missing = { ...facts, values: { 'exposure-settled': true } };
  assert.throws(() => validateRequiredFacts(onRow, missing, names), { message: 'fixture-room-demo-on: required fact:exposure-ev is missing or invalid' });
});

test('validateRequiredFacts refuses a false assertion with its detail', () => {
  const failing = { ...facts, assertions: [{ name: 'exposure-settled', passed: false, detail: 'still moving by 0.4 EV' }] };
  assert.throws(() => validateRequiredFacts(onRow, failing, names), { message: 'fixture-room-demo-on: required assertion:exposure-settled is missing or invalid: still moving by 0.4 EV' });
});

test('validateRequiredFacts refuses a failed readiness', () => {
  const broken = { ...facts, readiness: { status: 'failed', stage: 'exposure', error: 'device lost' } };
  assert.throws(() => validateRequiredFacts(onRow, broken, names), { message: 'fixture-room-demo-on: required runtime:ready is missing or invalid' });
});

test('validateRequiredFacts refuses a facts object without values', () => {
  const noValues = { ...facts };
  delete noValues.values;
  assert.throws(() => validateRequiredFacts(onRow, noValues, names), { message: 'fixture-room-demo-on: invalid PostObservation' });
});

// Readiness gate (waitExposureSettled) on an engine double: autoExposureState answers the given states in turn (the
// last one repeats), and each render is counted.
const PENDING = { pending: true, settled: false, exposureEv: null, averageLuminance: null };
const UNSETTLED = { pending: false, settled: false, exposureEv: 0.5, averageLuminance: 0.2 };
const SETTLED = { pending: false, settled: true, exposureEv: 1, averageLuminance: 0.18 };
const budgetOf = remainingMs => ({ remainingMs: () => remainingMs });
function exposureDouble(states) {
  const double = { renders: 0, reads: 0 };
  double.render = async () => { double.renders += 1; };
  double.engine = {
    autoExposureState() {
      const state = states[Math.min(double.reads, states.length - 1)];
      double.reads += 1;
      if (state instanceof Error) throw state;
      return state;
    },
  };
  return double;
}

test('waitExposureSettled is ready on the second settled read, after exactly 6 renders', async () => {
  const double = exposureDouble([PENDING, UNSETTLED, SETTLED, UNSETTLED, SETTLED, SETTLED]);
  const verdict = await waitExposureSettled(double.engine, double.render, budgetOf(Infinity));
  assert.deepEqual(verdict, { status: 'ready', state: SETTLED, frames: 6, error: null });
  assert.equal(double.renders, 6);
});

test('a never-settled exposure fails after 600 renders and never reports ready', async () => {
  const double = exposureDouble([UNSETTLED]);
  const verdict = await waitExposureSettled(double.engine, double.render, budgetOf(Infinity));
  assert.deepEqual(verdict, { status: 'failed', state: UNSETTLED, frames: 600, error: 'exposure did not settle in 600 frames' });
  assert.equal(double.renders, 600);
});

test('a pending exposure keeps rendering and is not ready until a settled pair arrives', async () => {
  const stuck = exposureDouble([PENDING]);
  const failed = await waitExposureSettled(stuck.engine, stuck.render, budgetOf(Infinity), 25);
  assert.deepEqual([failed.status, failed.error, stuck.renders], ['failed', 'exposure did not settle in 25 frames', 25]);
  const late = exposureDouble([PENDING, PENDING, PENDING, PENDING, PENDING, SETTLED, SETTLED]);
  const verdict = await waitExposureSettled(late.engine, late.render, budgetOf(Infinity));
  assert.deepEqual([verdict.status, verdict.frames, late.renders], ['ready', 7, 7]);
});

test('a throwing autoExposureState fails with the described error', async () => {
  const broken = exposureDouble([new Error('device lost')]);
  const verdict = await waitExposureSettled(broken.engine, broken.render, budgetOf(Infinity));
  assert.deepEqual([verdict.status, verdict.error], ['failed', 'device lost']);
});

test('a budget with no time left fails before the first render', async () => {
  const double = exposureDouble([SETTLED]);
  const verdict = await waitExposureSettled(double.engine, double.render, budgetOf(0));
  assert.deepEqual([verdict.status, verdict.error, double.renders], ['failed', 'exposure: not settled before the operation deadline', 0]);
});

test('applyVariantSettings applies the keys in SETTING_ORDER and refuses an unknown key or a missing method', () => {
  const calls = [];
  const engine = {
    setToneMapping: value => calls.push(['setToneMapping', value]),
    setPassEnabled: (name, enabled) => calls.push(['setPassEnabled', name, enabled]),
    setAutoExposure: value => calls.push(['setAutoExposure', value]),
  };
  applyVariantSettings(engine, { autoExposure: { enabled: true }, toneMapping: { exposureEv: 1 }, passes: { taa: true } });
  assert.deepEqual(calls, [
    ['setToneMapping', { exposureEv: 1 }],
    ['setPassEnabled', 'taa', true],
    ['setAutoExposure', { enabled: true }],
  ]);
  assert.throws(() => applyVariantSettings(engine, { toneMapping: { exposureEv: 2 }, blur: 1 }), { message: 'post runtime: unknown setting blur' });
  assert.throws(() => applyVariantSettings(engine, { sharpening: { enabled: true } }), { message: 'post runtime: engine has no setSharpening' });
  assert.equal(calls.length, 3);
});

test('captureDisplayed captures a 1280 x 720 canvas as it is, and a smaller one through a 1280 x 720 proxy', async () => {
  const captured = [];
  const capture = async (canvas, status) => { captured.push([canvas, status]); return 'PNG'; };
  const status = { textContent: '' };
  const full = { width: 1280, height: 720 };
  assert.equal(await captureDisplayed(full, status, { capture }), 'PNG');
  assert.equal(captured[0][0], full);
  const draws = [];
  const context = { imageSmoothingEnabled: false, drawImage: (...args) => draws.push([args, context.imageSmoothingEnabled]) };
  const proxy = { width: 0, height: 0, getContext: kind => (kind === '2d' ? context : null) };
  const small = { width: 858, height: 482, ownerDocument: { createElement: tag => (tag === 'canvas' ? proxy : null) } };
  await captureDisplayed(small, status, { capture });
  assert.deepEqual(draws, [[[small, 0, 0, 1280, 720], true]]);
  assert.equal(captured[1][0], proxy);
  assert.deepEqual([proxy.width, proxy.height], [1280, 720]);
});

test('createPostPageRunner refuses a missing canvas before any clock is read', () => {
  assert.throws(() => createPostPageRunner({ status: { textContent: '' }, load: async () => ({}), scenes: new Map(), lib: {} }), { message: 'post runtime: options.canvas is required' });
});

test('advancingRender renders settle frame k at the variant time plus (k + 1) frame steps, after scene.frame', async () => {
  const calls = [];
  const clock = { now: () => 0, requestFrame: callback => { callback(0); return 1; }, cancelFrame() {} };
  const engine = { render: timeMs => calls.push(['render', timeMs]) };
  const scene = { frame: (target, timeMs) => calls.push(['frame', timeMs]) };
  const render = advancingRender({ engine, scene, variant: { id: 'on', timeMs: 1000 }, state: {}, clock });
  for (const k of [0, 1, 2]) await render(k);
  assert.deepEqual(calls.map(([kind]) => kind), ['frame', 'render', 'frame', 'render', 'frame', 'render']);
  const times = calls.filter(([kind]) => kind === 'render').map(([, timeMs]) => timeMs);
  [1, 2, 3].forEach((steps, i) => assert.ok(Math.abs(times[i] - (1000 + steps * (1000 / 60))) < 1e-9));
  assert.ok(1000 < times[0] && times[0] < times[1] && times[1] < times[2]);
});

test('createPostPageRunner applies the settings after the build, then settles on advancing frames', async () => {
  const log = [];
  const times = [];
  const clock = { now: () => 0, requestFrame: callback => { queueMicrotask(() => callback(0)); return 1; }, cancelFrame() {} };
  const engine = {
    setToneMapping: value => log.push(['setToneMapping', value.exposureEv]),
    setAutoExposure: value => log.push(['setAutoExposure', value.enabled]),
    autoExposureState: () => ({ ...UNSETTLED }),
    render: timeMs => times.push(timeMs),
    resize() {},
    stats: () => ({ hardware: false, adapter: {} }),
    passes: () => [],
    dispose() {},
  };
  const scene = {
    id: 'probe',
    milestone: 'exposure',
    title: 'Probe',
    variants: [{ id: 'on', settle: true, settings: { toneMapping: { exposureEv: 1 }, autoExposure: { enabled: true } } }],
    build: async () => { log.push(['build']); return {}; },
  };
  const row = { id: 'probe-on', scenario: 'probe', mode: 'on', filename: 'probe-on.png', expectedVisibleBehavior: 'Probe.', requiredFacts: [], heldConstant: [] };
  const scenes = new Map([['probe', { default: scene, inventory: [row], factNames: [], observe: async () => ({ values: {}, assertions: [] }) }]]);
  const runner = createPostPageRunner({ canvas: { width: 0, height: 0 }, status: { textContent: '' }, load: async () => ({ createEngine: async () => engine }), scenes, lib: {}, clock });
  const input = { selectionId: 's-1', attemptId: 'a-1', buildId: 'b-1', scenario: 'probe', mode: 'on', width: 1280, height: 720, remainingMs: 300000 };
  await assert.rejects(runner.run(input), { message: 'post runtime: readiness: exposure did not settle in 600 frames' });
  runner.dispose();
  assert.deepEqual(log, [['build'], ['setToneMapping', 1], ['setAutoExposure', true]]);
  assert.ok(times.slice(0, 30).every(timeMs => timeMs === 1000));
  const settle = times.slice(30);
  assert.equal(settle.length, 600);
  assert.ok(settle.every((timeMs, k) => Math.abs(timeMs - (1000 + (k + 1) * SETTLE_FRAME_MS)) < 1e-9));
});

// Page entry (main.js) discovery and the managed-page declaration.
const { discoverScenes } = await import('./main.js');
const exposureScene = await import('./scenes/exposure.js');
const indexOf = rows => () => Promise.resolve({ ok: true, json: () => Promise.resolve(rows) });
const exposureRow = { id: 'exposure-room', milestone: 'exposure', file: 'scenes/exposure.js' };

test('discoverScenes keys the real exposure scene by its id', async () => {
  const scenes = await discoverScenes({ fetchIndex: indexOf([exposureRow]), importScene: async () => exposureScene });
  assert.deepEqual([...scenes.keys()], ['exposure-room']);
});

test('discoverScenes refuses an unknown milestone and unsorted ids', async () => {
  const importScene = async () => exposureScene;
  await assert.rejects(discoverScenes({ fetchIndex: indexOf([{ ...exposureRow, milestone: 'blur' }]), importScene }), /invalid/);
  const pair = [exposureRow, { ...exposureRow, id: 'a-room' }];
  await assert.rejects(discoverScenes({ fetchIndex: indexOf(pair), importScene }), /unsorted/);
});

test('discoverScenes refuses a namespace whose default.id differs', async () => {
  const importScene = async () => ({ ...exposureScene, default: { ...exposureScene.default, id: 'other-room' } });
  await assert.rejects(discoverScenes({ fetchIndex: indexOf([exposureRow]), importScene }), /namespace identity mismatch/);
});

test('nmc.gpu.json declares the run op of index.html', async () => {
  const { readFile } = await import('node:fs/promises');
  const text = await readFile(new URL('./nmc.gpu.json', import.meta.url), 'utf8');
  assert.deepEqual(JSON.parse(text), { page: 'index.html', ops: ['run'] });
});
