import test from 'node:test';
import assert from 'node:assert/strict';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import { createReadiness } from './lib/readiness.js';
import { discoverScenes, attachSkyPage } from './main.js';
import { buildSkySample, sampleSelection } from './lib/scene.js';
import * as foundation from './scenes/foundation.js';

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

const foundationVariant = foundation.default.variants[0];
const foundationIndex = [{ id: foundation.default.id, milestone: 'foundation', file: 'scenes/foundation.js' }];
const indexResponse = value => async () => ({ ok: true, json: async () => value });

test('foundation inventory selects the real sample without executing its builder', async () => {
  let imported = '';
  const scenes = await discoverScenes({ fetchIndex: indexResponse(foundationIndex), importScene: async file => {
    imported = file; return foundation;
  } });
  assert.equal(imported, 'scenes/foundation.js');
  assert.equal(scenes.get('foundation'), foundation);
  for (const key of ['default', 'inventory', 'factNames', 'observe']) assert.equal(scenes.get('foundation')[key], foundation[key]);
  const rows = validateInventory([...scenes.values()], 'foundation');
  assert.equal(rows[0].filename, 'foundation-sky-ground-and-reflective-sphere.png');
  assert.equal(rows[0].expectedVisibleBehavior, foundationVariant.description);
  assert.equal(foundationVariant.judge, true);
  assert.deepEqual(foundationVariant.references, []);
  assert.equal(foundationVariant.selection.subjects.length, 6);
});

test('discovery rejects broken indexes and namespaces before attaching a runnable producer', async () => {
  for (const index of [null, [], [{ ...foundationIndex[0], file: '../outside.js' }],
    [{ ...foundationIndex[0], file: 'scenes/../foundation.js' }], [...foundationIndex, ...foundationIndex],
    [{ ...foundationIndex[0], id: 'different' }], [{ ...foundationIndex[0], milestone: 'weather' }]]) {
    await assert.rejects(discoverScenes({ fetchIndex: indexResponse(index), importScene: async () => foundation }), /scenes.json/);
  }
  for (const change of [{ observe: undefined }, { factNames: [] }, { inventory: [{ ...foundation.inventory[0], requiredFacts: ['sky.intensity'] }] }]) {
    await assert.rejects(discoverScenes({ fetchIndex: indexResponse(foundationIndex), importScene: async () => ({ ...foundation, ...change }) }));
  }
  await assert.rejects(discoverScenes({ fetchIndex: async () => ({ ok: false, status: 503 }) }), /HTTP 503/);
  const cause = new Error('originating scene import failure');
  await assert.rejects(discoverScenes({ fetchIndex: indexResponse(foundationIndex), importScene: async () => { throw cause; } }), error => error === cause);
});

test('discovery consumes every delivered scene generically from the built index', async () => {
  const later = scene('atmosphere');
  const index = [{ id: later.default.id, milestone: 'atmosphere', file: 'scenes/delivered.js' }, ...foundationIndex];
  const scenes = await discoverScenes({ fetchIndex: indexResponse(index),
    importScene: async file => file === 'scenes/delivered.js' ? later : foundation });
  assert.deepEqual([...scenes.keys()], ['atmosphere-scene', 'foundation']);
  assert.equal(scenes.get('atmosphere-scene').observe, later.observe);
  assert.equal(validateInventory([...scenes.values()], 'atmosphere').length, 2);
});

function sampleEngine() {
  const meshes = new Map(), materials = new Map(), nodes = new Map(), instances = [];
  const engine = {
    meshes, materials, nodes, instances,
    setCamera(desc) { this.camera = desc; },
    primitiveMesh(kind, params) { const id = meshes.size + 1; meshes.set(id, { kind, params }); return id; },
    addMaterial(desc) { const id = materials.size + 1; materials.set(id, desc); return id; },
    addNode(parent, t, r, s) { const id = nodes.size + 1; nodes.set(id, { parent, t, r, s }); return id; },
    addInstance(node, mesh, material) { instances.push({ node, mesh, material }); },
  };
  return engine;
}

test('sample geometry and selection share actual metre placements, materials and camera', async () => {
  const engine = sampleEngine();
  // No sky/time/weather setter exists on this engine: the helper must only build geometry.
  await buildSkySample(engine, {}, foundationVariant.selection.camera);
  const { aspect, ...expectedCamera } = foundationVariant.selection.camera;
  assert.deepEqual(engine.camera, expectedCamera);
  assert.equal(aspect, 16 / 9);
  for (const [i, instance] of engine.instances.entries()) {
    const mesh = engine.meshes.get(instance.mesh), node = engine.nodes.get(instance.node);
    const half = mesh.kind === 'sphere' ? Array(3).fill(mesh.params.radius) : mesh.params.halfExtents;
    const subject = foundationVariant.selection.subjects[i];
    assert.deepEqual(subject.min, node.t.map((v, axis) => v - half[axis]));
    assert.deepEqual(subject.max, node.t.map((v, axis) => v + half[axis]));
    assert.deepEqual(node.s, [1, 1, 1]);
    assert.deepEqual(node.r, [0, 0, 0, 1]);
  }
  assert.equal(engine.instances.length, foundationVariant.selection.subjects.length);
  const matte = engine.materials.get(engine.instances[1].material), chrome = engine.materials.get(engine.instances[2].material);
  assert.equal(matte.roughness, 1); assert.equal(matte.metallic, 0);
  assert.equal(chrome.metallic, 1); assert.ok(chrome.roughness < matte.roughness);
  const alias = sampleSelection({ position: [0, 1, 5], target: [0, 1, 0] });
  assert.deepEqual(alias.camera, { ...camera, aspect: 16 / 9, far: 200 });
  alias.subjects[0].min[0] = 123;
  assert.notEqual(sampleSelection(camera).subjects[0].min[0], 123);
});

test('foundation observations use actual getter, pass and submission assertions', async () => {
  const sky = { preset: 'day', intensity: 1.2 };
  const engine = {
    sky: () => sky,
    passes: () => ['sky.env', 'sky.background'].map(name => ({ name, enabled: true, role: 'pass', order: 10, required: false })),
    stats: () => ({ frames: 31, instances: 6, triangles: 4096 }),
  };
  const observation = await foundation.observe(engine, foundationVariant);
  assert.equal(observation.sky, sky);
  assert.deepEqual([observation.time, observation.weather, observation.diagnostics], [null, null, null]);
  assert.deepEqual(observation.edits, []);
  const runtime = { passes: engine.passes(), viewSizes: [{ view: 0, width: 1280, height: 720 }], readiness: { status: 'ready', stage: 'foundation', error: null } };
  validateRequiredFacts(foundation.inventory[0], { ...observation, ...runtime }, foundation.factNames);
  for (const change of [
    { sky: () => ({ intensity: 0 }) }, { passes: () => [] },
    { stats: () => ({ frames: 0, instances: 6, triangles: 4096 }) },
    { stats: () => ({ frames: 31, instances: 5, triangles: 4096 }) },
  ]) {
    const failed = await foundation.observe({ ...engine, ...change }, foundationVariant);
    assert.throws(() => validateRequiredFacts(foundation.inventory[0], { ...failed, ...runtime }, foundation.factNames), /assertion:/);
  }
  const cause = new Error('sky getter failed');
  await assert.rejects(foundation.observe({ ...engine, sky() { throw cause; } }, foundationVariant), error => error === cause);
});

function pageElements() {
  const document = { createElement };
  function createElement(tag) {
    return {
      tag, ownerDocument: document, children: [], textContent: '', value: '', disabled: false, listeners: {},
      appendChild(child) { this.children.push(child); if (tag === 'select' && this.children.length === 1) this.value = child.value; },
      replaceChildren() { this.children = []; },
      addEventListener(name, handler) { this.listeners[name] = handler; },
    };
  }
  return {
    target: { crypto: { randomUUID: () => 'interactive-id' }, listeners: {}, addEventListener(name, handler) { this.listeners[name] = handler; } },
    canvas: createElement('canvas'), status: createElement('div'), picker: createElement('select'),
    controls: createElement('div'), runButton: createElement('button'),
  };
}

test('page keeps discovery pending and then attaches the exact runner operation and namespaces', async () => {
  const page = pageElements();
  let resolve, options;
  const pending = new Promise(r => { resolve = r; });
  const runner = { run: async () => ({}), engine: () => null, dispose() {} };
  const preparation = attachSkyPage({ ...page, discover: () => pending, createRunner: value => { options = value; return runner; } });
  assert.match(page.status.textContent, /^pending:/);
  assert.equal(page.target.nmcGpu.ready, false);
  assert.equal(page.target.nmcGpu.ops.run, undefined);
  assert.equal(page.runButton.disabled, true);
  const namespaces = new Map([['foundation', foundation]]);
  resolve(namespaces);
  assert.equal(await preparation, runner);
  assert.equal(page.target.nmcGpu.ops.run, runner.run);
  assert.equal(options.scenes, namespaces);
  assert.equal(options.scenes.get('foundation').observe, foundation.observe);
  assert.equal(options.canvas, page.canvas); assert.equal(options.status, page.status);
  assert.equal(page.target.nmcGpu.ready, true);
  assert.match(page.status.textContent, /^pending:.*press Run/); // No Engine is ready yet.
  assert.equal(page.picker.children.length, 1);
  page.target.listeners.pagehide();
});

test('page preserves an originating rejected discovery in both visible status and operation', async () => {
  const page = pageElements();
  const cause = Object.assign(new Error('missing delivered namespace'), { code: 'scene.import' });
  await assert.rejects(attachSkyPage({ ...page, discover: async () => { throw cause; } }), error => error === cause);
  assert.equal(page.status.textContent, 'failed: scene.import: missing delivered namespace');
  assert.equal(page.runButton.disabled, true);
  await assert.rejects(page.target.nmcGpu.ops.run({}), error => error === cause);
});

test('direct managed operations populate controls from the runner ready status', async () => {
  const page = pageElements();
  let changed, observed, disconnected = false;
  page.target.MutationObserver = class {
    constructor(callback) { changed = callback; }
    observe(element) { observed = element; }
    disconnect() { disconnected = true; }
  };
  const runner = {
    run: async () => { page.status.textContent = 'ready: foundation/day'; },
    engine: () => ({ sky: () => ({ intensity: 1.2, clouds: { coverage: .25 } }), passes: () => [] }),
    dispose() {},
  };
  await attachSkyPage({ ...page, discover: async () => new Map([['foundation', foundation]]), createRunner: () => runner });
  assert.equal(observed, page.status);
  assert.equal(page.target.nmcGpu.ops.run, runner.run);
  await page.target.nmcGpu.ops.run({});
  changed();
  assert.equal(page.controls.children.length, foundation.default.controls.length);
  const intensity = page.controls.children.flatMap(label => label.children).find(input => input.id === 'control-intensity');
  assert.equal(intensity.value, '1.2');
  page.status.textContent = 'pending: preparing foundation/day'; changed();
  assert.equal(page.controls.children.length, 0);
  page.target.listeners.pagehide(); assert.equal(disconnected, true);
});

test('page controls read accepted values and show originating invalid-edit errors', async () => {
  const page = pageElements();
  let sky = { intensity: 1.2, clouds: { coverage: .25 } }, rendered = 0, disposed = 0;
  const calls = [];
  const engine = {
    sky: () => structuredClone(sky),
    passes: () => ['sky.env', 'sky.background'].map(name => ({ name, enabled: true })),
    setSky(value) {
      if (!Number.isFinite(value.intensity) || value.intensity < 0) throw Object.assign(new Error('intensity must be finite and >= 0'), { code: 'sky.invalid' });
      sky = { ...sky, ...value };
    },
    render() { rendered++; },
  };
  const runner = {
    run: async input => { calls.push(input); page.status.textContent = 'ready: foundation/day'; },
    engine: () => engine, dispose: () => { disposed++; },
  };
  await attachSkyPage({ ...page, discover: async () => new Map([['foundation', foundation]]), readBuild: async () => 'actual-built-id', createRunner: () => runner });
  await page.runButton.listeners.click();
  assert.equal(calls[0].scenario, 'foundation'); assert.equal(calls[0].mode, 'day');
  assert.equal(calls[0].remainingMs, 270000);
  assert.equal(calls[0].buildId, 'actual-built-id');
  const inputs = page.controls.children.map(label => label.children[0]);
  const intensity = inputs.find(input => input.id === 'control-intensity');
  assert.equal(intensity.value, '1.2');
  intensity.value = '-1'; intensity.listeners.change();
  assert.equal(sky.intensity, 1.2); assert.equal(intensity.value, '1.2'); assert.equal(rendered, 0);
  assert.match(page.status.textContent, /failed: sky.invalid: intensity must be finite/);
  intensity.value = ''; intensity.listeners.change();
  assert.equal(sky.intensity, 1.2); assert.equal(intensity.value, '1.2');
  intensity.value = '2'; intensity.listeners.change();
  assert.equal(sky.intensity, 2); assert.equal(intensity.value, '2'); assert.equal(rendered, 1);
  assert.match(page.status.textContent, /^ready:/);
  page.target.listeners.pagehide(); assert.equal(disposed, 1);
});
