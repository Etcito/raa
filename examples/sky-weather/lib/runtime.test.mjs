import test from 'node:test';
import assert from 'node:assert/strict';
import { createOperationBudget, createSkyPageRunner } from './runtime.js';
import { captureCurrent } from './capture.js';

const SELECTION = {
  camera: { eye: [0, 1, 5], target: [0, 1, 0], fovYDeg: 60, near: .1, far: 100 },
  subjects: [
    { name: 'ground', min: [-100, -1, -100], max: [100, 0, 100] },
    { name: 'sphere', min: [-1, 0, -1], max: [1, 2, 1] },
  ],
};
const ROW = {
  id: 'foundation-image', criterionName: 'Sky', scenario: 'foundation-scene', mode: 'default',
  filename: 'foundation-sky-and-sphere.png', expectedVisibleBehavior: 'Blue sky above the sphere',
  requiredFacts: ['snapshot:sky', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:sky-visible'],
  pair: null,
};
const ADAPTER = { vendor: 'NVIDIA', architecture: 'Ada', device: 'RTX 3090', description: 'managed', isFallbackAdapter: false };

function observation() {
  return {
    sky: { preset: 'day' },
    time: null,
    weather: null,
    diagnostics: null,
    assertions: [{ name: 'sky-visible', passed: true, detail: 'Sky is visible' }],
    edits: [],
  };
}

function sceneNamespace(overrides = {}) {
  const variants = overrides.variants ?? [{ id: 'default', selection: structuredClone(SELECTION) }];
  const scene = {
    id: 'foundation-scene', milestone: 'foundation', title: 'Foundation',
    variants,
    build: overrides.build ?? (async () => {}),
  };
  if (overrides.frame) scene.frame = overrides.frame;
  return {
    default: scene,
    inventory: overrides.inventory ?? [structuredClone(ROW)],
    factNames: overrides.factNames ?? ['sky-visible'],
    observe: overrides.observe ?? (async () => observation()),
  };
}

function measuredNamespace(overrides = {}) {
  const variants = [
    { id: 'default', selection: structuredClone(SELECTION), measurement: { pair: 'fog', state: 'on', preset: 'day', features: { fog: true } } },
    { id: 'fog-off', selection: structuredClone(SELECTION), measurement: { pair: 'fog', state: 'off', preset: 'day', features: { fog: false } } },
  ];
  return sceneNamespace({ ...overrides, variants, inventory: overrides.inventory ?? [{ ...structuredClone(ROW), pair: 'fog' }] });
}

function input(overrides = {}) {
  return {
    selectionId: 'selection-1', attemptId: 'attempt-1', buildId: 'build-1',
    scenario: 'foundation-scene', mode: 'default', width: 1280, height: 720, remainingMs: 10000,
    ...overrides,
  };
}

function installClock(t, { start = 1000, step = 1 } = {}) {
  const state = { now: start, step, frames: 0, handles: 0 };
  const originalPerformance = globalThis.performance;
  const originalFrame = globalThis.requestAnimationFrame;
  const originalCancel = globalThis.cancelAnimationFrame;
  globalThis.performance = { now: () => state.now };
  globalThis.requestAnimationFrame = callback => {
    state.handles += 1;
    state.now += state.step;
    state.frames += 1;
    callback(state.now);
    return state.handles;
  };
  globalThis.cancelAnimationFrame = () => {};
  t.after(() => {
    globalThis.performance = originalPerformance;
    if (originalFrame === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = originalFrame;
    if (originalCancel === undefined) delete globalThis.cancelAnimationFrame;
    else globalThis.cancelAnimationFrame = originalCancel;
  });
  return state;
}

function createCanvas(width = 1280, height = 720) {
  const record = { drawImage: [], fillRect: [], text: [] };
  const context = {
    font: '', fillStyle: '', textBaseline: '',
    fillRect(...args) { record.fillRect.push(args); },
    drawImage(...args) { record.drawImage.push(args); },
    fillText(text, x, y) { record.text.push({ text, x, y }); },
    measureText(text) { return { width: text.length * 8 }; },
  };
  const output = {
    width: 0,
    height: 0,
    getContext() { return context; },
    toDataURL() { return `data:image/png;base64,${Buffer.from(`${this.width}x${this.height}`).toString('base64')}`; },
  };
  const canvas = {
    width,
    height,
    style: {},
    ownerDocument: { createElement(tag) { assert.equal(tag, 'canvas'); return output; } },
  };
  return { canvas, output, record, context };
}

function createStatus(text = '') {
  return { textContent: text, getBoundingClientRect: () => ({ width: 1280, height: 20 }) };
}

function engineFactory({ viewports = [{ x: 0, y: 0, width: 1, height: 1 }] } = {}) {
  const engines = [];
  function create(options = {}) {
    const state = { disposed: 0, renders: [], resizes: [], options, viewports };
    const engine = {
      resize(width, height) { state.resizes.push([width, height]); },
      render(timeMs) { state.renders.push(timeMs); },
      passes() { return [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }]; },
      stats() { return { adapter: ADAPTER, hardware: true }; },
      setPassEnabled(name, enabled) { (state.passes ??= []).push([name, enabled]); },
      setPassOrder(order) { state.order = order; },
      viewports() { return state.viewports; },
      probeNonEmpty: async () => true,
      dispose() { state.disposed += 1; },
    };
    state.engine = engine;
    engines.push(state);
    return engine;
  }
  return { engines, create };
}

function createRunner(t, config = {}) {
  const clock = installClock(t, { start: config.start ?? 1000, step: config.step ?? 1 });
  const { canvas, output, record } = createCanvas(config.width ?? 1280, config.height ?? 720);
  const status = createStatus('');
  const factory = engineFactory({ viewports: config.viewports });
  const namespace = config.namespace ?? sceneNamespace();
  const scenes = config.scenes ?? new Map([[namespace.default.id, namespace]]);
  const load = config.load ?? (async () => ({
    createEngine: async (element, options) => { factory.onCreate?.(); return factory.create(options); },
  }));
  const runner = createSkyPageRunner({ canvas, status, load, scenes, lib: { marker: true } });
  t.after(() => runner.dispose());
  return { runner, clock, canvas, output, record, status, factory, scenes, namespace };
}

test('createOperationBudget fixes one absolute monotonic deadline and clamps the remainder', () => {
  const state = { now: 500 };
  const clock = { now: () => state.now };
  const budget = createOperationBudget(1000, clock);
  assert.equal(budget.deadlineMs, 1500);
  assert.equal(budget.remainingMs(), 1000);
  state.now = 1200;
  assert.equal(budget.remainingMs(), 300);
  state.now = 1500;
  assert.equal(budget.remainingMs(), 0);
  state.now = 2500;
  assert.equal(budget.remainingMs(), 0);
  state.now = NaN;
  assert.equal(budget.remainingMs(), 0);
  for (const bad of [0, -1, NaN, Infinity, '100']) {
    assert.throws(() => createOperationBudget(bad, clock), /remainingMs/);
  }
  assert.throws(() => createOperationBudget(10, null), /clock/);
  assert.throws(() => createOperationBudget(10, { now: () => NaN }), /finite millisecond/);
});

test('run validates the amended input before preparing a scene', async t => {
  const setup = createRunner(t);
  const bad = [
    null,
    { ...input(), selectionId: '' },
    { ...input(), attemptId: ' ' },
    { ...input(), buildId: undefined },
    { ...input(), scenario: '' },
    { ...input(), mode: null },
    { ...input(), width: 0 },
    { ...input(), height: 1.5 },
    { ...input(), width: '1280' },
    { ...input(), remainingMs: 0 },
    { ...input(), remainingMs: -5 },
    { ...input(), remainingMs: NaN },
    { ...input(), remainingMs: 300001 },
  ];
  for (const value of bad) {
    await assert.rejects(setup.runner.run(value), /sky page runner: input/);
  }
  assert.equal(setup.factory.engines.length, 0);
  assert.match(setup.status.textContent, /^failed: sky page runner: input/);
});

test('runs the selected scene with actual passes, view sizes, readiness and a captured image', async t => {
  const setup = createRunner(t);
  let pendingDuringBuild = '';
  setup.namespace.default.build = async () => { pendingDuringBuild = setup.status.textContent; };
  const result = await setup.runner.run(input());

  assert.match(pendingDuringBuild, /^pending: preparing foundation-scene\/default/);
  assert.match(setup.status.textContent, /^ready: foundation-scene\/default/);
  assert.deepEqual(result.provenance, {
    selectionId: 'selection-1', attemptId: 'attempt-1', buildId: 'build-1',
    scenario: 'foundation-scene', mode: 'default',
  });
  assert.equal(result.browser.devicePixelRatio > 0, true);
  assert.equal(typeof result.browser.userAgent, 'string');
  assert.equal(typeof result.browser.platform, 'string');
  assert.deepEqual(result.adapter, ADAPTER);
  assert.equal(result.hardware, true);
  assert.equal(result.images.length, 1);
  assert.deepEqual(Object.keys(result.images[0]).sort(), ['data', 'filename', 'id', 'mime', 'observation']);
  assert.equal(result.images[0].id, 'foundation-image');
  assert.equal(result.images[0].filename, 'foundation-sky-and-sphere.png');
  assert.equal(result.images[0].mime, 'image/png');
  assert.equal(result.images[0].observation, 'Blue sky above the sphere');
  assert.ok(result.images[0].data.length > 0);
  assert.deepEqual(result.facts.passes, [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }]);
  assert.deepEqual(result.facts.viewSizes, [{ view: 0, width: 1280, height: 720 }]);
  assert.deepEqual(result.facts.readiness, { status: 'ready', stage: 'foundation', error: null });
  assert.deepEqual(result.facts.assertions, [{ name: 'sky-visible', passed: true, detail: 'Sky is visible' }]);
  assert.deepEqual(result.facts.sky, { preset: 'day' });
  assert.deepEqual(result.measurements, []);
  assert.deepEqual(result.errors, { console: [], page: [], gpu: [] });

  const engineState = setup.factory.engines[0];
  assert.equal(setup.runner.engine(), engineState.engine);
  assert.deepEqual(engineState.resizes, [[1280, 720]]);
  assert.ok(engineState.renders.length >= 31);
  assert.equal(engineState.renders.at(-1), 1000);
  assert.equal(setup.output.width, 1280);
  assert.ok(setup.output.height > 720);
  assert.match(setup.record.text.map(entry => entry.text).join(' '), /ready: foundation-scene\/default/);
});

test('renders every frame and the capture at the variant time', async t => {
  const variants = [{ id: 'default', selection: structuredClone(SELECTION), timeMs: 2500 }];
  const namespace = sceneNamespace({ variants });
  const setup = createRunner(t, { namespace });
  const frameCalls = [];
  namespace.default.frame = (engine, timeMs, variant) => { frameCalls.push([timeMs, variant.id]); };
  await setup.runner.run(input());
  assert.ok(setup.factory.engines[0].renders.length >= 31);
  assert.ok(setup.factory.engines[0].renders.every(timeMs => timeMs === 2500));
  assert.deepEqual(frameCalls[0], [2500, 'default']);
  assert.equal(frameCalls.length, setup.factory.engines[0].renders.length);
});

test('pending preparation stays visible and failure keeps the originating cause until a new preparation', async t => {
  const setup = createRunner(t);
  let release = null;
  let entered = null;
  const gate = new Promise(resolve => { release = resolve; });
  const enteredPromise = new Promise(resolve => { entered = resolve; });
  const failure = new Error('ground mesh preparation failed');
  let attempts = 0;
  setup.namespace.default.build = async () => {
    attempts += 1;
    if (attempts === 1) {
      entered();
      await gate;
      throw failure;
    }
  };
  const first = setup.runner.run(input());
  await enteredPromise;
  assert.match(setup.status.textContent, /^pending: preparing foundation-scene\/default/);
  release();
  await assert.rejects(first, error => error === failure);
  assert.match(setup.status.textContent, /^failed:/);
  assert.match(setup.status.textContent, /ground mesh preparation failed/);
  assert.equal(setup.runner.engine(), setup.factory.engines[0].engine);

  const second = await setup.runner.run(input());
  assert.match(setup.status.textContent, /^ready:/);
  assert.deepEqual(second.facts.readiness, { status: 'ready', stage: 'foundation', error: null });
});

test('an injected Engine failure rejects visibly and leaves no resident engine', async t => {
  const failure = new Error('adapter request failed');
  const namespace = sceneNamespace();
  const setup = createRunner(t, {
    namespace,
    load: async () => ({ createEngine: async () => { throw failure; } }),
  });
  await assert.rejects(setup.runner.run(input()), error => error === failure);
  assert.equal(setup.runner.engine(), null);
  assert.match(setup.status.textContent, /^failed:/);
  assert.match(setup.status.textContent, /adapter request failed/);
});

test('elapsed setup time is charged to the one budget and expires before observation', async t => {
  const setup = createRunner(t);
  let observed = 0;
  setup.factory.onCreate = () => { setup.clock.now += 200; };
  setup.namespace.observe = async () => { observed += 1; return observation(); };
  await assert.rejects(setup.runner.run(input({ remainingMs: 100 })), /deadline expired during Engine creation/);
  assert.equal(observed, 0);
  assert.equal(setup.record.drawImage.length, 0);
  assert.match(setup.status.textContent, /^failed:/);
  assert.match(setup.status.textContent, /deadline expired/);
});

test('feature measurement keeps the run-entry deadline after setup time elapsed', async t => {
  const setup = createRunner(t, { namespace: measuredNamespace(), step: 6 });
  let observed = 0;
  setup.namespace.default.build = async () => { setup.clock.now += 1500; };
  setup.namespace.observe = async () => { observed += 1; return observation(); };
  await assert.rejects(setup.runner.run(input({ remainingMs: 2000 })), /feature measure: operation deadline expired/);
  assert.equal(observed, 1);
  assert.equal(setup.record.drawImage.length, 0);
  assert.ok(setup.factory.engines[0].renders.length < 330);
  assert.match(setup.status.textContent, /^failed:/);
  assert.match(setup.status.textContent, /deadline expired/);
});

test('collects the selected feature state and preserves its pair metadata', async t => {
  const setup = createRunner(t, { namespace: measuredNamespace() });
  const result = await setup.runner.run(input({ remainingMs: 5000 }));
  assert.equal(result.measurements.length, 1);
  const [measurement] = result.measurements;
  assert.equal(measurement.pair, 'fog');
  assert.equal(measurement.state, 'on');
  assert.equal(measurement.preset, 'day');
  assert.deepEqual(measurement.features, { fog: true });
  assert.equal(measurement.width, 1280);
  assert.equal(measurement.height, 720);
  assert.equal(measurement.warmup, 30);
  assert.equal(measurement.frames, 300);
  assert.equal(measurement.frameTimesMs.length, 300);
  assert.equal(setup.clock.frames, 330);
});

test('rejects concurrent runs while the first run owns the budget', async t => {
  const setup = createRunner(t);
  let release = null;
  let entered = null;
  const gate = new Promise(resolve => { release = resolve; });
  const enteredPromise = new Promise(resolve => { entered = resolve; });
  setup.namespace.default.build = async () => { entered(); await gate; };
  const first = setup.runner.run(input());
  await enteredPromise;
  await assert.rejects(setup.runner.run(input({ attemptId: 'attempt-2' })), /already in progress/);
  release();
  const result = await first;
  assert.equal(result.provenance.attemptId, 'attempt-1');
});

test('disposal is idempotent and clears the resident engine', async t => {
  const setup = createRunner(t);
  await setup.runner.run(input());
  const engineState = setup.factory.engines[0];
  assert.equal(setup.runner.engine(), engineState.engine);
  setup.runner.dispose();
  setup.runner.dispose();
  assert.equal(setup.runner.engine(), null);
  assert.equal(engineState.disposed, 1);
  await assert.rejects(setup.runner.run(input()), /disposed/);
});

test('reports actual engine viewport sizes instead of guessing a single view', async t => {
  const viewports = [{ x: 0, y: 0, width: .5, height: 1 }, { x: .5, y: 0, width: .5, height: 1 }];
  const setup = createRunner(t, { viewports });
  const result = await setup.runner.run(input());
  assert.deepEqual(result.facts.viewSizes, [
    { view: 0, width: 640, height: 720 },
    { view: 1, width: 640, height: 720 },
  ]);
});

test('rejects unknown selections and observations that miss required selectors', async t => {
  const setup = createRunner(t);
  setup.namespace.observe = async () => ({ ...observation(), sky: null });
  await assert.rejects(setup.runner.run(input()), /foundation-image: required snapshot:sky is missing or invalid/);
  assert.equal(setup.record.drawImage.length, 0);
  assert.match(setup.status.textContent, /^failed:/);

  await assert.rejects(setup.runner.run(input({ scenario: 'missing-scene' })), /unknown scenario/);
  await assert.rejects(setup.runner.run(input({ mode: 'missing-mode' })), /has no mode/);
  assert.equal(setup.record.drawImage.length, 0);
  assert.match(setup.status.textContent, /^failed:/);
});

test('captureCurrent composes the actual canvas with readable status text', async () => {
  const { canvas, output, record } = createCanvas(640, 360);
  const status = createStatus('ready: foundation-scene/default');
  status.getBoundingClientRect = () => ({ width: 640, height: 22 });
  const data = await captureCurrent(canvas, status);
  assert.equal(output.width, 640);
  assert.equal(output.height, 394);
  assert.equal(record.drawImage.length, 1);
  assert.equal(record.drawImage[0][0], canvas);
  assert.deepEqual(record.drawImage[0].slice(1), [0, 0, 640, 360]);
  assert.ok(record.text.some(entry => entry.text.includes('ready: foundation-scene/default')));
  assert.equal(data, Buffer.from('640x394').toString('base64'));
});

test('captureCurrent rejects invalid canvases and unusable documents', async () => {
  const status = createStatus('failed: sky.invalid: bad value');
  await assert.rejects(captureCurrent(null, status), /canvas is required/);
  await assert.rejects(captureCurrent({ width: 0, height: 10 }, status), /positive integer/);
  const { canvas } = createCanvas();
  canvas.ownerDocument = null;
  const originalDocument = globalThis.document;
  delete globalThis.document;
  try {
    await assert.rejects(captureCurrent(canvas, status), /document is required/);
  } finally {
    if (originalDocument !== undefined) globalThis.document = originalDocument;
  }
});

// Run 13659 (atmosphere/env-off): readiness and observation are submission facts, so a mode
// whose observation awaits no readback reached the rAF collector while the GPU still drained
// the prepared work, and Chrome delivered no warm-up frame 1 within 10 s. The collector must
// start only after a readback proves the GPU completed every earlier submission.
test('feature measurement starts only after the GPU completed the prepared frames', async t => {
  const setup = createRunner(t, { namespace: measuredNamespace() });
  const order = [];
  let drained = false;
  const frame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = callback => {
    order.push(drained ? 'frame' : 'frame-before-gpu-completion');
    return frame(callback);
  };
  setup.factory.onCreate = () => queueMicrotask(() => {
    const engine = setup.factory.engines.at(-1).engine;
    engine.probeNonEmpty = async () => {
      order.push('probe');
      await new Promise(resolve => setTimeout(resolve, 5));
      drained = true;
      return true;
    };
  });
  const result = await setup.runner.run(input({ remainingMs: 5000 }));
  assert.equal(result.measurements.length, 1);
  assert.equal(order[0], 'probe');
  assert.ok(!order.includes('frame-before-gpu-completion'), 'no animation frame was requested before GPU completion');
  assert.equal(order.filter(entry => entry === 'frame').length, 330);
});

test('a GPU that never completes the prepared frames fails at the operation deadline, not as a missing frame', async t => {
  const setup = createRunner(t, { namespace: measuredNamespace() });
  setup.factory.onCreate = () => queueMicrotask(() => {
    setup.factory.engines.at(-1).engine.probeNonEmpty = () => new Promise(() => {});
  });
  await assert.rejects(setup.runner.run(input({ remainingMs: 200 })),
    /the GPU did not complete the prepared frames within the remaining \d+ ms of the operation deadline/);
  assert.equal(setup.clock.frames, 0);
  assert.equal(setup.record.drawImage.length, 0);
  assert.match(setup.status.textContent, /^failed:/);
});

test('scenes without a feature measurement do not wait for a GPU completion probe', async t => {
  const setup = createRunner(t);
  let probes = 0;
  setup.factory.onCreate = () => queueMicrotask(() => {
    setup.factory.engines.at(-1).engine.probeNonEmpty = async () => { probes += 1; return true; };
  });
  const result = await setup.runner.run(input());
  assert.deepEqual(result.measurements, []);
  assert.equal(probes, 0);
});
