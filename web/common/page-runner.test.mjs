// Runner tests of contract 'page-runner' (web/common/page-runner.js, shares 1 and 2). Node only, no
// browser. Each test builds its own fakes (canvas and status, a frame clock, an Engine double with a
// frame counter and one scene namespace), so no state crosses between tests.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createOperationBudget, createPageRunner, validateInput } from './page-runner.js';

const PNG = 'iVBORw0KGgo=';
const ADAPTER = { vendor: 'NVIDIA', architecture: 'Ada', device: 'RTX 3090', description: 'managed', isFallbackAdapter: false };

// The clock double of contract 'page-runner': requestFrame queues a callback; deliver() advances the time and runs the queued frames.
function frameClock() {
  const callbacks = new Map();
  let time = 0;
  let nextId = 1;
  return {
    now: () => time,
    requestFrame(callback) {
      const id = nextId++;
      callbacks.set(id, callback);
      return id;
    },
    cancelFrame(id) { callbacks.delete(id); },
    waiting: () => callbacks.size,
    deliver(stepMs = 100) {
      time += stepMs;
      const due = [...callbacks.values()];
      callbacks.clear();
      for (const callback of due) callback(time);
    },
  };
}

// A status element that records every text assigned to it.
function statusDouble() {
  const texts = [];
  return { set textContent(text) { texts.push(text); }, get textContent() { return texts.at(-1); }, texts };
}

// The Engine double; `state` is the test's record of its frames (the counter), resizes and disposal.
function engineDouble(state) {
  return {
    resize(width, height) { state.resizes.push([width, height]); },
    render(timeMs) {
      state.frames += 1;
      state.times.push(timeMs);
    },
    stats: () => ({ frames: state.frames, triangles: 12, hardware: true, adapter: ADAPTER }),
    passes: () => [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }],
    dispose() { state.disposed += 1; },
  };
}

// A runner over the given scenes with this file's doubles; `extra` overrides any of its options.
function harness(scenes, extra = {}) {
  const clock = frameClock();
  const canvas = { width: 1280, height: 720, style: {} };
  const status = statusDouble();
  const engines = [];
  const captures = [];
  const load = async () => ({
    createEngine: async () => {
      const state = { frames: 0, times: [], resizes: [], disposed: 0 };
      engines.push(state);
      return engineDouble(state);
    },
  });
  const capture = async (...args) => {
    captures.push(args);
    return PNG;
  };
  const options = {
    topic: 'gi',
    canvas,
    status,
    load,
    scenes,
    lib: {},
    clock,
    capture,
    collectFacts: () => ({ settings: { enabled: true } }),
    validateRequiredFacts: () => {},
    ...extra,
  };
  return { runner: createPageRunner(options), clock, status, engines, captures };
}

// One scene namespace of milestone 'bake' with one inventory row per variant; hooks override the defaults.
function sceneOf(id, hooks = {}) {
  const variants = hooks.variants ?? [{ id: 'default', description: 'Default', references: [], judge: true }];
  return {
    default: { id, milestone: 'bake', title: 'Scene', variants, build: hooks.build ?? (async () => ({})) },
    inventory: variants.map(variant => ({
      id: `${id}-${variant.id}`,
      scenario: id,
      mode: variant.id,
      filename: `${id}-${variant.id}.png`,
      expectedVisibleBehavior: 'Visible',
      requiredFacts: [],
    })),
    factNames: [],
    observe: hooks.observe ?? (async () => ({})),
  };
}

function runInput(scenario, overrides = {}) {
  return { selectionId: 'selection-1', attemptId: 'attempt-1', buildId: 'build-1', scenario, mode: 'default', width: 1280, height: 720, remainingMs: 300000, ...overrides };
}

async function flush() {
  await new Promise(resolve => setImmediate(resolve));
}

// Delivers one frame whenever the run waits for one, until it settles; returns { value } or { error }.
async function drive(clock, promise) {
  let outcome = null;
  promise.then(value => { outcome = { value }; }, error => { outcome = { error }; });
  for (let turn = 0; outcome === null; turn += 1) {
    assert.ok(turn < 1000, 'the run never settled');
    await flush();
    if (outcome === null && clock.waiting() > 0) clock.deliver();
  }
  return outcome;
}

test('createOperationBudget: remainingMs reaches 0 once the clock passes the deadline, and 0 or 300001 is refused', () => {
  const clock = frameClock();
  const budget = createOperationBudget(1000, clock);
  assert.equal(budget.deadlineMs, 1000);
  clock.deliver(600);
  assert.equal(budget.remainingMs(), 400);
  clock.deliver(400);
  assert.equal(budget.remainingMs(), 0);
  clock.deliver(100);
  assert.equal(budget.remainingMs(), 0);
  assert.equal(createOperationBudget(300000, clock).remainingMs(), 300000);
  for (const remainingMs of [0, 300001]) {
    assert.throws(() => createOperationBudget(remainingMs, clock), { message: 'page runner: remainingMs must be a finite number in (0, 300000]' });
  }
});

test('validateInput refuses null, a missing key and an extra key with messages starting page runner: input', () => {
  const valid = runInput('scene-a');
  validateInput(valid);
  const missing = { ...valid };
  delete missing.remainingMs;
  for (const input of [null, missing, { ...valid, extra: true }]) {
    assert.throws(() => validateInput(input), { name: 'Error', message: /^page runner: input/ });
  }
});

test('readiness: a second run while the first is pending is refused, and the first run still completes', async () => {
  const scene = sceneOf('pending-scene');
  const h = harness(new Map([[scene.default.id, scene]]));
  const first = h.runner.run(runInput('pending-scene'));
  await flush();
  assert.equal(h.clock.waiting(), 1);
  await assert.rejects(h.runner.run(runInput('pending-scene')), { message: 'gi runtime: a run is already in progress' });
  const outcome = await drive(h.clock, first);
  assert.ok('value' in outcome, 'the first run must still resolve');
  assert.equal(outcome.value.images.length, 1);
  assert.equal(h.status.textContent, 'ready: pending-scene/default');
});

// The readiness of a failed run is published only through the status text: facts.readiness exists on success.
test('readiness: a failing scene.build rejects with its own error, shows failed: boom and renders nothing', async () => {
  const boom = new Error('boom');
  let observed = 0;
  const scene = sceneOf('failing-scene', {
    build: async () => { throw boom; },
    observe: async () => {
      observed += 1;
      return {};
    },
  });
  const h = harness(new Map([[scene.default.id, scene]]));
  const outcome = await drive(h.clock, h.runner.run(runInput('failing-scene')));
  assert.strictEqual(outcome.error, boom);
  assert.equal(h.status.textContent, 'failed: boom');
  assert.equal(h.status.texts.some(text => text.startsWith('ready')), false);
  assert.equal(h.engines[0].frames, 0);
  assert.equal(observed, 0);
  assert.equal(h.captures.length, 0);
});

test('readiness: a settle reporting failed rejects with readiness: boom, observe never runs and the status stays failed', async () => {
  let observed = 0;
  const scene = sceneOf('settle-scene', {
    observe: async () => {
      observed += 1;
      return {};
    },
  });
  const settle = async () => ({ status: 'failed', stats: null, error: 'boom' });
  const h = harness(new Map([[scene.default.id, scene]]), { settle });
  const outcome = await drive(h.clock, h.runner.run(runInput('settle-scene')));
  assert.equal(outcome.error?.message, 'gi runtime: readiness: boom');
  assert.equal(observed, 0);
  assert.equal(h.status.textContent, 'failed: gi runtime: readiness: boom');
  assert.equal(h.status.texts.some(text => text.startsWith('ready')), false);
  assert.equal(h.captures.length, 0);
});

test('readiness: a settle that stays pending until the operation deadline rejects and never reaches observe', async () => {
  let observed = 0;
  let renders = 0;
  const scene = sceneOf('settle-scene', {
    observe: async () => {
      observed += 1;
      return {};
    },
  });
  // The runner does not cut a settle short, so this settle watches its own budget and reports failed when it runs out.
  const settle = async (_engine, ctx) => {
    while (ctx.budget.remainingMs() > 0) {
      await ctx.render();
      renders += 1;
    }
    return { status: 'failed', stats: null, error: 'not settled before the operation deadline' };
  };
  const h = harness(new Map([[scene.default.id, scene]]), { settle });
  const outcome = await drive(h.clock, h.runner.run(runInput('settle-scene', { remainingMs: 5000 })));
  assert.equal(outcome.error?.message, 'gi runtime: readiness: not settled before the operation deadline');
  assert.ok(renders > 0, 'the settle rendered until the deadline');
  assert.equal(observed, 0);
  assert.equal(h.status.textContent, `failed: ${outcome.error?.message}`);
});

test('a passing run returns facts in collectFacts order then passes, viewSizes and readiness, and a measured record carries measurementExtras', async () => {
  const scene = sceneOf('measured-scene', {
    variants: [{ id: 'on', description: 'On', references: [], judge: true, measurement: { pair: 'gi', state: 'on', preset: 'room', features: { gi: true } } }],
  });
  const h = harness(new Map([[scene.default.id, scene]]), {
    collectFacts: (engine, observation, variant) => ({ settings: { mode: variant.id }, stats: engine.stats() }),
    measurementExtras: () => ({ quality: { resolution: 96, levels: 5, probeRays: 8 }, allocatedBytes: 0 }),
  });
  const outcome = await drive(h.clock, h.runner.run(runInput('measured-scene', { mode: 'on' })));
  assert.ok('value' in outcome, 'the run must resolve');
  const { facts, measurements } = outcome.value;
  assert.deepEqual(Object.keys(facts), ['settings', 'stats', 'passes', 'viewSizes', 'readiness']);
  assert.deepEqual(facts.readiness, { status: 'ready', stage: 'bake', error: null });
  assert.equal(measurements.length, 1);
  assert.equal(measurements[0].pair, 'gi');
  assert.equal(measurements[0].frames, 300);
  assert.deepEqual(measurements[0].quality, { resolution: 96, levels: 5, probeRays: 8 });
  assert.equal(measurements[0].allocatedBytes, 0);
});

test('dispose disposes the Engine once and a later run rejects with runner is disposed', async () => {
  const scene = sceneOf('bake-scene');
  const h = harness(new Map([[scene.default.id, scene]]));
  assert.ok('value' in (await drive(h.clock, h.runner.run(runInput('bake-scene')))));
  h.runner.dispose();
  h.runner.dispose();
  assert.equal(h.runner.engine(), null);
  assert.equal(h.engines[0].disposed, 1);
  await assert.rejects(h.runner.run(runInput('bake-scene')), { message: 'gi runtime: runner is disposed' });
});

test('createPageRunner refuses a missing or blank topic with the page runner message before any other option', () => {
  assert.throws(() => createPageRunner({}), { message: 'page runner: options.topic must be a nonblank string' });
  assert.throws(() => createPageRunner({ topic: '  ' }), { message: 'page runner: options.topic must be a nonblank string' });
  assert.throws(() => createPageRunner({ topic: 'gi' }), { message: 'gi runtime: options.canvas is required' });
});
