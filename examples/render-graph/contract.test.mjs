import test from 'node:test';
import assert from 'node:assert/strict';
import { RG_CAPTURE_T_S, ROOM_SUBJECTS, cameraAt, selectionAt } from './lib/scene.js';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import { captionText, collectRgFacts, createRgPageRunner, halfFrameTimes } from './lib/runtime.js';
import * as interpreter from './scenes/interpreter.js';
import { runNeuralBaseline } from '../global-illumination/lib/neural-baseline-runner.js';
import { BASELINE_EVENTS, BASELINE_FRAMES, BASELINE_WARMUP_FRAMES } from '../global-illumination/lib/neural-baseline.js';
import { browserFrameFeedback } from '../../web/common/browser-frame-feedback.js';
import { attachGiPage } from '../global-illumination/main.js';
import { analyze } from '../../scripts/neural-baseline-latency.mjs';

// Browser-free contract checks of the render-graph page share (contracts rg-room-scene and
// rg-page-facts): the RG_CAMERA_PATH formula, the room subjects in the capture view, the
// inventory rules and the required-facts gate.

test('cameraAt follows RG_CAMERA_PATH at t = 0 and at the capture time', () => {
  // By hand: a = 0.6 + 0.1 t, position = (9 sin a, 3, 9 cos a), target (0, 1, 0), fov 50.
  const hand = t => {
    const a = 0.6 + 0.1 * t;
    return [9 * Math.sin(a), 3, 9 * Math.cos(a)];
  };
  assert.deepEqual(cameraAt(0), { position: hand(0), target: [0, 1, 0], fovDeg: 50 });
  assert.deepEqual(cameraAt(RG_CAPTURE_T_S), { position: hand(RG_CAPTURE_T_S), target: [0, 1, 0], fovDeg: 50 });
  // The captured frame is 240 warm-up frames later: 240 x (1000 / 60) ms = 4 s.
  assert.equal(RG_CAPTURE_T_S, 4);
  assert.deepEqual(cameraAt(4).position, [9 * Math.sin(1), 3, 9 * Math.cos(1)]);
});

const ROW = {
  id: 'interpreter-room-visible',
  criterionName: 'P5-Q2 lit room',
  scenario: 'interpreter',
  mode: 'room',
  filename: 'interpreter-room-lit-shadowed.png',
  expectedVisibleBehavior: 'The lit room with sun shadows under the boxes on the visible floor.',
  requiredFacts: ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready', 'assertion:room-visible'],
  pair: null,
  heldConstant: ['room geometry', 'camera'],
  producer: 'raa-graph',
};

// A minimal scene namespace shaped like the interpreter scene of the S61 page.
function fixture() {
  return {
    default: {
      id: 'interpreter',
      milestone: 'interpreter',
      title: 'fixture',
      variants: [{ id: 'room', selection: selectionAt(RG_CAPTURE_T_S) }],
      build() {},
    },
    factNames: ['room-visible'],
    inventory: [ROW],
    observe() {},
  };
}

test('every ROOM_SUBJECTS box overlaps the capture view volume through validateInventory', () => {
  assert.equal(selectionAt(RG_CAPTURE_T_S).subjects, ROOM_SUBJECTS);
  assert.deepEqual(ROOM_SUBJECTS.map(s => s.name),
    ['floor', 'back-wall', 'left-wall', 'box-a', 'box-b', 'box-c', 'sphere', 'emissive-panel']);
  assert.equal(validateInventory([fixture()], 'interpreter').length, 1);
  for (const subject of ROOM_SUBJECTS) {
    const ns = fixture();
    ns.default.variants[0].selection = { ...selectionAt(RG_CAPTURE_T_S), subjects: [subject] };
    assert.equal(validateInventory([ns], 'interpreter').length, 1, subject.name);
  }
});

test('validateInventory refuses a duplicate filename and a non-descriptive filename', () => {
  const duplicate = fixture();
  duplicate.inventory = [ROW, { ...ROW, id: 'interpreter-room-copy' }];
  assert.throws(() => validateInventory([duplicate], 'interpreter'), /invalid inventory row interpreter-room-copy/);
  const vague = fixture();
  vague.inventory = [{ ...ROW, filename: 'capture.png' }];
  assert.throws(() => validateInventory([vague], 'interpreter'), /invalid inventory row/);
});

test('validateInventory refuses an unknown stage and any measurement', () => {
  assert.throws(() => validateInventory([fixture()], 'volume'), /invalid stage or scenes: volume/);
  const measured = fixture();
  measured.default.variants[0].measurement = { pair: 'room', state: 'on', preset: 'room', features: {} };
  assert.throws(() => validateInventory([measured], 'interpreter'), /measurement is not allowed/);
});

function facts() {
  return {
    renderGraphActive: true,
    graph: 'Simple forward',
    issues: [{ code: 'graph.input_missing', severity: 'warning', nodeId: 'forward', port: 'ao', message: 'unlinked', fix: 'link ao' }],
    assertions: [{ name: 'room-visible', passed: true, detail: 'eight subjects inside the view volume' }],
    edits: [{ operation: 'setRenderGraph', accepted: true, error: null }],
    passes: [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }],
    viewSizes: [{ view: 0, width: 1280, height: 720 }],
    readiness: { status: 'ready', stage: 'interpreter', error: null },
  };
}

test('validateRequiredFacts accepts valid facts and refuses a false assertion', () => {
  validateRequiredFacts(ROW, facts(), ['room-visible']);
  const failed = { ...facts(), assertions: [{ name: 'room-visible', passed: false, detail: 'back wall outside the view' }] };
  assert.throws(() => validateRequiredFacts(ROW, failed, ['room-visible']),
    error => /required assertion:room-visible is missing or invalid/.test(error.message)
      && error.message.includes('back wall outside the view'));
});

test('halfFrameTimes() is the 241 RG_RUN camera times from 0 to RG_CAPTURE_T_S', () => {
  const times = halfFrameTimes();
  assert.equal(times.length, 241);
  assert.equal(times[0], 0);
  assert.equal(times.at(-1), RG_CAPTURE_T_S);
  assert.ok(times.every((time, frame) => frame === 0 || time > times[frame - 1]));
});

test('captionText names the refused code, node and parameter', () => {
  assert.equal(
    captionText({ code: 'graph.param_range', nodeId: 'aoGtao', param: 'radius', message: 'm', severity: 'error', fix: 'f' }),
    'refused: graph.param_range on node aoGtao (param radius): m');
  assert.equal(
    captionText({ code: 'graph.cycle', nodeId: 'forward', message: 'link cycle', severity: 'error', fix: 'f' }),
    'refused: graph.cycle on node forward: link cycle');
});

test('collectRgFacts maps an observation and yields nulls for absent fields', () => {
  const observation = {
    renderGraphActive: true,
    graph: 'Simple forward',
    issues: [{ code: 'graph.param_range', severity: 'error', nodeId: 'aoGtao', param: 'radius', message: 'm', fix: 'f' }],
    assertions: [{ name: 'room-visible', passed: true, detail: 'eight subjects inside the view volume' }],
    edits: [{ operation: 'setRenderGraph', accepted: true, error: null }],
  };
  assert.deepEqual(collectRgFacts(null, observation, null), observation);
  assert.deepEqual(collectRgFacts(null, {}, null),
    { renderGraphActive: null, graph: null, issues: null, assertions: null, edits: null });
  assert.deepEqual(collectRgFacts(null, null, null),
    { renderGraphActive: null, graph: null, issues: null, assertions: null, edits: null });
});

// Contract rg-interpreter-inventory: the interpreter scene is the stage 'interpreter' (rank 0).
test('validateInventory returns the four interpreter rows of stage interpreter', () => {
  const rows = validateInventory([interpreter], 'interpreter');
  assert.equal(rows.length, 4);
  assert.deepEqual(rows.map(row => row.id),
    ['interpreter-simple-forward', 'interpreter-full-gi-pair', 'interpreter-null-pair', 'interpreter-refused']);
  assert.equal(new Set(rows.map(row => row.filename)).size, 4);
  assert.ok(rows.every(row => row.filename.startsWith('interpreter-') && row.scenario === 'render-graph-interpreter'));
  assert.equal(interpreter.default.id, 'render-graph-interpreter');
  assert.equal(interpreter.default.milestone, 'interpreter');
  // The interpreter scene provides stage interpreter only: no later stage is available from it.
  assert.throws(() => validateInventory([interpreter], 'views'), /stage views is unavailable/);
});

test('every ROOM_SUBJECTS box lies in the RG_CAPTURE_T_S view of the interpreter rows', () => {
  for (const subject of ROOM_SUBJECTS) {
    const selection = { ...selectionAt(RG_CAPTURE_T_S), subjects: [subject] };
    const variants = interpreter.default.variants.map(variant => ({ ...variant, selection }));
    const scene = { ...interpreter, default: { ...interpreter.default, variants } };
    assert.equal(validateInventory([scene], 'interpreter').length, 4, subject.name);
  }
});

function fakeCanvas(trace = null) {
  const context = {
    drawImage(...args) { trace?.push({ op: 'drawImage', args }); },
    fillRect(...args) { trace?.push({ op: 'fillRect', args, fillStyle: context.fillStyle }); },
    fillText(...args) { trace?.push({ op: 'fillText', args, fillStyle: context.fillStyle, font: context.font }); },
    measureText: text => ({ width: text.length * 8 }),
  };
  return { width: 0, height: 0, ownerDocument: { createElement: () => fakeCanvas(trace) },
    getContext: () => context, toDataURL: () => 'data:image/png;base64,aGVsbG8=' };
}

function runtimeDouble(isolatedPair, trace, graph = { format: 'raa.render-graph', version: 1, name: 'pair', nodes: [], links: [] }) {
  let id = 0;
  const engines = [];
  const load = async () => ({ createEngine: async (canvas, options) => {
    const identity = ++id, engine = {
      identity, options, canvas, frameIndex: null, count: 0, disposed: false,
      resize(width, height) { canvas.width = width; canvas.height = height; },
      setCamera(pose) { trace.cameras.push([identity, pose]); },
      render(time) { this.count++; this.frameIndex = Math.round(time / (1000 / 60)); trace.renders.push([identity, time]); },
      renderGraph() { return graph; },
      renderGraphActive() { return true; }, renderGraphIssues() { return []; },
      renderGraphStats() { return { frameIndex: this.frameIndex, nodes: [] }; },
      passes() { return [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }]; },
      stats() { return { hardware: false, adapter: {} }; }, dispose() { this.disposed = true; trace.disposed.push(identity); },
    };
    engines.push(engine);
    trace.created.push({ identity, options, canvas });
    return engine;
  } });
  const variant = { id: 'pair', pair: { order: ['left', 'right'], left: 'frozen', right: 'interpreted' }, selection: { camera: {}, subjects: [] } };
  const row = { ...ROW, scenario: 'fixture', mode: 'pair', requiredFacts: [] };
  const entry = { default: { id: 'fixture', milestone: 'interpreter', variants: [variant], isolatedPair,
    build() { trace.setupBuild++; } },
    observe(engine) {
      trace.observed = engine.identity;
      return facts();
    }, inventory: [row], factNames: [] };
  const canvas = fakeCanvas(trace.canvasOps ??= []);
  const runner = createRgPageRunner({ canvas, status: { textContent: '' }, load, scenes: new Map([['fixture', entry]]), lib: {},
    clock: { now: () => 0, requestFrame: cb => { cb(); return 1; }, cancelFrame() {} } });
  return { runner, trace, engines, canvas };
}

test('isolatedPair creates two independent 241-frame engines and routes observation to interpreted half', async () => {
  const trace = { created: [], renders: [], cameras: [], disposed: [], setupBuild: 0 };
  const seeds = (_engine, _variant, _state, half) => ({ weatherSeed: 7, weatherChanging: false,
    timePlaying: false, freshEngine: true, initialRenderCount: 0, source: 'explicit-room-and-fresh-render-sequence' });
  const { runner, engines } = runtimeDouble({
    engineOptions: (_variant, _state, half) => half === 'left' ? { pipeline: 'full_gi' } : {},
    prepare: async (engine, _lib, _variant, _state, half) => { trace.prepared = [...(trace.prepared ?? []), [engine.identity, half]]; },
    seeds,
  }, trace);
  const previousFrame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = callback => { callback(); return 1; };
  let result;
  try {
    result = await runner.run({ selectionId: 's', attemptId: 'a', buildId: 'b', scenario: 'fixture', mode: 'pair', width: 1280, height: 720, remainingMs: 300000 });
  } finally {
    if (previousFrame === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousFrame;
  }
  assert.equal(engines.length, 3); // page-kit setup handle plus the two measured engines
  assert.deepEqual(trace.created.slice(1).map(row => row.options), [{ pipeline: 'full_gi' }, {}]);
  assert.deepEqual(trace.prepared.map(row => row[1]), ['left', 'right']);
  assert.equal(trace.renders.filter(([identity]) => identity === 2).length, 241);
  assert.equal(trace.renders.filter(([identity]) => identity === 3).length, 242); // page-kit final observation render
  assert.equal(trace.observed, 3);
  assert.equal(result.images[0].data, 'aGVsbG8=');
  assert.equal(engines[1].disposed, true);
  assert.equal(engines[2].count, 242); // capture is copied before page-kit observation render
  runner.dispose();
  assert.equal(engines[2].disposed, true);
});

test('Debug pair annotations restore all tile labels below headers inside both halves', async () => {
  const labels = ['depth', 'normal', 'motion', 'ao', 'area shadow mask', 'giRawVsDenoised', 'history length', 'final'];
  const nodes = labels.map((label, index) => ({ id: `tile${index}`, type: 'view.image', params: { label } }));
  nodes.push({ id: 'grid', type: 'view.grid', params: { labels: true, columns: 4 } });
  const graph = { format: 'raa.render-graph', version: 1, name: 'Debug', nodes,
    links: labels.map((_, index) => ({ from: `tile${index}.color`, to: `grid.t${index}` })) };
  const trace = { created: [], renders: [], cameras: [], disposed: [], setupBuild: 0 };
  const seeds = () => ({ weatherSeed: 7, weatherChanging: false, timePlaying: false,
    freshEngine: true, initialRenderCount: 0, source: 'explicit-room-and-fresh-render-sequence' });
  const { runner } = runtimeDouble({
    engineOptions: (_variant, _state, half) => half === 'left' ? { pipeline: 'debug' } : {},
    prepare: async () => {}, seeds,
  }, trace, graph);
  const previousFrame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = callback => { callback(); return 1; };
  try {
    await runner.run({ selectionId: 's', attemptId: 'a', buildId: 'b', scenario: 'fixture', mode: 'pair', width: 1280, height: 720, remainingMs: 300000 });
  } finally {
    if (previousFrame === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousFrame;
    runner.dispose();
  }

  const ops = trace.canvasOps;
  const labelOps = ops.filter(op => op.op === 'fillText' && labels.some(label => op.args[0] === (label === 'ao' ? 'ambient occlusion' : label)));
  const expected = [...labels, ...labels].map(label => label === 'ao' ? 'ambient occlusion' : label);
  assert.deepEqual(labelOps.map(op => op.args[0]), expected);
  const lastImage = ops.reduce((last, op, index) => op.op === 'drawImage' && op.args.length === 3 ? index : last, -1);
  const lastHeader = ops.findLastIndex(op => op.op === 'fillText' && ['frozen', 'interpreted'].includes(op.args[0]));
  assert.ok(lastImage < lastHeader && lastHeader < ops.indexOf(labelOps[0]));
  const boxOps = ops.filter(op => op.op === 'fillRect' && op.fillStyle === '#000000' && op.args[3] === 18);
  assert.equal(boxOps.length, 16);
  for (let half = 0; half < 2; half++) {
    for (let tile = 0; tile < 8; tile++) {
      const box = boxOps[half * 8 + tile].args;
      const x = half * 1280 + (tile % 4) * 320;
      const y = Math.max(28, Math.floor(tile / 4) * 360);
      assert.equal(box[0], x);
      assert.equal(box[1], y);
      assert.ok(box[2] <= 320 && box[3] === 18);
      assert.ok(box[0] + box[2] <= x + 320 && box[1] + box[3] <= y + 360);
    }
  }
  const finalAnnotation = Math.max(...labelOps.map((_, index) => ops.indexOf(labelOps[index])));
  assert.equal(ops.slice(finalAnnotation + 1).some(op => ['drawImage', 'fillRect', 'fillText'].includes(op.op)), false);
});

test('non-grid pair composition keeps its image blits and title band without annotations', async () => {
  const trace = { created: [], renders: [], cameras: [], disposed: [], setupBuild: 0 };
  const { runner } = runtimeDouble({
    engineOptions: () => ({}), prepare: async () => {},
    seeds: () => ({ weatherSeed: 7, weatherChanging: false, timePlaying: false,
      freshEngine: true, initialRenderCount: 0, source: 'explicit-room-and-fresh-render-sequence' }),
  }, trace);
  const previousFrame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = callback => { callback(); return 1; };
  try {
    await runner.run({ selectionId: 's', attemptId: 'a', buildId: 'b', scenario: 'fixture', mode: 'pair', width: 1280, height: 720, remainingMs: 300000 });
  } finally {
    if (previousFrame === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = previousFrame;
    runner.dispose();
  }
  assert.equal(trace.canvasOps.filter(op => op.op === 'drawImage' && op.args.length === 3).length, 2);
  assert.equal(trace.canvasOps.filter(op => op.op === 'fillRect').length, 2); // output background and title band
  assert.deepEqual(trace.canvasOps.filter(op => op.op === 'fillText').map(op => op.args[0]), ['frozen', 'interpreted']);
});

function baselineRuntime(beforeFrame = () => {}) {
  let time = 0;
  let keyboard = null;
  const timeOrigin = 1795470000000;
  const status = { textContent: '' };
  const engines = [];
  const canvas = {
    style: {},
    focus() {},
    getBoundingClientRect: () => ({ toJSON: () => ({ x: 0, y: 0, width: canvas.width, height: canvas.height }) }),
    addEventListener(name, listener) { assert.equal(name, 'keydown'); keyboard = listener; },
    removeEventListener(name, listener) { if (listener === keyboard) keyboard = null; },
  };
  const target = {
    performance: { timeOrigin, now: () => time += 0.05 },
    document: { visibilityState: 'visible' },
    navigator: { userAgent: 'test browser' },
    devicePixelRatio: 1,
    requestAnimationFrame(callback) {
      return setImmediate(() => {
        time += 16;
        beforeFrame(engines.at(-1), overrides => keyboard?.({
          code: 'Space', timeStamp: time - 1, isTrusted: true, preventDefault() {}, ...overrides,
        }));
        callback(time);
      });
    },
    cancelAnimationFrame: clearImmediate,
    setTimeout,
    clearTimeout,
  };
  const raa = { async createEngine() {
    const engine = {
      mutations: [], renders: [], light: null, blocker: null, disposed: false,
      setSky() {}, setWeather() {}, setEnvironmentLighting() {}, setAmbient() {}, setLight() {},
      setToneMapping() {}, setGlobalIllumination() {}, setAreaLightShadows() {}, setGiQuality() {},
      setTimeOfDay() {}, setAutoExposure() {}, setPassEnabled() {}, setRenderScale() {},
      addMaterial: () => 1,
      addLight(light) { assert.equal(this.light, null); this.light = light; return 4; },
      addNode() { return ++this.nodes; },
      nodes: 0,
      addMesh: mesh => mesh.positions.length,
      addMeshInstance(_node, mesh) { return mesh; },
      setInstanceGlobalIllumination() {},
      setCamera(pose) { this.mutations.push(['camera', pose]); },
      setTransform(node, position) {
        assert.equal(node, this.nodes);
        this.mutations.push(['blocker', position]);
      },
      updateLight(handle, change) {
        assert.equal(handle, 4);
        this.mutations.push(['light', change.intensity]);
      },
      giStats: () => ({ active: true, pendingRebuild: false }),
      areaLightShadows: () => ({
        tier: 'auto', activeHeroes: 0, activeStochastic: 1, stochasticCells: 4, stochasticMode: 'stratified',
      }),
      lightStats: () => ({ activeLights: 1 }),
      passes: () => [{ name: 'gi.area-shadow', enabled: true }],
      stats: () => ({ hardware: true, adapter: { vendor: 'test' } }),
      render(nowMs) { this.renders.push(nowMs); },
      renderInstrumented(nowMs, ctx) {
        const frameIndex = this.renders.length;
        this.render(nowMs);
        return {
          ...ctx, nowMs, timeOriginMs: timeOrigin, frameIndex, error: null,
          renderStartMs: target.performance.now(), acquireStartMs: target.performance.now(),
          acquireEndMs: target.performance.now(), submitStartMs: target.performance.now(),
          submitEndMs: target.performance.now(), presentSelectedMs: target.performance.now(),
          presentReturnedMs: target.performance.now(),
        };
      },
      dispose() { this.disposed = true; },
    };
    engines.push(engine);
    return engine;
  } };
  return { target, canvas, raa, status, engines, hasKeyListener: () => keyboard !== null };
}

test('neural baseline replays identical blocker, flash and edge events on fresh engines', async () => {
  const harness = baselineRuntime();
  const trials = await runNeuralBaseline(harness, { captureId: 'playback', loadClass: 'idle' });
  assert.deepEqual(trials.map(trial => trial.trialId), ['playback:1', 'playback:2']);
  assert.deepEqual(harness.engines[0].mutations, harness.engines[1].mutations);
  assert.deepEqual(harness.engines[0].renders, harness.engines[1].renders);
  for (const engine of harness.engines) {
    assert.equal(engine.light.kind, 'area');
    assert.deepEqual(engine.light.size, [1.05, 1.30]);
    assert.equal(engine.renders.length, BASELINE_WARMUP_FRAMES + BASELINE_FRAMES);
    assert.equal(engine.disposed, true);
    const [initialCamera, blocker, flashOn, flashOff, edgeCamera, restoredCamera] = engine.mutations;
    assert.deepEqual(blocker, ['blocker', [0.85, 0, 0]]);
    assert.deepEqual(flashOn, ['light', engine.light.intensity * 2]);
    assert.deepEqual(flashOff, ['light', engine.light.intensity]);
    assert.notDeepEqual(edgeCamera, initialCamera);
    assert.deepEqual(restoredCamera, initialCamera);
  }
  for (const trial of trials) {
    assert.equal(trial.status, 'complete');
    assert.deepEqual(trial.missingEventIds, []);
    assert.deepEqual(trial.facts.features, { denoise: false, upscale: false, predict: false });
    assert.equal(trial.facts.areaShadows.stochasticMode, 'stratified');
    assert.equal(trial.frames.length, BASELINE_FRAMES);
    assert.deepEqual(trial.events.map(event => [event.eventId, event.presentId]),
      BASELINE_EVENTS.map(event => [event.eventId, String(event.frame)]));
    assert.ok(trial.events.every(event => event.trigger === 'scripted'));
    assert.ok(trial.frames.every(frame => frame.frameIndex === BASELINE_WARMUP_FRAMES + frame.fixtureFrame));
  }
});

test('neural baseline binds trusted key timing to the first affected frame', async () => {
  const keyFrames = [1, 19, 37, 55, 73];
  const keyTimes = [];
  const harness = baselineRuntime((engine, key) => {
    const frame = engine.renders.length - BASELINE_WARMUP_FRAMES;
    if (frame === 0) {
      key({ isTrusted: false });
      key({ repeat: true });
      key({ code: 'Escape' });
    }
    if (keyFrames.includes(frame)) {
      const eventMs = harness.target.performance.now() - 0.5;
      keyTimes.push(eventMs);
      key({ timeStamp: eventMs });
      key(); // The pending key already owns this frame.
    }
  });
  const [trial] = await runNeuralBaseline(harness, { captureId: 'keys', mode: 'keyboard', loadClass: 'gpu-load' });
  assert.equal(trial.status, 'complete');
  assert.equal(trial.loadClass, 'gpu-load');
  assert.equal(harness.hasKeyListener(), false);
  assert.deepEqual(trial.missingEventIds, []);
  assert.deepEqual(trial.events.map(event => event.eventMs), keyTimes);
  assert.deepEqual(trial.frames.map(frame => frame.fixtureFrame), keyFrames);
  const [summary] = analyze([trial]).loadClasses;
  assert.equal(summary.keyboardEventCount, keyFrames.length);
  assert.equal(summary.keyToSubmitMs.count, keyFrames.length);
  assert.equal(summary.keyToBrowserPresentationMs[0].latencyMs.count, keyFrames.length);
  assert.equal(summary.screenRefresh, 'unavailable');
  for (const [index, event] of trial.events.entries()) {
    const frame = trial.frames[index];
    assert.deepEqual([event.eventId, event.viewId, event.trialId, event.presentId],
      [frame.eventId, frame.viewId, frame.trialId, frame.presentId]);
    assert.equal(event.trigger, 'keyboard');
    assert.equal(event.keyCode, 'Space');
    assert.equal(frame.eventMs, event.eventMs);
    assert.equal(frame.acceptedMs, event.acceptedMs);
    assert.ok(event.eventMs <= event.acceptedMs && event.acceptedMs <= event.appliedMs);
    assert.ok(event.appliedMs <= frame.inputCutoffMs && frame.inputCutoffMs <= frame.submitStartMs);
    assert.deepEqual(event.presentation, [{
      source: 'next-animation-frame', kind: 'estimate', atMs: frame.nextCallbackStartMs,
      eventId: event.eventId, timeOriginMs: trial.timeOriginMs, viewId: frame.viewId,
      trialId: frame.trialId, presentId: frame.presentId,
    }]);
  }
});

test('browser presentation feedback is scoped to the affected callback and avoids unconfirmed paint matches', () => {
  const frame = { previousRenderEndMs: 90, callbackStartMs: 110, presentReturnedMs: 115, nextCallbackStartMs: 126 };
  const event = { trigger: 'keyboard', eventMs: 99, acceptedMs: 102 };
  const key = { entryType: 'event', name: 'keydown', startTime: 99, processingStart: 100, processingEnd: 103,
    duration: 16, presentationTime: 116 };
  const draw = { entryType: 'long-animation-frame', renderStart: 109, styleAndLayoutStart: 120,
    paintTime: 121, presentationTime: 125 };
  const later = { ...draw, renderStart: 123, presentationTime: 180 };
  const result = browserFrameFeedback(frame, event, [key, draw, later]);
  assert.equal(result.documentKey.presentationMs, 116);
  assert.deepEqual(result.presentation, [
    { source: 'long-animation-frame', kind: 'browser-feedback', atMs: 125 },
    { source: 'next-animation-frame', kind: 'estimate', atMs: 126 },
  ]);
  const onlyLater = browserFrameFeedback(frame, event, [key, later]);
  assert.equal(onlyLater.documentFrame, null);
  assert.deepEqual(onlyLater.presentation, [result.presentation[1]]);
  assert.deepEqual(browserFrameFeedback({ ...frame, presentReturnedMs: null }, event, [key, draw]).presentation, []);
});

test('neural baseline preserves a key accepted before the trial times out without a frame', async () => {
  let expire;
  const harness = baselineRuntime((engine, key) => {
    if (engine.renders.length === BASELINE_WARMUP_FRAMES) {
      key();
      expire();
    }
  });
  harness.target.setTimeout = callback => { expire = callback; return 1; };
  harness.target.clearTimeout = () => {};
  const [trial] = await runNeuralBaseline(harness, { captureId: 'miss', mode: 'keyboard' });
  assert.equal(trial.status, 'timed-out');
  assert.equal(trial.frames.length, 0);
  assert.equal(trial.events.length, 1);
  assert.equal(trial.events[0].eventId, 'blocker-step');
  assert.equal(trial.events[0].presentId, null);
  assert.equal(trial.events[0].appliedMs, null);
  assert.deepEqual(trial.events[0].presentation, []);
  assert.deepEqual(trial.missingEventIds, BASELINE_EVENTS.slice(1).map(event => event.eventId));
  assert.equal(harness.hasKeyListener(), false);
  assert.equal(harness.engines[0].disposed, true);
});

test('the GI page exposes baseline replay through its managed operation', async () => {
  const harness = baselineRuntime();
  const { target, canvas, status } = harness;
  target.document.createElement = () => ({});
  target.addEventListener = () => {};
  const control = () => ({ addEventListener() {}, appendChild() {}, replaceChildren() {} });
  const scenePicker = control();
  const variantPicker = control();
  const runButton = control();
  let runnerCount = 0;
  let disposed = 0;
  const formerWindow = globalThis.window;
  try {
    globalThis.window = target;
    await attachGiPage({
      target, canvas, status, scenePicker, variantPicker, runButton,
      load: async () => harness.raa,
      discover: async () => new Map([['gi', {
        default: { id: 'gi', title: 'GI', variants: [{ id: 'still' }] },
        inventory: [{ mode: 'still' }],
      }]]),
      createRunner() {
        runnerCount++;
        return { run: input => input, dispose() { disposed++; } };
      },
    });
    const baseline = target.nmcGpu.ops.baseline({ args: { captureId: 'managed', repeat: 1, loadClass: 'idle' } });
    await assert.rejects(target.nmcGpu.ops.run({ scenario: 'gi' }), /run is already in progress/);
    const result = await baseline;
    assert.deepEqual(result.files, []);
    assert.equal(result.json[0].trialId, 'managed:1');
    assert.equal(result.json[0].status, 'complete');
    assert.equal(disposed, 1);
    assert.equal(runButton.disabled, false);
    assert.deepEqual(await target.nmcGpu.ops.run({ scenario: 'gi' }), { scenario: 'gi' });
    assert.equal(runnerCount, 2);
  } finally {
    if (formerWindow === undefined) delete globalThis.window;
    else globalThis.window = formerWindow;
  }
});
