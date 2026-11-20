import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as frozen from './scenes/frozen.js';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import { collectRgFacts } from './lib/runtime.js';
import { stageInventory } from '../../web/common/render-graph-job.mjs';

// The frozen stage (contract rg-frozen-scene) against engine doubles: frozen names come from
// catalog.frozen and interpreted documents from catalog.starters, the isolated-pair hooks use
// one fresh engine per half, and observe consumes only the saved half snapshots.

const readJson = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const SIMPLE = readFileSync(new URL('../../crates/raa-graph/graphs/simple-forward.json', import.meta.url), 'utf8');
const FULL_GI = readFileSync(new URL('../../crates/raa-graph/graphs/full-gi.json', import.meta.url), 'utf8');
const DEBUG = readFileSync(new URL('../../crates/raa-graph/graphs/debug.json', import.meta.url), 'utf8');
const STARTERS = [
  { name: 'Simple forward', file: 'simple-forward.json', json: SIMPLE },
  { name: 'Full GI', file: 'full-gi.json', json: FULL_GI },
  { name: 'Debug', file: 'debug.json', json: DEBUG },
];
const FROZEN = ['debug', 'full_gi', 'simple_forward'];
const PASS = [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }];

function engineDouble({ frozen: names = FROZEN, starters = STARTERS, active = null } = {}) {
  const calls = [];
  let current = active, issues = [];
  let weather = { seed: 1, changing: false }, time = { playing: false };
  return {
    calls,
    renderGraphCatalog: () => ({ frozen: names, starters, presets: [] }),
    setRenderGraph(value) {
      calls.push(['setRenderGraph', value]);
      const doc = typeof value === 'string' ? JSON.parse(value) : value;
      if (doc && typeof doc === 'object') current = doc;
      issues = [];
      return { ok: true, rebuilt: 0, issues };
    },
    renderGraphActive: () => current !== null,
    renderGraph: () => current,
    renderGraphIssues: () => issues,
    // A live value the observe path must never use for the saved pair evidence.
    renderGraphStats: () => ({ frameIndex: 999, nodes: [] }),
    passes: () => PASS.map(pass => ({ ...pass })),
    setWeather(value) { calls.push(['setWeather', value]); weather = { ...weather, ...value }; },
    weather: () => ({ ...weather }),
    setTimeOfDay(value) { calls.push(['setTimeOfDay', value]); time = { ...time, ...value }; },
    timeOfDay: () => ({ ...time }),
  };
}

const lib = () => ({
  scene: { buildRoom: async engine => { engine.calls.push(['buildRoom']); return { subjects: [] }; } },
});
const variant = id => frozen.default.variants.find(entry => entry.id === id);
const baseSeeds = () => ({
  weatherSeed: 1, weatherChanging: false, timePlaying: false, freshEngine: true,
  initialRenderCount: 0, source: 'explicit weather seed 1, changing false and paused time on a fresh engine',
});
const renderTimes = () => Array.from({ length: 241 }, (_, frame) => frame * (1000 / 60));
const cameraTimes = () => Array.from({ length: 241 }, (_, frame) => frame / 60);
function snapshot(half, overrides = {}) {
  return {
    half, graphFrameIndex: 240, renderTimesMs: renderTimes(), cameraTimesS: cameraTimes(),
    seeds: { ...baseSeeds() }, graph: { format: 'raa.render-graph', version: 1, name: `frozen ${half}`, nodes: [], links: [] },
    active: true, issues: [], stats: { frameIndex: 240, nodes: [] }, passes: PASS.map(pass => ({ ...pass })),
    ...overrides,
  };
}
const stateOf = halves => ({ isolatedHalves: halves, assertions: [], edits: [] });
const gridDoc = labels => ({
  format: 'raa.render-graph', version: 1, name: 'Debug',
  nodes: [{ id: 'grid', type: 'view.grid', params: { columns: 4, labels } }], links: [],
});

test('the frozen rows, filenames, producer and paired variants are those of the contract', () => {
  const rows = validateInventory([frozen], 'frozen');
  assert.deepEqual(rows.map(({ id, mode, filename, pair }) => [id, mode, filename, pair]), [
    ['frozen-simple-forward', 'simple-forward', 'frozen-simple-forward-left-interpreted-right.png', null],
    ['frozen-full-gi', 'full-gi', 'frozen-full-gi-left-interpreted-right.png', null],
    ['frozen-debug', 'debug', 'frozen-debug-left-interpreted-right.png', null],
  ]);
  assert.equal(new Set(rows.map(row => row.filename)).size, 3);
  assert.ok(rows.every(row => row.scenario === 'render-graph-frozen'
    && row.producer === 'Build the three frozen comparison scenes'
    && row.heldConstant.join(', ') === 'room geometry, materials, lights, sky, camera, seeds'));
  assert.deepEqual(frozen.factNames,
    ['simple-forward-frozen-pair', 'full-gi-frozen-pair', 'debug-frozen-pair']);
  assert.deepEqual(frozen.default.variants.map(entry => [entry.id, entry.pair]), [
    ['simple-forward', { left: 'frozen simple_forward', right: 'interpreted Simple forward', order: ['left', 'right'] }],
    ['full-gi', { left: 'frozen full_gi', right: 'interpreted Full GI', order: ['left', 'right'] }],
    ['debug', { left: 'frozen debug', right: 'interpreted Debug', order: ['left', 'right'] }],
  ]);
  assert.equal(frozen.default.id, 'render-graph-frozen');
  assert.equal(frozen.default.milestone, 'frozen');
});

test('stageInventory selects only the three frozen rows', () => {
  assert.deepEqual(stageInventory([frozen], 'frozen', validateInventory).map(row => row.id),
    ['frozen-simple-forward', 'frozen-full-gi', 'frozen-debug']);
});

test('build reads catalog metadata only and refuses a missing frozen or starter entry', async () => {
  const engine = engineDouble(), helper = lib();
  const state = await frozen.default.build(engine, helper.scene);
  assert.deepEqual(engine.calls, []);
  assert.deepEqual(state.frozen, FROZEN);
  assert.deepEqual(state.mapping.map(({ mode, frozen: name, starter }) => [mode, name, starter]), [
    ['simple-forward', 'simple_forward', 'Simple forward'],
    ['full-gi', 'full_gi', 'Full GI'],
    ['debug', 'debug', 'Debug'],
  ]);
  assert.deepEqual(state.mapping.map(entry => entry.json), [SIMPLE, FULL_GI, DEBUG]);
  const metadata = await frozen.default.buildMetadata(engine, helper.scene);
  assert.deepEqual(metadata.mapping, state.mapping);
  await assert.rejects(frozen.default.build(engineDouble({ frozen: ['debug', 'simple_forward'] }), helper.scene),
    /catalog\.frozen has no full_gi pipeline/);
  await assert.rejects(frozen.default.build(engineDouble({ starters: STARTERS.filter(entry => entry.name !== 'Debug') }), helper.scene),
    /catalog\.starters has no Debug document/);
});

test('isolatedPair freezes the left engine with pipeline and applies the starter on the right', async () => {
  const fullGi = variant('full-gi'), engine = engineDouble();
  const state = await frozen.default.build(engine, lib().scene);
  const hooks = frozen.default.isolatedPair;
  assert.deepEqual(hooks.engineOptions(fullGi, state, 'left'), { pipeline: 'full_gi' });
  assert.deepEqual(hooks.engineOptions(fullGi, state, 'right'), {});

  const left = engineDouble({ active: readJson('../../crates/raa-graph/graphs/full-gi.json') }), leftLib = lib();
  await hooks.prepare(left, leftLib, fullGi, state, 'left');
  assert.deepEqual(left.calls, [
    ['buildRoom'], ['setWeather', { seed: 1, changing: false }], ['setTimeOfDay', { playing: false }],
  ]);
  const provenance = hooks.seeds(left, fullGi, state, 'left');
  assert.deepEqual({ ...provenance, source: 'explicit' }, { ...baseSeeds(), source: 'explicit' });
  assert.match(provenance.source, /fresh engine rendered zero\.\.240 from zero/);

  const right = engineDouble(), rightLib = lib();
  await hooks.prepare(right, rightLib, fullGi, state, 'right');
  assert.deepEqual(right.calls.map(([method]) => method), ['buildRoom', 'setWeather', 'setTimeOfDay', 'setRenderGraph']);
  assert.equal(right.calls.at(-1)[1], FULL_GI);
  assert.equal(right.renderGraph().name, 'Full GI');
  // The runtime requires equal provenance across halves, so the source text is half-independent.
  assert.deepEqual(hooks.seeds(right, fullGi, state, 'right'), provenance);

  await assert.rejects(hooks.prepare(engineDouble(), lib(), fullGi, state, 'left'),
    /frozen pipeline full_gi is not active/);
  const wrongSeed = engineDouble();
  wrongSeed.setWeather({ seed: 2 });
  assert.throws(() => hooks.seeds(wrongSeed, fullGi, state, 'right'), /explicit weather seed 1/);
});

test('observe reads the saved half snapshots and retains them through collectRgFacts', () => {
  const fullGi = variant('full-gi');
  const state = stateOf([snapshot('left'), snapshot('right')]);
  const engine = engineDouble({ active: readJson('../../crates/raa-graph/graphs/full-gi.json') });
  const observation = frozen.observe(engine, fullGi, state);
  const assertions = observation.assertions.filter(entry => entry.name === frozen.factNames[1]);
  assert.equal(assertions.length, 1);
  assert.equal(assertions[0].passed, true);
  const detail = JSON.parse(assertions[0].detail);
  assert.deepEqual(detail.halfSnapshots.map(entry => entry.graphFrameIndex), [240, 240]);
  assert.deepEqual(detail.halfSnapshots.map(entry => entry.renderTimesMs.length), [241, 241]);
  assert.deepEqual(detail.halfSnapshots.map(entry => entry.seeds), [baseSeeds(), baseSeeds()]);
  // The live engine reports frame 999 and no pair evidence, so the values can only come from the snapshot.
  assert.equal(observation.assertions.find(entry => entry.name === frozen.factNames[1]).detail.includes('999'), false);

  const facts = {
    ...collectRgFacts(engine, observation, fullGi),
    passes: engine.passes(),
    viewSizes: [{ view: 0, width: 1280, height: 720 }],
    readiness: { status: 'ready', stage: 'frozen', error: null },
  };
  const row = frozen.inventory.find(entry => entry.mode === 'full-gi');
  validateRequiredFacts(row, facts, frozen.factNames);
});

test('observe refuses missing or unequal frame, trace and seed evidence', () => {
  const mode = variant('simple-forward'), engine = engineDouble();
  const left = snapshot('left'), right = snapshot('right');
  assert.throws(() => frozen.observe(engine, mode, stateOf([left])), /exactly two halves/);
  assert.throws(() => frozen.observe(engine, mode, { isolatedHalves: undefined }),
    /exactly two halves/);
  assert.throws(() => frozen.observe(engine, mode, stateOf([left,
    { ...right, graphFrameIndex: 239, stats: { frameIndex: 239, nodes: [] } }])), /graph frame index is not 240/);
  assert.throws(() => frozen.observe(engine, mode, stateOf([left,
    { ...right, stats: { frameIndex: 239, nodes: [] } }])), /stats frame index differs/);
  assert.throws(() => frozen.observe(engine, mode, stateOf([left,
    { ...right, renderTimesMs: right.renderTimesMs.map((time, frame) => frame === 5 ? time + 1 : time) }])),
  /frame trace differs at frame 5/);
  assert.throws(() => frozen.observe(engine, mode, stateOf([left,
    { ...right, cameraTimesS: right.cameraTimesS.map((time, frame) => frame === 3 ? time + 0.01 : time) }])),
  /frame trace differs at frame 3/);
  assert.throws(() => frozen.observe(engine, mode, stateOf([left,
    { ...right, seeds: { ...right.seeds, weatherSeed: 2 } }])), /different seed provenance/);
  assert.throws(() => frozen.observe(engine, mode, stateOf([left,
    { ...right, seeds: { ...right.seeds, weatherChanging: true } }])), /different seed provenance/);
  assert.throws(() => frozen.observe(engine, mode, stateOf([left,
    { ...right, seeds: { ...right.seeds, source: '' } }])), /seed provenance is incomplete/);
});

test('the Debug pair keeps the same readable labelled grid tiles on both halves', () => {
  const debug = variant('debug'), engine = engineDouble();
  const good = stateOf([snapshot('left', { graph: gridDoc(true) }), snapshot('right', { graph: gridDoc(true) })]);
  const passed = frozen.observe(engine, debug, good).assertions.find(entry => entry.name === frozen.factNames[2]);
  assert.equal(passed.passed, true);
  assert.deepEqual(JSON.parse(passed.detail).labeledTiles, [true, true]);
  const bad = stateOf([snapshot('left', { graph: gridDoc(true) }), snapshot('right', { graph: gridDoc(false) })]);
  const failed = frozen.observe(engine, debug, bad).assertions.find(entry => entry.name === frozen.factNames[2]);
  assert.equal(failed.passed, false);
  assert.deepEqual(JSON.parse(failed.detail).labeledTiles, [true, false]);
});
