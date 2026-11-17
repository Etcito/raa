import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import scene, { factNames, inventory, observe } from './scenes/interpreter.js';
import { ROOM_SUBJECTS } from './lib/scene.js';
import { validateRequiredFacts } from './lib/inventory.js';
import { collectRgFacts } from './lib/runtime.js';

// The interpreter scene (contract rg-interpreter-inventory) against an engine double: a plain
// object that records, in order, the calls which change the engine. Its catalog holds the two real
// starter texts; setRenderGraph answers the scripted results (accepted unless scripted), and an
// accepted graph sets the pass list the way the engine does.
const SIMPLE = readFileSync(new URL('../../crates/raa-graph/graphs/simple-forward.json', import.meta.url), 'utf8');
const FULL_GI = readFileSync(new URL('../../crates/raa-graph/graphs/full-gi.json', import.meta.url), 'utf8');
const ACCEPTED = { ok: true, rebuilt: 1, issues: [] };
const REFUSAL = {
  code: 'graph.param_range', severity: 'error', nodeId: 'aoGtao', param: 'radius',
  message: 'radius 25 is outside (0, 10]', fix: 'set radius within (0, 10]',
};
const REFUSED = { ok: false, rebuilt: 0, issues: [REFUSAL] };
const CAPTION = 'refused: graph.param_range on node aoGtao (param radius): radius 25 is outside (0, 10]';
// The engine's passes with no graph, all enabled; the non-required ones in this order.
const PASSES = [
  { name: 'sky.atmosphere', role: 'pass', enabled: true, order: 5, required: false },
  { name: 'shadow.csm', role: 'pass', enabled: true, order: 20, required: false },
  { name: 'prepass', role: 'pass', enabled: true, order: 30, required: true },
  { name: 'light.tiles', role: 'pass', enabled: true, order: 35, required: false },
  { name: 'ao.gtao', role: 'pass', enabled: true, order: 40, required: false },
  { name: 'forward', role: 'pass', enabled: true, order: 60, required: true },
  { name: 'ssr', role: 'pass', enabled: true, order: 65, required: false },
  { name: 'dof', role: 'pass', enabled: true, order: 110, required: false },
  { name: 'tonemap', role: 'pass', enabled: true, order: 200, required: true },
  { name: 'sky.background', role: 'contributor', enabled: true, order: 900, required: false },
];

function engineDouble(results) {
  const calls = [], script = [...results];
  const defaults = PASSES.map(pass => ({ ...pass }));
  let passes = defaults.map(pass => ({ ...pass })), active = null, issues = [];
  return {
    calls,
    renderGraphCatalog: () => ({
      starters: [
        { name: 'Simple forward', file: 'simple-forward.json', json: SIMPLE },
        { name: 'Full GI', file: 'full-gi.json', json: FULL_GI },
      ],
    }),
    setRenderGraph(graph) {
      calls.push(['setRenderGraph', graph]);
      const result = script.shift() ?? ACCEPTED;
      if (result.ok) {
        active = typeof graph === 'string' ? JSON.parse(graph) : graph;
        const entries = active && new Set(active.nodes.filter(node => node.type.startsWith('pass.')).map(node => node.type.slice(5)));
        passes = defaults.map(pass => ({ ...pass, enabled: entries ? pass.required || entries.has(pass.name) : true }));
      }
      issues = result.issues ?? [];
      return result;
    },
    renderGraphActive: () => active !== null,
    renderGraph: () => active,
    renderGraphIssues: () => issues,
    passes: () => passes.map(pass => ({ ...pass })),
    setPassEnabled(name, enabled) {
      calls.push(['setPassEnabled', name, enabled]);
      passes.find(pass => pass.name === name).enabled = enabled;
    },
    setGlobalIllumination(patch) { calls.push(['setGlobalIllumination', patch]); },
    setGiQuality(patch) { calls.push(['setGiQuality', patch]); },
  };
}

// Runs one mode the way the page runner does: build, then prepareHalf for each half of its order.
async function run(mode, results = [], overrides = {}) {
  const engine = { ...engineDouble(results), ...overrides }, frames = [];
  const lib = { scene: { buildRoom: async target => { target.calls.push(['buildRoom']); return { subjects: ROOM_SUBJECTS }; } } };
  const variant = scene.variants.find(item => item.id === mode);
  const state = await scene.build(engine, lib, variant);
  for (const half of variant.pair?.order ?? ['single']) {
    await scene.prepareHalf(engine, lib, variant, state, half, { renderFrames: async count => { frames.push(count); } });
  }
  return { engine, state, variant, frames };
}
const verdicts = state => state.assertions.map(({ name, passed }) => [name, passed]);

test('simple-forward sets the Simple forward starter and asserts its enabled passes', async () => {
  const { engine, state } = await run('simple-forward');
  assert.equal(state.starters[0].json, SIMPLE);
  assert.deepEqual(engine.calls, [['buildRoom'], ['setRenderGraph', SIMPLE]]);
  assert.deepEqual(verdicts(state), [['simple-forward-accepted', true]]);
});

test('full-gi-pair sets the Full GI graph on the left, then clears and configures the engine on the right', async () => {
  const { engine, state } = await run('full-gi-pair');
  assert.deepEqual(engine.calls, [
    ['buildRoom'],
    ['setRenderGraph', FULL_GI],
    ['setRenderGraph', null],
    // Non-required passes in engine order, enabled when the Full GI graph has their pass node.
    ['setPassEnabled', 'sky.atmosphere', true],
    ['setPassEnabled', 'shadow.csm', true],
    ['setPassEnabled', 'light.tiles', true],
    ['setPassEnabled', 'ao.gtao', true],
    ['setPassEnabled', 'ssr', true],
    ['setPassEnabled', 'dof', false],
    ['setPassEnabled', 'sky.background', true],
    // The S61 Full GI values.
    ['setGlobalIllumination', {
      historyMax: 128, clampGamma: 3, spatialPasses: 4, restirMCap: 30, voxelFullRelight: true, historyMin: 1,
      voxelHistoryMax: 64, voxelRays: 3, debugView: 'none', enabled: true, screenSpace: true,
    }],
    ['setGiQuality', { resolution: 96, levels: 5, probeRays: 8 }],
  ]);
  assert.deepEqual(verdicts(state), [['full-gi-accepted', true], ['full-gi-engine-configured', true]]);
});

test('null-pair records the never-set passes first, then restores them after 30 frames', async () => {
  const { engine, state, frames } = await run('null-pair');
  assert.deepEqual(JSON.parse(state.neverSet), PASSES);
  assert.deepEqual(engine.calls, [['buildRoom'], ['setRenderGraph', FULL_GI], ['setRenderGraph', null]]);
  assert.deepEqual(frames, [30]);
  assert.deepEqual(verdicts(state), [['null-restored', true]]);
});

test('refused keeps the Simple forward graph and records the refused document', async () => {
  const { engine, state } = await run('refused', [ACCEPTED, REFUSED]);
  const graphs = engine.calls.filter(([method]) => method === 'setRenderGraph').map(([, graph]) => graph);
  assert.equal(graphs[0], SIMPLE);
  assert.equal(graphs[1].name, 'Simple forward with invalid AO radius');
  assert.deepEqual(graphs[1].nodes.at(-1), { id: 'aoGtao', type: 'pass.ao.gtao', params: { radius: 25 } });
  assert.deepEqual(verdicts(state), [['refused-kept-previous', true]]);
  assert.deepEqual(state.edits.at(-1), {
    operation: 'setRenderGraph', accepted: false, error: { code: 'graph.param_range', message: 'radius 25 is outside (0, 10]' },
  });
});

test('caption is the refusal line in the refused mode and null in every other mode', async () => {
  const refused = await run('refused', [ACCEPTED, REFUSED]);
  assert.equal(scene.caption(refused.engine, refused.state, refused.variant), CAPTION);
  for (const mode of ['simple-forward', 'full-gi-pair', 'null-pair']) {
    const other = await run(mode);
    assert.equal(scene.caption(other.engine, other.state, other.variant), null, mode);
  }
});

test('each assertion fails on a wrong engine result', async () => {
  const cases = [
    ['simple-forward', [REFUSED], 'simple-forward-accepted'],
    ['full-gi-pair', [REFUSED], 'full-gi-accepted'],
    ['full-gi-pair', [ACCEPTED, REFUSED], 'full-gi-engine-configured'],
    ['null-pair', [ACCEPTED, REFUSED], 'null-restored'],
    ['refused', [ACCEPTED, ACCEPTED], 'refused-kept-previous'],
  ];
  for (const [mode, results, name] of cases) {
    const { state } = await run(mode, results);
    assert.equal(state.assertions.find(item => item.name === name).passed, false, `${mode}: ${name}`);
  }
});

test('a refused setter is recorded as a rejected edit and fails full-gi-engine-configured', async () => {
  const refusal = Object.assign(new Error('quality out of range'), { code: 'gi.invalid' });
  const { state } = await run('full-gi-pair', [], { setGiQuality: () => { throw refusal; } });
  assert.deepEqual(state.edits.at(-1), {
    operation: 'setGiQuality', accepted: false, error: { code: 'gi.invalid', message: 'quality out of range' },
  });
  assert.deepEqual(verdicts(state), [['full-gi-accepted', true], ['full-gi-engine-configured', false]]);
});

test('the observed facts of every inventory row pass validateRequiredFacts', async () => {
  for (const row of inventory) {
    const { engine, state, variant } = await run(row.mode, row.mode === 'refused' ? [ACCEPTED, REFUSED] : []);
    const facts = {
      ...collectRgFacts(engine, observe(engine, variant, state), variant),
      passes: engine.passes(),
      viewSizes: [{ view: 0, width: 1280, height: 720 }],
      readiness: { status: 'ready', stage: 'interpreter', error: null },
    };
    validateRequiredFacts(row, facts, factNames);
  }
});

test('the rows, fact names and pairs are those of contract rg-interpreter-inventory', () => {
  assert.deepEqual(factNames, ['simple-forward-accepted', 'full-gi-accepted', 'full-gi-engine-configured', 'null-restored', 'refused-kept-previous']);
  assert.deepEqual(inventory.map(row => [row.id, row.mode, row.filename, row.pair]), [
    ['interpreter-simple-forward', 'simple-forward', 'interpreter-simple-forward-lit-shadowed-room.png', null],
    ['interpreter-full-gi-pair', 'full-gi-pair', 'interpreter-full-gi-graph-left-engine-default-right.png', null],
    ['interpreter-null-pair', 'null-pair', 'interpreter-null-graph-left-never-set-right-matches.png', null],
    ['interpreter-refused', 'refused', 'interpreter-refused-graph-keeps-simple-forward.png', null],
  ]);
  assert.ok(inventory.every(row => row.heldConstant.join(', ') === 'room geometry, materials, lights, sky, camera'));
  assert.deepEqual(scene.variants.map(variant => variant.pair), [
    null,
    { left: 'graph', right: 'engine default', order: ['left', 'right'] },
    { left: 'null graph', right: 'never set', order: ['right', 'left'] },
    null,
  ]);
});
