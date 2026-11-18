import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as interpreter from './scenes/interpreter.js';
import * as views from './scenes/views.js';
import * as parameters from './scenes/parameters.js';
import * as wiring from './scenes/wiring.js';
import { validateInventory } from './lib/inventory.js';

// The wiring scene against an engine double whose catalog holds the repository starter graphs.
const graph = name => readFileSync(new URL(`../../crates/raa-graph/graphs/${name}.json`, import.meta.url), 'utf8');
const STARTERS = [['Simple forward', 'simple-forward'], ['Full GI', 'full-gi'], ['Debug', 'debug']]
  .map(([name, file]) => ({ name, file: `${file}.json`, json: graph(file) }));
const RESOURCE = { port: 'color', width: 1280, height: 720, layers: 1, mips: 1, format: 'rgba16float', bytes: 1 };
const pass = (name, enabled = true) => ({ name, role: 'pass', enabled, order: 0, required: false });
const stat = (id, type = 'view.image', ran = true) => ({ id, type, ran, resources: ran ? [RESOURCE] : [] });

function engineDouble({ passes = [], nodes = [] } = {}) {
  const documents = [];
  let active = null;
  return {
    documents,
    passList: passes,
    nodes,
    renderGraphCatalog: () => ({ starters: STARTERS }),
    setRenderGraph(doc) { documents.push(doc); active = typeof doc === 'string' ? JSON.parse(doc) : doc; return { ok: true, issues: [], rebuilt: 0 }; },
    renderGraphActive: () => active !== null,
    renderGraph: () => active,
    renderGraphIssues: () => [],
    renderGraphStats() { return { frameIndex: 2, nodes: this.nodes }; },
    passes() { return this.passList; },
  };
}

const variant = id => wiring.default.variants.find(entry => entry.id === id);
async function run(engine, mode, halves = ['left'], between = () => {}) {
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
  const state = await wiring.default.build(engine, lib, variant(mode));
  const frames = [];
  for (const half of halves) {
    await wiring.default.prepareHalf(engine, lib, variant(mode), state, half, { renderFrames: async n => frames.push(n) });
    between(half);
  }
  return { state, frames, observation: wiring.observe(engine, variant(mode), state) };
}
const link = (doc, to) => doc.links.find(entry => entry.to === to)?.from;
const passed = state => state.assertions.map(entry => [entry.name, entry.passed]);

test('the wiring inventory adds five rows after the earlier stages, filenames unique', () => {
  const rows = validateInventory([interpreter, views, parameters, wiring], 'wiring');
  const mine = rows.slice(-5);
  assert.equal(rows.length, rows.length);
  assert.deepEqual(mine.map(row => [row.id, row.mode, row.filename]), [
    ['wiring-default-simple-forward', 'default-simple-forward', 'wiring-default-simple-forward.png'],
    ['wiring-default-full-gi', 'default-full-gi', 'wiring-default-full-gi.png'],
    ['wiring-default-debug', 'default-debug', 'wiring-default-debug.png'],
    ['wiring-bloom-before-dof-pair', 'bloom-before-dof-pair', 'wiring-post-chain-bloom-before-dof-left-default-right.png'],
    ['wiring-two-taa-split', 'two-taa-split', 'wiring-two-taa-instances-compare-split.png'],
  ]);
  assert.equal(new Set(rows.map(row => row.filename)).size, rows.length);
  assert.deepEqual(variant('bloom-before-dof-pair').pair, { left: 'bloom before dof', right: 'default order', order: ['left', 'right'] });
});

test('the default modes set the starter texts unchanged', async () => {
  for (const [mode, index, fact] of [['default-simple-forward', 0, 'simple-forward-wired'], ['default-full-gi', 1, 'full-gi-wired']]) {
    const engine = engineDouble();
    const { frames, state } = await run(engine, mode);
    assert.deepEqual(engine.documents, [STARTERS[index].json]);
    assert.deepEqual(frames, [2]);
    assert.deepEqual(passed(state), [[fact, true]]);
  }
});

test('default-debug passes only when every view node ran with a resource', async () => {
  const engine = engineDouble({ nodes: [stat('vDepth'), stat('grid', 'view.grid')] });
  assert.deepEqual(engine.documents, []);
  assert.deepEqual(passed((await run(engine, 'default-debug')).state), [['debug-wired', true]]);
  assert.deepEqual(engine.documents, [STARTERS[2].json]);
  const bad = engineDouble({ nodes: [stat('vDepth', 'view.image', false)] });
  assert.deepEqual(passed((await run(bad, 'default-debug')).state), [['debug-wired', false]]);
});

test('bloom-before-dof-pair sends the two reordered documents', async () => {
  const engine = engineDouble({ passes: [pass('bloom'), pass('dof')] });
  await run(engine, 'bloom-before-dof-pair', ['left', 'right']);
  const [left, right] = engine.documents;
  for (const doc of [left, right]) {
    assert.deepEqual(doc.nodes.find(node => node.id === 'bloom').params, { strength: 0.6, radius: 1 });
    assert.deepEqual(doc.nodes.find(node => node.id === 'dof'), { id: 'dof', type: 'pass.dof', params: { focusDistance: 2, fStop: 0.7, focalLength: 85, maxCoc: 48 } });
  }
  assert.deepEqual(['bloom.color', 'dof.color', 'tonemap.color'].map(to => link(left, to)), ['taa.color', 'bloom.color', 'dof.color']);
  assert.deepEqual(['dof.color', 'bloom.color', 'tonemap.color'].map(to => link(right, to)), ['taa.color', 'dof.color', 'bloom.color']);
  assert.equal(left.links.length, JSON.parse(STARTERS[1].json).links.length + 1);
});

test('bloom-before-dof-order follows the passes read after each half', async () => {
  const orders = { left: [pass('bloom'), pass('dof')], right: [pass('dof'), pass('bloom')] };
  const engine = engineDouble({ passes: orders.left });
  const good = await run(engine, 'bloom-before-dof-pair', ['left', 'right'], half => { if (half === 'left') engine.passList = orders.right; });
  assert.deepEqual(passed(good.state), [['bloom-before-dof-order', true]]);
  const same = engineDouble({ passes: orders.left });
  assert.deepEqual(passed((await run(same, 'bloom-before-dof-pair', ['left', 'right'])).state), [['bloom-before-dof-order', false]]);
});

test('two-taa-split sends the taaB and cmp graph', async () => {
  const engine = engineDouble();
  await run(engine, 'two-taa-split');
  const doc = engine.documents[0];
  assert.ok(!doc.nodes.some(node => node.id === 'bloom'));
  assert.ok(!doc.links.some(entry => /^bloom\./.test(entry.from) || /^bloom\./.test(entry.to)));
  assert.deepEqual(doc.nodes.find(node => node.id === 'taaB'), { id: 'taaB', type: 'pass.taa', params: { blendBase: 1, blendMin: 1, blendMax: 1 } });
  assert.deepEqual(doc.nodes.find(node => node.id === 'cmp'), { id: 'cmp', type: 'view.compare', params: { mode: 'split', position: 0.5 } });
  assert.deepEqual(['taaB.color', 'cmp.a', 'cmp.b', 'tonemap.color'].map(to => link(doc, to)), ['fog.color', 'taa.color', 'taaB.color', 'cmp.color']);
});

test('two-taa-instances-ran needs both entries and the cmp node', async () => {
  const nodes = [stat('cmp', 'view.compare')];
  const good = await run(engineDouble({ passes: [pass('taa'), pass('taa#taaB')], nodes }), 'two-taa-split');
  assert.deepEqual(passed(good.state), [['two-taa-instances-ran', true]]);
  const missing = await run(engineDouble({ passes: [pass('taa')], nodes }), 'two-taa-split');
  assert.deepEqual(passed(missing.state), [['two-taa-instances-ran', false]]);
  const off = await run(engineDouble({ passes: [pass('taa'), pass('taa#taaB', false)], nodes }), 'two-taa-split');
  assert.deepEqual(passed(off.state), [['two-taa-instances-ran', false]]);
  const idle = await run(engineDouble({ passes: [pass('taa'), pass('taa#taaB')], nodes: [stat('cmp', 'view.compare', false)] }), 'two-taa-split');
  assert.deepEqual(passed(idle.state), [['two-taa-instances-ran', false]]);
});

test('observe returns the SceneObservation fields', async () => {
  const { observation } = await run(engineDouble(), 'default-simple-forward');
  assert.deepEqual(Object.keys(observation).sort(), ['assertions', 'edits', 'graph', 'issues', 'renderGraphActive']);
  assert.equal(observation.graph, 'Simple forward');
  assert.deepEqual(observation.edits, [{ operation: 'setRenderGraph', accepted: true, error: null }]);
});
