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
    added: [],
    addMaterial(desc) { this.added.push(['material', desc]); return 7; },
    primitiveMesh(kind, params) { this.added.push(['mesh', kind, params]); return 8; },
    addNode(parent, position) { this.added.push(['node', position]); return 9; },
    addInstance(node, mesh, material) { this.added.push(['instance', node, mesh, material]); return 10; },
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
    assert.deepEqual(doc.nodes.find(node => node.id === 'bloom').params, { strength: 0.35, radius: 1 });
    assert.deepEqual(doc.nodes.find(node => node.id === 'dof'), { id: 'dof', type: 'pass.dof', params: { focusDistance: 2, fStop: 0.7, focalLength: 85, maxCoc: 48 } });
  }
  assert.deepEqual(['bloom.color', 'dof.color', 'tonemap.color'].map(to => link(left, to)), ['taa.color', 'bloom.color', 'dof.color']);
  assert.deepEqual(['dof.color', 'bloom.color', 'tonemap.color'].map(to => link(right, to)), ['taa.color', 'dof.color', 'bloom.color']);
  assert.equal(left.links.length, JSON.parse(STARTERS[1].json).links.length + 1);
});

// The two orders differ only on small bright emitters out of focus (bloom and dof are linear in
// colour): the pair adds them once, beyond the dof focus distance and inside the capture view.
test('bloom-before-dof-pair adds bright emitters out of focus at the panel, other modes none', async () => {
  const engine = engineDouble({ passes: [pass('bloom'), pass('dof')] });
  await run(engine, 'bloom-before-dof-pair', ['left', 'right']);
  const instances = engine.added.filter(entry => entry[0] === 'instance');
  assert.equal(instances.length, wiring.PAIR_EMITTERS.positions.length);
  const [, material] = engine.added.find(entry => entry[0] === 'material');
  assert.ok(Math.min(...material.emissive) > 40, 'the emitters are far brighter than the panel (4, 3, 1.5)');
  const { camera } = variant('bloom-before-dof-pair').selection;
  const focus = engine.documents.map(docOf)[0].nodes.find(node => node.id === 'dof').params.focusDistance;
  for (const [, position] of engine.added.filter(entry => entry[0] === 'node')) {
    const distance = Math.hypot(...position.map((v, i) => v - camera.eye[i]));
    assert.ok(distance > 3 * focus, `emitter at ${position} is out of focus (${distance.toFixed(1)} m)`);
  }
  for (const mode of ['default-full-gi', 'two-taa-split']) {
    const other = engineDouble({ passes: [pass('taa'), pass('taa#taaB')] });
    await run(other, mode);
    assert.deepEqual(other.added, [], `${mode} adds no emitter`);
  }
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
  assert.deepEqual(doc.nodes.find(node => node.id === 'taaB'), { id: 'taaB', type: 'pass.taa', params: { blendBase: 1, blendMin: 1, blendMax: 1, filterSharpness: 8 } });
  // Single-sample forward on an msaa 1 engine: msaa 4 would smooth the edges before either TAA.
  assert.equal(doc.nodes.find(node => node.type === 'pass.forward').params.msaa, 1);
  assert.deepEqual(variant('two-taa-split').engine, { msaa: 1 });
  assert.equal(variant('default-full-gi').engine, undefined);
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

// The graph documents of the wiring stage must be complete for the Rust validator: every colour
// input of a chain pass is linked (`generic` colour inputs read no default source since M6), and
// the view nodes each mode shows are present. The engine's acceptance itself is checked by the GPU
// run; this test checks the documents each mode sends and that the engine accepted them.
const CHAIN_TYPES = [
  'pass.gi.history', 'pass.ssr', 'pass.fog', 'pass.sunshafts', 'pass.opaque.copy', 'pass.transparent.oit',
  'pass.particles.blend', 'pass.taa', 'pass.dof', 'pass.bloom', 'pass.grading', 'pass.cel.resolve', 'pass.tonemap',
];
const DEBUG_VIEWS = ['vDepth', 'vNormal', 'vMotion', 'vAo', 'vAreaMask', 'giRawVsDenoised', 'vHistory', 'vFinal', 'grid'];
const docOf = sent => (typeof sent === 'string' ? JSON.parse(sent) : sent);
const unlinkedColour = doc => {
  const linked = new Set(doc.links.map(entry => entry.to));
  return doc.nodes.filter(node => CHAIN_TYPES.includes(node.type) && !linked.has(`${node.id}.color`)).map(node => node.id);
};
const EXPECTED_VIEWS = {
  'default-simple-forward': [],
  'default-full-gi': [],
  'default-debug': DEBUG_VIEWS,
  'bloom-before-dof-pair': [],
  'two-taa-split': ['cmp'],
};

test('every wiring mode sends accepted documents with every colour input linked and its view nodes', async () => {
  for (const scene of wiring.default.variants) {
    const engine = engineDouble({ passes: [pass('taa'), pass('taa#taaB'), pass('bloom'), pass('dof')],
      nodes: DEBUG_VIEWS.map(id => stat(id)).concat([stat('cmp', 'view.compare')]) });
    const halves = scene.pair ? scene.pair.order : ['single'];
    const { observation } = await run(engine, scene.id, halves);
    assert.ok(engine.documents.length > 0, `${scene.id} sends a graph`);
    assert.ok(observation.edits.every(edit => edit.accepted === true), `${scene.id} accepted: ${JSON.stringify(observation.edits)}`);
    for (const sent of engine.documents) {
      const doc = docOf(sent);
      assert.deepEqual(unlinkedColour(doc), [], `${scene.id}: unlinked colour inputs in ${doc.name}`);
      const ids = new Set(doc.nodes.map(node => node.id));
      for (const view of EXPECTED_VIEWS[scene.id]) assert.ok(ids.has(view), `${scene.id}: ${doc.name} has view node ${view}`);
    }
  }
});

test('a wiring document with an unlinked colour input is reported', () => {
  const doc = docOf(STARTERS[2].json);
  assert.deepEqual(unlinkedColour(doc), []);
  const cut = { ...doc, links: doc.links.filter(entry => entry.to !== 'fog.color') };
  assert.deepEqual(unlinkedColour(cut), ['fog']);
});
