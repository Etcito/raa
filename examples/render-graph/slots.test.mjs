import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as interpreter from './scenes/interpreter.js';
import * as views from './scenes/views.js';
import * as parameters from './scenes/parameters.js';
import * as wiring from './scenes/wiring.js';
import * as slots from './scenes/slots.js';
import { validateInventory } from './lib/inventory.js';
import { stageInventory } from '../../web/common/render-graph-job.mjs';

// The slots scene against an engine double whose catalog holds the repository starter graphs.
const graph = name => readFileSync(new URL(`../../crates/raa-graph/graphs/${name}.json`, import.meta.url), 'utf8');
const STARTERS = [['Simple forward', 'simple-forward'], ['Full GI', 'full-gi']]
  .map(([name, file]) => ({ name, file: `${file}.json`, json: graph(file) }));
const RESOURCE = { port: 'color', width: 1280, height: 720, layers: 1, mips: 1, format: 'rgba16float', bytes: 1 };
const pass = (name, enabled = true) => ({ name, role: 'pass', enabled, order: 0, required: false });
const stat = (id, type = 'view.image', ran = true) => ({ id, type, ran, resources: ran ? [RESOURCE] : [] });

function engineDouble({ passes = [], nodes = [] } = {}) {
  const documents = [];
  let active = null;
  return {
    documents,
    renderGraphCatalog: () => ({ starters: STARTERS }),
    setRenderGraph(doc) { documents.push(doc); active = typeof doc === 'string' ? JSON.parse(doc) : doc; return { ok: true, issues: [], rebuilt: 0 }; },
    renderGraphActive: () => active !== null,
    renderGraph: () => active,
    renderGraphIssues: () => [],
    renderGraphStats: () => ({ frameIndex: 2, nodes }),
    passes: () => passes,
  };
}

const variant = id => slots.default.variants.find(entry => entry.id === id);
async function run(engine, mode, halves = ['left']) {
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
  const state = await slots.default.build(engine, lib, variant(mode));
  const frames = [];
  for (const half of halves) {
    await slots.default.prepareHalf(engine, lib, variant(mode), state, half, { renderFrames: async n => frames.push(n) });
  }
  return { state, frames, observation: slots.observe(engine, variant(mode), state) };
}
const passed = state => state.assertions.map(entry => [entry.name, entry.passed]);
const linkTo = (doc, to) => doc.links.find(entry => entry.to === to)?.from;
const docOf = sent => (typeof sent === 'string' ? JSON.parse(sent) : sent);

// The contract document slots-two-ao-doc, written out.
const SLOTS_TWO_AO_DOC = {
  format: 'raa.render-graph', version: 1, name: 'Simple forward two AO branches',
  nodes: [
    { id: 'skyAtmosphere', type: 'pass.sky.atmosphere' },
    { id: 'skyCloudLighting', type: 'pass.sky.cloud-lighting' },
    { id: 'skyEnv', type: 'pass.sky.env' },
    { id: 'shadowCsm', type: 'pass.shadow.csm' },
    { id: 'prepass', type: 'pass.prepass' },
    { id: 'lightTiles', type: 'pass.light.tiles' },
    { id: 'aoNarrow', type: 'pass.ao.gtao', params: { radius: 0.3, intensity: 1, power: 2 } },
    { id: 'aoWide', type: 'pass.ao.gtao', params: { radius: 2, intensity: 1, power: 2 } },
    { id: 'forward', type: 'pass.forward' },
    { id: 'skyBackground', type: 'pass.sky.background' },
    { id: 'tonemap', type: 'pass.tonemap', params: { mapper: 'agx', exposureEv: 0 } },
    { id: 'vNarrow', type: 'view.image', params: { channels: 'r', label: 'ao 0.3 m' } },
    { id: 'vWide', type: 'view.image', params: { channels: 'r', label: 'ao 2 m' } },
    { id: 'aoNarrowVsWide', type: 'view.compare', params: { mode: 'split', position: 0.5 } },
    { id: 'vFinal', type: 'view.image', params: { label: 'final, forward ao from 2 m' } },
    { id: 'grid', type: 'view.grid', params: { columns: 2 } },
    { id: 'out', type: 'output.display' },
  ],
  links: [
    ['prepass.depth', 'lightTiles.depth'], ['shadowCsm.shadowMap', 'forward.shadowMap'],
    ['lightTiles.lightTiles', 'forward.lightTiles'], ['aoWide.ao', 'forward.ao'], ['forward.color', 'tonemap.color'],
    ['aoNarrow.ao', 'vNarrow.src'], ['aoWide.ao', 'vWide.src'], ['vNarrow.color', 'aoNarrowVsWide.a'],
    ['vWide.color', 'aoNarrowVsWide.b'], ['tonemap.color', 'vFinal.src'], ['aoNarrowVsWide.color', 'grid.t0'],
    ['vFinal.color', 'grid.t1'], ['grid.color', 'out.color'],
  ].map(([from, to]) => ({ from, to })),
};

test('the slots stage selects exactly the four slots images, never the wiring rows', () => {
  const rows = stageInventory([interpreter, views, parameters, wiring, slots], 'slots', validateInventory);
  assert.deepEqual(rows.map(row => row.filename), [
    'slots-two-ao-branches-compare-split.png',
    'slots-forward-ao-from-wide-radius-branch.png',
    'slots-default-full-gi.png',
    'slots-default-simple-forward.png',
  ]);
  assert.deepEqual(rows.map(row => row.scenario), Array(4).fill('render-graph-slots'));
  assert.ok(rows.every(row => !row.filename.startsWith('wiring-')));
});

test('the slots inventory adds four rows after the earlier stages, filenames unique', () => {
  const rows = validateInventory([interpreter, views, parameters, wiring, slots], 'slots');
  assert.deepEqual(rows.slice(-4).map(row => [row.id, row.mode, row.filename]), [
    ['slots-two-ao-split', 'two-ao-split', 'slots-two-ao-branches-compare-split.png'],
    ['slots-forward-ao-pair', 'forward-ao-pair', 'slots-forward-ao-from-wide-radius-branch.png'],
    ['slots-default-full-gi', 'default-full-gi', 'slots-default-full-gi.png'],
    ['slots-default-simple-forward', 'default-simple-forward', 'slots-default-simple-forward.png'],
  ]);
  assert.equal(new Set(rows.map(row => row.filename)).size, rows.length);
  assert.deepEqual(variant('forward-ao-pair').pair, { left: 'forward ao from 2 m branch', right: 'forward ao from 0.3 m branch', order: ['left', 'right'] });
});

test('two-ao-split sends one document whose nodes and links equal slots-two-ao-doc', async () => {
  const engine = engineDouble({ passes: [pass('ao.gtao'), pass('ao.gtao#aoWide')] });
  await run(engine, 'two-ao-split');
  assert.equal(engine.documents.length, 1);
  assert.deepEqual(engine.documents[0], SLOTS_TWO_AO_DOC);
});

test('forward-ao-pair sends two halves whose forward.ao reads the wide and the narrow branch', async () => {
  const engine = engineDouble({ passes: [pass('ao.gtao')] });
  await run(engine, 'forward-ao-pair', ['left', 'right']);
  const feed = doc => {
    const from = linkTo(doc, 'forward.ao');
    const node = doc.nodes.find(entry => entry.id === from.split('.')[0]);
    return [from, node.params.radius];
  };
  assert.deepEqual(engine.documents.map(feed), [['aoWide.ao', 2], ['aoNarrow.ao', 0.3]]);
  assert.deepEqual(engine.documents.map(doc => doc.name), ['Simple forward forward ao from wide', 'Simple forward forward ao from narrow']);
});

test('the default modes send the starter texts unchanged', async () => {
  for (const [mode, index, fact] of [['default-simple-forward', 0, 'simple-forward-slots'], ['default-full-gi', 1, 'full-gi-slots']]) {
    const engine = engineDouble();
    const { frames, state } = await run(engine, mode);
    assert.deepEqual(engine.documents, [STARTERS[index].json]);
    assert.deepEqual(frames, [2]);
    assert.deepEqual(passed(state), [[fact, true]]);
  }
});

test('two-ao-branches-ran needs both AO instances and the compare and grid nodes ran', async () => {
  const both = [pass('ao.gtao'), pass('ao.gtao#aoWide')];
  const nodes = [stat('aoNarrowVsWide', 'view.compare'), stat('grid', 'view.grid')];
  assert.deepEqual(passed((await run(engineDouble({ passes: both, nodes }), 'two-ao-split')).state), [['two-ao-branches-ran', true]]);
  assert.deepEqual(passed((await run(engineDouble({ passes: [pass('ao.gtao')], nodes }), 'two-ao-split')).state), [['two-ao-branches-ran', false]]);
  const off = [pass('ao.gtao'), pass('ao.gtao#aoWide', false)];
  assert.deepEqual(passed((await run(engineDouble({ passes: off, nodes }), 'two-ao-split')).state), [['two-ao-branches-ran', false]]);
  const idle = [stat('aoNarrowVsWide', 'view.compare', false), stat('grid', 'view.grid')];
  assert.deepEqual(passed((await run(engineDouble({ passes: both, nodes: idle }), 'two-ao-split')).state), [['two-ao-branches-ran', false]]);
});

test('forward-follows-ao-link needs both halves accepted with ao.gtao enabled', async () => {
  const good = await run(engineDouble({ passes: [pass('ao.gtao')] }), 'forward-ao-pair', ['left', 'right']);
  assert.deepEqual(passed(good.state), [['forward-follows-ao-link', true]]);
  const none = await run(engineDouble(), 'forward-ao-pair', ['left', 'right']);
  assert.deepEqual(passed(none.state), [['forward-follows-ao-link', false]]);
});

// The wiring test's rule: every chain colour input of a sent document is linked.
const CHAIN_TYPES = [
  'pass.gi.history', 'pass.ssr', 'pass.fog', 'pass.sunshafts', 'pass.opaque.copy', 'pass.transparent.oit',
  'pass.particles.blend', 'pass.taa', 'pass.dof', 'pass.bloom', 'pass.grading', 'pass.cel.resolve', 'pass.tonemap',
];
const unlinkedColour = doc => {
  const linked = new Set(doc.links.map(entry => entry.to));
  return doc.nodes.filter(node => CHAIN_TYPES.includes(node.type) && !linked.has(`${node.id}.color`)).map(node => node.id);
};

test('every slots mode sends accepted documents with every chain colour input linked', async () => {
  for (const scene of slots.default.variants) {
    const engine = engineDouble({ passes: [pass('ao.gtao'), pass('ao.gtao#aoWide')],
      nodes: [stat('aoNarrowVsWide', 'view.compare'), stat('grid', 'view.grid')] });
    const halves = scene.pair ? scene.pair.order : ['left'];
    const { observation } = await run(engine, scene.id, halves);
    assert.ok(engine.documents.length > 0, `${scene.id} sends a graph`);
    assert.ok(observation.edits.every(edit => edit.accepted === true), `${scene.id} accepted`);
    for (const sent of engine.documents) assert.deepEqual(unlinkedColour(docOf(sent)), [], `${scene.id}: unlinked colour inputs`);
  }
});

test('observe returns the SceneObservation fields', async () => {
  const { observation } = await run(engineDouble(), 'default-simple-forward');
  assert.deepEqual(Object.keys(observation).sort(), ['assertions', 'edits', 'graph', 'issues', 'renderGraphActive']);
  assert.equal(observation.graph, 'Simple forward');
  assert.deepEqual(observation.edits, [{ operation: 'setRenderGraph', accepted: true, error: null }]);
});
