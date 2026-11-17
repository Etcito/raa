import test from 'node:test';
import assert from 'node:assert/strict';
import * as interpreter from './scenes/interpreter.js';
import * as views from './scenes/views.js';
import { validateInventory } from './lib/inventory.js';

// The views scene (contract views-scene) against an engine double whose catalog holds fixture
// starter texts and whose stats list a configurable node list.
const DEBUG_TEXT = JSON.stringify({
  format: 'raa.render-graph', version: 1, name: 'Debug',
  nodes: [{ id: 'grid', type: 'view.grid' }, { id: 'out', type: 'output.display' }],
  links: [{ from: 'grid.color', to: 'out.color' }],
});
const FULL_GI_TEXT = JSON.stringify({
  format: 'raa.render-graph', version: 1, name: 'Full GI',
  nodes: [
    { id: 'aoGtao', type: 'pass.ao.gtao' }, { id: 'giVolume', type: 'pass.gi.volume', params: { intensity: 1 } },
    { id: 'giSsgi', type: 'pass.gi.ssgi' }, { id: 'tonemap', type: 'pass.tonemap' }, { id: 'out', type: 'output.display' },
  ],
  links: [{ from: 'tonemap.color', to: 'out.color' }],
});
const DEBUG_VIEWS = ['vDepth', 'vNormal', 'vMotion', 'vAo', 'vAreaMask', 'giRawVsDenoised', 'vHistory', 'vFinal', 'grid'];
const RESOURCE = { port: 'color', width: 1280, height: 720, layers: 1, mips: 1, format: 'rgba16float', bytes: 7372800 };
const viewNode = (id, ran = true) => ({ id, type: id === 'grid' ? 'view.grid' : 'view.image', ran, resources: ran ? [RESOURCE] : [] });
const PASS_NODES = [{ id: 'tonemap', type: 'pass.tonemap', ran: true, resources: [] }, { id: 'out', type: 'output.display', ran: true, resources: [] }];

function engineDouble(nodes = []) {
  const documents = [];
  let active = null;
  return {
    documents,
    nodes,
    renderGraphCatalog: () => ({
      starters: [{ name: 'Debug', file: 'debug.json', json: DEBUG_TEXT }, { name: 'Full GI', file: 'full-gi.json', json: FULL_GI_TEXT }],
    }),
    setRenderGraph(graph) {
      documents.push(graph);
      active = typeof graph === 'string' ? JSON.parse(graph) : graph;
      return { ok: true, issues: [], rebuilt: 0 };
    },
    renderGraphActive: () => active !== null,
    renderGraph: () => active,
    renderGraphIssues: () => [],
    renderGraphStats() { return { frameIndex: 2, nodes: this.nodes }; },
    passes: () => [],
  };
}

const variant = id => views.default.variants.find(entry => entry.id === id);
async function run(engine, mode, halves = ['left']) {
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
  const state = await views.default.build(engine, lib);
  const frames = [];
  for (const half of halves) {
    await views.default.prepareHalf(engine, lib, variant(mode), state, half, { renderFrames: async n => frames.push(n) });
  }
  return { state, frames, observation: views.observe(engine, variant(mode), state) };
}
const docOf = value => (typeof value === 'string' ? JSON.parse(value) : value);

test('the views inventory adds the three contract rows after the interpreter rows', () => {
  const rows = validateInventory([interpreter, views], 'views');
  assert.equal(rows.length, 7);
  assert.deepEqual(rows.slice(4).map(row => [row.id, row.mode, row.filename]), [
    ['views-debug-grid', 'debug-grid', 'views-debug-grid-every-tile-readable.png'],
    ['views-show-in-view-ao', 'show-in-view-ao', 'views-show-in-view-ao-tap.png'],
    ['views-final-direct-pair', 'final-direct-pair', 'views-final-left-direct-light-only-right.png'],
  ]);
  assert.equal(new Set(rows.map(row => row.filename)).size, 7);
  assert.throws(() => validateInventory([interpreter], 'views'), /stage views is unavailable/);
});

test('debug-grid sets the Debug starter text and renders two frames', async () => {
  const engine = engineDouble();
  const { frames, observation } = await run(engine, 'debug-grid');
  assert.deepEqual(engine.documents, [DEBUG_TEXT]);
  assert.deepEqual(frames, [2]);
  assert.deepEqual(observation.edits, [{ operation: 'setRenderGraph', accepted: true, error: null }]);
  assert.equal(observation.renderGraphActive, true);
});

test('show-in-view-ao sets Full GI with the AO view linked to out', async () => {
  const engine = engineDouble();
  await run(engine, 'show-in-view-ao');
  const doc = docOf(engine.documents[0]);
  assert.equal(doc.name, 'Full GI with AO in view');
  const index = doc.nodes.findIndex(node => node.id === 'show_aoGtao_ao');
  assert.deepEqual(doc.nodes[index], { id: 'show_aoGtao_ao', type: 'view.image', params: { channels: 'r', label: 'ao' } });
  assert.equal(doc.nodes[index + 1].id, 'out');
  const links = doc.links.map(link => `${link.from} -> ${link.to}`);
  assert.ok(links.includes('aoGtao.ao -> show_aoGtao_ao.src'));
  assert.ok(links.includes('show_aoGtao_ao.color -> out.color'));
  assert.ok(!links.includes('tonemap.color -> out.color'));
});

test('final-direct-pair sets Full GI, then GI intensity 0 without the SSGI node', async () => {
  const engine = engineDouble();
  const { state } = await run(engine, 'final-direct-pair', ['left', 'right']);
  assert.deepEqual(engine.documents[0], FULL_GI_TEXT);
  const right = docOf(engine.documents[1]);
  assert.equal(right.name, 'Full GI direct light only');
  assert.equal(right.nodes.find(node => node.id === 'giVolume').params.intensity, 0);
  assert.ok(!right.nodes.some(node => node.id === 'giSsgi'));
  assert.ok(!right.links.some(link => link.from.startsWith('giSsgi.') || link.to.startsWith('giSsgi.')));
  assert.deepEqual(state.assertions.map(a => [a.name, a.passed]), [['final-pair-accepted', true]]);
});

test('debug-views-ran passes when the nine views ran with a resource and fails naming a view that did not', async () => {
  const good = await run(engineDouble([...PASS_NODES, ...DEBUG_VIEWS.map(id => viewNode(id))]), 'debug-grid');
  assert.equal(good.state.assertions[0].name, 'debug-views-ran');
  assert.equal(good.state.assertions[0].passed, true);
  const bad = await run(engineDouble([...PASS_NODES, ...DEBUG_VIEWS.map(id => viewNode(id, id !== 'vAo'))]), 'debug-grid');
  assert.equal(bad.state.assertions[0].passed, false);
  assert.match(bad.state.assertions[0].detail, /vAo/);
});

test('ao-view-ran passes when the AO view ran with a resource and fails naming it when it did not', async () => {
  const good = await run(engineDouble([...PASS_NODES, viewNode('show_aoGtao_ao')]), 'show-in-view-ao');
  assert.deepEqual([good.state.assertions[0].name, good.state.assertions[0].passed], ['ao-view-ran', true]);
  const bad = await run(engineDouble([...PASS_NODES, viewNode('show_aoGtao_ao', false)]), 'show-in-view-ao');
  assert.equal(bad.state.assertions[0].passed, false);
  assert.match(bad.state.assertions[0].detail, /show_aoGtao_ao/);
});

test('final-pair-accepted fails when the right half graph is refused', async () => {
  const engine = engineDouble();
  const set = engine.setRenderGraph.bind(engine);
  engine.setRenderGraph = graph => (engine.documents.length ? (set(graph), { ok: false, issues: [], rebuilt: 0 }) : set(graph));
  const { state } = await run(engine, 'final-direct-pair', ['left', 'right']);
  assert.equal(state.assertions[0].passed, false);
  assert.match(state.assertions[0].detail, /direct light only ok false/);
});
