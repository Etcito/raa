import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as interpreter from './scenes/interpreter.js';
import * as views from './scenes/views.js';
import * as parameters from './scenes/parameters.js';
import * as wiring from './scenes/wiring.js';
import * as slots from './scenes/slots.js';
import * as giStages from './scenes/gi-stages.js';
import { validateInventory } from './lib/inventory.js';
import { stageInventory } from '../../web/common/render-graph-job.mjs';

const graph = name => readFileSync(new URL(`../../crates/raa-graph/graphs/${name}.json`, import.meta.url), 'utf8');
const STARTERS = [['Full GI stages', 'full-gi-stages'], ['Full GI', 'full-gi']]
  .map(([name, file]) => ({ name, file: `${file}.json`, json: graph(file) }));
const resource = { port: 'output', width: 1280, height: 720, layers: 1, mips: 1, format: 'rgba16float', bytes: 1 };
const pass = (name, enabled = true) => ({ name, role: 'pass', enabled, order: 0, required: false });
const stat = (id, ran = true) => ({ id, type: 'gi.restir.temporal', ran, resources: ran ? [resource] : [] });

function engineDouble({ passes = [], nodes = [] } = {}) {
  const documents = [];
  let active = null, currentPasses = passes, currentNodes = nodes;
  return {
    documents,
    setPasses(value) { currentPasses = value; },
    set nodes(value) { currentNodes = value; },
    renderGraphCatalog: () => ({ starters: STARTERS }),
    setRenderGraph(doc) {
      documents.push(doc);
      active = typeof doc === 'string' ? JSON.parse(doc) : doc;
      return { ok: true, issues: [], rebuilt: 0 };
    },
    renderGraphActive: () => active !== null,
    renderGraph: () => active,
    renderGraphIssues: () => [],
    renderGraphStats: () => ({ frameIndex: 2, nodes: currentNodes }),
    passes: () => currentPasses,
  };
}
const variant = id => giStages.default.variants.find(entry => entry.id === id);
async function setup(engine, mode, halves = ['left']) {
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
  const state = await giStages.default.build(engine, lib);
  const frames = [];
  for (const half of halves) await giStages.default.prepareHalf(engine, lib, variant(mode), state, half,
    { renderFrames: async count => frames.push(count) });
  return { state, frames, observation: giStages.observe(engine, variant(mode), state) };
}
const assertions = state => state.assertions.map(({ name, passed }) => [name, passed]);
const docOf = doc => typeof doc === 'string' ? JSON.parse(doc) : doc;
const link = (doc, from, to) => doc.links.some(entry => entry.from === from && entry.to === to);
const ownRows = () => validateInventory([interpreter, views, parameters, wiring, slots, giStages], 'gi-stages').slice(-5);

test('GI stages inventory rows satisfy the contract and stage selection', () => {
  const rows = ownRows();
  assert.deepEqual(rows.map(({ id, mode, filename, criterionName, requiredFacts, producer }) =>
    [id, mode, filename, criterionName, requiredFacts, producer]), [
    ['gi-stages-full-gi-pair', 'full-gi-pair', 'gi-stages-full-gi-from-stages-left-pass-graph-right.png',
      'P5-Q10 Full GI from stage nodes matches the pass-level graph',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:stages-and-pass-graph-accepted'],
      'raa-graph exec gi_stages 3/3: graph frame binding stage inputs, executor-owned histories and forward slots'],
    ['gi-stages-spatial-skipped', 'spatial-skipped', 'gi-stages-spatial-skipped-raw-noisier.png',
      'P5-Q10 skipping the spatial stage shows the raw, noisier result',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:spatial-skipped-ran'],
      'raa-gi stages 7/11: gi.restir.spatial stage and gi.accumulate irradiance mode'],
    ['gi-stages-atrous-pair', 'atrous-pair', 'gi-stages-two-atrous-left-four-atrous-right.png',
      'P5-Q10 the a-trous count is the number of gi.atrous nodes',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:atrous-count-follows-nodes'],
      'raa-gi stages 4/11: gi.atrous and gi.resolve stages with the resolve kernel'],
    ['gi-stages-area-no-spatial', 'area-no-spatial', 'gi-stages-area-shadow-without-spatial.png',
      'P5-Q10 area shadow stages can be removed',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:area-without-spatial-ran'],
      'raa-gi stages 9/11: area.spatial and area.mask stages'],
    ['gi-stages-ssgi-raw', 'ssgi-raw', 'gi-stages-ssgi-without-resolve-raw.png',
      'P5-Q10 SSGI stages can be removed',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:ssgi-raw-ran'],
      'raa-gi stages 11/11: ssgi.trace and ssgi.resolve stages'],
  ]);
  assert.equal(new Set(rows.map(row => row.filename)).size, rows.length);
  assert.deepEqual(stageInventory([interpreter, views, parameters, wiring, slots, giStages], 'gi-stages', validateInventory)
    .map(row => row.filename), rows.map(row => row.filename));
});

test('full-gi-pair uses the exact starter texts and requires the stage and pass graph passes', async () => {
  const engine = engineDouble({ passes: [pass('gi.pyramid#giPyramid')] });
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
  const state = await giStages.default.build(engine, lib);
  await giStages.default.prepareHalf(engine, lib, variant('full-gi-pair'), state, 'left', { renderFrames: async () => {} });
  engine.setPasses([pass('gi.screen')]);
  await giStages.default.prepareHalf(engine, lib, variant('full-gi-pair'), state, 'right', { renderFrames: async () => {} });
  assert.deepEqual(engine.documents.map(docOf), STARTERS.map(entry => JSON.parse(entry.json)));
  assert.deepEqual(assertions(state), [['stages-and-pass-graph-accepted', true]]);
  const missing = await setup(engineDouble({ passes: [pass('gi.screen')] }), 'full-gi-pair', ['left', 'right']);
  assert.deepEqual(assertions(missing.state), [['stages-and-pass-graph-accepted', false]]);
});

test('spatial-skipped removes spatial work, inserts ReSTIR links and checks ran stats', async () => {
  const good = engineDouble({ nodes: [stat('restirTemporal'), stat('giAccumulate')] });
  const result = await setup(good, 'spatial-skipped');
  const doc = docOf(good.documents[0]);
  assert.equal(doc.name, 'Full GI stages spatial skipped');
  assert.ok(doc.nodes.some(node => node.id === 'restirInitial') && doc.nodes.some(node => node.id === 'restirTemporal'));
  assert.ok(!doc.nodes.some(node => ['giProbes', 'giAtrous1', 'giAtrous2', 'giAtrous3', 'giAtrous4'].includes(node.id)));
  for (const [from, to] of [['giPyramid.pyramid', 'restirInitial.pyramid'], ['restirInitial.reservoirs', 'restirTemporal.initial'],
    ['restirTemporal.irradiance', 'giAccumulate.irradiance'], ['giAccumulate.output', 'giResolve.irradiance']]) assert.ok(link(doc, from, to));
  assert.deepEqual(assertions(result.state), [['spatial-skipped-ran', true]]);
  good.nodes = [stat('restirTemporal', false), stat('giAccumulate')];
  assert.deepEqual(assertions((await setup(good, 'spatial-skipped')).state), [['spatial-skipped-ran', false]]);
});

test('atrous pair removes only the last two filters and checks pass counts', async () => {
  const engine = engineDouble({ passes: [pass('gi.atrous#one'), pass('gi.atrous#two')] });
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } }, state = await giStages.default.build(engine, lib);
  await giStages.default.prepareHalf(engine, lib, variant('atrous-pair'), state, 'left', { renderFrames: async () => {} });
  let doc = docOf(engine.documents[0]);
  assert.deepEqual(doc.nodes.filter(node => node.id.startsWith('giAtrous')).map(node => node.id), ['giAtrous1', 'giAtrous2']);
  assert.ok(link(doc, 'giAtrous2.dst', 'giResolve.irradiance'));
  engine.setPasses(Array.from({ length: 4 }, (_, i) => pass(`gi.atrous#${i}`)));
  await giStages.default.prepareHalf(engine, lib, variant('atrous-pair'), state, 'right', { renderFrames: async () => {} });
  assert.deepEqual(assertions(state), [['atrous-count-follows-nodes', true]]);
  const bad = await setup(engineDouble({ passes: Array.from({ length: 4 }, (_, i) => pass(`gi.atrous#${i}`)) }), 'atrous-pair', ['left', 'right']);
  assert.deepEqual(assertions(bad.state), [['atrous-count-follows-nodes', false]]);
});

test('area and SSGI variants bypass their stage and require the remaining stage to run', async () => {
  const area = engineDouble({ nodes: [stat('areaTemporal'), stat('areaMask')] });
  const areaResult = await setup(area, 'area-no-spatial');
  let doc = docOf(area.documents[0]);
  assert.ok(!doc.nodes.some(node => node.id === 'areaSpatial'));
  assert.ok(link(doc, 'areaTemporal.filtered', 'areaMask.src'));
  assert.deepEqual(assertions(areaResult.state), [['area-without-spatial-ran', true]]);
  area.nodes = [stat('areaTemporal'), stat('areaMask', false)];
  assert.deepEqual(assertions((await setup(area, 'area-no-spatial')).state), [['area-without-spatial-ran', false]]);

  const ssgi = engineDouble({ nodes: [stat('ssgiTrace')] });
  const ssgiResult = await setup(ssgi, 'ssgi-raw');
  doc = docOf(ssgi.documents[0]);
  assert.ok(!doc.nodes.some(node => node.id === 'ssgiResolve'));
  assert.ok(link(doc, 'ssgiTrace.raw', 'forward.gi'));
  assert.deepEqual(assertions(ssgiResult.state), [['ssgi-raw-ran', true]]);
  ssgi.nodes = [stat('ssgiTrace', false)];
  assert.deepEqual(assertions((await setup(ssgi, 'ssgi-raw')).state), [['ssgi-raw-ran', false]]);
});

test('observe exposes exactly the scene observation keys', async () => {
  const { observation } = await setup(engineDouble(), 'ssgi-raw');
  assert.deepEqual(Object.keys(observation).sort(), ['assertions', 'edits', 'graph', 'issues', 'renderGraphActive']);
});

test('noise rows run without pass.taa and with one-frame temporal stages; the parity pair keeps the starters', async () => {
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
  const param = (doc, id, key) => doc.nodes.find(node => node.id === id)?.params?.[key];
  const noTaa = doc => !doc.nodes.some(node => node.type === 'pass.taa') && link(doc, 'fog.color', 'bloom.color') &&
    !doc.links.some(entry => entry.from.startsWith('taa.') || entry.to.startsWith('taa.'));
  const docs = {};
  for (const [mode, halves] of [['spatial-skipped', ['left']], ['atrous-pair', ['left', 'right']],
    ['area-no-spatial', ['left']], ['ssgi-raw', ['left']]]) {
    const engine = engineDouble();
    const state = await giStages.default.build(engine, lib);
    for (const half of halves) await giStages.default.prepareHalf(engine, lib, variant(mode), state, half, { renderFrames: async () => {} });
    docs[mode] = engine.documents.map(docOf);
  }
  for (const doc of Object.values(docs).flat()) assert.ok(noTaa(doc), doc.name);
  assert.equal(param(docs['spatial-skipped'][0], 'giAccumulate', 'historyMax'), 1);
  const [two, four] = docs['atrous-pair'];
  assert.deepEqual([param(two, 'giAccumulate', 'historyMax'), param(four, 'giAccumulate', 'historyMax')], [1, 1]);
  assert.equal(four.nodes.filter(node => node.type === 'gi.atrous').length, 4);
  assert.deepEqual(two.nodes.filter(node => node.id !== 'giAtrous3' && node.id !== 'giAtrous4'),
    four.nodes.filter(node => node.id !== 'giAtrous3' && node.id !== 'giAtrous4'));
  assert.equal(param(docs['area-no-spatial'][0], 'areaTemporal', 'temporalAlpha'), 1);
  assert.equal(param(docs['ssgi-raw'][0], 'ssgiTrace', 'quality'), 'medium');
  const full = engineDouble();
  const state = await giStages.default.build(full, lib);
  for (const half of ['left', 'right']) await giStages.default.prepareHalf(full, lib, variant('full-gi-pair'), state, half, { renderFrames: async () => {} });
  assert.ok(full.documents.map(docOf).every(doc => doc.nodes.some(node => node.type === 'pass.taa')));
});
