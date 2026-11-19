import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as interpreter from './scenes/interpreter.js';
import * as views from './scenes/views.js';
import * as parameters from './scenes/parameters.js';
import * as wiring from './scenes/wiring.js';
import * as slots from './scenes/slots.js';
import * as giStages from './scenes/gi-stages.js';
import * as postStages from './scenes/post-stages.js';
import { validateInventory } from './lib/inventory.js';
import { stageInventory } from '../../web/common/render-graph-job.mjs';

const STARTER_TEXT = readFileSync(new URL('../../crates/raa-graph/graphs/full-gi-stages.json', import.meta.url), 'utf8');
const STARTER = JSON.parse(STARTER_TEXT);
const STARTERS = [{ name: 'Full GI stages', file: 'full-gi-stages.json', json: STARTER_TEXT }];
const RESOURCE = { port: 'output', width: 1280, height: 720, layers: 1, mips: 1, format: 'rgba16float', bytes: 1 };
const pass = name => ({ name, role: 'pass', enabled: true, order: 0, required: false });
const stat = (id, ran = true) => ({ id, type: id === 'aoTrace' ? 'ao.trace' : 'ssr.trace', ran, resources: ran ? [RESOURCE] : [] });

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
      const downs = active.nodes.filter(node => node.type === 'bloom.down').length;
      if (downs) currentPasses = Array.from({ length: downs }, (_, i) => pass(`bloom.down#${i + 1}`));
      if (active.nodes.some(node => node.type === 'ssr.trace')) currentPasses = [pass('ssr.trace#ssrTrace')];
      return { ok: true, issues: [], rebuilt: 0 };
    },
    renderGraphActive: () => active !== null,
    renderGraph: () => active,
    renderGraphIssues: () => [],
    renderGraphStats: () => ({ frameIndex: 2, nodes: currentNodes }),
    passes: () => currentPasses,
  };
}
const variant = id => postStages.default.variants.find(entry => entry.id === id);
async function run(engine, mode, halves = ['left']) {
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
  const state = await postStages.default.build(engine, lib);
  const frames = [];
  for (const half of halves) await postStages.default.prepareHalf(engine, lib, variant(mode), state, half,
    { renderFrames: async count => frames.push(count) });
  return { state, frames, observation: postStages.observe(engine, variant(mode), state) };
}
const assertions = state => state.assertions.map(({ name, passed }) => [name, passed]);
const docOf = value => typeof value === 'string' ? JSON.parse(value) : value;
const link = (doc, from, to) => doc.links.some(entry => entry.from === from && entry.to === to);
const ownRows = () => validateInventory([interpreter, views, parameters, wiring, slots, giStages, postStages], 'post-stages').slice(-4);

test('post-stage inventory rows and stage selection match the contract', () => {
  const rows = ownRows();
  assert.deepEqual(rows.map(({ id, mode, filename, criterionName, requiredFacts }) =>
    [id, mode, filename, criterionName, requiredFacts]), [
    ['post-stages-ao-raw', 'ao-raw', 'post-stages-ao-blur-removed-raw-ao.png',
      'P5-Q10 removing the AO blur stage shows the raw AO',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:ao-trace-without-blur-ran']],
    ['post-stages-ssr-trace', 'ssr-trace', 'post-stages-ssr-trace-tap.png',
      'P5-Q10 the SSR trace stage can be viewed alone',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:ssr-trace-shown-alone']],
    ['post-stages-bloom-pair', 'bloom-pair', 'post-stages-bloom-two-levels-left-five-right.png',
      'P5-Q10 the bloom chain length follows the number of level nodes',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:bloom-levels-follow-nodes']],
    ['post-stages-default', 'default', 'post-stages-default-full-gi-stages.png',
      'P5-Q3 the default Full GI stages graph renders unchanged',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:default-full-gi-stages-accepted']],
  ]);
  assert.equal(new Set(rows.map(row => row.filename)).size, 4);
  assert.deepEqual(stageInventory([interpreter, views, parameters, wiring, slots, giStages, postStages], 'post-stages', validateInventory)
    .map(row => row.id), rows.map(row => row.id));
});

test('AO raw document removes the pass, adds only ao.trace, and shows its raw output', async () => {
  const engine = engineDouble({ passes: [pass('ao.trace#aoTrace')], nodes: [stat('aoTrace')] });
  const result = await run(engine, 'ao-raw'), doc = docOf(engine.documents[0]);
  assert.equal(result.frames.length, 1);
  assert.ok(doc.nodes.some(node => node.id === 'aoTrace' && node.type === 'ao.trace'));
  assert.ok(!doc.nodes.some(node => ['ao.blur', 'pass.ao.gtao'].includes(node.type)));
  assert.ok(link(doc, 'aoTrace.raw', 'showPostStage.src'));
  assert.deepEqual(doc.nodes.find(node => node.id === 'showPostStage'),
    { id: 'showPostStage', type: 'view.image', params: { channels: 'r', label: 'raw AO' } });
  assert.deepEqual(assertions(result.state), [['ao-trace-without-blur-ran', true]]);
  engine.nodes = [stat('aoTrace', false)];
  assert.deepEqual(assertions((await run(engine, 'ao-raw')).state), [['ao-trace-without-blur-ran', false]]);
});

test('SSR trace replaces pass.ssr, preserves its colour chain and is shown in view', async () => {
  const engine = engineDouble({ passes: [pass('ssr.trace#ssrTrace')], nodes: [stat('ssrTrace')] });
  const result = await run(engine, 'ssr-trace'), doc = docOf(engine.documents[0]);
  for (const type of ['ssr.hiz', 'ssr.trace', 'ssr.composite']) assert.ok(doc.nodes.some(node => node.type === type));
  assert.ok(!doc.nodes.some(node => node.type === 'pass.ssr'));
  for (const [from, to] of [['ssrHiz.hiz', 'ssrTrace.hiz'], ['ssrTrace.trace', 'ssrComposite.trace'],
    ['giHistory.color', 'ssrComposite.color'], ['ssrComposite.color', 'fog.color'],
    ['ssrTrace.trace', 'showPostStage.src'], ['showPostStage.color', 'out.color']]) assert.ok(link(doc, from, to));
  assert.deepEqual(assertions(result.state), [['ssr-trace-shown-alone', true]]);
});

test('bloom pair links levels and terminal composite to the former colour chain', async () => {
  const engine = engineDouble({ passes: Array.from({ length: 2 }, (_, i) => pass(`bloom.down#${i}`)) });
  const result = await run(engine, 'bloom-pair', ['left', 'right']);
  const [left, right] = engine.documents.map(docOf);
  assert.deepEqual([left.nodes.filter(node => node.type === 'bloom.down').length, left.nodes.filter(node => node.type === 'bloom.up').length], [2, 1]);
  assert.deepEqual([right.nodes.filter(node => node.type === 'bloom.down').length, right.nodes.filter(node => node.type === 'bloom.up').length], [5, 4]);
  for (const doc of [left, right]) {
    assert.ok(link(doc, 'taa.color', 'bloomDown1.src'));
    assert.ok(link(doc, 'taa.color', 'bloomUp1.color'));
    assert.ok(link(doc, 'bloomUp1.color', 'tonemap.color'));
    const levels = doc.nodes.filter(node => node.type === 'bloom.down').length;
    for (let i = 2; i <= levels; i++) assert.ok(link(doc, `bloomDown${i - 1}.level`, `bloomDown${i}.src`));
    assert.ok(link(doc, `bloomDown${levels}.level`, `bloomUp${levels - 1}.src`));
    for (let i = 1; i < levels; i++) assert.ok(link(doc, `bloomDown${i}.level`, `bloomUp${i}.base`));
  }
  assert.deepEqual(left.nodes.find(node => node.id === 'bloomUp1').params, { strength: 0.3 });
  assert.deepEqual(assertions(result.state), [['bloom-levels-follow-nodes', true]]);
});

test('default mode submits the parsed starter document unchanged and checks its passes', async () => {
  const engine = engineDouble({ passes: [pass('gi.pyramid#giPyramid'), pass('ssr'), pass('bloom')] });
  const result = await run(engine, 'default');
  assert.deepEqual(docOf(engine.documents[0]), STARTER);
  assert.deepEqual(assertions(result.state), [['default-full-gi-stages-accepted', true]]);
});

test('observe returns the standard scene observation shape', async () => {
  const result = await run(engineDouble(), 'default');
  assert.deepEqual(Object.keys(result.observation).sort(), ['assertions', 'edits', 'graph', 'issues', 'renderGraphActive']);
});
