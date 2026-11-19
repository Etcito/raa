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
import * as compute from './scenes/compute.js';
import { validateInventory } from './lib/inventory.js';
import { stageInventory } from '../../web/common/render-graph-job.mjs';
import { captionText } from './lib/runtime.js';

const readJson = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const STARTERS = [
  { name: 'Custom denoise', json: readFileSync(new URL('../../crates/raa-graph/graphs/custom-denoise.json', import.meta.url), 'utf8') },
  { name: 'Full GI stages', json: readFileSync(new URL('../../crates/raa-graph/graphs/full-gi-stages.json', import.meta.url), 'utf8') },
];
const PRESETS = [{ name: 'Gaussian blur', json: readFileSync(new URL('../../crates/raa-graph/presets/gaussian-blur.json', import.meta.url), 'utf8') }];
const RESOURCE = { port: 'dst', width: 1280, height: 720, layers: 1, mips: 1, format: 'rgba16float', bytes: 1 };
const pass = name => ({ name, role: 'pass', enabled: true, order: 0, required: false });
const stat = (id, type = 'compute.wgsl', ran = true) => ({ id, type, ran, resources: ran ? [RESOURCE] : [] });

function engineDouble({ passes = [], nodes = [], refuse = null, omitCompute = false } = {}) {
  const documents = [];
  let active = null, currentPasses = passes, currentNodes = nodes, issues = [];
  return {
    documents,
    renderGraphCatalog: () => ({ starters: STARTERS, presets: PRESETS }),
    setRenderGraph(value) {
      const doc = typeof value === 'string' ? JSON.parse(value) : value;
      documents.push(doc);
      if (refuse?.(doc)) {
        issues = [{ code: refuse.code ?? 'graph.wgsl', severity: 'error', nodeId: 'giDenoise',
          message: refuse.message, fix: 'repair WGSL' }];
        return { ok: false, issues, rebuilt: 0 };
      }
      active = doc;
      issues = [];
      const types = new Set(doc.nodes.map(node => node.type));
      if (types.has('compute.wgsl')) {
        currentPasses = omitCompute ? [] : doc.nodes.filter(node => node.type === 'compute.wgsl').map(node => pass(`compute#${node.id}`));
        currentNodes = doc.nodes.filter(node => node.type === 'compute.wgsl').map(node => stat(node.id));
      }
      if (doc.name === 'Full GI stages') currentPasses = [pass('gi.atrous#giAtrous1')];
      return { ok: true, issues, rebuilt: 0 };
    },
    renderGraphActive: () => active !== null,
    renderGraph: () => active,
    renderGraphIssues: () => issues,
    renderGraphStats: () => ({ frameIndex: 2, nodes: currentNodes }),
    passes: () => currentPasses,
  };
}
const variant = id => compute.default.variants.find(entry => entry.id === id);
async function run(engine, mode, halves = ['left']) {
  const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
  const state = await compute.default.build(engine, lib), frames = [];
  for (const half of halves) await compute.default.prepareHalf(engine, lib, variant(mode), state, half,
    { renderFrames: async count => frames.push(count) });
  return { state, frames, observation: compute.observe(engine, variant(mode), state) };
}
const assertions = state => state.assertions.map(({ name, passed }) => [name, passed]);
const docOf = value => typeof value === 'string' ? JSON.parse(value) : value;
const link = (doc, from, to) => doc.links.some(entry => entry.from === from && entry.to === to);
const scenes = [interpreter, views, parameters, wiring, slots, giStages, postStages, compute];

test('compute inventory rows, filenames and stage selection match the contract', () => {
  const rows = validateInventory(scenes, 'compute').slice(-3);
  assert.deepEqual(rows.map(({ id, mode, filename, criterionName, requiredFacts }) =>
    [id, mode, filename, criterionName, requiredFacts]), [
    ['compute-denoise-pair', 'denoise-pair', 'compute-custom-denoise-bilateral-left-atrous-right.png',
      'P5-Q11 a custom compute denoiser replaces the a-trous stage with a clean result',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:custom-denoise-replaces-atrous']],
    ['compute-gaussian-ao', 'gaussian-ao', 'compute-preset-gaussian-blur-on-ao.png',
      'P5-Q11 a preset compute node runs on an engine image',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:gaussian-preset-ran']],
    ['compute-wgsl-error', 'wgsl-error', 'compute-wgsl-error-reported-on-node.png',
      'P5-Q11 a WGSL error is reported on its node with its line',
      ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:wgsl-error-on-node']],
  ]);
  assert.equal(new Set(rows.map(row => row.filename)).size, 3);
  assert.deepEqual(stageInventory(scenes, 'compute', validateInventory).map(row => row.id), rows.map(row => row.id));
});

test('denoise pair submits both catalog starters and checks compute against a-trous', async () => {
  const engine = engineDouble({ passes: [pass('gi.atrous#giAtrous1')] });
  const result = await run(engine, 'denoise-pair', ['left', 'right']);
  const [left, right] = engine.documents.map(docOf);
  assert.deepEqual(left, readJson('../../crates/raa-graph/graphs/custom-denoise.json'));
  assert.deepEqual(right, readJson('../../crates/raa-graph/graphs/full-gi-stages.json'));
  assert.ok(left.nodes.some(node => node.id === 'giDenoise' && node.type === 'compute.wgsl'));
  assert.deepEqual(result.frames, [2, 2]);
  assert.deepEqual(assertions(result.state), [['custom-denoise-replaces-atrous', true]]);
  const missing = engineDouble({ omitCompute: true });
  assert.deepEqual(assertions((await run(missing, 'denoise-pair', ['left', 'right'])).state),
    [['custom-denoise-replaces-atrous', false]]);
});

test('Gaussian preset document blurs AO and sends only that image to display', async () => {
  const engine = engineDouble({ passes: [pass('compute#aoBlur')], nodes: [stat('aoBlur')] });
  const result = await run(engine, 'gaussian-ao'), doc = docOf(engine.documents[0]);
  const presetNode = readJson('../../crates/raa-graph/presets/gaussian-blur.json').nodes[0];
  assert.deepEqual(doc.nodes.find(node => node.id === 'aoBlur'), { ...presetNode, id: 'aoBlur' });
  assert.ok(link(doc, 'aoGtao.ao', 'aoBlur.src'));
  assert.ok(link(doc, 'aoBlur.dst', 'showCompute.src'));
  assert.ok(link(doc, 'showCompute.color', 'out.color'));
  assert.ok(!link(doc, 'tonemap.color', 'out.color'));
  assert.deepEqual(assertions(result.state), [['gaussian-preset-ran', true]]);
  engine.renderGraphStats = () => ({ nodes: [stat('aoBlur', 'compute.wgsl', false)] });
  assert.deepEqual(assertions((await run(engine, 'gaussian-ao')).state), [['gaussian-preset-ran', false]]);
  assert.deepEqual(assertions((await run(engineDouble({ omitCompute: true }), 'gaussian-ao')).state),
    [['gaussian-preset-ran', false]]);
});

test('WGSL error refusal keeps Custom denoise active and caption includes its line', async () => {
  const starter = JSON.parse(STARTERS[0].json), wgsl = starter.nodes.find(node => node.id === 'giDenoise').wgsl;
  const line = (wgsl.match(/\n/g) ?? []).length + 2;
  const message = `WGSL error at line ${line}, column 11: expected expression`;
  const engine = engineDouble({ refuse: Object.assign(doc => doc.name === 'Custom denoise broken', { message }) });
  const result = await run(engine, 'wgsl-error'), [accepted, rejected] = engine.documents;
  assert.equal(compute.default.caption(engine, result.state), result.state.caption);
  assert.equal(accepted.nodes.find(node => node.id === 'giDenoise').wgsl, wgsl);
  assert.equal(rejected.nodes.find(node => node.id === 'giDenoise').wgsl,
    `${wgsl}fn brokenLine() -> f32 {\n  let x = ;\n  return 1.0;\n}\n`);
  assert.deepEqual(assertions(result.state), [['wgsl-error-on-node', true]]);
  assert.equal(result.state.caption, captionText({ code: 'graph.wgsl', nodeId: 'giDenoise', message }));
  assert.match(result.state.caption, new RegExp(`line ${line},`));
  const wrong = engineDouble({ refuse: Object.assign(doc => doc.name === 'Custom denoise broken',
    { message, code: 'graph.schema' }) });
  const failed = await run(wrong, 'wgsl-error');
  assert.deepEqual(assertions(failed.state), [['wgsl-error-on-node', false]]);
});

