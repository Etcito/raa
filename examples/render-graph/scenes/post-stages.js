// Post-stage scenes always derive their documents from the engine's catalog starters.
import { RG_CAPTURE_T_S, selectionAt } from '../lib/scene.js';

const FULL_GI_STAGES = 'Full GI stages';
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera'];
const FACTS = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready'];
const producer = 'raa-graph post_stages extension 2/2: graph frame binding inputs, bloom composite chain and view aliases';

const row = (id, mode, filename, criterionName, fact, source, expectedVisibleBehavior) => ({
  id, criterionName, scenario: 'render-graph-post-stages', mode, filename, expectedVisibleBehavior,
  requiredFacts: [...FACTS, `assertion:${fact}`], pair: null, heldConstant: [...HELD_CONSTANT], producer: source,
});

export const factNames = ['ao-trace-without-blur-ran', 'ssr-trace-shown-alone', 'bloom-levels-follow-nodes', 'default-full-gi-stages-accepted'];
export const inventory = [
  row('post-stages-ao-raw', 'ao-raw', 'post-stages-ao-blur-removed-raw-ao.png',
    'P5-Q10 removing the AO blur stage shows the raw AO', factNames[0],
    'raa-gtao stage passes: ao.trace and ao.blur',
    'The view shows a full-screen greyscale ambient-occlusion image of the room instead of the lit scene: open floor and wall areas near white, darker grey where the walls meet the floor, in the room corners and around the bases of the boxes and the sphere. The darkening is grainy and speckled with fine pixel noise rather than smooth, as unblurred ambient occlusion is. No colours and no direct lighting.'),
  row('post-stages-ssr-trace', 'ssr-trace', 'post-stages-ssr-trace-tap.png',
    'P5-Q10 the SSR trace stage can be viewed alone', factNames[1],
    'raa-ssr stage passes 1/2: ssr.hiz and ssr.trace',
    'The view shows a full-screen image of the screen-space reflection trace instead of the lit scene: mostly black over the rough floor, the walls and the sky, with coloured pixels (smooth mixtures of red, green and blue that encode the hit position and confidence) over the glossy floor strip and over the metal sphere where reflection rays found a surface.'),
  row('post-stages-bloom-pair', 'bloom-pair', 'post-stages-bloom-two-levels-left-five-right.png',
    'P5-Q10 the bloom chain length follows the number of level nodes', factNames[2],
    'raa-bloom stage passes 2/2: bloom.up composite',
    'Two halves of the same view of the room, labelled two bloom levels (left) and five bloom levels (right). In both halves the emissive panel on the back wall glows; in the left half the glow is a tight halo hugging the panel edges, in the right half it spreads much wider and softer over the wall around the panel. Geometry, colours and lighting otherwise match.'),
  row('post-stages-default', 'default', 'post-stages-default-full-gi-stages.png',
    'P5-Q3 the default Full GI stages graph renders unchanged', factNames[3],
    'post_stages extension 2/2: graph frame binding inputs, bloom composite chain and view aliases',
    'The room lit by the sun with soft indirect light: shadowed areas dim but not black, sun shadows under the boxes, a reflection on the glossy floor strip, a soft glow around the emissive panel on the back wall and a faint red tint on the floor near the red wall. No black screen, no missing geometry and no colour artefacts.'),
];

function makeVariant(id, description, pair = null) {
  return { id, description, references: [], judge: true, pair, selection: selectionAt(RG_CAPTURE_T_S) };
}
function starter(state) {
  const found = state.starters.find(entry => entry.name === FULL_GI_STAGES);
  if (!found) throw new Error(`render-graph post-stages: catalog has no ${FULL_GI_STAGES} starter`);
  return found.json;
}
const check = (state, name, passed, detail) => state.assertions.push({ name, passed: passed === true, detail });
function setGraph(engine, state, doc) {
  const result = engine.setRenderGraph(doc), [first] = result.issues ?? [];
  state.edits.push({ operation: 'setRenderGraph', accepted: result.ok === true,
    error: result.ok === true ? null : { code: first?.code ?? 'graph.refused', message: first?.message ?? 'refused' } });
  return result;
}
const statNode = (engine, id) => (engine.renderGraphStats()?.nodes ?? []).find(node => node.id === id);
const passNames = engine => engine.passes().filter(pass => pass.enabled !== false).map(pass => pass.name);
const withoutNodes = (doc, ids) => {
  const removed = new Set(ids);
  doc.nodes = doc.nodes.filter(node => !removed.has(node.id));
  doc.links = doc.links.filter(link => ![...removed].some(id => link.from.startsWith(`${id}.`) || link.to.startsWith(`${id}.`)));
};
const nodeOf = (doc, type) => doc.nodes.find(node => node.type === type);
const link = (doc, from, to) => doc.links.some(item => item.from === from && item.to === to);

function showInView(doc, source, channels, label) {
  const out = doc.nodes.find(node => node.type === 'output.display');
  const view = { id: 'showPostStage', type: 'view.image', params: { channels, label } };
  const outIndex = doc.nodes.indexOf(out);
  doc.nodes.splice(outIndex, 0, view);
  doc.links = doc.links.filter(item => item.to !== `${out.id}.color`);
  doc.links.push({ from: source, to: 'showPostStage.src' }, { from: 'showPostStage.color', to: `${out.id}.color` });
  return doc;
}
function aoRaw(text) {
  const doc = JSON.parse(text), old = nodeOf(doc, 'pass.ao.gtao');
  withoutNodes(doc, [old.id]);
  doc.nodes.splice(doc.nodes.indexOf(doc.nodes.find(node => node.type === 'pass.prepass')), 0, { id: 'aoTrace', type: 'ao.trace' });
  doc.name = 'Full GI stages raw AO';
  return showInView(doc, 'aoTrace.raw', 'r', 'raw AO');
}
function ssrTrace(text) {
  const doc = JSON.parse(text), old = nodeOf(doc, 'pass.ssr');
  const input = doc.links.find(item => item.to === `${old.id}.color`)?.from;
  const consumers = doc.links.filter(item => item.from === `${old.id}.color`).map(item => item.to);
  withoutNodes(doc, [old.id]);
  doc.nodes.push({ id: 'ssrHiz', type: 'ssr.hiz' }, { id: 'ssrTrace', type: 'ssr.trace' },
    { id: 'ssrComposite', type: 'ssr.composite' });
  doc.links.push({ from: 'ssrHiz.hiz', to: 'ssrTrace.hiz' }, { from: 'ssrTrace.trace', to: 'ssrComposite.trace' });
  if (input) doc.links.push({ from: input, to: 'ssrComposite.color' });
  for (const to of consumers) doc.links.push({ from: 'ssrComposite.color', to });
  doc.name = 'Full GI stages SSR trace';
  return showInView(doc, 'ssrTrace.trace', 'rgb', 'SSR trace');
}
function bloomLevels(text, count) {
  const doc = JSON.parse(text), old = nodeOf(doc, 'pass.bloom');
  const input = doc.links.find(item => item.to === `${old.id}.color`)?.from;
  const consumers = doc.links.filter(item => item.from === `${old.id}.color`).map(item => item.to);
  withoutNodes(doc, [old.id]);
  for (let i = 1; i <= count; i++) doc.nodes.push({ id: `bloomDown${i}`, type: 'bloom.down' });
  for (let i = 1; i < count; i++) doc.nodes.push({ id: `bloomUp${i}`, type: 'bloom.up', params: { strength: 0.3 } });
  if (input) doc.links.push({ from: input, to: 'bloomDown1.src' });
  for (let i = 2; i <= count; i++) doc.links.push({ from: `bloomDown${i - 1}.level`, to: `bloomDown${i}.src` });
  doc.links.push({ from: `bloomDown${count}.level`, to: `bloomUp${count - 1}.src` });
  for (let i = count - 2; i >= 1; i--) doc.links.push({ from: `bloomUp${i + 1}.level`, to: `bloomUp${i}.src` });
  for (let i = 1; i < count; i++) doc.links.push({ from: `bloomDown${i}.level`, to: `bloomUp${i}.base` });
  if (input) doc.links.push({ from: input, to: 'bloomUp1.color' });
  for (const to of consumers) doc.links.push({ from: 'bloomUp1.color', to });
  doc.name = `Full GI stages bloom ${count} levels`;
  return doc;
}

async function prepareHalf(engine, _lib, variant, state, half, { renderFrames }) {
  const text = starter(state);
  if (variant.id === 'default') {
    const result = setGraph(engine, state, text);
    await renderFrames(2);
    check(state, factNames[3], result.ok === true && passNames(engine).includes('gi.pyramid#giPyramid') &&
      passNames(engine).includes('ssr') && passNames(engine).includes('bloom'), `ok ${result.ok}; passes ${passNames(engine).join(', ')}`);
    return;
  }
  let doc;
  if (variant.id === 'ao-raw') doc = aoRaw(text);
  else if (variant.id === 'ssr-trace') doc = ssrTrace(text);
  else if (variant.id === 'bloom-pair') doc = bloomLevels(text, half === 'left' ? 2 : 5);
  else throw new Error(`render-graph post-stages: unknown mode ${variant.id}`);
  const result = setGraph(engine, state, doc);
  await renderFrames(2);
  const graph = engine.renderGraph(), types = new Set(graph.nodes.map(node => node.type)), names = passNames(engine);
  if (variant.id === 'ao-raw') check(state, factNames[0], result.ok === true && statNode(engine, 'aoTrace')?.ran === true &&
    !types.has('ao.blur') && !types.has('pass.ao.gtao') && link(graph, 'aoTrace.raw', 'showPostStage.src'),
    `ok ${result.ok}; aoTrace ran ${statNode(engine, 'aoTrace')?.ran}; types ${[...types].join(', ')}`);
  if (variant.id === 'ssr-trace') check(state, factNames[1], result.ok === true && statNode(engine, 'ssrTrace')?.ran === true &&
    names.includes('ssr.trace#ssrTrace') && !names.includes('ssr'), `ok ${result.ok}; passes ${names.join(', ')}`);
  if (variant.id === 'bloom-pair') {
    const count = half === 'left' ? 2 : 5, actual = names.filter(name => name.startsWith('bloom.down#')).length;
    state.pair.push({ half, ok: result.ok === true, count: actual });
    if (half === 'right') {
      const left = state.pair.find(item => item.half === 'left');
      check(state, factNames[2], !!left?.ok && result.ok === true && left.count === 2 && actual === 5,
        `left ${left?.count ?? 'missing'}, right ${actual}`);
    }
  }
}

export function observe(engine, _variant, state) {
  return { renderGraphActive: engine.renderGraphActive(), graph: engine.renderGraphActive() ? engine.renderGraph().name : null,
    issues: engine.renderGraphIssues(), assertions: state.assertions, edits: state.edits };
}

export default {
  id: 'render-graph-post-stages', milestone: 'post-stages',
  title: 'Render graph post stages: AO, SSR and bloom from stage nodes', engine: {},
  variants: [
    makeVariant('ao-raw', 'Raw AO stage shown in the view'),
    makeVariant('ssr-trace', 'SSR trace stage shown in the view'),
    makeVariant('bloom-pair', 'Two bloom levels (left) and five bloom levels (right)',
      { left: 'two bloom levels', right: 'five bloom levels', order: ['left', 'right'] }),
    makeVariant('default', 'Full GI stages starter graph unchanged'),
  ],
  async build(engine, lib) {
    await lib.scene.buildRoom(engine);
    return { starters: engine.renderGraphCatalog().starters, assertions: [], edits: [], pair: [] };
  },
  prepareHalf,
};
