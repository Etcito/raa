// GI stage-node scenes for the render-graph page. Each edited document is derived from parsed
// catalog starter text; starter graphs and room content are never copied into this page module.
import { RG_CAPTURE_T_S, selectionAt } from '../lib/scene.js';

const FULL_GI_STAGES = 'Full GI stages';
const FULL_GI = 'Full GI';
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera'];
const PRODUCER = 'raa-gi stages 11/11: ssgi.trace and ssgi.resolve stages';
const FACTS = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready'];

const row = (id, mode, filename, criterionName, fact, producer, expectedVisibleBehavior) => ({
  id, criterionName, scenario: 'render-graph-gi-stages', mode, filename, expectedVisibleBehavior,
  requiredFacts: [...FACTS, `assertion:${fact}`], pair: null, heldConstant: [...HELD_CONSTANT], producer,
});

export const factNames = [
  'stages-and-pass-graph-accepted', 'spatial-skipped-ran', 'atrous-count-follows-nodes',
  'area-without-spatial-ran', 'ssgi-raw-ran',
];

export const inventory = [
  row('gi-stages-full-gi-pair', 'full-gi-pair', 'gi-stages-full-gi-from-stages-left-pass-graph-right.png',
    'P5-Q10 Full GI from stage nodes matches the pass-level graph', factNames[0],
    'raa-graph exec gi_stages 3/3: graph frame binding stage inputs, executor-owned histories and forward slots',
    'Two halves of the same view of the room, labelled Full GI from stage nodes (left) and Full GI pass graph (right). Both show the room lit by the sun with soft indirect light: shadowed areas dim but not black, sun shadows under the boxes, a reflection on the glossy floor strip and a faint red tint on the floor near the red wall. The two halves match in brightness, colour and indirect light.'),
  row('gi-stages-spatial-skipped', 'spatial-skipped', 'gi-stages-spatial-skipped-raw-noisier.png',
    'P5-Q10 skipping the spatial stage shows the raw, noisier result', factNames[1],
    'raa-gi stages 7/11: gi.restir.spatial stage and gi.accumulate irradiance mode',
    'The room lit by the sun whose indirect light is visibly grainy: speckled, noisy patches of bounce light on the floor, walls and box sides in the shadowed areas instead of smooth soft gradients. The image is neither black nor washed out and no geometry is missing.'),
  row('gi-stages-atrous-pair', 'atrous-pair', 'gi-stages-two-atrous-left-four-atrous-right.png',
    'P5-Q10 the a-trous count is the number of gi.atrous nodes', factNames[2],
    'raa-gi stages 4/11: gi.atrous and gi.resolve stages with the resolve kernel',
    'Two halves of the same view, labelled two a-trous passes (left) and four a-trous passes (right). In the left half the indirect light in the shadowed areas shows visible blotchy noise; in the right half it is smoother and softer. Geometry, colours and direct sun light match.'),
  row('gi-stages-area-no-spatial', 'area-no-spatial', 'gi-stages-area-shadow-without-spatial.png',
    'P5-Q10 area shadow stages can be removed', factNames[3],
    'raa-gi stages 9/11: area.spatial and area.mask stages',
    'The room lit by the sun and an overhead area light, where the soft shadows of the area light on the floor around the boxes look grainy and speckled rather than smooth. The image is neither black nor washed out and no geometry is missing.'),
  row('gi-stages-ssgi-raw', 'ssgi-raw', 'gi-stages-ssgi-without-resolve-raw.png',
    'P5-Q10 SSGI stages can be removed', factNames[4],
    'raa-gi stages 11/11: ssgi.trace and ssgi.resolve stages',
    'The room lit by the sun whose indirect light in the shadowed areas is very noisy and speckled, without the smooth soft bounce light of a denoised result. The image is neither black nor washed out and no geometry is missing.'),
];

function makeVariant(id, description, pair = null) {
  return { id, description, references: [], judge: true, pair, selection: selectionAt(RG_CAPTURE_T_S) };
}

function starter(state, name) {
  const found = state.starters.find(entry => entry.name === name);
  if (!found) throw new Error(`render-graph gi-stages: the catalog has no ${name} starter`);
  return found.json;
}
const check = (state, name, passed, detail) => state.assertions.push({ name, passed: passed === true, detail });
function setGraph(engine, state, graph) {
  const result = engine.setRenderGraph(graph), [first] = result.issues ?? [];
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

// The noise rows show a stage's per-frame output. Every stage noise is reseeded each frame
// (ssgi/trace.wgsl interleaved gradient noise shifted by the frame index, the area and probe
// traces likewise), and the capture comes after RG_CAPTURE_T_S of a still camera, so any
// temporal accumulator left in the graph converges that noise to a smooth image: pass.taa
// (blendBase 0.06), gi.accumulate (historyMax 128) and area.temporal (temporalAlpha 0.1).
// A raw row therefore drops pass.taa (fog feeds bloom) and runs its remaining temporal stage
// with a one-frame history. Both halves of a pair get the same edit, so only the subject differs.
const setParam = (doc, id, key, value) => {
  const node = doc.nodes.find(entry => entry.id === id);
  if (!node) throw new Error(`render-graph gi-stages: graph ${doc.name} has no node ${id}`);
  node.params = { ...(node.params ?? {}), [key]: value };
};
function raw(doc, { accumulate = false, areaTemporal = false } = {}) {
  const taa = doc.nodes.find(node => node.type === 'pass.taa');
  if (taa) {
    const into = doc.links.find(link => link.to === `${taa.id}.color`)?.from;
    const out = doc.links.filter(link => link.from === `${taa.id}.color`).map(link => link.to);
    withoutNodes(doc, [taa.id]);
    for (const to of out) if (into) doc.links.push({ from: into, to });
  }
  if (accumulate) setParam(doc, 'giAccumulate', 'historyMax', 1);
  if (areaTemporal) setParam(doc, 'areaTemporal', 'temporalAlpha', 1);
  return doc;
}

function makeDocuments(state) {
  const stages = JSON.parse(starter(state, FULL_GI_STAGES));
  const passGraph = JSON.parse(starter(state, FULL_GI));
  const spatial = JSON.parse(starter(state, FULL_GI_STAGES));
  withoutNodes(spatial, ['giProbes', 'giAtrous1', 'giAtrous2', 'giAtrous3', 'giAtrous4']);
  const pyramid = spatial.nodes.findIndex(node => node.id === 'giPyramid');
  spatial.nodes.splice(pyramid + 1, 0,
    { id: 'restirInitial', type: 'gi.restir.initial' },
    { id: 'restirTemporal', type: 'gi.restir.temporal' });
  spatial.links.push(
    { from: 'giPyramid.pyramid', to: 'restirInitial.pyramid' },
    { from: 'restirInitial.reservoirs', to: 'restirTemporal.initial' },
    { from: 'restirTemporal.irradiance', to: 'giAccumulate.irradiance' },
    { from: 'giAccumulate.output', to: 'giResolve.irradiance' },
  );
  spatial.name = 'Full GI stages spatial skipped';
  raw(spatial, { accumulate: true });

  const two = JSON.parse(starter(state, FULL_GI_STAGES));
  withoutNodes(two, ['giAtrous3', 'giAtrous4']);
  two.links.push({ from: 'giAtrous2.dst', to: 'giResolve.irradiance' });
  two.name = 'Full GI stages two a-trous';
  raw(two, { accumulate: true });
  const four = raw(JSON.parse(starter(state, FULL_GI_STAGES)), { accumulate: true });
  four.name = 'Full GI stages four a-trous';

  const area = JSON.parse(starter(state, FULL_GI_STAGES));
  withoutNodes(area, ['areaSpatial']);
  area.links.push({ from: 'areaTemporal.filtered', to: 'areaMask.src' });
  area.name = 'Full GI stages area without spatial';
  raw(area, { areaTemporal: true });

  const ssgi = JSON.parse(starter(state, FULL_GI_STAGES));
  withoutNodes(ssgi, ['ssgiResolve']);
  ssgi.links.push({ from: 'ssgiTrace.raw', to: 'forward.gi' });
  ssgi.name = 'Full GI stages SSGI without resolve';
  raw(ssgi);
  return { stages, passGraph, spatial, two, four, area, ssgi };
}

async function prepareHalf(engine, _lib, variant, state, half, { renderFrames }) {
  const docs = makeDocuments(state);
  let doc;
  switch (variant.id) {
    case 'full-gi-pair': doc = half === 'left' ? docs.stages : docs.passGraph; break;
    case 'spatial-skipped': doc = docs.spatial; break;
    case 'atrous-pair': doc = half === 'left' ? docs.two : docs.four; break;
    case 'area-no-spatial': doc = docs.area; break;
    case 'ssgi-raw': doc = docs.ssgi; break;
    default: throw new Error(`render-graph gi-stages: unknown mode ${variant.id}`);
  }
  const result = setGraph(engine, state, doc);
  await renderFrames(2);
  if (variant.id === 'full-gi-pair') {
    const names = passNames(engine);
    state.pair.push({ half, ok: result.ok === true, passes: names });
    if (half === 'right') {
      const left = state.pair.find(entry => entry.half === 'left');
      check(state, factNames[0], !!left?.ok && result.ok === true && left.passes.includes('gi.pyramid#giPyramid') &&
        !left.passes.includes('gi.screen') && names.includes('gi.screen'), `left ${left?.passes.join(', ') ?? 'missing'}; right ${names.join(', ')}`);
    }
  } else if (variant.id === 'spatial-skipped') {
    const stats = engine.renderGraphStats(), graph = engine.renderGraph();
    const passed = result.ok === true && statNode(engine, 'restirTemporal')?.ran === true &&
      statNode(engine, 'giAccumulate')?.ran === true && !graph.nodes.some(node => node.type === 'gi.restir.spatial');
    check(state, factNames[1], passed, `ok ${result.ok}; temporal ${statNode(engine, 'restirTemporal')?.ran}; accumulate ${statNode(engine, 'giAccumulate')?.ran}; spatial absent ${!graph.nodes.some(node => node.type === 'gi.restir.spatial')}; frame ${stats?.frameIndex}`);
  } else if (variant.id === 'atrous-pair') {
    const count = names => names.filter(name => name.startsWith('gi.atrous#')).length;
    state.pair.push({ half, ok: result.ok === true, count: count(passNames(engine)) });
    if (half === 'right') {
      const left = state.pair.find(entry => entry.half === 'left');
      check(state, factNames[2], !!left?.ok && result.ok === true && left.count === 2 && count(passNames(engine)) === 4,
        `left ${left?.count ?? 'missing'}, right ${count(passNames(engine))}`);
    }
  } else {
    const graph = engine.renderGraph();
    const nodes = new Set(graph.nodes.map(node => node.type));
    const ids = new Set(graph.nodes.map(node => node.id));
    const fact = variant.id === 'area-no-spatial' ? factNames[3] : factNames[4];
    const passed = result.ok === true && (variant.id === 'area-no-spatial'
      ? statNode(engine, 'areaTemporal')?.ran === true && statNode(engine, 'areaMask')?.ran === true && !nodes.has('area.spatial')
      : statNode(engine, 'ssgiTrace')?.ran === true && !nodes.has('ssgi.resolve'));
    check(state, fact, passed, variant.id === 'area-no-spatial'
      ? `ok ${result.ok}; temporal ${statNode(engine, 'areaTemporal')?.ran}; mask ${statNode(engine, 'areaMask')?.ran}; spatial absent ${!nodes.has('area.spatial')}`
      : `ok ${result.ok}; trace ${statNode(engine, 'ssgiTrace')?.ran}; resolve absent ${!nodes.has('ssgi.resolve')}; active nodes ${ids.size}`);
  }
}

export function observe(engine, _variant, state) {
  return { renderGraphActive: engine.renderGraphActive(), graph: engine.renderGraphActive() ? engine.renderGraph().name : null,
    issues: engine.renderGraphIssues(), assertions: state.assertions, edits: state.edits };
}

/**
 * Handoff: render-graph page: gi-stages scene test drives `default.build(engine, lib)`,
 * `default.prepareHalf(engine, lib, variant, state, half, { renderFrames })` and
 * `observe(engine, variant, state)` from an engine double. Build state holds catalog starters,
 * assertions, edits and pair observations; each half applies a parsed starter-derived graph and
 * renders two frames.
 */
export default {
  id: 'render-graph-gi-stages', milestone: 'gi-stages',
  title: 'Render graph GI stages: compare stage graphs and remove GI stages', engine: {},
  variants: [
    makeVariant('full-gi-pair', 'Full GI from stage nodes (left) and Full GI pass graph (right)',
      { left: 'Full GI from stage nodes', right: 'Full GI pass graph', order: ['left', 'right'] }),
    makeVariant('spatial-skipped', 'Full GI stages with the spatial stage skipped'),
    makeVariant('atrous-pair', 'Two a-trous passes (left) and four a-trous passes (right)',
      { left: 'two a-trous passes', right: 'four a-trous passes', order: ['left', 'right'] }),
    makeVariant('area-no-spatial', 'Full GI stages with area spatial filtering removed'),
    makeVariant('ssgi-raw', 'Full GI stages with SSGI resolve removed'),
  ],
  async build(engine, lib) {
    await lib.scene.buildRoom(engine);
    return { starters: engine.renderGraphCatalog().starters, assertions: [], edits: [], pair: [] };
  },
  prepareHalf,
};
