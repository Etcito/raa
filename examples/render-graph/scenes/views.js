// Views stage of the render-graph page (contract views-scene, module render-graph-views-scene).
// A mode sets render graphs through the engine and records every setRenderGraph as an edit;
// lib/runtime.js renders RG_RUN for each half that prepareHalf prepares. The starter graphs come
// from engine.renderGraphCatalog(), never copied here. Importing this module has no top-level
// side effect.
import { RG_CAPTURE_T_S, selectionAt } from '../lib/scene.js';

const DEBUG = 'Debug';
const FULL_GI = 'Full GI';
const VIEW_PREFIX = 'view.';
const DEBUG_VIEWS = ['vDepth', 'vNormal', 'vMotion', 'vAo', 'vAreaMask', 'giRawVsDenoised', 'vHistory', 'vFinal', 'grid'];
const AO_VIEW = 'show_aoGtao_ao';
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera'];
const VIEWS_PRODUCER = 'View nodes 5/6: views graph frame, pass installation and node stats';
const SCENE_PRODUCER = 'Render-graph page: views scene, its test and the rg-views commands';

export const factNames = ['debug-views-ran', 'ao-view-ran', 'final-pair-accepted'];

function row(id, mode, filename, criterionName, requiredFacts, producer, expectedVisibleBehavior) {
  return {
    id, criterionName, scenario: 'render-graph-views', mode, filename, expectedVisibleBehavior,
    requiredFacts, pair: null, heldConstant: [...HELD_CONSTANT], producer,
  };
}

// The three rows of contract views-scene: the only source of filenames and judge text.
export const inventory = [
  row('views-debug-grid', 'debug-grid', 'views-debug-grid-every-tile-readable.png',
    'P5-Q7 Debug grid shows every tile as a readable labelled image',
    ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:debug-views-ran'],
    VIEWS_PRODUCER,
    'A grid of eight tiles in four columns and two rows fills the image, each tile with a readable white label at its top-left. Top row: depth (a greyscale depth image of the room, nearer surfaces darker and farther surfaces lighter), normal (the room coloured by surface orientation, floor and walls in distinct flat colours), motion (a nearly uniform tile), ao (mostly white surfaces with soft dark occlusion in the corners and under the boxes). Bottom row: area shadow mask (the area light shadow mask as a light and dark image), giRawVsDenoised (one tile split by a thin white vertical line, noisier indirect light on the left, smoother on the right), history length (a greyscale tile), final (the lit room in colour).'),
  row('views-show-in-view-ao', 'show-in-view-ao', 'views-show-in-view-ao-tap.png',
    'P5-Q7 Show in view puts the AO tap in the view',
    ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:ao-view-ran'],
    VIEWS_PRODUCER,
    'The whole frame shows the ambient-occlusion image of the room in greyscale instead of the lit colour image: mostly light grey and white surfaces with soft darker occlusion where the walls meet the floor, in the room corners and at the base of the boxes and the sphere; no lit colours.'),
  row('views-final-direct-pair', 'final-direct-pair', 'views-final-left-direct-light-only-right.png',
    'P5-Q7 final frame beside direct light only (page-composed pair: a view.compare of two forward branches needs multi-instance forward, a later milestone)',
    ['snapshot:renderGraphActive', 'runtime:passes', 'runtime:ready', 'assertion:final-pair-accepted'],
    SCENE_PRODUCER,
    'Two halves of the same view. The left half, labelled final, shows the room with indirect bounce light: a red tint from the red left wall on the floor and the nearby boxes, and shadowed areas that are not black. The right half, labelled direct light only, shows the same sunlit areas and shadow shapes but darker shadowed areas and no red bounce tint.'),
];

function makeVariant(id, description, pair = null) {
  return { id, description, references: [], judge: true, pair, selection: selectionAt(RG_CAPTURE_T_S) };
}

// The starter graph text with this name, from the engine catalog.
function starter(state, name) {
  const found = state.starters.find(entry => entry.name === name);
  if (!found) throw new Error(`render-graph views: the catalog has no ${name} starter`);
  return found.json;
}
const graphName = engine => (engine.renderGraphActive() ? engine.renderGraph().name : null);
const check = (state, name, passed, detail) => state.assertions.push({ name, passed: passed === true, detail });

// setRenderGraph reports a refusal in its result: record it as an edit with the first issue's code.
function setGraph(engine, state, graph) {
  const result = engine.setRenderGraph(graph);
  const [first] = result.issues ?? [];
  state.edits.push({
    operation: 'setRenderGraph',
    accepted: result.ok === true,
    error: result.ok === true ? null : { code: first?.code ?? 'graph.refused', message: first?.message ?? 'refused' },
  });
  return result;
}

// The stats entry of a view node is valid when it ran with at least one resource.
const viewRan = node => node?.ran === true && Array.isArray(node.resources) && node.resources.length > 0;
const statNodes = engine => engine.renderGraphStats()?.nodes ?? [];

async function debugGrid(engine, state, renderFrames) {
  const result = setGraph(engine, state, starter(state, DEBUG));
  await renderFrames(2);
  const active = engine.renderGraphActive();
  // The stats list every plan node; only the `view.` entries are checked.
  const views = statNodes(engine).filter(node => String(node.type).startsWith(VIEW_PREFIX));
  const ids = views.map(node => node.id);
  const exact = ids.length === DEBUG_VIEWS.length && DEBUG_VIEWS.every(id => ids.includes(id));
  const failed = views.filter(node => !viewRan(node)).map(node => node.id);
  check(state, 'debug-views-ran', result.ok === true && active && exact && failed.length === 0,
    `ok ${result.ok}, active ${active}, view nodes [${ids.join(', ')}], not ran or without resource [${failed.join(', ')}]`);
}

// The Full GI document with the AO tap shown in the view instead of the final colour.
function aoInView(text) {
  const doc = JSON.parse(text);
  doc.name = 'Full GI with AO in view';
  const outIndex = doc.nodes.findIndex(node => node.type === 'output.display');
  const out = doc.nodes[outIndex];
  doc.nodes.splice(outIndex, 0, { id: AO_VIEW, type: 'view.image', params: { channels: 'r', label: 'ao' } });
  doc.links = doc.links.filter(link => link.to !== `${out.id}.color`);
  doc.links.push({ from: 'aoGtao.ao', to: `${AO_VIEW}.src` }, { from: `${AO_VIEW}.color`, to: `${out.id}.color` });
  return doc;
}

async function showInViewAo(engine, state, renderFrames) {
  const result = setGraph(engine, state, aoInView(starter(state, FULL_GI)));
  await renderFrames(2);
  const node = statNodes(engine).find(entry => entry.id === AO_VIEW);
  check(state, 'ao-view-ran', result.ok === true && viewRan(node),
    `ok ${result.ok}, ${AO_VIEW} ran ${node?.ran}, resources ${node?.resources?.length ?? 0}`);
}

// The Full GI document with GI intensity 0 and no SSGI node.
function directOnly(text) {
  const doc = JSON.parse(text);
  doc.name = 'Full GI direct light only';
  const ssgi = doc.nodes.filter(node => node.type === 'pass.gi.ssgi').map(node => node.id);
  doc.nodes = doc.nodes.filter(node => !ssgi.includes(node.id));
  doc.links = doc.links.filter(link => !ssgi.includes(link.from.split('.')[0]) && !ssgi.includes(link.to.split('.')[0]));
  for (const node of doc.nodes) {
    if (node.type === 'pass.gi.volume') node.params = { ...node.params, intensity: 0 };
  }
  return doc;
}

function finalDirect(engine, state, half) {
  if (half === 'left') {
    state.finalOk = setGraph(engine, state, starter(state, FULL_GI)).ok === true;
    return;
  }
  const right = setGraph(engine, state, directOnly(starter(state, FULL_GI)));
  check(state, 'final-pair-accepted', state.finalOk === true && right.ok === true,
    `final ok ${state.finalOk}, direct light only ok ${right.ok}`);
}

async function prepareHalf(engine, _lib, variant, state, half, { renderFrames }) {
  switch (variant.id) {
    case 'debug-grid': return debugGrid(engine, state, renderFrames);
    case 'show-in-view-ao': return showInViewAo(engine, state, renderFrames);
    case 'final-direct-pair': return finalDirect(engine, state, half);
    default: throw new Error(`render-graph views: unknown mode ${variant.id}`);
  }
}

// The SceneObservation of contract rg-page-facts; collectRgFacts maps it to the page facts.
export function observe(engine, _variant, state) {
  return {
    renderGraphActive: engine.renderGraphActive(),
    graph: graphName(engine),
    issues: engine.renderGraphIssues(),
    assertions: state.assertions,
    edits: state.edits,
  };
}

export default {
  id: 'render-graph-views',
  milestone: 'views',
  title: 'Render graph views: Debug grid, Show in view and the final / direct light pair',
  engine: {},
  variants: [
    makeVariant('debug-grid', 'Debug starter graph: the view grid of the taps'),
    makeVariant('show-in-view-ao', 'Full GI with the AO tap shown in the view'),
    makeVariant('final-direct-pair', 'Full GI final frame (left) beside Full GI without indirect light (right)',
      { left: 'final', right: 'direct light only', order: ['left', 'right'] }),
  ],
  async build(engine, lib) {
    await lib.scene.buildRoom(engine);
    return { starters: engine.renderGraphCatalog().starters, assertions: [], edits: [], finalOk: null };
  },
  prepareHalf,
};
