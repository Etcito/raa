// Interpreter stage of the render-graph page (contract rg-interpreter-inventory, module
// render-graph-page). A mode sets render graphs through the engine and records every
// setRenderGraph and setter call as an edit; lib/runtime.js renders RG_RUN for each half that
// prepareHalf prepares. The starter graphs come from engine.renderGraphCatalog(), never copied
// here. Importing this module has no top-level side effect.
import { RG_CAPTURE_T_S, selectionAt } from '../lib/scene.js';
import { captionText } from '../lib/runtime.js';

const PASS_PREFIX = 'pass.';
const SIMPLE_FORWARD = 'Simple forward';
const FULL_GI = 'Full GI';
const REFUSED_NAME = 'Simple forward with invalid AO radius';
// The S61 Full GI values (sdfm Quality preset), written to an engine that has no graph.
const FULL_GI_GLOBAL = {
  historyMax: 128, clampGamma: 3, spatialPasses: 4, restirMCap: 30, voxelFullRelight: true, historyMin: 1,
  voxelHistoryMax: 64, voxelRays: 3, debugView: 'none', enabled: true, screenSpace: true,
};
const FULL_GI_QUALITY = { resolution: 96, levels: 5, probeRays: 8 };
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera'];
const PRODUCER = 'render-graph page 3/5: interpreter scene and its test';

export const factNames = [
  'simple-forward-accepted', 'full-gi-accepted', 'full-gi-engine-configured', 'null-restored', 'refused-kept-previous',
];

function row(id, mode, filename, criterionName, requiredFacts, expectedVisibleBehavior) {
  return {
    id, criterionName, scenario: 'render-graph-interpreter', mode, filename, expectedVisibleBehavior,
    requiredFacts, pair: null, heldConstant: [...HELD_CONSTANT], producer: PRODUCER,
  };
}

// The four rows of contract rg-interpreter-inventory: the only source of filenames and judge text.
export const inventory = [
  row('interpreter-simple-forward', 'simple-forward', 'interpreter-simple-forward-lit-shadowed-room.png',
    'P5-Q2 Simple forward lit and shadowed',
    ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:simple-forward-accepted'],
    'The room is correctly lit by the sun: surfaces facing the sun are brighter than the others and each box casts a distinct dark sun shadow on the floor under and beside it; the image is neither black nor washed out and no geometry is missing.'),
  row('interpreter-full-gi-pair', 'full-gi-pair', 'interpreter-full-gi-graph-left-engine-default-right.png',
    'P5-Q3 Full GI graph equals the engine configured the same way',
    ['snapshot:renderGraphActive', 'runtime:passes', 'runtime:ready', 'assertion:full-gi-accepted', 'assertion:full-gi-engine-configured'],
    'Two halves labelled graph (left) and engine default (right) show the same picture: the same direct and indirect lighting, the same shadows and reflections; no visible difference in brightness, colour or shadowing between the halves.'),
  row('interpreter-null-pair', 'null-pair', 'interpreter-null-graph-left-never-set-right-matches.png',
    'P5-Q4 setRenderGraph(null) renders as an engine that never had a graph',
    ['snapshot:renderGraphActive', 'runtime:passes', 'runtime:ready', 'assertion:null-restored'],
    'Two halves labelled null graph (left) and never set (right) show the same picture: identical lighting, shadows and colours, with no visible difference between the halves.'),
  row('interpreter-refused', 'refused', 'interpreter-refused-graph-keeps-simple-forward.png',
    'P5-Q5 A refused graph keeps the previous graph and names the node and code',
    ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:refused-kept-previous'],
    'The room still renders with sun lighting and box shadows on the floor (the previous graph), and a caption strip under the image reads graph.param_range and names the node aoGtao.'),
];

function makeVariant(id, description, pair = null) {
  return { id, description, references: [], judge: true, pair, selection: selectionAt(RG_CAPTURE_T_S) };
}

// The starter graph text with this name, from the engine catalog.
function starter(state, name) {
  const found = state.starters.find(entry => entry.name === name);
  if (!found) throw new Error(`render-graph interpreter: the catalog has no ${name} starter`);
  return found.json;
}
// The pass names a graph document drives: its `pass.` node types without the prefix.
function passEntries(text) {
  return new Set(JSON.parse(text).nodes
    .filter(node => node.type.startsWith(PASS_PREFIX))
    .map(node => node.type.slice(PASS_PREFIX.length)));
}
const enabledNames = engine => engine.passes().filter(pass => pass.enabled).map(pass => pass.name);
const graphName = engine => (engine.renderGraphActive() ? engine.renderGraph().name : null);
function sameNames(names, expected) {
  const actual = new Set(names);
  return actual.size === expected.size && [...expected].every(name => actual.has(name));
}
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
// A setter that throws is refused: record it as an edit, accepted or not.
function setting(state, operation, run) {
  try {
    run();
    state.edits.push({ operation, accepted: true, error: null });
  } catch (error) {
    state.edits.push({
      operation,
      accepted: false,
      error: { code: String(error?.code ?? 'api.error'), message: String(error?.message ?? error) },
    });
  }
}

function simpleForward(engine, state) {
  const text = starter(state, SIMPLE_FORWARD);
  const result = setGraph(engine, state, text);
  const names = enabledNames(engine), expected = passEntries(text), active = engine.renderGraphActive();
  check(state, 'simple-forward-accepted', result.ok === true && active && sameNames(names, expected),
    `ok ${result.ok}, active ${active}, enabled [${names.join(', ')}], graph passes [${[...expected].join(', ')}]`);
}

function fullGiGraph(engine, state) {
  const result = setGraph(engine, state, starter(state, FULL_GI));
  check(state, 'full-gi-accepted', result.ok === true, `ok ${result.ok}`);
}

// The engine half: no graph, then every non-required pass and the GI settings set by hand.
function fullGiEngine(engine, state) {
  const start = state.edits.length;
  setGraph(engine, state, null);
  const entries = passEntries(starter(state, FULL_GI));
  for (const pass of engine.passes()) {
    if (!pass.required) setting(state, 'setPassEnabled', () => engine.setPassEnabled(pass.name, entries.has(pass.name)));
  }
  setting(state, 'setGlobalIllumination', () => engine.setGlobalIllumination(FULL_GI_GLOBAL));
  setting(state, 'setGiQuality', () => engine.setGiQuality(FULL_GI_QUALITY));
  const accepted = state.edits.slice(start).every(edit => edit.accepted);
  check(state, 'full-gi-engine-configured', accepted && engine.renderGraphActive() === false,
    `every call accepted ${accepted}, active ${engine.renderGraphActive()}`);
}

// The right half is prepared first: its passes are the reference the left half must restore.
function neverSet(engine, state) {
  state.neverSet = JSON.stringify(engine.passes());
}

async function nullGraph(engine, state, renderFrames) {
  const r1 = setGraph(engine, state, starter(state, FULL_GI));
  await renderFrames(30);
  const r2 = setGraph(engine, state, null);
  const restored = JSON.stringify(engine.passes()) === state.neverSet;
  check(state, 'null-restored', r1.ok === true && r2.ok === true && engine.renderGraphActive() === false && restored,
    `ok ${r1.ok}/${r2.ok}, active ${engine.renderGraphActive()}, passes equal the never-set engine ${restored}`);
}

// An AO radius outside (0, 10] is refused: the Simple forward graph keeps rendering.
function refused(engine, state) {
  const text = starter(state, SIMPLE_FORWARD);
  const a = setGraph(engine, state, text);
  const before = JSON.stringify(enabledNames(engine));
  const doc = JSON.parse(text);
  doc.name = REFUSED_NAME;
  doc.nodes.push({ id: 'aoGtao', type: 'pass.ao.gtao', params: { radius: 25 } });
  const b = setGraph(engine, state, doc);
  const [issue] = b.issues ?? [];
  state.caption = issue ? captionText(issue) : null;
  const unchanged = JSON.stringify(enabledNames(engine)) === before;
  check(state, 'refused-kept-previous',
    a.ok === true && b.ok === false && issue?.code === 'graph.param_range' && issue.nodeId === 'aoGtao'
      && issue.param === 'radius' && engine.renderGraphActive() === true && graphName(engine) === SIMPLE_FORWARD && unchanged,
    `ok ${a.ok}/${b.ok}, code ${issue?.code}, node ${issue?.nodeId}, param ${issue?.param}, active graph ${graphName(engine)}, enabled passes unchanged ${unchanged}`);
}

async function prepareHalf(engine, _lib, variant, state, half, { renderFrames }) {
  switch (variant.id) {
    case 'simple-forward': return simpleForward(engine, state);
    case 'full-gi-pair': return half === 'left' ? fullGiGraph(engine, state) : fullGiEngine(engine, state);
    case 'null-pair': return half === 'right' ? neverSet(engine, state) : nullGraph(engine, state, renderFrames);
    case 'refused': return refused(engine, state);
    default: throw new Error(`render-graph interpreter: unknown mode ${variant.id}`);
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
  id: 'render-graph-interpreter',
  milestone: 'interpreter',
  title: 'Render graph interpreter: starter graphs, paired halves and a refused graph',
  engine: {},
  variants: [
    makeVariant('simple-forward', 'Simple forward starter graph on the lit room'),
    makeVariant('full-gi-pair', 'Full GI graph (left) beside the engine configured the same way without a graph (right)',
      { left: 'graph', right: 'engine default', order: ['left', 'right'] }),
    makeVariant('null-pair', 'setRenderGraph(null) (left) beside an engine that never had a graph (right)',
      { left: 'null graph', right: 'never set', order: ['right', 'left'] }),
    makeVariant('refused', 'Refused AO radius: the Simple forward graph keeps rendering'),
  ],
  async build(engine, lib) {
    await lib.scene.buildRoom(engine);
    return { starters: engine.renderGraphCatalog().starters, assertions: [], edits: [], caption: null, neverSet: null };
  },
  prepareHalf,
  caption: (_engine, state) => state.caption,
};
