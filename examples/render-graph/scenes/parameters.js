// Parameters stage of the render-graph page (contract rg-m4-parameters-rows, scene
// render-graph-parameters). Row 0 pairs the Full GI starter with the same graph where every
// catalog parameter a node lacks is written at its catalog default. Each family row pairs the
// starter, plus the row's add-on nodes, with one parameter changed on the right. Every starter
// text comes from engine.renderGraphCatalog().starters, never copied here, and every
// setRenderGraph is recorded in state.edits. Importing this module has no top-level side effect.
import { RG_CAPTURE_T_S, cameraAt, selectionAt } from '../lib/scene.js';

const SCENE_ID = 'render-graph-parameters';
const SCENE_PRODUCER = 'Render-graph page: parameters scene, its test and the rg-parameters commands';
const DEFAULTS_MODE = 'full-gi-defaults';
const FULL_GI = 'Full GI';
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera'];
const DEFAULTS_FACTS = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:defaults-accepted'];
const FAMILY_FACTS = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready', 'assertion:parameter-changed'];
const PANE_TARGET = [0.4, 1, -0.4];
const PANE_SPACING_M = 0.25;
// Each pane also steps sideways (across the view) and up by PANE_FAN_M, so the eight panes fan out
// like a hand of cards: every colour shows at its own edge and a common core still holds all eight.
// Without the fan the panes hid exactly behind one another and read as one square (run 15631).
export const PANE_FAN_M = { side: 0.09, up: 0.05 };
const PANE_COLOURS = [
  [0.9, 0.2, 0.2], [0.95, 0.5, 0.15], [0.95, 0.85, 0.2], [0.2, 0.8, 0.3],
  [0.2, 0.75, 0.9], [0.2, 0.35, 0.95], [0.6, 0.25, 0.9], [0.85, 0.85, 0.85],
];

export const factNames = ['defaults-accepted', 'parameter-changed'];

// The 19 family rows of contract rg-m4-parameters-rows. A setting row takes its default and range
// from the cited Rust lines; a tunable row has null default and range here and takes them from
// docs/render-graph/tunables.md. `minExclusive` marks a `(` bound of the stated range.
export const PARAMETER_CHANGES = [
  { family: 'shadow', nodeType: 'pass.shadow.csm', nodeId: 'shadowCsm', key: 'softness', value: 8, binding: 'setting', default: 1, min: 0, max: 16, minExclusive: true },
  { family: 'gtao', nodeType: 'pass.ao.gtao', nodeId: 'aoGtao', key: 'power', value: 4, binding: 'setting', default: 1, min: 0, max: 8, minExclusive: true },
  { family: 'contact', nodeType: 'pass.shadow.contact', nodeId: 'shadowContact', key: 'distance', value: 2, binding: 'setting', default: 0.35, min: 0, max: 5, minExclusive: true },
  { family: 'ssr', nodeType: 'pass.ssr', nodeId: 'ssr', key: 'maxSteps', value: 4, binding: 'tunable', default: null, min: null, max: null, minExclusive: false },
  { family: 'dof', nodeType: 'pass.dof', nodeId: 'dof', key: 'fStop', value: 0.7, binding: 'setting', default: 2.8, min: 0.7, max: 64, minExclusive: false },
  { family: 'fog', nodeType: 'pass.fog', nodeId: 'fog', key: 'density', value: 0.1, binding: 'setting', default: 0, min: 0, max: null, minExclusive: false },
  { family: 'shafts', nodeType: 'pass.sunshafts', nodeId: 'sunshafts', key: 'density', value: 0.2, binding: 'setting', default: 0.02, min: 0, max: 1, minExclusive: false },
  { family: 'bloom', nodeType: 'pass.bloom', nodeId: 'bloom', key: 'strength', value: 0.5, binding: 'setting', default: 0.04, min: 0, max: 1, minExclusive: false },
  { family: 'grading', nodeType: 'pass.grading', nodeId: 'grading', key: 'saturation', value: 0, binding: 'setting', default: 1, min: 0, max: 4, minExclusive: false },
  { family: 'tonemap', nodeType: 'pass.tonemap', nodeId: 'tonemap', key: 'exposureEv', value: 2, binding: 'setting', default: 0, min: null, max: null, minExclusive: false },
  { family: 'taa', nodeType: 'pass.taa', nodeId: 'taa', key: 'filterSharpness', value: 0.25, binding: 'tunable', default: null, min: null, max: null, minExclusive: false },
  { family: 'oit', nodeType: 'pass.transparent.oit', nodeId: 'transparentOit', key: 'maxFragments', value: 4, binding: 'setting', default: 16, min: 4, max: 32, minExclusive: false },
  { family: 'gi-volume', nodeType: 'pass.gi.volume', nodeId: 'giVolume', key: 'intensity', value: 4, binding: 'setting', default: 1, min: 0, max: 16, minExclusive: false },
  { family: 'gi-screen', nodeType: 'pass.gi.screen', nodeId: 'giScreen', key: 'spatialPasses', value: 0, binding: 'setting', default: 3, min: 0, max: 5, minExclusive: false },
  { family: 'gi-restir', nodeType: 'pass.gi.screen', nodeId: 'giScreen', key: 'restirMCap', value: 1, binding: 'setting', default: 20, min: 1, max: 64, minExclusive: false },
  { family: 'area-shadow', nodeType: 'pass.gi.area-shadow', nodeId: 'giAreaShadow', key: 'temporalAlpha', value: 1, binding: 'tunable', default: null, min: null, max: null, minExclusive: false },
  { family: 'ssgi', nodeType: 'pass.gi.ssgi', nodeId: 'giSsgi', key: 'stepsMedium', value: 1, binding: 'tunable', default: null, min: null, max: null, minExclusive: false },
  { family: 'sky', nodeType: 'pass.sky.atmosphere', nodeId: 'skyAtmosphere', key: 'skyViewSteps', value: 2, binding: 'tunable', default: null, min: null, max: null, minExclusive: false },
  { family: 'clouds', nodeType: 'pass.sky.cloud-lighting', nodeId: 'skyCloudLighting', key: 'cloudAlbedo', value: 0.3, binding: 'tunable', default: null, min: null, max: null, minExclusive: false },
];

// Per family: the producer and the judge text of its contract row, and the add-on objects its
// row holds constant (the oit row's transparent panes).
const FAMILY_TEXT = {
  shadow: { producer: SCENE_PRODUCER, expected: 'Two halves of the same view, labelled default and softness = 8. In the right half the sun shadows of the boxes and the sphere on the floor have visibly wider, blurrier edges than the crisper shadows of the left half.' },
  gtao: { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and power = 4. In the right half the ambient occlusion where the walls meet the floor, in the room corner and at the bases of the boxes and the sphere is visibly darker and more contrasted than in the left half.' },
  contact: { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and distance = 2. In the right half dark contact shadowing next to the bases of the boxes and the sphere reaches visibly further across the floor than in the left half.' },
  ssr: { producer: 'raa-ssr: SSR tunables', expected: 'Two halves, labelled default and maxSteps = 4. The left half shows sharp reflections of the boxes and the sphere in the glossy floor strip; in the right half those reflections are missing, cut short or replaced by blurry coarse reflections.' },
  dof: { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and fStop = 0.7. The left half is mostly sharp; in the right half the floor in the foreground and the back wall are strongly blurred while the objects near the focus distance stay sharper.' },
  fog: { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and density = 0.1. The right half is veiled by a pale haze that thickens with distance: the back wall and far objects are washed out and lower in contrast than in the clear left half.' },
  shafts: { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and density = 0.2. The right half shows bright hazy sunlit air with visible streaks of light and darker shadowed volumes behind the boxes; the air of the left half is nearly clear.' },
  bloom: { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and strength = 0.5. In the right half a strong soft glow spreads around the warm emissive panel on the back wall and around bright highlights; the left half has only a faint glow.' },
  grading: { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and saturation = 0. The left half is in colour (red wall, warm panel, blue sky); the right half shows the same view entirely in greyscale.' },
  tonemap: { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and exposureEv = 2. The right half is visibly brighter overall, its bright areas close to white, compared with the left half.' },
  taa: { producer: 'raa-taa: TAA settings as tunables', expected: 'Two halves, labelled default and filterSharpness = 0.25. The right half is visibly softer: the edges of the boxes, the outline of the sphere and the floor strip are smeared compared with the crisper left half.' },
  oit: { producer: SCENE_PRODUCER, addOn: ['eight transparent coloured panes'], expected: 'Two halves, labelled default and maxFragments = 4. Eight overlapping coloured transparent panes stand in front of the room; where they all overlap, the left half blends all their colours, while the right half shows a visibly different, simpler stacked colour with some pane colours missing.' },
  'gi-volume': { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and intensity = 4. The right half has much stronger indirect light: shadowed areas are brighter and a strong red tint from the red wall lies on the floor and the nearby boxes.' },
  'gi-screen': { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and spatialPasses = 0. The indirect light on the walls and floor is smooth in the left half and visibly blotchy or noisy in the right half.' },
  'gi-restir': { producer: SCENE_PRODUCER, expected: 'Two halves, labelled default and restirMCap = 1. The indirect light on the walls and floor is smooth in the left half and visibly grainier, speckled, in the right half.' },
  'area-shadow': { producer: 'raa-gi 4/5: area-shadow tunables', expected: 'Two halves, labelled default and temporalAlpha = 1. The soft area-light shadows under the boxes are smooth in the left half and visibly grainy, speckled, in the right half.' },
  ssgi: { producer: 'raa-gi 5/5: SSGI per-quality slices and steps', expected: 'Two halves, labelled default and stepsMedium = 1. In the right half the soft darkening and colour bleeding where the boxes and the sphere meet the floor and the walls is weaker than in the left half.' },
  sky: { producer: 'raa-sky 1/3: atmosphere tunables and SkyHandle::set_tunable', expected: 'Two halves, labelled default and skyViewSteps = 2. The sky visible beyond the walls is a smooth blue gradient in the left half and has a visibly different colour or brightness, with banding near the horizon, in the right half.' },
  clouds: { producer: 'raa-sky 2/3: cloud tunables in CloudFrame and the cloud volume', expected: 'Two halves, labelled default and cloudAlbedo = 0.3. The clouds visible in the sky beyond the walls are bright white in the left half and dark grey in the right half.' },
};

const DEFAULTS_EXPECTED = 'Two halves of the same view of the room, labelled starter and all parameters at default, that look the same: the lit room with the red left wall, the white back wall with a warm glowing panel, three boxes and a metallic sphere on a grey floor with a glossy strip, with the same sun shadows, soft indirect light and floor reflections in both halves.';

// The nodes each family row adds to the starter, by id, and the setting a row writes into the
// starter in both halves before its change (the gi-restir row's ReSTIR indirect path).
const ADDED = { dof: 'pass.dof', sunshafts: 'pass.sunshafts', grading: 'pass.grading', transparentOit: 'pass.transparent.oit' };
const ADD_ONS = { dof: ['dof'], shafts: ['sunshafts'], grading: ['grading'], oit: ['transparentOit'] };
const SETUP = { 'gi-restir': { nodeId: 'giScreen', key: 'indirect', value: 'restir' } };

const row = (id, mode, filename, criterionName, requiredFacts, producer, expectedVisibleBehavior, heldConstant = HELD_CONSTANT) => ({
  id, criterionName, scenario: SCENE_ID, mode, filename, expectedVisibleBehavior, requiredFacts, pair: null,
  heldConstant: [...heldConstant], producer,
});

const defaultsRow = row('parameters-defaults', DEFAULTS_MODE, 'parameters-full-gi-all-defaults.png',
  'P5-Q8 every promoted parameter at its default leaves the Full GI image unchanged', DEFAULTS_FACTS,
  'Executor extension: apply Tunable bindings', DEFAULTS_EXPECTED);

function familyRow(change) {
  const text = FAMILY_TEXT[change.family];
  return row(`parameters-${change.family}`, change.family, `parameters-${change.family}-default-left-changed-right.png`,
    `P5-Q8 ${change.family}: one changed parameter visibly changes the Full GI image`, FAMILY_FACTS,
    text.producer, text.expected, [...HELD_CONSTANT, ...(text.addOn ?? [])]);
}

// The contract rows of stage parameters: the defaults row, then one row per family.
export const inventory = [defaultsRow, ...PARAMETER_CHANGES.map(familyRow)];

// The eight transparent panes of the oit row: 1 x 1 m, stacked along the camera's view through
// PANE_TARGET at PANE_SPACING_M, each turned to face the camera's yaw. Pure geometry, so the
// variant's subjects and the built scene agree.
export function paneLayout() {
  const eye = cameraAt(RG_CAPTURE_T_S).position;
  const toward = PANE_TARGET.map((value, axis) => value - eye[axis]);
  const length = Math.hypot(...toward);
  const yaw = Math.atan2(eye[0], eye[2]);
  const side = [toward[2] / Math.hypot(toward[0], toward[2]), 0, -toward[0] / Math.hypot(toward[0], toward[2])];
  return PANE_COLOURS.map((color, index) => {
    const step = index - (PANE_COLOURS.length - 1) / 2;
    const offset = step * PANE_SPACING_M;
    const centre = PANE_TARGET.map((value, axis) => value + toward[axis] / length * offset
      + side[axis] * step * PANE_FAN_M.side + (axis === 1 ? step * PANE_FAN_M.up : 0));
    return { name: `oit-pane-${index}`, color, centre, yaw };
  });
}

const paneSubjects = () => paneLayout().map(({ name, centre }) => ({
  name, min: centre.map(value => value - 0.6), max: centre.map(value => value + 0.6),
}));

function addPanes(engine) {
  for (const pane of paneLayout()) {
    const node = engine.addNode(null, pane.centre, [0, Math.sin(pane.yaw / 2), 0, Math.cos(pane.yaw / 2)], [1, 1, 1]);
    const material = engine.addSurfaceMaterial({ color: pane.color, roughness: 0.5, blend: 'transparent', opacity: 0.35 });
    engine.addInstance(node, engine.primitiveMesh('box', { halfExtents: [0.5, 0.5, 0.01] }), material);
  }
}

function makeVariant(id, description, pair, extraSubjects = []) {
  const selection = selectionAt(RG_CAPTURE_T_S);
  selection.subjects = [...selection.subjects, ...extraSubjects];
  return { id, description, references: [], judge: true, pair, selection };
}

const variants = [
  makeVariant(DEFAULTS_MODE, 'Full GI starter (left) beside the same graph with every catalog parameter written at its default (right)',
    { left: 'starter', right: 'all parameters at default', order: ['left', 'right'] }),
  ...PARAMETER_CHANGES.map(change => makeVariant(change.family,
    `Full GI with ${change.key} = ${change.value} (right) beside the default (left)`,
    { left: 'default', right: `${change.key} = ${change.value}`, order: ['left', 'right'] },
    change.family === 'oit' ? paneSubjects() : [])),
];
const accepted = result => result.ok === true && !(result.issues ?? []).some(issue => issue.severity === 'error');
const errorCodes = result => (result.issues ?? []).filter(issue => issue.severity === 'error').map(issue => issue.code).join(', ') || 'none';
const within = (value, min, max) => (min === null || min === undefined || value >= min) && (max === null || max === undefined || value <= max);
const check = (state, name, passed, detail) => state.assertions.push({ name, passed: passed === true, detail });

// setRenderGraph reports a refusal in its result: record it as an edit with the first issue's code.
function setGraph(engine, state, graph) {
  const result = engine.setRenderGraph(graph);
  const [first] = result.issues ?? [];
  const ok = accepted(result);
  state.edits.push({
    operation: 'setRenderGraph',
    accepted: ok,
    error: ok ? null : { code: first?.code ?? 'graph.refused', message: first?.message ?? 'refused' },
  });
  return result;
}

const catalogNode = (state, type) => state.catalog.nodes.find(node => node.id === type);
const catalogParams = (state, type) => catalogNode(state, type)?.params ?? [];
const catalogParam = (state, type, key) => catalogParams(state, type).find(param => param.key === key);

// The Full GI starter document, parsed from the engine catalog (never copied into this file).
function starterDocument(state) {
  const found = state.catalog.starters.find(entry => entry.name === FULL_GI);
  if (!found) throw new Error(`render-graph parameters: the catalog has no ${FULL_GI} starter`);
  return JSON.parse(found.json);
}

function nodeOf(doc, id) {
  const node = doc.nodes.find(entry => entry.id === id);
  if (!node) throw new Error(`render-graph parameters: the graph has no node ${id}`);
  return node;
}

function setParam(doc, nodeId, key, value) {
  const node = nodeOf(doc, nodeId);
  node.params = { ...node.params, [key]: value };
}

// An added node takes every catalog parameter at its default, placed before the output node.
function addNode(doc, state, id) {
  const type = ADDED[id];
  const params = Object.fromEntries(catalogParams(state, type).map(param => [param.key, param.default]));
  const output = doc.nodes.findIndex(node => node.type === 'output.display');
  doc.nodes.splice(output < 0 ? doc.nodes.length : output, 0, { id, type, params });
}

// The colour links: each colour input reads the first defaultSource entry whose node type is in
// the graph (the chain order of M4). Since M6 the chain inputs are generic, so nothing links them
// by default: an added post node is spliced in here, or it stays unlinked (graph.input_missing).
// Added and affected colour inputs are rewritten in place, a missing link is appended.
function wireColour(doc, state) {
  const idOfType = new Map(doc.nodes.map(node => [node.type, node.id]));
  for (const node of doc.nodes) {
    const port = catalogNode(state, node.type)?.inputs.find(input => input.key === 'color');
    if (!port) continue;
    const source = port.defaultSource.find(entry => idOfType.has(entry.slice(0, entry.lastIndexOf('.'))));
    if (source === undefined) continue;
    const type = source.slice(0, source.lastIndexOf('.'));
    const link = { from: `${idOfType.get(type)}.${source.slice(type.length + 1)}`, to: `${node.id}.color` };
    const at = doc.links.findIndex(entry => entry.to === link.to);
    if (at < 0) doc.links.push(link);
    else doc.links[at] = link;
  }
}

// The starter with a family row's add-on nodes and setup, and the changed value on the right.
function familyDocument(state, change, half) {
  const doc = starterDocument(state);
  for (const id of ADD_ONS[change.family] ?? []) addNode(doc, state, id);
  const setup = SETUP[change.family];
  if (setup) setParam(doc, setup.nodeId, setup.key, setup.value);
  wireColour(doc, state);
  if (half === 'right') setParam(doc, change.nodeId, change.key, change.value);
  return doc;
}

// The starter where every catalog parameter a node lacks is written at its catalog default.
function withCatalogDefaults(doc, state) {
  for (const node of doc.nodes) {
    const missing = catalogParams(state, node.type).filter(param => !(param.key in (node.params ?? {})));
    if (missing.length) node.params = { ...node.params, ...Object.fromEntries(missing.map(param => [param.key, param.default])) };
  }
  return doc;
}

const missingParams = (state, doc) => doc.nodes.flatMap(node => catalogParams(state, node.type)
  .filter(param => !(param.key in (node.params ?? {}))).map(param => `${node.id}.${param.key}`));

function defaultsHalf(engine, state, half) {
  if (half === 'left') {
    const result = setGraph(engine, state, starterDocument(state));
    state.left = { ok: accepted(result), value: null };
    return;
  }
  const result = setGraph(engine, state, withCatalogDefaults(starterDocument(state), state));
  const missing = missingParams(state, engine.renderGraph());
  check(state, 'defaults-accepted', state.left?.ok === true && accepted(result) && missing.length === 0,
    `starter ok ${state.left?.ok}, all-defaults ok ${accepted(result)} (errors: ${errorCodes(result)}), parameters missing from renderGraph(): [${missing.join(', ')}]`);
}

function changeHalf(engine, state, change, half) {
  const doc = familyDocument(state, change, half);
  const result = setGraph(engine, state, doc);
  if (half === 'left') {
    const value = nodeOf(doc, change.nodeId).params?.[change.key] ?? catalogParam(state, change.nodeType, change.key)?.default ?? null;
    state.left = { ok: accepted(result), value };
    return;
  }
  const shown = engine.renderGraph().nodes.find(node => node.id === change.nodeId)?.params?.[change.key];
  const param = catalogParam(state, change.nodeType, change.key);
  const fallback = param?.default ?? null;
  const passed = state.left?.ok === true && accepted(result) && shown === change.value
    && change.value !== fallback && change.value !== state.left?.value && within(change.value, param?.min, param?.max);
  check(state, 'parameter-changed', passed,
    `${change.nodeId}.${change.key} = ${shown} (catalog default ${fallback}, left ${state.left?.value}, range [${param?.min ?? 'none'}, ${param?.max ?? 'none'}]), ok ${state.left?.ok} and ${accepted(result)}`);
}

export async function build(engine, lib, variant) {
  await lib.scene.buildRoom(engine);
  if (variant?.id === 'oit') addPanes(engine);
  return { catalog: engine.renderGraphCatalog(), assertions: [], edits: [], left: null };
}

export async function prepareHalf(engine, _lib, variant, state, half) {
  if (variant.id === DEFAULTS_MODE) return defaultsHalf(engine, state, half);
  const change = PARAMETER_CHANGES.find(entry => entry.family === variant.id);
  if (!change) throw new Error(`render-graph parameters: unknown mode ${variant.id}`);
  return changeHalf(engine, state, change, half);
}

// The SceneObservation of contract rg-page-facts; collectRgFacts maps it to the page facts.
export function observe(engine, _variant, state) {
  const active = engine.renderGraphActive();
  return {
    renderGraphActive: active,
    graph: active ? engine.renderGraph().name : null,
    issues: engine.renderGraphIssues(),
    assertions: state.assertions,
    edits: state.edits,
  };
}

export default {
  id: SCENE_ID,
  milestone: 'parameters',
  title: 'Render graph parameters: every parameter at its default, then one changed parameter per family',
  engine: {},
  variants,
  build,
  prepareHalf,
};
