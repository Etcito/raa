// Wiring stage of the render-graph page (module render-graph-wiring-scene): the three default
// graphs on the wired colour chain, the bloom / dof reorder pair and two TAA instances in a split
// compare. Starters come from engine.renderGraphCatalog(), never copied here.
import { RG_CAPTURE_T_S, selectionAt } from '../lib/scene.js';

const SIMPLE = 'Simple forward';
const FULL_GI = 'Full GI';
const DEBUG = 'Debug';
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera'];
const CHAIN_PRODUCER = 'Wiring 3/5: colour-chain graph frame, resource pool and order from links';
const MULTI_PRODUCER = 'Wiring 4/5: multi-instance post passes';
const DEFAULT_FACTS = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready'];
const PAIR_FACTS = ['snapshot:renderGraphActive', 'runtime:passes', 'runtime:ready', 'assertion:bloom-before-dof-order'];
const DOF = { id: 'dof', type: 'pass.dof', params: { focusDistance: 2, fStop: 0.7, focalLength: 85, maxCoc: 48 } };
const BLOOM = { strength: 0.35, radius: 1 };
// The bloom has no threshold and the DOF blurs by depth only: both are linear in colour, so on the
// flat back wall the two orders give the same image. The order shows on small, very bright warm
// emitters at the panel's corners, all beyond the 2 m focus: the bloom's first (Karis) downsample
// damps a sharp small emitter (bloom before dof: a faint patch, then blurred), while a dof-blurred
// emitter keeps its whole glow (dof before bloom: a brighter halo around the blurred panel).
export const PAIR_EMITTERS = {
  radius: 0.08,
  emissive: [120, 90, 45],
  positions: [[1.6, 3.0, -4.75], [3.6, 3.0, -4.75], [1.6, 1.8, -4.75], [3.6, 1.8, -4.75]],
};
// The second TAA instance: no history blending and a nearest-sample resolve, on a single-sample
// forward (msaa 4 would anti-alias the edges before any TAA, so neither half could show steps).
const TAA_B = { blendBase: 1, blendMin: 1, blendMax: 1, filterSharpness: 8 };
const TWO_TAA_ENGINE = { msaa: 1 };

export const factNames = ['simple-forward-wired', 'full-gi-wired', 'debug-wired', 'bloom-before-dof-order', 'two-taa-instances-ran'];

const row = (id, mode, filename, criterionName, requiredFacts, producer, expectedVisibleBehavior) => ({
  id, criterionName, scenario: 'render-graph-wiring', mode, filename, expectedVisibleBehavior,
  requiredFacts, pair: null, heldConstant: [...HELD_CONSTANT], producer,
});

export const inventory = [
  row('wiring-default-simple-forward', 'default-simple-forward', 'wiring-default-simple-forward.png',
    'P5-Q2 Simple forward still renders with the wired colour chain',
    [...DEFAULT_FACTS, 'assertion:simple-forward-wired'], CHAIN_PRODUCER,
    'The room is correctly lit by the sun: surfaces facing the sun are brighter than the others and each box casts a distinct dark sun shadow on the floor under and beside it; the image is neither black nor washed out and no geometry is missing.'),
  row('wiring-default-full-gi', 'default-full-gi', 'wiring-default-full-gi.png',
    'P5-Q3 Full GI still renders with the wired colour chain',
    [...DEFAULT_FACTS, 'assertion:full-gi-wired'], CHAIN_PRODUCER,
    'The room renders lit by the sun with soft indirect light: shadowed areas that are dim but not black, sun shadows from the boxes and a reflection on the glossy floor strip; the image is neither black nor washed out and no geometry is missing.'),
  row('wiring-default-debug', 'default-debug', 'wiring-default-debug.png',
    'P5-Q7 Debug grid still renders with the wired colour chain',
    [...DEFAULT_FACTS, 'assertion:debug-wired'], CHAIN_PRODUCER,
    'A grid of eight tiles in four columns and two rows fills the image, each tile with a readable white label at its top-left: depth, normal, motion, ao, area shadow mask, giRawVsDenoised (one tile split by a thin vertical line), history length and final (the lit room in colour); no tile is empty or black.'),
  row('wiring-bloom-before-dof-pair', 'bloom-before-dof-pair', 'wiring-post-chain-bloom-before-dof-left-default-right.png',
    'P5-Q9 a reordered post chain changes the image as expected', PAIR_FACTS, CHAIN_PRODUCER,
    'Two halves of the same strongly blurred view of the room. In the left half, labelled bloom before dof, the warm glow of the bright emissive panel on the back wall is blurred together with the panel into one wide, faint, soft patch. In the right half, labelled default order, the same blurred room shows a brighter, more concentrated glow halo around the blurred panel. The two halves visibly differ around the emissive panel.'),
  row('wiring-two-taa-split', 'two-taa-split', 'wiring-two-taa-instances-compare-split.png',
    'P5-Q9 two TAA instances compared side by side',
    [...DEFAULT_FACTS, 'assertion:two-taa-instances-ran'], MULTI_PRODUCER,
    'One view of the room split by a thin vertical line in the middle. Left of the line (the default TAA instance) the edges of the boxes, walls and sphere are smooth; right of the line (the second TAA instance with no history blending) the same edges show visible jagged stair-step aliasing. Lighting, colours and geometry match across the line.'),
];

function makeVariant(id, description, pair = null, engine = undefined) {
  const variant = { id, description, references: [], judge: true, pair, selection: selectionAt(RG_CAPTURE_T_S) };
  return engine ? { ...variant, engine } : variant;
}

// The pair's emitters (PAIR_EMITTERS), added once before both halves render.
function addPairEmitters(engine) {
  const material = engine.addMaterial({ color: [1, 0.9, 0.75], roughness: 0.5, emissive: PAIR_EMITTERS.emissive });
  const mesh = engine.primitiveMesh('sphere', { radius: PAIR_EMITTERS.radius }, 16);
  for (const position of PAIR_EMITTERS.positions) {
    engine.addInstance(engine.addNode(null, position, [0, 0, 0, 1], [1, 1, 1]), mesh, material);
  }
}

function starter(state, name) {
  const found = state.starters.find(entry => entry.name === name);
  if (!found) throw new Error(`render-graph wiring: the catalog has no ${name} starter`);
  return found.json;
}
const check = (state, name, passed, detail) => state.assertions.push({ name, passed: passed === true, detail });

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

// Both halves of the pair: Full GI with a strong bloom, a dof node and the chosen post links.
function reordered(text, chain) {
  const doc = JSON.parse(text);
  doc.name = chain === 'bloom-first' ? 'Full GI bloom before dof' : 'Full GI dof before bloom';
  const bloom = doc.nodes.find(node => node.id === 'bloom');
  bloom.params = { ...bloom.params, ...BLOOM };
  doc.nodes.splice(doc.nodes.indexOf(bloom) + 1, 0, structuredClone(DOF));
  const links = chain === 'bloom-first'
    ? [['taa', 'bloom'], ['bloom', 'dof'], ['dof', 'tonemap']]
    : [['taa', 'dof'], ['dof', 'bloom'], ['bloom', 'tonemap']];
  doc.links = doc.links.filter(link => !['taa.color', 'bloom.color'].includes(link.from));
  doc.links.push(...links.map(([from, to]) => ({ from: `${from}.color`, to: `${to}.color` })));
  return doc;
}

// Full GI without bloom, a second TAA with no history blending, compared by a split.
function twoTaa(text) {
  const doc = JSON.parse(text);
  doc.name = 'Full GI two TAA instances';
  doc.nodes = doc.nodes.filter(node => node.id !== 'bloom');
  const forward = doc.nodes.find(node => node.type === 'pass.forward');
  forward.params = { ...forward.params, msaa: TWO_TAA_ENGINE.msaa };
  doc.nodes.splice(doc.nodes.findIndex(node => node.id === 'tonemap'), 0,
    { id: 'taaB', type: 'pass.taa', params: { ...TAA_B } },
    { id: 'cmp', type: 'view.compare', params: { mode: 'split', position: 0.5 } });
  doc.links = doc.links.filter(link => !link.from.startsWith('bloom.') && !link.to.startsWith('bloom.'));
  doc.links.push(
    { from: 'fog.color', to: 'taaB.color' }, { from: 'taa.color', to: 'cmp.a' },
    { from: 'taaB.color', to: 'cmp.b' }, { from: 'cmp.color', to: 'tonemap.color' });
  return doc;
}

const statNode = (engine, id) => (engine.renderGraphStats()?.nodes ?? []).find(node => node.id === id);
const passNames = engine => engine.passes().filter(pass => pass.enabled !== false).map(pass => pass.name);

async function defaultGraph(engine, state, renderFrames, name, fact, extra = () => true) {
  const result = setGraph(engine, state, starter(state, name));
  await renderFrames(2);
  const active = engine.renderGraphActive();
  check(state, fact, result.ok === true && active && extra(), `ok ${result.ok}, active ${active}`);
}

async function debugWired(engine, state, renderFrames) {
  await defaultGraph(engine, state, renderFrames, DEBUG, 'debug-wired');
  const views = (engine.renderGraphStats()?.nodes ?? []).filter(node => String(node.type).startsWith('view.'));
  const failed = views.filter(node => !(node.ran === true && node.resources?.length > 0)).map(node => node.id);
  const entry = state.assertions.at(-1);
  entry.passed = entry.passed && views.length > 0 && failed.length === 0;
  entry.detail += `, view nodes ${views.length}, not ran [${failed.join(', ')}]`;
}

async function bloomDofPair(engine, state, half, renderFrames) {
  const left = half === 'left';
  const result = setGraph(engine, state, reordered(starter(state, FULL_GI), left ? 'bloom-first' : 'dof-first'));
  await renderFrames(2);
  const names = passNames(engine);
  const [bloom, dof] = [names.indexOf('bloom'), names.indexOf('dof')];
  const ordered = bloom >= 0 && dof >= 0 && (left ? bloom < dof : dof < bloom);
  state.pair.push({ ok: result.ok === true, ordered, half });
  if (!left) {
    check(state, 'bloom-before-dof-order', state.pair.length === 2 && state.pair.every(entry => entry.ok && entry.ordered),
      state.pair.map(entry => `${entry.half}: ok ${entry.ok}, ordered ${entry.ordered}`).join('; '));
  }
}

async function twoTaaSplit(engine, state, renderFrames) {
  const result = setGraph(engine, state, twoTaa(starter(state, FULL_GI)));
  await renderFrames(2);
  const names = passNames(engine);
  const cmp = statNode(engine, 'cmp');
  const ran = cmp?.ran === true && cmp.resources?.length > 0;
  check(state, 'two-taa-instances-ran', result.ok === true && names.includes('taa') && names.includes('taa#taaB') && ran,
    `ok ${result.ok}, taa ${names.includes('taa')}, taa#taaB ${names.includes('taa#taaB')}, cmp ran ${ran}`);
}

async function prepareHalf(engine, _lib, variant, state, half, { renderFrames }) {
  switch (variant.id) {
    case 'default-simple-forward': return defaultGraph(engine, state, renderFrames, SIMPLE, 'simple-forward-wired');
    case 'default-full-gi': return defaultGraph(engine, state, renderFrames, FULL_GI, 'full-gi-wired');
    case 'default-debug': return debugWired(engine, state, renderFrames);
    case 'bloom-before-dof-pair': return bloomDofPair(engine, state, half, renderFrames);
    case 'two-taa-split': return twoTaaSplit(engine, state, renderFrames);
    default: throw new Error(`render-graph wiring: unknown mode ${variant.id}`);
  }
}

export function observe(engine, _variant, state) {
  return {
    renderGraphActive: engine.renderGraphActive(),
    graph: engine.renderGraphActive() ? engine.renderGraph().name : null,
    issues: engine.renderGraphIssues(),
    assertions: state.assertions,
    edits: state.edits,
  };
}

export default {
  id: 'render-graph-wiring',
  milestone: 'wiring',
  title: 'Render graph wiring: default graphs on the wired chain, reordered post chain, two TAA instances',
  engine: {},
  variants: [
    makeVariant('default-simple-forward', 'Simple forward starter on the wired colour chain'),
    makeVariant('default-full-gi', 'Full GI starter on the wired colour chain'),
    makeVariant('default-debug', 'Debug starter on the wired colour chain'),
    makeVariant('bloom-before-dof-pair', 'Full GI with bloom before dof (left) and dof before bloom (right)',
      { left: 'bloom before dof', right: 'default order', order: ['left', 'right'] }),
    makeVariant('two-taa-split', 'Full GI with two TAA instances compared in a split', null, TWO_TAA_ENGINE),
  ],
  async build(engine, lib, variant) {
    await lib.scene.buildRoom(engine);
    if (variant?.id === 'bloom-before-dof-pair') addPairEmitters(engine);
    return { starters: engine.renderGraphCatalog().starters, assertions: [], edits: [], pair: [] };
  },
  prepareHalf,
};
