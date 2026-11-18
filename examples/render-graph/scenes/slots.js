// Slots stage of the render-graph page (module render-graph-slots-scene): two AO branches with radii
// 0.3 m and 2 m in a split compare beside the final frame, a forward input fed by either branch as a
// pair, and the Full GI and Simple forward starters unchanged. Starters come from
// engine.renderGraphCatalog(), never copied here.
import { RG_CAPTURE_T_S, selectionAt } from '../lib/scene.js';

const SIMPLE = 'Simple forward';
const FULL_GI = 'Full GI';
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera'];
const SLOT_PRODUCER = 'Slots 2/3: executor extension slots, graph frame binding slot and G-buffer inputs';
const INSTANCE_PRODUCER = 'Slots 3/3: gtao and contact instance passes in the slots extension';
const DEFAULT_FACTS = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready'];
const PAIR_FACTS = ['snapshot:renderGraphActive', 'runtime:passes', 'runtime:ready', 'assertion:forward-follows-ao-link'];
// Each half feeds forward.ao from one AO branch; the other branch is left unreachable (accepted warning).
const FEED = {
  left: { node: 'aoWide', name: 'Simple forward forward ao from wide' },
  right: { node: 'aoNarrow', name: 'Simple forward forward ao from narrow' },
};
const SKY = [
  { id: 'skyAtmosphere', type: 'pass.sky.atmosphere' },
  { id: 'skyCloudLighting', type: 'pass.sky.cloud-lighting' },
  { id: 'skyEnv', type: 'pass.sky.env' },
];
const VIEWS = [
  { id: 'vNarrow', type: 'view.image', params: { channels: 'r', label: 'ao 0.3 m' } },
  { id: 'vWide', type: 'view.image', params: { channels: 'r', label: 'ao 2 m' } },
  { id: 'aoNarrowVsWide', type: 'view.compare', params: { mode: 'split', position: 0.5 } },
  { id: 'vFinal', type: 'view.image', params: { label: 'final, forward ao from 2 m' } },
  { id: 'grid', type: 'view.grid', params: { columns: 2 } },
];
const SPLIT_LINKS = [
  ['prepass.depth', 'lightTiles.depth'], ['shadowCsm.shadowMap', 'forward.shadowMap'],
  ['lightTiles.lightTiles', 'forward.lightTiles'], ['aoWide.ao', 'forward.ao'], ['forward.color', 'tonemap.color'],
  ['aoNarrow.ao', 'vNarrow.src'], ['aoWide.ao', 'vWide.src'], ['vNarrow.color', 'aoNarrowVsWide.a'],
  ['vWide.color', 'aoNarrowVsWide.b'], ['tonemap.color', 'vFinal.src'], ['aoNarrowVsWide.color', 'grid.t0'],
  ['vFinal.color', 'grid.t1'], ['grid.color', 'out.color'],
].map(([from, to]) => ({ from, to }));

export const factNames = ['two-ao-branches-ran', 'forward-follows-ao-link', 'full-gi-slots', 'simple-forward-slots'];

const row = (id, mode, filename, criterionName, requiredFacts, producer, expectedVisibleBehavior) => ({
  id, criterionName, scenario: 'render-graph-slots', mode, filename, expectedVisibleBehavior,
  requiredFacts, pair: null, heldConstant: [...HELD_CONSTANT], producer,
});

export const inventory = [
  row('slots-two-ao-split', 'two-ao-split', 'slots-two-ao-branches-compare-split.png',
    'P5-Q9 two AO branches compare side by side',
    [...DEFAULT_FACTS, 'assertion:two-ao-branches-ran'], INSTANCE_PRODUCER,
    'A grid of two tiles side by side, each with a readable white label at its top-left. The left tile, labelled aoNarrowVsWide, is a greyscale ambient-occlusion image of the room split by a thin vertical line: left of the line the darkening is confined to thin lines in the creases where the boxes meet the floor and where walls meet; right of the line the darkening is broad and soft, spreading well across the floor around the boxes and up the wall corners. The right tile, labelled final, forward ao from 2 m, shows the lit room in colour. No tile is empty or black.'),
  row('slots-forward-ao-pair', 'forward-ao-pair', 'slots-forward-ao-from-wide-radius-branch.png',
    'P5-Q9 the forward input follows the chosen AO branch', PAIR_FACTS, SLOT_PRODUCER,
    'Two halves of the same view of the lit room, each labelled at its top. In the left half, labelled forward ao from 2 m branch, broad soft darkening spreads across the floor around and between the boxes and into the wall corners, most visible where the sun does not reach. In the right half, labelled forward ao from 0.3 m branch, the darkening is limited to thin contact lines where the boxes meet the floor and to tight corners. Geometry, colours and sun lighting otherwise match, and the left half is visibly darker around the boxes.'),
  row('slots-default-full-gi', 'default-full-gi', 'slots-default-full-gi.png',
    'P5-Q3 Full GI still renders with slot and G-buffer binding',
    [...DEFAULT_FACTS, 'assertion:full-gi-slots'], SLOT_PRODUCER,
    'The room renders lit by the sun with soft indirect light: shadowed areas that are dim but not black, sun shadows from the boxes and a reflection on the glossy floor strip; the image is neither black nor washed out and no geometry is missing.'),
  row('slots-default-simple-forward', 'default-simple-forward', 'slots-default-simple-forward.png',
    'P5-Q2 Simple forward still renders with slot and G-buffer binding',
    [...DEFAULT_FACTS, 'assertion:simple-forward-slots'], SLOT_PRODUCER,
    'The room is correctly lit by the sun: surfaces facing the sun are brighter than the others and each box casts a distinct dark sun shadow on the floor under and beside it; the image is neither black nor washed out and no geometry is missing.'),
];

function makeVariant(id, description, pair = null) {
  return { id, description, references: [], judge: true, pair, selection: selectionAt(RG_CAPTURE_T_S) };
}

const pick = (base, id) => base.nodes.find(node => node.id === id);
const ao = (id, radius) => ({ id, type: 'pass.ao.gtao', params: { radius, intensity: 1, power: 2 } });

// The contract document slots-two-ao-doc: Simple forward nodes with the sky, the two AO nodes and
// the view nodes inserted at their positions; the tonemap-to-out link becomes grid-to-out.
function twoAoDoc(text) {
  const base = JSON.parse(text);
  const nodes = [
    ...SKY, pick(base, 'shadowCsm'), pick(base, 'prepass'), pick(base, 'lightTiles'),
    ao('aoNarrow', 0.3), ao('aoWide', 2), pick(base, 'forward'), pick(base, 'skyBackground'),
    pick(base, 'tonemap'), ...VIEWS, pick(base, 'out'),
  ];
  return { ...base, name: 'Simple forward two AO branches', nodes, links: SPLIT_LINKS };
}

// One half of the pair: Simple forward with the sky before shadowCsm and both AO nodes after
// lightTiles, and forward.ao fed by the half's branch.
function pairDoc(text, half) {
  const base = JSON.parse(text);
  const nodes = [
    ...SKY, pick(base, 'shadowCsm'), pick(base, 'prepass'), pick(base, 'lightTiles'),
    ao('aoNarrow', 0.3), ao('aoWide', 2), pick(base, 'forward'), pick(base, 'skyBackground'),
    pick(base, 'tonemap'), pick(base, 'out'),
  ];
  const links = [...base.links, { from: `${FEED[half].node}.ao`, to: 'forward.ao' }];
  return { ...base, name: FEED[half].name, nodes, links };
}

function starter(state, name) {
  const found = state.starters.find(entry => entry.name === name);
  if (!found) throw new Error(`render-graph slots: the catalog has no ${name} starter`);
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

const statNode = (engine, id) => (engine.renderGraphStats()?.nodes ?? []).find(node => node.id === id);
const ranWithResource = (engine, id) => statNode(engine, id)?.ran === true && statNode(engine, id).resources?.length > 0;
const passNames = engine => engine.passes().filter(pass => pass.enabled !== false).map(pass => pass.name);
const feedOf = doc => doc.links.find(entry => entry.to === 'forward.ao')?.from;

async function defaultGraph(engine, state, renderFrames, name, fact) {
  const result = setGraph(engine, state, starter(state, name));
  await renderFrames(2);
  const active = engine.renderGraphActive();
  check(state, fact, result.ok === true && active, `ok ${result.ok}, active ${active}`);
}

async function twoAoSplit(engine, state, renderFrames) {
  const result = setGraph(engine, state, twoAoDoc(starter(state, SIMPLE)));
  await renderFrames(2);
  const names = passNames(engine);
  const ran = ranWithResource(engine, 'aoNarrowVsWide') && ranWithResource(engine, 'grid');
  check(state, 'two-ao-branches-ran', result.ok === true && names.includes('ao.gtao') && names.includes('ao.gtao#aoWide') && ran,
    `ok ${result.ok}, ao.gtao ${names.includes('ao.gtao')}, ao.gtao#aoWide ${names.includes('ao.gtao#aoWide')}, compare and grid ran ${ran}`);
}

async function forwardAoPair(engine, state, half, renderFrames) {
  const doc = pairDoc(starter(state, SIMPLE), half);
  const result = setGraph(engine, state, doc);
  await renderFrames(2);
  const aoEnabled = passNames(engine).includes('ao.gtao');
  state.pair.push({ ok: result.ok === true, source: feedOf(doc), ao: aoEnabled, half });
  if (half === 'right') {
    const followed = state.pair.length === 2 && state.pair.every(entry => entry.ok && entry.ao && entry.source === `${FEED[entry.half].node}.ao`);
    check(state, 'forward-follows-ao-link', followed,
      state.pair.map(entry => `${entry.half}: ok ${entry.ok}, forward.ao from ${entry.source}, ao.gtao ${entry.ao}`).join('; '));
  }
}

async function prepareHalf(engine, _lib, variant, state, half, { renderFrames }) {
  switch (variant.id) {
    case 'two-ao-split': return twoAoSplit(engine, state, renderFrames);
    case 'forward-ao-pair': return forwardAoPair(engine, state, half, renderFrames);
    case 'default-full-gi': return defaultGraph(engine, state, renderFrames, FULL_GI, 'full-gi-slots');
    case 'default-simple-forward': return defaultGraph(engine, state, renderFrames, SIMPLE, 'simple-forward-slots');
    default: throw new Error(`render-graph slots: unknown mode ${variant.id}`);
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
  id: 'render-graph-slots',
  milestone: 'slots',
  title: 'Render graph slots: two AO branches compared, forward fed by a chosen AO branch, default graphs',
  engine: {},
  variants: [
    makeVariant('two-ao-split', 'Simple forward with two AO branches (0.3 m, 2 m) in a split compare beside the final frame'),
    makeVariant('forward-ao-pair', 'Simple forward fed by the 2 m AO branch (left) and by the 0.3 m branch (right)',
      { left: 'forward ao from 2 m branch', right: 'forward ao from 0.3 m branch', order: ['left', 'right'] }),
    makeVariant('default-full-gi', 'Full GI starter unchanged'),
    makeVariant('default-simple-forward', 'Simple forward starter unchanged'),
  ],
  async build(engine, lib) {
    await lib.scene.buildRoom(engine);
    return { starters: engine.renderGraphCatalog().starters, assertions: [], edits: [], pair: [] };
  },
  prepareHalf,
};
