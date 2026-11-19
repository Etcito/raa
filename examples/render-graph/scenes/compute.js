// Compute-stage scenes derive graph documents only from the engine catalog.
import { RG_CAPTURE_T_S, selectionAt } from '../lib/scene.js';
import { captionText } from '../lib/runtime.js';

const FULL_GI_STAGES = 'Full GI stages';
const CUSTOM_DENOISE = 'Custom denoise';
const GAUSSIAN_BLUR = 'Gaussian blur';
const HELD_CONSTANT = ['room geometry', 'materials', 'lights', 'sky', 'camera'];
const FACTS = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:ready'];

const row = (id, mode, filename, criterionName, fact, producer, expectedVisibleBehavior) => ({
  id, criterionName, scenario: 'render-graph-compute', mode, filename, expectedVisibleBehavior,
  requiredFacts: [...FACTS, `assertion:${fact}`], pair: null,
  heldConstant: [...HELD_CONSTANT], producer,
});

export const factNames = ['custom-denoise-replaces-atrous', 'gaussian-preset-ran', 'wgsl-error-on-node'];
export const inventory = [
  row('compute-denoise-pair', 'denoise-pair', 'compute-custom-denoise-bilateral-left-atrous-right.png',
    'P5-Q11 a custom compute denoiser replaces the a-trous stage with a clean result', factNames[0],
    'raa custom denoise starter: bilateral compute node replaces the a-trous stages',
    'Two halves of the same view of the room, labelled custom bilateral denoise (left) and a-trous denoise (right). Both halves show the room lit by the sun with soft indirect light: shadowed areas dim but not black, sun shadows under the boxes, the emissive panel glowing on the back wall and a faint red tint on the floor near the red wall. The indirect light on the walls and floor is smooth in both halves, without grain, speckles, blotches or black holes; the two halves may differ slightly in softness.'),
  row('compute-gaussian-ao', 'gaussian-ao', 'compute-preset-gaussian-blur-on-ao.png',
    'P5-Q11 a preset compute node runs on an engine image', factNames[1],
    'raa presets 1/2: blend, gaussian blur, box blur',
    'The view shows a full-screen greyscale ambient-occlusion image of the room instead of the lit scene: open floor and wall areas near white, darker grey where the walls meet the floor, in the room corners and around the bases of the boxes and the sphere. The darkening is soft and smoothly blurred, with no fine pixel noise. No colours and no direct lighting.'),
  row('compute-wgsl-error', 'wgsl-error', 'compute-wgsl-error-reported-on-node.png',
    'P5-Q11 a WGSL error is reported on its node with its line', factNames[2],
    'raa-graph compute 2/7: declaration and WGSL validation',
    'The room still renders lit by the sun with soft indirect light and sun shadows under the boxes (the previous Custom denoise graph keeps rendering), and a caption strip under the image reads graph.wgsl, names the node giDenoise and gives the WGSL line number of the error.'),
];

function makeVariant(id, description, pair = null) {
  return { id, description, references: [], judge: true, pair, selection: selectionAt(RG_CAPTURE_T_S) };
}
function catalogEntry(state, entries, name) {
  const found = entries.find(entry => entry.name === name);
  if (!found) throw new Error(`render-graph compute: catalog has no ${name}`);
  return found.json;
}
const check = (state, name, passed, detail) => state.assertions.push({ name, passed: passed === true, detail });
function setGraph(engine, state, doc) {
  const result = engine.setRenderGraph(doc), [first] = result.issues ?? [];
  state.edits.push({ operation: 'setRenderGraph', accepted: result.ok === true,
    error: result.ok === true ? null : { code: first?.code ?? 'graph.refused', message: first?.message ?? 'refused' } });
  return result;
}
const nodeOf = (doc, type) => doc.nodes.find(node => node.type === type);
const statNode = (engine, id) => (engine.renderGraphStats()?.nodes ?? []).find(node => node.id === id);
const passNames = engine => engine.passes().filter(pass => pass.enabled !== false).map(pass => pass.name);
const linksTo = (doc, target) => doc.links.filter(link => link.to !== target);

function showInView(doc, source, channels, label) {
  const out = nodeOf(doc, 'output.display');
  const view = { id: 'showCompute', type: 'view.image', params: { channels, label } };
  doc.nodes.splice(doc.nodes.indexOf(out), 0, view);
  doc.links = linksTo(doc, `${out.id}.color`);
  doc.links.push({ from: source, to: 'showCompute.src' },
    { from: 'showCompute.color', to: `${out.id}.color` });
  return doc;
}

function gaussianAo(state) {
  const doc = JSON.parse(catalogEntry(state, state.starters, FULL_GI_STAGES));
  const preset = JSON.parse(catalogEntry(state, state.presets, GAUSSIAN_BLUR));
  const ao = doc.nodes.find(node => node.type === 'pass.ao.gtao');
  const out = nodeOf(doc, 'output.display');
  const blur = { ...preset.nodes[0], id: 'aoBlur' };
  doc.nodes.splice(doc.nodes.indexOf(out), 0, blur);
  doc.links = linksTo(doc, `${out.id}.color`);
  doc.links.push({ from: `${ao.id}.ao`, to: 'aoBlur.src' });
  doc.name = 'Full GI stages gaussian AO';
  return showInView(doc, 'aoBlur.dst', 'r', 'gaussian AO');
}

async function prepareHalf(engine, _lib, variant, state, half, { renderFrames }) {
  if (variant.id === 'denoise-pair') {
    const name = half === 'left' ? CUSTOM_DENOISE : FULL_GI_STAGES;
    const result = setGraph(engine, state, catalogEntry(state, state.starters, name));
    await renderFrames(2);
    state.pair.push({ half, accepted: result.ok === true, graph: engine.renderGraphActive() ? engine.renderGraph().name : null,
      passes: passNames(engine), denoiseRan: statNode(engine, 'giDenoise')?.ran === true });
    if (half === 'right') {
      const left = state.pair.find(item => item.half === 'left');
      const passed = !!left?.accepted && result.ok === true && left.passes.includes('compute#giDenoise')
        && !left.passes.some(name => name.startsWith('gi.atrous#')) && left.denoiseRan
        && passNames(engine).includes('gi.atrous#giAtrous1');
      check(state, factNames[0], passed, `left passes ${left?.passes.join(', ') ?? 'missing'}; compute ran ${left?.denoiseRan}; right passes ${passNames(engine).join(', ')}`);
    }
    return;
  }
  if (variant.id === 'gaussian-ao') {
    const doc = gaussianAo(state), result = setGraph(engine, state, doc);
    await renderFrames(2);
    const passed = result.ok === true && passNames(engine).includes('compute#aoBlur')
      && statNode(engine, 'aoBlur')?.ran === true;
    check(state, factNames[1], passed, `ok ${result.ok}; compute#aoBlur ${passNames(engine).includes('compute#aoBlur')}; ran ${statNode(engine, 'aoBlur')?.ran}`);
    return;
  }
  if (variant.id === 'wgsl-error') {
    const text = catalogEntry(state, state.starters, CUSTOM_DENOISE);
    const first = setGraph(engine, state, text);
    const doc = JSON.parse(text), denoise = doc.nodes.find(node => node.id === 'giDenoise');
    const expectedLine = (denoise.wgsl.match(/\n/g) ?? []).length + 2;
    denoise.wgsl += `fn brokenLine() -> f32 {\n  let x = ;\n  return 1.0;\n}\n`;
    doc.name = 'Custom denoise broken';
    const result = setGraph(engine, state, doc), [issue] = result.issues ?? [];
    state.caption = issue ? captionText(issue) : null;
    await renderFrames(2);
    check(state, factNames[2], first.ok === true && result.ok === false && issue?.code === 'graph.wgsl'
      && issue.nodeId === 'giDenoise' && issue.message.includes(`line ${expectedLine},`)
      && engine.renderGraphActive() === true && engine.renderGraph().name === CUSTOM_DENOISE,
    `first ${first.ok}; second ${result.ok}; issue ${issue?.code} on ${issue?.nodeId}: ${issue?.message}; active ${engine.renderGraph()?.name}`);
    return;
  }
  throw new Error(`render-graph compute: unknown mode ${variant.id}`);
}

export function observe(engine, _variant, state) {
  return { renderGraphActive: engine.renderGraphActive(),
    graph: engine.renderGraphActive() ? engine.renderGraph().name : null,
    issues: engine.renderGraphIssues(), assertions: state.assertions, edits: state.edits };
}

function caption(_engine, state) {
  return state.caption;
}

export default {
  id: 'render-graph-compute', milestone: 'compute',
  title: 'Render graph compute nodes: custom denoise, presets and WGSL errors', engine: {},
  variants: [
    makeVariant('denoise-pair', 'Custom bilateral denoise (left) and a-trous denoise (right)',
      { left: 'custom bilateral denoise', right: 'a-trous denoise', order: ['left', 'right'] }),
    makeVariant('gaussian-ao', 'Gaussian blur compute preset on ambient occlusion'),
    makeVariant('wgsl-error', 'WGSL validation error reported on its compute node'),
  ],
  async build(engine, lib) {
    await lib.scene.buildRoom(engine);
    const catalog = engine.renderGraphCatalog();
    return { starters: catalog.starters, presets: catalog.presets, assertions: [], edits: [], pair: [], caption: null };
  },
  prepareHalf,
  caption,
};
