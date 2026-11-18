import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as interpreter from './scenes/interpreter.js';
import * as views from './scenes/views.js';
import * as parameters from './scenes/parameters.js';
import { validateInventory } from './lib/inventory.js';

// The parameters stage against an engine double. The double's catalog holds the Full GI starter
// text (crates/raa-graph/graphs/full-gi.json, read from the repository) and the parameters this
// stage reads, at the defaults of the cited Rust lines and of the tunables table. Its colour
// inputs carry the chain_sources defaults of crates/raa-graph/src/nodes/passes.rs.

const { PARAMETER_CHANGES } = parameters;
const ROOT = new URL('../../', import.meta.url);
const STARTER_TEXT = readFileSync(new URL('crates/raa-graph/graphs/full-gi.json', ROOT), 'utf8');
const TABLE_TEXT = readFileSync(new URL('docs/render-graph/tunables.md', ROOT), 'utf8');

const param = ([key, value, min = null, max = null]) => ({ key, default: value, min, max });
const PARAMS = {
  'pass.shadow.csm': [['softness', 1, 0, 16]],
  'pass.ao.gtao': [['power', 1, 0, 8]],
  'pass.shadow.contact': [['distance', 0.35, 0, 5]],
  'pass.ssr': [['maxSteps', 64]],
  'pass.dof': [['focusDistance', 10], ['fStop', 2.8, 0.7, 64], ['focalLength', 50], ['maxCoc', 24]],
  'pass.fog': [['density', 0, 0]],
  'pass.sunshafts': [['intensity', 1, 0, 8], ['density', 0.02, 0, 1], ['maxDistance', 60]],
  'pass.bloom': [['strength', 0.04, 0, 1]],
  'pass.grading': [['temperature', 6500], ['tint', 0], ['exposure', 0], ['contrast', 1], ['saturation', 1, 0, 4],
    ['vibrance', 0], ['lift', [0, 0, 0]], ['gamma', [1, 1, 1]], ['gain', [1, 1, 1]], ['shadowsTint', [0.5, 0.5, 0.5]],
    ['highlightsTint', [0.5, 0.5, 0.5]], ['splitBalance', 0], ['vignette', 0]],
  'pass.tonemap': [['exposureEv', 0]],
  'pass.taa': [['filterSharpness', 2.3]],
  'pass.transparent.oit': [['maxFragments', 16, 4, 32], ['poolFactor', 8, 2, 32]],
  'pass.gi.volume': [['intensity', 1, 0, 16]],
  'pass.gi.screen': [['indirect', 'probes'], ['spatialPasses', 3, 0, 5], ['restirMCap', 20, 1, 64]],
  'pass.gi.area-shadow': [['temporalAlpha', 0.1]],
  'pass.gi.ssgi': [['stepsMedium', 8]],
  'pass.sky.atmosphere': [['skyViewSteps', 32]],
  'pass.sky.cloud-lighting': [['cloudAlbedo', 0.98]],
};
// The colour chain in order (type, order); a chain pass reads every earlier chain pass, nearest first.
const CHAIN = [
  ['pass.forward', 60], ['pass.gi.history', 64], ['pass.ssr', 65], ['pass.fog', 68], ['pass.sunshafts', 69],
  ['pass.opaque.copy', 70], ['pass.transparent.oit', 80], ['pass.particles.blend', 85], ['pass.taa', 100],
  ['pass.dof', 110], ['pass.bloom', 120], ['pass.grading', 130], ['pass.cel.resolve', 190], ['pass.tonemap', 200],
  ['pass.overlay', 220],
];
const colourSources = order => CHAIN.filter(([, rank]) => rank < order).map(([type]) => `${type}.color`).reverse();
const TYPES = [...new Set([...CHAIN.map(([type]) => type), ...Object.keys(PARAMS)])];
const CATALOG_NODES = [
  ...TYPES.map(id => {
    const order = CHAIN.find(([type]) => type === id)?.[1];
    const inputs = order === undefined ? [] : [{ key: 'color', generic: false, defaultSource: colourSources(order) }];
    return { id, inputs, params: (PARAMS[id] ?? []).map(entry => param(entry)) };
  }),
  { id: 'output.display', inputs: [{ key: 'color', generic: false, defaultSource: ['pass.overlay.color', 'pass.tonemap.color'] }], params: [] },
];

// The nodes each added family adds to the starter, at the fixture's catalog defaults.
const ADDS = { dof: ['dof'], shafts: ['sunshafts'], grading: ['grading'], oit: ['transparentOit'] };
const ADDED_EXPECTED = {
  dof: { id: 'dof', type: 'pass.dof', params: { focusDistance: 10, fStop: 2.8, focalLength: 50, maxCoc: 24 } },
  sunshafts: { id: 'sunshafts', type: 'pass.sunshafts', params: { intensity: 1, density: 0.02, maxDistance: 60 } },
  grading: {
    id: 'grading', type: 'pass.grading',
    params: {
      temperature: 6500, tint: 0, exposure: 0, contrast: 1, saturation: 1, vibrance: 0, lift: [0, 0, 0],
      gamma: [1, 1, 1], gain: [1, 1, 1], shadowsTint: [0.5, 0.5, 0.5], highlightsTint: [0.5, 0.5, 0.5],
      splitBalance: 0, vignette: 0,
    },
  },
  transparentOit: { id: 'transparentOit', type: 'pass.transparent.oit', params: { maxFragments: 16, poolFactor: 8 } },
};
// The colour chain of each family: the starter's chain with the added nodes in place.
const BASE_CHAIN = ['forward', 'giHistory', 'ssr', 'fog', 'taa', 'bloom', 'tonemap', 'out'];
const CHAIN_OF = {
  shafts: ['forward', 'giHistory', 'ssr', 'fog', 'sunshafts', 'taa', 'bloom', 'tonemap', 'out'],
  dof: ['forward', 'giHistory', 'ssr', 'fog', 'taa', 'dof', 'bloom', 'tonemap', 'out'],
  grading: ['forward', 'giHistory', 'ssr', 'fog', 'taa', 'bloom', 'grading', 'tonemap', 'out'],
  oit: ['forward', 'giHistory', 'ssr', 'fog', 'transparentOit', 'taa', 'bloom', 'tonemap', 'out'],
};

// The engine double: a catalog, the starter, and the documents each setRenderGraph was given.
function engineDouble() {
  const documents = [];
  let active = null;
  return {
    documents,
    renderGraphCatalog: () => ({ nodes: CATALOG_NODES, starters: [{ name: 'Full GI', file: 'full-gi.json', json: STARTER_TEXT }] }),
    addNode: () => 1,
    addSurfaceMaterial: () => 2,
    primitiveMesh: () => 3,
    addInstance: () => undefined,
    setRenderGraph(graph) {
      active = JSON.parse(typeof graph === 'string' ? graph : JSON.stringify(graph));
      documents.push(structuredClone(active));
      return { ok: true, issues: [], rebuilt: 0 };
    },
    renderGraphActive: () => active !== null,
    renderGraph: () => structuredClone(active),
    renderGraphIssues: () => [],
  };
}

const lib = { scene: { buildRoom: async () => ({ subjects: [] }) } };
async function prepare(engine, mode, halves) {
  const variant = parameters.default.variants.find(entry => entry.id === mode);
  const state = await parameters.default.build(engine, lib, variant);
  for (const half of halves) {
    await parameters.default.prepareHalf(engine, lib, variant, state, half, { renderFrames: async () => {} });
  }
  return state;
}
const byId = doc => Object.fromEntries(doc.nodes.map(node => [node.id, node]));
const linksOf = doc => doc.links.map(link => `${link.from} -> ${link.to}`).sort();
const chainLinks = ids => ids.slice(1).map((to, index) => `${ids[index]}.color -> ${to}.color`).sort();

// docs/render-graph/tunables.md: the triage table, rows keyed by its header.
function tableRows(text) {
  const lines = text.split('\n');
  const start = lines.findIndex(line => line.startsWith('| crate | file:line |'));
  const cells = line => line.split('|').slice(1, -1).map(cell => cell.trim());
  const header = cells(lines[start]);
  return lines.slice(start + 2).filter(line => line.startsWith('|')).map(cells)
    .filter(row => row.length === header.length)
    .map(row => Object.fromEntries(row.map((cell, index) => [header[index], cell])));
}
const TABLE = tableRows(TABLE_TEXT);

// A stated range such as [0, 1] or (0, 16]; null for `unstated` or any other text.
function statedRange(text) {
  const match = /^([[(])\s*(-?[\d.]+(?:e[-+]?\d+)?)\s*,\s*(-?[\d.]+(?:e[-+]?\d+)?)\s*([\])])$/.exec(text);
  return match ? { min: Number(match[2]), max: Number(match[3]), minOpen: match[1] === '(', maxOpen: match[4] === ')' } : null;
}

test('the parameters inventory has the defaults row and one row per family, 20 unique rows', () => {
  const rows = validateInventory([interpreter, views, parameters], 'parameters').filter(row => row.scenario === 'render-graph-parameters');
  assert.equal(rows.length, 20);
  assert.equal(parameters.inventory.length, 20);
  assert.deepEqual(rows.map(row => row.id), ['parameters-defaults', ...PARAMETER_CHANGES.map(change => `parameters-${change.family}`)]);
  assert.equal(rows[0].filename, 'parameters-full-gi-all-defaults.png');
  for (const change of PARAMETER_CHANGES) {
    const row = rows.find(entry => entry.id === `parameters-${change.family}`);
    assert.equal(row.filename, `parameters-${change.family}-default-left-changed-right.png`);
    assert.equal(row.mode, change.family);
  }
  assert.equal(new Set(rows.map(row => row.filename)).size, 20);
  assert.throws(() => validateInventory([interpreter, views], 'parameters'), /stage parameters is unavailable/);
});

test('the family rows follow the contract order', () => {
  assert.deepEqual(PARAMETER_CHANGES.map(change => change.family), ['shadow', 'gtao', 'contact', 'ssr', 'dof', 'fog',
    'shafts', 'bloom', 'grading', 'tonemap', 'taa', 'oit', 'gi-volume', 'gi-screen', 'gi-restir', 'area-shadow', 'ssgi',
    'sky', 'clouds']);
});

for (const change of PARAMETER_CHANGES.filter(entry => entry.binding === 'setting')) {
  test(`setting row ${change.family}: ${change.key} = ${change.value} differs from its default and lies in its range`, () => {
    assert.ok(Number.isFinite(change.value));
    assert.notEqual(change.value, change.default);
    if (change.min !== null) assert.ok(change.minExclusive ? change.value > change.min : change.value >= change.min);
    if (change.max !== null) assert.ok(change.value <= change.max);
  });
}

for (const change of PARAMETER_CHANGES.filter(entry => entry.binding === 'tunable')) {
  test(`tunable row ${change.family}: ${change.nodeType} ${change.key} differs from its triage default and lies in its range`, () => {
    const matches = TABLE.filter(row => row.class === 'tunable' && row['node type'] === change.nodeType && row.key === change.key);
    assert.equal(matches.length, 1, `one triage row for ${change.nodeType} ${change.key}`);
    const [row] = matches;
    const fallback = Number.parseFloat(row.value);
    assert.ok(Number.isFinite(fallback));
    assert.notEqual(change.value, fallback);
    const range = statedRange(row.range);
    if (range) {
      assert.ok(range.minOpen ? change.value > range.min : change.value >= range.min);
      assert.ok(range.maxOpen ? change.value < range.max : change.value <= range.max);
    }
  });
}

test('the defaults row: the starter is accepted and the all-defaults half writes every parameter of every node', async () => {
  const engine = engineDouble();
  const state = await prepare(engine, 'full-gi-defaults', ['left', 'right']);
  const [left, right] = engine.documents;
  assert.deepEqual(left, JSON.parse(STARTER_TEXT));
  assert.deepEqual(state.edits.map(edit => edit.accepted), [true, true]);
  assert.deepEqual(state.assertions.map(item => [item.name, item.passed]), [['defaults-accepted', true]]);
  for (const node of right.nodes) {
    for (const entry of CATALOG_NODES.find(spec => spec.id === node.type)?.params ?? []) {
      assert.ok(entry.key in node.params, `${node.id}.${entry.key}`);
    }
  }
});

for (const change of PARAMETER_CHANGES) {
  test(`${change.family} row: the change reaches the right half only, and the left half is the starter with its add-ons`, async () => {
    const engine = engineDouble();
    const state = await prepare(engine, change.family, ['left', 'right']);
    const [left, right] = engine.documents;
    assert.deepEqual(state.edits.map(edit => edit.accepted), [true, true]);
    assert.deepEqual(state.assertions.map(item => [item.name, item.passed]), [['parameter-changed', true]]);

    const expected = JSON.parse(STARTER_TEXT);
    if (change.family === 'gi-restir') expected.nodes.find(node => node.id === 'giScreen').params.indirect = 'restir';
    const starterIds = new Set(expected.nodes.map(node => node.id));
    const leftNodes = byId(left);
    assert.deepEqual(Object.fromEntries(Object.entries(leftNodes).filter(([id]) => starterIds.has(id))), byId(expected));
    for (const id of ADDS[change.family] ?? []) assert.deepEqual(leftNodes[id], ADDED_EXPECTED[id]);
    assert.deepEqual(linksOf(left), chainLinks(CHAIN_OF[change.family] ?? BASE_CHAIN));

    assert.equal(byId(right)[change.nodeId].params[change.key], change.value);
    const restored = structuredClone(right);
    restored.nodes.find(node => node.id === change.nodeId).params[change.key] = leftNodes[change.nodeId].params[change.key];
    assert.deepEqual(restored, left);
  });
}

// pass.forward's opaqueColor/opaqueDepth inputs default to pass.opaque.copy (crates/raa-graph/src/nodes/passes.rs),
// which reads forward's colour chain: adding that node makes the engine refuse the graph with graph.cycle
// (run 15402: the oit row's both halves were refused and renderGraph() showed the no-graph value).
test('no family row adds pass.opaque.copy, which would close a graph.cycle through forward', async () => {
  for (const change of PARAMETER_CHANGES) {
    const engine = engineDouble();
    await prepare(engine, change.family, ['left', 'right']);
    for (const doc of engine.documents) assert.ok(!doc.nodes.some(node => node.type === 'pass.opaque.copy'), change.family);
  }
});

// Run 15631: the eight oit panes stood exactly behind one another on the view ray and read as one
// square. Seen from the capture camera, each pane must step sideways from the previous one while
// the eight 1 m panes still share a common core (the overlap where maxFragments shows).
test('the oit panes fan out across the view and keep a common core of all eight', async () => {
  const { cameraAt } = await import('./lib/scene.js');
  const { RG_CAPTURE_T_S } = await import('./lib/scene.js');
  const eye = cameraAt(RG_CAPTURE_T_S).position;
  const panes = parameters.paneLayout();
  assert.equal(panes.length, 8);
  // Direction from the eye to each centre, split into a horizontal angle and an elevation.
  const angles = panes.map(({ centre }) => {
    const d = centre.map((value, axis) => value - eye[axis]);
    return { yaw: Math.atan2(d[0], d[2]), pitch: Math.atan2(d[1], Math.hypot(d[0], d[2])), dist: Math.hypot(...d) };
  });
  for (let i = 1; i < angles.length; i++) {
    const shift = Math.hypot(angles[i].yaw - angles[i - 1].yaw, angles[i].pitch - angles[i - 1].pitch);
    // At least 0.05 m of the nearer pane's distance: a visible stripe of every colour.
    assert.ok(shift * Math.min(angles[i].dist, angles[i - 1].dist) > 0.05, `pane ${i} hides behind pane ${i - 1}`);
  }
  // The whole fan, in metres at the nearest pane, stays well inside a pane's 1 m width.
  const nearest = Math.min(...angles.map(a => a.dist));
  const spreadYaw = Math.max(...angles.map(a => a.yaw)) - Math.min(...angles.map(a => a.yaw));
  const spreadPitch = Math.max(...angles.map(a => a.pitch)) - Math.min(...angles.map(a => a.pitch));
  assert.ok(spreadYaw * nearest < 0.8 && spreadPitch * nearest < 0.8, 'the eight panes keep no common core');
});

test('capping the oit row at 4 fragments visibly changes the common core', () => {
  const panes = parameters.paneLayout();
  assert.ok(panes.length >= 6, 'the common core needs at least six panes');
  const colourKeys = panes.map(({ color }) => color.join(','));
  assert.equal(new Set(colourKeys).size, panes.length, 'pane colours must all be distinct');

  // raa-oit uses back-face culling for ordinary transparent materials, so each thin box writes
  // one fragment per pixel in the core (crates/raa-oit/src/feature.rs:204).
  const fragmentsPerPane = 1;
  // Pane colours enter the JS API as sRGB and are blended in linear space by the GPU.
  const srgbToLinear = channel => channel <= 0.04045
    ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  const fragments = panes.flatMap(({ color, opacity }) =>
    Array.from({ length: fragmentsPerPane }, () => ({ color: color.map(srgbToLinear), opacity })));
  const blendFarToNear = source => source.slice().reverse().reduce((dst, { color, opacity }) =>
    color.map((channel, index) => channel * opacity + dst[index] * (1 - opacity)), [0.5, 0.5, 0.5]);
  const allFragments = blendFarToNear(fragments);
  const nearestFour = blendFarToNear(fragments.slice(0, 4));
  const maxDifference = Math.max(...allFragments.map((channel, index) => Math.abs(channel - nearestFour[index])));
  assert.ok(maxDifference >= 0.12, `max channel difference ${maxDifference} is below 0.12`);
});
