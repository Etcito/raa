import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { validateInventory } from './lib/inventory.js';
import * as bakeNamespace from './scenes/bake.js';
import * as passesNamespace from './scenes/passes.js';

// Inventory tests of the passes stage (scenes/passes.js), then the rules of its rows and variants.
// Node only; no engine, no WebGPU.

const scene = passesNamespace.default;
const { factNames, inventory } = passesNamespace;

const FILENAMES = [
  'dense-field-vg-on-many-high-poly-instances.png',
  'dense-field-vg-off-forward-path.png',
  'dense-field-vg-on-hardware-raster-only.png',
  'continuous-lod-vg-on-five-distances-no-cracks.png',
  'continuous-lod-vg-off-forward-path.png',
  'occlusion-vg-on-walls-hide-most-instances.png',
  'occlusion-vg-off-forward-path.png',
  'occlusion-vg-on-occlusion-test-off-same-content.png',
  'shadows-vg-on-cast-and-received-in-cascades.png',
  'shadows-vg-off-forward-path.png',
  'shadows-vg-on-virtual-casters-off.png',
  'mixed-vg-on-beside-forward-skinned-transparent-fur.png',
  'mixed-vg-off-forward-path.png',
  'runtime-bake-forward-while-baking.png',
  'runtime-bake-virtual-when-ready.png',
  'runtime-bake-vg-off-forward-path.png',
  'debug-clusters-coloured-patches.png',
  'debug-lod-level-ramp-by-distance.png',
  'debug-raster-hardware-versus-software.png',
  'debug-occlusion-first-versus-second-phase.png',
  'debug-views-off-normal-shading.png',
];
const VARIANT_IDS = [
  'dense-field-on', 'dense-field-off', 'dense-field-hardware-only', 'occlusion-test-off', 'shadows-vg-shadows-off',
  'continuous-lod-on', 'continuous-lod-off', 'occlusion-on', 'occlusion-off', 'shadows-on', 'shadows-off',
  'mixed-on', 'mixed-off', 'runtime-bake-baking', 'runtime-bake-on', 'runtime-bake-off',
  'debug-clusters-on', 'debug-lod', 'debug-raster', 'debug-occlusion', 'debug-views-off',
];
const PAIRS = ['dense-field', 'continuous-lod', 'occlusion', 'shadows', 'mixed', 'runtime-bake', 'debug-views'];
const FACT_NAMES = [
  'vg-active', 'vg-inactive-forward', 'lod-triangles-fall-with-distance', 'occlusion-reduces-drawn-clusters',
  'shadow-clusters-drawn', 'refusals-reported', 'bake-states-in-order', 'debug-view-set',
];
// Rows that need vg-active without an -on suffix: the three feature toggles (their -off id is the toggled feature).
const TOGGLES = ['dense-field-hardware-only', 'occlusion-test-off', 'shadows-vg-shadows-off'];
const sorted = values => [...values].sort();

// A filename starts with its preset: the scene id, or 'debug' for the debug-view rows (their filenames start with 'debug-').
const presetOf = variant => (variant.id.startsWith('debug-') ? 'debug' : variant.scene);

const rows = validateInventory([bakeNamespace, passesNamespace], 'passes');
const passesRows = rows.filter(row => row.scenario === 'passes');
const variantOf = new Map(scene.variants.map(variant => [variant.id, variant]));

test('scene id, milestone, msaa and exports', () => {
  assert.equal(scene.id, 'passes');
  assert.equal(scene.milestone, 'passes');
  assert.equal(scene.engine.msaa, 1);
  assert.deepEqual(factNames, FACT_NAMES);
  assert.equal(typeof passesNamespace.observe, 'function');
  assert.equal(typeof scene.build, 'function');
  assert.equal(typeof scene.ready, 'function');
});

test('inventory of stage passes: twenty-one unique filenames and variant ids', () => {
  assert.equal(passesRows.length, 21);
  assert.deepEqual(sorted(passesRows.map(row => row.filename)), sorted(FILENAMES));
  assert.equal(new Set(passesRows.map(row => row.filename)).size, 21);
  assert.deepEqual(sorted(passesRows.map(row => row.mode)), sorted(VARIANT_IDS));
  assert.deepEqual(sorted(scene.variants.map(variant => variant.id)), sorted(VARIANT_IDS));
  for (const row of passesRows) assert.equal(row.pair, variantOf.get(row.mode).measurement?.pair ?? null);
});

test('every measurement pair has one on and one off variant with the same preset', () => {
  const pairs = new Set(scene.variants.map(variant => variant.measurement?.pair).filter(Boolean));
  assert.deepEqual(sorted(pairs), sorted(PAIRS));
  for (const pair of PAIRS) {
    const members = scene.variants.filter(variant => variant.measurement?.pair === pair);
    assert.equal(members.length, 2, `${pair} has ${members.length} variants`);
    assert.deepEqual(sorted(members.map(variant => variant.measurement.state)), ['off', 'on'], pair);
    assert.equal(members[0].measurement.preset, members[1].measurement.preset, `${pair} presets differ`);
    for (const member of members) assert.ok(member.id.endsWith(`-${member.measurement.state}`), member.id);
  }
});

test('row requirements: pair rows, -off rows, -on rows and the three toggles', () => {
  for (const row of passesRows) {
    if (row.pair !== null) {
      assert.ok(row.requiredFacts.includes(`measurement:${row.pair}`), `${row.id} lacks measurement:${row.pair}`);
      assert.ok(row.requiredFacts.includes('snapshot:stats'), `${row.id} lacks snapshot:stats`);
    }
    if (row.mode.endsWith('-off') && !TOGGLES.includes(row.mode)) {
      assert.ok(row.requiredFacts.includes('assertion:vg-inactive-forward'), `${row.id} lacks vg-inactive-forward`);
    }
    if (row.mode.endsWith('-on') || TOGGLES.includes(row.mode)) {
      assert.ok(row.requiredFacts.includes('assertion:vg-active'), `${row.id} lacks vg-active`);
    }
  }
});

test('every variant has subjects in view and the inventory rows are the same set', () => {
  for (const variant of scene.variants) {
    assert.ok(variant.selection.subjects.length >= 1, `${variant.id} has no subject`);
  }
  assert.deepEqual(sorted(inventory.map(row => row.filename)), sorted(FILENAMES));
});

test('filenames are descriptive: at least four hyphenated words and they start with their preset', () => {
  for (const row of passesRows) {
    assert.ok(row.filename.endsWith('.png'), row.filename);
    const stem = row.filename.slice(0, -'.png'.length);
    assert.ok(stem.split('-').length >= 4, `${row.filename} has fewer than four words`);
    const variant = variantOf.get(row.mode);
    assert.ok(stem.startsWith(`${presetOf(variant)}-`), `${row.filename} does not start with ${presetOf(variant)}`);
  }
});

test('no row uses more than one viewport', () => {
  const doc = fs.readFileSync(new URL('../../docs/api/m21-views.md', import.meta.url), 'utf8');
  const multiView = /`(set\w+)\(rects\)`/.exec(doc)?.[1];
  assert.equal(multiView, 'setViewports', 'the m21 multi-view method is not named in its doc');
  const source = fs.readFileSync(new URL('./scenes/passes.js', import.meta.url), 'utf8');
  assert.doesNotMatch(source, new RegExp(`\\b${multiView}\\s*\\(`));
  for (const variant of scene.variants) assert.equal(Object.hasOwn(variant, 'views'), false, `${variant.id} has views`);
});
