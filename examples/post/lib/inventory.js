// Browser-free validation of the P5 post-page scene inventories (S49, module post-page).
// Form follows examples/global-illumination/lib/inventory.js; the stages, settings and selectors are the post ones.
// Each variant's selection is framed by assertSubjectsInView (web/common/view-check.mjs:80).
import { assertSubjectsInView } from '../../../web/common/view-check.mjs';

const stages = ['exposure', 'sharpen', 'mips'];
const fixedSelectors = ['runtime:passes', 'runtime:viewSizes', 'runtime:ready'];
const namePattern = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
const filePattern = /^[a-z0-9]+(?:-[a-z0-9]+)+\.png$/;

// The settings a variant may name, in the order the runtime applies them; the index is the rank.
export const SETTING_ORDER = Object.freeze(['toneMapping', 'passes', 'renderScale', 'autoExposure', 'sharpening']);

const text = value => typeof value === 'string' && value.trim().length > 0;
const string = value => typeof value === 'string';
const named = value => string(value) && namePattern.test(value);
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const count = value => Number.isSafeInteger(value) && value >= 0;
const positive = value => Number.isSafeInteger(value) && value > 0;
const bool = value => typeof value === 'boolean';
const fields = (o, schema) => record(o) && Object.entries(schema).every(([key, test]) => test(o[key]));
const nullable = test => value => value === null || test(value);
const enumeration = values => value => values.includes(value);
const list = test => value => Array.isArray(value) && value.every(test);
const nonEmpty = test => value => Array.isArray(value) && value.length > 0 && value.every(test);
const features = value => record(value) && Object.values(value).every(v => bool(v) || string(v) || finite(v));
const factValue = value => value === null || bool(value) || finite(value);
const factValues = value => record(value) && Object.values(value).every(factValue);
const assertion = value => fields(value, { name: string, passed: bool, detail: string });
const pass = value => fields(value, { name: text, role: enumeration(['pass', 'contributor']), enabled: bool, order: finite, required: bool });
const view = value => fields(value, { view: count, width: positive, height: positive });
const readiness = value => fields(value, { status: enumeration(['pending', 'ready', 'failed']), stage: string, error: nullable(string) });
const observation = value => fields(value, { values: factValues, assertions: list(assertion), passes: list(pass), viewSizes: list(view), readiness });
function fail(message) { throw new Error(message); }

// The fact names of a scene: unique and matching the name pattern.
function registry(names, context) {
  if (!Array.isArray(names)) fail(`${context}: factNames must be an array`);
  const set = new Set();
  for (const name of names) {
    if (!named(name) || set.has(name)) fail(`${context}: invalid or duplicate fact name ${name}`);
    set.add(name);
  }
  return set;
}

function selectorKnown(value, names) {
  if (fixedSelectors.includes(value)) return true;
  return ['fact:', 'assertion:'].some(prefix => string(value) && value.startsWith(prefix) && names.has(value.slice(prefix.length)));
}

function assertionsNamed(facts, name) {
  return facts.assertions.filter(entry => entry.name === name);
}

// Whether a known selector holds in the facts (the S49 selector rules).
function holds(key, facts) {
  if (key === 'runtime:passes') return nonEmpty(pass)(facts.passes);
  if (key === 'runtime:viewSizes') return nonEmpty(view)(facts.viewSizes);
  if (key === 'runtime:ready') return facts.readiness.status === 'ready' && facts.readiness.error === null;
  if (key.startsWith('fact:')) return Object.hasOwn(facts.values, key.slice(5)) && factValue(facts.values[key.slice(5)]);
  const matches = assertionsNamed(facts, key.slice(10));
  return matches.length === 1 && matches[0].passed === true;
}

// The assertion detail appended to a refusal; named only when exactly one assertion carries the name.
function detailOf(key, facts) {
  if (!key.startsWith('assertion:')) return '';
  const matches = assertionsNamed(facts, key.slice(10));
  return matches.length === 1 && text(matches[0].detail) ? `: ${matches[0].detail}` : '';
}

// A selection has at least one subject, and each subject lies in the camera frustum (view-check).
function checkSelection(selection, context) {
  if (!record(selection) || !Array.isArray(selection.subjects) || selection.subjects.length === 0) fail(`${context}: invalid selection`);
  assertSubjectsInView(selection, context);
}

// The settings name only keys of SETTING_ORDER; `passes` is an object of booleans.
function checkSettings(settings, context) {
  if (!record(settings)) fail(`${context}: invalid settings`);
  for (const [key, value] of Object.entries(settings)) {
    if (!SETTING_ORDER.includes(key)) fail(`${context}: unknown setting ${key}`);
    if (key === 'passes' && !(record(value) && Object.values(value).every(bool))) fail(`${context}: invalid setting passes`);
  }
}

// One inventory row against its scene: the S49 row fields, unique selectors, and the pair of its mode's measurement.
function rowFields(row, sceneId, variants) {
  const valid = fields(row, {
    id: named,
    criterionName: text,
    scenario: value => value === sceneId,
    mode: value => variants.has(value),
    filename: value => string(value) && filePattern.test(value),
    expectedVisibleBehavior: text,
    requiredFacts: list(string),
    pair: nullable(text),
    heldConstant: nonEmpty(text),
    producer: text,
  });
  return valid && new Set(row.requiredFacts).size === row.requiredFacts.length
    && (variants.get(row.mode).measurement?.pair ?? null) === row.pair;
}

/**
 * `validateInventory` validates the scene namespaces up to `stage` and returns their rows, in namespace then inventory order.
 * Cumulative: a scene whose milestone rank is above the stage is skipped, and at least one scene must have the stage as its
 * milestone. Today it refuses an unknown stage or a non-array with `invalid stage or scenes: <stage>`, and `stage <stage> is
 * unavailable` when no scene has the stage; each variant's selection goes through assertSubjectsInView (web/common/view-check.mjs).
 * Handoff: post-page: exposure-room scene takes `validateInventory(namespaces: object[], stage: string) -> object[]`
 * exported by examples/post/lib/inventory.js:120 validateInventory; exposure.test.mjs attaches it by import and passes
 * [the exposure scene namespace] with stage 'exposure'; state/keys: the namespace (default, inventory, factNames, observe);
 * it returns the rows.
 * Handoff: post-page: page entry, build script and p5 commands takes `validateInventory(namespaces: object[], stage: string) -> object[]`
 * exported by examples/post/lib/inventory.js:120 validateInventory; main.js attaches it in discoverScenes, passing the discovered
 * namespaces in scenes.json order and the highest stage present; state/keys: as above.
 */
export function validateInventory(namespaces, stage) {
  const limit = stages.indexOf(stage);
  if (limit < 0 || !Array.isArray(namespaces)) fail(`invalid stage or scenes: ${stage}`);
  const sceneIds = new Set(), rowIds = new Set(), filenames = new Set(), rows = [], pairs = new Map();
  let available = false;
  for (const ns of namespaces) {
    if (!record(ns) || !record(ns.default)) fail('invalid scene namespace');
    const scene = ns.default, rank = stages.indexOf(scene.milestone);
    if (rank < 0 || !named(scene.id) || sceneIds.has(scene.id)) fail(`invalid or duplicate scene ${scene.id}`);
    sceneIds.add(scene.id);
    if (rank > limit) continue;
    if (rank === limit) available = true;
    const names = registry(ns.factNames, scene.id);
    if (!Array.isArray(ns.inventory) || !Array.isArray(scene.variants) || typeof scene.build !== 'function' || typeof ns.observe !== 'function') fail(`${scene.id}: incomplete scene exports`);
    const variants = new Map();
    for (const variant of scene.variants) {
      const context = `${scene.id}/${variant?.id}`;
      if (!record(variant) || !named(variant.id) || variants.has(variant.id)) fail(`${scene.id}: invalid or duplicate variant`);
      checkSelection(variant.selection, context);
      checkSettings(variant.settings, context);
      if (variant.settle !== undefined && !bool(variant.settle)) fail(`${context}: settle must be a boolean`);
      variants.set(variant.id, variant);
      const m = variant.measurement;
      if (m === undefined) continue;
      // The state is named by the variant id suffix (-on / -off).
      if (!fields(m, { pair: text, state: enumeration(['on', 'off']), preset: text, features }) || !variant.id.endsWith(`-${m.state}`)) fail(`${context}: invalid measurement or variant name`);
      pairs.set(m.pair, [...(pairs.get(m.pair) ?? []), m]);
    }
    for (const row of ns.inventory) {
      if (!rowFields(row, scene.id, variants) || rowIds.has(row.id) || filenames.has(row.filename)) fail(`${scene.id}: invalid inventory row ${row?.id}`);
      for (const key of row.requiredFacts) {
        if (!selectorKnown(key, names)) fail(`${row.id}: unknown selector ${key}`);
      }
      rowIds.add(row.id);
      filenames.add(row.filename);
      rows.push(row);
    }
  }
  if (!available) fail(`stage ${stage} is unavailable`);
  for (const [pair, entries] of pairs) {
    const states = new Set(entries.map(entry => entry.state));
    if (entries.length !== 2 || states.size !== 2 || entries[0].preset !== entries[1].preset) fail(`measurement pair ${pair} requires on and off states with the same preset`);
  }
  return rows;
}

/**
 * `validateRequiredFacts` checks each `requiredFacts` selector of one inventory row against the facts of its scene and throws on
 * the first that does not hold; it returns undefined otherwise. Today it refuses a facts object of the wrong shape with
 * `<id>: invalid PostObservation` (checked first), an unknown selector with `<id>: unknown selector <value>`, and a selector that
 * does not hold with `<id>: required <selector> is missing or invalid`, followed by `: <detail>` for an assertion selector whose
 * single assertion has a detail.
 * Handoff: post-page: runtime (createPostPageRunner) takes `validateRequiredFacts(row: object, facts: object, factNames: string[]) -> void`
 * exported by examples/post/lib/inventory.js:177 validateRequiredFacts; runtime.js attaches it by passing it to createPageRunner
 * as its validateRequiredFacts option; state/keys: the facts object { values, assertions, passes, viewSizes, readiness } and the
 * scene's factNames.
 */
export function validateRequiredFacts(row, facts, factNames) {
  const id = row?.id ?? '<unknown image>';
  const names = registry(factNames, id);
  if (!record(row) || !Array.isArray(row.requiredFacts)) fail(`${id}: invalid requiredFacts`);
  if (!observation(facts)) fail(`${id}: invalid PostObservation`);
  for (const key of row.requiredFacts) {
    if (!selectorKnown(key, names)) fail(`${id}: unknown selector ${key}`);
    if (!holds(key, facts)) fail(`${id}: required ${key} is missing or invalid${detailOf(key, facts)}`);
  }
}
