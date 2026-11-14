// Browser-free validation of the global-illumination scene inventories (S40, module gi-page).
// Form follows examples/virtual-geometry/lib/inventory.js; the stages and selectors are the GI ones.
const stages = ['volume', 'lighting', 'virtual'];
const selectors = ['snapshot:settings', 'snapshot:quality', 'snapshot:stats', 'snapshot:instances', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready'];
const namePattern = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
const filePattern = /^[a-z0-9]+(?:-[a-z0-9]+)+\.png$/;
const text = value => typeof value === 'string' && value.trim().length > 0;
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const count = value => Number.isSafeInteger(value) && value >= 0;
const positive = value => Number.isSafeInteger(value) && value > 0;
const bool = value => typeof value === 'boolean';
const string = value => typeof value === 'string';
const vec = value => Array.isArray(value) && value.length === 3 && value.every(finite);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const unit = v => { const n = Math.hypot(...v); return n > 1e-12 ? v.map(x => x/n) : null; };
const fields = (o, schema) => record(o) && Object.entries(schema).every(([key, test]) => test(o[key]));
const nullable = test => value => value === null || test(value);
const enumeration = values => value => values.includes(value);
const list = test => value => Array.isArray(value) && value.every(test);
const nonEmpty = test => value => Array.isArray(value) && value.length > 0 && value.every(test);
function fail(message) { throw new Error(message); }
function registry(names, context) {
  if (!Array.isArray(names)) fail(`${context}: factNames must be an array`);
  const set = new Set();
  for (const name of names) {
    if (typeof name !== 'string' || !namePattern.test(name) || set.has(name)) fail(`${context}: invalid or duplicate fact name ${name}`);
    set.add(name);
  }
  return set;
}
function selector(value, names, context) {
  if (selectors.includes(value)) return;
  if (typeof value === 'string' && value.startsWith('assertion:') && names.has(value.slice(10))) return;
  fail(`${context}: unknown selector ${String(value)}`);
}
function selection(value, context) {
  if (!record(value) || !record(value.camera) || !Array.isArray(value.subjects) || !value.subjects.length) fail(`${context}: camera and subjects required`);
  const c = value.camera;
  const aspect = c.aspect === undefined ? 16/9 : c.aspect;
  if (!vec(c.eye) || !vec(c.target) || !finite(c.fovYDeg) || c.fovYDeg <= 0 || c.fovYDeg >= 180 || !finite(c.near) || !finite(c.far) || c.near <= 0 || c.far <= c.near || !finite(aspect) || aspect <= 0) fail(`${context}: invalid camera`);
  const forward = unit(sub(c.target, c.eye));
  const right = forward && unit(cross(forward, [0, 1, 0]));
  if (!right) fail(`${context}: degenerate camera`);
  const up = cross(right, forward), vertical = Math.tan(c.fovYDeg * Math.PI / 360), horizontal = vertical * aspect;
  const seen = new Set();
  for (const subject of value.subjects) {
    if (!record(subject) || !text(subject.name) || seen.has(subject.name) || !vec(subject.min) || !vec(subject.max) || subject.min.some((v, i) => v > subject.max[i])) fail(`${context}: invalid subject`);
    seen.add(subject.name);
    const corners = [];
    for (let mask = 0; mask < 8; mask++) {
      const p = sub(subject.min.map((v, i) => mask & (1 << i) ? subject.max[i] : v), c.eye);
      const x = dot(p, right), y = dot(p, up), z = dot(p, forward);
      corners.push([z-c.near, c.far-z, z*horizontal+x, z*horizontal-x, z*vertical+y, z*vertical-y]);
    }
    // A box is out only when all eight corners lie beyond the same frustum plane, so a floor larger than the frame stays in view.
    if (corners[0].some((_, plane) => corners.every(corner => corner[plane] < 0))) fail(`${context}: subject ${subject.name} outside camera frustum`);
  }
}
const features = value => record(value) && Object.values(value).every(v => bool(v) || string(v) || finite(v));
const quality = value => fields(value, { resolution: finite, levels: finite, probeRays: finite });
const statistics = value => fields(value, { active: bool, allocatedBytes: finite });
const instanceGi = value => fields(value, { requested: nullable(bool), voxelized: bool, driver: nullable(enumeration(['forward', 'virtual'])), code: nullable(string), message: nullable(string) });
const instanceFact = value => fields(value, { name: text, instance: count, globalIllumination: nullable(instanceGi) });
const assertion = value => fields(value, { name: v => string(v) && namePattern.test(v), passed: bool, detail: string });
const edit = value => fields(value, { operation: string, accepted: bool, error: nullable(e => fields(e, { code: string, message: string })) });
const pass = value => fields(value, { name: text, role: enumeration(['pass', 'contributor']), enabled: bool, order: finite, required: bool });
const view = value => fields(value, { view: count, width: positive, height: positive });
const readiness = value => fields(value, { status: enumeration(['pending', 'ready', 'failed']), stage: string, error: nullable(string) });

/**
 * `validateInventory` validates the S40 inventories of the scenes up to the stage (cumulative: later stages are
 * skipped) and returns their rows. Today it throws `invalid stage or scenes: <stage>` for an unknown stage or a
 * non-array, and `stage <stage> is unavailable` when no scene has `stage` as its milestone.
 * Handoff: gi-page: contract.test.mjs takes `validateInventory(scenes: object[], stage: string) -> object[]`
 * exported by examples/global-illumination/lib/inventory.js:87 validateInventory; the card calls it on the scenes/volume.js
 * namespace and on fixtures, and calls validateRequiredFacts on facts fixtures; state/keys: the scene namespaces
 * (default, inventory, factNames, observe) and the stage name.
 * Handoff: gi-page: volume.test.mjs takes `validateInventory(scenes: object[], stage: string) -> object[]`
 * exported by examples/global-illumination/lib/inventory.js:87 validateInventory; attached by import; state/keys: as above.
 * Handoff: gi-page: page entry, build script and p4-gi commands takes `validateInventory(scenes: object[], stage: string) -> object[]`
 * exported by examples/global-illumination/lib/inventory.js:87 validateInventory; attached by import in main.js to check
 * scenes.json; state/keys: as above.
 */
export function validateInventory(scenes, stage) {
  const limit = stages.indexOf(stage);
  if (limit < 0 || !Array.isArray(scenes)) fail(`invalid stage or scenes: ${stage}`);
  const ids = new Set(), images = new Set(), filenames = new Set(), rows = [], pairs = new Map();
  let available = false;
  for (const ns of scenes) {
    if (!record(ns) || !record(ns.default)) fail('invalid scene namespace');
    const scene = ns.default, rank = stages.indexOf(scene.milestone);
    if (rank < 0 || !text(scene.id) || !namePattern.test(scene.id) || ids.has(scene.id)) fail(`invalid or duplicate scene ${scene.id}`);
    ids.add(scene.id);
    if (rank > limit) continue;
    if (rank === limit) available = true;
    const names = registry(ns.factNames, scene.id);
    if (!Array.isArray(ns.inventory) || !Array.isArray(scene.variants) || typeof scene.build !== 'function' || typeof ns.observe !== 'function') fail(`${scene.id}: incomplete scene exports`);
    const variants = new Map();
    for (const variant of scene.variants) {
      const context = `${scene.id}/${variant?.id}`;
      if (!record(variant) || !namePattern.test(variant.id ?? '') || variants.has(variant.id)) fail(`${scene.id}: invalid or duplicate variant`);
      selection(variant.selection, context);
      if (variant.giExpected !== undefined && !['active', 'disabled'].includes(variant.giExpected)) fail(`${context}: giExpected must be active or disabled`);
      variants.set(variant.id, variant);
      const m = variant.measurement;
      if (m === undefined) continue;
      // A measurement state is named by the variant id suffix (-on / -off).
      if (!fields(m, { pair: text, state: enumeration(['on', 'off']), preset: text, features }) || !variant.id.endsWith(`-${m.state}`)) fail(`${context}: invalid measurement or variant name`);
      pairs.set(m.pair, [...(pairs.get(m.pair) ?? []), m]);
    }
    for (const row of ns.inventory) {
      if (!fields(row, { id: v => text(v) && namePattern.test(v), criterionName: text, scenario: v => v === scene.id, mode: v => variants.has(v), filename: v => string(v) && filePattern.test(v), expectedVisibleBehavior: text, requiredFacts: list(string), pair: nullable(text), heldConstant: list(text), producer: text }) || images.has(row.id) || filenames.has(row.filename) || (variants.get(row.mode).measurement?.pair ?? null) !== row.pair) fail(`${scene.id}: invalid inventory row ${row?.id}`);
      row.requiredFacts.forEach(fact => selector(fact, names, row.id));
      if (new Set(row.requiredFacts).size !== row.requiredFacts.length) fail(`${row.id}: duplicate selector`);
      images.add(row.id); filenames.add(row.filename); rows.push(row);
    }
  }
  if (!available) fail(`stage ${stage} is unavailable`);
  for (const [pair, entries] of pairs) {
    if (entries.length !== 2 || new Set(entries.map(e => e.state)).size !== 2 || entries[0].preset !== entries[1].preset) fail(`measurement pair ${pair} requires on and off states with the same preset`);
  }
  return rows;
}

/**
 * `validateRequiredFacts` checks each requiredFacts selector of one inventory row against the facts of its scene (contract gi-facts).
 * Today it returns nothing and throws `<id>: required <selector> is missing or invalid[: detail]` on missing, null,
 * duplicate or false evidence, and `<id>: invalid SceneObservation` when the whole facts shape is wrong.
 * Handoff: gi-page: lib/runtime.js and lib/facts.js takes `validateRequiredFacts(row: object, facts: object, factNames: string[]) -> void`
 * exported by examples/global-illumination/lib/inventory.js:137 validateRequiredFacts; attached by import in lib/runtime.js,
 * which passes it to createPageRunner as the validateRequiredFacts option; state/keys: the facts of contract gi-facts and
 * the scene's factNames registry.
 */
export function validateRequiredFacts(row, facts, factNames) {
  const id = row?.id ?? '<unknown image>';
  const names = registry(factNames, id);
  if (!record(row) || !Array.isArray(row.requiredFacts)) fail(`${id}: invalid requiredFacts`);
  const seen = new Set();
  for (const key of row.requiredFacts) {
    selector(key, names, id);
    if (seen.has(key)) fail(`${id}: duplicate selector ${key}`);
    seen.add(key);
    let valid = record(facts), detail = '';
    if (valid && key === 'snapshot:settings') valid = record(facts.settings);
    else if (valid && key === 'snapshot:quality') valid = quality(facts.quality);
    else if (valid && key === 'snapshot:stats') valid = statistics(facts.stats);
    else if (valid && key === 'snapshot:instances') valid = nonEmpty(instanceFact)(facts.instances);
    else if (valid && key === 'runtime:passes') valid = nonEmpty(pass)(facts.passes);
    else if (valid && key === 'runtime:viewSizes') valid = nonEmpty(view)(facts.viewSizes);
    else if (valid && key === 'runtime:ready') {
      valid = fields(facts.readiness, { status: v => v === 'ready', stage: text, error: v => v === null });
      if (!valid && record(facts.readiness)) detail = `: readiness ${String(facts.readiness.status)}${facts.readiness.error ? ` (${facts.readiness.error})` : ''}`;
    } else if (valid && key.startsWith('assertion:')) {
      const matches = Array.isArray(facts.assertions) ? facts.assertions.filter(a => a?.name === key.slice(10)) : [];
      valid = matches.length === 1 && assertion(matches[0]) && matches[0].passed === true;
      // A false assertion reports its own detail; a repeated report is refused.
      if (!valid) detail = matches.length === 1 ? `: ${String(matches[0]?.detail)}` : ` (${matches.length} reported)`;
    }
    if (!valid) fail(`${id}: required ${key} is missing or invalid${detail}`);
  }
  if (!record(facts) || !nullable(record)(facts.settings) || !nullable(quality)(facts.quality) || !nullable(record)(facts.stats) || !list(instanceFact)(facts.instances) || !list(assertion)(facts.assertions) || !list(edit)(facts.edits) || !list(pass)(facts.passes) || !list(view)(facts.viewSizes) || !readiness(facts.readiness)) fail(`${id}: invalid SceneObservation`);
}
