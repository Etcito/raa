// Browser-free validation of the render-graph scene inventories (contract rg-page-facts, module
// render-graph-page). Form and messages follow examples/global-illumination/lib/inventory.js:
// the S26 rules plus its frustum overlap check (a subject box is out only when all eight corners
// lie beyond the same frustum plane). The stages are the nine S61 stages in rank order.
const stages = ['interpreter', 'views', 'parameters', 'wiring', 'slots', 'gi-stages', 'post-stages', 'compute', 'frozen'];
const selectors = ['snapshot:renderGraphActive', 'snapshot:issues', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready'];
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
// Contract rg-page-runtime: a variant's pair labels the two halves and names their order.
const halfOrder = value => Array.isArray(value) && value.length === 2
  && ((value[0] === 'left' && value[1] === 'right') || (value[0] === 'right' && value[1] === 'left'));
const variantPair = value => value === null || (record(value) && text(value.left) && text(value.right) && halfOrder(value.order));
const optionalText = value => value === undefined || text(value);
const optionalLink = value => value === undefined || count(value);
const issue = value => fields(value, {
  code: string, severity: enumeration(['error', 'warning']), nodeId: optionalText, port: optionalText,
  param: optionalText, link: optionalLink, message: string, fix: string,
});
const assertion = value => fields(value, { name: v => string(v) && namePattern.test(v), passed: bool, detail: string });
const edit = value => fields(value, { operation: string, accepted: bool, error: nullable(e => fields(e, { code: string, message: string })) });
const pass = value => fields(value, { name: text, role: enumeration(['pass', 'contributor']), enabled: bool, order: finite, required: bool });
const view = value => fields(value, { view: count, width: positive, height: positive });
const readiness = value => fields(value, { status: enumeration(['pending', 'ready', 'failed']), stage: string, error: nullable(string) });

/**
 * `validateInventory` validates the render-graph inventories of the scenes up to `stage`
 * (cumulative: later stages are skipped) and returns their rows. It follows the S26 rules:
 * scene ids, variants and row ids are unique and descriptive, every row names an existing
 * variant of its own scene, filenames match `^[a-z0-9]+(?:-[a-z0-9]+)+\.png$` and are unique,
 * every requiredFacts selector is known, and each variant `selection` must have its subject
 * boxes overlap the camera view volume. A variant with a `measurement` is refused (the page
 * takes none); a variant `pair` is null or `{ left, right, order }`, and every row `pair` is
 * null. Today it throws `invalid stage or scenes: <stage>` for an unknown stage or a non-array,
 * `stage <stage> is unavailable` when no scene is at `stage`, and
 * `<scene>: invalid inventory row <id>` for a malformed row.
 */
export function validateInventory(scenes, stage) {
  const limit = stages.indexOf(stage);
  if (limit < 0 || !Array.isArray(scenes)) fail(`invalid stage or scenes: ${stage}`);
  const ids = new Set(), images = new Set(), filenames = new Set(), rows = [];
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
      if (variant.measurement !== undefined) fail(`${context}: measurement is not allowed`);
      if (variant.pair !== undefined && !variantPair(variant.pair)) fail(`${context}: invalid pair`);
      variants.set(variant.id, variant);
    }
    for (const row of ns.inventory) {
      if (!fields(row, { id: v => text(v) && namePattern.test(v), criterionName: text, scenario: v => v === scene.id, mode: v => variants.has(v), filename: v => string(v) && filePattern.test(v), expectedVisibleBehavior: text, requiredFacts: list(string), pair: v => v === null, heldConstant: list(text), producer: text }) || images.has(row.id) || filenames.has(row.filename)) fail(`${scene.id}: invalid inventory row ${row?.id}`);
      row.requiredFacts.forEach(fact => selector(fact, names, row.id));
      if (new Set(row.requiredFacts).size !== row.requiredFacts.length) fail(`${row.id}: duplicate selector`);
      images.add(row.id); filenames.add(row.filename); rows.push(row);
    }
  }
  if (!available) fail(`stage ${stage} is unavailable`);
  return rows;
}

/**
 * `validateRequiredFacts` checks each requiredFacts selector of one inventory row against the
 * facts of contract rg-page-facts. Today it returns nothing and throws
 * `<id>: required <selector> is missing or invalid[: detail]` on missing, null, duplicate or
 * false evidence, and `<id>: invalid SceneObservation` when the whole facts shape is wrong.
 * `snapshot:renderGraphActive` requires a boolean, `snapshot:issues` an array of valid issues,
 * `runtime:passes` a non-empty array of valid PassInfo, `runtime:viewSizes` a non-empty array,
 * `runtime:ready` a ready readiness with a null error, and `assertion:<name>` exactly one
 * declared assertion with `passed === true`.
 */
// Handoff: examples/render-graph/lib/scene.js:56 cameraAt covers this validator (rg-page-facts).
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
    if (valid && key === 'snapshot:renderGraphActive') valid = bool(facts.renderGraphActive);
    else if (valid && key === 'snapshot:issues') valid = list(issue)(facts.issues);
    else if (valid && key === 'runtime:passes') valid = nonEmpty(pass)(facts.passes);
    else if (valid && key === 'runtime:viewSizes') valid = nonEmpty(view)(facts.viewSizes);
    else if (valid && key === 'runtime:ready') {
      valid = fields(facts.readiness, { status: v => v === 'ready', stage: text, error: v => v === null });
      if (!valid && record(facts.readiness)) detail = `: readiness ${String(facts.readiness.status)}${facts.readiness.error ? ` (${facts.readiness.error})` : ''}`;
    } else if (valid && key.startsWith('assertion:')) {
      const matches = Array.isArray(facts.assertions) ? facts.assertions.filter(a => a?.name === key.slice(10)) : [];
      valid = matches.length === 1 && assertion(matches[0]) && matches[0].passed === true;
      if (!valid) detail = matches.length === 1 ? `: ${String(matches[0]?.detail)}` : ` (${matches.length} reported)`;
    }
    if (!valid) fail(`${id}: required ${key} is missing or invalid${detail}`);
  }
  if (!record(facts) || !nullable(bool)(facts.renderGraphActive) || !nullable(string)(facts.graph) || !nullable(list(issue))(facts.issues) || !list(assertion)(facts.assertions) || !list(edit)(facts.edits) || !list(pass)(facts.passes) || !list(view)(facts.viewSizes) || !readiness(facts.readiness)) fail(`${id}: invalid SceneObservation`);
}
