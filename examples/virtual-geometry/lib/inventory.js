// Browser-free validation of the virtual-geometry scene inventories (contract vg-scene-module).
// Form follows examples/sky-weather/lib/inventory.js.
const stages = ['bake', 'passes'];
const selectors = ['snapshot:settings', 'snapshot:stats', 'snapshot:meshes', 'snapshot:instances', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready'];
const namePattern = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
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
function selector(value, row, names, context) {
  if (selectors.includes(value)) return;
  if (typeof value === 'string' && value.startsWith('assertion:') && names.has(value.slice(10))) return;
  if (typeof value === 'string' && value.startsWith('measurement:') && value.slice(12) === row.pair) return;
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
const meshFact = v => fields(v, { name: text, mesh: nullable(count), clusters: nullable(record), cuts: list(c => fields(c, { errorMetres: finite, clusters: count, triangles: count })), states: list(s => fields(s, { state: enumeration(['none', 'baking', 'ready', 'refused']), progress: finite })) });
const instanceFact = v => fields(v, { name: text, instance: count, virtualGeometry: nullable(g => fields(g, { requested: nullable(bool), path: enumeration(['virtual', 'forward']), pending: bool, code: nullable(string), message: nullable(string) })) });
const assertion = v => fields(v, { name: v => string(v) && namePattern.test(v), passed: bool, detail: string });
const edit = v => fields(v, { operation: string, accepted: bool, error: nullable(e => fields(e, { code: string, message: string })) });
const pass = v => fields(v, { name: text, role: enumeration(['pass', 'contributor']), enabled: bool, order: finite, required: bool });
const view = v => fields(v, { view: count, width: positive, height: positive });
const readiness = v => fields(v, { status: enumeration(['pending', 'ready', 'failed']), stage: string, error: nullable(string) });

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
      variants.set(variant.id, variant);
      const m = variant.measurement;
      if (m === undefined) continue;
      // Q15: a measurement state is named by the variant id suffix (-on / -off).
      if (!fields(m, { pair: text, state: enumeration(['on', 'off']), preset: text, features }) || !variant.id.endsWith(`-${m.state}`)) fail(`${context}: invalid measurement or variant name`);
      pairs.set(m.pair, [...(pairs.get(m.pair) ?? []), m]);
    }
    for (const row of ns.inventory) {
      if (!record(row) || !text(row.id) || !namePattern.test(row.id) || images.has(row.id) || !text(row.criterionName) || !text(row.expectedVisibleBehavior) || !Array.isArray(row.heldConstant) || !row.heldConstant.every(text) || row.scenario !== scene.id || !variants.has(row.mode) || typeof row.filename !== 'string' || !/^[a-z0-9]+(?:-[a-z0-9]+)+\.png$/.test(row.filename) || filenames.has(row.filename) || !Array.isArray(row.requiredFacts) || !(row.pair === null || text(row.pair))) fail(`${scene.id}: invalid inventory row ${row?.id}`);
      const required = new Set();
      for (const fact of row.requiredFacts) {
        selector(fact, row, names, row.id);
        if (required.has(fact)) fail(`${row.id}: duplicate selector ${fact}`);
        required.add(fact);
      }
      if ((variants.get(row.mode).measurement?.pair ?? null) !== row.pair) fail(`${row.id}: measurement pair mismatch`);
      images.add(row.id); filenames.add(row.filename); rows.push(row);
    }
  }
  if (!available) fail(`stage ${stage} is unavailable`);
  for (const [pair, entries] of pairs) {
    if (entries.length !== 2 || new Set(entries.map(e => e.state)).size !== 2 || entries[0].preset !== entries[1].preset) fail(`measurement pair ${pair} requires on and off states with the same preset`);
  }
  return rows;
}

export function validateRequiredFacts(row, facts, factNames) {
  const id = row?.id ?? '<unknown image>';
  const names = registry(factNames, id);
  if (!record(row) || !Array.isArray(row.requiredFacts)) fail(`${id}: invalid requiredFacts`);
  const seen = new Set();
  for (const key of row.requiredFacts) {
    selector(key, row, names, id);
    if (seen.has(key)) fail(`${id}: duplicate selector ${key}`);
    seen.add(key);
    // measurement:<pair> needs no fact here: its record is in the job result's measurements, checked by the fresh step.
    let valid = record(facts), detail = '';
    if (valid && key === 'snapshot:settings') valid = record(facts.settings);
    else if (valid && key === 'snapshot:stats') valid = record(facts.stats);
    else if (valid && key === 'snapshot:meshes') valid = nonEmpty(meshFact)(facts.meshes);
    else if (valid && key === 'snapshot:instances') valid = nonEmpty(instanceFact)(facts.instances);
    else if (valid && key === 'runtime:passes') valid = nonEmpty(pass)(facts.passes);
    else if (valid && key === 'runtime:viewSizes') valid = nonEmpty(view)(facts.viewSizes);
    else if (valid && key === 'runtime:ready') {
      valid = fields(facts.readiness, { status: v => v === 'ready', stage: text, error: v => v === null });
      if (!valid && record(facts.readiness)) detail = `: readiness ${String(facts.readiness.status)}${facts.readiness.error ? ` (${facts.readiness.error})` : ''}`;
    } else if (valid && key.startsWith('assertion:')) {
      const matches = Array.isArray(facts.assertions) ? facts.assertions.filter(a => a?.name === key.slice(10)) : [];
      valid = matches.length === 1 && assertion(matches[0]) && matches[0].passed === true;
      // The failed assertion's own detail names what was not drawn; a repeated report is refused.
      if (!valid) detail = matches.length === 1 ? `: ${String(matches[0]?.detail)}` : ` (${matches.length} reported)`;
    }
    if (!valid) fail(`${id}: required ${key} is missing or invalid${detail}`);
  }
  if (!record(facts) || !nullable(record)(facts.settings) || !nullable(record)(facts.stats) || !list(meshFact)(facts.meshes) || !list(instanceFact)(facts.instances) || !list(assertion)(facts.assertions) || !list(edit)(facts.edits) || !list(pass)(facts.passes) || !list(view)(facts.viewSizes) || !readiness(facts.readiness)) fail(`${id}: invalid SceneObservation`);
}
