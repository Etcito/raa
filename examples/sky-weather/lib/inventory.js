// Browser-free validation shared by selection, runtime and freshness.
const stages = ['foundation', 'atmosphere', 'weather'];
const namePattern = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/;
const text = value => typeof value === 'string' && value.trim().length > 0;
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const finite = value => typeof value === 'number' && Number.isFinite(value);
const vec = value => Array.isArray(value) && value.length === 3 && value.every(finite);
const sub = (a, b) => a.map ( (v, i) => v - b[i]);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const unit = v => { const n = Math.hypot(...v); return n > 1e-12 ? v.map(x => x/n) : null; };
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
  if (['snapshot:sky', 'snapshot:time', 'snapshot:weather', 'snapshot:diagnostics', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready'].includes(value)) return;
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
    if (corners[0].some((_, plane) => corners.every(corner => corner[plane] < 0))) fail(`${context}: subject ${subject.name} outside camera frustum`);
  }
}

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
      if (!record(variant) || !namePattern.test(variant.id ?? '') || variants.has(variant.id)) fail(`${scene.id}: invalid or duplicate variant`);
      selection(variant.selection, `${scene.id}/${variant.id}`);
      variants.set(variant.id, variant);
      if (variant.measurement !== undefined) {
        const m = variant.measurement;
        if (!record(m) || !text(m.pair) || !['on', 'off'].includes(m.state) || !text(m.preset) || !record(m.features) || Object.values(m.features).some(v => !(['boolean', 'string'].includes(typeof v) || finite(v)))) fail(`${scene.id}/${variant.id}: invalid measurement`);
        const pair = pairs.get(m.pair) ?? [];
        pair.push(m); pairs.set(m.pair, pair);
      }
    }
    for (const row of ns.inventory) {
      if (!record(row) || !text(row.id) || !namePattern.test(row.id) || images.has(row.id) || !text(row.criterionName) || !text(row.expectedVisibleBehavior) || row.scenario !== scene.id || !variants.has(row.mode) || typeof row.filename !== 'string' || !/^[a-z0-9]+(?:-[a-z0-9]+)+\.png$/.test(row.filename) || filenames.has(row.filename) || !Array.isArray(row.requiredFacts) || !(row.pair === null || text(row.pair))) fail(`${scene.id}: invalid inventory row ${row?.id}`);
      const required = new Set();
      for (const fact of row.requiredFacts) {
        selector(fact, names, row.id);
        if (required.has(fact)) fail(`${row.id}: duplicate selector ${fact}`);
        required.add(fact);
      }
      const measurement = variants.get(row.mode).measurement;
      if ((measurement?.pair ?? null) !== row.pair) fail(`${row.id}: measurement pair mismatch`);
      images.add(row.id); filenames.add(row.filename); rows.push(row);
    }
  }
  if (!available) fail(`stage ${stage} is unavailable`);
  for (const [pair, entries] of pairs) {
    if (entries.length !== 2 || new Set(entries.map(e => e.state)).size !== 2 || entries[0].preset !== entries[1].preset) fail(`measurement pair ${pair} requires on and off states with the same preset`);
  }
  return rows;
}

const fields = (o, schema) => record(o) && Object.entries(schema).every(([key, test]) => test(o[key]));
const bool = v => typeof v === 'boolean';
const string = v => typeof v === 'string';
const optional = test => v => v === undefined || test(v);
const enumeration = values => v => values.includes(v);
const vector2 = v => Array.isArray(v) && v.length === 2 && v.every(finite);
const sky = v => fields(v, {
  preset: optional(enumeration(['day', 'snow', 'jungle', 'night'])), profile: optional(enumeration(['day', 'snow', 'jungle', 'night'])),
  model: optional(enumeration(['physical', 'legacy'])), zenith: optional(vec), horizon: optional(vec), ground: optional(vec),
  intensity: optional(finite), horizonSoftness: optional(finite), sunDiscSize: optional(finite), sunDiscIntensity: optional(finite),
  envUpdate: optional(enumeration(['onChange', 'everyFrame'])),
  clouds: optional(c => fields(c, { coverage: optional(finite), density: optional(finite), altitude: optional(finite), scale: optional(finite), wind: optional(vector2), color: optional(vec), silverLining: optional(finite) })),
  atmosphere: optional(a => fields(a, { groundAlbedo: optional(vec), mieMultiplier: optional(finite) })),
});
const time = v => fields(v, { day: finite, hours: finite, latitude: finite, north: finite, enabled: bool, playing: bool, speed: finite, currentDay: finite, currentHours: finite, sunDirection: vec, moonDirection: vec, moonLit: finite, starPole: vec, starEquinox: vec, dominant: enumeration(['manual', 'sun', 'moon']) });
const weatherState = v => fields(v, { coverage: finite, type: finite, base: finite, thickness: finite, density: finite, cirrus: finite, wind: w => fields(w, { speed: finite, from: finite }), evolution: finite, rain: finite });
const weather = v => fields(v, { enabled: bool, shadows: bool, changing: bool, seed: Number.isSafeInteger, transitionSeconds: finite, target: weatherState, current: weatherState, cyclePreset: v => v === null || enumeration(['clear', 'fair', 'broken', 'overcast', 'storm', 'cirrus'])(v), text: string });
// Diagnostics evolve with the renderer, but every reported counter must be a safe integer.
function diagnosticValue(v) {
  if (typeof v === 'number') return Number.isSafeInteger(v);
  if (v === null || typeof v === 'string' || typeof v === 'boolean') return true;
  if (Array.isArray(v)) return v.every(diagnosticValue);
  return record(v) && Object.values(v).every(diagnosticValue);
}
const diagnostics = v => record(v) && Object.keys(v).length > 0 && diagnosticValue(v);
const nullable = test => v => v === null || test(v);
const state = v => fields(v, { sky: nullable(sky), time: nullable(time), weather: nullable(weather) });
const assertion = v => fields(v, { name: v => string(v) && namePattern.test(v), passed: bool, detail: string });
const edit = v => fields(v, { operation: string, accepted: bool, error: nullable(e => fields(e, { code: string, message: string })), before: state, after: state });
const pass = v => fields(v, { name: text, role: enumeration(['pass', 'contributor']), enabled: bool, order: finite, required: bool });
const view = v => fields(v, { view: v => Number.isSafeInteger(v) && v >= 0, width: v => Number.isSafeInteger(v) && v > 0, height: v => Number.isSafeInteger(v) && v > 0 });

export function validateRequiredFacts(row, facts, factNames) {
  const id = row?.id ?? '<unknown image>';
  const names = registry(factNames, id);
  if (!record(row) || !Array.isArray(row.requiredFacts)) fail(`${id}: invalid requiredFacts`);
  const seen = new Set();
  for (const key of row.requiredFacts) {
    selector(key, names, id);
    if (seen.has(key)) fail(`${id}: duplicate selector ${key}`);
    seen.add(key);
    let valid = record(facts);
    if (valid && key.startsWith('snapshot:')) {
      const field = key.slice(9);
      valid = ({ sky, time, weather, diagnostics })[field](facts[field]);
    } else if (valid && key === 'runtime:passes') valid = Array.isArray(facts.passes) && facts.passes.length > 0 && facts.passes.every(pass);
    else if (valid && key === 'runtime:viewSizes') valid = Array.isArray(facts.viewSizes) && facts.viewSizes.length > 0 && facts.viewSizes.every(view);
    else if (valid && key === 'runtime:ready') valid = fields(facts.readiness, { status: v => v === 'ready', stage: text, error: v => v === null });
    else if (valid && key.startsWith('assertion:')) {
      const matches = Array.isArray(facts.assertions) ? facts.assertions.filter(a => a?.name === key.slice(10)) : [];
      valid = matches.length === 1 && assertion(matches[0]) && matches[0].passed === true;
    }
    if (!valid) fail(`${id}: required ${key} is missing or invalid`);
  }
  // Validate the observation envelope even when a particular nullable snapshot is not required.
  if (!record(facts) || !state(facts) || !nullable(diagnostics)(facts.diagnostics) || !Array.isArray(facts.assertions) || !facts.assertions.every(assertion) || !Array.isArray(facts.edits) || !facts.edits.every(edit)) fail(`${id}: invalid SceneObservation`);
}
