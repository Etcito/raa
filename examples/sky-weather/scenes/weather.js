import { buildSkySample, sampleSelection } from '../lib/scene.js';
import { weatherTargets, NEAR_TOLERANCE } from './atmosphere.js';

// Volumetric weather presets, weather on/off and ground cloud shadows through the delivered
// public API only. Every assertion is computed by observe from Engine getters, diagnostics and
// the pass lists the build read around its weather edit; the only authored values compared are
// the static preset targets (crates/raa-sky/weather/*.weather) the assertions name.

// The two built-ins missing from atmosphere.js, from storm.weather and cirrus.weather (omitted keys keep the fair defaults).
const fair = weatherTargets.fair;
export const targets = {
  ...weatherTargets,
  storm: { coverage: .85, type: 1, base: 800, thickness: 7000, density: 2, cirrus: .6, wind: { speed: 16, from: 230 }, evolution: 2, rain: 1 },
  cirrus: { ...fair, coverage: .04, type: .9, cirrus: .75, wind: { speed: 20, from: 260 } },
};

// A sky-facing camera still overlaps the ground, spheres and distant forms; the shadow camera looks down across the ground.
const skyCamera = { eye: [0, 2.6, 9], target: [0, 6, -8], fovYDeg: 60, near: .1, far: 300 };
const groundCamera = { eye: [0, 14, 16], target: [0, 0, -10], fovYDeg: 60, near: .1, far: 300 };
const moment = { day: 172, hours: 17.5, latitude: 38, north: 0, enabled: true, playing: false, speed: 1 };
const PRESETS = ['fair', 'broken', 'overcast', 'storm', 'cirrus', 'clear'];
export const MODES = [...PRESETS, 'weather-off', 'shadows-on', 'shadows-off'];
const modes = {
  ...Object.fromEntries(PRESETS.map(preset => [preset, { request: { preset, transitionSeconds: 0, changing: false }, camera: skyCamera }])),
  'weather-off': { request: { preset: 'broken', enabled: false }, camera: skyCamera },
  'shadows-on': { request: { preset: 'broken', shadows: true }, camera: groundCamera, shadow: true },
  'shadows-off': { request: { preset: 'broken', shadows: false }, camera: groundCamera, shadow: true },
};

const shared = 'A broad neutral ground with a terracotta matte sphere on the left, a chrome sphere on the right '
  + 'and three distant block forms behind them. The status below the view reads ready, with no displayed failure.';
const behaviors = {
  fair: 'The fair weather preset: separated bright cumulus with clear blue sky between them. ',
  broken: 'The broken weather preset: large cumulus masses covering most of the sky with blue gaps. ',
  overcast: 'The overcast weather preset: a continuous low grey sheet with no blue sky and soft shadowless light. ',
  storm: 'The storm weather preset: dark, tall, heavy cloud with a dim scene. ',
  cirrus: 'The cirrus weather preset: a high thin fibrous streaked veil with blue sky showing through and no low cumulus. ',
  clear: 'The clear weather preset: open blue sky with no cumulus or sheet. ',
  'weather-off': 'The same scene with weather disabled: clear sky, no clouds at all. ',
  'shadows-on': 'Cloud shadows enabled: soft large cloud-shadow patches darken parts of the ground while the spheres still cast their own sharp shadows. ',
  'shadows-off': 'Cloud shadows disabled: the ground is evenly sunlit with no cloud-shadow patches while the spheres still cast their own shadows. ',
};
for (const mode of MODES) behaviors[mode] += shared;

// Measurement pairs: weather on/off and cloud shadows on/off, both on the broken preset.
const measurements = {
  broken: { pair: 'weather', state: 'on', preset: 'broken', features: { weather: true } },
  'weather-off': { pair: 'weather', state: 'off', preset: 'broken', features: { weather: false } },
  'shadows-on': { pair: 'cloud-shadows', state: 'on', preset: 'broken', features: { shadows: true } },
  'shadows-off': { pair: 'cloud-shadows', state: 'off', preset: 'broken', features: { shadows: false } },
};

export const factNames = ['weather-preset-applied', 'clouds-marched', 'clouds-neutral', 'weather-disabled', 'shadows-enabled',
  'shadows-disabled-csm-kept', 'atmosphere-submitted', 'environment-published', 'view-submitted'];
const common = ['snapshot:sky', 'snapshot:time', 'snapshot:weather', 'snapshot:diagnostics', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready',
  'assertion:atmosphere-submitted', 'assertion:environment-published', 'assertion:view-submitted'];
const row = (mode, criterionName, filename, extra) => ({
  id: mode === 'weather-off' ? mode : `weather-${mode}`, criterionName, scenario: 'weather', mode, filename, expectedVisibleBehavior: behaviors[mode],
  requiredFacts: [...common, ...extra.map(name => `assertion:${name}`)], pair: measurements[mode]?.pair ?? null,
});
const preset = ['weather-preset-applied', 'clouds-marched'];
export const inventory = [
  row('fair', 'Fair separated cumulus', 'fair-separated-cumulus.png', preset),
  row('broken', 'Broken cloud with blue gaps', 'broken-cloud-blue-gaps.png', preset),
  row('overcast', 'Overcast low grey sheet', 'overcast-low-grey-sheet.png', preset),
  row('storm', 'Storm dark towering cloud', 'storm-dark-towering-cloud.png', preset),
  row('cirrus', 'Cirrus high fibrous veil', 'cirrus-high-fibrous-veil.png', preset),
  row('clear', 'Clear preset cloudless blue sky', 'clear-preset-cloudless-blue-sky.png', preset),
  row('weather-off', 'Weather disabled clear sky', 'weather-off-clear-sky.png', ['clouds-neutral', 'weather-disabled']),
  row('shadows-on', 'Ground cloud shadows on', 'ground-cloud-shadow-on.png', ['shadows-enabled']),
  row('shadows-off', 'Ground cloud shadows off, CSM retained', 'ground-cloud-shadow-off-csm-retained.png', ['shadows-disabled-csm-kept']),
];

const snapshot = engine => ({ sky: engine.sky(), time: engine.timeOfDay(), weather: engine.weather() });
const errorOf = error => ({ code: typeof error?.code === 'string' ? error.code : 'error', message: String(error?.message ?? error) });
// Tagged public edits performed by build, keyed by the engine they ran on.
const recorded = new WeakMap();
// The pass lists read immediately before and after the shadow modes' weather edit, keyed by engine.
const passLists = new WeakMap();
/** Runs one public edit, recording the request it passed plus the actual before/after getter snapshots. */
function record(engine, tag, operation, request, apply) {
  const before = snapshot(engine);
  let accepted = true, error = null;
  try { apply(); } catch (cause) { accepted = false; error = errorOf(cause); }
  const edit = { operation, accepted, error, before, after: snapshot(engine) };
  if (!recorded.has(engine)) recorded.set(engine, []);
  recorded.get(engine).push({ tag, request: structuredClone(request), edit });
  if (!accepted) throw new Error(`${operation}: ${error.code}: ${error.message}`);
  return edit;
}

const near = (a, b) => typeof a === 'number' && Number.isFinite(a) && Math.abs(a - b) <= NEAR_TOLERANCE * Math.max(1, Math.abs(b));
const targetKeys = ['coverage', 'type', 'base', 'thickness', 'density', 'cirrus', 'evolution', 'rain'];
/** Field-by-field comparison of an observed weather target with a static preset target. */
const targetMatches = (t, want) => t != null && targetKeys.every(k => near(t[k], want[k])) && near(t.wind?.speed, want.wind.speed) && near(t.wind?.from, want.wind.from);
const flagsOn = d => d.views.length > 0 && d.views.every(v => v.cloudPanoramaValid === true && v.cloudScreenValid === true);
const flagsOff = d => d.views.length > 0 && d.views.every(v => v.cloudPanoramaValid === false && v.cloudScreenValid === false);
const flagDetail = d => JSON.stringify(d.views.map(v => [v.view, v.cloudScreenValid, v.cloudPanoramaValid]));

const FRAME_LIMIT = 600;
const yieldTask = () => new Promise(resolve => setTimeout(resolve, 0));
// Accepted-state publication each capture waits for; marching modes also wait for both cloud flags.
const settled = (d, marching) => d.readiness.status === 'ready' && d.readiness.error === null && d.atmosphereGeneration > 0
  && d.environment.publishedGeneration !== null && d.environment.publishedGeneration >= d.environment.requestedGeneration && d.environment.pending === false
  && d.views.length > 0 && d.views.every(v => v.submittedInvocation !== null && (!marching || (v.cloudScreenValid === true && v.cloudPanoramaValid === true)));
async function awaitPublication(engine, variant) {
  const marching = variant.id !== 'weather-off';
  let diagnostics = engine.skyWeatherDiagnostics();
  for (let frame = 0; frame < FRAME_LIMIT && !settled(diagnostics, marching); frame++) {
    if (diagnostics.readiness.status === 'failed') break;
    engine.render(variant.timeMs ?? 1000);
    await yieldTask();
    diagnostics = engine.skyWeatherDiagnostics();
  }
  return diagnostics;
}

/** Shadow elements of the pass lists stored around the weather edit, with their enabled values before and after. */
function shadowPasses(lists) {
  const find = (list, name) => list?.find(p => p.name === name)?.enabled;
  const names = [...new Set([...(lists?.before ?? []), ...(lists?.after ?? [])].map(p => p.name).filter(n => n.startsWith('shadow.')))];
  return names.map(name => ({ name, before: find(lists?.before, name), after: find(lists?.after, name) }));
}

export async function observe(engine, variant) {
  const mode = variant.id, spec = modes[mode];
  if (!spec) throw new Error(`weather has no mode ${mode}`);
  const diagnostics = await awaitPublication(engine, variant);
  const { sky, time, weather } = snapshot(engine);
  const env = diagnostics.environment, records = recorded.get(engine) ?? [];
  const want = targets[spec.request.preset], edit = records.find(r => r.tag === 'weather')?.edit;
  const assertions = [
    { name: 'atmosphere-submitted', passed: diagnostics.readiness.status === 'ready' && diagnostics.atmosphereGeneration > 0,
      detail: `readiness ${JSON.stringify(diagnostics.readiness)}; atmosphereGeneration ${diagnostics.atmosphereGeneration}` },
    { name: 'environment-published', passed: env.publishedGeneration !== null && env.publishedGeneration >= env.requestedGeneration && env.pending === false,
      detail: `environment ${JSON.stringify(env)}` },
    { name: 'view-submitted', passed: diagnostics.views.length > 0 && diagnostics.views.every(v => v.submittedInvocation !== null),
      detail: `submittedInvocation ${diagnostics.submittedInvocation}; views ${JSON.stringify(diagnostics.views.map(v => [v.view, v.submittedInvocation]))}` },
  ];
  if (PRESETS.includes(mode)) {
    assertions.push({ name: 'weather-preset-applied', passed: targetMatches(weather.target, want),
      detail: `${mode}: target ${JSON.stringify(weather.target)}; expected ${JSON.stringify(want)}` });
    assertions.push({ name: 'clouds-marched', passed: flagsOn(diagnostics), detail: `[view, screen, panorama] ${flagDetail(diagnostics)}` });
  } else if (mode === 'weather-off') {
    assertions.push({ name: 'clouds-neutral', passed: flagsOff(diagnostics) && diagnostics.readiness.status === 'ready' && diagnostics.readiness.error === null,
      detail: `[view, screen, panorama] ${flagDetail(diagnostics)}; readiness ${JSON.stringify(diagnostics.readiness)}` });
    assertions.push({ name: 'weather-disabled', passed: edit?.accepted === true && weather.enabled === false && targetMatches(weather.target, want),
      detail: `edit accepted ${edit?.accepted}; enabled ${weather.enabled}; target ${JSON.stringify(weather.target)}; expected ${JSON.stringify(want)}` });
  } else if (mode === 'shadows-on') {
    assertions.push({ name: 'shadows-enabled', passed: weather.shadows === true && flagsOn(diagnostics),
      detail: `shadows ${weather.shadows}; [view, screen, panorama] ${flagDetail(diagnostics)}` });
  } else {
    const lists = passLists.get(engine), elements = shadowPasses(lists);
    const csm = lists?.after?.find(p => p.name === 'shadow.csm');
    assertions.push({ name: 'shadows-disabled-csm-kept',
      passed: weather.shadows === false && flagsOn(diagnostics) && csm?.enabled === true && csm.order === 20 && elements.every(e => e.before === e.after),
      detail: `shadows ${weather.shadows}; [view, screen, panorama] ${flagDetail(diagnostics)}; shadow.csm after the edit ${JSON.stringify(csm ?? null)}; `
        + `shadow passes before/after the weather edit ${elements.map(e => `${e.name}: ${e.before}->${e.after}`).join(', ') || 'none'}` });
  }
  return { sky, time, weather, diagnostics, assertions, edits: records.map(r => r.edit) };
}

const flag = (id, label, key) => ({ id, label, kind: 'checkbox', unit: '', choices: [],
  read: engine => engine.weather()[key] === true, apply: (engine, value) => engine.setWeather({ [key]: value }) });
const variant = id => ({ id, description: behaviors[id], references: [], judge: true, timeMs: 1000, engine: { exposureEv: 0 },
  selection: sampleSelection(modes[id].camera), ...(measurements[id] ? { measurement: measurements[id] } : {}) });

export default {
  id: 'weather', milestone: 'weather', title: 'Volumetric weather presets and ground cloud shadows',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0 },
  variants: MODES.map(variant),
  controls: [flag('weather-enabled', 'Volumetric weather', 'enabled'), flag('cloud-shadows', 'Cloud shadows on the ground', 'shadows')],
  async build(engine, lib, selected) {
    const mode = selected.id, spec = modes[mode];
    if (!spec) throw new Error(`weather has no mode ${mode}`);
    engine.setBackground([.05, .06, .08]);
    engine.setLight({ direction: [-.35, -.6, -.55], color: [1, .95, .88], intensity: 3 });
    engine.setAmbient({ color: [.55, .62, .75], intensity: .15 });
    const sky = { preset: 'day', model: 'physical' };
    record(engine, 'sky', "setSky({preset:'day',model:'physical'})", sky, () => engine.setSky(sky));
    // The shadow modes store the pass lists around their weather edit; no shadow call is made.
    const before = spec.shadow ? structuredClone(engine.passes()) : null;
    record(engine, 'weather', `setWeather(${JSON.stringify(spec.request)})`, spec.request, () => engine.setWeather(spec.request));
    if (spec.shadow) passLists.set(engine, { before, after: structuredClone(engine.passes()) });
    engine.setTimeOfDay(moment);
    await buildSkySample(engine, lib, selected.selection.camera);
  },
};
