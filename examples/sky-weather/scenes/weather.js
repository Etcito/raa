import { buildSkySample, sampleSelection } from '../lib/scene.js';
import { weatherTargets, NEAR_TOLERANCE, preCutCamera, RESIZE_TO } from './atmosphere.js';

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
// Looks at the chrome sphere from close range with the sky above the horizon in frame.
const reflectionCamera = { eye: [.6, 1.5, 4.2], target: [1.35, 1.5, .4], fovYDeg: 60, near: .1, far: 300 };
const moment = { day: 172, hours: 17.5, latitude: 38, north: 0, enabled: true, playing: false, speed: 1 };
const PRESETS = ['fair', 'broken', 'overcast', 'storm', 'cirrus', 'clear'];
const SCENARIOS = ['drift', 'reflections', 'resize', 'camera-cut', 'weather-change', 'model-switch'];
export const MODES = [...PRESETS, 'weather-off', 'shadows-on', 'shadows-off', ...SCENARIOS];
const BROKEN = { preset: 'broken', transitionSeconds: 0, changing: false };
const modes = {
  ...Object.fromEntries(PRESETS.map(preset => [preset, { request: { preset, transitionSeconds: 0, changing: false }, camera: skyCamera }])),
  'weather-off': { request: { preset: 'broken', enabled: false }, camera: skyCamera },
  'shadows-on': { request: { preset: 'broken', shadows: true }, camera: groundCamera, shadow: true },
  'shadows-off': { request: { preset: 'broken', shadows: false }, camera: groundCamera, shadow: true },
  drift: { request: BROKEN, camera: skyCamera },
  reflections: { request: BROKEN, camera: reflectionCamera },
  resize: { request: BROKEN, camera: skyCamera },
  'camera-cut': { request: BROKEN, camera: skyCamera },
  'weather-change': { request: { preset: 'fair', transitionSeconds: 0, changing: false }, camera: skyCamera },
  'model-switch': { request: BROKEN, camera: skyCamera },
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
  drift: 'Broken clouds two minutes later: crisp, fully formed cloud shapes in new positions with no smearing, doubled edges or streaks. ',
  reflections: 'The chrome sphere shows the current cloudy sky: cloud masses and blue gaps are recognisable in the reflection, not a cloudless or black reflection. ',
  resize: 'After resizing smaller and back, clouds are rendered cleanly at the current size with no blocky, stretched or stale cloud image. ',
  'camera-cut': 'After an abrupt camera cut the clouds match the new view with no ghost of the previous view and no trails. ',
  'weather-change': 'After switching from fair to overcast the sky is a continuous grey sheet, with no leftover separated cumulus or stale brighter patches. ',
  'model-switch': 'After legacy and back, volumetric broken clouds are present again, not the flat legacy layer, with no stale or smeared cloud image. ',
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
  'shadows-disabled-csm-kept', 'atmosphere-submitted', 'environment-published', 'view-submitted',
  'drift-advanced', 'environment-current-generation', 'reset-resize', 'view-size-restored', 'reset-camera-discontinuity',
  'reset-weather', 'reset-model', 'history-restored'];
const common = ['snapshot:sky', 'snapshot:time', 'snapshot:weather', 'snapshot:diagnostics', 'runtime:passes', 'runtime:viewSizes', 'runtime:ready',
  'assertion:atmosphere-submitted', 'assertion:environment-published', 'assertion:view-submitted'];
// The reflections capture waits for its own pinned generation, so it does not require the settled-state publication.
const commonFor = mode => mode === 'reflections' ? common.filter(f => f !== 'assertion:environment-published') : common;
const row = (mode, criterionName, filename, extra) => ({
  id: mode === 'weather-off' ? mode : `weather-${mode}`, criterionName, scenario: 'weather', mode, filename, expectedVisibleBehavior: behaviors[mode],
  requiredFacts: [...commonFor(mode), ...extra.map(name => `assertion:${name}`)], pair: measurements[mode]?.pair ?? null,
});
const preset = ['weather-preset-applied', 'clouds-marched'];
const transition = ['history-restored'];
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
  row('drift', 'Clouds later drift and evolve', 'clouds-later-drift-and-evolution.png', ['drift-advanced']),
  row('reflections', 'Current clouds in chrome reflection', 'panorama-current-clouds-in-chrome-reflection.png', ['environment-current-generation']),
  row('resize', 'Resized sky with cloud history reset', 'resized-sky-cloud-history-reset.png', ['reset-resize', 'view-size-restored', ...transition]),
  row('camera-cut', 'Camera cut with current clouds and no trails', 'camera-cut-current-clouds-no-trails.png', ['reset-camera-discontinuity', ...transition]),
  row('weather-change', 'Weather change with current clouds and no stale panorama', 'weather-change-current-clouds-no-stale-panorama.png', ['reset-weather', ...transition]),
  row('model-switch', 'Model switch restores volumetric clouds', 'model-switch-volumetric-clouds-restored.png', ['reset-model', ...transition]),
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

const FRAME_LIMIT = 600, DRIFT_FRAMES = 480, TRANSITION_FRAMES = 4, REFLECTION_MIN_FRAMES = 60, STEP_MS = 250;
const yieldTask = () => new Promise(resolve => setTimeout(resolve, 0));
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const sameJson = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// Accepted-state publication each capture waits for; marching modes also wait for both cloud flags.
const settled = (d, marching) => d.readiness.status === 'ready' && d.readiness.error === null && d.atmosphereGeneration > 0
  && d.environment.publishedGeneration !== null && d.environment.publishedGeneration >= d.environment.requestedGeneration && d.environment.pending === false
  && d.views.length > 0 && d.views.every(v => v.submittedInvocation !== null && (!marching || (v.cloudScreenValid === true && v.cloudPanoramaValid === true)));
// Frames after the first settled wait advance the timestamp by STEP_MS each, so the simulated clock moves with them.
const tick = (engine, variant, clock) => { clock.now += STEP_MS; engine.render(clock.now); };
async function advance(engine, variant, clock, count) {
  for (let frame = 0; frame < count; frame++) { tick(engine, variant, clock); await yieldTask(); }
  return engine.skyWeatherDiagnostics();
}
async function awaitPublication(engine, variant, clock = null) {
  const marching = variant.id !== 'weather-off';
  let diagnostics = engine.skyWeatherDiagnostics();
  for (let frame = 0; frame < FRAME_LIMIT && !settled(diagnostics, marching); frame++) {
    if (diagnostics.readiness.status === 'failed') break;
    if (clock) tick(engine, variant, clock); else engine.render(variant.timeMs ?? 1000);
    await yieldTask();
    diagnostics = engine.skyWeatherDiagnostics();
  }
  return diagnostics;
}
// Newer submitted frames after the last transition edit, then the settled state.
async function settleAfter(engine, variant, clock) {
  await advance(engine, variant, clock, TRANSITION_FRAMES);
  return awaitPublication(engine, variant, clock);
}
/** A reset reason absent from every view before the action and present on every view after it, with newer submitted work. */
function reasonAdded(name, reason, before, after) {
  const viewFacts = d => JSON.stringify(d.views.map(v => ({ view: v.view, size: [v.width, v.height], submitted: v.submittedInvocation, reasons: v.resetReasons })));
  const passed = before.views.length > 0 && after.views.length === before.views.length
    && before.views.every(v => !v.resetReasons.includes(reason)) && after.views.every(v => v.resetReasons.includes(reason))
    && isNum(after.submittedInvocation) && isNum(before.submittedInvocation) && after.submittedInvocation > before.submittedInvocation;
  return { name, passed, detail: `${reason}: submitted ${before.submittedInvocation}->${after.submittedInvocation}; before ${viewFacts(before)}; after ${viewFacts(after)}` };
}
/** At least TRANSITION_FRAMES newer submitted frames after the final edit, with valid history and both cloud flags on every view. */
function historyRestored(mark, after) {
  const newer = isNum(mark) && isNum(after.submittedInvocation) ? after.submittedInvocation - mark : -1;
  const passed = newer >= TRANSITION_FRAMES && after.views.length > 0
    && after.views.every(v => v.historyValid === true && v.cloudScreenValid === true && v.cloudPanoramaValid === true);
  return { name: 'history-restored', passed, detail: `newer submitted frames ${newer}; [view, historyValid, screen, panorama] ${JSON.stringify(after.views.map(v => [v.view, v.historyValid, v.cloudScreenValid, v.cloudPanoramaValid]))}` };
}

/** Shadow elements of the pass lists stored around the weather edit, with their enabled values before and after. */
function shadowPasses(lists) {
  const find = (list, name) => list?.find(p => p.name === name)?.enabled;
  const names = [...new Set([...(lists?.before ?? []), ...(lists?.after ?? [])].map(p => p.name).filter(n => n.startsWith('shadow.')))];
  return names.map(name => ({ name, before: find(lists?.before, name), after: find(lists?.after, name) }));
}

const EVERY_FRAME = { envUpdate: 'everyFrame' };
const CHANGE = { preset: 'overcast' };
// Scenario modes run their public actions after the first settled wait. Each returns the diagnostics of its final
// state and a judge of that state; actions are recorded through `record` so their getter snapshots reach `edits`.
const scenarios = {
  async drift(engine, variant, before, clock) {
    const target = structuredClone(engine.weather().target);
    const end = await advance(engine, variant, clock, DRIFT_FRAMES);
    const endTarget = structuredClone(engine.weather().target);
    const diagnostics = await awaitPublication(engine, variant, clock);
    const advanced = end.weatherGeneration - before.weatherGeneration;
    return { diagnostics, judge: () => [{ name: 'drift-advanced',
      passed: advanced >= 400 && isNum(before.submittedInvocation) && isNum(end.submittedInvocation) && end.submittedInvocation > before.submittedInvocation
        && flagsOn(end) && sameJson(endTarget, target),
      detail: `weatherGeneration ${before.weatherGeneration}->${end.weatherGeneration} over ${DRIFT_FRAMES} frames; submitted ${before.submittedInvocation}->${end.submittedInvocation}; `
        + `[view, screen, panorama] ${flagDetail(end)}; authored target unchanged ${sameJson(endTarget, target)}` }] };
  },
  async reflections(engine, variant, before, clock) {
    const P0 = before.environment.publishedGeneration;
    record(engine, 'transition', "setSky({envUpdate:'everyFrame'})", EVERY_FRAME, () => engine.setSky(EVERY_FRAME));
    tick(engine, variant, clock);
    await yieldTask();
    let diagnostics = engine.skyWeatherDiagnostics();
    // The pinned request is the first generation requested under everyFrame; publication must reach it and rise above P0.
    const G = diagnostics.environment.requestedGeneration;
    const reached = d => d.environment.publishedGeneration !== null && d.environment.publishedGeneration >= G;
    let rendered = 1;
    while (rendered < FRAME_LIMIT && !(rendered >= REFLECTION_MIN_FRAMES && reached(diagnostics))) {
      tick(engine, variant, clock);
      await yieldTask();
      rendered++;
      diagnostics = engine.skyWeatherDiagnostics();
    }
    const P = diagnostics.environment.publishedGeneration;
    return { diagnostics, judge: () => [{ name: 'environment-current-generation',
      passed: P !== null && P >= G && isNum(P0) && P > P0 && diagnostics.views.length > 0 && diagnostics.views.every(v => v.cloudPanoramaValid === true),
      detail: `requested G ${G}; published ${P} (P0 ${P0}) after ${rendered} frames; environment ${JSON.stringify(diagnostics.environment)}; `
        + `[view, panorama] ${JSON.stringify(diagnostics.views.map(v => [v.view, v.cloudPanoramaValid]))}` }] };
  },
  async resize(engine, variant, before, clock) {
    const [width, height] = [before.views[0]?.width, before.views[0]?.height];
    record(engine, 'transition', `resize(${RESIZE_TO.join(',')})`, { size: RESIZE_TO }, () => engine.resize(...RESIZE_TO));
    const mid = await advance(engine, variant, clock, TRANSITION_FRAMES);
    record(engine, 'transition', `resize(${width},${height})`, { size: [width, height] }, () => engine.resize(width, height));
    const mark = engine.skyWeatherDiagnostics().submittedInvocation;
    const diagnostics = await settleAfter(engine, variant, clock);
    return { diagnostics, judge: after => [reasonAdded('reset-resize', 'Resize', before, after), { name: 'view-size-restored',
      passed: mid.views[0]?.width === RESIZE_TO[0] && mid.views[0]?.height === RESIZE_TO[1] && after.views[0]?.width === width && after.views[0]?.height === height,
      detail: `view 0 ${width}x${height} -> ${mid.views[0]?.width}x${mid.views[0]?.height} -> ${after.views[0]?.width}x${after.views[0]?.height}` },
    historyRestored(mark, after)] };
  },
  async 'camera-cut'(engine, variant, before, clock) {
    const { aspect, ...camera } = variant.selection.camera;
    record(engine, 'transition', `setCamera(${JSON.stringify(camera)})`, camera, () => engine.setCamera(camera));
    const mark = engine.skyWeatherDiagnostics().submittedInvocation;
    const diagnostics = await settleAfter(engine, variant, clock);
    return { diagnostics, judge: after => [reasonAdded('reset-camera-discontinuity', 'CameraDiscontinuity', before, after), historyRestored(mark, after)] };
  },
  async 'model-switch'(engine, variant, before, clock) {
    const a = record(engine, 'transition', "setSky({model:'legacy'})", { model: 'legacy' }, () => engine.setSky({ model: 'legacy' }));
    const legacy = await advance(engine, variant, clock, TRANSITION_FRAMES);
    const b = record(engine, 'transition', "setSky({model:'physical'})", { model: 'physical' }, () => engine.setSky({ model: 'physical' }));
    const mark = engine.skyWeatherDiagnostics().submittedInvocation;
    const diagnostics = await settleAfter(engine, variant, clock);
    return { diagnostics, judge: after => {
      const reason = reasonAdded('reset-model', 'Model', before, after);
      const legacyOff = legacy.views.length > 0 && legacy.views.every(v => v.cloudPanoramaValid === false && v.cloudScreenValid === false);
      const kept = [a, b].every(e => sameJson(e.after.weather, e.before.weather) && sameJson(e.after.time, e.before.time));
      return [{ ...reason, passed: reason.passed && a.accepted && b.accepted && legacyOff && kept,
        detail: `${reason.detail}; legacy [view, screen, panorama] ${flagDetail(legacy)}; weather and time kept by both switches ${kept}` }, historyRestored(mark, after)];
    } };
  },
  async 'weather-change'(engine, variant, before, clock) {
    record(engine, 'transition', "setWeather({preset:'overcast'})", CHANGE, () => engine.setWeather(CHANGE));
    const mark = engine.skyWeatherDiagnostics().submittedInvocation;
    const diagnostics = await settleAfter(engine, variant, clock);
    return { diagnostics, judge: (after, current) => {
      const reason = reasonAdded('reset-weather', 'Weather', before, after);
      const overcast = targetMatches(current.weather?.target, targets.overcast);
      return [{ ...reason, passed: reason.passed && after.weatherGeneration > before.weatherGeneration && overcast,
        detail: `${reason.detail}; weatherGeneration ${before.weatherGeneration}->${after.weatherGeneration}; target ${JSON.stringify(current.weather?.target)}; overcast ${overcast}` }, historyRestored(mark, after)];
    } };
  },
};

export async function observe(engine, variant) {
  const mode = variant.id, spec = modes[mode];
  if (!spec) throw new Error(`weather has no mode ${mode}`);
  const settledBefore = await awaitPublication(engine, variant);
  const clock = { now: variant.timeMs ?? 1000 };
  const scenario = scenarios[mode] ? await scenarios[mode](engine, variant, settledBefore, clock) : null;
  const diagnostics = scenario ? scenario.diagnostics : settledBefore;
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
  if (scenario) {
    assertions.push(...scenario.judge(diagnostics, { sky, time, weather }));
  } else if (PRESETS.includes(mode)) {
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
    // The camera-cut mode starts on the pre-cut camera; its transition then cuts to the capture camera.
    if (mode === 'camera-cut') record(engine, 'camera', 'setCamera(pre-cut)', preCutCamera, () => engine.setCamera(preCutCamera));
  },
};
