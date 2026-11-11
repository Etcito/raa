import test from 'node:test';
import assert from 'node:assert/strict';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import * as foundation from './scenes/foundation.js';
import * as atmosphere from './scenes/atmosphere.js';
import * as weatherScene from './scenes/weather.js';

const scene = weatherScene.default;
const table = [['fair', 'weather-fair', 'fair-separated-cumulus.png'], ['broken', 'weather-broken', 'broken-cloud-blue-gaps.png'],
  ['overcast', 'weather-overcast', 'overcast-low-grey-sheet.png'], ['storm', 'weather-storm', 'storm-dark-towering-cloud.png'],
  ['cirrus', 'weather-cirrus', 'cirrus-high-fibrous-veil.png'], ['clear', 'weather-clear', 'clear-preset-cloudless-blue-sky.png'],
  ['weather-off', 'weather-off', 'weather-off-clear-sky.png'], ['shadows-on', 'weather-shadows-on', 'ground-cloud-shadow-on.png'],
  ['shadows-off', 'weather-shadows-off', 'ground-cloud-shadow-off-csm-retained.png'],
  ['drift', 'weather-drift', 'clouds-later-drift-and-evolution.png'], ['reflections', 'weather-reflections', 'panorama-current-clouds-in-chrome-reflection.png'],
  ['resize', 'weather-resize', 'resized-sky-cloud-history-reset.png'], ['camera-cut', 'weather-camera-cut', 'camera-cut-current-clouds-no-trails.png'],
  ['weather-change', 'weather-weather-change', 'weather-change-current-clouds-no-stale-panorama.png'],
  ['model-switch', 'weather-model-switch', 'model-switch-volumetric-clouds-restored.png']];
const TRANSITIONS = ['resize', 'camera-cut', 'weather-change', 'model-switch'];
const RESET_FACT = { resize: 'reset-resize', 'camera-cut': 'reset-camera-discontinuity', 'weather-change': 'reset-weather', 'model-switch': 'reset-model' };
// Targets from crates/raa-sky/weather/*.weather; omitted keys keep the fair defaults.
const fair = { coverage: .3, type: .9, base: 1200, thickness: 1600, density: 1, cirrus: .15, wind: { speed: 7, from: 250 }, evolution: 1, rain: 0 };
const presets = { fair, clear: { ...fair, coverage: 0, cirrus: .05, wind: { speed: 5, from: 250 } },
  broken: { coverage: .62, type: .95, base: 1000, thickness: 2600, density: 1.15, cirrus: .3, wind: { speed: 10, from: 240 }, evolution: 1.3, rain: 0 },
  overcast: { coverage: .97, type: .12, base: 600, thickness: 900, density: 1.3, cirrus: 0, wind: { speed: 8, from: 220 }, evolution: .6, rain: 0 },
  storm: { coverage: .85, type: 1, base: 800, thickness: 7000, density: 2, cirrus: .6, wind: { speed: 16, from: 230 }, evolution: 2, rain: 1 },
  cirrus: { ...fair, coverage: .04, type: .9, cirrus: .75, wind: { speed: 20, from: 260 } } };

// Public-API fake. `touch(passes)` runs inside setWeather to model a weather edit that changes pass state. Reset reasons
// are recorded only after frames have rendered, as the engine does for transitions; the options script false evidence:
// noReset (no reason), historyLost (history never valid), targetDrift (authored target changes per frame),
// frozenWeather (weatherGeneration stays), stuckPublication (everyFrame publication stays at its first generation).
function fakeEngine({ touch = null, readyAfter = 2, noReset = false, historyLost = false, targetDrift = false, frozenWeather = false, stuckPublication = false } = {}) {
  const calls = [], renderTimes = [], reasons = new Set();
  let renders = 0, lastReset = null, size = { width: 1280, height: 720 };
  let passes = [{ name: 'shadow.csm', role: 'pass', enabled: true, order: 20, required: false }, { name: 'shadow.contact', role: 'pass', enabled: true, order: 45, required: false },
    { name: 'forward', role: 'pass', enabled: true, order: 60, required: true }];
  const time = { day: 172, hours: 9, latitude: 38, north: 0, enabled: false, playing: false, speed: 1 };
  let weather = { enabled: true, shadows: true, changing: false, seed: 1, transitionSeconds: 0, target: structuredClone(fair), cyclePreset: null };
  const sky = { model: 'physical', profile: 'day', envUpdate: 'onChange', intensity: 1, atmosphere: { groundAlbedo: [.3, .3, .3], mieMultiplier: 1 } };
  const note = name => (...args) => { calls.push([name, ...args]); };
  const reset = reason => { if (renders > 0 && !noReset) { reasons.add(reason); lastReset = renders; } };
  const ready = () => renders >= readyAfter;
  return {
    calls, renderTimes, setBackground: note('setBackground'), setLight: note('setLight'), setAmbient: note('setAmbient'),
    setCamera: (...args) => { calls.push(['setCamera', ...args]); reset('CameraDiscontinuity'); },
    setShadows: note('setShadows'), setPassEnabled: note('setPassEnabled'),
    setSky: v => {
      calls.push(['setSky', v]);
      if (v.model && v.model !== sky.model) { sky.model = v.model; reset('Model'); }
      if (v.envUpdate) sky.envUpdate = v.envUpdate;
    },
    resize: (width, height) => { calls.push(['resize', width, height]); size = { width, height }; reset('Resize'); },
    setTimeOfDay: v => { calls.push(['setTimeOfDay', v]); Object.assign(time, v); },
    setWeather: v => {
      calls.push(['setWeather', v]);
      const { preset, ...rest } = v;
      weather = { ...weather, ...rest, target: structuredClone(presets[preset] ?? weather.target) };
      reset('Weather');
      touch?.(passes, list => { passes = list; });
    },
    primitiveMesh: () => 1, addMaterial: () => 1, addNode: () => 1, addInstance: () => calls.push(['addInstance']),
    passes: () => structuredClone(passes),
    render: timeMs => {
      renders++; renderTimes.push(timeMs);
      if (targetDrift) weather = { ...weather, target: { ...weather.target, coverage: weather.target.coverage + 1e-3 } };
    },
    sky: () => structuredClone(sky),
    timeOfDay: () => ({ ...time, currentDay: time.day, currentHours: time.hours, sunDirection: [0, .5, -.8], moonDirection: [0, -.5, .8], moonLit: .5,
      starPole: [0, .6, -.8], starEquinox: [1, 0, 0], dominant: 'sun' }),
    weather: () => ({ ...structuredClone(weather), current: structuredClone(weather.target), text: '' }),
    skyWeatherDiagnostics: () => {
      const isReady = ready(), flags = weather.enabled && sky.model === 'physical' && isReady;
      // Under everyFrame each frame requests one more generation and publication follows one frame behind.
      const environment = sky.envUpdate === 'everyFrame' && isReady
        ? { requestedGeneration: 1 + renders, publishedGeneration: stuckPublication ? 1 : renders, pending: true }
        : { requestedGeneration: 1, publishedGeneration: isReady ? 1 : null, pending: !isReady };
      return { invocation: renders || null, submittedInvocation: isReady ? renders : null, weatherGeneration: frozenWeather ? 1 : 1 + renders,
        atmosphereGeneration: isReady ? 1 : 0, environment,
        views: [{ view: 0, width: size.width, height: size.height, invocation: renders, submittedInvocation: isReady ? renders : null, skyViewGeneration: 1,
          historyValid: lastReset === null || (!historyLost && renders - lastReset >= 4), resetReasons: [...reasons], cloudScreenValid: flags, cloudPanoramaValid: flags }],
        readiness: isReady ? { status: 'ready', stage: 'submission', error: null } : { status: 'pending', stage: 'atmosphere', error: null } };
    },
  };
}
const runtime = { passes: [{ name: 'forward', role: 'pass', enabled: true, order: 60, required: true }],
  viewSizes: [{ view: 0, width: 1280, height: 720 }], readiness: { status: 'ready', stage: 'weather', error: null } };

/** Builds and observes one mode; `patch(engine)` tampers with the fake's getters between build and observe. */
async function run(mode, options, patch) {
  const variant = scene.variants.find(v => v.id === mode);
  const engine = fakeEngine(options);
  await scene.build(engine, {}, variant);
  patch?.(engine);
  return { engine, variant, observation: await weatherScene.observe(engine, variant) };
}
const assertion = (observation, name) => observation.assertions.find(a => a.name === name);

test('the weather scene resolves through validateInventory for the weather stage', () => {
  const rows = validateInventory([foundation, atmosphere, weatherScene], 'weather').filter(r => r.scenario === 'weather');
  assert.equal(rows.length, 15);
  for (const v of scene.variants) {
    assert.deepEqual(v.selection.subjects.map(s => s.name).sort(), ['distant-centre', 'distant-left', 'distant-right', 'ground', 'matte-reference', 'reflective-sphere']);
    assert.equal(v.engine.exposureEv, 0);
  }
  const moved = { ...weatherScene, default: { ...scene, variants: scene.variants.map(v => ({ ...v, selection: { ...v.selection,
    camera: { ...v.selection.camera, eye: [0, 1, 20], target: [0, 1, 30] } } })) } };
  assert.throws(() => validateInventory([moved], 'weather'), /frustum/);
});

test('filenames, modes and pairs match the declared table', () => {
  assert.deepEqual(scene.variants.map(v => v.id), table.map(t => t[0]));
  assert.deepEqual(weatherScene.inventory.map(r => [r.mode, r.id, r.filename]), table);
  assert.equal(scene.id, 'weather'); assert.equal(scene.milestone, 'weather');
  assert.deepEqual(weatherScene.inventory.filter(r => r.pair).map(r => [r.mode, r.pair]),
    [['broken', 'weather'], ['weather-off', 'weather'], ['shadows-on', 'cloud-shadows'], ['shadows-off', 'cloud-shadows']]);
  for (const r of weatherScene.inventory) for (const fact of r.requiredFacts.filter(f => f.startsWith('assertion:'))) assert.ok(weatherScene.factNames.includes(fact.slice(10)));
});

test('each preset mode authors its named preset and passes weather-preset-applied', async () => {
  for (const mode of ['fair', 'broken', 'overcast', 'storm', 'cirrus', 'clear']) {
    const { engine, observation } = await run(mode);
    assert.deepEqual(engine.calls.find(c => c[0] === 'setSky')[1], { preset: 'day', model: 'physical' });
    assert.deepEqual(engine.calls.find(c => c[0] === 'setWeather')[1], { preset: mode, transitionSeconds: 0, changing: false });
    assert.equal(assertion(observation, 'weather-preset-applied').passed, true, mode);
    assert.equal(assertion(observation, 'clouds-marched').passed, true, mode);
    assert.equal(observation.edits.length, 2);
    assert.ok(observation.edits.every(e => e.accepted && e.before.weather && e.after.weather));
    validateRequiredFacts(weatherScene.inventory.find(r => r.mode === mode), { ...observation, ...runtime }, weatherScene.factNames);
  }
  const { observation } = await run('storm', {}, engine => { const real = engine.weather; engine.weather = () => { const w = real(); w.target.thickness = 1600; return w; }; });
  assert.equal(assertion(observation, 'weather-preset-applied').passed, false);
});

test('weather-off and shadows-off are positive assertions of the disabled state', async () => {
  const off = await run('weather-off', { readyAfter: 2 });
  assert.equal(assertion(off.observation, 'clouds-neutral').passed, true);
  assert.equal(assertion(off.observation, 'weather-disabled').passed, true);
  assert.deepEqual(off.engine.calls.find(c => c[0] === 'setWeather')[1], { preset: 'broken', enabled: false });
  const flagsOn = e => { const real = e.skyWeatherDiagnostics; e.skyWeatherDiagnostics = () => { const d = real(); d.views.forEach(v => { v.cloudScreenValid = v.cloudPanoramaValid = true; }); return d; }; };
  const enabledOn = e => { const real = e.weather; e.weather = () => ({ ...real(), enabled: true }); };
  const a = (await run('weather-off', {}, flagsOn)).observation;
  assert.deepEqual([assertion(a, 'clouds-neutral').passed, assertion(a, 'weather-disabled').passed], [false, true]);
  const b = (await run('weather-off', {}, enabledOn)).observation;
  assert.deepEqual([assertion(b, 'clouds-neutral').passed, assertion(b, 'weather-disabled').passed], [true, false]);
  const on = await run('shadows-on'), no = await run('shadows-off');
  assert.equal(assertion(on.observation, 'shadows-enabled').passed, true);
  assert.equal(assertion(no.observation, 'shadows-disabled-csm-kept').passed, true);
  assert.deepEqual(no.engine.calls.find(c => c[0] === 'setWeather')[1], { preset: 'broken', shadows: false });
  const stillOn = await run('shadows-off', {}, e => { const real = e.weather; e.weather = () => ({ ...real(), shadows: true }); });
  assert.equal(assertion(stillOn.observation, 'shadows-disabled-csm-kept').passed, false);
});

test('shadows-off fails when a shadow pass changed its enabled value or shadow.csm is missing', async () => {
  const verdict = async touch => assertion((await run('shadows-off', { touch })).observation, 'shadows-disabled-csm-kept');
  const flipped = await verdict(passes => { passes.find(p => p.name === 'shadow.contact').enabled = false; });
  assert.equal(flipped.passed, false);
  assert.match(flipped.detail, /shadow\.contact: true->false/);
  const missing = await verdict((passes, set) => set(passes.filter(p => p.name !== 'shadow.csm')));
  assert.equal(missing.passed, false);
  assert.match(missing.detail, /shadow\.csm: true->undefined/);
  const csmOff = await verdict((passes, set) => set(passes.map(p => ({ ...p, enabled: p.name === 'shadow.csm' ? false : p.enabled }))));
  assert.equal(csmOff.passed, false);
  const added = await verdict((passes, set) => set([...passes, { name: 'shadow.extra', role: 'pass', enabled: true, order: 46, required: false }]));
  assert.equal(added.passed, false);
});

test('the shadow modes make no setShadows or setPassEnabled call', async () => {
  for (const mode of ['shadows-on', 'shadows-off']) {
    const { engine } = await run(mode);
    assert.deepEqual(engine.calls.filter(c => ['setShadows', 'setPassEnabled'].includes(c[0])), [], mode);
    assert.equal(engine.calls.filter(c => c[0] === 'setWeather').length, 1);
  }
});

test('missing, null and false evidence rejects each required selector', async () => {
  for (const r of weatherScene.inventory) {
    const { observation } = await run(r.mode);
    const base = () => ({ ...structuredClone(observation), ...structuredClone(runtime) });
    validateRequiredFacts(r, base(), weatherScene.factNames);
    for (const key of r.requiredFacts) {
      const [kind, name] = key.split(':'), field = name === 'ready' ? 'readiness' : name;
      const mutations = kind === 'assertion'
        ? [f => { f.assertions = f.assertions.filter(a => a.name !== name); }, f => { f.assertions.find(a => a.name === name).passed = null; }, f => { f.assertions.find(a => a.name === name).passed = false; }]
        : [f => { f[field] = null; }, f => { delete f[field]; }];
      for (const mutate of mutations) { const f = base(); mutate(f); assert.throws(() => validateRequiredFacts(r, f, weatherScene.factNames), new RegExp(r.id), `${r.id} ${key}`); }
    }
  }
});

test('measurement pairs are complete on/off pairs with the same preset', () => {
  const pairs = {};
  for (const v of scene.variants.filter(v => v.measurement)) (pairs[v.measurement.pair] ??= []).push(v);
  assert.deepEqual(Object.keys(pairs), ['weather', 'cloud-shadows']);
  for (const list of Object.values(pairs)) {
    assert.deepEqual(list.map(v => v.measurement.state), ['on', 'off']);
    assert.equal(new Set(list.map(v => v.measurement.preset)).size, 1);
    assert.equal(list[0].measurement.preset, 'broken');
  }
  assert.deepEqual(pairs.weather.map(v => [v.id, v.measurement.features]), [['broken', { weather: true }], ['weather-off', { weather: false }]]);
  assert.deepEqual(pairs['cloud-shadows'].map(v => [v.id, v.measurement.features]), [['shadows-on', { shadows: true }], ['shadows-off', { shadows: false }]]);
  const broken = { ...scene, variants: scene.variants.map(v => v.id === 'weather-off' ? { ...v, measurement: { ...v.measurement, preset: 'fair' } } : v) };
  assert.throws(() => validateInventory([{ ...weatherScene, default: broken }], 'weather'), /measurement pair/);
});

const rowOf = mode => weatherScene.inventory.find(r => r.mode === mode);

test('the complete weather inventory has fifteen uniquely named rows and resolves for the weather stage', () => {
  const rows = validateInventory([foundation, atmosphere, weatherScene], 'weather').filter(r => r.scenario === 'weather');
  assert.deepEqual(rows.map(r => r.id), table.map(t => t[1]));
  for (const key of ['id', 'criterionName', 'filename']) assert.equal(new Set(rows.map(r => r[key])).size, 15, key);
  assert.equal(weatherScene.factNames.length, 17);
  assert.equal(new Set(weatherScene.factNames).size, 17);
});

test('transition modes add their reset reason and require history-restored', async () => {
  for (const mode of TRANSITIONS) {
    const { observation } = await run(mode);
    assert.equal(assertion(observation, RESET_FACT[mode]).passed, true, mode);
    assert.equal(assertion(observation, 'history-restored').passed, true, mode);
    assert.ok(rowOf(mode).requiredFacts.includes('assertion:history-restored'), mode);
    assert.ok(observation.edits.length >= 3 && observation.edits.every(e => e.accepted), mode);
    validateRequiredFacts(rowOf(mode), { ...observation, ...runtime }, weatherScene.factNames);
  }
  const resize = await run('resize');
  assert.deepEqual(resize.engine.calls.filter(c => c[0] === 'resize').map(c => c.slice(1)), [atmosphere.RESIZE_TO, [1280, 720]]);
  assert.equal(assertion(resize.observation, 'view-size-restored').passed, true);
  const model = await run('model-switch');
  assert.deepEqual(model.engine.calls.filter(c => c[0] === 'setSky' && c[1].model).map(c => c[1].model), ['physical', 'legacy', 'physical']);
  const camera = await run('camera-cut');
  assert.equal(camera.engine.calls.filter(c => c[0] === 'setCamera').length, 3); // build, pre-cut, cut
  const change = await run('weather-change');
  assert.deepEqual(change.engine.calls.filter(c => c[0] === 'setWeather').map(c => c[1]), [{ preset: 'fair', transitionSeconds: 0, changing: false }, { preset: 'overcast' }]);
});

test('drift requires advancing generations and an unchanged authored target', async () => {
  const { engine, observation } = await run('drift');
  assert.equal(assertion(observation, 'drift-advanced').passed, true);
  validateRequiredFacts(rowOf('drift'), { ...observation, ...runtime }, weatherScene.factNames);
  const drift = engine.renderTimes.slice(-480);
  assert.equal(drift.length, 480);
  assert.ok(drift.slice(1).every((t, i) => t - drift[i] === 250));
  assert.equal(engine.calls.filter(c => c[0] === 'setWeather').length, 1);
  const frozen = await run('drift', { frozenWeather: true });
  assert.equal(assertion(frozen.observation, 'drift-advanced').passed, false);
  assert.throws(() => validateRequiredFacts(rowOf('drift'), { ...frozen.observation, ...runtime }, weatherScene.factNames), /weather-drift/);
  const moving = await run('drift', { targetDrift: true });
  assert.equal(assertion(moving.observation, 'drift-advanced').passed, false);
});

test('reflections require a published generation that reached the pinned request and rose', async () => {
  const { engine, observation } = await run('reflections');
  assert.equal(assertion(observation, 'environment-current-generation').passed, true);
  assert.deepEqual(engine.calls.filter(c => c[0] === 'setSky' && c[1].envUpdate).map(c => c[1]), [{ envUpdate: 'everyFrame' }]);
  validateRequiredFacts(rowOf('reflections'), { ...observation, ...runtime }, weatherScene.factNames);
  const stuck = await run('reflections', { stuckPublication: true });
  assert.equal(assertion(stuck.observation, 'environment-current-generation').passed, false);
  assert.throws(() => validateRequiredFacts(rowOf('reflections'), { ...stuck.observation, ...runtime }, weatherScene.factNames), /weather-reflections/);
});

test('the reflections row does not require environment-published', async () => {
  const row = rowOf('reflections');
  assert.equal(row.requiredFacts.includes('assertion:environment-published'), false);
  for (const fact of ['assertion:environment-current-generation', 'assertion:atmosphere-submitted', 'assertion:view-submitted', 'snapshot:diagnostics', 'runtime:ready'])
    assert.ok(row.requiredFacts.includes(fact), fact);
  assert.ok(rowOf('drift').requiredFacts.includes('assertion:environment-published'));
  const { observation } = await run('reflections');
  assert.equal(assertion(observation, 'environment-published').passed, false); // everyFrame publication lags its request
  validateRequiredFacts(row, { ...observation, ...runtime }, weatherScene.factNames);
});

test('false or missing reset evidence rejects the transition rows', async () => {
  for (const mode of TRANSITIONS) {
    const row = rowOf(mode), reset = RESET_FACT[mode];
    const noReason = await run(mode, { noReset: true });
    assert.equal(assertion(noReason.observation, reset).passed, false, mode);
    assert.throws(() => validateRequiredFacts(row, { ...noReason.observation, ...runtime }, weatherScene.factNames), new RegExp(row.id), mode);
    const noHistory = await run(mode, { historyLost: true });
    assert.equal(assertion(noHistory.observation, 'history-restored').passed, false, mode);
    assert.throws(() => validateRequiredFacts(row, { ...noHistory.observation, ...runtime }, weatherScene.factNames), new RegExp(row.id), mode);
    const { observation } = await run(mode);
    const without = name => ({ ...observation, ...runtime, assertions: observation.assertions.filter(a => a.name !== name) });
    assert.throws(() => validateRequiredFacts(row, without(reset), weatherScene.factNames), new RegExp(row.id), mode);
    assert.throws(() => validateRequiredFacts(row, without('history-restored'), weatherScene.factNames), new RegExp(row.id), mode);
  }
});
