import test from 'node:test';
import assert from 'node:assert/strict';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import * as foundation from './scenes/foundation.js';
import * as atmosphere from './scenes/atmosphere.js';

const scene = atmosphere.default;
const filenames = ['clear-day-warm-horizon-and-reflective-sphere.png', 'twilight-warm-horizon-distant-forms.png',
  'night-moon-phase-stars-and-milky-way.png', 'night-later-celestial-positions.png',
  'snow-preset-cold-overcast-bright-ground.png', 'jungle-preset-humid-broken-cloud.png', 'legacy-gradient-layer-comparison.png',
  'fog-on-sky-coherent-distant-forms.png', 'fog-off-distant-forms.png', 'ibl-on-current-sky-reflection.png', 'ibl-off-flat-ambient.png',
  'ibl-diffuse-only-sky-fill-no-reflection.png', 'ibl-specular-only-chrome-sky-reflection.png', 'environment-disabled-cleared-ibl-sky-kept.png',
  'background-disabled-flat-colour.png', 'error-invalid-sky-time-weather-edits-last-valid-state.png', 'multi-view-two-cameras-same-sky-clock.png',
  'transition-resize-current-sky-history-reset.png', 'transition-camera-cut-current-sky-no-trails.png', 'transition-model-switch-physical-sky-restored.png',
  'transition-weather-cycle-start-and-restart.png', 'transition-re-enable-sky-environment-restored.png'];
const modes = ['day', 'twilight', 'night', 'later', 'snow', 'jungle', 'legacy', 'fog-on', 'fog-off', 'ibl-on', 'ibl-off', 'ibl-diffuse',
  'ibl-specular', 'env-off', 'background-off', 'invalid', 'multi-view', 'resize', 'camera-cut', 'model-switch', 'weather-change', 're-enable'];

// Public-API fake: records delivered calls and evaluates the authored moments with the
// source celestial formulas only to supply plausible getter values for the observer.
function fakeEngine({ readyAfter = 3, failGetter = null } = {}) {
  const calls = [];
  let time = { day: 172, hours: 9, latitude: 38, north: 0, enabled: false, playing: false, speed: 1 };
  let cursor = { currentDay: time.day, currentHours: time.hours };
  let renders = 0;
  // Renderer lifecycle model of the documented diagnostics: view slots, accumulated reset
  // reasons, environment request/publication generations and pass enablement.
  let slots = [], rects = 1, size = [1280, 720], requested = 2, published = null, weatherGeneration = 1, cameraSet = false;
  const passes = new Map([['fog', true], ['sky.env', true], ['sky.background', true], ['sky.cloud-lighting', true], ['forward', true]]);
  const reason = r => { for (const slot of slots) if (!slot.reasons.includes(r)) slot.reasons.push(r); };
  const sourceChanged = () => { requested++; };
  const invalid = msg => Object.assign(new Error(msg), { code: 'sky.invalid' });
  const validate = t => {
    for (const k of ['day', 'hours', 'latitude', 'north', 'speed']) if (typeof t[k] !== 'number' || !Number.isFinite(t[k])) throw invalid(`${k} must be finite`);
    if (!Number.isInteger(t.day) || t.day < 1 || t.day > 365) throw invalid('day must be an integer 1..365');
    if (t.hours < 0 || t.hours >= 24) throw invalid('hours must lie in [0,24)');
    if (t.latitude < -90 || t.latitude > 90) throw invalid('latitude must lie in [-90,90]');
    if (t.speed < 0) throw invalid('speed must be >= 0');
    for (const k of ['enabled', 'playing']) if (typeof t[k] !== 'boolean') throw invalid(`${k} must be boolean`);
  };
  const T = 2 * Math.PI, r = Math.PI / 180;
  const aa = (d, h, l) => [Math.asin(Math.sin(l) * Math.sin(d) + Math.cos(l) * Math.cos(d) * Math.cos(h)),
    Math.atan2(-Math.sin(h) * Math.cos(d), Math.sin(d) * Math.cos(l) - Math.cos(d) * Math.sin(l) * Math.cos(h))];
  const hz = ([a, z], n) => { const l = [Math.cos(a) * Math.sin(z), Math.sin(a), -Math.cos(a) * Math.cos(z)];
    return [Math.cos(n) * l[0] + Math.sin(n) * l[2], l[1], -Math.sin(n) * l[0] + Math.cos(n) * l[2]]; };
  const state = () => {
    const L = time.latitude * r, sd = -23.44 * r * Math.cos(T * (time.day + 10) / 365), sh = T * (time.hours - 12) / 24;
    const age = (((time.day + time.hours / 24 - 6) / 29.53059) % 1 + 1) % 1;
    const sun = hz(aa(sd, sh, L), time.north), moon = hz(aa(sd * Math.cos(T * age), sh - T * age, L), time.north);
    const dominant = !time.enabled ? 'manual' : sun[1] > -.17 ? 'sun' : 'moon';
    return { ...time, ...cursor, sunDirection: sun, moonDirection: moon,
      moonLit: .5 - .5 * Math.cos(T * age), starPole: [0, .6, -.8], starEquinox: hz(aa(0, sh + 1.5 * time.day / 100, L), time.north), dominant };
  };
  // Documented m08 sky/weather contract: preset first, then overrides; physical presets set
  // weather target and day 172 preset hour; legacy presets and model-only switches keep both.
  const fair = { coverage: .3, type: .9, base: 1200, thickness: 1600, density: 1, cirrus: .15, wind: { speed: 7, from: 250 }, evolution: 1, rain: 0 };
  const weatherPresets = { fair, clear: { ...fair, coverage: 0, cirrus: .05, wind: { speed: 5, from: 250 } },
    broken: { coverage: .62, type: .95, base: 1000, thickness: 2600, density: 1.15, cirrus: .3, wind: { speed: 10, from: 240 }, evolution: 1.3, rain: 0 },
    overcast: { coverage: .97, type: .12, base: 600, thickness: 900, density: 1.3, cirrus: 0, wind: { speed: 8, from: 220 }, evolution: .6, rain: 0 } };
  const skyPresets = { day: ['fair', 9, [.3, .3, .3], 1], snow: ['overcast', 12, [.9, .9, .9], 1.5], jungle: ['broken', 15, [.18, .26, .14], 1.2], night: ['clear', 0, [.3, .3, .3], 1] };
  let sky = { model: 'physical', profile: 'day', intensity: 1, zenith: [0, .2, 1], horizon: [1, .8, .6], ground: [.3, .3, .3], atmosphere: { groundAlbedo: [.3, .3, .3], mieMultiplier: 1 } };
  let weather = { enabled: true, shadows: true, changing: false, seed: 1, transitionSeconds: 0, target: structuredClone(fair), cyclePreset: null };
  const engine = {
    calls,
    setBackground: v => calls.push(['setBackground', v]), setLight: v => calls.push(['setLight', v]),
    setAmbient: v => calls.push(['setAmbient', v]),
    setSky: v => {
      calls.push(['setSky', v]);
      if (v.atmosphere?.mieMultiplier !== undefined && !(v.atmosphere.mieMultiplier > 0)) throw invalid('atmosphere.mieMultiplier must be > 0');
      const next = structuredClone(sky);
      if (v.preset) {
        if (!skyPresets[v.preset]) throw invalid('preset unknown');
        const [w, hours, groundAlbedo, mieMultiplier] = skyPresets[v.preset];
        Object.assign(next, { model: 'physical', profile: v.preset, atmosphere: { groundAlbedo, mieMultiplier } });
        if ((v.model ?? 'physical') === 'physical') {
          time = { ...time, day: 172, hours }; cursor = { currentDay: 172, currentHours: hours };
          weather = { ...weather, target: structuredClone(weatherPresets[w]), cyclePreset: w };
        }
      }
      if (v.model) next.model = v.model;
      if (next.model !== sky.model) reason('Model');
      sky = next; sourceChanged();
    },
    setWeather: v => {
      calls.push(['setWeather', v]);
      const { preset, wind, changing, seed, ...rest } = v;
      const base = preset ? structuredClone(weatherPresets[preset]) : structuredClone(weather.target);
      const target = { ...base, ...rest, wind: { ...base.wind, ...wind } };
      if (!(target.coverage >= 0 && target.coverage <= 1)) throw invalid('coverage must lie in [0,1]');
      // Explicit preset is the cycle start; manual target edits stop changing unless changing:true is in the patch.
      const targetEdit = preset !== undefined || Object.keys(rest).length > 0 || wind !== undefined;
      const nextChanging = changing ?? (targetEdit ? false : weather.changing);
      const start = preset ?? weather.start ?? 'fair';
      weather = { ...weather, target, changing: nextChanging, seed: seed ?? weather.seed, start, cyclePreset: nextChanging ? start : null };
      weatherGeneration++; reason('Weather'); sourceChanged();
    }, setEnvironmentLighting: v => calls.push(['setEnvironmentLighting', v]),
    setTimeOfDay: v => {
      calls.push(['setTimeOfDay', v]);
      const next = { ...time, ...v };
      validate(next);
      const moved = next.day !== time.day || next.hours !== time.hours;
      time = next;
      if (moved) cursor = { currentDay: time.day, currentHours: time.hours };
    },
    // Test-only: move the playback cursor away from the authored moment.
    moveCursor: (currentDay, currentHours) => { cursor = { currentDay, currentHours }; },
    setCamera: v => { calls.push(['setCamera', v]); if (cameraSet) reason('CameraDiscontinuity'); },
    setFog: v => calls.push(['setFog', v]),
    setPassEnabled: (name, enabled) => {
      calls.push(['setPassEnabled', name, enabled]);
      if (!passes.has(name)) throw Object.assign(new Error(`unknown pass ${name}`), { code: 'pipeline.unknown_pass' });
      if (name === 'sky.env' && !enabled) published = null;
      if (name === 'sky.env' && enabled && !passes.get(name)) sourceChanged();
      if (name === 'sky.cloud-lighting' && enabled && !passes.get(name)) reason('Enable');
      passes.set(name, enabled);
    },
    passes: () => [...passes].map(([name, enabled], order) => ({ name, role: name === 'sky.background' ? 'contributor' : 'pass', enabled, order, required: name === 'forward' })),
    setViewports: r => { calls.push(['setViewports', r]); rects = r.length; },
    setViewportCamera: (index, camera) => calls.push(['setViewportCamera', index, camera]),
    resize: (w, h) => { calls.push(['resize', w, h]); size = [w, h]; }, primitiveMesh: () => 1, addMaterial: () => 1, addNode: () => 1,
    addInstance: () => calls.push(['addInstance']),
    render: () => {
      renders++; cameraSet = true;
      const width = size[0] / rects;
      slots = Array.from({ length: rects }, (_, view) => {
        const slot = slots[view] ?? { view, width, height: size[1], reasons: [] };
        if (slot.width !== width || slot.height !== size[1]) slot.reasons.includes('Resize') || slot.reasons.push('Resize');
        return { ...slot, width, height: size[1] };
      });
      if (renders >= readyAfter && passes.get('sky.env')) published = requested;
    },
    sky: () => (failGetter === 'sky' ? null : structuredClone(sky)),
    timeOfDay: () => state(),
    weather: () => { const { start, ...w } = structuredClone(weather); return { ...w, current: structuredClone(weather.target), text: '' }; },
    skyWeatherDiagnostics: () => {
      const ready = renders >= readyAfter;
      return { invocation: renders || null, submittedInvocation: ready ? renders : null, weatherGeneration, atmosphereGeneration: ready ? 1 : 0,
        environment: { requestedGeneration: requested, publishedGeneration: published, pending: passes.get('sky.env') && published !== requested },
        views: slots.map(slot => ({ view: slot.view, width: slot.width, height: slot.height, invocation: renders, submittedInvocation: ready ? renders : null,
          skyViewGeneration: 1, historyValid: true, resetReasons: [...slot.reasons], cloudScreenValid: false, cloudPanoramaValid: false })),
        readiness: ready ? { status: 'ready', stage: 'submission', error: null } : { status: 'pending', stage: 'milkyWay', error: null } };
    },
  };
  return engine;
}
const runtime = { passes: [{ name: 'sky.background', role: 'contributor', enabled: true, order: 900, required: false }],
  viewSizes: [{ view: 0, width: 1280, height: 720 }], readiness: { status: 'ready', stage: 'atmosphere', error: null } };

async function run(mode, options) {
  const variant = scene.variants.find(v => v.id === mode);
  const engine = fakeEngine(options);
  await scene.build(engine, {}, variant);
  return { engine, variant, observation: await atmosphere.observe(engine, variant) };
}

test('atmosphere scenes resolve cumulatively with exact selectors and framed subjects', () => {
  const rows = validateInventory([foundation, atmosphere], 'atmosphere');
  assert.deepEqual(rows.filter(r => r.scenario === 'atmosphere').map(r => r.filename), filenames);
  assert.deepEqual(scene.variants.map(v => v.id), modes);
  assert.deepEqual(atmosphere.inventory.filter(r => r.pair).map(r => [r.mode, r.pair]), [['day', 'sky-model'], ['legacy', 'sky-model'],
    ['fog-on', 'fog'], ['fog-off', 'fog'], ['ibl-on', 'ibl'], ['ibl-off', 'ibl'], ['env-off', 'environment'], ['background-off', 'sky-background'],
    ['camera-cut', 'sky-background'], ['re-enable', 'environment']]);
  assert.deepEqual(scene.variants.filter(v => v.measurement && v.measurement.pair === 'sky-model').map(v => [v.id, v.measurement.state, v.measurement.features.model]), [['day', 'on', 'physical'], ['legacy', 'off', 'legacy']]);
  for (const row of atmosphere.inventory) {
    for (const fact of row.requiredFacts) assert.match(fact, /^(snapshot:(sky|time|weather|diagnostics)|runtime:(passes|viewSizes|ready)|assertion:[a-z0-9-]+)$/);
    for (const fact of row.requiredFacts.filter(f => f.startsWith('assertion:'))) assert.ok(atmosphere.factNames.includes(fact.slice(10)));
  }
  assert.throws(() => validateInventory([atmosphere], 'weather'), /unavailable/);
  for (const v of scene.variants) for (const name of ['ground', 'matte-reference', 'reflective-sphere', 'distant-centre']) assert.ok(v.selection.subjects.some(s => s.name === name));
  const moved = { ...atmosphere, default: { ...scene, variants: scene.variants.map(v => ({ ...v, selection: { ...v.selection, camera: { ...v.selection.camera, target: [0, 1, 30], eye: [0, 1, 20] } } })) } };
  moved.default.variants[0].selection.subjects = [{ name: 'behind', min: [0, 0, -10], max: [1, 1, -9] }];
  assert.throws(() => validateInventory([moved], 'atmosphere'), /frustum/);
});

test('build delivers public sky, weather and time calls with automatic paused lighting', async () => {
  for (const mode of modes) {
    const { engine, variant } = await run(mode);
    const names = engine.calls.map(c => c[0]);
    for (const name of ['setSky', 'setTimeOfDay', 'setCamera', 'addInstance']) assert.ok(names.includes(name), `${mode} ${name}`);
    assert.equal(names.includes('setWeather'), mode !== 'snow', `${mode} setWeather (snow keeps its preset weather)`);
    assert.equal(engine.calls.find(c => c[0] === 'setSky')[1].model, 'physical');
    assert.equal(engine.calls.find(c => c[0] === 'setWeather')?.[1].preset ?? 'overcast', { snow: 'overcast', jungle: 'broken' }[mode] ?? 'clear');
    // The invalid mode's observe adds a refused hours-only edit; the authored moment is the last full one.
    const last = engine.calls.filter(c => c[0] === 'setTimeOfDay' && Object.hasOwn(c[1], 'enabled')).at(-1)[1];
    assert.equal(last.enabled, true); assert.equal(last.playing, false);
    assert.deepEqual(engine.calls.find(c => c[0] === 'setCamera')[1].position ?? engine.calls.find(c => c[0] === 'setCamera')[1].eye, variant.selection.camera.eye);
  }
});

test('observe waits for publication and satisfies every row from actual getters', async () => {
  for (const row of atmosphere.inventory) {
    const { observation } = await run(row.mode);
    validateRequiredFacts(row, { ...observation, ...runtime }, atmosphere.factNames);
  }
  const { observation } = await run('later');
  const later = observation.edits.find(e => e.operation.startsWith('setTimeOfDay'));
  assert.equal(later.before.time.day, 13); assert.equal(later.after.time.day, 14);
});

test('missing, null and false required facts reject', async () => {
  const row = atmosphere.inventory[2];
  const { observation } = await run('night');
  const base = () => ({ ...structuredClone(observation), ...structuredClone(runtime) });
  for (const mutate of [
    f => { f.time = null; }, f => { f.diagnostics = null; }, f => { f.weather = null; },
    f => { f.assertions = f.assertions.filter(a => a.name !== 'moon-in-view'); },
    f => { f.assertions.find(a => a.name === 'sun-below-night').passed = false; },
    f => { f.assertions.find(a => a.name === 'moon-phase-partial').passed = null; },
  ]) { const f = base(); mutate(f); assert.throws(() => validateRequiredFacts(row, f, atmosphere.factNames), /night-moon-stars-milky-way/); }
  const stalled = await run('day', { readyAfter: Infinity });
  assert.throws(() => validateRequiredFacts(atmosphere.inventory[0], { ...stalled.observation, ...runtime }, atmosphere.factNames), /environment-published|atmosphere-submitted|view-submitted/);
});

test('controls are exactly the seven time/place/playback controls', () => {
  assert.deepEqual(scene.controls.map(c => c.id), ['day', 'hours', 'latitude', 'north', 'enabled', 'playing', 'speed']);
  for (const c of scene.controls) {
    assert.notEqual(c.kind, 'button');
    assert.ok(!/restart/i.test(c.id) && !/restart/i.test(c.label), c.id);
  }
});

const newValues = { day: 200, hours: 17.5, latitude: -12, north: 0.7, enabled: true, playing: true, speed: 60 };
const control = id => scene.controls.find(c => c.id === id);

test('every control applies exactly one setTimeOfDay patch and reads the accepted value', () => {
  for (const [key, value] of Object.entries(newValues)) {
    const engine = fakeEngine();
    control(key).apply(engine, value);
    const calls = engine.calls.filter(c => c[0] === 'setTimeOfDay');
    assert.equal(calls.length, 1, key);
    assert.deepEqual(calls[0][1], { [key]: value });
    assert.equal(control(key).read(engine), value);
    assert.equal(engine.timeOfDay()[key], value);
  }
});

test('changed day or hours moves the moved cursor to the merged authored moment', () => {
  for (const key of ['day', 'hours']) {
    const engine = fakeEngine();
    engine.moveCursor(3, 4.25);
    control(key).apply(engine, newValues[key]);
    const t = engine.timeOfDay();
    assert.deepEqual([t.currentDay, t.currentHours], [t.day, t.hours], key);
    assert.equal(t[key], newValues[key]);
  }
});

test('place and playback edits keep the moved cursor', () => {
  for (const key of ['latitude', 'north', 'enabled', 'playing', 'speed']) {
    const engine = fakeEngine();
    engine.moveCursor(3, 4.25);
    control(key).apply(engine, newValues[key]);
    const t = engine.timeOfDay();
    assert.deepEqual([t.currentDay, t.currentHours], [3, 4.25], key);
  }
});

test('out-of-domain edits throw sky.invalid and keep the last accepted state', () => {
  const bad = { day: 366, hours: 24, latitude: 91, speed: -1 };
  for (const [key, value] of Object.entries(bad)) {
    const engine = fakeEngine();
    control(key).apply(engine, newValues[key]);
    const before = engine.timeOfDay();
    assert.throws(() => control(key).apply(engine, value), e => e.code === 'sky.invalid', key);
    assert.equal(control(key).read(engine), newValues[key]);
    assert.deepEqual(engine.timeOfDay(), before);
  }
});

test('night and twilight captures author the dark-adaptation exposure; day stays at 0 EV', () => {
  assert.equal(atmosphere.DARK_ADAPTATION_EV, 7);
  const ev = Object.fromEntries(scene.variants.map(v => [v.id, { ...scene.engine, ...v.engine }.exposureEv]));
  assert.deepEqual(ev, { day: 0, twilight: 7, night: 7, later: 7, snow: 0, jungle: 0, legacy: 0, ...Object.fromEntries(modes.slice(7).map(m => [m, 0])) });
  for (const v of scene.variants) {
    const options = { ...scene.engine, ...v.engine };
    assert.equal(options.toneMapper, 'agx');
    assert.equal(options.msaa, 4);
  }
});

test('physical presets map profile, weather, preset hour and atmosphere; day/night captures reuse them', async () => {
  for (const [mode, preset] of [['day', 'day'], ['night', 'night'], ['snow', 'snow'], ['jungle', 'jungle']]) {
    const { observation } = await run(mode);
    const edit = observation.edits.find(e => e.operation.startsWith('setSky({preset'));
    assert.equal(edit.after.sky.profile, preset);
    assert.equal(edit.after.time.hours, atmosphere.physicalPresets[preset].hours);
    assert.equal(edit.after.weather.target.coverage, atmosphere.physicalPresets[preset].coverage);
    assert.equal(observation.assertions.find(a => a.name === 'physical-preset-mapping').passed, true, mode);
  }
});

test('explicit preset resets the moment while an omitted preset retains it', async () => {
  const { observation } = await run('snow');
  const omitted = observation.edits.find(e => e.operation.startsWith('setSky({})'));
  const explicit = observation.edits.find(e => e.operation === "setSky({preset:'snow'})");
  assert.deepEqual([omitted.before.time.hours, omitted.after.time.hours, explicit.after.time.hours], [13, 13, 12]);
  assert.deepEqual(explicit.after.sky, omitted.after.sky);
  for (const name of ['omitted-preset-retains-moment', 'explicit-preset-resets-moment']) assert.equal(observation.assertions.find(a => a.name === name).passed, true, name);
});

test('named weather preset plus overrides keeps the remaining preset values', async () => {
  const { observation } = await run('jungle');
  const t = observation.weather.target;
  assert.deepEqual([t.coverage, t.wind.speed, t.wind.from, t.base, t.thickness], [.7, 12, 240, 1000, 2600]);
  assert.equal(observation.assertions.find(a => a.name === 'weather-preset-overrides').passed, true);
});

test('legacy presets and model switches retain authored weather and moment', async () => {
  const { observation } = await run('legacy');
  assert.deepEqual(observation.edits.filter(e => e.operation.includes("model:'legacy'") && e.operation.includes('preset')).map(e => e.after.sky.profile), ['snow', 'jungle', 'night', 'day']);
  assert.equal(observation.sky.model, 'legacy');
  assert.equal(observation.weather.target.cirrus, .2);
  for (const name of ['legacy-preset-mappings', 'weather-retained-across-model-switches', 'sky-model-legacy']) assert.equal(observation.assertions.find(a => a.name === name).passed, true, name);
});

test('a lost weather target across a model switch fails the retention assertion', async () => {
  const variant = scene.variants.find(v => v.id === 'legacy');
  const engine = fakeEngine();
  const setSky = engine.setSky;
  engine.setSky = v => { setSky(v); if (v.model === 'physical' && !v.preset) engine.setWeather({ preset: 'fair' }); };
  await scene.build(engine, {}, variant);
  const observation = await atmosphere.observe(engine, variant);
  assert.equal(observation.assertions.find(a => a.name === 'weather-retained-across-model-switches').passed, false);
  const row = atmosphere.inventory.find(r => r.mode === 'legacy');
  assert.throws(() => validateRequiredFacts(row, { ...observation, ...runtime }, atmosphere.factNames), /legacy-comparison/);
});

// Synthetic records fed straight to the selectors: expectations must come from the request.
function presetRecord(request, mutate = () => {}) {
  const engine = fakeEngine();
  engine.setTimeOfDay({ latitude: 38 });
  const before = { sky: engine.sky(), time: engine.timeOfDay(), weather: engine.weather() };
  engine.setSky(request);
  const after = { sky: engine.sky(), time: engine.timeOfDay(), weather: engine.weather() };
  mutate(after);
  return { tag: request.model === 'legacy' ? 'legacy-preset' : 'preset', request, edit: { operation: 'synthetic', accepted: true, error: null, before, after } };
}
const verdict = (records, name) => {
  const last = records[records.length - 1].edit.after;
  return atmosphere.editAssertions(records, last).find(a => a.name === name).passed;
};
const resolvedAs = preset => presetRecord({ preset, model: 'physical' }).edit.after;

test('preset mapping selector passes the faithful snow and jungle mappings', () => {
  for (const preset of ['snow', 'jungle', 'day', 'night']) assert.equal(verdict([presetRecord({ preset, model: 'physical' })], 'physical-preset-mapping'), true, preset);
});

test('requested snow resolved to a self-consistent jungle profile fails', () => {
  const jungle = resolvedAs('jungle');
  assert.equal(verdict([presetRecord({ preset: 'snow', model: 'physical' }, a => Object.assign(a, jungle))], 'physical-preset-mapping'), false);
});

test('requested snow with a wrong albedo or mie fails', () => {
  assert.equal(verdict([presetRecord({ preset: 'snow', model: 'physical' }, a => { a.sky.atmosphere.groundAlbedo = [.3, .3, .3]; })], 'physical-preset-mapping'), false);
  assert.equal(verdict([presetRecord({ preset: 'snow', model: 'physical' }, a => { a.sky.atmosphere.mieMultiplier = 1; })], 'physical-preset-mapping'), false);
});

test('requested jungle with wrong hours or weather fails', () => {
  assert.equal(verdict([presetRecord({ preset: 'jungle', model: 'physical' }, a => { a.time.hours = 9; a.time.currentHours = 9; })], 'physical-preset-mapping'), false);
  assert.equal(verdict([presetRecord({ preset: 'jungle', model: 'physical' }, a => { a.weather.target = structuredClone(resolvedAs('snow').weather.target); })], 'physical-preset-mapping'), false);
  assert.equal(verdict([presetRecord({ preset: 'jungle', model: 'physical' }, a => { a.weather.target.type = .5; })], 'physical-preset-mapping'), false);
});

test('a legacy preset whose observed model is physical, or whose profile differs from the request, fails', () => {
  const order = ['snow', 'jungle', 'night', 'day'];
  const good = order.map(p => presetRecord({ preset: p, model: 'legacy' }));
  assert.equal(verdict(good, 'legacy-preset-mappings'), true);
  const physical = order.map(p => presetRecord({ preset: p, model: 'legacy' }, a => { if (p === 'night') a.sky.model = 'physical'; }));
  assert.equal(verdict(physical, 'legacy-preset-mappings'), false);
  const swapped = order.map(p => presetRecord({ preset: p, model: 'legacy' }, a => { if (p === 'jungle') a.sky.profile = 'snow'; }));
  assert.equal(verdict(swapped, 'legacy-preset-mappings'), false);
});

test('weather overrides, omitted/explicit presets and model switches check requests, not observations', async () => {
  const { engine } = await run('jungle');
  const w = { tag: 'weather', request: { preset: 'broken', coverage: .7, wind: { speed: 12 } } };
  const weatherAfter = engine.weather();
  const edit = after => ({ operation: 'synthetic', accepted: true, error: null, before: { sky: engine.sky(), time: engine.timeOfDay(), weather: weatherAfter }, after });
  const wrong = { ...weatherAfter, target: { ...weatherAfter.target, base: 600 } };
  const cur = weather => ({ sky: engine.sky(), time: engine.timeOfDay(), weather });
  assert.equal(atmosphere.editAssertions([{ ...w, edit: edit(cur(weatherAfter)) }], cur(weatherAfter)).find(a => a.name === 'weather-preset-overrides').passed, true);
  assert.equal(atmosphere.editAssertions([{ ...w, edit: edit(cur(wrong)) }], cur(wrong)).find(a => a.name === 'weather-preset-overrides').passed, false);
  const snow = await run('snow');
  const recs = name => snow.observation.edits.filter(e => e.operation.startsWith(name));
  const [omitted] = recs('setSky({})'), [explicit] = recs("setSky({preset:'snow'})");
  const cs = { sky: snow.observation.sky, time: snow.observation.time, weather: snow.observation.weather };
  const pick = (records, name) => atmosphere.editAssertions(records, cs).find(a => a.name === name).passed;
  assert.equal(pick([{ tag: 'omitted', request: {}, edit: omitted }], 'omitted-preset-retains-moment'), true);
  assert.equal(pick([{ tag: 'omitted', request: { preset: 'snow' }, edit: omitted }], 'omitted-preset-retains-moment'), false);
  const ex = [{ tag: 'omitted', request: {}, edit: omitted }];
  assert.equal(pick([...ex, { tag: 'explicit', request: { preset: 'snow' }, edit: explicit }], 'explicit-preset-resets-moment'), true);
  assert.equal(pick([...ex, { tag: 'explicit', request: { preset: 'jungle' }, edit: explicit }], 'explicit-preset-resets-moment'), false);
  const legacy = await run('legacy');
  const switches = legacy.observation.edits.filter(e => /^setSky\(\{model:/.test(e.operation));
  const lc = { sky: legacy.observation.sky, time: legacy.observation.time, weather: legacy.observation.weather };
  const sw = models => switches.map((e, i) => ({ tag: 'model', request: { model: models[i] }, edit: e }));
  assert.equal(atmosphere.editAssertions(sw(['legacy', 'physical', 'legacy']), lc).find(a => a.name === 'weather-retained-across-model-switches').passed, true);
  assert.equal(atmosphere.editAssertions(sw(['legacy', 'legacy', 'legacy']), lc).find(a => a.name === 'weather-retained-across-model-switches').passed, false);
});

test('snow expectation describes only what the sky stage controls', async () => {
  const { inventory } = await import('./scenes/atmosphere.js');
  const text = inventory.find(r => r.mode === 'snow').expectedVisibleBehavior;
  for (const word of ['ground albedo', 'neutral ground']) assert.ok(!text.includes(word), word);
  assert.match(text, /pale whitish-blue hazy sky/);
  assert.match(text, /milky horizon/);
});

// ---- Controls, atomic edits, multi-view and transitions ----
const rowOf = mode => atmosphere.inventory.find(r => r.mode === mode);
const factsOf = observation => ({ ...structuredClone(observation), ...structuredClone(runtime) });
async function runWith(mode, wrap, options) {
  const variant = scene.variants.find(v => v.id === mode);
  const engine = fakeEngine(options);
  wrap?.(engine);
  await scene.build(engine, {}, variant);
  return { engine, variant, observation: await atmosphere.observe(engine, variant) };
}
const named = (observation, name) => observation.assertions.filter(a => a.name === name);

test('inventory covers every controls, error, view and transition criterion with explicit filenames and pairs', () => {
  const criteria = { 'fog-on': 'fog-on', 'fog-off': 'fog-off', 'ibl-on': 'ibl-on', 'ibl-off': 'ibl-off', 'ibl-diffuse': 'ibl-diffuse-only',
    'ibl-specular': 'ibl-specular-only', 'env-off': 'environment-disabled', 'background-off': 'background-disabled', invalid: 'invalid-edits',
    'multi-view': 'multi-view-same-clock', resize: 'transition-resize', 'camera-cut': 'transition-camera-cut',
    'model-switch': 'transition-model-switch', 'weather-change': 'transition-weather-change', 're-enable': 'transition-re-enable' };
  for (const [mode, id] of Object.entries(criteria)) assert.equal(rowOf(mode)?.id, id, mode);
  for (const mode of atmosphere.TRANSITION_MODES) assert.match(rowOf(mode).filename, /^transition-[a-z0-9-]+\.png$/);
  assert.match(rowOf('invalid').filename, /^error-invalid-/);
  // Every pair is exactly one on and one off variant of the same preset, each with an inventory row.
  const pairs = Map.groupBy(scene.variants.filter(v => v.measurement), v => v.measurement.pair);
  assert.deepEqual([...pairs.keys()].sort(), ['environment', 'fog', 'ibl', 'sky-background', 'sky-model']);
  for (const [pair, variants] of pairs) {
    assert.deepEqual(variants.map(v => v.measurement.state).sort(), ['off', 'on'], pair);
    for (const v of variants) assert.equal(rowOf(v.id).pair, pair);
  }
  // Selector registry is exact: unique names, every name required by some row, every selector registered.
  assert.equal(new Set(atmosphere.factNames).size, atmosphere.factNames.length);
  const used = new Set(atmosphere.inventory.flatMap(r => r.requiredFacts).filter(f => f.startsWith('assertion:')).map(f => f.slice(10)));
  assert.deepEqual([...used].sort(), [...atmosphere.factNames].sort());
  assert.ok(!rowOf('env-off').requiredFacts.includes('assertion:environment-published'));
  const unknown = { ...rowOf('fog-on'), requiredFacts: ['assertion:fog-pass-unknown'] };
  assert.throws(() => validateInventory([{ ...atmosphere, inventory: [unknown] }], 'atmosphere'), /unknown selector/);
  assert.throws(() => validateRequiredFacts(unknown, factsOf({ assertions: [], edits: [] }), atmosphere.factNames), /unknown selector/);
});

test('second view and pre-cut cameras overlap every framed subject', () => {
  for (const camera of [atmosphere.secondViewCamera, atmosphere.preCutCamera]) {
    const aspect = camera === atmosphere.secondViewCamera ? atmosphere.MULTI_VIEW_ASPECT : 16 / 9;
    const variants = scene.variants.map(v => ({ ...v, selection: { ...v.selection, camera: { ...camera, aspect } } }));
    assert.doesNotThrow(() => validateInventory([{ ...atmosphere, default: { ...scene, variants } }], 'atmosphere'));
  }
  assert.ok(Math.abs(scene.variants.find(v => v.id === 'multi-view').selection.camera.aspect - 640 / 720) < 1e-12);
});

test('feature controls apply fog, independent IBL, environment disable and flat background through public calls', async () => {
  for (const mode of atmosphere.FEATURE_MODES) {
    const { engine, observation } = await runWith(mode);
    validateRequiredFacts(rowOf(mode), factsOf(observation), atmosphere.factNames);
    const pass = name => engine.passes().find(p => p.name === name).enabled;
    if (mode.startsWith('fog')) {
      assert.deepEqual(engine.calls.find(c => c[0] === 'setFog')[1], atmosphere.FOG_SETTINGS);
      assert.equal(pass('fog'), mode === 'fog-on');
    }
    if (mode.startsWith('ibl')) assert.deepEqual(engine.calls.filter(c => c[0] === 'setEnvironmentLighting').at(-1)[1], atmosphere.IBL_SETTINGS[mode]);
    assert.equal(pass('sky.env'), mode !== 'env-off');
    assert.equal(pass('sky.background'), mode !== 'background-off');
  }
  const { engine, observation } = await runWith('env-off');
  assert.equal(observation.diagnostics.environment.publishedGeneration, null);
  assert.deepEqual(engine.calls.find(c => c[0] === 'setPassEnabled').slice(1), ['sky.env', false]);
  const flat = (await runWith('background-off')).engine.calls.filter(c => c[0] === 'setBackground').at(-1)[1];
  assert.deepEqual(flat, atmosphere.FLAT_BACKGROUND);
});

test('invalid edits reject atomically, keep accepted and rendered state and display their actual errors', async () => {
  const status = { textContent: '' };
  const previous = globalThis.document;
  globalThis.document = { getElementById: id => (id === 'status' ? status : null) };
  try {
    const { engine, variant, observation } = await runWith('invalid');
    validateRequiredFacts(rowOf('invalid'), factsOf(observation), atmosphere.factNames);
    const refused = observation.edits.filter(e => !e.accepted);
    assert.equal(refused.length, atmosphere.INVALID_EDITS.length);
    for (const e of refused) { assert.equal(e.error.code, 'sky.invalid'); assert.deepEqual(e.after, e.before); }
    status.textContent = 'ready: atmosphere/invalid';
    scene.frame(engine, 1000, variant);
    scene.frame(engine, 1000, variant);
    const lines = status.textContent.split('\n');
    assert.equal(lines.length, 2 + atmosphere.INVALID_EDITS.length, 'appended exactly once');
    for (const e of refused) assert.ok(status.textContent.includes(e.error.message), e.operation);
    status.textContent = 'pending: preparing';
    scene.frame(engine, 1000, variant);
    assert.equal(status.textContent, 'pending: preparing');
  } finally { globalThis.document = previous; }
  // An engine that accepts and applies an invalid coverage fails rejection and preservation.
  const lax = await runWith('invalid', engine => {
    const setWeather = engine.setWeather;
    engine.setWeather = v => (v.coverage === 2 ? setWeather({ coverage: .9 }) : setWeather(v));
  });
  assert.equal(named(lax.observation, 'invalid-edits-rejected')[0].passed, false);
  assert.equal(named(lax.observation, 'invalid-edits-preserve-state')[0].passed, false);
  assert.equal(named(lax.observation, 'invalid-edits-preserve-rendering')[0].passed, false);
  assert.throws(() => validateRequiredFacts(rowOf('invalid'), factsOf(lax.observation), atmosphere.factNames), /invalid-edits/);
});

test('multi-view observes two submitted views from one clock snapshot', async () => {
  const { engine, observation } = await runWith('multi-view');
  validateRequiredFacts(rowOf('multi-view'), factsOf(observation), atmosphere.factNames);
  assert.deepEqual(engine.calls.find(c => c[0] === 'setViewports')[1], atmosphere.MULTI_VIEW_RECTS);
  assert.deepEqual(engine.calls.find(c => c[0] === 'setViewportCamera').slice(1), [1, atmosphere.secondViewCamera]);
  assert.deepEqual(observation.diagnostics.views.map(v => [v.view, v.width, v.height]), [[0, 640, 720], [1, 640, 720]]);
  const skewed = await runWith('multi-view', engine => {
    const read = engine.skyWeatherDiagnostics;
    engine.skyWeatherDiagnostics = () => { const d = read(); if (d.views[1]) d.views[1].invocation = (d.invocation ?? 0) - 1; return d; };
  });
  assert.equal(named(skewed.observation, 'views-same-clock')[0].passed, false);
});

test('transitions record actual reset reasons, restored sizes, cycle start/restart and republication', async () => {
  for (const mode of atmosphere.TRANSITION_MODES) {
    const { observation } = await runWith(mode);
    validateRequiredFacts(rowOf(mode), factsOf(observation), atmosphere.factNames);
  }
  const weather = (await runWith('weather-change')).observation.weather;
  assert.deepEqual([weather.cyclePreset, weather.changing, weather.seed, weather.target.coverage], ['broken', true, 10, .7]);
  const resize = (await runWith('resize')).engine.calls.filter(c => c[0] === 'resize').map(c => c.slice(1));
  assert.deepEqual(resize, [atmosphere.RESIZE_TO, [1280, 720]]);
  // Reasons that never appear, or appear already before the action, fail.
  for (const mode of atmosphere.TRANSITION_MODES) {
    const silent = await runWith(mode, engine => {
      const read = engine.skyWeatherDiagnostics;
      engine.skyWeatherDiagnostics = () => { const d = read(); for (const v of d.views) v.resetReasons = []; return d; };
    });
    const name = rowOf(mode).requiredFacts.find(f => f.startsWith('assertion:reset-')).slice(10);
    assert.equal(named(silent.observation, name)[0].passed, false, mode);
    assert.throws(() => validateRequiredFacts(rowOf(mode), factsOf(silent.observation), atmosphere.factNames), new RegExp(name));
  }
});

test('neutral backend volume flags stay truthful and new selectors reject missing, null and false', async () => {
  const lying = await runWith('resize', engine => {
    const read = engine.skyWeatherDiagnostics;
    engine.skyWeatherDiagnostics = () => { const d = read(); for (const v of d.views) v.cloudScreenValid = true; return d; };
  });
  assert.equal(named(lying.observation, 'neutral-cloud-flags')[0].passed, false);
  for (const mode of ['fog-off', 'env-off', 'multi-view', 'weather-change', 're-enable']) {
    const row = rowOf(mode), { observation } = await runWith(mode);
    for (const name of row.requiredFacts.filter(f => f.startsWith('assertion:')).map(f => f.slice(10)).slice(-2)) {
      for (const mutate of [f => { f.assertions = f.assertions.filter(a => a.name !== name); },
        f => { f.assertions.find(a => a.name === name).passed = null; },
        f => { f.assertions.find(a => a.name === name).passed = false; },
        f => { f.assertions.push({ ...f.assertions.find(a => a.name === name) }); }]) {
        const facts = factsOf(observation); mutate(facts);
        assert.throws(() => validateRequiredFacts(row, facts, atmosphere.factNames), new RegExp(row.id), `${mode} ${name}`);
      }
    }
    const facts = factsOf(observation); facts.diagnostics = null;
    assert.throws(() => validateRequiredFacts(row, facts, atmosphere.factNames), /snapshot:diagnostics/);
  }
});
