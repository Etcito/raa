import test from 'node:test';
import assert from 'node:assert/strict';
import { validateInventory, validateRequiredFacts } from './lib/inventory.js';
import * as foundation from './scenes/foundation.js';
import * as atmosphere from './scenes/atmosphere.js';

const scene = atmosphere.default;
const filenames = ['clear-day-warm-horizon-and-reflective-sphere.png', 'twilight-warm-horizon-distant-forms.png',
  'night-moon-phase-stars-and-milky-way.png', 'night-later-celestial-positions.png'];

// Public-API fake: records delivered calls and evaluates the authored moments with the
// source celestial formulas only to supply plausible getter values for the observer.
function fakeEngine({ readyAfter = 3, failGetter = null } = {}) {
  const calls = [];
  let time = { day: 172, hours: 9, latitude: 38, north: 0, enabled: false, playing: false, speed: 1 };
  let cursor = { currentDay: time.day, currentHours: time.hours };
  let renders = 0;
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
  const ws = { coverage: 0, type: .9, base: 1200, thickness: 1600, density: 1, cirrus: .05, wind: { speed: 5, from: 250 }, evolution: 1, rain: 0 };
  const engine = {
    calls,
    setBackground: v => calls.push(['setBackground', v]), setLight: v => calls.push(['setLight', v]),
    setAmbient: v => calls.push(['setAmbient', v]), setSky: v => calls.push(['setSky', v]),
    setWeather: v => calls.push(['setWeather', v]), setEnvironmentLighting: v => calls.push(['setEnvironmentLighting', v]),
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
    setCamera: v => calls.push(['setCamera', v]), primitiveMesh: () => 1, addMaterial: () => 1, addNode: () => 1,
    addInstance: () => calls.push(['addInstance']),
    render: () => { renders++; },
    sky: () => (failGetter === 'sky' ? null : { model: 'physical', profile: 'day', intensity: 1, zenith: [0, .2, 1], horizon: [1, .8, .6], ground: [.3, .3, .3] }),
    timeOfDay: () => state(),
    weather: () => ({ enabled: true, shadows: true, changing: false, seed: 1, transitionSeconds: 0, target: ws, current: ws, cyclePreset: null, text: '' }),
    skyWeatherDiagnostics: () => {
      const ready = renders >= readyAfter;
      return { invocation: renders || null, submittedInvocation: ready ? renders : null, weatherGeneration: 1, atmosphereGeneration: ready ? 1 : 0,
        environment: { requestedGeneration: 2, publishedGeneration: ready ? 2 : null, pending: !ready },
        views: [{ view: 0, width: 1280, height: 720, invocation: renders, submittedInvocation: ready ? renders : null, skyViewGeneration: 1, historyValid: true, resetReasons: [], cloudScreenValid: false, cloudPanoramaValid: false }],
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
  assert.deepEqual(scene.variants.map(v => v.id), ['day', 'twilight', 'night', 'later']);
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
  for (const mode of ['day', 'twilight', 'night', 'later']) {
    const { engine, variant } = await run(mode);
    const names = engine.calls.map(c => c[0]);
    for (const name of ['setSky', 'setWeather', 'setTimeOfDay', 'setCamera', 'addInstance']) assert.ok(names.includes(name), `${mode} ${name}`);
    assert.equal(engine.calls.find(c => c[0] === 'setSky')[1].model, 'physical');
    assert.deepEqual(engine.calls.find(c => c[0] === 'setWeather')[1], { preset: 'clear' });
    const last = engine.calls.filter(c => c[0] === 'setTimeOfDay').at(-1)[1];
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
  assert.equal(observation.edits.length, 1);
  assert.equal(observation.edits[0].before.time.day, 13); assert.equal(observation.edits[0].after.time.day, 14);
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
  assert.deepEqual(ev, { day: 0, twilight: 7, night: 7, later: 7 });
  for (const v of scene.variants) {
    const options = { ...scene.engine, ...v.engine };
    assert.equal(options.toneMapper, 'agx');
    assert.equal(options.msaa, 4);
  }
});
