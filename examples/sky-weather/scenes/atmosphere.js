import { buildSkySample, sampleSelection } from '../lib/scene.js';

// Physical atmosphere and celestial time through the delivered public m08 API only.
// Every observation below is read back from Engine getters; nothing is inferred from
// the authored constants except the comparisons the assertions name.

const dayCamera = { eye: [0, 2.6, 9], target: [0, 1.9, -8], fovYDeg: 47, near: .1, far: 300 };
// Night cameras tilt up so the moon (placed ahead by the authored north rotation) is framed.
const nightCamera = { eye: [0, 2.2, 9], target: [0, 12, -30], fovYDeg: 60, near: .1, far: 300 };

// Authored moments (latitude 38 N). north rotates the celestial sphere about +Y so the
// observed bodies lie toward the camera's -Z view; the runtime asserts it from getters.
const moments = {
  day: { day: 172, hours: 17.5, latitude: 38, north: 0 },
  twilight: { day: 172, hours: 19.9, latitude: 38, north: -.6 },
  night: { day: 13, hours: 22, latitude: 38, north: -1.8 },
  later: { day: 14, hours: 23, latitude: 38, north: -1.8 },
};

const shared = 'A broad neutral ground with a terracotta matte sphere on the left, a chrome sphere on the right '
  + 'and three distant block forms behind them. The status below the view reads ready, with no displayed failure.';
const behaviors = {
  day: 'A clear late-afternoon daytime sky: readable deep blue overhead brightening to a warm pale golden horizon, '
    + 'with no clouds. The chrome sphere clearly reflects the blue sky and the warm horizon band. ' + shared,
  twilight: 'Twilight just after sunset: a darkening blue-violet upper sky above a warm orange glow along the horizon, '
    + 'the distant block forms silhouetted against that glow and the spheres dimly lit. ' + shared,
  night: 'A dark night sky that stays dark overall, with a visibly partially lit moon showing its phase above the horizon, '
    + 'many small stars and a faint band of the Milky Way readable against the dark sky. The ground and spheres are dim, softly moonlit. ' + shared,
  later: 'The same dark night one day and one hour later: the moon is at a different place in the sky and shows a fuller phase, '
    + 'the stars and the faint Milky Way band have rotated to new positions, and the sky remains dark. ' + shared,
};

export const factNames = [
  'atmosphere-submitted', 'environment-published', 'view-submitted', 'time-authored',
  'sun-daylight', 'sun-twilight-band', 'sun-below-night', 'moon-in-view', 'moon-phase-partial',
  'celestial-positions-changed',
];

const common = ['snapshot:sky', 'snapshot:time', 'snapshot:weather', 'snapshot:diagnostics',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready',
  'assertion:atmosphere-submitted', 'assertion:environment-published', 'assertion:view-submitted', 'assertion:time-authored'];
const row = (id, criterionName, mode, filename, extra) => ({
  id, criterionName, scenario: 'atmosphere', mode, filename, expectedVisibleBehavior: behaviors[mode],
  requiredFacts: [...common, ...extra.map(name => `assertion:${name}`)], pair: null,
});

export const inventory = [
  row('clear-day-warm-horizon', 'Clear day with warm horizon', 'day', 'clear-day-warm-horizon-and-reflective-sphere.png', ['sun-daylight']),
  row('twilight-warm-horizon', 'Continuous twilight', 'twilight', 'twilight-warm-horizon-distant-forms.png', ['sun-twilight-band']),
  row('night-moon-stars-milky-way', 'Dark readable night', 'night', 'night-moon-phase-stars-and-milky-way.png', ['sun-below-night', 'moon-in-view', 'moon-phase-partial']),
  row('night-later-celestial-positions', 'Changed celestial time and date', 'later', 'night-later-celestial-positions.png', ['sun-below-night', 'moon-in-view', 'moon-phase-partial', 'celestial-positions-changed']),
];

const snapshot = engine => ({ sky: engine.sky(), time: engine.timeOfDay(), weather: engine.weather() });
// Before/after states of the documented later-time action, keyed by the engine it ran on.
const actions = new WeakMap();

function errorOf(error) {
  return { code: typeof error?.code === 'string' ? error.code : 'error', message: String(error?.message ?? error) };
}

/** Applies the authored moment with automatic celestial lighting, paused for capture. */
function applyMoment(engine, mode) {
  engine.setTimeOfDay({ ...moments[mode], enabled: true, playing: false, speed: 1 });
}

const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = v => { const l = Math.hypot(...v); return v.map(x => x / l); };
const altitudeDeg = v => Math.asin(Math.max(-1, Math.min(1, v[1]))) * 180 / Math.PI;
const angleDeg = (a, b) => Math.acos(Math.max(-1, Math.min(1, dot(unit(a), unit(b))))) * 180 / Math.PI;

/** Whether a direction at infinity lies inside the variant's capture frustum. */
export function directionInView(direction, camera) {
  const forward = unit(sub(camera.target, camera.eye));
  const right = unit(cross(forward, [0, 1, 0]));
  const up = cross(right, forward);
  const d = unit(direction), z = dot(d, forward);
  const vertical = Math.tan(camera.fovYDeg * Math.PI / 360), horizontal = vertical * (camera.aspect ?? 16 / 9);
  return z > 0 && Math.abs(dot(d, right)) <= z * horizontal && Math.abs(dot(d, up)) <= z * vertical;
}

const FRAME_LIMIT = 600;
const yieldTask = () => new Promise(resolve => setTimeout(resolve, 0));
const settled = d => d.readiness.status === 'ready' && d.readiness.error === null && d.atmosphereGeneration > 0
  && d.environment.publishedGeneration !== null && d.environment.publishedGeneration >= d.environment.requestedGeneration
  && d.environment.pending === false && d.views.length > 0 && d.views.every(view => view.submittedInvocation !== null);

/**
 * Renders until the submitted atmosphere/Milky Way (readiness ready) and the environment
 * generation requested by the accepted state are published, or the frame limit passes;
 * the assertions then report the actual diagnostics either way.
 */
async function awaitPublication(engine, variant) {
  let diagnostics = engine.skyWeatherDiagnostics();
  for (let frame = 0; frame < FRAME_LIMIT && !settled(diagnostics); frame++) {
    if (diagnostics.readiness.status === 'failed') break;
    engine.render(variant.timeMs ?? 1000);
    await yieldTask();
    diagnostics = engine.skyWeatherDiagnostics();
  }
  return diagnostics;
}

export async function observe(engine, variant) {
  const mode = variant.id, authored = moments[mode];
  if (!authored) throw new Error(`atmosphere has no mode ${mode}`);
  const diagnostics = await awaitPublication(engine, variant);
  const { sky, time, weather } = snapshot(engine);
  const env = diagnostics.environment, camera = variant.selection.camera;
  const sunAlt = altitudeDeg(time.sunDirection), moonAlt = altitudeDeg(time.moonDirection);
  const assertions = [
    { name: 'atmosphere-submitted', passed: diagnostics.readiness.status === 'ready' && diagnostics.atmosphereGeneration > 0,
      detail: `readiness ${JSON.stringify(diagnostics.readiness)}; atmosphereGeneration ${diagnostics.atmosphereGeneration} (ready includes the Milky Way bake)` },
    { name: 'environment-published', passed: env.publishedGeneration !== null && env.publishedGeneration >= env.requestedGeneration && env.pending === false,
      detail: `environment ${JSON.stringify(env)}` },
    { name: 'view-submitted', passed: diagnostics.views.length > 0 && diagnostics.views.every(view => view.submittedInvocation !== null),
      detail: `submittedInvocation ${diagnostics.submittedInvocation}; views ${JSON.stringify(diagnostics.views.map(v => [v.view, v.submittedInvocation]))}` },
    { name: 'time-authored', passed: ['day', 'hours', 'latitude', 'north'].every(key => Math.abs(time[key] - authored[key]) < 1e-5)
        && time.enabled === true && time.playing === false && time.currentDay === authored.day && Math.abs(time.currentHours - authored.hours) < 1e-5,
      detail: `authored ${JSON.stringify(authored)}; observed day ${time.day} hours ${time.hours} current ${time.currentDay}/${time.currentHours} enabled ${time.enabled} playing ${time.playing}` },
    { name: 'sun-daylight', passed: sunAlt > 10 && time.dominant === 'sun', detail: `sun altitude ${sunAlt.toFixed(2)} deg; dominant ${time.dominant}` },
    { name: 'sun-twilight-band', passed: sunAlt < 0 && sunAlt > -10, detail: `sun altitude ${sunAlt.toFixed(2)} deg; dominant ${time.dominant}` },
    { name: 'sun-below-night', passed: sunAlt < -18 && time.dominant === 'moon', detail: `sun altitude ${sunAlt.toFixed(2)} deg; dominant ${time.dominant}` },
    { name: 'moon-in-view', passed: moonAlt > 0 && directionInView(time.moonDirection, camera),
      detail: `moon direction ${JSON.stringify(time.moonDirection)}, altitude ${moonAlt.toFixed(2)} deg, camera ${JSON.stringify(camera)}` },
    { name: 'moon-phase-partial', passed: time.moonLit > .2 && time.moonLit < .95, detail: `moonLit ${time.moonLit}` },
  ];
  const edits = [];
  const action = actions.get(engine);
  if (mode === 'later' && action) {
    edits.push(action.edit);
    const before = action.edit.before.time, after = action.edit.after.time;
    const moved = before && after ? angleDeg(before.moonDirection, after.moonDirection) : NaN;
    const rotated = before && after ? angleDeg(before.starEquinox, after.starEquinox) : NaN;
    assertions.push({ name: 'celestial-positions-changed',
      passed: action.edit.accepted && after.day !== before.day && after.hours !== before.hours && moved > 1 && rotated > 1
        && after.day === time.day && after.hours === time.hours,
      detail: `day ${before?.day}->${after?.day}, hours ${before?.hours}->${after?.hours}; moon moved ${moved.toFixed?.(2)} deg; star equinox rotated ${rotated.toFixed?.(2)} deg` });
  }
  return { sky, time, weather, diagnostics, assertions, edits };
}

const timeNumber = (id, label, unit, key) => ({ id, label, kind: 'number', unit, choices: [],
  read: engine => engine.timeOfDay()[key], apply: (engine, value) => engine.setTimeOfDay({ [key]: value }) });
const timeFlag = (id, label, key) => ({ id, label, kind: 'checkbox', unit: '', choices: [],
  read: engine => engine.timeOfDay()[key] === true, apply: (engine, value) => engine.setTimeOfDay({ [key]: value }) });

// Time, place and playback controls edit the accepted Engine state; the runtime displays
// a rejected edit's sky.invalid error and the getters keep the last valid settings.
// Editing day or hours to a new value moves the playback cursor to the authored moment.
const controls = [
  timeNumber('day', 'Day of year', 'day 1–365', 'day'),
  timeNumber('hours', 'Local time', 'hours 0–24', 'hours'),
  timeNumber('latitude', 'Latitude', '° north', 'latitude'),
  timeNumber('north', 'North rotation', 'rad about +Y', 'north'),
  timeFlag('enabled', 'Automatic sun/moon lighting', 'enabled'),
  timeFlag('playing', 'Play time (pause when off)', 'playing'),
  timeNumber('speed', 'Playback speed', 'sim s per real s', 'speed'),
];

// Camera exposure per moment. The physical sky keeps absolute radiance: with the daylight
// baseline of intensity 3 below, the night glow (~1e-4), stars and moon (~1e-2) and the
// moonlight (3/2500 x phase) sit under the AgX floor at 0 EV and the frame is black. The
// Solstice source shows them through its automatic exposure, which meters the frame to
// middle grey between a brightest average of 0.06 and a darkest average of 0.06/128 per
// unit of daylight sun (crates/solstice-render/src/post.rs:12-20 at solstice@7e465b5).
// For intensity 3 the brightest clamp is exactly 0 EV (0.18 / (3 x 0.06) = 1), and twilight
// below the civil band and night meter under the darkest clamp, i.e. 128x = +7 EV. RAA has
// no automatic exposure in P2, so each capture authors that fixed equivalent through the
// public tone-mapping option; the sky still renders dark, only readable.
export const DARK_ADAPTATION_EV = Math.log2(128);
export const exposureEv = { day: 0, twilight: DARK_ADAPTATION_EV, night: DARK_ADAPTATION_EV, later: DARK_ADAPTATION_EV };

const variant = (id, camera) => ({ id, description: behaviors[id], references: [], judge: true, timeMs: 1000,
  engine: { exposureEv: exposureEv[id] }, selection: sampleSelection(camera) });

export default {
  id: 'atmosphere', milestone: 'atmosphere', title: 'Physical atmosphere and celestial time',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0 },
  variants: [variant('day', dayCamera), variant('twilight', dayCamera), variant('night', nightCamera), variant('later', nightCamera)],
  controls,
  async build(engine, lib, selected) {
    const mode = selected.id;
    if (!moments[mode]) throw new Error(`atmosphere has no mode ${mode}`);
    const night = mode === 'night' || mode === 'later';
    engine.setBackground([.05, .06, .08]);
    // Manual daylight baseline; automatic celestial control supplies the direction.
    engine.setLight({ direction: [-.35, -.6, -.55], color: [1, .95, .88], intensity: 3 });
    engine.setAmbient({ color: [.55, .62, .75], intensity: night ? .02 : .15 });
    engine.setSky({ preset: night ? 'night' : 'day', model: 'physical' });
    engine.setWeather({ preset: 'clear' });
    engine.setEnvironmentLighting({ diffuse: 1, specular: 1 });
    if (mode === 'later') {
      // Documented action: start from the night moment, then advance date and time.
      applyMoment(engine, 'night');
      const before = snapshot(engine);
      let accepted = true, error = null;
      try { applyMoment(engine, 'later'); } catch (cause) { accepted = false; error = errorOf(cause); }
      actions.set(engine, { edit: { operation: 'setTimeOfDay(day 13 22:00 -> day 14 23:00)', accepted, error, before, after: snapshot(engine) } });
      if (!accepted) throw new Error(`${error.code}: ${error.message}`);
    } else {
      applyMoment(engine, mode);
    }
    await buildSkySample(engine, lib, selected.selection.camera);
  },
};
