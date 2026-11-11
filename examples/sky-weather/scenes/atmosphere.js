import { buildSkySample, sampleSelection } from '../lib/scene.js';

// Physical atmosphere and celestial time through the delivered public m08 API only.
// Every observation below is read back from Engine getters; nothing is inferred from
// the authored constants except the comparisons the assertions name.
// The M2 volumetric backend renders the physical model's clouds for the authored weather (see the
// mode descriptions below). The legacy model is the retained gradient/layer evaluator; it keeps
// the full authored weather but only approximates it as a planar layer, shows no volumetric clouds
// and never claims volumetric parity with the physical model.

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
  // Physical snow/jungle keep their preset moment (day 172 at 12:00 / 15:00).
  snow: { day: 172, hours: 12, latitude: 38, north: 0 },
  jungle: { day: 172, hours: 15, latitude: 38, north: 0 },
  // Same moment as the physical day capture, which is its comparison partner.
  legacy: { day: 172, hours: 17.5, latitude: 38, north: 0 },
};
// Controls, atomic-edit, multi-view and transition modes share the physical day moment.
export const FEATURE_MODES = ['fog-on', 'fog-off', 'ibl-on', 'ibl-off', 'ibl-diffuse', 'ibl-specular', 'env-off', 'background-off'];
export const TRANSITION_MODES = ['resize', 'camera-cut', 'model-switch', 'weather-change', 're-enable'];
const EXTRA_MODES = [...FEATURE_MODES, 'invalid', 'multi-view', ...TRANSITION_MODES];
for (const mode of EXTRA_MODES) moments[mode] = { ...moments.day };

// S24 built-in weather targets (solstice@7e465b5 crates/solstice-render/weather/*.weather).
const fairTarget = { coverage: .3, type: .9, base: 1200, thickness: 1600, density: 1, cirrus: .15, wind: { speed: 7, from: 250 }, evolution: 1, rain: 0 };
export const weatherTargets = {
  fair: fairTarget,
  clear: { ...fairTarget, coverage: 0, cirrus: .05, wind: { speed: 5, from: 250 } },
  broken: { coverage: .62, type: .95, base: 1000, thickness: 2600, density: 1.15, cirrus: .3, wind: { speed: 10, from: 240 }, evolution: 1.3, rain: 0 },
  overcast: { coverage: .97, type: .12, base: 600, thickness: 900, density: 1.3, cirrus: 0, wind: { speed: 8, from: 220 }, evolution: .6, rain: 0 },
};
// S24 physical preset mappings: weather target, preset hour on day 172, ground albedo, Mie.
// Values are compared with the relative tolerance NEAR_TOLERANCE declared here (f32 round trip).
export const NEAR_TOLERANCE = 1e-4;
const mapping = (weather, hours, groundAlbedo, mieMultiplier) => ({ weather, ...weatherTargets[weather], target: weatherTargets[weather], day: 172, hours, groundAlbedo, mieMultiplier });
export const physicalPresets = {
  day: mapping('fair', 9, [.3, .3, .3], 1),
  snow: mapping('overcast', 12, [.9, .9, .9], 1.5),
  jungle: mapping('broken', 15, [.18, .26, .14], 1.2),
  night: mapping('clear', 0, [.3, .3, .3], 1),
};
const OMITTED_HOURS = 13;
const LEGACY_ORDER = ['snow', 'jungle', 'night', 'day'];
const presetOf = { day: 'day', twilight: 'day', night: 'night', later: 'night', snow: 'snow', jungle: 'jungle', legacy: 'day',
  ...Object.fromEntries(EXTRA_MODES.map(mode => [mode, 'day'])) };
// The authored weather of each mode (named preset, then overrides).
const authoredWeather = {
  snow: null, // keeps the overcast target the snow preset supplied
  jungle: { preset: 'broken', coverage: .7, wind: { speed: 12 } },
  legacy: { preset: 'clear', cirrus: .2, wind: { speed: 9 } },
};

const shared = 'A broad neutral ground with a terracotta matte sphere on the left, a chrome sphere on the right '
  + 'and three distant block forms behind them. The status below the view reads ready, with no displayed failure.';
const behaviors = {
  day: 'A clear late-afternoon daytime sky: readable deep blue overhead brightening to a warm pale golden horizon, '
    + 'with no cumulus or cloud sheet, at most a few faint thin high wisps. The chrome sphere clearly reflects the blue sky and the warm horizon band. ' + shared,
  twilight: 'Twilight just after sunset: a darkening blue-violet upper sky above a warm orange glow along the horizon, '
    + 'the distant block forms silhouetted against that glow and the spheres dimly lit. ' + shared,
  night: 'A dark night sky that stays dark overall, with a visibly partially lit moon showing its phase above the horizon, '
    + 'many small stars and a faint band of the Milky Way readable against the dark sky. The ground and spheres are dim, softly moonlit. ' + shared,
  later: 'The same dark night one day and one hour later: the moon is at a different place in the sky and shows a fuller phase, '
    + 'the stars and the faint Milky Way band have rotated to new positions, and the sky remains dark. ' + shared,
  // The M2 volumetric backend renders the overcast and broken weather targets as cloud structure,
  // so the snow and jungle descriptions below state the clouds that backend draws for them.
  // The ground is the scene's fixed ground plane (lib/scene.js), not controlled by the sky stage.
  snow: 'Snow preset at midday: a pale whitish-blue hazy sky with a bright milky horizon from the stronger haze, '
    + 'under a low pale grey-white overcast sheet covering the sky with soft, nearly shadowless light, and the chrome '
    + 'sphere reflecting the pale sky. The scene\'s fixed ground plane carries a terracotta matte sphere on the left, a chrome '
    + 'sphere on the right and three distant block forms behind them, all readable. The status below the view reads ready, '
    + 'with no displayed failure.',
  jungle: 'Jungle preset in mid-afternoon: a humid, slightly hazy blue sky with a soft bright greenish-warm horizon haze, '
    + 'broken volumetric cumulus with blue gaps under warm afternoon light, warm sunlight on the spheres and the chrome '
    + 'sphere reflecting the sky. ' + shared,
  legacy: 'The retained legacy sky model for comparison: a smooth artistic gradient from a blue zenith to a pale warm horizon '
    + 'with a flat painted look rather than physical scattering, at most a soft flat planar cloud layer and no volumetric '
    + 'clouds; the chrome sphere reflects that gradient. ' + shared,
};

// M2 volumetric backend: each mode below describes the cloud structure its authored weather renders.
const dayLook = 'a clear late-afternoon blue sky brightening to a warm pale horizon';
Object.assign(behaviors, {
  'fog-on': `Height fog switched on under ${dayLook}: the three distant block forms are visibly softened and washed toward `
    + 'the same pale warm horizon colour as the sky behind them, the far ground fades into that haze, while the near '
    + 'terracotta and chrome spheres stay crisp. ' + shared,
  'fog-off': `Fog switched off under ${dayLook}: the three distant block forms stand out with crisp, full-contrast edges `
    + 'against the horizon and the far ground stays sharp, with no haze veil. ' + shared,
  'ibl-on': `Sky image-based lighting fully on under ${dayLook}: the chrome sphere shows a clear reflection of the current `
    + 'blue sky above and the warm horizon band, and the matte sphere\'s shadowed side is gently filled with bluish sky light. ' + shared,
  'ibl-off': `Sky image-based lighting switched off (diffuse and specular zero) under ${dayLook}: the background sky still `
    + 'shows, but the chrome sphere no longer reflects it and looks dark and flat apart from the direct sun highlight, and the '
    + 'spheres\' shadowed sides receive only a flat neutral ambient. ' + shared,
  'ibl-diffuse': `Diffuse sky lighting only, specular reflections off, under ${dayLook}: the matte sphere and ground keep their `
    + 'soft bluish sky fill, while the chrome sphere shows no sky reflection and looks dark apart from its sun highlight. ' + shared,
  'ibl-specular': `Specular sky reflections only, diffuse sky lighting off, under ${dayLook}: the chrome sphere clearly reflects `
    + 'the blue sky and warm horizon, while the matte sphere\'s shadowed side is darker with only a flat neutral fill. ' + shared,
  'env-off': `The sky environment pass is disabled under ${dayLook}: the background sky still renders, but the cleared image-based `
    + 'lighting leaves the chrome sphere without any sky reflection and the shadowed sides lit only by flat ambient. ' + shared,
  'background-off': 'The sky background is disabled: everything behind the scene is one flat uniform slate-grey colour with no '
    + 'gradient, sun or horizon, while the chrome sphere still reflects the current blue and warm sky through image-based lighting. '
    + 'The ground, a terracotta matte sphere on the left, a chrome sphere on the right and three distant block forms are readable. '
    + 'The status below the view reads ready, with no displayed failure.',
  invalid: `The unchanged clear late-afternoon sky scene (${dayLook}, chrome sphere reflecting it) after rejected edits. `
    + 'The status text below the view lists each rejected edit (an out-of-range weather coverage, an out-of-range hour and a '
    + 'negative Mie multiplier) with its sky.invalid error message and states that the last accepted settings and rendering are '
    + 'kept. A broad neutral ground with a terracotta matte sphere on the left, a chrome sphere on the right and three distant block forms behind them.',
  'multi-view': 'Two side-by-side views of the same scene from two different cameras, both under the same clear late-afternoon sky '
    + 'with the same sun position, blue zenith and warm horizon, and the chrome sphere reflecting it in both views. Each view shows '
    + 'the ground, the terracotta matte sphere, the chrome sphere and the distant block forms. The status below reads ready.',
  resize: `After the canvas was resized smaller and back, the full-size frame shows ${dayLook} rendered cleanly at the current `
    + 'size, with no stretched, stale or blocky sky and no smearing or trails. ' + shared,
  'camera-cut': `After an abrupt camera cut, the new view shows ${dayLook} matching the new camera direction, with no ghosted `
    + 'previous view, trails or smeared sky; the chrome sphere reflects the current sky. ' + shared,
  'model-switch': `After switching the sky model to legacy and back to physical, the frame shows the physical ${dayLook}, `
    + 'not the flat legacy gradient, with no stale panorama or trails; the chrome sphere reflects that current sky. ' + shared,
  'weather-change': 'After a weather change that starts a broken-weather cycle and restarts it with a new seed, the physical sky '
    + 'shows current broken volumetric clouds with blue gaps between them, with no stale panorama, doubled edges or trails. '
    + shared,
  're-enable': `After the sky environment and cloud-lighting passes were disabled and re-enabled, ${dayLook} is shown and the `
    + 'chrome sphere again clearly reflects the current sky, with no black or stale reflection. ' + shared,
});

export const factNames = [
  'atmosphere-submitted', 'environment-published', 'view-submitted', 'time-authored',
  'sun-daylight', 'sun-twilight-band', 'sun-below-night', 'moon-in-view', 'moon-phase-partial',
  'celestial-positions-changed', 'physical-preset-mapping', 'omitted-preset-retains-moment',
  'explicit-preset-resets-moment', 'weather-preset-overrides', 'legacy-preset-mappings',
  'weather-retained-across-model-switches', 'sky-model-legacy',
  'fog-pass-on', 'fog-pass-off', 'ibl-lighting-accepted', 'environment-cleared', 'background-pass-off',
  'invalid-edits-rejected', 'invalid-edits-preserve-state', 'invalid-edits-preserve-rendering',
  'views-two-submitted', 'views-same-clock', 'cloud-flags-current',
  'reset-resize', 'view-size-restored', 'reset-camera-discontinuity', 'reset-model', 'model-weather-retained',
  'reset-weather', 'weather-cycle-start', 'weather-cycle-restart', 'reset-enable', 'environment-republished',
];

const common = ['snapshot:sky', 'snapshot:time', 'snapshot:weather', 'snapshot:diagnostics',
  'runtime:passes', 'runtime:viewSizes', 'runtime:ready',
  'assertion:atmosphere-submitted', 'assertion:environment-published', 'assertion:view-submitted', 'assertion:time-authored',
  'assertion:physical-preset-mapping'];
// Physical/legacy pair for the shared 30/300-frame functional collector.
export const MODEL_PAIR = 'sky-model';
// Every on/off pair below is measured by the shared collector inside its functional render.
export const FOG_PAIR = 'fog', IBL_PAIR = 'ibl', ENVIRONMENT_PAIR = 'environment', BACKGROUND_PAIR = 'sky-background';
const measurements = {
  day: { pair: MODEL_PAIR, state: 'on', preset: 'day', features: { model: 'physical' } },
  legacy: { pair: MODEL_PAIR, state: 'off', preset: 'day', features: { model: 'legacy' } },
  'fog-on': { pair: FOG_PAIR, state: 'on', preset: 'day', features: { fog: true } },
  'fog-off': { pair: FOG_PAIR, state: 'off', preset: 'day', features: { fog: false } },
  'ibl-on': { pair: IBL_PAIR, state: 'on', preset: 'day', features: { diffuse: 1, specular: 1 } },
  'ibl-off': { pair: IBL_PAIR, state: 'off', preset: 'day', features: { diffuse: 0, specular: 0 } },
  're-enable': { pair: ENVIRONMENT_PAIR, state: 'on', preset: 'day', features: { 'sky.env': true } },
  'env-off': { pair: ENVIRONMENT_PAIR, state: 'off', preset: 'day', features: { 'sky.env': false } },
  'camera-cut': { pair: BACKGROUND_PAIR, state: 'on', preset: 'day', features: { 'sky.background': true } },
  'background-off': { pair: BACKGROUND_PAIR, state: 'off', preset: 'day', features: { 'sky.background': false } },
};
// The cleared environment never publishes, so env-off proves its cleared state instead.
const commonFor = mode => mode === 'env-off' ? common.filter(f => f !== 'assertion:environment-published') : common;
const row = (id, criterionName, mode, filename, extra) => ({
  id, criterionName, scenario: 'atmosphere', mode, filename, expectedVisibleBehavior: behaviors[mode],
  requiredFacts: [...commonFor(mode), ...extra.map(name => `assertion:${name}`)], pair: measurements[mode]?.pair ?? null,
});

export const inventory = [
  row('clear-day-warm-horizon', 'Clear day with warm horizon', 'day', 'clear-day-warm-horizon-and-reflective-sphere.png', ['sun-daylight']),
  row('twilight-warm-horizon', 'Continuous twilight', 'twilight', 'twilight-warm-horizon-distant-forms.png', ['sun-twilight-band']),
  row('night-moon-stars-milky-way', 'Dark readable night', 'night', 'night-moon-phase-stars-and-milky-way.png', ['sun-below-night', 'moon-in-view', 'moon-phase-partial']),
  row('night-later-celestial-positions', 'Changed celestial time and date', 'later', 'night-later-celestial-positions.png', ['sun-below-night', 'moon-in-view', 'moon-phase-partial', 'celestial-positions-changed']),
  row('snow-preset', 'Snow preset mapping', 'snow', 'snow-preset-cold-overcast-bright-ground.png',
    ['sun-daylight', 'omitted-preset-retains-moment', 'explicit-preset-resets-moment']),
  row('jungle-preset', 'Jungle preset mapping', 'jungle', 'jungle-preset-humid-broken-cloud.png', ['sun-daylight', 'weather-preset-overrides']),
  row('legacy-comparison', 'Physical/legacy comparison', 'legacy', 'legacy-gradient-layer-comparison.png',
    ['sun-daylight', 'legacy-preset-mappings', 'weather-retained-across-model-switches', 'sky-model-legacy']),
  row('fog-on', 'Fog on coherent with the sky', 'fog-on', 'fog-on-sky-coherent-distant-forms.png', ['sun-daylight', 'fog-pass-on']),
  row('fog-off', 'Fog off', 'fog-off', 'fog-off-distant-forms.png', ['sun-daylight', 'fog-pass-off']),
  row('ibl-on', 'IBL on with current sky reflection', 'ibl-on', 'ibl-on-current-sky-reflection.png', ['sun-daylight', 'ibl-lighting-accepted']),
  row('ibl-off', 'IBL off', 'ibl-off', 'ibl-off-flat-ambient.png', ['sun-daylight', 'ibl-lighting-accepted']),
  row('ibl-diffuse-only', 'Independent diffuse IBL', 'ibl-diffuse', 'ibl-diffuse-only-sky-fill-no-reflection.png', ['sun-daylight', 'ibl-lighting-accepted']),
  row('ibl-specular-only', 'Independent specular IBL', 'ibl-specular', 'ibl-specular-only-chrome-sky-reflection.png', ['sun-daylight', 'ibl-lighting-accepted']),
  row('environment-disabled', 'Environment disable', 'env-off', 'environment-disabled-cleared-ibl-sky-kept.png', ['sun-daylight', 'environment-cleared']),
  row('background-disabled', 'Flat background', 'background-off', 'background-disabled-flat-colour.png', ['sun-daylight', 'background-pass-off']),
  row('invalid-edits', 'Invalid input keeps last valid state', 'invalid', 'error-invalid-sky-time-weather-edits-last-valid-state.png',
    ['invalid-edits-rejected', 'invalid-edits-preserve-state', 'invalid-edits-preserve-rendering']),
  row('multi-view-same-clock', 'Multi-view same-time facts', 'multi-view', 'multi-view-two-cameras-same-sky-clock.png',
    ['sun-daylight', 'views-two-submitted', 'views-same-clock', 'cloud-flags-current']),
  row('transition-resize', 'Resize transition', 'resize', 'transition-resize-current-sky-history-reset.png',
    ['reset-resize', 'view-size-restored', 'cloud-flags-current']),
  row('transition-camera-cut', 'Camera discontinuity transition', 'camera-cut', 'transition-camera-cut-current-sky-no-trails.png',
    ['reset-camera-discontinuity', 'cloud-flags-current']),
  row('transition-model-switch', 'Model change transition', 'model-switch', 'transition-model-switch-physical-sky-restored.png',
    ['reset-model', 'model-weather-retained', 'cloud-flags-current']),
  row('transition-weather-change', 'Weather change transition', 'weather-change', 'transition-weather-cycle-start-and-restart.png',
    ['reset-weather', 'weather-cycle-start', 'weather-cycle-restart', 'cloud-flags-current']),
  row('transition-re-enable', 'Re-enable transition', 're-enable', 'transition-re-enable-sky-environment-restored.png',
    ['reset-enable', 'environment-republished', 'cloud-flags-current']),
];

const snapshot = engine => ({ sky: engine.sky(), time: engine.timeOfDay(), weather: engine.weather() });
// Before/after states of the documented later-time action, keyed by the engine it ran on.
const actions = new WeakMap();

function errorOf(error) {
  return { code: typeof error?.code === 'string' ? error.code : 'error', message: String(error?.message ?? error) };
}

// Tagged public edits performed by build, keyed by the engine they ran on.
const recorded = new WeakMap();
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
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const near = (a, b) => isNum(a) && isNum(b) && Math.abs(a - b) <= NEAR_TOLERANCE * Math.max(1, Math.abs(b));
const sameJson = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const placeKeys = ['latitude', 'north', 'enabled', 'playing', 'speed'];
const weatherKeys = ['coverage', 'type', 'base', 'thickness', 'density', 'cirrus', 'evolution', 'rain'];
/** Field-by-field comparison of an observed weather target with an expected static target. */
const targetMatches = (t, want) => t != null && weatherKeys.every(k => near(t[k], want[k])) && near(t.wind?.speed, want.wind.speed) && near(t.wind?.from, want.wind.from);
/** Expected target of a named weather preset plus structured overrides, from the static table. */
export function expectedWeather(request) {
  const { preset, wind, ...rest } = request;
  const base = weatherTargets[preset];
  return base ? { ...base, ...rest, wind: { ...base.wind, ...wind } } : null;
}

/** The accepted physical preset transaction matches the static S24 entry of the REQUESTED preset and keeps place/playback. */
function presetMapping(rec) {
  const preset = rec.request?.preset, want = Object.hasOwn(physicalPresets, preset ?? '') ? physicalPresets[preset] : null;
  const { edit } = rec, { sky, time, weather } = edit.after, t = weather?.target;
  const passed = want !== null && (rec.request.model ?? 'physical') === 'physical' && edit.accepted === true
    && sky?.model === 'physical' && sky.profile === preset
    && want.groundAlbedo.every((v, i) => near(sky.atmosphere?.groundAlbedo?.[i], v)) && near(sky.atmosphere?.mieMultiplier, want.mieMultiplier)
    && time?.day === want.day && near(time.hours, want.hours) && time.currentDay === want.day && near(time.currentHours, want.hours)
    && placeKeys.every(key => time[key] === edit.before.time?.[key])
    && targetMatches(t, want.target);
  return { name: 'physical-preset-mapping', passed,
    detail: `${edit.operation} requested ${JSON.stringify(rec.request)}: profile ${sky?.profile} model ${sky?.model} atmosphere ${JSON.stringify(sky?.atmosphere)}; `
      + `time ${time?.day}/${time?.hours} current ${time?.currentDay}/${time?.currentHours}; weather target ${JSON.stringify(t)}; expected ${JSON.stringify(want)}` };
}

/** Assertions over the recorded preset, override and model-switch edits; expectations come from requests and static tables. */
export function editAssertions(records, current) {
  const by = tag => records.filter(r => r.tag === tag);
  const out = [];
  const [mapping] = by('preset');
  if (mapping) out.push(presetMapping(mapping));
  const [omittedRec] = by('omitted'), [explicitRec] = by('explicit');
  const omitted = omittedRec?.edit, explicit = explicitRec?.edit;
  if (omitted) out.push({ name: 'omitted-preset-retains-moment',
    passed: !Object.hasOwn(omittedRec.request, 'preset') && omitted.accepted === true && omitted.before.time.hours === OMITTED_HOURS
      && omitted.after.time.hours === OMITTED_HOURS && omitted.after.time.currentHours === OMITTED_HOURS
      && sameJson(omitted.after.sky, omitted.before.sky) && sameJson(omitted.after.weather.target, omitted.before.weather.target),
    detail: `${omitted.operation}: hours ${omitted.before.time.hours}->${omitted.after.time.hours}, current ${omitted.after.time.currentHours}, profile ${omitted.after.sky.profile}` });
  if (explicit) {
    const m = presetMapping(explicitRec);
    out.push({ name: 'explicit-preset-resets-moment',
      passed: m.passed && explicit.before.time.hours === OMITTED_HOURS && sameJson(explicit.after.sky, omitted?.after.sky),
      detail: `${explicit.operation}: hours ${explicit.before.time.hours}->${explicit.after.time.hours}, current ${explicit.after.time.currentHours}; mapping ${m.passed}; resolved sky identical to omitted-preset result ${sameJson(explicit.after.sky, omitted?.after.sky)}` });
  }
  const [overridesRec] = by('weather');
  if (overridesRec) {
    const overrides = overridesRec.edit, t = overrides.after.weather.target, want = expectedWeather(overridesRec.request);
    out.push({ name: 'weather-preset-overrides',
      passed: want !== null && overrides.accepted === true && targetMatches(t, want) && overrides.after.weather.changing === false && targetMatches(current.weather?.target, want),
      detail: `${overrides.operation}: target ${JSON.stringify(t)}; expected ${JSON.stringify(want)}; cyclePreset ${overrides.after.weather.cyclePreset}` });
  }
const legacy = by('legacy-preset');
  if (legacy.length) out.push({ name: 'legacy-preset-mappings',
    passed: sameJson(legacy.map(r => r.request.preset), LEGACY_ORDER) && legacy.every(({ request, edit: e }) => request.model === 'legacy'
      && e.accepted === true && e.after.sky?.model === 'legacy' && e.after.sky.profile === request.preset
      && sameJson(e.after.weather.target, e.before.weather.target) && e.after.time.day === e.before.time.day && e.after.time.hours === e.before.time.hours),
    detail: legacy.map(({ request, edit: e }) => `${e.operation}: requested ${request.preset}/${request.model}, profile ${e.after.sky?.profile} model ${e.after.sky?.model}, weather kept ${sameJson(e.after.weather.target, e.before.weather.target)}`).join('; ') });
  const switches = by('model');
  if (switches.length) {
    const want = expectedWeather(authoredWeather.legacy), authored = switches[0].edit.before.weather;
    const keep = w => w != null && targetMatches(w.target, want) && ['enabled', 'shadows', 'changing', 'seed', 'transitionSeconds'].every(k => w[k] === authored[k]);
    out.push({ name: 'weather-retained-across-model-switches',
      passed: sameJson(switches.map(r => r.request.model), ['legacy', 'physical', 'legacy']) && switches.every(({ request, edit: e }) => e.accepted === true
        && !Object.hasOwn(request, 'preset') && e.after.sky?.model === request.model && keep(e.after.weather)) && keep(authored) && keep(current.weather),
      detail: `expected ${JSON.stringify(want)}; ` + switches.map(({ request, edit: e }) => `${e.operation}: requested ${request.model}, model ${e.after.sky?.model}, kept ${keep(e.after.weather)}`).join('; ') });
    const last = switches[switches.length - 1].request;
    out.push({ name: 'sky-model-legacy', passed: last.model === 'legacy' && current.sky?.model === 'legacy' && current.sky.profile === 'day',
      detail: `requested ${last.model}; final sky model ${current.sky?.model} profile ${current.sky?.profile}` });
  }
  return out;
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
// The accepted-state publication each capture waits for: a cleared environment (env-off) waits
// for no publication and the multi-view capture waits for both submitted views.
const settled = (d, mode) => d.readiness.status === 'ready' && d.readiness.error === null && d.atmosphereGeneration > 0
  && (mode === 'env-off' ? d.environment.publishedGeneration === null && d.environment.pending === false
    : d.environment.publishedGeneration !== null && d.environment.publishedGeneration >= d.environment.requestedGeneration
      && d.environment.pending === false)
  && d.views.length >= (mode === 'multi-view' ? 2 : 1) && d.views.every(view => view.submittedInvocation !== null);

/**
 * Renders until the submitted atmosphere/Milky Way (readiness ready) and the environment
 * generation requested by the accepted state are published, or the frame limit passes;
 * the assertions then report the actual diagnostics either way.
 */
async function awaitPublication(engine, variant) {
  let diagnostics = engine.skyWeatherDiagnostics();
  for (let frame = 0; frame < FRAME_LIMIT && !settled(diagnostics, variant.id); frame++) {
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
  let diagnostics = await awaitPublication(engine, variant);
  const judgeTransition = transitions[mode] ? await transitions[mode](engine, variant, diagnostics) : null;
  if (judgeTransition) {
    await frames(engine, variant, TRANSITION_FRAMES); // the final state is observed only after newer submitted frames
    diagnostics = await awaitPublication(engine, variant);
  }
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
  const records = recorded.get(engine) ?? [];
  assertions.push(...editAssertions(records, { sky, time, weather }));
  assertions.push(...featureAssertions(engine, mode, records, diagnostics));
  if (judgeTransition) assertions.push(...judgeTransition(diagnostics, { sky, time, weather }));
  const edits = records.map(r => r.edit);
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

// ---- Controls, atomic edits, views and transitions (public API only) ----
export const FOG_SETTINGS = { preset: 'haze', density: .025, heightFalloff: .05, baseHeight: 0, start: 5 };
export const FLAT_BACKGROUND = [.32, .36, .42];
export const IBL_SETTINGS = { 'ibl-on': { diffuse: 1, specular: 1 }, 'ibl-off': { diffuse: 0, specular: 0 },
  'ibl-diffuse': { diffuse: 1, specular: 0 }, 'ibl-specular': { diffuse: 0, specular: 1 } };
export const MULTI_VIEW_RECTS = [{ x: 0, y: 0, width: .5, height: 1 }, { x: .5, y: 0, width: .5, height: 1 }];
export const MULTI_VIEW_ASPECT = (1280 * .5) / 720;
// View 1 looks at the same subjects from the right; tests prove its frustum overlap.
export const secondViewCamera = { eye: [4.5, 3, 9.5], target: [.5, 1.8, -8], fovYDeg: 55, near: .1, far: 300 };
// The camera-cut mode renders from this camera first and then cuts to its selection camera.
export const preCutCamera = { eye: [-5, 2, 7], target: [1, 1.6, -10], fovYDeg: 60, near: .1, far: 300 };
export const RESIZE_TO = [960, 540];
export const INVALID_EDITS = [
  ['setWeather({coverage:2})', engine => engine.setWeather({ coverage: 2 })],
  ['setTimeOfDay({hours:25})', engine => engine.setTimeOfDay({ hours: 25 })],
  ['setSky({atmosphere:{mieMultiplier:-1}})', engine => engine.setSky({ atmosphere: { mieMultiplier: -1 } })],
];
export const CYCLE_START = { preset: 'broken', coverage: .7, changing: true, seed: 9 };
export const CYCLE_RESTART = { seed: 10 };

const passEdit = (engine, name, enabled) => record(engine, 'pass', `setPassEnabled('${name}',${enabled})`, { name, enabled },
  () => engine.setPassEnabled(name, enabled));
const fogEdit = engine => record(engine, 'fog', `setFog(${JSON.stringify(FOG_SETTINGS)})`, FOG_SETTINGS, () => engine.setFog(FOG_SETTINGS));
const iblEdit = mode => engine => record(engine, 'ibl', `setEnvironmentLighting(${JSON.stringify(IBL_SETTINGS[mode])})`,
  IBL_SETTINGS[mode], () => engine.setEnvironmentLighting(IBL_SETTINGS[mode]));
const setup = {
  'fog-on': fogEdit,
  'fog-off': engine => { fogEdit(engine); passEdit(engine, 'fog', false); },
  'ibl-on': iblEdit('ibl-on'), 'ibl-off': iblEdit('ibl-off'), 'ibl-diffuse': iblEdit('ibl-diffuse'), 'ibl-specular': iblEdit('ibl-specular'),
  'env-off': engine => passEdit(engine, 'sky.env', false),
  'background-off': engine => { engine.setBackground(FLAT_BACKGROUND); passEdit(engine, 'sky.background', false); },
  'multi-view': engine => { engine.setViewports(MULTI_VIEW_RECTS); engine.setViewportCamera(1, secondViewCamera); },
  'camera-cut': engine => engine.setCamera(preCutCamera),
};

/** Records an edit that is expected to be refused; unlike record it keeps the refusal. */
function attempt(engine, tag, operation, apply) {
  const before = snapshot(engine);
  let accepted = true, error = null;
  try { apply(); } catch (cause) { accepted = false; error = errorOf(cause); }
  const edit = { operation, accepted, error, before, after: snapshot(engine) };
  if (!recorded.has(engine)) recorded.set(engine, []);
  recorded.get(engine).push({ tag, request: null, edit });
  return edit;
}
// Displayed error lines of the invalid mode, keyed by engine.
const displayedErrors = new WeakMap();
const TRANSITION_FRAMES = 4;
async function frames(engine, variant, count) {
  for (let frame = 0; frame < count; frame++) { engine.render(variant.timeMs ?? 1000); await yieldTask(); }
  return engine.skyWeatherDiagnostics();
}
const viewFacts = d => JSON.stringify(d.views.map(v => ({ view: v.view, size: [v.width, v.height], invocation: v.invocation,
  submitted: v.submittedInvocation, reasons: v.resetReasons })));
/** A reset reason absent from every view before the action and present on every view after it, with newer submitted work. */
function reasonAdded(name, reason, before, after) {
  const passed = before.views.length > 0 && after.views.length === before.views.length
    && before.views.every(v => !v.resetReasons.includes(reason)) && after.views.every(v => v.resetReasons.includes(reason))
    && isNum(after.submittedInvocation) && isNum(before.submittedInvocation) && after.submittedInvocation > before.submittedInvocation;
  return { name, passed, detail: `${reason}: submitted ${before.submittedInvocation}->${after.submittedInvocation}; before ${viewFacts(before)}; after ${viewFacts(after)}` };
}
const published = env => env.publishedGeneration !== null && env.publishedGeneration >= env.requestedGeneration && env.pending === false;

// Each transition runs after the first settled publication and returns a judge of the final settled state.
const transitions = {
  async resize(engine, variant, before) {
    const [width, height] = [before.views[0]?.width, before.views[0]?.height];
    record(engine, 'transition', `resize(${RESIZE_TO.join(',')})`, { size: RESIZE_TO }, () => engine.resize(...RESIZE_TO));
    const mid = await frames(engine, variant, TRANSITION_FRAMES);
    record(engine, 'transition', `resize(${width},${height})`, { size: [width, height] }, () => engine.resize(width, height));
    return after => [reasonAdded('reset-resize', 'Resize', before, after), { name: 'view-size-restored',
      passed: mid.views[0]?.width === RESIZE_TO[0] && mid.views[0]?.height === RESIZE_TO[1] && after.views[0]?.width === width && after.views[0]?.height === height,
      detail: `view 0 ${width}x${height} -> ${mid.views[0]?.width}x${mid.views[0]?.height} -> ${after.views[0]?.width}x${after.views[0]?.height}` }];
  },
  async 'camera-cut'(engine, variant, before) {
    const { aspect, ...camera } = variant.selection.camera;
    record(engine, 'transition', `setCamera(${JSON.stringify(camera)})`, camera, () => engine.setCamera(camera));
    return after => [reasonAdded('reset-camera-discontinuity', 'CameraDiscontinuity', before, after)];
  },
  async 'model-switch'(engine, variant, before) {
    const a = record(engine, 'transition', "setSky({model:'legacy'})", { model: 'legacy' }, () => engine.setSky({ model: 'legacy' }));
    await frames(engine, variant, TRANSITION_FRAMES);
    const b = record(engine, 'transition', "setSky({model:'physical'})", { model: 'physical' }, () => engine.setSky({ model: 'physical' }));
    return (after, current) => [reasonAdded('reset-model', 'Model', before, after), { name: 'model-weather-retained',
      passed: a.after.sky?.model === 'legacy' && b.after.sky?.model === 'physical' && current.sky?.model === 'physical' && current.sky.profile === 'day'
        && [a, b].every(e => sameJson(e.after.weather, e.before.weather) && sameJson(e.after.time, e.before.time)),
      detail: `models ${a.after.sky?.model}->${b.after.sky?.model}; final ${current.sky?.model}/${current.sky?.profile}; weather kept ${[a, b].map(e => sameJson(e.after.weather, e.before.weather))}` }];
  },
  async 'weather-change'(engine, variant, before) {
    const start = record(engine, 'transition', `setWeather(${JSON.stringify(CYCLE_START)})`, CYCLE_START, () => engine.setWeather(CYCLE_START));
    await frames(engine, variant, TRANSITION_FRAMES);
    const restart = record(engine, 'transition', `setWeather(${JSON.stringify(CYCLE_RESTART)})`, CYCLE_RESTART, () => engine.setWeather(CYCLE_RESTART));
    const want = expectedWeather({ preset: CYCLE_START.preset, coverage: CYCLE_START.coverage });
    const w1 = start.after.weather, w2 = restart.after.weather;
    return (after, current) => [reasonAdded('reset-weather', 'Weather', before, after),
      { name: 'weather-cycle-start', passed: w1?.changing === true && w1.cyclePreset === CYCLE_START.preset && w1.seed === CYCLE_START.seed
          && targetMatches(w1.target, want) && after.weatherGeneration > before.weatherGeneration,
        detail: `${start.operation}: changing ${w1?.changing}, cyclePreset ${w1?.cyclePreset}, seed ${w1?.seed}, target ${JSON.stringify(w1?.target)}; weatherGeneration ${before.weatherGeneration}->${after.weatherGeneration}` },
      { name: 'weather-cycle-restart', passed: w2?.changing === true && w2.seed === CYCLE_RESTART.seed && w2.cyclePreset === CYCLE_START.preset
          && targetMatches(w2.target, want) && sameJson(current.weather?.target, w2.target) && current.weather.cyclePreset === CYCLE_START.preset,
        detail: `${restart.operation}: seed ${w1?.seed}->${w2?.seed}, cyclePreset ${w2?.cyclePreset}, final cyclePreset ${current.weather?.cyclePreset}` }];
  },
  async 're-enable'(engine, variant, before) {
    passEdit(engine, 'sky.env', false); passEdit(engine, 'sky.cloud-lighting', false);
    const mid = await frames(engine, variant, TRANSITION_FRAMES);
    passEdit(engine, 'sky.env', true); passEdit(engine, 'sky.cloud-lighting', true);
    return after => [reasonAdded('reset-enable', 'Enable', before, after), { name: 'environment-republished',
      passed: published(before.environment) && mid.environment.publishedGeneration === null && published(after.environment)
        && ['sky.env', 'sky.cloud-lighting'].every(name => engine.passes().find(p => p.name === name)?.enabled === true),
      detail: `environment before ${JSON.stringify(before.environment)}, disabled ${JSON.stringify(mid.environment)}, re-enabled ${JSON.stringify(after.environment)}` }];
  },
  async invalid(engine, variant, before) {
    const edits = INVALID_EDITS.map(([operation, apply]) => attempt(engine, 'invalid', operation, () => apply(engine)));
    displayedErrors.set(engine, [...edits.map(e => `Rejected ${e.operation}: ${e.error ? `${e.error.code}: ${e.error.message}` : 'unexpectedly accepted'}`),
      'Last accepted sky, time and weather settings and the rendered sky are kept.']);
    await frames(engine, variant, TRANSITION_FRAMES);
    return (after, current) => [
      { name: 'invalid-edits-rejected', passed: edits.length === INVALID_EDITS.length && edits.every(e => e.accepted === false && e.error?.code === 'sky.invalid'),
        detail: edits.map(e => `${e.operation}: accepted ${e.accepted}, ${e.error?.code}: ${e.error?.message}`).join('; ') },
      { name: 'invalid-edits-preserve-state', passed: edits.every(e => sameJson(e.after, e.before)) && sameJson(edits[0].before, current),
        detail: `before ${JSON.stringify(edits[0].before.time)}; current ${JSON.stringify(current.time)}; per-edit unchanged ${edits.map(e => sameJson(e.after, e.before))}` },
      { name: 'invalid-edits-preserve-rendering', passed: after.weatherGeneration === before.weatherGeneration
          && after.atmosphereGeneration === before.atmosphereGeneration && after.environment.requestedGeneration === before.environment.requestedGeneration
          && published(after.environment) && after.submittedInvocation > before.submittedInvocation,
        detail: `weatherGeneration ${before.weatherGeneration}->${after.weatherGeneration}, atmosphereGeneration ${before.atmosphereGeneration}->${after.atmosphereGeneration}, `
          + `environment ${JSON.stringify(before.environment)}->${JSON.stringify(after.environment)}, submitted ${before.submittedInvocation}->${after.submittedInvocation}` }];
  },
};

/** Feature-control and view assertions from recorded edits, actual pass states and diagnostics. */
export function featureAssertions(engine, mode, records, d) {
  const out = [], env = d.environment;
  const pass = name => engine.passes().find(p => p.name === name);
  const tagged = tag => records.filter(r => r.tag === tag).map(r => r.edit);
  const accepted = edits => edits.length > 0 && edits.every(e => e.accepted === true && e.error === null);
  const passFacts = names => names.map(name => `${name} ${pass(name)?.enabled}`).join(', ');
  if (mode === 'fog-on' || mode === 'fog-off') {
    const on = mode === 'fog-on';
    out.push({ name: on ? 'fog-pass-on' : 'fog-pass-off', passed: accepted(tagged('fog')) && pass('fog')?.enabled === on
      && (on || accepted(tagged('pass'))), detail: `${tagged('fog').map(e => e.operation)}; passes ${passFacts(['fog'])}` });
  }
  if (Object.hasOwn(IBL_SETTINGS, mode)) {
    const edits = tagged('ibl');
    out.push({ name: 'ibl-lighting-accepted', passed: edits.length === 1 && accepted(edits) && pass('sky.env')?.enabled === true && published(env),
      detail: `${edits.map(e => `${e.operation} accepted ${e.accepted}`)}; passes ${passFacts(['sky.env'])}; environment ${JSON.stringify(env)}` });
  }
  if (mode === 'env-off') out.push({ name: 'environment-cleared', passed: accepted(tagged('pass')) && pass('sky.env')?.enabled === false
      && pass('sky.background')?.enabled === true && env.publishedGeneration === null && env.pending === false,
    detail: `passes ${passFacts(['sky.env', 'sky.background'])}; environment ${JSON.stringify(env)}` });
  if (mode === 'background-off') out.push({ name: 'background-pass-off', passed: accepted(tagged('pass')) && pass('sky.background')?.enabled === false
      && pass('sky.env')?.enabled === true && published(env),
    detail: `passes ${passFacts(['sky.background', 'sky.env'])}; environment ${JSON.stringify(env)}; flat ${JSON.stringify(FLAT_BACKGROUND)}` });
  if (mode === 'multi-view') {
    const [a, b] = d.views;
    out.push({ name: 'views-two-submitted', passed: d.views.length === 2 && a.view === 0 && b.view === 1
        && d.views.every(v => v.submittedInvocation !== null && v.width > 0 && v.height > 0) && a.width === b.width && a.height === b.height,
      detail: viewFacts(d) });
    out.push({ name: 'views-same-clock', passed: d.views.length === 2 && isNum(d.invocation) && isNum(d.submittedInvocation)
        && d.views.every(v => v.invocation === d.invocation && v.submittedInvocation === d.submittedInvocation),
      detail: `one timeOfDay snapshot serves both views; engine invocation ${d.invocation}, submitted ${d.submittedInvocation}; views ${viewFacts(d)}` });
  }
  // The M2 volumetric backend renders the cloud screen and panorama: every view must publish both as current.
  out.push({ name: 'cloud-flags-current', passed: d.views.length > 0 && d.views.every(v => v.cloudPanoramaValid === true && v.cloudScreenValid === true),
    detail: JSON.stringify(d.views.map(v => [v.view, v.cloudScreenValid, v.cloudPanoramaValid])) });
  return out;
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
// public tone-mapping option; the sky still renders dark, only readable. Headroom check from
// run 13615 at 0 EV (AgX, MSAA 4): 99.5% of the twilight/night pixels were under 14/255, i.e.
// linear scene radiance under ~(14/255)^2.2 = 1.7e-3; x128 keeps them under ~0.22, below
// middle grey, so night stays night while the moon disc (3 x 0.0024 x 128 ~ 0.9) reads.
export const DARK_ADAPTATION_EV = Math.log2(128);
export const exposureEv = { day: 0, twilight: DARK_ADAPTATION_EV, night: DARK_ADAPTATION_EV, later: DARK_ADAPTATION_EV,
  snow: 0, jungle: 0, legacy: 0, ...Object.fromEntries(EXTRA_MODES.map(mode => [mode, 0])) };

// The physical day capture and the legacy comparison share the preset and moment; the
// collector measures each at 1280x720 within the functional render.
// Each selection is the camera that renders the capture; multi-view view 0 is the left half.
const selectionFor = (camera, aspect) => {
  const selection = sampleSelection(camera);
  return aspect === undefined ? selection : { ...selection, camera: { ...selection.camera, aspect } };
};
const variant = (id, camera, aspect) => ({ id, description: behaviors[id], references: [], judge: true, timeMs: 1000,
  engine: { exposureEv: exposureEv[id] }, selection: selectionFor(camera, aspect),
  ...(measurements[id] ? { measurement: measurements[id] } : {}) });

export default {
  id: 'atmosphere', milestone: 'atmosphere', title: 'Physical atmosphere and celestial time',
  engine: { msaa: 4, toneMapper: 'agx', exposureEv: 0 },
  variants: [variant('day', dayCamera), variant('twilight', dayCamera), variant('night', nightCamera), variant('later', nightCamera),
    variant('snow', dayCamera), variant('jungle', dayCamera), variant('legacy', dayCamera),
    ...FEATURE_MODES.map(mode => variant(mode, dayCamera)), variant('invalid', dayCamera),
    variant('multi-view', dayCamera, MULTI_VIEW_ASPECT), ...TRANSITION_MODES.map(mode => variant(mode, dayCamera))],
  controls,
  // The invalid mode shows each rejected edit's actual error in the displayed status region,
  // appended after the runtime publishes ready and before it captures the frame.
  frame(engine) {
    const lines = displayedErrors.get(engine);
    const status = globalThis.document?.getElementById?.('status');
    if (!lines || !status || !status.textContent.startsWith('ready:') || status.textContent.includes(lines[0])) return;
    status.textContent = [status.textContent, ...lines].join('\n');
  },
  async build(engine, lib, selected) {
    const mode = selected.id;
    if (!moments[mode]) throw new Error(`atmosphere has no mode ${mode}`);
    const night = mode === 'night' || mode === 'later';
    engine.setBackground([.05, .06, .08]);
    // Manual daylight baseline; automatic celestial control supplies the direction.
    engine.setLight({ direction: [-.35, -.6, -.55], color: [1, .95, .88], intensity: 3 });
    engine.setAmbient({ color: [.55, .62, .75], intensity: night ? .02 : .15 });
    const preset = presetOf[mode];
    const presetRequest = { preset, model: 'physical' };
    record(engine, 'preset', `setSky({preset:'${preset}',model:'physical'})`, presetRequest, () => engine.setSky(presetRequest));
    if (mode === 'snow') {
      // Omitted preset keeps the edited moment; an explicit preset resets it.
      engine.setTimeOfDay({ hours: OMITTED_HOURS });
      record(engine, 'omitted', 'setSky({}) after setTimeOfDay({hours:13})', {}, () => engine.setSky({}));
      record(engine, 'explicit', "setSky({preset:'snow'})", { preset: 'snow' }, () => engine.setSky({ preset: 'snow' }));
    } else if (mode === 'jungle') {
      record(engine, 'weather', "setWeather({preset:'broken',coverage:0.7,wind:{speed:12}})", authoredWeather.jungle, () => engine.setWeather(authoredWeather.jungle));
    } else if (mode === 'legacy') {
      engine.setWeather(authoredWeather.legacy);
    } else {
      engine.setWeather({ preset: 'clear' });
    }
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
    if (mode === 'legacy') {
      // Model-only switches and legacy presets retain the authored weather and moment.
      record(engine, 'model', "setSky({model:'legacy'})", { model: 'legacy' }, () => engine.setSky({ model: 'legacy' }));
      for (const p of LEGACY_ORDER) {
        record(engine, 'legacy-preset', `setSky({preset:'${p}',model:'legacy'})`, { preset: p, model: 'legacy' }, () => engine.setSky({ preset: p, model: 'legacy' }));
      }
      record(engine, 'model', "setSky({model:'physical'})", { model: 'physical' }, () => engine.setSky({ model: 'physical' }));
      record(engine, 'model', "setSky({model:'legacy'})", { model: 'legacy' }, () => engine.setSky({ model: 'legacy' }));
    }
    await buildSkySample(engine, lib, selected.selection.camera);
    setup[mode]?.(engine);
  },
};
