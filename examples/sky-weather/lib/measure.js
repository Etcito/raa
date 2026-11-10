// Functional on/off pair collector. Callers pass the single operation deadline
// (absolute monotonic milliseconds) so a pair never starts a fresh budget.
const WIDTH = 1280;
const HEIGHT = 720;
const WARMUP_FRAMES = 30;
const MEASURED_FRAMES = 300;
const FRAME_TIMEOUT_MS = 10000;

const states = ['on', 'off'];
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = value => typeof value === 'string' && value.trim().length > 0;
const finite = value => typeof value === 'number' && Number.isFinite(value);

function fail(message) { throw new Error(`feature measure: ${message}`); }

function validateFeatures(features, context) {
  if (!record(features)) fail(`${context}: features must be an object`);
  for (const [name, value] of Object.entries(features)) {
    if (!(typeof value === 'boolean' || typeof value === 'string' || finite(value))) {
      fail(`${context}: feature ${name} must be a boolean, string or finite number`);
    }
  }
}

function validateSettings(settings) {
  if (!record(settings)) fail('settings must be an object');
  if (!text(settings.pair)) fail('settings.pair must be a nonempty string');
  if (!states.includes(settings.state)) fail('settings.state must be on or off');
  if (!text(settings.preset)) fail('settings.preset must be a nonempty string');
  validateFeatures(settings.features, 'settings');
  if (!finite(settings.deadlineMs)) fail('settings.deadlineMs must be a finite deadline');
}

function summarize(frameTimesMs) {
  const ordered = [...frameTimesMs].sort((a, b) => a - b);
  const nearest = p => ordered[Math.ceil(p * ordered.length) - 1];
  const mean = frameTimesMs.reduce((sum, value) => sum + value, 0) / frameTimesMs.length;
  return { fps: 1000 / mean, p50Ms: nearest(.5), p95Ms: nearest(.95), p99Ms: nearest(.99) };
}

// Resolves with the animation-frame timestamp, or rejects when the frame is
// absent for the earlier of ten seconds and the operation's remaining budget.
function nextFrame(clock, deadlineMs, label) {
  return new Promise((resolve, reject) => {
    const startedAt = clock.now();
    if (!finite(startedAt)) {
      reject(new Error('feature measure: clock.now() must return a finite millisecond timestamp'));
      return;
    }
    if (startedAt >= deadlineMs) {
      reject(new Error(`feature measure: operation deadline expired before ${label}`));
      return;
    }
    const waitMs = Math.min(FRAME_TIMEOUT_MS, deadlineMs - startedAt);
    let handle = null;
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      if (handle !== null) { try { clock.cancelFrame(handle); } catch {} }
      reject(new Error(`feature measure: missing ${label} within ${waitMs} ms of the operation deadline`));
    }, waitMs);
    try {
      handle = clock.requestFrame(nowMs => {
        if (settled) return;
        const finishedAt = clock.now();
        if (!finite(finishedAt) || finishedAt >= deadlineMs) {
          settled = true;
          clearTimeout(timer);
          reject(new Error(`feature measure: operation deadline expired at ${label}`));
          return;
        }
        settled = true;
        clearTimeout(timer);
        resolve(nowMs);
      });
    } catch (error) {
      settled = true;
      clearTimeout(timer);
      reject(error);
    }
  });
}

function validateInterval(previous, timestamp, index) {
  if (!finite(timestamp) || !finite(previous) || !(timestamp > previous)) {
    fail(`frameTimesMs[${index}] requires a positive finite animation-frame interval`);
  }
  return timestamp - previous;
}

/**
 * Collects one on/off pair state: 30 warm-up frames, then exactly 300 measured
 * requestAnimationFrame intervals at 1280x720 with device pixel ratio 1.
 * @param {(nowMs:number)=>void} render
 * @param {{pair:string,state:'on'|'off',preset:string,features:Record<string,boolean|string|number>,deadlineMs:number}} settings
 * @param {{now:()=>number,requestFrame:(callback:(nowMs:number)=>void)=>number,cancelFrame:(handle:number)=>void}} clock
 */
export async function collectFeatureMeasurement(render, settings, clock) {
  if (typeof render !== 'function') fail('render must be a function');
  validateSettings(settings);
  if (!record(clock) || typeof clock.now !== 'function' ||
      typeof clock.requestFrame !== 'function' || typeof clock.cancelFrame !== 'function') {
    fail('clock must provide now, requestFrame and cancelFrame');
  }
  if (settings.deadlineMs <= clock.now()) fail('operation deadline already expired');

  let previous = null;
  for (let index = 0; index < WARMUP_FRAMES; index++) {
    previous = await nextFrame(clock, settings.deadlineMs, `warm-up frame ${index + 1}`);
    render(previous);
  }
  const frameTimesMs = [];
  for (let index = 0; index < MEASURED_FRAMES; index++) {
    const timestamp = await nextFrame(clock, settings.deadlineMs, `measured frame ${index + 1}`);
    frameTimesMs.push(validateInterval(previous, timestamp, index));
    previous = timestamp;
    render(timestamp);
  }

  return validateFeatureMeasurement({
    pair: settings.pair,
    state: settings.state,
    preset: settings.preset,
    features: { ...settings.features },
    width: WIDTH,
    height: HEIGHT,
    warmup: WARMUP_FRAMES,
    frames: MEASURED_FRAMES,
    frameTimesMs,
    ...summarize(frameTimesMs),
  });
}

/**
 * Rejects any incomplete or inconsistent pair measurement without filling,
 * truncating or re-deriving stored intervals.
 * @param {unknown} value
 */
export function validateFeatureMeasurement(value) {
  if (!record(value)) fail('measurement must be an object');
  if (!text(value.pair)) fail('measurement.pair must be a nonempty string');
  if (!states.includes(value.state)) fail('measurement.state must be on or off');
  if (!text(value.preset)) fail('measurement.preset must be a nonempty string');
  validateFeatures(value.features, 'measurement');
  if (value.width !== WIDTH || value.height !== HEIGHT) {
    fail(`measurement must be ${WIDTH}x${HEIGHT} physical pixels at device pixel ratio 1`);
  }
  if (value.warmup !== WARMUP_FRAMES) fail(`measurement.warmup must be ${WARMUP_FRAMES}`);
  if (value.frames !== MEASURED_FRAMES) fail(`measurement.frames must be ${MEASURED_FRAMES}`);
  if (!Array.isArray(value.frameTimesMs) || value.frameTimesMs.length !== MEASURED_FRAMES) {
    fail(`measurement requires exactly ${MEASURED_FRAMES} frame intervals`);
  }
  for (let index = 0; index < value.frameTimesMs.length; index++) {
    const interval = value.frameTimesMs[index];
    if (!finite(interval) || interval <= 0) {
      fail(`measurement.frameTimesMs[${index}] must be a positive finite number`);
    }
  }
  const expected = summarize(value.frameTimesMs);
  if (!finite(expected.fps) || expected.fps <= 0) fail('measurement frame intervals do not produce a finite positive fps');
  for (const key of ['fps', 'p50Ms', 'p95Ms', 'p99Ms']) {
    if (value[key] !== expected[key]) fail(`measurement.${key} must be ${expected[key]}, got ${value[key]}`);
  }
  return value;
}
