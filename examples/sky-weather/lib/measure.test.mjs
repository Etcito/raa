import test from 'node:test';
import assert from 'node:assert/strict';
import { collectFeatureMeasurement, validateFeatureMeasurement } from './measure.js';

const START = 1000;
const DEADLINE = START + 60000;
const SETTINGS = { pair: 'fog', state: 'on', preset: 'day', features: { fog: true, model: 'physical', samples: 4 } };

function shuffledMeasured() {
  const values = Array.from({ length: 300 }, (_, index) => index + 1);
  for (let index = values.length - 1; index > 0; index--) {
    const swap = (index * 41 + 7) % (index + 1);
    [values[index], values[swap]] = [values[swap], values[index]];
  }
  return values;
}

function steps() { return [...Array(30).fill(4), ...shuffledMeasured()]; }

function build(frameTimesMs, overrides = {}) {
  const ordered = [...frameTimesMs].sort((a, b) => a - b);
  const nearest = p => ordered[Math.ceil(p * ordered.length) - 1];
  const mean = frameTimesMs.reduce((sum, value) => sum + value, 0) / frameTimesMs.length;
  return {
    pair: 'fog', state: 'off', preset: 'night', features: {}, width: 1280, height: 720,
    warmup: 30, frames: 300, frameTimesMs, fps: 1000 / mean,
    p50Ms: nearest(.5), p95Ms: nearest(.95), p99Ms: nearest(.99), ...overrides,
  };
}

function sequenceClock(sequence, start = START) {
  let now = start;
  let index = 0;
  const record = { requested: 0, canceled: [], timestamps: [] };
  return {
    record,
    advance(ms) { now += ms; },
    clock: {
      now: () => now,
      requestFrame(callback) {
        const handle = ++record.requested;
        if (index >= sequence.length) throw new Error(`unexpected frame ${index + 1}`);
        now += sequence[index++];
        record.timestamps.push(now);
        callback(now);
        return handle;
      },
      cancelFrame(handle) { record.canceled.push(handle); },
    },
  };
}

function fixedClock(at) {
  const record = { requested: 0, canceled: [] };
  return {
    record,
    clock: {
      now: () => at,
      requestFrame() { record.requested += 1; return record.requested; },
      cancelFrame(handle) { record.canceled.push(handle); },
    },
  };
}

test('collects 30 warm-up frames and exactly 300 measured intervals', async () => {
  const sequence = steps();
  const { clock, record } = sequenceClock(sequence);
  const rendered = [];
  const measurement = await collectFeatureMeasurement(now => rendered.push(now), { ...SETTINGS, deadlineMs: DEADLINE }, clock);
  assert.equal(record.requested, 330);
  assert.deepEqual(rendered, record.timestamps);
  assert.deepEqual(measurement.frameTimesMs, sequence.slice(30));
  assert.equal(measurement.pair, 'fog');
  assert.equal(measurement.state, 'on');
  assert.equal(measurement.preset, 'day');
  assert.deepEqual(measurement.features, SETTINGS.features);
  assert.equal(measurement.width, 1280);
  assert.equal(measurement.height, 720);
  assert.equal(measurement.warmup, 30);
  assert.equal(measurement.frames, 300);
  assert.equal(measurement.fps, 1000 / 150.5);
  assert.equal(measurement.p50Ms, 150);
  assert.equal(measurement.p95Ms, 285);
  assert.equal(measurement.p99Ms, 297);
  assert.equal(validateFeatureMeasurement(measurement), measurement);
});

test('nearest-rank percentiles select ceil(p*N)-1', () => {
  const split = [...Array(150).fill(1), ...Array(150).fill(2)];
  const measurement = build(split);
  assert.equal(measurement.p50Ms, 1);
  assert.equal(measurement.p95Ms, 2);
  assert.equal(validateFeatureMeasurement(measurement), measurement);
  const distinct = build(Array.from({ length: 300 }, (_, index) => index + 1));
  assert.equal(distinct.p50Ms, 150);
  assert.equal(distinct.p95Ms, 285);
  assert.equal(distinct.p99Ms, 297);
  assert.throws(() => validateFeatureMeasurement({ ...measurement, p50Ms: 2 }), /p50Ms/);
  assert.throws(() => validateFeatureMeasurement({ ...distinct, p95Ms: 286 }), /p95Ms/);
});

test('an already expired deadline rejects immediately without frames', async () => {
  for (const deadlineMs of [START, START - 1]) {
    const { clock, record } = fixedClock(START);
    await assert.rejects(collectFeatureMeasurement(() => {}, { ...SETTINGS, deadlineMs }, clock), /expired/);
    assert.equal(record.requested, 0);
  }
});

test('the shared deadline rejects during warm-up after the first state frame', async () => {
  const { clock } = sequenceClock(Array(400).fill(1000));
  let renders = 0;
  await assert.rejects(collectFeatureMeasurement(() => { renders += 1; }, { ...SETTINGS, deadlineMs: START + 1500 }, clock), /deadline expired/);
  assert.equal(renders, 1);
});

test('paired states reuse one deadline instead of starting a fresh budget', async () => {
  const { clock, advance, record } = sequenceClock(Array(330).fill(1));
  const shared = START + 1000;
  const on = await collectFeatureMeasurement(() => {}, { ...SETTINGS, deadlineMs: shared }, clock);
  assert.equal(on.state, 'on');
  advance(shared - clock.now() + 1);
  await assert.rejects(collectFeatureMeasurement(() => {}, { ...SETTINGS, state: 'off', deadlineMs: shared }, clock), /expired/);
  assert.equal(record.requested, 330);
});

test('a missing warm-up frame expires at the remaining deadline and cancels its callback', async () => {
  const { clock, record } = fixedClock(START);
  await assert.rejects(collectFeatureMeasurement(() => {}, { ...SETTINGS, deadlineMs: START + 20 }, clock), /missing warm-up frame 1/);
  assert.equal(record.requested, 1);
  assert.deepEqual(record.canceled, [1]);
});

test('a missing measured frame rejects without a partial measurement', async () => {
  let now = START;
  let requested = 0;
  const canceled = [];
  const clock = {
    now: () => now,
    requestFrame(callback) {
      requested += 1;
      if (requested <= 30) { now += 0.01; callback(now); }
      return requested;
    },
    cancelFrame(handle) { canceled.push(handle); },
  };
  const rendered = [];
  await assert.rejects(collectFeatureMeasurement(nowMs => rendered.push(nowMs), { ...SETTINGS, deadlineMs: START + 30 }, clock), /missing measured frame 1/);
  assert.equal(rendered.length, 30);
  assert.equal(requested, 31);
  assert.deepEqual(canceled, [31]);
});

test('render failures reject with the originating error during warm-up and measurement', async () => {
  for (const failAt of [3, 34]) {
    const failure = new Error(`render failed at frame ${failAt}`);
    const { clock } = sequenceClock(steps());
    let calls = 0;
    await assert.rejects(
      collectFeatureMeasurement(() => { calls += 1; if (calls === failAt) throw failure; }, { ...SETTINGS, deadlineMs: DEADLINE }, clock),
      error => error === failure,
    );
    assert.equal(calls, failAt);
  }
});

test('zero, reversed and nonfinite frame timestamps reject without filling', async () => {
  const mutations = [
    timestamps => { timestamps[30] = timestamps[29]; },
    timestamps => { timestamps[40] = timestamps[39]; },
    timestamps => { timestamps[100] = Number.NaN; },
    timestamps => { timestamps[200] = timestamps[199] - 1; },
  ];
  for (const mutate of mutations) {
    const timestamps = Array.from({ length: 330 }, (_, index) => START + index);
    mutate(timestamps);
    let index = 0;
    const clock = {
      now: () => START,
      requestFrame(callback) { const handle = ++index; callback(timestamps[index - 1]); return handle; },
      cancelFrame() {},
    };
    await assert.rejects(collectFeatureMeasurement(() => {}, { ...SETTINGS, deadlineMs: DEADLINE }, clock), /positive finite/);
  }
});

test('settings, render and clock are required and feature values stay bounded', async () => {
  const { clock } = sequenceClock(steps());
  const base = { ...SETTINGS, deadlineMs: DEADLINE };
  for (const bad of [null, [], {}, { ...base, pair: ' ' }, { ...base, state: 'maybe' },
    { ...base, preset: '' }, { ...base, features: null }, { ...base, features: { fog: NaN } },
    { ...base, features: { fog: Infinity } }, { ...base, features: { fog: { enabled: true } } },
    { ...base, deadlineMs: NaN }]) {
    await assert.rejects(collectFeatureMeasurement(() => {}, bad, clock), /feature measure/);
  }
  await assert.rejects(collectFeatureMeasurement(null, base, clock), /render must be a function/);
  await assert.rejects(collectFeatureMeasurement(() => {}, base, null), /clock must provide/);
});

test('validateFeatureMeasurement rejects malformed results and wrong interval counts', () => {
  const valid = build(shuffledMeasured());
  assert.equal(validateFeatureMeasurement(valid), valid);
  const mutations = [
    ['missing intervals', m => { delete m.frameTimesMs; }],
    ['truncated', m => { m.frameTimesMs.pop(); }],
    ['filled', m => { m.frameTimesMs.push(1); }],
    ['zero', m => { m.frameTimesMs[5] = 0; }],
    ['negative', m => { m.frameTimesMs[5] = -4; }],
    ['nan', m => { m.frameTimesMs[5] = NaN; }],
    ['infinity', m => { m.frameTimesMs[5] = Infinity; }],
    ['string', m => { m.frameTimesMs[5] = '16'; }],
    ['width', m => { m.width = 1920; }],
    ['height', m => { m.height = 1080; }],
    ['warmup', m => { m.warmup = 29; }],
    ['frames', m => { m.frames = 299; }],
    ['fps', m => { m.fps += 1; }],
    ['p50', m => { m.p50Ms = 1; }],
    ['p95', m => { m.p95Ms = 1; }],
    ['p99', m => { m.p99Ms = 1; }],
    ['state', m => { m.state = 'maybe'; }],
    ['pair', m => { m.pair = ' '; }],
    ['preset', m => { m.preset = ''; }],
    ['features', m => { m.features = null; }],
    ['feature value', m => { m.features = { fog: NaN }; }],
    ['feature object', m => { m.features = { fog: {} }; }],
  ];
  for (const [name, mutate] of mutations) {
    const value = structuredClone(valid);
    mutate(value);
    assert.throws(() => validateFeatureMeasurement(value), /feature measure/, name);
  }
  for (const bad of [null, [], 'measurement']) assert.throws(() => validateFeatureMeasurement(bad), /feature measure/);
  assert.throws(() => validateFeatureMeasurement(build(Array(300).fill(Number.MIN_VALUE))), /fps/);
  assert.throws(() => validateFeatureMeasurement(build(Array(300).fill(Number.MAX_VALUE))), /fps/);
});
