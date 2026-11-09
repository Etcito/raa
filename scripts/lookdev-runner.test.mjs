// node --test scripts/lookdev-runner.test.mjs: the shared lookdev runner timing contract,
// without a browser or GPU. requestAnimationFrame is a deterministic fixture here.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  FRAMES, WARMUP, measureAnimationFrames, percentile, runScene, summarizeFrameIntervals,
} from '../examples/lookdev/lib/runner.js';

const DEFAULT_KEYS = [
  'adapter', 'hardware', 'canvas', 'nonEmpty', 'firstFrameMs', 'warmup', 'frames',
  'cpuMsMedian', 'cpuMsP95', 'instances', 'drawCalls', 'triangles', 'allocsPerFrameTotal',
  'resize', 'scene', 'variant', 'passes',
];
const TIMING_KEYS = ['frameIntervalMsMedian', 'frameIntervalMsP95', 'fpsMean', 'fpsMedian', 'fpsP95'];

function close(actual, expected) {
  assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`);
}

function harness({ onRender } = {}) {
  const calls = { render: 0, frames: [] };
  const engine = {
    resize() {},
    render(timeMs) { calls.render++; if (onRender) onRender(timeMs); },
    setPassEnabled() {},
    setPassOrder() {},
    async probeNonEmpty() { return true; },
    stats() {
      return {
        adapter: 'fixture-adapter', hardware: true, firstFrameMs: 1,
        cpuMsMedian: 2, cpuMsP95: 3, instances: 4, drawCalls: 5, triangles: 6,
      };
    },
    passes() { return []; },
  };
  const raa = { async createEngine() { return engine; } };
  const canvas = { style: {} };
  const scene = {
    id: 'fixture-scene',
    async build() {},
    frame(engine, timeMs) { calls.frames.push(timeMs); },
  };
  const variant = { id: 'fixture-variant' };
  return { calls, raa, canvas, scene, variant };
}

const originalRaf = globalThis.requestAnimationFrame;

function withRaf(t, timestamps) {
  const queue = [...timestamps];
  globalThis.requestAnimationFrame = (callback) => {
    assert.ok(queue.length > 0, 'requestAnimationFrame called more than the fixture allows');
    callback(queue.shift());
  };
  t.after(() => {
    if (originalRaf === undefined) delete globalThis.requestAnimationFrame;
    else globalThis.requestAnimationFrame = originalRaf;
  });
}

test('nearest-rank percentiles use ceil(p*n)-1 on ascending values', () => {
  const ten = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  assert.equal(percentile(ten, 0.5), 5);
  assert.equal(percentile(ten, 0.95), 10);
  const hundred = Array.from({ length: 100 }, (_, i) => i);
  assert.equal(percentile(hundred, 0.95), 94);
  const threeHundred = Array.from({ length: 300 }, (_, i) => i);
  assert.equal(percentile(threeHundred, 0.95), 284);
});

test('interval arithmetic and FPS percentiles follow the contract', () => {
  const timing = summarizeFrameIntervals([10, 20, 30, 40]);
  close(timing.frameIntervalMsMedian, 20);
  close(timing.frameIntervalMsP95, 40);
  close(timing.fpsMean, 40);
  close(timing.fpsMedian, 1000 / 30);
  close(timing.fpsP95, 100);
  const values = Object.values(timing);
  assert.ok(values.every((value) => Number.isFinite(value)));
});

test('missing and nonpositive intervals reject instead of falling back', () => {
  for (const intervals of [[], [0], [-5], [Number.NaN], [Infinity], [undefined], [10, 0]]) {
    assert.throws(() => summarizeFrameIntervals(intervals), /frame intervals/);
  }
});

test('omitting measure keeps the default capture and the existing keys', async () => {
  const { calls, raa, canvas, scene, variant } = harness();
  assert.equal(globalThis.requestAnimationFrame, undefined, 'the default path needs no rAF');
  const result = await runScene(raa, canvas, scene, variant);
  assert.deepEqual(Object.keys(result).sort(), [...DEFAULT_KEYS].sort());
  assert.equal(result.warmup, WARMUP);
  assert.equal(result.frames, FRAMES);
  assert.equal(calls.render, WARMUP + FRAMES);
  assert.equal(result.cpuMsMedian, 2);
  assert.equal(result.cpuMsP95, 3);
  for (const key of TIMING_KEYS) assert.ok(!(key in result), `${key} must be absent`);
});

test('measured mode excludes warm-up and schedules exactly frames intervals', async (t) => {
  const { calls, raa, canvas, scene, variant } = harness();
  // One boundary timestamp, then one timestamp per measured frame.
  withRaf(t, [100, 110, 130, 150, 180]);
  const result = await runScene(raa, canvas, scene, variant, { measure: { warmup: 2, frames: 4 } });
  assert.equal(result.warmup, 2);
  assert.equal(result.frames, 4);
  assert.equal(calls.render, 2 + 4, 'warm-up renders plus measured renders');
  close(result.frameIntervalMsMedian, 20);
  close(result.frameIntervalMsP95, 30);
  close(result.fpsMean, 4 / 0.08);
  close(result.fpsMedian, 50);
  close(result.fpsP95, 100);
  assert.equal(result.cpuMsMedian, 2);
  assert.equal(result.cpuMsP95, 3);
  for (const key of DEFAULT_KEYS) assert.ok(key in result, `${key} is retained`);
});

test('measured mode rejects a missing or nonpositive timestamp', async (t) => {
  const missing = harness();
  withRaf(t, [100, undefined, 120, 140]);
  await assert.rejects(
    runScene(missing.raa, missing.canvas, missing.scene, missing.variant,
      { measure: { warmup: 1, frames: 2 } }),
    /frame intervals/,
  );

  const reversed = harness();
  withRaf(t, [100, 100, 90, 80]);
  await assert.rejects(
    runScene(reversed.raa, reversed.canvas, reversed.scene, reversed.variant,
      { measure: { warmup: 1, frames: 2 } }),
    /frame intervals/,
  );
});

test('measured mode requires requestAnimationFrame and never uses a timer', async () => {
  const { raa, canvas, scene, variant } = harness();
  assert.equal(globalThis.requestAnimationFrame, undefined);
  await assert.rejects(
    runScene(raa, canvas, scene, variant, { measure: { warmup: 1, frames: 1 } }),
    /requires requestAnimationFrame/,
  );
});

test('measure counts must be positive integers', async () => {
  for (const measure of [
    { warmup: 0, frames: 1 }, { warmup: 1, frames: 0 }, { warmup: -1, frames: 1 },
    { warmup: 1.5, frames: 1 }, { warmup: '2', frames: 1 }, { warmup: 1, frames: 3.2 },
  ]) {
    const { raa, canvas, scene, variant } = harness();
    await assert.rejects(runScene(raa, canvas, scene, variant, { measure }), /positive integer/);
  }
});

test('measureAnimationFrames returns the same summary for a fixed schedule', async (t) => {
  const { raa, canvas, scene, variant } = harness();
  withRaf(t, [0, 16, 32, 48]);
  const engine = await raa.createEngine(canvas);
  const timing = await measureAnimationFrames(engine, scene, variant, 1000, 3);
  assert.deepEqual(Object.keys(timing).sort(), [...TIMING_KEYS].sort());
  close(timing.frameIntervalMsMedian, 16);
  close(timing.fpsMean, 3 / 0.048);
});
