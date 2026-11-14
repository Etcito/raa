import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parseArgs, storeRoot, STAGES, validateGiMeasurement } from './gi-job.mjs';

delete process.env.NMC_CACHE_DIR;

const PAGE = 'dist/global-illumination';

test('STAGES and parseArgs follow the global-illumination stages and usage', () => {
  assert.deepEqual(STAGES, ['volume', 'lighting', 'virtual']);
  const args = ['check', '--page', PAGE, '--out', 'out/global-illumination/volume', '--stage', 'volume'];
  assert.deepEqual(parseArgs(args),
    { action: 'check', page: PAGE, out: 'out/global-illumination/volume', stage: 'volume' });
  assert.throws(() => parseArgs(['check', '--page', PAGE, '--stage', 'bake']),
    (error) => error.message === 'invalid stage: bake');
  assert.throws(() => parseArgs(['check', '--stage', 'volume']), /usage: node web\/common\/gi-job\.mjs/);
});

test('storeRoot follows NMC_CACHE_DIR and the resolved out otherwise', (t) => {
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), 'gi-job-'));
  t.after(() => { delete process.env.NMC_CACHE_DIR; fs.rmSync(cache, { recursive: true, force: true }); });
  const out = 'out/global-illumination/volume';
  assert.equal(storeRoot(out, 'volume'), path.resolve(out));
  process.env.NMC_CACHE_DIR = cache;
  assert.equal(storeRoot(out, 'volume'), path.join(path.resolve(cache), 'global-illumination', 'volume'));
});

/** A valid PAIR_RUN record: 300 constant 16 ms intervals, so every statistic is exactly 16 ms. */
function record(extra = {}) {
  return {
    pair: 'volume-voxels', state: 'on', preset: 'volume-voxels', features: { gi: true },
    width: 1280, height: 720, warmup: 30, frames: 300,
    frameTimesMs: Array(300).fill(16), fps: 62.5, p50Ms: 16, p95Ms: 16, p99Ms: 16,
    quality: { resolution: 96, levels: 5, probeRays: 8 }, allocatedBytes: 148635648,
    ...extra,
  };
}

test('validateGiMeasurement accepts the quality record and refuses the others', () => {
  const ok = record();
  assert.equal(validateGiMeasurement(ok), ok);
  const refused = [
    [{ quality: { resolution: 80, levels: 5, probeRays: 8 } }, 'quality.resolution'],
    [{ quality: { resolution: 96, levels: 2, probeRays: 8 } }, 'quality.levels'],
    [{ quality: { resolution: 96, levels: 5, probeRays: 7 } }, 'quality.probeRays'],
    [{ quality: undefined }, 'quality'],
    [{ allocatedBytes: -1 }, 'allocatedBytes'],
    [{ allocatedBytes: 1.5 }, 'allocatedBytes'],
  ];
  for (const [change, field] of refused) {
    assert.throws(() => validateGiMeasurement(record(change)),
      (error) => error.message.startsWith(`gi-job: measurement.${field}`), field);
  }
});
