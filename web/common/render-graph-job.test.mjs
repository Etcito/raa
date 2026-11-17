import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { report, storeRoot, STAGES, validateMeasurement } from './render-graph-job.mjs';

delete process.env.NMC_CACHE_DIR;

test('STAGES are the nine render-graph stages in rank order', () => {
  assert.deepEqual(STAGES, ['interpreter', 'views', 'parameters', 'wiring', 'slots',
    'gi-stages', 'post-stages', 'compute', 'frozen']);
});

test('storeRoot is NMC_CACHE_DIR/render-graph/<stage>, the stage root otherwise', (t) => {
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), 'rg-job-'));
  t.after(() => { delete process.env.NMC_CACHE_DIR; fs.rmSync(cache, { recursive: true, force: true }); });
  const out = 'out/render-graph/interpreter';
  assert.equal(storeRoot(out, 'interpreter'), path.resolve(out));
  process.env.NMC_CACHE_DIR = cache;
  assert.equal(storeRoot(out, 'interpreter'), path.join(path.resolve(cache), 'render-graph', 'interpreter'));
});

test('validateMeasurement refuses every record', () => {
  assert.throws(() => validateMeasurement({}),
    (error) => error.message === 'render-graph-job: the render-graph page takes no measurement');
});

test('report lines of check and render', () => {
  const inventory = [{ id: 'a', filename: 'a.png' }, { id: 'b', filename: 'b.png' }];
  assert.deepEqual(report('check', 'interpreter', {}, inventory),
    ['a a.png', 'b b.png', 'render-graph-job check: ok (interpreter, 2 rows)']);
  assert.deepEqual(report('render', 'interpreter', { kept: ['a'], images: ['a', 'b'] }, inventory),
    ['render-graph-job render: kept 1, rendered 1', 'render-graph-job render: ok (interpreter, 2 rows)']);
});
