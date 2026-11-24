import assert from 'node:assert/strict';
import test from 'node:test';
import { createLiveStats } from './lib/live-stats.js';

test('FPS uses elapsed wall time and the submitted frame count across skipped updates', () => {
  let frames = 7;
  let reads = 0;
  const engine = { stats() { reads++; return { frames }; } };
  const container = {};
  const { update, reset } = createLiveStats(container);

  update(engine, 1000, 1280, 720);
  assert.match(container.textContent, /Rendered FPS —/);
  frames = 10;
  update(engine, 1250, 1280, 720);
  assert.equal(reads, 1);
  frames = 22;
  update(engine, 1600, 1280, 720);
  assert.match(container.textContent, /Rendered FPS 25\.0.*Submitted frames 22/);

  update(engine, 1700, 640, 360);
  assert.match(container.textContent, /Rendered FPS 25\.0.*Output 640 × 360/);
  frames = 34;
  update(engine, 2200, 640, 360);
  assert.match(container.textContent, /Rendered FPS 20\.0/);

  const replacement = { stats: () => ({ frames: 100 }) };
  update(replacement, 2250, 640, 360);
  assert.match(container.textContent, /Rendered FPS —.*Submitted frames 100/);
  reset();
  assert.equal(container.textContent, '');
  update(replacement, 3000, 640, 360);
  assert.match(container.textContent, /Rendered FPS —/);
});

test('reports actual engine counters and current internal texture dimensions', () => {
  const container = {};
  const { update } = createLiveStats(container);
  const image = (pass, port, width, height, layers = 1, mips = 1) =>
    ({ pass, port, width, height, layers, mips });
  let images = [
    image('gi.screen', 'probes', 41, 23),
    image('gi.screen', 'pyramid', 641, 361, 1, 10),
    image('gi.area-shadow', 'mask', 641, 361, 2),
    image('tonemap', 'color', 641, 361),
  ];
  const engine = {
    stats: () => ({ frames: 1, cpuMsMedian: 2.1, cpuMsP95: 3.47, instances: 12, drawCalls: 8, triangles: 321 }),
    giStats: () => ({ active: true, pendingRebuild: true, allocatedBytes: 2 * 1024 * 1024 }),
    passImageExtents: () => images,
  };

  update(engine, 0, 641, 361);
  assert.match(container.textContent, /CPU frame median 2\.10 ms.*p95 3\.47 ms/);
  assert.match(container.textContent, /Instances 12.*Draw calls 8.*Triangles 321/);
  assert.match(container.textContent, /GI volume rebuilding.*GI memory 2\.0 MiB/);
  assert.match(container.textContent, /gi\.screen +probes 41 × 23 · pyramid 641 × 361 \(10 mips\)/);
  assert.match(container.textContent, /gi\.area-shadow +mask 641 × 361 × 2/);
  assert.doesNotMatch(container.textContent, /tonemap/);

  images = [];
  update(engine, 500, 641, 361);
  assert.doesNotMatch(container.textContent, /Internal pass textures|gi\.screen/);
});

test('GPU timings distinguish unsupported, pending, sampled and invalid timestamp data', () => {
  const container = {};
  const { update } = createLiveStats(container);
  let timings = { status: 'unavailable', reason: 'timestamp queries unsupported' };
  let reads = 0;
  const engine = {
    stats: () => ({ frames: 1, cpuMsMedian: 2.1 }),
    passGpuTimings: () => { reads++; return timings; },
  };

  update(engine, 0, 1280, 720);
  assert.match(container.textContent, /GPU timings unavailable \(timestamp queries unsupported\)/);
  timings = { status: 'pending' };
  update(engine, 250, 1280, 720);
  assert.equal(reads, 1);
  update(engine, 500, 1280, 720);
  assert.match(container.textContent, /GPU timings pending/);

  timings = {
    status: 'sample', frame: 17, width: 1280, height: 720,
    passes: [
      { pass: 'gi.area-shadow', ms: 2.3456 },
      { pass: 'tonemap', ms: 0 },
      { pass: 'forward', ms: null },
    ],
  };
  update(engine, 1000, 640, 360);
  assert.match(container.textContent, /Output 640 × 360/);
  assert.match(container.textContent, /GPU timings · frame 17 · 1280 × 720/);
  assert.match(container.textContent, /gi\.area-shadow +2\.346 ms\ntonemap +0\.000 ms\nforward +unavailable/);

  timings.passes = [{ pass: 'gi.area-shadow', ms: null }];
  update(engine, 1500, 640, 360);
  assert.match(container.textContent, /GPU timings unavailable \(invalid timestamp data\)/);
  update({ stats: engine.stats }, 1600, 640, 360);
  assert.match(container.textContent, /GPU timings unavailable \(timestamp data unavailable\)/);
});
