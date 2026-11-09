import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { constants, runInNewContext } from 'node:vm';
import test from 'node:test';
import { variantProblems } from '../../web/common/lights-job.mjs';
import * as geometry from '../lookdev/lib/geometry.js';
import { runScene, renderAt, variantTimeMs } from '../lookdev/lib/runner.js';
import motion from './scenes/motion.js';

const read = (file) => readFileSync(new URL(file, import.meta.url), 'utf8');
const plain = (value) => JSON.parse(JSON.stringify(value));

function fakeEngine(canvas = {}) {
  let next = 0;
  let views = 1;
  const nodes = new Map();
  const calls = [];
  return {
    calls, nodes,
    addMaterial: () => ++next,
    addMesh: () => ++next,
    addNode(_parent, position) {
      const id = ++next;
      nodes.set(id, position.slice());
      return id;
    },
    addMeshInstance() {},
    removeNode: (id) => nodes.delete(id),
    setTransform: (id, position) => nodes.set(id, position.slice()),
    setCamera() {}, setBackground() {}, setLight() {}, setAmbient() {},
    setPassEnabled: (name, enabled) => calls.push({ op: 'pass', name, enabled }),
    setViewports(rects) {
      views = rects.length;
      calls.push({ op: 'views', rects: plain(rects) });
    },
    setViewportCamera: (index, camera) => calls.push({ op: 'camera', index, camera }),
    resize(width, height) {
      Object.assign(canvas, { width, height });
      calls.push({ op: 'resize', size: [width, height] });
    },
    render(timeMs) {
      calls.push({ op: 'render', timeMs, views, nodes: plain([...nodes.values()]) });
    },
    probeNonEmpty: async () => { calls.push({ op: 'probe' }); return true; },
    stats: () => ({}),
    passes: () => [],
    dispose: () => calls.push({ op: 'dispose' }),
  };
}

// Follow examples/lights/page.test.mjs: execute the actual page and transport in a VM.
function page() {
  const engines = [];
  const canvas = { style: {}, toDataURL: () => 'data:image/png;base64,cG5n' };
  const status = {};
  const context = {
    window: {}, navigator: {}, geometry, objects: {}, runScene, renderAt, variantTimeMs,
    document: { getElementById: (id) => id === 'c' ? canvas : status },
    fetch: async () => ({ ok: true, json: async () => [{ id: motion.id, file: 'scenes/motion.js' }] }),
    loadRaa: async () => ({
      createEngine: async () => {
        const engine = fakeEngine(canvas);
        engines.push(engine);
        return engine;
      },
    }),
  };
  const source = read('./main.js').replace(/^import .*;\n/gm, '');
  const transport = read('../../web/common/nmc-gpu.js').replace('export function', 'function');
  runInNewContext(transport + '\n' + source, context, {
    filename: fileURLToPath(new URL('./main.js', import.meta.url)),
    importModuleDynamically: constants.USE_MAIN_CONTEXT_DEFAULT_LOADER,
  });
  return { gpu: context.window.nmcGpu, engines, status };
}

test('motion selection covers every captured subject and documents the signed legend', () => {
  assert.equal(motion.id, 'motion');
  assert.deepEqual(motion.variants.map((variant) => variant.id), ['on', 'off']);
  for (const variant of motion.variants) {
    assert.deepEqual(variantProblems(variant), []);
    assert.equal(variant.selection.subjects.length, 6);
    assert.match(variant.description, /[Ll]ocal lights cast no shadow yet/);
  }
  assert.match(motion.variants[0].description, /Stationary floor.*newly introduced.*zero motion/);
  assert.match(motion.variants[1].description, /Ordinarily shaded.*diagnostic is disabled/);
  for (const signed of ['+X right: red', '-X left: cyan', '+Y down: green', '-Y up: magenta']) {
    assert.ok(motion.legend.includes(signed));
  }
});

test('submitted frames exercise resize and two views before first-frame sphere capture', async () => {
  const engine = fakeEngine();
  const variant = motion.variants[0];
  const lib = { geometry, pbr: (e, color, opts) => e.addMaterial({ color, ...opts }) };
  await motion.build(engine, lib, variant);
  const lifecycle = await motion.prepareCapture(engine, variant, { width: 1280, height: 720 });
  const frames = engine.calls.filter((call) => call.op === 'render');
  assert.deepEqual(frames.map((f) => f.timeMs), [0, 16, 32, 48, 64, 80, 96, 112, 128]);
  assert.deepEqual(frames.map((f) => f.views), [1, 1, 2, 2, 2, 2, 1, 1, 1]);
  assert.deepEqual(engine.calls.filter((c) => c.op === 'resize').map((c) => c.size), [[640, 360], [1280, 720]]);
  assert.equal(engine.calls.find((c) => c.op === 'camera').index, 1);
  assert.ok(lifecycle.frames.slice(0, -1).every((f) => !f.spawned));
  assert.equal(lifecycle.frames.at(-1).spawned, true);
  assert.ok(frames.slice(0, -1).every((f) => f.nodes.length === 5));
  assert.equal(frames.at(-1).nodes.length, 6);
  assert.ok(frames.every((f) => JSON.stringify(f.nodes[0]) === '[0,0,0]'));
  const previous = frames.at(-2).nodes;
  const current = frames.at(-1).nodes;
  const deltas = current.slice(1, 5).map((p, i) => p.map((v, k) => +(v - previous[i + 1][k]).toFixed(6)));
  assert.deepEqual(deltas, [[0.6, 0, 0], [-0.6, 0, 0], [0, -0.6, 0], [0, 0.6, 0]]);
  motion.frame(engine, variant.timeMs);
  assert.deepEqual([...engine.nodes.values()], current);
});

test('page captures the final submitted frame once and isolates on/off engine state', async () => {
  const { gpu, engines, status } = page();
  for (const variant of motion.variants) {
    const result = await gpu.ops.render({ args: { scene: motion.id, variant: variant.id } });
    const engine = engines.at(-1);
    assert.equal(result.json.lifecycle.frames.at(-1).timeMs, variant.timeMs);
    assert.deepEqual(plain(result.json.lifecycle.size), [1280, 720]);
    assert.equal(result.files[0].base64, 'cG5n');
    assert.equal(engine.calls.at(-1).op, 'render');
    assert.equal(engine.calls.at(-1).nodes.length, 6);
    const probe = engine.calls.findIndex((c) => c.op === 'probe');
    assert.equal(engine.calls.slice(probe + 1).filter((c) => c.op === 'render').length, 9);
    const passes = engine.calls.filter((c) => c.op === 'pass');
    assert.equal(passes[0].enabled, variant.id === 'on');
    assert.ok(passes.every((c) => c.name === 'motion.display'));
  }
  assert.equal(gpu.ready, true);
  assert.equal(status.textContent, motion.legend);
  assert.equal(engines[0].calls.at(-1).op, 'dispose');
  assert.notEqual(engines[0], engines[1]);
});

test('page propagates lifecycle failures without publishing a completed capture', async () => {
  const { gpu, engines } = page();
  const prepare = motion.prepareCapture;
  try {
    motion.prepareCapture = async () => { throw new Error('lifecycle failed'); };
    await assert.rejects(gpu.ops.render({ scene: motion.id, variant: 'on' }), /lifecycle failed/);
  } finally {
    motion.prepareCapture = prepare;
  }
  const result = await gpu.ops.render({ scene: motion.id, variant: 'off' });
  assert.equal(result.json.variant, 'off');
  assert.equal(engines[0].calls.at(-1).op, 'dispose');
});
