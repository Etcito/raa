import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { constants, runInNewContext } from 'node:vm';
import test from 'node:test';
import { variantProblems } from '../../web/common/lights-job.mjs';
import * as geometry from '../lookdev/lib/geometry.js';
import { runScene, renderAt, variantTimeMs } from '../lookdev/lib/runner.js';
import attached from './scenes/attached-light.js';

const names = ['initial', 'translated', 'rotated', 'reattached', 'detached',
  'reattached-before-delete', 'parent-deleted', 'light-removed', 'attached-final'];
const read = (file) => readFileSync(new URL(file, import.meta.url), 'utf8');
const plain = (value) => JSON.parse(JSON.stringify(value));
const rounded = (values) => values.map((v) => Math.round(v * 1e6) / 1e6 || 0);

// Public Engine test double, following point.test.mjs; render records world state.
function fakeEngine(canvas) {
  let next = 0;
  const nodes = new Map();
  const instances = new Map();
  const lights = new Map();
  const calls = [];
  const frames = [];
  const world = (light) => {
    if (!light.parent) return plain(light);
    const { t, r } = nodes.get(instances.get(light.parent));
    const angle = 2 * Math.atan2(r[1], r[3]);
    const rotate = ([x, y, z]) => [x * Math.cos(angle) + z * Math.sin(angle), y,
      -x * Math.sin(angle) + z * Math.cos(angle)];
    return { position: rotate(light.offset).map((v, i) => v + t[i]),
      direction: rotate(light.direction).map((v) => v / Math.hypot(...light.direction)) };
  };
  return {
    calls, frames,
    addMaterial: () => ++next,
    addMesh: () => ++next,
    addNode(_parent, t, r) { const id = ++next; nodes.set(id, { t, r }); return id; },
    addMeshInstance(node) { const id = ++next; instances.set(id, node); return id; },
    setTransform: (node, t, r) => nodes.set(node, { t, r }),
    addLight(desc) {
      calls.push('add');
      const id = ++next;
      lights.set(id, plain(desc));
      return id;
    },
    attachLight(id, parent, offset) {
      assert.ok(instances.has(parent), 'attachment uses an instance, not a node');
      calls.push('attach');
      Object.assign(lights.get(id), { parent, offset });
    },
    detachLight(id) { calls.push('detach'); lights.set(id, world(lights.get(id))); },
    removeNode(node) {
      calls.push('delete-parent');
      for (const [id, light] of lights) {
        if (instances.get(light.parent) === node) lights.set(id, world(light));
      }
      for (const [id, owner] of instances) if (owner === node) instances.delete(id);
      nodes.delete(node);
    },
    removeLight(id) { calls.push('remove'); assert.ok(lights.delete(id)); },
    lightStats: () => ({ activeLights: lights.size,
      attachedLights: [...lights.values()].filter((light) => light.parent).length }),
    render() {
      const frame = [...lights.values()].map(world);
      for (const light of frame) assert.deepEqual(rounded(light.position), rounded([...nodes.values()].at(-1).t));
      frames.push(frame);
    },
    setCamera() {}, setBackground() {}, setLight() {}, setAmbient() {}, setVisible() {},
    resize: (width, height) => Object.assign(canvas, { width, height }),
    probeNonEmpty: async () => true,
    stats: () => ({}), passes: () => [],
    dispose: () => calls.push('dispose'),
  };
}

// Execute the actual page and shared envelope as in page.test.mjs and motion.test.mjs.
function page() {
  const engines = [];
  const canvas = { style: {}, toDataURL: () => 'data:image/png;base64,cG5n' };
  const context = {
    window: {}, navigator: {}, geometry, objects: {}, runScene, renderAt, variantTimeMs,
    document: { getElementById: (id) => id === 'c' ? canvas : {} },
    fetch: async () => ({ ok: true, json: async () => [{ id: attached.id, file: 'scenes/attached-light.js' }] }),
    loadRaa: async () => ({ createEngine: async () => {
      const engine = fakeEngine(canvas);
      engines.push(engine);
      return engine;
    } }),
  };
  runInNewContext(read('../../web/common/nmc-gpu.js').replace('export function', 'function')
    + '\n' + read('./main.js').replace(/^import .*;\n/gm, ''), context, {
    filename: fileURLToPath(new URL('./main.js', import.meta.url)),
    importModuleDynamically: constants.USE_MAIN_CONTEXT_DEFAULT_LOADER,
  });
  return { gpu: context.window.nmcGpu, engines };
}

test('attached-light selects visible surface, parents and source for on/off', () => {
  assert.equal(attached.id, 'attached-light');
  assert.deepEqual(attached.variants.map((v) => v.id), ['on', 'off']);
  for (const variant of attached.variants) {
    assert.deepEqual(variantProblems(variant), []);
    assert.equal(variant.selection.subjects.length, 3);
    assert.match(variant.description, /Local lights cast no shadow yet/);
  }
});

test('page returns static lifecycle PNGs and checks counts without duplicate attachments', async () => {
  const { gpu, engines } = page();
  for (const variant of attached.variants) {
    const result = await gpu.ops.render({ args: { scene: attached.id, variant: variant.id } });
    assert.deepEqual(plain(result.files.map((f) => f.name)), ['capture.png', ...names.map((n) => `${n}.png`)]);
    assert.ok(result.files.every((f) => f.base64 === 'cG5n'));
    assert.deepEqual(plain(result.json.lifecycle.frames.map((f) => f.name)), names);
    const engine = engines.at(-1);
    const frames = engine.frames.slice(-names.length);
    const counts = result.json.lifecycle.frames.map(({ stats }) => [stats.activeLights, stats.attachedLights]);
    if (variant.id === 'off') {
      assert.deepEqual(plain(counts), names.map(() => [0, 0]));
      assert.ok(frames.every((f) => f.length === 0));
      continue;
    }
    assert.deepEqual(plain(counts), [[1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 1], [1, 0], [0, 0], [1, 1]]);
    assert.deepEqual(engine.calls, ['add', 'attach', 'attach', 'detach', 'attach', 'delete-parent', 'remove', 'add', 'attach']);
    assert.deepEqual(frames.slice(0, 4).map((f) => rounded(f[0].position)),
      [[-0.7, 1.15, 0], [0.8, 1.15, 0], [0, 1.15, -0.8], [1.5, 1.15, 0]]);
    assert.deepEqual(rounded(frames[2][0].direction), [0, -0.707107, -0.707107]);
    assert.deepEqual(rounded(frames[3][0].direction), [0.707107, -0.707107, 0]);
    assert.deepEqual(frames[4][0].position, frames[3][0].position);
    assert.deepEqual(frames[4][0].direction, frames[3][0].direction);
    assert.deepEqual(frames[6][0], frames[5][0]);
    assert.deepEqual(frames[8], frames[2]);
  }
  assert.equal(engines[0].calls.at(-1), 'dispose');
});

test('a count mismatch rejects the page operation before captures are published', async () => {
  const { gpu } = page();
  const prepare = attached.prepareCapture;
  try {
    attached.prepareCapture = function (engine, variant, options) {
      engine.lightStats = () => ({ activeLights: 2, attachedLights: 1 });
      return prepare.call(this, engine, variant, options);
    };
    await assert.rejects(gpu.ops.render({ scene: attached.id, variant: 'on' }), /initial: unexpected light counts/);
  } finally {
    attached.prepareCapture = prepare;
  }
});
