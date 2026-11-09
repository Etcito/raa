// The 64-light preset (S20, RAA-LIGHTS-PAGE, QB8). CPU-only checks: scene metadata,
// the 64-instance/64-light build through the public Engine.addLight API, and the page's
// measured on/off operation with a deterministic runner fixture. No GPU result is
// compared with a CPU one; look acceptance belongs to the managed description judge.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { constants, runInNewContext } from 'node:vm';
import test from 'node:test';
import { variantProblems } from '../../web/common/lights-job.mjs';
import * as geometry from '../lookdev/lib/geometry.js';
import many from './scenes/many-lights.js';

const read = (file) => readFileSync(new URL(file, import.meta.url), 'utf8');
const plain = (value) => JSON.parse(JSON.stringify(value));
const BASE = {
  adapter: {
    vendor: 'nvidia', architecture: 'ada', device: 'RTX 3090',
    description: 'NVIDIA GeForce RTX 3090', isFallbackAdapter: false,
  },
  hardware: true,
  nonEmpty: true,
  frameIntervalMsMedian: 16.6,
  frameIntervalMsP95: 17.2,
  fpsMean: 60.2,
  fpsMedian: 60.1,
  fpsP95: 58.9,
  frameIntervalsMs: Array(300).fill(16.6),
};

// The S17 public light methods plus the page's scene-building calls; the preset must
// create each light through addLight(desc) and never touch the sun setLight path.
function lightEngine() {
  const lights = new Map();
  const calls = { meshes: 0, nodes: 0, instances: 0, positions: [], camera: null };
  return {
    calls,
    localLights: () => [...lights.values()].map((light) => structuredClone(light)),
    addLight(desc) {
      const handle = lights.size + 1;
      lights.set(handle, structuredClone(desc));
      return handle;
    },
    lightStats: () => ({ activeLights: lights.size, attachedLights: 0 }),
    setCamera(camera) { calls.camera = camera; },
    setBackground() {},
    setLight() {},
    setAmbient() {},
    addMaterial: () => 1,
    addMesh() { calls.meshes += 1; return calls.meshes; },
    addNode(_parent, position) { calls.positions.push(position); calls.nodes += 1; return calls.nodes; },
    addMeshInstance() { calls.instances += 1; },
  };
}

const lib = { geometry, pbr: (engine, color, opts = {}) => engine.addMaterial({ ...opts, color, model: 'pbr' }) };

// The runner fixture: main.js must call the landed runScene separately for on/off with
// the scene's measured configuration, and keep both engines resident for the operation.
function measured(fps = { on: BASE.fpsMean, off: BASE.fpsMean }) {
  const calls = [];
  const runScene = async (_raa, _canvas, scene, variant, args) => {
    const engine = { disposed: 0, dispose() { this.disposed += 1; } };
    args.onEngine(engine);
    calls.push({ scene: scene.id, variant: variant.id, args, engine });
    return {
      ...BASE, fpsMean: fps[variant.id], scene: scene.id, variant: variant.id,
      canvas: [args.width, args.height], gpuErrors: [],
    };
  };
  return { calls, runScene };
}

// Execute the actual page and shared envelope as in page.test.mjs and motion.test.mjs.
function page(runScene, gpu) {
  const canvas = { style: {}, toDataURL: () => 'data:image/png;base64,cG5n' };
  const context = {
    window: {},
    navigator: {
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/142.0.0.0 Safari/537.36',
      platform: 'Win32', gpu,
    },
    devicePixelRatio: 1, geometry, objects: {}, runScene, renderAt: () => {}, variantTimeMs: (variant) => variant.timeMs,
    document: { getElementById: (id) => id === 'c' ? canvas : {} },
    fetch: async () => ({ ok: true, json: async () => [{ id: many.id, file: 'scenes/many-lights.js' }] }),
    loadRaa: async () => ({ createEngine: async () => { throw new Error('measured runs must use runScene'); } }),
  };
  const transport = read('../../web/common/nmc-gpu.js').replace('export function', 'function');
  runInNewContext(transport + '\n' + read('./main.js').replace(/^import .*;\n/gm, ''), context, {
    filename: fileURLToPath(new URL('./main.js', import.meta.url)),
    importModuleDynamically: constants.USE_MAIN_CONTEXT_DEFAULT_LOADER,
  });
  return { gpu: context.window.nmcGpu };
}

test('many-lights selects on/off variants with the 1920x1080 measurement configuration', () => {
  assert.equal(many.id, 'many-lights');
  assert.equal(many.milestone, 'm28');
  assert.deepEqual(many.variants.map((variant) => variant.id), ['on', 'off']);
  const config = many.measure;
  assert.deepEqual(
    [config.warmup, config.frames, config.width, config.height, config.devicePixelRatio, config.minFpsMean],
    [60, 300, 1920, 1080, 1, 60],
  );
  assert.match(config.measurementFor, /60 FPS at 1920x1080 with 64 lights on/);
  for (const variant of many.variants) {
    assert.deepEqual(variantProblems(variant), []);
    assert.match(variant.description, /Local lights cast no shadow yet/);
    assert.match(variant.description, /prepass coverage/);
    assert.equal(variant.selection.subjects.length, 65);
  }
  assert.match(many.variants[0].description, /64.*distinct.*point light/);
  assert.match(many.variants[1].description, /no local lights/);
});

test('on adds 64 distinct lights over the 64-instance field and off adds none', async () => {
  const geometries = [];
  for (const variant of many.variants) {
    const engine = lightEngine();
    await many.build(engine, lib, variant);
    assert.equal(engine.calls.meshes, 2, 'one floor mesh and one box mesh');
    assert.equal(engine.calls.nodes, 65, 'one floor node plus 64 field instances');
    assert.equal(engine.calls.instances, 65);
    assert.deepEqual(engine.calls.camera, variant.selection.camera);
    geometries.push(engine.calls.positions);
    const lights = engine.localLights();
    if (variant.id === 'off') {
      assert.equal(lights.length, 0);
      assert.equal(engine.lightStats().activeLights, 0);
      continue;
    }
    assert.equal(lights.length, 64);
    assert.equal(new Set(lights.map((light) => light.position.join(','))).size, 64, 'one light per instance');
    assert.equal(new Set(lights.map((light) => light.color.join(','))).size, 64, 'the 64 lights are distinct');
    for (const light of lights) {
      assert.equal(light.kind, 'point');
      assert.ok(light.intensity > 0 && light.range > 0 && light.range < 6);
    }
    assert.equal(engine.lightStats().activeLights, 64);
  }
  assert.deepEqual(geometries[0], geometries[1], 'off keeps the same instance field');
});

test('the operation returns both measured runs with adapter and browser metadata', async () => {
  const fixture = measured();
  const { gpu } = page(fixture.runScene);
  const result = await gpu.ops.render({ args: { scene: many.id, variant: 'on' } });
  assert.deepEqual(fixture.calls.map((call) => call.variant), ['off', 'on']);
  for (const call of fixture.calls) {
    assert.deepEqual(call.scene, many.id);
    assert.deepEqual(plain(call.args.measure), { warmup: 60, frames: 300 });
    assert.deepEqual([call.args.width, call.args.height], [1920, 1080]);
  }
  const { json, files } = result;
  assert.equal(json.scene, many.id);
  assert.equal(json.variant, 'on');
  assert.deepEqual(Object.keys(json.runs).sort(), ['off', 'on']);
  assert.equal(json.runs.on.fpsMean, 60.2);
  assert.equal(json.runs.on.frameIntervalsMs.length, 300);
  assert.equal(json.runs.off.frameIntervalsMs.length, 300);
  assert.deepEqual(plain(json.adapter), BASE.adapter);
  assert.match(json.browser.userAgent, /Chrome/);
  assert.equal(json.browser.platform, 'Win32');
  assert.equal(json.browser.devicePixelRatio, 1);
  assert.equal(json.measurement_for, many.measure.measurementFor);
  assert.deepEqual(plain(json.gpuErrors), []);
  assert.deepEqual(plain(files.map((file) => file.name)), ['capture.png']);
  const engines = fixture.calls.map((call) => call.engine);
  assert.deepEqual(engines.map((engine) => engine.disposed), [0, 0], 'both devices stay resident');
  const off = await gpu.ops.render({ args: { scene: many.id, variant: 'off' } });
  assert.deepEqual(fixture.calls.slice(-2).map((call) => call.variant), ['on', 'off']);
  assert.equal(off.json.variant, 'off');
  assert.deepEqual(Object.keys(off.json.runs).sort(), ['off', 'on']);
  assert.deepEqual(engines.map((engine) => engine.disposed), [1, 1], 'the next operation replaces them');
});

test('the operation retains a slow measurement for downstream judgment and rejects a wrong canvas', async () => {
  const slow = measured({ on: 59.4, off: 61 });
  const { gpu } = page(slow.runScene);
  for (const variant of ['on', 'off']) {
    const result = await gpu.ops.render({ args: { scene: many.id, variant } });
    assert.equal(result.json.runs.on.fpsMean, 59.4);
  }
  const small = measured();
  const smallRun = async (raa, canvas, scene, variant, args) => ({
    ...await small.runScene(raa, canvas, scene, variant, args), canvas: [1280, 720],
  });
  await assert.rejects(
    page(smallRun).gpu.ops.render({ args: { scene: many.id, variant: 'on' } }),
    /is not 1920x1080/,
  );
});


test('measured on/off runs retain their own creating adapter metadata', async () => {
  let info;
  const browserGpu = { requestAdapter: async () => ({
    info, requestDevice: async () => ({ lost: new Promise(() => {}), addEventListener() {} }),
  }) };
  const fixture = measured();
  const run = async (...args) => {
    const variant = args[3].id;
    info = { vendor: variant === 'on' ? 'nvidia' : 'google', architecture: '',
      device: '', description: '', isFallbackAdapter: variant === 'off' };
    await (await browserGpu.requestAdapter()).requestDevice();
    return fixture.runScene(...args);
  };
  const { json } = await page(run, browserGpu).gpu.ops.render({ args: { scene: many.id, variant: 'on' } });
  assert.equal(json.runs.on.adapter.vendor, 'nvidia');
  assert.equal(json.runs.on.hardware, true);
  assert.equal(json.runs.off.adapter.vendor, 'google');
  assert.equal(json.runs.off.adapter.isFallbackAdapter, true);
  assert.equal(json.runs.off.hardware, false);
});
