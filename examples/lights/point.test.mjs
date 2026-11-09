// Bounded point preset page checks (S20, RAA-LIGHTS-PAGE). The scene is exercised through
// the documented S17 public engine light API only: a small CPU reference of addLight,
// updateLight, removeLight and lightStats enforces the documented atomicity, 256-light
// capacity and pending-readback shapes, so these checks need no GPU and compare no
// CPU result with a GPU render.
import assert from 'node:assert/strict';
import test from 'node:test';
import { variantProblems } from '../../web/common/lights-job.mjs';
import * as geometry from '../lookdev/lib/geometry.js';
import point from './scenes/point.js';

const KINDS = ['point', 'spot', 'directional', 'area'];
const MAX_LIGHTS = 256;
const DEFAULTS = {
  kind: 'point', position: [0, 0, 0], direction: [0, 0, -1], color: [1, 1, 1], intensity: 2,
  range: 6, angleDeg: 35, size: [1, 1], radius: 0, shadows: false,
};
const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const triple = (value) => Array.isArray(value) && value.length === 3 && value.every(finite);
const coded = (code, message) => Object.assign(new Error(`${code}: ${message}`), { code });
const fail = (key) => { throw coded('light.invalid_desc', `'${key}' is invalid`); };

function validate(light) {
  if (!KINDS.includes(light.kind)) fail('kind');
  if (!triple(light.position)) fail('position');
  if (!triple(light.direction) || Math.hypot(...light.direction) === 0) fail('direction');
  if (!triple(light.color) || light.color.some((c) => c < 0 || c > 1)) fail('color');
  if (!finite(light.intensity) || light.intensity < 0) fail('intensity');
  if (!finite(light.range) || light.range <= 0) fail('range');
  if (!finite(light.angleDeg) || light.angleDeg < 1 || light.angleDeg > 179) fail('angleDeg');
  if (light.innerAngleDeg !== undefined
    && (!finite(light.innerAngleDeg) || light.innerAngleDeg < 1 || light.innerAngleDeg > light.angleDeg)) fail('innerAngleDeg');
  if (!Array.isArray(light.size) || light.size.length !== 2 || !light.size.every((v) => finite(v) && v > 0)) fail('size');
  if (!finite(light.radius) || light.radius < 0) fail('radius');
  if (typeof light.shadows !== 'boolean') fail('shadows');
}

function merge(base, changes) {
  const out = { ...base };
  for (const [key, value] of Object.entries(changes)) out[key] = value === undefined ? DEFAULTS[key] : value;
  return out;
}

// The documented S17 engine light methods plus the page's scene-building calls. Validate
// before mutation, so a refusal leaves the stored light exactly as it was.
function lightEngine() {
  const lights = new Map();
  const calls = { sun: null, ambient: null, meshes: 0, nodes: 0 };
  let nextHandle = 1;
  let completed = { dropped: null, frame: null, pending: true };
  return {
    calls,
    localLights: () => [...lights.values()].map((light) => structuredClone(light)),
    addLight(desc = {}) {
      if (lights.size >= MAX_LIGHTS) throw coded('lights.invalid', 'capacity is 256 local lights');
      const light = merge(DEFAULTS, desc);
      validate(light);
      const handle = nextHandle++;
      lights.set(handle, light);
      return handle;
    },
    updateLight(handle, changes) {
      if (!lights.has(handle)) throw coded('api.unknown_handle', 'unknown handle');
      const light = merge(lights.get(handle), changes);
      validate(light);
      lights.set(handle, light);
    },
    removeLight(handle) {
      if (!lights.delete(handle)) throw coded('api.unknown_handle', 'unknown handle');
    },
    lightStats: () => ({
      activeLights: lights.size, attachedLights: 0, tileColumns: 0, tileRows: 0, tileCapacity: 127,
      droppedLightReferences: completed.dropped, statsFrameIndex: completed.frame, pending: completed.pending,
    }),
    complete(dropped, frame) { completed = { dropped, frame, pending: false }; },
    setCamera() {}, setBackground() {},
    setLight(light) { calls.sun = structuredClone(light); },
    setAmbient(light) { calls.ambient = structuredClone(light); },
    addMaterial: () => 1,
    addMesh() { calls.meshes += 1; return calls.meshes; },
    addNode() { calls.nodes += 1; return calls.nodes; },
    addMeshInstance() {},
  };
}

const lib = { geometry, pbr: (engine, color, opts = {}) => engine.addMaterial({ ...opts, color, model: 'pbr' }) };

test('point variants are bounded on/off selections', () => {
  assert.equal(point.id, 'point');
  assert.deepEqual(point.variants.map((variant) => variant.id), ['on', 'off']);
  for (const variant of point.variants) {
    assert.deepEqual(variantProblems(variant), []);
    assert.match(variant.description, /bounded point lighting/);
    assert.match(variant.description, /local lights cast no shadow yet/);
  }
  assert.match(point.variants[0].description, /finite range.*dims with distance/);
  assert.match(point.variants[1].description, /no local point lights/);
  assert.match(point.legend, /fades to zero at the range limit/);
});

test('on adds bounded point lights through the public API and off adds none', async () => {
  for (const variant of point.variants) {
    const engine = lightEngine();
    await point.build(engine, lib, variant);
    assert.equal(engine.calls.meshes, 4);
    assert.equal(engine.calls.nodes, 4);
    assert.deepEqual(engine.calls.sun, { direction: [-0.5, -1, -0.5], color: [1, 1, 1], intensity: 2 });
    assert.ok(engine.calls.ambient);
    const lights = engine.localLights();
    if (variant.id === 'off') {
      assert.equal(lights.length, 0);
      continue;
    }
    assert.equal(lights.length, 2);
    for (const light of lights) {
      assert.equal(light.kind, 'point');
      assert.ok(Number.isFinite(light.range) && light.range > 0 && light.range <= 4.5);
    }
    const key = lights.find((light) => light.intensity === 8);
    assert.ok(key, 'the on scene leaves its atomic update applied');
    assert.deepEqual([key.position, key.color, key.range], [[1.4, 1.5, 2.4], [1, 0.75, 0.5], 4.5]);
  }
});

test('public updateLight merges atomically and a refusal leaves the light untouched', () => {
  const engine = lightEngine();
  const handle = engine.addLight({ kind: 'point', position: [1, 2, 3], intensity: 2, range: 4 });
  engine.updateLight(handle, { intensity: 9 });
  const updated = engine.localLights()[0];
  assert.deepEqual([updated.position, updated.color, updated.intensity, updated.range], [[1, 2, 3], [1, 1, 1], 9, 4]);
  assert.throws(() => engine.updateLight(handle, { range: 0 }), /light\.invalid_desc/);
  assert.throws(() => engine.updateLight(handle, { intensity: -1 }), /light\.invalid_desc/);
  assert.deepEqual(engine.localLights()[0], updated);
  engine.updateLight(handle, { angleDeg: 40, innerAngleDeg: undefined });
  assert.equal(engine.localLights()[0].angleDeg, 40);
});

test('the public capacity refuses the 257th light without mutating the registry', () => {
  const engine = lightEngine();
  for (let i = 0; i < MAX_LIGHTS; i++) engine.addLight({ range: 1 + i / MAX_LIGHTS });
  assert.equal(engine.lightStats().activeLights, MAX_LIGHTS);
  assert.throws(() => engine.addLight({}), /lights\.invalid.*256/);
  assert.equal(engine.lightStats().activeLights, MAX_LIGHTS);
  engine.removeLight(1);
  assert.equal(engine.lightStats().activeLights, MAX_LIGHTS - 1);
  assert.ok(engine.addLight({}) > MAX_LIGHTS, 'a removed handle is never reused');
});

test('public stats count synchronously and publish overflow only after a completed readback', () => {
  const engine = lightEngine();
  engine.addLight({});
  assert.deepEqual(engine.lightStats(), {
    activeLights: 1, attachedLights: 0, tileColumns: 0, tileRows: 0, tileCapacity: 127,
    droppedLightReferences: null, statsFrameIndex: null, pending: true,
  });
  engine.complete(3, 7);
  const { droppedLightReferences, statsFrameIndex, pending } = engine.lightStats();
  assert.deepEqual([droppedLightReferences, statsFrameIndex, pending], [3, 7, false]);
});
