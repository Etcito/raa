import { test } from 'node:test';
import assert from 'node:assert/strict';
import { INSTANCE_PARAM_SLOTS, instanceParams } from '../src/instance-params.js';
import { fieldPartShading } from '../src/field-shading.js';

const near = (actual, expected, eps = 1e-3) => {
  assert.equal(actual.length, expected.length);
  expected.forEach((v, i) => assert.ok(Math.abs(actual[i] - v) <= eps, `float ${i}: ${actual[i]} vs ${v}`));
};

test('INSTANCE_PARAM_SLOTS is the S-layout of the 16 floats', () => {
  assert.deepEqual(INSTANCE_PARAM_SLOTS, { tint: 0, wind: 4, scale: 5, free: 6 });
  assert.ok(Object.isFrozen(INSTANCE_PARAM_SLOTS));
});

test('no attributes give null', () => {
  assert.equal(instanceParams(), null);
  assert.equal(instanceParams(null), null);
  assert.equal(instanceParams({}), null);
});

test('tint channels convert from sRGB, float 3 flags the tint, the rest is zero', () => {
  const p = instanceParams({ 'tint.r': 1, 'tint.g': 0.5, 'tint.b': 0 });
  assert.equal(p.length, 16);
  near(p.subarray(0, 4), [1, 0.214, 0, 1]);
  near(p.subarray(4), new Array(12).fill(0));
});

test('a missing tint channel is 1 and the others convert', () => {
  near(instanceParams({ 'tint.r': 0.5 }).subarray(0, 4), [0.214, 1, 1, 1]);
});

test('wind and scale go to slots 4 and 5', () => {
  const p = instanceParams({ wind: 0.3, scale: 1.25 });
  assert.equal(p[INSTANCE_PARAM_SLOTS.wind], Math.fround(0.3));
  assert.equal(p[INSTANCE_PARAM_SLOTS.scale], 1.25);
  assert.equal(p[INSTANCE_PARAM_SLOTS.tint + 3], 0, 'no tint key, no tint flag');
  assert.deepEqual(Array.from(p.subarray(0, 4)), [0, 0, 0, 0]);
});

test('free attributes fill 6..15 in byte order and drop the last two', () => {
  const attrs = {};
  for (let i = 0; i < 12; i++) attrs[`k${String(11 - i).padStart(2, '0')}`] = i + 1;
  const p = instanceParams(attrs);
  const keys = Object.keys(attrs).sort();
  assert.equal(keys.length, 12);
  for (let i = 0; i < 10; i++) assert.equal(p[6 + i], attrs[keys[i]], `slot ${6 + i} = ${keys[i]}`);
  assert.equal(p[4], 0);
  assert.equal(p[5], 0);
});

test('field part wgsl reads the instance tint', () => {
  const FIELD = `struct PmfSample { d: f32, g: vec3f, la: u32, lb: u32, t: f32 }
fn pmf_eval(base: u32, p: vec3f) -> PmfSample { return PmfSample(length(p) - pmf_f(base + 3u), p, 0u, 1u, 0.5); }`;
  const ROWS = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
  const shading = {
    program: [1, 0, 8, 0.25, 0, 0, 0, 0], single: false, look: { roughness: 0.4, metallic: 0 },
    operands: [{ color: [0, 0, 1], material: null, toNode: ROWS }],
  };
  const { wgsl } = fieldPartShading(shading, [], { field: FIELD });
  assert.ok(wgsl.includes('in.instance_params[0].w > 0.5'));
  assert.ok(wgsl.includes('o.base_color = o.base_color * in.instance_params[0].xyz;'));
});
