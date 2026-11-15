import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import scene from '../scenes/m31-taa.js';
import * as geometry from '../lib/geometry.js';

const lib = { geometry, pbr: (engine, color, opts) => engine.addMaterial({ ...opts, color, model: 'pbr' }) };
const OFF = 'Temporal anti-aliasing is off with no multisampling: the thin dark fence posts and rails against the bright sky and ground, the slanted white plank and the red ball show jagged stair-stepped pixel edges along their outlines.';
const ON = 'Temporal anti-aliasing is on: the thin dark fence posts and rails, the slanted white plank and the red ball have smooth anti-aliased outlines without stair steps; the thin posts stay continuous with no broken segments; the red ball is crisp with no smeared trail or doubled ghost copy behind it; the image is not blurred overall.';

// A recording engine: every method returns a fresh id and is logged in `calls`.
function recordingEngine() {
  const calls = [];
  let next = 1;
  return new Proxy({}, {
    get: (_, name) => (name === 'calls' ? calls : (...args) => { calls.push([name, ...args]); return next++; }),
  });
}
const named = (engine, name) => engine.calls.filter(call => call[0] === name).map(call => call.slice(1));

test('id, milestone, engine options and variants equal the contract', () => {
  assert.equal(scene.id, 'm31-taa');
  assert.equal(scene.milestone, 'm31');
  assert.equal(scene.title, 'Temporal anti-aliasing over a swaying camera');
  assert.deepEqual(scene.engine, { msaa: 1, toneMapper: 'agx' });
  assert.deepEqual(scene.variants, [
    { id: 'taa-off', taa: false, judge: true, references: [], description: OFF },
    { id: 'taa-on', taa: true, judge: true, references: [], description: ON },
  ]);
});

test('build toggles taa once per variant and places the content', async () => {
  for (const variant of scene.variants) {
    const engine = recordingEngine();
    await scene.build(engine, lib, variant);
    assert.deepEqual(named(engine, 'setPassEnabled'), [['taa', variant.taa]]);
    assert.deepEqual(named(engine, 'setOrbit'), [[[0, 1.0, 0], Math.hypot(7, 0.6), 0, Math.atan2(0.6, 7)]]);
    assert.equal(named(engine, 'addMeshInstance').length, 16);
  }
});

async function sway() {
  const engine = recordingEngine();
  await scene.build(engine, lib, scene.variants[1]);
  for (let i = 0; i < 64; i++) scene.frame(engine, 1000, scene.variants[1]);
  return engine.calls;
}

test('64 frame-hook calls sway within the yaw bound and slide the ball in range', async () => {
  const calls = await sway();
  const orbits = calls.filter(c => c[0] === 'orbit').map(c => c.slice(1));
  assert.equal(orbits.length, 64);
  let yaw = 0;
  for (const [dyaw, dpitch, zoom] of orbits) {
    assert.ok(Math.abs(Math.abs(dyaw) - 0.003) < 1e-12);
    assert.equal(dpitch, 0);
    assert.equal(zoom, 1);
    yaw += dyaw;
    assert.ok(Math.abs(yaw) <= 0.015 + 1e-9, `yaw ${yaw}`);
  }
  const moves = calls.filter(c => c[0] === 'setTransforms').map(c => c.slice(1));
  assert.equal(moves.length, 64);
  for (const [, values] of moves) assert.ok(values[0] >= -1.5 && values[0] <= 1.5);
});

// The box overlaps the view volume unless all eight corners are beyond one frustum plane.
function inView(min, max, yaw) {
  const d = Math.hypot(7, 0.6), pitch = Math.atan2(0.6, 7);
  const eye = [d * Math.cos(pitch) * Math.sin(yaw), 1 + d * Math.sin(pitch), d * Math.cos(pitch) * Math.cos(yaw)];
  const f = [-eye[0], 1 - eye[1], -eye[2]];
  const fl = Math.hypot(...f);
  const fwd = f.map(v => v / fl);
  const rl = Math.hypot(fwd[2], fwd[0]);
  const right = [-fwd[2] / rl, 0, fwd[0] / rl];
  const up = [fwd[1] * right[2] - fwd[2] * right[1], fwd[2] * right[0] - fwd[0] * right[2], fwd[0] * right[1] - fwd[1] * right[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const vertical = Math.tan(40 * Math.PI / 360), horizontal = vertical * 16 / 9;
  const corners = [];
  for (let m = 0; m < 8; m++) {
    const p = [0, 1, 2].map(i => (m & (1 << i) ? max[i] : min[i]) - eye[i]);
    const x = dot(p, right), y = dot(p, up), z = dot(p, fwd);
    corners.push([z - 0.1, 200 - z, z * horizontal + x, z * horizontal - x, z * vertical + y, z * vertical - y]);
  }
  return !corners[0].some((_, plane) => corners.every(c => c[plane] < 0));
}

test('posts, rails, plank and the ball travel box stay in view over the sway', () => {
  const subjects = [];
  for (let i = 0; i < 11; i++) {
    const x = -2.5 + 0.5 * i;
    subjects.push([[x - 0.03, 0, -0.03], [x + 0.03, 1.6, 0.03]]);
  }
  for (const y of [0.5, 1.25]) subjects.push([[-2.6, y - 0.025, -0.025], [2.6, y + 0.025, 0.025]]);
  subjects.push([[-1.2, -0.15, 0.9], [1.2, 0.85, 1.5]]);
  subjects.push([[-1.8, 0, 1.7], [1.8, 0.6, 2.3]]);
  assert.equal(subjects.length, 15);
  for (const yaw of [-0.015, 0, 0.015]) {
    subjects.forEach(([min, max], i) => assert.ok(inView(min, max, yaw), `subject ${i} at yaw ${yaw}`));
  }
});

const dist = process.env.LOOKDEV_DIST;
test('the staged build lists the scene and stays within the wasm size limit', { skip: !dist }, async () => {
  const list = JSON.parse(readFileSync(`${dist}/scenes.json`, 'utf8'));
  const entries = list.filter(e => e.id === 'm31-taa');
  assert.equal(entries.length, 1);
  assert.deepEqual(entries[0], { id: 'm31-taa', milestone: 'm31', file: 'scenes/m31-taa.js' });
  const staged = (await import(new URL(`${dist}/scenes/m31-taa.js`, `file://${process.cwd()}/`))).default;
  assert.deepEqual(staged.variants.map(v => v.id), ['taa-off', 'taa-on']);
  const { gzip } = JSON.parse(readFileSync(`${dist}/sizes.json`, 'utf8'));
  assert.ok(Number.isInteger(gzip) && gzip <= 300000, `wasm gzip is ${gzip}, limit 300000`);
});
