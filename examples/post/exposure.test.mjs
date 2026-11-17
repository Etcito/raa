// Exposure scene and its room geometry: content builders.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  box,
  checker,
  facadeCourses,
  plane,
  roomWithOpening,
  stripes,
  tiledGround,
} from './lib/content.js';

test('box has 24 vertices, 36 indices and unit normals', () => {
  const m = box([1, 2, 3], [0.5, 1, 1.5]);
  assert.equal(m.positions.length, 72);
  assert.equal(m.indices.length, 36);
  for (let i = 0; i < m.normals.length; i += 3) {
    assert.ok(Math.abs(Math.hypot(m.normals[i], m.normals[i + 1], m.normals[i + 2]) - 1) < 1e-6);
  }
});

test('plane(4, 2) is four vertices at y 0 with +Y normals', () => {
  const m = plane(4, 2);
  assert.equal(m.positions.length, 12);
  for (let i = 0; i < 4; i++) {
    assert.equal(m.positions[i * 3 + 1], 0);
    assert.deepEqual([...m.normals.slice(i * 3, i * 3 + 3)], [0, 1, 0]);
  }
});

const room = {
  width: 6,
  height: 3,
  depth: 6,
  wall: 0.3,
  opening: { zMin: -4, zMax: -2, yMin: 0.3, yMax: 2.5 },
};

test('roomWithOpening: sub-meshes, indices, opening and bounds', () => {
  const m = roomWithOpening(room);
  assert.equal(m.subMeshes.length, 3);
  let next = 0;
  for (const s of m.subMeshes) {
    assert.equal(s.first, next);
    next += s.count;
  }
  assert.equal(next, m.indices.length);
  const n = m.positions.length / 3;
  assert.ok(m.indices.every((i) => i < n));
  const p = (i) => [m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]];
  for (let t = 0; t < m.indices.length; t += 3) {
    const c = [0, 1, 2].map((a) => (p(m.indices[t])[a] + p(m.indices[t + 1])[a] + p(m.indices[t + 2])[a]) / 3);
    const inside = c[0] >= 3 && c[0] <= 3.3 && c[2] > -4 && c[2] < -2 && c[1] > 0.3 && c[1] < 2.5;
    assert.ok(!inside, `triangle ${t / 3} centroid in the opening`);
  }
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    p(i).forEach((v, a) => {
      lo[a] = Math.min(lo[a], v);
      hi[a] = Math.max(hi[a], v);
    });
  }
  const near = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-5);
  assert.ok(near(lo, [-3.3, -0.3, -6.3]));
  assert.ok(near(hi, [3.3, 3.3, 0.3]));
});

test('tiledGround splits tiles into even and odd', () => {
  const m = tiledGround({ x0: 0, z0: 0, columns: 3, rows: 2, size: 1 });
  assert.deepEqual(m.subMeshes.map((s) => s.count), [18, 18]);
});

test('facadeCourses leaves the skip rect free', () => {
  const skip = { zMin: -1, zMax: 1, yMin: 1, yMax: 2.5 };
  const m = facadeCourses({ x: 0, zMin: -4, zMax: 4, yMin: 0, yMax: 4, course: 0.5, depth: 0.2, skip });
  assert.ok(m.indices.length > 0);
  // Every box is 24 vertices; compare each box's z/y extent with the skip rect.
  for (let b = 0; b < m.positions.length / 72; b++) {
    const lo = [Infinity, Infinity];
    const hi = [-Infinity, -Infinity];
    for (let v = 0; v < 24; v++) {
      const o = b * 72 + v * 3;
      lo[0] = Math.min(lo[0], m.positions[o + 1]);
      hi[0] = Math.max(hi[0], m.positions[o + 1]);
      lo[1] = Math.min(lo[1], m.positions[o + 2]);
      hi[1] = Math.max(hi[1], m.positions[o + 2]);
    }
    const overlaps = lo[0] < skip.yMax && hi[0] > skip.yMin && lo[1] < skip.zMax && hi[1] > skip.zMin;
    assert.ok(!overlaps, `box ${b} overlaps the skip rect`);
  }
});

test('stripes and checker texels', () => {
  const s = stripes(8, 4);
  assert.deepEqual([...Array(8).keys()].map((x) => s.data[x * 4]), [255, 255, 0, 0, 255, 255, 0, 0]);
  assert.ok([...Array(64).keys()].every((i) => s.data[i * 4 + 3] === 255));
  const c = checker(4, 2);
  const at = (x, y) => c.data[(y * 4 + x) * 4];
  assert.deepEqual([at(0, 0), at(1, 0), at(0, 1), at(1, 1)], [255, 0, 0, 255]);
});

test('stripes(8, 1) throws', () => {
  assert.throws(
    () => stripes(8, 1),
    /content: stripes needs an integer size > 0 and an integer period >= 2/,
  );
});
