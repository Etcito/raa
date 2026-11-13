import test from 'node:test';
import assert from 'node:assert/strict';
import { foliagePatch, groundPlane, highPolyMesh, meshBounds, occluderWall } from './lib/content.js';
import { clusterColor, cutPreviewMesh } from './lib/cluster-preview.js';
import { collectVgFacts } from './lib/facts.js';

// Content tests (the four preset meshes and meshBounds), then the cluster-preview and facts tests.
// Node only; no engine, no WebGPU.

const bytes = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const meshBytes = (m) => Buffer.concat(['positions', 'normals', 'uvs', 'indices'].map((k) => bytes(m[k])));
const vertex = (a, i) => [a[3 * i], a[3 * i + 1], a[3 * i + 2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// Exact extents of the seeded generators, measured once from the current highPolyMesh() and
// foliagePatch() (the generators are not tuned to these numbers; they pin them).
const HIGH_POLY_EXTENT = { min: [-1.005219, -1.010367, -1.027208], max: [1.005219, 1.010367, 1.027208] };
const FOLIAGE_EXTENT = { min: [-2.012192, 0, -2.014094], max: [2.013464, 0.25, 2.009901] };

// Undirected edge usage on the position-welded topology (S2): vertices at the same position
// (hard edges duplicate them) share one id. Each edge counts its triangle uses and the net
// direction (+1 per lo->hi use, -1 per hi->lo use). A closed, consistently wound mesh has
// count 2 and net 0.
function edgeUse(mesh) {
  const ids = new Map();
  const weld = Array.from({ length: mesh.positions.length / 3 }, (_, i) => {
    const key = vertex(mesh.positions, i).map((v) => v.toFixed(6)).join(',');
    if (!ids.has(key)) ids.set(key, ids.size);
    return ids.get(key);
  });
  const use = new Map();
  const indices = mesh.indices;
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = weld[indices[t + k]];
      const b = weld[indices[t + ((k + 1) % 3)]];
      const key = `${Math.min(a, b)}:${Math.max(a, b)}`;
      const e = use.get(key) ?? { count: 0, dir: 0 };
      e.count++;
      e.dir += a < b ? 1 : -1;
      use.set(key, e);
    }
  }
  return use;
}

function assertClosed(mesh) {
  for (const [key, e] of edgeUse(mesh)) {
    assert.ok(e.count === 2 && e.dir === 0, `edge ${key} used ${e.count} times, net ${e.dir}`);
  }
}

function assertUnitNormals(mesh) {
  for (let i = 0; i < mesh.normals.length; i += 3) {
    const len = Math.hypot(mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]);
    assert.ok(Math.abs(len - 1) <= 1e-4, `normal ${i / 3} has length ${len}`);
  }
}

function assertWellFormed(mesh, vertexCount, triangleCount) {
  assert.equal(mesh.positions.length, 3 * vertexCount);
  assert.equal(mesh.normals.length, 3 * vertexCount);
  assert.equal(mesh.uvs.length, 2 * vertexCount);
  assert.equal(mesh.indices.length, 3 * triangleCount);
  for (const k of ['positions', 'normals', 'uvs']) {
    assert.ok(mesh[k].every(Number.isFinite), `${k} has a non-finite value`);
  }
  assert.ok(mesh.indices.every((i) => i < vertexCount), 'an index is out of range');
}

function assertBounds(mesh, min, max, tol = 1e-3) {
  const b = meshBounds(mesh);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(b.min[k] - min[k]) <= tol, `min[${k}] ${b.min[k]} != ${min[k]}`);
    assert.ok(Math.abs(b.max[k] - max[k]) <= tol, `max[${k}] ${b.max[k]} != ${max[k]}`);
  }
}

// Every triangle's geometric normal (counter-clockwise) agrees with the vertex normals.
function assertFacesAgreeWithNormals(mesh) {
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const [a, b, c] = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]];
    const p0 = vertex(mesh.positions, a);
    const geo = cross(sub(vertex(mesh.positions, b), p0), sub(vertex(mesh.positions, c), p0));
    for (const i of [a, b, c]) {
      assert.ok(dot(geo, vertex(mesh.normals, i)) > 0, `triangle ${t / 3} is wound against its normal`);
    }
  }
}

test('highPolyMesh has 81920 triangles, 40962 shared vertices and a closed displaced icosphere', () => {
  const m = highPolyMesh();
  assertWellFormed(m, 40962, 81920);
  assertUnitNormals(m);
  assertClosed(m);
  assertFacesAgreeWithNormals(m);
  // Envelope: the displacement is at most 0.06 of the unit radius, so every vertex lies at radius
  // between 1 - 0.06 and 1 + 0.06, and every coordinate stays inside [-1.06, 1.06].
  for (let i = 0; i < m.positions.length; i += 3) {
    const r = Math.hypot(m.positions[i], m.positions[i + 1], m.positions[i + 2]);
    assert.ok(r >= 0.94 - 1e-6 && r <= 1.06 + 1e-6, `vertex ${i / 3} at radius ${r}`);
  }
  assert.ok(m.positions.every((v) => Math.abs(v) <= 1.06 + 1e-6), 'a coordinate leaves [-1.06, 1.06]');
  // Exact extents of the generator, within 1e-3.
  assertBounds(m, HIGH_POLY_EXTENT.min, HIGH_POLY_EXTENT.max);
});

test('foliagePatch has 4096 blades of two triangles and four own vertices each', () => {
  const m = foliagePatch();
  assertWellFormed(m, 4096 * 4, 4096 * 2);
  assertUnitNormals(m);
  for (let blade = 0; blade < 4096; blade++) {
    for (let t = 2 * blade; t < 2 * blade + 2; t++) {
      for (let k = 0; k < 3; k++) {
        const i = m.indices[3 * t + k];
        assert.ok(i >= 4 * blade && i < 4 * blade + 4, `blade ${blade} triangle ${t} uses vertex ${i}`);
      }
    }
    const p0 = vertex(m.positions, 4 * blade);
    const p1 = vertex(m.positions, 4 * blade + 1);
    const p3 = vertex(m.positions, 4 * blade + 3);
    assert.ok(Math.abs(Math.hypot(...sub(p1, p0)) - 0.03) <= 1e-5, `blade ${blade} width`);
    assert.ok(Math.abs(Math.hypot(...sub(p3, p0)) - 0.25) <= 1e-5, `blade ${blade} height`);
  }
});

test('foliagePatch stands on y = 0 inside the 4 m square and its blades keep their face normals', () => {
  const m = foliagePatch();
  const b = meshBounds(m);
  assert.ok(Math.abs(b.min[1]) <= 1e-3 && Math.abs(b.max[1] - 0.25) <= 1e-3, 'height range');
  // Envelope: cell origins lie in [-2, 2) on x and z; a blade's corners add its half-width,
  // BLADE_WIDTH / 2 = 0.015 m, so x and z stay within [-2.015, 2.015].
  for (const k of [0, 2]) {
    assert.ok(b.min[k] >= -2.015 && b.max[k] <= 2.015, `axis ${k} leaves the 4 m square plus a half blade`);
  }
  // Exact extents of the generator, within 1e-3.
  assertBounds(m, FOLIAGE_EXTENT.min, FOLIAGE_EXTENT.max);
  for (let i = 0; i < m.normals.length; i += 3) {
    assert.ok(Math.abs(m.normals[i + 1]) <= 1e-6, 'blade normal is not horizontal');
  }
});

test('occluderWall is a closed 6 m x 3 m x 0.3 m box with hard edges and outward normals', () => {
  const m = occluderWall();
  assertWellFormed(m, 24, 12);
  assertUnitNormals(m);
  assertClosed(m);
  assertFacesAgreeWithNormals(m);
  assertBounds(m, [-3, 0, -0.15], [3, 3, 0.15]);
  for (let i = 0; i < 24; i++) {
    const n = vertex(m.normals, i);
    const centred = sub(vertex(m.positions, i), [0, 1.5, 0]);
    assert.ok(dot(n, centred) > 0, `vertex ${i} normal points inward`);
  }
});

test('groundPlane is a 40 m square on y = 0 with normal +Y and uvs 0..1', () => {
  const m = groundPlane();
  assertWellFormed(m, 4, 2);
  assertBounds(m, [-20, 0, -20], [20, 0, 20]);
  for (let i = 0; i < m.normals.length; i += 3) {
    assert.deepEqual([m.normals[i], m.normals[i + 1], m.normals[i + 2]], [0, 1, 0]);
  }
  assert.deepEqual(Array.from(m.uvs), [0, 0, 0, 1, 1, 1, 1, 0]);
  assertFacesAgreeWithNormals(m);
});

test('every builder returns equal bytes on two calls', () => {
  for (const build of [highPolyMesh, foliagePatch, occluderWall, groundPlane]) {
    assert.equal(Buffer.compare(meshBytes(build()), meshBytes(build())), 0, build.name);
  }
});

test('builders return fresh arrays: a mutated result does not change the next call', () => {
  const first = highPolyMesh();
  const reference = Buffer.from(meshBytes(first));
  first.positions.fill(0);
  first.indices.fill(0);
  assert.equal(Buffer.compare(meshBytes(highPolyMesh()), reference), 0);
});

test('meshBounds gives the component-wise min and max of the positions', () => {
  const b = meshBounds({ positions: new Float32Array([1, -2, 3, -4, 5, -6]) });
  assert.deepEqual(b, { min: [-4, -2, -6], max: [1, 5, 3] });
});

// A hand-built cut of three clusters and seven triangles over twelve vertices: cluster 0 holds
// triangles 0-1 on vertices 0-3, cluster 1 holds triangles 2-3 on vertices 4-7, cluster 2 holds
// triangles 4-6 on vertices 8-11. Each cluster's vertices are its own.
function handCut() {
  return {
    positions: Float32Array.from({ length: 36 }, (_, i) => i * 0.25),
    normals: Float32Array.from({ length: 36 }, (_, i) => (i % 3 === 1 ? 1 : 0)),
    uvs: Float32Array.from({ length: 24 }, (_, i) => i / 24),
    indices: Uint32Array.from([0, 1, 2, 0, 2, 3, 4, 5, 6, 5, 7, 6, 8, 9, 10, 9, 11, 10, 10, 11, 8]),
    clusterOfTriangle: Uint32Array.from([0, 0, 1, 1, 2, 2, 2]),
    clusters: 3,
    triangles: 7,
  };
}

// A module double: its clusterCutMesh returns the cut and records its arguments.
function raaDouble(cut, calls = []) {
  return { clusterCutMesh: (section, errorMetres) => (calls.push([section, errorMetres]), cut) };
}

test('cutPreviewMesh keeps positions, normals, uvs and indices unchanged: every cut triangle once', () => {
  const cut = handCut();
  const calls = [];
  const section = { algorithm: 1 };
  const m = cutPreviewMesh(raaDouble(cut, calls), section, 0.5);
  assert.deepEqual(calls, [[section, 0.5]]);
  for (const k of ['positions', 'normals', 'uvs', 'indices']) {
    assert.deepEqual(Array.from(m[k]), Array.from(cut[k]), k);
  }
  assert.equal(m.indices.length, 3 * cut.triangles);
});

test('cutPreviewMesh gives four colour floats per vertex: equal inside a cluster, distinct between clusters', () => {
  const m = cutPreviewMesh(raaDouble(handCut()), {}, 0);
  assert.equal(m.colors.length, 4 * 12);
  assert.ok(m.colors.every((v) => Number.isFinite(v) && v >= 0 && v <= 1), 'a colour leaves [0, 1]');
  const colourOf = (v) => Array.from(m.colors.subarray(4 * v, 4 * v + 3));
  for (let v = 0; v < 12; v++) {
    assert.equal(m.colors[4 * v + 3], 1, `vertex ${v} alpha`);
    const c = Math.floor(v / 4);
    assert.ok(colourOf(v).every((x, k) => Math.abs(x - clusterColor(c)[k]) <= 1e-6), `vertex ${v} colour`);
  }
  const first = [0, 4, 8].map((v) => colourOf(v).join(','));
  assert.equal(new Set(first).size, 3, 'two clusters share a colour');
});

test('clusterColor is the same on two calls, distinct per index, and linear sRGB of the hue', () => {
  for (let c = 0; c < 8; c++) assert.deepEqual(clusterColor(c), clusterColor(c));
  const seen = new Set(Array.from({ length: 16 }, (_, c) => clusterColor(c).join(',')));
  assert.equal(seen.size, 16);
  // Index 0: hue 0, so HSV (0.9, 0.315, 0.315) in sRGB; the channels are converted to linear.
  const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const [r, g, b] = clusterColor(0);
  assert.ok(Math.abs(r - lin(0.9)) <= 1e-12 && Math.abs(g - lin(0.315)) <= 1e-12 && Math.abs(b - lin(0.315)) <= 1e-12);
});

test('collectVgFacts reports null settings, stats and instance state for an engine without the getters', () => {
  const engine = {
    meshClusters: (mesh) => ({ state: mesh === 3 ? 'ready' : 'none', progress: 0, source: null, code: null, message: null, sections: 0, clusters: 0, sourceTriangles: 0, levels: 0 }),
  };
  const facts = collectVgFacts(
    engine,
    [{ name: 'high', mesh: 3, cuts: [{ errorMetres: 0.1, clusters: 5, triangles: 9 }], states: [{ state: 'ready', progress: 1 }] }, { name: 'none', mesh: null }],
    [{ name: 'a', instance: 7 }],
  );
  assert.equal(facts.settings, null);
  assert.equal(facts.stats, null);
  assert.equal(facts.meshes[0].clusters.state, 'ready');
  assert.equal(facts.meshes[1].clusters, null);
  assert.deepEqual(facts.meshes[0].cuts, [{ errorMetres: 0.1, clusters: 5, triangles: 9 }]);
  assert.deepEqual(facts.meshes[0].states, [{ state: 'ready', progress: 1 }]);
  assert.deepEqual(facts.meshes[1].cuts, []);
  assert.deepEqual(facts.meshes[1].states, []);
  assert.deepEqual(facts.instances, [{ name: 'a', instance: 7, virtualGeometry: null }]);
});

test('collectVgFacts copies getter results into fresh objects and lets a throwing getter propagate', () => {
  const settings = { enabled: true, errorPixels: 1 };
  const engine = {
    virtualGeometry: () => settings,
    virtualGeometryStats: () => ({ active: true }),
    instanceVirtualGeometry: () => ({ requested: null, path: 'virtual', pending: false, code: null, message: null }),
  };
  const facts = collectVgFacts(engine, [], [{ name: 'a', instance: 1 }]);
  assert.deepEqual(facts.settings, settings);
  assert.notEqual(facts.settings, settings);
  assert.equal(facts.stats.active, true);
  assert.equal(facts.instances[0].virtualGeometry.path, 'virtual');
  const broken = { virtualGeometry() { throw new Error('getter failed'); } };
  assert.throws(() => collectVgFacts(broken, [], []), /getter failed/);
});
