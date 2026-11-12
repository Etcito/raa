import { test } from 'node:test';
import assert from 'node:assert/strict';

import { decodeMeshFile, meshFilePayload } from '../src/mesh-file.js';
import { validateClusterSection } from '../src/mesh-clusters.js';

// The S24 encoder of test/procedural.test.js (byte 0 = 1, big-endian header and buffer counts, {$buffer} / {$array}
// markers in the JSON header, then the buffers), written again so this file stands alone.
function encodeMeshFile(value) {
  const buffers = [];
  const index = (buffer) => {
    let i = buffers.indexOf(buffer);
    if (i < 0) {
      i = buffers.length;
      buffers.push(buffer);
    }
    return i;
  };
  const json = JSON.stringify(value, (_key, v) => {
    if (v instanceof ArrayBuffer) return { $buffer: index(v) };
    if (ArrayBuffer.isView(v)) return { $array: v.constructor.name, buffer: index(v.buffer), offset: v.byteOffset, length: v.byteLength };
    return v;
  });
  const header = new TextEncoder().encode(json);
  const out = new Uint8Array(9 + header.length + buffers.reduce((n, b) => n + 4 + b.byteLength, 0));
  out[0] = 1;
  const view = new DataView(out.buffer);
  view.setUint32(1, header.length);
  view.setUint32(5, buffers.length);
  out.set(header, 9);
  let at = 9 + header.length;
  for (const buffer of buffers) {
    view.setUint32(at, buffer.byteLength);
    out.set(new Uint8Array(buffer), at + 4);
    at += 4 + buffer.byteLength;
  }
  return out;
}

// The fixture section: 3 vertices, 1 triangle, 1 cluster (3 vertices, 1 triangle), 1 leaf node over that cluster.
function section() {
  const clusters = new Uint8Array(48);
  new DataView(clusters.buffer).setUint32(44, 3 | (1 << 8), true);
  const nodes = new Uint8Array(48);
  const nodeView = new DataView(nodes.buffer);
  nodeView.setUint32(40, 1, true);
  nodeView.setUint32(44, 1, true);
  return {
    version: 1,
    algorithm: 1,
    vertices: new Uint16Array([1, 2, 3, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0]),
    triangles: new Uint32Array([0 | (1 << 8) | (2 << 16)]),
    clusters,
    nodes,
    rootNode: 0,
    hierarchyDepth: 1,
    levels: 1,
    quantOrigin: [0, 0, 0],
    quantStep: 0.5,
    uvOrigin: [0, 0],
    uvStep: [0.25, 0.25],
    boundsCenter: [0.5, 0.5, 0.5],
    boundsRadius: 1,
    sourceTriangles: 1,
  };
}

const edit = (fn) => {
  const value = section();
  fn(value);
  return value;
};
const setU32 = (array, offset, value) => new DataView(array.buffer).setUint32(offset, value, true);

function meshFile(meshes) {
  const frame = new Float64Array(16);
  frame[0] = frame[5] = frame[10] = frame[15] = 1;
  return {
    transfer: [],
    parts: 1,
    triangles: 1,
    data: {
      groups: [{ group: 'root', parts: [{ partId: '1', name: 'Body', frame, meshes }] }],
      materials: [{ id: 'paint', kind: 'standard' }],
    },
  };
}

function plainMesh(extra = {}) {
  return {
    material: 'paint',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: null,
    indices: new Uint32Array([0, 1, 2]),
    ...extra,
  };
}

function assertRefused(bytes, label) {
  assert.throws(() => meshFilePayload(decodeMeshFile(bytes)), (error) => {
    assert.ok(error && error.name === 'SdfmError', String(error));
    assert.equal(error.code, 'sdfm.scene_resolve');
    assert.ok(error.message.includes(label), `${label} not in: ${error.message}`);
    return true;
  });
}

test('a file without the clusters key gives clusters: null for every mesh', () => {
  const payload = meshFilePayload(decodeMeshFile(encodeMeshFile(meshFile([plainMesh(), plainMesh()]))));
  assert.deepEqual(payload.groups[0].parts[0].meshes.map((mesh) => mesh.clusters), [null, null]);
});

test('a valid section is returned as the same object and survives decode and payload with equal arrays', () => {
  const sent = section();
  assert.equal(validateClusterSection(sent, 'root/1#0'), sent);
  const decoded = decodeMeshFile(encodeMeshFile(meshFile([plainMesh({ clusters: sent }), plainMesh()])));
  const [first, second] = meshFilePayload(decoded).groups[0].parts[0].meshes;
  assert.deepStrictEqual(first.clusters, sent);
  assert.ok(first.clusters.vertices instanceof Uint16Array);
  assert.ok(first.clusters.triangles instanceof Uint32Array);
  assert.ok(first.clusters.clusters instanceof Uint8Array);
  assert.equal(second.clusters, null);
});

test('the refusal names the index of the mesh that carries the bad section', () => {
  const bytes = encodeMeshFile(meshFile([plainMesh({ clusters: section() }), plainMesh({ clusters: edit((s) => { s.levels = 0; }) })]));
  assertRefused(bytes, 'mesh: part root/1#1 clusters.levels:');
});

// One refusal per wire rule, in rule order. Each value is the section (or other value) that the rule refuses.
const RULES = [
  ['keys', () => []],
  ['keys', () => edit((s) => { delete s.levels; })],
  ['keys', () => edit((s) => { s.extra = 1; })],
  ['version', () => edit((s) => { s.version = 2; })],
  ['algorithm', () => edit((s) => { s.algorithm = 0; })],
  ['vertices', () => edit((s) => { s.vertices = new Uint8Array(8); })],
  ['vertices', () => edit((s) => { s.vertices = new Uint16Array(9); })],
  ['triangles', () => edit((s) => { s.triangles = new Uint32Array(0); })],
  ['nodes', () => edit((s) => { s.nodes = new Uint8Array(47); })],
  ['rootNode', () => edit((s) => { s.rootNode = 1; })],
  ['hierarchyDepth', () => edit((s) => { s.hierarchyDepth = 65; })],
  ['levels', () => edit((s) => { s.levels = 0; })],
  ['sourceTriangles', () => edit((s) => { s.sourceTriangles = -1; })],
  ['quantOrigin', () => edit((s) => { s.quantOrigin = [0, 0, Infinity]; })],
  ['quantStep', () => edit((s) => { s.quantStep = 0; })],
  ['uvStep', () => edit((s) => { s.uvStep = [-1, 0]; })],
  ['boundsRadius', () => edit((s) => { s.boundsRadius = NaN; })],
  ['clusters[0]', () => edit((s) => setU32(s.clusters, 44, 1 << 8))],
  ['clusters[0]', () => edit((s) => setU32(s.clusters, 44, 3))],
  ['clusters[0]', () => edit((s) => setU32(s.clusters, 44, 3 | (129 << 8)))],
  ['clusters[0]', () => edit((s) => setU32(s.clusters, 36, 1))],
  ['clusters[0]', () => edit((s) => setU32(s.clusters, 40, 1))],
  ['nodes[0]', () => edit((s) => setU32(s.nodes, 40, 0))],
  ['nodes[0]', () => edit((s) => setU32(s.nodes, 44, 2))],
  ['nodes[0]', () => edit((s) => setU32(s.nodes, 36, 1))],
];

for (const [index, [field, value]] of RULES.entries()) {
  test(`refuses clusters.${field} (case ${index + 1}) with the part, mesh index and field named`, () => {
    const bytes = encodeMeshFile(meshFile([plainMesh({ clusters: value() })]));
    assertRefused(bytes, `mesh: part root/1#0 clusters.${field}:`);
  });
}
