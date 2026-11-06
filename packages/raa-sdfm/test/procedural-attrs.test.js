import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createFakeEngine } from './fake-engine.js';
import { loadProceduralScene } from '../src/procedural.js';

// S24 encoder copied from test/procedural.test.js: byte 0 = 1, big-endian u32 header length at 1 and buffer count at 5,
// the UTF-8 JSON header with {$buffer} / {$array} markers, then per buffer a big-endian u32 length and its bytes.
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

// Two parts, so one scene instance owns two mesh instances and the forwarding must reach each of them.
function validMeshFile() {
  const frame = new Float64Array(16);
  frame[0] = frame[5] = frame[10] = frame[15] = 1;
  const mesh = () => ({
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: null,
    indices: new Uint32Array([0, 1, 2]),
  });
  return {
    transfer: [],
    parts: 2,
    triangles: 2,
    data: {
      groups: [{
        group: 'root',
        parts: [
          { partId: '1', name: 'A', frame, meshes: [mesh()] },
          { partId: '2', name: 'B', frame, meshes: [mesh()] },
        ],
      }],
      materials: [],
    },
  };
}

const TINT = [1, 0.214, 0, 1];
const near = (actual, expected, eps = 1e-3) => {
  assert.equal(actual.length, expected.length);
  expected.forEach((v, i) => assert.ok(Math.abs(actual[i] - v) <= eps, `float ${i}: ${actual[i]} vs ${v}`));
};

const mama = async () => ({ parseProceduralScene: (text) => JSON.parse(text) });

function recordingEngine({ lod = true } = {}) {
  const engine = createFakeEngine();
  const calls = { setInstanceParams: [], setInstanceLodBias: [], removeNode: [], removeMesh: [] };
  engine.setInstanceParams = (handle, params) => calls.setInstanceParams.push([handle, Array.from(params)]);
  if (lod) engine.setInstanceLodBias = (handle, bias) => calls.setInstanceLodBias.push([handle, bias]);
  engine.setVisible = () => {};
  engine.removeNode = (node) => calls.removeNode.push(node);
  engine.removeMesh = (mesh) => calls.removeMesh.push(mesh);
  engine.setMaterialTintMode = () => {};
  return { engine, calls };
}

function meshScene() {
  return {
    format: 'sdfm.procedural-scene',
    version: 1,
    variant: 'mesh-refs',
    name: 'attrs',
    game: 'fixture',
    generator: { document: 'fixture', node: 'scatter', seed: 1, params: {} },
    sources: [{
      kind: 'mesh',
      mesh: 'procedural/attrs/meshes/00000000.mesh',
      meshKey: '00000000',
      bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    }],
    instances: [
      {
        key: 1, source: 0, t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], group: -1,
        attrs: { 'tint.r': 1, 'tint.g': 0.5, 'tint.b': 0 }, lod: { bias: 2 },
      },
      { key: 2, source: 0, t: [1, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], group: -1 },
    ],
    groups: [],
    colliders: [],
    visual: [],
    runtime: null,
  };
}

function fixtureAsset() {
  return {
    format: 'sdfm.game-asset',
    version: 1,
    game: 'fixture',
    id: 'common.post.test',
    name: 'common.post.test',
    category: 'prop',
    world: 'fixture',
    mesher: { output: 'paramesh' },
    document: { format: 'sdfm', version: 2, name: 'common.post.test', root: [], nodes: {}, materials: {} },
    collision: [],
    anchors: {},
    bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    hash: '0'.repeat(64),
  };
}

function sdfScene() {
  return {
    format: 'sdfm.procedural-scene',
    version: 1,
    variant: 'sdf-refs',
    name: 'attrs',
    game: 'fixture',
    generator: { document: 'fixture', node: 'scatter', seed: 1, params: {} },
    sources: [{
      kind: 'asset',
      file: 'assets/common/post.test.asset.json',
      ref: { game: 'fixture', id: 'common.post.test' },
      bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    }],
    instances: [
      {
        key: 1, source: 0, t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], group: -1,
        attrs: { 'tint.r': 1, 'tint.g': 0.5, 'tint.b': 0 }, lod: { bias: 2 },
      },
      { key: 2, source: 0, t: [1, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], group: -1 },
    ],
    groups: [],
    colliders: [],
    visual: [],
    runtime: null,
  };
}

// The loader.spawn fake gives each part its own engine instance handle, derived from the instance node id.
function sdfLoader() {
  return {
    load: () => ({ parts: [{ name: 'p', mesh: 1 }] }),
    spawn: (_asset, parent) => ({
      root: parent,
      parts: {
        a: { node: parent, instance: parent * 100 + 1 },
        b: { node: parent, instance: parent * 100 + 2 },
      },
    }),
  };
}

test('mesh-refs forwards attributes and the detail hint to every part of the instance', async () => {
  const { engine, calls } = recordingEngine();
  const scene = meshScene();
  const handle = await loadProceduralScene(engine, null, scene, { mama, resolveMesh: async () => encodeMeshFile(validMeshFile()) });
  const owned = engine.calls.addMeshInstance.slice(0, 2).map(({ id }) => id);
  const foreign = engine.calls.addMeshInstance.slice(2).map(({ id }) => id);

  assert.deepStrictEqual(calls.setInstanceParams.map(([id]) => id), owned);
  for (const [, params] of calls.setInstanceParams) {
    assert.equal(params.length, 16);
    near(params.slice(0, 4), TINT);
  }
  assert.deepStrictEqual(calls.setInstanceLodBias, owned.map((id) => [id, 2]));
  assert.ok(calls.setInstanceParams.every(([id]) => !foreign.includes(id)));
  assert.equal(handle.stats().attributed, 1);
  assert.equal(handle.stats().instances, 2);
});

test('sdf-refs forwards attributes and the detail hint through spawned parts', async () => {
  const { engine, calls } = recordingEngine();
  const handle = await loadProceduralScene(engine, sdfLoader(), sdfScene(), {
    mama, resolveAsset: async () => fixtureAsset(),
  });
  const instanceNodes = engine.calls.addNode.filter((node) => node.parent === handle.root);
  assert.equal(instanceNodes.length, 2);
  const owned = instanceNodes[0].id * 100 + 1;
  const ownedSecond = instanceNodes[0].id * 100 + 2;

  assert.deepStrictEqual(calls.setInstanceParams.map(([id]) => id), [owned, ownedSecond]);
  for (const [, params] of calls.setInstanceParams) near(params.slice(0, 4), TINT);
  assert.deepStrictEqual(calls.setInstanceLodBias, [[owned, 2], [ownedSecond, 2]]);
  assert.equal(handle.stats().attributed, 1);
});

test('an engine without setInstanceLodBias still loads', async () => {
  const { engine, calls } = recordingEngine({ lod: false });
  const handle = await loadProceduralScene(engine, null, meshScene(), {
    mama, resolveMesh: async () => encodeMeshFile(validMeshFile()),
  });
  assert.equal(engine.setInstanceLodBias, undefined);
  assert.equal(calls.setInstanceParams.length, 2);
  assert.equal(handle.stats().attributed, 1);
});
