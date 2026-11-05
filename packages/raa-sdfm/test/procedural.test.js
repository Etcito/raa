import { test } from 'node:test';
import assert from 'node:assert/strict';

import { decodeMeshFile, meshFilePayload } from '../src/mesh-file.js';
import { CODES, SdfmError } from '../src/errors.js';

// A small S24 encoder, written from the saved-mesh layout: byte 0 = 1, big-endian u32 header length at 1 and buffer
// count at 5, the UTF-8 JSON header with {$buffer} / {$array} markers, then per buffer a big-endian u32 length and its
// bytes. Buffers are deduplicated by ArrayBuffer identity in first-seen order, so two views can share one buffer.
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

// positions and normals are two views on one buffer; indices live in a second buffer.
function validMeshFile() {
  const frame = new Float64Array(16);
  frame[0] = frame[5] = frame[10] = frame[15] = 1;
  const shared = new ArrayBuffer(72);
  const positions = new Float32Array(shared, 0, 9);
  positions.set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const normals = new Float32Array(shared, 36, 9);
  normals.set([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  return {
    assetId: 'raatest.mesh',
    hash: 'hash',
    mesher: 'paramesh',
    parts: 1,
    triangles: 1,
    key: 'key',
    transfer: [],
    data: {
      groups: [{
        group: 'root',
        parts: [{
          partId: '1',
          name: 'Body',
          frame,
          meshes: [{
            material: 'paint',
            positions,
            normals,
            uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
            indices: new Uint32Array([0, 1, 2]),
          }],
        }],
      }],
      materials: [{ id: 'paint', kind: 'standard' }],
    },
  };
}

function sceneError(fn) {
  assert.throws(fn, (error) => {
    assert.ok(error instanceof SdfmError, `SdfmError expected, got ${error}`);
    assert.equal(error.code, 'sdfm.scene_resolve');
    assert.ok(error.message.length > 'sdfm.scene_resolve: '.length, 'non-empty message');
    return true;
  });
}

test('the eleven existing codes keep their order and the two scene codes are added', () => {
  const existing = [
    'sdfm.format', 'sdfm.version', 'sdfm.schema', 'sdfm.not_finite', 'sdfm.id', 'sdfm.cap_document',
    'sdfm.hash_mismatch', 'sdfm.mesher_refused', 'sdfm.material_unmappable', 'sdfm.worker', 'sdfm.disposed',
  ];
  assert.deepEqual(CODES.slice(0, existing.length), existing);
  assert.ok(CODES.includes('sdfm.scene_format'));
  assert.ok(CODES.includes('sdfm.scene_resolve'));
  assert.ok(CODES.indexOf('sdfm.scene_format') < CODES.indexOf('sdfm.scene_resolve'));
});

test('decodes an S24 file, sharing the buffer behind two views, and wraps its payload', () => {
  const decoded = decodeMeshFile(encodeMeshFile(validMeshFile()));
  const payload = meshFilePayload(decoded);
  const mesh = payload.groups[0].parts[0].meshes[0];
  assert.equal(payload.groups[0].group, 'root');
  assert.equal(payload.groups[0].parts[0].name, 'Body');
  assert.ok(mesh.positions instanceof Float32Array);
  assert.ok(mesh.normals instanceof Float32Array);
  assert.ok(mesh.indices instanceof Uint32Array);
  assert.ok(mesh.uvs instanceof Float32Array);
  assert.equal(mesh.positions.buffer, mesh.normals.buffer);
  assert.deepEqual(Array.from(mesh.indices), [0, 1, 2]);
  assert.deepEqual(payload.materials, [{ id: 'paint', kind: 'standard' }]);
});

test('format byte 2 is sdfm.scene_resolve', () => {
  const bytes = encodeMeshFile(validMeshFile());
  bytes[0] = 2;
  sceneError(() => decodeMeshFile(bytes));
});

test('a truncated last buffer is sdfm.scene_resolve', () => {
  const bytes = encodeMeshFile(validMeshFile());
  sceneError(() => decodeMeshFile(bytes.slice(0, bytes.length - 1)));
});

test('one extra trailing byte is sdfm.scene_resolve', () => {
  const bytes = encodeMeshFile(validMeshFile());
  const longer = new Uint8Array(bytes.length + 1);
  longer.set(bytes);
  sceneError(() => decodeMeshFile(longer));
});

test('an unknown $array BigInt64Array is sdfm.scene_resolve', () => {
  const bytes = encodeMeshFile({
    transfer: [],
    parts: 0,
    triangles: 0,
    data: { groups: [], materials: [], probe: { $array: 'BigInt64Array', buffer: 0, offset: 0, length: 8 } },
  });
  sceneError(() => decodeMeshFile(bytes));
});

test('a 15-number frame is refused, naming the part', () => {
  const value = validMeshFile();
  value.data.groups[0].parts[0].frame = new Float64Array(15);
  sceneError(() => meshFilePayload(value));
});

test('an index equal to the vertex count is refused, naming the part', () => {
  const value = validMeshFile();
  value.data.groups[0].parts[0].meshes[0].indices = new Uint32Array([0, 1, 3]);
  sceneError(() => meshFilePayload(value));
});

import { readFile } from 'node:fs/promises';
import { createFakeEngine } from './fake-engine.js';
import { loadProceduralScene } from '../src/procedural.js';
import { importProcedural } from '../src/mama-bridge.js';

const fixture = JSON.parse(await readFile(new URL('./fixtures/procedural-mesh.json', import.meta.url), 'utf8'));
const material = JSON.parse(await readFile(new URL('./fixtures/compiled-basic.json', import.meta.url), 'utf8'));
const mama = async () => ({
  parseProceduralScene(text) {
    if (text === 'marker') throw { code: 'PSCENE_SCHEMA', message: 'bad source', path: '/sources/0' };
    return JSON.parse(text);
  },
});
function recordingEngine() {
  const engine = createFakeEngine();
  for (const name of ['setVisible', 'removeNode', 'removeMesh', 'setMaterialTintMode']) {
    engine.calls[name] = [];
    engine[name] = (...args) => engine.calls[name].push(args);
  }
  return engine;
}
function sceneBytes() {
  const value = validMeshFile();
  const part = value.data.groups[0].parts[0];
  part.meshes[0].material = material.name;
  part.meshes.push({ ...part.meshes[0], uvs: null });
  value.data.materials = [material];
  return encodeMeshFile(value);
}
const resolveMesh = async () => sceneBytes();

test('loads mesh references deterministically, preserves transforms, visibility and disposal', async () => {
  const engines = [recordingEngine(), recordingEngine()];
  const handles = [];
  for (const engine of engines) handles.push(await loadProceduralScene(engine, null, fixture, { mama, resolveMesh }));
  assert.deepStrictEqual(engines[0].calls, engines[1].calls);
  const { calls } = engines[0];
  const root = handles[0].root;
  const nodes = calls.addNode.filter((node) => node.parent === root);
  assert.deepStrictEqual(nodes.map(({ t, r, s }) => ({ t, r, s })), fixture.instances.map(({ t, r, s }) => ({ t, r, s })));
  assert.equal(calls.addMeshInstance.length, 6);
  assert.equal(calls.addMesh.length, 4);
  assert.equal(calls.addMaterialFunction.length, 1);
  assert.equal(handles[0].instanceCount, 3);
  assert.equal(handles[0].sourceCount, 2);
  assert.deepStrictEqual(handles[0].stats(), { instances: 3, meshes: 4, pending: 0 });
  handles[0].setVisible(false);
  assert.deepStrictEqual(calls.setVisible, calls.addMeshInstance.map(({ id }) => [id, false]));
  handles[0].dispose();
  handles[0].dispose();
  assert.deepStrictEqual(calls.removeNode, [[root]]);
  assert.deepStrictEqual(calls.removeMesh, calls.addMesh.map(({ id }) => [id]));
  for (const fn of [() => handles[0].stats(), () => handles[0].setVisible(true)]) {
    assert.throws(fn, { code: 'sdfm.disposed' });
  }
});

test('starts all mesh resolutions before waiting, and reports failures with their paths', async () => {
  const paths = [];
  await assert.rejects(loadProceduralScene(recordingEngine(), null, fixture, {
    mama, resolveMesh: (path) => {
      paths.push(path);
      return path === fixture.sources[0].mesh ? Promise.reject(new Error('unavailable')) : Promise.resolve(sceneBytes());
    },
  }), (e) => e.code === 'sdfm.scene_resolve' && e.message.includes(fixture.sources[0].mesh));
  assert.deepStrictEqual(paths, fixture.sources.map((source) => source.mesh));
  await assert.rejects(loadProceduralScene(recordingEngine(), null, fixture, { mama, resolveMesh: async () => 'bad' }),
    (e) => e.code === 'sdfm.scene_resolve' && e.message.includes(fixture.sources[0].mesh));
});

test('refuses invalid scenes, trace mode, mesh deforms and sdf references', async () => {
  const engine = recordingEngine();
  await assert.rejects(loadProceduralScene(engine, null, 'marker', { mama }),
    (e) => e.code === 'sdfm.scene_format' && e.message.includes('PSCENE_SCHEMA') && e.path === '/sources/0');
  await assert.rejects(loadProceduralScene(engine, null, fixture, { mama, mode: 'trace' }), { code: 'sdfm.scene_format' });
  const bent = structuredClone(fixture);
  bent.instances[0].deform = {};
  await assert.rejects(loadProceduralScene(engine, null, bent, { mama }), { code: 'sdfm.scene_format' });
  await assert.rejects(loadProceduralScene(engine, null, { ...fixture, variant: 'sdf-refs' }, { mama }),
    (e) => e.code === 'sdfm.scene_format' && e.message.includes('SDF reference scenes load in the next version'));
});

test('procedural bridge reports the failing module specifier', async () => {
  await assert.rejects(importProcedural(new URL('./missing/', import.meta.url)),
    (e) => e.code === 'sdfm.worker' && e.message.includes('cannot import MAMA module') && e.message.includes('/missing/'));
});

test('handles existing material functions, graph tint and part frames', async () => {
  const engine = recordingEngine();
  const graph = structuredClone(material);
  graph.body = `  let tint = mat_instance_tint;\n${graph.body}`;
  const value = validMeshFile();
  const part = value.data.groups[0].parts[0];
  part.frame[0] = 2;
  part.frame[12] = 4;
  part.meshes[0].material = graph.name;
  value.data.materials = [graph];
  const { mapMaterial } = await import('../src/map-material.js');
  const mapped = mapMaterial(graph);
  engine.addMaterialFunction(mapped.functionName, mapped.wgsl);
  const handle = await loadProceduralScene(engine, null, fixture, {
    mama, resolveMesh: async () => encodeMeshFile(value),
  });
  assert.equal(engine.calls.addMaterialFunction.length, 1);
  assert.equal(engine.calls.setMaterialTintMode.length, 2);
  assert.ok(engine.calls.setMaterialTintMode.every(([, mode]) => mode === 'graph'));
  const firstInstanceNode = engine.calls.addNode.find((node) => node.parent === handle.root);
  const partNode = engine.calls.addNode.find((node) => node.parent === firstInstanceNode.id);
  assert.deepStrictEqual(partNode.t, [4, 0, 0]);
  assert.deepStrictEqual(partNode.r, [0, 0, 0, 1]);
  assert.deepStrictEqual(partNode.s, [1, 1, 1]);
});
