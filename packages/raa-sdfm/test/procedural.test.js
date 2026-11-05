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
