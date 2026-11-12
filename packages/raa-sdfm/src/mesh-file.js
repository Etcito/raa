// S24 saved mesh (`assets/<world>/<id>.mesh`): the meshWire encoding of a meshed asset.
// Byte 0 = 1, big-endian u32 header length @1 and buffer count @5, a UTF-8 JSON header where an ArrayBuffer is
// {$buffer: i} and a view is {$array: name, buffer: i, offset: byteOffset, length: byteLength}, then per buffer a
// big-endian u32 length and its bytes. Plain JS port of MAMA src/core/game/mesh-file.ts decodeMeshFile; it does not
// import MAMA. Every refusal is an SdfmError `sdfm.scene_resolve`.
import { SdfmError } from './errors.js';
import { validateClusterSection } from './mesh-clusters.js';

const ARRAY_TYPES = Object.freeze({
  Float32Array,
  Float64Array,
  Uint32Array,
  Uint8Array,
  Int32Array,
  Uint16Array,
  Int16Array,
  Int8Array,
});

function fail(message) {
  throw new SdfmError('sdfm.scene_resolve', message);
}

function arrayBufferAt(buffers, index, where) {
  if (!Number.isInteger(index) || index < 0 || index >= buffers.length) fail(`${where}: buffer ${index} is not in the file`);
  return buffers[index];
}

/**
 * Decode an S24 saved mesh file. The value keeps the header fields and a `data` payload whose typed arrays are views
 * on the decoded buffers (two views sharing one buffer stay shared).
 * @param {Uint8Array} bytes the saved mesh bytes
 * @returns {object} the decoded mesh file value
 * @throws {SdfmError} `sdfm.scene_resolve` for a bad format byte, a truncated buffer, trailing bytes, an unknown array
 *   type or an invalid JSON header
 */
export function decodeMeshFile(bytes) {
  if (!(bytes instanceof Uint8Array)) fail('mesh: pass the saved file as a Uint8Array');
  if (bytes.length < 9) fail('mesh: the file is shorter than its 9-byte header');
  if (bytes[0] !== 1) fail(`mesh: format byte ${bytes[0]} is not 1`);

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = view.getUint32(1);
  const bufferCount = view.getUint32(5);
  const headerStart = 9;
  const buffersStart = headerStart + headerLength;
  if (buffersStart > bytes.length) fail('mesh: the JSON header runs past the end of the file');

  const buffers = [];
  let offset = buffersStart;
  for (let i = 0; i < bufferCount; i++) {
    if (offset + 4 > bytes.length) fail(`mesh: buffer ${i} has a truncated length`);
    const size = view.getUint32(offset);
    offset += 4;
    if (offset + size > bytes.length) fail(`mesh: buffer ${i} is truncated`);
    buffers.push(bytes.slice(offset, offset + size).buffer);
    offset += size;
  }
  if (offset !== bytes.length) fail(`mesh: ${bytes.length - offset} trailing bytes after the last buffer`);

  const header = new TextDecoder().decode(bytes.subarray(headerStart, buffersStart));
  try {
    return JSON.parse(header, (_key, value) => {
      if (value === null || typeof value !== 'object') return value;
      if ('$buffer' in value) return arrayBufferAt(buffers, value.$buffer, 'mesh header');
      if ('$array' in value) {
        const Type = ARRAY_TYPES[value.$array];
        if (!Type) fail(`mesh: unsupported mesh array '${value.$array}'`);
        const buffer = arrayBufferAt(buffers, value.buffer, `mesh array '${value.$array}'`);
        const byteOffset = value.offset;
        const byteLength = value.length;
        if (!Number.isInteger(byteOffset) || !Number.isInteger(byteLength) || byteOffset < 0 || byteLength < 0
          || byteOffset + byteLength > buffer.byteLength) {
          fail(`mesh: array '${value.$array}' is outside its buffer`);
        }
        if (byteLength % Type.BYTES_PER_ELEMENT !== 0) fail(`mesh: array '${value.$array}' length ${byteLength} is not a whole number of elements`);
        return new Type(buffer, byteOffset, byteLength / Type.BYTES_PER_ELEMENT);
      }
      return value;
    });
  } catch (error) {
    if (error instanceof SdfmError) throw error;
    fail(`mesh: the JSON header is invalid (${error && error.message ? error.message : String(error)})`);
  }
}

function partLabel(group, part, index) {
  const groupName = typeof group === 'object' && group !== null && typeof group.group === 'string' ? group.group : `#${index}`;
  const partName = typeof part === 'object' && part !== null
    ? (typeof part.partId === 'string' ? part.partId : typeof part.name === 'string' ? part.name : `#${index}`)
    : `#${index}`;
  return `${groupName}/${partName}`;
}

function meshArrays(mesh, where, index) {
  if (!(mesh.positions instanceof Float32Array)) fail(`mesh: part ${where} positions are not a Float32Array`);
  if (!(mesh.normals instanceof Float32Array)) fail(`mesh: part ${where} normals are not a Float32Array`);
  if (!(mesh.indices instanceof Uint32Array)) fail(`mesh: part ${where} indices are not a Uint32Array`);
  if (!(mesh.uvs === null || mesh.uvs instanceof Float32Array)) fail(`mesh: part ${where} uvs are neither null nor a Float32Array`);
  if (mesh.positions.length !== mesh.normals.length) fail(`mesh: part ${where} has ${mesh.positions.length} positions but ${mesh.normals.length} normals`);
  if (mesh.positions.length % 3 !== 0) fail(`mesh: part ${where} positions length ${mesh.positions.length} is not a multiple of 3`);
  const vertices = mesh.positions.length / 3;
  for (let i = 0; i < mesh.indices.length; i++) {
    if (mesh.indices[i] >= vertices) fail(`mesh: part ${where} index ${i} is ${mesh.indices[i]}, not below ${vertices} vertices`);
  }
  const clusters = mesh.clusters === undefined ? null : validateClusterSection(mesh.clusters, `${where}#${index}`);
  return { material: mesh.material, positions: mesh.positions, normals: mesh.normals, uvs: mesh.uvs, indices: mesh.indices, clusters };
}

/**
 * Validate the `data` payload of a decoded S24 file with the saved-asset rules and wrap it with its materials.
 * @param {object} value a decoded mesh file value (`decodeMeshFile`)
 * @returns {{groups: {group: string, parts: {partId: string, name: string, frame: Float64Array, meshes: object[]}[]}[], materials: object[]}}
 * @throws {SdfmError} `sdfm.scene_resolve` naming the group and part of a bad record
 */
export function meshFilePayload(value) {
  if (value === null || typeof value !== 'object') fail('mesh: the decoded mesh is not an object');
  if (!Array.isArray(value.transfer) || typeof value.parts !== 'number' || typeof value.triangles !== 'number') fail('mesh: the mesh header is invalid');
  const data = value.data;
  if (data === null || typeof data !== 'object' || !Array.isArray(data.groups) || !Array.isArray(data.materials)) fail('mesh: the mesh payload is invalid');
  const groups = data.groups.map((group, groupIndex) => {
    if (group === null || typeof group !== 'object' || !Array.isArray(group.parts)) fail(`mesh: group ${groupIndex} has no parts array`);
    const parts = group.parts.map((part, partIndex) => {
      const where = partLabel(group, part, groupIndex);
      if (part === null || typeof part !== 'object' || !(part.frame instanceof Float64Array) || part.frame.length !== 16
        || !part.frame.every(Number.isFinite) || !Array.isArray(part.meshes)) {
        fail(`mesh: part ${where} needs a finite 16-number frame and a meshes array`);
      }
      const meshes = part.meshes.map((mesh, meshIndex) => {
        if (mesh === null || typeof mesh !== 'object') fail(`mesh: part ${where} has a mesh that is not an object`);
        return meshArrays(mesh, where, meshIndex);
      });
      return { partId: part.partId, name: part.name, frame: part.frame, meshes };
    });
    return { group: group.group, parts };
  });
  return { groups, materials: data.materials };
}
