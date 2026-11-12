// S30 saved-mesh clusters section, read by raa-sdfm (P3-SDFM-READER). A plain JS validator: no MAMA import, no wasm.
// The six wire rules are checked in order and the first failure wins. A refusal is an SdfmError `sdfm.scene_resolve`
// whose message names the mesh (`<group>/<part>#<index>`) and the field (`clusters.<field>`). A valid section is
// returned as the same object, so the typed arrays stay the ones the decoder built.
import { SdfmError } from './errors.js';

const KEYS = [
  'version', 'algorithm', 'vertices', 'triangles', 'clusters', 'nodes', 'rootNode', 'hierarchyDepth', 'levels',
  'quantOrigin', 'quantStep', 'uvOrigin', 'uvStep', 'boundsCenter', 'boundsRadius', 'sourceTriangles',
];
const ARRAYS = [['vertices', Uint16Array], ['triangles', Uint32Array], ['clusters', Uint8Array], ['nodes', Uint8Array]];
const CLUSTER_BYTES = 48;
const NODE_BYTES = 48;

function fail(message) {
  throw new SdfmError('sdfm.scene_resolve', message);
}

const isCount = (value) => Number.isSafeInteger(value) && value >= 0;
const isFiniteTuple = (value, length) => Array.isArray(value) && value.length === length && value.every(Number.isFinite);

/**
 * Validate one S30 clusters section against the six wire rules and return it unchanged.
 * @param {unknown} value the `clusters` key of a saved mesh record
 * @param {string} where `<group>/<part>#<mesh index>`, named in every refusal
 * @returns {object} the same section object
 * @throws {SdfmError} `sdfm.scene_resolve` with `mesh: part <where> clusters.<field>: <reason>`
 */
export function validateClusterSection(value, where) {
  const refuse = (field, reason) => fail(`mesh: part ${where} clusters.${field}: ${reason}`);
  if (value === null || typeof value !== 'object' || Array.isArray(value)
    || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) refuse('keys', 'not a plain object');
  const missing = KEYS.find((key) => !Object.hasOwn(value, key));
  if (missing !== undefined) refuse('keys', `missing key '${missing}'`);
  const extra = Object.keys(value).find((key) => !KEYS.includes(key));
  if (extra !== undefined) refuse('keys', `unknown key '${extra}'`);
  if (value.version !== 1) refuse('version', `${value.version} is not 1`);
  if (value.algorithm !== 1) refuse('algorithm', `${value.algorithm} is not 1`);

  for (const [name, Type] of ARRAYS) if (!(value[name] instanceof Type)) refuse(name, `is not a ${Type.name}`);
  if (value.vertices.length % 8 !== 0) refuse('vertices', `length ${value.vertices.length} is not a multiple of 8`);
  if (value.clusters.length % CLUSTER_BYTES !== 0) refuse('clusters', `length ${value.clusters.length} is not a multiple of 48`);
  if (value.nodes.length % NODE_BYTES !== 0) refuse('nodes', `length ${value.nodes.length} is not a multiple of 48`);
  for (const [name] of ARRAYS) if (value[name].length === 0) refuse(name, 'is empty');

  const clusterCount = value.clusters.length / CLUSTER_BYTES;
  const nodeCount = value.nodes.length / NODE_BYTES;
  if (!isCount(value.rootNode) || value.rootNode >= nodeCount) refuse('rootNode', `${value.rootNode} is not an integer below ${nodeCount}`);
  if (!isCount(value.hierarchyDepth) || value.hierarchyDepth < 1 || value.hierarchyDepth > 64) refuse('hierarchyDepth', `${value.hierarchyDepth} is not in 1..64`);
  if (!isCount(value.levels) || value.levels < 1 || value.levels > 256) refuse('levels', `${value.levels} is not in 1..256`);
  if (!isCount(value.sourceTriangles)) refuse('sourceTriangles', `${value.sourceTriangles} is not a safe non-negative integer`);

  if (!isFiniteTuple(value.quantOrigin, 3)) refuse('quantOrigin', 'is not three finite numbers');
  if (!(Number.isFinite(value.quantStep) && value.quantStep > 0)) refuse('quantStep', `${value.quantStep} is not a finite number above 0`);
  if (!isFiniteTuple(value.uvOrigin, 2)) refuse('uvOrigin', 'is not two finite numbers');
  if (!isFiniteTuple(value.uvStep, 2) || value.uvStep.some((step) => step < 0)) refuse('uvStep', 'is not two finite numbers >= 0');
  if (!isFiniteTuple(value.boundsCenter, 3)) refuse('boundsCenter', 'is not three finite numbers');
  if (!(Number.isFinite(value.boundsRadius) && value.boundsRadius >= 0)) refuse('boundsRadius', `${value.boundsRadius} is not a finite number >= 0`);

  const records = (array) => new DataView(array.buffer, array.byteOffset, array.byteLength);
  const clusters = records(value.clusters);
  const vertexTotal = value.vertices.length / 8;
  for (let i = 0; i < clusterCount; i++) {
    const at = i * CLUSTER_BYTES;
    const vertexOffset = clusters.getUint32(at + 36, true);
    const triangleOffset = clusters.getUint32(at + 40, true);
    const counts = clusters.getUint32(at + 44, true);
    const vertexCount = counts & 0xff;
    const triangleCount = (counts >>> 8) & 0xff;
    const field = `clusters[${i}]`;
    if (vertexCount < 1 || vertexCount > 255) refuse(field, `vertex count ${vertexCount} is not in 1..255`);
    if (triangleCount < 1 || triangleCount > 128) refuse(field, `triangle count ${triangleCount} is not in 1..128`);
    if (vertexOffset + vertexCount > vertexTotal) refuse(field, `vertices ${vertexOffset}+${vertexCount} exceed ${vertexTotal}`);
    if (triangleOffset + triangleCount > value.triangles.length) refuse(field, `triangles ${triangleOffset}+${triangleCount} exceed ${value.triangles.length}`);
  }

  const nodes = records(value.nodes);
  for (let i = 0; i < nodeCount; i++) {
    const at = i * NODE_BYTES;
    const first = nodes.getUint32(at + 36, true);
    const count = nodes.getUint32(at + 40, true);
    const leaf = nodes.getUint32(at + 44, true);
    const field = `nodes[${i}]`;
    if (count < 1) refuse(field, `count ${count} is below 1`);
    if (leaf !== 0 && leaf !== 1) refuse(field, `is_leaf ${leaf} is not 0 or 1`);
    const limit = leaf === 1 ? clusterCount : nodeCount;
    if (first + count > limit) refuse(field, `first ${first} + count ${count} exceeds the ${leaf === 1 ? 'cluster' : 'node'} count ${limit}`);
  }
  return value;
}
