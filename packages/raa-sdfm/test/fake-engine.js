// Fake RAA engine for the loader tests: records the Engine calls the loader makes (addMesh,
// addMaterialFunction, addSurfaceMaterial, addNode, addMeshInstance) and refuses like the real
// engine does for the rules the loader must respect (S13 sub-meshes, duplicate function names,
// unknown handles). Handles increase across every kind, starting at 1.

/** The engine's refusal (JS `RaaError`): message prefixed with the code. */
export class FakeRaaError extends Error {
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = 'RaaError';
    this.code = code;
  }
}

export function createFakeEngine() {
  let next = 1;
  const handle = () => next++;
  const calls = { addMesh: [], addMaterialFunction: [], addSurfaceMaterial: [], addNode: [], addMeshInstance: [] };
  const meshes = new Map();
  const functions = new Map();
  const materials = new Map();
  const nodes = new Map();
  const instances = new Map();

  return {
    calls,
    meshes,
    functions,
    materials,
    nodes,
    instances,

    addMesh(desc) {
      const { positions, normals, indices } = desc;
      if (!(positions instanceof Float32Array) || !(normals instanceof Float32Array) || !(indices instanceof Uint32Array)) {
        throw new FakeRaaError('api.invalid_argument', 'positions, normals (Float32Array) and indices (Uint32Array) are required');
      }
      const vertices = positions.length / 3;
      if (vertices === 0 || indices.length === 0) throw new FakeRaaError('mesh.empty', 'no vertex or no index');
      if (positions.length % 3 !== 0) throw new FakeRaaError('mesh.length_mismatch', 'positions');
      if (normals.length !== positions.length) throw new FakeRaaError('mesh.length_mismatch', 'normals');
      if (desc.uvs !== undefined && desc.uvs.length !== vertices * 2) throw new FakeRaaError('mesh.length_mismatch', 'uvs');
      if (indices.length % 3 !== 0) throw new FakeRaaError('mesh.length_mismatch', 'indices');
      const bad = indices.findIndex((i) => i >= vertices);
      if (bad >= 0) throw new FakeRaaError('mesh.index_range', `indices[${bad}] = ${indices[bad]}`);
      const subs = desc.subMeshes ?? [{ first: 0, count: indices.length }];
      const ranges = [];
      subs.forEach((s, k) => {
        const ok = Number.isInteger(s.first) && Number.isInteger(s.count) && s.first % 3 === 0 && s.count % 3 === 0
          && s.first >= 0 && s.count > 0 && s.first + s.count <= indices.length;
        if (!ok) throw new FakeRaaError('mesh.submesh_range', `subMeshes[${k}] = { first: ${s.first}, count: ${s.count} }`);
        if (s.material !== undefined && !materials.has(s.material)) {
          throw new FakeRaaError('api.unknown_handle', `subMeshes[${k}].material ${s.material}`);
        }
        ranges.push([s.first, s.first + s.count, k]);
      });
      ranges.sort((a, b) => a[0] - b[0]);
      for (let i = 1; i < ranges.length; i += 1) {
        if (ranges[i][0] < ranges[i - 1][1]) {
          throw new FakeRaaError('mesh.submesh_overlap', `subMeshes[${ranges[i - 1][2]}] and subMeshes[${ranges[i][2]}]`);
        }
      }
      const id = handle();
      calls.addMesh.push({ id, desc });
      meshes.set(id, desc);
      return id;
    },

    addMaterialFunction(name, wgsl) {
      if (typeof name !== 'string' || name === '' || typeof wgsl !== 'string' || wgsl === '') {
        throw new FakeRaaError('api.invalid_argument', 'name and wgsl must be non-empty strings');
      }
      if (functions.has(name)) throw new FakeRaaError('material.fn_duplicate', `material function '${name}' is already registered`);
      calls.addMaterialFunction.push({ name, wgsl });
      functions.set(name, wgsl);
    },

    addSurfaceMaterial(desc) {
      if (!desc || typeof desc !== 'object') throw new FakeRaaError('api.invalid_argument', 'desc is not an object');
      if (desc.function !== undefined && !functions.has(desc.function)) {
        throw new FakeRaaError('material.unknown_fn', `no material function '${desc.function}'`);
      }
      const id = handle();
      calls.addSurfaceMaterial.push({ id, desc });
      materials.set(id, desc);
      return id;
    },

    addNode(parent, t = [0, 0, 0], r = [0, 0, 0, 1], s = [1, 1, 1]) {
      if (parent !== null && !nodes.has(parent)) throw new FakeRaaError('api.unknown_handle', `parent node ${parent}`);
      const id = handle();
      const node = { id, parent, t: [...t], r: [...r], s: [...s] };
      calls.addNode.push(node);
      nodes.set(id, node);
      return id;
    },

    addMeshInstance(node, mesh) {
      if (!nodes.has(node)) throw new FakeRaaError('api.unknown_handle', `node ${node}`);
      if (!meshes.has(mesh)) throw new FakeRaaError('api.unknown_handle', `mesh ${mesh}`);
      const id = handle();
      calls.addMeshInstance.push({ id, node, mesh });
      instances.set(id, { node, mesh });
      return id;
    },
  };
}
