import { SdfmError } from './errors.js';
import { importProcedural } from './mama-bridge.js';
import { mapMaterial } from './map-material.js';
import { decodeMeshFile, meshFilePayload } from './mesh-file.js';

/** Load a procedural scene whose sources are saved meshes. */
export async function loadProceduralScene(engine, loader, input, options = {}) {
  const mama = await (options.mama ?? importProcedural)();
  let scene;
  try {
    scene = mama.parseProceduralScene(typeof input === 'string' ? input : JSON.stringify(input));
  } catch (e) {
    throw new SdfmError('sdfm.scene_format', `${e?.code ?? 'PSCENE_FORMAT'}: ${e?.message ?? String(e)}`, e?.path ?? '');
  }
  if (options.mode !== undefined && options.mode !== 'mesh') {
    throw new SdfmError('sdfm.scene_format', 'Raytrace loading is not available in this version');
  }
  if (scene.variant === 'sdf-refs') {
    throw new SdfmError('sdfm.scene_format', 'SDF reference scenes load in the next version');
  }
  if (scene.instances.some((instance) => instance.deform !== undefined)) {
    throw new SdfmError('sdfm.scene_format', 'Mesh reference instances cannot carry deform');
  }

  const root = engine.addNode(options.parent ?? null);
  const meshes = [];
  const instances = [];
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    engine.removeNode(root);
    for (const mesh of meshes) engine.removeMesh(mesh);
  };
  try {
    const payloads = await Promise.all(scene.sources.map(async (source) => {
      let bytes;
      try {
        bytes = await options.resolveMesh(source.mesh);
        if (!(bytes instanceof Uint8Array) && !(bytes instanceof ArrayBuffer)) {
          throw new Error('expected Uint8Array or ArrayBuffer');
        }
      } catch (e) {
        throw new SdfmError('sdfm.scene_resolve', `cannot resolve mesh ${source.mesh}: ${e?.message ?? String(e)}`);
      }
      return meshFilePayload(decodeMeshFile(bytes));
    }));
    const registered = new Set();
    const sources = payloads.map((payload) => {
      const materials = new Map();
      for (const material of payload.materials) {
        const mapped = mapMaterial(material);
        if (!registered.has(mapped.functionName)) {
          try {
            engine.addMaterialFunction(mapped.functionName, mapped.wgsl);
          } catch (e) {
            if (e?.code !== 'material.fn_duplicate') throw e;
          }
          registered.add(mapped.functionName);
        }
        const handle = engine.addSurfaceMaterial({ ...mapped.desc });
        if (mapped.tintMode === 'graph') engine.setMaterialTintMode(handle, 'graph');
        materials.set(material.name, handle);
      }
      return payload.groups.flatMap((group) => group.parts.map((part) => {
        const handles = part.meshes.map((mesh) => {
          const material = materials.get(mesh.material);
          const handle = engine.addMesh({
            positions: mesh.positions, normals: mesh.normals, indices: mesh.indices,
            ...(mesh.uvs != null ? { uvs: mesh.uvs } : {}),
            subMeshes: [{ first: 0, count: mesh.indices.length, ...(material !== undefined ? { material } : {}) }],
          });
          meshes.push(handle);
          return handle;
        });
        return { ...frameToTr(part.frame), meshes: handles };
      }));
    });
    for (const instance of scene.instances) {
      const node = engine.addNode(root, instance.t, instance.r, instance.s);
      for (const part of sources[instance.source]) {
        const partNode = engine.addNode(node, part.t, part.r);
        for (const mesh of part.meshes) instances.push(engine.addMeshInstance(partNode, mesh));
      }
    }
  } catch (e) {
    dispose();
    throw e;
  }
  const checkLive = () => {
    if (disposed) throw new SdfmError('sdfm.disposed', 'procedural scene has been disposed');
  };
  return {
    root,
    instanceCount: scene.instances.length,
    sourceCount: scene.sources.length,
    setVisible(visible) {
      checkLive();
      for (const instance of instances) engine.setVisible(instance, visible);
    },
    stats() {
      checkLive();
      return { instances: scene.instances.length, meshes: meshes.length, pending: 0 };
    },
    dispose,
  };
}
function frameToTr(f) {
  const t = [f[12], f[13], f[14]];
  const col = (c) => {
    const x = f[c * 4];
    const y = f[c * 4 + 1];
    const z = f[c * 4 + 2];
    const len = Math.hypot(x, y, z) || 1;
    return [x / len, y / len, z / len];
  };
  const [c0, c1, c2] = [col(0), col(1), col(2)];
  // m(row, col) = c<col>[row]
  const m00 = c0[0], m10 = c0[1], m20 = c0[2];
  const m01 = c1[0], m11 = c1[1], m21 = c1[2];
  const m02 = c2[0], m12 = c2[1], m22 = c2[2];
  const trace = m00 + m11 + m22;
  let q;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s];
  }
  const n = Math.hypot(...q) || 1;
  return { t, r: q.map((v) => v / n) };
}
