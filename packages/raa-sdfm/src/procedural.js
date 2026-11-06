import { SdfmError } from './errors.js';
import { importProcedural } from './mama-bridge.js';
import { loadProceduralTrace } from './procedural-trace.js';
import { mapMaterial } from './map-material.js';
import { decodeMeshFile, meshFilePayload } from './mesh-file.js';
import { canonicalJson, sha256Hex } from './canonical.js';
import { instanceParams } from './instance-params.js';

/**
 * Load one `sdfm.procedural-scene` v1 file (section 66). Mesh sources are decoded and uploaded;
 * SDF reference sources are resolved into in-memory game assets and loaded through `loader`.
 * Instances are never flattened: each one is its own engine node.
 */
export async function loadProceduralScene(engine, loader, input, options = {}) {
  const mama = await (options.mama ?? importProcedural)();
  let scene;
  try {
    scene = mama.parseProceduralScene(typeof input === 'string' ? input : JSON.stringify(input));
  } catch (e) {
    throw new SdfmError('sdfm.scene_format', `${e?.code ?? 'PSCENE_FORMAT'}: ${e?.message ?? String(e)}`, e?.path ?? '');
  }
  if (options.mode !== undefined && options.mode !== 'mesh' && options.mode !== 'trace') {
    throw new SdfmError('sdfm.scene_format', `unknown procedural scene mode '${options.mode}'`);
  }
  if (options.mode === 'trace') return loadProceduralTrace(engine, loader, scene, mama, options);
  if (scene.variant === 'mesh-refs' && scene.instances.some((instance) => instance.deform !== undefined)) {
    throw new SdfmError('sdfm.scene_format', 'Mesh reference instances cannot carry deform');
  }

  const root = engine.addNode(options.parent ?? null);
  const meshes = [];
  const instances = [];
  let meshCount = 0;
  let attributed = 0;
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    engine.removeNode(root);
    for (const mesh of meshes) engine.removeMesh(mesh);
  };
  try {
    if (scene.variant === 'sdf-refs') {
      const loaded = await loadSdfRefs(engine, loader, scene, mama, options, root);
      instances.push(...loaded.instances);
      meshCount = loaded.meshes;
      attributed = loaded.attributed;
    } else {
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
        const spawned = [];
        for (const part of sources[instance.source]) {
          const partNode = engine.addNode(node, part.t, part.r);
          for (const mesh of part.meshes) {
            const handle = engine.addMeshInstance(partNode, mesh);
            instances.push(handle);
            spawned.push(handle);
          }
        }
        attributed += applyInstanceParams(engine, instance, spawned);
      }
      meshCount = meshes.length;
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
      const stats = { instances: scene.instances.length, meshes: meshCount, pending: 0 };
      if (attributed > 0) stats.attributed = attributed;
      return stats;
    },
    dispose,
  };
}

/**
 * Resolve an sdf-refs scene: asset sources become game-asset loads, composites (csg, realized) and
 * bent instances become in-memory game assets built by MAMA's sceneSourceDocument.
 */
async function loadSdfRefs(engine, loader, scene, mama, options, root) {
  const files = [];
  for (const source of scene.sources) {
    if (source.kind === 'asset' && !files.includes(source.file)) files.push(source.file);
  }
  const parsedAssets = new Map();
  await Promise.all(files.map(async (file) => {
    let value;
    try {
      value = await options.resolveAsset(file);
    } catch (e) {
      throw new SdfmError('sdfm.scene_resolve', `cannot resolve asset ${file}: ${messageOf(e)}`);
    }
    if (typeof value === 'string') {
      try {
        value = JSON.parse(value);
      } catch (e) {
        throw new SdfmError('sdfm.scene_resolve', `asset ${file} is not valid JSON: ${messageOf(e)}`);
      }
    }
    parsedAssets.set(file, value);
  }));
  const lookup = (file) => parsedAssets.get(file) ?? null;

  const loaded = [];
  const remember = (asset) => {
    if (!loaded.includes(asset)) loaded.push(asset);
    return asset;
  };

  const assetByFile = new Map();
  for (const file of files) assetByFile.set(file, remember(await loader.load(parsedAssets.get(file))));

  const sourceAssets = new Map();
  for (const [index, source] of scene.sources.entries()) {
    if (source.kind === 'asset') {
      sourceAssets.set(index, assetByFile.get(source.file));
      continue;
    }
    if (source.kind !== 'csg' && source.kind !== 'realized') continue;
    const doc = sourceDocument(mama, scene, index, lookup);
    const envelope = await buildEnvelope(mama, scene, index, doc, parsedAssets,
      `procedural.${scene.name}.c${index}`, `${scene.name} source ${index}`);
    sourceAssets.set(index, remember(await loader.load(envelope)));
  }

  const bendAssets = new Map();
  const bendCounts = new Map();
  const instanceAssets = [];
  for (const instance of scene.instances) {
    if (instance.deform === undefined) {
      instanceAssets.push(sourceAssets.get(instance.source));
      continue;
    }
    const paramsKey = canonicalJson(instance.deform.params);
    let perSource = bendAssets.get(instance.source);
    if (!perSource) {
      perSource = new Map();
      bendAssets.set(instance.source, perSource);
    }
    let asset = perSource.get(paramsKey);
    if (asset === undefined) {
      const k = bendCounts.get(instance.source) ?? 0;
      bendCounts.set(instance.source, k + 1);
      const doc = sourceDocument(mama, scene, instance.source, lookup);
      doc.nodes.source.deformers = [{ id: 'bend', kind: 'bend', enabled: true, params: instance.deform.params }];
      const envelope = await buildEnvelope(mama, scene, instance.source, doc, parsedAssets,
        `procedural.${scene.name}.b${instance.source}-${k}`, `${scene.name} source ${instance.source} bend ${k}`);
      asset = remember(await loader.load(envelope));
      perSource.set(paramsKey, asset);
    }
    instanceAssets.push(asset);
  }

  const instances = [];
  let attributed = 0;
  for (const [index, instance] of scene.instances.entries()) {
    const node = engine.addNode(root, instance.t, instance.r, instance.s);
    const spawned = loader.spawn(instanceAssets[index], node);
    const parts = [];
    for (const part of Object.values(spawned?.parts ?? {})) {
      if (part && part.instance !== undefined) {
        instances.push(part.instance);
        parts.push(part.instance);
      }
    }
    attributed += applyInstanceParams(engine, instance, parts);
  }
  const meshes = loaded.reduce((count, asset) => count + (Array.isArray(asset?.parts) ? asset.parts.length : 0), 0);
  return { instances, meshes, attributed };
}

/**
 * Forward one scene instance's procgen attributes and detail hint to every engine instance it spawned.
 * `attributed` counts the scene instances whose attributes were set (once, however many parts they have).
 */
function applyInstanceParams(engine, instance, handles) {
  const params = instanceParams(instance.attrs);
  const bias = instance.lod?.bias;
  const setBias = Number.isInteger(bias) && bias !== 0 && typeof engine.setInstanceLodBias === 'function';
  if (params === null && !setBias) return 0;
  for (const handle of handles) {
    if (params !== null) engine.setInstanceParams(handle, params);
    if (setBias) engine.setInstanceLodBias(handle, bias);
  }
  return params !== null && handles.length > 0 ? 1 : 0;
}

function sourceDocument(mama, scene, index, lookup) {
  try {
    return mama.sceneSourceDocument(scene, index, lookup);
  } catch (e) {
    throw new SdfmError('sdfm.scene_resolve', messageOf(e));
  }
}

/** The in-memory sdfm.game-asset envelope, keys in validate.js ASSET_KEYS order. */
async function buildEnvelope(mama, scene, index, doc, parsedAssets, id, name) {
  const source = scene.sources[index];
  const file = firstAssetFile(scene, index);
  const asset = file === null ? null : parsedAssets.get(file);
  const hash = await sha256Hex(mama.serializeDocument(doc));
  return {
    format: 'sdfm.game-asset',
    version: 1,
    game: scene.game,
    id,
    name,
    category: asset?.category ?? 'prop',
    world: asset?.world ?? scene.game,
    mesher: asset?.mesher ?? { output: 'paramesh' },
    document: doc,
    collision: [],
    anchors: {},
    bounds: { min: [...source.bounds.min], max: [...source.bounds.max] },
    hash,
  };
}

/** The first asset file a composite reaches, depth-first: csg a before b, members in order. */
function firstAssetFile(scene, index, seen = new Set()) {
  if (seen.has(index)) return null;
  seen.add(index);
  const source = scene.sources[index];
  if (!source) return null;
  if (source.kind === 'asset') return source.file;
  if (source.kind === 'csg') return firstAssetFile(scene, source.a, seen) ?? firstAssetFile(scene, source.b, seen);
  if (source.kind === 'realized') {
    for (const member of source.members) {
      const file = firstAssetFile(scene, member.source, seen);
      if (file !== null) return file;
    }
  }
  return null;
}

function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
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
