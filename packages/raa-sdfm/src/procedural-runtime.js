// Runtime adapter of one loaded procedural scene (section 68): drives the MAMA runtime handle
// (ProceduralRuntime via mama-bridge, Rust/WASM) and applies its sorted delta to the scene's
// engine instances. Stable keys address instances; moved instances keep their node, added
// instances are attached under the scene root, removed instances lose their subtree, and a new
// mesh key is meshed through the loader worker while every earlier mesh is kept.
//
// The handle wiring (ProceduralSceneHandle.runtime) supplies the initial key -> engine entry
// registry and the attach/detach callbacks; this file never loads a scene itself.
import { SdfmError } from './errors.js';
import { instanceParams } from './instance-params.js';

/**
 * Create the runtime adapter of one scene. `options.runtime` is a MAMA `Runtime` (or any object
 * with the same `set`/`step`/`dispose`); without it `options.mama.Runtime.create(procgen,
 * scene.runtime.input)` builds one (the object `importProcedural` returns in mama-bridge.js).
 *
 * @param {{
 *   engine: object,
 *   scene: object,
 *   runtime?: { set(values: object): void, step(tick: number): object, dispose(): void },
 *   mama?: { Runtime?: { create(procgen: object, input: object): object } },
 *   procgen?: object,
 *   root?: number | null,
 *   instances?: Map<number, object> | object[] | Record<string, object>,
 *   attach?: (instance: object, mesh: unknown) => object | number | Promise<object | number>,
 *   detach?: (entry: object) => void,
 *   move?: (entry: object, t: number[], r: number[], s: number[]) => void | Promise<void>,
 *   reattach?: (entry: object, mesh: unknown, item: object) => void | Promise<void>,
 *   remesh?: (item: { meshKey: string, source: number, deform?: object }) => unknown | Promise<unknown>,
 *   release?: (mesh: unknown) => void,
 * }} options
 *   `instances` holds one entry per already loaded scene instance: `{key, node, parts?, source?,
 *   deform?, meshKey?}`. A custom `detach` replaces the default `engine.removeNode(entry.node)`;
 *   a custom `move` replaces `engine.setTransform(entry.node, t, r, s)`. `attach` creates the
 *   hierarchy of an added instance (default: one node under `root`) and receives the freshly
 *   meshed token when its `meshKey` was just built. `reattach` swaps an existing instance to a new
 *   mesh without releasing the old one. `remesh` builds a new mesh key through the loader worker.
 * @returns {{ step(tick: number): Promise<object>, set(values: object): void, dispose(): void }}
 * @throws {SdfmError} `sdfm.scene_format` when the scene has no runtime input, `sdfm.scene_resolve`
 *   when MAMA, the runtime or a remesh/attach callback fails
 */
export function createProceduralRuntime(options = {}) {
  const { engine, scene } = options;
  if (!engine || typeof engine !== 'object') throw new TypeError('createProceduralRuntime: options.engine is required');
  if (!scene || typeof scene !== 'object') throw new TypeError('createProceduralRuntime: options.scene is required');
  if (scene.runtime == null || typeof scene.runtime.input !== 'object') {
    throw new SdfmError('sdfm.scene_format', 'the scene has no runtime input');
  }
  const runtime = options.runtime ?? createRuntime(options);
  const root = options.root ?? null;
  const entries = new Map();
  const meshes = new Map();
  let first = true;
  let disposed = false;
  let queue = Promise.resolve();

  for (const entry of initialEntries(options.instances)) {
    if (!Number.isInteger(entry.key)) throw new TypeError('createProceduralRuntime: every instance entry needs an integer key');
    entries.set(entry.key, { ...entry });
  }

  const disposedError = () => new SdfmError('sdfm.disposed', 'procedural runtime has been disposed');
  const checkLive = () => {
    if (disposed) throw disposedError();
  };

  function step(tick) {
    if (disposed) return Promise.reject(disposedError());
    if (!Number.isInteger(tick) || tick < 0) {
      return Promise.reject(new SdfmError('sdfm.scene_format', `tick must be a non-negative integer, got ${tick}`));
    }
    const run = queue.then(() => apply(tick));
    queue = run.then(() => undefined, () => undefined);
    return run;
  }

  function set(values) {
    checkLive();
    try {
      runtime.set(values);
    } catch (e) {
      throw wrap(e);
    }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    const errors = [];
    for (const entry of entries.values()) {
      try {
        detach(entry);
      } catch (e) {
        errors.push(e);
      }
    }
    entries.clear();
    for (const token of meshes.values()) {
      try {
        options.release?.(token);
      } catch (e) {
        errors.push(e);
      }
    }
    meshes.clear();
    try {
      runtime.dispose();
    } catch (e) {
      errors.push(e);
    }
    if (errors.length === 1) throw errors[0];
    if (errors.length > 1) throw new AggregateError(errors, 'procedural runtime disposal failed');
  }

  const settled = () => ({ moved: 0, remeshed: 0, added: 0, removed: 0, ok: false, issues: [], disposed: true });

  async function apply(tick) {
    checkLive();
    let delta;
    try {
      delta = runtime.step(tick);
    } catch (e) {
      throw wrap(e);
    }
    if (!delta || typeof delta !== 'object') {
      throw new SdfmError('sdfm.scene_resolve', 'the runtime step returned no delta');
    }
    const issues = Array.isArray(delta.issues) ? delta.issues : [];
    // A rejected evaluation changes nothing: the previous valid state stays.
    if (delta.ok === false || !Array.isArray(delta.added) || !Array.isArray(delta.moved)) {
      return { moved: 0, remeshed: 0, added: 0, removed: 0, ok: false, issues, delta };
    }

    // Build every genuinely new mesh key before touching the scene, so a failed worker leaves the
    // previous valid state. The first step compares against no output and so lists the meshes the
    // loader already made; those are not remeshed.
    const remesh = first ? [] : (Array.isArray(delta.remesh) ? delta.remesh : []);
    const tokens = new Map();
    const built = [];
    for (const item of remesh) {
      if (!item || typeof item.meshKey !== 'string' || tokens.has(item.meshKey)) continue;
      if (typeof options.remesh !== 'function') break;
      if (meshes.has(item.meshKey)) {
        tokens.set(item.meshKey, meshes.get(item.meshKey));
        continue;
      }
      let token;
      try {
        token = await options.remesh?.(item);
      } catch (e) {
        if (disposed) return settled();
        throw wrap(e);
      }
      if (disposed) {
        try { options.release?.(token); } catch { /* the adapter is already going away */ }
        return settled();
      }
      meshes.set(item.meshKey, token);
      tokens.set(item.meshKey, token);
      built.push(item);
    }
    if (disposed) return settled();

    let removed = 0;
    for (const key of Array.isArray(delta.removed) ? delta.removed : []) {
      const entry = entries.get(key);
      if (entry === undefined) continue;
      detach(entry);
      entries.delete(key);
      removed += 1;
    }

    let moved = 0;
    for (const item of delta.moved) {
      const entry = entries.get(item?.key);
      if (entry === undefined) continue;
      await move(entry, item);
      if (disposed) return settled();
      moved += 1;
    }

    for (const item of built) {
      const token = tokens.get(item.meshKey);
      for (const entry of entries.values()) {
        if (entry.meshKey === item.meshKey || entry.source !== item.source) continue;
        if (!sameDeform(entry.deform, item.deform)) continue;
        if (typeof options.reattach !== 'function') continue;
        try {
          await options.reattach(entry, token, item);
        } catch (e) {
          if (disposed) return settled();
          throw wrap(e);
        }
        if (disposed) return settled();
        entry.meshKey = item.meshKey;
        entry.mesh = token;
      }
    }

    let added = 0;
    for (const instance of delta.added) {
      if (!instance || !Number.isInteger(instance.key) || entries.has(instance.key)) continue;
      const token = tokens.get(instance.meshKey) ?? meshes.get(instance.meshKey);
      let entry;
      try {
        entry = asEntry(await attach(instance, token));
      } catch (e) {
        if (disposed) return settled();
        throw wrap(e);
      }
      if (disposed) {
        try { detach(entry); } catch { /* the adapter is already going away */ }
        return settled();
      }
      entries.set(instance.key, {
        ...entry,
        key: instance.key,
        source: instance.source,
        deform: instance.deform,
        meshKey: instance.meshKey,
        mesh: token,
      });
      applyInstance(entry, instance);
      added += 1;
    }

    first = false;
    return { moved, remeshed: built.length, added, removed, ok: true, issues, delta };
  }

  function attach(instance, token) {
    if (typeof options.attach === 'function') return options.attach(instance, token);
    return engine.addNode(root, instance.t, instance.r, instance.s);
  }

  function detach(entry) {
    if (typeof options.detach === 'function') return options.detach(entry);
    if (typeof entry?.dispose === 'function') return entry.dispose();
    if (entry?.node != null) return engine.removeNode(entry.node);
  }

  function move(entry, item) {
    if (typeof options.move === 'function') return options.move(entry, item.t, item.r, item.s);
    if (typeof engine.setTransform !== 'function') {
      throw new SdfmError('sdfm.scene_resolve', 'the engine has no setTransform to move runtime instances');
    }
    return engine.setTransform(entry.node, item.t, item.r, item.s);
  }

  // An added instance carries procgen attributes and a LOD bias like a loaded one (S69); apply
  // them to every engine instance the attach callback spawned for it.
  function applyInstance(entry, instance) {
    const params = instanceParams(instance.attrs);
    const bias = instance.lod?.bias;
    const setBias = Number.isInteger(bias) && bias !== 0 && typeof engine.setInstanceLodBias === 'function';
    if (params === null && !setBias) return;
    for (const handle of instanceHandles(entry.parts)) {
      if (params !== null && typeof engine.setInstanceParams === 'function') engine.setInstanceParams(handle, params);
      if (setBias) engine.setInstanceLodBias(handle, bias);
    }
  }

  return { step, set, dispose };
}

function instanceHandles(parts) {
  if (Array.isArray(parts)) return parts;
  if (!parts || typeof parts !== 'object') return [];
  const values = parts instanceof Map ? [...parts.values()] : Object.values(parts);
  return values.map((part) => (part && typeof part === 'object' ? part.instance : part)).filter((part) => part != null);
}

function createRuntime(options) {
  const create = options.mama?.Runtime?.create;
  if (typeof create !== 'function') {
    throw new SdfmError('sdfm.scene_resolve',
      'runtime scenes need MAMA Runtime: pass options.runtime, or options.mama with Runtime and options.procgen');
  }
  try {
    return create(options.procgen, options.scene.runtime.input);
  } catch (e) {
    throw wrap(e);
  }
}

function initialEntries(value) {
  if (value == null) return [];
  if (value instanceof Map) return [...value.entries()].map(([key, entry]) => ({ key, ...entry }));
  if (Array.isArray(value)) return value;
  return Object.entries(value).map(([key, entry]) => ({ key: Number(key), ...entry }));
}

function asEntry(value) {
  if (typeof value === 'number') return { node: value };
  if (value && typeof value === 'object') return { ...value };
  throw new SdfmError('sdfm.scene_resolve', 'the attach callback returned no instance entry');
}

function sameDeform(a, b) {
  return stable(a) === stable(b);
}

function stable(value) {
  if (value == null || typeof value !== 'object') return JSON.stringify(value ?? null);
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(',')}}`;
}

function wrap(e) {
  if (e && typeof e === 'object' && typeof e.code === 'string') return e;
  return new SdfmError('sdfm.scene_resolve', e instanceof Error ? e.message : String(e));
}
