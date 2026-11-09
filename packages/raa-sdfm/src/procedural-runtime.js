// Runtime adapter of one loaded procedural scene (section 68): drives the MAMA runtime handle
// (ProceduralRuntime via mama-bridge, Rust/WASM) and applies its sorted delta to the scene's
// engine instances or to the engine's trace scene. Stable keys address instances; moved
// instances keep their node, added instances are attached under the scene root, removed
// instances lose their subtree, and a new mesh key is meshed through the existing loader while
// every earlier mesh stays visible until its replacement succeeded. Surviving instances also
// reconcile their source, mesh key and deform from the delta's complete output snapshot, so a
// metadata-only step (no moved/added/removed) retargets remesh replacements and trace updates by
// stable key, never by a remesh request's source.
//
// The runtime handle is stateful, so `step` evaluates every tick in call order. Application is
// ordered too, but a later tick's evaluation never waits for an earlier tick's slow mesh build:
// each tick records the keys and mesh keys its output still needs, a build whose key the newest
// tick no longer needs is dropped and its resource released, and one build is shared by every
// instance (and every overlapping step) that asks for the same new key.
//
// The handle wiring (ProceduralSceneHandle.runtime) supplies the initial key -> engine entry
// registry, the attach/detach callbacks and, in trace mode, the scene's trace input; this file
// never loads a scene itself.
import { SdfmError } from './errors.js';
import { instanceParams } from './instance-params.js';
import { createMamaLightOwner } from './mama-light-owner.js';

const TRACE_HEADER_WORDS = 8;
const TRACE_SOURCE_WORDS = 12;
const TRACE_MATERIAL_WORDS = 8;
const TRACE_INSTANCE_WORDS = 32;

/**
 * Create the runtime adapter of one scene. `options.runtime` is a MAMA `Runtime` (or any object
 * with the same `set`/`step`/`dispose`); without it `options.mama.Runtime.create(procgen,
 * scene.runtime.input)` builds one (the object `importProcedural` returns in mama-bridge.js).
 * M3_RUNTIME_LIGHT_HANDOFF: both loaders pass options.lights, the exact scene-owned registry
 * from M3_LIGHT_OWNER; direct callers may omit it for a zero-placement owner seeded from
 * scene.lights. M3_RUNTIME_SNAPSHOT supplies complete successful outputs: absent output leaves
 * lights unchanged, omitted/empty lights clear them, and failed evaluation preserves them.
 * Raw output.lights go to prepare before geometry application, including the first baseline;
 * commit follows successful application, rollback follows failure/disposal. Only the owner
 * applies its captured placement (including trace placement); runtime never translates lights.
 * Runtime disposal also disposes this owner, even when supplied by a loader.
 *
 * @param {{
 *   engine: object,
 *   scene: object,
 *   runtime?: { set(values: object): void, step(tick: number): object, dispose(): void },
 *   mama?: { Runtime?: { create(procgen: object, input: object): object } },
 *   procgen?: object,
 *   lights?: import('./mama-light-owner.js').MamaLightOwner,
 *   root?: number | null,
 *   instances?: Map<number, object> | object[] | Record<string, object>,
 *   attach?: (instance: object, mesh: unknown) => object | number | Promise<object | number>,
 *   detach?: (entry: object) => void,
 *   move?: (entry: object, t: number[], r: number[], s: number[]) => void | Promise<void>,
 *   reattach?: (entry: object, mesh: unknown, item: object) => void | Promise<void>,
 *   remesh?: (item: { meshKey: string, source: number, deform?: object }) => unknown | Promise<unknown>,
 *   loader?: { load(asset: object): Promise<unknown> },
 *   meshSource?: (item: { meshKey: string, source: number, deform?: object }) => object | Promise<object>,
 *   release?: (mesh: unknown) => void,
 *   trace?: {
 *     input: { sources: object[], instances: object[] },
 *     procgen: object,
 *     packTrace: (procgen: object, input: object) => Uint8Array,
 *     instancesOf?: (program: Uint8Array) => Float32Array,
 *     commit?: (program: Uint8Array, change: { topologyChanged: boolean, instances: Float32Array }) => void | Promise<void>,
 *   },
 * }} options
 *   `instances` holds one entry per already loaded scene instance: `{key, node, parts?, source?,
 *   deform?, meshKey?}`. A custom `detach` replaces the default `engine.removeNode(entry.node)`;
 *   a custom `move` replaces `engine.setTransform(entry.node, t, r, s)`. `attach` creates the
 *   hierarchy of an added instance (default: one node under `root`) and receives the freshly
 *   meshed token when its `meshKey` was just built. `reattach` swaps an existing instance to a new
 *   mesh without releasing the old one. `remesh` builds a new mesh key directly; without it,
 *   `meshSource(item)` builds the asset envelope and the existing `loader` (its worker pool)
 *   meshes it. `trace` switches stepping to the trace scene: the adapter keeps the trace input's
 *   instances in step with the delta and installs the repacked program through the public trace
 *   API (`updateTraceInstances` when only transforms changed, `setTraceScene` when the instance
 *   topology changed, `clearTraceScene` when none remain), or through `commit` when the handle
 *   merges several trace scenes.
 * @returns {{ step(tick: number): Promise<object>, set(values: object): void, dispose(): void }}
 * @throws {SdfmError} `sdfm.scene_format` when the scene has no runtime input or the trace input
 *   is missing, `sdfm.scene_resolve` when MAMA, the runtime, a remesh/attach callback or the trace
 *   packer fails
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
  const builds = new Map();
  const released = new Set();
  const liveKeys = new Set();
  const keyMesh = new Map();
  const meshUse = new Map();
  const trace = options.trace != null ? createTraceStepper(options, engine) : null;
  let seenOutput = false;
  let disposed = false;
  let queue = Promise.resolve();

  for (const entry of initialEntries(options.instances)) {
    if (!Number.isInteger(entry.key)) throw new TypeError('createProceduralRuntime: every instance entry needs an integer key');
    entries.set(entry.key, { ...entry });
  }
  const lights = options.lights ?? createMamaLightOwner(engine);
  if (options.lights == null) {
    try {
      lights.prepare(scene.lights).commit();
    } catch (e) {
      lights.dispose();
      runtime.dispose();
      throw e;
    }
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
    let delta;
    try {
      delta = runtime.step(tick);
    } catch (e) {
      return Promise.reject(wrap(e));
    }
    if (!delta || typeof delta !== 'object') {
      return Promise.reject(new SdfmError('sdfm.scene_resolve', 'the runtime step returned no delta'));
    }
    // Every tick settles which keys and mesh keys the newest output needs; an earlier tick's
    // pending build can then be recognised as obsolete while it is still running. The first
    // accepted output is the baseline the loader already installed, not a change.
    const baseline = !seenOutput;
    noteOutput(delta);
    const run = queue.then(() => apply(delta, baseline));
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
    try {
      lights.dispose();
    } catch (e) {
      errors.push(e);
    }
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
        releaseToken(token);
      } catch (e) {
        errors.push(e);
      }
    }
    meshes.clear();
    builds.clear();
    try {
      runtime.dispose();
    } catch (e) {
      errors.push(e);
    }
    if (errors.length === 1) throw errors[0];
    if (errors.length > 1) throw new AggregateError(errors, 'procedural runtime disposal failed');
  }

  const settled = () => ({ moved: 0, remeshed: 0, added: 0, removed: 0, ok: false, issues: [], disposed: true });

  async function apply(delta, baseline) {
    checkLive();
    const issues = Array.isArray(delta.issues) ? delta.issues : [];
    // A rejected evaluation changes nothing: the previous valid state stays.
    if (delta.ok === false || delta.output?.ok === false || !Array.isArray(delta.added) || !Array.isArray(delta.moved)) {
      return { moved: 0, remeshed: 0, added: 0, removed: 0, ok: false, issues, delta };
    }
    const transaction = delta.output ? lights.prepare(delta.output.lights) : null;
    try {
      // Geometry's first baseline is already installed; its lights still need reconciliation.
      let result;
      if (trace === null) result = await applyNodes(delta, issues, baseline);
      else if (baseline) result = { moved: 0, remeshed: 0, added: 0, removed: 0, ok: true, issues, delta };
      else result = await trace.apply(delta, issues);
      if (disposed) {
        transaction?.rollback();
        return settled();
      }
      transaction?.commit();
      return result;
    } catch (e) {
      transaction?.rollback();
      throw e;
    }
  }

  async function applyNodes(delta, issues, baseline) {
    // Build every genuinely new mesh key before touching the scene, so a failed worker leaves the
    // previous valid state. The baseline delta compares against no output and so lists the meshes
    // the loader already made; those are not remeshed.
    const remesh = baseline ? [] : (Array.isArray(delta.remesh) ? delta.remesh : []);
    const tokens = new Map();
    const built = [];
    const replacements = [];
    for (const item of remesh) {
      if (!item || typeof item.meshKey !== 'string' || tokens.has(item.meshKey)) continue;
      if (meshes.has(item.meshKey)) {
        tokens.set(item.meshKey, meshes.get(item.meshKey));
        replacements.push(item);
        continue;
      }
      if (!canRemesh()) break;
      if (isObsolete(item.meshKey)) continue;
      let pending = builds.get(item.meshKey);
      const starter = pending === undefined;
      if (starter) {
        pending = Promise.resolve().then(() => requestMesh(item));
        builds.set(item.meshKey, pending);
        pending.catch(() => {
          if (builds.get(item.meshKey) === pending) builds.delete(item.meshKey);
        });
      }
      let token;
      try {
        token = await pending;
      } catch (e) {
        if (builds.get(item.meshKey) === pending) builds.delete(item.meshKey);
        if (disposed) return settled();
        // A failed build whose newest output no longer uses the key is abandoned, not fatal.
        if (isObsolete(item.meshKey)) continue;
        throw wrap(e);
      }
      if (builds.get(item.meshKey) === pending) builds.delete(item.meshKey);
      if (disposed) {
        try { releaseToken(token); } catch { /* the adapter is already going away */ }
        return settled();
      }
      if (isObsolete(item.meshKey)) {
        // A newer tick removed every instance of this key: the completion is stale.
        try { releaseToken(token); } catch { /* abandoned by a newer tick */ }
        continue;
      }
      if (meshes.has(item.meshKey)) token = meshes.get(item.meshKey);
      else meshes.set(item.meshKey, token);
      tokens.set(item.meshKey, token);
      replacements.push(item);
      if (starter) built.push(item);
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

    // The complete output snapshot, keyed by stable instance key, says which surviving instance
    // needs which mesh. A remesh request names only a mesh key's source and deform, so two
    // instances sharing a source but bent differently are told apart by their own keyed metadata,
    // never by the request alone. A transform-only delta carries no snapshot and changes no mesh.
    const snapshot = outputInstances(delta);
    for (const item of replacements) {
      const token = tokens.get(item.meshKey);
      for (const entry of entries.values()) {
        if (entry.meshKey === item.meshKey) continue;
        if (typeof options.reattach !== 'function') continue;
        const target = snapshot?.get(entry.key);
        if (target === undefined || target.meshKey !== item.meshKey) continue;
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

    // Reconcile every surviving instance's source, mesh key and deform from the snapshot, so a
    // later delta knows the instance's current metadata even when no transform moved.
    if (snapshot !== null) {
      for (const entry of entries.values()) {
        const target = snapshot.get(entry.key);
        if (target === undefined) continue;
        entry.source = target.source;
        entry.deform = target.deform;
        if (typeof target.meshKey === 'string') entry.meshKey = target.meshKey;
      }
    }

    let added = 0;
    for (const instance of delta.added) {
      if (!instance || !Number.isInteger(instance.key) || entries.has(instance.key)) continue;
      // A newer tick already removed this key: its mesh completion is stale, do not attach it.
      if (!liveKeys.has(instance.key)) continue;
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

    return { moved, remeshed: built.length, added, removed, ok: true, issues, delta };
  }

  function noteOutput(delta) {
    if (delta.ok === false || delta.output?.ok === false) return;
    const added = Array.isArray(delta.added) ? delta.added : [];
    const removed = Array.isArray(delta.removed) ? delta.removed : [];
    if (!seenOutput) seenOutput = true;
    // The complete output snapshot (on the first step and whenever a surviving instance changes
    // its source, mesh key or deform) is the newest statement of what every live key needs; the
    // compact added/removed lists describe a transform-only step.
    const snapshot = outputInstances(delta);
    if (snapshot !== null) {
      liveKeys.clear();
      keyMesh.clear();
      meshUse.clear();
      for (const [key, instance] of snapshot) {
        liveKeys.add(key);
        if (typeof instance.meshKey !== 'string') continue;
        keyMesh.set(key, instance.meshKey);
        meshUse.set(instance.meshKey, (meshUse.get(instance.meshKey) ?? 0) + 1);
      }
      return;
    }
    for (const key of removed) {
      liveKeys.delete(key);
      const meshKey = keyMesh.get(key);
      if (meshKey === undefined) continue;
      keyMesh.delete(key);
      const count = (meshUse.get(meshKey) ?? 0) - 1;
      if (count > 0) meshUse.set(meshKey, count);
      else meshUse.delete(meshKey);
    }
    for (const instance of added) {
      if (!instance || !Number.isInteger(instance.key)) continue;
      liveKeys.add(instance.key);
      if (typeof instance.meshKey === 'string') {
        keyMesh.set(instance.key, instance.meshKey);
        meshUse.set(instance.meshKey, (meshUse.get(instance.meshKey) ?? 0) + 1);
      }
    }
  }

  function isObsolete(meshKey) {
    return (meshUse.get(meshKey) ?? 0) === 0;
  }

  function canRemesh() {
    if (typeof options.remesh === 'function') return true;
    return Boolean(options.loader && typeof options.loader.load === 'function' && typeof options.meshSource === 'function');
  }

  function requestMesh(item) {
    if (typeof options.remesh === 'function') return options.remesh(item);
    return Promise.resolve(options.meshSource(item)).then((asset) => options.loader.load(asset));
  }

  function releaseToken(token) {
    if (token === undefined || token === null || released.has(token)) return;
    released.add(token);
    options.release?.(token);
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

/**
 * Trace-mode stepping: the trace program is repacked in Rust (MAMA op 4) from the scene's static
 * sources and the delta-updated instance list, then installed through the engine's public trace
 * API. `updateTraceInstances` keeps the installed program and its bend associations (the engine
 * validates the count and either refuses the whole update or invalidates the world volume);
 * `setTraceScene` replaces the program when the instance topology changed; an empty list clears
 * it. A refused pack or install leaves the previously installed program in place.
 */
function createTraceStepper(options, engine) {
  const spec = options.trace;
  if (spec == null || typeof spec !== 'object' || spec.input == null || typeof spec.input !== 'object') {
    throw new SdfmError('sdfm.scene_format', 'trace stepping needs options.trace.input (the loaded scene trace input)');
  }
  const input = spec.input;
  if (!Array.isArray(input.instances)) {
    throw new SdfmError('sdfm.scene_format', 'trace stepping needs trace.input.instances');
  }
  if (typeof spec.packTrace !== 'function') {
    throw new SdfmError('sdfm.scene_resolve', 'trace stepping needs trace.packTrace (MAMA packTrace) and trace.procgen');
  }
  const sources = Array.isArray(input.sources) ? input.sources : [];
  const list = input.instances.map(traceInstanceInput);
  const byKey = new Map(list.map((instance) => [instance.key, instance]));
  const instancesOf = typeof spec.instancesOf === 'function' ? spec.instancesOf : traceInstancesOf;
  let installed = true;
  let installedTopology = topologyOf(list);

  async function apply(delta, issues) {
    const removedKeys = new Set(Array.isArray(delta.removed) ? delta.removed : []);
    let removed = 0;
    if (removedKeys.size > 0) {
      for (let i = list.length - 1; i >= 0; i -= 1) {
        if (!removedKeys.has(list[i].key)) continue;
        byKey.delete(list[i].key);
        list.splice(i, 1);
        removed += 1;
      }
    }
    let moved = 0;
    for (const item of delta.moved) {
      const instance = byKey.get(item?.key);
      if (instance === undefined) continue;
      instance.t = [...item.t];
      instance.r = [...item.r];
      instance.s = [...item.s];
      moved += 1;
    }
    let added = 0;
    for (const instance of delta.added) {
      if (!instance || !Number.isInteger(instance.key) || byKey.has(instance.key)) continue;
      const entry = traceInstanceInput(instance);
      list.push(entry);
      byKey.set(entry.key, entry);
      added += 1;
    }
    // Reconcile the surviving trace instances from the complete output snapshot by stable key:
    // a metadata-only step (empty moved/added/removed) can change one instance's source or bend
    // while several instances share a source. The trace record's identity is its key, so only
    // that key is retargeted and the program is repacked.
    let retargeted = false;
    const snapshot = outputInstances(delta);
    if (snapshot !== null) {
      for (const [key, instance] of snapshot) {
        const target = byKey.get(key);
        if (target === undefined) continue;
        let touched = false;
        if (target.source !== instance.source) {
          target.source = instance.source;
          touched = true;
        }
        if (!sameDeform(target.deform, instance.deform)) {
          if (instance.deform === undefined) delete target.deform;
          else target.deform = instance.deform;
          touched = true;
        }
        if (touched) retargeted = true;
      }
    }
    if (moved + added + removed > 0 || retargeted) await commit();
    return { moved, remeshed: 0, added, removed, ok: true, issues, delta };
  }

  async function commit() {
    if (list.length === 0) {
      if (installed) {
        engine.clearTraceScene();
        installed = false;
        installedTopology = '';
      }
      return;
    }
    let program;
    try {
      program = spec.packTrace(spec.procgen, { sources, instances: list.map(traceInstanceInput) });
    } catch (e) {
      throw new SdfmError('sdfm.scene_resolve', `cannot pack the trace program: ${messageOf(e)}`);
    }
    const nextTopology = topologyOf(list);
    const instances = instancesOf(program);
    const topologyChanged = !installed || nextTopology !== installedTopology;
    if (typeof spec.commit === 'function') {
      await spec.commit(program, { topologyChanged, instances });
    } else if (topologyChanged || typeof engine.updateTraceInstances !== 'function') {
      engine.setTraceScene(program);
    } else {
      engine.updateTraceInstances(instances);
    }
    installedTopology = nextTopology;
    installed = true;
  }

  return { apply };
}

function traceInstanceInput(instance) {
  return {
    source: instance.source,
    t: [...instance.t],
    r: [...instance.r],
    s: [...instance.s],
    key: instance.key,
    ...(instance.deform !== undefined ? { deform: instance.deform } : {}),
  };
}

/** Instance topology that `updateTraceInstances` may reuse: source, key and bend per record. */
function topologyOf(list) {
  return JSON.stringify(list.map((instance) => [instance.source, instance.key, instance.deform ?? null]));
}

/**
 * The 32-word instance records of one packed TraceProgram v1, copied word for word (source,
 * material, key and axes words keep their bits) for the public `updateTraceInstances`.
 */
function traceInstancesOf(program) {
  const bytes = program instanceof Uint8Array ? program : new Uint8Array(program);
  if (bytes.byteLength % 4 !== 0) {
    throw new SdfmError('sdfm.scene_resolve', 'the packed trace program is not a whole number of words');
  }
  const raw = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
  const words = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  if (words.length < TRACE_HEADER_WORDS) {
    throw new SdfmError('sdfm.scene_resolve', 'the packed trace program is truncated');
  }
  const start = TRACE_HEADER_WORDS + words[2] * TRACE_SOURCE_WORDS + words[4] + words[5]
    + words[6] * TRACE_MATERIAL_WORDS;
  const count = words[3];
  if (start + count * TRACE_INSTANCE_WORDS > words.length) {
    throw new SdfmError('sdfm.scene_resolve', 'the packed trace program ends before its placed objects');
  }
  return new Float32Array(words.slice(start, start + count * TRACE_INSTANCE_WORDS).buffer);
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

/**
 * The complete instance list of a delta's output snapshot, keyed by stable instance key. The
 * corrected MAMA delta includes the output on the first step and whenever a surviving instance
 * changes its source, mesh key or deform; a transform-only delta carries none.
 * @returns {Map<number, object> | null}
 */
function outputInstances(delta) {
  const output = delta?.output;
  if (!output || output.ok === false || !Array.isArray(output.instances)) return null;
  const map = new Map();
  for (const instance of output.instances) {
    if (!instance || !Number.isInteger(instance.key)) continue;
    if (!map.has(instance.key)) map.set(instance.key, instance);
  }
  return map;
}

function sameDeform(a, b) {
  return stable(a) === stable(b);
}

function stable(value) {
  if (value == null || typeof value !== 'object') return JSON.stringify(value ?? null);
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(',')}}`;
}

function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}

function wrap(e) {
  if (e && typeof e === 'object' && typeof e.code === 'string') return e;
  return new SdfmError('sdfm.scene_resolve', e instanceof Error ? e.message : String(e));
}
