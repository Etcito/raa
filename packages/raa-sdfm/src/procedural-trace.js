import { SdfmError } from './errors.js';
import { createProceduralRuntime } from './procedural-runtime.js';

// One engine has a single trace slot: each handle keeps its MAMA TraceInput, a change merges the
// visible inputs, and MAMA op 4 packs them in Rust. Installing a program selects the engine's
// world-volume trace scene (GPU-stamped R8Snorm volume with five mips, full-resolution trace and
// exact-program refinement); no geometry is composed here.
const sessions = new WeakMap();

export async function loadProceduralTrace(engine, loader, scene, mama, options) {
  if (scene.variant !== 'sdf-refs') {
    throw new SdfmError('sdfm.scene_format', 'Trace mode loads SDF reference scenes only; a mesh reference scene has no SDF sources to trace');
  }
  if (typeof engine.setTraceScene !== 'function' || typeof engine.clearTraceScene !== 'function') {
    throw new SdfmError('sdfm.scene_resolve', 'Trace mode needs the engine trace scene (setTraceScene and clearTraceScene); this engine has none');
  }
  const procgen = options.procgen;
  if (typeof mama.buildSceneTraceInput !== 'function' || typeof mama.packTrace !== 'function' || procgen == null) {
    throw new SdfmError('sdfm.scene_resolve', 'Trace mode needs the MAMA trace functions and options.procgen (the loaded MAMA evaluator)');
  }
  let input = await traceInput(scene, mama, options);
  if (options.placement !== undefined) input = placed(input, options.placement);
  const session = sessions.get(engine) ?? new Set();
  sessions.set(engine, session);
  const entry = { input, visible: true, procgen, packTrace: mama.packTrace };
  const root = engine.addNode(options.parent ?? null, options.placement ?? [0, 0, 0]);
  session.add(entry);
  try {
    commit(engine, session);
  } catch (e) {
    session.delete(entry);
    engine.removeNode(root);
    throw e;
  }
  let disposed = false;
  const check = () => {
    if (disposed) throw new SdfmError('sdfm.disposed', 'procedural scene has been disposed');
  };
  return {
    root,
    instanceCount: scene.instances.length,
    sourceCount: scene.sources.length,
    setVisible(visible) {
      check();
      if (entry.visible === visible) return;
      entry.visible = visible;
      try {
        commit(engine, session);
      } catch (e) {
        entry.visible = !visible;
        throw e;
      }
    },
    stats() {
      check();
      return { instances: scene.instances.length, meshes: 0, pending: 0 };
    },
    runtime(procgen) {
      check();
      // The adapter repacks the delta-updated instance list through MAMA op 4. The merged trace
      // slot is rebuilt from this handle's entry, so the newest packed input is stored on it
      // before the shared commit merges every visible handle.
      let packed = null;
      return createProceduralRuntime({
        engine, scene, procgen, mama,
        trace: {
          input: entry.input,
          procgen,
          packTrace(p, value) {
            packed = value;
            return mama.packTrace(p, value);
          },
          commit() {
            if (packed !== null) entry.input = packed;
            commit(engine, session);
          },
        },
      });
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      session.delete(entry);
      engine.removeNode(root);
      commit(engine, session);
    },
  };
}
// Pack the visible inputs into the one trace slot; no visible handle clears it. Every accepted
// install replaces the program, so the engine invalidates and rebuilds the world volume before it
// draws the changed scene.
function commit(engine, session) {
  const entries = [...session].filter((entry) => entry.visible);
  if (entries.length === 0) return engine.clearTraceScene();
  const { procgen, packTrace } = entries[0];
  let program;
  try {
    program = packTrace(procgen, merge(entries.map((entry) => entry.input)));
  } catch (e) {
    throw new SdfmError('sdfm.scene_resolve', `cannot pack the trace program: ${messageOf(e)}`);
  }
  engine.setTraceScene(program);
}
function merge(inputs) {
  const sources = [];
  const instances = [];
  for (const input of inputs) {
    const base = sources.length;
    sources.push(...input.sources);
    for (const instance of input.instances) instances.push({ ...instance, source: base + instance.source });
  }
  return { sources, instances };
}
function placed(input, placement) {
  const [x, y, z] = placement;
  return { sources: input.sources, instances: input.instances.map((instance) => ({ ...instance, t: [instance.t[0] + x, instance.t[1] + y, instance.t[2] + z] })) };
}
async function traceInput(scene, mama, options) {
  const files = [];
  for (const source of scene.sources) {
    if (source.kind === 'asset' && !files.includes(source.file)) files.push(source.file);
  }
  const assets = new Map();
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
    assets.set(file, value);
  }));
  try {
    return mama.buildSceneTraceInput(scene, (file) => assets.get(file) ?? null);
  } catch (e) {
    throw new SdfmError('sdfm.scene_resolve', `cannot build the trace inputs: ${messageOf(e)}`);
  }
}
const messageOf = (e) => (e instanceof Error ? e.message : String(e));
