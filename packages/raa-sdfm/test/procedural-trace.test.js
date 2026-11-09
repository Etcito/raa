import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadProceduralScene } from '../src/procedural.js';

const BOUNDS = { min: [-1, -1, -1], max: [1, 1, 1] };
const pose = (t = [0, 0, 0]) => ({ t, r: [0, 0, 0, 1], s: [1, 1, 1] });
const scene = (overrides = {}) => ({
  format: 'sdfm.procedural-scene', version: 1, variant: 'sdf-refs', name: 'trace', game: 'test',
  generator: { document: 'trace.sdfm.json', node: 'host', seed: 1, params: {} },
  sources: [{ kind: 'asset', file: 'assets/test/a.asset.json', ref: { game: 'test', id: 'a.test' }, bounds: BOUNDS }],
  instances: [{ key: 1, source: 0, ...pose() }, { key: 2, source: 0, ...pose([1, 0, 0]) }],
  groups: [], colliders: [], visual: [], runtime: null, ...overrides,
});
const programs = new WeakMap();
const lastProgram = (engine) => programs.get(engine.calls.setTraceScene.at(-1));
const light = (id = 'lamp', t = [1, 2, 3]) => ({
  id, node: 'light', key: 99, ...pose(t), kind: 'point', color: [1, 1, 1], intensity: 2, range: 6, shadows: false,
});
const delta = (extra = {}) => ({ ok: true, issues: [], added: [], moved: [], removed: [], remesh: [], ...extra });
const snapshot = (extra = {}) => ({ ok: true, instances: [], ...extra });
const liveScene = (extra = {}) => scene({ runtime: { input: {} }, ...extra });

function fakeEngine() {
  const calls = { addNode: [], removeNode: [], setTraceScene: [], clearTraceScene: [], traceArgs: [], addLight: [], removeLight: [] };
  const lights = new Map();
  let trace = null;
  let volume = 0;
  let next = 1;
  return {
    calls,
    lights,
    addLight(desc) { const id = next++; lights.set(id, structuredClone(desc)); calls.addLight.push(id); return id; },
    updateLight(id, changes) { assert.ok(lights.has(id)); Object.assign(lights.get(id), structuredClone(changes)); },
    removeLight(id) { assert.ok(lights.delete(id)); calls.removeLight.push(id); },
    get trace() { return trace; },
    get volume() { return volume; },
    addNode(parent, t = [0, 0, 0]) { const id = next++; calls.addNode.push({ id, parent, t: [...t] }); return id; },
    removeNode(id) { calls.removeNode.push(id); },
    setTraceScene(bytes, ...rest) {
      calls.setTraceScene.push(bytes);
      calls.traceArgs.push(rest.length);
      trace = structuredClone(programs.get(bytes));
      volume += 1;
    },
    clearTraceScene() { calls.clearTraceScene.push(true); trace = null; },
  };
}

function fakeMama(deltas = []) {
  const calls = { build: 0, pack: [], create: 0, dispose: 0 };
  const tree = { root: '$root', nodes: { $root: { kind: 'group', children: [], transform: { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] } } } };
  return {
    calls,
    mama: {
      parseProceduralScene: (text) => JSON.parse(text),
      Runtime: { create() {
        calls.create++;
        return { set() {}, step() { return deltas.shift(); }, dispose() { calls.dispose++; } };
      } },
      buildSceneTraceInput(parsed, lookup) {
        calls.build++;
        if (parsed.sources.some((source) => source.kind === 'asset' && lookup(source.file) === null)) throw new Error('asset not resolved');
        return { sources: parsed.sources.map((_source, index) => ({ key: `source:${index}`, tree })),
          instances: parsed.instances.map((instance) => ({ ...instance })) };
      },
      packTrace(procgen, input) {
        calls.pack.push({ procgen, input });
        const words = new Uint32Array(8 + input.sources.length * 12 + input.instances.length * 32);
        words.set([0x50544453, 1, input.sources.length, input.instances.length]);
        const bytes = new Uint8Array(words.buffer);
        programs.set(bytes, structuredClone(input));
        return bytes;
      },
    },
  };
}

const fakeLoader = () => ({
  calls: { load: 0, spawn: 0 },
  async load() { this.calls.load++; throw new Error('loader.load must not run in trace mode'); },
  spawn() { this.calls.spawn++; throw new Error('loader.spawn must not run in trace mode'); },
});
const options = (fake, extra = {}) => ({
  mode: 'trace', procgen: { call() {} }, mama: async () => fake.mama, resolveAsset: async (file) => ({ file }), ...extra,
});
const failureCode = (promise) => promise.then(() => 'no error', (error) => error.code);
const sourcesOf = (input) => [input.sources.length, ...input.instances.map((instance) => instance.source)];

test('trace mode packs through MAMA op 4 without meshing and keeps the handle lifecycle', async () => {
  const [fake, engine, loader] = [fakeMama(), fakeEngine(), fakeLoader()];
  const value = options(fake);
  const handle = await loadProceduralScene(engine, loader, scene(), value);
  assert.deepEqual(sourcesOf(lastProgram(engine)), [1, 0, 0]);
  assert.equal(fake.calls.pack[0].procgen, value.procgen);
  assert.deepEqual(handle.stats(), { instances: 2, meshes: 0, pending: 0 });
  assert.deepEqual([loader.calls.load, 'addMesh' in engine], [0, false]);
  handle.setVisible(false);
  assert.equal(engine.trace, null);
  handle.setVisible(true);
  assert.equal(lastProgram(engine).instances.length, 2);
  const root = handle.root;
  handle.dispose();
  assert.deepEqual(engine.calls.removeNode, [root]);
  assert.equal(engine.trace, null);
  assert.throws(() => handle.stats(), (error) => error.code === 'sdfm.disposed');
});

test('trace mode installs the world-volume trace scene and invalidates it on every accepted change', async () => {
  const [fake, engine, loader] = [fakeMama(), fakeEngine(), fakeLoader()];
  const value = options(fake);
  const handle = await loadProceduralScene(engine, loader, scene(), value);
  assert.equal(engine.calls.setTraceScene.length, 1);
  assert.deepEqual(engine.calls.traceArgs, [0]);
  assert.ok(engine.calls.setTraceScene[0] instanceof Uint8Array);
  assert.equal(engine.volume, 1);
  assert.deepEqual(['addMesh', 'addMeshInstance', 'addSurfaceMaterial', 'addMaterialFunction'].filter((name) => name in engine), []);
  handle.setVisible(false);
  assert.equal(engine.trace, null);
  handle.setVisible(true);
  assert.equal(engine.calls.setTraceScene.length, 2);
  assert.equal(engine.volume, 2);
  const b = await loadProceduralScene(engine, loader, scene({ name: 'b', instances: [{ key: 9, source: 0, ...pose() }] }), value);
  assert.equal(engine.volume, 3);
  b.dispose();
  assert.equal(engine.calls.setTraceScene.length, 4);
  assert.equal(engine.volume, 4);
  handle.dispose();
  assert.equal(engine.trace, null);
  assert.equal(engine.volume, 4);
});

test('trace handles share one slot: placement, visibility and disposal repack the merged input', async () => {
  const [fake, engine, loader] = [fakeMama(), fakeEngine(), fakeLoader()];
  const base = options(fake);
  const a = await loadProceduralScene(engine, loader, scene(), { ...base, placement: [1, 0, 0] });
  const b = await loadProceduralScene(engine, loader, scene({ name: 'b', instances: [{ key: 9, source: 0, ...pose([4, 0, 0]) }] }),
    { ...base, placement: [-2, 0, 0] });
  let input = lastProgram(engine);
  assert.deepEqual(sourcesOf(input), [2, 0, 0, 1]);
  assert.deepEqual([input.instances[0].t, input.instances[2].t], [[1, 0, 0], [2, 0, 0]]);
  b.setVisible(false);
  assert.deepEqual(sourcesOf(lastProgram(engine)), [1, 0, 0]);
  b.setVisible(true);
  a.dispose();
  input = lastProgram(engine);
  assert.deepEqual(sourcesOf(input), [1, 0]);
  assert.deepEqual(input.instances[0].t, [2, 0, 0]);
  b.dispose();
  assert.equal(engine.trace, null);
  assert.ok(engine.calls.clearTraceScene.length >= 1);
});

test('trace mode refuses invalid scenes and a rejected program keeps the previous handle', async () => {
  const [fake, engine, loader] = [fakeMama(), fakeEngine(), fakeLoader()];
  const value = options(fake);
  const meshRefs = scene({ variant: 'mesh-refs', sources: [{ kind: 'mesh', mesh: 'procedural/trace/meshes/x.mesh', meshKey: 'x', bounds: BOUNDS }],
    instances: [{ key: 1, source: 0, ...pose() }] });
  assert.equal(await failureCode(loadProceduralScene(engine, loader, meshRefs, value)), 'sdfm.scene_format');
  const noTrace = fakeEngine();
  delete noTrace.setTraceScene;
  assert.equal(await failureCode(loadProceduralScene(noTrace, loader, scene(), value)), 'sdfm.scene_resolve');
  assert.deepEqual([noTrace.calls.setTraceScene.length, noTrace.calls.addNode.length], [0, 0]);
  assert.equal(await failureCode(loadProceduralScene(engine, loader, scene(), {
    ...value, resolveAsset: async () => { throw new Error('offline'); } })), 'sdfm.scene_resolve');
  assert.equal(await failureCode(loadProceduralScene(engine, loader, scene(), { ...value, procgen: undefined })), 'sdfm.scene_resolve');
  assert.equal(await failureCode(loadProceduralScene(engine, loader, scene(), {
    ...value, mama: async () => ({ parseProceduralScene: fake.mama.parseProceduralScene }) })), 'sdfm.scene_resolve');
  assert.deepEqual([engine.calls.setTraceScene.length, engine.calls.addNode.length], [0, 0]);
  const a = await loadProceduralScene(engine, loader, scene(), value);
  const previous = engine.trace;
  const error = Object.assign(new Error('trace.format: refused'), { code: 'trace.format' });
  const set = engine.setTraceScene;
  engine.setTraceScene = () => { throw error; };
  await assert.rejects(loadProceduralScene(engine, loader, scene({ name: 'b' }), value), (thrown) => thrown === error);
  engine.setTraceScene = set;
  assert.equal(engine.trace, previous);
  assert.equal(engine.calls.removeNode.length, 1);
  a.dispose();
  assert.equal(engine.trace, null);
});

test('static trace lights retain per-handle identity, placement and visibility without runtime input', async () => {
  const [fake, engine, loader] = [fakeMama(), fakeEngine(), fakeLoader()];
  const source = scene({ lights: [light()] });
  const original = structuredClone(source);
  const placement = [10, -20, 30];
  const a = await loadProceduralScene(engine, loader, source, options(fake, { placement }));
  const b = await loadProceduralScene(engine, loader, source, options(fake));
  const [aId, bId] = engine.lights.keys();
  assert.deepEqual(engine.lights.get(aId).position, [11, -18, 33]);
  assert.deepEqual(engine.lights.get(bId).position, [1, 2, 3]);
  assert.deepEqual(source, original);
  assert.throws(() => a.runtime({}), (error) => error.code === 'sdfm.scene_format');
  a.setVisible(false);
  assert.deepEqual([...engine.lights.values()].map((desc) => desc.intensity), [0, 2]);
  a.setVisible(true);
  assert.deepEqual([...engine.lights.keys()], [aId, bId]);
  a.dispose();
  a.dispose();
  assert.deepEqual([...engine.lights.keys()], [bId]);
  for (const lights of [[], undefined, [light()]]) {
    const reloaded = await loadProceduralScene(engine, loader, scene({ lights, sources: [], instances: [] }), options(fake));
    assert.equal(engine.lights.size, lights?.length ? 2 : 1);
    reloaded.dispose();
    reloaded.dispose();
    assert.deepEqual([...engine.lights.keys()], [bId]);
  }
  b.dispose();
  assert.equal(engine.lights.size, 0);
});

test('runtime access shares the trace owner and translates every complete light snapshot once', async () => {
  const first = light();
  const moved = light('lamp', [4, 5, 6]);
  const changed = { ...moved, intensity: 7 };
  const outputs = [snapshot({ lights: [light('lamp', [2, 3, 4])] }), snapshot({ lights: [moved] }), snapshot({ lights: [changed] })];
  const original = structuredClone(outputs);
  const fake = fakeMama([
    ...outputs.map((output) => delta({ output })), delta(),
    delta({ ok: false, output: snapshot() }), delta({ output: snapshot() }),
    delta({ output: snapshot({ lights: [first] }) }), delta({ output: snapshot({ lights: [] }) }),
  ]);
  const [engine, loader] = [fakeEngine(), fakeLoader()];
  const placement = [10, -20, 30];
  const a = await loadProceduralScene(engine, loader, liveScene({ lights: [first] }), options(fake, { placement }));
  const b = await loadProceduralScene(engine, loader, scene({ lights: [first] }), options(fake));
  const [aId, bId] = engine.lights.keys();
  const runtime = a.runtime({});
  a.runtime({});
  assert.equal(engine.calls.addLight.length, 2, 'runtime access must not reconstruct the registry');
  placement[0] = 1000;
  const baseline = await runtime.step(0);
  assert.deepEqual([baseline.moved, baseline.added, baseline.removed], [0, 0, 0]);
  assert.deepEqual(engine.lights.get(aId).position, [12, -17, 34]);
  await runtime.step(1);
  assert.deepEqual(engine.lights.get(aId).position, [14, -15, 36]);
  a.setVisible(false);
  await runtime.step(2);
  assert.equal(engine.lights.get(aId).intensity, 0);
  a.setVisible(true);
  assert.equal(engine.lights.get(aId).intensity, 7);
  await runtime.step(3);
  assert.equal((await runtime.step(4)).ok, false);
  assert.equal(engine.lights.get(aId).intensity, 7);
  const publications = engine.calls.setTraceScene.length;
  await runtime.step(5);
  assert.deepEqual([...engine.lights.keys()], [bId]);
  await runtime.step(6);
  assert.deepEqual([...engine.lights.values()][1].position, [11, -18, 33]);
  await runtime.step(7);
  assert.deepEqual([...engine.lights.keys()], [bId]);
  assert.equal(engine.calls.setTraceScene.length, publications, 'light removal is independent of geometry');
  assert.deepEqual(outputs, original);
  runtime.dispose();
  a.dispose();
  assert.equal(fake.calls.dispose, 2);
  assert.deepEqual([...engine.lights.keys()], [bId]);
  b.dispose();
});

test('light creation and refused trace publications roll back only the attempted handle changes', async () => {
  const refusal = Object.assign(new Error('trace.format: refused'), { code: 'trace.format' });
  const fake = fakeMama([delta(), delta({
    moved: [{ key: 1, ...pose([9, 0, 0]) }], output: snapshot({ lights: [light('replacement')] }),
  })]);
  const [engine, loader] = [fakeEngine(), fakeLoader()];
  const value = options(fake);
  const a = await loadProceduralScene(engine, loader, liveScene({ lights: [light()] }), value);
  const b = await loadProceduralScene(engine, loader, scene({ lights: [light()] }), value);
  const before = structuredClone([...engine.lights]);
  const previous = engine.trace;
  const add = engine.addLight;
  let additions = 0;
  engine.addLight = (desc) => { if (++additions === 2) throw refusal; return add(desc); };
  await assert.rejects(loadProceduralScene(engine, loader, scene({ lights: [light('x'), light('y')] }), value),
    (error) => error === refusal);
  engine.addLight = add;
  assert.deepEqual([...engine.lights], before);
  assert.equal(engine.trace, previous);
  const set = engine.setTraceScene;
  engine.setTraceScene = () => { throw refusal; };
  await assert.rejects(loadProceduralScene(engine, loader, scene({ lights: [light('x')] }), value),
    (error) => error === refusal);
  assert.deepEqual([...engine.lights], before);
  const runtime = a.runtime({});
  await runtime.step(0);
  await assert.rejects(runtime.step(1), (error) => error === refusal);
  assert.equal(engine.trace, previous);
  assert.deepEqual([...engine.lights], before);
  assert.throws(() => a.setVisible(false), (error) => error === refusal);
  assert.deepEqual([...engine.lights], before);
  engine.setTraceScene = set;
  b.setVisible(false);
  assert.deepEqual(lastProgram(engine).instances[0].t, [0, 0, 0], 'failed runtime input cannot leak into a later merge');
  b.setVisible(true);
  a.dispose();
  assert.deepEqual([...engine.lights], before.slice(1));
  b.dispose();
});

test('failure activating staged lights restores the prior merged trace publication', async () => {
  const [fake, engine, loader] = [fakeMama(), fakeEngine(), fakeLoader()];
  const value = options(fake);
  const a = await loadProceduralScene(engine, loader, scene({ lights: [light()] }), value);
  const before = structuredClone([...engine.lights]);
  const previous = engine.trace;
  const update = engine.updateLight;
  const refusal = Object.assign(new Error('light activation refused'), { code: 'light.invalid_desc' });
  engine.updateLight = (id, changes) => {
    if (changes.intensity === 2) throw refusal;
    update(id, changes);
  };
  await assert.rejects(loadProceduralScene(engine, loader, scene({ lights: [light('new')] }), value),
    (error) => error === refusal);
  engine.updateLight = update;
  assert.deepEqual(engine.trace, previous);
  assert.deepEqual([...engine.lights], before);
  a.dispose();
});

test('disposal aborts staged runtime lights and queued runtime work cannot resurrect them', async () => {
  const fake = fakeMama([delta(), delta({
    moved: [{ key: 1, ...pose([9, 0, 0]) }], output: snapshot({ lights: [light('staged')] }),
  }), delta({ output: snapshot({ lights: [light('queued')] }) })]);
  const [engine, loader] = [fakeEngine(), fakeLoader()];
  const a = await loadProceduralScene(engine, loader, liveScene({ lights: [light()] }), options(fake));
  const b = await loadProceduralScene(engine, loader, scene({ lights: [light()] }), options(fake));
  const bId = [...engine.lights.keys()][1];
  const runtime = a.runtime({});
  await runtime.step(0);
  const set = engine.setTraceScene;
  engine.setTraceScene = (bytes) => { set(bytes); engine.setTraceScene = set; a.dispose(); };
  const pending = runtime.step(1);
  const queued = runtime.step(2);
  await assert.rejects(queued, (error) => error.code === 'sdfm.disposed');
  assert.equal((await pending).disposed, true);
  assert.deepEqual([...engine.lights.keys()], [bId]);
  assert.equal(fake.calls.dispose, 1);
  assert.equal(lastProgram(engine).instances.length, 2);
  a.dispose();
  b.dispose();
  assert.equal(engine.lights.size, 0);
});
