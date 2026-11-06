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
const lastProgram = (engine) => JSON.parse(new TextDecoder().decode(engine.calls.setTraceScene.at(-1)));

function fakeEngine() {
  const calls = { addNode: [], removeNode: [], setTraceScene: [], clearTraceScene: [] };
  let trace = null;
  let next = 1;
  return {
    calls,
    get trace() { return trace; },
    addNode(parent, t = [0, 0, 0]) { const id = next++; calls.addNode.push({ id, parent, t: [...t] }); return id; },
    removeNode(id) { calls.removeNode.push(id); },
    setTraceScene(bytes) { calls.setTraceScene.push(bytes); trace = JSON.parse(new TextDecoder().decode(bytes)); },
    clearTraceScene() { calls.clearTraceScene.push(true); trace = null; },
  };
}

function fakeMama() {
  const calls = { build: 0, pack: [] };
  const tree = { root: '$root', nodes: { $root: { kind: 'group', children: [], transform: { position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] } } } };
  return {
    calls,
    mama: {
      parseProceduralScene: (text) => JSON.parse(text),
      buildSceneTraceInput(parsed, lookup) {
        calls.build++;
        if (parsed.sources.some((source) => source.kind === 'asset' && lookup(source.file) === null)) throw new Error('asset not resolved');
        return { sources: parsed.sources.map((_source, index) => ({ key: `source:${index}`, tree })),
          instances: parsed.instances.map((instance) => ({ ...instance })) };
      },
      packTrace(procgen, input) { calls.pack.push({ procgen, input }); return new TextEncoder().encode(JSON.stringify(input)); },
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
