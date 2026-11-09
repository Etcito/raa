import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createFakeEngine } from './fake-engine.js';
import { createProceduralRuntime } from '../src/procedural-runtime.js';
import { createMamaLightOwner } from '../src/mama-light-owner.js';
import { loadProceduralScene } from '../src/procedural.js';
import { SdfmError } from '../src/errors.js';

function runtimeEngine() {
  const engine = createFakeEngine();
  engine.calls.setTransform = [];
  engine.calls.removeNode = [];
  engine.setTransform = (node, t, r, s) => {
    engine.calls.setTransform.push({ node, t: [...t], r: [...r], s: [...s] });
    Object.assign(engine.nodes.get(node), { t: [...t], r: [...r], s: [...s] });
  };
  engine.removeNode = (node) => {
    engine.calls.removeNode.push(node);
    const stack = [node];
    while (stack.length > 0) {
      const id = stack.pop();
      engine.nodes.delete(id);
      for (const [child, value] of engine.nodes) if (value.parent === id) stack.push(child);
    }
  };
  return engine;
}

function runtimeScene() {
  return {
    format: 'sdfm.procedural-scene',
    version: 1,
    variant: 'sdf-refs',
    name: 'demo',
    game: 'fixture',
    generator: {},
    sources: [{ kind: 'asset', file: 'common.post.test', ref: 'common.post.test', bounds: { min: [0, 0, 0], max: [1, 1, 1] } }],
    instances: [],
    groups: [],
    colliders: [],
    visual: [],
    runtime: { input: { version: 1, graph: {}, paramValues: {}, timeTicks: 0, sources: {} } },
  };
}

function outInstance(key, extra = {}) {
  return { key, source: 0, t: [0, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1], group: -1, meshKey: 'm', ...extra };
}

function delta(extra = {}) {
  return { ok: true, issues: [], moved: [], added: [], removed: [], remesh: [], colliderFrames: [], ...extra };
}

function fakeRuntime(deltas) {
  return {
    lastTick: null,
    disposed: false,
    sets: [],
    step(tick) {
      this.lastTick = tick;
      const next = deltas.shift();
      if (next === undefined) throw new Error(`no delta for tick ${tick}`);
      return { ...(typeof next === 'function' ? next(tick) : next), tick };
    },
    set(values) {
      this.sets.push(values);
    },
    dispose() {
      this.disposed = true;
    },
  };
}

const BEND = { kind: 'bend', params: { axis: 'y', angle: 20 }, angle: 0.349, length: 1, along: 'x', toward: 'y' };

test('step applies moved, added and removed instances under the scene root, keeping stable keys', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root, [0, 0, 0]);
  const nodeB = engine.addNode(root, [0, 0, 0]);
  const runtime = fakeRuntime([delta({
    moved: [{ key: 7, t: [1, 2, 3], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [2, 2, 2] }],
    added: [outInstance(11, { t: [4, 5, 6] })],
    removed: [9],
  }), delta({
    moved: [{ key: 11, t: [0, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }],
    removed: [7],
  })]);
  const attached = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: new Map([[7, { node: nodeA, source: 0 }], [9, { node: nodeB, source: 0 }]]),
    attach(instance, token) {
      const node = engine.addNode(root, instance.t, instance.r, instance.s);
      attached.push({ instance, token, node });
      return { node };
    },
  });

  const first = await adapter.step(5);
  assert.equal(runtime.lastTick, 5);
  assert.deepStrictEqual(
    [first.moved, first.added, first.removed, first.remeshed],
    [1, 1, 1, 0],
  );
  assert.deepStrictEqual(engine.calls.setTransform, [
    { node: nodeA, t: [1, 2, 3], r: [0, 0, 0, 1], s: [2, 2, 2] },
  ]);
  assert.deepStrictEqual(engine.calls.removeNode, [nodeB]);
  assert.equal(attached.length, 1);
  const addedNode = attached[0].node;
  assert.equal(engine.nodes.get(addedNode).parent, root);
  assert.deepStrictEqual(engine.nodes.get(addedNode).t, [4, 5, 6]);
  assert.equal(engine.nodes.has(nodeB), false);

  const second = await adapter.step(6);
  assert.deepStrictEqual([second.moved, second.added, second.removed], [1, 0, 1]);
  assert.deepStrictEqual(engine.calls.setTransform.at(-1), { node: addedNode, t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1] });
  assert.deepStrictEqual(engine.calls.removeNode, [nodeB, nodeA]);

  adapter.dispose();
  assert.deepStrictEqual(engine.calls.removeNode, [nodeB, nodeA, addedNode]);
  assert.equal(runtime.disposed, true);
  adapter.dispose();
  assert.strictEqual(engine.calls.removeNode.length, 3);
});

test('the first step neither duplicates the loaded instances nor remeshes the loaded meshes', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  const nodeB = engine.addNode(root);
  const initial = [{ key: 1, node: nodeA, source: 0 }, { key: 2, node: nodeB, source: 0 }];
  const runtime = fakeRuntime([
    delta({
      added: [outInstance(1), outInstance(2)],
      remesh: [{ meshKey: 'm1', source: 0 }, { meshKey: 'm2', source: 0 }],
    }),
    delta({
      remesh: [{ meshKey: 'm3', source: 0 }],
      added: [outInstance(3, { meshKey: 'm3', t: [9, 0, 0] })],
    }),
  ]);
  const remeshed = [];
  const attached = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: initial,
    remesh(item) {
      remeshed.push(item.meshKey);
      if (item.meshKey !== 'm3') throw new Error('the loaded meshes must not be remeshed');
      return { mesh: item.meshKey };
    },
    attach(instance, token) {
      attached.push({ key: instance.key, token });
      return { node: engine.addNode(root, instance.t, instance.r, instance.s) };
    },
  });

  const first = await adapter.step(0);
  assert.deepStrictEqual([first.added, first.remeshed], [0, 0]);
  assert.equal(engine.calls.addNode.length, 3);
  assert.deepStrictEqual(remeshed, []);

  const second = await adapter.step(1);
  assert.deepStrictEqual(remeshed, ['m3']);
  assert.deepStrictEqual(attached, [{ key: 3, token: { mesh: 'm3' } }]);
  assert.equal(second.remeshed, 1);
  assert.equal(second.added, 1);
});

test('a new mesh key is built once, handed to added instances and old meshes stay alive', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  const nodeB = engine.addNode(root);
  const runtime = fakeRuntime([
    delta(),
    delta({
      remesh: [{ meshKey: 'k2', source: 0, deform: BEND }],
      added: [outInstance(30, { meshKey: 'k2', deform: BEND, t: [2, 0, 0] })],
      output: {
        ok: true,
        instances: [
          outInstance(7, { meshKey: 'k2', deform: BEND }),
          outInstance(8, { meshKey: 'k2', deform: BEND }),
          outInstance(30, { meshKey: 'k2', deform: BEND, t: [2, 0, 0] }),
        ],
      },
    }),
  ]);
  const released = [];
  const reattached = [];
  const attached = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: new Map([
      [7, { node: nodeA, source: 0, deform: BEND, mesh: { old: 1 } }],
      [8, { node: nodeB, source: 0, deform: BEND, mesh: { old: 2 } }],
    ]),
    remesh: async (item) => ({ mesh: item.meshKey }),
    reattach(entry, token, item) {
      reattached.push({ key: entry.key, token, meshKey: item.meshKey });
    },
    attach(instance, token) {
      attached.push({ key: instance.key, token });
      return { node: engine.addNode(root, instance.t, instance.r, instance.s) };
    },
    release(token) {
      released.push(token);
    },
  });

  await adapter.step(0);
  const result = await adapter.step(1);
  assert.equal(result.remeshed, 1);
  assert.deepStrictEqual(reattached, [
    { key: 7, token: { mesh: 'k2' }, meshKey: 'k2' },
    { key: 8, token: { mesh: 'k2' }, meshKey: 'k2' },
  ]);
  assert.deepStrictEqual(attached, [{ key: 30, token: { mesh: 'k2' } }]);
  assert.deepStrictEqual(released, []);

  adapter.dispose();
  assert.deepStrictEqual(released, [{ mesh: 'k2' }]);
});

test('a metadata-only step reattaches each surviving instance by its own stable key, never the request source', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  const nodeB = engine.addNode(root);
  const bendA = { ...BEND, params: { axis: 'y', angle: 20 }, angle: 0.349 };
  const bendB = { ...BEND, params: { axis: 'y', angle: 45 }, angle: 0.785 };
  const runtime = fakeRuntime([
    delta({
      output: {
        ok: true,
        instances: [
          outInstance(7, { meshKey: 'a', deform: bendA }),
          outInstance(8, { meshKey: 'b', deform: bendB }),
        ],
      },
    }),
    delta({
      remesh: [
        { meshKey: 'c', source: 0, deform: bendB },
        { meshKey: 'd', source: 0, deform: bendA },
      ],
      output: {
        ok: true,
        instances: [
          outInstance(7, { meshKey: 'c', deform: bendB }),
          outInstance(8, { meshKey: 'd', deform: bendA }),
        ],
      },
    }),
  ]);
  const built = [];
  const reattached = [];
  const released = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: new Map([
      [7, { node: nodeA, source: 0, deform: bendA, meshKey: 'a', mesh: { old: 7 } }],
      [8, { node: nodeB, source: 0, deform: bendB, meshKey: 'b', mesh: { old: 8 } }],
    ]),
    remesh(item) {
      built.push(item.meshKey);
      return { mesh: item.meshKey };
    },
    reattach(entry, token, item) {
      reattached.push({ key: entry.key, token, meshKey: item.meshKey });
    },
    release(token) {
      released.push(token);
    },
  });

  await adapter.step(0);
  const result = await adapter.step(1);
  assert.deepStrictEqual([result.moved, result.added, result.removed, result.remeshed], [0, 0, 0, 2]);
  assert.deepStrictEqual(built, ['c', 'd']);
  // The bends are swapped between two instances of one source: a source+deform inference would
  // give 8 the "c" mesh and 7 the "d" mesh. Their stable keys say 7 -> c and 8 -> d.
  assert.deepStrictEqual(reattached, [
    { key: 7, token: { mesh: 'c' }, meshKey: 'c' },
    { key: 8, token: { mesh: 'd' }, meshKey: 'd' },
  ]);
  assert.deepStrictEqual(released, [], 'the old meshes stay alive while the replacements are used');

  adapter.dispose();
  assert.deepStrictEqual(released, [{ mesh: 'c' }, { mesh: 'd' }]);
});

test('a metadata-only step reconciles source, mesh key and deform for later steps', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  const runtime = fakeRuntime([
    delta(),
    delta({
      output: { ok: true, instances: [outInstance(7, { source: 1, meshKey: 'k1', deform: BEND })] },
    }),
    delta({
      remesh: [{ meshKey: 'k2', source: 1, deform: BEND }],
      output: { ok: true, instances: [outInstance(7, { source: 1, meshKey: 'k2', deform: BEND })] },
    }),
  ]);
  const reattached = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: new Map([[7, { node: nodeA, source: 0, deform: BEND, meshKey: 'k1', mesh: { old: 1 } }]]),
    remesh: (item) => ({ mesh: item.meshKey }),
    reattach(entry, token, item) {
      reattached.push({ key: entry.key, meshKey: item.meshKey });
    },
  });

  await adapter.step(0);
  const clean = await adapter.step(1);
  assert.deepStrictEqual([clean.moved, clean.added, clean.removed, clean.remeshed], [0, 0, 0, 0]);
  const replaced = await adapter.step(2);
  assert.deepStrictEqual([replaced.moved, replaced.added, replaced.removed, replaced.remeshed], [0, 0, 0, 1]);
  assert.deepStrictEqual(reattached, [{ key: 7, meshKey: 'k2' }]);
  assert.deepStrictEqual(engine.calls.removeNode, []);
});

test('a surviving instance returns to an already built mesh without rebuilding it', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  const bendA = { ...BEND, angle: 0.1 };
  const bendB = { ...BEND, angle: 0.2 };
  const step = (meshKey, deform) => delta({
    remesh: [{ meshKey, source: 0, deform }],
    output: { ok: true, instances: [outInstance(7, { meshKey, deform })] },
  });
  const runtime = fakeRuntime([delta(), step('kB', bendB), step('kA', bendA), step('kB', bendB)]);
  const builds = [];
  const reattached = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: new Map([[7, { node: nodeA, source: 0, deform: bendA, meshKey: 'kA', mesh: { old: 1 } }]]),
    remesh(item) {
      builds.push(item.meshKey);
      return { mesh: item.meshKey };
    },
    reattach(entry, token, item) {
      reattached.push({ key: entry.key, meshKey: item.meshKey });
    },
  });

  await adapter.step(0);
  await adapter.step(1);
  await adapter.step(2);
  const result = await adapter.step(3);
  assert.equal(result.remeshed, 0, 'the key was built by the first change');
  assert.deepStrictEqual(builds, ['kB', 'kA']);
  assert.deepStrictEqual(reattached, [
    { key: 7, meshKey: 'kB' },
    { key: 7, meshKey: 'kA' },
    { key: 7, meshKey: 'kB' },
  ]);
});

test('a rejected evaluation changes nothing and reports its issues', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  const issues = [{ severity: 'error', code: 'PROC_PARAM', node: 'move', message: 'bad value', fix: 'fix it' }];
  const runtime = fakeRuntime([delta({
    ok: false,
    issues,
    moved: [{ key: 1, t: [5, 5, 5], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }],
    added: [outInstance(2)],
    removed: [1],
    remesh: [{ meshKey: 'k2', source: 0 }],
  })]);
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: new Map([[1, { node: nodeA }]]),
    remesh() { throw new Error('must not remesh a rejected evaluation'); },
    attach() { throw new Error('must not add to a rejected evaluation'); },
  });

  const result = await adapter.step(3);
  assert.equal(result.ok, false);
  assert.deepStrictEqual(result.issues, issues);
  assert.deepStrictEqual([result.moved, result.added, result.removed, result.remeshed], [0, 0, 0, 0]);
  assert.deepStrictEqual(engine.calls.setTransform, []);
  assert.deepStrictEqual(engine.calls.removeNode, []);
  assert.equal(engine.calls.addNode.length, 2);

  adapter.dispose();
  assert.deepStrictEqual(engine.calls.removeNode, [nodeA]);
});

test('disposing during a pending remesh removes the scene and applies nothing afterwards', async () => {
  const engine = runtimeLightEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  let releaseRemesh;
  const pendingMesh = new Promise((resolve) => { releaseRemesh = resolve; });
  let remeshStarted;
  const started = new Promise((resolve) => { remeshStarted = resolve; });
  const runtime = fakeRuntime([
    delta(),
    delta({
      output: { ok: true, instances: [outInstance(40, { meshKey: 'k9' })], lights: [runtimeLight('b')] },
      remesh: [{ meshKey: 'k9', source: 0, deform: BEND }],
      added: [outInstance(40, { meshKey: 'k9', deform: BEND })],
    }),
  ]);
  const released = [];
  let attached = 0;
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: { ...runtimeScene(), lights: [runtimeLight('a')] },
    runtime,
    instances: new Map([[7, { node: nodeA, source: 0, deform: BEND }]]),
    remesh() {
      remeshStarted();
      return pendingMesh;
    },
    attach() {
      attached += 1;
      return { node: engine.addNode(root) };
    },
    release(token) {
      released.push(token);
    },
  });

  await adapter.step(0);
  const stepping = adapter.step(1);
  await started;
  assert.equal(engine.lights.size, 2, 'the addition is staged while the published light stays live');
  adapter.dispose();
  assert.equal(engine.lights.size, 0);
  assert.deepStrictEqual(engine.calls.removeNode, [nodeA]);
  assert.equal(runtime.disposed, true);
  releaseRemesh({ mesh: 'k9' });
  const result = await stepping;
  assert.equal(result.disposed, true);
  assert.equal(engine.lights.size, 0, 'a stale mesh completion cannot recreate lights');
  assert.equal(attached, 0);
  assert.deepStrictEqual(released, [{ mesh: 'k9' }]);
  assert.equal(engine.calls.addNode.length, 2);
  await assert.rejects(adapter.step(2), (error) => error instanceof SdfmError && error.code === 'sdfm.disposed');
  assert.throws(() => adapter.set({}), (error) => error instanceof SdfmError && error.code === 'sdfm.disposed');
});

test('builds the MAMA Runtime from mama and procgen, and forwards set', () => {
  const engine = runtimeEngine();
  const calls = [];
  const runtime = fakeRuntime([]);
  const adapter = createProceduralRuntime({
    engine,
    scene: runtimeScene(),
    procgen: { call() {} },
    mama: {
      Runtime: {
        create(procgen, input) {
          calls.push({ procgen, input });
          return runtime;
        },
      },
    },
  });
  assert.equal(calls.length, 1);
  assert.deepStrictEqual(calls[0].input, runtimeScene().runtime.input);
  adapter.set({ spin: 4 });
  assert.deepStrictEqual(runtime.sets, [{ spin: 4 }]);
  adapter.dispose();
  assert.equal(runtime.disposed, true);
});

test('refuses a scene without runtime input and a missing MAMA Runtime', () => {
  const engine = runtimeEngine();
  assert.throws(
    () => createProceduralRuntime({ engine, scene: { runtime: null } }),
    (error) => error instanceof SdfmError && error.code === 'sdfm.scene_format',
  );
  assert.throws(
    () => createProceduralRuntime({ engine, scene: runtimeScene() }),
    (error) => error instanceof SdfmError && error.code === 'sdfm.scene_resolve',
  );
});

test('attaches an added instance under the scene root without a custom attach callback', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const runtime = fakeRuntime([delta({ added: [outInstance(5, { t: [7, 0, 0] })] })]);
  const adapter = createProceduralRuntime({ engine, root, scene: runtimeScene(), runtime, instances: [] });

  const result = await adapter.step(2);
  assert.equal(result.added, 1);
  const node = engine.calls.addNode.at(-1);
  assert.equal(node.parent, root);
  assert.deepStrictEqual(node.t, [7, 0, 0]);

  adapter.dispose();
  assert.deepStrictEqual(engine.calls.removeNode, [node.id]);
});

test('an added instance forwards its procgen attributes and LOD bias to every spawned part', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const partA = engine.addMeshInstance(engine.addNode(root), engine.addMesh({
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
  }));
  const partB = engine.addMeshInstance(engine.addNode(root), engine.addMesh({
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices: new Uint32Array([0, 1, 2]),
  }));
  engine.calls.setInstanceParams = [];
  engine.calls.setInstanceLodBias = [];
  engine.setInstanceParams = (handle, params) => engine.calls.setInstanceParams.push([handle, Array.from(params)]);
  engine.setInstanceLodBias = (handle, bias) => engine.calls.setInstanceLodBias.push([handle, bias]);
  const runtime = fakeRuntime([delta({
    added: [outInstance(5, { attrs: { 'tint.r': 1, wind: 2, moss: 3 }, lod: { bias: 1 } })],
  })]);
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: [],
    attach(instance) {
      return { node: engine.addNode(root, instance.t, instance.r, instance.s), parts: [partA, partB] };
    },
  });

  await adapter.step(0);
  assert.deepStrictEqual(engine.calls.setInstanceParams.map(([handle]) => handle), [partA, partB]);
  assert.deepStrictEqual(engine.calls.setInstanceLodBias, [[partA, 1], [partB, 1]]);
  const params = engine.calls.setInstanceParams[0][1];
  assert.equal(params[3], 1);
  assert.equal(params[4], 2);
  assert.equal(params[6], 3);
});

test('a new mesh key listed twice is requested once and shared by every added instance', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const runtime = fakeRuntime([
    delta(),
    delta({
      remesh: [{ meshKey: 'k2', source: 0 }, { meshKey: 'k2', source: 0 }],
      added: [outInstance(1, { meshKey: 'k2', t: [1, 0, 0] }), outInstance(2, { meshKey: 'k2', t: [2, 0, 0] })],
    }),
  ]);
  let builds = 0;
  const tokens = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    remesh(item) {
      builds += 1;
      return { mesh: item.meshKey };
    },
    attach(instance, token) {
      tokens.push(token);
      return { node: engine.addNode(root, instance.t, instance.r, instance.s) };
    },
  });

  await adapter.step(0);
  const result = await adapter.step(1);
  assert.equal(builds, 1);
  assert.equal(result.remeshed, 1);
  assert.equal(result.added, 2);
  assert.deepStrictEqual(tokens, [{ mesh: 'k2' }, { mesh: 'k2' }]);
});

test('overlapping steps evaluate in call order and apply the newest transforms with one build', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  let releaseBuild;
  const build = new Promise((resolve) => { releaseBuild = resolve; });
  let startedBuild;
  const started = new Promise((resolve) => { startedBuild = resolve; });
  const runtime = fakeRuntime([
    delta(),
    delta({ remesh: [{ meshKey: 'k2', source: 0 }], added: [outInstance(1, { meshKey: 'k2', t: [1, 0, 0] })] }),
    delta({ moved: [{ key: 1, t: [2, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }] }),
  ]);
  let builds = 0;
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    remesh() {
      builds += 1;
      startedBuild();
      return build;
    },
    attach(instance) {
      return { node: engine.addNode(root, instance.t, instance.r, instance.s) };
    },
  });

  await adapter.step(0);
  const first = adapter.step(1);
  await started;
  const second = adapter.step(2);
  assert.equal(runtime.lastTick, 2, 'the second tick is evaluated while the first mesh is still building');
  releaseBuild({ mesh: 'k2' });
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a.added, 1);
  assert.equal(b.moved, 1);
  assert.equal(builds, 1);
  assert.deepStrictEqual(engine.calls.setTransform.at(-1).t, [2, 0, 0]);
});

test('a failed remesh rejects the step and leaves the old mesh attached', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  const runtime = fakeRuntime([
    delta(),
    delta({ remesh: [{ meshKey: 'k9', source: 0, deform: BEND }], added: [outInstance(40, { meshKey: 'k9', deform: BEND })] }),
  ]);
  const attached = [];
  let reattached = 0;
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    instances: new Map([[7, { node: nodeA, source: 0, deform: BEND, meshKey: 'k1', mesh: { old: 1 } }]]),
    remesh() {
      throw new Error('the mesher refused the document');
    },
    reattach() {
      reattached += 1;
    },
    attach() {
      attached.push(1);
      return { node: engine.addNode(root) };
    },
  });

  await adapter.step(0);
  await assert.rejects(
    adapter.step(1),
    (error) => error instanceof SdfmError && error.code === 'sdfm.scene_resolve',
  );
  assert.equal(reattached, 0);
  assert.deepStrictEqual(attached, []);
  assert.deepStrictEqual(engine.calls.removeNode, []);
  assert.equal(engine.nodes.has(nodeA), true);

  adapter.dispose();
  assert.deepStrictEqual(engine.calls.removeNode, [nodeA]);
});

test('a mesh completion whose instances a newer step removed is dropped and released', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  let releaseBuild;
  const build = new Promise((resolve) => { releaseBuild = resolve; });
  let startedBuild;
  const started = new Promise((resolve) => { startedBuild = resolve; });
  const runtime = fakeRuntime([
    delta(),
    delta({ remesh: [{ meshKey: 'k9', source: 0 }], added: [outInstance(40, { meshKey: 'k9' })] }),
    delta({ removed: [40] }),
  ]);
  const attached = [];
  const released = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    remesh() {
      startedBuild();
      return build;
    },
    attach(instance) {
      attached.push(instance.key);
      return { node: engine.addNode(root) };
    },
    release(token) {
      released.push(token);
    },
  });

  await adapter.step(0);
  const first = adapter.step(1);
  await started;
  const second = adapter.step(2);
  releaseBuild({ mesh: 'k9' });
  const [a, b] = await Promise.all([first, second]);
  assert.deepStrictEqual([a.added, a.remeshed], [0, 0]);
  assert.equal(b.removed, 0);
  assert.deepStrictEqual(attached, []);
  assert.deepStrictEqual(released, [{ mesh: 'k9' }]);

  adapter.dispose();
  assert.deepStrictEqual(released, [{ mesh: 'k9' }], 'a released completion is not released twice');
});

test('a new mesh key is requested through the existing loader when no remesh callback is given', async () => {
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const loads = [];
  const loader = {
    async load(asset) {
      loads.push(asset);
      return { id: asset.id, parts: [] };
    },
  };
  const runtime = fakeRuntime([
    delta(),
    delta({ remesh: [{ meshKey: 'k2', source: 0, deform: BEND }], added: [outInstance(1, { meshKey: 'k2', deform: BEND })] }),
  ]);
  const tokens = [];
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
    runtime,
    loader,
    meshSource(item) {
      return { id: `procedural.demo.b0-${item.meshKey}`, hash: 'h', document: {} };
    },
    attach(instance, token) {
      tokens.push(token);
      return { node: engine.addNode(root) };
    },
  });

  await adapter.step(0);
  const result = await adapter.step(1);
  assert.equal(loads.length, 1);
  assert.equal(loads[0].id, 'procedural.demo.b0-k2');
  assert.deepStrictEqual(tokens, [{ id: 'procedural.demo.b0-k2', parts: [] }]);
  assert.equal(result.remeshed, 1);
});

function runtimeTraceEngine() {
  const engine = runtimeEngine();
  engine.trace = { sets: [], updates: [], clears: 0 };
  engine.setTraceScene = (program) => engine.trace.sets.push(program);
  engine.updateTraceInstances = (instances) => engine.trace.updates.push(instances);
  engine.clearTraceScene = () => { engine.trace.clears += 1; };
  return engine;
}

function runtimeTraceProgram(input) {
  const sources = Array.isArray(input.sources) ? input.sources.length : 0;
  const instances = Array.isArray(input.instances) ? input.instances : [];
  const words = new Uint32Array(8 + sources * 12 + instances.length * 32);
  words[0] = 0x50544453;
  words[1] = 1;
  words[2] = sources;
  words[3] = instances.length;
  let at = 8 + sources * 12;
  for (const instance of instances) {
    words[at] = instance.key;
    words[at + 1] = instance.source;
    at += 32;
  }
  return new Uint8Array(words.buffer);
}

function runtimeTraceInput(instances) {
  return { sources: [{ key: 'node:one', tree: { root: '', nodes: {} } }], instances };
}

function runtimeTraceInstance(key, extra = {}) {
  return { source: 0, t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], key, ...extra };
}

test('trace stepping moves instances through updateTraceInstances, rebuilds on a topology change and clears when empty', async () => {
  const engine = runtimeTraceEngine();
  const runtime = fakeRuntime([
    delta({ added: [outInstance(1), outInstance(2)] }),
    delta({ moved: [{ key: 1, t: [3, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }] }),
    delta({ added: [outInstance(3, { t: [1, 0, 0] })] }),
    delta({ removed: [1, 2, 3] }),
  ]);
  const packed = [];
  const adapter = createProceduralRuntime({
    engine,
    scene: runtimeScene(),
    runtime,
    trace: {
      input: runtimeTraceInput([runtimeTraceInstance(1), runtimeTraceInstance(2)]),
      procgen: {},
      packTrace(procgen, input) {
        packed.push(input);
        return runtimeTraceProgram(input);
      },
    },
  });

  const baseline = await adapter.step(0);
  assert.deepStrictEqual([baseline.moved, baseline.added, baseline.removed], [0, 0, 0]);
  assert.equal(engine.trace.sets.length + engine.trace.updates.length, 0);

  const moved = await adapter.step(1);
  assert.equal(moved.moved, 1);
  assert.equal(engine.trace.updates.length, 1);
  assert.equal(engine.trace.sets.length, 0);
  const records = new Uint32Array(engine.trace.updates[0].buffer);
  assert.deepStrictEqual([...records].filter((_, index) => index % 32 === 0), [1, 2]);
  assert.deepStrictEqual(packed.at(-1).instances[0].t, [3, 0, 0]);

  const added = await adapter.step(2);
  assert.equal(added.added, 1);
  assert.equal(engine.trace.sets.length, 1);
  assert.equal(engine.trace.updates.length, 1);

  const removed = await adapter.step(3);
  assert.equal(removed.removed, 3);
  assert.equal(engine.trace.clears, 1);
});

test('trace stepping reconciles a metadata-only output by key and repacks the program', async () => {
  const engine = runtimeTraceEngine();
  const bendA = { ...BEND, params: { axis: 'y', angle: 20 }, angle: 0.349 };
  const bendB = { ...BEND, params: { axis: 'y', angle: 45 }, angle: 0.785 };
  const runtime = fakeRuntime([
    delta(),
    delta({
      output: {
        ok: true,
        instances: [
          outInstance(1, { source: 1, meshKey: 'm1' }),
          outInstance(2, { source: 0, meshKey: 'm2', deform: bendB }),
          outInstance(3, { source: 0, meshKey: 'm3', deform: bendA }),
        ],
      },
    }),
  ]);
  const packed = [];
  const adapter = createProceduralRuntime({
    engine,
    scene: runtimeScene(),
    runtime,
    trace: {
      input: runtimeTraceInput([
        runtimeTraceInstance(1),
        runtimeTraceInstance(2, { deform: bendA }),
        runtimeTraceInstance(3, { deform: bendB }),
      ]),
      procgen: {},
      packTrace(procgen, input) {
        packed.push(input);
        return runtimeTraceProgram(input);
      },
    },
  });

  await adapter.step(0);
  const result = await adapter.step(1);
  assert.deepStrictEqual([result.moved, result.added, result.removed, result.remeshed], [0, 0, 0, 0]);
  assert.equal(engine.trace.sets.length, 1, 'the changed source and bends replace the program');
  assert.equal(engine.trace.updates.length, 0);
  assert.deepStrictEqual(
    packed.at(-1).instances.map((instance) => [instance.key, instance.source, instance.deform?.angle]),
    [
      [1, 1, undefined],
      [2, 0, bendB.angle],
      [3, 0, bendA.angle],
    ],
  );
});

test('a refused trace install and a failed trace pack both leave the installed program in place', async () => {
  const engine = runtimeTraceEngine();
  const runtime = fakeRuntime([
    delta(),
    delta({ moved: [{ key: 1, t: [1, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }] }),
    delta({ moved: [{ key: 1, t: [2, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }] }),
    delta({ moved: [{ key: 1, t: [3, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }] }),
  ]);
  let packFails = false;
  const adapter = createProceduralRuntime({
    engine,
    scene: runtimeScene(),
    runtime,
    trace: {
      input: runtimeTraceInput([runtimeTraceInstance(1)]),
      procgen: {},
      packTrace(procgen, input) {
        if (packFails) throw new Error('PROC_LIMIT: too many objects');
        return runtimeTraceProgram(input);
      },
    },
  });
  const install = engine.updateTraceInstances;

  await adapter.step(0);
  engine.updateTraceInstances = () => {
    throw new Error('trace.instance_count: refused');
  };
  await assert.rejects(adapter.step(1), /trace\.instance_count/);
  assert.equal(engine.trace.sets.length, 0);
  assert.equal(engine.trace.updates.length, 0);

  engine.updateTraceInstances = install;
  packFails = true;
  await assert.rejects(
    adapter.step(2),
    (error) => error instanceof SdfmError && error.code === 'sdfm.scene_resolve' && /PROC_LIMIT/.test(error.message),
  );
  assert.equal(engine.trace.sets.length, 0);
  assert.equal(engine.trace.updates.length, 0);

  packFails = false;
  const accepted = await adapter.step(3);
  assert.equal(accepted.moved, 1);
  assert.equal(engine.trace.updates.length, 1);
  assert.equal(new Uint32Array(engine.trace.updates[0].buffer)[0], 1);
});

test('disposing a trace runtime disposes the handle and applies no further step', async () => {
  const engine = runtimeTraceEngine();
  const runtime = fakeRuntime([
    delta(),
    delta({ moved: [{ key: 1, t: [1, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }] }),
  ]);
  const adapter = createProceduralRuntime({
    engine,
    scene: runtimeScene(),
    runtime,
    trace: {
      input: runtimeTraceInput([runtimeTraceInstance(1)]),
      procgen: {},
      packTrace(procgen, input) {
        return runtimeTraceProgram(input);
      },
    },
  });

  await adapter.step(0);
  adapter.dispose();
  assert.equal(runtime.disposed, true);
  await assert.rejects(adapter.step(1), (error) => error instanceof SdfmError && error.code === 'sdfm.disposed');
  assert.equal(engine.trace.updates.length, 0);
  assert.equal(engine.trace.sets.length, 0);
});

// --- Public attachment: loadProceduralScene(...).runtime(procgen) ---------------------------------

const BEND_A = { kind: 'bend', params: { axis: 'y', angle: 20 }, angle: 0.349, length: 1, along: 'x', toward: 'y' };
const BEND_B = { kind: 'bend', params: { axis: 'y', angle: 45 }, angle: 0.785, length: 1, along: 'x', toward: 'y' };

function publicScene(overrides = {}) {
  return {
    format: 'sdfm.procedural-scene', version: 1, variant: 'sdf-refs', name: 'demo', game: 'fixture',
    generator: { document: 'demo.sdfm.json', node: 'host', seed: 1, params: {} },
    sources: [{
      kind: 'asset', file: 'assets/common/post.test.asset.json', ref: { game: 'fixture', id: 'common.post.test' },
      bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    }],
    instances: [{ key: 1, source: 0, t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], group: -1 }],
    groups: [], colliders: [], visual: [],
    runtime: { input: { version: 1, graph: {}, paramValues: {}, timeTicks: 0, sources: {} } },
    ...overrides,
  };
}

function publicMama(deltas, { trace = false } = {}) {
  const calls = { create: [], packed: [] };
  const runtime = fakeRuntime(deltas);
  const mama = {
    parseProceduralScene: (text) => JSON.parse(text),
    sceneSourceDocument(scene) {
      return {
        format: 'sdfm', version: 2, name: scene.name, root: ['source'],
        nodes: { source: { id: 'source', kind: 'group', children: [] } }, materials: {},
      };
    },
    serializeDocument: (doc) => JSON.stringify(doc),
    Runtime: {
      create(procgen, input) {
        calls.create.push({ procgen, input });
        return runtime;
      },
    },
  };
  if (trace) {
    mama.buildSceneTraceInput = (scene) => ({
      sources: scene.sources.map((_source, index) => ({ key: `source:${index}`, tree: { root: '', nodes: {} } })),
      instances: scene.instances.map((instance) => ({ ...instance })),
    });
    mama.packTrace = (procgen, input) => {
      calls.packed.push(input);
      return runtimeTraceProgram(input);
    };
  }
  return { mama, runtime, calls };
}

function publicLoader(log) {
  return {
    async load(asset) {
      log.load.push(asset);
      return { id: asset.id, parts: [{ name: 'p', mesh: 1 }] };
    },
    spawn(asset, parent) {
      const root = 5000 + log.spawn.length;
      log.spawn.push({ id: asset.id, parent, root });
      return { root, parts: { p: { node: root, instance: root + 1 } } };
    },
  };
}

const loadOptions = (fake) => ({ mama: async () => fake.mama, resolveAsset: async (file) => ({ file }) });

test('the public handle builds the MAMA runtime, applies deltas and forwards set', async () => {
  const engine = runtimeEngine();
  const log = { load: [], spawn: [] };
  const fake = publicMama([
    delta(),
    delta({ moved: [{ key: 1, t: [2, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }] }),
  ]);
  const handle = await loadProceduralScene(engine, publicLoader(log), publicScene(), loadOptions(fake));
  const runtime = handle.runtime({ call() {} });
  assert.equal(fake.calls.create.length, 1);
  assert.deepStrictEqual(fake.calls.create[0].input, publicScene().runtime.input);

  const baseline = await runtime.step(0);
  assert.deepStrictEqual([baseline.moved, baseline.remeshed], [0, 0]);
  const moved = await runtime.step(1);
  assert.equal(moved.moved, 1);
  assert.deepStrictEqual(engine.calls.setTransform.at(-1).t, [2, 0, 0]);

  runtime.set({ spin: 2 });
  assert.deepStrictEqual(fake.runtime.sets, [{ spin: 2 }]);
  runtime.dispose();
  assert.equal(fake.runtime.disposed, true);
  handle.dispose();
});

test('the public handle remeshes a metadata-only change by stable key, never by source', async () => {
  const engine = runtimeEngine();
  const log = { load: [], spawn: [] };
  const fake = publicMama([
    delta({
      output: {
        ok: true,
        instances: [outInstance(1, { meshKey: 'a', deform: BEND_A }), outInstance(2, { meshKey: 'b', deform: BEND_B })],
      },
    }),
    delta({
      remesh: [{ meshKey: 'c', source: 0, deform: BEND_B }, { meshKey: 'd', source: 0, deform: BEND_A }],
      output: {
        ok: true,
        instances: [outInstance(1, { meshKey: 'c', deform: BEND_B }), outInstance(2, { meshKey: 'd', deform: BEND_A })],
      },
    }),
  ]);
  const scene = publicScene({
    instances: [
      { key: 1, source: 0, t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], group: -1, deform: BEND_A },
      { key: 2, source: 0, t: [1, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], group: -1, deform: BEND_B },
    ],
  });
  const handle = await loadProceduralScene(engine, publicLoader(log), scene, loadOptions(fake));
  const nodes = engine.calls.addNode.filter((node) => node.parent === handle.root).map((node) => node.id);
  assert.equal(nodes.length, 2);

  const runtime = handle.runtime({ call() {} });
  await runtime.step(0);
  const result = await runtime.step(1);
  assert.deepStrictEqual([result.moved, result.added, result.removed, result.remeshed], [0, 0, 0, 2]);

  const runtimeLoads = log.load.filter((asset) => typeof asset.id === 'string' && asset.id.includes('.r0-'));
  assert.deepStrictEqual(runtimeLoads.map((asset) => asset.id), ['procedural.demo.r0-c', 'procedural.demo.r0-d']);
  assert.deepStrictEqual(runtimeLoads[0].document.nodes.source.deformers,
    [{ id: 'bend', kind: 'bend', enabled: true, params: BEND_B.params }]);
  assert.deepStrictEqual(runtimeLoads[1].document.nodes.source.deformers,
    [{ id: 'bend', kind: 'bend', enabled: true, params: BEND_A.params }]);

  const initialRoots = log.spawn.filter((spawn) => !spawn.id.includes('.r0-')).map((spawn) => spawn.root);
  assert.deepStrictEqual(engine.calls.removeNode, initialRoots);
  const runtimeSpawns = log.spawn.filter((spawn) => spawn.id.includes('.r0-'));
  assert.deepStrictEqual(runtimeSpawns.map((spawn) => [spawn.id, spawn.parent]), [
    ['procedural.demo.r0-c', nodes[0]],
    ['procedural.demo.r0-d', nodes[1]],
  ]);
  runtime.dispose();
  handle.dispose();
});

test('the public handle applies a trace scene runtime delta through the shared trace slot', async () => {
  const engine = runtimeTraceEngine();
  const log = { load: [], spawn: [] };
  const fake = publicMama([
    delta(),
    delta({ moved: [{ key: 1, t: [3, 0, 0], r: [0, 0, 0, 1], rDeg: [0, 0, 0], s: [1, 1, 1] }] }),
  ], { trace: true });
  const handle = await loadProceduralScene(engine, publicLoader(log), publicScene(), {
    ...loadOptions(fake), mode: 'trace', procgen: { call() {} },
  });
  assert.equal(engine.trace.sets.length, 1);

  const runtime = handle.runtime({ call() {} });
  assert.equal(fake.calls.create.length, 1);
  const baseline = await runtime.step(0);
  assert.deepStrictEqual([baseline.moved, baseline.added, baseline.removed], [0, 0, 0]);
  assert.equal(engine.trace.sets.length, 1, 'the loader installed the baseline program');

  const moved = await runtime.step(1);
  assert.equal(moved.moved, 1);
  assert.equal(engine.trace.sets.length, 2, 'the delta repacks the shared trace scene');
  assert.deepStrictEqual(fake.calls.packed.at(-1).instances[0].t, [3, 0, 0]);
  runtime.dispose();
  assert.equal(fake.runtime.disposed, true);
  handle.dispose();
  assert.ok(engine.trace.clears >= 1, 'the last handle clears the trace slot');
});

test('the public handle refuses runtime for a scene without runtime input', async () => {
  const engine = runtimeEngine();
  const fake = publicMama([]);
  const handle = await loadProceduralScene(engine, publicLoader({ load: [], spawn: [] }),
    publicScene({ runtime: null }), loadOptions(fake));
  assert.throws(() => handle.runtime({ call() {} }),
    (error) => error instanceof SdfmError && error.code === 'sdfm.scene_format');
  handle.dispose();
});

const runtimeLight = (id, fields = {}) => ({
  id, node: 'light', key: 1, t: [1, 2, 3], r: [0, 0, 0, 1], s: [1, 1, 1],
  kind: 'spot', color: [1, 1, 1], intensity: 3, range: 8, angleDeg: 60, shadows: false, ...fields,
});
const lightOutput = (sections = {}) => ({ ok: true, instances: [outInstance(1)], ...sections });
const lightRefusal = Object.assign(new Error('light creation refused'), { code: 'light.invalid_desc' });

// Local S17 double keeps handle identities and explicit optional-field resets observable.
function runtimeLightEngine() {
  const engine = runtimeTraceEngine();
  engine.lights = new Map();
  engine.lightAdds = 0;
  engine.addLight = (desc) => {
    const handle = ++engine.lightAdds;
    if (handle === engine.failAddAt) throw lightRefusal;
    engine.lights.set(handle, structuredClone(desc));
    return handle;
  };
  engine.updateLight = (handle, changes) => {
    assert.ok(engine.lights.has(handle));
    Object.assign(engine.lights.get(handle), structuredClone(changes));
  };
  engine.removeLight = (handle) => assert.ok(engine.lights.delete(handle), 'remove each handle once');
  return engine;
}

function lightAdapter(engine, deltas, trace, options = {}) {
  const node = engine.addNode(null);
  return createProceduralRuntime({
    engine, runtime: fakeRuntime(deltas),
    scene: { ...runtimeScene(), lights: [runtimeLight('a')] },
    instances: [{ key: 1, node }],
    ...(trace ? { trace: {
      input: runtimeTraceInput([runtimeTraceInstance(1)]), procgen: {}, packTrace: (_, input) => runtimeTraceProgram(input),
    } } : {}),
    ...options,
  });
}

for (const trace of [false, true]) {
  test(`${trace ? 'trace' : 'mesh'} reconciles complete light snapshots with stable identities and placement`, async () => {
    const engine = runtimeLightEngine();
    const placement = trace ? [10, -20, 30] : [0, 0, 0];
    const options = {};
    if (trace) {
      options.lights = createMamaLightOwner(engine, placement);
      options.lights.prepare([runtimeLight('a')]).commit();
    }
    const first = runtimeLight('a', { t: [2, 3, 4], innerAngleDeg: 20 });
    const changed = runtimeLight('a', { t: [-2, 5, 8], intensity: 9, range: 12 });
    const original = structuredClone([first, changed]);
    const adapter = lightAdapter(engine, [
      delta({ output: lightOutput({ lights: [first] }), added: [outInstance(1)], remesh: [{ meshKey: 'm', source: 0 }] }),
      delta(),
      delta({ ok: false, output: lightOutput(), issues: ['delta refused'] }),
      delta({ output: lightOutput({ ok: false }), issues: ['output refused'] }),
      delta({ output: lightOutput({ lights: [first] }) }),
      delta({ output: lightOutput({ lights: [changed] }) }),
      delta({ output: lightOutput({ lights: [runtimeLight('b')] }) }),
      delta({ output: lightOutput() }),
      delta({ output: lightOutput({ lights: [runtimeLight('c')] }) }),
      delta({ output: lightOutput({ lights: [] }) }),
    ], trace, { ...options, remesh() { throw new Error('baseline geometry must not rebuild'); } });
    const initialHandle = [...engine.lights.keys()][0];
    const baseline = await adapter.step(0);
    assert.deepEqual([baseline.moved, baseline.added, baseline.removed, baseline.remeshed], [0, 0, 0, 0]);
    assert.deepEqual(Object.keys(baseline).sort(), ['added', 'delta', 'issues', 'moved', 'ok', 'remeshed', 'removed']);
    assert.deepEqual(engine.lights.get(initialHandle).position, first.t.map((v, i) => v + placement[i]));
    for (let tick = 1; tick <= 4; tick += 1) {
      const result = await adapter.step(tick);
      assert.equal(result.ok, tick !== 2 && tick !== 3);
      if (!result.ok) assert.deepEqual(result.issues, [tick === 2 ? 'delta refused' : 'output refused']);
      assert.deepEqual([...engine.lights.keys()], [initialHandle]);
      assert.equal(engine.lights.get(initialHandle).innerAngleDeg, 20);
    }
    await adapter.step(5);
    assert.deepEqual([...engine.lights.keys()], [initialHandle]);
    assert.deepEqual(engine.lights.get(initialHandle).position, changed.t.map((v, i) => v + placement[i]));
    assert.equal(engine.lights.get(initialHandle).intensity, 9);
    assert.equal(engine.lights.get(initialHandle).range, 12);
    assert.equal(engine.lights.get(initialHandle).innerAngleDeg, undefined);
    assert.deepEqual([first, changed], original);
    await adapter.step(6);
    assert.equal(engine.lights.has(initialHandle), false);
    assert.equal(engine.lights.size, 1);
    await adapter.step(7);
    assert.equal(engine.lights.size, 0);
    await adapter.step(8);
    assert.equal(engine.lights.size, 1);
    await adapter.step(9);
    assert.equal(engine.lights.size, 0);
    assert.equal(engine.lightAdds, 3, 'the supplied registry is reused and repeated snapshots allocate nothing');
    assert.equal(engine.calls.setTransform.length + engine.trace.updates.length + engine.trace.sets.length, 0);
    adapter.dispose();
    adapter.dispose();
    if (options.lights) assert.throws(() => options.lights.prepare([]), { code: 'sdfm.disposed' });
  });

  test(`${trace ? 'trace' : 'mesh'} prepares before geometry and rolls lights back on application failure`, async () => {
    const engine = runtimeLightEngine();
    const refusal = Object.assign(new Error('application refused'), { code: 'api.unknown_handle' });
    const snapshot = lightOutput({ lights: [runtimeLight('a', { intensity: 9 }), runtimeLight('b'), runtimeLight('c')] });
    const moved = [outInstance(1, { t: [5, 0, 0] })];
    const adapter = lightAdapter(engine, [
      delta(), delta({ moved, removed: trace ? [] : [1], output: snapshot }),
      delta({ moved, output: snapshot }), delta({ moved, output: snapshot }),
    ], trace);
    await adapter.step(0);
    engine.failAddAt = 3;
    await assert.rejects(adapter.step(1), (error) => error === lightRefusal);
    assert.equal(engine.calls.removeNode.length + engine.calls.setTransform.length + engine.trace.updates.length, 0);
    assert.deepEqual([...engine.lights.keys()], [1]);
    assert.equal(engine.lights.get(1).intensity, 3);
    const apply = trace ? engine.updateTraceInstances : engine.setTransform;
    engine[trace ? 'updateTraceInstances' : 'setTransform'] = () => { throw refusal; };
    await assert.rejects(adapter.step(2), (error) => error === refusal);
    assert.deepEqual([...engine.lights.keys()], [1]);
    assert.equal(engine.lights.get(1).intensity, 3);
    engine[trace ? 'updateTraceInstances' : 'setTransform'] = apply;
    const accepted = await adapter.step(3);
    assert.equal(accepted.moved, 1);
    assert.equal(accepted.ok, true);
    assert.equal(engine.lights.size, 3);
    assert.equal(engine.lights.get(1).intensity, 9);
    adapter.dispose();
    assert.equal(engine.lights.size, 0);
  });
}

test('disposal aborts lights while a trace application is pending and queued steps cannot recreate them', async () => {
  const engine = runtimeLightEngine();
  const lights = createMamaLightOwner(engine, [10, 20, 30]);
  lights.prepare([runtimeLight('a')]).commit();
  let started;
  let finish;
  const applying = new Promise((resolve) => { started = resolve; });
  const pending = new Promise((resolve) => { finish = resolve; });
  const adapter = lightAdapter(engine, [
    delta(),
    delta({ moved: [outInstance(1)], output: lightOutput({ lights: [runtimeLight('b')] }) }),
    delta({ output: lightOutput({ lights: [runtimeLight('c')] }) }),
  ], true, {
    lights,
    trace: {
      input: runtimeTraceInput([runtimeTraceInstance(1)]), procgen: {}, packTrace: (_, input) => runtimeTraceProgram(input),
      commit() {
        assert.equal(engine.lights.get(1).intensity, 3);
        assert.equal([...engine.lights.values()][1].intensity, 0, 'staged additions stay dark until application succeeds');
        started();
        return pending;
      },
    },
  });
  await adapter.step(0);
  const stepping = adapter.step(1);
  await applying;
  const queued = adapter.step(2);
  const rejected = assert.rejects(queued, { code: 'sdfm.disposed' });
  adapter.dispose();
  adapter.dispose();
  assert.equal(engine.lights.size, 0);
  finish();
  assert.equal((await stepping).disposed, true);
  await rejected;
  assert.equal(engine.lightAdds, 2);
  assert.equal(engine.lights.size, 0);
  assert.throws(() => lights.prepare([]), { code: 'sdfm.disposed' });
});

test('direct runtime initialization releases staged lights and its evaluator on creation failure', () => {
  const engine = runtimeLightEngine();
  const runtime = fakeRuntime([]);
  engine.failAddAt = 2;
  assert.throws(() => createProceduralRuntime({
    engine, runtime, scene: { ...runtimeScene(), lights: [runtimeLight('a'), runtimeLight('b')] },
  }), (error) => error === lightRefusal);
  assert.equal(engine.lights.size, 0);
  assert.equal(runtime.disposed, true);
  assert.equal(engine.calls.addNode.length + engine.calls.setTransform.length, 0);
});
