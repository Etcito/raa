import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createFakeEngine } from './fake-engine.js';
import { createProceduralRuntime } from '../src/procedural-runtime.js';
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
  const engine = runtimeEngine();
  const root = engine.addNode(null);
  const nodeA = engine.addNode(root);
  let releaseRemesh;
  const pendingMesh = new Promise((resolve) => { releaseRemesh = resolve; });
  let remeshStarted;
  const started = new Promise((resolve) => { remeshStarted = resolve; });
  const runtime = fakeRuntime([
    delta(),
    delta({
      remesh: [{ meshKey: 'k9', source: 0, deform: BEND }],
      added: [outInstance(40, { meshKey: 'k9', deform: BEND })],
    }),
  ]);
  const released = [];
  let attached = 0;
  const adapter = createProceduralRuntime({
    engine,
    root,
    scene: runtimeScene(),
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
  adapter.dispose();
  assert.deepStrictEqual(engine.calls.removeNode, [nodeA]);
  assert.equal(runtime.disposed, true);
  releaseRemesh({ mesh: 'k9' });
  const result = await stepping;
  assert.equal(result.disposed, true);
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
