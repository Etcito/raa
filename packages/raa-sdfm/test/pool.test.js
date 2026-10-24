import { test } from 'node:test';
import assert from 'node:assert/strict';

import * as fakeMama from './fake-mama.js';
import { createPool, defaultPoolSize } from '../src/pool.js';
import { createWorkerHandler } from '../worker.js';
import { SdfmError } from '../src/errors.js';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** A fake-MAMA document with one group named `name` (one part, one sub-mesh). */
function doc(name, extra = {}) {
  return {
    nodes: {
      [name]: {
        kind: 'group', name, asset: {},
        fake: { parts: [{ partId: 'p', name: `${name}-part`, frame: IDENTITY, subMeshes: [{ material: 'm', vertices: 3 }] }] },
        ...extra,
      },
    },
    materials: {
      m: { name: 'M', fake: { wgslBody: '  return MaterialFull(vec3f(1.0), 0.0, 0.5, vec3f(0.0), 0.0);', helpers: [], constants: [] } },
    },
  };
}

/**
 * In-process Worker: runs worker.js's handler with the fake MAMA. A document whose `crash` key is
 * set makes the worker fire an `error` event (an uncaught exception) instead of answering.
 */
class FakeWorker extends EventTarget {
  constructor(url, { delayMs = 0 } = {}) {
    super();
    this.url = url;
    this.received = [];
    this.transfers = [];
    this.terminated = false;
    this.delayMs = delayMs;
    this.handle = createWorkerHandler({
      loadMama: async () => fakeMama,
      post: (message, transfer) => {
        if (this.terminated) return;
        this.transfers.push(transfer);
        const data = structuredClone(message, { transfer });
        this.dispatchEvent(new MessageEvent('message', { data }));
      },
    });
  }

  postMessage(message) {
    const data = structuredClone(message);
    this.received.push(data);
    setTimeout(() => {
      if (this.terminated) return;
      if (data.type === 'mesh' && data.document.crash) {
        const event = new Event('error', { cancelable: true });
        event.message = 'Uncaught RangeError: boom';
        this.dispatchEvent(event);
        return;
      }
      this.handle(data);
    }, data.type === 'mesh' ? this.delayMs : 0);
  }

  terminate() {
    this.terminated = true;
  }
}

function fakeFactory(delays = []) {
  const created = [];
  const createWorker = (url) => {
    const w = new FakeWorker(url, { delayMs: delays[created.length] ?? 0 });
    created.push(w);
    return w;
  };
  return { created, createWorker };
}

const meshes = (w) => w.received.filter((m) => m.type === 'mesh');
const inits = (w) => w.received.filter((m) => m.type === 'init');

test('requests spread round-robin over the pool, one init per worker', async () => {
  const { created, createWorker } = fakeFactory();
  const pool = createPool({ size: 3, workerUrl: 'worker.js', mamaUrl: 'https://example.test/mama/', createWorker });
  assert.equal(pool.size, 3);
  assert.equal(created.length, 3);
  await Promise.all(Array.from({ length: 7 }, (_, i) => pool.mesh(doc(`g${i}`), { resolution: 16 })));
  assert.deepEqual(created.map((w) => meshes(w).length), [3, 2, 2]);
  for (const w of created) {
    assert.equal(w.url, 'worker.js');
    assert.deepEqual(inits(w), [{ type: 'init', mamaUrl: 'https://example.test/mama/' }]);
    assert.deepEqual(meshes(w)[0].mesher, { resolution: 16 });
  }
  pool.terminate();
});

test('results resolve by request id even when workers answer out of order', async () => {
  const { created, createWorker } = fakeFactory([30, 0]);
  const pool = createPool({ size: 2, workerUrl: 'worker.js', createWorker });
  const order = [];
  const names = ['slow0', 'fast0', 'slow1', 'fast1'];
  const results = await Promise.all(names.map((n) => pool.mesh(doc(n), {}).then((r) => (order.push(n), r))));
  assert.deepEqual(order.slice(0, 2).sort(), ['fast0', 'fast1']);
  results.forEach((r, i) => {
    assert.equal(r.groups.length, 1);
    assert.equal(r.groups[0].group, names[i]);
    const mesh = r.groups[0].parts[0].meshes[0];
    assert.ok(mesh.positions instanceof Float32Array && mesh.indices instanceof Uint32Array);
    assert.equal(r.materials[0].name, 'm');
  });
  const ids = created.flatMap(meshes).map((m) => m.id);
  assert.equal(new Set(ids).size, 4);
  // positions, normals, indices and the frame travel as transferables
  assert.equal(created[1].transfers[0].length, 4);
  pool.terminate();
});

test('a worker refusal rejects with its SdfmError code, message and path', async () => {
  const { createWorker } = fakeFactory();
  const pool = createPool({ size: 1, workerUrl: 'worker.js', createWorker });
  const refused = doc('wheel', { fake: { refuse: { message: 'no triangle' } } });
  await assert.rejects(pool.mesh(refused, {}), (e) => {
    assert.ok(e instanceof SdfmError);
    assert.equal(e.code, 'sdfm.mesher_refused');
    assert.equal(e.path, '/document');
    assert.match(e.message, /^sdfm\.mesher_refused: wheel: .*no triangle/);
    return true;
  });
  const thrown = doc('wheel', { fake: { throw: 'mesher exploded' } });
  await assert.rejects(pool.mesh(thrown, {}), (e) => e.code === 'sdfm.worker' && /mesher exploded/.test(e.message));
  pool.terminate();
});

test('a crashing worker rejects its pending requests with sdfm.worker and is replaced', async () => {
  const { created, createWorker } = fakeFactory([20, 0]);
  const pool = createPool({ size: 2, workerUrl: 'worker.js', createWorker });
  const pendingOnCrasher = pool.mesh(doc('a'), {}); // worker 0, slow: still pending at the crash
  const other = pool.mesh(doc('b'), {}); // worker 1
  const crash = pool.mesh({ ...doc('c'), crash: true }, {}); // worker 0
  await assert.rejects(crash, (e) => e instanceof SdfmError && e.code === 'sdfm.worker' && /boom/.test(e.message));
  await assert.rejects(pendingOnCrasher, (e) => e.code === 'sdfm.worker');
  assert.equal((await other).groups[0].group, 'b');
  assert.equal(created.length, 3);
  assert.equal(created[0].terminated, true);
  assert.equal(inits(created[2]).length, 1);
  const after = await Promise.all([pool.mesh(doc('d'), {}), pool.mesh(doc('e'), {})]);
  assert.deepEqual(after.map((r) => r.groups[0].group), ['d', 'e']);
  assert.equal(meshes(created[2]).length + meshes(created[1]).length, 3);
  pool.terminate();
});

test('a messageerror event also rejects with sdfm.worker', async () => {
  const { created, createWorker } = fakeFactory([50]);
  const pool = createPool({ size: 1, workerUrl: 'worker.js', createWorker });
  const pending = pool.mesh(doc('a'), {});
  created[0].dispatchEvent(new Event('messageerror'));
  await assert.rejects(pending, (e) => e.code === 'sdfm.worker' && /deserialised/.test(e.message));
  assert.equal(created.length, 2);
  pool.terminate();
});

test('after terminate pending and later calls reject sdfm.disposed', async () => {
  const { created, createWorker } = fakeFactory([50, 50]);
  const pool = createPool({ size: 2, workerUrl: 'worker.js', createWorker });
  const pending = [pool.mesh(doc('a'), {}), pool.mesh(doc('b'), {})];
  pool.terminate();
  for (const p of pending) await assert.rejects(p, (e) => e instanceof SdfmError && e.code === 'sdfm.disposed');
  await assert.rejects(pool.mesh(doc('c'), {}), (e) => e.code === 'sdfm.disposed');
  assert.ok(created.every((w) => w.terminated));
  pool.terminate(); // idempotent
});

test('default size is min(4, navigator.hardwareConcurrency || 2)', (t) => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  t.after(() => Object.defineProperty(globalThis, 'navigator', original));
  const cases = [[16, 4], [3, 3], [0, 2], [undefined, 2]];
  for (const [cores, expected] of cases) {
    Object.defineProperty(globalThis, 'navigator', { value: { hardwareConcurrency: cores }, configurable: true });
    assert.equal(defaultPoolSize(), expected);
    const { created, createWorker } = fakeFactory();
    const pool = createPool({ workerUrl: 'worker.js', createWorker });
    assert.equal(pool.size, expected);
    assert.equal(created.length, expected);
    pool.terminate();
  }
  assert.throws(() => createPool({ size: 0, workerUrl: 'w.js', createWorker: fakeFactory().createWorker }), RangeError);
});

test('worker handler: mesh requests that arrive before init wait for it', async () => {
  const posted = [];
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  let loadedWith;
  const handle = createWorkerHandler({
    loadMama: async (url) => {
      loadedWith = url;
      await gate;
      return fakeMama;
    },
    post: (message) => posted.push(message),
  });
  const early = handle({ type: 'mesh', id: 7, document: doc('early'), mesher: {} });
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(posted.length, 0);
  handle({ type: 'init', mamaUrl: 'mama/' });
  handle({ type: 'init', mamaUrl: 'other/' }); // ignored: one init per worker
  release();
  await early;
  assert.equal(loadedWith, 'mama/');
  assert.equal(posted.length, 1);
  assert.equal(posted[0].type, 'meshed');
  assert.equal(posted[0].id, 7);
  assert.equal(posted[0].groups[0].group, 'early');
});

test('worker handler: a failed init answers each mesh request with an error', async () => {
  const posted = [];
  const handle = createWorkerHandler({
    loadMama: async () => {
      throw new SdfmError('sdfm.worker', 'cannot import MAMA module x');
    },
    post: (message) => posted.push(message),
  });
  handle({ type: 'init', mamaUrl: 'x' });
  await handle({ type: 'mesh', id: 1, document: doc('a'), mesher: {} });
  assert.deepEqual(posted, [{ type: 'error', id: 1, code: 'sdfm.worker', message: 'cannot import MAMA module x' }]);
});
