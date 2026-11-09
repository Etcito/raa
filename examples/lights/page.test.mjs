import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { constants, runInNewContext } from 'node:vm';
import test from 'node:test';
import { variantProblems } from '../../web/common/lights-job.mjs';
import point from './scenes/point.js';

const read = (file) => readFileSync(new URL(file, import.meta.url), 'utf8');
const tick = () => new Promise((resolve) => setImmediate(resolve));

function page(index = [], overrides = {}) {
  const packageReady = Promise.withResolvers();
  const context = {
    window: {}, navigator: { userAgent: 'fixture browser', platform: 'fixture platform' },
    devicePixelRatio: 2, geometry: {}, objects: {},
    document: { getElementById: () => ({ toDataURL: () => 'data:image/png;base64,cG5n' }) },
    loadRaa: () => packageReady.promise,
    fetch: async () => ({ ok: true, json: async () => index }),
    runScene: async (_raa, _canvas, scene, variant, args) => {
      args.onEngine({ dispose() {} });
      return { scene: scene.id, variant: variant.id, nonEmpty: true };
    },
    renderAt: () => {}, variantTimeMs: (variant) => variant.timeMs, ...overrides,
  };
  // Same page-function CPU execution convention as scripts/anim-capture.test.mjs.
  const transport = read('../../web/common/nmc-gpu.js').replace('export function', 'function');
  const source = read('./main.js').replace(/^import .*;\n/gm, '');
  runInNewContext(transport + '\n' + source, context, {
    filename: fileURLToPath(new URL('./main.js', import.meta.url)),
    importModuleDynamically: constants.USE_MAIN_CONTEXT_DEFAULT_LOADER,
  });
  return { context, packageReady, gpu: context.window.nmcGpu };
}

test('ops register while preparation is pending and wait for its failure', async () => {
  const { gpu, packageReady } = page();
  assert.equal(gpu.ready, false);
  assert.equal(gpu.preparation.status, 'pending');
  let settled = false;
  const result = gpu.ops.render({ args: { scene: 'point', variant: 'on' } });
  result.then(() => { settled = true; }, () => { settled = true; });
  await tick();
  assert.equal(settled, false);
  packageReady.reject(new Error('package preparation failed'));
  await assert.rejects(result, /package preparation failed/);
  await tick();
  assert.equal(gpu.ready, false);
  assert.equal(gpu.preparation.status, 'failed');
  assert.equal(gpu.preparation.error, 'package preparation failed');
});

test('successful preparation marks readiness and rejects unknown metadata keys', async () => {
  const { gpu, packageReady } = page();
  packageReady.resolve({});
  await tick();
  assert.equal(gpu.ready, true);
  assert.equal(gpu.preparation.status, 'ready');
  await assert.rejects(gpu.ops.render({ scene: 'missing', variant: 'on' }), /unknown scene\/variant/);
});

test('neutral point variants have visible subjects and a built protocol entry', () => {
  assert.deepEqual(point.variants.map((v) => v.id), ['on', 'off']);
  for (const variant of point.variants) {
    assert.deepEqual(variantProblems(variant), []);
    assert.match(variant.description, /Neutral.*local lights cast no shadow yet/);
  }
  assert.deepEqual(JSON.parse(read('./nmc.gpu.json')), { page: 'index.html', ops: ['render'] });
});

test('discovered scene dispatch returns the shared transport capture envelope', async () => {
  const { gpu, packageReady } = page([{ id: 'point', file: 'scenes/point.js' }]);
  packageReady.resolve({});
  const result = await gpu.ops.render({ args: { scene: 'point', variant: 'off' } });
  assert.equal(gpu.ready, true);
  assert.equal(result.json.scene, 'point');
  assert.equal(result.json.variant, 'off');
  assert.deepEqual({ ...result.json.browser }, { userAgent: 'fixture browser', platform: 'fixture platform', devicePixelRatio: 2 });
  assert.equal(result.files[0].name, 'capture.png');
  assert.equal(result.files[0].base64, 'cG5n');
});

test('missing browser identity rejects instead of inventing metadata', async () => {
  const { gpu, packageReady } = page([{ id: 'point', file: 'scenes/point.js' }], { navigator: {} });
  packageReady.resolve({});
  await assert.rejects(gpu.ops.render({ args: { scene: 'point', variant: 'on' } }), /browser metadata missing/);
});

test('uncaptured errors and device loss retain their causes and reject the operation', async () => {
  for (const kind of ['uncaptured', 'lost']) {
    const loss = Promise.withResolvers();
    let listener;
    const device = { lost: loss.promise, addEventListener: (_name, callback) => { listener = callback; } };
    const navigator = { userAgent: 'browser', platform: 'platform', gpu: {
      requestAdapter: async () => ({ requestDevice: async () => device }),
    } };
    const { gpu, packageReady } = page([{ id: 'point', file: 'scenes/point.js' }], {
      navigator,
      runScene: async (_raa, _canvas, scene, variant, args) => {
        args.onEngine({ dispose() {} });
        await (await navigator.gpu.requestAdapter()).requestDevice();
        if (kind === 'uncaptured') listener({ error: new Error('shader failed', { cause: new Error('invalid binding') }) });
        else { loss.resolve({ reason: 'unknown', message: 'device vanished', cause: new Error('driver reset') }); await Promise.resolve(); }
        return { scene: scene.id, variant: variant.id };
      },
    });
    packageReady.resolve({});
    await assert.rejects(gpu.ops.render({ args: { scene: 'point', variant: 'on' } }),
      kind === 'uncaptured' ? /shader failed: invalid binding/ : /unknown: device vanished: driver reset/);
  }
});

test('capture preserves the creating adapter identity when wgpu loses its labels', async () => {
  let current;
  const navigator = { userAgent: 'browser', platform: 'platform', gpu: {
    requestAdapter: async () => ({ info: current, requestDevice: async () => ({
      lost: new Promise(() => {}), addEventListener() {},
    }) }),
  } };
  const { gpu, packageReady } = page([{ id: 'point', file: 'scenes/point.js' }], {
    navigator,
    runScene: async (_raa, _canvas, scene, variant, args) => {
      await (await navigator.gpu.requestAdapter()).requestDevice();
      args.onEngine({ dispose() {} });
      return { scene: scene.id, variant: variant.id, nonEmpty: true,
        adapter: { vendor: '', architecture: '', device: '', description: '', isFallbackAdapter: false }, hardware: true };
    },
  });
  packageReady.resolve({});
  for (const info of [
    { vendor: 'nvidia', architecture: 'ampere', device: '', description: '', isFallbackAdapter: false },
    { vendor: 'google', architecture: '', device: '', description: 'SwiftShader', isFallbackAdapter: false },
    { vendor: 'nvidia', architecture: 'ampere', device: '', description: '', isFallbackAdapter: true },
    { vendor: '', architecture: '', device: '', description: '', isFallbackAdapter: false },
  ]) {
    current = info;
    const { json } = await gpu.ops.render({ args: { scene: 'point', variant: 'on' } });
    assert.deepEqual({ ...json.adapter }, info);
    assert.equal(json.hardware, !info.isFallbackAdapter && !/SwiftShader/.test(info.description));
  }
});


test('one whole-flow operation continues after failures and collects errors and captures', async () => {
  const events = new Map();
  const window = {
    addEventListener: (type, handler) => { const handlers = events.get(type) ?? new Set(); handlers.add(handler); events.set(type, handlers); },
    removeEventListener: (type, handler) => events.get(type)?.delete(handler),
  };
  const console = { error() {} };
  const calls = [];
  const { gpu, packageReady, context } = page([{ id: 'point', file: 'scenes/point.js' }], {
    window, console,
    runScene: async (_raa, _canvas, scene, variant, args) => {
      calls.push(variant.id);
      if (variant.id === 'on') {
        console.error('shader diagnostic');
        for (const handler of events.get('error')) handler({ error: new Error('page broke') });
        for (const handler of events.get('unhandledrejection')) handler({ reason: new Error('async broke') });
        await context.fetch('missing-asset');
        throw new Error('expected state absent');
      }
      args.onEngine({ dispose() {} });
      return { scene: scene.id, variant: variant.id, nonEmpty: true };
    },
  });
  packageReady.resolve({});
  await tick();
  context.fetch = async () => ({ ok: false, status: 404 });
  const fetch = context.fetch, error = console.error;
  const result = await gpu.ops.render({ args: { pairs: [
    { scene: 'point', variant: 'on' }, { scene: 'point', variant: 'off' },
  ] } });
  assert.deepEqual(calls, ['on', 'off']);
  assert.match(result.json.steps[0].error, /expected state absent/);
  assert.equal(result.json.steps[1].result.variant, 'off');
  assert.equal(result.files[0].name, 's-point/off/capture.png');
  for (const kind of ['console.error', 'pageerror', 'failed-request', 'unmet-state']) {
    assert.ok(result.json.diagnostics.some(d => d.kind === kind), kind);
  }
  assert.equal(context.fetch, fetch);
  assert.equal(console.error, error);
  assert.ok([...events.values()].every(handlers => handlers.size === 0));
});
