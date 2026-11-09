import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { constants, runInNewContext } from 'node:vm';
import test from 'node:test';
import { variantProblems } from '../../web/common/lights-job.mjs';
import point from './scenes/point.js';

const read = (file) => readFileSync(new URL(file, import.meta.url), 'utf8');
const tick = () => new Promise((resolve) => setImmediate(resolve));

function page(index = []) {
  const packageReady = Promise.withResolvers();
  const context = {
    window: {}, navigator: {}, geometry: {}, objects: {},
    document: { getElementById: () => ({ toDataURL: () => 'data:image/png;base64,cG5n' }) },
    loadRaa: () => packageReady.promise,
    fetch: async () => ({ ok: true, json: async () => index }),
    runScene: async (_raa, _canvas, scene, variant, args) => {
      args.onEngine({ dispose() {} });
      return { scene: scene.id, variant: variant.id, nonEmpty: true };
    },
    renderAt: () => {}, variantTimeMs: (variant) => variant.timeMs,
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
  assert.equal(result.files[0].name, 'capture.png');
  assert.equal(result.files[0].base64, 'cG5n');
});
