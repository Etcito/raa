import assert from 'node:assert/strict';
import { copyFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

const here = dirname(fileURLToPath(import.meta.url));

async function stubbedPackage(result) {
  const dir = await mkdtemp(join(tmpdir(), 'raa-frozen-pipeline-'));
  await mkdir(join(dir, 'pkg'));
  await copyFile(join(here, '..', 'index.js'), join(dir, 'index.js'));
  await copyFile(join(here, '..', 'sky-weather.js'), join(dir, 'sky-weather.js'));
  await writeFile(join(dir, 'pkg', 'raa_js.js'), `
    export const calls = { create: [], frozen: [], dispose: 0 };
    export default async function init() {}
    export class Engine {
      static create(canvas, options) {
        calls.create.push([canvas, options]);
        if (options.sync) return 'sync-engine';
        return Promise.resolve(engine);
      }
    }
    export const engine = {
      setFrozenRenderGraph(name) { calls.frozen.push(name); return ${JSON.stringify(result)}; },
      dispose() { calls.dispose++; },
    };
  `);
  return dir;
}

test('createEngine strips pipeline and applies a frozen plan once', async () => {
  const dir = await stubbedPackage({ ok: true, issues: [] });
  try {
    const { loadRaa } = await import(pathToFileURL(join(dir, 'index.js')));
    const wasm = await import(pathToFileURL(join(dir, 'pkg', 'raa_js.js')));
    const { calls } = wasm;
    const raa = await loadRaa();
    const options = { pipeline: 'simple_forward', msaa: 1 };
    const engine = await raa.createEngine('canvas', options);
    assert.deepEqual(calls.create, [['canvas', { msaa: 1 }]]);
    assert.deepEqual(calls.frozen, ['simple_forward']);
    assert.equal(calls.dispose, 0);
    assert.deepEqual(options, { pipeline: 'simple_forward', msaa: 1 });
    assert.strictEqual(engine, wasm.engine);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('createEngine disposes and rejects when a frozen plan is refused', async () => {
  const dir = await stubbedPackage({
    ok: false,
    issues: [
      { code: 'graph.frozen_unknown', message: 'unknown frozen plan' },
      { code: 'graph.frozen_stale', message: 'frozen plan is stale' },
    ],
  });
  try {
    const { loadRaa } = await import(pathToFileURL(join(dir, 'index.js')));
    const { calls } = await import(pathToFileURL(join(dir, 'pkg', 'raa_js.js')));
    const raa = await loadRaa();
    await assert.rejects(raa.createEngine('canvas', { pipeline: 'missing', msaa: 4 }), {
      name: 'Error',
      message: 'graph.frozen_unknown: unknown frozen plan\ngraph.frozen_stale: frozen plan is stale',
    });
    assert.deepEqual(calls.create, [['canvas', { msaa: 4 }]]);
    assert.deepEqual(calls.frozen, ['missing']);
    assert.equal(calls.dispose, 1);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('createEngine without a pipeline keeps the synchronous direct return', async () => {
  const dir = await stubbedPackage({ ok: true, issues: [] });
  try {
    const { loadRaa } = await import(pathToFileURL(join(dir, 'index.js')));
    const { calls } = await import(pathToFileURL(join(dir, 'pkg', 'raa_js.js')));
    const raa = await loadRaa();
    const options = { sync: true, msaa: 4 };
    const engine = raa.createEngine('canvas', options);
    assert.equal(engine, 'sync-engine');
    assert.deepEqual(calls.create, [['canvas', options]]);
    assert.deepEqual(calls.frozen, []);
    assert.deepEqual(options, { sync: true, msaa: 4 });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
