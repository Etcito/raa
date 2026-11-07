// Overlapping loadRaa calls share one wasm instantiation (sdfm card 7126: the app's displays and the capture
// scenario loaded RAA at the same time, the module was instantiated twice and engines of the first instance
// faulted in the second: "memory access out of bounds"). The wasm-bindgen glue is replaced by a counting stub.
import assert from 'node:assert/strict';
import { copyFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

const here = dirname(fileURLToPath(import.meta.url));

async function stubbedPackage(failFirst) {
  const dir = await mkdtemp(join(tmpdir(), 'raa-load-once-'));
  await mkdir(join(dir, 'pkg'));
  await copyFile(join(here, '..', 'index.js'), join(dir, 'index.js'));
  await writeFile(join(dir, 'pkg', 'raa_js.js'), `
    export const calls = { init: 0 };
    let failures = ${failFirst ? 1 : 0};
    export default async function init() {
      calls.init++;
      await new Promise((done) => setTimeout(done, 20));
      if (failures-- > 0) throw new Error('fetch failed');
    }
    export class Engine { static create() { return 'engine'; } }
  `);
  return dir;
}

test('overlapping loadRaa calls instantiate the wasm module once', async () => {
  const dir = await stubbedPackage(false);
  try {
    const { loadRaa } = await import(pathToFileURL(join(dir, 'index.js')));
    const { calls } = await import(pathToFileURL(join(dir, 'pkg', 'raa_js.js')));
    const modules = await Promise.all([loadRaa(), loadRaa({ wasmUrl: 'x.wasm' }), loadRaa()]);
    await loadRaa();
    assert.equal(calls.init, 1);
    assert.equal(new Set(modules).size, 3);
    assert.equal(modules[0].createEngine(), 'engine');
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('a failed instantiation is retried by the next call', async () => {
  const dir = await stubbedPackage(true);
  try {
    const { loadRaa } = await import(pathToFileURL(join(dir, 'index.js')));
    const { calls } = await import(pathToFileURL(join(dir, 'pkg', 'raa_js.js')));
    const first = await Promise.allSettled([loadRaa(), loadRaa()]);
    assert.deepEqual(first.map((r) => r.status), ['rejected', 'rejected']);
    assert.equal(calls.init, 1);
    assert.equal((await loadRaa()).version, '0.5.0');
    assert.equal(calls.init, 2);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
