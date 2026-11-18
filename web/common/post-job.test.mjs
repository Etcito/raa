import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs, report, storeRoot, STAGES, validators } from './post-job.mjs';

delete process.env.NMC_CACHE_DIR;
delete process.env.POST_RERENDER_ALL;

const USAGE = 'usage: node web/common/post-job.mjs <check|select|clean|render|fresh|judge> --page dist/post --out out/post/<stage> --stage <exposure|sharpen|mips>';

function scratch(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'post-job-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function put(root, name, text) {
  const file = path.join(root, name);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, text);
  return file;
}

test('STAGES follows the post-page stages in rank order', () => {
  assert.deepEqual(STAGES, ['exposure', 'sharpen', 'mips']);
});

test('parseArgs refuses an unknown action with the post-job usage line', () => {
  assert.throws(() => parseArgs(['nope']), (error) => error.message.includes(USAGE));
});

test('storeRoot is the out directory without NMC_CACHE_DIR and $NMC_CACHE_DIR/post/<stage> with it', (t) => {
  delete process.env.NMC_CACHE_DIR;
  assert.equal(storeRoot('out/post/exposure', 'exposure'), path.resolve('out/post/exposure'));
  process.env.NMC_CACHE_DIR = '/cache';
  t.after(() => { delete process.env.NMC_CACHE_DIR; });
  assert.equal(storeRoot('out/post/exposure', 'exposure'), path.join(path.resolve('/cache'), 'post', 'exposure'));
});

test('report prints the rows of check and the kept and rendered counts of render', () => {
  assert.deepEqual(report('check', 'exposure', {}, [{ id: 'exposure-interior-on', filename: 'exposure-interior-on-room-readable.png' }]), [
    'exposure-interior-on exposure-interior-on-room-readable.png',
    'post-job check: ok (exposure, 1 rows)',
  ]);
  assert.deepEqual(report('render', 'exposure', { kept: ['a'], images: ['a', 'b', 'c'] }, [{ id: 'a' }, { id: 'b' }, { id: 'c' }]), [
    'post-job render: kept 1, rendered 2',
    'post-job render: ok (exposure, 3 rows)',
  ]);
});

test('validators returns both functions of a built page and refuses a module that exports one', async (t) => {
  const page = scratch(t);
  put(page, 'lib/inventory.js', 'export function validateInventory() { return []; }\nexport function validateRequiredFacts() {}\n');
  const namespace = await import(pathToFileURL(path.join(page, 'lib/inventory.js')).href);
  const loaded = await validators(page);
  assert.equal(loaded.validateInventory, namespace.validateInventory);
  assert.equal(loaded.validateRequiredFacts, namespace.validateRequiredFacts);

  const partial = scratch(t);
  put(partial, 'lib/inventory.js', 'export function validateInventory() { return []; }\n');
  const file = path.join(path.resolve(partial), 'lib/inventory.js');
  await assert.rejects(validators(partial),
    (error) => error.message === `${file}: validateInventory and validateRequiredFacts must be exported`);
});
