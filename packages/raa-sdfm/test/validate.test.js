import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateAsset, checkHash } from '../src/validate.js';
import { documentHash } from '../src/canonical.js';
import { SdfmError } from '../src/errors.js';

const fixture = () => JSON.parse(readFileSync(new URL('./fixtures/valid.asset.json', import.meta.url), 'utf8'));

function expectError(asset, code, path) {
  const r = validateAsset(asset);
  assert.equal(r.ok, false);
  for (const e of r.errors) assert.ok(typeof e.message === 'string' && e.message.length > 0);
  const hit = r.errors.find((e) => e.code === code && (path === undefined || e.path === path));
  assert.ok(hit, `expected ${code} at ${path}, got ${JSON.stringify(r.errors)}`);
  return r;
}

test('fixture validates and its hash matches', async () => {
  const a = fixture();
  assert.deepEqual(validateAsset(a), { ok: true, errors: [] });
  await checkHash(a);
});

test('non-object root is sdfm.schema at the root', () => {
  expectError([], 'sdfm.schema', '');
});

test('wrong format is sdfm.format', () => {
  expectError({ ...fixture(), format: 'raa.prims' }, 'sdfm.format', '/format');
});

test('wrong version is sdfm.version', () => {
  expectError({ ...fixture(), version: 2 }, 'sdfm.version', '/version');
});

test('missing key is sdfm.schema', () => {
  const a = fixture();
  delete a.bounds;
  expectError(a, 'sdfm.schema', '/bounds');
});

test('unknown key is sdfm.schema', () => {
  expectError({ ...fixture(), extra: 1 }, 'sdfm.schema', '/extra');
});

test('wrong key order is sdfm.schema', () => {
  const { format, version, ...rest } = fixture();
  expectError({ version, format, ...rest }, 'sdfm.schema', '/version');
});

test('wrong type is sdfm.schema with its pointer', () => {
  const a = fixture();
  a.anchors.top.t = [0, 1];
  expectError(a, 'sdfm.schema', '/anchors/top/t');
  expectError({ ...fixture(), name: 3 }, 'sdfm.schema', '/name');
  expectError({ ...fixture(), hash: 'ABC' }, 'sdfm.schema', '/hash');
});

test('NaN anywhere is sdfm.not_finite with its pointer', () => {
  const a = fixture();
  a.document.root.halfExtents[1] = NaN;
  expectError(a, 'sdfm.not_finite', '/document/root/halfExtents/1');
});

test('bad id is sdfm.id', () => {
  expectError({ ...fixture(), id: 'Crate' }, 'sdfm.id', '/id');
});

test('document over 2 MB is sdfm.cap_document', () => {
  const a = fixture();
  a.document = { blob: 'x'.repeat(2 * 1024 * 1024) };
  expectError(a, 'sdfm.cap_document', '/document');
});

test('tampered document is sdfm.hash_mismatch at /hash', async () => {
  const a = fixture();
  a.document.root.halfExtents[0] = 0.75;
  assert.notEqual(await documentHash(a.document), a.hash);
  await assert.rejects(checkHash(a), (e) => {
    assert.ok(e instanceof SdfmError);
    assert.equal(e.code, 'sdfm.hash_mismatch');
    assert.equal(e.path, '/hash');
    assert.ok(e.message.length > 0);
    return true;
  });
});
