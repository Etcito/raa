import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { canonicalJson, documentHash, sha256Bytes, sha256Hex } from '../src/canonical.js';
import { SdfmError } from '../src/errors.js';

const nodeSha256 = (text) => createHash('sha256').update(text, 'utf8').digest('hex');

test('sha256Hex matches the known digest of "abc" and node:crypto', async () => {
  assert.equal(
    await sha256Hex('abc'),
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
  );
  const text = 'sdfm.game-asset canonical text ✓';
  assert.equal(await sha256Hex(text), nodeSha256(text));
});

test('sha256Bytes (no crypto.subtle) matches node:crypto across block boundaries', () => {
  const hex = (bytes) => Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  for (const length of [0, 1, 55, 56, 63, 64, 65, 119, 1000]) {
    const text = 'é'.repeat(length >> 1) + 'x'.repeat(length & 1);
    assert.equal(hex(sha256Bytes(new TextEncoder().encode(text))), nodeSha256(text), `length ${length}`);
  }
});

test('sha256Hex falls back to JS where crypto.subtle is missing (opaque-origin worker)', async () => {
  const subtle = Object.getOwnPropertyDescriptor(globalThis.crypto, 'subtle')
    ?? Object.getOwnPropertyDescriptor(Object.getPrototypeOf(globalThis.crypto), 'subtle');
  Object.defineProperty(globalThis.crypto, 'subtle', { value: undefined, configurable: true });
  try {
    assert.equal(await sha256Hex('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  } finally {
    delete globalThis.crypto.subtle;
    if (!('subtle' in globalThis.crypto)) Object.defineProperty(globalThis.crypto, 'subtle', subtle);
  }
});

test('canonicalJson uses two-space indent, LF and a trailing newline', () => {
  assert.equal(canonicalJson({ a: 1, b: [true, null] }), '{\n  "a": 1,\n  "b": [\n    true,\n    null\n  ]\n}\n');
});

test('canonicalJson writes integers without a fraction and -0 as 0', () => {
  assert.equal(canonicalJson(3), '3\n');
  assert.equal(canonicalJson(2.0), '2\n');
  assert.equal(canonicalJson(-0), '0\n');
  assert.equal(canonicalJson([-0, 100, 9007199254740992]), '[\n  0,\n  100,\n  9007199254740992\n]\n');
});

test('canonicalJson uses JS Number-to-string outside the plain range', () => {
  assert.equal(canonicalJson([0.5, 1e-6, 1e-7, 1e21]), '[\n  0.5,\n  0.000001,\n  1e-7,\n  1e+21\n]\n');
});

test('canonicalJson keeps object keys in insertion order', () => {
  assert.equal(canonicalJson({ z: 1, a: { y: 2, b: 3 } }), '{\n  "z": 1,\n  "a": {\n    "y": 2,\n    "b": 3\n  }\n}\n');
});

test('canonicalJson of canonical text round-trips byte-identical', () => {
  const text = canonicalJson({ format: 'sdfm.game-asset', version: 1, nodes: [{ id: 'a', r: 0.5, s: -0 }] });
  assert.equal(canonicalJson(JSON.parse(text)), text);
});

test('canonicalJson refuses NaN and infinities with sdfm.not_finite and the JSON pointer', () => {
  for (const bad of [NaN, Infinity, -Infinity]) {
    assert.throws(
      () => canonicalJson({ parts: [{ name: 'a' }, { name: 'b', t: [0, bad, 0] }] }),
      (err) =>
        err instanceof SdfmError &&
        err.code === 'sdfm.not_finite' &&
        err.path === '/parts/1/t/1' &&
        err.message.startsWith('sdfm.not_finite: '),
    );
  }
});

test('canonicalJson escapes JSON pointer tokens in the path of a non-finite number', () => {
  assert.throws(
    () => canonicalJson({ 'a/b~c': { x: NaN } }),
    (err) => err.code === 'sdfm.not_finite' && err.path === '/a~1b~0c/x',
  );
});

test('canonicalJson refuses a non-finite root with the empty pointer', () => {
  assert.throws(
    () => canonicalJson(Number.NaN),
    (err) => err.code === 'sdfm.not_finite' && err.path === '',
  );
});

test('canonicalJson refuses values that are not JSON with sdfm.schema', () => {
  assert.throws(
    () => canonicalJson({ a: [1, undefined] }),
    (err) => err instanceof SdfmError && err.code === 'sdfm.schema' && err.path === '/a/1',
  );
  assert.throws(() => canonicalJson(new Date(0)), (err) => err.code === 'sdfm.schema');
});

test('documentHash is the sha256 of the canonical text and depends on key order', async () => {
  const document = { name: 'chair', parts: [{ name: 'seat', t: [0, 0.5, 0] }] };
  assert.equal(await documentHash(document), nodeSha256(canonicalJson(document)));
  assert.notEqual(await documentHash({ a: 1, b: 2 }), await documentHash({ b: 2, a: 1 }));
});
