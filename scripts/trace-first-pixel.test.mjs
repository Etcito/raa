// node --test scripts/trace-first-pixel.test.mjs: the first-pixel scene and capture checks,
// without a browser or GPU.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SCENE, programBytes, programWords } from '../examples/trace-first-pixel/scene.js';
import { pngSize } from './trace-first-pixel.mjs';

test('the box and sphere pack as a TraceProgram v1 with two sources and two instances', () => {
  const w = programWords();
  assert.equal(w[0], 0x50544453);
  assert.equal(w[1], 1);
  const [sources, instances, code, data, materials] = [w[2], w[3], w[4], w[5], w[6]];
  assert.deepEqual([sources, instances, code, data, materials], [2, 2, 2, 33, 2]);
  assert.equal(w.length, 8 + sources * 12 + code + data + materials * 8 + instances * 32);
  assert.deepEqual(Array.from(w.subarray(8 + 24, 8 + 26)), [1, 2]); // BOX, SPHERE
});

test('the program is deterministic and follows the scene parameters', () => {
  assert.deepEqual(programBytes(), programBytes());
  const moved = { ...SCENE, sphere: { ...SCENE.sphere, radius: 0.5 } };
  assert.notDeepEqual(programBytes(moved), programBytes());
});

test('pngSize reads the IHDR dimensions and refuses other data', () => {
  const png = Buffer.alloc(24);
  Buffer.from('89504e470d0a1a0a', 'hex').copy(png, 0);
  png.write('IHDR', 12, 'ascii');
  png.writeUInt32BE(1280, 16);
  png.writeUInt32BE(720, 20);
  assert.deepEqual(pngSize(png), [1280, 720]);
  assert.equal(pngSize(Buffer.from('not a png at all, really')), null);
});
