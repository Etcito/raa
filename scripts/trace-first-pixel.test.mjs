// node --test scripts/trace-first-pixel.test.mjs: the first-pixel scene and capture checks,
// without a browser or GPU.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { OCCLUSION, SCENE, programBytes, programWords } from '../examples/trace-first-pixel/scene.js';
import { deflateSync } from 'node:zlib';
import {
  checkCapture, checkOcclusion, decodePng, meshCoverage, pngSize, projectPixel,
} from './trace-first-pixel.mjs';

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

// 8-bit RGB PNG with filter 0 rows, enough for the capture check tests.
function encodePng(width, height, at) {
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    Buffer.from(at(x, y)).copy(raw, y * (width * 3 + 1) + 1 + x * 3);
  }
  const chunk = (kind, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    return Buffer.concat([len, Buffer.from(kind, 'ascii'), data, Buffer.alloc(4)]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

function frame({ background = [5, 6, 9], speckle = 0, slab = false } = {}) {
  const [w, h] = [320, 180];
  const box = projectPixel(SCENE, SCENE.box.center, w, h);
  const sphere = projectPixel(SCENE, SCENE.sphere.center, w, h);
  const slabPixels = [OCCLUSION.mesh.center, OCCLUSION.hidden].map((p) => projectPixel(SCENE, p, w, h));
  return decodePng(encodePng(w, h, (x, y) => {
    const near = ([cx, cy]) => Math.abs(x - cx) < 25 && Math.abs(y - cy) < 25;
    const dark = speckle && (x * 7 + y * 13) % speckle === 0;
    if (slab && slabPixels.some(([cx, cy]) => Math.abs(x - cx) <= 1 && Math.abs(y - cy) <= 1)) return [51, 191, 77];
    if (near(box)) return dark ? [42, 13, 12] : [216, 63, 49];
    if (near(sphere)) return dark ? [10, 24, 56] : [105, 230, 255];
    return background;
  }));
}

test('the capture check accepts both clean shapes on a dark background', () => {
  assert.deepEqual(checkCapture(frame(), SCENE), []);
});

test('the capture check refuses a light background and speckled surfaces', () => {
  assert.equal(checkCapture(frame({ background: [189, 183, 176] }), SCENE).filter((p) => /not dark/.test(p)).length, 4);
  const speckled = checkCapture(frame({ speckle: 9 }), SCENE);
  assert.ok(speckled.some((p) => /box surface has/.test(p)) && speckled.some((p) => /sphere surface has/.test(p)), speckled.join('; '));
});

test('the capture check refuses swapped shapes', () => {
  const swapped = { ...SCENE, box: { ...SCENE.box, center: SCENE.sphere.center }, sphere: { ...SCENE.sphere, center: SCENE.box.center } };
  assert.equal(checkCapture(frame(), swapped).filter((p) => /not (red|blue)/.test(p)).length, 2);
});

test('the occlusion check accepts a green slab over the box surface behind it', () => {
  assert.deepEqual(checkOcclusion(frame({ slab: true }), SCENE, OCCLUSION), []);
});

test('the occlusion check refuses a capture where the trace wins the hidden pixel', () => {
  assert.ok(checkOcclusion(frame(), SCENE, OCCLUSION).filter((p) => /not the green mesh/.test(p)).length === 2);
});

test('the occlusion preflight keeps the slab over the hidden surface and off the box centre', () => {
  assert.deepEqual(meshCoverage(SCENE, OCCLUSION), { hidden: true, centre: false });
  const away = { ...OCCLUSION, mesh: { ...OCCLUSION.mesh, center: [100, 100, 100] } };
  assert.deepEqual(meshCoverage(SCENE, away), { hidden: false, centre: false });
});
