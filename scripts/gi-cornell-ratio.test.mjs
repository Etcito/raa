// Tests of scripts/gi-cornell-ratio.mjs (contract cornell-ratio-record), without a browser or GPU.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';
import test from 'node:test';
import assert from 'node:assert/strict';
import { decodePng } from './trace-first-pixel.mjs';
import { patchLuminance, predictedRatios, projectPixel } from './gi-cornell-ratio.mjs';
import * as lighting from '../examples/global-illumination/scenes/lighting.js';

const SCRIPT = fileURLToPath(new URL('./gi-cornell-ratio.mjs', import.meta.url));
const CAMERA = { eye: [2.78, 2.73, -8], target: [2.78, 2.73, 0], fovYDeg: 39.3 };
const POINTS = { 'floor-front': [2.78, 0, 0.8], 'floor-front-clear': [4.2, 0, 0.8], 'back-wall': [2.78, 4.4, 5.592] };
const PATCH = 21;
const OFF_FILENAME = lighting.inventory.find(row => row.mode === 'cornell-box-off').filename;
const ROWS = ['cornell-box-linear-on', 'cornell-box-linear-off', 'cornell-box-on', 'cornell-box-off'];

const decodeSrgb = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
function srgbByte(linear) {
  const value = linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
  return Math.round(255 * value);
}

function chunk(name, data) {
  const body = Buffer.concat([Buffer.from(name), data]);
  let crc = 0xffffffff;
  for (const byte of body) { crc ^= byte; for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0); }
  const size = Buffer.alloc(4), sum = Buffer.alloc(4);
  size.writeUInt32BE(data.length); sum.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
  return Buffer.concat([size, body, sum]);
}

function encodePng(width, height, channels, at) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width); ihdr.writeUInt32BE(height, 4); ihdr[8] = 8; ihdr[9] = channels === 4 ? 6 : 2;
  const stride = width * channels, raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixel = at(x, y), base = y * (stride + 1) + 1 + x * channels;
      for (let c = 0; c < channels; c++) raw[base + c] = pixel[c];
    }
  }
  return Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

/** A 1280x720 RGBA scene with `backByte` on the back wall and `floorByte` on the two floor patches. */
function sceneImage(width, height, floorByte, backByte) {
  const half = Math.floor(PATCH / 2);
  const pixels = Object.fromEntries(Object.entries(POINTS).map(([name, point]) => [name, projectPixel(CAMERA, point, width, height)]));
  return encodePng(width, height, 4, (x, y) => {
    for (const [name, pixel] of Object.entries(pixels)) {
      if (Math.abs(x - pixel[0]) <= half && Math.abs(y - pixel[1]) <= half) {
        const value = name === 'back-wall' ? backByte : floorByte;
        return [value, value, value, 255];
      }
    }
    return [0, 0, 0, 255];
  });
}

function cli(args, env) {
  return spawnSync(process.execPath, [SCRIPT, ...args], {
    encoding: 'utf8', env: { ...process.env, ...env }, cwd: process.cwd(),
  });
}

function writeCurrent(root, attempt) {
  const directory = `attempts/${attempt}`;
  mkdirSync(join(root, directory, 'images'), { recursive: true });
  writeFileSync(join(root, 'current.json'), JSON.stringify({ selectionId: 'selection-1', attemptId: attempt, directory }));
  writeFileSync(join(root, 'selection.json'), JSON.stringify({ selectionId: 'selection-1', buildId: 'build-1', stage: 'lighting' }));
  return directory;
}

function writeRows(root, directory, bytes) {
  for (const mode of ROWS) {
    const filename = lighting.inventory.find(row => row.mode === mode).filename;
    writeFileSync(join(root, directory, 'images', filename), bytes);
  }
}

test('projectPixel frames the Cornell floor and back wall at 1280 x 720', () => {
  assert.deepEqual(projectPixel(CAMERA, POINTS['floor-front'], 1280, 720), [640, 673]);
  assert.deepEqual(projectPixel(CAMERA, POINTS['floor-front-clear'], 1280, 720), [477, 673]);
  assert.deepEqual(projectPixel(CAMERA, POINTS['back-wall'], 1280, 720), [640, 236]);
});

test('patchLuminance decodes sRGB', () => {
  const image = decodePng(encodePng(64, 64, 4, () => [128, 128, 128, 255]));
  const got = patchLuminance(image, [32, 32], PATCH);
  assert.ok(Math.abs(got - decodeSrgb(128 / 255)) < 1e-4, `got ${got}`);
});

test('mode current writes the record under the cache and under --out', () => {
  const root = mkdtempSync(join(tmpdir(), 'cornell-'));
  const out = join(root, 'out'), cache = join(root, 'cache');
  mkdirSync(out, { recursive: true });
  const attempt = '0f6b3c2e-5d1a-4e8b-9c7d-2a1b3c4d5e6f';
  const directory = writeCurrent(out, attempt);
  const backByte = 128;
  const floorByte = srgbByte(decodeSrgb(backByte / 255) / 2);
  const bytes = sceneImage(1280, 720, floorByte, backByte);
  writeRows(out, directory, bytes);
  const reference = join(out, 'cornell-box-reference.png');
  writeFileSync(reference, bytes);

  const first = cli(['--out', out, '--reference', reference], { NMC_CACHE_DIR: cache });
  assert.equal(first.status, 0, first.stderr);
  const cached = join(cache, 'global-illumination', 'cornell-ratio.json');
  assert.ok(existsSync(cached), 'record under NMC_CACHE_DIR');
  const record = JSON.parse(readFileSync(cached, 'utf8'));
  assert.equal(record.mode, 'current');
  assert.deepEqual(record.missing, []);
  for (const name of ['linear-on', 'linear-off']) {
    const image = record.images.find(entry => entry.name === name);
    assert.ok(image, name);
    assert.ok(Math.abs(image.ratio - 0.5) < 1e-2, `${name} ratio ${image.ratio}`);
  }
  for (const name of ['linear-on', 'linear-off', 'agx-on', 'agx-off', 'reference']) {
    assert.ok(record.images.some(entry => entry.name === name), name);
  }
  assert.match(first.stdout, /^cornell-ratio linear-on floor-front \S+ floor-front-clear \S+ back-wall \S+ ratio \S+ ratioClear \S+$/m);
  assert.match(first.stdout, /^cornell-ratio predicted karis \S+ \S+ formFactor \S+ \S+$/m);
  assert.match(first.stdout, /^cornell-ratio: ok \(.+cornell-ratio\.json\)$/m);

  const second = cli(['--out', out, '--reference', reference], { NMC_CACHE_DIR: '' });
  assert.equal(second.status, 0, second.stderr);
  assert.ok(existsSync(join(out, 'cornell-ratio.json')), 'record under --out');
});

test('mode current names a missing image and exits 1', () => {
  const root = mkdtempSync(join(tmpdir(), 'cornell-'));
  const out = join(root, 'out');
  mkdirSync(out, { recursive: true });
  const directory = writeCurrent(out, '0f6b3c2e-5d1a-4e8b-9c7d-2a1b3c4d5e6f');
  const bytes = sceneImage(1280, 720, 96, 128);
  const missingMode = 'cornell-box-linear-off';
  const missingFile = lighting.inventory.find(row => row.mode === missingMode).filename;
  for (const mode of ROWS.filter(mode => mode !== missingMode)) {
    writeFileSync(join(out, directory, 'images', lighting.inventory.find(row => row.mode === mode).filename), bytes);
  }
  const reference = join(out, 'cornell-box-reference.png');
  writeFileSync(reference, bytes);
  const result = cli(['--out', out, '--reference', reference], { NMC_CACHE_DIR: join(root, 'cache') });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /cornell-ratio: /);
  assert.ok(result.stderr.includes(missingFile), result.stderr);
});

test('predictedRatios puts the form factor above the representative point', () => {
  const light = {
    position: [lighting.CORNELL_LIGHT_X, lighting.CORNELL_LIGHT_Y, lighting.CORNELL_LIGHT_Z],
    direction: [0, -1, 0], size: lighting.CORNELL_LIGHT_SIZE, range: 30,
  };
  const predicted = predictedRatios(light);
  for (const model of [predicted.karis, predicted.formFactor]) {
    for (const value of [model.ratio, model.ratioClear]) {
      assert.ok(Number.isFinite(value) && value > 0, String(value));
    }
  }
  assert.ok(predicted.formFactor.ratio > predicted.karis.ratio, JSON.stringify(predicted));
});

test('mode stored uses the newest AgX off image and records the missing on row', () => {
  const root = mkdtempSync(join(tmpdir(), 'cornell-'));
  const cache = join(root, 'cache'), out = join(root, 'out');
  const store = join(cache, 'global-illumination', 'lighting');
  mkdirSync(out, { recursive: true });
  const oldId = '11111111-1111-4111-8111-111111111111';
  const newId = '22222222-2222-4222-8222-222222222222';
  const oldFile = join(store, 'attempts', oldId, 'images', OFF_FILENAME);
  const newFile = join(store, 'attempts', newId, 'images', OFF_FILENAME);
  for (const [id, file, backByte, buildId] of [[oldId, oldFile, 96, 'build-old'], [newId, newFile, 192, 'build-new']]) {
    mkdirSync(join(store, 'attempts', id, 'images'), { recursive: true });
    writeFileSync(file, sceneImage(1280, 720, srgbByte(decodeSrgb(backByte / 255) / 2), backByte));
    writeFileSync(join(store, 'attempts', id, 'build.json'), JSON.stringify({ buildId }));
  }
  utimesSync(oldFile, new Date(1_000_000), new Date(1_000_000));
  utimesSync(newFile, new Date(2_000_000), new Date(2_000_000));
  const reference = join(root, 'reference.png');
  writeFileSync(reference, sceneImage(1280, 720, 96, 160));

  const result = cli(['--stored', 'lighting', '--out', out, '--reference', reference], { NMC_CACHE_DIR: cache });
  assert.equal(result.status, 0, result.stderr);
  const recordFile = join(cache, 'global-illumination', 'cornell-ratio-stored.json');
  assert.ok(existsSync(recordFile));
  const record = JSON.parse(readFileSync(recordFile, 'utf8'));
  assert.equal(record.mode, 'stored');
  assert.equal(record.attemptId, newId);
  assert.equal(record.buildId, 'build-new');
  assert.deepEqual(record.missing, ['agx-on']);
  const off = record.images.find(entry => entry.name === 'agx-off');
  assert.ok(off.file.includes(newId), off.file);
  assert.ok(Math.abs(off.luminance['back-wall'] - decodeSrgb(192 / 255)) < 1e-6);
  assert.match(result.stdout, /^cornell-ratio missing agx-on$/m);
});
