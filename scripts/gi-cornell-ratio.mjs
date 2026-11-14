#!/usr/bin/env node
// Cornell floor-front / back-wall ratio of the lighting stage (contract cornell-ratio-record).
//
//   node scripts/gi-cornell-ratio.mjs --out <dir> --reference <png> [--stored <stage>]
//
// Mode current reads <dir>/current.json, <dir>/selection.json and the five images of the current
// attempt. Mode stored reads the newest AgX Cornell images of the GI page's resumable store. Neither
// mode starts a GPU job or builds anything. The record is written to $NMC_CACHE_DIR/global-illumination/
// cornell-ratio[-stored].json, or under --out when NMC_CACHE_DIR is unset or blank.
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePng } from './trace-first-pixel.mjs';
import * as lighting from '../examples/global-illumination/scenes/lighting.js';

const PATCH_PX = 21;
const POINTS = {
  'floor-front': [2.78, 0, 0.8],
  'floor-front-clear': [4.2, 0, 0.8],
  'back-wall': [2.78, 4.4, 5.592],
};
const NORMALS = { 'floor-front': [0, 1, 0], 'floor-front-clear': [0, 1, 0], 'back-wall': [0, 0, -1] };
const IMAGES = [
  { name: 'linear-on', mode: 'cornell-box-linear-on', encoding: 'linear', linear: true },
  { name: 'linear-off', mode: 'cornell-box-linear-off', encoding: 'linear', linear: true },
  { name: 'agx-on', mode: 'cornell-box-on', encoding: 'display', linear: false },
  { name: 'agx-off', mode: 'cornell-box-off', encoding: 'display', linear: false },
];
const DEFAULT_LIGHT = {
  position: [lighting.CORNELL_LIGHT_X, lighting.CORNELL_LIGHT_Y, lighting.CORNELL_LIGHT_Z],
  direction: [0, -1, 0],
  size: lighting.CORNELL_LIGHT_SIZE,
  range: 30,
};

const sub = (a, b) => a.map((v, i) => v - b[i]);
const add = (a, b) => a.map((v, i) => v + b[i]);
const scale = (v, s) => v.map(x => x * s);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const lengthOf = v => Math.hypot(...v);
const unit = v => { const n = lengthOf(v); return n > 1e-12 ? v.map(x => x / n) : [0, 0, 0]; };
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const nonblank = value => typeof value === 'string' && value.trim().length > 0;

/** sRGB OETF inverse of the 8-bit stored value. */
function decodeSrgb(value) {
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

/**
 * Pixel of a world point under a camera, following the convention of examples/global-illumination/
 * lib/inventory.js `selection`: forward = normalize(target - eye), right = normalize(cross(forward, Y)),
 * up = cross(right, forward), and x, y, z are the dot products of (point - eye) with right, up and forward.
 */
export function projectPixel(camera, point, width, height) {
  const forward = unit(sub(camera.target, camera.eye));
  const right = unit(cross(forward, [0, 1, 0]));
  const up = cross(right, forward);
  const d = sub(point, camera.eye);
  const x = dot(d, right), y = dot(d, up), z = dot(d, forward);
  const tan = Math.tan(camera.fovYDeg * Math.PI / 360);
  const aspect = width / height;
  const px = Math.round((1 + x / (z * tan * aspect)) / 2 * width);
  const py = Math.round((1 - y / (z * tan)) / 2 * height);
  return [px, py];
}

/** Mean Rec. 709 luminance of the `size` x `size` patch centred on `pixel`, sRGB-decoded and clipped to the image. */
export function patchLuminance(image, pixel, size) {
  const half = Math.floor(size / 2);
  let sum = 0, count = 0;
  for (let y = pixel[1] - half; y <= pixel[1] + half; y++) {
    if (y < 0 || y >= image.height) continue;
    for (let x = pixel[0] - half; x <= pixel[0] + half; x++) {
      if (x < 0 || x >= image.width) continue;
      const at = (y * image.width + x) * image.channels;
      const r = decodeSrgb(image.data[at] / 255);
      const g = decodeSrgb(image.data[at + 1] / 255);
      const b = decodeSrgb(image.data[at + 2] / 255);
      sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
      count++;
    }
  }
  return count ? sum / count : 0;
}

/** Karis representative-point irradiance of one world point (S11 area shading before LTC). */
function karisIrradiance(light, point, normal) {
  const direction = unit(light.direction);
  const v = sub(point, light.position);
  if (dot(v, direction) <= 0) return 0;
  const up0 = Math.abs(dot(direction, [0, 1, 0])) < 0.99 ? [0, 1, 0] : [1, 0, 0];
  const tangent = unit(cross(up0, direction));
  const bitangent = cross(direction, tangent);
  const representative = add(light.position, add(
    scale(tangent, clamp(dot(v, tangent), -light.size[0] / 2, light.size[0] / 2)),
    scale(bitangent, clamp(dot(v, bitangent), -light.size[1] / 2, light.size[1] / 2)),
  ));
  const toLight = sub(representative, point);
  const distance = lengthOf(toLight);
  if (distance <= 0) return 0;
  const l = scale(toLight, 1 / distance);
  const window = clamp(1 - (distance / light.range) ** 4, 0, 1);
  return window ** 2 / Math.max(distance * distance, 0.04) * Math.max(dot(normal, l), 0);
}

/** Rectangle corners of contract area-form-factor-wgsl for one origin (counter-clockwise from the emitting side). */
function areaCorners(light, origin) {
  const direction = unit(light.direction);
  const up0 = Math.abs(dot(direction, [0, 1, 0])) >= 0.99 ? [1, 0, 0] : [0, 1, 0];
  const tangent = unit(cross(up0, direction));
  const bitangent = cross(direction, tangent);
  const t = scale(tangent, Math.max(light.size[0], 0) * 0.5);
  const b = scale(bitangent, Math.max(light.size[1], 0) * 0.5);
  const p = sub(light.position, origin);
  return [sub(sub(p, t), b), add(sub(p, t), b), add(add(p, t), b), sub(add(p, t), b)];
}

/** three.js r180 LTC_EdgeVectorFormFactor. */
function edgeVectorFormFactor(a, b) {
  const x = dot(a, b), y = Math.abs(x);
  const v = (0.8543985 + (0.4965155 + 0.0145206 * y) * y) / (3.4175940 + (4.1616724 + y) * y);
  const theta = x > 0 ? v : 0.5 / Math.sqrt(Math.max(1 - x * x, 1e-7)) - v;
  return scale(cross(a, b), theta);
}

/** three.js LTC_ClippedSphereFormFactor, the horizon clipping. */
const clippedSphereFormFactor = (lengthF, z) => Math.max((lengthF * lengthF + z) / (lengthF + 1), 0);

/** Area form factor of contract area-form-factor-wgsl, in f64. */
function areaFormFactor(light, origin, normal) {
  if (dot(sub(origin, light.position), unit(light.direction)) <= 0) return 0;
  const corners = areaCorners(light, origin).map(unit);
  let f = [0, 0, 0];
  for (let i = 0; i < 4; i++) f = add(f, edgeVectorFormFactor(corners[i], corners[(i + 1) % 4]));
  return clippedSphereFormFactor(lengthOf(f), dot(f, normal));
}

/**
 * Predicted floor-front / back-wall ratios of the Karis representative point and of the area form
 * factor, both evaluated in f64 on the Cornell light. The form factor carries the emitter cosine that
 * the representative-point model lacks, so it lowers the back wall relative to the floor.
 * no handoff: Cornell ratio cause fix and exposure re-derivation for the form-factor model takes nothing from this card. It reads the CORNELL_* exports of lighting.js under contract cornell-linear-probe-rows.
 */
export function predictedRatios(light) {
  const ratios = model => {
    const wall = model('back-wall');
    return { ratio: wall === 0 ? 0 : model('floor-front') / wall, ratioClear: wall === 0 ? 0 : model('floor-front-clear') / wall };
  };
  return {
    karis: ratios(name => karisIrradiance(light, POINTS[name], NORMALS[name])),
    formFactor: ratios(name => Math.PI * areaFormFactor(light, POINTS[name], NORMALS[name])),
  };
}

function readJson(file) {
  try { return JSON.parse(readFileSync(file, 'utf8')); }
  catch { throw new Error(`cannot read ${file}`); }
}

function inventoryFilename(mode) {
  const row = lighting.inventory.find(entry => entry.mode === mode);
  if (!row) throw new Error(`inventory has no ${mode} row`);
  return row.filename;
}

function cameraOf() {
  const variant = lighting.default.variants.find(entry => entry.id === 'cornell-box-linear-off');
  if (!variant) throw new Error('lighting.js has no cornell-box-linear-off variant');
  return variant.selection.camera;
}

function measureImage(name, file, encoding, camera) {
  let image;
  try { image = decodePng(readFileSync(file)); }
  catch { throw new Error(`not an 8-bit RGB/RGBA PNG: ${file}`); }
  const pixels = {}, luminance = {};
  for (const [point, value] of Object.entries(POINTS)) pixels[point] = projectPixel(camera, value, image.width, image.height);
  for (const [point, pixel] of Object.entries(pixels)) luminance[point] = patchLuminance(image, pixel, PATCH_PX);
  if (luminance['back-wall'] === 0) throw new Error(`back-wall patch is 0 in ${file}`);
  return {
    name, file, encoding, width: image.width, height: image.height, pixels, luminance,
    ratio: luminance['floor-front'] / luminance['back-wall'],
    ratioClear: luminance['floor-front-clear'] / luminance['back-wall'],
  };
}

function sortedAttempts(storeRoot) {
  const attempts = join(storeRoot, 'attempts');
  if (!existsSync(attempts)) return [];
  return readdirSync(attempts, { withFileTypes: true })
    .filter(entry => entry.isDirectory())
    .map(entry => entry.name)
    .sort();
}

/** Newest candidate by mtime; a tie goes to the later name in ascending order. */
function newestCandidate(storeRoot, filename) {
  let best = null;
  for (const attempt of sortedAttempts(storeRoot)) {
    const file = join(storeRoot, 'attempts', attempt, 'images', filename);
    if (!existsSync(file)) continue;
    let mtimeMs;
    try { mtimeMs = statSync(file).mtimeMs; }
    catch { continue; }
    if (!best || mtimeMs > best.mtimeMs || (mtimeMs === best.mtimeMs && attempt > best.attempt)) best = { attempt, file, mtimeMs };
  }
  return best;
}

function recordPath(mode, out, env) {
  const name = mode === 'stored' ? 'cornell-ratio-stored.json' : 'cornell-ratio.json';
  return nonblank(env.NMC_CACHE_DIR)
    ? resolve(env.NMC_CACHE_DIR, 'global-illumination', name)
    : resolve(out, name);
}

function format(value) {
  return Number(value).toPrecision(4);
}

function printRecord(record, file) {
  for (const image of record.images) {
    console.log(`cornell-ratio ${image.name} floor-front ${format(image.luminance['floor-front'])} ` +
      `floor-front-clear ${format(image.luminance['floor-front-clear'])} back-wall ${format(image.luminance['back-wall'])} ` +
      `ratio ${format(image.ratio)} ratioClear ${format(image.ratioClear)}`);
  }
  console.log(`cornell-ratio predicted karis ${format(record.predicted.karis.ratio)} ${format(record.predicted.karis.ratioClear)} ` +
    `formFactor ${format(record.predicted.formFactor.ratio)} ${format(record.predicted.formFactor.ratioClear)}`);
  if (record.mode === 'stored') console.log(`cornell-ratio missing ${record.missing.length ? record.missing.join(',') : 'none'}`);
  console.log(`cornell-ratio: ok (${file})`);
}

function parseArgs(argv) {
  const valueOf = flag => {
    const at = argv.indexOf(flag);
    return at < 0 ? undefined : argv[at + 1];
  };
  const out = valueOf('--out'), reference = valueOf('--reference'), stored = valueOf('--stored');
  if (!nonblank(out)) throw new Error('usage: node scripts/gi-cornell-ratio.mjs --out <dir> --reference <png> [--stored <stage>]');
  if (!nonblank(reference)) throw new Error('missing --reference <png>');
  if (argv.includes('--stored') && !nonblank(stored)) throw new Error('missing --stored <stage>');
  return { out, reference, stored: argv.includes('--stored') ? stored : null };
}

/**
 * Run one mode of contract cornell-ratio-record and write its record.
 * @param {string[]} argv
 * @param {{NMC_CACHE_DIR?: string}} env
 */
export async function run(argv, env) {
  const { out, reference, stored } = parseArgs(argv);
  const mode = stored === null ? 'current' : 'stored';
  const camera = cameraOf();
  const referenceImage = measureImage('reference', reference, 'display', camera);
  let attemptId, buildId, images, missing;

  if (mode === 'current') {
    const currentFile = join(out, 'current.json');
    if (!existsSync(currentFile)) throw new Error(`missing ${currentFile}`);
    const current = readJson(currentFile);
    const attemptIdValue = current.attemptId;
    const directory = current.directory;
    const selectionFile = join(out, 'selection.json');
    if (!existsSync(selectionFile)) throw new Error(`missing ${selectionFile}`);
    const selection = readJson(selectionFile);
    attemptId = attemptIdValue;
    buildId = typeof selection.buildId === 'string' ? selection.buildId : null;
    images = [];
    for (const row of IMAGES) {
      const file = join(out, directory, 'images', inventoryFilename(row.mode));
      if (!existsSync(file)) throw new Error(`missing image ${file}`);
      images.push(measureImage(row.name, file, row.encoding, camera));
    }
    images.push(referenceImage);
    missing = [];
  } else {
    const storeRoot = nonblank(env.NMC_CACHE_DIR)
      ? resolve(env.NMC_CACHE_DIR, 'global-illumination', stored)
      : resolve('out/global-illumination', stored);
    const found = {};
    missing = [];
    for (const row of IMAGES.filter(entry => !entry.linear)) {
      const candidate = newestCandidate(storeRoot, inventoryFilename(row.mode));
      if (candidate) found[row.name] = { row, candidate };
      else missing.push(row.name);
    }
    const off = found['agx-off'], on = found['agx-on'];
    attemptId = off ? off.candidate.attempt : on ? on.candidate.attempt : '';
    buildId = null;
    if (attemptId) {
      try { buildId = readJson(join(storeRoot, 'attempts', attemptId, 'build.json')).buildId ?? null; }
      catch { buildId = null; }
    }
    images = [];
    for (const row of IMAGES.filter(entry => !entry.linear)) {
      const hit = found[row.name];
      if (hit) images.push(measureImage(row.name, hit.candidate.file, row.encoding, camera));
    }
    images.push(referenceImage);
  }

  const record = {
    mode, attemptId, buildId,
    camera: { eye: camera.eye, target: camera.target, fovYDeg: camera.fovYDeg },
    points: Object.fromEntries(Object.entries(POINTS).map(([name, point]) => [name, point])),
    patchPx: PATCH_PX,
    images,
    missing,
    predicted: predictedRatios(DEFAULT_LIGHT),
  };
  const file = recordPath(mode, out, env);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(record, null, 2) + '\n');
  printRecord(record, file);
  return record;
}

async function main() {
  try {
    await run(process.argv.slice(2), process.env);
  } catch (error) {
    console.error(`cornell-ratio: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
