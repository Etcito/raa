#!/usr/bin/env node
// Lights-page selection and visibility validation (S20, M2-PAGE-SELECTION): `select`
// reads only the requested page's produced build, validates every scene/variant record,
// checks each authored subject box against its camera and writes selection.json; `check`
// validates without writing. GPU transport and freshness checks belong to card 7263.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs as parse } from 'node:util';

const OPS = ['select', 'check'];
const USAGE = 'usage: node web/common/lights-job.mjs <select|check> --page dist/lights [--out out/lights]';
const ASPECT = 16 / 9;
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const vec3 = (v) => Array.isArray(v) && v.length === 3 && v.every(isNum);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const add = (a, b) => a.map((v, i) => v + b[i]);
const mul = (a, s) => a.map((v) => v * s);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const ensure = (ok, message) => { if (!ok) throw new Error(message); };
const unit = (a) => { const len = Math.hypot(...a); return len > 0 ? a.map((v) => v / len) : null; };

function readJson(path, hint) {
  ensure(existsSync(path), `${path} is missing: ${hint}`);
  try { return JSON.parse(readFileSync(path, 'utf8')); } catch (e) { throw new Error(`${path} is not valid JSON (${e.message}): ${hint}`); }
}

function parseArgs(argv) {
  const { values, positionals } = parse({ args: argv, allowPositionals: true,
    options: { page: { type: 'string' }, out: { type: 'string' } } });
  ensure(positionals.length === 1 && OPS.includes(positionals[0]) && values.page, USAGE);
  ensure(positionals[0] !== 'select' || values.out, USAGE);
  return { op: positionals[0], page: values.page, out: values.out };
}

/** True when `box` overlaps the camera's view volume (conservative AABB/frustum test). */
export function inView(camera, box) {
  const forward = unit(sub(camera.target, camera.eye));
  const right = forward && unit(cross(forward, [0, 1, 0]));
  if (!right) return false;
  const up = cross(right, forward);
  const tanY = Math.tan((camera.fovYDeg * Math.PI) / 360);
  const tanX = tanY * (camera.aspect > 0 ? camera.aspect : ASPECT);
  const planes = [
    [forward, add(camera.eye, mul(forward, camera.near))],
    [mul(forward, -1), add(camera.eye, mul(forward, camera.far))],
    [sub(mul(forward, tanX), right), camera.eye],
    [add(mul(forward, tanX), right), camera.eye],
    [sub(mul(forward, tanY), up), camera.eye],
    [add(mul(forward, tanY), up), camera.eye],
  ];
  const center = box.min.map((v, i) => (v + box.max[i]) / 2);
  const half = box.min.map((v, i) => (box.max[i] - v) / 2);
  return planes.every(([normal, point]) => dot(normal, center) - dot(normal, point)
    + half.reduce((sum, h, i) => sum + Math.abs(normal[i]) * h, 0) >= 0);
}

function cameraProblems(camera) {
  if (!camera || !vec3(camera.eye) || !vec3(camera.target)) return ['no camera'];
  const problems = [];
  if (!isNum(camera.fovYDeg) || camera.fovYDeg <= 0 || camera.fovYDeg >= 180 || !isNum(camera.near)
    || camera.near <= 0 || !isNum(camera.far) || camera.far <= camera.near) problems.push('camera frustum is invalid');
  if (camera.aspect !== undefined && (!isNum(camera.aspect) || camera.aspect <= 0)) problems.push('camera aspect is invalid');
  const forward = vec3(camera.eye) && vec3(camera.target) ? unit(sub(camera.target, camera.eye)) : null;
  if (!forward || !unit(cross(forward, [0, 1, 0]))) problems.push('camera direction is invalid');
  return problems;
}

export function variantProblems(variant) {
  const selection = variant?.selection;
  const problems = [];
  if (typeof variant?.description !== 'string' || !variant.description.trim()) problems.push('no description');
  if (!isNum(variant?.timeMs) || variant.timeMs < 0) problems.push('timeMs is not a duration');
  if (variant?.judge !== true) problems.push('judge must be true');
  const cameraErrors = cameraProblems(selection?.camera);
  problems.push(...cameraErrors);
  if (!Array.isArray(selection?.subjects) || selection.subjects.length === 0) return [...problems, 'selection needs at least one subject'];
  selection.subjects.forEach((box, i) => {
    if (!box || !vec3(box.min) || !vec3(box.max) || box.min.some((x, k) => x > box.max[k])) problems.push(`subjects[${i}] is not a finite min<=max box`);
    else if (cameraErrors.length === 0 && !inView(selection.camera, box)) problems.push(`subjects[${i}] is outside the camera view`);
  });
  return problems;
}

// The module of an index entry must live inside the requested page build.
function sceneModulePath(page, file) {
  const root = resolve(page);
  const path = resolve(root, file);
  const rel = relative(root, path);
  return rel && !rel.startsWith('..') && !isAbsolute(rel) && existsSync(path) ? path : null;
}

async function loadScenes(page) {
  const indexPath = join(page, 'scenes.json');
  const index = readJson(indexPath, 'build the page first');
  ensure(Array.isArray(index) && index.length > 0, `${indexPath} is not a nonempty array`);
  const scenes = new Map();
  for (const [i, entry] of index.entries()) {
    ensure(entry && typeof entry.id === 'string' && entry.id.trim() && typeof entry.milestone === 'string'
      && typeof entry.file === 'string', `${indexPath} entry ${i} is not { id, milestone, file }`);
    const file = sceneModulePath(page, entry.file);
    ensure(file, `scene ${entry.id}: module ${entry.file} is not in the page build`);
    let def;
    try { def = (await import(pathToFileURL(file).href)).default; } catch (e) { throw new Error(`scene ${entry.id}: import failed (${e.message})`); }
    ensure(def?.id === entry.id && def.milestone === entry.milestone && typeof def.build === 'function', `scene ${entry.id}: metadata is invalid`);
    ensure(Array.isArray(def.variants) && def.variants.length > 0, `scene ${entry.id}: no variants`);
    const ids = def.variants.map((v) => v?.id);
    ids.forEach((id) => ensure(id === 'on' || id === 'off', `scene ${entry.id}: variant ${id} is not on or off`));
    ensure(new Set(ids).size === ids.length, `scene ${entry.id}: duplicate variant`);
    ensure(!scenes.has(def.id), `scene ${entry.id}: duplicate scene`);
    scenes.set(def.id, def);
  }
  return scenes;
}

function collect(scenes) {
  const pairs = [];
  const problems = [];
  for (const def of scenes.values()) for (const variant of def.variants) {
    const invalid = variantProblems(variant).map((p) => `${def.id}--${variant.id}: ${p}`);
    if (invalid.length > 0) problems.push(...invalid);
    else pairs.push({ scene: def.id, variant: variant.id, description: variant.description, timeMs: variant.timeMs });
  }
  ensure(problems.length === 0, `invalid selection:\n  ${problems.join('\n  ')}`);
  ensure(pairs.length > 0, 'selection is empty');
  return pairs;
}

export async function check(page) {
  return collect(await loadScenes(page));
}

// The selection records its produced build so later commands cannot run a stale one.
export async function select(page, out) {
  const pairs = await check(page);
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, 'selection.json'), `${JSON.stringify({ page: resolve(page), pairs }, null, 2)}\n`);
  return pairs;
}

export async function main(argv = process.argv.slice(2)) {
  const { op, page, out } = parseArgs(argv);
  const pairs = op === 'check' ? await check(page) : await select(page, out);
  process.stdout.write(`lights-job: ${op} ${pairs.length} pairs\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    process.stderr.write(`lights-job: ${error?.message ?? error}\n`);
    process.exit(1);
  });
}
