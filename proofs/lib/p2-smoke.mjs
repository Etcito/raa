// P2 project smoke: every committed lookdev record has the S14 shape, and every API tag in
// packages/raa/types/ is documented in docs/api/. Fast (no build, no GPU). Exit 1 on failure.
import { existsSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { apiDocCheck, RECORDS_DIR } from './p2.mjs';
import { check, readJson } from './proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const SOFTWARE_RE = /swiftshader|llvmpipe|lavapipe|software|basic render/i;
const ID_RE = /^[a-z0-9-]+$/;
const SCENE_RE = /^m\d{2}-[a-z0-9-]+$/;
const SHA_RE = /^[0-9a-f]{64}$/;
const JUDGE_OUTPUT_MAX = 4000;

const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const isStr = (v) => typeof v === 'string';
const isIso = (v) => isStr(v) && !Number.isNaN(Date.parse(v));
const isCount = (v) => Number.isInteger(v) && v >= 0;

// S14 shape violations of one record file named `<scene>--<variant>.json`.
function recordErrors(file, r) {
  if (!isObj(r)) return ['not an object'];
  const e = [];
  if (!isStr(r.scene) || !SCENE_RE.test(r.scene)) e.push('scene');
  if (!isStr(r.variant) || !ID_RE.test(r.variant)) e.push('variant');
  if (file !== `${r.scene}--${r.variant}.json`) e.push('file name != <scene>--<variant>.json');
  if (!isStr(r.milestone) || !/^m\d{2}$/.test(r.milestone)) e.push('milestone');
  else if (isStr(r.scene) && !r.scene.startsWith(`${r.milestone}-`)) e.push('milestone != scene prefix');
  if (!isIso(r.capturedAt)) e.push('capturedAt');
  const ad = r.adapter;
  const adOk = isObj(ad) && ['vendor', 'architecture', 'device', 'description'].every((k) => isStr(ad[k]));
  if (!adOk) e.push('adapter');
  else if (typeof ad.isFallbackAdapter !== 'boolean') e.push('adapter.isFallbackAdapter');
  if (typeof r.hardware !== 'boolean') e.push('hardware');
  else if (adOk) {
    const looksSoftware =
      ad.isFallbackAdapter === true ||
      [ad.vendor, ad.architecture, ad.device, ad.description].some((s) => SOFTWARE_RE.test(s));
    if (r.hardware === looksSoftware) e.push('hardware disagrees with adapter');
  }
  if (!Array.isArray(r.canvas) || r.canvas.length !== 2 || !r.canvas.every((n) => Number.isInteger(n) && n > 0)) {
    e.push('canvas');
  }
  if (typeof r.nonEmpty !== 'boolean') e.push('nonEmpty');
  if (!Array.isArray(r.passes) || !r.passes.every(isStr)) e.push('passes');
  for (const k of ['instances', 'triangles']) if (!isCount(r[k])) e.push(k);
  if (!(r.firstFrameMs === null || isNum(r.firstFrameMs))) e.push('firstFrameMs');
  const w = r.wasm;
  if (!isObj(w) || !isCount(w.raw) || !isCount(w.gzip) || !isStr(w.sha256) || !SHA_RE.test(w.sha256)) {
    e.push('wasm');
  }
  if (r.capture !== `proofs/captures/lookdev/${r.scene}--${r.variant}.png`) e.push('capture');
  if (!isStr(r.captureSha256) || !SHA_RE.test(r.captureSha256)) e.push('captureSha256');
  if (!('judge' in r)) e.push('judge missing (null when not judged)');
  else if (r.judge !== null) {
    const j = r.judge;
    if (!isObj(j)) e.push('judge');
    else {
      if (!isStr(j.description) || j.description.trim() === '') e.push('judge.description');
      if (!Array.isArray(j.references) || !j.references.every((x) => isStr(x) && /^ref-[1-4]$/.test(x))) {
        e.push('judge.references');
      }
      if (j.minScore !== 6) e.push('judge.minScore');
      if (!Number.isInteger(j.exit)) e.push('judge.exit');
      if (!(j.score === null || isNum(j.score))) e.push('judge.score');
      if (!isStr(j.output) || j.output.length > JUDGE_OUTPUT_MAX) e.push('judge.output');
      if (!isIso(j.at)) e.push('judge.at');
    }
  }
  return e;
}

await check('lookdev-records', () => {
  const dir = join(ROOT, RECORDS_DIR);
  const files = existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith('.json')).sort() : [];
  const bad = [];
  for (const f of files) {
    let r;
    try {
      r = readJson(join(dir, f));
    } catch (err) {
      bad.push(`${f}: invalid JSON (${err.message})`);
      continue;
    }
    const errs = recordErrors(f, r);
    if (errs.length) bad.push(`${f}: ${errs.join(',')}`);
  }
  return {
    measured: `${files.length - bad.length}/${files.length} records valid${bad.length ? `; ${bad.join('; ')}` : ''}`,
    threshold: `every ${RECORDS_DIR}/*.json matches S14`,
    ok: bad.length === 0,
  };
});

const typesDir = join(ROOT, 'packages/raa/types');
const tags = existsSync(typesDir)
  ? [...new Set(readdirSync(typesDir).map((f) => /^(m\d{2})-.*\.d\.ts$/.exec(f)?.[1]).filter(Boolean))].sort()
  : [];
await check('api-doc-tags', () => ({
  measured: `${tags.length} tag(s)${tags.length ? `: ${tags.join(',')}` : ''} in packages/raa/types/`,
  threshold: 'listed (each tag checked below)',
  ok: true,
}));
for (const tag of tags) await check(`api-doc ${tag}`, () => apiDocCheck(tag));
