// Shared helpers for milestone proofs (proofs/mN-<name>.mjs). Node built-ins only.
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

// Values decided once (ARCHITECTURE.md, Conventions, "Values decided once").
export const VALUES = Object.freeze({
  WASM_GZIP_MAX: 300000,
  FIRST_FRAME_MAX_MS: 500,
  CPU_MS_MEDIAN_MAX: 4.0,
  ALLOC_RUN: Object.freeze({ instances: 1000, warmup: 120, frames: 600, allowed: 0 }),
  BROWSER_RUN: Object.freeze({ warmup: 60, frames: 300 }),
  CAPTURE: Object.freeze({ width: 1280, height: 720, resizeTo: Object.freeze([960, 540]), resizeFrames: 30 }),
  CLEAR_COLOR: Object.freeze([0.05, 0.06, 0.08, 1.0]),
  QUALITY: Object.freeze({ LOW: 8, LOD0: 24, HIGH: 64, MIN: 3, MAX: 256 }),
  TRIANGLE_BUDGET_DEFAULT: 1500,
});

// `--quick` (default) | `--full`; `--only <regex>` takes the next argument.
export function parseArgs(argv) {
  let full = false;
  let only = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--full') full = true;
    else if (a === '--quick') full = false;
    else if (a === '--only') {
      if (i + 1 >= argv.length) throw new Error('--only needs a regex argument');
      only = new RegExp(argv[++i]);
    }
  }
  return { quick: !full, full, only };
}

const ARGS = parseArgs(process.argv.slice(2));

// Runs one check; prints `CHECK <label> | measured: <m> | threshold: <t> | PASS|FAIL`.
// Skipped when `--only` does not match the label. Exits 1 on failure.
export async function check(label, fn) {
  if (ARGS.only && !ARGS.only.test(label)) return;
  let measured;
  let threshold = '-';
  let ok = false;
  try {
    const r = await fn();
    measured = r.measured;
    threshold = r.threshold;
    ok = r.ok === true;
  } catch (e) {
    measured = e instanceof Error ? e.message : String(e);
    ok = false;
  }
  console.log(`CHECK ${label} | measured: ${measured} | threshold: ${threshold} | ${ok ? 'PASS' : 'FAIL'}`);
  if (!ok) process.exit(1);
}

// spawnSync without a shell, inheriting the environment.
export function run(cmd, args = [], opts = {}) {
  const r = spawnSync(cmd, args, {
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
    env: process.env,
    ...opts,
    shell: false,
  });
  const code = r.error ? -1 : (r.status ?? (r.signal ? 128 : -1));
  const stderr = (r.stderr ?? '') + (r.error ? `\n${r.error.message}` : '');
  return { code, stdout: r.stdout ?? '', stderr };
}

function cpus() {
  const n = Math.floor(Number(process.env.NMC_CPUS));
  return n >= 1 ? n : 1;
}

// Runs `worker(item, index)` with at most NMC_CPUS in flight; results in input order.
export async function pool(items, worker) {
  const list = Array.from(items);
  const results = new Array(list.length);
  let next = 0;
  async function lane() {
    while (next < list.length) {
      const i = next++;
      results[i] = await worker(list[i], i);
    }
  }
  const lanes = [];
  for (let k = 0; k < Math.min(cpus(), list.length); k++) lanes.push(lane());
  await Promise.all(lanes);
  return results;
}

export function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

export function gzipSize(path) {
  return gzipSync(readFileSync(path), { level: 9 }).length;
}

const SOFTWARE_RE = /swiftshader|llvmpipe|lavapipe|software|basic render/i;

// Validates a committed BrowserResult (S6) at proofs/results/<name>.json.
export function browserResultCheck(name, { requireHardware = false, maxCpuMs, instances } = {}) {
  const rel = `proofs/results/${name}.json`;
  const ffMax = VALUES.FIRST_FRAME_MAX_MS;
  const thresholdParts = ['nonEmpty true', 'adapter named', 'hardware stated', `firstFrameMs <= ${ffMax}`];
  if (requireHardware) thresholdParts.push('hardware true');
  if (maxCpuMs !== undefined) thresholdParts.push(`cpuMsMedian <= ${maxCpuMs}`);
  if (instances !== undefined) thresholdParts.push(`instances == ${instances}`);
  const threshold = thresholdParts.join(', ');
  const path = join(REPO_ROOT, rel);
  if (!existsSync(path)) return { measured: `missing ${rel}`, threshold, ok: false };
  let r;
  try {
    r = readJson(path);
  } catch (e) {
    return { measured: `invalid ${rel}: ${e.message}`, threshold, ok: false };
  }
  if (r === null || typeof r !== 'object' || Array.isArray(r)) {
    return { measured: `invalid ${rel}: expected an object`, threshold, ok: false };
  }
  const ad = r.adapter && typeof r.adapter === 'object' ? r.adapter : {};
  const adapterName = [ad.vendor, ad.device, ad.description]
    .filter((s) => typeof s === 'string' && s.trim() !== '')
    .join(' / ');
  const hwWord = r.hardware === true ? 'hardware' : r.hardware === false ? 'software' : 'hardware-unstated';
  const parts = [
    `nonEmpty=${r.nonEmpty}`,
    `adapter=${adapterName || '(none)'}`,
    hwWord,
    `firstFrameMs=${r.firstFrameMs}`,
  ];
  let ok =
    r.nonEmpty === true &&
    adapterName !== '' &&
    typeof r.hardware === 'boolean' &&
    typeof r.firstFrameMs === 'number' &&
    Number.isFinite(r.firstFrameMs) &&
    r.firstFrameMs <= ffMax;
  if (typeof r.hardware === 'boolean') {
    const looksSoftware =
      ad.isFallbackAdapter === true ||
      [ad.vendor, ad.architecture, ad.device, ad.description].some((s) => typeof s === 'string' && SOFTWARE_RE.test(s));
    if (r.hardware && looksSoftware) parts.push('(adapter looks software)');
  }
  if (requireHardware && r.hardware !== true) ok = false;
  if (maxCpuMs !== undefined) {
    parts.push(`cpuMsMedian=${r.cpuMsMedian}`);
    if (!(typeof r.cpuMsMedian === 'number' && Number.isFinite(r.cpuMsMedian) && r.cpuMsMedian <= maxCpuMs)) ok = false;
  }
  if (instances !== undefined) {
    parts.push(`instances=${r.instances}`);
    if (r.instances !== instances) ok = false;
  }
  return { measured: parts.join(', '), threshold, ok };
}
