// M4 primitive asset loader proof. --quick and --full are identical: every case is cheap.
// Cases run sequentially inside prims-check, so there is no NMC_CPUS fan-out.
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserResultCheck, check, gzipSize, readJson, run, VALUES } from './lib/proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const BIN = join(ROOT, 'target/release/examples/prims-check');
const TMP = join(ROOT, 'target/proofs');
const TIMEOUT_MS = 30 * 60 * 1000;
const OPTS = { cwd: ROOT, timeout: TIMEOUT_MS };

function tail(text, lines = 12) {
  return text.trim().split(/\r?\n/).slice(-lines).join(' / ');
}

// Each artefact is built at most once per run, lazily, so a skipped check builds nothing.
let primsBuilt = false;
function primsCheck() {
  if (!primsBuilt) {
    const b = run('cargo', ['build', '--release', '-p', 'raa-loader', '--example', 'prims-check'], OPTS);
    if (b.code !== 0) throw new Error(`cargo build prims-check exit ${b.code}: ${tail(b.stderr)}`);
    primsBuilt = true;
  }
}

function runCases(name, pattern) {
  primsCheck();
  mkdirSync(TMP, { recursive: true });
  const out = join(TMP, `m4-${name}.json`);
  const r = run(BIN, ['--only', pattern, '--json', out], OPTS);
  if (!existsSync(out)) throw new Error(`prims-check ${name} exit ${r.code}, no JSON: ${tail(r.stderr + r.stdout)}`);
  const cases = readJson(out);
  return { code: r.code, cases, stdout: r.stdout };
}

let cargoTest = null;
function loaderTests() {
  if (!cargoTest) {
    const t = run('cargo', ['test', '-p', 'raa-loader'], OPTS);
    cargoTest = { ok: t.code === 0, detail: t.code === 0 ? '' : `cargo test exit ${t.code}: ${tail(t.stderr + t.stdout)}` };
  }
  return cargoTest;
}

let webBuilt = false;
function webSizes() {
  if (!webBuilt) {
    const b = run(join(ROOT, 'scripts/build-web.sh'), ['demos/m4-asset'], OPTS);
    if (b.code !== 0) throw new Error(`build-web.sh demos/m4-asset exit ${b.code}: ${tail(b.stderr + b.stdout)}`);
    webBuilt = true;
  }
  return readJson(join(ROOT, 'dist/m4-asset/sizes.json'));
}

// Timing thresholds come first and alone.
await check('browser-run', () => {
  const r = browserResultCheck('m4-asset', { requireHardware: false });
  const j = readJson(join(ROOT, 'proofs/results/m4-asset.json'));
  return {
    measured: `${r.measured}, cpuMsMedian=${Number(j.cpuMsMedian).toFixed(2)} ms/frame`,
    threshold: `firstFrameMs <= ${VALUES.FIRST_FRAME_MAX_MS}`,
    ok: r.ok,
  };
});

await check('roundtrip', () => {
  const { code, cases } = runCases('roundtrip', '^roundtrip:');
  const canon = cases.filter((c) => /^roundtrip:(lamp|critter|crate)$/.test(c.label));
  const non = cases.filter((c) => c.label.startsWith('roundtrip:noncanonical-'));
  const a = canon.filter((c) => c.ok).length;
  const b = non.filter((c) => c.ok).length;
  const ok = code === 0 && cases.length > 0 && cases.every((c) => c.ok) && canon.length === 3 && non.length >= 6;
  return {
    measured: `canonical identical ${a}/${canon.length}, non-canonical idempotent ${b}/${non.length}`,
    threshold: 'exit 0, all ok',
    ok,
  };
});

await check('refusals', () => {
  const { code, cases } = runCases('refusals', '^refusal:');
  const good = cases.filter((c) => c.ok && c.code === c.expected && c.detail).length;
  return {
    measured: `${good}/${cases.length} cases refused with their expected code`,
    threshold: 'all',
    ok: code === 0 && cases.length > 0 && good === cases.length,
  };
});

await check('build', () => {
  const { code, cases } = runCases('build', '^build:');
  const t = loaderTests();
  const per = cases.map(
    (c) => `${c.label.slice(6)} ${c.triangles}/${c.budget} tris, meshes ${c.meshes}/${c.unionNodes}, colliders ${c.colliders}`,
  );
  const ok =
    code === 0 &&
    t.ok &&
    cases.length === 3 &&
    cases.every((c) => c.ok && c.triangles <= c.budget && c.meshes === c.unionNodes && c.colliders > 0);
  return {
    measured: `${per.join('; ')}; ${t.ok ? 'loader tests pass' : t.detail}`,
    threshold: 'meshes = union nodes, triangles <= budget, colliders derived, loader tests pass',
    ok,
  };
});

await check('wasm-size', () => {
  const sizes = webSizes();
  const path = join(ROOT, 'dist/m4-asset', sizes.wasm);
  const gz = gzipSize(path);
  return {
    measured: `raw ${sizes.raw}, gzip ${gz}`,
    threshold: String(VALUES.WASM_GZIP_MAX),
    ok: gz <= VALUES.WASM_GZIP_MAX,
  };
});
