// M16 (M1 S1-0) animation-brick proof: node proofs/m16-anim-brick.mjs [--quick|--full] [--only <regex>]
// Five functional checks: unit, bricks, det-guard, node-load, browser-load. No hash or SHA
// value is compared with anything (owner rule: no hash proofs); each record is validated for
// its exact shape and requested seed/ticks. --quick (default) and --full are identical.
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';
import { captureAnimBrick } from '../scripts/anim-capture.mjs';
import { check, pool, run, VALUES } from './lib/proof.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, 'dist/bricks');
const CASES = [[0, 0], [1, 60], [305419896, 600]];
const HASH_RE = /^[0-9a-f]{8}$/;
const U32_MAX = 0xffffffff;
const TIMEOUT_MS = 15 * 60 * 1000;
const EXPECTED = [
  { name: 'raa', package: '@etcito/raa', version: '0.4.1', dir: 'raa', wasm: true, cap: null },
  { name: 'raa-sdfm', package: '@etcito/raa-sdfm', version: '0.4.1', dir: 'raa-sdfm', wasm: false, cap: null },
  { name: 'raa-anim', package: '@etcito/raa-anim', version: '0.5.0', dir: 'raa-anim', wasm: true, cap: VALUES.WASM_GZIP_MAX },
];

const fail = (message) => { throw new Error(message); };
const tail = (text, lines = 10) => text.trim().split(/\r?\n/).slice(-lines).join(' / ');
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const recordsText = (list) => list.map((r) => `${r.seed}/${r.ticks}:${r.hash}`).join(' ');
const u32 = (value) => Number.isInteger(value) && value >= 0 && value <= U32_MAX;

// One canonical result object: exact keys, case, u32 seed/ticks, eight lowercase hex digits,
// and the seed/ticks the producer was asked for. The hash string is the simulation's output
// data: its shape is checked, its value is never compared with another environment.
function one(record, seed, ticks, label) {
  const keys = record && typeof record === 'object' && !Array.isArray(record) ? Object.keys(record).sort().join(',') : '';
  if (keys !== 'case,hash,seed,ticks' || record.case !== 'det-selftest' || !u32(record.seed) || !u32(record.ticks)) {
    fail(`${label}: malformed result (keys, case or u32 seed/ticks)`);
  }
  if (typeof record.hash !== 'string' || !HASH_RE.test(record.hash)) fail(`${label}: malformed hash`);
  if (record.seed !== seed || record.ticks !== ticks) fail(`${label}: wrong input seed=${record.seed} ticks=${record.ticks}`);
  return { case: record.case, seed: record.seed, ticks: record.ticks, hash: record.hash };
}

function measure(rel) {
  const file = join(OUT, rel);
  if (!existsSync(file)) fail(`missing staged wasm ${rel}`);
  const bytes = readFileSync(file);
  return { raw: bytes.length, gzip: gzipSync(bytes, { level: 9 }).length };
}

// Validates the staged packages, their entrypoints and the manifest; recomputes every wasm
// raw/gzip size. The manifest's sha256 field is data and is deliberately not compared.
function validateBricks() {
  const manifest = JSON.parse(readFileSync(join(OUT, 'bricks.json'), 'utf8'));
  if (manifest.format !== 'raa.bricks' || manifest.version !== 1 || manifest.wasmGzipMax !== VALUES.WASM_GZIP_MAX
      || !Array.isArray(manifest.bricks) || manifest.bricks.length !== EXPECTED.length) {
    fail('bricks.json is not a raa.bricks v1 manifest with three bricks and the Q2 budget');
  }
  const sizes = [];
  let anim = null;
  for (let i = 0; i < EXPECTED.length; i++) {
    const want = EXPECTED[i];
    const brick = manifest.bricks[i];
    if (!brick || brick.name !== want.name || brick.package !== want.package || brick.version !== want.version
        || brick.dir !== want.dir || brick.wasmGzipMax !== want.cap) {
      fail(`bricks.json[${i}] is not ${want.name} ${want.package}@${want.version}`);
    }
    if (!brick.source || brick.source.repository !== 'raa' || typeof brick.source.path !== 'string'
        || (want.name === 'raa-anim' && brick.source.crate !== 'raa-anim-js')) {
      fail(`${want.name}: manifest source provenance is incomplete`);
    }
    const pkgFile = join(OUT, want.dir, 'package.json');
    if (!existsSync(pkgFile)) fail(`${want.name}: missing package.json`);
    const pkg = JSON.parse(readFileSync(pkgFile, 'utf8'));
    if (pkg.name !== want.package || pkg.version !== want.version) fail(`${want.name}: package.json is ${pkg.name}@${pkg.version}`);
    const targets = [pkg.main, pkg.types, ...(pkg.files ?? []),
      ...Object.values(pkg.exports ?? {}).flatMap((value) => typeof value === 'string' ? [value] : Object.values(value ?? {}))]
      .filter((value) => typeof value === 'string' && !value.includes('*'));
    for (const target of targets) {
      if (!existsSync(join(OUT, want.dir, target.replace(/^\.\//, '')))) fail(`${want.name}: missing entrypoint ${target}`);
    }
    if (!want.wasm) {
      if (brick.wasm !== null) fail(`${want.name}: expected no wasm`);
      continue;
    }
    if (!brick.wasm || typeof brick.wasm !== 'object' || Array.isArray(brick.wasm) || typeof brick.wasm.path !== 'string'
        || !brick.wasm.path.endsWith('.wasm') || !brick.wasm.path.startsWith(`${want.dir}/`)) {
      fail(`${want.name}: invalid wasm record in bricks.json`);
    }
    const size = measure(brick.wasm.path);
    for (const key of ['raw', 'gzip']) {
      if (brick.wasm[key] !== size[key]) fail(`${want.name}: manifest ${key} differs from the staged wasm`);
    }
    if (want.cap !== null && size.gzip > want.cap) fail(`${want.name}: gzip ${size.gzip} B exceeds ${want.cap} B`);
    if (want.name === 'raa-anim') anim = { ...size, path: brick.wasm.path };
    sizes.push(`${want.name} ${size.raw}/${size.gzip} B raw/gzip`);
  }
  if (!anim) fail('raa-anim wasm is missing');
  return { anim, measured: `3 installable bricks, manifest ok; ${sizes.join(', ')}; raa-anim cap ${VALUES.WASM_GZIP_MAX} B` };
}

// Removed and rebuilt once by the bricks check; later checks reuse the same staged outputs.
let bricksPromise = null;
function ensureBricks(fresh = false) {
  if (!bricksPromise || fresh) {
    bricksPromise = (async () => {
      if (fresh) rmSync(OUT, { recursive: true, force: true });
      if (fresh || !existsSync(join(OUT, 'bricks.json'))) {
        const built = run('bash', ['scripts/build-bricks.sh'], { cwd: ROOT, timeout: TIMEOUT_MS });
        if (built.code !== 0) fail(`build-bricks.sh exit ${built.code}: ${tail(built.stderr + built.stdout)}`);
      }
      return validateBricks();
    })();
  }
  return bricksPromise;
}

async function stagedApi(source, label) {
  const mod = await import(pathToFileURL(join(OUT, 'raa-anim', 'index.js')).href);
  const api = await mod.loadRaaAnim(source);
  if (!same(api.brickInfo(), { name: 'raa-anim', version: '0.5.0' })) fail(`${label}: brickInfo is not the 0.5.0 raa-anim metadata`);
  return api;
}

// Selftest cases are independent; pool bounds them by NMC_CPUS (default 1).
function selftestRecords(api, label) {
  return pool(CASES, async ([seed, ticks], i) => {
    const record = api.selftest('det-selftest', seed, ticks);
    return one(record, seed, ticks, `${label}[${i}]`);
  });
}

await check('unit', () => {
  const r = run('cargo', ['test', '-p', 'raa-det', '-p', 'raa-anim-js'],
    { cwd: ROOT, timeout: TIMEOUT_MS });
  return {
    measured: `cargo test -p raa-det -p raa-anim-js exit ${r.code}`
      + (r.code === 0 ? '' : `: ${tail(r.stderr + r.stdout)}`),
    threshold: 'cargo test -p raa-det -p raa-anim-js exit 0',
    ok: r.code === 0,
  };
});

await check('bricks', async () => {
  const built = await ensureBricks(true);
  return {
    measured: built.measured,
    threshold: `dist/bricks removed then built once; 3 packages and bricks.json valid; raa-anim gzip <= ${VALUES.WASM_GZIP_MAX}`,
    ok: true,
  };
});

await check('det-guard', () => {
  const r = run('bash', ['scripts/check-det.sh'], { cwd: ROOT, timeout: 120_000 });
  const zero = /check-det: 0 hits across (\d+) Rust file\(s\) in [^\n]+/.exec(r.stdout + r.stderr);
  return {
    measured: r.code === 0 && zero ? zero[0] : `exit ${r.code}: ${tail(r.stderr + r.stdout)}`,
    threshold: 'scripts/check-det.sh exit 0 with an explicit 0-hit result',
    ok: r.code === 0 && Boolean(zero),
  };
});

await check('node-load', async () => {
  await ensureBricks();
  const api = await stagedApi(undefined, 'node default');
  const node = await selftestRecords(api, 'node');
  // Staged loader source forms: default (above), directory, paths, URL, and preloaded bytes.
  const wasmFile = join(OUT, 'raa-anim', 'pkg', 'raa_anim_js_bg.wasm');
  const bytes = readFileSync(wasmFile);
  const forms = [
    ['package directory', join(OUT, 'raa-anim')],
    ['absolute wasm path', wasmFile],
    ['relative wasm path', relative(process.cwd(), wasmFile)],
    ['file URL', pathToFileURL(wasmFile)],
    ['file URL string', pathToFileURL(wasmFile).href],
    ['Uint8Array bytes', new Uint8Array(bytes)],
    ['ArrayBuffer bytes', bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)],
  ];
  for (const [label, source] of forms) {
    const formApi = await stagedApi(source, label);
    await selftestRecords(formApi, label);
  }
  return {
    measured: `node=${recordsText(node)}; selftest returned 3/3 records; loader forms ${forms.length + 1}/${forms.length + 1}`,
    threshold: 'loadRaaAnim() loads the staged brick in Node and selftest returns the three canonical records (hash values not compared)',
    ok: true,
  };
});

await check('browser-load', async () => {
  await ensureBricks();
  const chrome = await captureAnimBrick({ cases: CASES });
  if (chrome.records.length !== CASES.length) fail(`Chrome returned ${chrome.records.length} records, expected ${CASES.length}`);
  for (let i = 0; i < CASES.length; i++) {
    one(chrome.records[i], CASES[i][0], CASES[i][1], `chrome[${i}]`);
  }
  return {
    measured: `chrome=${recordsText(chrome.records)}; run() returned 3/3 records; adapter ${chrome.adapterText} (${chrome.hardware ? 'hardware' : 'software'})`,
    threshold: 'examples/anim-brick loads the staged brick in Chrome and run() returns the three canonical records (hash values not compared)',
    ok: true,
  };
});
